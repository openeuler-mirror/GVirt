#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""gather_sparse_kv_cache kernel test.

For each (batch, topK) case, builds a paged kCache/peCache + block_tables +
topk_indices (token-index semantics), runs the xlite gather kernel, and
compares against a host-side reference gather that maps each token index to
its physical block via block_tables.
"""
from __future__ import absolute_import

import logging
import numpy as np
import torch
from xlite._C import Runtime, gather_sparse_kv_cache

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True

BLOCK_SIZE = 128

# model configs: name, kv_lora_rank, rope_head_dim, dtype
models = [
    ("glm5", 512, 64, torch.bfloat16),
]

# work configs: batch, cached_lens (decode => query_len=1 per batch)
work = [
    (1, [16384]),
    (3, [32751] * 3),
    (3, [2047, 8191, 45841]),
    (4, [2047] * 4),
    (1, [131071]),
    (1, [127]),
    (1, [4096]),
]

# topK values (must be <= MAX_TOPK_NUM=2048)
topk_values = [512, 2048]


def max_blocks(cached_lens, block_size):
    max_sum = max(cached_lens) + 1  # decode adds 1 query token
    return (max_sum + block_size - 1) // block_size


def run_test(name, kv_lora_rank, rope_head_dim, test_dtype, batch, cached_lens_list, topK):
    max_num_blocks = max_blocks(cached_lens_list, BLOCK_SIZE)
    max_seq_len = max_num_blocks * BLOCK_SIZE
    kv_heads = 1

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        kvcache_block_num = max_num_blocks * batch
        k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, kv_heads, kv_lora_rank)
        pe_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, kv_heads, rope_head_dim)

        # block_tables: batch x max_num_blocks, contiguous layout
        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(max_num_blocks, dtype=np.uint32)
        block_tables_array = batch_indices * max_num_blocks + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(),
                                    dtype=torch.int32).flatten()

        # topk_indices: [batch, topK] token-index semantics (decode => 1 query/batch)
        topk_indices_list = []
        for i in range(batch):
            clen = cached_lens_list[i]
            total_len = clen + 1  # decode query token
            valid_len = total_len
            if valid_len >= topK:
                perm = torch.randperm(valid_len)[:topK]
            else:
                perm = torch.randperm(valid_len)
                pad = perm[-1:].expand(topK - valid_len)
                perm = torch.cat([perm, pad])
            topk_indices_list.append(perm)
        topk_indices_tensor = torch.stack(topk_indices_list).to(dtype=torch.int32)
        # sort per row (kernel does not require sort, but reference matches either way)
        topk_indices_tensor, _ = torch.sort(topk_indices_tensor, dim=-1)

        k_dense_xlite = torch.zeros(batch, topK, kv_heads, kv_lora_rank, dtype=test_dtype)
        pe_dense_xlite = torch.zeros(batch, topK, kv_heads, rope_head_dim, dtype=test_dtype)

        # decode => 1 query token per batch; cached_lens = seq len prior to query
        query_lens = torch.tensor([1] * batch, dtype=torch.int32).flatten()
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()

    # reference gather on host — only fill valid slots. Kernel processes
    # slots [0, min(totalLen, topK)) per batch where totalLen = queryLen + cacheLen;
    # slots beyond that are skipped (dense cache stays zero-init).
    k_dense_ref = torch.zeros(batch, topK, kv_heads, kv_lora_rank, dtype=test_dtype)
    pe_dense_ref = torch.zeros(batch, topK, kv_heads, rope_head_dim, dtype=test_dtype)
    valid_mask = torch.zeros(batch, topK, dtype=torch.bool)
    for b in range(batch):
        clen = cached_lens_list[b]
        total_len = clen + 1  # query_len == 1 in decode
        valid_cnt = min(total_len, topK)
        for i in range(valid_cnt):
            tok = int(topk_indices_tensor[b, i].item())
            block_id = tok // BLOCK_SIZE
            rem = tok % BLOCK_SIZE
            phys = int(block_tables[b * max_num_blocks + block_id].item())
            k_dense_ref[b, i] = k_cache_xlite[phys, rem]
            pe_dense_ref[b, i] = pe_cache_xlite[phys, rem]
            valid_mask[b, i] = True

    torch.npu.synchronize()
    gather_sparse_kv_cache(rt, k_cache_xlite, pe_cache_xlite, block_tables,
                           topk_indices_tensor, query_lens, cached_lens, k_dense_xlite,
                           pe_dense_xlite, batch, topK, BLOCK_SIZE,
                           kv_lora_rank, rope_head_dim)
    torch.npu.synchronize()

    logging.info(
        "gather_sparse_kv_cache %s (kv_lora_rank=%d, rope_head_dim=%d, %s) "
        "work (batch=%d, cached_lens=%s, topK=%d) executed!",
        name, kv_lora_rank, rope_head_dim, test_dtype, batch, cached_lens_list, topK)

    # Compare only valid slots; invalid slots are zero-init and not written.
    k_xlite = k_dense_xlite.cpu()[valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(k_dense_ref)]
    k_ref = k_dense_ref[valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(k_dense_ref)]
    pe_xlite = pe_dense_xlite.cpu()[valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pe_dense_ref)]
    pe_ref = pe_dense_ref[valid_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pe_dense_ref)]
    try:
        torch.testing.assert_close(k_xlite, k_ref, atol=0, rtol=0)
        torch.testing.assert_close(pe_xlite, pe_ref, atol=0, rtol=0)
        logging.info("gather_sparse_kv_cache case passed (exact match on %d valid slots).",
                     int(valid_mask.sum().item()))
    except AssertionError as e:
        logging.error(f"gather_sparse_kv_cache mismatch: {e}")
        logging.error(f"k_dense_xlite (valid): {k_xlite}")
        logging.error(f"k_dense_ref   (valid): {k_ref}")
        logging.error(f"pe_dense_xlite (valid): {pe_xlite}")
        logging.error(f"pe_dense_ref   (valid): {pe_ref}")
        raise


def main():
    for name, kv_lora_rank, rope_head_dim, test_dtype in models:
        for batch, cached_lens_list in work:
            assert len(cached_lens_list) == batch
            for topK in topk_values:
                # skip configs where cached_len+1 < topK is too small to be meaningful
                min_total = min(c + 1 for c in cached_lens_list)
                if min_total < 1:
                    continue
                run_test(name, kv_lora_rank, rope_head_dim, test_dtype, batch,
                         cached_lens_list, topK)


if __name__ == "__main__":
    main()
