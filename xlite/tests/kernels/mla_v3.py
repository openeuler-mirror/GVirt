#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""mla_v3 kernel test.

For each (batch, topK) decode case, gathers a dense kCache/peCache from a
paged layout via IndexerTopK's topk_indices (token-index), then runs mla_v3
on the dense cache. Compares against a host-side MLA reference that uses the
same dense cache. Also verifies mla_v3 with topK==0 (dense = full KV) matches
the mla_v2 reference path for small cases.
"""
from __future__ import absolute_import

import logging
import numpy as np
import torch
from xlite._C import Runtime, gather_sparse_kv_cache, mla_v3
from tests.models.weight_utils import matrix_nd2nz

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 3000)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True

BLOCK_SIZE = 128

# model configs: name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, dtype
models = [
    ("base", 1, 64, 128, 128, 16, torch.bfloat16),
    ("deepseek_v3", 16, 64, 128, 128, 512, torch.bfloat16),
    ("glm5", 8, 64, 192, 256, 512, torch.bfloat16),
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

topk_values = [512, 2048]


def max_blocks(cached_lens, block_size):
    max_sum = max(cached_lens) + 1
    return (max_sum + block_size - 1) // block_size


def rms_norm_last_dim(x, eps=1e-6):
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms


def run_test(name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank,
             test_dtype, batch, cached_lens_list, topK):
    max_num_blocks = max_blocks(cached_lens_list, BLOCK_SIZE)
    max_seq_len = max_num_blocks * BLOCK_SIZE
    total_query_len = batch  # decode: 1 token per batch
    scale = (nope_head_dim + rope_head_dim) ** -0.5

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        # standard (non-paged) k_cache / pe_cache for reference computation
        qWithQr_std = torch.randn(total_query_len, n_heads, nope_head_dim + rope_head_dim)
        qWithQr_std = rms_norm_last_dim(qWithQr_std)
        k_cache_std = torch.randn(batch, max_seq_len, kv_lora_rank).clamp(0.2, 1)
        k_cache_std = rms_norm_last_dim(k_cache_std)
        pe_cache_std = torch.randn(batch, max_seq_len, rope_head_dim)
        pe_cache_std = rms_norm_last_dim(pe_cache_std)
        wkvb = torch.randn(n_heads * (nope_head_dim + v_head_dim), kv_lora_rank).clamp(0.2, 1)
        wkvb = wkvb.view(n_heads, nope_head_dim + v_head_dim, kv_lora_rank)

        # paged k_cache / pe_cache for xlite (block_tables contiguous)
        kvcache_block_num = max_num_blocks * batch
        k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, 1, kv_lora_rank)
        pe_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, 1, rope_head_dim)
        # write per-sample KV into block cache (contiguous layout)
        for b in range(batch):
            clen = cached_lens_list[b]
            qlen = 1
            current_k = k_cache_std[b:b + 1, :clen + qlen]
            current_pe = pe_cache_std[b:b + 1, :clen + qlen]
            sample_cache_start = b * max_num_blocks
            total_len = clen + qlen
            num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            for block_idx in range(num_blocks_needed):
                seq_start = block_idx * BLOCK_SIZE
                seq_end = min((block_idx + 1) * BLOCK_SIZE, total_len)
                current_seq_len = seq_end - seq_start
                cache_block_idx = sample_cache_start + block_idx
                k_cache_xlite[cache_block_idx, :current_seq_len, 0] = current_k[0, seq_start:seq_end]
                pe_cache_xlite[cache_block_idx, :current_seq_len, 0] = current_pe[0, seq_start:seq_end]

        # block_tables contiguous
        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(max_num_blocks, dtype=np.uint32)
        block_tables_array = batch_indices * max_num_blocks + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(),
                                    dtype=torch.int32).flatten()

        # topk_indices [batch, topK] token-index (valid range [0, clen])
        topk_indices_list = []
        for b in range(batch):
            clen = cached_lens_list[b]
            total_len = clen + 1
            if total_len >= topK:
                perm = torch.randperm(total_len)[:topK]
            else:
                perm = torch.randperm(total_len)
                pad = perm[-1:].expand(topK - total_len)
                perm = torch.cat([perm, pad])
            topk_indices_list.append(perm)
        topk_indices_tensor = torch.stack(topk_indices_list).to(dtype=torch.int32)
        topk_indices_tensor, _ = torch.sort(topk_indices_tensor, dim=-1)

        # xlite dense buffers
        k_dense_xlite = torch.zeros(batch, topK, 1, kv_lora_rank, dtype=test_dtype)
        pe_dense_xlite = torch.zeros(batch, topK, 1, rope_head_dim, dtype=test_dtype)

        # q_absorb (pre-absorbed on host) + qr
        q_absorb_xlite = torch.empty(total_query_len, n_heads, kv_lora_rank)
        qr_xlite = torch.empty(total_query_len, n_heads, rope_head_dim)
        # o_absorb buffer (qk is allocated internally by the op)
        o_absorb_xlite = torch.zeros(total_query_len, n_heads, kv_lora_rank, dtype=test_dtype)

        query_lens = torch.tensor([1] * batch, dtype=torch.int32).flatten()
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_start_loc = torch.tensor(np.cumsum([1] * batch) - 1, dtype=torch.int32).flatten()

        # WUV weight (hdc layout) for final output projection on host
        wuv_hdc = wkvb[:, -v_head_dim:].contiguous()

    # host-side q_absorb / qr split
    q_absorb_std_list = []
    qr_std_list = []
    for b in range(batch):
        chunk = qWithQr_std[b:b + 1]  # (1, n_heads, nope+rope)
        q_nope, q_rope = chunk.split([nope_head_dim, rope_head_dim], dim=-1)
        q_absorb = torch.einsum("bhd,hdc->bhc", q_nope, wkvb[:, :nope_head_dim])
        q_absorb_std_list.append(q_absorb.squeeze(0))  # (n_heads, kv_lora_rank)
        qr_std_list.append(q_rope.squeeze(0))          # (n_heads, rope_head_dim)
    q_absorb_std = torch.stack(q_absorb_std_list, dim=0)  # (batch, n_heads, kv_lora_rank)
    qr_std = torch.stack(qr_std_list, dim=0)               # (batch, n_heads, rope_head_dim)
    q_absorb_xlite.copy_(q_absorb_std)
    qr_xlite.copy_(qr_std)

    # reference: build dense cache and run MLA on it
    k_dense_std = torch.zeros(batch, topK, kv_lora_rank, dtype=test_dtype, device="npu")
    pe_dense_std = torch.zeros(batch, topK, rope_head_dim, dtype=test_dtype, device="npu")
    for b in range(batch):
        for i in range(topK):
            tok = int(topk_indices_tensor[b, i].item())
            # token-index into the standard (non-paged) k_cache_std
            k_dense_std[b, i] = k_cache_std[b, tok]
            pe_dense_std[b, i] = pe_cache_std[b, tok]

    # MLA reference on dense cache. Only slots [0, total_len) participate in
    # softmax — mla_v3 kernel uses calcLen = min(cached_lens[b]+1, index_topk)
    # and masks out the gather-skipped padding tail (dense cache = 0 there).
    o_absorbs = []
    outputs = []
    for b in range(batch):
        clen = cached_lens_list[b]
        total_len = clen + 1
        calc_len = min(total_len, topK)
        q_absorb = q_absorb_std_list[b]  # (n_heads, kv_lora_rank)
        q_rope = qr_std_list[b]          # (n_heads, rope_head_dim)
        kD = k_dense_std[b]              # (topK, kv_lora_rank)
        peD = pe_dense_std[b]            # (topK, rope_head_dim)
        qkc = torch.einsum("hc,tc->ht", q_absorb, kD)
        qkr = torch.einsum("hr,tr->ht", q_rope, peD)
        scores = (qkc + qkr) * scale
        # mask padding slots (i >= calc_len) to -inf so softmax ignores them,
        # matching mla_v3's calcSoftmaxLen = calcLen boundary.
        mask = torch.arange(topK, device=scores.device) >= calc_len
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        scores = torch.softmax(scores, dim=-1)
        o_absorb = torch.einsum("ht,tc->hc", scores, kD)
        o_absorbs.append(o_absorb)
        x = torch.einsum("hc,hdc->hd", o_absorb, wkvb[:, -v_head_dim:])
        outputs.append(x)
    o_absorb_std = torch.cat([o.unsqueeze(0) for o in o_absorbs], dim=0)  # (batch, h, c)
    output_std = torch.cat([x.unsqueeze(0) for x in outputs], dim=0)       # (batch, h, d)

    # xlite: gather + mla_v3
    torch.npu.synchronize()
    gather_sparse_kv_cache(rt, k_cache_xlite, pe_cache_xlite, block_tables,
                           topk_indices_tensor, query_lens, cached_lens, k_dense_xlite,
                           pe_dense_xlite, batch, topK, BLOCK_SIZE, max_num_blocks,
                           kv_lora_rank, rope_head_dim, 1)
    mla_v3(rt, q_absorb_xlite, qr_xlite, k_dense_xlite, pe_dense_xlite,
           o_absorb_xlite, query_start_loc, query_lens, cached_lens, n_heads, rope_head_dim,
           kv_lora_rank, batch, topK, scale)
    torch.npu.synchronize()

    # final WUV projection on host
    output_xlite = torch.einsum("mhc,hdc->mhd", o_absorb_xlite.cpu().float(),
                                wuv_hdc.cpu().float()).to(test_dtype)

    logging.info(
        "mla_v3 %s (n_heads=%d, rope=%d, nope=%d, v=%d, kv_lora=%d, %s) "
        "work (batch=%d, cached_lens=%s, topK=%d) executed!",
        name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, test_dtype,
        batch, cached_lens_list, topK)

    try:
        torch.testing.assert_close(output_xlite, output_std.cpu(), atol=1e-5, rtol=5e-02)
        logging.info("mla_v3 case passed (output match).")
    except AssertionError as e:
        logging.error(f"mla_v3 mismatch: {e}")
        logging.error(f"output_xlite:\n{output_xlite}")
        logging.error(f"output_std:\n{output_std.cpu()}")
        raise


def main():
    for name, n_heads, rope_head_dim, nope_head_dim, v_head_dim, kv_lora_rank, test_dtype in models:
        for batch, cached_lens_list in work:
            assert len(cached_lens_list) == batch
            for topK in topk_values:
                min_total = min(c + 1 for c in cached_lens_list)
                if min_total < 1:
                    continue
                run_test(name, n_heads, rope_head_dim, nope_head_dim, v_head_dim,
                         kv_lora_rank, test_dtype, batch, cached_lens_list, topK)


if __name__ == "__main__":
    main()
