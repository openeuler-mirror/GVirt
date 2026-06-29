#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import

import logging
import numpy as np
import torch
from xlite._C import Runtime, indexer_topk

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

BLOCK_SIZE = 128

# model configurations: name, n_heads, head_dim, dtype
models = [
    ("indexer_64", 64, 128, torch.bfloat16),
    ("indexer_64_fp16", 64, 128, torch.float16),
]

# work configurations: batch_size, cached_lens, query_lens
work = [
    (1, [0], [1]),
    (1, [0], [4]),
    (1, [2049], [1]),
    (1, [3542], [1]),
    (1, [0], [8124]),
    (1, [9728], [1]),
    (1, [10184], [1]),
    (1, [0], [13542]),
    (1, [0], [30]),
    (1, [0], [128]),
    (1, [10], [4000]),
    (1, [100], [30]),
    (2, [0] * 2, [4, 8]),
    (2, [8123, 0], [1, 8231]),
    (4, [5012, 127, 2189, 500], [4, 2, 6, 8]),
]

# topK values to exercise (must be <= MAX_TOPK_NUM=2048)
topk_values = [512, 2048]

# Max percentage of differing indices tolerated per row. Computation order
# and precision can swap near-tie indices, so we compare as sets (position
# within a row doesn't matter) and allow a small fraction to differ. Set to
# 0.0 to require exact set-equality.
DIFF_PCT_THRESHOLD = 3.0  # percent


def max_blocks(query_lens, cached_lens, block_size):
    max_sum = max(a + b for a, b in zip(query_lens, cached_lens))
    return (max_sum + block_size - 1) // block_size


def run_test(name, n_heads, head_dim, test_dtype, batch, cached_lens_list,
             query_len_list, topK):
    max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE)
    max_seq_len = max_num_blocks * BLOCK_SIZE
    total_query_len = sum(query_len_list)

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        q_standard = torch.randn(total_query_len, n_heads * head_dim)
        k_cache = torch.randn(batch, max_seq_len, head_dim)
        weight_standard = torch.randn(total_query_len, head_dim + n_heads)

        q_xlite = q_standard.clone()
        weight_xlite = weight_standard.clone()

        kvcache_block_num = max_num_blocks * batch
        k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, head_dim)

        query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_lens_np = np.array(query_len_list)
        query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
        query_start_loc = torch.tensor(query_start_loc_np.tolist(),
                                       dtype=torch.int32).flatten()

        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(max_num_blocks, dtype=np.uint32)
        block_tables_array = batch_indices * max_num_blocks + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(),
                                    dtype=torch.int32).flatten()

        indices = torch.arange(max_seq_len, dtype=torch.int32).npu()
        topk_indices_xlite = torch.empty(total_query_len, topK,
                                         dtype=torch.int32).npu()

    # standard indexer_scores forward: process each sample
    index_scores_standard_list = []
    offset = 0
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        q_chunk = q_standard[offset: offset + qlen]
        weight_chunk = weight_standard[offset: offset + qlen,
                                       head_dim: head_dim + n_heads]
        offset += qlen

        q_chunk = q_chunk.view(qlen, n_heads, head_dim)
        k_cache_slice = k_cache[i:i + 1, :clen + qlen]
        scores = torch.einsum("bshd,btd->bsht", q_chunk.unsqueeze(0),
                              k_cache_slice)
        scores = scores.squeeze(0)
        index_score = torch.einsum("sht,sh->st", scores, weight_chunk)
        index_scores_standard_list.append(index_score)

    # xlite: write per-sample KV into block cache
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        current_k = k_cache[i:i + 1, :qlen + clen]
        sample_cache_start = i * max_num_blocks
        total_len = clen + qlen
        num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for block_idx in range(num_blocks_needed):
            seq_start = block_idx * BLOCK_SIZE
            seq_end = min((block_idx + 1) * BLOCK_SIZE, total_len)
            current_seq_len = seq_end - seq_start
            cache_block_idx = sample_cache_start + block_idx
            if cache_block_idx >= (i + 1) * max_num_blocks:
                break
            k_cache_xlite[cache_block_idx, :current_seq_len] = \
                current_k[:, seq_start:seq_end]

    # standard topk: for each query position, select topK indices from
    # [0, clen+qlen) range. When total_len < topK, pad with the last valid
    # index (this matches the kernel's behavior of only writing kvLen
    # indices when total_len <= topK — we compare only the valid prefix).
    standard_topk_list = []
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        total_len = clen + qlen
        index_score = index_scores_standard_list[i]  # [qlen, total_len]
        k = min(topK, total_len)
        # topk returns (values, indices); we only need indices
        _, tk_idx = torch.topk(index_score, k=k, dim=-1)
        if k < topK:
            # pad with the last selected index to reach topK columns
            pad = tk_idx[:, -1:].expand(qlen, topK - k)
            tk_idx = torch.cat([tk_idx, pad], dim=-1)
        standard_topk_list.append(tk_idx.to(dtype=torch.int32))
    standard_topk = torch.cat(standard_topk_list, dim=0)

    torch.npu.synchronize()
    indexer_topk(rt, q_xlite, k_cache_xlite, weight_xlite, indices,
                 topk_indices_xlite, query_start_loc, query_lens, cached_lens,
                 block_tables, n_heads, head_dim, BLOCK_SIZE, batch,
                 max_num_blocks, topK)
    torch.npu.synchronize()

    logging.info(
        "indexer_topk %s (%d heads, %d head_dim, %s) work (%d batch, "
        "cached_lens=%s, query_lens=%s, topK=%d) executed!",
        name, n_heads, head_dim, test_dtype, batch, cached_lens_list,
        query_len_list, topK)

    # Compare per batch. When total_len < topK, the kernel only writes the
    # first total_len indices; remaining columns are uninitialized. We
    # therefore compare only the first min(topK, total_len) columns.
    all_match = True
    offset = 0
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        total_len = clen + qlen
        k = min(topK, total_len)

        sample_standard = standard_topk[offset: offset + qlen, :k]
        sample_xlite = topk_indices_xlite[offset: offset + qlen, :k]

        # topk ties can produce different index orderings; compute order
        # and precision can also swap near-tie indices. Compare as sets per
        # row: position within a row doesn't matter, only the fraction of
        # indices that differ (set difference / k). Sort on CPU — aclnnSort
        # on NPU fails for tiny slices (e.g. k=1) with "Dst tensor size is
        # less than src tensor size".
        std_sorted, _ = torch.sort(sample_standard.cpu(), dim=-1)
        xlite_sorted, _ = torch.sort(sample_xlite.cpu(), dim=-1)

        # For each std index, searchsorted in xlite_sorted. Since topk
        # indices are unique within a row, a hit means set membership.
        # diff_count[i] = number of std indices absent from xlite row i.
        pos = torch.searchsorted(xlite_sorted, std_sorted)
        pos = pos.clamp_max(k - 1)
        hits = (xlite_sorted.gather(-1, pos) == std_sorted).to(torch.int32)
        diff_count = k - hits.sum(dim=-1).to(torch.int32)
        diff_pct = diff_count.to(torch.float32) * (100.0 / k)

        max_pct = diff_pct.max().item()
        mean_pct = diff_pct.mean().item()

        if max_pct > DIFF_PCT_THRESHOLD:
            logging.error(
                f"Sample {i} mismatch: max diff pct={max_pct:.2f}% "
                f"(mean={mean_pct:.2f}%), threshold={DIFF_PCT_THRESHOLD:.2f}%")
            logging.error(f"torch (sorted): {std_sorted}")
            logging.error(f"xlite (sorted): {xlite_sorted}")
            all_match = False
        else:
            logging.info(
                f"Sample {i} ok: max diff pct={max_pct:.2f}%, "
                f"mean={mean_pct:.2f}%")
        offset += qlen

    if all_match:
        logging.info("All samples passed for topK=%d!", topK)


def main():
    for name, n_heads, head_dim, test_dtype in models:
        for batch, cached_lens_list, query_len_list in work:
            assert len(cached_lens_list) == batch
            assert len(query_len_list) == batch
            for topK in topk_values:
                # skip configurations where topK is larger than the
                # scratch constraint MAX_TOPK_NUM or doesn't make sense
                run_test(name, n_heads, head_dim, test_dtype, batch,
                         cached_lens_list, query_len_list, topK)


if __name__ == "__main__":
    main()
