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

from collections import defaultdict

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


def run_test(name, n_heads, head_dim, test_dtype, batch, cached_lens_list, query_len_list, topK):
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
        query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(max_num_blocks, dtype=np.uint32)
        block_tables_array = batch_indices * max_num_blocks + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

        topk_indices_xlite = torch.empty(total_query_len, topK, dtype=torch.int32).npu()

    # standard indexer_scores forward: process each sample. The reference score
    # tensor [qlen, seq_len, n_heads] is fp32 and grows ~qlen*seq_len*n_heads;
    # for large prefill (qlen up to 13542) that is tens of GiB, which OOMs the
    # device. Compute it on the host instead, chunked over query rows so peak
    # host memory stays ~chunk*seq_len*n_heads*4B (a few hundred MiB).
    q_std_cpu = q_standard.cpu().float()
    k_cache_cpu = k_cache.cpu().float()
    weight_std_cpu = weight_standard.cpu().float()

    index_scores_standard_list = []
    offset = 0
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        seq_len = clen + qlen
        q_chunk = q_std_cpu[offset : offset + qlen].view(qlen, n_heads, head_dim)
        weight_chunk = weight_std_cpu[offset : offset + qlen, head_dim : head_dim + n_heads]
        offset += qlen

        k_slice = k_cache_cpu[i, :seq_len]  # [seq_len, head_dim]
        index_score = torch.empty(qlen, seq_len, dtype=torch.float32)
        # scores[s] = clamp(q[s] @ k.T, 0); index_score[s] = scores[s] @ weight[s]
        chunk = 256
        for s0 in range(0, qlen, chunk):
            s1 = min(s0 + chunk, qlen)
            scores = torch.einsum("chd,td->cht", q_chunk[s0:s1], k_slice)  # [chunk, seq_len, n_heads]
            scores = scores.clamp(min=0)  # ReLU filter to match kernel's L0C copy with reluEn=1
            index_score[s0:s1] = torch.einsum("cht,ch->ct", scores, weight_chunk[s0:s1])
        index_scores_standard_list.append(index_score)

    # xlite: scatter per-sample KV into contiguous block cache (one slice-copy
    # per sample; blocks are contiguous so flatten to [nblk*BLOCK, head_dim]).
    for i in range(batch):
        clen = cached_lens_list[i]
        total_len = clen + query_len_list[i]
        num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        start = i * max_num_blocks
        dst = k_cache_xlite[start : start + num_blocks_needed].view(-1, head_dim)
        dst[:total_len] = k_cache[i, :total_len]

    # standard topk: select topK from the causal range [0, p0] for each query
    # row (p0 = clen + q). Sparse rows (p0 >= topK) get a real topk; dense rows
    # (p0 < topK) use the default template 0..topK-1 and are compared exactly.
    # Vectorized: pad to a common width, apply a per-row causal mask, one
    # batched torch.topk. Dense-row results are unused (template-compared).
    score_dim = max(max_seq_len, topK)
    padded_scores = []
    p0_per_row = []
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        seq_len = clen + qlen
        s = index_scores_standard_list[i]  # [qlen, seq_len]
        if seq_len < score_dim:
            s = torch.cat([s, torch.full((qlen, score_dim - seq_len), float("-inf"))], dim=1)
        padded_scores.append(s)
        p0_per_row.append(torch.arange(clen, clen + qlen, dtype=torch.int64))
    all_scores = torch.cat(padded_scores, dim=0)  # [total_query_len, score_dim]
    p0_all = torch.cat(p0_per_row)  # [total_query_len]
    pos_grid = torch.arange(score_dim, dtype=torch.int64).unsqueeze(0)
    causal_mask = pos_grid <= p0_all.unsqueeze(1)  # [total, score_dim], True where valid
    all_scores.masked_fill_(~causal_mask, float("-inf"))
    _, standard_topk = torch.topk(all_scores, k=topK, dim=-1)
    standard_topk = standard_topk.to(dtype=torch.int32)

    logging.info(
        "indexer_topk %s (%d heads, %d head_dim, %s) work (%d batch, cached_lens=%s, query_lens=%s, topK=%d) executed!",
        name,
        n_heads,
        head_dim,
        test_dtype,
        batch,
        cached_lens_list,
        query_len_list,
        topK,
    )

    torch.npu.synchronize()
    indexer_topk(
        rt,
        q_xlite,
        k_cache_xlite,
        weight_xlite,
        topk_indices_xlite,
        query_start_loc,
        query_lens,
        cached_lens,
        block_tables,
        n_heads,
        head_dim,
        BLOCK_SIZE,
        batch,
        topK,
    )
    torch.npu.synchronize()

    # Vectorized comparison over [total_query_len, topK]. Sparse rows: order-
    # tolerant set comparison. Dense rows: exact template match. Causal sanity:
    # sparse rows must only emit indices <= p0.
    xlite_cpu = topk_indices_xlite.cpu()
    row_default = torch.arange(0, topK, dtype=torch.int32)
    is_dense = p0_all < topK

    causal_fail = (~is_dense) & ~(xlite_cpu.to(torch.int64) <= p0_all.unsqueeze(1)).all(dim=-1)

    dense_exact = (xlite_cpu == row_default).all(dim=-1)
    dense_pct = torch.zeros(len(p0_all), dtype=torch.float32)
    dense_pct[~dense_exact] = 100.0

    std_sorted, _ = torch.sort(standard_topk, dim=-1)
    xlite_sorted, _ = torch.sort(xlite_cpu, dim=-1)
    pos = torch.searchsorted(xlite_sorted, std_sorted).clamp_max(topK - 1)
    hits = xlite_sorted.gather(-1, pos) == std_sorted
    diff_count = topK - hits.sum(dim=-1)
    sparse_pct = diff_count.to(torch.float32) * (100.0 / topK)

    row_pct = torch.where(is_dense, dense_pct, sparse_pct)
    row_pct = torch.where(causal_fail, torch.full_like(row_pct, 100.0), row_pct)

    row_pct_np = row_pct.numpy()
    causal_fail_np = causal_fail.numpy()
    xlite_cpu_np = xlite_cpu.numpy()

    all_match = True
    compared = 0
    offset = 0
    errors: dict[int, dict[int, float]] = defaultdict(dict)
    max_pcts: list[float] = []
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        bs = slice(offset, offset + qlen)
        batch_pct = row_pct_np[bs]
        batch_fail = causal_fail_np[bs]

        # causal violations: log + mark, excluded from compared/max_pct
        for q in np.nonzero(batch_fail)[0]:
            q = int(q)
            p0_q = clen + q
            all_match = False
            errors[i][q] = 100.0
            logging.error(
                f"\tSample {i} row {q} (p0={p0_q}) has non-causal topk indices: "
                f"{xlite_cpu_np[offset + q]}"
            )

        ok = ~batch_fail
        compared += int(ok.sum())
        batch_ok_pct = batch_pct[ok]
        max_pct_per_batch = float(batch_ok_pct.max()) if batch_ok_pct.size else 0.0
        max_pcts.append(max_pct_per_batch)

        # threshold violations among non-causal rows
        ok_local = np.nonzero(ok)[0]
        over = ok_local[batch_pct[ok] > DIFF_PCT_THRESHOLD]
        for q in over:
            q = int(q)
            all_match = False
            errors[i][q] = float(batch_pct[q])
        offset += qlen

    if compared == 0:
        logging.info(f"\tNo rows to compare for topK={topK}.")
    elif all_match:
        logging.info(f"\tAll samples passed for topK={topK}! max diff pct per batch: {max_pcts}")
    else:
        logging.error(f"\tSome samples failed for topK={topK}! max diff pct per batch: {max_pcts}")
        for i, err in errors.items():
            error_info = ", ".join(f"{row} ({pct:.2f}%)" for row, pct in err.items())
            logging.error(f"\tBatch {i}: {error_info}")


def main():
    for name, n_heads, head_dim, test_dtype in models:
        for batch, cached_lens_list, query_len_list in work:
            assert len(cached_lens_list) == batch
            assert len(query_len_list) == batch
            for topK in topk_values:
                # skip configurations where topK is larger than the
                # scratch constraint MAX_TOPK_NUM or doesn't make sense
                run_test(name, n_heads, head_dim, test_dtype, batch, cached_lens_list, query_len_list, topK)


if __name__ == "__main__":
    main()
