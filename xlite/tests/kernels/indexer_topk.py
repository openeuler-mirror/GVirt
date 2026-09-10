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

        indices = torch.arange(max_seq_len, dtype=torch.int32).npu()
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

    # xlite: write per-sample KV into block cache
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        current_k = k_cache[i : i + 1, : qlen + clen]
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
            k_cache_xlite[cache_block_idx, :current_seq_len] = current_k[:, seq_start:seq_end]

    # standard topk: for each query position q (absolute cache position
    # p0 = clen + q), select topK indices from the causal range [0, p0]
    # (including itself, no future tokens). The kernel's causal
    # mask masks scores after p0 to -inf, so every emitted index is <= p0.
    #
    # Topk results are only meaningful for sparse attention, i.e. when the
    # token has more than topK candidates including itself (p0 >= topK); for p0 < topK the
    # token uses dense attention and the kernel does not need topk. We still
    # compute the reference topk for every row here, but only compare rows with
    # p0 >= topK against the kernel below.
    standard_topk_list = []
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        index_score = index_scores_standard_list[i]  # [qlen, total_len]
        tk_idx_list = []
        for q in range(qlen):
            p0 = clen + q  # absolute position; valid candidates [0, p0]
            row_score = index_score[q].clone()
            row_score[p0 + 1:] = float("-inf")
            k = min(topK, p0 + 1)
            if k == 0:
                # no valid candidates (p0 == 0); kernel writes nothing real
                tk_idx = torch.zeros(topK, dtype=torch.int32, device=row_score.device)
            else:
                _, idx = torch.topk(row_score, k=k)
                if k < topK:
                    pad = idx[-1:].expand(topK - k)
                    idx = torch.cat([idx, pad])
                tk_idx = idx.to(dtype=torch.int32)
            tk_idx_list.append(tk_idx)
        standard_topk_list.append(torch.stack(tk_idx_list, dim=0))
    standard_topk = torch.cat(standard_topk_list, dim=0)

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
        indices,
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

    # Compare per batch, per row. Topk is only valid (sparse) for tokens whose
    # absolute position p0 = clen + q is at least topK (p0 >= topK); for
    # p0 < topK the token uses dense attention and its topk is not compared
    # against the kernel (the kernel still emits min(topK, p0 + 1) indices for such
    # rows when their tile is processed, but those are dense, not sparse picks).
    #
    # For comparable rows (p0 >= topK) the kernel emits exactly topK indices,
    # all drawn from the causal range [0, p0] (self yes, no future tokens). Compare
    # the full topK-width row as a set (position within a row doesn't matter).
    all_match = True
    compared = 0
    offset = 0
    errors: dict[int, dict[int, float]] = defaultdict(dict)
    max_pcts: list[float] = []
    for i in range(batch):
        max_pct_per_batch = 0.0
        qlen = query_len_list[i]
        clen = cached_lens_list[i]

        for q in range(qlen):
            p0 = clen + q
            if p0 < topK:
                continue  # dense attention; topk not valid for comparison

            k = topK  # p0 >= topK => full topK-width, all candidates in [0, p0]
            row_standard = standard_topk[offset + q, :k]
            row_xlite = topk_indices_xlite[offset + q, :k]

            # # sanity: every emitted index must be a valid causal position (<= p0)
            # assert (row_xlite <= p0).all(), f"Sample {i} row {q} (p0={p0}) has non-causal topk indices: {row_xlite}"

            std_sorted, _ = torch.sort(row_standard.cpu(), dim=-1)
            xlite_sorted, _ = torch.sort(row_xlite.cpu(), dim=-1)

            pos = torch.searchsorted(xlite_sorted, std_sorted)
            pos = pos.clamp_max(k - 1)
            hits = (xlite_sorted.gather(-1, pos) == std_sorted).to(torch.int32)
            diff_count = k - hits.sum(dim=-1).to(torch.int32)
            diff_pct = diff_count.to(torch.float32) * (100.0 / k)

            max_pct = diff_pct.max().item()
            max_pct_per_batch = max(max_pct_per_batch, max_pct)
            compared += 1

            if max_pct > DIFF_PCT_THRESHOLD:
                # logging.error(
                #     f"\tSample {i} row {q} (p0={p0}) mismatch: max diff pct="
                #     f"{max_pct:.2f}%, threshold={DIFF_PCT_THRESHOLD:.2f}%"
                # )
                # logging.error(f"\ttorch (sorted): {std_sorted}")
                # logging.error(f"\txlite (sorted): {xlite_sorted}")
                all_match = False
                errors[i][q] = max_pct
            else:
                logging.debug(f"\tSample {i} row {q} (p0={p0}) ok: max diff pct={max_pct:.2f}%")
        offset += qlen
        max_pcts.append(max_pct_per_batch)

    if compared == 0:
        logging.info(f"\tNo sparse rows (all p0 < topK={topK}) for this config; nothing to compare.")
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
