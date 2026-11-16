#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import logging
from collections import defaultdict

import numpy as np
import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, indexer_topk

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, size=500)
torch.npu.set_device(0)

BLOCK_SIZE = 128

# model configurations: name, n_heads, head_dim, dtype
models = [
    ("indexer_64", 64, 128, torch.bfloat16),
    ("indexer_64_fp16", 64, 128, torch.float16),
]

# work configurations: batch_size, cached_lens, query_lens. One shape per
# branch: dense/sparse/mixed rows, single/multi KV-tile relay, decode/prefill/
# multi-batch mixes, and the ratio==4 edges (last three entries).
work = [
    (1, [0], [1]),  # single uncached token: dense rows (host template path)
    (1, [0], [30]),  # small prefill, mixed tile
    (1, [100], [30]),  # cached prefix, mixed dense/sparse rows
    (1, [2049], [1]),  # decode, p0 == topK boundary (topK=2048)
    (1, [16384], [1]),  # decode, 5 KV tiles: cross-chunk lastTopk relay chain
    (1, [0], [4352]),  # prefill, 2 KV tiles
    (1, [10], [4000]),  # dense prefix + sparse tail
    (2, [0, 98], [4096, 8192]),  # multi-batch, mixed dense/sparse
    (4, [5012, 127, 2189, 500], [4, 2, 6, 8]),  # 4-batch short queries
    (1, [0], [3]),  # ratio==4: qlen < ratio, first block incomplete (p0 == -1)
    (1, [0], [2048]),  # ratio==4: 512 blocks == topK, dense boundary
    (1, [98], [4096]),  # ratio==4: clen not divisible by ratio
]

# fp16 only re-instantiates the kernel template; a representative subset is enough
work_fp16 = [
    work[2],
    work[4],
    work[10],
]

topk_values = [2048]
topk_values_compressed = [512]

# Max percentage of differing indices tolerated per row. Computation order
# and precision can swap near-tie indices, so we compare as sets (position
# within a row doesn't matter) and allow a small fraction to differ. Set to
# 0.0 to require exact set-equality.
DIFF_PCT_THRESHOLD = 3.0  # percent


def max_blocks(query_lens, cached_lens, block_size, compress_ratio=1):
    # When compress_ratio > 1 the KV cache holds one row per compressed block, so
    # the paged-block count derives from (clen+qlen)//ratio, not the token count.
    max_sum = max((a + b) // compress_ratio for a, b in zip(query_lens, cached_lens))
    return (max_sum + block_size - 1) // block_size


def run_test(
    name,
    n_heads,
    head_dim,
    test_dtype,
    batch,
    cached_lens_list,
    query_len_list,
    topK,
    compress_ratio=1,
    dense_weight=False,
):
    if dense_weight:
        name = name + "_dense_w"
    ratio = compress_ratio
    max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE, ratio)
    max_seq_len = max_num_blocks * BLOCK_SIZE
    total_query_len = sum(query_len_list)

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        q_standard = torch.randn(total_query_len, n_heads * head_dim)
        k_cache = torch.randn(batch, max_seq_len, head_dim)
        if dense_weight:
            # CXA-style dense head weights: [T, nHeads] with no fused K columns.
            weight_standard = torch.randn(total_query_len, n_heads)
        else:
            weight_standard = torch.randn(total_query_len, head_dim + n_heads)

        q_xlite = q_standard.clone()
        weight_xlite = weight_standard.clone()

        kvcache_block_num = max_num_blocks * batch
        k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, head_dim)

        query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_start_loc = (query_lens.cumsum(0) - query_lens).int()

        # persistent buffer with stale trails (the op narrows to the
        # lens-derived width, pitch unchanged)
        valid = torch.arange(batch)[:, None] * max_num_blocks + torch.arange(max_num_blocks)
        padded = torch.zeros(batch, max_num_blocks + 3, dtype=torch.int32)
        padded[:, :max_num_blocks] = valid
        block_tables = padded

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
        seq_len = (clen + qlen) // ratio if ratio > 1 else clen + qlen
        q_chunk = q_std_cpu[offset : offset + qlen].view(qlen, n_heads, head_dim)
        if dense_weight:
            weight_chunk = weight_std_cpu[offset : offset + qlen]
        else:
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
        total_len = (clen + query_len_list[i]) // ratio if ratio > 1 else clen + query_len_list[i]
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
        # index_scores width is the compressed-block count when ratio > 1
        # (one column per KV block), not the token count.
        seq_len = (clen + qlen) // ratio if ratio > 1 else clen + qlen
        s = index_scores_standard_list[i]  # [qlen, seq_len]
        if seq_len < score_dim:
            s = torch.cat([s, torch.full((qlen, score_dim - seq_len), float("-inf"))], dim=1)
        padded_scores.append(s)
        P = torch.arange(clen, clen + qlen, dtype=torch.int64)
        # ratio==1: causal includes own position (p0 = P).
        # ratio>1: causal excludes own incomplete block (p0 = (P+1)//ratio - 1).
        p0_per_row.append((P + 1) // ratio - 1 if ratio > 1 else P)
    all_scores = torch.cat(padded_scores, dim=0)  # [total_query_len, score_dim]
    p0_all = torch.cat(p0_per_row)  # [total_query_len]
    pos_grid = torch.arange(score_dim, dtype=torch.int64).unsqueeze(0)
    causal_mask = pos_grid <= p0_all.unsqueeze(1)  # [total, score_dim], True where valid
    all_scores.masked_fill_(~causal_mask, float("-inf"))
    _, standard_topk = torch.topk(all_scores, k=topK, dim=-1)
    standard_topk = standard_topk.to(dtype=torch.int32)

    logging.info(
        "indexer_topk %s (%d heads, %d head_dim, %s) work (%d batch, cached_lens=%s, query_lens=%s, topK=%d, compress_ratio=%d) executed!",
        name,
        n_heads,
        head_dim,
        test_dtype,
        batch,
        cached_lens_list,
        query_len_list,
        topK,
        compress_ratio,
    )

    # Both partitioning modes and every host tile tier must give identical
    # results on every shape. q_tile 0 = the compile-time default (no
    # override); 1 = extreme small tile (many tail blocks); 17/22 = the
    # chunked-prefill / mixed-batch host tiers; 24 = the macro ceiling. The
    # work set's query lens are mostly non-multiples of 17/22/24, so the
    # per-tile tail blocks are exercised throughout.
    for kv_split in (0, 1):
        for q_tile in (0, 1, 17, 22, 24):
            # Poison the output to catch rows the kernel fails to write; sync so
            # the fill cannot race the op's non-default stream.
            topk_indices_xlite.fill_(-1)
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
                compress_ratio=ratio,
                kv_split=kv_split,
                q_tile=q_tile,
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
                        f"\tSample {i} row {q} (p0={p0_q}) has non-causal topk indices: {xlite_cpu_np[offset + q]}"
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
                logging.info(f"\tkv_split={kv_split}, q_tile={q_tile}: no rows to compare for topK={topK}.")
            elif all_match:
                logging.info(
                    f"\tkv_split={kv_split}, q_tile={q_tile}: all samples passed for topK={topK}! "
                    f"max diff pct per batch: {max_pcts}"
                )
            else:
                logging.error(
                    f"\tkv_split={kv_split}, q_tile={q_tile}: some samples failed for topK={topK}! "
                    f"max diff pct per batch: {max_pcts}"
                )
                for i, err in errors.items():
                    error_info = ", ".join(f"{row} ({pct:.2f}%)" for row, pct in err.items())
                    logging.error(f"\tBatch {i}: {error_info}")
                # The tier-sweep contract above must fail the run, not just log
                # (mirrors indexer_c8.py's asserts with a non-zero exit code).
                raise AssertionError(
                    f"kv_split={kv_split}, q_tile={q_tile}, topK={topK}: mismatched rows "
                    + "; ".join(
                        f"batch {i} (row (diff pct)): " + ", ".join(f"{row} ({pct:.2f}%)" for row, pct in err.items())
                        for i, err in errors.items()
                    )
                )


def main():
    for name, n_heads, head_dim, test_dtype in models:
        selected = work if test_dtype == torch.bfloat16 else work_fp16
        for batch, cached_lens_list, query_len_list in selected:
            assert len(cached_lens_list) == batch
            assert len(query_len_list) == batch
            # DSA indexer (ratio==1)
            for topK in topk_values:
                run_test(name, n_heads, head_dim, test_dtype, batch, cached_lens_list, query_len_list, topK)
            # Compressed-block indexer (ratio==4)
            for topK in topk_values_compressed:
                run_test(
                    name, n_heads, head_dim, test_dtype, batch, cached_lens_list, query_len_list, topK, compress_ratio=4
                )

    # Dense [T, nHeads] weight layout (CXA path): the kernel derives the row
    # pitch from weight.shape[1]; verify it matches the fused-layout reference
    # on both ratio=1 and ratio=4.
    run_test(models[0][0], 64, 128, models[0][3], 1, [3542], [1], 2048, dense_weight=True)
    run_test(models[0][0], 64, 128, models[0][3], 1, [0], [8192], 512, compress_ratio=4, dense_weight=True)
    run_test(models[1][0], 64, 128, models[1][3], 1, [3542], [1], 2048, dense_weight=True)

    # q_tile above the kernel's build-time maximum must be rejected before any
    # launch (the kernel's L1 q/w staging is macro-sized at build time).
    n_heads, head_dim, topK = 64, 128, 2048
    with torch.device("npu"):
        q = torch.randn(1, n_heads * head_dim)
        k_cache = torch.randn(1, BLOCK_SIZE, head_dim)
        weight = torch.randn(1, head_dim + n_heads)
        topk_indices = torch.empty(1, topK, dtype=torch.int32)
    query_start_loc = torch.tensor([0], dtype=torch.int32)
    lens = torch.tensor([1], dtype=torch.int32)
    cached_lens = torch.tensor([0], dtype=torch.int32)
    block_tables = torch.zeros(1, 1, dtype=torch.int32)
    for bad_q_tile in (25, 26):
        try:
            indexer_topk(
                rt,
                q,
                k_cache,
                weight,
                topk_indices,
                query_start_loc,
                lens,
                cached_lens,
                block_tables,
                n_heads,
                head_dim,
                BLOCK_SIZE,
                1,
                topK,
                q_tile=bad_q_tile,
            )
        except RuntimeError as e:
            assert "qTile must be in" in str(e), f"unexpected error for q_tile={bad_q_tile}: {e}"
        else:
            raise AssertionError(f"q_tile={bad_q_tile} was not rejected")
    logging.info("q_tile values above the build-time maximum are rejected")


if __name__ == "__main__":
    main()
