#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""gather_sparse_kv_cache kernel test (packed variable-length queries).

The kernel gathers, for each query token q of batch b (laid out contiguously at
inOffset = queryStartLoc[b] + q), the top-K KV rows its indexer selected into a
contiguous dense cache. A row is dense (num_candidates = (cached_lens[b] + q + 1)
// compress_ratio <= topK: identity template, dummy indices not read) or sparse
(indices read in any order; consecutive ascending/descending runs in one block
form a burst, and a descending run lands reversed — set semantics). Each case is
checked against an exact-match reference mirroring the kernel's grouping plus an
order-independent byte-multiset oracle.

Coverage: decode (query_len=1) and variable-length (prefill/mixed) queries x
orders x burst sizes (512..16384) x topK (1..2048, including partial chunks and
the 4B-aligned-only topkIndices GM source), compress_ratio 1/2, a deterministic
grouping-edge pattern, permuted block tables, host channel swap, nope
single-channel variants, cachedLens window reload (44929 rows), and host-side
rejection of a channel emptiness mismatch.
"""

from __future__ import absolute_import

import itertools
import logging
import numpy as np
import torch
from xlite._C import Runtime, gather_sparse_kv_cache

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True

BLOCK_SIZE = 128
CNT_ONE_LOOP = 32  # kernel processes topK in chunks of 32 slots per task
GM_UB_BURST_SIZE = 16 * 1024
GM_UB_BURST_SIZE_TILING = 8 * 1024

# model configs: name, kv_lora_rank, rope_head_dim, dtype
models = [
    ("glm5", 512, 64, torch.bfloat16),
]

# decode work configs: batch, cached_lens, query_lens (query_len=1 per batch);
# cached_lens[b] < topK rows exercise the dense-attention template path (a decode
# row is dense iff cached_lens[b] + 1 <= topK, i.e. cached_lens[b] < topK)
work = [
    (1, [16384], [1]),
    (3, [32751] * 3, [1] * 3),
    (3, [2047, 8191, 45841], [1, 1, 1]),  # mixed: 2047 < topK=2048 => row 0 dense (template)
    (4, [2047] * 4, [1] * 4),  # all rows dense (template) for topK=2048
    (1, [131071], [1]),
    (1, [127], [1]),  # dense for both topKs; only prefix [0, 128) gathered
    (1, [4096], [1]),
    (2, [2048, 2049], [1, 1]),  # boundary: 2048+1=2049 > 2048, 2049+1=2050 > 2048 => both sparse
    # variable-length (prefill/mixed) queries: the causal candidate range
    # (cached_len + q + 1) // ratio grows with q, so one row can transition
    # dense->sparse across its query tokens (e.g. [2047] + qlen 64 at topK=2048:
    # q=0 dense (2048==topK), q=63 sparse (2111>2048)) — exercises the packed
    # inOffset = queryStartLoc[b] + q addressing
    (1, [0], [128]),          # pure prefill, no cached KV (always dense up to topK)
    (1, [100], [30]),         # prefill after a prefix cache
    (1, [2047], [64]),        # q=0 dense -> q=63 sparse at topK=2048
    (2, [4000, 0], [1, 77]),  # mixed decode + prefill in one batch
    (2, [8191, 127], [64, 1]),
]

# compress_ratio=2 work configs (CXA compressed-KV classifier): dense iff
# (cached_len + 1) // 2 <= topK; 4096/4097 straddle the topK=2048 boundary
work_ratio2 = [
    (1, [16384], [1]),
    (3, [1023, 1024, 4097], [1, 1, 1]),
    (2, [4096, 4097], [1, 1]),  # boundary: (4096+1)//2 = 2048 == topK=2048 => dense
    (1, [127], [1]),  # dense: numCandidates = 64 <= 512; only prefix [0, 64)
    # compress_ratio=2 variable-length: causal candidate (cached + q + 1) // 2
    (1, [4096], [64]),  # q=0 (4097//2=2048==topK) dense -> q=63 (4160//2=2080) sparse
    (2, [1023, 5000], [40, 1]),
]

# deterministic pattern work config (see make_pattern_indices); topK=2048 only
work_pattern = (2, [8191, 2047], [1, 1])  # row 0 sparse pattern, row 1 dense template

topk_values = [512, 2048]

# burst copy sizes (bytes): 512 => cnt 1 (token-by-token), 5120 => cnt 5
# (non-power-of-2), 8192 => cnt 8 (GM_UB_BURST_SIZE_TILING), 16384 => still
# cnt 8 (confined by GM_UB_BURST_SIZE_TILING)
burst_sizes = [512, 5120, 8192, 16384]

# topk index orders: ascending (indexer_topk contract), descending (legacy), random
orders = ["asc", "desc", "random"]


def max_blocks(cached_lens, query_lens, block_size):
    # cache must hold every batch's cached + query tokens; the last query token of
    # batch b attends causally to cached_lens[b] + query_lens[b] positions
    max_sum = max(c + q for c, q in zip(cached_lens, query_lens))
    return (max_sum + block_size - 1) // block_size


def burst_copy_cnt(first_dim, burst_copy_size=16 * 1024):
    """Mirror the kernel's burst count (tokens per burst copy-in), derived from
    the first channel — the larger per-token footprint — so K and PE bursts
    cover the same run: cnt = CLAMP(MIN(burstCopySize, 16384, 8192) /
    perKBytes, 1, MIN(32, 128))."""
    burst = min(burst_copy_size, GM_UB_BURST_SIZE, GM_UB_BURST_SIZE_TILING)
    per_bytes = first_dim * 2  # bf16
    return min(max(burst // per_bytes, 1), min(CNT_ONE_LOOP, BLOCK_SIZE))


def gen_topk_indices(valid_len, topK, clustered, order):
    """Sample topK distinct token ids from [0, valid_len) in the requested
    order (asc/desc/random). clustered=True biases toward consecutive runs;
    dense rows (valid_len <= topK) get dummy ids the kernel must not read."""
    if valid_len <= topK:
        # dummy indices for dense-attention rows (kernel skips these)
        return torch.randint(0, valid_len, (topK,), dtype=torch.int32)
    if not clustered:
        ids = torch.randperm(valid_len)[:topK].tolist()
    else:
        # seed with ~topK/2 random ids, extend each into a short consecutive
        # run (clamped to valid_len), dedup, then top up randomly.
        chosen = set()
        seeds = torch.randperm(valid_len)[: topK // 2 + 1].tolist()
        for t in seeds:
            run = int(torch.randint(1, 9, (1,)).item())
            for u in range(t, min(t + run, valid_len)):
                chosen.add(u)
                if len(chosen) >= topK:
                    break
            if len(chosen) >= topK:
                break
        while len(chosen) < topK:
            chosen.add(int(torch.randint(0, valid_len, (1,)).item()))
        ids = list(chosen)[:topK]
    ids = torch.tensor(ids, dtype=torch.int32)
    if order == "asc":
        ids, _ = torch.sort(ids)
    elif order == "desc":
        ids, _ = torch.sort(ids, descending=True)
    else:
        ids = ids[torch.randperm(len(ids))]
    return ids


def make_pattern_indices(valid_len, topK):
    """Deterministic sparse-row indices (topK == 2048, valid_len >= 4096)
    pinning the grouping edges: asc/desc runs, runs split by the burst cap,
    a run crossing a block boundary (must split), and runs split by the
    32-slot chunk boundary; tail filled with scattered singles."""
    assert valid_len >= 4096 and topK == 2048
    ids = []
    # chunk 0 (slots 0..31)
    ids += [100, 101, 102, 103, 104]  # asc run 5
    ids += [500]  # single
    ids += [300, 299, 298, 297, 296, 295, 294]  # desc run 7
    ids += [600, 601]  # asc run 2
    ids += list(range(700, 716))  # asc 16: splits at the burst cap
    ids += [125]  # head of a run that crosses a block boundary
    # chunk 1 (slots 32..63)
    ids += [126, 127, 128]  # asc 3: 126/127 in block 0, 128 in block 1 -> splits
    ids += [400, 399, 398, 397]  # desc 4
    ids += list(range(800, 816))  # asc 16: splits at the burst cap
    ids += [501]  # single
    ids += list(range(900, 892, -1))  # desc 8 ending at the chunk edge
    # chunk 2 (slots 64..95)
    ids += list(range(892, 884, -1))  # desc 8: continues across the chunk boundary
    ids += list(range(1000, 1008))  # asc run of exactly the max cap (8)
    ids += list(range(1008, 1016))  # adjacent asc run (capped again)
    ids += [200, 201]  # asc 2
    ids += [3000]  # single
    ids += list(range(435, 430, -1))  # desc 5: chunk-2 tail of a longer run
    # chunk 3 (slots 96..127)
    ids += list(range(430, 426, -1))  # desc 4: continuation across the chunk boundary
    ids += [385, 384]  # desc 2 ending exactly at a block edge (offset 0)
    # scattered singles to fill (deterministic)
    used = set(ids)
    t = 2000
    while len(ids) < topK:
        if t not in used:
            ids.append(t)
            used.add(t)
        t += 3
    assert len(ids) == topK and len(used) == topK and max(ids) < valid_len
    return torch.tensor(ids, dtype=torch.int32)


def gather_chunk_groups(phys, off, chunk, cnt, burst_cnt):
    """Mirror of the kernel's burst grouping within one chunk: direction from
    slot i+1 (+1/-1/0), extend while same block, same direction and below
    burst_cnt; a run fetches tokens [src, src+n) ascending into slots [i, i+n),
    so a descending run lands reversed. Yields (i, blk, src, n)."""
    groups = []
    i = 0
    while i < cnt:
        blk, blk_offset = phys[chunk + i], off[chunk + i]
        direction = 0
        if i + 1 < cnt and phys[chunk + i + 1] == blk:
            if off[chunk + i + 1] == blk_offset + 1:
                direction = 1
            elif off[chunk + i + 1] + 1 == blk_offset:
                direction = -1
        n = 1
        while (
            direction != 0
            and i + n < cnt
            and n < burst_cnt
            and phys[chunk + i + n] == blk
            and off[chunk + i + n] == blk_offset + direction * n
        ):
            n += 1
        src = blk_offset if direction >= 0 else blk_offset + 1 - n
        groups.append((i, blk, src, n))
        i += n
    return groups


def assert_row_set(out_rows, expected_rows):
    """Order-independent multiset comparison of gathered token rows (raw bytes):
    membership must be exact, order within a row is unspecified."""
    a = sorted(r.view(torch.uint8).numpy().tobytes() for r in out_rows)
    b = sorted(r.view(torch.uint8).numpy().tobytes() for r in expected_rows)
    assert a == b, "gathered rows are not the expected token-row multiset"


def run_test(
    name,
    kv_lora_rank,
    rope_head_dim,
    test_dtype,
    batch,
    cached_lens_list,
    query_lens_list,
    topK,
    order,
    compress_ratio=1,
    burst_copy_size=16 * 1024,
    channels="both",
    pattern=False,
    shared_single_block=False,
    permuted_blocks=False,
    pitched=False,
):
    total_query_len = sum(query_lens_list)
    max_num_blocks = max_blocks(cached_lens_list, query_lens_list, BLOCK_SIZE)
    kv_heads = 1
    # the burst count derives from the first channel (active channel for nope,
    # larger of the two otherwise — the host swaps so the first is never smaller)
    if channels == "both":
        first_dim = max(kv_lora_rank, rope_head_dim)
    elif channels == "k":
        first_dim = kv_lora_rank
    else:
        first_dim = rope_head_dim
    burst_cnt = burst_copy_cnt(first_dim, burst_copy_size)

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        # shared_single_block: every row maps to physical block 0 (huge batch, one block)
        kvcache_block_num = 1 if shared_single_block else max_num_blocks * batch
        # pitched: keep every other physical block so dim-0 stride doubles — the
        # kernels must address blocks via the passed cache stride, not the
        # contiguous pitch (InitXTensor forwards the non-contiguous stride)
        pitch = 2 if pitched else 1
        if channels == "pe":
            k_cache_xlite = torch.zeros(0, dtype=test_dtype)
            pe_cache_xlite = torch.randn(kvcache_block_num * pitch, BLOCK_SIZE, 1,
                                         rope_head_dim)[::pitch]
        elif channels == "k":
            k_cache_xlite = torch.randn(kvcache_block_num * pitch, BLOCK_SIZE, 1,
                                        kv_lora_rank)[::pitch]
            pe_cache_xlite = torch.zeros(0, dtype=test_dtype)
        else:
            k_cache_xlite = torch.randn(kvcache_block_num * pitch, BLOCK_SIZE, 1,
                                        kv_lora_rank)[::pitch]
            pe_cache_xlite = torch.randn(kvcache_block_num * pitch, BLOCK_SIZE, 1,
                                         rope_head_dim)[::pitch]

        # block_tables: batch x max_num_blocks (contiguous / all-zero / random permutation)
        if shared_single_block:
            block_tables = torch.zeros(batch * max_num_blocks, dtype=torch.int32)
        elif permuted_blocks:
            block_tables = (
                torch.randperm(max_num_blocks * batch).to(torch.int32).flatten()
            )
        else:
            batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
            block_indices = np.arange(max_num_blocks, dtype=np.uint32)
            block_tables_array = batch_indices * max_num_blocks + block_indices
            block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

        # topk_indices: [total_query_len, topK] (packed, one row per query token);
        # for query token q of batch b the causal candidate range is
        # [0, (cached_lens[b] + q + 1) // compress_ratio), so dense rows
        # (candidate <= topK) get dummy in-range indices the kernel must not read
        topk_indices_list = []
        gq = 0  # global query-token index, for varying the clustering seed
        for i in range(batch):
            clen = cached_lens_list[i]
            qlen = query_lens_list[i]
            for q in range(qlen):
                num_candidates = (clen + q + 1) // compress_ratio
                if pattern and num_candidates > topK and topK == 2048 and qlen == 1:
                    topk_indices_list.append(make_pattern_indices(num_candidates, topK))
                    gq += 1
                    continue
                clustered = gq % 2 == 0
                topk_indices_list.append(gen_topk_indices(num_candidates, topK, clustered, order))
                gq += 1
        topk_indices_tensor = torch.stack(topk_indices_list).to(dtype=torch.int32)

        if channels == "pe":
            k_dense_xlite = torch.zeros(0, dtype=test_dtype)
            pe_dense_xlite = torch.zeros(total_query_len, topK, 1, rope_head_dim, dtype=test_dtype)
        elif channels == "k":
            k_dense_xlite = torch.zeros(total_query_len, topK, 1, kv_lora_rank, dtype=test_dtype)
            pe_dense_xlite = torch.zeros(0, dtype=test_dtype)
        else:
            k_dense_xlite = torch.zeros(total_query_len, topK, 1, kv_lora_rank, dtype=test_dtype)
            pe_dense_xlite = torch.zeros(total_query_len, topK, 1, rope_head_dim, dtype=test_dtype)
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_lens = torch.tensor(query_lens_list, dtype=torch.int32).flatten()
        # query_start_loc: exclusive prefix sum (cumsum - lens), as in mla_v2;
        # decode (all-1) => [0, 1, ..., batch-1]
        query_lens_np = np.array(query_lens_list)
        query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
        query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

    # reference gather mirroring the kernel chunk-by-chunk over packed query
    # tokens: sparse rows fill all topK slots from topk_indices; dense rows take
    # the prefix via the identity template (slot i -> token i); suffix slots stay
    # untouched (zero-init here)
    k_dense_ref = torch.zeros(total_query_len, topK, 1, kv_lora_rank, dtype=test_dtype)
    pe_dense_ref = torch.zeros(total_query_len, topK, 1, rope_head_dim, dtype=test_dtype)
    # stage the caches on host once: the reference below gathers group by group
    k_cache_src = k_cache_xlite.cpu()
    pe_cache_src = pe_cache_xlite.cpu()
    bt = block_tables.tolist()  # plain list: the reference below hits it per token
    sparse_rows = 0
    dense_rows = 0
    row_valid = []  # per query-token valid slot count, for the set-semantics oracle
    row_flat_idx = []  # per query-token flat cache token index per slot, ditto
    in_offset = 0
    for b in range(batch):
        clen = cached_lens_list[b]
        qlen = query_lens_list[b]
        for q in range(qlen):
            num_candidates = (clen + q + 1) // compress_ratio
            valid_cnt = min(topK, num_candidates)
            if num_candidates <= topK:
                dense_rows += 1
                toks = None  # identity template: slot i -> token i
            else:
                sparse_rows += 1
                toks = topk_indices_tensor[in_offset].tolist()
            phys = [
                bt[b * max_num_blocks + (t if toks is None else toks[t]) // BLOCK_SIZE]
                for t in range(valid_cnt)
            ]
            off = [(t if toks is None else toks[t]) % BLOCK_SIZE for t in range(valid_cnt)]
            row_valid.append(valid_cnt)
            row_flat_idx.append([p * BLOCK_SIZE + o for p, o in zip(phys, off)])
            for chunk in range(0, valid_cnt, CNT_ONE_LOOP):
                cnt = min(CNT_ONE_LOOP, valid_cnt - chunk)
                for i, blk, src, n in gather_chunk_groups(phys, off, chunk, cnt, burst_cnt):
                    if channels != "pe":
                        k_dense_ref[in_offset, chunk + i : chunk + i + n] = k_cache_src[blk, src : src + n]
                    if channels != "k":
                        pe_dense_ref[in_offset, chunk + i : chunk + i + n] = pe_cache_src[blk, src : src + n]
            in_offset += 1

    torch.npu.synchronize()
    gather_sparse_kv_cache(
        rt,
        k_cache_xlite,
        pe_cache_xlite,
        block_tables,
        topk_indices_tensor,
        query_start_loc,
        query_lens,
        cached_lens,
        k_dense_xlite,
        pe_dense_xlite,
        batch,
        topK,
        BLOCK_SIZE,
        kv_lora_rank,
        rope_head_dim,
        kv_heads,
        compress_ratio,
        burst_copy_size,
    )
    torch.npu.synchronize()

    clens_repr = (
        str(cached_lens_list)
        if len(cached_lens_list) <= 8
        else f"[{', '.join(str(c) for c in cached_lens_list[:4])}, ... x{len(cached_lens_list)}]"
    )
    qlens_repr = (
        str(query_lens_list)
        if len(query_lens_list) <= 8
        else f"[{', '.join(str(q) for q in query_lens_list[:4])}, ... x{len(query_lens_list)}]"
    )
    logging.info(
        "gather_sparse_kv_cache %s (kv_lora_rank=%d, rope_head_dim=%d, %s) "
        "work (batch=%d, cached_lens=%s, query_lens=%s, total_q=%d, topK=%d, %s, "
        "ratio=%d, burst=%d, channels=%s) executed! (%d sparse, %d dense)",
        name,
        kv_lora_rank,
        rope_head_dim,
        test_dtype,
        batch,
        clens_repr,
        qlens_repr,
        total_query_len,
        topK,
        order,
        compress_ratio,
        burst_copy_size,
        channels,
        sparse_rows,
        dense_rows,
    )

    try:
        if channels != "pe":
            torch.testing.assert_close(k_dense_xlite.cpu(), k_dense_ref, atol=0, rtol=0)
        if channels != "k":
            torch.testing.assert_close(pe_dense_xlite.cpu(), pe_dense_ref, atol=0, rtol=0)
        # grouping-independent oracle: gathered slots must hold the expected
        # token rows as a byte multiset (catches bugs mirrored into the reference)
        k_out = k_dense_xlite.cpu() if channels != "pe" else None
        pe_out = pe_dense_xlite.cpu() if channels != "k" else None
        for in_offset in range(total_query_len):
            # index by (block, offset): row_flat_idx holds contiguous-layout flat
            # slots, which resolve to the same rows on a pitched cache view too
            flat = torch.tensor(row_flat_idx[in_offset], dtype=torch.int64)
            blk_idx, off_idx = flat // BLOCK_SIZE, flat % BLOCK_SIZE
            if k_out is not None:
                k_rows = k_cache_src[blk_idx, off_idx].reshape(-1, kv_lora_rank)
                assert_row_set(k_out[in_offset, : row_valid[in_offset]].reshape(-1, kv_lora_rank), k_rows)
            if pe_out is not None:
                pe_rows = pe_cache_src[blk_idx, off_idx].reshape(-1, rope_head_dim)
                assert_row_set(pe_out[in_offset, : row_valid[in_offset]].reshape(-1, rope_head_dim), pe_rows)
        logging.info(
            "gather_sparse_kv_cache case passed (exact match + set semantics; %d dense rows).",
            dense_rows,
        )
    except AssertionError as e:
        logging.error(f"gather_sparse_kv_cache mismatch: {e}")
        logging.error(f"k_dense_xlite: {k_dense_xlite.cpu()}")
        logging.error(f"k_dense_ref:   {k_dense_ref}")
        logging.error(f"pe_dense_xlite: {pe_dense_xlite.cpu()}")
        logging.error(f"pe_dense_ref:   {pe_dense_ref}")
        raise


def run_host_check_cases():
    """The op must reject a channel input/output emptiness mismatch (it would
    write through a null dense-cache pointer or leave the output unwritten)."""
    kv, pe, topK, clen = 512, 64, 2048, 8191
    mnb = max_blocks([clen], [1], BLOCK_SIZE)
    with torch.device("npu"):
        k_cache = torch.randn(mnb, BLOCK_SIZE, 1, kv, dtype=torch.bfloat16)
        pe_cache = torch.randn(mnb, BLOCK_SIZE, 1, pe, dtype=torch.bfloat16)
        block_tables = torch.arange(mnb, dtype=torch.int32)
        topk_indices = torch.randperm(clen)[:topK].to(torch.int32).unsqueeze(0)
        cached_lens = torch.tensor([clen], dtype=torch.int32)
        query_lens = torch.tensor([1], dtype=torch.int32)
        query_start_loc = torch.tensor([0], dtype=torch.int32)
        k_dense = torch.zeros(1, topK, 1, kv, dtype=torch.bfloat16)

    def expect_rejected(desc, *args):
        try:
            gather_sparse_kv_cache(rt, *args)
        except RuntimeError as e:
            logging.info("host check rejected %s as expected: %s", desc, e)
            return
        raise AssertionError(f"host check accepted invalid input: {desc}")

    expect_rejected(
        "channel mismatch", k_cache, pe_cache, block_tables, topk_indices, query_start_loc, query_lens,
        cached_lens, k_dense, torch.zeros(0, dtype=torch.bfloat16), 1, topK, BLOCK_SIZE, kv, pe, 1, 1, 16 * 1024,
    )


def main():
    name, kv_lora_rank, rope_head_dim, test_dtype = models[0]
    # primary matrix: glm5 shapes, both channels, every order and burst size
    work_cases = [
        (compress_ratio, batch, cached_lens, query_lens)
        for compress_ratio, cases in ((1, work), (2, work_ratio2))
        for batch, cached_lens, query_lens in cases
    ]
    assert all(
        len(cached_lens) == batch and len(query_lens) == batch
        for _, batch, cached_lens, query_lens in work_cases
    )
    for (compress_ratio, batch, cached_lens, query_lens), topK, order, burst_copy_size in itertools.product(
        work_cases, topk_values, orders, burst_sizes
    ):
        run_test(
            name,
            kv_lora_rank,
            rope_head_dim,
            test_dtype,
            batch,
            cached_lens,
            query_lens,
            topK,
            order,
            compress_ratio,
            burst_copy_size,
        )

    # deterministic grouping edges (topK=2048 only, both channels)
    for burst_copy_size in burst_sizes:
        run_test(
            name,
            kv_lora_rank,
            rope_head_dim,
            test_dtype,
            *work_pattern,
            2048,
            "pattern",
            1,
            burst_copy_size,
            "both",
            True,
        )

    # topK not a multiple of 8 (misaligned topkIndices GM source) or of 32
    # (partial tail chunk; cached_len=99 is dense at topK=100); topK=16/1 edge
    for topK in (100, 16, 1):
        for order, burst_copy_size in itertools.product(orders, (512, 8192)):
            run_test(
                name,
                kv_lora_rank,
                rope_head_dim,
                test_dtype,
                3,
                [99, 8191, 45841],
                [1, 1, 1],
                topK,
                order,
                1,
                burst_copy_size,
            )

    # fragmented cache: random logical -> physical block mapping
    for order in ("asc", "random"):
        run_test(
            name,
            kv_lora_rank,
            rope_head_dim,
            test_dtype,
            3,
            [2047, 8191, 45841],
            [1, 1, 1],
            2048,
            order,
            1,
            8192,
            permuted_blocks=True,
        )

    # swapped channels: kv_lora_rank < rope_head_dim exercises the host swap
    for order, burst_copy_size in itertools.product(orders, burst_sizes):
        run_test(
            "swapped",
            64,
            512,
            torch.bfloat16,
            2,
            [8191, 2047],
            [1, 1],
            2048,
            order,
            1,
            burst_copy_size,
        )

    # nope variants: single-channel gathers (empty tensor -> null channel)
    for channels, compress_ratio, batch, cached_lens, query_lens in (
        ("k", 1, 2, [8191, 2047], [1, 1]),
        ("pe", 1, 2, [8191, 2047], [1, 1]),
        ("k", 2, 2, [4096, 4097], [1, 1]),  # CXA-style compressed single-channel gather
    ):
        for order, burst_copy_size in itertools.product(("asc", "random"), (512, 8192)):
            run_test(
                name,
                kv_lora_rank,
                rope_head_dim,
                test_dtype,
                batch,
                cached_lens,
                query_lens,
                2048,
                order,
                compress_ratio,
                burst_copy_size,
                channels,
            )

    # cachedLens window reload: a batch one row past the nope-k window capacity
    # ((UB_SIZE - 4*(4096 + 128))/4) forces a mid-run GM reload; all rows share
    # one physical block to keep the oversized batch cheap
    ub_size = 196608  # 192KB, kernel_macro.h (NpuArch 2201)
    nope_per_buf = 4096 + 128  # kHeadBytesPad (32 slots * 64 * bf16) + topkBytesPad
    reload_batch = (ub_size - 4 * nope_per_buf) // 4 + 1
    assert reload_batch == 44929  # re-derive if the kernel layout changes
    run_test(
        "swapped",
        64,
        512,
        torch.bfloat16,
        reload_batch,
        [63] * reload_batch,
        [1] * reload_batch,
        128,
        "asc",
        1,
        8192,
        "k",
        shared_single_block=True,
    )

    # pitched caches: dim-0 stride double the contiguous pitch (buf[::2] views);
    # kernels must address blocks via the passed stride — mixed sparse/dense rows,
    # every order, the swapped-channel and single-channel (nope) variants
    for order, channels, swapped in (
        ("asc", "both", False),
        ("random", "both", False),
        ("desc", "pe", False),
        ("random", "k", True),
    ):
        run_test(
            "swapped" if swapped else name,
            64 if swapped else kv_lora_rank,
            512 if swapped else rope_head_dim,
            test_dtype,
            3,
            [2047, 8191, 45841],
            [1, 1, 1],
            2048,
            order,
            1,
            8192,
            channels,
            pitched=True,
        )

    run_host_check_cases()


if __name__ == "__main__":
    main()
