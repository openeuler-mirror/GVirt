#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Candidate two-level top-k level 1 (V4.1): the candidate-source layer pools
indexer scores into groups of `candidate_block_size` KV rows, pins the causal
boundary group, and emits the top `candidate_topk_blocks` group ids per query
row ([-1, ..., -1, ascending ids]); consumer layers degrade every
non-candidate position's score to NEG_HUGE (-1e30) ahead of their own
position top-k. Oracle semantics follow select_candidate_blocks from
vllm-ascend's DeepSeek V4.1 reference implementation.

Runs on the C8 (int8) instance — the production V4.1 indexer path — so the
scores follow the C8 formula from tests/kernels/indexer_c8.py: per-head
relu(int8 dot)/1024 rounded to fp16, weighted per head, summed, scaled per
KV row by k_scale_cache. The candidate source pools those post-staging fp32
scores, i.e. after the k-scale multiply.

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_indexer_topk_candidate.py``.
"""

import logging

import pytest
import torch
import torch_npu  # noqa: F401

from xlite._C import indexer_topk

logging.getLogger().setLevel(logging.INFO)

BLOCK_SIZE = 128
N_HEADS, HEAD_DIM = 64, 128
TOPK = 512

# production V4.1 config: indexTopK=512, candidate block 8, top blocks 2048
# (= MAX_TOPK_NUM, the emit's fixed-size totalSort borrow width)
CAND_TOPK_BLOCKS = 2048
CAND_BLOCK_SIZE = 8
NEG_HUGE = -1e30

K_CACHE_FILL = -101  # int8 fill of the unseen pool rows
SCALE_CACHE_FILL = -17.0  # fp16 fill of the unseen pool scales

# Active rows (gTotal > W) rank W real groups; pooled maxima are ulp-stable per
# group but adjacent groups near the W-cutoff can flip between the NPU head
# sum and the host einsum reference (the same pairing the indexer_c8 test
# tolerates through its score-boundary gap check).
MEMBERSHIP_TOLERANCE = 4  # groups per row
DIFF_PCT_THRESHOLD = 3.0  # percent of topk indices per row

# name, batch, cached_lens, query_lens. One shape per regime:
#   mixed_small        -- dense rows (emit template) + passive sparse rows
#   boundary           -- gTotal == W exactly (zero -1 pads) straddling active
#   sparse_passive     -- 3-KV-tile relay chain in the keep-everything regime
#   active_cross_chunks-- real selection across 5 KV tiles (gTotal up to 2423)
work = [
    ("mixed_small", 1, [0], [800]),
    ("boundary", 1, [16376], [16]),  # p0=16376..16391: gTotal 2048, 2048, 2049..
    ("sparse_passive", 1, [9000], [1000]),
    ("active_cross_chunks", 1, [16384], [3000]),
    # chunked-prefill serving shape: a mid-chunk continuation (cached 16360 +
    # 64-token chunk) batched with an unaligned decode request (cached 5003 +
    # 1 token). Mid-chunk rows keep the strict per-row causal clip [0, p0]
    # the serving path always applies (mask_mode=3), and the qlen crosses
    # the W*candidateBlockSize threshold mid-qlen: rows 0..23 passive (incl.
    # gTotal == W, zero -1 pads), rows 24..63 active. batch > 1 exercises the
    # per-sample block-table slots and tokenRowBase row mapping, and the
    # mixed query lens drive the cube's prefill-tile and decode-task query
    # shapes in one launch.
    ("chunked_mixed_batch", 2, [16360, 5003], [64, 1]),
    # realistic mixed serving batch, one request per regime: a fresh
    # first-chunk prompt (dense rows), an odd-aligned mid-chunk continuation
    # (137 % 8 == 1: partial last group), two decode tokens -- one passive,
    # one active (gTotal 2376: real selection on a decode row at batch > 1)
    # -- the threshold-crossing continuation and a long active prefill. One
    # launch spans dense/sparse/passive/active rows, VEC-written dense
    # candidate templates at nonzero row offsets, and both query-task shapes
    # (tiles never cross requests, so each decode request is its own
    # queryTaskLen == 1 task).
    ("multi_len_batch", 6, [0, 137, 5003, 19000, 16360, 20000], [40, 17, 1, 1, 64, 100]),
    # continuous-batching decode step: 16 requests, one token each, sweeping
    # the per-row state boundaries as 16 separate decode tasks --
    # p0=511 (last dense row) vs 512 (first sparse), 4095 (odd group tail:
    # last group holds one valid position) vs 4096 (aligned), 16383 (gTotal
    # == W, zero -1 pads) vs 16384 (first active), 19000 (2376 groups).
    (
        "decode_heavy_batch",
        16,
        [0, 3, 137, 511, 512, 513, 1000, 2000, 4095, 4096, 8191, 9000, 12000, 16383, 16384, 19000],
        [1] * 16,
    ),
    # smallest prefill tiles: request A is the sharpest regime pair (row 0 the
    # last zero-pad passive row at gTotal == W, row 1 the first active row),
    # request B an all-active 3-row tile. qlen 2/3 drive the prefill tile at
    # the minimum query count (K-reuse across just 2-3 queries) -- every
    # earlier prefill fixture had qlen >= 17.
    ("tiny_prefill", 2, [16383, 20000], [2, 3]),
    # exact qTile fits at the default tile size: 24 = one full tile, 48 = two
    # (passive and active regimes); no earlier fixture lands on a tile
    # boundary.
    ("tile_exact_fit", 2, [9000, 20000], [24, 48]),
    # the W-crossing stretched across ~13 tiles (chunked_mixed_batch's
    # 64-token chunk spans only 3): rows 0..83 passive, the crossing deep
    # inside tile 3, then a 216-row active tail into the final chunk.
    ("crossing_deep_prefill", 1, [16300], [300]),
    # two active decode tasks in one launch (multi_len_batch carries only one).
    ("dual_active_decode", 2, [17000, 25000], [1, 1]),
    # deepest context in the suite: a 25-chunk relay chain, gTotal 12501 --
    # stresses the mode-1 serial relay merge and the mode-2 mask far beyond
    # active_cross_chunks' 5 chunks.
    ("deep_context_decode", 1, [100000], [1]),
]

# mode 1 always relays serially: the op forces kvSplit=0 for source mode (the
# cross-core sync-flag ring isn't extended to a second relay payload), so the
# kv_split=1 point merely re-covers (0, 0), pinning that force. The sweep
# varies the host q-tile tiers (tokenRowBase mapping).
MODE1_SWEEP = ((0, 0), (0, 1), (0, 17), (1, 0))
# mode 2 is chunk-local, so it must hold under both partitioning modes and
# under host tiling (row-indexed candidate reads)
MODE2_SWEEP = ((0, 0), (0, 17), (1, 0), (1, 17))

# integration on the fixtures whose emit output feeds a nontrivial consume:
# the multi-request batches, the cross-chunk selection shape, the sharpest
# regime pair, the deep W-crossing, and the longest relay chain
INTEGRATION_WORK = [work[i] for i in (3, 4, 5, 6, 7, 9, 11)]


def max_blocks(query_lens, cached_lens, block_size):
    max_sum = max(a + b for a, b in zip(query_lens, cached_lens))
    return (max_sum + block_size - 1) // block_size


def shape_seed(batch, cached_lens_list, query_len_list):
    """Stable per-shape seed so every runner (mode 1, mode 2, integration)
    rebuilds the identical fixture."""
    digits = [batch] + list(cached_lens_list) + list(query_len_list)
    return int("".join(str(d) for d in digits)) % (2**31)


def build_inputs(batch, cached_lens_list, query_len_list):
    """C8-format inputs (int8 q/k pools, fp16 k-scale, fp16 per-head weights)
    plus a host fp32 [T, max_seq_len] score reference. Row r's causal span is
    [0, p0(r)] with p0 = clen + in-block offset; blocks of sample i live at
    block-table slots [i*max_num_blocks, (i+1)*max_num_blocks) so sample rows
    map to contiguous pool rows i*max_num_blocks*BLOCK_SIZE + r."""
    max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE)
    max_seq_len = max_num_blocks * BLOCK_SIZE
    total_query_len = sum(query_len_list)
    num_cache_blocks = max_num_blocks * batch

    # host-side fixture data: explicit cpu context so the module also runs
    # under an ambient npu default (the suite's default_device fixture)
    with torch.device("cpu"):
        gen = torch.Generator().manual_seed(shape_seed(batch, cached_lens_list, query_len_list))
        q8 = torch.randint(-127, 128, (total_query_len, N_HEADS, HEAD_DIM), generator=gen).char()
        scaled_w = (torch.randn(total_query_len, N_HEADS, generator=gen) * 0.1).half()
        k8_list, k_scale_list = [], []
        for clen, qlen in zip(cached_lens_list, query_len_list):
            span = clen + qlen
            k8_list.append(torch.randint(-127, 128, (span, HEAD_DIM), generator=gen).char())
            k_scale_list.append((torch.rand(span, generator=gen) * 0.1).half())

    with torch.device("npu"):
        q = q8.reshape(total_query_len, N_HEADS * HEAD_DIM).npu()
        weight = scaled_w.npu()
        k_cache = torch.full((num_cache_blocks, BLOCK_SIZE, 1, HEAD_DIM), K_CACHE_FILL, dtype=torch.int8)
        k_scale_cache = torch.full((num_cache_blocks, BLOCK_SIZE, 1, 1), SCALE_CACHE_FILL, dtype=torch.float16)
        query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_start_loc = (query_lens.cumsum(0) - query_lens).int()
        # persistent buffer with stale trails (the op narrows to the
        # lens-derived width, pitch unchanged)
        valid = torch.arange(batch)[:, None] * max_num_blocks + torch.arange(max_num_blocks)
        padded = torch.zeros(batch, max_num_blocks + 3, dtype=torch.int32)
        padded[:, :max_num_blocks] = valid
        block_tables = padded
        topk_indices = torch.full((total_query_len, TOPK), -111, dtype=torch.int32)
        candidates = torch.full((total_query_len, CAND_TOPK_BLOCKS), -111, dtype=torch.int32)
        candidate_relay = torch.full((total_query_len, 2 * CAND_TOPK_BLOCKS), -111, dtype=torch.int32)

    # scatter per-sample KV rows and scales into the contiguous pool
    k_flat = k_cache.view(-1, HEAD_DIM)
    ks_flat = k_scale_cache.view(-1)
    for i in range(batch):
        base = i * max_num_blocks * BLOCK_SIZE
        span = cached_lens_list[i] + query_len_list[i]
        k_flat[base : base + span] = k8_list[i]
        ks_flat[base : base + span] = k_scale_list[i]

    # host score reference [T, max_seq_len]: C8 math per head —
    # fp16(relu(int8 dot)/1024) * weight, summed over heads, * per-row k scale
    with torch.device("cpu"):
        ref_scores = torch.full((total_query_len, max_seq_len), float("-inf"))
        offset = 0
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]
            span = clen + qlen
            q_chunk = q8[offset : offset + qlen]
            w_chunk = scaled_w[offset : offset + qlen].float()
            k_slice = k8_list[i]
            ks_slice = k_scale_list[i].float()
            chunk = 128
            for s0 in range(0, qlen, chunk):
                s1 = min(s0 + chunk, qlen)
                dot = q_chunk[s0:s1].reshape(-1, HEAD_DIM).int() @ k_slice.int().T
                dot = dot.reshape(s1 - s0, N_HEADS, span).clamp_min(0)
                per_head = (dot.float() / 1024).half().float() * w_chunk[s0:s1][..., None]
                ref_scores[offset + s0 : offset + s1, :span] = per_head.sum(1) * ks_slice[None, :]
            offset += qlen

        p0_all = torch.tensor(
            [cl + q for cl, ql in zip(cached_lens_list, query_len_list) for q in range(ql)],
            dtype=torch.int64,
        )
    return {
        "q": q,
        "k_cache": k_cache,
        "k_scale_cache": k_scale_cache,
        "weight": weight,
        "query_start_loc": query_start_loc,
        "query_lens": query_lens,
        "cached_lens": cached_lens,
        "block_tables": block_tables,
        "topk_indices": topk_indices,
        "candidates": candidates,
        "candidate_relay": candidate_relay,
        "ref_scores": ref_scores,
        "p0_all": p0_all,
        "batch": batch,
    }


def candidate_row_oracle(ref_row, p0):
    """Expected source-mode candidate row: top-W group ids ascending, -1 padded
    to W. Passive rows (num_blocks <= W) keep every group; the reference's
    `values > -inf` keep-filter drops only the padded tail group columns."""
    with torch.device("cpu"):
        width = p0 + 1
        logits = ref_row[:width]
        scores = torch.nn.functional.pad(logits, (0, -width % CAND_BLOCK_SIZE), value=float("-inf"))
        scores = scores.unflatten(-1, (-1, CAND_BLOCK_SIZE)).amax(dim=-1)
        num_blocks = scores.numel()
        last = p0 // CAND_BLOCK_SIZE
        scores = scores.masked_fill(torch.arange(num_blocks) == last, float("inf"))
        k = min(CAND_TOPK_BLOCKS, num_blocks)
        top = torch.topk(scores, k)
        keep = top.values > float("-inf")
        ids = top.indices[keep]
        row = torch.full((CAND_TOPK_BLOCKS,), -1, dtype=torch.int64)
        row[CAND_TOPK_BLOCKS - ids.numel() :] = ids.sort().values
        return row


def degraded_topk_oracle(ref_row, p0, cand_row):
    """Expected consumer-mode position topk: every position outside the kept
    candidate blocks drops to NEG_HUGE before the top-K."""
    with torch.device("cpu"):
        width = p0 + 1
        degraded = ref_row[:width].clone()
        nblocks = (width + CAND_BLOCK_SIZE - 1) // CAND_BLOCK_SIZE
        keep_blocks = torch.zeros(nblocks, dtype=torch.bool)
        real = cand_row[cand_row >= 0]
        keep_blocks[real[real < nblocks]] = True
        keep_pos = keep_blocks.repeat_interleave(CAND_BLOCK_SIZE)[:width]
        degraded[~keep_pos] = NEG_HUGE
    k = min(TOPK, width)
    return torch.topk(degraded, k).indices.to(torch.int64)


def compare_topk_sets(xlite_cpu, oracle_idx, p0_all, tag, extra=""):
    """Per-row comparison. Dense rows (p0 < topK): exact identity template.
    Sparse rows: causal sanity + order-tolerant set comparison vs the oracle."""
    with torch.device("cpu"):
        total = xlite_cpu.shape[0]
        is_dense = p0_all < TOPK
        row_default = torch.arange(0, TOPK, dtype=torch.int32)
        worst = 0.0
        failures = []
        for r in range(total):
            got = xlite_cpu[r].to(torch.int64)
            if bool(is_dense[r]):
                ok = bool((xlite_cpu[r] == row_default).all())
                pct = 0.0 if ok else 100.0
            else:
                p0 = int(p0_all[r])
                if not bool((got <= p0).all()):
                    pct = 100.0  # non-causal indices: always a failure
                else:
                    exp = oracle_idx[r]
                    got_sorted, _ = torch.sort(got)
                    pos = torch.searchsorted(got_sorted, exp).clamp_max(got.numel() - 1)
                    hits = got_sorted.gather(0, pos) == exp
                    pct = float(100.0 * (exp.numel() - hits.sum()) / max(1, exp.numel()))
            if pct > DIFF_PCT_THRESHOLD:
                failures.append((r, pct))
            worst = max(worst, pct)
        if failures:
            logging.error(
                "\t%s %s: %d rows over %.1f%%: %s", tag, extra, len(failures), DIFF_PCT_THRESHOLD, failures[:8]
            )
            raise AssertionError(f"{tag} {extra}: topk mismatch on {len(failures)} rows")
        logging.info("\t%s %s: all %d rows pass (max diff pct %.2f)", tag, extra, total, worst)


def check_candidate_rows(cand_cpu, oracle_rows, p0_all, tag, extra=""):
    """Source-mode candidate rows. Passive rows (gTotal <= W, including all
    dense rows): exact match required -- selection is all-groups, immune to
    score ulp noise. Active rows: structural checks (all W slots real,
    strictly ascending) plus membership tolerance at the W-cutoff."""
    total = cand_cpu.shape[0]
    bad = []
    for r in range(total):
        got = cand_cpu[r].to(torch.int64)
        exp = oracle_rows[r]
        g_total = int(p0_all[r]) // CAND_BLOCK_SIZE + 1
        if g_total <= CAND_TOPK_BLOCKS:
            if not bool((got == exp).all()):
                bad.append((r, "exact-mismatch"))
            continue
        # active row: exactly W real ids, ascending (emit sort contract)
        if bool((got < 0).any()) or not bool((got.diff() > 0).all()):
            bad.append((r, "structure"))
            continue
        exp_set = set(exp[exp >= 0].tolist())
        got_set = set(got.tolist())
        if len(exp_set - got_set) > MEMBERSHIP_TOLERANCE or len(got_set - exp_set) > (MEMBERSHIP_TOLERANCE):
            bad.append((r, f"cutoff-diff {len(exp_set ^ got_set)}"))
    if bad:
        raise AssertionError(f"{tag} {extra}: candidate rows mismatch: {bad[:8]}")
    logging.info("\t%s %s: all %d candidate rows pass", tag, extra, total)


def call_indexer_topk(b, rt, **kwargs):
    """Common C8 call: positional slots plus the candidate/partition kwargs."""
    args = dict(
        rt=rt,
        q=b["q"],
        k_cache=b["k_cache"],
        weight=b["weight"],
        topk_indices=b["topk_indices"],
        query_start_loc=b["query_start_loc"],
        lens=b["query_lens"],
        cached_lens=b["cached_lens"],
        block_tables=b["block_tables"],
        n_heads=N_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        batch=int(b["batch"]),
        top_k=TOPK,
        k_scale_cache=b["k_scale_cache"],
        compress_ratio=1,
    )
    args.update(kwargs)
    return indexer_topk(**args)


def run_mode1(rt, name, b):
    total = int(b["query_lens"].sum())
    p0_all = b["p0_all"]
    oracle_rows = [candidate_row_oracle(b["ref_scores"][r], int(p0_all[r])) for r in range(total)]
    oracle_pos = [
        torch.topk(b["ref_scores"][r, : p0_all[r] + 1], min(TOPK, int(p0_all[r]) + 1)).indices.to(torch.int64)
        for r in range(total)
    ]
    for kv_split, q_tile in MODE1_SWEEP:
        b["candidates"].fill_(-2)  # poison: unwritten slots must not survive
        b["topk_indices"].fill_(-1)
        torch.npu.synchronize()
        call_indexer_topk(
            b,
            rt,
            candidates=b["candidates"],
            candidate_relay=b["candidate_relay"],
            candidate_mode=1,
            candidate_topk_blocks=CAND_TOPK_BLOCKS,
            candidate_block_size=CAND_BLOCK_SIZE,
            kv_split=kv_split,
            q_tile=q_tile,
        )
        torch.npu.synchronize()
        check_candidate_rows(
            b["candidates"].cpu(),
            oracle_rows,
            p0_all,
            "mode1-candidates",
            f"kv_split={kv_split} q_tile={q_tile} {name}",
        )
        compare_topk_sets(
            b["topk_indices"].cpu(), oracle_pos, p0_all, "mode1-topk", f"kv_split={kv_split} q_tile={q_tile} {name}"
        )
    logging.info("mode1 %s: pass", name)


def run_mode2_with(rt, b, cand_npu, cand_cpu_i64, name):
    """Consumer mode on a prebuilt fixture, feeding cand_npu as the candidate
    rows (standalone: host oracle rows; integration: mode 1's NPU output)."""
    total = int(b["query_lens"].sum())
    p0_all = b["p0_all"]
    oracle_pos = [degraded_topk_oracle(b["ref_scores"][r], int(p0_all[r]), cand_cpu_i64[r]) for r in range(total)]
    for kv_split, q_tile in MODE2_SWEEP:
        b["topk_indices"].fill_(-1)
        torch.npu.synchronize()
        call_indexer_topk(
            b,
            rt,
            candidates=cand_npu,
            candidate_mode=2,
            candidate_topk_blocks=CAND_TOPK_BLOCKS,
            candidate_block_size=CAND_BLOCK_SIZE,
            kv_split=kv_split,
            q_tile=q_tile,
        )
        torch.npu.synchronize()
        compare_topk_sets(
            b["topk_indices"].cpu(),
            oracle_pos,
            p0_all,
            "mode2-topk",
            f"kv_split={kv_split} q_tile={q_tile} {name}",
        )
    logging.info("mode2 %s: pass", name)


def run_mode2(rt, name, b):
    total = int(b["query_lens"].sum())
    p0_all = b["p0_all"]
    oracle_rows = torch.stack([candidate_row_oracle(b["ref_scores"][r], int(p0_all[r])) for r in range(total)])
    b["candidates"].copy_(oracle_rows.to(torch.int32).npu())
    torch.npu.synchronize()
    run_mode2_with(rt, b, b["candidates"], oracle_rows, name)


def run_mode2_fullchunk_degrade(rt):
    """Force the zero-candidate-chunk branch: candidate rows cover only groups
    [0, 1600), so the KV chunk holding groups 2048+ has no candidates at all
    (full-chunk NEG_HUGE degrade) and the chunk at 1536..2047 straddles the
    boundary (partial intersection)."""
    batch, clen, qlen = 1, 20000, 64
    total = qlen
    b = build_inputs(batch, [clen], [qlen])
    with torch.device("cpu"):
        first = torch.arange(0, 1600, dtype=torch.int64)
        keep = torch.full((CAND_TOPK_BLOCKS,), -1, dtype=torch.int64)
        keep[CAND_TOPK_BLOCKS - first.numel() :] = first
        cand_rows = keep.unsqueeze(0).expand(total, -1).contiguous()
    b["candidates"].copy_(cand_rows.to(torch.int32).npu())
    torch.npu.synchronize()
    run_mode2_with(rt, b, b["candidates"], cand_rows, "fullchunk-degrade")


def run_mode2_decode(rt):
    """queryLen == 1 (decode): the score-all cube scores every KV row of the
    single query task (the cube is candidate-unaware), then the VEC's
    RunCandidateMask degrades every non-candidate position to NEG_HUGE ahead
    of the position topk, so the result must match the oracle's degraded
    topk. clen large enough that gTotal (2501) > W (2048) => real selection."""
    batch, clen, qlen = 1, 20000, 1
    b = build_inputs(batch, [clen], [qlen])
    total = qlen
    p0_all = b["p0_all"]
    oracle_rows = torch.stack([candidate_row_oracle(b["ref_scores"][r], int(p0_all[r])) for r in range(total)])
    b["candidates"].copy_(oracle_rows.to(torch.int32).npu())
    torch.npu.synchronize()
    run_mode2_with(rt, b, b["candidates"], oracle_rows, "decode")


def run_mode2_prefill(rt):
    """queryLen > 1 (prefill): the score-all cube keeps its K-reuse across
    the query tile (candidate-unaware), and the VEC's RunCandidateMask
    degrades every non-candidate position ahead of the position topk. The
    pool is two bands of groups -- [0, 1024) and [2100, 3124), band 2
    starting mid-block -- shared by all 16 queries on a 40000-row cache, so
    the gap chunks between the bands hit the mask's whole-chunk degrade
    branch while band 2's edge blocks straddle it partially. Real candidate
    positions number 2048*8 > TOPK, so no NEG_HUGE fillers enter the
    comparison."""
    batch, clen, qlen = 1, 40000, 16
    total = qlen
    b = build_inputs(batch, [clen], [qlen])
    with torch.device("cpu"):
        half = CAND_TOPK_BLOCKS // 2
        pool = torch.cat(
            [
                torch.arange(0, half, dtype=torch.int64),
                torch.arange(2 * half + 52, 3 * half + 52, dtype=torch.int64),
            ]
        )
        assert pool.numel() == CAND_TOPK_BLOCKS and bool((pool.diff() > 0).all())
        cand_rows = pool.unsqueeze(0).expand(total, -1).contiguous()
    b["candidates"].copy_(cand_rows.to(torch.int32).npu())
    torch.npu.synchronize()
    run_mode2_with(rt, b, b["candidates"], cand_rows, "prefill-twoband")


def run_integration(rt, name, b):
    """mode 1's NPU-emitted candidates feed mode 2 directly (no host round
    trip): validates the emit contract the consumer's binary search needs,
    on the same fixture both stages ran against."""
    b["candidates"].fill_(-2)
    torch.npu.synchronize()
    call_indexer_topk(
        b,
        rt,
        candidates=b["candidates"],
        candidate_relay=b["candidate_relay"],
        candidate_mode=1,
        candidate_topk_blocks=CAND_TOPK_BLOCKS,
        candidate_block_size=CAND_BLOCK_SIZE,
    )
    torch.npu.synchronize()
    cand_cpu = b["candidates"].cpu().to(torch.int64)
    run_mode2_with(rt, b, b["candidates"], cand_cpu, name + "-integrated")


def run_validation_errors(rt):
    """op-level config gates fire before any launch."""
    batch, clen, qlen = 1, 21000, 8  # span > W blocks: real candidate regime
    b = build_inputs(batch, [clen], [qlen])
    narrow_candidates = torch.empty(qlen, CAND_TOPK_BLOCKS // 2, dtype=torch.int32).npu()
    wide_candidates = torch.empty(qlen, CAND_TOPK_BLOCKS * 2, dtype=torch.int32).npu()
    wide_relay = torch.empty(qlen, 4 * CAND_TOPK_BLOCKS, dtype=torch.int32).npu()
    flat_candidates = torch.empty(CAND_TOPK_BLOCKS, dtype=torch.int32).npu()
    short_candidates = torch.empty(qlen - 1, CAND_TOPK_BLOCKS, dtype=torch.int32).npu()
    short_relay = torch.empty(qlen - 1, 2 * CAND_TOPK_BLOCKS, dtype=torch.int32).npu()

    def call(topk=TOPK, cand="default", relay_t="default", mode=1, w=CAND_TOPK_BLOCKS, bs=CAND_BLOCK_SIZE):
        call_indexer_topk(
            b,
            rt,
            top_k=topk,
            candidates=b["candidates"] if cand == "default" else cand,
            candidate_relay=b["candidate_relay"] if relay_t == "default" else relay_t,
            candidate_mode=mode,
            candidate_topk_blocks=w,
            candidate_block_size=bs,
        )

    cases = [
        ("mode 2 without candidates", dict(mode=2, cand=None), "candidate mode requires"),
        ("unknown mode 3", dict(mode=3), "candidate mode requires"),
        ("rank-1 candidates", dict(cand="flat"), "candidate mode requires"),
        ("narrow candidates", dict(cand="narrow"), "at least candidateTopkBlocks wide"),
        ("candidates rows < query tokens", dict(cand="short"), "one row per query token"),
        ("relay rows < query tokens", dict(relay_t="short"), "one row per query token"),
        ("mode 1 without relay", dict(relay_t=None), "relay workspace"),
        (
            "emit scratch too small",
            dict(w=CAND_TOPK_BLOCKS * 2, cand="wide", relay_t="wide"),
            "candidateTopkBlocks <= MAX_TOPK_NUM",
        ),
        (
            "emit width not multiple of 8",
            dict(w=CAND_TOPK_BLOCKS - 4, cand="wide", relay_t="wide"),
            "multiple of 8",
        ),
        ("bad candidate block size", dict(bs=4), "candidateBlockSize 8"),
        ("mode 2 bad block size", dict(mode=2, bs=4), "candidateBlockSize 8"),
    ]
    for label, kwargs, needle in cases:
        if kwargs.get("cand") == "narrow":
            kwargs["cand"] = narrow_candidates
        elif kwargs.get("cand") == "wide":
            kwargs["cand"] = wide_candidates
        elif kwargs.get("cand") == "flat":
            kwargs["cand"] = flat_candidates
        elif kwargs.get("cand") == "short":
            kwargs["cand"] = short_candidates
        if kwargs.get("relay_t") == "wide":
            kwargs["relay_t"] = wide_relay
        elif kwargs.get("relay_t") == "short":
            kwargs["relay_t"] = short_relay
        try:
            call(**kwargs)
        except (RuntimeError, ValueError) as e:
            assert needle in str(e), f"{label}: unexpected error: {e}"
            logging.info("\trejected %s: %s", label, str(e).splitlines()[0][:110])
        else:
            raise AssertionError(f"{label} was not rejected")

    # all-dense cache (topK >= capacity): mode 1 rejected, host template taken
    with torch.device("npu"):
        q_d = torch.randint(-127, 128, (28, N_HEADS * HEAD_DIM)).char()
        k_d = torch.full((1, BLOCK_SIZE, 1, HEAD_DIM), K_CACHE_FILL, dtype=torch.int8)
        k_scale_d = torch.full((1, BLOCK_SIZE, 1, 1), SCALE_CACHE_FILL, dtype=torch.float16)
        w_d = (torch.randn(28, N_HEADS) * 0.1).half()
        idx_d = torch.empty(28, TOPK, dtype=torch.int32)
        cand_d = torch.empty(28, CAND_TOPK_BLOCKS, dtype=torch.int32)
        relay_d = torch.empty(28, 2 * CAND_TOPK_BLOCKS, dtype=torch.int32)
    # metadata tensors stayed host in the legacy script; keep that exact
    # contract (the op's validation path accepted them there)
    with torch.device("cpu"):
        lens_d = torch.tensor([28], dtype=torch.int32)
        cached_d = torch.tensor([100], dtype=torch.int32)
        start_d = torch.tensor([0], dtype=torch.int32)
        tables_d = torch.zeros(1, 1, dtype=torch.int32)
    try:
        indexer_topk(
            rt,
            q_d,
            k_d,
            w_d,
            idx_d,
            start_d,
            lens_d,
            cached_d,
            tables_d,
            N_HEADS,
            HEAD_DIM,
            BLOCK_SIZE,
            1,
            TOPK,
            k_scale_cache=k_scale_d,
            compress_ratio=1,
            candidates=cand_d,
            candidate_relay=relay_d,
            candidate_mode=1,
            candidate_topk_blocks=CAND_TOPK_BLOCKS,
            candidate_block_size=CAND_BLOCK_SIZE,
        )
    except (RuntimeError, ValueError) as e:
        assert "incompatible with the dense-row" in str(e), f"dense cache: {e}"
        logging.info("\trejected dense cache: %s", str(e).splitlines()[0][:110])
    else:
        raise AssertionError("dense-cache mode 1 was not rejected")
    logging.info("candidate config validation: all invalid configs rejected")


@pytest.mark.parametrize("name,batch,cached_lens,query_lens", work, ids=[w[0] for w in work])
def test_mode1_candidate_source(runtime, name, batch, cached_lens, query_lens):
    b = build_inputs(batch, cached_lens, query_lens)
    run_mode1(runtime, name, b)


@pytest.mark.parametrize("name,batch,cached_lens,query_lens", work, ids=[w[0] for w in work])
def test_mode2_consumer(runtime, name, batch, cached_lens, query_lens):
    b = build_inputs(batch, cached_lens, query_lens)
    run_mode2(runtime, name, b)


def test_mode2_fullchunk_degrade(runtime):
    run_mode2_fullchunk_degrade(runtime)


def test_mode2_decode(runtime):
    run_mode2_decode(runtime)


def test_mode2_prefill(runtime):
    run_mode2_prefill(runtime)


@pytest.mark.parametrize("name,batch,cached_lens,query_lens", INTEGRATION_WORK, ids=[w[0] for w in INTEGRATION_WORK])
def test_integration_emit_feeds_consumer(runtime, name, batch, cached_lens, query_lens):
    b = build_inputs(batch, cached_lens, query_lens)
    run_integration(runtime, name, b)


def test_config_validation(runtime):
    run_validation_errors(runtime)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
