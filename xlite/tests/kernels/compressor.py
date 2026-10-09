#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY of even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Unit test for ``csrc/kernels/compressor.h`` (fused DeepSeek-V4 KV Compressor).

Covers prefill (overlap_transform + softmax-weighted-sum + RMSNorm + rotary +
compress_kv cache write) and decode (streaming state machine), against a
per-request torch reference mirroring ``Compressor.forward``.
"""
from __future__ import absolute_import

import logging

import torch

from xlite._C import Runtime, compressor

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

HEAD_DIM = 512
ROPE_HEAD_DIM = 64
NORM_EPS = 1e-6
ROPE_THETA = 160000.0
BLOCK_SIZE = 128
# state cache paged with its own block size; keep == BLOCK_SIZE here.
STATE_BLOCK_SIZE = 128
# bf16/fp16 round-trips through fp32 softmax/pool/norm/rope; the kernel's accumulation
# order differs from torch's, so results diverge at the BF16/FP16 ULP level
# (compress_kv max ~1.56e-2 = 1 BF16 ULP at |v|~3, spill state ~1e-3). Use the loose
# bounds for compress_kv/weightedSum (accumulated outputs); STRICT for state writes
# (kv/score+ape copies, which are bit-exact up to the dtype round).
STRICT_ATOL = 1e-5
STRICT_RTOL = 1e-3
ATOL = 1e-2
RTOL = 1e-2

# ---------------------------------------------------------------------------
# models × work cross-product (mirrors tests/kernels/attention.py):
#   `models` = the orthogonal model config (ratio, overlap, dtype) — 3 ratios
#   × 2 dtypes. `work` = the workload, each item tagged by category {prefill,
#   mix, decode} with a ratio-relative spec_builder so the SAME item runs under
#   r=2/4/128, plus an optional flags dict (rotate / norope / no-ape /
#   first-window). `__main__` is `for model in models: for work_item in work:`.
# ---------------------------------------------------------------------------

# (name, ratio, overlap, dtype). ratio2/128 overlap=0; ratio4 overlap=1.
models = [
    ("ratio2", 2, 0, torch.bfloat16),
    ("ratio2", 2, 0, torch.float16),
    ("ratio4_overlap", 4, 1, torch.bfloat16),
    ("ratio4_overlap", 4, 1, torch.float16),
    ("ratio128", 128, 0, torch.bfloat16),
    ("ratio128", 128, 0, torch.float16),
]

# (category, suffix, spec_builder, flags).
#   spec_builder: (ratio) -> list[(qlen, cached)]. run_batch classifies each
#   returned (qlen, cached) pair: qlen>1,cached==0 -> fresh prefill;
#   qlen>1,cached>0 -> single-launch continuation chunk (front window from
#   state, tail from gmKv); qlen==1,cached>0 -> decode compress (cached=decode_pos).
#   decode_pos uses ratio*k-1 so (pos+1)%ratio==0 (compress with history).
work = [
    # ── PREFILL (single launch) ──
    ("prefill", "single_block", lambda r: [(r, 0)], {}),
    ("prefill", "block_spill", lambda r: [(r + 1, 0)], {}),
    ("prefill", "two_blocks", lambda r: [(2 * r, 0)], {}),
    ("prefill", "mid8", lambda r: [(8 * r, 0)], {}),
    ("prefill", "long16", lambda r: [(16 * r, 0)], {}),
    # longer block-count progressions (ratio-relative: N blocks for EVERY ratio;
    # ratio128 -> N*128 tokens, stressing the tiled col-slice path).
    ("prefill", "long32", lambda r: [(32 * r, 0)], {}),
    ("prefill", "long64", lambda r: [(64 * r, 0)], {}),
    # absolute-token long sequences (ratio-INdependent: N tokens for every ratio;
    # gives genuine long-token coverage for small ratios -- ratio2/ratio4 get many
    # blocks, ratio128 stays modest). qlen>=ratio holds for all models (max r=128).
    ("prefill", "tok1024", lambda r: [(1024, 0)], {}),
    ("prefill", "tok2048", lambda r: [(2048, 0)], {}),
    ("prefill", "tok4096", lambda r: [(4096, 0)], {}),
    ("prefill", "tok8192", lambda r: [(8192, 0)], {}),
    # short prefill segments (qlen<ratio): nTotalBlocks=0 path — only spill to state,
    # no compress block. ratio128: 64/96<128 -> nTotalBlocks=0 (情形 B: chunked-prefill
    # tail / short prompt). ratio2/4: 64/96>=ratio -> normal prefill (still valid, not
    # the nTotalBlocks=0 case but exercises the same code path).
    ("prefill", "tok64", lambda r: [(64, 0)], {}),
    ("prefill", "tok96", lambda r: [(96, 0)], {}),
    ("prefill", "remainder3", lambda r: [(8 * r + 3, 0)], {}),
    ("prefill", "batch2_div", lambda r: [(8 * r, 0), (4 * r, 0)], {}),
    ("prefill", "batch2_mixdiv", lambda r: [(8 * r, 0), (8 * r + 2, 0)], {}),
    ("prefill", "batch3_varrem", lambda r: [(8 * r, 0), (8 * r + 2, 0), (8 * r + 4, 0)], {}),
    # single-launch continuation chunk (cached>0); C∈{r,r+2,4r} all ≥ratio.
    ("prefill", "chunk_div", lambda r: [(r, r)], {}),          # head=0 (case b)
    ("prefill", "chunk_nondiv", lambda r: [(r, r + 2)], {}),   # head=2 (case c)
    ("prefill", "chunk_long", lambda r: [(2 * r, 4 * r)], {}),
    ("prefill", "chunk_long8", lambda r: [(8 * r, 4 * r)], {}),  # 8 cont blocks from 4*r state
    # longer continuation chunks (mirror prefill long16/long32; cached>=ratio holds).
    ("prefill", "chunk_long16", lambda r: [(16 * r, 8 * r)], {}),   # 16 cont blocks from 8*r state
    ("prefill", "chunk_long32", lambda r: [(32 * r, 16 * r)], {}),  # 32 cont blocks from 16*r state
    # cont-chunk tail segment: cached=4r (>=ratio), qlen<ratio -> no completed ratio window
    # -> nTotalBlocks=0 (情形 B: chunked-prefill last segment). ratio128: 64<128 -> nTotalBlocks=0,
    # only spill tail+pred-window to state. ratio2/4: 64>=ratio -> normal cont-chunk.
    ("prefill", "chunk_tail64", lambda r: [(64, 4 * r)], {}),
    ("prefill", "chunk_plus_pure", lambda r: [(r, r), (2 * r, 0)], {}),
    # flag variants (rotate / norope / no-ape).
    ("prefill", "rotate", lambda r: [(8 * r, 0)], {"do_rotate": True}),
    ("prefill", "norope", lambda r: [(8 * r, 0)], {"freqs_none": True}),
    ("prefill", "norope_noape", lambda r: [(8 * r, 0)], {"freqs_none": True, "ape_enabled": False}),
    ("prefill", "noape", lambda r: [(8 * r, 0)], {"ape_enabled": False}),
    # ── MIX (single launch: prefill req(s) + decode req(s)) ──
    ("mix", "pure_dec1", lambda r: [(4 * r, 0), (1, 6 * r - 1)], {}),
    ("mix", "pure_dec2", lambda r: [(4 * r, 0), (1, 6 * r - 1), (1, 8 * r - 1)], {}),
    ("mix", "multi_pf_dec3", lambda r: [(8 * r, 0), (4 * r, 0), (1, 6 * r - 1),
                                         (1, 8 * r - 1), (1, 10 * r - 1)], {}),
    ("mix", "short_pf_dec2", lambda r: [(r, 0), (1, 4 * r - 1), (1, 6 * r - 1)], {}),
    ("mix", "chunk_pure_dec", lambda r: [(r, r), (2 * r, 0), (1, 6 * r - 1)], {}),
    ("mix", "chunk_dec", lambda r: [(r, r), (1, 6 * r - 1)], {}),
    # longer mix sequences (long prefill / long cont-chunk + decode step).
    ("mix", "long_pf_dec1", lambda r: [(64 * r, 0), (1, 6 * r - 1)], {}),     # long prefill + 1 decode
    ("mix", "long_pf_dec2", lambda r: [(64 * r, 0), (1, 6 * r - 1), (1, 8 * r - 1)], {}),  # + 2nd decode
    ("mix", "chunk_long_dec", lambda r: [(8 * r, 4 * r), (1, 6 * r - 1)], {}),  # long cont-chunk + decode
    ("mix", "tok2048_dec", lambda r: [(2048, 0), (1, 6 * r - 1)], {}),         # absolute long prefill + decode
    ("mix", "tok4096_dec", lambda r: [(4096, 0), (1, 6 * r - 1)], {}),
    ("mix", "tok8192_dec", lambda r: [(8192, 0), (1, 6 * r - 1)], {}),
    # short prefill (qlen<ratio -> nTotalBlocks=0 for the prefill req) + decode step.
    # ratio128: prefill 64<128 -> nTotalBlocks=0 (spill only) + decode compress step.
    ("mix", "short_pf_dec", lambda r: [(64, 0), (1, 6 * r - 1)], {}),
    ("mix", "noape", lambda r: [(4 * r, 0), (1, 6 * r - 1)], {"ape_enabled": False}),
    # ── DECODE (multi-launch state machine -> run_decode_test) ──
    ("decode", "basic", lambda r: [], {}),
    ("decode", "long3", lambda r: [], {"n_compress": 3}),
    ("decode", "long8", lambda r: [], {"n_compress": 8}),
    ("decode", "long16", lambda r: [], {"n_compress": 16}),
    ("decode", "multi2", lambda r: [], {"n_decode_reqs": 2}),
    ("decode", "multi2_long3", lambda r: [], {"n_compress": 3, "n_decode_reqs": 2}),
    ("decode", "noape", lambda r: [], {"ape_enabled": False}),
    ("decode", "firstwin", lambda r: [], {"first_window": True}),  # overlap-only
]


def precompute_freqs_cis(dim: int, end: int, theta: float = ROPE_THETA):
    """Complex freqs_cis of shape ``[end, dim // 2]`` on NPU (TTTWWW layout)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) RoPE over the last ``rope_dim`` of ``x``.

    Output is interleaved [r0,i0,r1,i1,...] (view_as_real(...).flatten(-2)),
    matching the kernel's outInterleaved=true rope output.
    """
    dtype = x.dtype
    *lead, rope_dim = x.shape
    x = torch.view_as_complex(x.float().reshape(*lead, rope_dim // 2, 2))
    if freqs_cis.dim() < x.dim():
        freqs_cis = freqs_cis.unsqueeze(-2)
    freqs_cis = freqs_cis.expand_as(x).to(x.dtype)
    y = torch.view_as_real(x * freqs_cis).flatten(-2)
    return y.to(dtype)


def overlap_transform(tensor: torch.Tensor, ratio: int, head_dim: int, value=0):
    """``[1, nBlocks, ratio, coff*head_dim]`` -> ``[1, nBlocks, 2*ratio, head_dim]``.

    Front half = previous block's [:head_dim] left-shifted (block 0 padded with
    ``value``), back half = current block's [head_dim:].
    """
    _, s, _, _ = tensor.size()
    new_tensor = tensor.new_full((1, s, 2 * ratio, head_dim), value)
    new_tensor[:, :, ratio:] = tensor[:, :, :, head_dim:]
    new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :head_dim]
    return new_tensor


def reference_one_request(kv_req, score_req, ape, norm_w, freqs_cis, ratio, overlap,
                          head_dim, rope_head_dim, norm_eps, test_dtype, do_rotate=False):
    """Pure-torch reference for ONE request's prefill.

    kv_req/score_req: ``[qlen, coff*head_dim]`` fp32. Returns
    ``(blocks, state_rows)``: ``blocks`` is ``[nBlocks, head_dim]`` (post
    weighted-sum + RMSNorm + rotary); ``state_rows`` is ``[2*ratio, 2*full_dim]``
    holding spilled [kv|score] rows — slots [0,ratio)=last full window (overlap
    only), slots [offset, offset+remainder)=tail remainder.
    """
    qlen = kv_req.shape[0]
    remainder = qlen % ratio
    cutoff = qlen - remainder
    offset = ratio if overlap else 0
    full_dim = kv_req.shape[1]
    n_blocks = cutoff // ratio
    # spill state (raw GEMM kv/score, score+ape). One row per slot: [kv|score].
    state_rows = torch.zeros(2 * ratio, 2 * full_dim, dtype=torch.float32,
                             device=kv_req.device)
    if overlap and cutoff >= ratio:
        last_win_kv = kv_req[cutoff - ratio:cutoff]       # [ratio, full_dim]
        last_win_sc = score_req[cutoff - ratio:cutoff]
        if ape is not None:
            last_win_sc = last_win_sc + ape
        state_rows[:ratio, :full_dim] = last_win_kv
        state_rows[:ratio, full_dim:] = last_win_sc
    if remainder > 0:
        tail_kv = kv_req[cutoff:cutoff + remainder]       # [rem, full_dim]
        tail_sc = score_req[cutoff:cutoff + remainder]
        if ape is not None:
            tail_sc = tail_sc + ape[:remainder]
        state_rows[offset:offset + remainder, :full_dim] = tail_kv
        state_rows[offset:offset + remainder, full_dim:] = tail_sc
    # compress the cutoff prefix into n_blocks.
    kv = kv_req[:cutoff].unsqueeze(0).unflatten(1, (-1, ratio))
    score = score_req[:cutoff].unsqueeze(0).unflatten(1, (-1, ratio))
    if ape is not None:
        score = score + ape  # +ape per row
    if overlap:
        kv = overlap_transform(kv, ratio, head_dim, 0)
        score = overlap_transform(score, ratio, head_dim, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [1, nBlocks, head_dim]
    # Mirror the kernel's bf16 round of the weighted sum before norm (else the
    # norm sees extra-precision input and outputs drift by 1-2 ulp).
    kv = kv.to(test_dtype).float()
    var = kv.pow(2).mean(-1, keepdim=True)
    kv = kv * torch.rsqrt(var + norm_eps)
    if norm_w is not None:
        kv = kv * norm_w.float()
    kv = kv.to(test_dtype)  # [1, nBlocks, head_dim]
    # rotary on last rope_head_dim dims; position = block j's rope pos j*ratio.
    block_positions = torch.arange(n_blocks, device=kv.device) * ratio
    rope_in = kv[..., -rope_head_dim:]
    # aclnnIndex does not support complex64: index the real view, then rebuild.
    freqs = torch.view_as_complex(
        torch.view_as_real(freqs_cis)[block_positions].contiguous())  # [n_blocks, rope_head_dim//2]
    rope_out = apply_rotary_emb(rope_in.float(), freqs.unsqueeze(0))
    kv = torch.cat([kv[..., :-rope_head_dim].float(), rope_out], dim=-1).to(test_dtype)
    # rotate (x * 1/sqrt(head_dim)) after rope, before return.
    if do_rotate:
        kv = kv * (head_dim ** -0.5)
    return kv.squeeze(0), state_rows  # [n_blocks, head_dim], [2*ratio, 2*full_dim]


def reference_one_request_norope(kv_req, score_req, ape, norm_w, ratio, overlap,
                                 head_dim, norm_eps, test_dtype):
    """Reference for ONE request's prefill with RoPE disabled: pooled latent
    after RMSNorm only (no rotary, no cache write). Mirrors the kernel's
    ``freqs=None`` (hasRope=false) output — the pre-RoPE latent the indexer
    consumes. ``ape=None`` exercises the kernel's ``ape == nullptr`` path.

    Returns ``[nBlocks, head_dim]`` in ``test_dtype``.
    """
    qlen = kv_req.shape[0]
    cutoff = qlen - (qlen % ratio)
    full_dim = kv_req.shape[1]
    n_blocks = cutoff // ratio
    kv = kv_req[:cutoff].unsqueeze(0).unflatten(1, (-1, ratio))
    score = score_req[:cutoff].unsqueeze(0).unflatten(1, (-1, ratio))
    if ape is not None:
        score = score + ape
    if overlap:
        kv = overlap_transform(kv, ratio, head_dim, 0)
        score = overlap_transform(score, ratio, head_dim, float("-inf"))
    kv = (kv * score.softmax(dim=2)).sum(dim=2)  # [1, nBlocks, head_dim]
    kv = kv.to(test_dtype).float()
    var = kv.pow(2).mean(-1, keepdim=True)
    kv = kv * torch.rsqrt(var + norm_eps)
    if norm_w is not None:
        kv = kv * norm_w.float()
    return kv.to(test_dtype).squeeze(0)  # [n_blocks, head_dim]


def build_paged_state(full_dim, batch, max_pos, tokens_total, query_lens, ratio, overlap):
    """Allocate the paged streaming ``state`` cache + 1-based ``state_block_table`` and
    build the PREFILL ``state_slot_mapping`` (the only spill rule that is identical
    across run_batch's fresh-prefill path / run_decode_test-prefill).

    Spill rule (per request, keyed by ABSOLUTE token position):
      - overlap & cutoff>=ratio: last full window [cutoff-ratio, cutoff) -> abs pos.
      - remainder>0:             tail           [cutoff,      cutoff+rem) -> abs pos.
    Physical block 0 is reserved (1-based table sentinel; kernel writes skipped).
    Returns ``(state, state_block_table, state_slot_mapping)``.

    NOTE: ``query_lens`` must be the PREFILL requests only. Decode-token slot mappings
    (run_mixed_batch) and per-chunk ssm (run_chunked_prefill) use different rules and
    stay inline at their call sites — do NOT force them through this helper.
    """
    state_blocks_per_req = max(1, (max_pos + STATE_BLOCK_SIZE - 1) // STATE_BLOCK_SIZE)
    n_state_blocks = 1 + batch * state_blocks_per_req
    state = torch.zeros(n_state_blocks, STATE_BLOCK_SIZE, 1, 2 * full_dim, dtype=torch.float32)
    state_block_table = (1 + torch.arange(batch * state_blocks_per_req,
                          dtype=torch.int32)).reshape(batch, state_blocks_per_req)
    state_slot_mapping = torch.zeros(tokens_total, dtype=torch.int32)
    off = 0
    for ql in query_lens:
        remainder = ql % ratio
        cutoff = ql - remainder
        if overlap and cutoff >= ratio:
            win_start = off + cutoff - ratio
            for k in range(ratio):
                state_slot_mapping[win_start + k] = cutoff - ratio + k
        if remainder > 0:
            tail_start = off + cutoff
            for k in range(remainder):
                state_slot_mapping[tail_start + k] = cutoff + k
        off += ql
    return state, state_block_table, state_slot_mapping


def check_close(name, ref, out, conf, atol=ATOL, rtol=RTOL):
    """``assert_close`` + PASS/FAIL logging, returning True on pass.

    Mirrors the swallowed-AssertionError->logging pattern the harness counts FAIL lines
    on (see [[xlite-build-deploy-test-workflow]]). Sites needing richer FAIL diagnostics
    (e.g. run_mixed_batch per-block breakdown) keep their own inline except block.
    """
    try:
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)
        logging.info(f'  PASS {name}: {conf}')
        return True
    except AssertionError as e:
        logging.error(f'  FAIL {name}: {conf}: {e}')
        logging.error(f'    ref:  {ref.flatten()[:16]}')
        logging.error(f'    out:  {out.flatten()[:16]}')
        logging.error(f'    max abs diff: {(ref - out).abs().max().item()}')
        return False


def run_batch(name, ratio, overlap, test_dtype, request_specs, flags=None):
    """Unified single-launch body for the models×work cross-product (mirrors
    attention.py's single body). Classifies each request spec (qlen, cached):
      qlen>1, cached==0  -> fresh prefill        (reference_one_request)
      qlen>1, cached>0   -> single-launch continuation chunk (front window from
                            pre-filled state + tail from gmKv; ref over
                            cat([pred,chunk]) sliced at cached//ratio)
      qlen==1, cached>0  -> decode compress       (reference_decode_compress,
                            cached = decode_pos; state window pre-filled)

    Flags: ape_enabled, do_rotate (prefill only), freqs_none (prefill only ->
    hasRope=false, compares weightedSum not compress_kv, no state check).
    """
    flags = flags or {}
    ape_enabled = flags.get("ape_enabled", True)
    do_rotate = flags.get("do_rotate", False)
    freqs_none = flags.get("freqs_none", False)

    coff = 1 + overlap
    full_dim = coff * HEAD_DIM
    merge_size = (1 + overlap) * ratio

    # 1. classify + reorder: fresh prefill, then continuation chunks, then decode.
    fresh = [(b, ql, 0) for b, (ql, c) in enumerate(request_specs) if ql > 1 and c == 0]
    cont = [(b, ql, c) for b, (ql, c) in enumerate(request_specs) if ql > 1 and c > 0]
    dec = [(b, 1, c) for b, (ql, c) in enumerate(request_specs) if ql == 1 and c > 0]
    ordered = fresh + cont + dec
    batch = len(ordered)
    if batch == 0:
        return
    query_lens = [ql for _, ql, _ in ordered]
    cached_lens = [c for _, _, c in ordered]
    for orig, ql, c in ordered:
        if ql > 1:                                     # prefill / continuation chunk
            # overlap==1 requires ql>=ratio (bStart-ratio front window underflows otherwise).
            # overlap==0 allows ql<ratio: a short prefill segment with no completed ratio
            # window yields nTotalBlocks=0 (only spill to state; kernel guards the empty
            # norm+rope tail). This covers chunked-prefill tail segments / short prompts.
            if overlap:
                assert ql >= ratio, f"overlap query len {ql} must be >= ratio {ratio}"
            if c > 0:
                assert c >= ratio, f"continuation cached {c} must be >= ratio {ratio} (overlap underflow)"
        # decode (ql==1) is unconstrained by ratio (single-token compress step).

    # 2. build gmKv/score stream + per-request reference inputs.
    #    fresh: ql tokens. cont: clen=ql chunk tokens AND pred_kv/pred_score of
    #    length c (kept for the reference's concatenated input). decode: 1 token.
    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        torch.manual_seed(11)
        chunk_kv, chunk_score, pred_kv, pred_score = [], [], [], []
        dec_kv, dec_score = [], []
        for oi, (orig, ql, c) in enumerate(ordered):
            if ql > 1 and c == 0:                      # fresh prefill
                chunk_kv.append((torch.randn(ql, full_dim, dtype=torch.float32) / 10).contiguous())
                chunk_score.append((torch.randn(ql, full_dim, dtype=torch.float32) / 10).contiguous())
                pred_kv.append(None); pred_score.append(None)
            elif ql > 1 and c > 0:                     # continuation chunk
                pk = (torch.randn(c, full_dim, dtype=torch.float32) / 10).contiguous()
                ps = (torch.randn(c, full_dim, dtype=torch.float32) / 10).contiguous()
                pred_kv.append(pk); pred_score.append(ps)
                ck = (torch.randn(ql, full_dim, dtype=torch.float32) / 10).contiguous()
                cs = (torch.randn(ql, full_dim, dtype=torch.float32) / 10).contiguous()
                chunk_kv.append(ck); chunk_score.append(cs)
            else:                                      # decode (ql==1, c>0)
                chunk_kv.append((torch.randn(1, full_dim, dtype=torch.float32) / 10).contiguous())
                chunk_score.append((torch.randn(1, full_dim, dtype=torch.float32) / 10).contiguous())
                pred_kv.append(None); pred_score.append(None)
                dec_kv.append(chunk_kv[-1]); dec_score.append(chunk_score[-1])
        kv = torch.cat(chunk_kv, dim=0).contiguous()           # [batchedTokens, full_dim]
        score = torch.cat(chunk_score, dim=0).contiguous()
        ape = (torch.randn(ratio, full_dim, dtype=torch.float32) / 10) if ape_enabled else None
        norm_w = torch.randn(HEAD_DIM, dtype=test_dtype) / 10

        query_start_loc = torch.zeros(batch, dtype=torch.int32)
        acc = 0
        for b in range(batch):
            query_start_loc[b] = acc; acc += query_lens[b]
        query_lens_t = torch.tensor(query_lens, dtype=torch.int32)
        cached_lens_t = torch.tensor(cached_lens, dtype=torch.int32)

    # 3. compress tables (fresh blocks, then cont blocks, then decode blocks).
    n_blocks_per_req = []
    block_offset = 0
    compress_slot_mapping_l, compress_positions_l = [], []
    for oi, (orig, ql, c) in enumerate(ordered):
        if ql > 1:                                     # prefill (fresh or cont)
            n_b = ((c + ql) // ratio) - (c // ratio)   # cont includes cross-boundary block
            for j in range(n_b):
                compress_slot_mapping_l.append(block_offset + j)
                compress_positions_l.append((c // ratio + j) * ratio)   # abs block first-token pos
        else:                                          # decode
            n_b = 1 if (c + 1) % ratio == 0 else 0
            for j in range(n_b):
                compress_slot_mapping_l.append(block_offset + j)
                compress_positions_l.append(c + 1 - ratio)
        n_blocks_per_req.append(n_b)
        block_offset += n_b
    n_total_blocks = sum(n_blocks_per_req)
    batched_tokens = sum(query_lens)
    max_blocks = max(n_total_blocks, BLOCK_SIZE)

    # 4. state allocation + ssm (inline; build_paged_state's ssm only knows fresh prefill).
    max_state_pos = 0
    for oi, (orig, ql, c) in enumerate(ordered):
        max_state_pos = max(max_state_pos, c + ql)
    state_blocks_per_req = max(1, (max_state_pos + STATE_BLOCK_SIZE - 1) // STATE_BLOCK_SIZE)
    n_state_blocks = 1 + batch * state_blocks_per_req
    with torch.device("npu"):
        compress_kv = torch.zeros(max_blocks, BLOCK_SIZE, 1, HEAD_DIM, dtype=test_dtype)
        compress_slot_mapping = torch.tensor(compress_slot_mapping_l, dtype=torch.int32)
        compress_positions = torch.tensor(compress_positions_l, dtype=torch.int64)
        state = torch.zeros(n_state_blocks, STATE_BLOCK_SIZE, 1, 2 * full_dim, dtype=torch.float32)
        state_block_table = (1 + torch.arange(batch * state_blocks_per_req,
                                              dtype=torch.int32)).reshape(batch, state_blocks_per_req)
        weightedSum = torch.zeros(n_total_blocks, HEAD_DIM, dtype=test_dtype)
        # freqs table must span the largest rope position = the largest
        # compress_position (= (c//ratio+j)*ratio for prefill/cont, c+1-ratio for
        # decode) + a margin. max(c+ql) bounds the prefill positions; max(cached)+2
        # bounds the decode positions; batched_tokens covers fresh prefill.
        freqs_end = max(batched_tokens, max(c + ql for _, ql, c in ordered),
                        max(cached_lens) + 2)
        freqs_cis = precompute_freqs_cis(ROPE_HEAD_DIM, freqs_end, ROPE_THETA)

    def state_phys(b, p):
        blk = state_block_table[b][p // STATE_BLOCK_SIZE].item()
        return blk * STATE_BLOCK_SIZE + p % STATE_BLOCK_SIZE
    state_flat = state.view(-1, 2 * full_dim)

    # 5. state pre-fill (host, default stream) for continuation chunks + decode reqs.
    #    Fresh prefill needs none (kernel writes via ssm).
    cont_ref_blocks = {}   # oi -> ref_blocks over cat([pred,chunk])
    for oi, (orig, ql, c) in enumerate(ordered):
        if ql > 1 and c > 0:                            # continuation chunk
            kv_full = torch.cat([pred_kv[oi], chunk_kv[oi]], dim=0)
            sc_full = torch.cat([pred_score[oi], chunk_score[oi]], dim=0)
            rb, _ = reference_one_request(
                kv_full, sc_full, ape, norm_w, freqs_cis, ratio, overlap,
                HEAD_DIM, ROPE_HEAD_DIM, NORM_EPS, test_dtype, do_rotate=False)
            cont_ref_blocks[oi] = rb
            # Pre-fill the state window this chunk's blocks will READ (not the spill
            # st_rows, which is the LAST-window tail written for the NEXT chunk). A
            # cont-chunk block j has bStart=(c//ratio+j)*ratio; its overlap window
            # spans abs [bStart-ratio, bStart+ratio) (overlap) or [bStart, bStart+ratio)
            # (non-overlap). Every abs < c in that window is read from state, so fill
            # ALL of [fill_lo, c) with the predecessor's RAW per-token [kv | score+ape].
            # NB: score+ape (not raw score) — the kernel's hasStateRows path assumes
            # state rows already carry ape (spill baked it in), so gmKv rows are
            # pre-added and softmax runs with ape=null.
            bStart_first = (c // ratio) * ratio
            fill_lo = max(0, bStart_first - ratio) if overlap else bStart_first
            for ap in range(fill_lo, c):
                row = state_flat[state_phys(oi, ap)]
                row[:full_dim] = pred_kv[oi][ap]
                if ape is not None:
                    row[full_dim:] = pred_score[oi][ap] + ape[ap % ratio]
                else:
                    row[full_dim:] = pred_score[oi][ap]
        elif ql == 1 and c > 0:                         # decode
            p = c
            win0 = p + 1 - merge_size
            for k in range(merge_size):
                posk = win0 + k
                if posk == p:                          # decode token itself
                    tk = chunk_kv[oi][0]
                    ts = chunk_score[oi][0] + (ape[p % ratio] if ape is not None else 0)
                else:                                  # synthetic history (NPU: added to ape[...])
                    tk = torch.randn(full_dim, dtype=torch.float32, device="npu") / 10
                    ts = torch.randn(full_dim, dtype=torch.float32, device="npu") / 10
                    if ape is not None:
                        ts = ts + ape[posk % ratio]
                row = state_flat[state_phys(oi, posk)]
                row[:full_dim] = tk
                row[full_dim:] = ts

    # 6. state_slot_mapping (indexed by gmKv row, value = request-local abs pos).
    #    MUST be an NPU tensor (baseline built it inside run_test's `with
    #    torch.device("npu")` via build_paged_state; CPU tensor here caused a
    #    507057 vector-core exception at the compressor() launch).
    with torch.device("npu"):
        state_slot_mapping = torch.zeros(batched_tokens, dtype=torch.int32)
        off = 0
        for oi, (orig, ql, c) in enumerate(ordered):
            if ql > 1 and c == 0:                          # fresh prefill spill rule
                remainder = ql % ratio; cutoff = ql - remainder
                if overlap and cutoff >= ratio:
                    for k in range(ratio):
                        state_slot_mapping[off + cutoff - ratio + k] = cutoff - ratio + k
                if remainder > 0:
                    for k in range(remainder):
                        state_slot_mapping[off + cutoff + k] = cutoff + k
            elif ql > 1 and c > 0:                         # cont chunk spill (mirror L705-716)
                abs_end = c + ql; cutoff = abs_end // ratio * ratio
                spill_begin = cutoff
                if overlap and cutoff >= ratio + c:
                    spill_begin = cutoff - ratio
                if spill_begin < c:
                    spill_begin = c
                for k in range(ql):
                    abs_pos = c + k
                    if abs_pos >= spill_begin:
                        state_slot_mapping[off + k] = abs_pos
            else:                                          # decode single token
                state_slot_mapping[off] = c
            off += ql

    # 7. the ONE compressor() launch. Sync first (ssm/state built on default
    #    stream, compressor reads on rt.stream — see [[compressor-spill-test-stream-race]]).
    torch.npu.synchronize()
    freqs_arg = None if freqs_none else freqs_cis
    compressor(rt, kv, score, ape, norm_w, freqs_arg, weightedSum, query_start_loc,
               query_lens_t, cached_lens_t, batch, n_total_blocks, ratio, overlap, HEAD_DIM,
               ROPE_HEAD_DIM, NORM_EPS, compress_kv, compress_positions, compress_slot_mapping,
               BLOCK_SIZE, state, state_slot_mapping, state_block_table, STATE_BLOCK_SIZE,
               STATE_BLOCK_SIZE * 2 * full_dim, do_rotate)
    torch.npu.synchronize()
    cat = 'mixed' if any(ql == 1 for _, ql, _ in ordered) and any(ql > 1 for _, ql, _ in ordered) else 'batch'
    logging.info(f'compressor {cat} ({test_dtype}) ratio={ratio} overlap={overlap} '
                 f'query_lens={query_lens} cached={cached_lens} executed!')

    # 8. per-request reference + compare.
    flat = compress_slot_mapping.to(torch.int64)
    out_all = compress_kv.view(-1, 1, HEAD_DIM)[flat, 0]   # [n_total_blocks, head_dim]
    ws_out = weightedSum                                  # norope compares this
    block_base = 0
    off = 0
    ok_all = True
    for oi, (orig, ql, c) in enumerate(ordered):
        n_b = n_blocks_per_req[oi]
        cmp_conf = f'ratio={ratio} overlap={overlap} dtype={test_dtype} req={oi} qlen={ql} cached={c}'
        if ql > 1 and c == 0:                             # fresh prefill
            if freqs_none:
                ref = reference_one_request_norope(chunk_kv[oi], chunk_score[oi], ape, norm_w,
                                                    ratio, overlap, HEAD_DIM, NORM_EPS, test_dtype)
                out = ws_out[block_base:block_base + n_b].to(ref.dtype)
                ok_all &= check_close(f'weightedSum_norope ({name})', ref, out,
                                       f'kernel weightedSum (freqs=None) vs ref; {cmp_conf}')
            else:
                rb, st_rows = reference_one_request(chunk_kv[oi], chunk_score[oi], ape, norm_w,
                                                    freqs_cis, ratio, overlap, HEAD_DIM,
                                                    ROPE_HEAD_DIM, NORM_EPS, test_dtype, do_rotate)
                out = out_all[block_base:block_base + n_b].to(rb.dtype)
                ok_all &= check_close(f'compress_kv ({name})', rb, out,
                                       f'kernel compress_kv vs ref (norm+rope+cache); {cmp_conf}')
                # verify state spill (STRICT — bit-exact kv/score+ape copies).
                remainder = ql % ratio; cutoff = ql - remainder
                offset = ratio if overlap else 0
                spilled = []
                if overlap and cutoff >= ratio:
                    for k in range(ratio):
                        spilled.append((off + cutoff - ratio + k, k))
                if remainder > 0:
                    for k in range(remainder):
                        spilled.append((off + cutoff + k, offset + k))
                if spilled:
                    tok_ids = torch.tensor([t for t, _ in spilled], dtype=torch.int64)
                    ref_slot_ids = torch.tensor([s for _, s in spilled], dtype=torch.int64)
                    mapping = state_slot_mapping[tok_ids].to(torch.int64)
                    tbl = state_block_table[oi].cpu()
                    phys = torch.tensor(
                        [tbl[s // STATE_BLOCK_SIZE].item() * STATE_BLOCK_SIZE + s % STATE_BLOCK_SIZE
                         for s in mapping.tolist()], dtype=torch.int64)
                    got = state_flat[phys].to(torch.float32)
                    exp = st_rows[ref_slot_ids].to(torch.float32)
                    try:
                        torch.testing.assert_close(got, exp, atol=STRICT_ATOL, rtol=STRICT_RTOL)
                        logging.info(f'  PASS state ({name}) req={oi} slots={len(spilled)}: '
                                     f'kernel state[table-resolved] vs ref state_rows; {cmp_conf}')
                    except AssertionError as e:
                        ok_all = False
                        logging.error(f'  FAIL state ({name}) req={oi}: {cmp_conf}: {e}')
                        logging.error(f'    ref: {exp.flatten()[:16]}')
                        logging.error(f'    out: {got.flatten()[:16]}')
                        logging.error(f'    max abs diff: {(exp - got).abs().max().item()}')
        elif ql > 1 and c > 0:                           # continuation chunk
            rb = cont_ref_blocks[oi]
            ref_slice = rb[c // ratio:c // ratio + n_b]
            out = out_all[block_base:block_base + n_b].to(ref_slice.dtype)
            ok_all &= check_close(f'compress_kv_cont ({name})', ref_slice, out,
                                   f'kernel cont-chunk compress_kv vs ref[c//ratio:]; {cmp_conf}')
        else:                                            # decode
            p = c
            sr = torch.stack([state_flat[state_phys(oi, pp)] for pp in range(p + 1)])
            ref = reference_decode_compress(sr, norm_w, freqs_cis, ratio, overlap, HEAD_DIM,
                                            ROPE_HEAD_DIM, NORM_EPS, test_dtype, p)
            if n_b >= 1:                                 # compress position: 1 block written
                out = out_all[block_base].to(ref.dtype)   # [head_dim], 1-D (matches ref)
                ok_all &= check_close(f'compress_kv_decode ({name})', ref, out,
                                       f'kernel decode compress_kv vs ref; {cmp_conf}')
        block_base += n_b
        off += ql

    # mixed-batch per-block diagnostic on failure (mirrors run_mixed_batch L1415-1439).
    if not ok_all and cat == 'mixed':
        ref_all = []
        block_base = 0
        for oi, (orig, ql, c) in enumerate(ordered):
            n_b = n_blocks_per_req[oi]
            if ql > 1 and c == 0:
                rb, _ = reference_one_request(chunk_kv[oi], chunk_score[oi], ape, norm_w, freqs_cis,
                                              ratio, overlap, HEAD_DIM, ROPE_HEAD_DIM, NORM_EPS,
                                              test_dtype, False)
                ref_all.extend(rb)
            elif ql > 1 and c > 0:
                ref_all.extend(cont_ref_blocks[oi][c // ratio:c // ratio + n_b])
            else:
                p = c
                sr = torch.stack([state_flat[state_phys(oi, pp)] for pp in range(p + 1)])
                ref_all.append(reference_decode_compress(sr, norm_w, freqs_cis, ratio, overlap,
                                                          HEAD_DIM, ROPE_HEAD_DIM, NORM_EPS,
                                                          test_dtype, p))
            block_base += n_b
        ref_t = torch.stack(ref_all, dim=0).to(out_all.dtype)
        for bi in range(n_total_blocks):
            bd = (ref_t[bi] - out_all[bi]).abs().max().item()
            logging.error(f'    block {bi}: max_abs_diff={bd:.6f} '
                          f'ref_norm={ref_t[bi].abs().max().item():.4f} '
                          f'out_norm={out_all[bi].abs().max().item():.4f}')
        _ckv_dump = __import__('os').environ.get('XLITE_DUMP_CKV')
        if _ckv_dump:
            torch.save({'compress_kv': compress_kv.cpu(), 'state': state.cpu(),
                        'weightedSum': weightedSum.cpu()}, _ckv_dump)


def run_decode_test(name, ratio, overlap, test_dtype, ape_enabled=True, n_compress=1,
                    n_decode_reqs=1):
    """Decode branch: a prefill fills the streaming state window, then per-step
    decode launches (qLen==1) accumulate tokens; the step where
    (pos+1)%ratio==0 compresses the window (weighted-sum + norm + rope + cache).
    Both non-compress and compress steps verified. State rows are addressed by
    ABSOLUTE token position, so nothing is slid: the window a compress step
    assembles is the state rows [T-mergeSize, T), T = pos+1, read in place.
    ratio=128 (overlap==0) exercises the kernel's channel-sliced tiled path.

    ``ape_enabled=False`` passes ``ape=None``, exercising the ``ape == nullptr``
    decode paths (raw score staged via copy instead of vadd).

    ``n_compress`` (default 1): each decode request runs ``n_compress*ratio``
    decode steps, triggering ``n_compress`` compress points. n_compress>1
    exercises repeated window slides + repeated compress_kv writes on the same
    request (long decode stream).

    ``n_decode_reqs`` (default 1): multiple decode requests (batch =
    n_decode_reqs) share one compressor() per step. Each request starts from a
    distinct cached_len offset so their compress points interleave. Exercises the
    cross-core barrier (KERNEL_TASK_TYPE_DEFAULT + ffts_cross_core_sync) under
    multiple concurrent decode requests — the multi-decode-req stability path.
    """
    coff = 1 + overlap
    full_dim = coff * HEAD_DIM
    n_steps = n_compress * ratio
    batch = n_decode_reqs
    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        torch.manual_seed(7)
        # Each decode request has its OWN prefill stream of length ql_b, packed
        # contiguously into the batched token buffer (query_lens = [ql_b]).
        # ql_b is a MULTIPLE of ratio (ratio*(4+b)) so the overlap prefill spill
        # window [ql_b-ratio, ql_b) aligns exactly with the first decode compress's
        # front half — and so that prefill runs cleanly with cached_lens=0 (the
        # kernel's prefill overlap path reads prevStart = blockTokStart - ratio
        # and underflows when cachedLens%ratio!=0, i.e. head!=0; the decode path
        # handles nonzero cachedLens by reading from state, not gmKv). Different
        # magnitudes give distinct compress_kv slots; all requests compress in the
        # same step (concurrent), stressing the cross-core barrier. n_compress>1
        # runs repeated window slides + repeated compress_kv writes per request.
        ql_b = [ratio * (4 + b) for b in range(batch)]
        n_pf_blocks = sum(q // ratio for q in ql_b)   # prefill compressed blocks
        kv_list = [(torch.randn(q, full_dim, dtype=torch.float32) / 10).contiguous()
                   for q in ql_b]
        score_list = [(torch.randn(q, full_dim, dtype=torch.float32) / 10).contiguous()
                      for q in ql_b]
        kv = torch.cat(kv_list, dim=0).contiguous()
        score = torch.cat(score_list, dim=0).contiguous()
        ape = (torch.randn(ratio, full_dim, dtype=torch.float32) / 10) if ape_enabled else None
        norm_w = torch.randn(HEAD_DIM, dtype=test_dtype) / 10
        # cached_lens=0 for prefill (see NOTE above); decode steps advance it per
        # request via cl_d = cur_cached (nonzero — decode path handles it).
        cached_lens = torch.zeros(batch, dtype=torch.int32)
        query_start_loc = torch.zeros(batch, dtype=torch.int32)
        acc = 0
        for b in range(batch):
            query_start_loc[b] = acc
            acc += ql_b[b]
        query_lens_t = torch.tensor(ql_b, dtype=torch.int32)
        # compress_kv covers prefill blocks + each req's n_compress compress rows.
        max_compress_slots = n_pf_blocks + batch * n_compress
        compress_kv = torch.zeros(max(max_compress_slots, BLOCK_SIZE), BLOCK_SIZE, 1,
                                  HEAD_DIM, dtype=test_dtype)
        # prefill compress tables: contiguous slots, request-local rope pos j*ratio.
        compress_slot_mapping = torch.zeros(n_pf_blocks, dtype=torch.int32)
        compress_positions = torch.zeros(n_pf_blocks, dtype=torch.int64)
        g = 0
        for b in range(batch):
            nb_b = ql_b[b] // ratio
            for j in range(nb_b):
                compress_slot_mapping[g] = g
                compress_positions[g] = j * ratio
                g += 1
        max_ql = max(ql_b)
        freqs_cis = precompute_freqs_cis(ROPE_HEAD_DIM, max_ql + n_steps + ratio, ROPE_THETA)
        weightedSum = torch.zeros(n_pf_blocks, HEAD_DIM, dtype=test_dtype)
        # state cache: rows indexed by ABSOLUTE token position; each request owns
        # its own block-table columns covering [0, max_ql + n_steps). Block 0 is
        # reserved (sentinel). build_paged_state fills the prefill spill mapping
        # (per-request: overlap last window [ql_b-ratio,ql_b) + tail remainder),
        # keyed by absolute position — only the PREFILL requests (ql_b) are passed;
        # decode-step token rows are addressed by absolute pos, not slot mapping.
        max_pos = max_ql + n_steps + ratio
        state, state_block_table, state_slot_mapping = build_paged_state(
            full_dim, batch, max_pos, sum(ql_b), ql_b, ratio, overlap)
        torch.npu.synchronize()
        compressor(rt, kv, score, ape, norm_w, freqs_cis, weightedSum, query_start_loc,
                   query_lens_t, cached_lens, batch, n_pf_blocks, ratio, overlap, HEAD_DIM,
                   ROPE_HEAD_DIM, NORM_EPS, compress_kv, compress_positions, compress_slot_mapping,
                   BLOCK_SIZE, state, state_slot_mapping, state_block_table, STATE_BLOCK_SIZE,
                   STATE_BLOCK_SIZE * 2 * full_dim)
        torch.npu.synchronize()

        kv_cpu = [t.cpu() for t in kv_list]
        score_cpu = [t.cpu() for t in score_list]
        ape_cpu = ape.cpu() if ape is not None else None
        ok_all = True

        def dstate_row(req, p):
            """Physical row index of absolute position ``p`` for request ``req``:
            column p // STATE_BLOCK_SIZE, row p % STATE_BLOCK_SIZE (via the req's
            block-table row)."""
            blk = state_block_table[req][p // STATE_BLOCK_SIZE].item()
            return blk * STATE_BLOCK_SIZE + p % STATE_BLOCK_SIZE

        # per-request running cached_len (advances one per step) and compress
        # row counter (next compress_kv slot for this request). Each request b owns
        # a DISTINCT compress_kv slot range [n_pf_blocks + b*n_compress, ...) so
        # concurrent compresses in one launch (the cross-core barrier case) land
        # in separate slots — matches the kernel's per-block compress_slot_mapping.
        cur_cached = list(ql_b)                              # post-prefill: cached = ql_b
        next_compress_row = [n_pf_blocks + b * n_compress for b in range(batch)]
        # Each step writes every request's token row at its (request-local) pos;
        # the compress step reads that request's window [T-mergeSize, T).
        for step in range(n_steps):
            tok_kv = (torch.randn(1, full_dim, dtype=torch.float32) / 10).contiguous()
            tok_sc = (torch.randn(1, full_dim, dtype=torch.float32) / 10).contiguous()
            # shared token stream across requests (each request advances its own
            # abs position; correctness is per-request by abs position).
            # pos = the absolute position this step WRITES (= cachedLens passed
            # to the kernel; kernel writes the token at state row `pos` and
            # compresses when (pos+1)%ratio==0). cur_cached tracks the next
            # position to write, starting at ql_b after prefill.
            pos_list = [cur_cached[b] for b in range(batch)]
            compress_flags = [(p + 1) % ratio == 0 for p in pos_list]
            n_compress_this = sum(compress_flags)
            nb = n_compress_this
            if n_compress_this > 0:
                cs_d = torch.tensor(
                    [next_compress_row[b] for b in range(batch) if compress_flags[b]],
                    dtype=torch.int32)
                cp_d = torch.tensor(
                    [pos_list[b] + 1 - ratio for b in range(batch) if compress_flags[b]],
                    dtype=torch.int64)
            else:
                cs_d = torch.zeros(0, dtype=torch.int32)
                cp_d = torch.zeros(0, dtype=torch.int64)
            ws_d = torch.zeros(nb, HEAD_DIM, dtype=test_dtype)
            # shared single decode token at gmKv row 0; every request reads it.
            qsl_d = torch.zeros(batch, dtype=torch.int32)
            qlt_d = torch.ones(batch, dtype=torch.int32)
            cl_d = torch.tensor(pos_list, dtype=torch.int32)
            # decode ignores stateSlotMapping (rows come from absolute positions).
            ssm_d = torch.zeros(batch, dtype=torch.int32)
            torch.npu.synchronize()
            compressor(rt, tok_kv, tok_sc, ape, norm_w, freqs_cis, ws_d, qsl_d,
                       qlt_d, cl_d, batch, nb, ratio, overlap, HEAD_DIM, ROPE_HEAD_DIM,
                       NORM_EPS, compress_kv, cp_d, cs_d, BLOCK_SIZE, state, ssm_d,
                       state_block_table, STATE_BLOCK_SIZE,
                       STATE_BLOCK_SIZE * 2 * full_dim)
            torch.npu.synchronize()

            for b in range(batch):
                pos = pos_list[b]
                compressing = compress_flags[b]
                row = pos % ratio
                sc_ape = tok_sc.cpu()[0] + (ape_cpu[row] if ape_cpu is not None else 0)
                if not compressing:
                    got = state.cpu().view(-1, 2 * full_dim)[dstate_row(b, pos)]
                    try:
                        torch.testing.assert_close(got[:full_dim], tok_kv.cpu()[0],
                                                   atol=STRICT_ATOL, rtol=STRICT_RTOL)
                        torch.testing.assert_close(got[full_dim:], sc_ape, atol=STRICT_ATOL, rtol=STRICT_RTOL)
                        logging.info(f'  PASS decode state ({name}) req={b} pos={pos}: '
                                     f'state row [{pos}] == '
                                     f'[kv | score+ape[{row}]]; '
                                     f'ratio={ratio} overlap={overlap} dtype={test_dtype}')
                    except AssertionError as e:
                        ok_all = False
                        logging.error(f'  FAIL decode state ({name}) req={b} pos={pos}: {e}')
                        logging.error(f'  kv max diff: '
                                      f'{(got[:full_dim] - tok_kv.cpu()[0]).abs().max().item()}')
                        logging.error(f'  sc+ape max diff: '
                                      f'{(got[full_dim:] - sc_ape).abs().max().item()}')
                else:
                    # reference via reference_decode_compress — the same helper
                    # run_decode_first_window_test uses (handles has_history
                    # true/false). ql_b stays ratio-aligned so prefill runs cleanly
                    # with cached_lens=0; the bStart=0 decode first-window case is
                    # covered by run_decode_first_window_test.
                    st_now = state.view(-1, 2 * full_dim)
                    sr = torch.stack([st_now[dstate_row(b, pp)] for pp in range(pos + 1)])
                    ref = reference_decode_compress(sr, norm_w, freqs_cis, ratio, overlap,
                                                    HEAD_DIM, ROPE_HEAD_DIM, NORM_EPS,
                                                    test_dtype, pos)
                    slot = next_compress_row[b]
                    got = compress_kv.view(-1, HEAD_DIM)[slot].float()
                    # bf16 one-ulp tolerance: accumulation order differs from torch.
                    ok_all &= check_close(
                        f'decode compress ({name}) req={b} pos={pos}', ref, got,
                        f'compress_kv[slot={slot}] == weighted-sum+norm+rope of assembled '
                        f'window; ratio={ratio} overlap={overlap} dtype={test_dtype}')
                    next_compress_row[b] += 1
                cur_cached[b] = pos + 1  # advance to next position to write
        _ckv_dump = __import__('os').environ.get('XLITE_DUMP_CKV')
        if _ckv_dump:
            torch.save({'compress_kv': compress_kv.cpu(), 'state': state.cpu(),
                         'weightedSum': weightedSum.cpu()}, _ckv_dump)


def reference_decode_compress(state_rows, norm_w, freqs_cis, ratio, overlap,
                              head_dim, rope_head_dim, norm_eps, test_dtype, pos):
    """Pure-torch reference for ONE decode compress step at absolute position
    ``pos`` ((pos+1) % ratio == 0). ``state_rows[p]`` holds [kv|score+ape] for
    absolute position p (pre-filled by prior steps, ape already baked into the
    score half — no ``ape`` arg). Returns ``[head_dim]`` post weighted-sum +
    RMSNorm + rotary (mirrors the unified compressor compress path).
    """
    full_dim = (1 + overlap) * head_dim
    merge_size = (1 + overlap) * ratio
    win_base = pos + 1 - ratio          # window start T-ratio
    has_history = (pos + 1) >= 2 * ratio
    if overlap:
        # front [0,ratio) = prev window's [:head_dim] (first window padded 0/-inf),
        # back [ratio,2*ratio) = this window's [head_dim:].
        kv_rows, sc_rows = [], []
        if has_history:
            for k in range(ratio):
                front = state_rows[win_base - ratio + k]
                kv_rows.append(front[:head_dim])
                sc_rows.append(front[full_dim:full_dim + head_dim])
        else:
            for k in range(ratio):
                kv_rows.append(torch.zeros(head_dim, dtype=torch.float32))
                sc_rows.append(torch.full((head_dim,), float('-inf'), dtype=torch.float32))
        for k in range(ratio):
            back = state_rows[win_base + k]
            kv_rows.append(back[head_dim:full_dim])
            sc_rows.append(back[full_dim + head_dim:])
    else:
        kv_rows = [state_rows[win_base + k][:full_dim] for k in range(ratio)]
        sc_rows = [state_rows[win_base + k][full_dim:] for k in range(ratio)]
    kv_rows = torch.stack(kv_rows)      # [merge_size, head_dim]
    sc_rows = torch.stack(sc_rows)      # [merge_size, head_dim]
    w = (kv_rows * sc_rows.softmax(dim=0)).sum(dim=0)
    # mirror the kernel's bf16 round of the weighted sum before norm.
    w = w.to(test_dtype).float()
    var = w.pow(2).mean()
    wn = (w * torch.rsqrt(var + norm_eps)) * norm_w.float()
    # interleaved rope (matches apply_rotary_emb), pos = pos + 1 - ratio.
    xc = torch.view_as_complex(wn[-rope_head_dim:].unflatten(-1, (-1, 2)))
    fc_row = freqs_cis[pos + 1 - ratio]
    yr = torch.view_as_real(xc * fc_row)
    ref = wn.clone()
    ref[-rope_head_dim:] = yr.flatten(-2).flatten()
    return ref.to(test_dtype).float()


def run_decode_first_window_test(name, ratio, overlap, test_dtype, ape_enabled=True):
    """Regression: decode compress at the FIRST-EVER window (no predecessor).

    A short prefill (qlen = ratio-1 < ratio) compresses NOTHING and spills
    `ratio-1` tokens to state. Then decode steps run in sequence:

    1. pos = ratio-1   — first compress point (bStart = 0, noPredecessor = true,
       cachedLen = ratio-1 ∈ (0, ratio)). The pre-fix bug: load skipped the front
       half (noPredecessor) but the pad guard was ``cachedLen == 0`` (false here),
       leaving the front half as UB residue fed into softmax. Reference uses
       reference_decode_compress (has_history=false → front padded 0/-inf).
    2. pos = ratio, ratio+1, ..., 2*ratio-2 — non-compress decode steps that fill
       the SECOND compress window's back half [ratio, 2*ratio) with real tokens.
       (A gap here leaves zero state rows in the window — an unrealistic scenario
       that perturbs the ape softmax weights and can masquerade as a divergence.)
    3. pos = 2*ratio-1 — second compress point (bStart = ratio, has_history=true);
       front half read from the previous window's state rows (NOT padded), back
       half fully populated by step 2.

    The prefill spill (state[0..ratio-2] == [kv|score+ape]) is verified BEFORE the
    decode compress, so a wrong spill cannot make ref and got agree on garbage
    (false PASS). Each decode step's token-state write is verified too.

    Only meaningful for overlap (front/back split): ratio=4 overlap=1.
    """
    coff = 1 + overlap
    full_dim = coff * HEAD_DIM
    batch = 1
    ql_b = ratio - 1                       # < ratio: prefill produces 0 blocks
    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        torch.manual_seed(7)
        kv = (torch.randn(ql_b, full_dim, dtype=torch.float32) / 10).contiguous()
        score = (torch.randn(ql_b, full_dim, dtype=torch.float32) / 10).contiguous()
        ape = (torch.randn(ratio, full_dim, dtype=torch.float32) / 10) if ape_enabled else None
        norm_w = torch.randn(HEAD_DIM, dtype=test_dtype) / 10
        cached_lens = torch.zeros(batch, dtype=torch.int32)
        query_start_loc = torch.zeros(batch, dtype=torch.int32)
        query_lens_t = torch.tensor([ql_b], dtype=torch.int32)
        compress_kv = torch.zeros(BLOCK_SIZE, BLOCK_SIZE, 1, HEAD_DIM, dtype=test_dtype)
        compress_slot_mapping = torch.zeros(0, dtype=torch.int32)        # prefill: 0 blocks
        compress_positions = torch.zeros(0, dtype=torch.int64)    # prefill: 0 blocks
        freqs_cis = precompute_freqs_cis(ROPE_HEAD_DIM, 2 * ratio, ROPE_THETA)
        weightedSum = torch.zeros(0, HEAD_DIM, dtype=test_dtype)
        # state + 1-based table + prefill spill mapping. ql_b=ratio-1 < ratio so
        # cutoff=0: overlap's last-window spill does not apply, only the remainder
        # [0, ql_b) spills to abs pos 0..ql_b-1 — build_paged_state reproduces this.
        max_pos = 2 * ratio
        state, state_block_table, state_slot_mapping = build_paged_state(
            full_dim, batch, max_pos, ql_b, [ql_b], ratio, overlap)
        # state_slot_mapping (and the other int control tensors above) are built on
        # the default NPU stream, but compressor launches on rt.stream. Without a
        # sync the kernel reads a stale/unmapped mapping: rows collapse to slot 0
        # (last-writer-wins, a false "spill bug") or the vector core faults (507035).
        torch.npu.synchronize()
        compressor(rt, kv, score, ape, norm_w, freqs_cis, weightedSum, query_start_loc,
                   query_lens_t, cached_lens, batch, 0, ratio, overlap, HEAD_DIM,
                   ROPE_HEAD_DIM, NORM_EPS, compress_kv, compress_positions, compress_slot_mapping,
                   BLOCK_SIZE, state, state_slot_mapping, state_block_table, STATE_BLOCK_SIZE,
                   STATE_BLOCK_SIZE * 2 * full_dim)
        torch.npu.synchronize()

        def dstate_row(req, p):
            blk = state_block_table[req][p // STATE_BLOCK_SIZE].item()
            return blk * STATE_BLOCK_SIZE + p % STATE_BLOCK_SIZE

        st = state.view(-1, 2 * full_dim)

        def check_token_state(pos, tok_kv, tok_sc):
            row = pos % ratio
            exp_kv = tok_kv[0]
            exp_sc = tok_sc[0] + (ape[row] if ape is not None else 0)
            got = st[dstate_row(0, pos)]
            try:
                torch.testing.assert_close(got[:full_dim], exp_kv, atol=STRICT_ATOL, rtol=STRICT_RTOL)
                torch.testing.assert_close(got[full_dim:], exp_sc, atol=STRICT_ATOL, rtol=STRICT_RTOL)
                logging.info(f'  PASS first-window decode state ({name}) pos={pos}: '
                             f'state[{pos}] == [kv | score+ape[{row}]]')
            except AssertionError as e:
                logging.error(f'  FAIL first-window decode state ({name}) pos={pos}: {e}')
                logging.error(f'    kv max diff: {(got[:full_dim] - exp_kv).abs().max().item()}')
                logging.error(f'    sc+ape max diff: '
                              f'{(got[full_dim:] - exp_sc).abs().max().item()}')

        def decode_step(pos, compress, slot):
            tok_kv = (torch.randn(1, full_dim, dtype=torch.float32) / 10).contiguous()
            tok_sc = (torch.randn(1, full_dim, dtype=torch.float32) / 10).contiguous()
            nb = 1 if compress else 0
            cs_d = (torch.tensor([slot], dtype=torch.int32) if compress
                    else torch.zeros(0, dtype=torch.int32))
            cp_d = (torch.tensor([pos + 1 - ratio], dtype=torch.int64) if compress
                    else torch.zeros(0, dtype=torch.int64))
            ws_d = torch.zeros(nb, HEAD_DIM, dtype=test_dtype)
            qsl_d = torch.zeros(batch, dtype=torch.int32)
            qlt_d = torch.ones(batch, dtype=torch.int32)
            cl_d = torch.tensor([pos], dtype=torch.int32)
            ssm_d = torch.zeros(batch, dtype=torch.int32)
            torch.npu.synchronize()
            compressor(rt, tok_kv, tok_sc, ape, norm_w, freqs_cis, ws_d, qsl_d,
                       qlt_d, cl_d, batch, nb, ratio, overlap, HEAD_DIM, ROPE_HEAD_DIM,
                       NORM_EPS, compress_kv, cp_d, cs_d, BLOCK_SIZE, state, ssm_d,
                       state_block_table, STATE_BLOCK_SIZE,
                       STATE_BLOCK_SIZE * 2 * full_dim)
            torch.npu.synchronize()
            check_token_state(pos, tok_kv, tok_sc)
            if not compress:
                return
            # reference over state (now includes the decode token at pos, written
            # by compressor_decode_token_state in this launch).
            sr = torch.stack([st[dstate_row(0, p)] for p in range(pos + 1)])
            ref = reference_decode_compress(sr, norm_w, freqs_cis, ratio, overlap,
                                            HEAD_DIM, ROPE_HEAD_DIM, NORM_EPS, test_dtype, pos)
            got = compress_kv.view(-1, HEAD_DIM)[slot].float()
            has_history = (pos + 1) >= 2 * ratio
            tag = 'second' if has_history else 'first-window'
            front = 'prev window' if has_history else 'padded 0/-inf'
            check_close(
                f'{tag} decode compress ({name}) pos={pos}', ref, got,
                f'compress_kv[{slot}] == reference (front {front}); '
                f'ratio={ratio} overlap={overlap} dtype={test_dtype}')

        # ---- #1: verify prefill spill BEFORE any decode (state[0..ql_b-1] ==
        # [kv|score+ape]). Done first so a wrong spill cannot make ref and got
        # agree on garbage (false PASS) downstream. ----
        for k in range(ql_b):
            exp_kv = kv[k]
            exp_sc = score[k] + (ape[k] if ape is not None else 0)
            got = st[dstate_row(0, k)]
            try:
                torch.testing.assert_close(got[:full_dim], exp_kv, atol=STRICT_ATOL, rtol=STRICT_RTOL)
                torch.testing.assert_close(got[full_dim:], exp_sc, atol=STRICT_ATOL, rtol=STRICT_RTOL)
            except AssertionError as e:
                logging.error(f'  FAIL first-window prefill spill ({name}) abs_pos={k}: {e}')
                logging.error(f'    kv max diff: {(got[:full_dim] - exp_kv).abs().max().item()}')
                logging.error(f'    sc+ape max diff: '
                              f'{(got[full_dim:] - exp_sc).abs().max().item()}')
        logging.info(f'  prefill spill verified ({name}): {ql_b} rows [kv|score+ape] '
                     f'at abs_pos 0..{ql_b - 1}')

        # ---- step 1: first compress, pos=ratio-1 (bStart=0, noPredecessor) ----
        decode_step(ratio - 1, compress=True, slot=0)
        # ---- step 2: non-compress steps fill the second compress window's back
        # half [ratio, 2*ratio) with real tokens (pos=ratio..2*ratio-2), so the
        # second compress reads a fully-populated window (no zero/gap rows). ----
        for fill_pos in range(ratio, 2 * ratio - 1):
            decode_step(fill_pos, compress=False, slot=0)
        # ---- step 3: second compress, pos=2*ratio-1 (bStart=ratio, has_history) ----
        decode_step(2 * ratio - 1, compress=True, slot=1)


if __name__ == "__main__":
    # models × work cross-product (mirrors tests/kernels/attention.py).
    for mname, ratio, overlap, dtype in models:
        for category, suffix, spec_builder, flags in work:
            name = f"{category}_{suffix}_{mname}"
            specs = spec_builder(ratio)
            if category in ("prefill", "mix"):
                run_batch(name, ratio, overlap, dtype, specs, flags)
            elif category == "decode":
                if flags.get("first_window"):
                    # first-window is overlap-only (front/back split; see
                    # run_decode_first_window_test docstring).
                    if overlap == 0:
                        continue
                    run_decode_first_window_test(name, ratio, overlap, dtype,
                                                 ape_enabled=flags.get("ape_enabled", True))
                else:
                    run_decode_test(name, ratio, overlap, dtype,
                                    ape_enabled=flags.get("ape_enabled", True),
                                    n_compress=flags.get("n_compress", 1),
                                    n_decode_reqs=flags.get("n_decode_reqs", 1))

    logging.info("compressor test done.")

