#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
# ===============================================================================
"""
flash_mla_v2 sparse softmax (topK>0) alignment test.

GLM-5.3-Flash kpool widens the topk row to a logical 2051 slots, then pads
it to topkWidth = 2176 so the sparse softmax in softmax_attn_aiv.h always
walks full 128-slot mask blocks; slots [2051, 2176) are -1 padding. Multi-tile
geometry (cached+query > TILE_SIZE) adds the online-softmax merge path on top.

Cases (top_k/topk_stride = 2048/2048 control, 2051/2176 GLM5):
  A. decode (2, [4000, 4000], [1, 1])  -> calcLen > top_k activates the
     sparse softmax branch; seqIdx stays 0.
  B. prefill (1, [2100], [64])         -> single-tile 2176-row/2051-select.
  C. prefill (1, [9000], [64])         -> cached+query crosses TILE_SIZE=8192,
     kvNum>=2; mirrors the 9k standalone crash geometry.
  D. neg_tail variants                 -> slots [2048, topkWidth) are -1
     tail/padding (GLM5 kpool layout).
  E. decode long context (batch>=4, ~21k cached) -> kvNum=3 tiles +
     cross-core online-softmax ring merge, the real GLM5 decode path
     (model.cpp mla_v2 call). Hit patterns "tail"/"two-blocks" cluster
     hits like the score-clustered indexer topk, so trailing tiles can
     be all-miss and clusters straddle tile boundaries.
"""

import argparse
import logging
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

from tests.models.weight_utils import matrix_nd2nz
from xlite._C import Runtime, mla_v2

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, size=3000)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True

BLOCK_SIZE = 128
TILE_SIZE = 8192
ENABLE_FLASH = True
WEIGHT_NZ = True

# glm5 shape family from mla.py
N_HEADS, ROPE, NOPE, V_DIM, KV_LORA = 8, 64, 192, 256, 512
DTYPE = torch.bfloat16

# (name, cached_lens, query_lens, neg_tail, tile1_only, hit_pattern)
WORKS = [
    ("decode-b2-long", [4000, 4000], [1, 1], False, False, "rand"),
    ("prefill-64q", [2100], [64], False, False, "rand"),
    # cached+query crosses TILE_SIZE=8192 -> kvNum>=2 (multi-tile merge path)
    ("prefill-64q-multitile", [9000], [64], False, False, "rand"),
    # GLM5 PKI tail: slots [2048, topkWidth) are -1 tail/padding
    ("prefill-64q-negtail", [2100], [64], True, False, "rand"),
    ("prefill-64q-multitile-negtail", [9000], [64], True, False, "rand"),
    # all hits in tile1: tile2 is an all-masked row (dead-tile merge path)
    ("prefill-64q-multitile-tile1only", [9000], [64], False, True, "rand"),
    # only part of the rows straddle the tile boundary (cross-tile merge)
    ("prefill-100q-straddle", [8100], [100], False, False, "rand"),
    # decode trigger geometry: batch>=4 x ~21k cached prompt
    ("decode-b4-21k", [21000] * 4, [1] * 4, False, False, "rand"),
    ("decode-b4-21k-negtail", [21000] * 4, [1] * 4, True, False, "rand"),
    ("decode-b20-21k", [21000] * 20, [1] * 20, False, False, "rand"),
    ("decode-b20-21k-negtail", [21000] * 20, [1] * 20, True, False, "rand"),
    # real indexer topk is score-clustered, not uniform; these patterns
    # produce trailing all-miss tiles / boundary-straddling hit clusters
    ("decode-b20-21k-idxtail", [21000] * 20, [1] * 20, False, False, "tail"),
    # known bf16 round-placement edge (see KNOWN_EDGE_RELTOL)
    ("decode-b20-21k-idxtwoblock", [21000] * 20, [1] * 20, False, False, "two-blocks"),
    # two leading all-miss tiles fold to newSum=0; hits at tile3 must
    # recover from the empty state (zero-sum guard regression)
    ("prefill-64q-2miss-hit", [18600], [64], False, False, "late-tile"),
    # -1 padding row, no hit at all: output must be a zero row
    ("decode-b4-21k-allmiss", [21000] * 4, [1] * 4, False, False, "all-miss"),
]

# (top_k logical selection width, topk_stride physical row width)
TOPK_CASES = [(2048, 2048), (2051, 2176)]

# Both sides sit ~2.5% from the fp32 truth on these ill-conditioned elements
# (bf16 round placement), so the default 5% tolerance is too tight there.
KNOWN_EDGE_RELTOL = {
    "decode-b20-21k-idxtwoblock": 8e-2,
}


def rms_norm_last_dim(x):
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
    return x / rms


def max_blocks(query_lens, cached_lens, block_size):
    max_sum = max(a + b for a, b in zip(query_lens, cached_lens))
    return (max_sum + block_size - 1) // block_size


def sample_pool_indices(pool_cap, count, pattern):
    """Draw one row of `count` indices from pool positions [0, pool_cap).

    "rand": uniform. "tail": the highest 2*count positions. "two-blocks": the
    start of tile2 plus the last 2*count. "late-tile": one block at the start
    of tile3, so tiles 1-2 are all-miss. Requires pool_cap >= count
    ("late-tile" >= 2*TILE_SIZE + count).
    """
    assert pool_cap >= count, "hit patterns require pool_cap >= count"
    if pattern == "tail":
        pool = torch.arange(max(0, pool_cap - 2 * count), pool_cap, device="npu")
    elif pattern == "two-blocks":
        pool = torch.cat(
            [
                torch.arange(TILE_SIZE, TILE_SIZE + count, device="npu"),
                torch.arange(pool_cap - 2 * count, pool_cap - count, device="npu"),
            ]
        )
    elif pattern == "late-tile":
        assert pool_cap >= 2 * TILE_SIZE + count, "late-tile pattern requires pool_cap >= 2*TILE_SIZE + count"
        pool = torch.arange(2 * TILE_SIZE, 2 * TILE_SIZE + count, device="npu")
        return pool
    elif pattern == "all-miss":
        # -1 everywhere: the kernel-side "no hit in any kv tile" row shape.
        return torch.full((count,), -1, dtype=torch.int64, device="npu")
    else:
        return torch.randperm(pool_cap, device="npu")[:count]
    pick = torch.randperm(pool.numel(), device="npu")[:count]
    return pool[pick]


def generate_topk_indices(cached_lens_list, query_len_list, topk_stride, tile1_only, neg_tail, pattern):
    """Per-token topk index rows [total_query, topk_stride], int32, ascending.

    Rows are padded with -1 when the pool is smaller than topk_stride so the
    boolean-src scatter never sees an out-of-range index. With neg_tail slots
    [2048, topk_stride) are -1 (GLM5 kpool tail + 128-slot row padding), applied
    after the ascending sort.
    """
    rows = []
    for clen, qlen in zip(cached_lens_list, query_len_list):
        for q_idx in range(qlen):
            valid_len = clen + q_idx + 1
            pool_cap = min(valid_len, TILE_SIZE) if tile1_only else valid_len
            if pool_cap < topk_stride:
                perm = torch.randperm(pool_cap, device="npu")
                perm = torch.cat([perm, perm.new_full((topk_stride - pool_cap,), -1)])
            else:
                perm = sample_pool_indices(pool_cap, topk_stride, pattern)
            rows.append(perm)
    indices = torch.stack(rows)
    indices, _ = torch.sort(indices, dim=-1)
    indices = indices.to(dtype=torch.int32)
    if neg_tail:
        tail_start = 2048 if topk_stride > 2048 else topk_stride - 3
        indices[:, tail_start:] = -1
    return indices


def run_case(
    name,
    cached_lens_list,
    query_len_list,
    top_k,
    topk_stride,
    neg_tail=False,
    tile1_only=False,
    seed=0,
    pattern="rand",
):
    torch.manual_seed(seed)
    batch = len(cached_lens_list)
    max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE)
    max_seq_len = max_num_blocks * BLOCK_SIZE
    total_query_len = sum(query_len_list)
    scale = (NOPE + ROPE) ** -0.5

    torch.set_default_dtype(DTYPE)
    with torch.device("npu"):
        qWithQr = rms_norm_last_dim(torch.randn(total_query_len, N_HEADS, NOPE + ROPE))
        k_cache = rms_norm_last_dim(torch.randn(batch, max_seq_len, KV_LORA).clamp(0.2, 1))
        pe_cache = rms_norm_last_dim(torch.randn(batch, max_seq_len, ROPE))
        wkvb = torch.randn(N_HEADS * (NOPE + V_DIM), KV_LORA).clamp(0.2, 1)
        wkvb = wkvb.view(N_HEADS, NOPE + V_DIM, KV_LORA)

        kvcache_block_num = max_num_blocks * batch
        k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, KV_LORA)
        pe_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, ROPE)

        query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_lens_np = np.array(query_len_list)
        query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
        query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(max_num_blocks, dtype=np.uint32)
        block_tables_array = batch_indices * max_num_blocks + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

        qr_xlite = torch.empty(total_query_len, N_HEADS, ROPE)
        wukT_xlite = wkvb[:, :NOPE].reshape(N_HEADS, NOPE, KV_LORA).contiguous()
        wuv_xlite = wkvb[:, -V_DIM:].transpose(1, 2).contiguous()
        wukT_xlite = matrix_nd2nz(wukT_xlite.reshape(N_HEADS * NOPE, KV_LORA))
        wuv_xlite = matrix_nd2nz(wuv_xlite.reshape(N_HEADS * KV_LORA, V_DIM))

    # qr split per sample (rope part of qWithQr)
    offset = 0
    for i in range(batch):
        qlen = query_len_list[i]
        chunk = qWithQr[offset : offset + qlen].unsqueeze(0)
        _, q_rope = chunk.split([NOPE, ROPE], dim=-1)
        qr_xlite[offset : offset + qlen] = q_rope.squeeze(0)
        offset += qlen

    # xlite paged cache fill
    for i in range(batch):
        clen = cached_lens_list[i]
        qlen = query_len_list[i]
        total_len = clen + qlen
        num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
        for b in range(num_blocks_needed):
            s, e = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, total_len)
            k_cache_xlite[i * max_num_blocks + b, : e - s] = k_cache[i, s:e]
            pe_cache_xlite[i * max_num_blocks + b, : e - s] = pe_cache[i, s:e]

    topk_indices = generate_topk_indices(cached_lens_list, query_len_list, topk_stride, tile1_only, neg_tail, pattern)

    # torch reference: topk-masked softmax per sample. Scatter only valid
    # indices; clamping -1 to column 0 would clear a hit on token 0.
    outputs_ref = []
    q_offset = 0
    indices_offset = 0
    for i in range(batch):
        qlen = query_len_list[i]
        clen = cached_lens_list[i]
        total_len = clen + qlen

        qWithQr_chunk = qWithQr[q_offset : q_offset + qlen].unsqueeze(0)
        q_offset += qlen
        q_nope, q_rope = qWithQr_chunk.split([NOPE, ROPE], dim=-1)
        q_absorb = torch.einsum("bshd,hdc->bshc", q_nope, wkvb[:, :NOPE])
        qkc = torch.einsum("bshc,btc->bsht", q_absorb, k_cache[i : i + 1, :total_len])
        qkr = torch.einsum("bshr,btr->bsht", q_rope, pe_cache[i : i + 1, :total_len])
        scores = qkc + qkr
        scores = scores * scale

        sample_indices = topk_indices[indices_offset : indices_offset + qlen]
        indices_offset += qlen

        hit = torch.zeros((qlen, total_len), dtype=torch.bool, device="npu")
        valid = sample_indices >= 0
        rows = torch.arange(qlen, device="npu").unsqueeze(1).expand_as(sample_indices)
        hit[rows[valid], sample_indices[valid]] = True
        index_mask = torch.where(hit, 0.0, float("-inf")).unsqueeze(0)

        cols = torch.arange(total_len, device="npu").unsqueeze(0)
        rows = torch.arange(qlen, device="npu").unsqueeze(1)
        mask = torch.where(cols > clen + rows, float("-inf"), 0.0)

        scores = scores + mask.unsqueeze(0).unsqueeze(1).permute(0, 2, 1, 3)
        scores = scores + index_mask.unsqueeze(2)
        scores = torch.softmax(scores, dim=-1)
        o_absorb = torch.einsum("bsht,btc->bshc", scores, k_cache[i : i + 1, :total_len])
        x = torch.einsum("bshc,hdc->bshd", o_absorb, wkvb[:, -V_DIM:])
        # All-miss rows fold to the empty state (output 0); a torch softmax
        # over an all-masked row would be NaN, so mirror the kernel contract.
        no_hit = (sample_indices >= 0).sum(dim=1) == 0
        if bool(no_hit.any()):
            x[no_hit] = 0
        outputs_ref.append(x.squeeze(0))
    ref = torch.cat(outputs_ref, dim=0)

    output_xlite = torch.zeros(total_query_len, N_HEADS, V_DIM, device="npu", dtype=DTYPE)
    torch.npu.synchronize()
    mla_v2(
        rt,
        qWithQr,
        qr_xlite,
        k_cache_xlite,
        pe_cache_xlite,
        wukT_xlite,
        wuv_xlite,
        output_xlite,
        query_start_loc,
        query_lens,
        cached_lens,
        block_tables,
        N_HEADS,
        ROPE,
        NOPE,
        V_DIM,
        KV_LORA,
        BLOCK_SIZE,
        batch,
        scale,
        topk_indices,
        top_k=top_k,
        topk_stride=topk_stride,
        nz=WEIGHT_NZ,
        enable_flash_attention=ENABLE_FLASH,
        tile_size_of_cached_kv=TILE_SIZE,
    )
    torch.npu.synchronize()

    try:
        # NaN > tol is False: check isnan first or 0/0 outputs pass silently.
        n_nan = int(torch.isnan(output_xlite.float()).sum().item())
        if n_nan:
            logging.error(
                "FAIL %s topk_stride=%d top_k=%d seed=%d: %d NaN elements in output",
                name,
                topk_stride,
                top_k,
                seed,
                n_nan,
            )
            return False
        diff = (output_xlite.float() - ref.float()).abs()
        tol = 1e-5 + KNOWN_EDGE_RELTOL.get(name, 5e-2) * ref.float().abs()
        bad = diff > tol
        n_bad = int(bad.sum().item())
        if n_bad == 0:
            logging.info(
                "PASS %s topk_stride=%d top_k=%d seed=%d (max_seq_len=%d)",
                name,
                topk_stride,
                top_k,
                seed,
                max_seq_len,
            )
            return True
        coords = bad.nonzero()[:8]
        pairs = {}
        for c in bad.nonzero().tolist():
            key = (c[0], c[1])
            pairs[key] = pairs.get(key, 0) + 1
        msg = [
            f"Mismatched elements: {n_bad} / {bad.numel()}",
            "  per (seq,head): " + ", ".join(f"({i},{h})x{n}" for (i, h), n in sorted(pairs.items())[:20]),
        ]
        for c in coords.tolist():
            i, h, d = c
            msg.append(f"  ({i},{h},{d}) xlite={output_xlite[i, h, d].item():.6f} ref={ref[i, h, d].item():.6f}")
        logging.error(
            "FAIL %s topk_stride=%d top_k=%d seed=%d (max_seq_len=%d):\n%s",
            name,
            topk_stride,
            top_k,
            seed,
            max_seq_len,
            "\n".join(msg),
        )
        return False
    except Exception as e:
        logging.error(
            "FAIL %s topk_stride=%d top_k=%d seed=%d (max_seq_len=%d): %s",
            name,
            topk_stride,
            top_k,
            seed,
            max_seq_len,
            repr(e)[:2000],
        )
        return False


def main():
    # One process per case: an aicore exception poisons the device for the rest.
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None, help="run a single work case by name")
    ap.add_argument("--topk-stride", type=int, default=None, help="run one physical topk row width")
    ap.add_argument("--seed", type=int, default=0, help="rng seed for determinism")
    args = ap.parse_args()

    results = []
    sweep = [t for t in TOPK_CASES if not args.topk_stride or t[1] == args.topk_stride]
    if args.topk_stride and not sweep:
        ap.error(f"unsupported --topk-stride {args.topk_stride}; choose from {[t[1] for t in TOPK_CASES]}")
    for w in WORKS:
        name, cached, query, neg_tail, tile1_only, pattern = w
        if args.work and name != args.work:
            continue
        for top_k, topk_stride in sweep:
            try:
                ok = run_case(
                    name,
                    cached,
                    query,
                    top_k,
                    topk_stride,
                    neg_tail,
                    tile1_only,
                    args.seed,
                    pattern,
                )
            except Exception as e:  # aicore exception surfaces here
                logging.error(
                    "CRASH %s top_k=%d topk_stride=%d: %s",
                    name,
                    top_k,
                    topk_stride,
                    repr(e)[:500],
                )
                ok = False
            results.append((name, top_k, topk_stride, ok))
    logging.info("==== SUMMARY ====")
    for name, top_k, topk_stride, ok in results:
        logging.info(
            "%s top_k=%d topk_stride=%d -> %s",
            name,
            top_k,
            topk_stride,
            "PASS" if ok else "FAIL/CRASH",
        )
    bad = [r for r in results if not r[3]]
    # A stray atexit TypeError after SUMMARY is cosmetic; verdicts above are final.
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
