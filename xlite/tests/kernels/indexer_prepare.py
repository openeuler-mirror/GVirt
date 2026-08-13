#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Unit test for ``csrc/kernels/indexer_prepare.h``.

Fused DSA indexer prepare kernel (``norm_ropex_cache_muls``):
  * LayerNorm over ``kw[:, :index_head_dim]`` (in place).
  * Interleaved (GPT-J) RoPE on the ``rope_head_dim`` prefix of ``kw`` and
    scatter of the post-LN ``index_head_dim`` slice into a paged
    ``index_k_cache`` (skipped when ``block_size == 0``).
  * When ``is_long``:
      - Interleaved RoPE on the ``rope_head_dim`` prefix of each of the
        ``index_n_heads`` heads of ``q`` (in place).
      - ``muls`` on ``kw[:, index_head_dim:index_head_dim + index_n_heads]``
        by ``scale`` (in place), but only for tokens past ``top_k``.

The reference is computed with plain torch (LayerNorm + complex-pair RoPE).
"""

from __future__ import absolute_import

import logging
import math

import torch
import torch.nn.functional as F

from xlite._C import Runtime, indexer_prepare

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

BLOCK_SIZE = 128
NORM_EPS = 1e-6
ROPE_THETA = 10000.0
# Must exceed the largest position id exercised by `work` (max cached+query
# ≈ 13542). Sized to the next power of two for a tidy freqs table.
MAX_SEQ_LEN = 16384

# model configurations: name, index_head_dim, index_n_heads, rope_head_dim, dtype
models = [
    ("indexer_128_64", 128, 64, 64, torch.bfloat16),
    ("indexer_128_64_fp16", 128, 64, 64, torch.float16),
]

# work configurations: batch_size, cached_lens, query_lens (mirrors indexer_topk)
work = [
    (1, [0], [1]),
    (1, [0], [4]),
    (1, [0], [30]),
    (1, [0], [128]),
    (1, [2049], [1]),
    (1, [3542], [1]),
    (1, [0], [8124]),
    (1, [9728], [1]),
    (1, [0], [13542]),
    (1, [10], [4000]),
    (2, [0] * 2, [4, 8]),
    (2, [8123, 0], [1, 8231]),
    (4, [5012, 127, 2189, 500], [4, 2, 6, 8]),
]

# topK values to exercise (must be <= 2048). Only affects the ``muls`` gate on
# the long path: tokens with a position id > top_k are scaled.
topk_values = [512, 2048]

# absolute/relative tolerances for torch.testing.assert_close. bf16 has ~3
# digits of precision; the fused norm+rope path accumulates FP32 round-trips,
# so we keep the bounds loose but tight enough to catch real regressions.
ATOL = 2e-2
RTOL = 1.5e-2


def precompute_freqs_cis(dim: int, end: int, theta: float = ROPE_THETA):
    """Return complex freqs_cis of shape ``[end, dim // 2]`` on NPU.

    The kernel consumes an interleaved (GPT-J) freqs table laid out as
    ``[cos(t,0), cos(t,2), ... | sin(t,0), sin(t,2), ...]`` (TTTWWW) per
    token. ``torch.polar`` produces exactly this when the pairs are treated
    as complex numbers, so a single complex tensor is what both the kernel's
    ``vreducev2`` cos/sin split and this reference expect.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) RoPE.

    ``x``: ``[..., rope_dim]`` with adjacent pairs ``(x0, x1), (x2, x3), ...``
    ``freqs_cis``: ``[...]`` or ``[..., rope_dim // 2]`` complex, broadcastable.
    Returns ``[..., rope_dim]`` in the same interleaved layout.
    """
    dtype = x.dtype
    *lead, rope_dim = x.shape
    x = torch.view_as_complex(x.float().reshape(*lead, rope_dim // 2, 2))
    if freqs_cis.dim() < x.dim():
        freqs_cis = freqs_cis.unsqueeze(-2)
    freqs_cis = freqs_cis.expand_as(x).to(x.dtype)
    y = torch.view_as_real(x * freqs_cis)
    # interleave real/imag back to (x0', x1', x2', x3', ...)
    y = torch.cat([y[..., 0], y[..., 1]], dim=-1)
    return y.to(dtype)


def run_test(
    name, index_head_dim, index_n_heads, rope_head_dim, test_dtype, batch, cached_lens_list, query_len_list, topK
):
    num_tokens = sum(query_len_list)
    total_dim = index_head_dim + index_n_heads
    scale = 1.0 / math.sqrt(index_n_heads * index_head_dim)

    # is_long mirrors model.cpp: when the worst-case paged length exceeds
    # topK, the long path (q-RoPE + muls) is taken.
    max_num_blocks = max((cl + ql + BLOCK_SIZE - 1) // BLOCK_SIZE for cl, ql in zip(cached_lens_list, query_len_list))
    max_seq_len = max_num_blocks * BLOCK_SIZE
    is_long = (max_num_blocks * BLOCK_SIZE) > topK

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        kw = torch.randn(num_tokens, total_dim) / 10
        kw_ref = kw.clone()

        k_norm = torch.randn(index_head_dim) / 10
        k_norm_bias = torch.randn(index_head_dim) / 10

        # positions: per-token absolute sequence position within each sample.
        # Sample i's tokens occupy positions [cached_lens[i], cached+qlen).
        pos_list = []
        for cl, ql in zip(cached_lens_list, query_len_list):
            pos_list.extend(range(cl, cl + ql))
        position = torch.tensor(pos_list, dtype=torch.int64)

        freqs_cis = precompute_freqs_cis(rope_head_dim, MAX_SEQ_LEN, ROPE_THETA)

        # paged indexer k-cache: one block row per token (slot_mapping maps
        # token -> flat slot; with BLOCK_SIZE rows we pack tokens across
        # blocks). block_size=0 disables caching, so keep BLOCK_SIZE>0.
        block_num = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
        index_k_cache = torch.zeros(block_num, BLOCK_SIZE, index_head_dim, dtype=test_dtype)
        # slot = block_idx * BLOCK_SIZE + in_block_offset; the per-token
        # scatter in the reference uses the same flat slot arithmetic.
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)

        if is_long:
            q = torch.randn(num_tokens, index_n_heads * index_head_dim) / 10
            q_ref = q.clone()
        else:
            q = torch.zeros(num_tokens, index_n_heads * index_head_dim, dtype=test_dtype, device="npu")

    # ---------------- reference ----------------
    # When block_size>0 the kernel writes the LayerNorm+RoPE result into the
    # paged index_k_cache (output_norm_gm = kcache + slot*norm_dim) and does
    # NOT modify kw in place. kw[:, :index_head_dim] is read as the norm input
    # and is otherwise left untouched. Build the cache reference accordingly.
    freqs_per_token = freqs_cis.cpu()[pos_list].to("npu")

    # 1) LayerNorm over kw[:, :index_head_dim].
    ln_slice = F.layer_norm(
        kw_ref[:, :index_head_dim], (index_head_dim,), k_norm.float(), k_norm_bias.float(), NORM_EPS
    )

    # 2) RoPE on the rope_head_dim prefix of the LN result.
    pe_rot = apply_rotary_emb(
        ln_slice[:, :rope_head_dim].reshape(num_tokens, rope_head_dim).contiguous(), freqs_per_token
    )
    ln_rot = ln_slice.clone()
    ln_rot[:, :rope_head_dim] = pe_rot

    # 3) Scatter the post-LN (post-rope) index_head_dim slice into the cache.
    index_k_cache_ref = torch.zeros(block_num, BLOCK_SIZE, index_head_dim, dtype=test_dtype, device="npu")
    for i in range(num_tokens):
        slot = int(slot_mapping[i].item())
        b = slot // BLOCK_SIZE
        off = slot % BLOCK_SIZE
        index_k_cache_ref[b, off] = ln_rot[i]

    # 4) Long path: q-RoPE (per-head, on the rope_head_dim prefix) in place;
    #    muls scales kw[:, index_head_dim:index_head_dim + index_n_heads] (the
    #    per-token head weights that indexer_topk later reads at offset
    #    index_head_dim, see csrc/kernels/indexer_topk.h `wOffset = ... +
    #    headDim`) IN PLACE on kw, only for tokens whose position > topK. The
    #    kernel gates with `ipos > top_k` and runs the scale in FP32 before
    #    casting back to dtype, so match that here.
    if is_long:
        need_muls_mask = torch.tensor([p > topK for p in pos_list], device="npu", dtype=torch.float32).view(
            num_tokens, 1
        )
        tail = kw_ref[:, index_head_dim : index_head_dim + index_n_heads]
        scaled = (tail.float() * scale).to(test_dtype)
        kw_ref[:, index_head_dim : index_head_dim + index_n_heads] = torch.where(need_muls_mask.bool(), scaled, tail)

        q_heads = q_ref.reshape(num_tokens, index_n_heads, index_head_dim)
        q_rot_slice = apply_rotary_emb(q_heads[..., :rope_head_dim].contiguous(), freqs_per_token)
        q_heads[..., :rope_head_dim] = q_rot_slice
        q_ref = q_heads.reshape(num_tokens, index_n_heads * index_head_dim)

    # ---------------- xlite ----------------
    torch.npu.synchronize()
    indexer_prepare(
        rt,
        kw,
        k_norm,
        k_norm_bias,
        freqs_cis,
        position,
        index_head_dim,
        index_n_heads,
        rope_head_dim,
        BLOCK_SIZE,
        index_k_cache,
        slot_mapping,
        NORM_EPS,
        q,
        scale,
        topK,
        is_long,
    )
    torch.npu.synchronize()

    logging.info("=" * 80)
    logging.info(
        "indexer_prepare %s (hd=%d, nh=%d, rope=%d, %s) work "
        "(batch=%d, cached=%s, query=%s, topK=%d, is_long=%s) executed!",
        name,
        index_head_dim,
        index_n_heads,
        rope_head_dim,
        test_dtype,
        batch,
        cached_lens_list,
        query_len_list,
        topK,
        is_long,
    )

    ok = True
    # kw is read-only input when caching is on; it must be unchanged.
    try:
        torch.testing.assert_close(kw, kw_ref, atol=ATOL, rtol=RTOL)
    except AssertionError as e:
        ok = False
        logging.info("-" * 80)
        logging.error(f"kw modified ({name}, long={is_long}): {e}")

    # index_k_cache: compare the rows we actually wrote.
    try:
        torch.testing.assert_close(index_k_cache, index_k_cache_ref, atol=ATOL, rtol=RTOL)
    except AssertionError as e:
        ok = False
        logging.info("-" * 80)
        logging.error(f"index_k_cache mismatch ({name}, long={is_long}): {e}")

    if is_long:
        try:
            torch.testing.assert_close(q, q_ref, atol=ATOL, rtol=RTOL)
        except AssertionError as e:
            ok = False
            logging.info("-" * 80)
            logging.error(f"q mismatch ({name}, long={is_long}): {e}")

    if ok:
        logging.info(f"All tensors match for {name} (topK={topK}, is_long={is_long})!")


def main():
    for name, index_head_dim, index_n_heads, rope_head_dim, test_dtype in models:
        for batch, cached_lens_list, query_len_list in work:
            assert len(cached_lens_list) == batch
            assert len(query_len_list) == batch
            for topK in topk_values:
                run_test(
                    name,
                    index_head_dim,
                    index_n_heads,
                    rope_head_dim,
                    test_dtype,
                    batch,
                    cached_lens_list,
                    query_len_list,
                    topK,
                )


if __name__ == "__main__":
    main()
