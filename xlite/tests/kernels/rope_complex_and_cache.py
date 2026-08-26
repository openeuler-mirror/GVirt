#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Single-op test for rope_complex_and_cache (CXA kv path).

Covers the [remain | rope] full-head layout with v-cache writeback, which the
non-cache rope_complex test cannot reach (it forces need_v_cache=false).

The kernel loads the whole head [remain | rope] into UB, rotates the rope
region (last `rope_dim` elements), and writes the whole head back into vCache
at the slot given by slot_mapping, plus an in-place writeback of the rope
region to the input tensor.
"""
from __future__ import absolute_import
import logging
import torch
from xlite._C import Runtime, rope_complex_and_cache

logging.getLogger().setLevel(logging.INFO)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute complex frequencies (matches tests/models/deepseek_v3.py)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Apply rotary embedding. x: (..., rope_dim). Output layout per `interleaved`.

    interleaved=False -> deinterleaved half layout [r0..r(half-1) | i0..i(half-1)]
                        (MLA kv-cache convention, matches xlite out_interleaved=False)
    interleaved=True  -> interleaved layout [r0,i0,r1,i1,...]
                        (torch view_as_real().flatten convention, matches xlite out_interleaved=True)
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis)  # (..., half_dim, 2)
    if interleaved:
        y = y.flatten(-2)
    else:
        y = torch.cat([y[..., 0], y[..., 1]], dim=-1)
    return y.to(dtype)


rt = Runtime(0, 500)
torch.npu.set_device(0)

ROPE_THETA = 10000.0
MAX_SEQ_LEN = 1024
NUM_TOKENS = 10  # = BATCH_SIZE * SEQ_LEN

# CXA layout: full head = [remain | rope], rope at the tail.
# head_dim=512, rope_head_dim=64 -> remain=448, offset=448.
CXA_CASES = [
    (torch.float16, 64, 512),
    (torch.bfloat16, 64, 512),
    # smaller heads to exercise UB sizing and alignment edges
    (torch.float16, 64, 128),
    (torch.bfloat16, 64, 128),
]

# MLA-style rope-only cache layout (remain==0, offset at the head start): the
# vCache stores only the rope region. Guards the fullHeadLoad=false + vcache path.
MLA_CASES = [
    (torch.float16, 64, 64),
    (torch.bfloat16, 64, 64),
]

passed = 0
total = 0

for test_dtype, rope_dim, head_dim in CXA_CASES + MLA_CASES:
    for out_interleaved in (False, True):
        total += 1
        n_local_heads = 1  # CXA kv / MLA pe cache assert nLocalHeads==1 when need_v_cache
        offset = head_dim - rope_dim
        vdim = head_dim  # vCache stores a full head per slot (matches CXA swaKv shape)
        block_size = 16
        # enough blocks for all tokens at distinct slots
        num_blocks = (NUM_TOKENS + block_size - 1) // block_size

        torch.set_default_dtype(test_dtype)
        with torch.device("npu"):
            # input head: [remain | rope], shape (num_tokens, n_local_heads, head_dim)
            kv = torch.randn(NUM_TOKENS, n_local_heads, head_dim)
            kv_ref = kv.clone()

            freqs_cis = precompute_freqs_cis(rope_dim, MAX_SEQ_LEN, ROPE_THETA)[0:NUM_TOKENS]

            # position per token (0..NUM_TOKENS-1) and identity slot mapping
            position = torch.arange(NUM_TOKENS, dtype=torch.int64)
            slot_mapping = torch.arange(NUM_TOKENS, dtype=torch.int32)

            # vCache: (num_blocks, block_size, n_local_heads, vdim), zero-initialized
            v_cache = torch.zeros(num_blocks, block_size, n_local_heads, vdim)

        # ----- reference (torch) -----
        # apply rope to the last rope_dim elements, in-place on a full-head copy.
        kv_ref_3d = kv_ref.view(NUM_TOKENS, n_local_heads, head_dim)
        rope_part = kv_ref_3d[..., -rope_dim:].contiguous()  # (T, H, rope_dim)
        # apply_rotary_emb expects (bsz, seqlen, n_local_heads, rope_dim)
        rope_in = rope_part.view(1, NUM_TOKENS, n_local_heads, rope_dim)
        rope_out = apply_rotary_emb(rope_in, freqs_cis, interleaved=out_interleaved)
        rope_out = rope_out.view(NUM_TOKENS, n_local_heads, rope_dim)
        # rebuild full head: remain (unchanged) + rotated rope
        if rope_dim == head_dim:
            ref_head = rope_out  # rope-only, no remain
        else:
            remain_part = kv_ref_3d[..., :-rope_dim]
            ref_head = torch.cat([remain_part, rope_out], dim=-1)
        ref_head = ref_head.view(NUM_TOKENS, n_local_heads, vdim)

        # reference vCache: scatter each token's full head to its slot
        ref_v_cache = torch.zeros(num_blocks, block_size, n_local_heads, vdim)
        for t in range(NUM_TOKENS):
            slot = slot_mapping[t].item()
            b = slot // block_size
            i = slot % block_size
            ref_v_cache[b, i] = ref_head[t]

        # ----- xlite -----
        torch.npu.synchronize()
        rope_complex_and_cache(rt, n_local_heads, head_dim, rope_dim, offset, vdim, kv, freqs_cis,
                               position, block_size, v_cache, slot_mapping,
                               out_interleaved=out_interleaved)
        torch.npu.synchronize()

        lay = "interleaved" if out_interleaved else "deinterleaved"
        tag = "CXA" if offset != 0 else "MLA-style"
        name = f"rope_complex_and_cache ({tag}, {lay}, rope={rope_dim}, head={head_dim}, {test_dtype})"

        ok = True
        try:
            # vCache matches reference (whole head per slot).
            # Note: the op does NOT write back to the input kv tensor in-place
            # (op.cpp passes output_ptr=nullptr), so only vCache is validated.
            torch.testing.assert_close(v_cache.cpu(), ref_v_cache.cpu(), atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            ok = False
            logging.error(f"{name} vCache mismatch:\n{e}")

        if ok:
            passed += 1
            logging.info(f"{name} executed!")
        else:
            logging.error(f"{name} FAILED")

logging.info(f"\n==== {passed}/{total} cases passed ====")
assert passed == total, f"{total - passed} case(s) failed"
