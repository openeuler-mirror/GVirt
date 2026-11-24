#!/usr/bin/python3
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

With do_rotate=True the kernel additionally applies x * 1/sqrt(vdim) to the
whole head after rope and before the cache write (scale derived internally).
rotate only fires on the fullHeadLoad path (CXA: remain > 0); the MLA-style
rope-only layout (remain == 0) skips it.
"""

import logging

import torch
import torch_npu  # noqa: F401

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


rt = Runtime(0, size=500)
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

freqs_cis = precompute_freqs_cis(64, MAX_SEQ_LEN, ROPE_THETA)[0:NUM_TOKENS]

for test_dtype, rope_dim, head_dim in CXA_CASES + MLA_CASES:
    for out_interleaved in (False, True):
        for do_rotate in (False, True):
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
            ref_head = ref_head.view(NUM_TOKENS, n_local_heads, vdim).float()

            # rotate (x * 1/sqrt(vdim)) after rope, before cache write. Skipped when
            # remain == 0 (MLA-style) to mirror the kernel's fullHeadLoad guard.
            full_head_load = (head_dim - rope_dim) > 0
            rotate_scale = vdim**-0.5
            if do_rotate and full_head_load:
                ref_head = ref_head * rotate_scale

            # reference vCache: scatter each token's full head to its slot
            ref_v_cache = torch.zeros(num_blocks, block_size, n_local_heads, vdim)
            slot_cpu = slot_mapping.cpu()
            ref_head_cpu = ref_head.cpu()
            for t in range(NUM_TOKENS):
                slot = slot_cpu[t].item()
                b = slot // block_size
                i = slot % block_size
                ref_v_cache[b, i] = ref_head_cpu[t]
            ref_v_cache = ref_v_cache.to(test_dtype)

            # ----- xlite -----
            torch.npu.synchronize()
            rope_complex_and_cache(
                rt,
                n_local_heads,
                head_dim,
                rope_dim,
                offset,
                vdim,
                kv,
                freqs_cis,
                position,
                block_size,
                v_cache,
                slot_mapping,
                out_interleaved=out_interleaved,
                do_rotate=do_rotate,
            )
            torch.npu.synchronize()

            lay = "interleaved" if out_interleaved else "deinterleaved"
            tag = "CXA" if offset != 0 else "MLA-style"
            rot = "+rotate" if do_rotate else ""
            name = f"rope_complex_and_cache ({tag}, {lay}, rope={rope_dim}, head={head_dim}, {test_dtype}{rot})"

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

# ----- Sentinel slot test: -1 = "skip KV write" (vllm dummy/keepalive sentinel) -----
# vllm fills unused slots (execute_dummy_batch keepalive / uniform-decode dummies)
# with -1; as int32 = 0xFFFFFFFF, the flat-slot pointer arithmetic
# ((uint64_t)(slot_idx / block_size) * stride) would OOB-write GM (vector-core
# 507035) without the slot_idx != 0xFFFFFFFF guard in rope_complex_and_cache.h.
# Pre-fill vCache with NaN so a skipped write is distinguishable from a zero
# write: the sentinel token's row stays NaN, the real token's row holds its
# rope output.
logging.info("=== sentinel slot (-1 = skip KV write) ===")
_snt_dtype = torch.bfloat16  # V4.1 is a bf16 model
_snt_head_dim, _snt_rope_dim = 512, 64
_snt_offset = _snt_head_dim - _snt_rope_dim
_snt_n_heads, _snt_block_size, _snt_n_tokens = 1, 16, 2
torch.set_default_dtype(_snt_dtype)
with torch.device("npu"):
    _snt_kv = torch.randn(_snt_n_tokens, _snt_n_heads, _snt_head_dim)
    _snt_kv_ref = _snt_kv.clone()
    _snt_position = torch.arange(_snt_n_tokens, dtype=torch.int64)
    # token 0 -> slot 0 (write); token 1 -> -1 sentinel (skip the paged write)
    _snt_slot_mapping = torch.tensor([0, -1], dtype=torch.int32)
    # NaN sentinel: a skipped write leaves NaN; a real write overwrites it.
    _snt_v_cache = torch.full((1, _snt_block_size, _snt_n_heads, _snt_head_dim), float("nan"))

# reference for token 0 only (token 1 is the skipped sentinel)
_snt_ref = _snt_kv_ref.view(_snt_n_tokens, _snt_n_heads, _snt_head_dim)
_snt_rope_in = _snt_ref[..., -_snt_rope_dim:].contiguous().view(1, _snt_n_tokens, _snt_n_heads, _snt_rope_dim)
_snt_rope_out = apply_rotary_emb(_snt_rope_in, freqs_cis[:_snt_n_tokens], interleaved=False)
_snt_rope_out = _snt_rope_out.view(_snt_n_tokens, _snt_n_heads, _snt_rope_dim)
_snt_ref_head = torch.cat([_snt_ref[..., :-_snt_rope_dim], _snt_rope_out], dim=-1)
_snt_ref_head = _snt_ref_head.view(_snt_n_tokens, _snt_n_heads, _snt_head_dim).float()

# reference vCache, same shape/prefill as the real one: scatter ONLY token 0's
# head into slot 0; the sentinel token 1 (-1) is skipped by the guard, so its row
# stays at the NaN prefill. Comparing the whole vCache with equal_nan=True treats
# the untouched NaN rows as matching (a skipped write leaves NaN; a real write
# overwrites it), and would catch any stray write into the sentinel/unused rows.
_snt_ref_v_cache = torch.full((1, _snt_block_size, _snt_n_heads, _snt_head_dim), float("nan"))
_snt_ref_v_cache[0, 0, 0] = _snt_ref_head[0]  # token 0 -> slot 0 (block 0, row 0)
_snt_ref_v_cache = _snt_ref_v_cache.to(_snt_dtype)

torch.npu.synchronize()
rope_complex_and_cache(
    rt,
    _snt_n_heads,
    _snt_head_dim,
    _snt_rope_dim,
    _snt_offset,
    _snt_head_dim,
    _snt_kv,
    freqs_cis[:_snt_n_tokens],
    _snt_position,
    _snt_block_size,
    _snt_v_cache,
    _snt_slot_mapping,
    out_interleaved=False,
    do_rotate=False,
)
torch.npu.synchronize()

_snt_ok = True
# Whole-vCache compare (equal_nan=True): token 0's slot holds its rope output,
# the sentinel token 1's row and all unused rows stay NaN (guard skipped the
# paged write). A single assertion catches both the real-write correctness and
# any stray write into the sentinel/unused rows.
try:
    torch.testing.assert_close(_snt_v_cache.cpu(), _snt_ref_v_cache.cpu(), atol=1e-5, rtol=1e-3, equal_nan=True)
except AssertionError as e:
    _snt_ok = False
    logging.error(f"sentinel slot (-1 skip KV write) mismatch:\n{e}")

if _snt_ok:
    passed += 1
    logging.info("sentinel slot (-1 skip KV write) executed!")
else:
    logging.error("sentinel slot (-1 skip KV write) FAILED")
total += 1

logging.info(f"\n==== {passed}/{total} cases passed ====")
assert passed == total, f"{total - passed} case(s) failed"
