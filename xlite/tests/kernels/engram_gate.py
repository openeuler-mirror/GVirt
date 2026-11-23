#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Correctness test for the engram_gate kernel: the DeepSeek-V4.1 engram
post-wkv fused gate + residual inject (AIV), C++ counterpart of the triton kernel in
vllm_ascend/ops/triton/engram_gate.py.

For each (token, hc-branch) pair the kernel computes, over DIM:
  rstd = rsqrt(mean(h^2)+eps) * rsqrt(mean(key^2)+eps)
  dot  = sum(h * (qw*kw) * key) * rstd * dim^-0.5
  gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
  gate = mask ? gate : 0
  out  = hResid + gate * value

The torch reference mirrors engram_gate.py exactly (un-fused FP32 order), and
serves as the oracle since the triton kernel itself is not available on NPU.
"""

import torch
import torch_npu

from xlite._C import Runtime, engram_gate


def torch_engram_gate(hidden, kv, q_weight, k_weight, token_mask, eps):
    """Pure-torch oracle matching vllm_ascend/ops/triton/engram_gate.py:34-48.

    Run on CPU: the kernel inputs are bf16 NPU tensors; we compute the reference
    in fp32 on CPU (copysign is not an NPU op, and the oracle need not be fast).
    """
    dev = "cpu"
    h = hidden.to(dev).float()  # [T, HC, DIM]
    T, HC, DIM = h.shape
    key = kv.to(dev).float()[:, : HC * DIM].view(T, HC, DIM)
    value = kv.to(dev).float()[:, HC * DIM :]  # [T, DIM]
    qw = q_weight.to(dev).float()  # [HC, DIM]
    kw = k_weight.to(dev).float()  # [HC, DIM]
    mask = token_mask.to(dev).view(T, 1).float()  # [T, 1]

    # rstd = rsqrt(mean(h^2) + eps) * rsqrt(mean(key^2) + eps)
    rstd_h = torch.rsqrt((h * h).mean(-1) + eps)  # [T, HC]
    rstd_k = torch.rsqrt((key * key).mean(-1) + eps)  # [T, HC]
    rstd = rstd_h * rstd_k  # [T, HC]

    # dot = sum(h * (qw * kw) * key) * rstd * dim^-0.5
    qk = qw * kw  # [HC, DIM]
    dot = (h * qk * key).sum(-1) * rstd * (DIM**-0.5)  # [T, HC]

    # gate = sigmoid(copysign(sqrt(max(|dot|, 1e-6)), dot))
    mag = torch.sqrt(torch.clamp(dot.abs(), min=1e-6))
    # copysign: +mag when dot >= 0 (incl. +0), -mag when dot < 0. torch.copysign
    # honors the sign bit; for plain dot == 0 this yields +mag, matching the
    # triton bitcast(&0x80000000) which is 0 for +0.0.
    signed_mag = torch.copysign(mag, dot)
    gate = torch.sigmoid(signed_mag)  # [T, HC]

    # gate = mask ? gate : 0
    gate = gate * mask  # [T, HC]

    # out = h + gate * value
    out = h + gate.unsqueeze(-1) * value.unsqueeze(1)  # [T, HC, DIM]
    return out.to(torch.bfloat16).to(hidden.device)


def run_test(rt, num_tokens, hc_mult, dim, dtype, msg, mask_mode="all"):
    torch.manual_seed(1234 + num_tokens * 7 + hc_mult * 13 + dim)
    hidden = torch.randn(num_tokens, hc_mult, dim, dtype=dtype)
    # kv = [key(HC * DIM) + value(DIM)]; small values keep dot in a sane range.
    kv = torch.randn(num_tokens, (hc_mult + 1) * dim, dtype=dtype) * 0.1
    qw = torch.randn(hc_mult, dim, dtype=dtype) * 0.1
    kw = torch.randn(hc_mult, dim, dtype=dtype) * 0.1

    if mask_mode == "all":
        mask = torch.ones(num_tokens, dtype=torch.int32)
    elif mask_mode == "none":
        mask = torch.zeros(num_tokens, dtype=torch.int32)
    else:  # "partial"
        mask = (torch.arange(num_tokens) % 2 == 0).to(torch.int32)

    standard = torch_engram_gate(hidden, kv, qw, kw, mask, 1e-6)

    out = torch.empty(num_tokens, hc_mult, dim, dtype=dtype)
    torch.npu.synchronize()
    # Unrotated path: hidden_residual aliases hidden's storage (explicit pass).
    engram_gate(
        rt,
        hidden,
        kv=kv,
        q_weight=qw,
        k_weight=kw,
        mask=mask,
        out=out,
        hc_mult=hc_mult,
        dim=dim,
        eps=1e-6,
        hidden_residual=hidden,
    )
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(standard, out, rtol=1e-2, atol=1e-3)
        print(f"{msg}: PASS")
    except AssertionError as e:
        print(f"{msg}: FAILED")
        print(f"{e}")
        raise

    # Default path: omit hidden_residual (nullopt) -> host aliases hidden. Must
    # match the explicit-pass result above (same unrotated semantics).
    out2 = torch.empty(num_tokens, hc_mult, dim, dtype=dtype)
    torch.npu.synchronize()
    engram_gate(
        rt,
        hidden,
        kv=kv,
        q_weight=qw,
        k_weight=kw,
        mask=mask,
        out=out2,
        hc_mult=hc_mult,
        dim=dim,
        eps=1e-6,
    )
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(standard, out2, rtol=1e-2, atol=1e-3)
        print(f"{msg}/nullopt: PASS")
    except AssertionError as e:
        print(f"{msg}/nullopt: FAILED")
        print(f"{e}")
        raise


def torch_engram_gate_rotated(hidden, kv, q_weight, k_weight, token_mask, eps, rotation):
    """Rotated-basis oracle: un-rotate hidden for the gate math (dot/rstd on the
    original basis), then inject onto the rotated residual stream. Mirrors
    vllm_ascend/models/deepseek_v41/engram/common.py:engram_gate.
    rotation: [BS, BS] block applied to each BS-chunk of the last dim."""
    dev = "cpu"
    h = hidden.to(dev).float()  # [T, HC, DIM] rotated
    T, HC, DIM = h.shape
    BS = rotation.shape[0]
    nblk = DIM // BS
    key = kv.to(dev).float()[:, : HC * DIM].view(T, HC, DIM)  # original basis
    value = kv.to(dev).float()[:, HC * DIM :]  # rotated basis
    qw = q_weight.to(dev).float()
    kw = k_weight.to(dev).float()
    mask = token_mask.to(dev).view(T, 1).float()

    # original = hidden @ rotation  (block-diagonal: per BS-chunk matmul)
    h_blk = h.view(T, HC, nblk, BS)
    rot_cpu = rotation.to(dev).float()
    original = (h_blk @ rot_cpu).view(T, HC, DIM)

    rstd_h = torch.rsqrt((original * original).mean(-1) + eps)  # == rstd of h (R orthogonal)
    rstd_k = torch.rsqrt((key * key).mean(-1) + eps)
    rstd = rstd_h * rstd_k
    qk = qw * kw
    dot = (original * qk * key).sum(-1) * rstd * (DIM**-0.5)
    mag = torch.sqrt(torch.clamp(dot.abs(), min=1e-6))
    signed_mag = torch.copysign(mag, dot)
    gate = torch.sigmoid(signed_mag)
    gate = gate * mask
    # inject onto the ROTATED residual stream (h); value is already rotated.
    out = h + gate.unsqueeze(-1) * value.unsqueeze(1)
    return out.to(torch.bfloat16).to(hidden.device)


def run_test_rotated(rt, num_tokens, hc_mult, dim, dtype, msg, mask_mode="all"):
    """Rotated path: pass un-rotated hidden as `hidden` (gate math) and the
    rotated residual as `hidden_residual` (inject). The caller would normally
    derive the un-rotated hidden via matmul(hidden_rotated, R.T); here we
    construct both directly so the kernel under test sees two distinct tensors."""
    torch.manual_seed(4321 + num_tokens * 7 + hc_mult * 13 + dim)
    BS = 32
    assert dim % BS == 0, "rotated test requires dim divisible by 32"
    # rotated residual stream (what lives in the model)
    hidden_rot = torch.randn(num_tokens, hc_mult, dim, dtype=dtype)
    # block-diagonal orthogonal rotation (one [BS, BS] block, identity-diag-ish)
    r = torch.randn(BS, BS, dtype=torch.float32)
    rot, _ = torch.linalg.qr(r)  # orthogonal
    # un-rotated hidden = hidden_rot @ rot (matches common.py's @ rotation.T semantics
    # since rot here is the .T the native code loads)
    hidden_unrot = (hidden_rot.float().view(-1, dim // BS, BS) @ rot).view_as(hidden_rot).to(dtype)

    kv = torch.randn(num_tokens, (hc_mult + 1) * dim, dtype=dtype) * 0.1
    qw = torch.randn(hc_mult, dim, dtype=dtype) * 0.1
    kw = torch.randn(hc_mult, dim, dtype=dtype) * 0.1
    if mask_mode == "all":
        mask = torch.ones(num_tokens, dtype=torch.int32)
    elif mask_mode == "none":
        mask = torch.zeros(num_tokens, dtype=torch.int32)
    else:
        mask = (torch.arange(num_tokens) % 2 == 0).to(torch.int32)

    standard = torch_engram_gate_rotated(hidden_rot, kv, qw, kw, mask, 1e-6, rot)

    out = torch.empty(num_tokens, hc_mult, dim, dtype=dtype)
    torch.npu.synchronize()
    # hidden = un-rotated (gate math), hidden_residual = rotated (inject add)
    engram_gate(
        rt,
        hidden_unrot,
        kv=kv,
        q_weight=qw,
        k_weight=kw,
        mask=mask,
        out=out,
        hc_mult=hc_mult,
        dim=dim,
        eps=1e-6,
        hidden_residual=hidden_rot,
    )
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(standard, out, rtol=1e-2, atol=1e-3)
        print(f"{msg}: PASS")
    except AssertionError as e:
        print(f"{msg}: FAILED")
        print(f"{e}")
        raise


if __name__ == "__main__":
    rt = Runtime(0, size=500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")

    # V4.1 topology: HC = 4, DIM = 5120 (hidden_size, from the Flash-1017-w8a8
    # checkpoint config). The kernel loads the row whole (no tiling). Cover
    # decode (T = 1), prefill (T = 8, 41), and a small DIM = 128 for reduce sanity.
    # bf16 only — V4.1 is a w8a8 bf16 model (no fp16 variant).
    cases = [
        (1, 4, 128),
        (8, 4, 128),
        (1, 4, 5120),
        (8, 4, 5120),
        (41, 4, 5120),
    ]
    # Unrotated path (hidden_residual aliases hidden's storage, explicit pass).
    # The nullopt default path (hidden_residual omitted) is also covered inside run_test.
    for num_tokens, hc, dim in cases:
        for mask_mode in ["all", "partial", "none"]:
            msg = f"[unrotated/bf16/T={num_tokens}/HC={hc}/D={dim}/mask={mask_mode}]"
            run_test(rt, num_tokens, hc, dim, torch.bfloat16, msg, mask_mode=mask_mode)
    # Rotated path (distinct hidden vs hidden_residual, mirrors value_projection_rotated).
    for num_tokens, hc, dim in [(1, 4, 128), (8, 4, 128), (1, 4, 5120), (8, 4, 5120)]:
        for mask_mode in ["all", "partial", "none"]:
            msg = f"[rotated/bf16/T={num_tokens}/HC={hc}/D={dim}/mask={mask_mode}]"
            run_test_rotated(rt, num_tokens, hc, dim, torch.bfloat16, msg, mask_mode=mask_mode)

    # Negative cases: pin the host-side guards (dim % 64, UB budget,
    # hiddenResidual shape/dtype). Removing any guard without a matching
    # expect-throw here would silently re-open the kernel to UB corruption /
    # OOB reads on bad input.
    def expect_throw(rt, msg, dim, hidden_residual=None, hidden_dtype=torch.bfloat16):
        # Build a shape-consistent arg set so the target guard (not an earlier
        # shape check) fires: hidden/kv/qw/kw/out all match (T, HC, dim).
        T, HC = 2, 4
        hidden = torch.randn(T, HC, dim, dtype=hidden_dtype)
        kv = torch.randn(T, (HC + 1) * dim, dtype=hidden_dtype) * 0.1
        qw = torch.randn(HC, dim, dtype=hidden_dtype) * 0.1
        kw = torch.randn(HC, dim, dtype=hidden_dtype) * 0.1
        mask = torch.ones(T, dtype=torch.int32)
        out = torch.empty(T, HC, dim, dtype=hidden_dtype)
        args = dict(
            rt=rt, hidden=hidden, kv=kv, q_weight=qw, k_weight=kw, mask=mask, out=out, hc_mult=HC, dim=dim, eps=1e-6
        )
        if hidden_residual is not None:
            args["hidden_residual"] = hidden_residual
        try:
            engram_gate(**args)
            torch.npu.synchronize()
            raise AssertionError(f"{msg}: expected RuntimeError, got success")
        except RuntimeError as e:
            s = str(e)
            if msg not in s:
                raise AssertionError(f"{msg}: threw, but wrong message: {s}") from e
            print(f"[negative/{msg}]: PASS (threw)")

    # dim not a multiple of 64 -> "dim must be a multiple of 64"
    expect_throw(rt, "dim must be a multiple of 64", dim=100)
    # dim over the UB budget (EngramGateUbNeed(5504, 2) + 512 > UB_SIZE) -> "exceeds kernel UB budget"
    expect_throw(rt, "exceeds kernel UB budget", dim=5504)
    # hidden_residual shape mismatch -> "hiddenResidual must match hidden shape/dtype"
    expect_throw(
        rt,
        "hiddenResidual must match hidden shape/dtype",
        dim=128,
        hidden_residual=torch.randn(2, 4, 64, dtype=torch.bfloat16),
    )
    # hidden_residual dtype mismatch (fp16 vs bf16) -> same shape/dtype guard
    expect_throw(
        rt,
        "hiddenResidual must match hidden shape/dtype",
        dim=128,
        hidden_residual=torch.randn(2, 4, 128, dtype=torch.float16),
    )
