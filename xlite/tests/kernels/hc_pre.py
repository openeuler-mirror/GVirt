#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""End-to-end test for ForwardHcPre (DeepSeek-V4 Hyper-Connection pre-activation).

Composes the same host ops as model.cpp (rmsnorm -> matmul -> hc_act) and compares
every intermediate (x_norm, mixes, post, comb, y) against a pure-PyTorch reference.
Step 5 (merge) is folded into hc_act: pre is computed and consumed in-kernel (not
written to GM); y is written instead.

Dtype path (matches model.cpp): x is bf16 (the residual); x_norm is fp32. rmsnorm
does bf16-in -> fp32-out (null weight = no-affine path); matmul then runs fp32-in +
fp32-weight -> fp32-out (transpose=false == F.linear). hc_dim > 6144 exercises the
tiled two-pass rmsnorm path in norm.h.
"""
import sys

import torch
import torch.nn.functional as F

from xlite._C import Runtime, rmsnorm, matmul, hc_act


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Pure-PyTorch reference for the DeepSeek-V4 Hyper-Connection gate activation.

    Matches tests/models/deepseek_v4_kernel.py:hc_split_sinkhorn exactly, inlined
    here so this test is self-contained (same design as tests/kernels/hc_act.py).

    mixes       [b, s, mix_hc]   mix_hc = (2 + hc_mult) * hc_mult
    hc_scale    [3]              per-segment scale (pre / post / comb)
    hc_base     [mix_hc]         per-segment bias
    returns (pre [b,s,hc_mult], post [b,s,hc_mult], comb [b,s,hc_mult,hc_mult])
    """
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])

    comb = mixes[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
    b, s, _ = comb.shape
    comb = comb.view(b, s, hc_mult, hc_mult)
    comb = comb.softmax(dim=-1) + eps
    col_sum = comb.sum(dim=-2, keepdim=True) + eps
    comb = comb / col_sum
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / row_sum
        col_sum = comb.sum(dim=-2, keepdim=True) + eps
        comb = comb / col_sum
    return pre, post, comb


torch.npu.set_device(0)
rt = Runtime(0, 2048)

HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24 for hc_mult=4
NORM_EPS = 1e-6
HC_EPS = 1e-6

# (b, s, hidden, sinkhorn_iters). hidden is the per-branch feature dim (D in
# [b,s,hc_mult,D]); the matmul K = hc_mult * hidden.
CASES = [
    (2, 8, 256, 20),
    (1, 4, 512, 20),
    (1, 4, 4096, 20),
    (8, 1024, 4096, 20),
]


def run_case(b, s, hidden, sinkhorn_iters):
    n = b * s
    hc_dim = HC_MULT * hidden
    torch.manual_seed(1234)
    x_3d = torch.randn(b, s, HC_MULT, hidden, dtype=torch.bfloat16).npu()
    hc_fn = torch.randn(MIX_HC, hc_dim, dtype=torch.float32).npu()   # [outFeatures, hcDim]
    hc_scale = torch.randn(3, dtype=torch.float32).npu()
    hc_base = torch.randn(MIX_HC, dtype=torch.float32).npu()

    # ---- PyTorch reference — verbatim deepseek_v4.hc_pre ----
    shape = x_3d.size()
    x_ref = x_3d.flatten(2).float()
    rsqrt = torch.rsqrt(x_ref.square().mean(-1, keepdim=True) + NORM_EPS)
    mixes_ref = F.linear(x_ref, hc_fn) * rsqrt
    pre_ref, post_ref, comb_ref = hc_split_sinkhorn(
        mixes_ref, hc_scale, hc_base, hc_mult=HC_MULT,
        sinkhorn_iters=sinkhorn_iters, eps=HC_EPS)
    y_ref = torch.sum(pre_ref.unsqueeze(-1) * x_ref.view(shape), dim=2)

    # Flatten the 3D model-shape refs to 2D to match the xlite op outputs.
    x_norm_ref = (x_ref * rsqrt).reshape(n, hc_dim)
    pre_ref = pre_ref.reshape(n, HC_MULT)
    post_ref = post_ref.reshape(n, HC_MULT)
    comb_ref = comb_ref.reshape(n, HC_MULT * HC_MULT)
    mixes_ref = mixes_ref.reshape(n, MIX_HC)
    y_ref = y_ref.reshape(n, hidden).to(torch.bfloat16)

    # ---- xlite host-op chain (same order as ForwardHcPre) ----
    x_flat = x_3d.reshape(n, hc_dim).contiguous()             # bf16
    x_norm = torch.empty(n, hc_dim, dtype=torch.float32).npu()  # fp32, now that norm.h supports it
    mixes = torch.empty(n, MIX_HC, dtype=torch.float32).npu()
    post = torch.empty(n, HC_MULT, dtype=torch.float32).npu()
    comb = torch.empty(n, HC_MULT * HC_MULT, dtype=torch.float32).npu()
    x_residual = x_3d.reshape(n, HC_MULT, hidden).contiguous()   # bf16 [n, hc_mult, hidden]
    y = torch.empty(n, hidden, dtype=torch.bfloat16).npu()
    empty_norm = torch.empty(0, dtype=torch.float32, device="npu")
    torch.npu.synchronize()

    rmsnorm(rt, x_flat, empty_norm, x_norm, NORM_EPS, norm_dim=hc_dim, cnt_per_token=1)
    matmul(rt, x_norm, hc_fn, mixes, weight_nz=False, transpose=False)
    hc_act(rt, mixes, hc_scale, hc_base, post, comb,
           HC_MULT, HC_EPS, sinkhorn_iters, x_resid=x_residual, output=y)
    torch.npu.synchronize()

    # ---- compare ----
    tag = f"b={b} s={s} hidden={hidden} iters={sinkhorn_iters}"
    tol = {"x_norm": (1e-5, 1e-3), "mixes": (1e-5, 1e-3),
           "post": (1e-5, 1e-3), "comb": (1e-5, 1e-3),
           "y": (1e-5, 1e-3)}
    ok = True
    got_pairs = [("x_norm", x_norm, x_norm_ref), ("mixes", mixes, mixes_ref),
                 ("post", post, post_ref),
                 ("comb", comb, comb_ref), ("y", y, y_ref)]
    for name, got, want in got_pairs:
        atol, rtol = tol[name]
        try:
            torch.testing.assert_close(want.cpu(), got.cpu(), atol=atol, rtol=rtol)
            print(f"hc_pre {tag} {name} passed!")
        except AssertionError as e:
            ok = False
            print(f"{e}")
            print(f"torch_npu {name}: {want.cpu()}")
            print(f"xlite {name}: {got.cpu()}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for (b, s, hidden, iters) in CASES:
        all_ok &= run_case(b, s, hidden, iters)
    if all_ok:
        print("==== all hc_pre cases passed ====")
    else:
        print("==== some hc_pre cases FAILED ====")
        sys.exit(1)
