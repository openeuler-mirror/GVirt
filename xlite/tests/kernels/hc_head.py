#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""End-to-end test for ForwardHcPre in head mode (DeepSeek-V4 hc_head).

hc_head is the final Hyper-Connection head (deepseek_v4.py:725-732). Unlike the attn/ffn
hc_pre, it has no post/comb and no Sinkhorn — only a single pre gate:

    mixes = F.linear(x, hc_fn) * rsqrt              # [b, s, hc_mult]
    pre   = sigmoid(mixes * hc_scale + hc_base) + hc_eps    # hc_scale is [1]
    y     = sum(pre.unsqueeze(-1) * x.view(shape), dim=2)

ForwardHcPre (model.cpp:1487) computes this via the same op chain as attn/ffn hc_pre:
rmsnorm (no-affine) -> matmul -> hc_act(headOnly). The Step-5 pre-merge is folded into
hc_act (pre is consumed in-kernel, not written to GM; y is written instead). The hc_act
kernel auto-detects head mode from hc_base.numel() == hc_mult (the head bias is
[hc_mult], not the attn/ffn [(2+hc_mult)*hc_mult]); see tests/kernels/hc_act.py for the
kernel-level A/B.

This test composes the same host ops from Python and compares every observable
intermediate (x_norm, mixes, y) against a pure-PyTorch reference, end-to-end.
"""
import sys

import torch
import torch.nn.functional as F

from xlite._C import Runtime, rmsnorm, matmul, hc_act


def hc_head_ref(x_3d, hc_fn, hc_scale, hc_base, hc_mult, norm_eps, hc_eps):
    """Pure-PyTorch reference — verbatim deepseek_v4.hc_head (deepseek_v4.py:725-732).

    x_3d     [b, s, hc_mult, hidden] bf16  (the residual, expanded to hc_mult copies)
    hc_fn    [hc_mult, hc_mult*hidden] fp32
    hc_scale [1] fp32
    hc_base  [hc_mult] fp32
    returns (x_norm [n, hc_dim] fp32, mixes [n, hc_mult] fp32, pre [n, hc_mult] fp32,
             y [n, hidden] fp32)
    """
    b, s, _, hidden = x_3d.shape
    n = b * s
    hc_dim = hc_mult * hidden
    shape = x_3d.size()
    x = x_3d.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt                  # [b, s, hc_mult]
    pre = torch.sigmoid(mixes * hc_scale[0] + hc_base) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)   # [b, s, hidden]
    x_norm = (x * rsqrt).reshape(n, hc_dim)
    return x_norm, mixes.reshape(n, hc_mult), pre.reshape(n, hc_mult), y.reshape(n, hidden)


torch.npu.set_device(1)
rt = Runtime(1, 500)

HC_MULT = 4
NORM_EPS = 1e-6
HC_EPS = 1e-6

# (b, s, hidden).
CASES = [
    (2, 8, 256),
    (1, 4, 512),
    (1, 4, 4096),
    (8, 4096, 4096),
]


def run_case(b, s, hidden):
    n = b * s
    hc_dim = HC_MULT * hidden
    torch.manual_seed(4321)
    with torch.device("npu"):
        # x [b, s, hc_mult, hidden] bf16 (residual). hc_fn/scale/base stay fp32.
        x_3d = torch.randn(b, s, HC_MULT, hidden, dtype=torch.bfloat16)
        hc_fn = torch.randn(HC_MULT, hc_dim, dtype=torch.float32)   # [outFeatures=hc_mult, hcDim]
        hc_scale = torch.randn(1, dtype=torch.float32)              # [1] scalar for pre
        hc_base = torch.randn(HC_MULT, dtype=torch.float32)         # [hc_mult] pre bias

    # ---- PyTorch reference (fp32 throughout; the ground truth) ----
    x_norm_ref, mixes_ref, _, y_ref = hc_head_ref(
        x_3d, hc_fn, hc_scale, hc_base, hc_mult=HC_MULT, norm_eps=NORM_EPS, hc_eps=HC_EPS)
    y_ref = y_ref.to(torch.bfloat16).float()

    # ---- xlite host-op chain (same order as ForwardHcPre, head path) ----
    with torch.device("npu"):
        x_flat = x_3d.reshape(n, hc_dim).contiguous()             # bf16
        x_norm = torch.empty(n, hc_dim, dtype=torch.float32)     # bf16, as in model.cpp
        mixes = torch.empty(n, HC_MULT, dtype=torch.float32)      # [n, hc_mult]
        post = torch.empty(n, HC_MULT, dtype=torch.float32)
        comb = torch.empty(n, HC_MULT * HC_MULT, dtype=torch.float32)
        y_merge = torch.empty(n, hidden, dtype=torch.bfloat16)
        x_residual = x_3d.reshape(n, HC_MULT, hidden).contiguous()   # bf16 [n, hc_mult, hidden]
        empty_norm = torch.empty(0, dtype=torch.float32)           # no-affine placeholder
    torch.npu.synchronize()

    rmsnorm(rt, x_flat, empty_norm, x_norm, NORM_EPS, norm_dim=hc_dim, cnt_per_token=1)
    matmul(rt, x_norm, hc_fn, mixes, weight_nz=False, transpose=False)
    hc_act(rt, mixes, hc_scale, hc_base, post, comb, HC_MULT, HC_EPS, 20,
           x_resid=x_residual, output=y_merge)
    torch.npu.synchronize()

    # ---- compare ----
    tag = f"b={b} s={s} hidden={hidden}"
    tol = {"x_norm": (1e-5, 1e-3), "mixes": (1e-5, 1e-3), "y": (1e-5, 1e-3)}
    ok = True
    got_pairs = [("x_norm", x_norm.float(), x_norm_ref), ("mixes", mixes, mixes_ref),
                 ("y", y_merge.float(), y_ref)]
    for name, got, want in got_pairs:
        atol, rtol = tol.get(name, (1e-5, 1e-3))
        try:
            torch.testing.assert_close(want.cpu(), got.cpu(), atol=atol, rtol=rtol)
            print(f"hc_head {tag} {name} passed!")
        except AssertionError as e:
            ok = False
            print(f"{e}")
            print(f"torch_npu {name}: {want.cpu()}")
            print(f"xlite {name}: {got.cpu()}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for (b, s, hidden) in CASES:
        all_ok &= run_case(b, s, hidden)
    if all_ok:
        print("==== all hc_head cases passed ====")
    else:
        print("==== some hc_head cases FAILED ====")
        sys.exit(1)
