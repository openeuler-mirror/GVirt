#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Correctness test for the hc_split_sinkhorn kernel.

hc_split_sinkhorn is hc_act with preSum=0: per token, for hc_mult = K, it writes the
gate segments to GM —
  pre  [K]    = sigmoid(mixes[:, :K]       * scale[0] + base[:K])    + eps
  post [K]    = 2 * sigmoid(mixes[:, K:2K] * scale[1] + base[K:2K])
  comb [K*K]  = sinkhorn(softmax(mixes[:, 2K:] * scale[2] + base[2K:]) + eps)
No head path; hc_base is always the full [(2+K)*K] block.
"""
import sys

import torch

from xlite._C import Runtime, hc_split_sinkhorn


def hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Pure-PyTorch reference. Matches tests/models/deepseek_v4.py:hc_split_sinkhorn.

    mixes [b, s, mix_hc], mix_hc = (2 + hc_mult) * hc_mult.
    Returns (pre [b,s,hc_mult], post [b,s,hc_mult], comb [b,s,hc_mult,hc_mult]).
    Sinkhorn: col-norm once, then (row-norm, col-norm) * (iters-1); denom = sum + eps.
    """
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])

    comb = mixes[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
    b, s, _ = comb.shape
    comb = comb.view(b, s, hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + eps
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

# (b, s, sinkhorn_iters, eps). Gate-only; no merge I/O.
CASES = [
    (2, 8, 20, 1e-6),
    (1, 1, 20, 1e-6),
    (8, 4096, 20, 1e-6),
]


def run_case(b, s, sinkhorn_iters, eps):
    n = b * s
    torch.manual_seed(1234)
    with torch.device("npu"):
        mixes_3d = torch.randn(b, s, MIX_HC, dtype=torch.float32)
        hc_scale = torch.randn(3, dtype=torch.float32)
        hc_base = torch.randn(MIX_HC, dtype=torch.float32)

    mixes_flat = mixes_3d.view(n, MIX_HC).contiguous()

    ref_pre, ref_post, ref_comb = hc_split_sinkhorn_ref(
        mixes_3d, hc_scale, hc_base, hc_mult=HC_MULT, sinkhorn_iters=sinkhorn_iters, eps=eps)
    ref_pre = ref_pre.view(n, HC_MULT)
    ref_post = ref_post.view(n, HC_MULT)
    ref_comb = ref_comb.view(n, HC_MULT * HC_MULT)

    with torch.device("npu"):
        pre = torch.empty(n, HC_MULT, dtype=torch.float32)
        post = torch.empty(n, HC_MULT, dtype=torch.float32)
        comb = torch.empty(n, HC_MULT * HC_MULT, dtype=torch.float32)
    torch.npu.synchronize()
    hc_split_sinkhorn(rt, mixes_flat, hc_scale, hc_base, pre, post, comb, HC_MULT, eps,
                      sinkhorn_iters)
    torch.npu.synchronize()

    tag = f"b={b} s={s} iters={sinkhorn_iters} eps={eps}"
    tol = {"pre": (1e-5, 1e-3), "post": (1e-5, 1e-3), "comb": (1e-5, 1e-3)}
    ok = True
    got_pairs = [("pre", pre, ref_pre), ("post", post, ref_post), ("comb", comb, ref_comb)]
    for name, got, want in got_pairs:
        atol, rtol = tol[name]
        try:
            torch.testing.assert_close(want.cpu(), got.cpu(), atol=atol, rtol=rtol)
            print(f"hc_split_sinkhorn {tag} {name} passed!")
        except AssertionError as e:
            ok = False
            print(f"{e}")
            print(f"torch_npu {name}: {want.cpu().flatten()[:8]}")
            print(f"xlite {name}: {got.cpu().flatten()[:8]}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for (b, s, iters, eps) in CASES:
        all_ok &= run_case(b, s, iters, eps)
    if all_ok:
        print("==== all hc_split_sinkhorn cases passed ====")
    else:
        print("==== some hc_split_sinkhorn cases FAILED ====")
        sys.exit(1)
