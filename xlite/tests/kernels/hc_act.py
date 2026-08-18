#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Correctness test for the hc_act kernel (DeepSeek-V4 Hyper-Connection gates).

hc_act computes, per token, for hc_mult = K:
  pre  [K]    = sigmoid(mixes[:, :K]       * scale[0] + base[:K])           + eps
  post [K]    = 2 * sigmoid(mixes[:, K:2K] * scale[1] + base[K:2K])
  comb [K*K]  = sinkhorn(softmax(mixes[:, 2K:] * scale[2] + base[2K:]) + eps)

head mode (hc_head): mixes is [m, K], hc_scale is [1], hc_base is [K]. Only pre runs;
auto-detected from hc_base.numel() == hc_mult.
"""
import sys

import torch

from xlite._C import Runtime, hc_act


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Pure-PyTorch reference. Matches deepseek_v4_kernel.py:hc_split_sinkhorn.

    mixes [b, s, mix_hc], mix_hc = (2 + hc_mult) * hc_mult.
    Returns (pre [b,s,hc_mult], post [b,s,hc_mult], comb [b,s,hc_mult,hc_mult]).

    Sinkhorn: col-norm once, then (row-norm, col-norm) * (iters-1); each divide uses
    denom = sum + eps.
    """
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc_mult:2 * hc_mult] * hc_scale[1] + hc_base[hc_mult:2 * hc_mult])

    comb = mixes[..., 2 * hc_mult:] * hc_scale[2] + hc_base[2 * hc_mult:]
    b, s, _ = comb.shape
    comb = comb.view(b, s, hc_mult, hc_mult)
    # row softmax + eps
    comb = torch.softmax(comb, dim=-1) + eps
    # column normalization once
    col_sum = comb.sum(dim=-2, keepdim=True) + eps
    comb = comb / col_sum
    # (row-norm, col-norm) * (iters - 1)
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / row_sum
        col_sum = comb.sum(dim=-2, keepdim=True) + eps
        comb = comb / col_sum
    return pre, post, comb


def hc_head_ref(mixes, hc_scale, hc_base, hc_mult=4, eps=1e-6):
    """Pure-PyTorch reference for hc_head (pre-only). Matches deepseek_v4.py:hc_head.

    mixes [b, s, hc_mult], hc_scale [1], hc_base [hc_mult].
    Returns pre = sigmoid(mixes * hc_scale[0] + hc_base) + eps.
    """
    pre = torch.sigmoid(mixes * hc_scale[0] + hc_base) + eps
    return pre

torch.npu.set_device(0)
rt = Runtime(0, 2048)

HC_MULT = 4
MIX_HC = (2 + HC_MULT) * HC_MULT  # 24 for hc_mult=4
# bf16 pre-merge I/O dim. Small (256) for the gate cases: vecRep=4 > 1 exercises the
# vaxpy/convert multi-repeat path without the memory cost of 4096. The dedicated MERGE
# cases below exercise the real hidden=4096 path (vecRep=64, convert split into <=255
# chunks, UB boundary).
MERGE_HIDDEN_SMALL = 256
MERGE_HIDDEN = 4096

# (b, s, sinkhorn_iters, eps) gate cases. Merge I/O uses MERGE_HIDDEN_SMALL.
CASES = [
    (2, 8, 20, 1e-6),
    (1, 1, 20, 1e-6),
    (8, 4096, 20, 1e-6),
]
# Merge cases: small batch, real hidden=4096 — exercises convert chunk-split
# (totalRep = hcMult*vecRep = 4*64 = 256 > 255) and the UB boundary (169.5KB of 192KB).
MERGE_CASES = [
    (2, 8, 20, 1e-6),
    (1, 1, 20, 1e-6),
    (8, 4096, 20, 1e-6),
]


def run_case(b, s, sinkhorn_iters, eps, merge_hidden=MERGE_HIDDEN_SMALL):
    n = b * s
    torch.manual_seed(1234)
    with torch.device("npu"):
        # mixes [b, s, mix_hc] fp32; kernel takes flat [n, mix_hc].
        mixes_3d = torch.randn(b, s, MIX_HC, dtype=torch.float32)
        hc_scale = torch.randn(3, dtype=torch.float32)
        hc_base = torch.randn(MIX_HC, dtype=torch.float32)
        # x_resid [n, hc_mult, hidden] bf16 — the merge input (UN-normalized residual).
        x_resid = torch.randn(n, HC_MULT, merge_hidden, dtype=torch.bfloat16)

    mixes_flat = mixes_3d.view(n, MIX_HC).contiguous()

    # ---- PyTorch reference (3D input) ----
    ref_pre, ref_post, ref_comb = hc_split_sinkhorn(
        mixes_3d, hc_scale, hc_base, hc_mult=HC_MULT, sinkhorn_iters=sinkhorn_iters, eps=eps)
    # [b,s,K], [b,s,K], [b,s,K,K] -> flatten to [n,*] for kernel comparison
    ref_pre = ref_pre.view(n, HC_MULT)
    ref_post = ref_post.view(n, HC_MULT)
    ref_comb = ref_comb.view(n, HC_MULT * HC_MULT)
    y_ref = torch.sum(ref_pre.unsqueeze(-1) * x_resid.view(n, HC_MULT, merge_hidden).float(),
                     dim=1)
    y_ref = y_ref.to(torch.bfloat16)

    # ---- xlite kernel (2D input) ----
    with torch.device("npu"):
        post = torch.empty(n, HC_MULT, dtype=torch.float32)
        comb = torch.empty(n, HC_MULT * HC_MULT, dtype=torch.float32)
        y_out = torch.empty(n, merge_hidden, dtype=torch.bfloat16)
    torch.npu.synchronize()
    hc_act(rt, mixes_flat, hc_scale, hc_base, post, comb, HC_MULT, eps, sinkhorn_iters,
           x_resid=x_resid, output=y_out)
    torch.npu.synchronize()

    # ---- compare ----
    tag = f"b={b} s={s} iters={sinkhorn_iters} eps={eps} hidden={merge_hidden}"

    tol = {"post": (1e-5, 1e-3), "comb": (1e-5, 1e-3), "y": (1e-5, 1e-3)}
    ok = True
    got_pairs = [("post", post, ref_post), ("comb", comb, ref_comb),
                 ("y", y_out, y_ref)]
    for name, got, want in got_pairs:
        atol, rtol = tol[name]
        try:
            torch.testing.assert_close(want.cpu(), got.cpu(), atol=atol, rtol=rtol)
            print(f"hc_act {tag} {name} passed!")
        except AssertionError as e:
            ok = False
            print(f"{e}")
            print(f"torch_npu {name}: {want.cpu().flatten()[:8]}")
            print(f"xlite {name}: {got.cpu().flatten()[:8]}")
    return ok


# head-mode cases (hc_head: pre-only, mixes is [n, hc_mult]). sinkhorn_iters is
# irrelevant in head mode (no comb). Merge I/O uses MERGE_HIDDEN_SMALL.
HEAD_CASES = [
    (2, 8, 1e-6),
    (1, 4, 1e-6),
    (8, 4096, 1e-6),
]
# head-mode merge cases: real hidden=4096 — exercises the same convert chunk-split
# (totalRep = hcMult*vecRep = 4*64 = 256 > 255) as MERGE_CASES, but on the head path.
HEAD_MERGE_CASES = [
    (1, 8, 1e-6),
    (2, 1, 1e-6),  # m=2: two tokens exercise ping-pong across curr=0/1
]


def run_head_case(b, s, eps, merge_hidden=MERGE_HIDDEN_SMALL):
    n = b * s
    torch.manual_seed(4321)
    with torch.device("npu"):
        # mixes [b, s, hc_mult] fp32 (head: pre segment only); kernel takes flat [n, hc_mult].
        mixes_3d = torch.randn(b, s, HC_MULT, dtype=torch.float32)
        hc_scale = torch.randn(1, dtype=torch.float32)   # [1] scalar for pre
        hc_base = torch.randn(HC_MULT, dtype=torch.float32)  # [hc_mult] pre bias
        x_resid = torch.randn(n, HC_MULT, merge_hidden, dtype=torch.bfloat16)

    mixes_flat = mixes_3d.view(n, HC_MULT).contiguous()

    # ---- PyTorch reference (3D input) ----
    ref_pre = hc_head_ref(mixes_3d, hc_scale, hc_base, hc_mult=HC_MULT, eps=eps)
    ref_pre = ref_pre.view(n, HC_MULT)
    y_ref = torch.sum(ref_pre.unsqueeze(-1) * x_resid.view(n, HC_MULT, merge_hidden).float(),
                     dim=1)
    y_ref = y_ref.to(torch.bfloat16)

    # ---- xlite kernel (head mode, auto-detected from hc_base.numel()==hc_mult) ----
    # post/comb unused; scratch so the pybind signature is satisfied (kernel skips them).
    with torch.device("npu"):
        post = torch.empty(n, HC_MULT, dtype=torch.float32)
        comb = torch.empty(n, HC_MULT * HC_MULT, dtype=torch.float32)
        y_out = torch.empty(n, merge_hidden, dtype=torch.bfloat16)
    torch.npu.synchronize()
    hc_act(rt, mixes_flat, hc_scale, hc_base, post, comb, HC_MULT, eps, 20,
           x_resid=x_resid, output=y_out)
    torch.npu.synchronize()

    # ---- compare ----
    tag = f"[head] b={b} s={s} eps={eps} hidden={merge_hidden}"
    tol = {"y": (1e-5, 1e-3)}
    ok = True
    got_pairs = [("y", y_out, y_ref)]
    for name, got, want in got_pairs:
        atol, rtol = tol[name]
        try:
            torch.testing.assert_close(want.cpu(), got.cpu(), atol=atol, rtol=rtol)
            print(f"hc_act {tag} {name} passed!")
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
    for (b, s, iters, eps) in MERGE_CASES:
        all_ok &= run_case(b, s, iters, eps, merge_hidden=MERGE_HIDDEN)
    for (b, s, eps) in HEAD_CASES:
        all_ok &= run_head_case(b, s, eps)
    for (b, s, eps) in HEAD_MERGE_CASES:
        all_ok &= run_head_case(b, s, eps, merge_hidden=MERGE_HIDDEN)
    if all_ok:
        print("==== all hc_act cases passed ====")
    else:
        print("==== some hc_act cases FAILED ====")
        sys.exit(1)
