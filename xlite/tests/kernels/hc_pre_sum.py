#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY of even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Correctness test for the hc_pre kernel (hc_pre_sum): the DeepSeek-V4
Hyper-Connection pre-activation merge, non-fused counterpart of hc_act's merge
segment.

hc_pre collapses the hc copies into one, weighted by pre:
  x [m, hcMult, hidden] bf16  (the UN-normalized residual)
  pre [m, hcMult] fp32        (the hc_split_sinkhorn output, read from GM)
  y [m, hidden] bf16          = Σ_h pre[h] * x[h, hidden]   (fp32 accumulate, bf16 out)

This mirrors inference/model.py:hc_pre exactly:
  y = torch.sum(pre.unsqueeze(-1) * x.float(), dim=2)
"""

import sys

import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, hc_pre, hc_split_sinkhorn

torch.npu.set_device(0)
rt = Runtime(0, size=2048)

HC_MULT = 4
# Small (256), V4 real (4096, near the UB boundary), V4.1 real (5120). 5120 overflowed UB
# until the xFp32 single-row ping-pong fix (was [hcMult*hidden] fp32 = 80KB).

# (b, s, hidden).
CASES = [
    (2, 8, 256),
    (1, 1, 256),  # m=1: decode single-token path
    (2, 8, 4096),  # real-scale: convert chunk-split + UB boundary
    (1, 1, 4096),
    (8, 1024, 4096),
    (1, 1, 5120),  # V4.1 real hidden_size
    (8, 64, 5120),
]


def hc_split_sinkhorn_ref(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Pure-PyTorch reference. Matches tests/models/deepseek_v4.py:hc_split_sinkhorn.

    mixes [b, s, mix_hc], mix_hc = (2 + hc_mult) * hc_mult.
    Returns (pre [b,s,hc_mult], post [b,s,hc_mult], comb [b,s,hc_mult,hc_mult]).
    Sinkhorn: col-norm once, then (row-norm, col-norm) * (iters-1); denom = sum + eps.
    """
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
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


def run_case(b, s, hidden):
    n = b * s
    torch.manual_seed(4321)
    with torch.device("npu"):
        # x [b, s, hcMult, hidden] bf16 (UN-normalized residual). pre [n, hcMult] fp32.
        x_3d = torch.randn(b, s, HC_MULT, hidden, dtype=torch.bfloat16)
        # pre in a plausible range (sigmoid+eps ∈ (0,1]); keep magnitudes modest so the
        # fp32 accumulate stays accurate against the fp32 reference.
        pre = torch.sigmoid(torch.randn(b, s, HC_MULT, dtype=torch.float32)) + 1e-6

    # ---- PyTorch reference — verbatim inference/model.py:hc_pre ----
    # y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
    y_ref = torch.sum(pre.unsqueeze(-1) * x_3d.float(), dim=2)  # [b, s, hidden]
    y_ref = y_ref.to(torch.bfloat16).reshape(n, hidden)

    # ---- xlite kernel (2D flat inputs) ----
    x_flat = x_3d.reshape(n, HC_MULT, hidden).contiguous()  # bf16 [n, hcMult, hidden]
    pre_flat = pre.reshape(n, HC_MULT).contiguous()  # fp32 [n, hcMult]
    with torch.device("npu"):
        y_out = torch.empty(n, hidden, dtype=torch.bfloat16)
    torch.npu.synchronize()
    hc_pre(rt, x_flat, pre_flat, y_out, n, HC_MULT, hidden)
    torch.npu.synchronize()

    # ---- compare ----
    tag = f"b={b} s={s} hidden={hidden}"
    try:
        torch.testing.assert_close(y_ref.cpu(), y_out.cpu(), atol=1e-5, rtol=1e-3)
        print(f"hc_pre {tag} y passed!")
        return True
    except AssertionError as e:
        print(f"{e}")
        print(f"torch_npu y: {y_ref.cpu().flatten()[:8]}")
        print(f"xlite y: {y_out.cpu().flatten()[:8]}")
        return False


def run_premix_path_case(b, s, hidden):
    """Verify the ForwardHcPre preMix branch (hc_split_sinkhorn + hc_pre) against a
    pure-PyTorch reference.

    The preMix branch runs hc_split_sinkhorn (writes post/comb + a throwaway pre)
    then hc_pre (merges with the *external* preMix). When external preMix equals
    the internal pre the split produced, out = hc_collapse(x, pre) = Σ_h pre[h]*x[h].
    The reference uses the pure-PyTorch hc_split_sinkhorn (not the fused hc_act
    kernel, which overflows UB at hidden>4096 and is V4-only), so this validates
    the path at V4.1's hidden=5120.
    """
    n = b * s
    mix_hc = (2 + HC_MULT) * HC_MULT
    torch.manual_seed(99)
    with torch.device("npu"):
        x_3d = torch.randn(b, s, HC_MULT, hidden, dtype=torch.bfloat16)
        mixes = torch.randn(b, s, mix_hc, dtype=torch.float32)
        hc_scale = torch.randn(3, dtype=torch.float32)
        hc_base = torch.randn((2 + HC_MULT) * HC_MULT, dtype=torch.float32)

    # ---- PyTorch reference: hc_split_sinkhorn + hc_collapse (merge with internal pre). ----
    ref_pre, ref_post, ref_comb = hc_split_sinkhorn_ref(
        mixes, hc_scale, hc_base, hc_mult=HC_MULT, sinkhorn_iters=20, eps=1e-6
    )
    ref_pre = ref_pre.reshape(n, HC_MULT)
    ref_post = ref_post.reshape(n, HC_MULT)
    ref_comb = ref_comb.reshape(n, HC_MULT * HC_MULT)
    x_flat = x_3d.reshape(n, HC_MULT, hidden).contiguous()
    y_ref = torch.sum(ref_pre.unsqueeze(-1) * x_flat.float(), dim=1).to(torch.bfloat16)

    # ---- xlite preMix path: hc_split_sinkhorn (post/comb + pre) + hc_pre (merge). ----
    mixes_flat = mixes.reshape(n, mix_hc).contiguous()
    with torch.device("npu"):
        pre_scratch = torch.empty(n, HC_MULT, dtype=torch.float32)
        post_split = torch.empty(n, HC_MULT, dtype=torch.float32)
        comb_split = torch.empty(n, HC_MULT * HC_MULT, dtype=torch.float32)
        out_premix = torch.empty(n, hidden, dtype=torch.bfloat16)
    torch.npu.synchronize()
    hc_split_sinkhorn(rt, mixes_flat, hc_scale, hc_base, pre_scratch, post_split, comb_split, HC_MULT, 1e-6, 20)
    torch.npu.synchronize()
    hc_pre(rt, x_flat, pre_scratch, out_premix, n, HC_MULT, hidden)
    torch.npu.synchronize()

    tag = f"premix-path b={b} s={s} hidden={hidden}"
    ok = True
    try:
        torch.testing.assert_close(out_premix.cpu(), y_ref.cpu(), atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        ok = False
        print(f"{tag} output FAILED\n{e}")
    try:
        torch.testing.assert_close(post_split.cpu(), ref_post.cpu(), atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        ok = False
        print(f"{tag} post FAILED\n{e}")
    try:
        torch.testing.assert_close(comb_split.cpu(), ref_comb.cpu(), atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        ok = False
        print(f"{tag} comb FAILED\n{e}")
    if ok:
        print(f"hc_pre {tag} passed! (output/post/comb match torch ref)")
    return ok


if __name__ == "__main__":
    all_ok = True
    for b, s, hidden in CASES:
        all_ok &= run_case(b, s, hidden)
        all_ok &= run_premix_path_case(b, s, hidden)
    if all_ok:
        print("==== all hc_pre cases passed ====")
    else:
        print("==== some hc_pre cases FAILED ====")
        sys.exit(1)
