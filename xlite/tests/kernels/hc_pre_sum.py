#!/usr/bin/python3
# coding=utf-8
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

from xlite._C import Runtime, hc_pre


torch.npu.set_device(0)
rt = Runtime(0, 2048)

HC_MULT = 4
# Small (256) and real (4096) hidden. hidden=4096 exercises the convert chunk-split
# (totalRep = hcMult*vecRep = 4*64 = 256 > VECTOR_MAX_REPEAT=255) and the UB boundary,
# matching the merge path covered by tests/kernels/hc_act.py MERGE_CASES.

# (b, s, hidden).
CASES = [
    (2, 8, 256),
    (1, 1, 256),    # m=1: decode single-token path
    (2, 8, 4096),  # real-scale: convert chunk-split + UB boundary
    (1, 1, 4096),
    (8, 1024, 4096),
]


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
    y_ref = torch.sum(pre.unsqueeze(-1) * x_3d.float(), dim=2)   # [b, s, hidden]
    y_ref = y_ref.to(torch.bfloat16).reshape(n, hidden)

    # ---- xlite kernel (2D flat inputs) ----
    x_flat = x_3d.reshape(n, HC_MULT, hidden).contiguous()    # bf16 [n, hcMult, hidden]
    pre_flat = pre.reshape(n, HC_MULT).contiguous()          # fp32 [n, hcMult]
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


if __name__ == "__main__":
    all_ok = True
    for (b, s, hidden) in CASES:
        all_ok &= run_case(b, s, hidden)
    if all_ok:
        print("==== all hc_pre cases passed ====")
    else:
        print("==== some hc_pre cases FAILED ====")
        sys.exit(1)
