#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""End-to-end test for ForwardHcPost (DeepSeek-V4 Hyper-Connection post-activation merge).

ForwardHcPost implements the PyTorch reference:

    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)

Shapes are [b,s,...]; dim=2 contracts the source stream h:
  term1: y[b,s,k,d] = post[b,s,k] * x[b,s,d]
  term2: y[b,s,k,d] = sum_h comb[b,s,h,k] * residual[b,s,h,d]

term2 is a tiny-K=H batched GEMM the cube engine runs inefficiently, so the kernel
fuses term1 (vmuls) + term2 (H x vaxpy) + bf16<->fp32 cast in one AIV pass.

This test composes the host op (hc_post) from Python and compares y against the
PyTorch reference, in the in-place path (residual==y, as model.cpp:1618 calls it).
"""
import sys

import torch

from xlite._C import Runtime, hc_post


torch.npu.set_device(0)
rt = Runtime(0, 500)

HC_MULT = 4
HC_EPS = 1e-6

# hc_post tiles D over UB (unlike rmsnorm's whole-hc_dim-in-UB), so hidden=4096
# works; cases stay small to keep the run quick, with one real-scale smoke case.
CASES = [
    (2, 8, 256),
    (1, 4, 512),
    (8, 1024, 4096),
    (1, 1, 256),   # m=1: decode single-token path
    (1, 1, 4096),  # m=1 real-scale
]


def run_case(b, s, hidden):
    """Run one hc_post case and compare against the PyTorch reference."""
    n = b * s
    torch.manual_seed(5678)
    with torch.device("npu"):
        x = torch.randn(b, s, hidden, dtype=torch.bfloat16)                 # submodule output
        residual_ref = torch.randn(b, s, HC_MULT, hidden, dtype=torch.bfloat16)
        post = 2.0 * torch.sigmoid(torch.randn(b, s, HC_MULT, dtype=torch.float32))  # in [0, 2]
        # doubly-stochastic comb via softmax + a few Sinkhorn rounds
        comb = torch.randn(b, s, HC_MULT, HC_MULT, dtype=torch.float32)
        comb = torch.softmax(comb, dim=-1) + HC_EPS
        for _ in range(3):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + HC_EPS)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + HC_EPS)

    # PyTorch reference (fp32 throughout).
    x_ref, resid_ref = x.float(), residual_ref.float()
    y_ref = post.unsqueeze(-1) * x_ref.unsqueeze(-2)
    y_ref = y_ref + torch.sum(comb.unsqueeze(-1) * resid_ref.unsqueeze(-2), dim=2)
    y_ref = y_ref.type_as(x)

    # Kernel takes flattened [n=b*s, ...]; view without copying.
    x_n = x.view(n, hidden)
    post_n = post.view(n, HC_MULT)
    comb_n = comb.reshape(n, HC_MULT * HC_MULT)
    resid_n = residual_ref.view(n, HC_MULT, hidden)

    # in-place (residual == y)
    with torch.device("npu"):
        residual_inplace = resid_n.clone()
    torch.npu.synchronize()
    hc_post(rt, x_n, post_n, comb_n, residual_inplace, residual_inplace, n, HC_MULT, hidden)
    torch.npu.synchronize()

    tag = f"b={b} s={s} hidden={hidden}"
    ok = True
    try:
        torch.testing.assert_close(y_ref.cpu(), residual_inplace.view(b, s, HC_MULT, hidden).cpu(),
                                  atol=1e-5, rtol=1e-3)
        print(f"hc_post {tag} passed!")
    except AssertionError as e:
        ok = False
        print(f"{e}")
        print(f"torch_npu: {y_ref.cpu()}")
        print(f"xlite: {residual_inplace.view(b, s, HC_MULT, hidden).cpu()}")
    return ok


if __name__ == "__main__":
    all_ok = True
    for (b, s, hidden) in CASES:
        all_ok &= run_case(b, s, hidden)
    if all_ok:
        print("==== all hc_post cases passed ====")
    else:
        print("==== some hc_post cases FAILED ====")
        sys.exit(1)
