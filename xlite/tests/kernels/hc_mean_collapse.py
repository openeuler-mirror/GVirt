#!/usr/bin/env python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Test hc_mean_collapse: weight-free mean over hcMult MHC streams (GLM-5.3 head).

y[m, hidden] = (1/hcMult) * sum_h x[m, h*hidden + j], accumulated in fp32.
Only BF16/FP16 hidden-state dtypes are supported.
"""

import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, hc_mean_collapse


def torch_hc_mean_collapse(x, m, hc_mult, hidden):
    return x.view(m, hc_mult, hidden).float().mean(dim=1).to(x.dtype)


def run_case(rt, m, hc_mult, hidden, dtype, msg):
    x = torch.randn(m, hc_mult * hidden, dtype=dtype)
    ref = torch_hc_mean_collapse(x, m, hc_mult, hidden)
    out = torch.empty(m, hidden, dtype=dtype)
    torch.npu.synchronize()
    hc_mean_collapse(rt, x, out, m, hc_mult, hidden)
    torch.npu.synchronize()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)
    print(f"{msg}: PASS")


def check_rejections(rt, dtype):
    # Fail-closed host guards: the kernel's Dtype->fp32 convert runs in 64-element
    # repeats (no tail handling) and the fp32 working set must fit the 192KB UB
    # budget — both must raise instead of silently corrupting.
    cases = [
        ("hidden not multiple of 64", (2, 4, 100)),
        ("UB budget exceeded", (2, 4, 24576)),
    ]
    for msg, (m, hc_mult, hidden) in cases:
        x = torch.randn(m, hc_mult * hidden, dtype=dtype)
        out = torch.empty(m, hidden, dtype=dtype)
        try:
            torch.npu.synchronize()
            hc_mean_collapse(rt, x, out, m, hc_mult, hidden)
            raise AssertionError(f"{msg}: expected RuntimeError, call succeeded")
        except RuntimeError as e:
            assert "hidden" in str(e) or "UB" in str(e), f"{msg}: unexpected error {e}"
            print(f"hc_mean_collapse rejection [{msg}]: PASS")


if __name__ == "__main__":
    rt = Runtime(0, size=500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")
    torch.manual_seed(0)
    for dtype in [torch.bfloat16, torch.float16]:
        for m in [1, 4, 8]:
            run_case(rt, m, 4, 4096, dtype, f"[{dtype}/m={m}/hcMult=4/hidden=4096]")
        run_case(rt, 2, 2, 128, dtype, f"[{dtype}/m=2/hcMult=2/hidden=128]")
    check_rejections(rt, torch.bfloat16)
    print("hc_mean_collapse: ALL PASS")
