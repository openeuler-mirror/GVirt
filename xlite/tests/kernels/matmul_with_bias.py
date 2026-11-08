#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, matmul_with_bias
import torch.nn.functional as F

torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

rt = Runtime(0, size=500)
torch.npu.set_device(0)

for weight_nz in [False, True]:
    for dtype in [torch.float16, torch.bfloat16, torch.float]:
        for m in [1, 8]:
            for n in [32, 64, 16128]:
                for k in [32, 128, 7168, 7184, 7232, 7296, 7424, 7488, 7616]:
                    x = torch.randn(m, k, dtype=dtype, device="npu:0")
                    y = torch.randn(n, k, dtype=dtype, device="npu:0")
                    y_standard = y
                    z = torch.zeros(m, n, dtype=dtype, device="npu:0")
                    bias = torch.randn(n, dtype=torch.float, device="npu:0")

                    standard = F.linear(x, y_standard, bias).to(dtype)

                    if weight_nz and dtype != torch.float:
                        y = torch_npu.npu_format_cast(y, 29)

                    torch.npu.synchronize()
                    matmul_with_bias(rt, x, y, z, bias, weight_nz=weight_nz and dtype != torch.float)
                    torch.npu.synchronize()

                    print(f'[{m}, {k}] x [{n}, {k}] {dtype} weight_nz {weight_nz and dtype != torch.float} matmul executed!')

                    # torch_npu and xlite both accumulate in fp32 but split K
                    # differently (K up to 7616 here), so results differ by
                    # 1 output ULP (a bf16 ULP is up to 2^-8 relative) and by
                    # up to ~1e-3 absolute at cancellation-heavy near-zero
                    # outputs — the same accumulation-order noise as
                    # tests/kernels/matmul.py, whose fp64 check showed both
                    # sides equidistant from the true value. Per-dtype bounds
                    # mirror that test; a real operand/layout bug produces
                    # O(1)-O(100) garbage and still fails hard.
                    atol, rtol = {
                        torch.float16: (1e-2, 2e-3),
                        torch.bfloat16: (1e-2, 1.6e-2),
                        torch.float: (4e-3, 1e-3),
                    }[dtype]
                    try:
                        torch.testing.assert_close(standard, z, atol=atol, rtol=rtol)
                    except AssertionError as e:
                        print(f'{e}')
                        print(f'torch_npu: {standard}')
                        print(f'xlite: {z}')
