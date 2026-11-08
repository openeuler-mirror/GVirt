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
from xlite._C import Runtime, matmul
import torch.nn.functional as F


rt = Runtime(0, size=500)
torch.npu.set_device(0)
# Required for the NZ weight layout contract: with this flag off,
# npu_format_cast(y, 29) writes a different physical NZ layout than the one
# the xlite matmul kernel consumes (production sets it the same way in
# vllm-ascend's model runner).
torch.npu.config.allow_internal_format = True

# Per-dtype comparison tolerances. For k up to ~7.6k an element's exact result
# can land on an output-dtype rounding tie, which xlite and the torch_npu
# reference round in opposite directions (~1 output ulp apart), so allow ~2
# ulp relative error. The absolute floor covers fp32 accumulation-order noise
# on near-zero (cancelled) outputs, which is ~1e-3 for the largest k here.
ATOL_RTOL = {
    torch.float16: (1e-2, 2e-3),
    torch.bfloat16: (1e-2, 1.6e-2),
    torch.float: (4e-3, 1e-3),
}

for weight_nz in [False, True]:
    for transpose in [False, True]:
        for dtype in [torch.float16, torch.bfloat16, torch.float]:
            for m in [1, 8]:
                for n in [32, 64, 16128]:
                    for k in [32, 128, 7168, 7184, 7232, 7296, 7424, 7488, 7616]:
                        if transpose and dtype == torch.float:
                            continue
                        x = torch.randn(m, k, dtype=dtype, device="npu:0")
                        if transpose:
                            y = torch.randn(k, n, dtype=dtype, device="npu:0")
                            y_standard = y.transpose(0, 1).contiguous().reshape(n, k)
                        else:
                            y = torch.randn(n, k, dtype=dtype, device="npu:0")
                            y_standard = y
                        z = torch.zeros(m, n, dtype=dtype, device="npu:0")

                        standard = F.linear(x, y_standard, None)

                        if weight_nz and dtype != torch.float:
                            y = torch_npu.npu_format_cast(y, 29)

                        torch.npu.synchronize()
                        matmul(rt, x, y, z, weight_nz=weight_nz and dtype != torch.float, transpose=transpose)
                        torch.npu.synchronize()

                        if transpose:
                            print(f'[{m}, {k}] x [{k}, {n}] {dtype} weight_nz {weight_nz and dtype != torch.float} matmul executed!')
                        else:
                            print(f'[{m}, {k}] x [{n}, {k}] {dtype} weight_nz {weight_nz and dtype != torch.float} matmul executed!')

                        try:
                            atol, rtol = ATOL_RTOL[dtype]
                            torch.testing.assert_close(standard, z, atol=atol, rtol=rtol)
                        except AssertionError as e:
                            print(f'{e}')
                            print(f'torch_npu: {standard}')
                            print(f'xlite: {z}')
