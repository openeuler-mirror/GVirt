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
from xlite._C import Runtime, matmul
import torch.nn.functional as F
from tests.models.weight_utils import matrix_nd2nz


rt = Runtime(0, 500)
torch.npu.set_device(0)

weight_nz = True
for dtype in [torch.float16, torch.bfloat16, torch.float]:
    for m in [1, 8]:
        for n in [32, 64, 16128]:
            for k in [32, 128, 7168, 7184, 7232, 7296, 7424, 7488, 7616]:
                x = torch.randn(m, k, dtype=dtype, device="npu:0")
                y = torch.randn(n, k, dtype=dtype, device="npu:0")
                z = torch.zeros(m, n, dtype=dtype, device="npu:0")

                standard = F.linear(x, y, None)

                if weight_nz:
                    y.copy_(matrix_nd2nz(y))

                torch.npu.synchronize()
                matmul(rt, x, y, z, weight_nz)
                torch.npu.synchronize()
                print(f'[{m}, {k}] x [{n}, {k}] {dtype} matmul executed!')

                try:
                    torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
                except AssertionError as e:
                    print(f'{e}')
                    print(f'torch_npu: {standard}')
                    print(f'xlite: {z}')
