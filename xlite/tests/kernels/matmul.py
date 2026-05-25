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
torch.npu.config.allow_internal_format = True

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
                            y_standard = y.clone()
                        z = torch.zeros(m, n, dtype=dtype, device="npu:0")

                        standard = F.linear(x, y_standard, None)

                        if weight_nz and dtype != torch.float:
                            y = matrix_nd2nz(y)

                        torch.npu.synchronize()
                        matmul(rt, x, y, z, weight_nz and dtype != torch.float, transpose)
                        torch.npu.synchronize()

                        if transpose:
                            print(f'[{m}, {k}] x [{k}, {n}] {dtype} weight_nz {weight_nz and dtype != torch.float} matmul executed!')
                        else:
                            print(f'[{m}, {k}] x [{n}, {k}] {dtype} weight_nz {weight_nz and dtype != torch.float} matmul executed!')

                        try:
                            torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
                        except AssertionError as e:
                            print(f'{e}')
                            print(f'torch_npu: {standard}')
                            print(f'xlite: {z}')
