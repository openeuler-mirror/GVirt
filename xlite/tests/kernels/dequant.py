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
from xlite._C import Runtime, dequant

rt = Runtime(0, 500)
torch.npu.set_device(0)
cases = [
    [8192, 768],
    [16, 2048],
    [20000, 5120],
    [40, 192],
]

for m, n in cases:
    for has_scale in [False, True]:
        x = torch.randn(m, n, dtype=torch.float16, device="npu:0")
        scales = torch.randn(m, dtype=torch.float, device="npu:0")
        z = torch.empty(m, n, dtype=torch.bfloat16, device="npu:0")

        # torch 采用四舍五入，偶数优先的策略
        if has_scale:
            standard = (x.to(torch.float) * scales.unsqueeze(-1)).to(torch.bfloat16)
        else:
            standard = x.to(torch.float).to(torch.bfloat16)

        torch.npu.synchronize()
        dequant(rt, x, scales, z, has_scale)
        torch.npu.synchronize()
        print(f'Dequant [{m}, {n}] torch.float16 to torch.bfloat { "with scale " if has_scale else "" }executed!')

        try:
            torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'x: {x}')
            print(f'y: {scales}')
            print(f'torch_npu: {standard}')
            print(f'xlite: {z}')