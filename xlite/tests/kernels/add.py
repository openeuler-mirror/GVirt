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
import torch_npu
from xlite._C import runtime, add


rt = runtime(0, 500)
torch.npu.set_device(0)

x = torch.randn(8, 2048, dtype=torch.float16, device="npu:0")
y = torch.randn(8, 2048, dtype=torch.float16, device="npu:0")
z = torch.empty(8, 2048, dtype=torch.float16, device="npu:0")

standard = x + y

torch.npu.synchronize()
add(rt, x, y, z)
torch.npu.synchronize()

try:
    torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
except AssertionError as e:
    print(f'{e}')
    print(f'torch_npu: {standard}')
    print(f'xlite: {z}')