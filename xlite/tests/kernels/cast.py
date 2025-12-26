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
from xlite._C import Runtime, cast_up

rt = Runtime(0, 500)
torch.npu.set_device(0)

x = torch.randn(8, 2048, dtype=torch.bfloat16, device="npu:0")
y = torch.empty(8, 2048, dtype=torch.float, device="npu:0")
standard = x.float()

torch.npu.synchronize()
cast_up(rt, x, y)
torch.npu.synchronize()
print(f'cast torch.bfloat16 to torch.float executed!')

try:
    torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
except AssertionError as e:
    print(f'{e}')
    print(f'x: {x}')
    print(f'y: {y}')
    print(f'torch_npu: {standard}')
    print(f'xlite: {y}')