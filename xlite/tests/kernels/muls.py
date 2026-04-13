#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import Runtime, muls


rt = Runtime(0, 500)
torch.npu.set_device(0)

supported_dtype_list = [torch.float16, torch.bfloat16]

for dtype in supported_dtype_list:
    x = torch.randn(8, 2048, dtype=dtype, device="npu:0")
    y = torch.empty(8, 2048, dtype=dtype, device="npu:0")
    scale = 2.5

    standard = x * scale

    torch.npu.synchronize()
    muls(rt, x, scale, y)
    torch.npu.synchronize()
    print(f'muls {dtype} executed!')

    try:
        torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        print(f'{e}')
        print(f'x: {x}')
        print(f'scale: {scale}')
        print(f'torch_npu: {standard}')
        print(f'xlite: {y}')
