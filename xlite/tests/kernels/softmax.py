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
from xlite._C import Runtime, softmax

rt = Runtime(0, 500)
torch.npu.set_device(0)

supported_dtype_list = [torch.float16, torch.bfloat16]
size_list = [3, 78, 1035, 10489, 16321, 24320, 32640]

for dtype in supported_dtype_list:
    for size in  size_list:
        if dtype == torch.bfloat16 and size > 24320:
            continue
        x = torch.randn(1, size, dtype=dtype, device="npu:0")
        y = x.clone()
        standard = torch.nn.functional.softmax(x, dim=-1)

        torch.npu.synchronize()
        softmax(rt, y, False)

        print(f'softmax size={size} {dtype} executed!')

        try:
            torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'input: {x}')
            print(f'torch_npu: {standard}')
            print(f'xlite: {y}')

size_list = [3, 78, 1035, 10489, 32640, 48384, 64525, 129050, 145152, 2056320, 4145280]

for dtype in supported_dtype_list:
    for size in  size_list:
        if dtype == torch.bfloat16 and size > 2056320:
            continue
        x = torch.randn(1, size, dtype=dtype, device="npu:0")
        y = x.clone()
        standard = torch.nn.functional.softmax(x, dim=-1)

        torch.npu.synchronize()
        softmax(rt, y, True)

        print(f'long softmax size={size} {dtype} executed!')

        try:
            torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'input: {x}')
            print(f'torch_npu: {standard}')
            print(f'xlite: {y}')