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
size_list = [3, 78, 1035, 10489, 13952]
tokens = 1

for dtype in supported_dtype_list:
    for size in  size_list:
        x = torch.randn(tokens, (size + 127) // 128 * 128, dtype=dtype, device="npu:0")
        for m in range(tokens):
            x[m][size + m:] = float("-inf")
        y = x.clone()
        standard = torch.nn.functional.softmax(x, dim=-1)

        torch.npu.synchronize()
        softmax(rt, y, size, False)

        print(f'softmax size={size} {dtype} executed!')

        try:
            torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'input: {x}')
            print(f'torch_npu: {standard}')
            print(f'xlite: {y}')

size_list = [3, 78, 1035, 10489, 24322, 32640, 48384, 64525, 129050, 145152, 1056640, 2064512]

for dtype in supported_dtype_list:
    for size in  size_list:
        x = torch.randn(tokens, (size + 127) // 128 * 128, dtype=dtype, device="npu:0")
        for m in range(tokens):
            x[m][size + m:] = float("-inf")
        y = x.clone()
        standard = torch.nn.functional.softmax(x, dim=-1)

        torch.npu.synchronize()
        softmax(rt, y, size, True)

        print(f'long softmax size={size} {dtype} executed!')

        try:
            torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'input: {x}')
            print(f'torch_npu: {standard}')
            print(f'xlite: {y}')