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


# Large-N multi-iteration case: the 8x2048 case above runs one row per core (one iteration),
# which only exercises the priming flags. This drives many iterations per core over several
# seeds to surface a multi-iteration flag race.
print('=== muls large-N multi-iteration (nope-region scale shape) ===')
large_n_cases = [
    (5120, 128, 64),
    (5120, 192, 128),
    (5120, 576, 512),
]
n_seeds = 20
for dtype in supported_dtype_list:
    for N, D, calc_num in large_n_cases:
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            x = torch.randn(N, D, dtype=dtype, device="npu:0")
            scale = (D ** -0.5) * 2.0
            y = x.clone()

            torch.npu.synchronize()
            muls(rt, y, scale, y, calc_offset=0, calc_num=calc_num)
            torch.npu.synchronize()

            # rebuild the input from the same seed for the f32 reference
            torch.manual_seed(seed)
            x_ref = torch.randn(N, D, dtype=dtype, device="npu:0")
            standard = (x_ref.float() * scale).to(dtype)

            try:
                torch.testing.assert_close(standard[:, :calc_num], y[:, :calc_num],
                                           atol=1e-5, rtol=1e-3)
                torch.testing.assert_close(x_ref[:, calc_num:], y[:, calc_num:],
                                           atol=0, rtol=0)
                print(f'muls {dtype} N={N} D={D} cn={calc_num} seed={seed} executed!')
            except AssertionError as e:
                print(f'{e}')
                print(f'muls {dtype} N={N} D={D} cn={calc_num} seed={seed} FAILED')
