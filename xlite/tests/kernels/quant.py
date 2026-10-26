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
from xlite._C import Runtime, quant

npu_devid = 0
rt = Runtime(npu_devid, 500)
torch.npu.set_device(npu_devid)

supported_dtype_list = [torch.bfloat16]  # add torch.float16 here once fp16 kernel is instantiated

# All aligned cases (k % 32 == 0) go through the batched engine via the
# standalone tile dispatch (tileRows = ceil(m / block_num), one tile per
# block); the launch plan comes from tileRows under the actual-k kPad.
# K_TILE_MAX bounds kPad (k above it splits into kLoop chunks).
test_cases = [
    # ---- multi-row tiles (rows >= 2) ----
    [8192, 2048],    # kPad=2048, rows=7, tail group r=2
    [20000, 96],     # kPad=96,  rows=32, tail group r=16
    [20000, 12288],  # kLoop=2,  kPad=6144, rows=2, tail group r=2
    [8190, 2048],    # kPad=2048, rows=7, tail tile r=6
    [8192, 4096],    # kPad=4096, rows=3, full groups
    [20003, 96],     # kPad=96,  rows=32, tail group r=3
    [8192, 6144],    # kLoop=2,  kPad=3072, rows=5, 2D DMA row stride
    [2000, 6272],    # kLoop=2,  kPad=3136, rows=4, 50 full tiles
    [64, 6144],      # tileRows=2, kPad=3072, rows=2
    [99, 6144],      # tileRows=2, last tile single row inside a rows=2 plan
    [100, 6144],     # tileRows=2, full 50 tiles
    [2048, 6144],    # tileRows=41, rows=5, tail tile r=8
    # ---- one-row tiles (rows == 1, kPad > K_TILE_ONE_LINE) ----
    [1, 8192],       # kPad=8192, single chunk, rows=1
    [2, 8192],       # 2 blocks x one-row tiles, single chunk
    [1, 9216],       # kLoop=2, kPad=4608, rows=1
    [2, 6912],       # one-row tiles, kPad=3456, rows=1
    [8, 6144],       # 8 blocks x one-row tiles, 2 chunks each
    [8, 8192],       # 8 blocks x one-row tiles, 2 chunks (4096/4096)
]

for in_type in supported_dtype_list:
    for m, n in test_cases:
        out_type = torch.int8
        x = torch.rand(m, n, dtype=in_type, device=f"npu:{npu_devid}")
        z = torch.empty(m, n, dtype=out_type, device=f"npu:{npu_devid}")
        scale = torch.randn(n, dtype=in_type, device=f"npu:{npu_devid}")
        offset = torch.randn(n, dtype=in_type, device=f"npu:{npu_devid}")

        expected_z = torch_npu.npu_quantize(x, scale, offset, dtype=torch.qint8, div_mode=False)

        torch.npu.synchronize()
        quant(rt, x, scale, offset, z)
        torch.npu.synchronize()

        try:
            torch.testing.assert_close(expected_z, z, atol=0, rtol=0)
            print(f'Quant({in_type} -> i8) [{m}, {n}] check passed')
        except AssertionError as e:
            print(f'{e}')
            print(f'x: {x}, shape: {x.shape}')
            print(f'scale: {scale}')
            print(f'offset: {offset}')
            print(f'expected: {expected_z}, shape:{expected_z.shape}')
            print(f'xlite: {z}, shape: {z.shape}')
