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
from xlite._C import Runtime, quant_dynamic

npu_devid = 0
rt = Runtime(npu_devid, 500)
torch.npu.set_device(npu_devid)

supported_dtype_list = [torch.bfloat16]  # add torch.float16 here once fp16 kernel is instantiated

test_cases = [
    [8192, 2048],
    [8190, 2048],    # engine rows=7, tail group r=2
    [8192, 4096],    # engine boundary k=4096, rows=3
    [40, 96],
    [200000, 96],
    # k > K_TILE_MAX: chunked engine (kLoop > 1).
    [8, 12288],      # GLM-5 inter_dim, small m, large k: engine rows=1 x 2 chunks
    [8192, 12288],   # large m, large k: engine rows=2 x 2 chunks
    [8, 24576],      # multi-tile (3 tiles), engine rows=1 x 3 chunks
    [100, 12288],    # tileRows=2: engine rows=2 x 2 chunks per tile
    [128, 12288],    # tileRows=3: engine rows=2 with a tail r=1 group
    [51, 12288],     # engine tiles + 1-row tail tile mix at chunked width
    [200, 16384],    # rows=1 plan (kPad=8192): engine rows=1 x 2 chunks
    [100, 8224],     # aligned k, tail chunk 4000 not 128-balanced (kPad-width compute)
    [8, 12480],      # aligned k, tail chunk 6208 not 128-balanced, one-row tiles
    # ---- boundary cases (K_TILE_MAX = 8192) ----
    [1, 8192],       # single row, k == K_TILE (resident chunk, no tail)
    [8, 16384],      # chunked engine, exactly 2 full chunks, no tail
    [4096, 8192],    # large m, k == K_TILE (chunk-width boundary)
    # ---- m tile boundaries (standalone block_num = min(m, aivNum), aivNum=50) ----
    # tileRows crosses 1 -> 2 at m = aivNum + 1: m <= 50 takes one-row tiles
    # (engine rows=1), m > 50 takes multi-row tiles with a ragged tail tile
    # whenever m is odd
    [49, 2048],      # tileRows=1, blocks 0..48 each one row
    [50, 2048],      # tileRows=1 boundary exactly at aivNum
    [51, 2048],      # tileRows=2, tail 1-row tile (rows=1 group)
    [99, 2048],      # tileRows=2, tail 1-row tile at the last block
    [100, 2048],     # tileRows=2, no tail
    [101, 2048],     # tileRows=3, tiles=34 (blocks 34..49 idle)
]

for in_type in supported_dtype_list:
    for m, n in test_cases:
        out_type = torch.int8
        x = torch.randn(m, n, dtype=in_type, device=f"npu:{npu_devid}")
        z = torch.empty(m, n, dtype=out_type, device=f"npu:{npu_devid}")
        scale = torch.zeros(m, dtype=torch.float, device=f"npu:{npu_devid}")

        expected_z, expected_scale = torch_npu.npu_dynamic_quant(x)

        torch.npu.synchronize()
        quant_dynamic(rt, x, scale, z)
        torch.npu.synchronize()

        # torch.set_printoptions(threshold=1000000)

        try:
            torch.testing.assert_close(expected_z, z, atol=1, rtol=1/128)
            print(f'quant_dyn({in_type} -> i8) [{m}, {n}] output check passed')
        except AssertionError as e:
            print(f'{e}')
            print(f'x: {x}, shape: {x.shape}')
            print(f'expected z: {expected_z}, shape:{expected_z.shape}')
            print(f'xlite z: {z}, shape: {z.shape}')

        try:
            torch.testing.assert_close(expected_scale, scale, atol=1e-5, rtol=1e-3)
            print(f'quant_dyn({in_type} -> i8) [{m}, {n}] scale check passed')
        except AssertionError as e:
            print(f'{e}')
            print(f'x: {x}, shape: {x.shape}')
            print(f'expected scale: {expected_scale}, shape:{expected_scale.shape}')
            print(f'xlite scale: {scale}, shape: {scale.shape}')

    # ---- special-input edge cases ----
    # absmax == 0 (all-zero row): scaleRec = 127/0 division edge; the kernel must
    # still yield z == 0 / scale == 0, matching torch_npu.npu_dynamic_quant.
    special_inputs = [
        (torch.zeros(8, 8192, dtype=in_type, device=f"npu:{npu_devid}"),
         "zero-input [8, 8192] (engine rows=1)"),
        (torch.zeros(8192, 2048, dtype=in_type, device=f"npu:{npu_devid}"),
         "zero-input [8192, 2048] (batched groups, incl. tail group)"),
        (torch.zeros(100, 12288, dtype=in_type, device=f"npu:{npu_devid}"),
         "zero-input [100, 12288] (chunked engine)"),
    ]

    for x_sp, label in special_inputs:
        m, n = x_sp.shape
        z_sp = torch.empty(m, n, dtype=out_type, device=f"npu:{npu_devid}")
        scale_sp = torch.zeros(m, dtype=torch.float, device=f"npu:{npu_devid}")
        expected_z, expected_scale = torch_npu.npu_dynamic_quant(x_sp)
        torch.npu.synchronize()
        quant_dynamic(rt, x_sp, scale_sp, z_sp)
        torch.npu.synchronize()
        try:
            torch.testing.assert_close(expected_z, z_sp, atol=1, rtol=1/128)
            torch.testing.assert_close(expected_scale, scale_sp, atol=1e-5, rtol=1e-3)
            print(f'quant_dyn({in_type} -> i8) {label} check passed')
        except AssertionError as e:
            print(f'{e}')
            print(f'x: {x_sp}, shape: {x_sp.shape}')
            print(f'expected z: {expected_z}, shape:{expected_z.shape}')
            print(f'xlite z: {z_sp}, shape: {z_sp.shape}')
            print(f'expected scale: {expected_scale}, shape:{expected_scale.shape}')
            print(f'xlite scale: {scale_sp}, shape: {scale_sp.shape}')
