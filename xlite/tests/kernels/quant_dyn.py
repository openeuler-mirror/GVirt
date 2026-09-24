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
    [40, 96],
    [200000, 96],
    # k > QUANT_DYN_K_TILE(8192): exercises the k-tiling path.
    [8, 12288],      # GLM-5 inter_dim, small m, large k
    [8192, 12288],   # large m, large k (previously overflowed)
    [8, 24576],      # multi-tile (3 tiles), k_loop>1
    # ---- boundary cases (QUANT_DYN_K_TILE = 8192) ----
    [1, 1],          # minimal m and k
    [1, 8192],       # single row, k == K_TILE (fast path, no tail)
    [1, 8193],       # single row, smallest tiled path (tail k_size=1)
    [8, 8191],       # fast path, k not multiple of 128 (tail padding)
    [8, 16384],      # tiled path, exactly 2 full tiles, no tail
    [8, 100],        # small k, not multiple of 128
    [4096, 8192],    # large m, k == K_TILE (fast-path boundary)
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
    # sparse row: one dominant element among zeros exercises absmax reduction.
    special_inputs = [
        (torch.zeros(8, 8192, dtype=in_type, device=f"npu:{npu_devid}"),
         "zero-input [8, 8192] (fast path)"),
        (torch.zeros(4, 8193, dtype=in_type, device=f"npu:{npu_devid}"),
         "zero-input [4, 8193] (tiled path)"),
    ]
    x_sparse = torch.zeros(4, 8193, dtype=in_type, device=f"npu:{npu_devid}")
    x_sparse[1, 5000] = 100.0
    special_inputs.append((x_sparse, "sparse-input [4, 8193]"))

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
