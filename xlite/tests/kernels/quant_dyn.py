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

supported_dtype_list = [torch.bfloat16]

test_cases = [
    [8192, 2048],
    [40, 96],
    [200000, 96],
]

for m, n in test_cases:
    in_type = torch.bfloat16
    out_type = torch.int8
    x = torch.randn(m, n, dtype=in_type, device=f"npu:{npu_devid}")
    z = torch.empty(m, n, dtype=out_type, device=f"npu:{npu_devid}")
    scale = torch.zeros(m, dtype=torch.float, device=f"npu:{npu_devid}")

    expected_z, expected_scale = torch_npu.npu_dynamic_quant(x)

    torch.npu.synchronize()
    quant_dynamic(rt, x, scale, z)
    torch.npu.synchronize()

    # torch.set_printoptions(threshold=100000)

    try:
        torch.testing.assert_close(expected_z, z, atol=1, rtol=1/128)
        print(f'quant_dyn [{m}, {n}] output check passed')
    except AssertionError as e:
        print(f'{e}')
        print(f'x: {x}, shape: {x.shape}')
        print(f'expected z: {expected_z}, shape:{expected_z.shape}')
        print(f'xlite z: {z}, shape: {z.shape}')

    try:
        torch.testing.assert_close(expected_scale, scale, atol=1e-5, rtol=1e-3)
        print(f'quant_dyn [{m}, {n}] scale check passed')
    except AssertionError as e:
        print(f'{e}')
        print(f'x: {x}, shape: {x.shape}')
        print(f'expected scale: {expected_scale}, shape:{expected_scale.shape}')
        print(f'xlite scale: {scale}, shape:{scale.shape}')
