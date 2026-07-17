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
from xlite._C import Runtime, unpack_activation

npu_devid = 0
rt = Runtime(npu_devid, 500)
torch.npu.set_device(npu_devid)

supported_dtype_list = [torch.int8]

test_cases = [
    [20, 64],
    [100, 6144],
    [20000, 2048],
]

def pack_int4_to_int8(x):
    if x.dtype != torch.int8:
        raise TypeError(f"dtype of x should be int8, but get: {x.dtype}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dim should be even, but get: {x.shape}")
    # int8 采用小法端计算
    a = x[..., 1::2]
    b = x[..., 0::2]
    b_adjusted = torch.where(b < 0, b + 16, b)

    result = (a * 16 + b_adjusted)
    return result

def msd_split_activation_int8(activation_int8):
    """将 INT8 激活拆分为高 4bit 和低 4bit"""
    # 高 4bit: 算术右移，补符号位，值域 [-8, 7]
    high4 = (activation_int8.clone() / 16).to(torch.int8)
    # 低 4bit: 先转 uint8 消除符号位干扰，再取低 4bit，值域 [0, 15]
    low4 = (activation_int8.clone() & 0x0F).to(torch.float16) - 8
    low4 = pack_int4_to_int8(low4.to(torch.int8))
    high4 = pack_int4_to_int8(high4)
    merged = torch.cat([low4, high4], dim=0).reshape(activation_int8.shape)
    return merged


for m, n in test_cases:
    in_type = torch.int8
    out_type = torch.int8
    x = torch.randint(low=-128, high=128, size=(m, n), dtype=torch.int8, device="npu")
    z = torch.empty(m, n, dtype=torch.int8, device=f"npu:{npu_devid}")

    expected_z = msd_split_activation_int8(x)

    torch.npu.synchronize()
    unpack_activation(rt, x, z)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(expected_z, z, atol=1, rtol=1/128)
        print(f'unpack_activation [{m}, {n}] output check passed')
    except AssertionError as e:
        print(f'{e}')
        print(f'x: {x}, shape: {x.shape}')
        print(f'expected_z: {expected_z}, shape: {expected_z.shape}')
        print(f'xlite z: {z}, shape: {z.shape}')

