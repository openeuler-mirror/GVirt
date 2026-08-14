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
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

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
    """将 INT8 激活拆分为高 4bit 和低 4bit (interleaved per token, adjacent rows).

    Output layout [2m, k_half] (k_half = n/2): for token r, output row 2r holds the low
    nibbles and row 2r+1 holds the high nibbles — i.e. each token's low/high halves are
    placed in adjacent rows rather than split into two disjoint column regions. This
    matches the kernel's native [2m, k_half] interleaved-row output directly.
    """
    # 高 4bit: 算术右移 = floor(x/16), 值域 [-8, 7].
    # 必须用整型算术右移而非浮点除法+to(int8): 后者按向零截断(trunc)取整, 对负数
    # (如 x=-27: trunc=-1, floor=-2) 会与 kernel 的 floor 高 nibble 不一致, 破坏
    # 补码恒等式 x = high4*16 + (low4_raw & 0x0F).
    high4 = (activation_int8.to(torch.int32) >> 4).to(torch.int8)
    # 低 4bit: 先转 uint8 消除符号位干扰，再取低 4bit，值域 [0, 15]
    low4 = (activation_int8.clone() & 0x0F).to(torch.float16) - 8
    low4 = pack_int4_to_int8(low4.to(torch.int8))
    high4 = pack_int4_to_int8(high4)
    # 同一 token 的低/高 4bit 放相邻行: row 2r = low, row 2r+1 = high
    m = activation_int8.shape[0]
    merged = torch.stack([low4, high4], dim=1).reshape(2 * m, -1)
    return merged


for m, n in test_cases:
    in_type = torch.int8
    out_type = torch.int8
    x = torch.randint(low=-128, high=128, size=(m, n), dtype=torch.int8, device="npu")
    # 输出为 [2m, n/2]: 同一 token 的低/高 4bit 写入相邻行 (row 2r / 2r+1)
    z = torch.empty(2 * m, n // 2, dtype=torch.int8, device=f"npu:{npu_devid}")

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

