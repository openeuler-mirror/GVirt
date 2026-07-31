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
from xlite._C import Runtime, matmul_dequant, matmul, matmul_with_bias

# allow weight_nz
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

dev_id=2
debug = False

rt = Runtime(dev_id, 500)
torch.npu.set_device(dev_id)

# M = 64          # 输入行数
# K = 2048        # 输入列数 / 权重行数
# N = 64           # 权重列数（需 8 对齐，因为每 8 个 int4 打包为一个 int32）

test_cases = [
    [1078, 512, 1024],
    [64, 64, 2048],
    [5, 6144, 1024],
    [64, 256, 1024],
    [64, 64, 2048],
    [64, 2048, 1024],
    [64, 2048, 2048],
    [8192, 768, 2048],
    [64, 2048, 6144],
    [103, 3072, 4096],
    [512, 768, 2048],
]

DEVICE = f"npu:{dev_id}"

def pack_int4_to_int8(x):
    if x.dtype != torch.int8:
        raise TypeError(f"dtype of x should be int8, but get: {x.dtype}")
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"last dim should be even, but get: {x.shape}")
    # int8 采用小法端计算
    a = x[..., 1::2]
    b = x[..., 0::2]
    b_adjusted = torch.where(b < 0, b + 16, b)

    result = (a * 16 + b_adjusted).clone().contiguous()
    return result

def get_low_4bits_int8(activation_int8):
    """将 INT8 激活拆分为高 4bit 和低 4bit"""
    # 低 4bit: 先转 uint8 消除符号位干扰，再取低 4bit，值域 [0, 15]
    low4 = (activation_int8.clone() & 0x0F).to(torch.float16)
    low4 = torch.where(low4 >= 8, low4 - 16, low4)
    return pack_int4_to_int8(low4.to(torch.int8))


def npu_quant_matmul_cpu(
    x: torch.Tensor,                    # INT8 [M, K]
    weight: torch.Tensor,               # INT8 [N, K] (注意 NPU 通常存储为 [N, K])
    deq_scale: torch.Tensor,            # FP32, shape 可广播到 [M, N]
    bias: torch.Tensor,                 # INT32 量化偏置 [N]
    output_dtype: torch.dtype = torch.float16,
    transpose: bool = False
) -> torch.Tensor:
    """
    模拟 torch_npu.npu_quant_matmul
    公式: output = (matmul(x, weight^T) + bias) * deq_scale
    """
    x_i32 = x.to("cpu").to(torch.int32)
    w_i32 = weight.to("cpu").to(torch.int32)

    if transpose:
        acc_i32 = torch.matmul(x_i32, w_i32)
    else:
        acc_i32 = torch.matmul(x_i32, w_i32.t())

    if bias is not None:
        if bias.dim() == 1:
            bias_i32 = bias.to(torch.int32).to("cpu").view(1, -1)
        else:
            bias_i32 = bias.to(torch.int32).to("cpu")
        acc_i32 = acc_i32 + bias_i32

    if deq_scale.dim() == 1 and deq_scale.shape[0] == acc_i32.shape[-1]:
        deq_scale = deq_scale.to("cpu").view(1, -1)

    output = acc_i32.float() * deq_scale
    return output.to(output_dtype)


def block_sum_16x16_torch(tensor):
    m, n = tensor.shape
    m_valid = m // 16 * 16
    n_valid = n // 16 * 16
    padded = tensor[:m_valid, :n_valid]
    
    M = padded.shape[0] // 16
    N = padded.shape[1] // 16
    reshaped = padded.reshape(M, 16, N, 16)
    return reshaped.sum(dim=(1, 3))  # shape (M, N)

for M, N, K in test_cases:
    for transpose in [True, False]:
        for weight_nz in [True, False]:

            x = torch.randint(low=-8, high=8, size=(M, K), dtype=torch.int8, device=DEVICE)
            x_i32 = x.to(torch.int32)

            # 权重: int32 承载 int4 数据，值必须在 [-8, 7] 范围内
            if transpose:
                weight_int32 = torch.randint(low=-8, high=8, size=(K, N), dtype=torch.int32, device=DEVICE)
            else:
                weight_int32 = torch.randint(low=-8, high=8, size=(N, K), dtype=torch.int32, device=DEVICE)

            antiquant_scale = torch.randn(N, dtype=torch.bfloat16, device=f"npu:{dev_id}")
            scale = torch.zeros(N * 2, dtype=torch.float32, device=f"npu:{dev_id}")
            scale[0::2] = antiquant_scale.to(torch.float)[0::1]
            scale[1::2] = 0
            bias = torch.randn(N, dtype=torch.int32, device=f"npu:{dev_id}")

            standard = npu_quant_matmul_cpu(x, weight_int32, deq_scale=antiquant_scale, bias=bias, transpose=transpose)

            if weight_nz:
                ACL_FORMAT_FRACTAL_NZ = 29
                weight_int32 = torch_npu.npu_format_cast(weight_int32, ACL_FORMAT_FRACTAL_NZ)

            weight_int4pack = torch_npu.npu_convert_weight_to_int4pack(weight_int32)
            x_int4_pack = torch_npu.npu_convert_weight_to_int4pack(x_i32)
            z = torch.zeros(M, N, dtype=torch.float16, device=f"npu:{dev_id}")

            torch.npu.synchronize()
            matmul_dequant(rt, x_int4_pack, weight_int4pack, bias, scale, z, weight_nz, transpose)
            torch.npu.synchronize()

            try:
                torch.testing.assert_close(standard, z.to("cpu"), atol=1, rtol=1e-2)
                print(f'[{M}, {K}] x [{K}, {N}](transpose = {transpose}) int4 weight_nz: {weight_nz} matmul passed!')
            except AssertionError as e:
                print(f'[{M}, {K}] x [{K}, {N}](transpose = {transpose}) int4 weight_nz: {weight_nz} matmul failed!')
                print(f'{e}')
                print(f'torch_npu: {standard}')
                print(f'xlite: {z}')