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
from xlite._C import Runtime, matmul_quant

dev_id=0
debug = False

rt = Runtime(dev_id, 500)
torch.npu.set_device(dev_id)

def npu_quant_matmul_cpu(
    x: torch.Tensor,                    # INT8 [M, K]
    weight: torch.Tensor,               # INT8 [N, K] (注意 NPU 通常存储为 [N, K])
    deq_scale: torch.Tensor,            # FP32, shape 可广播到 [M, N]
    bias: torch.Tensor,                 # INT32 量化偏置 [N]
    output_dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    模拟 torch_npu.npu_quant_matmul
    公式: output = (matmul(x, weight^T) + bias) * deq_scale
    """
    x_i32 = x.to(torch.int32)
    w_i32 = weight.to(torch.int32)

    if weight.shape[0] == x.shape[-1]:
        acc_i32 = torch.matmul(x_i32, w_i32)
    else:
        acc_i32 = torch.matmul(x_i32, w_i32.t())

    if bias is not None:
        if bias.dim() == 1:
            bias_i32 = bias.to(torch.int32).view(1, -1)
        else:
            bias_i32 = bias.to(torch.int32)
        acc_i32 = acc_i32 + bias_i32

    if deq_scale.dim() == 1 and deq_scale.shape[0] == acc_i32.shape[-1]:
        deq_scale = deq_scale.view(1, -1)

    if debug:
        print(acc_i32)
    output = acc_i32.float() * deq_scale

    return output.to(output_dtype)

def gen_tf32_rand(n, device='cpu'):
    """
    fixpipe 采用 TF32 格式, 1 符号位, 8 指数位, 10 尾数位, 后 13 位要清零
    """
    sign = torch.randint(0, 2, (n,), dtype=torch.int32, device=device) << 31      # bit 31
    exp  = torch.randint(125, 130, (n,), dtype=torch.int32, device=device) << 23    # bits 23-30
    mant = torch.randint(0, 512, (n,), dtype=torch.int32, device=device) << 13   # bits 13-22
    return (sign | exp | mant).view(torch.float32).clone()

weight_nz = False
transpose = False
dtype = torch.int8
m = 8192
n = 768
k = 2048
x = torch.randn(m, k, dtype=dtype, device=f"npu:{dev_id}")
y = torch.randn(k, n, dtype=dtype, device=f"npu:{dev_id}")
y_standard = y.transpose(0, 1).contiguous().reshape(n, k)
y_in = y_standard.clone()
bias = torch.randn(n, dtype=torch.int32, device=f"npu:{dev_id}")
# deqscale 从 bf16 到 tf32 可以无损转换，因此可以确保 bf16 随机数的 cpu 计算的结果与 npu 一致
deqScale = torch.randn(n, dtype=torch.bfloat16, device=f"npu:{dev_id}").to(torch.float32)
# print(deqScale)
z = torch.zeros(m, n, dtype=torch.float16, device=f"npu:{dev_id}")

torch.set_printoptions(sci_mode=False, precision=4, threshold=1000)

standard = npu_quant_matmul_cpu(x.to("cpu"), y.to("cpu"), deqScale.to("cpu"), bias.to("cpu"), torch.float16)

torch.npu.synchronize()
# fixpipe硬件要求：以uint64_t存储fp32，高位为0，低位为fp32格式的二进制值
scale = torch.zeros(n * 2, dtype=torch.float32, device=f"npu:{dev_id}")
scale[0::2] = deqScale[0::1]
scale[1::2] = 0
matmul_quant(rt, x, y_in, bias, scale, z, weight_nz and dtype != torch.float, transpose)
torch.npu.synchronize()

if transpose:
    print(f'[{m}, {k}] x [{k}, {n}] {dtype} weight_nz {weight_nz and dtype != torch.float} matmul executed!')
else:
    print(f'[{m}, {k}] x [{n}, {k}] {dtype} weight_nz {weight_nz and dtype != torch.float} matmul executed!')
try:
    torch.testing.assert_close(standard, z.to("cpu"), atol=1e-5, rtol=1e-3)
except AssertionError as e:
    print(f'{e}')
    print(f'torch_npu: {standard}')
    print(f'xlite: {z}')
