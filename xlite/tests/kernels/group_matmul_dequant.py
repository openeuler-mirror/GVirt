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
from xlite._C import Runtime, group_matmul

npu_id = 0
npu_dev = f"npu:{npu_id}"
rt = Runtime(npu_id, 500)
torch.npu.set_device(npu_id)

m = 20
in_dim = 2048
out_dim = 512
group_num = 8

# Generate random counts that sum to m
counts = torch.ones(group_num, dtype=torch.int32, device=npu_dev)
remaining = m - group_num
if remaining > 0:
    random_vals = torch.rand(group_num, device=npu_dev)
    random_vals = (random_vals / random_vals.sum() * remaining).round().int()
    sum_random = random_vals.sum().item()
    if sum_random != remaining:
        diff = remaining - sum_random
        random_vals[0] += diff
    counts += random_vals

def int8_linear_with_dequant(x, weights, counts, dequantize_scales, transpose=False):
    group_num = len(weights)
    results = []
    start = 0

    for i in range(group_num):
        end = start + counts[i].item()
        x_slice = x[start:end, :].to("cpu")  # (count, k), int8
        weight = weights[i]  # int8
        if transpose:
            weight_t = weight.to("cpu")  # (k, n)
        else:
            weight_t = weight.t().to("cpu")  # (k, n)

        result_int32 = torch.matmul(x_slice.to(torch.int32), weight_t.to(torch.int32))  # (count, n), int32
        result_fp32 = result_int32.float() * dequantize_scales[i].to("cpu")  # (count, n), fp32
        result_fp16 = result_fp32.half()  # (count, n), fp16
        results.append(result_fp16)
        start = end

    result = torch.cat(results, dim=0)  # (m, n), fp16
    return result

for weight_nz in [True, False]:
    for transpose in [False, True]:
        n = out_dim if not transpose else in_dim
        k = in_dim if not transpose else out_dim
        dtype = torch.int8
        x = torch.randn(m, k, dtype=dtype, device=npu_dev)
        weights = []
        weights_standard = []
        deq_scales = []
        deq_scales_fixpipe = []
        for i in range(group_num):
            if transpose:
                weight = torch.randn(k, n, dtype=dtype, device=npu_dev)
                weight_standard = weight.transpose(0, 1).contiguous().reshape(n, k)
            else:
                weight = torch.randn(n, k, dtype=dtype, device=npu_dev)
                weight_standard = weight.clone()
            
            if weight_nz:
                ACL_FORMAT_FRACTAL_NZ = 29
                weight = torch_npu.npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ)
            deq_scale = torch.randn(n, dtype=torch.bfloat16, device=npu_dev)
            deq_scale = deq_scale.to(torch.float32).contiguous().clone()
            # fixpipe硬件要求：以uint64_t存储fp32，高位为0，低位为fp32格式的二进制值
            scale = torch.zeros(n * 2, dtype=torch.float32, device=npu_dev)
            scale[0::2] = deq_scale[0::1]
            scale[1::2] = 0
            deq_scales.append(deq_scale)
            deq_scales_fixpipe.append(scale)
            weights.append(weight)
            weights_standard.append(weight_standard)

        # standard
        start = 0
        results = []
        for i in range(group_num):
            end = start + counts[i].item()
            x_slice = x[start:end, :].to("cpu")  # (count, k), int8
            weight = weights[i]  # int8
            if transpose:
                weight_t = weight.to("cpu")  # (k, n)
            else:
                weight_t = weight.t().to("cpu")  # (k, n)

            result_int32 = torch.matmul(x_slice.to(torch.int32), weight_t.to(torch.int32))  # (count, n), int32
            if len(deq_scales) == group_num:
                result_fp32 = result_int32.float() * deq_scales[i].to("cpu")  # (count, n), fp32
            else:
                result_fp32 = result_int32.float()
            result_fp16 = result_fp32.half()  # (count, n), fp16
            results.append(result_fp16)
            start = end
        result = torch.cat(results, dim=0)

        # xlite
        z = torch.zeros(m, n, dtype=torch.float16, device=npu_dev)

        torch.npu.synchronize()
        group_matmul(rt, x, weights, deq_scales_fixpipe, counts, 0, group_num, n, k, z, weight_nz, transpose)
        torch.npu.synchronize()

        print(f'group_matmul_dequant ({dtype}, transpose={transpose}, nz={weight_nz}) executed!')

        # torch.set_printoptions(threshold=100000)

        try:
            torch.testing.assert_close(result, z.to("cpu"), atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'torch_npu: {result}, shape: {result.shape}')
            print(f'xlite: {z}, shape: {z.shape}')