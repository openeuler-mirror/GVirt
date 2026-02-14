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
from xlite._C import Runtime, group_matmul
import torch.nn.functional as F


rt = Runtime(0, 500)
torch.npu.set_device(0)

m = 20
in_dim = 7168
out_dim = 18432
group_num = 8

# Generate random counts that sum to m
counts = torch.ones(group_num, dtype=torch.int32, device="npu:0")
remaining = m - group_num
if remaining > 0:
    random_vals = torch.rand(group_num, device="npu:0")
    random_vals = (random_vals / random_vals.sum() * remaining).round().int()
    sum_random = random_vals.sum().item()
    if sum_random != remaining:
        diff = remaining - sum_random
        random_vals[0] += diff
    counts += random_vals

for transpose in [False, True]:
    for dtype in [torch.float16, torch.bfloat16, torch.float]:
        if transpose and dtype == torch.float:
            continue
        n = out_dim if not transpose else in_dim
        k = in_dim if not transpose else out_dim
        x = torch.randn(m, k, dtype=dtype, device="npu:0")
        weights = []
        weights_standard = []
        for i in range(group_num):
            if transpose:
                weight = torch.randn(k, n, dtype=dtype, device="npu:0")
                weight_standard = weight.transpose(0, 1).contiguous().reshape(n, k)
            else:
                weight = torch.randn(n, k, dtype=dtype, device="npu:0")
                weight_standard = weight.clone()
            weights.append(weight)
            weights_standard.append(weight_standard)

        # standard
        start = 0
        results = []
        for i in range(group_num):
            end = start + counts[i].item()
            x_slice = x[start:end, :]
            result = F.linear(x_slice, weights_standard[i], None)
            results.append(result)
            start = end
        result = torch.cat(results, dim=0)

        # xlite
        z = torch.zeros(m, n, dtype=dtype, device="npu:0")

        torch.npu.synchronize()
        group_matmul(rt, x, weights, [], counts, 0, group_num, n, k, z, False, transpose)
        torch.npu.synchronize()

        print(f'group_matmul ({dtype}, transpose={transpose}) executed!')

        try:
            torch.testing.assert_close(result, z, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'torch_npu: {result}')
            print(f'xlite: {z}')