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
from xlite._C import Runtime, permutation, unpermutation
import torch.nn.functional as F

rt = Runtime(0, 500)
torch.npu.set_device(0)

DIM = 7168
N_GROUPS = 8
TOPK_GROUPS = 4
TOPK = 8
BATCH_SIZE = 16

for n_routed_experts in [160, 256]:
    START = 0
    END = n_routed_experts
    for dtype in [torch.float, torch.bfloat16]:
        x = torch.randn(BATCH_SIZE, DIM, dtype=dtype, device="npu:0")
        weight = torch.randn(n_routed_experts, DIM, dtype=dtype, device="npu:0")
        bias = torch.randn(n_routed_experts, dtype=dtype, device="npu:0")

        # standard
        scores = F.linear(x, weight, None)
        scores = scores.sigmoid()
        original_scores = scores
        scores = scores + bias
        if N_GROUPS > 1:
            scores = scores.view(x.size(0), N_GROUPS, -1)
            group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(TOPK_GROUPS, dim=-1)[1]
            mask = torch.zeros_like(scores[...,0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, TOPK, dim=-1)[1]
        counts = torch.bincount(indices.flatten(), minlength=n_routed_experts)

        weights = torch.randn(BATCH_SIZE, n_routed_experts, dtype=dtype, device="npu:0")
        standard_token_sorted = torch.zeros_like(x)
        for i in range(START, END):
            if counts[i] == 0:
                continue
            idx, top = torch.where(indices == i)
            standard_token_sorted[idx] += x[idx] * weights[idx, i].unsqueeze(-1)


        # xlite
        unp_idx = torch.empty(n_routed_experts, BATCH_SIZE + 1, dtype=torch.int32, device="npu:0")
        experts_sorted = torch.empty(BATCH_SIZE * TOPK, DIM, dtype=dtype, device="npu:0")
        experts_counts = torch.empty(n_routed_experts, 1, dtype=torch.int32, device="npu:0")
        routing_xlite = torch.zeros((BATCH_SIZE, n_routed_experts // 32), dtype=torch.int32, device=indices.device)
        for batch in range(indices.shape[0]):
            for k in indices[batch]:
                k = k.item()
                m = k // 32
                b = k % 32
                mask = torch.tensor(1 << b, dtype=torch.int64, device=indices.device)
                routing_xlite[batch, m] |= mask

        torch.npu.synchronize()
        permutation(rt, x, routing_xlite, START, END, experts_sorted, unp_idx, experts_counts)
        torch.npu.synchronize()
        experts_counts = experts_counts.view(n_routed_experts).to(dtype=torch.int64)

        print(f'n_routed_experts: {n_routed_experts} dtype: {dtype} permutation executed!')

        try:
            torch.testing.assert_close(counts, experts_counts, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'torch_npu: {counts}')
            print(f'xlite: {experts_counts}')

        if dtype == torch.float:
            continue

        token_sorted = torch.zeros_like(x)
        torch.npu.synchronize()
        unpermutation(rt, experts_sorted, routing_xlite, weights, START, END, token_sorted, unp_idx)
        torch.npu.synchronize()

        print(f'n_routed_experts: {n_routed_experts} dtype: {dtype} unpermutation executed!')

        try:
            torch.testing.assert_close(standard_token_sorted, token_sorted, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            print(f'{e}')
            print(f'torch_npu: {standard_token_sorted}')
            print(f'xlite: {token_sorted}')