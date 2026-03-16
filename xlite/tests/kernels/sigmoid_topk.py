#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import random
from xlite._C import Runtime, sigmoid_topk
import numpy as np
import math

rt = Runtime(0, 500)
torch.npu.set_device(0)

n_tokens = 9

models = [
    (160, 1, 1, 8),
    (256, 8, 4, 8),
]

for dtype in [torch.bfloat16, torch.float32]:
    for n_routed_experts, n_group, topk_group, topK in models:
        n_indices_map = math.ceil(n_routed_experts / 32)
        scale = 2.5
        scores = torch.randn(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
        bias = torch.randn(n_routed_experts, dtype=torch.float32, device="npu:0")
        indices = torch.arange(n_routed_experts, dtype=torch.int32, device="npu:0")
        xlite_indices_out = torch.empty(n_tokens, n_indices_map, dtype=torch.int32, device="npu:0")
        xlite_weights_out = torch.empty(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")

        scores_sigmoid = scores.sigmoid()
        standard_sigmoid = scores_sigmoid + bias
        group_scores = (
            standard_sigmoid.view(-1, n_group, n_routed_experts // n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, n_routed_experts // n_group)
            .reshape(-1, n_routed_experts)
        )
        standard_sigmoid = standard_sigmoid.masked_fill(~score_mask.bool(), 0.0)
        standard_indices = torch.topk(standard_sigmoid, k=topK, dim=-1, sorted=False)[1]
        standard_indices, _ = torch.sort(standard_indices, descending=False)
        standard_weights = scores_sigmoid.gather(1, standard_indices)
        standard_weights = standard_weights / standard_weights.sum(dim=-1, keepdim=True)
        standard_weights = standard_weights * scale
        standard_weights = standard_weights.to(dtype)

        torch.npu.synchronize()
        sigmoid_topk(rt, scores, indices, bias, scale, xlite_weights_out, xlite_indices_out, n_group, topk_group, topK, True)
        torch.npu.synchronize()
        print(f'sigmoid topK (n_routed_experts={n_routed_experts}, n_group={n_group}, topk_group={topk_group}, dtype={dtype}) executed!')

        # Convert Xlite outputs from bitmaps to lists of indices
        xlite_indices = torch.zeros(*standard_indices.shape, dtype=torch.int32, device="npu:0")

        for i in range(n_tokens):
            bitmap = 0
            for j in reversed(range(xlite_indices_out.shape[1])):
                bitmap = (bitmap << 32) | int(np.uint32(xlite_indices_out[i][j].cpu()))

            for pos in range(n_routed_experts):
                if bitmap & (1 << pos):
                    xlite_indices[i][j] = pos
                    j += 1

        # Convert Xlite weights from sparse format to a sorted list
        mask = torch.ne(torch.zeros(xlite_weights_out.shape, device="npu:0"), xlite_weights_out)
        xlite_weights = torch.masked_select(xlite_weights_out, mask).reshape(standard_weights.shape)

        try:
            # Compare scores and weights. Score values may become very close after applying sigmoid
            # function. As long as both algorithms select elements with close sigmoid values it does
            # not matter which ones are choosen.
            torch.testing.assert_close(standard_indices.to(dtype=torch.int32), xlite_indices, atol=1e-4, rtol=1e-2)
            torch.testing.assert_close(standard_weights, xlite_weights, atol=1e-4, rtol=1e-2)
        except AssertionError as e:
            print(f'{e}')
            print(f'standard_indices:\n{standard_indices}')
            print(f'xlite_indices:\n{xlite_indices}')
            print(f'standard_weights:\n{standard_weights}')
            print(f'xlite_weights:\n{xlite_weights}')
