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
from xlite._C import Runtime, softmax_topk
import numpy as np

rt = Runtime(0, 500)
torch.npu.set_device(0)

topK = 8
n_routed_experts = 128
n_tokens = 8
n_indices_map = int(n_routed_experts / 64)

for dtype in [torch.bfloat16, torch.float32]:
    scores = torch.randn(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
    indices = torch.arange(n_routed_experts, dtype=torch.int32, device="npu:0")
    xlite_indices_out = torch.empty(n_tokens, n_indices_map, dtype=torch.int64, device="npu:0")
    xlite_weights_out = torch.empty(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")

    standard_indices_out = torch.zeros(n_tokens, n_indices_map, dtype=torch.int64, device="npu:0")
    standard_weights_out = torch.zeros(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
    standard_softmax = scores.softmax(dim=-1, dtype=dtype)
    standard_weights, standard_indices = torch.topk(standard_softmax, topK, dim=-1)
    standard_weights = standard_weights / standard_weights.sum(dim=-1, keepdim=True)

    for i in range(n_tokens):
        for j in range(topK):
            elem = np.int64(standard_indices[i][j].cpu())
            idx = int(elem / 64)
            value = np.uint64(standard_indices_out[i][idx].cpu()) | np.uint64(1 << (elem % 64))
            standard_indices_out[i][idx] = value.view(np.int64)
            standard_weights_out[i][elem] = standard_weights[i][j]

    torch.npu.synchronize()
    softmax_topk(rt, scores, indices, xlite_weights_out, xlite_indices_out, topK, True)
    torch.npu.synchronize()
    print(f'softmax topK (dtype={dtype}) executed!')

    try:
        torch.testing.assert_close(standard_indices_out, xlite_indices_out, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(standard_weights_out, xlite_weights_out, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        print(f'{e}')
        print(f'standard_indices_out: {standard_indices_out}')
        print(f'xlite_indices_out: {xlite_indices_out}')
        print(f'standard_weights_out: {standard_weights_out}')
        print(f'xlite_weights_out: {xlite_weights_out}')