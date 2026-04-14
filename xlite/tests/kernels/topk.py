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
import torch.nn.functional as F
from xlite._C import Runtime, topk as xlite_topk
import numpy as np
import random

rt = Runtime(0, 500)
torch.npu.set_device(0)
torch.set_default_device("npu:0")
torch.set_default_dtype(torch.bfloat16)

# n_tokens = 3000
K = 2048
n_routed_experts = 6144
max_seq_len = 5000
batches = 2
# TODO: check max_seq_len,max_seq_len
input_sizes = random.sample(range(1, max_seq_len+1), batches)
# input_sizes = [2447, 746]
# input_sizes = [2980, 208]
# input_sizes = [3379, 2885]
# input_sizes = [1397, 2882]
print(input_sizes)

batches = len(input_sizes)
input_sizes_tensor = torch.tensor(input_sizes, dtype=torch.int32)

def buildXliteInput(scores):
    paddedSize = torch.Size((max_seq_len, n_routed_experts))
    paddings = [paddedSize[0] - x.shape[0] for x in scores]
    res = []
    for s, pad in zip(scores, paddings):
        padding = (0,0,0,pad)
        res.append(F.pad(s,padding))

    return torch.concat(res)

standard_scores = []
standard_indices = []
for n_tokens in input_sizes:
    t = torch.randn(n_tokens, n_routed_experts)
    standard_scores.append(t)
    standard_indices.append(t.topk(K)[1])
standard_indices = torch.cat(standard_indices)

# values, torch_indices = standard_scores.topk(K)

scores = buildXliteInput(standard_scores)

standard_indices = standard_indices.to(dtype=torch.int32)
indices = torch.arange(n_routed_experts, dtype=torch.int32)

xlite_indices = torch.empty(sum(input_sizes), K, dtype=torch.int32)

torch.npu.synchronize()
xlite_topk(rt, scores, indices, xlite_indices, input_sizes_tensor, K, max_seq_len)
torch.npu.synchronize()

torch.testing.assert_close(standard_indices, xlite_indices)
