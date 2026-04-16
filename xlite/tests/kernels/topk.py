#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import Runtime, topk as xlite_topk
import random

K = 2048
n_routed_experts = 6144
max_seq_len = 5000
batches = 100
testIterations = 100


def buildXliteInput(scores):
    res = []
    for s in scores:
        pad = torch.zeros(max_seq_len - len(s))
        res.extend([s, pad])
    return torch.cat(res)


def generate_sequences(input_sizes):
    standard_scores = []
    standard_indices = []

    def get_topk(x):
        k = min(K, t.shape[-1])
        return t.topk(k)[1]

    for n_tokens in input_sizes:
        t = torch.randn(n_tokens)
        standard_scores.append(t)
        standard_indices.append(get_topk(K))
    standard_indices = torch.cat(standard_indices)
    return standard_scores, standard_indices.to(dtype=torch.int32)


def run_test(rt, input_sizes, msg):
    batches = len(input_sizes)
    batch_sizes = torch.tensor(input_sizes, dtype=torch.int32)
    standard_scores, standard_indices = generate_sequences(input_sizes)

    scores = buildXliteInput(standard_scores)
    indices = torch.arange(max_seq_len, dtype=torch.int32).repeat(batches)
    xlite_indices = torch.empty(K * batches, dtype=torch.int32)

    torch.npu.synchronize()
    xlite_topk(rt, scores, indices, xlite_indices, batch_sizes, K, max_seq_len)
    torch.npu.synchronize()

    torch.set_printoptions(edgeitems=7)
    try:
        torch.testing.assert_close(standard_indices, xlite_indices)
        print(f"{msg}: PASS")
    except Exception as e:
        s = ""
        first_bad = -1
        for i, (std, xlite) in enumerate(zip(standard_indices, xlite_indices)):
            if std != xlite and first_bad == -1:
                first_bad = i
            c = ' ' if std == xlite else 'x'
            s += f'[{c}]'
            if (i % 50 == 49):
                s += '\n'
        print(f"FAILED: {first_bad}")
        print(f"input_sizes: {input_sizes}")
        print("standard indices:")
        print(standard_indices)
        print("xlite indices:")
        print(xlite_indices)
        raise e


def main():
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")

    for dtype in [torch.bfloat16, torch.float32]:
        torch.set_default_dtype(torch.float32)
        for i in range(testIterations):
            input_sizes = random.sample(range(K, max_seq_len+1), batches)
            input_sizes = [x // 16 * 16 for x in input_sizes]
            msg = f'{dtype}: [{i}/{testIterations}]'
            run_test(rt, input_sizes, msg)


if __name__ == "__main__":
    main()
