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

    for n_tokens in input_sizes:
        t = torch.randn(n_tokens)
        standard_scores.append(t)
        if (n_tokens >= K):
            standard_indices.append(t.topk(K)[1])
        else:
            standard_indices.append(t.topk(n_tokens)[1])
            padding = torch.zeros(K - n_tokens, dtype=torch.int32)
            standard_indices.append(padding)

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
    xlite_topk(rt, scores, indices, xlite_indices, batch_sizes, K)
    torch.npu.synchronize()

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

    for dtype in reversed([torch.float32, torch.bfloat16]):
        torch.set_default_dtype(dtype)
        for i in range(testIterations):
            input_sizes = random.sample(range(1, max_seq_len+1), batches)
            msg = f'{dtype}: [{i+1}/{testIterations}]'
            run_test(rt, input_sizes, msg)

        input_sizes = list(range(1, K*2))
        msg = f'{dtype}: input < K'
        run_test(rt, input_sizes, msg)


if __name__ == "__main__":
    main()
