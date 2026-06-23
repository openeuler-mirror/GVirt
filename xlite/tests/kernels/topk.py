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

K = 2048
max_seq_len = 131072

# work configurations: batch_size, cached_lens, query_lens
work = [
    (1, [0], [1]),
    (1, [0], [30]),
    (1, [0], [77]),
    (1, [0], [128]),
    (1, [0], [256]),
    (1, [0], [1600]),
    (1, [0], [2528]),
    (1, [100], [30]),
    (8, [0] * 8, [789, 65, 13, 6545, 24, 190, 2432, 124]),
    (2, [4000] * 2, [1] * 2),
    (2, [6000] * 2, [1] * 2),
    (2, [131071] * 2, [1] * 2),
    (5, [8, 13, 65, 11, 5], [1] * 5),
    (1, [128], [128]),
]


def build_xlite_input(scores):
    res = torch.zeros(len(scores), max_seq_len)
    for i, s in enumerate(scores):
        res[i, :len(s)] = s
    return res


def generate_sequences(cached_lens, query_lens):
    standard_scores = []
    standard_indices = []

    for c, q in zip(cached_lens, query_lens):
        n_tokens = c + q
        for _ in range(q):
            t = torch.randn(n_tokens)
            standard_scores.append(t)
            if (n_tokens >= K):
                standard_indices.append(t.topk(K)[1])
            else:
                standard_indices.append(t.topk(n_tokens)[1])
                padding = torch.zeros(K - n_tokens, dtype=torch.int32)
                standard_indices[-1] = torch.cat([standard_indices[-1], padding])

    standard_indices = torch.stack(standard_indices)
    return standard_scores, standard_indices.to(dtype=torch.int32)


def run_test(rt, cached_lens, query_lens, msg):
    standard_scores, standard_indices = generate_sequences(cached_lens, query_lens)

    scores = build_xlite_input(standard_scores)
    indices = torch.arange(max_seq_len, dtype=torch.int32)
    xlite_indices = torch.empty(sum(query_lens), K, dtype=torch.int32)

    query_lens = torch.tensor(query_lens, dtype=torch.int32)
    cached_lens = torch.tensor(cached_lens, dtype=torch.int32)

    torch.npu.synchronize()
    xlite_topk(rt, scores, indices, xlite_indices, query_lens, cached_lens, K)
    torch.npu.synchronize()

    try:
        torch.testing.assert_close(standard_indices, xlite_indices)
        print(f"{msg}: PASS")
    except Exception as e:
        s = ""
        first_bad = -1
        for i, (std, xlite) in enumerate(zip(standard_indices.flatten(), xlite_indices.flatten())):
            if std != xlite and first_bad == -1:
                first_bad = i
            c = ' ' if std == xlite else 'x'
            s += f'[{c}]'
            if (i % 50 == 49):
                s += '\n'
        print(f"{msg}: FAILED: {first_bad}")
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
        for batch, cached_lens, query_lens in work:
            assert len(cached_lens) == batch, f"cached_lens length {len(cached_lens)} != batch {batch}"
            assert len(query_lens) == batch, f"query_lens length {len(query_lens)} != batch {batch}"
            msg = f'{dtype}: work(batch={batch}, cached_lens={cached_lens}, query_lens={query_lens})'
            run_test(rt, cached_lens, query_lens, msg)


if __name__ == "__main__":
    main()
