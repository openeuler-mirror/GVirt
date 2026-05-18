#!/usr/bin/env/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import torch.nn.functional as F
from xlite._C import Runtime, beta_decay

num_v_heads = 16


def run_test(rt, batch, seq_len, msg):
    b = torch.randn(batch, seq_len, num_v_heads)
    a = torch.randn(batch, seq_len, num_v_heads)
    A_log = torch.randn(num_v_heads)
    dt_bias = torch.randn(num_v_heads)

    my_beta = torch.zeros(batch, seq_len, num_v_heads)
    my_g = torch.zeros(batch, seq_len, num_v_heads)
    trch_beta = torch.zeros(batch, seq_len, num_v_heads)
    trch_g = torch.zeros(batch, seq_len, num_v_heads)

    torch.npu.synchronize()

    beta_decay(rt, b, a, A_log, dt_bias, my_beta, my_g, batch, seq_len,
               num_v_heads)

    torch.npu.synchronize()

    trch_beta = torch.sigmoid(b.float())
    trch_beta = trch_beta.to(b.dtype)
    soft = F.softplus(a.float() + dt_bias.float().unsqueeze(0).unsqueeze(0))
    trch_g = -A_log.float().exp().unsqueeze(0).unsqueeze(0) * soft
    trch_g = trch_g.to(a.dtype)

    torch.npu.synchronize()

    try:
        torch.testing.assert_close(trch_beta, my_beta, rtol=1e-2, atol=1e-3)
        print(f"{msg}beta: PASS")
    except Exception as e:
        print(f"b: {b}")
        print(f"my_beta: {my_beta}")
        print(f"trch_beta: {trch_beta}")
        print(f"{msg}: FAILED")
        raise e

    try:
        torch.testing.assert_close(trch_g, my_g, rtol=1e-2, atol=1e-3)
        print(f"{msg}g: PASS")
    except Exception as e:
        print(f"a: {a}")
        print(f"A_log: {A_log}")
        print(f"dt_bias: {dt_bias}")
        print(f"my_g: {my_g}")
        print(f"trch_g: {trch_g}")
        print(f"{msg}: FAILED")
        raise e


if __name__ == "__main__":
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")
    torch.set_printoptions(threshold=torch.inf)

    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        torch.set_default_dtype(dtype)
        for batch in range(1, 9):
            for i in range(13):
                seq_len = 2**i
                msg = f'[{dtype}/{batch}/{seq_len}]'
                run_test(rt, batch, seq_len, msg)
