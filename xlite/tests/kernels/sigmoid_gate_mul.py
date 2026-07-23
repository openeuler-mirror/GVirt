#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
from xlite._C import Runtime, sigmoid_gate_mul


def torch_sigmoid_gate_mul(attn, gate):
    return attn * torch.sigmoid(gate.float()).to(attn.dtype)


def run_test(rt, num_tokens, dim, dtype, inplace, msg):
    attn = torch.randn(num_tokens, dim, dtype=dtype)
    gate = torch.randn(num_tokens, dim, dtype=dtype)
    standard = torch_sigmoid_gate_mul(attn, gate)

    if inplace:
        out = attn.clone()
        torch.npu.synchronize()
        sigmoid_gate_mul(rt, out, gate, out)
        torch.npu.synchronize()
    else:
        out = torch.empty(num_tokens, dim, dtype=dtype)
        torch.npu.synchronize()
        sigmoid_gate_mul(rt, attn, gate, out)
        torch.npu.synchronize()

    try:
        torch.testing.assert_close(standard, out, rtol=1e-2, atol=1e-3)
        print(f"{msg}: PASS")
    except AssertionError as e:
        print(f"{msg}: FAILED")
        print(f"{e}")
        print(f"attn: {attn}")
        print(f"gate: {gate}")
        print(f"torch_npu: {standard}")
        print(f"xlite: {out}")
        raise


if __name__ == "__main__":
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")

    # Cover: small/large token counts, tile boundary (dim>2048), Qwen-like q_dim=2048,
    # and in-place path used by model ForwardAttn.
    shapes = [
        (1, 128),
        (8, 512),
        (41, 2048),
        (16, 4096),
        (64, 128),
    ]

    for dtype in [torch.bfloat16, torch.float16]:
        for num_tokens, dim in shapes:
            for inplace in [False, True]:
                mode = "inplace" if inplace else "out"
                msg = f"[{dtype}/T={num_tokens}/D={dim}/{mode}]"
                run_test(rt, num_tokens, dim, dtype, inplace, msg)
