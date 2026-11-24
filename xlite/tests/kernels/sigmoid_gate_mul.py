#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, sigmoid_gate_mul


def torch_sigmoid_gate_mul(attn, gate):
    return attn * torch.sigmoid(gate.float()).to(attn.dtype)


def torch_sigmoid_gate_mul_add(attn, gate, add):
    return (attn.float() * torch.sigmoid(gate.float()) + add.float()).to(attn.dtype)


def run_test(rt, num_tokens, dim, dtype, inplace, msg, broadcast=False, fused_add=False):
    attn = torch.randn(num_tokens, dim, dtype=dtype)
    gate = torch.randn(num_tokens, 1 if broadcast else dim, dtype=dtype)
    add = torch.randn(num_tokens, dim, dtype=dtype) if fused_add else None
    if fused_add:
        standard = torch_sigmoid_gate_mul_add(attn, gate, add)
    else:
        standard = torch_sigmoid_gate_mul(attn, gate)

    if inplace:
        out = attn.clone()
        torch.npu.synchronize()
        sigmoid_gate_mul(rt, out, gate, out, add)
        torch.npu.synchronize()
    else:
        out = torch.empty(num_tokens, dim, dtype=dtype)
        torch.npu.synchronize()
        sigmoid_gate_mul(rt, attn, gate, out, add)
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
    rt = Runtime(0, size=500)
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
        # Non-calcPad-aligned dims exercise the tail chunk: dim=2000 leaves a
        # 16-element tail (32B, whole-block tail copy); dim=1992 leaves an
        # 8-element tail (16B, b16 byte-exact tail copy in CopyGmToUbufAligned/
        # CopyUbufToGmAligned). Covers the fused-add tail path too (add uses
        # the same transfer shape as attn).
        (3, 2000),
        (3, 1992),
    ]

    for dtype in [torch.bfloat16, torch.float16]:
        for num_tokens, dim in shapes:
            for inplace in [False, True]:
                mode = "inplace" if inplace else "out"
                msg = f"[{dtype}/T={num_tokens}/D={dim}/{mode}]"
                run_test(rt, num_tokens, dim, dtype, inplace, msg)
                bmsg = f"[{dtype}/T={num_tokens}/D={dim}/{mode}/bcast]"
                run_test(rt, num_tokens, dim, dtype, inplace, bmsg, broadcast=True)
                # Fused addend (MoE shared-expert tail): broadcast + elementwise gate.
                fmsg = f"[{dtype}/T={num_tokens}/D={dim}/{mode}/fusedadd]"
                run_test(rt, num_tokens, dim, dtype, inplace, fmsg, fused_add=True)
                fbmsg = f"[{dtype}/T={num_tokens}/D={dim}/{mode}/fusedadd/bcast]"
                run_test(rt, num_tokens, dim, dtype, inplace, fbmsg, broadcast=True, fused_add=True)
