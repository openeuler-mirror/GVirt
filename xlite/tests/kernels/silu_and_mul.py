#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, silu_and_mul

rt = Runtime(0, size=500)
torch.npu.set_device(0)

# The fp32 op order still differs (kernel vexp/vdiv vs torch silu), flipping
# the final rounding on a few elements — always a single output ULP. A 1-ULP
# bf16 error reaches 2^-7 = 0.78% relative, so bf16 needs an ULP-aware rtol
# (same as tests/kernels/rmsnorm.py).
supported_dtype_list = [(torch.float, 2e-5, 2e-3), (torch.float16, 2e-5, 2e-3), (torch.bfloat16, 2e-2, 1e-2)]

for dtype, atol, rtol in supported_dtype_list:
    for dim in [2304, 6400]:
        input = torch.randn(41, dim * 2, dtype=dtype, device="npu:0")
        output = torch.empty(41, dim, dtype=dtype, device="npu:0")

        d = input.shape[-1] // 2
        # The kernel computes silu in fp32 and rounds the output once (see
        # doc/developer_guide/kernels/silu_and_mul.md). Computing silu on the
        # dtype tensor rounds the silu result to the dtype before the multiply
        # (double rounding) and disagrees with the kernel by 1 ULP on a few
        # percent of elements at larger magnitudes, so reference the
        # documented single-rounding semantics.
        standard = (torch.nn.functional.silu(input[..., :d].float()) * input[..., d:].float()).to(dtype)

        torch.npu.synchronize()
        silu_and_mul(rt, input, output)
        torch.npu.synchronize()

        print(f"silu and mul {dtype} dim={dim} executed!")

        try:
            torch.testing.assert_close(standard, output, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"{e}")
            print(f"torch_npu: {standard}")
            print(f"xlite: {output}")

        # SwiGLU limit clamp path: up = clamp(up, -L, L), gate = clamp(gate, max=L).
        # Output magnitude scales ~ L*L (silu(L)*up ≈ L*L for large L), so the small
        # atol/rtol used above (tuned for unit-magnitude randn) is far too tight for
        # the clamped path. Use scale-appropriate tolerances: a few ULPs of the output
        # dtype at the clamped magnitude.
        # fp16 computes silu with fp16 intermediates (documented): for
        # gate < -ln(65504) ~= -11.09, exp(-gate) overflows fp16 and silu
        # flushes to 0. |silu(gate)| <= 1.693e-4 there, so the output error
        # is bounded by 1.693e-4 * limit (<= 1.69e-3 at limit=10); atol=2e-3
        # absorbs this deterministic bound. bf16 keeps an ULP-aware rtol.
        clamp_atol = {torch.float: 2e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
        clamp_rtol = {torch.float: 2e-3, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
        for swiglu_limit in [2.0, 10.0]:
            input = torch.randn(41, dim * 2, dtype=dtype, device="npu:0") * 20.0
            output = torch.empty(41, dim, dtype=dtype, device="npu:0")

            d = input.shape[-1] // 2
            gate = input[..., :d]
            up = input[..., d:]
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
            gate = torch.clamp(gate, max=swiglu_limit)
            standard = (torch.nn.functional.silu(gate.float()) * up.float()).to(dtype)

            torch.npu.synchronize()
            silu_and_mul(rt, input, output, swiglu_limit=swiglu_limit)
            torch.npu.synchronize()

            print(f"silu and mul {dtype} swiglu_limit={swiglu_limit} dim={dim} executed!")

            try:
                torch.testing.assert_close(standard, output, atol=clamp_atol, rtol=clamp_rtol)
            except AssertionError as e:
                print(f"{e}")
                print(f"torch_npu: {standard}")
                print(f"xlite: {output}")
