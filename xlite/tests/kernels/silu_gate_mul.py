# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Keep GDN norm in FP32 through the output gate; reuse the MHA gate kernel."""
import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, silu_gate_mul, sigmoid_gate_mul

torch.npu.set_device(0)
torch.manual_seed(42)
rt = Runtime(0, 64)
for dtype in (torch.bfloat16, torch.float16):
    for shape in ((1, 1), (1, 128), (17, 6144), (129, 64), (2, 2080)):
        normed = torch.randn(shape, device='npu', dtype=torch.float32)
        gate = torch.randn(shape, device='npu', dtype=dtype)
        out = torch.empty_like(gate)
        expected = (normed * torch.nn.functional.silu(gate.float())).to(dtype)
        torch.npu.synchronize()
        silu_gate_mul(rt, normed, gate, out)
        torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=.004, atol=.004)
        # Existing MHA public entry keeps its sigmoid semantics.
        attn = normed.to(dtype)
        for raw_gate in (gate, gate[:, :1].contiguous()):
            expected_mha = (attn * torch.sigmoid(raw_gate.float()).to(dtype))
            torch.npu.synchronize()
            sigmoid_gate_mul(rt, attn, raw_gate, out)
            try:
                torch.testing.assert_close(out.cpu(), expected_mha.cpu(), rtol=.004, atol=.004)
            except AssertionError:
                print(f'FAILED MHA gate {dtype} shape={shape} gate={tuple(raw_gate.shape)}', flush=True)
                raise
        print(f'PASS GDN SiLU FP32 norm + MHA sigmoid: {dtype} {shape}', flush=True)
print('PASS 10 SiLU-gate and 20 MHA sigmoid regression cases', flush=True)
from xlite._C import rmsnorm
for dtype in (torch.bfloat16, torch.float16):
    for scale in (1.0, .001):
        x = (torch.randn(17, 48 * 128, device='npu') * scale).to(dtype)
        weight = torch.randn(128, device='npu', dtype=dtype)
        gate = torch.randn_like(x)
        normed = torch.empty_like(x, dtype=torch.float32)
        out = torch.empty_like(x)
        xf = x.float().reshape(17, 48, 128)
        expected = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6) * weight.float())
        expected = (expected.flatten(1) * torch.nn.functional.silu(gate.float())).to(dtype)
        torch.npu.synchronize()
        rmsnorm(rt, x, weight, normed, 1e-6, 128, 48)
        torch.npu.synchronize()
        silu_gate_mul(rt, normed, gate, out)
        torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=.005, atol=.005)
        print(f'PASS end-to-end GDN norm/gate {dtype} scale={scale}', flush=True)
