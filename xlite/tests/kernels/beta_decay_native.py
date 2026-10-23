# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Qwen3.5 FP32 A_log/dt_bias and FP32 decay-gate regression."""
import torch
import torch_npu  # noqa: F401
from xlite._C import Runtime, beta_decay

torch.npu.set_device(0)
torch.manual_seed(31)
rt = Runtime(0, 64)
for dtype in (torch.bfloat16, torch.float16, torch.float32):
    for bias_dtype in dict.fromkeys((dtype, torch.float32)):
        b, a = (torch.randn(67, 48, device='npu', dtype=dtype) for _ in range(2))
        log = torch.randn(48, device='npu', dtype=torch.float32) * .5
        bias = torch.randn(48, device='npu', dtype=bias_dtype)
        beta, gate = torch.empty_like(b), torch.empty_like(a, dtype=torch.float32)
        expected_beta = torch.sigmoid(b.float()).to(dtype)
        expected_gate = -log.exp()[None] * torch.nn.functional.softplus(a.float() + bias.float()[None])
        torch.npu.synchronize()
        beta_decay(rt, b, a, log, bias, beta, gate, 67, 1, 48)
        torch.testing.assert_close(beta.cpu(), expected_beta.cpu(), rtol=.004, atol=.002)
        torch.testing.assert_close(gate.cpu(), expected_gate.cpu(), rtol=1e-4, atol=1e-5)
        print(f'PASS native gating: activation={dtype} A_log=FP32 dt_bias={bias_dtype} g=FP32', flush=True)
