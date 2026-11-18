"""Minimal xlite kernel smoke test for the CANN simulator (no NPU).

Runs a single `muls` call per dtype on the camodel device and compares the
result with a CPU reference. Keep it small: the simulator is orders of
magnitude slower than real hardware.
"""

from __future__ import annotations

import sys

import torch
import torch_npu  # noqa: F401  (registers the `torch.npu` namespace)

from xlite._C import Runtime, muls


def main() -> int:
    rt = Runtime(0, size=500)
    torch.npu.set_device(0)
    torch.manual_seed(0)

    scale = 2.5
    all_ok = True
    for dtype in (torch.float16, torch.bfloat16):
        x_cpu = torch.randn(8, 2048, dtype=dtype)
        x = x_cpu.to("npu:0")
        y = torch.empty(8, 2048, dtype=dtype, device="npu:0")

        torch.npu.synchronize()
        muls(rt, x, scale, y)
        torch.npu.synchronize()

        expected = (x_cpu.float() * scale).to(dtype)
        ok = torch.allclose(expected.float(), y.cpu().float(), atol=1e-2, rtol=1e-2)
        print(f"muls {dtype}: {'ok' if ok else 'FAIL'}", flush=True)
        all_ok = all_ok and ok

    print("smoke test:", "PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
