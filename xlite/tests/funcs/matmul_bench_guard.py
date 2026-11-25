#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Value-guard regression test for the matmul_bench op (csrc/_C.cpp MatmulBench).

MatmulBench returns `res / iterations` after its timing loop. Before the
argument guard this had two failure modes, both demonstrated on the CANN
simulator:

  - iterations=0: integer division by zero raises SIGFPE and kills the whole
    process (shell status 136) — not a Python exception.
  - iterations=-1: both loops are skipped and the function silently returns a
    negative "average time", which a benchmark harness happily records.

The guard rejects iterations < 1 and warmup_iterations < 0 at the top of
MatmulBench, before any tensor or device work, so every negative probe below
must fail with a Python error naming the offending parameter and must not
launch anything on the device (torch surfaces the C++ std::invalid_argument
as ValueError; RuntimeError is accepted too, see the other negative probes).
Until the guard lands, the first probe kills this test process with SIGFPE —
a loud failure, not a silent one.

Run on an NPU machine:

    python tests/funcs/matmul_bench_guard.py [--device N]

or on the CANN simulator (no NPU required):

    tests/run_sim.sh run tests/funcs/matmul_bench_guard.py
"""

import argparse

import torch
import torch_npu  # noqa: F401  (registers the torch.npu namespace)

from xlite._C import Runtime, matmul_bench


def main(device: int) -> int:
    torch.npu.set_device(device)
    rt = Runtime(device, size=500)

    # Generate on the CPU and move to the device (simulator-compatible
    # pattern, see tests/sim_smoke.py). Smallest shape supported by the
    # matmul kernel per tests/kernels/matmul.py.
    x = torch.randn(1, 32, dtype=torch.float16).to(f"npu:{device}")
    y = torch.randn(32, 32, dtype=torch.float16).to(f"npu:{device}")
    z = torch.empty(1, 32, dtype=torch.float16, device=f"npu:{device}")

    # Negative probes: invalid iteration counts must be rejected with a
    # Python error naming the parameter (torch maps std::invalid_argument to
    # ValueError; RuntimeError is accepted too for robustness).
    for iterations in (0, -1):
        try:
            matmul_bench(rt, x, y, z, x, y, z, iterations, 0)
        except (RuntimeError, ValueError) as e:
            assert "iterations" in str(e), f"unexpected error: {e}"
            print(f"PASS iterations={iterations} rejected: {str(e)[:90]}", flush=True)
        else:
            raise AssertionError(f"iterations={iterations} was accepted")

    try:
        matmul_bench(rt, x, y, z, x, y, z, 1, -1)
    except (RuntimeError, ValueError) as e:
        assert "warmup" in str(e), f"unexpected error: {e}"
        print(f"PASS warmup_iterations=-1 rejected: {str(e)[:90]}", flush=True)
    else:
        raise AssertionError("warmup_iterations=-1 was accepted")

    # Positive smoke: one measured launch on the smallest supported shape —
    # the guard must not disturb the normal path.
    avg = matmul_bench(rt, x, y, z, x, y, z, 1, 0)
    assert isinstance(avg, int) and avg >= 0, f"bad average: {avg}"
    print(f"PASS iterations=1 measured {avg} ns/matmul", flush=True)

    print("PASS matmul_bench argument guard")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    raise SystemExit(main(parser.parse_args().device))
