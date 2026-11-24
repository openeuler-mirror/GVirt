#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Micro-benchmark: MoE shared-expert tail, split vs fused.

Compares the MoE shared-expert epilogue cost per layer:
  baseline: sigmoid_gate_mul(attn, gate, out) + add(out, h, out)   (2 launches)
  fused:    sigmoid_gate_mul(attn, gate, out, add=h)               (1 launch)

Both python wrappers synchronize per call, so wall time includes an identical
per-call sync; the delta isolates the extra launch + kernel + GM round trip.
Gate is broadcast [m, 1] (se_gate is Linear(H->1)), matching the model path.

Usage:
    python tests/performance/perf_sigmoid_gate_mul.py [--iters N]
"""

import argparse
import statistics
import time

import torch
import torch_npu  # noqa: F401  (torch.npu requires the torch_npu plugin import)

from xlite._C import Runtime, add, sigmoid_gate_mul


def bench(rt, m, dim, dtype, iters, warmup=20):
    attn = torch.randn(m, dim, dtype=dtype)
    gate = torch.randn(m, 1, dtype=dtype)
    h = torch.randn(m, dim, dtype=dtype)
    out = torch.empty(m, dim, dtype=dtype)
    torch.npu.synchronize()

    def run_split():
        sigmoid_gate_mul(rt, attn, gate, out)
        add(rt, out, h, out)

    def run_fused():
        sigmoid_gate_mul(rt, attn, gate, out, h)

    for _ in range(warmup):
        run_split()
        run_fused()

    results = {}
    for name, fn in [("split", run_split), ("fused", run_fused)]:
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
        results[name] = statistics.median(samples)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    rt = Runtime(0, size=500)
    torch.npu.set_device(0)
    torch.set_default_device("npu:0")

    # decode-like (small m) and prefill-like (large m) token counts;
    # dims cover common hidden sizes incl. tile-split (dim > 2048).
    shapes = [(1, 4096), (8, 4096), (8, 5120), (16, 7168), (64, 4096), (256, 4096)]

    print(f"{'dtype':<10}{'m':>5}{'dim':>7}{'split(us)':>12}{'fused(us)':>12}{'saved':>9}")
    for dtype in [torch.bfloat16, torch.float16]:
        for m, dim in shapes:
            r = bench(rt, m, dim, dtype, args.iters)
            saved = 100.0 * (r["split"] - r["fused"]) / r["split"]
            print(f"{str(dtype):<10}{m:>5}{dim:>7}{r['split'] * 1e3:>12.1f}{r['fused'] * 1e3:>12.1f}{saved:>8.1f}%")


if __name__ == "__main__":
    main()
