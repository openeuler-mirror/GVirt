#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Perf bench for the embed NPU kernel, meant to be run under msprof:

    msprof --delay=8 --duration=15 python tests/perf/embed.py -T 10

All inputs are built up front, then the script sleeps past the msprof delay so
only embed kernel launches land inside the capture window. Each case fires
`-N` launches; per-launch rows of the exported op_summary CSV (column
`Task Duration(us)`) map back onto the case plan by launch order.

Cases cover the real serving shapes: decode batches (1-64 tokens) and prefill
batches (512-8192 tokens) over the common hidden sizes. Vocab is sized to a
power of two so the weight fits comfortably next to other device users.
"""

import argparse
import time

import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, embed

SCRIPT_START_TIME = time.time()

VOCAB = 65536
DIMS = [2048, 4096, 5120, 7168]
BATCHES = [1, 8, 64, 512, 2048, 4096, 8192]

DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-T",
        "--time",
        type=int,
        default=10,
        help="Seconds to wait before firing the kernel calls; should match msprof "
        "--delay (a 0.5s margin is added on top). Default: 10",
    )
    parser.add_argument("-D", "--dtype", type=str, default="bf16", help="Kernel dtype: bf16 (default) or fp16")
    parser.add_argument(
        "-B", "--batches", type=int, nargs="*", default=None, help=f"Token counts to exercise; default: {BATCHES}"
    )
    parser.add_argument(
        "-H", "--dims", type=int, nargs="*", default=None, help=f"Hidden sizes to exercise; default: {DIMS}"
    )
    parser.add_argument("-N", "--num-duplicates", type=int, default=3, help="Number of launches per case (default: 3)")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype_key = args.dtype.lower()
    if dtype_key not in DTYPES:
        raise SystemExit(f"[FATAL] Unsupported dtype: {args.dtype} (expected one of {sorted(set(DTYPES))})")
    test_dtype = DTYPES[dtype_key]
    batches = args.batches if args.batches else BATCHES
    dims = args.dims if args.dims else DIMS

    rt = Runtime(0, size=500)
    torch.npu.set_device(0)

    cases = []
    for dim in dims:
        with torch.device("npu"):
            weight = torch.randn(VOCAB, dim, dtype=test_dtype)
        for batch in batches:
            with torch.device("npu"):
                ids = torch.randint(0, VOCAB, (batch,), dtype=torch.int32)
                out = torch.empty(batch, dim, dtype=test_dtype)
            cases.append((weight, ids, out))

    time_delta = args.time + 0.5 - (time.time() - SCRIPT_START_TIME)
    if time_delta > 0:
        time.sleep(time_delta)

    print(f"[INFO] Firing {len(cases)} embed kernel calls x{args.num_duplicates}...")
    t0 = time.time_ns()
    for weight, ids, out in cases:
        for _ in range(args.num_duplicates):
            embed(rt, weight, ids, out, 0, VOCAB)
    torch.npu.synchronize()
    t1 = time.time_ns()
    print(f"[INFO] Done. Total time: {(t1 - t0) / 1000:.3f} us")


if __name__ == "__main__":
    main()
