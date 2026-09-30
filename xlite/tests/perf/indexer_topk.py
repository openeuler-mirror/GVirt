#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Perf bench for the indexer_topk NPU kernel, meant to be run under msprof:

    msprof --delay=8 --duration=5 python tests/perf/indexer_topk.py -T 10

All inputs are built up front (random data, no accuracy reference), then the
script sleeps past the msprof delay so only indexer_topk kernel launches land
inside the capture window. No checks, no syncs, no prints.

Examples:

    # defaults (bf16, topK 512/2048, default work set)
    msprof --delay=8 --duration=5 python tests/perf/indexer_topk.py

    # custom shapes: 3 decode batches with 8k/16k/32k cached tokens, topK 2048
    python tests/perf/indexer_topk.py -T 10 -K 2048 -CL 8192 16384 32768 -QL 1 1 1

    # prefill-only, fp16
    python tests/perf/indexer_topk.py -D fp16 -K 512 -CL 0 0 -QL 4096 8192
"""

import argparse
import time

import numpy as np
import torch

from xlite._C import Runtime, indexer_topk

SCRIPT_START_TIME = time.time()

BLOCK_SIZE = 128
N_HEADS = 64
HEAD_DIM = 128
MAX_TOPK_NUM = 2048

# Debug strings are in. Running the failing repro (cached=4096, ql=12, topK=2048, nH=32):
# default work configurations: (batch, cached_lens, query_lens); kept small so
# all cases fit in device memory at once (built up front, before the delay)
DEFAULT_WORK = [
    (1, [0], [3142]),  # prefill
    (1, [0], [4096]),  # prefill
    (1, [0], [6348]),  # prefill
    (1, [0], [8192]),
    (1, [0], [12485]),
    (1, [0], [16384]),
    (1, [4096], [3183]),
    (1, [8192], [3183]),
    (1, [16384], [3183]),
    (1, [3186], [1]),
    (1, [6848], [1]),
    (1, [8265], [1]),
    (1, [12983], [1]),
    (1, [16384], [1]),  # decode, long cache
    (1, [16384, 16384, 16384], [1, 1, 1]),
    (3, [8192, 12243, 3186], [1, 1, 1]),
    (2, [8123, 0], [1, 3987]),  # mixed decode + prefill
    (2, [8123, 0], [1, 7196]),  # mixed decode + prefill
]
DEFAULT_TOPK = [2048]

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
        help="Time in seconds to wait for before firing the kernel calls; should match msprof --delay "
        "(a 0.5s margin is added on top). Default: 10",
    )
    parser.add_argument(
        "-D",
        "--dtype",
        type=str,
        default="bf16",
        help="Kernel input dtype: bf16 (default) or fp16",
    )
    parser.add_argument(
        "-K",
        "--topk",
        type=int,
        nargs="*",
        default=None,
        help=f"topK value(s) to exercise, each <= {MAX_TOPK_NUM}; default: {DEFAULT_TOPK}",
    )
    parser.add_argument(
        "-CL",
        "--cached-lens",
        type=int,
        nargs="*",
        default=None,
        help="Cached KV length(s); zipped pairwise with --query-lens to form the work set. "
        "When specified, --query-lens must be given with the same number of entries; "
        "when omitted, the built-in default work set is used",
    )
    parser.add_argument(
        "-QL",
        "--query-lens",
        type=int,
        nargs="*",
        default=None,
        help="Query length(s); zipped pairwise with --cached-lens (see above)",
    )
    parser.add_argument(
        "-N", "--num-duplicates", type=int, default=3, help="Number of times to duplicate the work set (default: 1)"
    )
    args = parser.parse_args()

    dtype_key = args.dtype.lower()
    if dtype_key not in DTYPES:
        raise SystemExit(f"[FATAL] Unsupported dtype: {args.dtype} (expected one of {sorted(set(DTYPES))})")
    args.dtype = DTYPES[dtype_key]

    if args.topk is None:
        args.topk = DEFAULT_TOPK
    for k in args.topk:
        if k <= 0 or k > MAX_TOPK_NUM:
            raise SystemExit(f"[FATAL] topK must be in [1, {MAX_TOPK_NUM}], got: {k}")

    if (args.cached_lens is None) != (args.query_lens is None):
        raise SystemExit("[FATAL] --cached-lens and --query-lens must be specified together")
    if args.cached_lens is not None:
        if len(args.cached_lens) != len(args.query_lens):
            raise SystemExit(
                f"[FATAL] --cached-lens ({len(args.cached_lens)} entries) and "
                f"--query-lens ({len(args.query_lens)} entries) must have the same length"
            )
        if any(cl < 0 for cl in args.cached_lens) or any(ql <= 0 for ql in args.query_lens):
            raise SystemExit("[FATAL] cached lens must be >= 0 and query lens must be > 0")
        args.work = [(len(args.cached_lens), list(args.cached_lens), list(args.query_lens))]
    else:
        args.work = DEFAULT_WORK
    return args


def max_blocks(query_lens, cached_lens, block_size):
    max_sum = max(a + b for a, b in zip(query_lens, cached_lens))
    return (max_sum + block_size - 1) // block_size


def build_case(test_dtype, batch, cached_lens_list, query_len_list, topK):
    max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE)
    max_seq_len = max_num_blocks * BLOCK_SIZE
    total_query_len = sum(query_len_list)

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        q = torch.randn(total_query_len, N_HEADS * HEAD_DIM)
        weight = torch.randn(total_query_len, HEAD_DIM + N_HEADS)
        k_cache = torch.randn(max_num_blocks * batch, BLOCK_SIZE, HEAD_DIM)

        query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_lens_np = np.array(query_len_list)
        query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
        query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(max_num_blocks, dtype=np.uint32)
        block_tables_array = batch_indices * max_num_blocks + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

        indices = torch.arange(max_seq_len, dtype=torch.int32).npu()
        topk_indices = torch.empty(total_query_len, topK, dtype=torch.int32).npu()

    return (
        q,
        k_cache,
        weight,
        indices,
        topk_indices,
        query_start_loc,
        query_lens,
        cached_lens,
        block_tables,
        N_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        batch,
        topK,
    )


def main(wait_time, test_dtype, work, topk_values):
    cases = []
    for batch, cached_lens_list, query_len_list in work:
        for topK in topk_values:
            cases.append(build_case(test_dtype, batch, cached_lens_list, query_len_list, topK))

    time_delta = wait_time + 0.5 - (time.time() - SCRIPT_START_TIME)
    if time_delta > 0:
        time.sleep(time_delta)

    print(f"[INFO] Firing {len(cases)} indexer_topk kernel calls...")
    t0 = time.time_ns()
    for case in cases:
        for _ in range(args.num_duplicates):
            indexer_topk(rt, *case)
    torch.npu.synchronize()
    t1 = time.time_ns()
    print(f"[INFO] Done. Total time: {(t1 - t0) / 1000:.3f} us")


if __name__ == "__main__":
    args = parse_args()
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    main(
        wait_time=args.time,
        test_dtype=args.dtype,
        work=args.work,
        topk_values=args.topk,
    )
