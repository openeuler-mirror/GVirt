#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Perf bench for the gather_sparse_kv_cache NPU kernel, meant to be run under msprof:

    msprof --delay=8 --duration=5 python tests/perf/gather_sparse_kv_cache.py -T 10

All inputs are built up front, then the script sleeps past the msprof delay so
only gather kernel launches land inside the capture window (no checks, syncs,
or prints). Decode-only; --order random measures the degenerate token-by-token
case, -B confines the burst copy-in size (512 => token-by-token).
"""

import argparse
import time

import numpy as np
import torch

from xlite._C import Runtime, gather_sparse_kv_cache

SCRIPT_START_TIME = time.time()

BLOCK_SIZE = 128
MAX_TOPK_NUM = 2048

# model configs: name, kv_lora_rank, rope_head_dim
MODELS = {
    "glm5": (512, 64),
}

# default work configurations: (batch, cached_lens, query_lens); decode => the
# default query_len=1 per batch. cached_lens[b] < topK rows exercise the
# dense-template gather path. Kept small so all cases fit in device memory at
# once (built before the delay).
DEFAULT_WORK = [
    (1, [16384], [1]),
    (3, [32751] * 3, [1] * 3),
    (3, [2047, 8191, 45841], [1, 1, 1]),  # mixed: 2047 < topK=2048 => row 0 dense (template)
    (4, [2047] * 4, [1] * 4),  # all rows dense (template) for topK=2048
    (1, [131071], [1]),
    (1, [127], [1]),  # dense (dummy indices must not be read)
    (1, [4096], [1]),
    (2, [2048, 2049], [1, 1]),  # boundary: 2048 >= 2048 processed, 2049 processed
]
DEFAULT_TOPK = [2048]

# the op is BF16-only (XliteOpGatherSparseKVCache dtype check)
DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
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
        help="Kernel input dtype: bf16 (default, the only dtype the op supports)",
    )
    parser.add_argument(
        "-M",
        "--model",
        type=str,
        default="glm5",
        help=f"Model shape preset (kv_lora_rank, rope_head_dim): {sorted(MODELS)}. Default: glm5",
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
        help="Cached KV length(s) of the decode batch (query_len is implicitly 1 per sequence); "
        "one batch is formed from all entries. When omitted, the built-in default work set is used",
    )
    parser.add_argument(
        "-N",
        "--num-duplicates",
        type=int,
        default=1,
        help="Number of times to duplicate the work set (default: 1; l2 cache hits affect multiple runs)",
    )
    parser.add_argument(
        "-O",
        "--order",
        type=str,
        choices=("asc", "desc", "random"),
        default="asc",
        help="Order of the generated topk_indices: ascending (the contract indexer_topk provides), "
        "descending (legacy), or random (degenerates the burst copy-in to token-by-token). "
        "Default: asc",
    )
    parser.add_argument(
        "-B",
        "--burst-size",
        type=int,
        default=16 * 1024,
        help="Burst copy-in size in bytes (tiling parameter), confined to the 8KB tiling optimum "
        "(GM_UB_BURST_SIZE_TILING); larger values behave like 8192. Default: 16KB.",
    )
    args = parser.parse_args()

    dtype_key = args.dtype.lower()
    if dtype_key not in DTYPES:
        raise SystemExit(f"[FATAL] Unsupported dtype: {args.dtype} (expected one of {sorted(set(DTYPES))})")
    args.dtype = DTYPES[dtype_key]

    if args.model not in MODELS:
        raise SystemExit(f"[FATAL] Unknown model preset: {args.model} (expected one of {sorted(MODELS)})")

    if args.topk is None:
        args.topk = DEFAULT_TOPK
    for k in args.topk:
        if k <= 0 or k > MAX_TOPK_NUM:
            raise SystemExit(f"[FATAL] topK must be in [1, {MAX_TOPK_NUM}], got: {k}")

    if args.cached_lens is not None:
        if any(cl <= 0 for cl in args.cached_lens):
            raise SystemExit("[FATAL] cached lens must be > 0")
        batch = len(args.cached_lens)
        args.work = [(batch, list(args.cached_lens), [1] * batch)]
    else:
        args.work = DEFAULT_WORK
    return args


def max_blocks(cached_lens, block_size):
    max_sum = max(cached_lens) + 1  # decode adds 1 query token
    return (max_sum + block_size - 1) // block_size


def gen_topk_indices(valid_len, topK, clustered=True, order="asc"):
    """Sample topK distinct token ids from [0, valid_len) in the requested order
    (asc/desc/random). clustered=True biases toward consecutive runs to exercise
    the burst copy-in; dense rows (valid_len <= topK) get dummy ids the kernel
    must not read."""
    if valid_len <= topK:
        # dummy indices for dense-attention rows (kernel skips these)
        return torch.randint(0, valid_len, (topK,), dtype=torch.int32)
    if not clustered:
        ids = torch.randperm(valid_len)[:topK].tolist()
    else:
        # seed with ~topK/2 random ids, extend each into a short consecutive
        # run (clamped to valid_len), dedup, then top up randomly.
        chosen = set()
        seeds = torch.randperm(valid_len)[: topK // 2 + 1].tolist()
        for t in seeds:
            run = int(torch.randint(1, 9, (1,)).item())
            for u in range(t, min(t + run, valid_len)):
                chosen.add(u)
                if len(chosen) >= topK:
                    break
            if len(chosen) >= topK:
                break
        while len(chosen) < topK:
            chosen.add(int(torch.randint(0, valid_len, (1,)).item()))
        ids = list(chosen)[:topK]
    ids = torch.tensor(ids, dtype=torch.int32)
    if order == "asc":
        ids, _ = torch.sort(ids)
    elif order == "desc":
        ids, _ = torch.sort(ids, descending=True)
    else:
        ids = ids[torch.randperm(len(ids))]
    return ids


def build_case(
    test_dtype,
    kv_lora_rank,
    rope_head_dim,
    batch,
    cached_lens_list,
    query_lens_list,
    topK,
    order="asc",
    burst_size=16 * 1024,
):
    max_num_blocks = max_blocks(cached_lens_list, BLOCK_SIZE)
    kv_heads = 1

    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        kvcache_block_num = max_num_blocks * batch
        k_cache = torch.randn(kvcache_block_num, BLOCK_SIZE, kv_heads, kv_lora_rank)
        pe_cache = torch.randn(kvcache_block_num, BLOCK_SIZE, kv_heads, rope_head_dim)

        # block_tables: batch x max_num_blocks, contiguous layout
        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(max_num_blocks, dtype=np.uint32)
        block_tables_array = batch_indices * max_num_blocks + block_indices
        block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32)

        # topk_indices: [batch, topK] token-index semantics (decode => 1
        # query/batch), ordered per `order` with short consecutive runs
        topk_indices_list = []
        for i in range(batch):
            total_len = cached_lens_list[i] + 1  # decode query token at abs pos clen
            topk_indices_list.append(gen_topk_indices(total_len, topK, clustered=True, order=order))
        topk_indices = torch.stack(topk_indices_list).to(dtype=torch.int32)

        k_dense = torch.zeros(batch, topK, kv_heads, kv_lora_rank, dtype=test_dtype)
        pe_dense = torch.zeros(batch, topK, kv_heads, rope_head_dim, dtype=test_dtype)
        cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
        query_lens = torch.tensor(query_lens_list, dtype=torch.int32).flatten()
        # query_start_loc: exclusive prefix sum (cumsum - lens), as in mla_v2;
        # decode (all-1) => [0, 1, ..., batch-1]
        query_lens_np = np.array(query_lens_list)
        query_start_loc = torch.tensor(
            (np.cumsum(query_lens_np) - query_lens_np).tolist(), dtype=torch.int32
        ).flatten()

    return (
        k_cache,
        pe_cache,
        block_tables,
        topk_indices,
        query_start_loc,
        query_lens,
        cached_lens,
        k_dense,
        pe_dense,
        batch,
        topK,
        BLOCK_SIZE,
        kv_lora_rank,
        rope_head_dim,
        1,  # kv_heads
        1,  # compress_ratio
        burst_size,
    )


def main(
    wait_time,
    test_dtype,
    kv_lora_rank,
    rope_head_dim,
    work,
    topk_values,
    num_duplicates,
    order,
    burst_size=16 * 1024,
):
    cases = []
    for batch, cached_lens_list, query_lens_list in work:
        for topK in topk_values:
            cases.append(
                build_case(
                    test_dtype,
                    kv_lora_rank,
                    rope_head_dim,
                    batch,
                    cached_lens_list,
                    query_lens_list,
                    topK,
                    order,
                    burst_size,
                )
            )

    time_delta = wait_time + 0.5 - (time.time() - SCRIPT_START_TIME)
    if time_delta > 0:
        time.sleep(time_delta)

    print(f"[INFO] Firing {len(cases)} gather_sparse_kv_cache kernel calls...")
    t0 = time.time_ns()
    for case in cases:
        for _ in range(num_duplicates):
            gather_sparse_kv_cache(rt, *case)
    torch.npu.synchronize()
    t1 = time.time_ns()
    print(f"[INFO] Done. Total time: {(t1 - t0) / 1000:.3f} us")


if __name__ == "__main__":
    args = parse_args()
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    kv_lora_rank, rope_head_dim = MODELS[args.model]
    main(
        wait_time=args.time,
        test_dtype=args.dtype,
        kv_lora_rank=kv_lora_rank,
        rope_head_dim=rope_head_dim,
        work=args.work,
        topk_values=args.topk,
        num_duplicates=args.num_duplicates,
        order=args.order,
        burst_size=args.burst_size,
    )
