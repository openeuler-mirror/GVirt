#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
# Test for the concat kernel: pack a variable number of input tensors (any
# dtype) into one contiguous flat byte buffer, matching torch.cat(..., dim=-1)
# on a common contiguous layout. The kernel is byte-wise, so inputs of
# different shapes/dtypes may be packed as long as the output byte count
# equals the sum of the input byte counts.
#
# Two regimes are exercised:
#   (1) small sizes (correctness sweep over dtypes / n_inputs / odd byte tails)
#   (2) large sizes (multi-core stride path + minimax_m2 real workload scale)
# ===============================================================================

import logging
import os
import numpy as np
import torch
from xlite._C import Runtime, concat

logging.getLogger().setLevel(logging.INFO)

# Pick a device: honor XLITE_TEST_DEVICE env var, default 0. Use a free device
# (one with most spare memory) when running the large/minimax_m2-scale cases,
# since the pool must hold ~2x the largest totalBytes (~230 MB for m=32768).
DEVICE = int(os.environ.get("XLITE_TEST_DEVICE", "0"))
# 500 MB pool fits the minimax_m2-scale case (single concat needs ~2x totalBytes,
# ~460 MB at m=32768). Bump XLITE_TEST_POOL_MB if you raise the size caps.
POOL_MB = int(os.environ.get("XLITE_TEST_POOL_MB", "500"))
rt = Runtime(DEVICE, POOL_MB)
torch.npu.set_device(DEVICE)
torch.set_printoptions(threshold=torch.inf)

DTYPE_LIST = [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int8]


def elem_size(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def to_numpy(t):
    """cpu().numpy() safe for all dtypes incl. bfloat16."""
    return t.cpu().to(torch.float32).numpy()


def run_test(n_inputs, dtype, seg_elems, msg):
    """Pack n_inputs flat tensors (each seg_elems elements) into one flat output."""
    es = elem_size(dtype)
    inputs = [torch.randn(seg_elems, dtype=dtype, device=f"npu:{DEVICE}") for _ in range(n_inputs)]
    if dtype in (torch.int32, torch.int8):
        for x in inputs:
            x.random_(-100, 100)
    out = torch.empty(seg_elems * n_inputs, dtype=dtype, device=f"npu:{DEVICE}")

    torch.npu.synchronize()
    concat(rt, inputs, out)
    torch.npu.synchronize()

    got = to_numpy(out)
    ref = np.concatenate([to_numpy(x) for x in inputs], axis=0)
    if not np.array_equal(got, ref):
        raise AssertionError(f"{msg}: mismatch\n  got={got[:16]}\n  ref={ref[:16]}")
    total_kb = seg_elems * n_inputs * es / 1024
    logging.info(f"{msg}: PASS (n={n_inputs}, dtype={dtype}, "
                 f"seg_elems={seg_elems}, total={total_kb:.0f}KB)")


def run_test_mixed_dtypes(input_specs, msg):
    """Pack inputs of different dtypes/shapes by raw bytes.

    input_specs: list of (dtype, numel). The output is a flat INT8 buffer whose
    byte count equals the sum of input byte counts.
    """
    inputs = []
    total_bytes = 0
    for dtype, numel in input_specs:
        es = elem_size(dtype)
        x = torch.randn(numel, dtype=dtype, device=f"npu:{DEVICE}")
        if dtype in (torch.int32, torch.int8):
            x.random_(-100, 100)
        inputs.append(x)
        total_bytes += numel * es

    out = torch.empty(total_bytes, dtype=torch.int8, device=f"npu:{DEVICE}")

    torch.npu.synchronize()
    concat(rt, inputs, out)
    torch.npu.synchronize()

    # Compare raw bytes: re-interpret each input as uint8 (byte view), not via
    # float32 (which would reinterpret the wrong bytes).
    got = out.cpu().view(torch.uint8).numpy()
    ref = np.concatenate([x.cpu().view(torch.uint8).numpy() for x in inputs], axis=0)
    if not np.array_equal(got, ref):
        raise AssertionError(f"{msg}: mismatch")
    logging.info(f"{msg}: PASS (total={total_bytes / 1024 / 1024:.1f}MB)")


if __name__ == "__main__":
    cases = 0

    # ---- (1) small correctness sweep: dtypes / n_inputs / odd byte tails ----
    for dtype in DTYPE_LIST:
        for n_inputs in [1, 2, 3, 4, 5, 6, 7, 8]:
            for seg_elems in [1, 3, 7, 16, 100, 1024, 10000]:
                if seg_elems * n_inputs == 0:
                    continue
                msg = f"[small/{dtype}/{n_inputs}/{seg_elems}]"
                run_test(n_inputs, dtype, seg_elems, msg)
                cases += 1
    logging.info(f"concat small sweep: {cases} cases done")

    # ---- (2) large sizes: exercise multi-core stride path ----
    # totalBytes crosses block_num*dataBufSize (~tens of MB) so all cores engage.
    # Cap each case so inputs+out (~2x totalBytes) fit in the tensor pool.
    max_bytes = POOL_MB * 1024 * 1024 // 2
    for dtype in [torch.float16, torch.bfloat16]:
        es = elem_size(dtype)
        for n_inputs in [1, 2, 3, 8]:
            for seg_elems in [100_000, 1_000_000, 5_000_000]:
                total_bytes = seg_elems * n_inputs * es
                if total_bytes > max_bytes:
                    continue
                msg = f"[large/{dtype}/{n_inputs}/{seg_elems}]"
                run_test(n_inputs, dtype, seg_elems, msg)
                cases += 1
    logging.info(f"concat large sweep done, running minimax_m2-scale case")

    # ---- (3) minimax_m2 real workload: packedSend scale ----
    # hiddenSize=3072, n_routed_experts=256, bfloat16, m up to 32768 tokens.
    #   bytesInput    = m * 3072 * 2        (bfloat16 hidden state)
    #   bytesWeights  = m * 256 * 4         (gate weights, FP32)
    #   bytesRouting  = m * 256 / 8         (BIT1, ~ m*32 bytes)
    # totalBytes ~ m * 7200 bytes.
    for m in [8, 512, 4096, 32768]:
        spec = [
            (torch.bfloat16, m * 3072),  # input
            (torch.float32, m * 256),    # weights (FP32)
            (torch.int8, m * 32),         # routing (~ m*n_experts/8 bytes)
        ]
        total_bytes = m * 3072 * 2 + m * 256 * 4 + m * 32
        if total_bytes > max_bytes:
            logging.info(f"[minimax_m2/m={m}] skipped (total={total_bytes/1024/1024:.1f}MB > pool)")
            continue
        msg = f"[minimax_m2/m={m}/total={total_bytes / 1024 / 1024:.1f}MB]"
        run_test_mixed_dtypes(spec, msg)
        cases += 1

    logging.info(f"concat: {cases} cases executed!")
