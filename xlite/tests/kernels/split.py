#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
# Test for the split kernel: deinterleave a flat byte buffer holding
# ``num_packets`` packets, each packet composed of several segments whose byte
# sizes are given by ``sizes``. For packet i and segment j:
#   copy sizes[j] bytes -> outputs[j] + i*sizes[j]
# This mirrors the MoE packed-recv deinterleaving path.
#
# Two regimes are exercised:
#   (1) small sizes (correctness sweep over dtypes / num_packets / segment counts)
#   (2) large sizes (multi-core stride path + minimax_m2 real workload scale)
# ===============================================================================

import logging
import os
import numpy as np
import torch
from xlite._C import Runtime, split

logging.getLogger().setLevel(logging.INFO)

# Pick a device: honor XLITE_TEST_DEVICE env var, default 0. Use a free device
# (one with most spare memory) when running the large/minimax_m2-scale cases,
# since the pool must hold ~2x the largest totalBytes.
DEVICE = int(os.environ.get("XLITE_TEST_DEVICE", "0"))
POOL_MB = int(os.environ.get("XLITE_TEST_POOL_MB", "500"))
rt = Runtime(DEVICE, POOL_MB)
torch.npu.set_device(DEVICE)
torch.set_printoptions(threshold=np.inf)

DTYPE_LIST = [torch.float16, torch.bfloat16, torch.float32, torch.int32, torch.int8]


def elem_size(dtype):
    return torch.tensor([], dtype=dtype).element_size()


def to_numpy(t):
    """cpu().numpy() safe for all dtypes incl. bfloat16."""
    return t.cpu().to(torch.float32).numpy()


def run_test(num_packets, sizes_elems, dtype, msg):
    """num_packets packets, each holding segments of sizes_elems[j] elements."""
    es = elem_size(dtype)
    sizes_bytes = [s * es for s in sizes_elems]
    total_elems = sum(sizes_elems) * num_packets

    in_ = torch.randn(total_elems, dtype=dtype, device=f"npu:{DEVICE}")
    if dtype in (torch.int32, torch.int8):
        in_.random_(-100, 100)
    in_host = to_numpy(in_)

    outputs = [
        torch.empty(sizes_elems[j] * num_packets, dtype=dtype, device=f"npu:{DEVICE}")
        for j in range(len(sizes_elems))
    ]

    torch.npu.synchronize()
    split(rt, in_, outputs, sizes_bytes, num_packets)
    torch.npu.synchronize()

    # CPU reference: walk packets, within each packet walk segments.
    ref = [np.zeros(sizes_elems[j] * num_packets, dtype=in_host.dtype) for j in range(len(sizes_elems))]
    off = 0
    for i in range(num_packets):
        for j in range(len(sizes_elems)):
            n = sizes_elems[j]
            ref[j][i * n:(i + 1) * n] = in_host[off:off + n]
            off += n

    for j in range(len(sizes_elems)):
        got = to_numpy(outputs[j])
        if not np.array_equal(got, ref[j]):
            raise AssertionError(
                f"{msg}: segment {j} mismatch\n  got={got[:16]}\n  ref={ref[j][:16]}"
            )
    total_kb = total_elems * es / 1024
    logging.info(
        f"{msg}: PASS (num_packets={num_packets}, "
        f"sizes_elems={sizes_elems}, dtype={dtype}, total={total_kb:.0f}KB)"
    )


if __name__ == "__main__":
    cases = 0

    # ---- (0) guard: too-small output must be rejected, not corrupt memory ----
    # Public API; an external caller passing an undersized output[j] would
    # otherwise trigger an out-of-bounds device write. The host wrapper must
    # raise before launching the kernel.
    in0 = torch.randint(-100, 100, (16,), dtype=torch.int8, device=f"npu:{DEVICE}")
    o_ok = torch.empty(8, dtype=torch.int8, device=f"npu:{DEVICE}")
    o_small = torch.empty(4, dtype=torch.int8, device=f"npu:{DEVICE}")  # needs 8
    raised = False
    try:
        split(rt, in0, [o_ok, o_small], [8, 8], 1)
        torch.npu.synchronize()
    except RuntimeError as e:
        raised = "output[1]" in str(e) and "bytes (4)" in str(e)
    logging.info(f"[guard/too-small-output]: {'PASS' if raised else 'FAIL'}")
    if not raised:
        raise AssertionError("split accepted an undersized output (expected RuntimeError)")
    cases += 1

    # ---- (1) small correctness sweep ----
    for dtype in DTYPE_LIST:
        for num_packets in [1, 2, 4, 8]:
            for sizes_elems in [
                [1],
                [100],
                [1, 1],
                [7, 3],
                [16, 32, 64],
                [1, 2, 3, 4, 5, 6, 7, 8],   # 8 segments (max supported)
                [1024, 256, 128],
            ]:
                msg = f"[small/{dtype}/{num_packets}/{sizes_elems}]"
                run_test(num_packets, sizes_elems, dtype, msg)
                cases += 1
    logging.info(f"split small sweep: {cases} cases done")

    # ---- (2) large sizes: exercise multi-core stride path ----
    # totalBytes (= total_elems*es) crosses block_num*dataBufSize so all cores engage.
    # Cap each case so input+outputs (~2x totalBytes) fit in the tensor pool.
    max_bytes = POOL_MB * 1024 * 1024 // 2
    for dtype in [torch.float16, torch.bfloat16]:
        es = elem_size(dtype)
        for num_packets in [1, 2, 4, 8]:
            # one dominant segment ~ minimax_m2 input segment, scaled
            for seg_elems in [100_000, 1_000_000, 3_000_000]:
                total_bytes = seg_elems * num_packets * es
                if total_bytes > max_bytes:
                    continue
                sizes_elems = [seg_elems, seg_elems // 4, seg_elems // 64]  # 3 segments
                msg = f"[large/{dtype}/{num_packets}/{seg_elems}]"
                run_test(num_packets, sizes_elems, dtype, msg)
                cases += 1
    logging.info(f"split large sweep done, running minimax_m2-scale case")

    # ---- (3) minimax_m2 real workload: packedRecv scale ----
    # After AllGather, packedRecv = totalBytes * defDpSize. With defDpSize packets,
    # each packet = [input m*3072*2 B, weights m*256*4 B, routing m*32 B].
    # Here m=32768 gives ~230 MB/packet; simulate defDpSize via num_packets.
    m = 32768
    # segment element counts (bf16 input / fp32 weights / int8 routing-ish)
    sizes_elems = [m * 3072, m * 256, m * 32]  # bf16 elems, fp32 elems, int8 elems
    for num_packets, dtype_pair in [(1, torch.bfloat16), (2, torch.float16)]:
        es = elem_size(dtype_pair)
        total_bytes = sum(sizes_elems) * es * num_packets
        if total_bytes > max_bytes:
            continue
        msg = f"[minimax_m2/m={m}/np={num_packets}/total={total_bytes/1024/1024:.0f}MB]"
        # all segments share dtype for the test harness; sizes in elements
        run_test(num_packets, sizes_elems, dtype_pair, msg)
        cases += 1

    logging.info(f"split: {cases} cases executed!")
