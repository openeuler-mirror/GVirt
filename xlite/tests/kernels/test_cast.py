#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""xlite cast_up (bfloat16 -> float32) vs torch_npu ``Tensor.float``.

Covers S1 cast_up accuracy plus the S2/S3/S4 tail-burst regressions for
0d31c55 (40-core ceil split, unaligned tail burst spills stale UB slots;
two-input alternation amplifies the race; S3 walks castedOut via matmul at
the i5 geometry). S4 covers the fp32->bf16 ``cast_down`` tail over-read.

Collected by the default pytest run; also directly executable:
``python tests/kernels/test_cast.py``.
"""

import sys

import pytest
import torch
import torch_npu  # noqa: F401

from xlite._C import cast_down, cast_up, matmul


@pytest.mark.parametrize("seed", range(20), ids=lambda x: f"seed{x}")
def test_cast_up(runtime, seed):
    torch.manual_seed(seed)
    x = torch.randn(8, 2048, dtype=torch.bfloat16)
    y = torch.empty(8, 2048, dtype=torch.float)
    standard = x.float()

    torch.npu.synchronize()
    cast_up(runtime, x, y)
    torch.npu.synchronize()

    # bf16 -> fp32 is exact (every bf16 value is representable in fp32), so
    # only a real kernel bug can miss this tolerance.
    assert torch.equal(standard, y)


def test_cast_up_tailburst(runtime):
    # ceil(16384/40)=410 elems/core, 410*4B % 32 != 0 -> tail burst spans 2 cores
    xa = torch.randn(16384, dtype=torch.bfloat16)
    xb = torch.randn(16384, dtype=torch.bfloat16) * -1.5
    ya = torch.empty(16384, dtype=torch.float)
    yb = torch.empty(16384, dtype=torch.float)
    torch.npu.synchronize()
    bad = 0
    for _ in range(20):
        cast_up(runtime, xa, ya)
        cast_up(runtime, xb, yb)
        torch.npu.synchronize()
        bad += int(not torch.equal(xa.float(), ya)) + int(not torch.equal(xb.float(), yb))
    assert bad == 0, f"cast_up tail-burst alternation x20 mismatches={bad}"


def test_matmul_after_cast_up_tailburst(runtime):
    # Walk castedOut via matmul at the i5 geometry: the unaligned tail burst
    # must not leave stale UB slots behind for the next kernel launch.
    m, n, k = 3784, 16, 5120
    xm = torch.randn(m, k, dtype=torch.bfloat16)
    wm = torch.randn(n, k, dtype=torch.float) * 0.05
    zm = torch.empty(m, n, dtype=torch.bfloat16)
    x2 = torch.randn(1024, k, dtype=torch.bfloat16)
    w2 = torch.randn(n, k, dtype=torch.float) * 0.05
    z2 = torch.empty(1024, n, dtype=torch.bfloat16)
    torch.npu.synchronize()
    matmul(runtime, xm, wm, zm)
    base = zm.clone()
    bad = 0
    for _ in range(40):
        matmul(runtime, x2, w2, z2)
        matmul(runtime, xm, wm, zm)
        torch.npu.synchronize()
        bad += int(not torch.equal(zm, base))
    ref = (xm.float() @ wm.t()).to(torch.bfloat16)
    close = torch.allclose(ref, zm, atol=1e-2, rtol=1e-2)
    assert bad == 0 and close, f"castedOut tail-burst bitdiff_iters={bad}/40 allclose_ref={close}"


# S4-cast_down-tailoverread: fp32->bf16 cast. vconv over-reads UB past the GM->UB load;
# with tile_size already 16-aligned the trigger is length mod 16 in (0,8], where the
# block-aligned write (ceil-to-16 bf16) reaches past the load (ceil-to-8 fp32) and flushes
# the garbage tail to GM. Corruption lands only in the padding lanes past `length`, so the
# kernel sees `length` via a narrowed view while the backing storage spans the rounded
# footprint, and the full footprint (not just [0,length)) is compared; two-input alternation
# ensures stale UB differs from the expected tail. Reverting cast.h's load-count fix
# re-corrupts the trigger lengths (8/20/24/40962/40966) but leaves the safe ones (16/12/32/40960).
@pytest.mark.parametrize("length", [8, 20, 24, 40962, 40966, 16, 12, 32, 40960], ids=lambda x: f"len{x}")
def test_cast_down_tailoverread(runtime, length):
    footprint = ((length + 15) // 16) * 16
    a = torch.randn(footprint, dtype=torch.float)
    b = torch.randn(footprint, dtype=torch.float) * -1.5
    ya = torch.empty(footprint, dtype=torch.bfloat16)
    yb = torch.empty(footprint, dtype=torch.bfloat16)
    torch.npu.synchronize()
    av, bv = a.narrow(0, 0, length), b.narrow(0, 0, length)
    yav, ybv = ya.narrow(0, 0, length), yb.narrow(0, 0, length)
    bad = 0
    for _ in range(20):
        cast_down(runtime, av, yav)
        cast_down(runtime, bv, ybv)
        torch.npu.synchronize()
        bad += int(not torch.allclose(a.to(torch.bfloat16), ya, atol=1e-2, rtol=1e-2))
        bad += int(not torch.allclose(b.to(torch.bfloat16), yb, atol=1e-2, rtol=1e-2))
    assert bad == 0, f"cast_down tail-overread footprint sweep mismatches={bad}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", *sys.argv[1:]]))
