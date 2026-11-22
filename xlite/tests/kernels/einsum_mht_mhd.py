#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import torch
import torch_npu  # noqa: F401

from tests.models.weight_utils import nd2nz_or_skip
from xlite._C import Runtime, einsum_mht_hdt_mhd, einsum_mht_htd_mhd

rt = Runtime(0, size=500)
torch.npu.set_device(0)
torch.npu.config.allow_internal_format = True

# (m, h, t, d)
test_cases = [
    (3, 4, 192, 512),
    (3, 4, 512, 192),
    (3, 4, 512, 256),
    (3, 4, 256, 512),
    (3, 4, 128, 512),
    (3, 4, 512, 128),
    (5230, 4, 192, 512),
    (5230, 4, 512, 192),
    (5230, 4, 512, 256),
    (5230, 4, 256, 512),
    (5230, 4, 128, 512),
    (5230, 4, 512, 128),
    (963, 8, 512, 128),
    # chunked-traversal coverage (mLoop >= 2 path): with aicNum = 24 cores,
    # h = 64 makes one core's contiguous chunk span ~2.7 whole heads (several
    # RebindGm head transitions inside a single core's range), h = 32 with
    # nLoop = 2 exercises the midx/nidx wrap inside a chunk; m = 257 adds a
    # 1-row tail m-tile
    (257, 64, 128, 128),
    (257, 32, 128, 512),
    # t > 8 * k0 (= 512): the B ping-pong slots cannot hold the whole K range,
    # so the chunked traversal is disarmed and the legacy round-robin runs
    # with mLoop >= 2 (kLoop = 9: the third B window recycles slot 0)
    (5230, 4, 576, 128),
    # 4*k0 < t < 8*k0 and t not a multiple of 4*k0 (bf16/fp16 k0=64 ->
    # t=384): one tile holds two UNEQUAL B windows (k-block counts 16/8),
    # so the window block count must be tracked per L1 slot — a member
    # shared by both slots carries the second window's count into a
    # first-window residency hit, and the transpose consumer
    # (CopyToL0BTCol srcStride) then reads L1 at the wrong n-fractal
    # stride. mLoop=3 x nLoop=1 x h=64 -> totalTiles=192 > aicNum, so the
    # chunked traversal really produces in-group hits (regression anchor:
    # hit tiles failed with ~1e2 garbage before the per-slot fix).
    (257, 64, 384, 128),
]


def run_case(einsum_op, weight_layout, dtype, m, h, t, d, weight_nz):
    mht = torch.randn(m, h, t, dtype=dtype, device="npu:0")
    htd = torch.randn(h, t, d, dtype=dtype, device="npu:0")
    mhd = torch.zeros(m, h, d, dtype=dtype, device="npu:0")

    standard = torch.einsum("mht,htd->mhd", mht, htd)

    if weight_layout == "htd":
        # kernel consumes right operand as [h, t, d], transpose path
        y = htd.contiguous()
    else:
        # weight_layout == "hdt": kernel consumes [h, d, t], non-transpose path
        y = htd.transpose(-1, -2).contiguous()

    if weight_nz:
        y = nd2nz_or_skip(y, f"[m={m}, h={h}, t={t}, d={d}, {dtype}, layout={weight_layout}]")
        if y is None:
            return True

    torch.npu.synchronize()
    einsum_op(rt, mht, y, mhd, m, h, t, d, weight_nz=weight_nz)

    # torch.einsum and the kernel both accumulate in fp32 but in different
    # orders (the kernel splits K into 64-wide Mmad chunks), so a rounded
    # output can land 1 ULP away from torch's. bf16: a 1-ULP flip is a
    # relative diff up to 2^-7, needing rtol=1e-2. fp16: 1-ULP flips
    # (rel diff 2^-10) stay within rtol=1e-3, but at large t catastrophic
    # cancellation leaves near-zero outputs where only absolute noise
    # remains (measured <5e-5 across 5 seeds), hence atol=5e-5. A real
    # bug (wrong tile / garbage, O(1) magnitude) still fails loudly.
    if dtype == torch.bfloat16:
        atol, rtol = 1e-3, 1e-2
    else:
        atol, rtol = 5e-5, 1e-3
    try:
        torch.testing.assert_close(standard, mhd, atol=atol, rtol=rtol)
    except AssertionError as e:
        print(f"[m={m}, h={h}, t={t}, d={d}, {dtype}, layout={weight_layout}, weight_nz={weight_nz}] {e}")
        print(f"torch npu: {standard}")
        print(f"xlite: {mhd}")
        return False
    print(f"[m={m}, h={h}, t={t}, d={d}, {dtype}, layout={weight_layout}, weight_nz={weight_nz}] einsum executed!")
    return True


if __name__ == "__main__":
    failed = False
    for dtype in [torch.float16, torch.bfloat16]:
        for weight_nz in [False, True]:
            for m, h, t, d in test_cases:
                ok = run_case(einsum_mht_htd_mhd, "htd", dtype, m, h, t, d, weight_nz)
                failed = failed or not ok
                ok = run_case(einsum_mht_hdt_mhd, "hdt", dtype, m, h, t, d, weight_nz)
                failed = failed or not ok
    if failed:
        raise SystemExit(1)
