#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Runtime recycle smoke test for the overflow-status buffer removal.

scripts/patch_host_stub.py removes the per-launch overflow-status device
malloc/free from the generated host stubs: the launch-argument slot is set
to null and never allocated (the kernel-side write is null-guarded, so
passing null just disables the debug-only overflow-status reporting; real
overflow reporting goes through AICore hardware traps /
aclrtGetOverflowStatus and does not involve this slot). This test exercises
the sensitive path: destroy a runtime (device reset) and build a new one in
the SAME process, then launch kernels again and check numerics -- the stubs
must keep working with no per-runtime state of any kind.
"""

import torch
from xlite._C import Runtime, add

DEVICE = 0
POOL_MB = 500


def run_round(tag):
    rt = Runtime(DEVICE, POOL_MB)
    x = torch.randn(8, 2048, dtype=torch.float16, device="npu:0")
    y = torch.randn(8, 2048, dtype=torch.float16, device="npu:0")
    z = torch.empty(8, 2048, dtype=torch.float16, device="npu:0")
    standard = x + y
    torch.npu.synchronize()
    add(rt, x, y, z)
    torch.npu.synchronize()
    torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
    # `del rt` runs the XRuntime destructor: aclrtResetDevice (and
    # aclFinalize when this process owned aclInit).
    del rt
    print(f"{tag}: add executed and matched after runtime teardown")


def main():
    torch.npu.set_device(0)
    run_round("round1")
    run_round("round2")
    run_round("round3")
    print("PASS: 3 runtime create/destroy cycles, kernels correct each time")


if __name__ == "__main__":
    main()
