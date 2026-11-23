#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
# S1 cast_up accuracy; S2/S3 tail-burst regression for 0d31c55 (40-core ceil
# split, unaligned tail burst spills stale UB slots; two-input alternation
# amplifies the race; S3 walks castedOut via matmul at the i5 geometry).
import sys

import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, cast_up, matmul

rt = Runtime(0, size=500)
torch.npu.set_device(0)
torch.manual_seed(0)

ok = True

x = torch.randn(8, 2048, dtype=torch.bfloat16, device="npu:0")
y = torch.empty(8, 2048, dtype=torch.float, device="npu:0")
torch.npu.synchronize()
cast_up(rt, x, y)
torch.npu.synchronize()
good = torch.equal(x.float(), y)
print(f"[S1-cast_up-basic] {'PASS' if good else 'FAIL'}")
ok &= good

# ceil(16384/40)=410 elems/core, 410*4B % 32 != 0 -> tail burst spans 2 cores
xa = torch.randn(16384, dtype=torch.bfloat16, device="npu:0")
xb = torch.randn(16384, dtype=torch.bfloat16, device="npu:0") * -1.5
ya = torch.empty(16384, dtype=torch.float, device="npu:0")
yb = torch.empty(16384, dtype=torch.float, device="npu:0")
torch.npu.synchronize()
bad2 = 0
for _ in range(20):
    cast_up(rt, xa, ya)
    cast_up(rt, xb, yb)
    torch.npu.synchronize()
    bad2 += int(not torch.equal(xa.float(), ya)) + int(not torch.equal(xb.float(), yb))
print(f"[S2-cast_up-tailburst] alternation x20 mismatches={bad2} -> {'PASS' if bad2 == 0 else 'FAIL'}")
ok &= bad2 == 0

M, N, K = 3784, 16, 5120
xm = torch.randn(M, K, dtype=torch.bfloat16, device="npu:0")
wm = torch.randn(N, K, dtype=torch.float, device="npu:0") * 0.05
zm = torch.empty(M, N, dtype=torch.bfloat16, device="npu:0")
x2 = torch.randn(1024, K, dtype=torch.bfloat16, device="npu:0")
w2 = torch.randn(N, K, dtype=torch.float, device="npu:0") * 0.05
z2 = torch.empty(1024, N, dtype=torch.bfloat16, device="npu:0")
torch.npu.synchronize()
matmul(rt, xm, wm, zm)
base = zm.clone()
bad3 = 0
for _ in range(40):
    matmul(rt, x2, w2, z2)
    matmul(rt, xm, wm, zm)
    torch.npu.synchronize()
    bad3 += int(not torch.equal(zm, base))
ref = (xm.float() @ wm.t()).to(torch.bfloat16)
close = torch.allclose(ref, zm, atol=1e-2, rtol=1e-2)
print(
    f"[S3-castedOut-tailburst] bitdiff_iters={bad3}/40 allclose_ref={close} -> "
    f"{'PASS' if bad3 == 0 and close else 'FAIL'}"
)
ok &= bad3 == 0 and close

sys.exit(0 if ok else 1)
