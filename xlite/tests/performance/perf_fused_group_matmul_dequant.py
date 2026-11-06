#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranties of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""fusion group_matmul+dequant 两段融合 vs 串行链路的 perf 被测脚本.

设计为被 tests/performance/perf_xlite_ops.py 包裹运行(逐调用 wall + device 计时):
    cd <非 xlite 仓库根的目录>
    python <xlite>/tests/performance/perf_fused_group_matmul_dequant.py

对照对象(量化在两臂之外先行——与 model.cpp ForwardMoE INT8 路径一致, INT8 输入由
QuantDyn 或 dispatch 前移产出, m 为 sum(counts) 的路由后总行数):
  串行2: group_matmul + dequant (XliteOpGroupMatmulDeQuant 未命中收益函数时的路径)
  融合2: fused_group_matmul_dequant (单次 MIX kernel, 两段全融合;
         即 XliteOpGroupMatmulDeQuant 收益函数命中时的派发目标)
  quant_dynamic 每轮先行产出共享的 (xq, out_scale), 不参与两臂对比.

测试矩阵(动态量化——MoE routed expert 为动态 W8A8):
  A. GLM MoE TP16 per-rank 尺寸: w13 (n=2*inter, k=hidden) 与 w2 (n=hidden,
     k=inter), hidden 7168/6144 两档, group_num=16;
  B. m 扫描覆盖 decode(128)/prefill chunk(1024,4096).

每个 (size, m) 配置: 构造一次张量, warmup 1 轮 + REPS 轮, 每轮固定顺序交错调用
4 个算子; warmup 轮做设备侧正确性校验(串行2 vs 融合2, allclose). ffts flag 积压
安全边界: 融合算子 tiling 固定 m0=128/n0=256/k0=128(int8), k_loop < 8 时按每核
tile 数×(8-k_loop) 估积压, 阈值 128 本机校准.
"""

import os
import time

import torch
import torch_npu

from xlite._C import (
    Runtime,
    dequant,
    fused_group_matmul_dequant,
    group_matmul,
    quant_dynamic,
)

# allow weight_nz
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

# 目标卡可经 XLITE_TEST_DEVICE 指定（等待任意空闲卡时用）
dev_id = int(os.environ.get("XLITE_TEST_DEVICE", "0"))
rt = Runtime(dev_id, size=500)
torch.npu.set_device(dev_id)

ACL_FORMAT_FRACTAL_NZ = 29
WARMUP_ROUNDS = 1
REPS = 10
OP_ORDER = [
    "quant_dynamic",
    "group_matmul",
    "dequant",
    "fused_group_matmul_dequant",
]

# m 扫描覆盖 decode(128)/prefill chunk(1024,4096); 可经 XLITE_PERF_M_VALUES 覆盖
# (如小 m 阈值标定 "8,16,32,64,128"); 共占设备 HBM 紧张时减小 m
M_VALUES = [int(v) for v in os.environ.get("XLITE_PERF_M_VALUES", "128,1024,4096").split(",")]

# ---- A. GLM MoE TP16 per-rank sizes ----
# (label, n, k, group_num, calls_per_forward); routed experts 均为动态 W8A8,
# 每 MoE 层一次 w13 + 一次 w2 (group 算子单调用覆盖全部本地专家).
MODEL_SIZES = [
    ("glm_w13", 4096, 7168, 16, 1),  # n=2*inter(2048), k=hidden(7168)
    ("glm_w2", 7168, 2048, 16, 1),  # n=hidden, k=inter
    ("glm_w13_h6k", 3072, 6144, 16, 1),  # hidden 6144 / inter 1536 档
    ("glm_w2_h6k", 6144, 1536, 16, 1),
]
CONFIGS = MODEL_SIZES

AIC_NUM = 25  # 本机 910B, 与 csrc 运行时 aclGetDeviceCapability 取值一致


def group_fused_safe(m, n, k, g):
    """ffts flag 积压安全边界(融合 group 算子 tiling 固定 m0=128/n0=256).

    每核 tile 数按均匀 counts 估算: g * ceil((m/g)/128) * ceil(n/256) / AIC_NUM;
    k_loop=ceil(k/128) < 8 时按 每核tile×(8-k_loop) 估积压, 阈值 128 本机校准.
    MoE k>=1536 时 k_loop>=12, 恒安全."""
    m_loop = -(-(m // g) // 128)
    n_loop = -(-n // 256)
    k_loop = -(-k // 128)
    tiles = g * m_loop * n_loop
    tiles_per_core = -(-tiles // AIC_NUM)
    return not (k_loop < 8 and tiles_per_core * (8 - k_loop) >= 128)


def run_config(label, m, n, k, g):
    dev = f"npu:{dev_id}"
    x = torch.randn(m, k, dtype=torch.bfloat16, device=dev) * 2.0
    # 精确求和到 m: m<g 时 torch.full((g,), m//g) 会全零 (sum=0 != m),
    # 串行/融合两侧都写 0 行, 对比的是两份未初始化内存
    counts = torch.full((g,), m // g, dtype=torch.int32, device=dev)
    counts[: m % g] += 1

    weights = []
    deq_scales = []
    for _ in range(g):
        weight_ref = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=dev)
        # 与模型部署一致: transpose=True + NZ 布局
        y_in = torch_npu.npu_format_cast(weight_ref.t().contiguous(), ACL_FORMAT_FRACTAL_NZ)
        weights.append(y_in)
        # deqscale 从 bf16 转换来的 fp32 可被 fixpipe 无损消费
        deq_scale = torch.rand(n, dtype=torch.float32, device=dev) * 0.015 + 0.001
        deq_scale = deq_scale.to(torch.bfloat16).to(torch.float32)
        # fixpipe 布局: uint64 槽位低 32 位存 fp32
        scale = torch.zeros(n * 2, dtype=torch.float32, device=dev)
        scale[0::2] = deq_scale
        deq_scales.append(scale)

    # 动态 W8A8: per-token out_scale 由 quant_dynamic 每轮写入, 两臂共用
    out_scale = torch.empty(m, dtype=torch.float32, device=dev)
    xq = torch.empty(m, k, dtype=torch.int8, device=dev)
    serial_tmp = torch.empty(m, n, dtype=torch.float16, device=dev)
    standard = torch.empty(m, n, dtype=torch.bfloat16, device=dev)
    fused_out = torch.empty(m, n, dtype=torch.bfloat16, device=dev)

    torch.npu.synchronize()
    try:
        for rep in range(WARMUP_ROUNDS + REPS):
            # 两臂共用的量化: quant_dynamic -> (xq, out_scale)
            quant_dynamic(rt, x, out_scale, xq)
            time.sleep(0.001)
            # 串行2 基线: group_matmul -> dequant
            group_matmul(rt, xq, weights, deq_scales, counts, 0, g, n, k, serial_tmp, True, True)
            time.sleep(0.001)
            dequant(rt, serial_tmp, out_scale, standard, True)
            time.sleep(0.001)
            # 两段融合 (XliteOpGroupMatmulDeQuant 收益函数命中时的派发目标)
            fused_group_matmul_dequant(
                rt, xq, weights, deq_scales, counts, 0, g, n, k, fused_out, out_scale, True, True
            )
            time.sleep(0.001)
            if rep == 0:
                ok = torch.allclose(standard, fused_out, atol=1e-2, rtol=1e-2)
                if not ok:
                    diff = (standard.float() - fused_out.float()).abs().max().item()
                    print(f"[WARN] {label} m={m} n={n} k={k} mismatch max_diff={diff:.4f}")
    finally:
        del x, counts, weights, deq_scales, out_scale, xq, serial_tmp
        del standard, fused_out


for label, n, k, g, _ in CONFIGS:
    for m in M_VALUES:
        try:
            safe = group_fused_safe(m, n, k, g)
            if not safe:
                print(f"[skip-fused] {label} m={m} n={n} k={k} g={g} (ffts flag 积压安全边界)")
            run_config(label, m, n, k, g)
        except torch.OutOfMemoryError:
            print(f"[skip-oom] {label} m={m} n={n} k={k} g={g}")

print("perf matrix done")
