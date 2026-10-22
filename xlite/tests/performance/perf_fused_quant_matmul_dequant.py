#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without any warranties of any kind.
# ===============================================================================
"""fusion quant+matmul+dequant 三段融合 vs 现有两条路径的 perf 被测脚本.

设计为被 tests/performance/perf_xlite_ops.py 包裹运行(逐调用 wall + device 计时):
    cd <非 xlite 仓库根的目录>
    python <xlite>/tests/performance/perf_xlite_ops.py \
        <xlite>/tests/performance/perf_quant_matmul_dequant_fusion.py

对照对象(全部从同一 BF16 x 出发, 与 model.cpp ForwardDense 的分派路径对应):
  串行3: quant_dynamic + matmul_dequant + dequant (XLITE_FUSED_DENSE_W8A8 关闭)
  quant+融合2: quant_dynamic + fusion_operator_matmul_dequant_pipeline (开关开启的现有路径)
  融合3: fused_quant_matmul_dequant (新算子, 三段全融合)

测试矩阵(动态量化为主——GLM dense/shared-expert 均为动态 W8A8):
  A. GLM-5.1 TP16 per-rank 真实动态 dense size (calls_per_forward 为每 forward 调用数);
  B. m 扫描覆盖 launch-bound 段(128~512)与 prefill 段(1024~4096).

每个 (size, m) 配置: 构造一次张量, warmup 1 轮 + REPS 轮, 每轮固定顺序交错调用
5 个算子; warmup 轮做设备侧正确性校验(串行3 vs 融合3, allclose). ffts flag 积压
安全边界: C2V per-tile flag 与现有融合算子一致, 同式判定 ffts 积压阈值.
"""
import os
import time

import torch
import torch_npu
from xlite._C import (
    Runtime,
    dequant,
    fusion_operator_matmul_dequant_pipeline,
    fused_quant_matmul_dequant,
    matmul_dequant,
    quant,
    quant_dynamic,
)

# allow weight_nz
torch.npu.set_option({"ALLOW_INTERNAL_FORMAT": True})

dev_id = 0
rt = Runtime(dev_id, 500)
torch.npu.set_device(dev_id)

ACL_FORMAT_FRACTAL_NZ = 29
WARMUP_ROUNDS = 1
REPS = 10
OP_ORDER = [
    "quant_dynamic",
    "matmul_dequant",
    "dequant",
    "fusion_operator_matmul_dequant_pipeline",
    "fused_quant_matmul_dequant",
]

# m 扫描覆盖 decode(128)/prefill chunk(1024,4096); 共占设备 HBM 紧张时减小 m
M_VALUES = [128, 1024, 4096]

# ---- A. GLM-5.1 TP16 per-rank real sizes ----
# (label, n, k, is_dyn, has_bias, calls_per_forward); attn = static W8A8
# (quant_bias, no out_scale), dense/shared-expert = dynamic W8A8.
MODEL_SIZES = [
    ("mla_qkv_a", 2624, 6144, False, True, 78),
    ("mla_q_b", 1024, 2048, False, True, 78),
    ("o_proj", 6144, 1024, False, True, 78),
    ("indexer_wq_b", 256, 2048, False, True, 78),
    ("se_gate_up", 256, 6144, True, False, 75),
    ("se_down", 6144, 128, True, False, 75),
    ("dense_gate_up", 1536, 6144, True, False, 3),
    ("dense_down", 6144, 768, True, False, 3),
]
CONFIGS = MODEL_SIZES

AIC_NUM = 25  # 本机 910B, 与 csrc 运行时 aclGetDeviceCapability 取值一致


def pick_matmul_tiling(m, n, k):
    """复刻 auto_tuner.cpp PickMatmulTiling 的 int8 无 bias/deq 主路径, 与 C++ 同步维护."""
    m0 = min((m + 31) // 32 * 32, 128)
    n0 = 256
    k0 = 512
    m_loop = -(-m // m0)
    n_loop = -(-n // n0)
    total = m_loop * n_loop
    last = total % AIC_NUM
    if total < 3 * AIC_NUM and last != 0 and last < AIC_NUM / 2:
        if n <= 32 * AIC_NUM:
            m0 = min(m0, 64)
            n0 = 64
        elif n <= 64 * AIC_NUM:
            n0 = 64
        elif n <= 128 * AIC_NUM:
            n0 = 128
        elif n <= 256 * AIC_NUM:
            n0 = 256
        else:
            m0 = min(m0, 64)
            n0 = 384
            k0 //= 2
    launch = max(min(m_loop * n_loop, AIC_NUM), 1)
    return m0, n0, k0, launch


def fused_safe(m, n, k):
    """ffts flag 积压安全边界(同 perf_matmul_dequant_fusion.py, 阈值 128 本机校准).
    新算子每 AIC tile 数 = ceil((mLoop-b)/N)*nLoop (midx-first), 与 grid-stride 的
    tiles/N 同阶, 同式判定."""
    m0, n0, k0, launch = pick_matmul_tiling(m, n, k)
    m_loop = -(-m // m0)
    n_loop = -(-n // n0)
    k_loop = -(-k // (k0 >> 2))
    tiles = -(-(m_loop * n_loop) // launch)
    return not (k_loop < 8 and tiles * (8 - k_loop) >= 128)


def run_config(label, m, n, k, is_dyn, has_bias):
    x = torch.randn(m, k, dtype=torch.bfloat16, device=f"npu:{dev_id}") * 2.0
    weight_ref = torch.randint(-8, 8, (n, k), dtype=torch.int8, device=f"npu:{dev_id}")
    # 与模型部署一致: transpose=True + NZ 布局
    y_in = torch_npu.npu_format_cast(weight_ref.t().contiguous(), ACL_FORMAT_FRACTAL_NZ)

    if has_bias:
        bias = torch.randint(-64, 64, (n,), dtype=torch.int32, device=f"npu:{dev_id}")
    else:
        bias = torch.empty(0, dtype=torch.int32, device=f"npu:{dev_id}")
    # deqscale 从 bf16 转换来的 fp32 可被 fixpipe 无损消费
    deq_scale = torch.rand(n, dtype=torch.float32, device=f"npu:{dev_id}") * 0.015 + 0.001
    deq_scale = deq_scale.to(torch.bfloat16).to(torch.float32)
    # fixpipe 布局: uint64 槽位低 32 位存 fp32
    scale = torch.zeros(n * 2, dtype=torch.float32, device=f"npu:{dev_id}")
    scale[0::2] = deq_scale
    num = torch.empty(0, dtype=torch.int32, device=f"npu:{dev_id}")

    # 静态 W8A8 (attn): per-channel scale/offset; 动态: per-token out_scale
    if is_dyn:
        srec = torch.empty(0, dtype=torch.bfloat16, device=f"npu:{dev_id}")
        soff = torch.empty(0, dtype=torch.bfloat16, device=f"npu:{dev_id}")
        out_scale = torch.rand(m, dtype=torch.float32, device=f"npu:{dev_id}") * 0.25 + 0.75
    else:
        srec = torch.rand(k, dtype=torch.bfloat16, device=f"npu:{dev_id}") * 0.05 + 0.01
        soff = torch.randn(k, dtype=torch.bfloat16, device=f"npu:{dev_id}") * 0.5
        out_scale = torch.empty(0, dtype=torch.float32, device=f"npu:{dev_id}")

    xq = torch.empty(m, k, dtype=torch.int8, device=f"npu:{dev_id}")
    serial_tmp = torch.empty(m, n, dtype=torch.float16, device=f"npu:{dev_id}")
    standard = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{dev_id}")
    fused2_out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{dev_id}")
    fused3_out = torch.empty(m, n, dtype=torch.bfloat16, device=f"npu:{dev_id}")
    x_quant = torch.empty(m, k, dtype=torch.int8, device=f"npu:{dev_id}")
    out_scale_ws = torch.empty(m, dtype=torch.float32, device=f"npu:{dev_id}")

    safe = fused_safe(m, n, k)

    # 新三段融合的可选参数按需传入, 其余省略 (接口侧默认 None)
    fused3_opt = {}
    if is_dyn:
        fused3_opt["out_scale"] = out_scale_ws
    else:
        fused3_opt["quant_scale_rec"] = srec
        fused3_opt["quant_offset"] = soff
    if has_bias:
        fused3_opt["bias"] = bias

    torch.npu.synchronize()
    try:
        for rep in range(WARMUP_ROUNDS + REPS):
            # 串行3 基线: quant(_dynamic) -> matmul_dequant -> dequant
            if is_dyn:
                quant_dynamic(rt, x, out_scale, xq)
            else:
                quant(rt, x, srec, soff, xq)
            time.sleep(0.001)
            matmul_dequant(rt, xq, y_in, bias, scale, serial_tmp, True, True)
            time.sleep(0.001)
            dequant(rt, serial_tmp, out_scale, standard, is_dyn)
            time.sleep(0.001)
            if safe:
                # quant + 现有两段融合 (XLITE_FUSED_DENSE_W8A8 开启的当前生产路径)
                if is_dyn:
                    quant_dynamic(rt, x, out_scale, xq)
                else:
                    quant(rt, x, srec, soff, xq)
                time.sleep(0.001)
                fusion_operator_matmul_dequant_pipeline(
                    rt, xq, y_in, fused2_out, bias, scale, True, True, out_scale, num
                )
                time.sleep(0.001)
                # 新三段融合 (可选参数按需省略)
                fused_quant_matmul_dequant(
                    rt, x, weight=y_in, out=fused3_out, deq_scale=scale, x_quant=x_quant,
                    dyn_quant=is_dyn, weight_nz=True, transpose=True, **fused3_opt
                )
                time.sleep(0.001)
            if rep == 0 and safe:
                ok = torch.allclose(standard, fused3_out, atol=1e-2, rtol=1e-2)
                if not ok:
                    diff = (standard.float() - fused3_out.float()).abs().max().item()
                    print(f"[WARN] {label} m={m} n={n} k={k} mismatch max_diff={diff:.4f}")
    finally:
        del x, weight_ref, y_in, bias, scale, num, xq, srec, soff
        del out_scale, serial_tmp, standard, fused2_out, fused3_out, x_quant, out_scale_ws


for label, n, k, is_dyn, has_bias, _ in CONFIGS:
    for m in M_VALUES:
        try:
            safe = fused_safe(m, n, k)
            if not safe:
                print(f"[skip-fused] {label} m={m} n={n} k={k} (ffts flag 积压安全边界)")
            run_config(label, m, n, k, is_dyn, has_bias)
        except torch.OutOfMemoryError:
            print(f"[skip-oom] {label} m={m} n={n} k={k}")

print("perf matrix done")
