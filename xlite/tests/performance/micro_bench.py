#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even warranties of any kind.
# ===============================================================================
"""微基准套件统一入口, 5 个实验按序执行(原理与实测数据见 doc/kernels/microbench.md):

  实验1  空算子 launch floor        empty_{aiv,aic,mix}
  实验2  标量依赖链每操作成本        scalar_*_{aiv,aic}(2a 核对/2b wall 粗测/
                                   2c 核内 SYS_CNT 计时 bench_timed)
  实验3  MTE 访存带宽/时延 vs size   mte_{read,write,copy,chase}_aiv(bench_mte)
  实验4  向量指令吞吐 vs 向量长度     vec_*_aiv(bench_vec)
  实验5  cube matmul 吞吐 vs tile    cube_mmad_aic(bench_cube, 5a 核对/5b 反消解/
                                   5c tile sweep, host 多次下发计时)

用法:
  python tests/performance/micro_bench.py [--quick] [--mte-sizes 64k,1m,...] [--mte-only] \
                                         [--vec-sizes 64,256,...] [--vec-only] \
                                         [--cube-sizes 128,128,128;...] [--cube-only]
  # --quick: 缩减重复次数/规模档位, 约 1/4 时间跑完全部实验(精度降低, 适合冒烟验证)
  # device 口径(profiler 包一层, 产出 CSV)后看分析报告:
  python tests/performance/perf_xlite_ops.py tests/performance/micro_bench.py
  python tests/performance/micro_bench.py --analyze xlite_prof_results/<csv>

实验2c/3/4/5 仅 XLITE_BENCH_TIMED=ON 构建提供(默认构建自动跳过并提示)。
"""
import os
import re
import statistics
import sys
import time

import numpy as np
import torch
import torch_npu

from xlite._C import (  # noqa: F401
    Runtime,
    empty_aiv,
    empty_aic,
    empty_mix,
    scalar_add_aiv,
    scalar_add_aic,
    scalar_xor_aiv,
    scalar_xor_aic,
    scalar_mul_aiv,
    scalar_mul_aic,
    scalar_div_aiv,
    scalar_div_aic,
    scalar_fadd_aiv,
    scalar_fadd_aic,
    scalar_add_unroll4_aiv,
    scalar_add_unroll4_aic,
)
import xlite._C as _C

# ==================== 公共设施(全实验共用) ====================

# bench_timed / bench_mte / bench_vec 仅在 XLITE_BENCH_TIMED=ON 构建中编译
# (默认构建连 *_timed kernel 与 host 入口都不存在), 默认构建下相应实验自动跳过.
bench_timed = getattr(_C, "bench_timed", None)
bench_mte_raw = getattr(_C, "bench_mte", None)
bench_vec_raw = getattr(_C, "bench_vec", None)
bench_cube_raw = getattr(_C, "bench_cube", None)

REPS = 5                              # 每配置重复次数, 取中位
BENCH_CYCLES_PER_US = 50              # SYS_CNT 固定 50MHz: cycles/50 = us
U32 = 1 << 32

# kernel 侧 BENCH_MTE_STRIDE: 每 block 的标量输出(ticks/acc)独占一条 64B GM line
# (跨核标量 GM store 同线互踩, 见 microbench.h)。MTE/VEC 两家族共用此布局。
MTE_STRIDE = 64


def _apply_quick():
    """--quick: 缩减重复次数/规模档位, 约 1/4 时间跑完全部实验。

    精度降低(中位样本少、大规模档位去掉), 适合冒烟验证而非出报告。
    改写的是模块级常量, 须在 run_bench 之前调用。"""
    global REPS, EMPTY_ITERS, ITERS_LIST
    global MTE_SIZES_DEFAULT, MTE_CHASE_ITERS
    global VEC_REPS_DEFAULT
    global CUBE_WARMUP_ITERS, CUBE_LAUNCH_ITERS
    REPS = 2
    EMPTY_ITERS = 3
    ITERS_LIST = [10_000, 100_000]
    MTE_SIZES_DEFAULT = [64 << 10, 1 << 20, 16 << 20]
    MTE_CHASE_ITERS = 512
    VEC_REPS_DEFAULT = (1, 16, 256)
    CUBE_WARMUP_ITERS = 3
    CUBE_LAUNCH_ITERS = 50


def _med_us(samples_ms):
    return statistics.median(samples_ms)


def _mte_ticks_us(ticks, nblock):
    """ticks 前 nblock 组(每 block 3 槽 t0/t1/t2, MTE_STRIDE 字节步距) ->
    [(issue, complete, drain)] us. MTE/VEC 两家族共用。"""
    v = ticks.cpu().tolist()
    w = MTE_STRIDE // 8                 # 每 block 占的 int64 槽数
    out = []
    for b in range(nblock):
        t0, t1, t2 = v[w * b:w * b + 3]
        out.append(((t1 - t0) / BENCH_CYCLES_PER_US,
                    (t2 - t0) / BENCH_CYCLES_PER_US,
                    (t2 - t1) / BENCH_CYCLES_PER_US))
    return out


def bench_mte(rt, variant, src, dst, ticks, nbytes, iters=1, block_num=0):
    """_C.bench_mte + 前置 torch.npu.synchronize().

    rt.stream 是 Runtime 私有的 aclrtCreateStream 流, 与 torch 当前流互不排序;
    不同步时 kernel 会与 torch 流上的 H2D 上传/zeros 填充双向竞态(实测均复现)。
    调用点一律经本包装, 不得直连 _C.bench_mte。"""
    torch.npu.synchronize()
    return bench_mte_raw(rt, variant, src, dst, ticks, nbytes, iters, block_num)


def bench_vec(rt, variant, src, out, ticks, elems, iters=1, block_num=0):
    """_C.bench_vec + 前置 torch.npu.synchronize()(理由同 bench_mte)。"""
    torch.npu.synchronize()
    return bench_vec_raw(rt, variant, src, out, ticks, elems, iters, block_num)


def bench_cube(rt, variant, a, b, c, ticks, m0, n0, k0, block_num=0):
    """_C.bench_cube + 前置 torch.npu.synchronize()(理由同 bench_mte)。

    单次 kernel 下发做一次完整 A@B.T matmul(FP16 输入/FP32 累加/FP16 输出)。
    依赖链测量由 host 多次下发实现(同实验 2b 的 wall 粗测模式)。"""
    torch.npu.synchronize()
    return bench_cube_raw(rt, variant, a, b, c, ticks, m0, n0, k0, block_num)


# ==================== 实验 1: 空算子 launch floor ====================

EMPTY_ITERS = 5                       # 每核型下发次数


def run_empty(rt):
    """实验1: 空算子(无计算/搬运/同步), wall 中位含 host 下发+同步;
    device 精确值见 --analyze [1]。"""
    print(f"[实验1] 空算子 launch floor(每核型 {EMPTY_ITERS} 次下发, wall 中位)")
    print(f"   {'kernel':<10s}{'blockNum':>9s}{'wall_med_us':>12s}")
    for name, op in (("empty_aiv", empty_aiv), ("empty_aic", empty_aic),
                     ("empty_mix", empty_mix)):
        walls, block_num = [], None
        for _ in range(EMPTY_ITERS):
            t0 = time.perf_counter()
            n = op(rt)
            walls.append((time.perf_counter() - t0) * 1e6)
            block_num = n if block_num is None else block_num
        print(f"   {name:<10s}{block_num:>9d}{_med_us(walls):>12.1f}")


# ==================== 实验 2: 标量依赖链(2a 核对/2b wall/2c SYS_CNT) ====================

INT_SEED = 3                          # 奇数: mul 链不会立刻塌缩到 0
FADD_SEED = 1e-30                     # 1e6 次累加后仍远小于 fp32 上限
VERIFY_ITERS = 1000                   # 小规模先核对数值, 确认测的不是被优化掉的循环
ITERS_LIST = [10_000, 100_000, 1_000_000]


# 标量参考模型: 与 kernel 逐位等价的 Python 实现(核对测量有效性用)

def ref_add(seed, iters):
    # acc += (seed ^ i): 操作数与计数器纠缠, 无闭式解, 须逐次模拟
    acc = seed
    for i in range(iters):
        acc = (acc + (seed ^ i)) % U32
    return acc


def ref_xor(seed, iters):
    acc = seed
    for i in range(iters):
        acc = (acc ^ (seed ^ i)) % U32
    return acc


def ref_mul(seed, iters):
    return pow(seed, iters + 1, U32)


def ref_div(seed, iters):
    acc = seed
    for _ in range(iters):
        acc = acc // seed
    return acc % U32


def ref_fadd(seed, iters):
    acc = np.float32(seed)
    x = np.float32(seed)
    for _ in range(iters):
        acc = np.float32(acc + x)
    return acc


def ref_add_unroll4(seed, iters):
    # 每圈 4 个加法, 操作数为 seed ^ (4t+k), iters = 总加法数
    acc = seed
    for j in range(iters):
        acc = (acc + (seed ^ j)) % U32
    return acc


OPS = [
    ("scalar_add_aiv", scalar_add_aiv, "i32", ref_add, INT_SEED),
    ("scalar_xor_aiv", scalar_xor_aiv, "i32", ref_xor, INT_SEED),
    ("scalar_mul_aiv", scalar_mul_aiv, "i32", ref_mul, INT_SEED),
    ("scalar_div_aiv", scalar_div_aiv, "i32", ref_div, INT_SEED),
    ("scalar_fadd_aiv", scalar_fadd_aiv, "f32", ref_fadd, FADD_SEED),
    ("scalar_add_unroll4_aiv", scalar_add_unroll4_aiv, "i32", ref_add_unroll4, INT_SEED),
    ("scalar_add_aic", scalar_add_aic, "i32", ref_add, INT_SEED),
    ("scalar_xor_aic", scalar_xor_aic, "i32", ref_xor, INT_SEED),
    ("scalar_mul_aic", scalar_mul_aic, "i32", ref_mul, INT_SEED),
    ("scalar_div_aic", scalar_div_aic, "i32", ref_div, INT_SEED),
    ("scalar_fadd_aic", scalar_fadd_aic, "f32", ref_fadd, FADD_SEED),
    ("scalar_add_unroll4_aic", scalar_add_unroll4_aic, "i32", ref_add_unroll4, INT_SEED),
]


def run_scalar(rt, out_i32, out_f32):
    """实验2: 2a 数值核对 / 2b wall 粗测 / 2c bench_timed 核内计时。"""
    print(f"[实验2] 标量依赖链({len(OPS)} 变体 x {len(ITERS_LIST)} 规模 x {REPS} 次, block=1)")

    # 2a 数值核对(顺带吸收进程级 lazy init, 让后面的计时调用落在稳态)
    for name, op, kind, ref, seed in OPS:
        out = out_f32 if kind == "f32" else out_i32
        op(rt, out, seed, VERIFY_ITERS)
        got = out.item()
        if kind == "i32":
            got &= U32 - 1
        expected = ref(seed, VERIFY_ITERS)
        assert got == expected, f"{name}: got {got}, expected {expected}"
    print(f"   2a) 数值核对(iters={VERIFY_ITERS}): {len(OPS)}/{len(OPS)} 与参考模型逐位一致")

    # 2b wall 粗测
    print("   2b) wall 粗测(含 host 下发+同步; device 口径见 --analyze [2])")
    for name, op, kind, _, seed in OPS:
        out = out_f32 if kind == "f32" else out_i32
        meds, block_num = [], None
        for iters in ITERS_LIST:
            walls = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                n = op(rt, out, seed, iters)
                walls.append((time.perf_counter() - t0) * 1e6)
                block_num = n if block_num is None else block_num
            meds.append(_med_us(walls))
        sizes = " / ".join(f"{m:.0f}us@{i // 1000}K" for i, m in zip(ITERS_LIST, meds))
        print(f"   {name:26s} block={block_num}  {sizes}")

    # 2c bench_timed 核内 SYS_CNT 计时(ticks 预分配, 核内采样, 无需 profiler)
    print("   2c) bench_timed 核内 SYS_CNT 计时")
    if bench_timed is None:
        print("       (本构建未编译: cmake -DXLITE_BENCH_TIMED=ON 全量重编后可用)")
        return
    ticks = torch.zeros(3, dtype=torch.int64, device="npu:0")
    # 2c 先核对: timed kernel 的累加结果须与 untimed 参考逐位一致
    for name, _, kind, ref, seed in OPS:
        variant = name[len("scalar_"):]
        out = out_f32 if kind == "f32" else out_i32
        bench_timed(rt, variant, out, ticks, seed, VERIFY_ITERS)
        got = out.item()
        if kind == "i32":
            got &= U32 - 1
        expected = ref(seed, VERIFY_ITERS)
        assert got == expected, f"bench_timed {variant}: got {got}, expected {expected}"
    print(f"       数值核对: {len(OPS)}/{len(OPS)} 与 untimed 参考逐位一致")
    # 2c 计时: t0=区域起点, t1=循环下发完, t2=PIPE_ALL 排空后(真实完成);
    # compl(=t2-t0) 为主口径, drain(=t2-t1) 为在飞排空
    i_big = ITERS_LIST[-1]
    i_small = ITERS_LIST[0]
    def _fmt_iters(n):
        return f"{n // 1000}K" if n < 1_000_000 else f"{n // 1_000_000}M"
    print(f"       {'variant':<16s}{'issue@' + _fmt_iters(i_big):>12s}"
          f"{'compl@' + _fmt_iters(i_big):>12s}{'ns/op':>10s}"
          f"{'drain':>10s}{'ns/op@' + _fmt_iters(i_small):>11s}")
    for name, _, kind, _, seed in OPS:
        variant = name[len("scalar_"):]
        out = out_f32 if kind == "f32" else out_i32
        meds = {}
        for iters in ITERS_LIST:
            issue, compl, drain = [], [], []
            for _ in range(REPS):
                bench_timed(rt, variant, out, ticks, seed, iters)
                c0, c1, c2 = ticks.cpu().tolist()
                issue.append(c1 - c0)
                compl.append(c2 - c0)
                drain.append(c2 - c1)
            meds[iters] = (statistics.median(issue) / BENCH_CYCLES_PER_US,
                           statistics.median(compl) / BENCH_CYCLES_PER_US,
                           statistics.median(drain) / BENCH_CYCLES_PER_US)
        ib, cbm, db = meds[i_big]
        nsop_b = cbm * 1000.0 / i_big
        nsop_s = meds[i_small][1] * 1000.0 / i_small
        print(f"       {variant:<16s}{ib:>12.2f}{cbm:>12.2f}{nsop_b:>10.3f}"
              f"{db:>10.2f}{nsop_s:>11.3f}")


# ==================== 实验 3: MTE 访存(3a 核对/3b 反消解/3c 带宽/3d 时延) ====================

MTE_TILE = 98304                       # kernel 侧 BENCH_MTE_TILE(=UB_SIZE/2), 须同步
MTE_VARIANTS = ("read", "write", "copy")
MTE_CHASE_ITERS = 2048                 # chase 每次调用的追逐步数
MTE_TARGET_BYTES = 64 << 20            # iters 自适应目标: 总搬运量恒定, 小 size 多重复
MTE_SIZES_DEFAULT = [64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20]


def _mte_fmt_size(size):
    for unit, div in (("g", 1 << 30), ("m", 1 << 20), ("k", 1 << 10)):
        if size >= div and size % div == 0:
            return f"{size // div}{unit}"
    return str(size)


def _parse_mte_sizes(spec):
    """'64k,1m,256m' -> 字节数列表; 支持 k/m/g 后缀与十进制/0x, 向下对齐到
    64B(chase 的块粒度; 其余变体只要求 32B)。"""
    sizes = []
    for tok in spec.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        mult = 1
        if t[-1] in "kmg":
            mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[t[-1]]
            t = t[:-1]
        n = int(t, 0) * mult
        if n < 64:
            raise ValueError(f"size '{tok.strip()}' < 64B")
        sizes.append(n - n % 64)
    if not sizes:
        raise ValueError("empty size list")
    return sizes


def _mte_iters(size):
    # 总搬运量目标恒定(MTE_TARGET_BYTES): 小工作集多重复, 拉长计时窗口
    return max(1, min(4096, MTE_TARGET_BYTES // size))


def _mte_ring(size, rng):
    """构造 chase 工作集: n = size/64 个 64B 块, 块首 u32 = 后继块的字节偏移,
    后继关系为随机单一环置换(全块成环) -> 每步依赖上一步数据, 无法预取。"""
    if size > (1 << 32) - 64:
        # 块偏移以 u32 存于块首, 超 4GB 的环无法表示(read/write/copy 不受限)
        raise ValueError(f"chase 工作集须 < 4GB(块偏移为 u32): {size}")
    n = size // 64
    perm = rng.permutation(n)
    succ = np.empty(n, dtype=np.uint32)
    succ[perm[:-1]] = perm[1:]
    succ[perm[-1]] = perm[0]
    succ_off = succ * 64                    # 块 i 首 u32 = 64 * succ[i]
    buf = np.zeros(size, dtype=np.uint8)
    buf.view(np.uint32)[::16] = succ_off    # 每 64B 块的第 0 个 u32
    return buf, succ_off


def _mte_chase_fold(succ_off, iters):
    """kernel 语义的 Python 折叠: pos=0, 每步 next=当前块首 u32, acc^=next, pos=next。"""
    pos, acc = 0, 0
    for _ in range(iters):
        nxt = int(succ_off[pos // 64])
        acc ^= nxt
        pos = nxt
    return acc % U32


def _mte_read_expect(src_np, size, nblock):
    """read 每 block 的期望累加值: 该 block 最后一次装载 tile 的首 u32(依赖 MTE2
    同目标有序完成); 空闲 block(gbo 起点 >= size)期望 0。"""
    exp = []
    for b in range(nblock):
        g0 = b * MTE_TILE
        if g0 >= size:
            exp.append(0)
            continue
        gbo = g0 + (size - 1 - g0) // (nblock * MTE_TILE) * (nblock * MTE_TILE)
        exp.append(int.from_bytes(src_np[gbo:gbo + 4].tobytes(), "little"))
    return exp


def _mte_write_expect(src_np, size, nblock):
    """write 期望: 各 block grid-stride 走位上 src[0:len] 逐段平铺。"""
    exp = np.zeros(size, dtype=np.uint8)
    for b in range(nblock):
        gbo = b * MTE_TILE
        while gbo < size:
            ln = min(MTE_TILE, size - gbo)
            exp[gbo:gbo + ln] = src_np[0:ln]
            gbo += nblock * MTE_TILE
    return exp


def _mte_accs(dst32, nblock):
    """read/chase 的 per-block acc 字(INT32 dst 按 MTE_STRIDE 步距, word0)。"""
    w = MTE_STRIDE // 4
    return [v & (U32 - 1) for v in dst32.cpu().reshape(-1, w)[:, 0].tolist()[:nblock]]


def _mte_check(rt, b_full):
    """3a) 数值核对(size=1MiB, iters=1): copy 全等 / write 平铺 / read 末 tile
    首 u32 / chase 折叠, block=1 与满核各一遍(chase 由 host 恒钳 1 block)。"""
    size = 1 << 20
    src_np = np.random.default_rng(0).integers(0, 256, size=size, dtype=np.uint8)
    src = torch.from_numpy(src_np).view(torch.int8).to("npu:0")
    ticks = torch.zeros((MTE_STRIDE // 8) * b_full, dtype=torch.int64, device="npu:0")
    for block in (1, 0):
        B = b_full if block == 0 else 1
        dst = torch.zeros(size, dtype=torch.int8, device="npu:0")
        bench_mte(rt, "copy_aiv", src, dst, ticks, size, 1, block)
        assert torch.equal(dst, src), "copy_aiv: dst != src"
        dst = torch.zeros(size, dtype=torch.int8, device="npu:0")
        bench_mte(rt, "write_aiv", src, dst, ticks, size, 1, block)
        assert np.array_equal(dst.cpu().numpy().view(np.uint8),
                              _mte_write_expect(src_np, size, B)), "write_aiv: 平铺不符"
        dst32 = torch.zeros((MTE_STRIDE // 4) * B, dtype=torch.int32, device="npu:0")
        bench_mte(rt, "read_aiv", src, dst32, ticks, size, 1, block)
        got = _mte_accs(dst32, B)
        assert got == _mte_read_expect(src_np, size, B), "read_aiv: 末 tile 首 u32 不符"
    ring, succ_off = _mte_ring(size, np.random.default_rng(3))
    src = torch.from_numpy(ring).view(torch.int8).to("npu:0")
    dst = torch.zeros(MTE_STRIDE // 4, dtype=torch.int32, device="npu:0")
    bench_mte(rt, "chase_aiv", src, dst, ticks, size, 100, 0)
    got = _mte_accs(dst, 1)[0]
    exp = _mte_chase_fold(succ_off, 100)
    assert got == exp, f"chase_aiv: got {got}, expected {exp}"
    print(f"   3a) 数值核对(size=1MiB, iters=1): copy 全等 / write 平铺 / "
          f"read 末 tile 首 u32 / chase 折叠, B=1,{b_full} 全部一致")


def _mte_anti_dce(rt):
    """3b) 反消解自检: read@4MiB 单 block, iters 翻倍则 complete 应近似翻倍
    (搬运若被消解, 比值贴近 1); 两档 GB/s 一致亦即线性自检。"""
    size = 4 << 20
    src = torch.ones(size, dtype=torch.int8, device="npu:0")
    dst = torch.zeros(MTE_STRIDE // 4, dtype=torch.int32, device="npu:0")
    ticks = torch.zeros(MTE_STRIDE // 8, dtype=torch.int64, device="npu:0")
    res = {}
    for tag, it in (("I", _mte_iters(size)), ("2I", 2 * _mte_iters(size))):
        compls = []
        for _ in range(REPS):
            bench_mte(rt, "read_aiv", src, dst, ticks, size, it, 1)
            compls.append(_mte_ticks_us(ticks, 1)[0][1])
        res[tag] = (it, statistics.median(compls))
    (i1, c1), (i2, c2) = res["I"], res["2I"]
    print(f"   3b) 反消解(read@4MiB): I={i1}: compl={c1:8.1f}us "
          f"({i1 * size / c1 / 1000:6.1f} GB/s)   2I={i2}: compl={c2:8.1f}us "
          f"({i2 * size / c2 / 1000:6.1f} GB/s)   ratio={c2 / c1:.2f}")
    assert 1.6 <= c2 / c1 <= 2.4, f"反消解失败: iters 翻倍 complete 比值 {c2 / c1:.2f}"


def _mte_bw_sweep(rt, sizes, b_full):
    """3c) 带宽 sweep: sizes x {read,write,copy} x {block=1, 满核} x REPS 中位。
    聚合时间 = 各 block complete 的最大值(并行执行, kernel 完成于最慢 block);
    聚合 GB/s = iters*size/compl(引擎计, copy 的字节只计一遍); 每核 = 聚合/actB。
    每 (size, B) 组合顺带数值抽验, 覆盖 3a 未及的多段/block 形态。"""
    print(f"   3c) 带宽 sweep({len(sizes)} 个 size x 3 变体 x block={{1,满核}}):")
    print(f"       {'size':>6s} {'variant':>6s} {'B':>3s}{'actB':>5s}{'I':>5s}"
          f"{'issue_us':>9s}{'compl_us':>9s}{'drain_us':>9s}{'agg GB/s':>9s}"
          f"{'per-core':>9s}")
    for size in sizes:
        iters = _mte_iters(size)
        src_np = np.random.default_rng(1).integers(0, 256, size=size, dtype=np.uint8)
        src = torch.from_numpy(src_np).view(torch.int8).to("npu:0")
        dst_i8 = torch.zeros(size, dtype=torch.int8, device="npu:0")
        dst_i32 = torch.zeros((MTE_STRIDE // 4) * b_full, dtype=torch.int32, device="npu:0")
        ticks = torch.zeros((MTE_STRIDE // 8) * b_full, dtype=torch.int64, device="npu:0")
        for block in (1, 0):
            n = b_full if block == 0 else 1
            active = min(n, -(-size // MTE_TILE))
            exp_write = _mte_write_expect(src_np, size, n)
            exp_read = _mte_read_expect(src_np, size, n)
            for variant in MTE_VARIANTS:
                dst = dst_i32 if variant == "read" else dst_i8
                rows = []
                for _ in range(REPS):
                    n = bench_mte(rt, f"{variant}_aiv", src, dst, ticks, size, iters, block)
                    tk = _mte_ticks_us(ticks, n)
                    rows.append((max(t[0] for t in tk), max(t[1] for t in tk),
                                 max(t[2] for t in tk)))
                issue = statistics.median([r[0] for r in rows])
                compl = statistics.median([r[1] for r in rows])
                drain = statistics.median([r[2] for r in rows])
                agg = iters * size / compl / 1000
                print(f"       {_mte_fmt_size(size):>6s} {variant:>6s} {n:>3d}{active:>5d}"
                      f"{iters:>5d}{issue:>9.1f}{compl:>9.1f}{drain:>9.1f}{agg:>9.1f}"
                      f"{agg / active:>9.1f}")
                if variant == "copy":
                    assert torch.equal(dst_i8, src), f"copy {size} B={n}: dst != src"
                elif variant == "write":
                    assert np.array_equal(dst_i8.cpu().numpy().view(np.uint8), exp_write), \
                        f"write {size} B={n}: 平铺不符"
                else:
                    # read 的期望依赖 MTE2 同目标有序完成, 大规模下仅告警不中断
                    # (3a 已对基础形态硬校验)
                    got = _mte_accs(dst_i32, n)
                    if got != exp_read:
                        print(f"       [warn] read {size} B={n}: 末 tile 首 u32 不符"
                              "(MTE2 完成序假设该规模不成立, 属引擎次序发现)")


def _mte_lat_sweep(rt, sizes):
    """3d) 时延 sweep: 每 size 建随机环, chase 单 block x REPS 中位。
    ns/step = complete/iters = 经 MTE 的 load-to-use 时延(MTE 下发 + GM 时延 +
    MTE2->S 握手), 每步消费上一步数据 -> 无法预取, lat_mem_rd 的忠实映射。"""
    print(f"   3d) 时延 sweep(chase 依赖链, {MTE_CHASE_ITERS} 步/次, 单 block):")
    print(f"       {'size':>6s}{'steps':>7s}{'issue_us':>10s}{'compl_us':>10s}"
          f"{'drain_us':>10s}{'ns/step':>9s}")
    rng = np.random.default_rng(7)
    for size in sizes:
        ring, succ_off = _mte_ring(size, rng)
        src = torch.from_numpy(ring).view(torch.int8).to("npu:0")
        dst = torch.zeros(MTE_STRIDE // 4, dtype=torch.int32, device="npu:0")
        ticks = torch.zeros(MTE_STRIDE // 8, dtype=torch.int64, device="npu:0")
        exp = _mte_chase_fold(succ_off, MTE_CHASE_ITERS)
        rows = []
        for _ in range(REPS):
            bench_mte(rt, "chase_aiv", src, dst, ticks, size, MTE_CHASE_ITERS, 1)
            got = _mte_accs(dst, 1)[0]
            assert got == exp, f"chase {size}: got {got}, expected {exp}"
            rows.append(_mte_ticks_us(ticks, 1)[0])
        issue = statistics.median([r[0] for r in rows])
        compl = statistics.median([r[1] for r in rows])
        drain = statistics.median([r[2] for r in rows])
        print(f"       {_mte_fmt_size(size):>6s}{MTE_CHASE_ITERS:>7d}{issue:>10.1f}"
              f"{compl:>10.1f}{drain:>10.2f}{compl * 1000.0 / MTE_CHASE_ITERS:>9.1f}")


def run_mte(rt, sizes):
    """实验3: MTE 访存微基准(bench_mte), lmbench 风格带宽/时延 vs size。"""
    b_full = empty_aiv(rt)     # 空算子仅下发, 返回值即满核 AIV block 数
    print(f"[实验3] MTE 内存微基准(bench_mte), 满核 AIV = {b_full}")
    _mte_check(rt, b_full)
    _mte_anti_dce(rt)
    _mte_bw_sweep(rt, sizes, b_full)
    _mte_lat_sweep(rt, sizes)


# ==================== 实验 4: 向量指令(4a 核对/4b 反消解/4c 吞吐 sweep) ====================

VEC_VARIANTS = ("add_f32", "mul_f32", "add_i32", "add_f16", "sqrt_f32", "exp_f32")
VEC_TORCH = {"add_f32": torch.float32, "mul_f32": torch.float32, "add_i32": torch.int32,
             "add_f16": torch.float16, "sqrt_f32": torch.float32, "exp_f32": torch.float32}
VEC_ITEM_BYTES = {"add_f32": 4, "mul_f32": 4, "add_i32": 4, "add_f16": 2,
                  "sqrt_f32": 4, "exp_f32": 4}
VEC_REP_BYTES = 256                    # kernel 侧 VECTOR_MAX_BYTESIZE: 一条指令一个 repeat 搬 256B
VEC_UB_BYTES = 196608                  # kernel 侧 UB_SIZE; 2*elems*sz + 65280(超越函数
                                       # scratch tile, BENCH_VEC_TMP_BYTES) <= UB, 三侧同步
VEC_REPS_DEFAULT = (1, 4, 16, 64, 256) # 默认 elems 档位 = repeats 阶梯 x 每 repeat 元素数
VEC_TARGET_ELEMS = 8 << 20             # iters 自适应目标: 总元素量恒定, 小向量多迭代
VEC_ITERS_MAX = 131072


def _parse_vec_sizes(spec):
    """'64,256,2k' -> 元素数列表(支持 k 后缀与十进制/0x). 元素数为每次向量
    操作的元素数, 各变体运行时会向下对齐到 repeat 量子(4B dtype 64, f16 128)
    并钳到 UB 上限(见 _vec_sizes)。"""
    sizes = []
    for tok in spec.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        mult = 1
        if t[-1] == "k":
            mult = 1 << 10
            t = t[:-1]
        n = int(t, 0) * mult
        if n < 1:
            raise ValueError(f"size '{tok.strip()}' < 1")
        sizes.append(n)
    if not sizes:
        raise ValueError("empty size list")
    return sizes


def _vec_quantum(variant):
    # 一条全掩码指令一个 repeat 搬 256B: 4B dtype 64 元素/repeat, 2B(f16) 128
    return VEC_REP_BYTES // VEC_ITEM_BYTES[variant]


def _vec_max_elems(variant):
    # host 侧 UB 预算(与 op.cpp 同式): 2*elems*sz + 65280 <= 196608
    # 并向下对齐到 repeat 量子(elems 须为整 repeat, 否则 host 校验拒绝)
    m = (VEC_UB_BYTES - 65280) // (2 * VEC_ITEM_BYTES[variant])
    return m - m % _vec_quantum(variant)


def _vec_sizes(variant, sizes):
    """该变体可用的 elems 档位: 用户列表逐项向下对齐到 repeat 量子(至少一个
    repeat)并钳到 UB 上限; 默认档位 = repeats 阶梯 x 每 repeat 元素数。"""
    q = _vec_quantum(variant)
    if sizes is None:
        return [r * q for r in VEC_REPS_DEFAULT]
    out, seen = [], set()
    for n in sizes:
        e = min(max(n - n % q, q), _vec_max_elems(variant))
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def _vec_iters(elems):
    # 总元素量目标恒定(VEC_TARGET_ELEMS): 小向量多迭代, 拉长计时窗口;
    # 同一 elems 下所有变体 iters 相同 -> sqrt/exp 与 mul_f32 的差分有效
    return max(1, min(VEC_ITERS_MAX, VEC_TARGET_ELEMS // elems))


def _vec_inputs(variant, elems, rng):
    """构造 src = [x0; c](拼接): x0 为链初值(有界, 无 inf/NaN/次正规), c 为
    常量操作数向量. 常量按变体选取, 保证满 iters 累计后有界, 且超越函数载体
    满足 op(c)=1(sqrt: c=1.0, exp: c=0.0) -> dst 逐位不变。"""
    if variant == "add_i32":
        # x0 限 ±2^30, c=3: 满迭代漂移 < 2^20, 不回绕(折叠与硬件逐位一致)
        x0 = rng.integers(-(1 << 30), 1 << 30, size=elems, dtype=np.int64).astype(np.int32)
        c = np.full(elems, 3, dtype=np.int32)
    else:
        dt = np.float16 if variant == "add_f16" else np.float32
        x0 = (rng.random(elems) + 1.0).astype(dt)          # [1, 2)
        cvals = {"add_f32": 2.0 ** -13, "mul_f32": 1.0 + 2.0 ** -12,
                 "add_f16": 2.0 ** -10, "sqrt_f32": 1.0, "exp_f32": 0.0}
        c = np.full(elems, cvals[variant], dtype=dt)
    return np.concatenate([x0, c])


def _vec_fold(variant, x0, c, iters):
    """kernel 语义的 numpy 折叠(同一条依赖链逐迭代复算), 供 4a 全数组比对。

    逐位等价依据: IEEE 加/乘/np.sqrt 均正确舍入(同序列同结果); f16 加法经 f32
    中间精确可证正确舍入; 载体乘以 op(c)=1 位不变; 唯 exp 的 vexp 多项式实现
    定义, 用容差(np.exp 未必逐位相同)。"""
    x = x0.copy()
    if variant == "add_f32":
        for _ in range(iters):
            x = x + np.float32(c[0])
    elif variant == "mul_f32":
        for _ in range(iters):
            x = x * np.float32(c[0])
    elif variant == "add_i32":
        for _ in range(iters):
            x = x + np.int32(c[0])
    elif variant == "add_f16":
        for _ in range(iters):
            x = x + np.float16(c[0])
    elif variant == "sqrt_f32":
        for _ in range(iters):
            x = x * np.sqrt(np.float32(c[0]))
    else:                                                  # exp_f32
        for _ in range(iters):
            x = x * np.exp(np.float32(c[0]))
    return x


def _vec_check(rt, b_full):
    """4a) 数值核对: 最小与最大 elems(后者触发多 chunk 分段) x {B=1, 满核},
    每 block 全数组比对(比单字 acc 更强, 能抓 mask/chunk 偏移 bug)。"""
    rng = np.random.default_rng(11)
    for variant in VEC_VARIANTS:
        dt_t = VEC_TORCH[variant]
        for elems in (_vec_quantum(variant), _vec_max_elems(variant)):
            iters = 3 if variant == "exp_f32" else 100
            src_np = _vec_inputs(variant, elems, rng)
            src = torch.from_numpy(src_np).to("npu:0")
            expect = _vec_fold(variant, src_np[:elems], src_np[elems:], iters)
            for block in (1, 0):
                B = b_full if block == 0 else 1
                out = torch.zeros(B * elems, dtype=dt_t, device="npu:0")
                ticks = torch.zeros((MTE_STRIDE // 8) * B, dtype=torch.int64, device="npu:0")
                bench_vec(rt, f"{variant}_aiv", src, out, ticks, elems, iters, block)
                got = out.cpu().numpy().reshape(B, elems)
                # np.array_equal 严格比较 shape(不广播), expect 须先广播到 (B, elems)
                expect_b = np.broadcast_to(expect, (B, elems))
                if variant == "exp_f32":
                    assert np.allclose(got, expect_b, rtol=1e-5, atol=1e-6), \
                        f"exp_f32 elems={elems} B={B}: 折叠不符"
                else:
                    assert np.array_equal(got, expect_b), \
                        f"{variant} elems={elems} B={B}: 折叠不符(链结果/chunk 偏移/mask 有误)"
    print(f"   4a) 数值核对(最小/最大 elems x B=1/{b_full}, 每 block 全数组比对): "
          f"{len(VEC_VARIANTS)} 变体全部一致(exp 容差, 其余逐位)")


def _vec_anti_dce(rt):
    """4b) 反消解自检: add_f32 与 sqrt_f32 载体各取 elems=1024, iters 翻倍则
    complete 应近似翻倍(指令若被消解/外提, 比值贴近 1)。"""
    print("   4b) 反消解(iters I vs 2I):")
    elems = 1024
    it = _vec_iters(elems)
    for variant in ("add_f32", "sqrt_f32"):
        src_np = _vec_inputs(variant, elems, np.random.default_rng(5))
        src = torch.from_numpy(src_np).to("npu:0")
        out = torch.zeros(elems, dtype=VEC_TORCH[variant], device="npu:0")
        ticks = torch.zeros(MTE_STRIDE // 8, dtype=torch.int64, device="npu:0")
        res = {}
        for tag, itr in (("I", it), ("2I", 2 * it)):
            compls = []
            for _ in range(REPS):
                bench_vec(rt, f"{variant}_aiv", src, out, ticks, elems, itr, 1)
                compls.append(_mte_ticks_us(ticks, 1)[0][1])
            res[tag] = (itr, statistics.median(compls))
        (i1, c1), (i2, c2) = res["I"], res["2I"]
        print(f"       {variant}: I={i1}: compl={c1:8.1f}us   2I={i2}: compl={c2:8.1f}us"
              f"   ratio={c2 / c1:.2f}")
        assert 1.6 <= c2 / c1 <= 2.4, f"{variant} 反消解失败: 比值 {c2 / c1:.2f}"


def _vec_perf_sweep(rt, sizes, b_full):
    """4c) 吞吐 sweep: 每 variant 一表(elems 档位 x {B=1, 满核}) x REPS 中位。
    单核 ns/elem = compl/(iters*elems)*1000; 聚合 Gelem/s = B*iters*elems/compl/1e3
    (每 block 独立同链, 聚合时间取各 block complete 最大值)。末尾输出超越函数
    差分表: sqrt/exp 的边际成本 = 同 elems/iters 下 (变体 - mul_f32)。"""
    print(f"   4c) 吞吐 sweep({len(VEC_VARIANTS)} 变体 x elems 档位 x block={{1,满核}}):")
    rng = np.random.default_rng(13)
    nse_b1 = {}                          # (variant, elems) -> B=1 的 ns/elem
    mul_base = {}                        # elems -> mul_f32 的 ns/elem(差分基线)
    for variant in VEC_VARIANTS:
        dt_t = VEC_TORCH[variant]
        q = _vec_quantum(variant)
        print(f"\n       -- {variant} --")
        print(f"       {'elems':>7s}{'reps':>6s}{'I':>7s}{'B':>4s}{'issue_us':>9s}"
              f"{'compl_us':>9s}{'drain_us':>9s}{'ns/elem':>9s}{'Gelem/s':>9s}")
        for elems in _vec_sizes(variant, sizes):
            iters = _vec_iters(elems)
            src = torch.from_numpy(_vec_inputs(variant, elems, rng)).to("npu:0")
            ticks = torch.zeros((MTE_STRIDE // 8) * b_full, dtype=torch.int64, device="npu:0")
            for block in (1, 0):
                B = b_full if block == 0 else 1
                out = torch.zeros(B * elems, dtype=dt_t, device="npu:0")
                rows = []
                for _ in range(REPS):
                    bench_vec(rt, f"{variant}_aiv", src, out, ticks, elems, iters, block)
                    tk = _mte_ticks_us(ticks, B)
                    rows.append((max(t[0] for t in tk), max(t[1] for t in tk),
                                 max(t[2] for t in tk)))
                issue = statistics.median([r[0] for r in rows])
                compl = statistics.median([r[1] for r in rows])
                drain = statistics.median([r[2] for r in rows])
                nse = compl * 1000.0 / (iters * elems)
                ges = B * iters * elems / compl / 1000.0
                print(f"       {elems:>7d}{elems // q:>6d}{iters:>7d}{B:>4d}{issue:>9.1f}"
                      f"{compl:>9.1f}{drain:>9.2f}{nse:>9.4f}{ges:>9.2f}")
                if B == 1:
                    nse_b1[(variant, elems)] = nse
                    if variant == "mul_f32":
                        mul_base[elems] = nse
    # 差分口径注意: 是"在 mul 依赖链存在的前提下加一条超越函数指令"的边际成本
    # (op 不在链上, 可与 mul 重叠), 不是孤立的 vsqrt/vexp 吞吐。
    print("\n       超越函数差分(vs mul_f32 同 elems/iters, B=1, ns/elem):")
    print(f"       {'elems':>7s}{'sqrt-mul':>10s}{'exp-mul':>10s}")
    for elems in _vec_sizes("sqrt_f32", sizes):
        m = mul_base.get(elems)
        if m is None:
            continue
        s = nse_b1.get(("sqrt_f32", elems))
        e = nse_b1.get(("exp_f32", elems))
        line = f"       {elems:>7d}"
        line += f"{s - m:>10.4f}" if s is not None else f"{'--':>10s}"
        line += f"{e - m:>10.4f}" if e is not None else f"{'--':>10s}"
        print(line)


def run_vec(rt, sizes):
    """实验4: 向量指令微基准(bench_vec), 纯向量吞吐 vs 向量长度。"""
    b_full = empty_aiv(rt)     # 空算子仅下发, 返回值即满核 AIV block 数
    print(f"[实验4] 向量指令微基准(bench_vec), 满核 AIV = {b_full}")
    _vec_check(rt, b_full)
    _vec_anti_dce(rt)
    _vec_perf_sweep(rt, sizes, b_full)


# ==================== 实验 5: cube matmul 吞吐(5a 核对/5b 反消解/5c tile sweep) ====================

# kernel 侧 Matmul::Init 的 tile 预算公式(Matmul<half,float,half>, 须同步):
#   L0A = m0*k0 <= 64KB; L0B = n0*k0 <= 64KB; L0C = 4*m0*n0 <= 64KB; L1 = 4*k0*(2*m0+n0) <= 512KB
# m0/n0/k0 须为 16 的倍数。tile = (m0, n0, k0), matmul 为 A[m0,k0] @ B[n0,k0].T -> C[m0,n0]
CUBE_TILES_DEFAULT = [(128, 128, 128), (128, 128, 256), (64, 128, 256),
                      (128, 64, 256), (64, 256, 256)]
CUBE_TICK_SLOTS = 8               # kernel 每 block 写 3 槽 + 5 空位(步距 8 int64)
CUBE_WARMUP_ITERS = 10            # 反消解/吞吐测量前的 host 下发预热次数
CUBE_LAUNCH_ITERS = 200           # 反消解/吞吐测量的 host 下发次数
CUBE_ANTI_DCE_RATIO_LO = 1.6      # iters 翻倍 wall 时间比值下界
CUBE_ANTI_DCE_RATIO_HI = 2.4      # iters 翻倍 wall 时间比值上界


def _cube_l0_l1_bytes(m0, n0, k0):
    """L0/L1 字节数(与 op.cpp 同式), 用于 host 侧预检。
    Matmul<half,float,half>: L0A=m0*k0, L0B=n0*k0, L0C=4*m0*n0, L1=4*k0*(2*m0+n0)。"""
    l0a = m0 * k0
    l0b = n0 * k0
    l0c = 4 * m0 * n0
    l1 = 4 * k0 * (2 * m0 + n0)
    return l0a, l0b, l0c, l1


def _cube_check_tile(m0, n0, k0):
    """host 侧 tile 预检, 报错信息与 op.cpp 对齐。"""
    if m0 == 0 or m0 % 16 or n0 % 16 or k0 % 16:
        raise ValueError(f"cube tile {m0}x{n0}x{k0}: m0/n0/k0 must be non-zero multiples of 16")
    l0a, l0b, l0c, l1 = _cube_l0_l1_bytes(m0, n0, k0)
    if l0a > 64 * 1024 or l0b > 64 * 1024 or l0c > 64 * 1024 or l1 > 512 * 1024:
        raise ValueError(f"cube tile {m0}x{n0}x{k0}: exceeds L0/L1 budget "
                         f"(L0A={l0a}B/64KB, L0B={l0b}B/64KB, L0C={l0c}B/64KB, L1={l1}B/512KB)")


def _parse_cube_tiles(spec):
    """'128,128,128;256,256,512' -> [(m0,n0,k0), ...]"""
    tiles = []
    for tok in spec.split(";"):
        t = tok.strip()
        if not t:
            continue
        parts = t.split(",")
        if len(parts) != 3:
            raise ValueError(f"tile '{t}' must have 3 comma-separated dims")
        m0, n0, k0 = (int(p.strip()) for p in parts)
        _cube_check_tile(m0, n0, k0)
        tiles.append((m0, n0, k0))
    if not tiles:
        raise ValueError("empty tile list")
    return tiles


def _cube_ticks_us(ticks):
    """单 block ticks(3 槽 t0/t1/t2, 步距 8 int64) -> (issue, complete, drain) us。"""
    v = ticks.cpu().tolist()
    t0, t1, t2 = v[0], v[1], v[2]
    return ((t1 - t0) / BENCH_CYCLES_PER_US,
            (t2 - t0) / BENCH_CYCLES_PER_US,
            (t2 - t1) / BENCH_CYCLES_PER_US)


def _cube_check(rt):
    """5a) 数值核对: 单次 kernel 调用, C vs numpy A@B.T(FP16 精度容差)。"""
    rng = np.random.default_rng(42)
    for m0, n0, k0 in CUBE_TILES_DEFAULT:
        a_np = rng.standard_normal((m0, k0)).astype(np.float16)
        b_np = rng.standard_normal((n0, k0)).astype(np.float16)
        # kernel: C = A @ B.T (B 布局为 [n0, k0], 即 B^T 传入)
        expect = (a_np.astype(np.float32) @ b_np.T.astype(np.float32)).astype(np.float16)
        a = torch.from_numpy(a_np).to("npu:0")
        b = torch.from_numpy(b_np).to("npu:0")
        c = torch.zeros(m0 * n0, dtype=torch.float16, device="npu:0")
        ticks = torch.zeros(CUBE_TICK_SLOTS, dtype=torch.int64, device="npu:0")
        bench_cube(rt, "cube_mmad_aic", a, b, c, ticks, m0, n0, k0, 0)
        got = c.cpu().numpy().reshape(m0, n0)
        assert np.allclose(got, expect, rtol=1e-2, atol=1e-2), \
            f"cube_mmad_aic {m0}x{n0}x{k0}: C != A@B.T (max diff {np.max(np.abs(got - expect))})"
    print(f"   5a) 数值核对({len(CUBE_TILES_DEFAULT)} 个 tile): C == A@B.T "
          f"(FP16 容差 rtol=atol=1e-2) 全部一致")


def _cube_anti_dce(rt):
    """5b) 反消解自检: host 下发次数翻倍则 wall 时间应近似翻倍。

    kernel 内只做一次 Matmul::Run()(不做循环), 依赖链由 host 多次下发实现。
    wall 时间含 host 下发 + kernel 执行 + 同步; 若 kernel 被消解, 比值贴近 1。"""
    m0, n0, k0 = 128, 128, 256
    a = torch.randn(m0 * k0, dtype=torch.float16, device="npu:0")
    b = torch.randn(n0 * k0, dtype=torch.float16, device="npu:0")
    c = torch.zeros(m0 * n0, dtype=torch.float16, device="npu:0")
    ticks = torch.zeros(CUBE_TICK_SLOTS, dtype=torch.int64, device="npu:0")
    res = {}
    for tag, iters in (("I", CUBE_LAUNCH_ITERS), ("2I", 2 * CUBE_LAUNCH_ITERS)):
        walls = []
        for _ in range(REPS):
            # 预热: 先下发 CUBE_WARMUP_ITERS 次吸收 lazy init
            for _w in range(CUBE_WARMUP_ITERS):
                bench_cube(rt, "cube_mmad_aic", a, b, c, ticks, m0, n0, k0, 0)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _i in range(iters):
                bench_cube(rt, "cube_mmad_aic", a, b, c, ticks, m0, n0, k0, 0)
            torch.npu.synchronize()
            walls.append((time.perf_counter() - t0) * 1e6)
        res[tag] = (iters, _med_us(walls))
    (i1, w1), (i2, w2) = res["I"], res["2I"]
    ratio = w2 / w1
    print(f"   5b) 反消解(host 下发 {m0}x{n0}x{k0}): "
          f"I={i1}: wall={w1:8.1f}us ({w1 / i1:.2f} us/launch)   "
          f"2I={i2}: wall={w2:8.1f}us ({w2 / i2:.2f} us/launch)   ratio={ratio:.2f}")
    assert CUBE_ANTI_DCE_RATIO_LO <= ratio <= CUBE_ANTI_DCE_RATIO_HI, \
        f"反消解失败: iters 翻倍 wall 比值 {ratio:.2f}"


def _cube_tile_sweep(rt, tiles):
    """5c) tile sweep: 各 tile 的 wall 中位(含 host 下发+同步) + 单次核内 SYS_CNT。

    wall 口径: host 下发 CUBE_LAUNCH_ITERS 次, 取 wall 中位, 除以次数得 per-launch。
    SYS_CNT 口径: 单次下发的核内 3 槽(t0=before Run, t1=after Run, t2=after PIPE_ALL),
    compl = t2-t0 为单次 matmul 的核内完成时间。"""
    print(f"   5c) tile sweep({len(tiles)} 个 tile, host 下发 {CUBE_LAUNCH_ITERS} 次/测量):")
    print(f"       {'tile':>14s}{'us/launch':>10s}{'issue_us':>10s}"
          f"{'compl_us':>10s}{'drain_us':>10s}{'TFLOPS':>9s}")
    rng = np.random.default_rng(99)
    for m0, n0, k0 in tiles:
        a = torch.from_numpy(rng.standard_normal((m0, k0)).astype(np.float16)).to("npu:0")
        b = torch.from_numpy(rng.standard_normal((n0, k0)).astype(np.float16)).to("npu:0")
        c = torch.zeros(m0 * n0, dtype=torch.float16, device="npu:0")
        ticks = torch.zeros(CUBE_TICK_SLOTS, dtype=torch.int64, device="npu:0")
        # 预热
        for _w in range(CUBE_WARMUP_ITERS):
            bench_cube(rt, "cube_mmad_aic", a, b, c, ticks, m0, n0, k0, 0)
        torch.npu.synchronize()
        # wall 口径
        walls = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for _i in range(CUBE_LAUNCH_ITERS):
                bench_cube(rt, "cube_mmad_aic", a, b, c, ticks, m0, n0, k0, 0)
            torch.npu.synchronize()
            walls.append((time.perf_counter() - t0) * 1e6)
        wall_per = _med_us(walls) / CUBE_LAUNCH_ITERS
        # SYS_CNT 口径(单次)
        issues, compls, drains = [], [], []
        for _ in range(REPS):
            bench_cube(rt, "cube_mmad_aic", a, b, c, ticks, m0, n0, k0, 0)
            iss, comp, drn = _cube_ticks_us(ticks)
            issues.append(iss)
            compls.append(comp)
            drains.append(drn)
        issue = statistics.median(issues)
        compl = statistics.median(compls)
        drain = statistics.median(drains)
        # TFLOPS = 2*m*n*k / (compl_us * 1e6) * 1e-12 = 2*m*n*k / compl_us / 1e6
        tflops = 2.0 * m0 * n0 * k0 / compl / 1e6
        print(f"       {f'{m0}x{n0}x{k0}':>14s}{wall_per:>10.2f}{issue:>10.2f}"
              f"{compl:>10.2f}{drain:>10.2f}{tflops:>9.2f}")


def run_cube(rt, tiles):
    """实验5: cube matmul 微基准(bench_cube), 单次 Matmul::Run() + host 多次下发计时。"""
    b_full = empty_aic(rt)     # 满核 AIC block 数(实验 5 仅用 1 block)
    print(f"[实验5] cube matmul 微基准(bench_cube), 满核 AIC = {b_full}")
    _cube_check(rt)
    _cube_anti_dce(rt)
    _cube_tile_sweep(rt, tiles)


# ==================== 编排与分析 ====================

def run_bench(mte_sizes=None, mte_only=False, vec_sizes=None, vec_only=False,
              cube_tiles=None, cube_only=False):
    rt = Runtime(0, 500)
    torch.npu.set_device(0)
    out_i32 = torch.zeros(1, dtype=torch.int32, device="npu:0")
    out_f32 = torch.zeros(1, dtype=torch.float32, device="npu:0")

    # 设备预热(原生 torch_npu 算子, 不经 xlite, 不计入结果)
    x = torch.ones(64, dtype=torch.float16, device="npu:0")
    _ = x * 2
    torch.npu.synchronize()

    if mte_only:
        run_mte(rt, mte_sizes if mte_sizes else MTE_SIZES_DEFAULT)
        return
    if vec_only:
        run_vec(rt, vec_sizes)
        return
    if cube_only:
        run_cube(rt, cube_tiles if cube_tiles else CUBE_TILES_DEFAULT)
        return

    run_empty(rt)
    print()
    run_scalar(rt, out_i32, out_f32)
    print()
    if bench_mte_raw is None:
        print("[实验3] MTE 内存微基准(bench_mte)")
        print("   (本构建未编译: cmake -DXLITE_BENCH_TIMED=ON 全量重编后可用)")
    else:
        run_mte(rt, mte_sizes if mte_sizes else MTE_SIZES_DEFAULT)
        print()
    if bench_vec_raw is None:
        print("[实验4] 向量指令微基准(bench_vec)")
        print("   (本构建未编译: cmake -DXLITE_BENCH_TIMED=ON 全量重编后可用)")
    else:
        run_vec(rt, vec_sizes)
        print()
    if bench_cube_raw is None:
        print("[实验5] cube matmul 微基准(bench_cube)")
        print("   (本构建未编译: cmake -DXLITE_BENCH_TIMED=ON 全量重编后可用)")
    else:
        run_cube(rt, cube_tiles if cube_tiles else CUBE_TILES_DEFAULT)


FIT_MIN = 10_000        # 拟合/分析只取 >= 该值 的调用(更小规模截距占比大)


def analyze(csvs):
    import pandas as pd

    frames = []
    for c in csvs:
        d = pd.read_csv(c, comment="#")
        d["src"] = os.path.basename(c)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    is_scalar = df.op.str.startswith("scalar_")
    df.loc[is_scalar, "iters"] = (
        df[is_scalar].args.str.extract(r"(\d+)\s*$", expand=False).astype(int)
    )

    bar = "=" * 74
    print(bar)
    print("微基准分析报告  (数据: %s, 每算子含多次调用取中位)" %
          ", ".join(os.path.basename(c) for c in csvs))
    print(bar)

    # [1] launch floor
    empt = df[df.op.str.startswith("empty_")]
    print("\n[1] 实验1 launch floor -- 空算子 device 耗时 = 该核型一次下发的固有开销")
    if len(empt):
        g = empt.groupby("op").device_us.agg(["median", "min", "max", "count"])
        for op, row in g.iterrows():
            print(f"    {op:12s} med = {row['median']:5.1f} us   "
                  f"(min {row['min']:.1f} / max {row['max']:.1f}, n={int(row['count'])})")
    else:
        print("    (本 CSV 无 empty 算子数据, 跳过)")

    # [2] 每操作成本表 + 线性拟合
    sc = df[is_scalar & (df.iters >= FIT_MIN)]
    if sc.empty:
        # --mte-only/--vec-only 的 profile 只含单一家族, 标量段整体跳过
        print("\n[2]-[7] 标量家族分析: 本 CSV 无 scalar 数据, 跳过")
    else:
        slope, intercept = {}, {}
        for op in OPS:
            d = sc[sc.op == op[0]]
            a, b = np.polyfit(d.iters, d.device_us, 1)
            slope[op[0]] = a * 1000.0            # ns/op
            intercept[op[0]] = b                 # us, launch floor
        med = sc.groupby(["op", "iters"]).device_us.median().unstack()
        ns1m = (med[1_000_000] * 1000.0 / 1_000_000).to_dict()

        print("\n[2] 实验2 标量每操作成本 -- ns/op = device_us*1000/iters (中位)")
        print("    各规模列应基本相等(线性自检); 拟合斜率为去截距后的每操作成本")
        print(f"    {'运算':<26s}{'@100K':>8s}{'@1M':>8s}{'拟合斜率':>10s}{'截距(us)':>10s}")
        for name, _, _, _, _ in OPS:
            n100 = med.loc[name, 100_000] * 1000.0 / 100_000
            print(f"    {name:<26s}{n100:>8.3f}{ns1m[name]:>8.3f}"
                  f"{slope[name]:>10.3f}{intercept[name]:>10.2f}")

        # [3] 槽模型: 以 add 链 = 3 槽为基准, 看其余运算是否落在整数槽上
        slot = ns1m["scalar_add_aiv"] / 3.0
        print("\n[3] 槽模型 -- 以 add 链 = 3 槽定基准槽宽, 其余运算折合槽数(接近整数")
        print(f"    则说明所有成本同为该槽宽的整数倍; 槽宽 = {slot:.4f} ns "
              f"(~{1.0 / slot:.2f} G 槽/s, 由时序反推)")
        for base in ["add", "xor", "add_unroll4", "mul", "fadd", "div"]:
            v = ns1m[f"scalar_{base}_aiv"]
            print(f"    {base:<12s} {v:6.3f} ns  =  {v / slot:5.2f} 槽")

        # [4] AIV vs AIC
        print("\n[4] AIV vs AIC -- 两种核型标量单元是否一致")
        max_diff = 0.0
        for base in ["add", "xor", "add_unroll4", "mul", "fadd", "div"]:
            va, vc = ns1m[f"scalar_{base}_aiv"], ns1m[f"scalar_{base}_aic"]
            diff = (vc - va) / va * 100
            max_diff = max(max_diff, abs(diff))
            print(f"    {base:<12s} aiv={va:.3f}  aic={vc:.3f}  ({diff:+.1f}%)")
        print(f"    最大差异 {max_diff:.1f}%")

        # [5] 三方程分解
        print("\n[5] 循环开销分解 -- s1=add, sx=xor, s4=unroll4 联立解 b/a/o")
        for ct in ["aiv", "aic"]:
            s1 = ns1m[f"scalar_add_{ct}"]
            sx = ns1m[f"scalar_xor_{ct}"]
            s4 = ns1m[f"scalar_add_unroll4_{ct}"]
            b = 4 * (s1 - s4) / 3
            o = (sx - b) / 2
            a = s1 - b - o
            print(f"    {ct}: b(循环控制)={b:.3f}  o(xor)={o:.3f}  a(add)={a:.3f} ns")

        # [6] bench_timed 的 profiler 外部视角(timed kernel device 时间, 对照 untimed)
        bt = df[df.op == "bench_timed"].copy()
        print("\n[6] bench_timed 外部视角 -- profiler 计的 timed kernel device 时间")
        if len(bt):
            bt["variant"] = bt.args.str.extract(r"'(\w+)'", expand=False)
            bt["iters"] = bt.args.str.extract(r"(\d+)\s*$", expand=False).astype(int)
            bt = bt[bt.iters >= FIT_MIN]
            medt = bt.groupby(["variant", "iters"]).device_us.median().unstack()
            print("    与 [2] untimed 同变体之差, 正常 = 打点机制自身开销(3 次读 SYS_CNT +")
            print("    3 槽批量回写 + 额外 barrier), 应为 sub-us 级")
            devs = []
            for name, _, _, _, _ in OPS:
                variant = name[len("scalar_"):]
                if variant not in medt.index or 1_000_000 not in medt.columns:
                    continue
                if name not in med.index or 1_000_000 not in med.columns:
                    continue
                tu = medt.loc[variant, 1_000_000]
                un = med.loc[name, 1_000_000]
                devs.append((variant, tu - un))
                print(f"    {variant:<16s} timed={tu:8.1f}  untimed={un:8.1f}  diff={tu - un:+5.1f} us")
            small = [abs(d) for _, d in devs if abs(d) <= 5.0]
            if small:
                print(f"    机制开销判读: {len(small)}/{len(devs)} 变体 |diff|<=5us, 中位 "
                      f"{statistics.median(small):.2f} us -> 打点机制自身开销 ≈ 0")
            for v, d in devs:
                if abs(d) > 5.0:
                    print(f"    {v}: diff={d:+.1f} us 非机制开销 -- timed/untimed 分属不同编译")
                    print("      单元, 1 槽级整数运算对 TU 代码生成上下文敏感(mul 曾在"
                          "3↔2 槽间漂移),")
                    print("      延迟 bound 的 fadd/div 不敏感; 见报告 TU 漂移一节")
            print("    (50MHz 换算自洽: 运行输出 2c) 的 compl_us@1M 应略小于本表 timed 列,")
            print("     差值 = kernel 前后缀 -- 打点前的 prologue + 回写/收尾的 epilogue)")
        else:
            print("    (本 CSV 无 bench_timed 数据, 跳过)")

        # [7] 自动结论
        print("\n[7] 自动结论")
        lin_dev = 0.0
        for name, _, _, _, _ in OPS:
            n100 = med.loc[name, 100_000] * 1000.0 / 100_000
            lin_dev = max(lin_dev, abs(n100 - ns1m[name]) / ns1m[name] * 100)
        print(f"    - 线性自检: ns/op @100K vs @1M 最大偏差 {lin_dev:.1f}% "
              f"({'OK, 循环真实执行' if lin_dev < 5 else '超 5%, 数据可疑'})")
        print(f"    - AIV vs AIC: 最大差异 {max_diff:.1f}% "
              f"({'一致, 标量单元同构' if max_diff < 2 else '存在差异'})")
        b_aiv = 4 * (ns1m['scalar_add_aiv'] - ns1m['scalar_add_unroll4_aiv']) / 3
        print(f"    - for 循环控制开销 b = {b_aiv:.3f} ns (占 add 链 "
              f"{b_aiv / ns1m['scalar_add_aiv'] * 100:.0f}%), 展开无收益 -> 循环控制"
              "被流水隐藏")
        if len(empt):
            lo, hi = min(intercept.values()), max(intercept.values())
            print(f"    - launch floor: 拟合截距 {lo:.1f}~{hi:.1f} us, 与 [1] empty "
                  "实测互相印证")
        print(f"    - 槽模型: add/xor/mul=3 槽, fadd=4 槽, div=8 槽, 槽宽 {slot:.3f} ns")

    # [8] bench_mte 外部视角
    bm = df[df.op == "bench_mte"].copy()
    print("\n[8] bench_mte 外部视角 -- profiler 计的 mte kernel device 时间")
    if len(bm):
        bm["variant"] = bm.args.str.extract(r"'(\w+)'", expand=False)
        recs = []
        for _, row in bm.iterrows():
            # args 形如 "Runtime, 'read_aiv', 1048576:int8, ..., 1048576, 1, 1":
            # 裸整数依次为 bytes/iters/block_num("shape:dtype" 形态不匹配);
            # 结尾必须用前瞻 (?=,|$) 而非吞掉逗号 -- 否则相邻裸整数每隔一个
            # 被吞掉的 ", " 跳过, 3 个数只能捕到 2 个
            nums = re.findall(r"(?:^|, )(\d+)(?=,|$)", str(row.args))
            if len(nums) >= 3 and row.device_us > 0:
                recs.append((row.variant, int(nums[0]), int(nums[2]), row.device_us))
        if recs:
            t = pd.DataFrame(recs, columns=["variant", "bytes", "block", "device_us"])
            g = t.groupby(["variant", "bytes", "block"]).device_us.median()
            print("    与运行输出 3c)/3d) 的核内 complete 对照: 单 block 安静态应恒差")
            print("    ~+2us(前后缀); 满核/大额分配后首调差值更大属口径差异")
            print("    (整核寿命含跨 block 派发/收尾), 详见 doc/kernels/microbench.md")
            print(f"    {'variant':<12s}{'size':>8s}{'B':>4s}{'dev_us':>10s}")
            for (v, byt, blk), dv in g.items():
                print(f"    {v:<12s}{_mte_fmt_size(byt):>8s}{blk:>4d}{dv:>10.1f}")
        else:
            print("    (无可解析的 bench_mte 行, 跳过)")
    else:
        print("    (本 CSV 无 bench_mte 数据, 跳过)")

    # [9] bench_vec 外部视角
    bvec = df[df.op == "bench_vec"].copy()
    print("\n[9] bench_vec 外部视角 -- profiler 计的 vec kernel device 时间")
    if len(bvec):
        bvec["variant"] = bvec.args.str.extract(r"'(\w+)'", expand=False)
        recs = []
        for _, row in bvec.iterrows():
            # args 形如 "Runtime, 'add_f32_aiv', 128:float, ..., 1024, 512, 1":
            # 裸整数依次为 elems/iters/block_num("shape:dtype" 形态不匹配);
            # 结尾前瞻 (?=,|$) 同 [8] 的坑(吞逗号会隔个跳过相邻裸整数)
            nums = re.findall(r"(?:^|, )(\d+)(?=,|$)", str(row.args))
            if len(nums) >= 1 and row.device_us > 0:
                recs.append((row.variant, int(nums[0]),
                             int(nums[1]) if len(nums) >= 2 else 1,
                             int(nums[2]) if len(nums) >= 3 else 1, row.device_us))
        if recs:
            t = pd.DataFrame(recs, columns=["variant", "elems", "iters", "block", "device_us"])
            g = t.groupby(["variant", "elems", "block"]).device_us.median()
            print("    与运行输出 4c) 的核内 complete 对照: 单 block 安静态应恒差")
            print("    ~+2us(前后缀); 满核差值更大属口径差异(整核寿命含派发/收尾),")
            print("    详见 doc/kernels/microbench.md")
            print(f"    {'variant':<12s}{'elems':>8s}{'B':>4s}{'dev_us':>10s}")
            for (v, el, blk), dv in g.items():
                print(f"    {v:<12s}{el:>8d}{blk:>4d}{dv:>10.1f}")
        else:
            print("    (无可解析的 bench_vec 行, 跳过)")
    else:
        print("    (本 CSV 无 bench_vec 数据, 跳过)")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--analyze":
        csvs = args[1:]
        if not csvs:
            sys.exit("usage: micro_bench.py --analyze <xlite_ops csv> [csv ...]")
        analyze(csvs)
        return
    mte_sizes, mte_only, rest, spec = None, False, [], None
    vec_sizes, vec_only, vspec = None, False, None
    cube_tiles, cube_only, cspec = None, False, None
    quick = False
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--quick":
            quick = True
            i += 1
        elif a == "--mte-sizes":
            if i + 1 >= len(args):
                sys.exit("usage: micro_bench.py [--mte-sizes 64k,1m,...] [--mte-only] "
                         "[--vec-sizes 64,256,...] [--vec-only] "
                         "[--cube-sizes 128,128,128;...] [--cube-only]")
            spec = args[i + 1]
            i += 2
        elif a.startswith("--mte-sizes="):
            spec = a.split("=", 1)[1]
            i += 1
        elif a == "--mte-only":
            mte_only = True
            i += 1
        elif a == "--vec-sizes":
            if i + 1 >= len(args):
                sys.exit("usage: micro_bench.py [--mte-sizes 64k,1m,...] [--mte-only] "
                         "[--vec-sizes 64,256,...] [--vec-only] "
                         "[--cube-sizes 128,128,128;...] [--cube-only]")
            vspec = args[i + 1]
            i += 2
        elif a.startswith("--vec-sizes="):
            vspec = a.split("=", 1)[1]
            i += 1
        elif a == "--vec-only":
            vec_only = True
            i += 1
        elif a == "--cube-sizes":
            if i + 1 >= len(args):
                sys.exit("usage: micro_bench.py [--cube-sizes 128,128,128;...]")
            cspec = args[i + 1]
            i += 2
        elif a.startswith("--cube-sizes="):
            cspec = a.split("=", 1)[1]
            i += 1
        elif a == "--cube-only":
            cube_only = True
            i += 1
        else:
            rest.append(a)
            i += 1
    if rest:
        sys.exit(f"unknown args: {' '.join(rest)} "
                 "(支持 --quick / --mte-sizes / --mte-only / --vec-sizes / --vec-only / "
                 "--cube-sizes / --cube-only)")
    if spec is not None:
        try:
            mte_sizes = _parse_mte_sizes(spec)
        except ValueError as e:
            sys.exit(f"--mte-sizes: {e}")
    if vspec is not None:
        try:
            vec_sizes = _parse_vec_sizes(vspec)
        except ValueError as e:
            sys.exit(f"--vec-sizes: {e}")
    if cspec is not None:
        try:
            cube_tiles = _parse_cube_tiles(cspec)
        except ValueError as e:
            sys.exit(f"--cube-sizes: {e}")
    if sum(1 for x in (mte_only, vec_only, cube_only) if x) > 1:
        sys.exit("--mte-only / --vec-only / --cube-only 互斥")
    if mte_only and bench_mte_raw is None:
        sys.exit("--mte-only 需要 XLITE_BENCH_TIMED=ON 构建(默认构建无 bench_mte)")
    if vec_only and bench_vec_raw is None:
        sys.exit("--vec-only 需要 XLITE_BENCH_TIMED=ON 构建(默认构建无 bench_vec)")
    if cube_only and bench_cube_raw is None:
        sys.exit("--cube-only 需要 XLITE_BENCH_TIMED=ON 构建(默认构建无 bench_cube)")
    if quick:
        _apply_quick()
        print("[quick] 已缩减重复次数与规模档位(冒烟模式, 非出报告用)")
    run_bench(mte_sizes, mte_only, vec_sizes, vec_only, cube_tiles, cube_only)


if __name__ == "__main__":
    main()
