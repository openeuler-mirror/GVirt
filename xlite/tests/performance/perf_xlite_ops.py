#!/usr/bin/env python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even warranties of any kind.
# ===============================================================================
"""xlite 单算子耗时 profiling 工具: 不改测试代码, 抓取现成测试用例中 xlite 算子的执行耗时.

用法(环境须能 import xlite._C; 注意在 xlite 仓库根目录运行会 import 到源码树包,
若未在源码树构建, 请从其他目录运行):
    python tests/performance/prof_xlite_ops.py tests/kernels/dequant.py

以 tests/kernels 单算子测试为设计对象, 任意 python 测试脚本均可套用, 两层口径:
  1) wall-clock 逐调用: patch xlite._C 全部导出算子, 每次 python 调用计时
     (含 pybind + host 派发 + launch + kernel + 算子内建 Synchronize), 同时记录
     入参 shape. 注意 DEVICE_TIME=True 时 wall 还含 profiler 采集开销(实测每调用
     +100us 量级), 干净的 wall 数字请置 DEVICE_TIME=False 单独跑一遍;
  2) device kernel time: torch_npu.profiler 采集, 每次调用包一层
     record_function("xlite::<op>") 标注, 解析 chrome trace 后归因到每次调用
     (CPU 版 torch 的 torch.profiler CUDA activity 不可用, 必须走
     torch_npu.profiler). device 口径不受采集开销影响. 归因三步(见
     attribute_device): 就近窗口投票 → kernel 名/op 名共享前缀校验 → 同名 kernel
     按时间序与调用序配对; 不依赖 ts 精确包含的原因: 实测 trace 中 kernel ts
     系统性偏晚 ~30us, 紧窗口下纯包含归因会失败.

输出: 终端一张 per-op 汇总表(wall/device 的 med/min/max + launch+sync); 单个
     CSV OUT_DIR/xlite_ops_<ts>.csv: 逐调用明细为主体(pandas 可直接读), 尾部
     # 注释行附每算子汇总(pd.read_csv(comment='#') 只读明细), CSV 始终保留.
     KEEP_PROF_DIR=False(默认): 运行结束清理 CANN profiling 落盘目录
     export_only_prof_dir/; 置 True 保留供排查.

限制: 单进程/单线程假设(调用与标注窗口按时间序对应); device 归因依赖单 stream
     按序执行(同名 kernel 事件序 = 调用序)与 xlite kernel 名由算子名派生的约定,
     若有算子例外可用配置区 KERNEL_OP_OVERRIDES 手工指定; 归因失败会打 warn,
     wall 口径不受影响.

所有实验变量在配置区修改; 命令行只有被测脚本路径.
"""
import argparse
import bisect
import contextlib
import csv
import functools
import inspect
import json
import os
import runpy
import shutil
import statistics
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime

import torch
import torch_npu

import xlite._C as _C

try:
    from torch.profiler import record_function
    from torch_npu.profiler import ProfilerActivity as NpuActivity
    from torch_npu.profiler import profile as npu_profile
    PROFILER_OK = True
except ImportError:
    PROFILER_OK = False

# ============================== 配置区: 所有实验变量在这里改 ==============================

DEVICE_TIME = True         # 采集 device kernel time(torch_npu.profiler, 约 +3s 解析开销)
EXCLUDE_FIRST_CALL = False  # 每算子首次调用含 lazy init(模块加载/descriptor 建立等),
                            # True 时汇总统计剔除首调(明细 CSV 仍保留)
EXCLUDE_OPS = set()        # 不 patch 不计时的算子名(如只想关注部分算子)
OUT_DIR = "xlite_prof_results"  # CSV 结果目录(始终保留)
KEEP_PROF_DIR = False      # False(默认): 运行结束清理 CANN profiling 落盘目录
                           # export_only_prof_dir/(profiler 原始数据, 跨运行累积);
                           # True: 保留该目录供排查
MAX_ARG_REPR = 64          # 明细 CSV 入参描述截断长度

# ---- device 归因参数(见 attribute_device 注释) ----
# 实测 torch_npu chrome trace 中 kernel 事件的 ts 系统性偏晚 ~30us(疑似 profiler
# 侧时间戳而非真正设备执行时刻), 紧窗口下纯 ts 包含归因会失败/串到下一调用窗口,
# 故采用: 就近窗口(容忍 ATTR_GAP_US) → 按 kernel 名多数投票 + kernel/op 名共享
# 前缀校验 → 同名 kernel 按时间序与调用序一一配对(单 stream 按序执行).
ATTR_GAP_US = 200          # kernel ts 晚于窗口结束多少 us 内仍认 belong 该窗口
ATTR_VOTE_MIN = 0.8        # name→op 多数票接受阈值
ATTR_PREFIX_MIN = 5        # kernel 名与 op 名共享前缀最小长度(防串扰, 如 aclnn* → op)
KERNEL_OP_OVERRIDES = {}   # 手工指定 kernel 名 → op 名(优先于自动发现), 键为全名或前缀

# 归因防串扰: 名字明显非 xlite 的 device 事件(调用窗口后紧跟的 torch kernel 拖尾
# 执行会就近落到刚结束的窗口), 不参与投票与归因, 仅计数
NON_XLITE_KERNEL_PATTERNS = ("aten::", "Memcpy", "Memset", "Distribution",
                             "TransData", "aclnn")

# torch_npu profiler 会把 CANN 原始数据落在 cwd 下 export_only_prof_dir/, 结束后
# 不自动清理且跨运行累积; 默认整目录移除(KEEP_PROF_DIR=True 时保留)
_PROF_SCRATCH_ROOT = "export_only_prof_dir"

# =========================================================================================


@dataclass
class Call:
    seq: int
    op: str
    wall_us: float
    args: str
    error: str = ""
    device_us: float = 0.0
    kernels: list = field(default_factory=list)  # [(kernel 名, dur_us)]


CALLS: list[Call] = []


def fmt_us(us):
    if us != us:  # NaN
        return "-"
    return f"{us:.1f}us" if us < 10000 else f"{us / 1000:.2f}ms"


def describe_args(args, kwargs):
    """入参摘要: tensor 记 shape×dtype, 标量记值, 其余记类型名."""
    parts = []
    for a in args:
        if isinstance(a, torch.Tensor):
            shape = "x".join(str(d) for d in a.shape)
            parts.append(f"{shape}:{str(a.dtype).removeprefix('torch.')}")
        elif isinstance(a, (bool, int, float)):
            parts.append(str(a))
        else:
            parts.append(type(a).__name__)
    for k, v in kwargs.items():
        parts.append(f"{k}={v}" if isinstance(v, (bool, int, float))
                     else f"{k}={type(v).__name__}")
    return ", ".join(parts)[:MAX_ARG_REPR]


def make_wrapper(orig, name):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter_ns()
        err = ""
        try:
            if DEVICE_TIME:
                # 标注窗口 = orig 调用全程; 算子内建 Synchronize 保证其 kernel
                # 生命周期落在窗口内, trace 解析时据此归因 device 时间
                with record_function(f"xlite::{name}"):
                    return orig(*args, **kwargs)
            return orig(*args, **kwargs)
        except Exception as e:
            err = type(e).__name__
            raise
        finally:
            wall = (time.perf_counter_ns() - t0) / 1e3
            CALLS.append(Call(len(CALLS) + 1, name, wall,
                              describe_args(args, kwargs), err))
    return wrapper


def patch_xlite_ops():
    """patch xlite._C 全部导出算子(在 runpy 执行测试前完成, 测试内的
    `from xlite._C import x` 与 `xlite._C.x(...)` 两种调用方式均被覆盖)."""
    patched = []
    for name in dir(_C):
        if name.startswith("_") or name in EXCLUDE_OPS:
            continue
        obj = getattr(_C, name)
        if not callable(obj) or inspect.isclass(obj):
            continue
        setattr(_C, name, make_wrapper(obj, name))
        patched.append(name)
    return patched


def parse_trace(path):
    """chrome trace → (标注窗口, device 事件).

    标注窗口: record_function 产生 cat=cpu_op 的 "xlite::<op>" 事件;
    device 事件: args 带 "Task Type" 的 kernel/memcpy 事件(cat 为 "?").
    """
    with open(path) as f:
        raw = json.load(f)
    events = raw.get("traceEvents", raw) if isinstance(raw, dict) else raw
    anns, devs = [], []
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        ts, dur = e.get("ts"), e.get("dur")
        if ts is None or dur is None:
            continue
        if name.startswith("xlite::"):
            anns.append((name.removeprefix("xlite::"), float(ts),
                         float(ts) + float(dur)))
        elif "Task Type" in (e.get("args") or {}):
            devs.append((name, float(ts), float(dur)))
    return anns, devs


def _nearest_window(anns_sorted, starts, ts):
    """ts 所属/紧随的窗口下标: 先找最后一个 start<=ts 的窗口(窗口互不重叠);
    ts 在其 [start,end] 内 → 即所属; ts 超出其 end 且延迟 <= ATTR_GAP_US → 也认
    belong 该窗口(吸收 trace ts 偏晚偏差); 否则返回 None."""
    i = bisect.bisect_right(starts, ts) - 1
    if i < 0:
        return None
    if ts <= anns_sorted[i][2] or ts - anns_sorted[i][2] <= ATTR_GAP_US:
        return i
    return None


def attribute_device(anns, devs):
    """把 device 事件归因到调用, 三步:

    1) 就近投票: 每个事件找最近前置窗口(容忍 ATTR_GAP_US 偏晚), 得到每个 kernel
       名对 op 的得票; 名字命中 NON_XLITE_KERNEL_PATTERNS 的事件不投票;
    2) name→op 映射: KERNEL_OP_OVERRIDES 优先; 否则要求得票率 >= ATTR_VOTE_MIN
       且 kernel 名与 op 名共享前缀 >= ATTR_PREFIX_MIN(xlite kernel 名由算子名
       派生, 如 matmul_dequant→matmul_int8_t_5 共享 "matmul"; aclnn* 等无共享
       前缀, 天然被拒);
    3) 配对: 映射成功的 kernel 名按 ts 排序与该 op 的调用序 zip(k-th ↔ k-th,
       单 stream 按序执行); 数量不匹配时该名退回逐事件就近归因并告警.

    返回 (per_call: (op, 第几次调用) -> [(kernel, dur)], 诊断信息 dict).
    """
    diag = {"excluded": [], "unattributed": [], "vote_failed": Counter(),
            "count_mismatch": []}
    anns_sorted = sorted(anns, key=lambda a: a[1])
    starts = [a[1] for a in anns_sorted]
    op_ann_idx, win_key = Counter(), []
    for op, _ts, _end in anns_sorted:
        win_key.append((op, op_ann_idx[op]))
        op_ann_idx[op] += 1

    # 步骤 1: 就近投票
    votes = {}  # kernel 名 -> Counter(op)
    near_owner = []  # (dev 事件, 就近窗口下标 or None), 与 devs 等长
    for name, ts, _dur in devs:
        i = _nearest_window(anns_sorted, starts, ts)
        near_owner.append(i)
        if i is None:
            diag["unattributed"].append(name)
            continue
        if any(p in name for p in NON_XLITE_KERNEL_PATTERNS):
            diag["excluded"].append(name)
            continue
        votes.setdefault(name, Counter())[win_key[i][0]] += 1

    # 步骤 2: name→op 映射
    name_to_op = {}
    for name, counter in votes.items():
        op, cnt = counter.most_common(1)[0]
        overridden = any(name.startswith(k) or k == name for k in KERNEL_OP_OVERRIDES)
        if overridden:
            name_to_op[name] = next(
                v for k, v in KERNEL_OP_OVERRIDES.items()
                if name.startswith(k) or k == name)
            continue
        if (cnt / sum(counter.values()) >= ATTR_VOTE_MIN
                and len(os.path.commonprefix([name, op])) >= ATTR_PREFIX_MIN):
            name_to_op[name] = op
        else:
            diag["vote_failed"][name] += cnt

    # 步骤 3: 归并到 op 后按时间序与调用序配对. 同一算子按 dtype 派生多个 kernel 名
    # (如 matmul → matmul_float16_t_4 / matmul_bfloat16_t_4), 须合并计数后配对;
    # 每次调用恰一个 kernel 时总数 == 调用数, zip 精确; 不匹配则退回逐事件就近.
    per_call = {}
    by_op = {}  # op -> [(ts, dur, name, devs 下标)]
    for idx, (name, ts, dur) in enumerate(devs):
        op = name_to_op.get(name)
        if op is not None:
            by_op.setdefault(op, []).append((ts, dur, name, idx))
    for op, evs in by_op.items():
        n_calls = op_ann_idx[op]
        if len(evs) == n_calls:
            for k, (_ts, dur, name, _idx) in enumerate(sorted(evs)):
                per_call.setdefault((op, k), []).append((name, dur))
        else:
            # 数量不匹配(每次调用多个 kernel, 或混入杂散/漏采事件): 退回逐事件就近
            diag["count_mismatch"].append(f"{op}: {len(evs)} kernels vs {n_calls} calls")
            for _ts, dur, name, idx in evs:
                i = near_owner[idx]
                if i is not None and win_key[i][0] == op:
                    per_call.setdefault(win_key[i], []).append((name, dur))
                else:
                    diag["unattributed"].append(name)
    return per_call, diag


def fill_calls_with_device(per_call):
    """把归因结果回填到 CALLS(同算子第 k 个窗口 ↔ 第 k 次调用)."""
    idx_by_op = Counter()
    for c in CALLS:
        k = idx_by_op[c.op]
        idx_by_op[c.op] += 1
        kers = per_call.get((c.op, k), [])
        c.kernels = kers
        c.device_us = sum(d for _, d in kers)


def summarize_calls(calls):
    """(统计口径调用集, wall 统计, device 统计, kernel 计数).
    wall 统计 = (med, mean, min, max, sum); device 统计 = (med, mean, max, sum)."""
    stat = calls[1:] if EXCLUDE_FIRST_CALL and len(calls) > 1 else calls
    walls = [c.wall_us for c in stat]
    devs = [c.device_us for c in stat if c.kernels]
    kers = Counter(name for c in stat for name, _ in c.kernels)
    return (stat, (statistics.median(walls), statistics.mean(walls), min(walls),
                   max(walls), sum(walls)),
            (statistics.median(devs), statistics.mean(devs), max(devs), sum(devs))
            if devs else None,
            kers)


def print_summary():
    """单张紧凑表: wall(pybind+launch+sync+kernel) 与 device(kernel) 并排,
    w-d 即 launch+sync+采集开销; mean/sum 只落 CSV, 终端留 med/min/max."""
    by_op = {}
    for c in CALLS:
        by_op.setdefault(c.op, []).append(c)

    print(f"\n== per-op: wall=pybind+launch+sync+kernel | dev=kernel | w-d≈launch+sync ==")
    name_w = min(40, max(len(op) for op in by_op)) + 1
    hdr = (f"{'op':<{name_w}}{'n':>5}{'wall_med':>11}{'wall_min':>11}"
           f"{'wall_max':>11}")
    if DEVICE_TIME:
        hdr += f"{'dev_med':>11}{'dev_max':>11}{'w-d':>10}  kernels"
    print(hdr)
    rows = sorted(by_op.items(), key=lambda kv: -sum(c.wall_us for c in kv[1]))
    for op, calls in rows:
        _, (w_med, _w_mean, w_min, w_max, _w_sum), dev, kers = summarize_calls(calls)
        line = (f"{op:<{name_w}}{len(calls):>5}{fmt_us(w_med):>11}"
                f"{fmt_us(w_min):>11}{fmt_us(w_max):>11}")
        if DEVICE_TIME:
            if dev is None:
                line += f"{'-':>11}{'-':>11}{'-':>10}"
            else:
                d_med, _d_mean, d_max, _d_sum = dev
                line += (f"{fmt_us(d_med):>11}{fmt_us(d_max):>11}"
                         f"{fmt_us(w_med - d_med):>10}")
            line += "  " + ", ".join(f"{n}x{cnt}" for n, cnt in kers.most_common())
        print(line)
        # 归因完整性: wall 调用数 vs 归因到 kernel 的调用数
        n_attr = len([c for c in calls if c.kernels])
        if DEVICE_TIME and n_attr < len(calls):
            print(f"[warn] {op}: {n_attr}/{len(calls)} 调用归因到 device kernel"
                  f"(调用抛异常, 或封装无内建 Synchronize; wall 口径不受影响)")


def write_csv():
    """单文件输出: 逐调用明细为主体(标准 CSV, pandas 可直接读), 尾部以 # 注释行
    附每算子汇总(pd.read_csv(path, comment='#') 只读明细, 汇总行供人看/grep)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"xlite_ops_{datetime.now():%Y%m%d_%H%M%S}.csv")
    by_op = {}
    for c in CALLS:
        by_op.setdefault(c.op, []).append(c)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["seq", "op", "wall_us", "device_us", "kernels", "args", "error"])
        for c in CALLS:
            w.writerow([c.seq, c.op, f"{c.wall_us:.1f}", f"{c.device_us:.1f}",
                        ";".join(f"{n}:{d:.1f}" for n, d in c.kernels),
                        c.args, c.error])
        f.write("\n# ---- summary per op (comment lines) ----\n")
        f.write("# op,calls,wall_med_us,wall_mean_us,wall_min_us,wall_max_us,"
                "wall_total_ms,device_calls,device_med_us,device_mean_us,"
                "device_max_us,device_total_ms,launch_sync_med_us,kernels\n")
        for op, calls in sorted(by_op.items()):
            _, (w_med, w_mean, w_min, w_max, w_sum), dev, kers = summarize_calls(calls)
            if dev is None:
                f.write(f"# {op},{len(calls)},{w_med:.1f},{w_mean:.1f},{w_min:.1f},"
                        f"{w_max:.1f},{w_sum / 1000:.3f},0,-,-,-,-,-,-\n")
                continue
            d_med, d_mean, d_max, d_sum = dev
            f.write(f"# {op},{len(calls)},{w_med:.1f},{w_mean:.1f},{w_min:.1f},"
                    f"{w_max:.1f},{w_sum / 1000:.3f},"
                    f"{len([c for c in calls if c.kernels])},"
                    f"{d_med:.1f},{d_mean:.1f},{d_max:.1f},{d_sum / 1000:.3f},"
                    f"{w_med - d_med:.1f},"
                    f"{';'.join(f'{n}x{cnt}' for n, cnt in kers.most_common())}\n")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="抓取现成测试用例中 xlite 算子的执行耗时(变量见脚本内配置区)")
    parser.add_argument("script", help="被测测试脚本路径, 如 tests/kernels/dequant.py")
    opts = parser.parse_args()
    script = os.path.abspath(opts.script)
    if not os.path.isfile(script):
        sys.exit(f"test script not found: {script}")

    global DEVICE_TIME
    if DEVICE_TIME and not PROFILER_OK:
        print("[warn] torch_npu.profiler 不可用, 关闭 device-time 采集")
        DEVICE_TIME = False

    patched = patch_xlite_ops()
    print(f"[prof] {len(patched)} xlite ops | device_time={DEVICE_TIME} "
          f"excl_first={EXCLUDE_FIRST_CALL} keep_prof_dir={KEEP_PROF_DIR}")
    if DEVICE_TIME:
        print("[prof] 注: profiler 使 wall +~100us/调用, "
              "干净 wall 请置 DEVICE_TIME=False 再跑")

    trace_path = os.path.join(tempfile.gettempdir(), f"xlite_prof_trace_{os.getpid()}.json")
    prof, test_exc, status = None, None, "OK"
    t0 = time.perf_counter()
    try:
        if DEVICE_TIME:
            prof = npu_profile(activities=[NpuActivity.CPU, NpuActivity.NPU])
            prof.start()
        sys.argv = [script]
        sys.path.insert(0, os.path.dirname(script))  # 测试脚本 import 同目录 helper 用
        try:
            runpy.run_path(script, run_name="__main__")
        except SystemExit as e:
            status = f"exit({e.code})"
        except BaseException as e:  # 测试抛错也要出结果, 输出完再重抛
            status = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            test_exc = e
    finally:
        if prof is not None:
            prof.stop()
            prof.export_chrome_trace(trace_path)
    elapsed = time.perf_counter() - t0

    if DEVICE_TIME:
        try:
            anns, devs = parse_trace(trace_path)
            per_call, diag = attribute_device(anns, devs)
            fill_calls_with_device(per_call)
            n_attr = sum(len(v) for v in per_call.values())
            print(f"[prof] device: {len(devs)} events → {n_attr} kernels 归因 | "
                  f"排除非 xlite {len(diag['excluded'])} | "
                  f"未归因 {len(diag['unattributed'])}(torch 侧)")
            if diag["vote_failed"]:
                print(f"[prof] [warn] 投票未过阈值不归因: "
                      f"{diag['vote_failed'].most_common(3)}")
            for msg in diag["count_mismatch"]:
                print(f"[prof] [warn] kernel 数≠调用数, 逐事件归因: {msg}")
        except Exception as e:
            print(f"[warn] trace 解析失败({type(e).__name__}: {e}), device 口径缺失")

    print(f"\n[prof] {os.path.basename(script)} | {status} | {elapsed:.2f}s | "
          f"{len(CALLS)} calls")
    if CALLS:
        print_summary()
        print(f"[prof] csv: {write_csv()}")

    if not KEEP_PROF_DIR and os.path.isdir(_PROF_SCRATCH_ROOT):
        # CANN profiling 原始落盘目录, 不自动清理且跨运行累积, 默认整目录清理
        shutil.rmtree(_PROF_SCRATCH_ROOT, ignore_errors=True)
        print(f"[prof] 已清理 {_PROF_SCRATCH_ROOT}/ (保留请置 KEEP_PROF_DIR=True)")
    with contextlib.suppress(OSError):
        os.remove(trace_path)
    if test_exc is not None:
        raise test_exc


if __name__ == "__main__":
    main()
