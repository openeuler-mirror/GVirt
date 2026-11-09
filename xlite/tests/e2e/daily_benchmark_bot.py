#!/usr/bin/env python3
"""
xlite 每日自动化测试机器人

功能说明:
1. 拉取最新代码
2. 编译项目生成 wheel 包
3. 执行基准测试脚本
4. 解析测试报告并与上一版本的数据对比
5. 检测性能劣化 (超过阈值时告警)
6. 发送通知到群组

架构说明:
- 在单一容器中完成拉取代码、编译项目、运行基准测试
- 从远程仓库拉取代码到 /workspaces/code/opencode/GVirt

使用方法:
    python3 daily_benchmark_bot.py --model moe --receiver "927280411401503971"

    可选参数:
    --model: 测试模型类型 (moe/moe_quant/moe_quant_a3/moe_mix/quant/all)，默认 moe
    --skip-pull: 跳过代码拉取
    --skip-build: 跳过编译
    --receiver: 接收者ID，默认 927280411401503971
    --threshold: 性能劣化阈值，默认 0.05 (即5%)
    --offline-threshold: 离线 bench 性能劣化阈值，默认 0.025 (即2.5%)
    --env-type: 环境类型 (blue/yellow)，默认 yellow
               - yellow 环境: 需要执行 source /home/env.sh
               - blue 环境: 不需要 source /home/env.sh
    --compare-ttft: 启用TTFT指标作为对比指标（默认不比较TTFT）
    --compare-tpot-p99: 启用TPOT P99指标作为对比指标（默认不比较TPOT P99）

退出码:
    0 - 成功，无性能劣化，且所有场景正常完成
    1 - 执行失败 (含: 拉取/编译/安装失败、测试阶段崩溃、对比报告缺失、
        或存在服务未起/压测报错等失败场景 —— 这些失败会单独发一条 ⚠️ 通知)
    2 - 成功，但检测到性能劣化 (仅当所有场景正常完成时才可能)

目录结构:
    PROJECT_ROOT = /workspaces/code/opencode/GVirt (容器内路径)
    REPORT_DIR = /home/daily_reports (容器内路径)
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 导入共享工具模块
from daily_bot_utils import (
    # 路径常量
    E2E_DIR,
    XLITE_DIR,
    build_project,
    # 环境检测
    detect_model_device,
    get_last_release_tag,
    # 版本信息
    get_vllm_ascend_version,
    # xlite 库路径
    get_xlite_lib_path,
    get_xlite_version,
    install_wheel,
    load_metrics_json,
    log_error,
    # 日志工具
    log_info,
    log_success,
    log_warning,
    move_log_file,
    # 代码拉取和编译
    pull_latest_code,
    # 数据存储
    save_metrics_json,
    # 通知发送
    send_notification,
    # 配置设置
    set_env_type,
    set_machine_ip,
    set_test_container,
    setup_logging,
    should_disable_xccl,
)

# ====================== 本地配置 ======================
# 每日报告存储目录 (宿主机路径)
REPORT_DIR = Path("/home/daily_reports")

# 性能劣化阈值，超过此比例视为性能下降
DEGRADATION_THRESHOLD = 0.05
# 离线 bench 性能劣化阈值 (与在线分离): 默认 0.025 = 2.5%。
# 离线 bench 噪声更小、对劣化更敏感, 故用更紧的阈值; 在线性能测试仍用上面
# 的 DEGRADATION_THRESHOLD (默认 5%)。
OFFLINE_DEGRADATION_THRESHOLD = 0.025
# TTFT指标是否作为对比指标（ttft_avg 和 ttft_p99）
COMPARE_TTFT_METRICS = True
# TPOT P99指标是否作为对比指标
COMPARE_TPOT_P99_METRICS = False

# ====================== 离线 bench 测试配置 ======================
# 用 run.sh 跑 run_glm5_w8a8 的 bench 模式，只跑 8 层（不关心精度），
# 记录性能数据并与基线对比。其余逻辑与 online 性能测试一致。
OFFLINE_BENCH_MODEL = "glm5_w8a8"
OFFLINE_BENCH_N_LAYERS = 8
# 用 8 卡跑：NPU id 列表（长度即卡数）。moe_ep_size 必须与卡数一致
OFFLINE_BENCH_NPUS = "0,1,2,3,4,5,6,7"
OFFLINE_BENCH_MOE_EP_SIZE = 8
# checkpoint 目录名 (models_base_path 下的子目录), 不同环境挂载的目录名可能不同。
OFFLINE_BENCH_CKPT_DIR = "GLM-5.1-w8a8"
OFFLINE_BENCH_ITERS = 10
# bench 场景: (input_len, output_len, batch_size)
# output_len==1 即 prefill 场景
OFFLINE_BENCH_SCENARIOS = [
    # 场景1: 512|512, concurrency 1 16 32 48 64
    (512, 512, 1),
    (512, 512, 16),
    (512, 512, 32),
    (512, 512, 48),
    (512, 512, 64),
    # 场景2: 3584|1536, concurrency 1 16
    (3584, 1536, 1),
    (3584, 1536, 16),
    # 场景3: 8192|1024, concurrency 1
    (8192, 1024, 1),
    # 场景4: 16384|1536, concurrency 1
    (16384, 1536, 1),
    # 场景5: 32768|3072, concurrency 1
    (32768, 3072, 1),
    # 场景6: 73728|8192, concurrency 1
    (73728, 8192, 1),
    # 场景7: prefill 场景
    (512, 1, 1),
    (512, 1, 16),
    (512, 1, 32),
    (512, 1, 48),
    (512, 1, 64),
    (3584, 1, 1),
    (3584, 1, 16),
    (8192, 1, 1),
    (16384, 1, 1),
    (32768, 1, 1),
    (73728, 1, 1),
]


# ====================== 基准测试执行函数 ======================
def update_model_paths(device: str) -> bool:
    """
    更新 online_server_compare.sh 中的模型路径

    参数:
        device: 设备名称 (如 'nvme0n1', 'sdb')

    返回:
        True - 更新成功
        False - 更新失败
    """
    script_path = E2E_DIR / "online_server_compare.sh"

    try:
        content = script_path.read_text(encoding="utf-8")

        # 替换模型路径中的设备名
        # 格式: /mnt/nvme0n1/models/... -> /mnt/{device}/models/...
        updated = re.sub(r"/mnt/\w+/models/", f"/mnt/{device}/models/", content)

        if updated != content:
            script_path.write_text(updated, encoding="utf-8")
            log_success(f"已更新模型路径为 /mnt/{device}/models/")
        else:
            log_info("模型路径无需更新")

        return True
    except Exception as e:
        log_error(f"更新模型路径失败: {e}")
        return False


def run_benchmark(model_type: str = "moe", report_subdir: Optional[Path] = None) -> Optional[Path]:
    """
    执行基准测试脚本

    参数:
        model_type: 模型类型 (moe/moe_quant/moe_quant_a3/moe_mix/quant/all)
        report_subdir: 报告目录路径。传入则复用, 否则按当前版本+日期新建。

    返回:
        报告目录路径，失败返回 None

    说明:
        测试结果会保存到 daily_reports/YYYYMMDD/ 目录下
    """
    log_info(f"开始执行基准测试 (模型类型: {model_type})...")

    # 检测并更新模型路径
    device = detect_model_device()
    if device:
        update_model_paths(device)
    else:
        log_warning("使用脚本中默认的模型路径")

    # 报告目录: 传入则复用, 否则按当前版本+日期新建
    if report_subdir is None:
        current_version = get_xlite_version()
        current_date = datetime.now().strftime("%Y%m%d")
        report_subdir = REPORT_DIR / f"xlite-{current_version}-{current_date}"
    report_subdir.mkdir(parents=True, exist_ok=True)

    # 将日志文件移动到报告目录
    move_log_file(report_subdir)

    try:
        # 设置环境变量，用于覆盖脚本中的输出目录
        env = os.environ.copy()
        env["MAIN_OUTPUT_DIR_OVERRIDE"] = str(report_subdir)

        # 根据 NPU-SMI 版本决定是否禁用 XCCL
        if should_disable_xccl():
            env["XLITE_DISABLE_XCCL"] = "True"

        # 执行基准测试脚本
        result = subprocess.run(
            ["bash", "online_server_compare.sh", model_type], cwd=E2E_DIR, env=env, capture_output=True, text=True
        )

        if result.returncode != 0:
            log_error(f"基准测试执行失败: {result.stderr}")
            return None

        log_success("基准测试执行完成!")
        return report_subdir

    except Exception as e:
        log_error(f"基准测试执行异常: {e}")
        return None


# ====================== 离线 bench 测试执行函数 ======================
def run_offline_bench(report_subdir: Optional[Path] = None) -> Optional[Path]:
    """
    执行离线 bench 测试 (run.sh bench 模式)

    用 run.sh 跑 run_glm5_w8a8 的 bench 模式，只跑 OFFLINE_BENCH_N_LAYERS 层。

    参数:
        report_subdir: 报告目录路径。传入则复用, 否则按当前版本+日期新建。

    返回:
        报告目录路径，失败返回 None

    说明:
        每个场景的完整 stdout 保存到报告目录下的 offline_bench_<scenario>.log，
        供 parse_offline_bench_report 解析。
    """
    log_info(
        f"开始执行离线 bench 测试 (模型: {OFFLINE_BENCH_MODEL}, 层数: {OFFLINE_BENCH_N_LAYERS}, "
        f"场景数: {len(OFFLINE_BENCH_SCENARIOS)})..."
    )

    # 检测模型存储设备以确定 models_base_path
    device = detect_model_device()
    if device:
        models_base_path = f"/mnt/{device}/models"
        log_info(f"模型路径: {models_base_path}")
    else:
        models_base_path = "/mnt/nvme0n1/models"
        log_warning(f"未检测到模型设备，使用默认路径: {models_base_path}")

    # 报告目录: 传入则复用, 否则按当前版本+日期新建。
    # current_date 与目录日期保持一致 (复用时从目录名提取, 保证日志文件名日期一致)
    if report_subdir is None:
        current_version = get_xlite_version()
        current_date = datetime.now().strftime("%Y%m%d")
        report_subdir = REPORT_DIR / f"xlite-{current_version}-{current_date}"
    else:
        match = re.search(r"(\d{8})$", str(report_subdir))
        current_date = match.group(1) if match else datetime.now().strftime("%Y%m%d")
    report_subdir.mkdir(parents=True, exist_ok=True)

    # 将日志文件移动到报告目录
    move_log_file(report_subdir)

    try:
        # run.sh 内部 cd "$(dirname "$0")/.." 到 XLITE_DIR 拉取仓根, 故以 XLITE_DIR 为 cwd
        for input_len, output_len, bs in OFFLINE_BENCH_SCENARIOS:
            scenario_tag = f"in{input_len}_out{output_len}_bs{bs}"
            log_file = report_subdir / f"offline_bench_{OFFLINE_BENCH_MODEL}_{scenario_tag}_{current_date}.log"
            log_info(
                f"执行离线 bench: in={input_len} out={output_len} bs={bs} "
                f"(iters={OFFLINE_BENCH_ITERS}) -> {log_file.name}"
            )

            env = os.environ.copy()
            env["XLITE_N_LAYERS"] = str(OFFLINE_BENCH_N_LAYERS)
            # 前置 site-packages: cwd=拉取仓根, 其下 xlite/ 源码会遮蔽已装 wheel (含 _C),
            # 前置 site-packages 即可; XLITE_DIR 留后让 tests 包仍可达。
            xlite_lib_path = get_xlite_lib_path()
            if xlite_lib_path and xlite_lib_path.exists():
                site_packages_dir = str(xlite_lib_path.parent)
                existing_pp = env.get("PYTHONPATH", "")
                env["PYTHONPATH"] = (
                    f"{site_packages_dir}:{XLITE_DIR}:{existing_pp}"
                    if existing_pp
                    else f"{site_packages_dir}:{XLITE_DIR}"
                )
            env["XLITE_NPUS"] = OFFLINE_BENCH_NPUS  # -> ASCEND_RT_VISIBLE_DEVICES
            env["XLITE_MOE_EP_SIZE"] = str(OFFLINE_BENCH_MOE_EP_SIZE)  # 须等于卡数
            env["XLITE_GLM5_W8A8_CKPT"] = OFFLINE_BENCH_CKPT_DIR
            env["RUN_MODE"] = "bench"
            env["MODEL"] = f"run_{OFFLINE_BENCH_MODEL}"
            env["BENCH_IT"] = str(OFFLINE_BENCH_ITERS)
            env["BENCH_BS"] = str(bs)
            env["BENCH_N1"] = str(input_len)
            env["BENCH_N2"] = str(output_len)
            if should_disable_xccl():
                env["XLITE_DISABLE_XCCL"] = "True"

            result = subprocess.run(
                ["bash", "tests/run.sh", models_base_path],
                cwd=str(XLITE_DIR),
                env=env,
                capture_output=True,
                text=True,
            )

            # 保存完整 stdout (含配置打印与 Iter 行) 供解析与排查
            log_file.write_text(result.stdout, encoding="utf-8")

            if result.returncode != 0:
                log_error(
                    f"离线 bench 场景 {scenario_tag} 执行失败 (返回码 {result.returncode}): {result.stderr[-500:]}"
                )
                continue

            # OOM 兜底: torchrun 吞掉 OOM 的 SIGTERM(-15) 仍返回 0, 仅看 returncode
            # 会把"空日志 + null 指标"误报成功。校验日志无 'tokens/s' 行则判失败。
            if "tokens/s" not in result.stdout:
                err_tail = (result.stderr or "")[-800:]
                log_error(
                    f"离线 bench 场景 {scenario_tag} 未产出有效结果 "
                    f"(返回码 {result.returncode} 但日志无 'tokens/s' 行, 疑似 OOM/崩溃):\n{err_tail}"
                )
                continue

            log_success(f"离线 bench 场景 {scenario_tag} 执行完成!")

        log_success("离线 bench 测试执行完成!")
        return report_subdir

    except Exception as e:
        log_error(f"离线 bench 测试执行异常: {e}")
        return None


# ====================== 离线 bench 报告解析函数 ======================
def parse_offline_bench_report(report_path: Path) -> dict:
    """
    解析离线 bench 报告文件 (run.sh bench 模式的 stdout)

    返回指标字典 (input_len/output_len/batch_size 从文件名提取, 其余从 Iter 行解析)。
    首轮含 warmup 偏差大, 丢弃; 对其余轮 total_ms / decode_tps / step_latency_ms
    取算术平均; generated_tokens 各轮稳定, 取末轮。
    decode_tps: prefill(output_len==1) 取日志 avg 字段 (prefilled/时间),
    decode 取 tokens/s 字段 (generated/时间)。
    """
    metrics = {
        "model": OFFLINE_BENCH_MODEL,
        "timestamp": "",
        "batch_size": None,
        "input_len": None,
        "output_len": None,
        "n_layers": OFFLINE_BENCH_N_LAYERS,
        "iters": OFFLINE_BENCH_ITERS,
        "decode_tps": None,
        "step_latency_ms": None,
        "total_ms": None,
    }

    if not report_path.exists():
        return metrics

    content = report_path.read_text(encoding="utf-8")
    metrics["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 场景信息从文件名解析:
    # offline_bench_<model>_in<input>_out<output>_bs<bs>_<date>.log
    name_match = re.search(r"offline_bench_\w+_in(\d+)_out(\d+)_bs(\d+)", report_path.name)
    if name_match:
        metrics["input_len"] = int(name_match.group(1))
        metrics["output_len"] = int(name_match.group(2))
        metrics["batch_size"] = int(name_match.group(3))

    # Iter 行: "Iter X/Y: ... generated N tokens in D.DD ms. avg: A.AA | B.BB tokens/s @ C.CC ms step bs: D"
    # 捕获 g1=generated g2=total_ms g3=avg(prefill) g4=tokens/s(decode) g5=step; .*? 兼容 DP 的 (≈... agg)
    iter_pattern = re.compile(
        r"Iter \d+/\d+:.*?generated (\d+).*?tokens in ([\d.]+) ms\."
        r".*?avg: ([\d.]+).*?([\d.]+) tokens/s.*?@ ([\d.]+) ms step bs: (\d+)"
    )
    rounds = list(iter_pattern.finditer(content))

    # 丢弃首轮(warmup), 对其余轮取平均; 只有一轮时退化为取该轮。
    if rounds:
        stable = rounds[1:] if len(rounds) > 1 else rounds
        metrics["generated_tokens"] = int(rounds[-1].group(1))  # 各轮稳定, 取末轮
        metrics["total_ms"] = sum(float(m.group(2)) for m in stable) / len(stable)
        # prefill(output_len==1) 取 avg(g3); decode 取 tokens/s(g4)
        tps_group = 3 if metrics.get("output_len") == 1 else 4
        metrics["decode_tps"] = sum(float(m.group(tps_group)) for m in stable) / len(stable)
        metrics["step_latency_ms"] = sum(float(m.group(5)) for m in stable) / len(stable)

    return metrics


# ====================== 离线 bench 性能对比函数 ======================
def compare_offline_metrics(current: dict, baseline: dict) -> dict:
    """
    对比当前版本和基线版本的离线 bench 性能指标

    返回 row_changes 风格的对比结果 (与 compare_metrics 对齐)。
    decode_tps 越高越好, step_latency_ms / total_ms 越低越好; 变化超过阈值记为劣化/提升。
    阈值用 OFFLINE_DEGRADATION_THRESHOLD (与在线的 DEGRADATION_THRESHOLD 分离)。
    """
    comparison = {
        "date_current": current.get("timestamp", ""),
        "date_baseline": baseline.get("timestamp", ""),
        "model": current.get("model", ""),
        "row_changes": [],
    }

    if not baseline.get("decode_tps") and not baseline.get("step_latency_ms"):
        comparison["error"] = "缺少基线对比数据"
        return comparison

    changes = {}
    # decode_tps 越高越好
    for metric in ["decode_tps", "step_latency_ms", "total_ms"]:
        current_val = current.get(metric)
        baseline_val = baseline.get(metric)

        if current_val is None or baseline_val is None or baseline_val == 0:
            changes[metric] = {
                "current": current_val,
                "baseline": baseline_val,
                "change_ratio": None,
                "is_degradation": False,
                "is_improvement": False,
            }
            continue

        change_ratio = (current_val - baseline_val) / baseline_val
        if metric == "decode_tps":
            # 越高越好: 下降为劣化，上升为提升
            is_degradation = change_ratio < -OFFLINE_DEGRADATION_THRESHOLD
            is_improvement = change_ratio > OFFLINE_DEGRADATION_THRESHOLD
        else:
            # 越低越好 (延迟): 上升为劣化，下降为提升
            is_degradation = change_ratio > OFFLINE_DEGRADATION_THRESHOLD
            is_improvement = change_ratio < -OFFLINE_DEGRADATION_THRESHOLD

        changes[metric] = {
            "current": current_val,
            "baseline": baseline_val,
            "change_ratio": change_ratio,
            "is_degradation": is_degradation,
            "is_improvement": is_improvement,
        }

    has_significant_change = any(c.get("is_degradation") or c.get("is_improvement") for c in changes.values())
    if has_significant_change:
        comparison["row_changes"].append(
            {
                "concurrency": current.get("batch_size"),
                "service": "prefill" if current.get("output_len") == 1 else "decode",
                "input_len": current.get("input_len"),
                "output_len": current.get("output_len"),
                "changes": changes,
            }
        )

    return comparison


# ====================== 离线 bench 历史报告查找函数 ======================
def get_previous_offline_report(
    input_len: int, output_len: int, batch_size: int
) -> tuple[Optional[Path], Optional[str]]:
    """
    获取基线版本指定离线 bench 场景的报告文件路径

    参数:
        input_len: 输入长度
        output_len: 输出长度
        batch_size: 批大小

    返回:
        (报告文件路径, 版本号)，不存在则返回 (None, None)

    说明:
        - 基线目录: xlite-{release_tag} (无日期，如 xlite-0.2.0rc0)，用最近发布 tag
          作为 key，在两次 tag 之间的所有 dev 构建间稳定，故每日构建都对比同一基线。
        - 当前目录: xlite-{dev_version}-{date} (有日期)，用 dev 版本号+日期。
    """
    baseline_version = get_last_release_tag()

    if baseline_version == "unknown":
        log_warning("无法获取基线版本号")
        return None, None

    baseline_version_dir = REPORT_DIR / f"xlite-{baseline_version}"
    if not baseline_version_dir.exists():
        log_warning(f"基线目录不存在: {baseline_version_dir}")
        return None, baseline_version

    pattern = f"offline_bench_{OFFLINE_BENCH_MODEL}_in{input_len}_out{output_len}_bs{batch_size}_*.log"
    reports = list(baseline_version_dir.glob(pattern))
    if reports:
        return reports[0], baseline_version

    log_warning(f"基线目录中未找到离线 bench 场景 in={input_len} out={output_len} bs={batch_size} 的报告")
    return None, baseline_version


# ====================== 离线 bench 报告生成函数 ======================
def generate_offline_report(
    model_name: str,
    comparisons: list[dict],
    report_dir: Optional[Path] = None,
    model_index: int = 0,
    total_models: int = 1,
) -> str:
    """
    为离线 bench 生成测试报告

    参数:
        model_name: 模型名称
        comparisons: 该模型的对比结果字典列表
        report_dir: 报告保存目录路径
        model_index: 当前模型索引（从0开始）
        total_models: 总模型数

    返回:
        格式化的报告字符串
    """
    lines = []

    vllm_ascend_version = get_vllm_ascend_version()
    xlite_version = get_xlite_version()
    baseline_info = get_baseline_info()

    current_date = "unknown"
    if report_dir:
        match = re.search(r"xlite-[\w.\+]+-(\d{8})$", str(report_dir))
        if match:
            current_date = match.group(1)

    lines.append(
        f'<font color="blue"><b>xlite {current_date} 离线 bench 性能测试报告 '
        f"[{model_index + 1}/{total_models}]</b></font>"
    )
    lines.append("")

    total_rows_with_change = 0
    total_degradations = 0
    total_improvements = 0

    detail_lines = []
    scenario_idx = 0
    for comparison in comparisons:
        row_changes = comparison.get("row_changes", [])
        if row_changes:
            scenario_idx += 1
            for row in row_changes:
                service = row["service"]
                concurrency = row["concurrency"]
                changes = row["changes"]
                input_len = row.get("input_len")
                output_len = row.get("output_len")

                detail_lines.append(f"{scenario_idx}. {service} in={input_len} out={output_len} bs={concurrency}")

                if comparison.get("error"):
                    detail_lines.append(f"  ⚠️  {comparison['error']}")
                    detail_lines.append("")
                    continue

                detail_lines.append(" | 场景 | throughput(tokens/s) | step latency(ms) | total(ms) |")
                detail_lines.append(" |------|------|------|------|")

                row_degradations = sum(1 for c in changes.values() if c.get("is_degradation"))
                row_improvements = sum(1 for c in changes.values() if c.get("is_improvement"))

                total_degradations += row_degradations
                total_improvements += row_improvements
                total_rows_with_change += 1

                def format_value(metric_name, change_info):
                    current_v = change_info["current"]
                    baseline_v = change_info["baseline"]
                    change_ratio = change_info["change_ratio"]

                    if current_v is None or baseline_v is None:
                        return "N/A"

                    if change_info["is_degradation"] or change_info["is_improvement"]:
                        change_percent = change_ratio * 100
                        return f"{current_v:.2f} ({change_percent:+.0f}%)"
                    else:
                        return f"{current_v:.2f}"

                metrics_order = ["decode_tps", "step_latency_ms", "total_ms"]
                formatted_parts = []
                i = 0
                while i < len(metrics_order):
                    metric = metrics_order[i]
                    change_info = changes[metric]

                    if change_info.get("is_degradation"):
                        red_parts = []
                        while i < len(metrics_order) and changes[metrics_order[i]].get("is_degradation"):
                            red_parts.append(format_value(metrics_order[i], changes[metrics_order[i]]))
                            i += 1
                        formatted_parts.append(f'<font color="red"><b>{" | ".join(red_parts)}</b></font>')
                    elif change_info.get("is_improvement"):
                        green_parts = []
                        while i < len(metrics_order) and changes[metrics_order[i]].get("is_improvement"):
                            green_parts.append(format_value(metrics_order[i], changes[metrics_order[i]]))
                            i += 1
                        formatted_parts.append(f'<font color="green"><b>{" | ".join(green_parts)}</b></font>')
                    else:
                        formatted_parts.append(format_value(metric, change_info))
                        i += 1

                detail_lines.append(f"  | {service} bs={concurrency} | {' | '.join(formatted_parts)} |")
            detail_lines.append("")

    status_icon = "⚠️" if total_degradations > 0 else "✅"
    if total_degradations > 0:
        status_text = f"发现性能劣化(超过{OFFLINE_DEGRADATION_THRESHOLD * 100:.1f}%阈值)，请关注！"
    elif total_improvements > 0:
        status_text = "发现性能优化"
    else:
        status_text = "性能正常，无劣化"

    ip_address = os.environ.get("MACHINE_IP", "")
    report_path = f"{ip_address}:{report_dir}" if report_dir else ""

    lines.append(
        f'📊 <font color="blue"><b>{model_name}</b></font> 离线 bench 性能统计: '
        f"<b>劣化项: {total_degradations} | 提升项: {total_improvements}</b>"
    )
    lines.append(f"【当前版本】 vllm-ascend 版本: {vllm_ascend_version}, xlite 版本: {xlite_version}")
    lines.append(
        f"【对比基线】 vllm-ascend 版本: {baseline_info['vllm_ascend_version']}, xlite 版本: {baseline_info['version']}"
    )
    lines.append(f"{status_icon} {status_text}")

    if report_path:
        lines.append(f"报告保存路径: {report_path}")
        lines.append("")

    if total_degradations > 0 or total_improvements > 0:
        lines.extend(detail_lines)

    return "\n".join(lines)


# ====================== 报告解析函数 ======================
def parse_benchmark_report(report_path: Path) -> dict:
    """
    解析基准测试报告文件

    参数:
        report_path: 报告文件路径

    返回:
        解析后的指标字典，结构如下:
        {
            "model": "模型名称",
            "timestamp": "报告生成时间",
            "scenarios": [
                {
                    "concurrency": 并发数,
                    "baseline": {"ttft_avg": ..., "qps": ..., ...},
                    "xlite_full": {...},
                    "xlite_decode_only": {...}
                },
                ...
            ]
        }

    说明:
        解析的指标包括:
        - TTFT (Time To First Token): 首token延迟
        - TPOT (Time Per Output Token): 每个输出token的时间
        - QPS: 每秒请求数
        - Output Speed: 输出token速度
    """
    # 初始化返回结构
    metrics = {"model": "", "timestamp": "", "scenarios": []}

    if not report_path.exists():
        return metrics

    content = report_path.read_text(encoding="utf-8")

    # 提取模型名称 (从标题行)
    model_match = re.search(r"## (.+?) TPS", content)
    if model_match:
        metrics["model"] = model_match.group(1).strip()

    # 提取报告生成时间
    time_match = re.search(r"Report Generated Time: (.+)", content)
    if time_match:
        metrics["timestamp"] = time_match.group(1).strip()

    # 定义表格行的正则表达式
    # 格式: | concurrency | item_type | ttft_avg | ttft_p99 | tpot_avg | tpot_p99 | qps | output_speed |
    table_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*(baseline-aclgraph|xlite-full|xlite-decode-only|diff1|diff2)\s*\|"
        r"\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|"
        r"\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|"
    )

    # 遍历匹配的每一行
    current_scenario = None
    for match in table_pattern.finditer(content):
        concurrency = int(match.group(1))  # 并发数
        item_type = match.group(2)  # 数据类型
        ttft_avg = match.group(3)  # TTFT 平均值
        ttft_p99 = match.group(4)  # TTFT P99值
        tpot_avg = match.group(5)  # TPOT 平均值
        tpot_p99 = match.group(6)  # TPOT P99值
        qps = match.group(7)  # 每秒请求数
        output_speed = match.group(8)  # 输出token速度

        # 只处理实际数据行，跳过 diff 行
        if item_type in ["baseline-aclgraph", "xlite-full", "xlite-decode-only"]:
            # 如果是新并发级别，保存上一个场景并创建新场景
            if current_scenario is None or current_scenario["concurrency"] != concurrency:
                if current_scenario:
                    metrics["scenarios"].append(current_scenario)
                current_scenario = {
                    "concurrency": concurrency,
                    "baseline": {},
                    "xlite_full": {},
                    "xlite_decode_only": {},
                }

            # 安全转换为浮点数
            def safe_float(val):
                try:
                    return float(val) if val != "N/A" else None
                except Exception:
                    return None

            # 构建指标数据
            data = {
                "ttft_avg": safe_float(ttft_avg),
                "ttft_p99": safe_float(ttft_p99),
                "tpot_avg": safe_float(tpot_avg),
                "tpot_p99": safe_float(tpot_p99),
                "qps": safe_float(qps),
                "output_speed": safe_float(output_speed),
            }

            # 根据类型存储到对应字段
            if item_type == "baseline-aclgraph":
                current_scenario["baseline"] = data
            elif item_type == "xlite-full":
                current_scenario["xlite_full"] = data
            elif item_type == "xlite-decode-only":
                current_scenario["xlite_decode_only"] = data

    # 添加最后一个场景
    if current_scenario:
        metrics["scenarios"].append(current_scenario)

    return metrics


# ====================== 性能对比函数 ======================
def compare_metrics(current: dict, baseline: dict) -> dict:
    """
    对比当前版本和基线版本的性能指标

    参数:
        current: 当前版本指标数据
        baseline: 基线版本指标数据

    返回:
        对比结果字典，结构如下:
        {
            "date_current": "当前版本测试时间",
            "date_baseline": "基线版本测试时间",
            "model": "模型名称",
            "input_len": 输入长度,
            "output_len": 输出长度,
            "row_changes": [...]  # 整行变化列表
        }

    说明:
        - 对于 QPS 和 Output Speed: 下降超过阈值视为劣化
        - 对于 TTFT 和 TPOT: 上升超过阈值视为劣化 (延迟越高越差)
        - 只要整行中有一项指标变化超过阈值，就记录整行数据
    """
    # 初始化对比结果
    comparison = {
        "date_current": current.get("timestamp", ""),
        "date_baseline": baseline.get("timestamp", ""),
        "model": current.get("model", ""),
        "input_len": current.get("input_len", "N/A"),
        "output_len": current.get("output_len", "N/A"),
        "row_changes": [],
    }

    # 检查数据完整性
    if not current.get("scenarios") or not baseline.get("scenarios"):
        comparison["error"] = "缺少对比数据"
        return comparison

    # 将场景列表转换为字典，便于按并发数查找
    current_scenarios = {s["concurrency"]: s for s in current["scenarios"]}
    baseline_scenarios = {s["concurrency"]: s for s in baseline["scenarios"]}

    # 遍历当前版本所有场景
    for concurrency, current_scenario in current_scenarios.items():
        # 跳过基线版本没有的场景
        if concurrency not in baseline_scenarios:
            continue

        baseline_scenario = baseline_scenarios[concurrency]

        # 对比两种 xlite 服务
        for service_type in ["xlite_full", "xlite_decode_only"]:
            service_name = "full" if service_type == "xlite_full" else "decode-only"

            current_data = current_scenario.get(service_type, {})
            baseline_data = baseline_scenario.get(service_type, {})

            # 检查是否有任何指标变化超过阈值
            has_significant_change = False
            changes = {}

            # 对比所有指标（始终计算所有指标，但TTFT和TPOT P99的劣化判断受配置控制）
            all_metrics = ["ttft_avg", "ttft_p99", "tpot_avg", "tpot_p99", "output_speed"]
            for metric in all_metrics:
                current_val = current_data.get(metric)
                baseline_val = baseline_data.get(metric)

                # 跳过无效数据
                if current_val is None or baseline_val is None or baseline_val == 0:
                    changes[metric] = {
                        "current": current_val,
                        "baseline": baseline_val,
                        "change_ratio": None,
                        "is_degradation": False,
                        "is_improvement": False,
                    }
                    continue

                # 计算变化比例
                change_ratio = (current_val - baseline_val) / baseline_val

                # 判断是否为劣化或提升
                # 对于 QPS 和 Output Speed: 下降为劣化，上升为提升
                # 对于 TTFT 和 TPOT: 上升为劣化，下降为提升
                # TTFT指标的判断受COMPARE_TTFT_METRICS配置控制
                # TPOT P99指标的判断受COMPARE_TPOT_P99_METRICS配置控制
                is_ttft_metric = metric in ["ttft_avg", "ttft_p99"]
                is_tpot_p99_metric = metric == "tpot_p99"
                if metric in ["qps", "output_speed"]:
                    is_degradation = change_ratio < -DEGRADATION_THRESHOLD
                    is_improvement = change_ratio > DEGRADATION_THRESHOLD
                elif is_ttft_metric and not COMPARE_TTFT_METRICS:
                    # TTFT指标不参与对比判断时，始终为False
                    is_degradation = False
                    is_improvement = False
                elif is_tpot_p99_metric and not COMPARE_TPOT_P99_METRICS:
                    # TPOT P99指标不参与对比判断时，始终为False
                    is_degradation = False
                    is_improvement = False
                else:
                    is_degradation = change_ratio > DEGRADATION_THRESHOLD
                    is_improvement = change_ratio < -DEGRADATION_THRESHOLD

                changes[metric] = {
                    "current": current_val,
                    "baseline": baseline_val,
                    "change_ratio": change_ratio,
                    "is_degradation": is_degradation,
                    "is_improvement": is_improvement,
                }

                # 如果有任何指标变化超过阈值，标记整行
                if is_degradation or is_improvement:
                    has_significant_change = True

            # 如果整行有显著变化，记录整行数据
            if has_significant_change:
                row_change = {"concurrency": concurrency, "service": service_name, "changes": changes}
                comparison["row_changes"].append(row_change)

    return comparison


# ====================== 报告生成函数 ======================
def generate_model_report(
    model_name: str,
    model_comparisons: list[dict],
    report_dir: Optional[Path] = None,
    model_index: int = 0,
    total_models: int = 1,
) -> str:
    """
    为单个模型生成测试报告

    参数:
        model_name: 模型名称
        model_comparisons: 该模型的对比结果字典列表
        report_dir: 报告保存目录路径
        model_index: 当前模型索引（从0开始）
        total_models: 总模型数

    返回:
        格式化的报告字符串（HTML格式）
    """
    lines = []

    vllm_ascend_version = get_vllm_ascend_version()
    xlite_version = get_xlite_version()
    baseline_info = get_baseline_info()

    # 获取当前测试日期（从report_dir中提取）
    current_date = "unknown"
    if report_dir:
        match = re.search(r"xlite-[\w.\+]+-(\d{8})$", str(report_dir))
        if match:
            current_date = match.group(1)

    lines.append(
        f'<font color="blue"><b>xlite {current_date} 在线性能测试报告 [{model_index + 1}/{total_models}]</b></font>'
    )
    lines.append("")

    # 汇总统计
    total_rows_with_change = 0
    total_degradations = 0
    total_improvements = 0

    # 按输入输出长度分别显示（按input长度排序）
    sorted_comparisons = sorted(model_comparisons, key=lambda x: x.get("input_len", 0))

    # 收集详细数据内容
    detail_lines = []
    scenario_idx = 0
    for comparison in sorted_comparisons:
        row_changes = comparison.get("row_changes", [])
        if row_changes:
            scenario_idx += 1
            input_len = comparison.get("input_len", "N/A")
            output_len = comparison.get("output_len", "N/A")

            detail_lines.append(f"{scenario_idx}. input={input_len}, output={output_len}")

            if comparison.get("error"):
                detail_lines.append(f"  ⚠️  {comparison['error']}")
                detail_lines.append("")
                continue

            # 显示表格头
            detail_lines.append(
                " | 并发 | 服务 | TTFT Avg(ms) | TTFT P99 | TPOT Avg(ms) | TPOT P99 | Throughput(toks/s) |"
            )
            detail_lines.append(" |------|------|------|------|------|------|------|")

            # 显示每一行数据
            for row in row_changes:
                concurrency = row["concurrency"]
                service = row["service"]
                changes = row["changes"]

                # 统计劣化和提升数量
                row_degradations = sum(1 for c in changes.values() if c.get("is_degradation"))
                row_improvements = sum(1 for c in changes.values() if c.get("is_improvement"))

                total_degradations += row_degradations
                total_improvements += row_improvements
                total_rows_with_change += 1

                # 格式化每个指标值
                def format_value(metric_name, change_info):
                    current = change_info["current"]
                    baseline = change_info["baseline"]
                    change_ratio = change_info["change_ratio"]

                    if current is None or baseline is None:
                        return "N/A"

                    if change_info["is_degradation"]:
                        # 劣化：红色标记
                        change_percent = change_ratio * 100
                        return f"{current:.0f} ({change_percent:+.0f}%)"
                    elif change_info["is_improvement"]:
                        # 提升：绿色标记
                        change_percent = change_ratio * 100
                        return f"{current:.0f} ({change_percent:+.0f}%)"
                    else:
                        # 正常：只显示当前值
                        return f"{current:.0f}"

                # 构建表格行，合并连续的相同颜色标签
                metrics_order = ["ttft_avg", "ttft_p99", "tpot_avg", "tpot_p99", "output_speed"]
                formatted_parts = []
                i = 0

                while i < len(metrics_order):
                    metric = metrics_order[i]
                    change_info = changes[metric]

                    if change_info.get("is_degradation"):
                        # 收集连续的红色指标
                        red_parts = []
                        while i < len(metrics_order) and changes[metrics_order[i]].get("is_degradation"):
                            red_parts.append(format_value(metrics_order[i], changes[metrics_order[i]]))
                            i += 1
                        formatted_parts.append(f'<font color="red"><b>{" | ".join(red_parts)}</b></font>')
                    elif change_info.get("is_improvement"):
                        # 收集连续的绿色指标
                        green_parts = []
                        while i < len(metrics_order) and changes[metrics_order[i]].get("is_improvement"):
                            green_parts.append(format_value(metrics_order[i], changes[metrics_order[i]]))
                            i += 1
                        formatted_parts.append(f'<font color="green"><b>{" | ".join(green_parts)}</b></font>')
                    else:
                        # 正常指标
                        formatted_parts.append(format_value(metric, change_info))
                        i += 1

                row_str = f"  | {concurrency} | {service} | {' | '.join(formatted_parts)} |"
                detail_lines.append(row_str)
            detail_lines.append("")

    # 汇总摘要（作为折叠标题）
    status_icon = "⚠️" if total_degradations > 0 else "✅"
    if total_degradations > 0:
        status_text = "发现性能劣化(超过5%阈值)，请关注！"
    elif total_improvements > 0:
        status_text = "发现性能优化"
    else:
        status_text = "性能正常，无劣化"

    ip_address = os.environ.get("MACHINE_IP", "")
    report_path = f"{ip_address}:{report_dir}" if report_dir else ""

    # 使用HTML折叠样式（内联样式）
    lines.append(
        f'📊 <font color="blue"><b>{model_name}</b></font> 性能统计: <b>劣化项: {total_degradations} | 提升项: {total_improvements}</b>'
    )
    lines.append(f"【当前版本】 vllm-ascend 版本: {vllm_ascend_version}, xlite 版本: {xlite_version}")
    lines.append(
        f"【对比基线】 vllm-ascend 版本: {baseline_info['vllm_ascend_version']}, xlite 版本: {baseline_info['version']}"
    )
    lines.append(f"{status_icon} {status_text}")

    # 添加报告路径
    if report_path:
        lines.append(f"报告保存路径: {report_path}")
        lines.append("")

    # 添加详细数据（仅在性能异常或提升时显示）
    if total_degradations > 0 or total_improvements > 0:
        lines.extend(detail_lines)

    return "\n".join(lines)


def build_no_change_report(
    report_dir: Optional[Path],
    model_names: Optional[list[str]] = None,
    is_offline: bool = False,
    failure_count: int = 0,
) -> str:
    """
    所有模型性能均无显著变化时生成汇总通知

    参数:
        report_dir: 报告保存目录路径
        model_names: 本次测试的模型名称列表
        is_offline: 是否为离线 bench 测试

    返回:
        格式化的报告字符串
    """
    vllm_ascend_version = get_vllm_ascend_version()
    xlite_version = get_xlite_version()
    baseline_info = get_baseline_info()

    # 获取当前测试日期（从report_dir中提取）
    current_date = "unknown"
    if report_dir:
        match = re.search(r"xlite-[\w.\+]+-(\d{8})$", str(report_dir))
        if match:
            current_date = match.group(1)

    if is_offline:
        header = f"xlite {current_date} 离线 bench 性能测试报告"
    else:
        header = f"xlite {current_date} 在线性能测试报告"
    lines = [f'<font color="blue"><b>{header}</b></font>', ""]

    lines.append(f"【当前版本】 vllm-ascend 版本: {vllm_ascend_version}, xlite 版本: {xlite_version}")
    lines.append(
        f"【对比基线】 vllm-ascend 版本: {baseline_info['vllm_ascend_version']}, xlite 版本: {baseline_info['version']}"
    )
    models_text = ""
    if model_names:
        models_text = f" (共 {len(model_names)} 个模型: {', '.join(model_names)})"
    if failure_count > 0:
        # 有失败场景时, "无变化"仅覆盖成功完成的场景, 避免与单独的失败通知矛盾、误导排查
        lines.append(
            f"✅ 成功完成的场景性能均无变化{models_text} "
            f"(另有 {failure_count} 个场景未能正常完成, 见单独的失败通知)"
        )
    else:
        lines.append(f"✅ 测试模型性能均无变化{models_text}")

    # 添加报告路径
    ip_address = os.environ.get("MACHINE_IP", "")
    report_path = f"{ip_address}:{report_dir}" if report_dir else ""
    if report_path:
        lines.append(f"报告保存路径: {report_path}")
        lines.append("")

    return "\n".join(lines)


# ====================== 历史报告查找函数 ======================
def extract_model_info(report_path: Path) -> Optional[dict]:
    """
    从报告文件名中提取模型信息

    参数:
        report_path: 报告文件路径

    返回:
        模型信息字典，提取失败返回 None
        {
            "model_name": "Qwen3-32B",
            "input_len": 512,
            "output_len": 512
        }

    文件名格式: benchmark_comparison_Qwen3-32B_input512_output512_tp8_20260331.log
    """
    match = re.search(r"benchmark_comparison_(.+?)_input(\d+)_output(\d+)", report_path.name)
    if match:
        return {"model_name": match.group(1), "input_len": int(match.group(2)), "output_len": int(match.group(3))}
    return None


def get_previous_version_report(
    model_name: str, input_len: int, output_len: int
) -> tuple[Optional[Path], Optional[str]]:
    """
    获取基线版本指定模型的测试报告文件路径

    参数:
        model_name: 模型名称 (如 'Qwen3-32B')
        input_len: 输入长度
        output_len: 输出长度

    返回:
        (报告文件路径, 版本号)，不存在则返回 (None, None)

    说明:
        - 基线目录: xlite-{release_tag} (无日期，如 xlite-0.2.0rc0)，用最近发布 tag
          作为 key，在两次 tag 之间的所有 dev 构建间稳定，故每日构建都对比同一基线。
        - 当前目录: xlite-{dev_version}-{date} (有日期，如 xlite-0.2.0rc1.dev146+...-20260828)。
    """
    baseline_version = get_last_release_tag()

    if baseline_version == "unknown":
        log_warning("无法获取基线版本号")
        return None, None

    # 基线目录使用最近发布 tag（不带日期）
    baseline_version_dir = REPORT_DIR / f"xlite-{baseline_version}"

    if not baseline_version_dir.exists():
        log_warning(f"基线目录不存在: {baseline_version_dir}")
        return None, baseline_version

    # 查找匹配模型名称、输入输出长度的报告文件
    pattern = f"benchmark_comparison_{model_name}_input{input_len}_output{output_len}_*.log"
    reports = list(baseline_version_dir.glob(pattern))
    if reports:
        return reports[0], baseline_version

    log_warning(f"基线目录中未找到模型 {model_name} (input={input_len}, output={output_len}) 的报告")
    return None, baseline_version


def get_baseline_info() -> dict:
    """
    获取基线版本信息

    返回:
        基线版本信息字典:
        {
            "version": "版本号",
            "date": "日期",
            "vllm_ascend_version": "vllm-ascend版本号"
        }
    """
    baseline_version = get_last_release_tag()
    baseline_version_dir = REPORT_DIR / f"xlite-{baseline_version}"

    baseline_info = {"version": baseline_version, "date": "unknown", "vllm_ascend_version": "unknown"}

    if not baseline_version_dir.exists():
        return baseline_info

    # 尝试从metrics JSON文件中获取信息
    metrics_files = list(baseline_version_dir.glob("metrics_*.json"))
    if metrics_files:
        try:
            metrics = load_metrics_json(metrics_files[0])
            if metrics.get("timestamp"):
                baseline_info["date"] = metrics["timestamp"]
        except Exception as e:
            log_warning(f"读取基线metrics文件失败: {e}")

    # 尝试从daily_summary.txt中获取commit信息
    summary_file = baseline_version_dir / "daily_summary.txt"
    if summary_file.exists():
        try:
            content = summary_file.read_text(encoding="utf-8")
            match = re.search(r"vllm-ascend 版本:\s*(\S+)", content)
            if match:
                baseline_info["vllm_ascend_version"] = match.group(1)
        except Exception as e:
            log_warning(f"读取基线summary文件失败: {e}")

    return baseline_info


def seed_baseline_if_release(report_dir: Path) -> None:
    """
    若本次构建恰好处于某个发布 tag 上 (get_xlite_version() == 最近 tag，
    即 distance==0 的干净 tag 构建)，则把本次报告目录镜像为基线目录
    xlite-{tag}/，使后续两次 tag 之间的每日 dev 构建有基线可对比。

    基线目录采用覆盖式更新: 每次在 tag 上构建都刷新基线为最新结果，
    因此基线始终反映该 tag 的最新一次运行。仅在 tag 构建时触发，dev
    构建只写带日期的报告目录、不更新基线，避免基线被未发布代码污染。

    参数:
        report_dir: 本次运行的报告目录 (xlite-{dev_version}-{date})
    """
    current_version = get_xlite_version()
    release_tag = get_last_release_tag()

    # dev 构建 (当前版本带 .dev/.+g... 后缀，与最近 tag 不同): 不更新基线。
    # 浅克隆等无 tag 情形 get_last_release_tag 退化为当前版本，二者相等会
    # 误判为 tag 构建；用 git describe --exact-match 再确认一次。
    if current_version != release_tag:
        return
    try:
        exact = subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=str(XLITE_DIR),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if exact.returncode != 0:
            # HEAD 不在某个 tag 上 (dev 构建): 不更新基线。
            return
    except Exception as e:
        log_warning(f"基线种子: 确认 tag 失败, 跳过基线更新: {e}")
        return

    baseline_dir = REPORT_DIR / f"xlite-{release_tag}"
    log_info(f"检测到发布 tag 构建 ({release_tag})，刷新基线目录: {baseline_dir}")

    # 覆盖式更新: 先清旧基线再镜像本次结果，保证基线即本次运行。
    if baseline_dir.exists():
        shutil.rmtree(baseline_dir)
    shutil.copytree(report_dir, baseline_dir)
    log_success(f"基线已更新: {baseline_dir}")


# ====================== 报告处理辅助函数 ======================
# ---- 失败场景收集 ----
# 服务起不来 / 压测报错时, 在线脚本 (online_server_compare.sh) 不 exit 1, 而是
# 跳过该服务或并发度继续跑, 导致对应 result_*/ 目录缺失或日志不含完成关键字;
# process_data.py 在任一服务目录缺失时整体中止, 该场景的 benchmark_comparison_*.log
# 不会生成。这些失败不会被 _process_online_reports 反映 (它只看成功生成的对比报告),
# 故单独扫描三类失败: 服务未就绪 / 压测未完成(含"假成功") / 结果目录有但对比报告缺失。
# "假成功"指压测日志含完成关键字 "Request throughput (req/s)", 但 Failed requests != 0
# 或 Mean TPOT == 0 —— vllm bench 仍打印指标块(值多为 0), 实为压测实质失败(请求没拿到
# 正常响应/模型无 decode 输出), 不能当作有效性能结果。
# 此外, 各类失败若日志含显存 OOM 串 ("NPU out of memory" 等, 见 _detect_oom), 会在
# 失败描述中标注 "(疑似显存 OOM)", 便于在通知里直接定位根因而非笼统的"未完成"。
_ONLINE_SERVER_READY_KEYWORD = "Application startup complete"
_ONLINE_BENCH_COMPLETE_KEYWORD = "Request throughput (req/s)"


def _bench_log_anomaly(content: str) -> Optional[str]:
    """
    检测一份在线压测日志是否为"假成功" (有完成关键字但压测实质失败)。

    判据 (任一命中即判异常, 返回原因; 无异常返回 None):
    - Failed requests != 0: vllm bench 的请求失败数, 非零说明有请求没拿到响应
      (如 OOM、超时、连接被拒)。注意压测可能只跑了部分请求就出错, 但仍打印完整指标块。
    - Mean TPOT (ms) == 0: 每输出 token 平均耗时为 0, 说明模型实际没产生 decode 输出
      (Total generated tokens 极少), 指标块是 vllm bench 兜底打印的零值, 非真实性能。

    仅在日志已含完成关键字 (_ONLINE_BENCH_COMPLETE_KEYWORD) 时调用才有意义;
    未含完成关键字的日志由调用方直接判"压测未完成", 不进这里。
    """
    failed_match = re.search(r"Failed requests:\s+(\d+)", content)
    if failed_match and int(failed_match.group(1)) > 0:
        return f"请求失败 {int(failed_match.group(1))} 个"

    tpot_match = re.search(r"Mean TPOT \(ms\):\s+([\d.]+)", content)
    if tpot_match:
        try:
            if float(tpot_match.group(1)) == 0.0:
                return "Mean TPOT 为 0 (无有效输出)"
        except ValueError:
            pass

    return None


# NPU/CUDA 显存 OOM 关键字。vllm-ascend 报 "NPU out of memory", GPU 环境报
# "CUDA out of memory" / "OutOfMemoryError"; 统一匹配, 任一命中即判定为显存 OOM。
# 用 compiled 模式一次扫多种串, 比多次 in 更省; 大小写不敏感 (OOM 错误信息大小写不一)。
_OOM_PATTERN = re.compile(
    r"(NPU out of memory|CUDA out of memory|OutOfMemoryError|out of memory)", re.IGNORECASE
)


def _detect_oom(content: str) -> bool:
    """
    检测日志内容是否含显存 OOM 错误。

    匹配 vllm-ascend 的 "NPU out of memory" 及常见 PyTorch/GPU 的 "CUDA out of memory"
    / "OutOfMemoryError" 串。用于把"服务未起/压测未完成/无有效结果"等笼统失败进一步
    细化为"显存 OOM", 便于在失败通知中直接定位根因。
    """
    return bool(_OOM_PATTERN.search(content))


def _collect_online_failures(report_dir: Path) -> list[str]:
    """
    扫描在线测试产物, 收集未能正常完成的场景。

    判据 (与 online_server_compare.sh 的 wait_server_ready / is_bench_log_complete 对齐):
    1. server_*.log 存在但不含启动就绪关键字 -> 服务未起来
    2. result_*/ 目录下某并发度日志缺失完成关键字 -> 该并发度压测未完成;
       或含完成关键字但"假成功" (Failed requests != 0 或 Mean TPOT == 0) -> 压测异常
    3. 三个服务 result_*/ 目录都齐全但无对应 benchmark_comparison_*.log -> process_data.py 中止
       (此处"齐全"指各含至少一份真正完成的并发度日志, 假成功的不计入)

    返回:
        失败描述字符串列表 (可直接渲染进通知)
    """
    failures: list[str] = []

    # 1. 服务未就绪: server_<model>_input<in>_output<out>_<service>.log
    #    未含就绪关键字即判失败; 若同时含 OOM, 标注为 OOM 以便直接定位根因。
    for server_log in sorted(report_dir.glob("server_*_input*_output*_*.log")):
        try:
            content = server_log.read_text(encoding="utf-8", errors="replace")
        except Exception:
            failures.append(f"服务日志读取失败: {server_log.name}")
            continue
        if _ONLINE_SERVER_READY_KEYWORD not in content:
            reason = " (疑似显存 OOM)" if _detect_oom(content) else ""
            failures.append(f"服务未启动就绪{reason}: {server_log.name}")

    # 2. 压测未完成/异常: result_input_<in>_output_<out>_<service>_<model>/
    #    目录下 input_<in>_output_<out>_concurrency<c>.log 应含完成关键字,
    #    且不出现"假成功" (Failed requests != 0 或 Mean TPOT == 0)。
    #    不含完成关键字时, 若日志含 OOM 则标注 (压测中被 OOM 打断的常见形态)。
    for result_dir in sorted(report_dir.glob("result_input_*_output_*_*_*")):
        if not result_dir.is_dir():
            continue
        for bench_log in sorted(result_dir.glob("input_*_output_*_concurrency*.log")):
            try:
                content = bench_log.read_text(encoding="utf-8", errors="replace")
            except Exception:
                failures.append(f"压测日志读取失败: {result_dir.name}/{bench_log.name}")
                continue
            if _ONLINE_BENCH_COMPLETE_KEYWORD not in content:
                reason = " (疑似显存 OOM)" if _detect_oom(content) else ""
                failures.append(f"压测未完成{reason}: {result_dir.name}/{bench_log.name}")
                continue
            anomaly = _bench_log_anomaly(content)
            if anomaly:
                # 假成功时也顺带查 OOM: 很多 OOM 场景请求失败 + TPOT=0, 但日志里有 OOM 串
                if _detect_oom(content):
                    anomaly = f"{anomaly}, 显存 OOM"
                failures.append(f"压测异常 ({anomaly}): {result_dir.name}/{bench_log.name}")

    # 3. process_data.py 中止: 三服务结果目录都存在 (各含至少一份完成的并发度日志),
    #    却没有对应的 benchmark_comparison_*.log -> 数据处理阶段失败。
    #    按场景 (model + input + output) 聚合判定, 复用上面的完成判据。
    completed_by_scenario: dict[tuple[str, int, int], dict[str, int]] = {}
    for result_dir in sorted(report_dir.glob("result_input_*_output_*_*_*")):
        if not result_dir.is_dir():
            continue
        m = re.match(
            r"result_input_(\d+)_output_(\d+)_(aclgraph|xlite_full|xlite_decode_only)_(.+)", result_dir.name
        )
        if not m:
            continue
        in_len, out_len, service, model_name = (
            int(m.group(1)),
            int(m.group(2)),
            m.group(3),
            m.group(4),
        )
        # 仅统计真正完成 (含完成关键字且无假成功) 的并发度, 与第 2 类判据对齐,
        # 避免假成功日志被算作"完成"从而掩盖 process_data.py 中止。
        completed = 0
        for log in result_dir.glob("input_*_output_*_concurrency*.log"):
            try:
                content = log.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if _ONLINE_BENCH_COMPLETE_KEYWORD in content and not _bench_log_anomaly(content):
                completed += 1
        if completed > 0:
            key = (model_name, in_len, out_len)
            completed_by_scenario.setdefault(key, {})[service] = completed

    for (model_name, in_len, out_len), services in completed_by_scenario.items():
        if {"aclgraph", "xlite_full", "xlite_decode_only"}.issubset(services.keys()):
            pattern = f"benchmark_comparison_{model_name}_input{in_len}_output{out_len}_*.log"
            if not list(report_dir.glob(pattern)):
                failures.append(
                    f"对比报告未生成 (process_data.py 中止): "
                    f"{model_name} input={in_len} output={out_len}"
                )

    return failures


def _collect_offline_failures(report_dir: Path) -> list[str]:
    """
    扫描离线 bench 产物, 收集未能正常完成的场景。

    判据 (与 run_offline_bench 的 "tokens/s" in stdout 对齐):
    OFFLINE_BENCH_SCENARIOS 中每个期望场景都应产出
    offline_bench_<model>_in<in>_out<out>_bs<bs>_*.log 且含 "tokens/s"。
    日志缺失 (场景未跑/中途崩溃且未落盘) 或不含完成关键字均记为失败;
    不含完成关键字时若日志含显存 OOM 串, 标注为 OOM 以便直接定位根因。
    """
    failures: list[str] = []

    for input_len, output_len, bs in OFFLINE_BENCH_SCENARIOS:
        pattern = f"offline_bench_{OFFLINE_BENCH_MODEL}_in{input_len}_out{output_len}_bs{bs}_*.log"
        logs = list(report_dir.glob(pattern))
        if not logs:
            failures.append(
                f"离线 bench 场景未产出日志: in={input_len} out={output_len} bs={bs}"
            )
            continue
        log_file = logs[0]
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            failures.append(f"离线 bench 日志读取失败: {log_file.name}")
            continue
        if "tokens/s" not in content:
            if _detect_oom(content):
                reason = "显存 OOM"
            else:
                reason = "无有效结果 (疑似崩溃)"
            failures.append(
                f"离线 bench 场景{reason}: in={input_len} out={output_len} bs={bs}"
            )

    return failures


def build_failure_report(
    report_dir: Path,
    online_failures: list[str],
    offline_failures: list[str],
) -> str:
    """
    汇总本次运行的失败场景, 生成独立的 ⚠️ 通知。

    仅在确实收集到失败时调用 (无失败时调用方不会触发)。
    单独成条发送, 与各阶段性能报告解耦: 即使某阶段性能报告正常, 失败仍可见。
    """
    vllm_ascend_version = get_vllm_ascend_version()
    xlite_version = get_xlite_version()

    current_date = "unknown"
    match = re.search(r"xlite-[\w.\+]+-(\d{8})$", str(report_dir))
    if match:
        current_date = match.group(1)

    lines = [
        f'<font color="orange"><b>⚠️ xlite {current_date} 测试失败场景汇总</b></font>',
        "",
        f"【当前版本】 vllm-ascend 版本: {vllm_ascend_version}, xlite 版本: {xlite_version}",
    ]

    ip_address = os.environ.get("MACHINE_IP", "")
    report_path = f"{ip_address}:{report_dir}" if report_dir else ""
    if report_path:
        lines.append(f"报告保存路径: {report_path}")
        lines.append("")

    total = len(online_failures) + len(offline_failures)
    lines.append(f"❗ 共发现 <b>{total}</b> 个失败项, 性能报告仅反映成功完成的场景, 请排查以下失败:")
    lines.append("")

    if online_failures:
        lines.append(f"<b>在线性能测试 ({len(online_failures)} 项):</b>")
        for item in online_failures:
            lines.append(f"  • {item}")
        lines.append("")

    if offline_failures:
        lines.append(f"<b>离线 bench 测试 ({len(offline_failures)} 项):</b>")
        for item in offline_failures:
            lines.append(f"  • {item}")
        lines.append("")

    lines.append(
        "提示: 服务未启动就绪请查 server_*.log; 压测未完成请查 result_*/ 下 *concurrency*.log; "
        "对比报告未生成说明 process_data.py 中止, 请查 process_data.py 的报错。"
    )

    return "\n".join(lines)


def _send_phase_crash_notification(
    report_dir: Path, receiver: str, phase_name: str, reason: str
) -> None:
    """
    某阶段整体崩溃 (run_benchmark/run_offline_bench 返回 None) 时发送的通知。

    到达这里说明测试脚本自身非零退出或异常, 报告阶段根本进不去, 故单独发一条,
    避免每日任务静默挂掉、用户无感知。发送失败仅记日志, 不影响已决定的 sys.exit(1)。
    """
    try:
        vllm_ascend_version = get_vllm_ascend_version()
        xlite_version = get_xlite_version()
    except Exception:
        vllm_ascend_version = "N/A"
        xlite_version = "N/A"

    ip_address = os.environ.get("MACHINE_IP", "")
    report_path = f"{ip_address}:{report_dir}" if report_dir else ""

    lines = [
        f'<font color="red"><b>❌ xlite {phase_name}阶段崩溃，任务终止</b></font>',
        "",
        f"【当前版本】 vllm-ascend 版本: {vllm_ascend_version}, xlite 版本: {xlite_version}",
        f"【失败原因】 {reason}",
    ]
    if report_path:
        lines.append(f"报告保存路径: {report_path}")
    lines.append("")
    lines.append("该阶段未能完成, 未生成性能对比报告, 请查 daily_benchmark 日志排查。")

    try:
        crash_file = report_dir / f"crash_{phase_name}.txt"
        crash_file.write_text("\n".join(lines), encoding="utf-8")
    except Exception:
        pass

    send_notification("\n".join(lines), receiver)


def _process_online_reports(report_dir: Path, current_reports: list[Path]) -> list[dict]:
    """
    解析 online 性能测试报告并与基线对比 (报告处理阶段)

    参数:
        report_dir: 当前版本报告目录
        current_reports: 当前版本的 benchmark_comparison_*.log 文件列表

    返回:
        对比结果字典列表
    """
    all_comparison_results = []

    for current_report in current_reports:
        model_info = extract_model_info(current_report)
        if not model_info:
            log_warning(f"无法从报告文件名提取模型信息: {current_report.name}")
            continue

        model_name = model_info["model_name"]
        input_len = model_info["input_len"]
        output_len = model_info["output_len"]

        log_info(f"处理模型: {model_name} (input={input_len}, output={output_len})")
        log_info(f"当前版本报告: {current_report}")

        # 解析当前版本指标
        current_metrics = parse_benchmark_report(current_report)
        current_metrics["input_len"] = input_len
        current_metrics["output_len"] = output_len
        # 确保模型名称被正确设置（优先使用从文件名提取的模型名称）
        if not current_metrics.get("model"):
            current_metrics["model"] = model_name

        # 保存当前版本指标
        current_json = report_dir / f"metrics_{model_name}_input{input_len}_output{output_len}.json"
        save_metrics_json(current_metrics, current_json)

        # 与基线版本数据对比
        baseline_report, baseline_version = get_previous_version_report(model_name, input_len, output_len)
        comparison_result = {}

        if baseline_report:
            log_info(f"对比基准: {baseline_report} (版本: {baseline_version})")
            baseline_metrics = parse_benchmark_report(baseline_report)
            comparison_result = compare_metrics(current_metrics, baseline_metrics)

            # 保存对比结果
            comparison_json = report_dir / f"comparison_{model_name}_input{input_len}_output{output_len}.json"
            save_metrics_json(comparison_result, comparison_json)
        else:
            log_warning(f"未找到模型 {model_name} (input={input_len}, output={output_len}) 的基线测试报告，跳过对比")
            comparison_result = {
                "error": "无基线数据",
                "date_current": current_metrics.get("timestamp", ""),
                "model": current_metrics.get("model", ""),
                "baseline_version": baseline_version or "unknown",
            }

        all_comparison_results.append(comparison_result)

    return all_comparison_results


def _process_offline_reports(report_dir: Path) -> list[dict]:
    """
    解析离线 bench 测试报告并与基线对比

    参数:
        report_dir: 当前版本报告目录

    返回:
        对比结果字典列表
    """
    all_comparison_results = []

    # 查找所有离线 bench 场景报告
    offline_reports = sorted(report_dir.glob(f"offline_bench_{OFFLINE_BENCH_MODEL}_*_*.log"))
    if not offline_reports:
        log_error("未找到离线 bench 测试报告文件")
        return all_comparison_results

    for current_report in offline_reports:
        log_info(f"处理离线 bench 报告: {current_report}")

        # 解析当前版本指标
        current_metrics = parse_offline_bench_report(current_report)
        input_len = current_metrics.get("input_len")
        output_len = current_metrics.get("output_len")
        batch_size = current_metrics.get("batch_size")

        if batch_size is None or input_len is None or output_len is None:
            log_warning(f"无法从报告文件名提取场景信息: {current_report.name}")
            continue

        # 误判检测: 无效结果 (OOM/崩溃导致日志无 Iter 行, decode_tps 为 None)
        # 不生成 metrics/对比/报告, 避免 null 指标被误报为"无劣化"。
        if current_metrics.get("decode_tps") is None:
            log_warning(
                f"离线 bench 场景 {current_report.name} 无有效性能指标 (decode_tps 为空, 疑似 OOM/崩溃), 跳过该场景"
            )
            continue

        scenario_tag = f"in{input_len}_out{output_len}_bs{batch_size}"

        # 保存当前版本指标
        current_json = report_dir / f"offline_metrics_{OFFLINE_BENCH_MODEL}_{scenario_tag}.json"
        save_metrics_json(current_metrics, current_json)

        # 与基线版本数据对比
        baseline_report, baseline_version = get_previous_offline_report(input_len, output_len, batch_size)
        comparison_result = {}

        if baseline_report:
            log_info(f"对比基准: {baseline_report} (版本: {baseline_version}, 场景: {scenario_tag})")
            baseline_metrics = parse_offline_bench_report(baseline_report)
            comparison_result = compare_offline_metrics(current_metrics, baseline_metrics)

            # 保存对比结果
            comparison_json = report_dir / f"offline_comparison_{OFFLINE_BENCH_MODEL}_{scenario_tag}.json"
            save_metrics_json(comparison_result, comparison_json)
        else:
            log_warning(f"未找到离线 bench 场景 {scenario_tag} 的基线报告，跳过对比")
            comparison_result = {
                "error": "无基线数据",
                "date_current": current_metrics.get("timestamp", ""),
                "model": current_metrics.get("model", ""),
                "baseline_version": baseline_version or "unknown",
            }

        all_comparison_results.append(comparison_result)

    return all_comparison_results


def _report_phase(
    report_dir: Path,
    all_comparison_results: list[dict],
    receiver: str,
    is_offline: bool,
    failure_count: int = 0,
) -> bool:
    """
    对一组对比结果 (online 或 offline) 生成报告并发送通知。

    将按模型分组 -> 生成报告 -> 发通知的逻辑复用于 online 与
    offline 两阶段, 使"默认先在线再离线"时两阶段各出一份通知。

    参数:
        report_dir: 当前版本报告目录
        all_comparison_results: 该阶段的对比结果列表
        receiver: 通知接收者 ID
        is_offline: True=离线 bench 阶段 (用 generate_offline_report / offline_summary),
                    False=在线阶段 (用 generate_model_report / daily_summary)

    返回:
        该阶段是否检测到任何劣化 (True=有劣化)
    """
    if not all_comparison_results:
        log_error("没有有效的对比结果")
        return False

    # 按模型分组
    model_groups = {}
    for comparison in all_comparison_results:
        model_name = comparison.get("model", "Unknown") or "Unknown"
        model_groups.setdefault(model_name, []).append(comparison)

    log_info(f"检测到 {len(model_groups)} 个模型:")
    for model_name, comparisons in model_groups.items():
        log_info(f"  - {model_name}: {len(comparisons)} 个测试场景")

    model_names = list(model_groups.keys())
    log_info(f"准备为 {len(model_names)} 个模型生成报告")
    # 先生成全部模型报告。有任一变化则每模型各发一条(含无变化), 否则只发一条合并通知。
    model_reports = []
    any_change = False
    for idx, model_name in enumerate(model_names):
        model_comparisons = model_groups[model_name]
        log_info(f"正在生成模型 {model_name} 的报告 [{idx + 1}/{len(model_names)}]")
        if is_offline:
            model_report = generate_offline_report(model_name, model_comparisons, report_dir, idx, len(model_names))
            summary_prefix = "offline_summary"
        else:
            model_report = generate_model_report(model_name, model_comparisons, report_dir, idx, len(model_names))
            summary_prefix = "daily_summary"

        model_report_file = report_dir / f"{summary_prefix}_{model_name}_{idx + 1}.txt"
        model_report_file.write_text(model_report, encoding="utf-8")
        log_info(f"模型 {model_name} 的报告已保存至: {model_report_file}")

        print("\n" + "=" * 60)
        print(f"模型 {model_name} 的报告 [{idx + 1}/{len(model_names)}]")
        print("=" * 60)
        print(model_report)
        print("=" * 60 + "\n")

        model_reports.append((model_name, model_report))
        if "发现性能优化" in model_report or "发现性能劣化" in model_report:
            any_change = True

    if any_change:
        for idx, (model_name, model_report) in enumerate(model_reports):
            log_info(f"正在发送模型 {model_name} 的通知 [{idx + 1}/{len(model_reports)}]")
            send_notification(model_report, receiver)
    else:
        log_info("测试模型性能均无变化")
        no_change_report = build_no_change_report(
            report_dir, list(model_groups.keys()), is_offline, failure_count
        )
        send_notification(no_change_report, receiver)

    # 该阶段是否检测到劣化
    has_degradation = False
    for comparison in all_comparison_results:
        for row in comparison.get("row_changes", []):
            for change in row["changes"].values():
                if change.get("is_degradation"):
                    has_degradation = True
                    break
            if has_degradation:
                break
        if has_degradation:
            break
    return has_degradation


# ====================== 主函数 ======================
def main():
    """
    主函数 - 执行完整的自动化测试流程

    流程:
    1. 解析命令行参数
    2. 拉取最新代码
    3. 编译项目
    4. 安装 wheel 包
    5. 执行基准测试
    6. 解析测试报告
    7. 与上一版本数据对比
    8. 生成并发送报告
    9. 返回退出码
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="xlite 每日自动化测试机器人")
    parser.add_argument(
        "--model",
        default="moe",
        choices=["moe", "moe_quant", "moe_quant_a3", "moe_mix", "quant", "all"],
        help="测试模型类型 (default: moe)",
    )
    parser.add_argument("--skip-pull", action="store_true", help="跳过代码拉取")
    parser.add_argument("--skip-build", action="store_true", help="跳过编译")
    parser.add_argument(
        "--receiver", type=str, default="927280411401503971", help="接收者ID (default: 927280411401503971)"
    )
    parser.add_argument("--threshold", type=float, default=0.05, help="性能劣化阈值 (default: 0.05 = 5%%)")
    parser.add_argument(
        "--offline-threshold",
        type=float,
        default=0.025,
        help="离线 bench 性能劣化阈值 (default: 0.025 = 2.5%%); 与在线 --threshold 分离",
    )
    parser.add_argument(
        "--report-dir", type=str, default=None, help="指定已有的报告目录（调试模式，跳过拉取、编译、测试步骤）"
    )
    parser.add_argument(
        "--env-type", type=str, default="yellow", choices=["blue", "yellow"], help="环境类型 (default: yellow)"
    )
    parser.add_argument("--compare-ttft", action="store_true", help="启用TTFT指标作为对比指标（默认不比较TTFT）")
    parser.add_argument(
        "--compare-tpot-p99", action="store_true", help="启用TPOT P99指标作为对比指标（默认不比较TPOT P99）"
    )
    parser.add_argument(
        "--offline-bench",
        action="store_true",
        help="仅跑离线 bench 测试（跳过在线性能测试）。默认流程为: 先在线性能测试, 再离线 bench。",
    )
    parser.add_argument(
        "--skip-offline-bench",
        action="store_true",
        help="跳过离线 bench 阶段, 只跑在线性能测试。",
    )

    args = parser.parse_args()

    # 更新全局配置
    global DEGRADATION_THRESHOLD, OFFLINE_DEGRADATION_THRESHOLD, COMPARE_TTFT_METRICS, COMPARE_TPOT_P99_METRICS
    DEGRADATION_THRESHOLD = args.threshold
    OFFLINE_DEGRADATION_THRESHOLD = args.offline_threshold
    COMPARE_TTFT_METRICS = args.compare_ttft
    COMPARE_TPOT_P99_METRICS = args.compare_tpot_p99

    # 设置配置
    set_env_type(args.env_type)
    set_machine_ip(os.environ.get("MACHINE_IP", ""))

    # 设置日志文件到默认目录（确保所有日志都能被保存）
    setup_logging(REPORT_DIR, log_prefix="daily_benchmark")
    log_info("日志文件路径已设置")

    # 打印启动信息
    log_info("=" * 60)
    log_info("xlite 每日自动化测试机器人启动")
    log_info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"环境类型: {args.env_type}")
    log_info("=" * 60)

    # 默认流程: 先在线性能测试, 再离线 bench 测试 (各出一份通知)。
    #   --offline-bench        : 仅离线 (跳过在线)
    #   --skip-offline-bench   : 仅在线 (跳过离线)
    if args.offline_bench:
        run_online = False
        run_offline = True
        if args.skip_offline_bench:
            log_warning("--offline-bench 与 --skip-offline-bench 同时指定, 互斥, 按 --offline-bench 处理 (仅离线)")
            args.skip_offline_bench = False
    else:
        run_online = True
        run_offline = not args.skip_offline_bench

    if args.report_dir:
        log_info("调试模式：使用指定的报告目录，跳过拉取、编译、测试步骤")
        report_dir = Path(args.report_dir)
        if not report_dir.exists():
            log_error(f"指定的报告目录不存在: {report_dir}")
            sys.exit(1)
        log_info(f"使用报告目录: {report_dir}")
    else:
        # 步骤1: 拉取最新代码
        if not args.skip_pull:
            if not pull_latest_code():
                log_error("代码拉取失败，终止执行")
                sys.exit(1)

        # 步骤2: 编译项目
        wheel_path = None
        if not args.skip_build:
            success, wheel_path = build_project(REPORT_DIR)
            if not success:
                log_error("编译失败，终止执行")
                sys.exit(1)

        # 步骤3: 安装 wheel 包
        if wheel_path:
            if not install_wheel(wheel_path):
                log_error("wheel 包安装失败，终止执行")
                sys.exit(1)

        # 步骤4: 执行基准测试
        # 一次性确定报告目录 (版本+日期), 供在线/离线两阶段复用,
        # 避免跨午夜跑测试时结果分到不同日期目录。
        current_version = get_xlite_version()
        current_date = datetime.now().strftime("%Y%m%d")
        report_dir = REPORT_DIR / f"xlite-{current_version}-{current_date}"
        report_dir.mkdir(parents=True, exist_ok=True)
        log_info(f"本次测试报告目录: {report_dir}")

        if run_online:
            log_info("=" * 60)
            log_info("阶段 1/2: 在线性能测试")
            log_info("=" * 60)
            online_report_dir = run_benchmark(args.model, report_dir)
            if not online_report_dir:
                log_error("在线基准测试失败，终止执行")
                # 在线阶段整体失败 (online_server_compare.sh 非零退出) 前必发通知,
                # 否则每日任务静默挂掉、用户无感知。
                _send_phase_crash_notification(
                    report_dir, args.receiver, "在线性能测试", "online_server_compare.sh 执行失败"
                )
                sys.exit(1)

        if run_offline:
            log_info("=" * 60)
            log_info(f"阶段 {'2/2' if run_online else '1/1'}: 离线 bench 测试")
            log_info("=" * 60)
            offline_report_dir = run_offline_bench(report_dir)
            if not offline_report_dir:
                log_error("离线 bench 测试失败，终止执行")
                _send_phase_crash_notification(
                    report_dir, args.receiver, "离线 bench 测试", "run.sh bench 执行异常"
                )
                sys.exit(1)

    # 步骤5: 处理报告 + 发送通知 (在线/离线各一份)
    has_any_degradation = False
    has_infra_failure = False  # 报告缺失/无效等基础设施故障 (与"有劣化"区分, 走 exit 1)
    # 收集各阶段"服务未起 / 压测报错 / 报告未生成"等失败场景, 单独发一条通知。
    online_failures: list[str] = []
    offline_failures: list[str] = []

    if run_online:
        # 无条件收集失败场景: 即使零对比报告 (如所有服务未起, online_server_compare.sh
        # 仍 exit 0), 也能据此发 ⚠️ 失败明细通知, 与离线侧对齐, 避免 exit 1 之外无群通知。
        online_failures = _collect_online_failures(report_dir)
        if online_failures:
            log_warning(f"在线测试发现 {len(online_failures)} 个失败场景")
            has_infra_failure = True
        current_reports = list(report_dir.glob("benchmark_comparison_*.log"))
        if not current_reports:
            log_error("未找到在线测试报告文件")
            has_infra_failure = True
        else:
            all_comparison_results = _process_online_reports(report_dir, current_reports)
            log_info("=" * 60)
            log_info("在线性能测试: 生成报告与通知")
            log_info("=" * 60)
            if _report_phase(
                report_dir,
                all_comparison_results,
                args.receiver,
                is_offline=False,
                failure_count=len(online_failures),
            ):
                has_any_degradation = True

    if run_offline:
        all_comparison_results = _process_offline_reports(report_dir)
        log_info("=" * 60)
        log_info("离线 bench 测试: 生成报告与通知")
        log_info("=" * 60)
        # 先收集失败场景, 以便把数量透传给 _report_phase (避免"无变化"通知与失败矛盾)
        offline_failures = _collect_offline_failures(report_dir)
        if offline_failures:
            log_warning(f"离线 bench 测试发现 {len(offline_failures)} 个失败场景")
            has_infra_failure = True
        if not all_comparison_results:
            # 一个有效场景都没有 = 基础设施故障 (单个场景 OOM 已在 _process_offline 内跳过)
            has_infra_failure = True
        elif _report_phase(
            report_dir,
            all_comparison_results,
            args.receiver,
            is_offline=True,
            failure_count=len(offline_failures),
        ):
            has_any_degradation = True

    # 失败场景汇总: 单独发一条 ⚠️ 通知, 与各阶段性能报告解耦。
    # 即使某阶段"无变化", 失败仍可见; 调度器靠 has_infra_failure->exit 1 触发 ❌。
    if online_failures or offline_failures:
        failure_report = build_failure_report(report_dir, online_failures, offline_failures)
        failure_report_file = report_dir / "failure_summary.txt"
        try:
            failure_report_file.write_text(failure_report, encoding="utf-8")
            log_info(f"失败汇总已保存至: {failure_report_file}")
        except Exception as e:
            log_warning(f"保存失败汇总失败: {e}")
        send_notification(failure_report, args.receiver)

    # 报告阶段后: 若本次恰为发布 tag 构建，镜像本次结果为基线，供后续 dev 构建对比。
    # 仅在非调试模式 (有真实本次运行报告目录) 时触发。
    if not args.report_dir:
        seed_baseline_if_release(report_dir)

    # 步骤6: 设置退出码
    #   0=成功无劣化 | 1=基础设施故障(报告缺失/场景失败/服务未起等) | 2=成功但有劣化
    #   有失败场景时走 exit 1, 使调度器 (run_benchmark_scheduler.py) 标记任务失败并关注;
    #   此时即便存在劣化也以 exit 1 优先 (失败场景下的劣化对比不可信)。
    if has_infra_failure:
        if online_failures or offline_failures:
            log_error(
                f"测试存在失败场景 (在线 {len(online_failures)} / 离线 {len(offline_failures)})。"
                f"报告已保存至: {report_dir}"
            )
        else:
            log_error(f"测试基础设施故障。报告已保存至: {report_dir}")
        log_success("自动化测试完成!")
        return 1
    if has_any_degradation:
        log_warning(f"检测到性能劣化。报告已保存至: {report_dir}")
        log_success("自动化测试完成!")
        return 2

    log_success("自动化测试完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
