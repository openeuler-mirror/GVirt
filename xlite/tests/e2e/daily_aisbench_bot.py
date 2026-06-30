#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlite 每日自动化精度测试机器人

功能说明:
1. 在编译容器和测试容器中拉取最新代码
2. 在编译容器中编译项目生成 wheel 包
3. 在测试容器中执行精度基准测试脚本 (batch_aisbench.py)
4. 解析测试报告并在单次运行内对比精度
5. 检测精度问题 (xlite 与 aclgraph 差异过大或精度低于阈值时告警)
6. 发送通知到群组

精度对比逻辑:
- 对比同一模型在不同 backend (xlite full, xlite decode-only, aclgraph) 下的精度
- 如果 xlite 精度比 aclgraph 低超过 4%，视为警告
- 如果任何模型精度低于 80%，视为警告

架构说明:
- 定时任务在测试容器中运行
- 编译容器: 用于拉取代码和编译项目
- 测试容器: 用于拉取代码和运行精度测试
- 两个容器都从远程仓库拉取代码到 /workspaces/code/opencode/GVirt

容器启动方式:

1. 启动编译容器 (xlite-build，可选，推荐直接使用测试容器编译):
   docker run -itd --shm-size=10.24gb --net=host --privileged --cap-add=SYS_PTRACE --user root \
     --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
     -v /usr/local/dcmi:/usr/local/dcmi:ro \
     -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
     -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common:ro \
     -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
     -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
     -v /etc/vnpu.cfg:/etc/vnpu.cfg:ro \
     -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
     -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
     -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
     --name xlite-build \
     -v /tmp:/tmp -v /home:/home -v /mnt/sdb/:/mnt/sdb \
     hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20260324 \
     /bin/bash

2. 启动测试容器 (daily-test，应直接再测试容器中执行定时任务):
   docker run -itd --shm-size=10.24gb --net=host --privileged --cap-add=SYS_PTRACE --user root \
     --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
     -v /usr/local/dcmi:/usr/local/dcmi:ro \
     -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
     -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common:ro \
     -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
     -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
     -v /etc/vnpu.cfg:/etc/vnpu.cfg:ro \
     -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
     -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
     -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
     --name daily-test \
     -v /tmp:/tmp -v /home:/home -v /mnt/sdb/:/mnt/sdb \
     -v /var/run/docker.sock:/var/run/docker.sock \
     quay.io/ascend/vllm-ascend:v0.20.2rc1-a3 \
     /bin/bash

使用方法:
    python3 daily_aisbench_bot.py --receiver "927280411401503971"

    可选参数:
    --model-args: 传递给 batch_aisbench.py 的模型参数字符串，格式参考 batch_aisbench.py 的参数格式
    --skip-pull: 跳过代码拉取
    --skip-build: 跳过编译
    --receiver: 接收者ID，默认 927280411401503971
    --xlite-threshold: xlite 与 aclgraph 精度差异阈值，默认 0.04 (即4%)
    --min-accuracy: 最低精度阈值，默认 0.80 (即80%)
    --build-container: 编译容器名称
    --env-type: 环境类型 (blue/yellow)，默认 yellow
               - yellow 环境: 需要执行 source /home/env.sh
               - blue 环境: 不需要 source /home/env.sh

退出码:
    0 - 成功，无精度警告
    1 - 执行失败
    2 - 成功，但检测到精度警告

目录结构:
    PROJECT_ROOT = /workspaces/code/opencode/GVirt (容器内路径)
    REPORT_DIR = /home/daily_aisbench_reports (容器内路径)
"""

import os
import sys
import re
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# 导入共享工具模块
from daily_bot_utils import (
    # 日志工具
    log_info,
    log_success,
    log_warning,
    log_error,
    setup_logging,
    move_log_file,
    # 代码拉取和编译
    pull_latest_code,
    build_project,
    install_wheel,
    # 版本信息
    get_current_version,
    get_current_commit,
    get_vllm_ascend_version,
    get_xlite_commit,
    # 环境检测
    detect_model_device,
    should_disable_xccl,
    # 数据存储
    save_metrics_json,
    # 通知发送
    send_notification,
    # 配置设置
    set_build_container,
    set_test_container,
    set_env_type,
    set_machine_ip,
)

# ====================== 本地配置 ======================
# 每日精度报告存储目录 (宿主机路径)
REPORT_DIR = Path("/home/daily_aisbench_reports")
E2E_DIR = Path(__file__).parent.resolve()

# xlite 与 aclgraph 精度差异阈值，超过此比例视为警告 (4%)
XLITE_ACLGRAPH_THRESHOLD = 4

# 最低精度阈值，低于此值视为警告 (80%)
MIN_ACCURACY_THRESHOLD = 80


# ====================== 精度基准测试执行函数 ======================
def run_aisbench_benchmark(
    model_args: str,
    debug: bool = False,
    reference_commit: str | None = None,
    timeout: int = 4 * 3600,
    retry: int = 3,
    vllm_timeout: int = 1800,
) -> Optional[Path]:
    """
    在测试容器中执行精度基准测试脚本 (batch_aisbench.py)

    参数:
        model_args: 模型参数字符串，格式为 "--models model1 model2 --tps tp1 tp2 --eps ep1 ep2 ..."，参考 batch_aisbench.py
            的参数格式
        debug: 是否为调试模式，调试模式下会直接将测试过程输出到stdout，非调试模式下会将错误输出到日志文件
        reference_commit: 指定复用前期某个 xlite commit hash 测试；仅当 model_args 中包含 --reuse 或 -r 参数时生效
        timeout: ais_bench 子进程超时时间（秒）
        retry: ais_bench 失败时的重试次数
        vllm_timeout: vLLM 服务器启动超时时间（秒）

    返回:
        报告目录路径，失败返回 None

    说明:
        测试结果会保存到 daily_aisbench_reports/YYYYMMDD/ 目录下
    """
    log_info("开始执行精度基准测试...")

    # 检测模型设备
    device = detect_model_device()
    model_dir = f"/mnt/{device}/models" if device else "/mnt/sdb/models"

    # 创建当前版本报告目录 (版本号+日期)
    current_version = get_current_version()
    current_commit = get_current_commit() or "unknown"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    aisbench_repo_dir = REPORT_DIR / "benchmark"
    if "--reuse" not in model_args and "-r" not in model_args:
        reference_commit = current_commit
    aisbench_output_dir = REPORT_DIR / "outputs" / (reference_commit or current_commit)
    report_subdir = REPORT_DIR / f"xlite-{current_version}" / f"{current_time}-{current_commit}"
    report_subdir.mkdir(parents=True, exist_ok=True)

    # 将日志文件移动到报告目录
    move_log_file(report_subdir)

    try:
        # 使用 batch_aisbench.py 进行精度测试
        aisbench_cmd = [
            "python",
            str(E2E_DIR / "batch_aisbench.py"),
            str(aisbench_repo_dir),
            "--num-prompts 8",
            f'--output-dir "{aisbench_output_dir}"',
            f'--log-file "{report_subdir / "aisbench_results.md"}"',
            f'--model-dir "{model_dir}"',
            f"--timeout {timeout}",
            f"--retry {retry}",
            f"--vllm-timeout {vllm_timeout}",
            str(model_args),
        ]
        if debug:
            aisbench_cmd.append("--debug")

        aisbench_cmd_str = " ".join(aisbench_cmd)
        log_info(f"执行命令: {aisbench_cmd_str}")

        # 根据是否禁用 XCCL 设置环境变量
        env = os.environ.copy()
        if should_disable_xccl():
            env["XLITE_DISABLE_XCCL"] = "1"
            log_info("已设置环境变量 XLITE_DISABLE_XCCL=1，禁用 XCCL 进行测试")
        else:
            env.pop("XLITE_DISABLE_XCCL", None)
            log_info("未设置环境变量 XLITE_DISABLE_XCCL，使用默认 XCCL 设置进行测试")

        # 执行精度测试脚本 (model_args 是 shell 风格参数字符串，需要 shell=True)
        result = subprocess.run(
            aisbench_cmd_str, cwd=str(E2E_DIR), env=env, shell=True, capture_output=not debug, text=True
        )

        if result.returncode != 0:
            log_error(f"精度基准测试执行失败: {result.stderr}")
            return None

        log_success("精度基准测试执行完成!")
        return report_subdir

    except Exception as e:
        log_error(f"精度基准测试执行异常: {e}")
        return None


# ====================== 报告解析函数 ======================
def parse_aisbench_report(report_path: Path) -> Dict:
    """
    解析精度基准测试报告文件 (CSV格式)

    参数:
        report_path: 报告文件路径 (CSV格式)

    返回:
        解析后的指标字典，结构如下:
        {
            "timestamp": "报告生成时间",
            "results": [
                {
                    "model_name": "模型名称",
                    "backend": "backend类型 (xlite full/xlite decode-only/aclgraph)",
                    "accuracy": 精度值,
                    "error": "错误信息"
                },
                ...
            ]
        }
    """
    import csv

    metrics = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "results": []}

    if not report_path.exists():
        log_warning(f"报告文件不存在: {report_path}")
        return metrics

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xlite = row.get("xlite", "").lower() in ["true", "y", "yes", "1"]
                xlite_full_mode = row.get("xlite_full_mode", "").lower() in ["true", "y", "yes", "1"]

                # 确定 backend 类型
                if xlite:
                    if xlite_full_mode:
                        backend = "xlite full"
                    else:
                        backend = "xlite decode-only"
                else:
                    backend = "aclgraph"

                model_name = row.get("model_name", "unknown-model")
                tp: int = int(row.get("tp_size", 0))
                ep: bool = row.get("ep", "").strip().lower() in ["true", "y", "yes", "1"]
                dp: int = int(row.get("dp_size", 0))
                result = {
                    "model_name": f"{model_name}+TP{tp}DP{dp}{'EP' if ep else ''}",
                    "backend": backend,
                    "accuracy": float(row.get("accuracy", 0.0)),
                    "error": row.get("error", "") if row.get("error") else None,
                }
                metrics["results"].append(result)

        log_info(f"解析报告成功，共 {len(metrics['results'])} 条结果")
    except Exception as e:
        log_error(f"解析报告失败: {e}")

    return metrics


def parse_aisbench_markdown_report(report_path: Path) -> Dict:
    """
    解析精度基准测试报告文件 (Markdown格式)

    参数:
        report_path: 报告文件路径 (Markdown格式)

    返回:
        解析后的指标字典
    """
    metrics = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "results": []}

    if not report_path.exists():
        log_warning(f"报告文件不存在: {report_path}")
        return metrics

    content = report_path.read_text(encoding="utf-8")

    # 定义表格行的正则表达式
    # 格式: | Model | Dataset | TP | EP | DP | Backend | Metric | Accuracy | Error |
    table_pattern = re.compile(
        r"\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*(\d+)\s*\|\s*([YN])\s*\|\s*(\d+)\s*\|"
        r"\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*([\d.]+)\s*\|\s*([^|]*)\s*\|"
    )

    for match in table_pattern.finditer(content):
        backend = match.group(6).strip()
        model_name = match.group(1).strip()
        tp: int = int(match.group(3).strip())
        ep: bool = match.group(4).strip().upper() == "Y"
        dp: int = int(match.group(5).strip())
        result = {
            "model_name": f"{model_name}+TP{tp}DP{dp}{'EP' if ep else ''}",
            "backend": backend,
            "accuracy": float(match.group(8)),
            "error": match.group(9).strip() if match.group(9).strip() else None,
        }
        metrics["results"].append(result)

    log_info(f"解析Markdown报告成功，共 {len(metrics['results'])} 条结果")
    return metrics


# ====================== 单次运行内精度对比函数 ======================
def analyze_accuracy_within_run(metrics: Dict, xlite_threshold: float, min_accuracy: float) -> Dict:
    """
    在单次运行内对比 xlite 与 aclgraph 的精度，并检查最低精度阈值

    参数:
        metrics: 当前运行指标数据
        xlite_threshold: xlite 与 aclgraph 精度差异阈值
        min_accuracy: 最低精度阈值

    返回:
        分析结果字典，结构如下:
        {
            "timestamp": "测试时间",
            "warnings": [
                {
                    "type": "xlite_vs_aclgraph" | "low_accuracy",
                    "model_name": "模型名称",
                    "backend": "backend类型",
                    "xlite_accuracy": 精度值 (仅 xlite_vs_aclgraph 类型),
                    "aclgraph_accuracy": 精度值 (仅 xlite_vs_aclgraph 类型),
                    "difference": 差异比例 (仅 xlite_vs_aclgraph 类型),
                    "accuracy": 精度值 (仅 low_accuracy 类型),
                },
                ...
            ]
        }

    说明:
        - xlite full 和 xlite decode-only 分别与对应的 aclgraph 对比
        - 如果 xlite 精度比 aclgraph 低超过 xlite_threshold，视为警告
        - 如果任何模型精度低于 min_accuracy，视为警告
    """
    analysis = {"timestamp": metrics.get("timestamp", ""), "warnings": []}

    # 按模型名称分组结果
    model_results: Dict[str, Dict[str, float]] = {}  # {model_name: {backend: accuracy}}
    for result in metrics.get("results", []):
        model_name = result["model_name"]
        backend = result["backend"]
        accuracy = result.get("accuracy", 0.0)
        error = result.get("error")

        model_results.setdefault(model_name, {})
        # 如果有错误，记录为 None
        model_results[model_name][backend] = None if error else accuracy  # type: ignore[assignment]

    # 对每个模型进行检查
    for model_name, backends in model_results.items():
        aclgraph_acc = backends.get("aclgraph")

        # 检查 xlite full vs aclgraph
        xlite_full_acc = backends.get("xlite full")
        if xlite_full_acc is not None and aclgraph_acc is not None and aclgraph_acc > 0:
            difference = aclgraph_acc - xlite_full_acc
            if difference > xlite_threshold:
                analysis["warnings"].append(
                    {
                        "type": "xlite_vs_aclgraph",
                        "model_name": model_name,
                        "backend": "xlite full",
                        "xlite_accuracy": xlite_full_acc,
                        "aclgraph_accuracy": aclgraph_acc,
                        "difference": difference,
                    }
                )
                log_warning(
                    f"{model_name} (xlite full): xlite={xlite_full_acc:.2f}%, aclgraph={aclgraph_acc:.2f}%, "
                    f"diff={difference:.2f}% [警告: xlite比aclgraph低超过{xlite_threshold}%]"
                )

        # 检查 xlite decode-only vs aclgraph
        xlite_decode_acc = backends.get("xlite decode-only")
        if xlite_decode_acc is not None and aclgraph_acc is not None and aclgraph_acc > 0:
            difference = aclgraph_acc - xlite_decode_acc
            if difference > xlite_threshold:
                analysis["warnings"].append(
                    {
                        "type": "xlite_vs_aclgraph",
                        "model_name": model_name,
                        "backend": "xlite decode-only",
                        "xlite_accuracy": xlite_decode_acc,
                        "aclgraph_accuracy": aclgraph_acc,
                        "difference": difference,
                    }
                )
                log_warning(
                    f"{model_name} (xlite decode-only): xlite={xlite_decode_acc:.2f}%, aclgraph={aclgraph_acc:.2f}%, "
                    f"diff={difference:.2f}% [警告: xlite比aclgraph低超过{xlite_threshold}%]"
                )

        # 检查所有 backend 的最低精度阈值
        for backend, accuracy in backends.items():
            if accuracy is not None and accuracy < min_accuracy:
                analysis["warnings"].append(
                    {
                        "type": "low_accuracy",
                        "model_name": model_name,
                        "backend": backend,
                        "accuracy": accuracy,
                    }
                )
                log_warning(f"{model_name} ({backend}): accuracy={accuracy:.2f}% [警告: 精度低于{min_accuracy}%]")

    return analysis


# ====================== 报告生成函数 ======================
def generate_accuracy_report(metrics: Dict, analysis: Dict, report_dir: Path) -> Tuple[str, Optional[str]]:
    """
    生成精度测试报告 (HTML格式，用于企业微信通知)

    返回两个字符串，分成两条通知发送:
    - 第一条: 标题 + 汇总 + 详细数据表格 (不含警告详情)
    - 第二条 (可选): 标题 + 警告详情 (仅在有警告时生成)

    参数:
        metrics: 当前测试指标数据
        analysis: 分析结果字典
        report_dir: 报告目录

    返回:
        (report_main, report_warnings)
        - report_main: 包含标题、汇总摘要和详细数据表格的报告
        - report_warnings: 包含标题和警告详情的报告，无警告时为 None
    """
    vllm_ascend_version = get_vllm_ascend_version()
    xlite_commit = get_xlite_commit()

    # 获取当前测试日期（从report_dir中提取）
    current_date = "unknown_date"
    if report_dir:
        match = re.search(r"^(\d{8})", str(report_dir.name))
        if match:
            current_date = match.group(1)

    # 构建标题块 (两条报告共用)
    header_lines = [
        f'<font color="blue"><b>xlite {current_date} 精度测试报告</b></font>',
        "",
        f"【当前版本】 vllm-ascend 版本: {vllm_ascend_version}, xlite commit: {xlite_commit}",
    ]

    # 按模型名称分组结果
    model_results: Dict[str, Dict[str, float]] = {}
    for result in metrics.get("results", []):
        model_name = result["model_name"]
        backend = result["backend"]
        accuracy = result.get("accuracy", 0.0)
        error = result.get("error")

        model_results.setdefault(model_name, {})
        model_results[model_name][backend] = None if error else accuracy  # type: ignore[assignment]

    # 统计警告数量
    warnings = analysis.get("warnings", [])
    xlite_warnings = [w for w in warnings if w["type"] == "xlite_vs_aclgraph"]
    low_acc_warnings = [w for w in warnings if w["type"] == "low_accuracy"]
    total_warnings = len(warnings)

    # 汇总摘要
    status_icon = "⚠️" if total_warnings > 0 else "✅"
    if total_warnings > 0:
        status_text = f"发现精度问题 ({total_warnings}项警告)，请关注！"
    else:
        status_text = "精度正常，所有模型符合要求"

    # 添加报告路径
    ip_address = os.environ.get("MACHINE_IP", "")
    if not ip_address:
        try:
            result = subprocess.run("hostname -I", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                ip_address = result.stdout.strip().split()[0]
        except Exception as e:
            log_error(f"获取IP地址失败: {e}")
    ip_address = ip_address or "unknown-ip"
    report_path = f"{ip_address}:{report_dir}" if report_dir else ""

    # ====== 第一条通知: 标题 + 汇总 + 详细数据 ======
    header_lines.append(f"📊 精度统计: <b>警告项: {total_warnings}</b>")
    header_lines.append(f"{status_icon} {status_text}")
    if report_path:
        header_lines.append(f"报告保存路径: {report_path}")
    header_lines.append("")

    # 详细数据表格
    main_lines = list(header_lines)
    if model_results:
        main_lines.append("【详细数据】")
        main_lines.append(" | 模型 | xlite full | xlite decode-only | aclgraph |")
        main_lines.append(" |------|------|------|------|")

        for model_name in sorted(model_results.keys()):
            backends = model_results[model_name]

            def format_accuracy(value: Optional[float], backend_name: str) -> str:
                if value is None:
                    return "N/A"
                has_xlite_warning = any(
                    w["model_name"] == model_name and w["backend"] == backend_name and w["type"] == "xlite_vs_aclgraph"
                    for w in xlite_warnings
                )
                has_low_acc_warning = any(
                    w["model_name"] == model_name and w["backend"] == backend_name and w["type"] == "low_accuracy"
                    for w in low_acc_warnings
                )
                formatted = f"{value:.2f}"
                if has_xlite_warning or has_low_acc_warning:
                    return f'<font color="red"><b>{formatted}</b></font>'
                return formatted

            xlite_full_str = format_accuracy(backends.get("xlite full"), "xlite full")
            xlite_decode_str = format_accuracy(backends.get("xlite decode-only"), "xlite decode-only")
            aclgraph_str = format_accuracy(backends.get("aclgraph"), "aclgraph")

            main_lines.append(f" | {model_name} | {xlite_full_str} | {xlite_decode_str} | {aclgraph_str} |")
        main_lines.append("")

    report_main = "\n".join(main_lines)

    # ====== 第二条通知: 标题 + 警告详情 (仅在有警告时生成) ======
    if total_warnings > 0:
        warning_lines = list(header_lines)
        if xlite_warnings:
            warning_lines.append(f"⚠️ <b>xlite 精度显著低于 aclgraph ({len(xlite_warnings)}项)</b>")
            for w in xlite_warnings:
                warning_lines.append(
                    f"  - {w['model_name']} ({str(w['backend']).lstrip('xlite ')}): "
                    f"xlite={w['xlite_accuracy']:.2f}%, aclgraph={w['aclgraph_accuracy']:.2f}%, "
                    f"差异 {w['difference']:.2f}%"
                )
            warning_lines.append("")

        if low_acc_warnings:
            warning_lines.append(f"⚠️ <b>精度低于最低阈值 ({len(low_acc_warnings)}项)</b>")
            for w in low_acc_warnings:
                warning_lines.append(
                    f"  - {w['model_name']} ({str(w['backend']).lstrip('xlite ')}): accuracy={w['accuracy']:.2f}%"
                )
            warning_lines.append("")

        report_warnings = "\n".join(warning_lines)
    else:
        report_warnings = None

    return report_main, report_warnings


def generate_summary_report(analysis: Dict, report_dir: Path, xlite_threshold: float, min_accuracy: float) -> str:
    """
    生成精度测试总结报告 (纯文本格式，用于保存到文件)

    参数:
        analysis: 分析结果字典
        report_dir: 报告目录
        xlite_threshold: xlite 与 aclgraph 精度差异阈值
        min_accuracy: 最低精度阈值

    返回:
        报告文本
    """
    current_version = get_current_version()
    current_commit = get_current_commit()
    vllm_ascend_version = get_vllm_ascend_version()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 60,
        "xlite 每日精度测试报告",
        "=" * 60,
        f"版本: {current_version}",
        f"xlite commit: {current_commit}",
        f"vllm-ascend 版本: {vllm_ascend_version}",
        f"测试时间: {current_date}",
        f"xlite与aclgraph差异阈值: {xlite_threshold}%",
        f"最低精度阈值: {min_accuracy}%",
        "=" * 60,
    ]

    # 添加警告信息
    warnings = analysis.get("warnings", [])
    if warnings:
        report_lines.append("\n[警告] 检测到以下问题:")

        # 分类警告
        xlite_warnings = [w for w in warnings if w["type"] == "xlite_vs_aclgraph"]
        low_acc_warnings = [w for w in warnings if w["type"] == "low_accuracy"]

        if xlite_warnings:
            report_lines.append("\n  xlite 精度低于 aclgraph:")
            for w in xlite_warnings:
                report_lines.append(
                    f"    - {w['model_name']} ({w['backend']}): "
                    f"xlite={w['xlite_accuracy']:.2f}%, aclgraph={w['aclgraph_accuracy']:.2f}%, "
                    f"差异 {w['difference']:.2f}%"
                )

        if low_acc_warnings:
            report_lines.append("\n  精度低于最低阈值:")
            for w in low_acc_warnings:
                report_lines.append(f"    - {w['model_name']} ({w['backend']}): accuracy={w['accuracy']:.2f}%")
    else:
        report_lines.append("\n[正常] 所有模型精度符合要求")

    report_lines.append("\n" + "=" * 60)
    report_lines.append(f"报告目录: {report_dir}")
    report_lines.append("=" * 60)

    return "\n".join(report_lines)


# ====================== 主函数 ======================
def main():
    """
    主函数 - 执行完整的自动化精度测试流程

    流程:
    1. 解析命令行参数
    2. 拉取最新代码 (编译容器)
    3. 编译项目 (编译容器)
    4. 安装 wheel 包
    5. 执行精度基准测试
    6. 解析测试报告
    7. 在单次运行内对比 xlite 与 aclgraph 精度
    8. 检查最低精度阈值
    9. 生成并发送报告
    10. 返回退出码
    """
    # 解析命令行参数
    default_model_args = (
        "--models Qwen3-30B-A3B --quantization 0 --tps 8 --eps 1 --dps 1 --xlite 2 1 0 --broadcast-xlite"
    )
    parser = argparse.ArgumentParser(description="xlite 每日自动化精度测试机器人")
    parser.add_argument(
        "-args",
        "--model-args",
        type=str,
        default=default_model_args,
        help=f"模型参数字符串，格式参考 batch_aisbench.py 的参数格式 (default: {default_model_args}）",
    )
    parser.add_argument("-p", "--skip-pull", action="store_true", help="跳过代码拉取")
    parser.add_argument("-b", "--skip-build", action="store_true", help="跳过编译")
    parser.add_argument(
        "-r", "--receiver", type=str, default="927280411401503971", help="接收者ID (default: 927280411401503971)"
    )
    parser.add_argument(
        "-t",
        "--xlite-threshold",
        type=float,
        default=XLITE_ACLGRAPH_THRESHOLD,
        help=f"xlite与aclgraph精度差异阈值 (default: {XLITE_ACLGRAPH_THRESHOLD}%%)",
    )
    parser.add_argument(
        "-m",
        "--min-accuracy",
        type=float,
        default=MIN_ACCURACY_THRESHOLD,
        help=f"最低精度阈值 (default: {MIN_ACCURACY_THRESHOLD}%%)",
    )
    parser.add_argument("-c", "--build-container", type=str, default="", help="编译容器名称")
    parser.add_argument(
        "-rd", "--report-dir", type=str, default=None, help="指定已有的报告目录（调试模式，跳过拉取、编译、测试步骤）"
    )
    parser.add_argument(
        "-e", "--env-type", type=str, default="yellow", choices=["blue", "yellow"], help="环境类型 (default: yellow)"
    )
    parser.add_argument(
        "--reference-commit",
        type=str,
        default=os.environ.get("XLITE_BOT_REFCOMMIT", ""),
        help="指定复用前期某个 xlite commit hash 测试",
    )
    parser.add_argument("-d", "--debug", action="store_true", help="调试模式，直接输出测试过程到stdout")
    parser.add_argument(
        "--timeout", type=int, default=4 * 3600, help="ais_bench 子进程超时时间（秒），默认 4h = 14400s"
    )
    parser.add_argument("--retry", type=int, default=3, help="ais_bench 失败时的重试次数，默认 3")
    parser.add_argument("--vllm-timeout", type=int, default=1800, help="vLLM 服务器启动超时时间（秒），默认 1800s")

    args = parser.parse_args()

    # 设置配置
    xlite_threshold = args.xlite_threshold
    min_accuracy = args.min_accuracy

    set_build_container(args.build_container or os.environ.get("BUILD_CONTAINER", ""))
    set_test_container(os.environ.get("TEST_CONTAINER", ""))
    set_env_type(args.env_type)
    set_machine_ip(os.environ.get("MACHINE_IP", ""))

    # 设置日志文件到默认目录（确保所有日志都能被保存）
    setup_logging(REPORT_DIR / "daily_logs", log_prefix="daily_aisbench")
    log_info("日志文件路径已设置")

    # 打印启动信息
    log_info("=" * 60)
    log_info("xlite 每日自动化精度测试机器人启动")
    log_info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"编译容器: {os.environ.get('BUILD_CONTAINER', '')}")
    log_info(f"环境类型: {args.env_type}")
    log_info(f"xlite 与 aclgraph 差异阈值: {xlite_threshold}%")
    log_info(f"最低精度阈值: {min_accuracy}%")
    log_info("=" * 60)

    # 调试模式：如果指定了报告目录，跳过前面的步骤
    if args.report_dir:
        log_info("调试模式：使用指定的报告目录，跳过拉取、编译、测试步骤")
        report_dir = Path(args.report_dir)
        if not report_dir.exists():
            log_error(f"指定的报告目录不存在: {report_dir}")
            sys.exit(1)
        log_info(f"使用报告目录: {report_dir}")
    else:
        # 步骤1: 拉取最新代码 (在编译容器中)
        if not args.skip_pull:
            if not pull_latest_code():
                log_error("代码拉取失败，终止执行")
                sys.exit(1)

        # 步骤2: 编译项目 (在编译容器中)
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

        # 步骤4: 执行精度基准测试 (在测试容器中)
        report_dir = run_aisbench_benchmark(
            model_args=args.model_args,
            debug=args.debug,
            reference_commit=args.reference_commit,
            timeout=args.timeout,
            retry=args.retry,
            vllm_timeout=args.vllm_timeout,
        )
        if not report_dir:
            log_error("精度基准测试失败，终止执行")
            sys.exit(1)

    # 步骤5: 查找并解析当前版本报告
    # 查找 CSV 报告
    csv_reports = list(report_dir.glob("aisbench_results.csv"))
    md_reports = list(report_dir.glob("aisbench_results.md"))

    if csv_reports:
        current_report = csv_reports[0]
        current_metrics = parse_aisbench_report(current_report)
    elif md_reports:
        current_report = md_reports[0]
        current_metrics = parse_aisbench_markdown_report(md_reports[0])
    else:
        log_error("未找到精度测试报告文件")
        sys.exit(1)

    log_info(f"当前版本报告: {current_report}")

    # 保存当前版本指标
    current_json = report_dir / "metrics_current.json"
    save_metrics_json(current_metrics, current_json)

    # 步骤6: 在单次运行内对比 xlite 与 aclgraph 精度，并检查最低精度阈值
    analysis_result = analyze_accuracy_within_run(current_metrics, xlite_threshold, min_accuracy)

    # 保存分析结果
    analysis_json = report_dir / "analysis.json"
    save_metrics_json(analysis_result, analysis_json)

    # 步骤7: 生成并发送报告
    summary_report = generate_summary_report(analysis_result, report_dir, xlite_threshold, min_accuracy)
    report_main, report_warnings = generate_accuracy_report(current_metrics, analysis_result, report_dir)

    # 保存报告到文件
    summary_file = report_dir / "daily_summary.txt"
    summary_file.write_text(summary_report, encoding="utf-8")
    log_info(f"总结报告已保存至: {summary_file}")

    # 打印报告
    print("\n" + summary_report)

    # 发送通知
    warnings = analysis_result.get("warnings", [])
    if report_warnings is not None:
        # 有警告: 发两条通知 [1/2] [2/2]
        log_warning("检测到精度警告，发送通知 [1/2] 数据 + [2/2] 警告详情")
        send_notification(f"<b>[1/2] </b>{report_main}", args.receiver)
        send_notification(f"<b>[2/2] </b>{report_warnings}", args.receiver)
    else:
        # 无警告: 发一条通知 [1/1]
        log_success("所有模型精度符合要求")
        send_notification(f"<b>[1/1] </b>{report_main}", args.receiver)

    # 步骤8: 设置退出码
    has_warnings = len(warnings) > 0

    if has_warnings:
        log_warning(f"检测到精度警告。报告已保存至: {report_dir}")
        log_success("自动化精度测试完成!")
        return 2

    log_success("自动化精度测试完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
