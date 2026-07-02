#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GVirt 每日测试机器人调度器

使用 schedule 库实现定时任务

安装依赖:
    pip install schedule

使用方法:
    python3 run_benchmark_scheduler.py

特性:
    - 每天凌晨 2:00 自动执行测试
    - 支持 Ctrl+C 退出
    - 自动记录日志
    - 支持自定义执行时间
"""

import sys
import signal
import schedule
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal

# ====================== 配置 ======================
# 脚本目录
SCRIPT_DIR = Path(__file__).parent
# Python 解释器路径
PYTHON_CMD = sys.executable

# 默认配置
DEFAULT_RECEIVER = "927280411401503971"
DEFAULT_BUILD_CONTAINER = ""  # 默认在当前容器构建 xlite 包
DEFAULT_RUN_TIME = "20:00"  # 默认晚上 8:00


class BenchmarkScheduler:
    """测试任务调度器"""

    def __init__(
        self,
        type: Literal["bench", "accuracy"] = "bench",
        script: Path = SCRIPT_DIR / "daily_benchmark_bot.py",
        model: str = "moe",
        receiver: str = DEFAULT_RECEIVER,
        build_container: str = DEFAULT_BUILD_CONTAINER,
        run_time: str = DEFAULT_RUN_TIME,
        log_file: Path = SCRIPT_DIR / "scheduler.log",
        debug: bool = False,
        timeout: int = 4 * 3600,
        retry: int = 3,
        vllm_timeout: int = 1800,
    ):
        self.type = type
        self.script = script.resolve()
        self.model = model
        self.receiver = receiver
        self.build_container = build_container
        self.run_time = run_time
        self.running = True
        self.log_file = log_file
        self.debug = debug
        self.timeout = timeout
        self.retry = retry
        self.vllm_timeout = vllm_timeout

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        self.log(f"收到信号 {signum}，准备退出...")
        self.running = False

    def log(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)

        # 写入日志文件
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"写入日志失败: {e}")

    def run_benchmark(self):
        """执行测试任务"""
        self.log("=" * 60)
        self.log(f"开始执行每日测试任务 - {self.type.upper()} - 模型: {self.model}")
        self.log("=" * 60)

        try:
            cmd = [
                PYTHON_CMD,
                str(self.script),
                "--model",
                self.model,
                "--receiver",
                self.receiver,
                "--build-container",
                self.build_container,
            ]

            self.log(f"执行命令: {' '.join(cmd)}")

            # 执行测试脚本
            result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=not self.debug, text=True)

            # 记录输出
            if result.stdout:
                self.log(f"标准输出:\n{result.stdout}")

            if result.stderr:
                self.log(f"标准错误:\n{result.stderr}")

            if result.returncode == 0:
                self.log("✅ 测试任务执行成功")
            elif result.returncode == 2:
                self.log("⚠️  测试完成但检测到性能劣化")
            else:
                self.log(f"❌ 测试任务执行失败，返回码: {result.returncode}")

        except Exception as e:
            self.log(f"❌ 执行测试任务时发生异常: {e}")

        self.log("=" * 60)
        self.log("测试任务结束")
        self.log("=" * 60)

    def run_accuracy(self):
        """执行精度测试任务"""
        self.log("=" * 60)
        self.log(f"开始执行每日精度测试任务 - {self.type.upper()}")
        self.log("=" * 60)

        try:
            cmd = [
                PYTHON_CMD,
                str(self.script),
                "--model-args",
                self.model,
                "--receiver",
                self.receiver,
                "--build-container",
                self.build_container,
                "--timeout",
                str(self.timeout),
                "--retry",
                str(self.retry),
                "--vllm-timeout",
                str(self.vllm_timeout),
            ]
            if self.debug:
                cmd.append("--debug")

            self.log(f"执行命令: {' '.join(cmd)}")

            # 执行测试脚本
            result = subprocess.run(cmd, cwd=SCRIPT_DIR, capture_output=not self.debug, text=True)

            # 记录输出
            if result.stdout:
                self.log(f"标准输出:\n{result.stdout}")

            if result.stderr:
                self.log(f"标准错误:\n{result.stderr}")

            if result.returncode == 0:
                self.log("✅ 精度测试任务执行成功，无警告")
            elif result.returncode == 2:
                self.log("⚠️  精度测试完成但检测到精度警告")
            else:
                self.log(f"❌ 精度测试任务执行失败，返回码: {result.returncode}")

        except Exception as e:
            self.log(f"❌ 执行精度测试任务时发生异常: {e}")

        self.log("=" * 60)
        self.log("精度测试任务结束")
        self.log("=" * 60)

    def setup_schedule(self):
        """设置定时任务"""
        self.log(f"设置定时任务: 每天 {self.run_time} 执行")
        task = self.run_benchmark if self.type == "bench" else self.run_accuracy
        schedule.every().day.at(self.run_time).do(task)

        # 显示下次执行时间
        next_run = schedule.next_run()
        if next_run:
            self.log(f"下次执行时间: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")

    def run(self):
        """运行调度器"""
        self.log("========================================")
        self.log("GVirt 每日测试机器人调度器启动")
        self.log("========================================")
        self.log(f"Python 解释器: {PYTHON_CMD}")
        self.log(f"测试脚本: {self.script}")
        self.log(f"模型类型: {self.model}")
        self.log(f"接收者ID: {self.receiver}")
        self.log(f"编译容器: {self.build_container}")
        self.log(f"执行时间: 每天 {self.run_time}")
        self.log(f"日志文件: {self.log_file}")
        self.log("========================================")

        # 设置定时任务
        self.setup_schedule()

        # 主循环
        self.log("调度器已启动，等待定时任务...")
        self.log("按 Ctrl+C 退出")

        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # 每分钟检查一次
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"调度器运行异常: {e}")
                time.sleep(60)

        self.log("调度器已停止")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="GVirt 每日测试机器人调度器")
    parser.add_argument(
        "-s",
        "--script",
        type=str,
        choices=["bench", "accuracy"],
        default="bench",
        help="测试脚本类型，bench: 性能测试，accuracy: 准确率测试，默认 bench",
    )
    parser.add_argument(
        "-l", "--log", type=Path, default=None, help=f"日志文件路径，默认 {SCRIPT_DIR}/scheduler-[bench|accuracy].log"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="moe",
        choices=["dense", "moe", "dense_quant", "moe_quant", "origin", "quant", "all"],
        help="*性能*测试模型类型 (dense/moe/dense_quant/moe_quant/origin/quant/all)，默认 moe；仅 bench 模式有效",
    )
    default_model_args = (
        "--models Qwen3-32B-w8a8-nopdmix Qwen3-30B-A3B-Instruct-2507 Qwen3-VL-8B-Instruct GLM-4.7-W8A8-floatmtp MiniMax-M2.7-w8a8-QuaRot",
        "--tps 4 4 4 8 8",
        "--eps 0 1 0 1 1",
        "--dps 1 1 1 1 1",
        "--xlite 2 1 0",
        "--broadcast-xlite",
        "--max-num-seqs 256",
    )
    parser.add_argument(
        "-args",
        "--model-args",
        type=str,
        default=" ".join(default_model_args),
        help=f"*精度*测试模型参数字符串，格式参考 batch_aisbench.py 的参数格式 (仅 accuracy 模式有效，默认: {default_model_args})",
    )
    parser.add_argument(
        "-r", "--receiver", type=str, default=DEFAULT_RECEIVER, help="接收者ID，默认 927280411401503971"
    )
    parser.add_argument(
        "-bc", "--build-container", type=str, default=DEFAULT_BUILD_CONTAINER, help="编译容器名称，默认在当前容器构建"
    )
    parser.add_argument("-rt", "--run-time", type=str, default=DEFAULT_RUN_TIME, help="执行时间 (HH:MM)，默认 02:00")
    parser.add_argument("-rn", "--run-now", action="store_true", help="立即执行一次测试，然后启动调度器")
    parser.add_argument("-d", "--debug", action="store_true", help="调试模式，直接输出测试过程到stdout")
    parser.add_argument(
        "--timeout", type=int, default=4 * 3600, help="ais_bench 子进程超时时间（秒），默认 4h = 14400s"
    )
    parser.add_argument("--retry", type=int, default=3, help="ais_bench 失败时的重试次数，默认 3")
    parser.add_argument("--vllm-timeout", type=int, default=1800, help="vLLM 服务器启动超时时间（秒），默认 1800s")

    args = parser.parse_args()

    if args.script == "bench":
        script_path = SCRIPT_DIR / "daily_benchmark_bot.py"
    else:
        script_path = SCRIPT_DIR / "daily_aisbench_bot.py"
    log_file = args.log if args.log else SCRIPT_DIR / f"scheduler-{args.script}.log"

    # 创建调度器
    scheduler = BenchmarkScheduler(
        type=args.script,
        script=script_path,
        model=args.model if args.script == "bench" else args.model_args,
        receiver=args.receiver,
        build_container=str(args.build_container).strip(),
        run_time=args.run_time,
        log_file=log_file,
        debug=args.debug,
        timeout=args.timeout,
        retry=args.retry,
        vllm_timeout=args.vllm_timeout,
    )

    # 如果需要立即执行
    if args.run_now:
        scheduler.log("立即执行测试...")
        if args.script == "bench":
            scheduler.run_benchmark()
        else:
            scheduler.run_accuracy()

    # 运行调度器
    scheduler.run()


if __name__ == "__main__":
    main()
