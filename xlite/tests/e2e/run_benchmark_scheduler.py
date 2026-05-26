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

import os
import sys
import signal
import schedule
import time
import subprocess
from datetime import datetime
from pathlib import Path

# ====================== 配置 ======================
# 脚本目录
SCRIPT_DIR = Path(__file__).parent
# Python 解释器路径
PYTHON_CMD = sys.executable
# 测试机器人脚本
BOT_SCRIPT = SCRIPT_DIR / "daily_benchmark_bot.py"
# 日志文件
LOG_FILE = SCRIPT_DIR / "scheduler.log"

# 默认配置
DEFAULT_MODEL = "moe"
DEFAULT_RECEIVER = "927280411401503971"
DEFAULT_BUILD_CONTAINER = "xlite-build"
DEFAULT_RUN_TIME = "20:00"  # 默认晚上 8:00


class BenchmarkScheduler:
    """测试任务调度器"""
    
    def __init__(self, model=DEFAULT_MODEL, receiver=DEFAULT_RECEIVER, 
                 build_container=DEFAULT_BUILD_CONTAINER, run_time=DEFAULT_RUN_TIME):
        self.model = model
        self.receiver = receiver
        self.build_container = build_container
        self.run_time = run_time
        self.running = True
        
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
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"写入日志失败: {e}")
    
    def run_benchmark(self):
        """执行测试任务"""
        self.log("=" * 60)
        self.log("开始执行每日测试任务")
        self.log("=" * 60)
        
        try:
            cmd = [
                PYTHON_CMD,
                str(BOT_SCRIPT),
                "--model", self.model,
                "--receiver", self.receiver,
                "--build-container", self.build_container
            ]
            
            self.log(f"执行命令: {' '.join(cmd)}")
            
            # 执行测试脚本
            result = subprocess.run(
                cmd,
                cwd=SCRIPT_DIR,
                capture_output=True,
                text=True
            )
            
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
    
    def setup_schedule(self):
        """设置定时任务"""
        self.log(f"设置定时任务: 每天 {self.run_time} 执行")
        schedule.every().day.at(self.run_time).do(self.run_benchmark)
        
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
        self.log(f"测试脚本: {BOT_SCRIPT}")
        self.log(f"模型类型: {self.model}")
        self.log(f"接收者ID: {self.receiver}")
        self.log(f"编译容器: {self.build_container}")
        self.log(f"执行时间: 每天 {self.run_time}")
        self.log(f"日志文件: {LOG_FILE}")
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
    parser.add_argument("--model", default=DEFAULT_MODEL, 
                       help="测试模型类型 (dense/moe/dense_quant/moe_quant/origin/quant/all)，默认 moe")
    parser.add_argument("--receiver", default=DEFAULT_RECEIVER,
                       help="接收者ID，默认 927280411401503971")
    parser.add_argument("--build-container", default=DEFAULT_BUILD_CONTAINER,
                       help="编译容器名称，默认 xlite-build")
    parser.add_argument("--run-time", default=DEFAULT_RUN_TIME,
                       help="执行时间 (HH:MM)，默认 02:00")
    parser.add_argument("--run-now", action="store_true",
                       help="立即执行一次测试，然后启动调度器")
    
    args = parser.parse_args()
    
    # 创建调度器
    scheduler = BenchmarkScheduler(
        model=args.model,
        receiver=args.receiver,
        build_container=args.build_container,
        run_time=args.run_time
    )
    
    # 如果需要立即执行
    if args.run_now:
        scheduler.log("立即执行测试...")
        scheduler.run_benchmark()
    
    # 运行调度器
    scheduler.run()


if __name__ == "__main__":
    main()
