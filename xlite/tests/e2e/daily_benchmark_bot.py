#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlite 每日自动化测试机器人

功能说明:
1. 在编译容器和测试容器中拉取最新代码
2. 在编译容器中编译项目生成 wheel 包
3. 在测试容器中执行基准测试脚本
4. 解析测试报告并与上一版本的数据对比
5. 检测性能劣化 (超过阈值时告警)
6. 发送通知到群组

架构说明:
- 定时任务在测试容器中运行
- 编译容器: 用于拉取代码和编译项目
- 测试容器: 用于拉取代码和运行基准测试
- 两个容器都从远程仓库拉取代码到 /workspaces/code/opencode/GVirt

容器启动方式:

1. 启动编译容器 (xlite-build):
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
     /bin/bash -c "while true;do echo hello;sleep 5;done"

2. 启动测试容器 (daily-test):
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
     quay.io/ascend/vllm-ascend:v0.17.0rc1-a3 \
     /bin/bash -c "while true;do echo hello;sleep 5;done"

使用方法:
    python3 daily_benchmark_bot.py --model moe --receiver "927280411401503971"
    
    可选参数:
    --model: 测试模型类型 (dense/moe/all)，默认 moe
    --skip-pull: 跳过代码拉取
    --skip-build: 跳过编译
    --receiver: 接收者ID，默认 927280411401503971
    --threshold: 性能劣化阈值，默认 0.05 (即5%)
    --build-container: 编译容器名称
    --env-type: 环境类型 (blue/yellow)，默认 yellow
               - yellow 环境: 需要执行 source /home/env.sh
               - blue 环境: 不需要 source /home/env.sh
    --compare-ttft: 启用TTFT指标作为对比指标（默认不比较TTFT）
    --compare-tpot-p99: 启用TPOT P99指标作为对比指标（默认不比较TPOT P99）

退出码:
    0 - 成功，无性能劣化
    1 - 执行失败
    2 - 成功，但检测到性能劣化

目录结构:
    PROJECT_ROOT = /workspaces/code/opencode/GVirt (容器内路径)
    REPORT_DIR = /home/daily_reports (容器内路径)
"""

import os
import sys
import re
import json
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ====================== 路径配置 ======================
# 项目根目录 (容器内路径)
PROJECT_ROOT = Path("/workspaces/code/opencode/GVirt")
# xlite 项目目录
XLITE_DIR = PROJECT_ROOT / "xlite"
# e2e 测试脚本目录 (使用容器内路径，用于运行基准测试)
E2E_DIR = XLITE_DIR / "tests" / "e2e"
# 每日报告存储目录 (宿主机路径)
REPORT_DIR = Path("/home/daily_reports")

# ====================== 容器配置 ======================
# 编译容器名称
BUILD_CONTAINER = os.environ.get("BUILD_CONTAINER", "xlite-build")
# 测试容器名称
TEST_CONTAINER = "daily-test"
# 容器内 wheel 包存放路径
CONTAINER_WHEEL_DIR = "/workspaces/code/opencode/GVirt/xlite/dist"

# ====================== 全局配置 ======================
# 性能劣化阈值，超过此比例视为性能下降
DEGRADATION_THRESHOLD = 0.05
# TTFT指标是否作为对比指标（ttft_avg 和 ttft_p99）
COMPARE_TTFT_METRICS = False
# TPOT P99指标是否作为对比指标
COMPARE_TPOT_P99_METRICS = False
# Webhook URL
WEBHOOK_URL = "http://cid-service.huawei.com/service-ldap/msg/espace"
# 机器IP地址
MACHINE_IP = os.environ.get("MACHINE_IP", "")
# 环境类型: blue 或 yellow，默认 yellow
# yellow 环境需要 source /home/env.sh，blue 环境不需要
ENV_TYPE = os.environ.get("ENV_TYPE", "yellow")


# ====================== 日志输出工具类 ======================
class Colors:
    """终端颜色常量，用于美化日志输出"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

# 日志文件路径
LOG_FILE = None


def setup_logging(report_dir: Optional[Path] = None):
    """
    设置日志文件路径
    
    参数:
        report_dir: 报告目录，如果为 None 则使用默认的 REPORT_DIR
    """
    global LOG_FILE
    if report_dir is None:
        report_dir = REPORT_DIR
    
    # 确保报告目录存在
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = report_dir / f"daily_benchmark_{timestamp}.log"


def _write_to_file(msg: str):
    """
    将消息写入日志文件
    
    参数:
        msg: 要写入的消息
    """
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"写入日志文件失败: {e}")


def log_info(msg: str):
    """输出信息级别日志 (蓝色)"""
    formatted_msg = f"[INFO] {msg}"
    print(f"{Colors.BLUE}{formatted_msg}{Colors.RESET}")
    _write_to_file(formatted_msg)


def log_success(msg: str):
    """输出成功日志 (绿色)"""
    formatted_msg = f"[SUCCESS] {msg}"
    print(f"{Colors.GREEN}{formatted_msg}{Colors.RESET}")
    _write_to_file(formatted_msg)


def log_warning(msg: str):
    """输出警告日志 (黄色)"""
    formatted_msg = f"[WARNING] {msg}"
    print(f"{Colors.YELLOW}{formatted_msg}{Colors.RESET}")
    _write_to_file(formatted_msg)


def log_error(msg: str):
    """输出错误日志 (红色)"""
    formatted_msg = f"[ERROR] {msg}"
    print(f"{Colors.RED}{formatted_msg}{Colors.RESET}")
    _write_to_file(formatted_msg)


# ====================== 命令执行工具函数 ======================
def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True, show_output: bool = False) -> Tuple[int, str, str]:
    """
    执行 shell 命令并返回结果
    
    参数:
        cmd: 命令及参数列表
        cwd: 工作目录
        check: 是否检查返回码，为 True 时非零返回码会抛出异常
        show_output: 是否实时显示命令输出
    
    返回:
        (返回码, 标准输出, 标准错误)
    """
    log_info(f"执行命令: {' '.join(cmd)}")
    
    if show_output:
        # 实时显示输出
        result = subprocess.run(
            cmd,
            cwd=cwd,
            text=True
        )
        if check and result.returncode != 0:
            log_error(f"命令执行失败，返回码: {result.returncode}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode, result.stdout or "", result.stderr or ""
    else:
        # 捕获输出
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if check and result.returncode != 0:
            log_error(f"命令执行失败: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode, result.stdout, result.stderr


# ====================== 容器操作工具函数 ======================
def run_in_container(container_name: str, cmd: str, check: bool = True, show_output: bool = False) -> Tuple[int, str, str]:
    """
    在指定容器中执行命令
    
    参数:
        container_name: 容器名称
        cmd: 要执行的命令
        check: 是否检查返回码
        show_output: 是否实时显示输出
    
    返回:
        (返回码, 标准输出, 标准错误)
    """
    log_info(f"在容器 {container_name} 中执行: {cmd}")
    docker_cmd = ["docker", "exec", container_name, "bash", "-c", cmd]
    return run_command(docker_cmd, check=check, show_output=show_output)


def copy_from_container(container_name: str, container_path: str, local_path: Path) -> bool:
    """
    从容器复制文件到本地
    
    参数:
        container_name: 容器名称
        container_path: 容器内文件路径
        local_path: 本地目标路径
    
    返回:
        True - 成功, False - 失败
    """
    log_info(f"从容器 {container_name} 复制 {container_path} 到 {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_command(["docker", "cp", f"{container_name}:{container_path}", str(local_path)])
        return True
    except Exception as e:
        log_error(f"复制文件失败: {e}")
        return False


# ====================== 代码拉取函数 ======================
def pull_latest_code() -> bool:
    """
    在编译容器和测试容器中从远程仓库拉取最新代码
    
    执行步骤:
    1. 在编译容器中删除项目根目录并克隆远程仓库
    2. 在测试容器中删除项目根目录并克隆远程仓库
    
    返回:
        True - 拉取成功
        False - 拉取失败
    """
    log_info("开始拉取最新代码...")
    
    REPO_URL = "https://atomgit.com/openeuler/GVirt.git"
    
    # 根据环境类型决定是否 source env.sh
    env_prefix = "source /home/env.sh && " if ENV_TYPE == "yellow" else ""
    
    try:
        # 在编译容器中拉取代码
        log_info("在编译容器中拉取代码...")
        run_in_container(BUILD_CONTAINER, f"rm -rf {PROJECT_ROOT}")
        run_in_container(BUILD_CONTAINER, f"mkdir -p {PROJECT_ROOT.parent}")
        run_in_container(
            BUILD_CONTAINER,
            f"{env_prefix}git clone {REPO_URL} {PROJECT_ROOT}"
        )
        log_success("编译容器仓库克隆成功!")
        
        # 在测试容器中拉取代码
        log_info("在测试容器中拉取代码...")
        run_in_container(TEST_CONTAINER, f"rm -rf {PROJECT_ROOT}")
        run_in_container(TEST_CONTAINER, f"mkdir -p {PROJECT_ROOT.parent}")
        run_in_container(
            TEST_CONTAINER,
            f"{env_prefix}git clone {REPO_URL} {PROJECT_ROOT}"
        )
        log_success("测试容器仓库克隆成功!")
        
        return True
    except Exception as e:
        log_error(f"仓库克隆失败: {e}")
        return False


# ====================== 项目编译函数 ======================
def get_python_version() -> str:
    """
    获取当前 Python 版本
    
    返回:
        Python 版本字符串 (如 '3.11.13')
    """
    import sys
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_xlite_lib_path() -> Path:
    """
    获取 xlite 库路径
    
    返回:
        xlite 库路径
    """
    import sys
    
    # 初始化 xlite 路径为 None
    xlite_path = None
    
    # 动态遍历 sys.path 查找 xlite
    for path in sys.path:
        if "site-packages" in path:
            current_path = Path(path) / "xlite"
            if current_path.exists() and current_path.is_dir():
                xlite_path = current_path
                log_success(f"成功找到 xlite 路径: {xlite_path}")
                break
    
    # 如果没找到，给出提示 + 拼接最可能的默认路径
    if not xlite_path:
        log_warning("未在 sys.path 中找到 xlite 库")
        full_ver = get_python_version()
        ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        xlite_path = Path(f"/usr/local/python{full_ver}/lib/python{ver}/site-packages/xlite")
        log_warning(f"使用默认备用路径: {xlite_path}")
    
    # 最终把 xlite_path 加入系统路径（解决导入错误）
    if xlite_path.exists():
        sys.path.append(str(xlite_path.parent))
    else:
        log_error(f"最终路径仍不存在: {xlite_path}")
    
    return xlite_path


def build_project() -> Tuple[bool, Optional[Path]]:
    """
    在编译容器中编译 xlite 项目，并将 wheel 包复制到本地
    
    执行步骤:
    1. 清理共享目录中的 Python 缓存
    2. 在编译容器中卸载旧版本
    3. 使用容器内独立目录构建 wheel 包
    4. 清理编译缓存
    5. 将 wheel 包复制到本地
    
    返回:
        (成功标志, wheel包路径)
    """
    log_info("开始编译项目...")
    try:
        # 根据环境类型决定是否 source env.sh
        env_prefix = "source /home/env.sh && " if ENV_TYPE == "yellow" else ""
        
        # 清理共享目录中的 Python 缓存，避免多容器环境冲突
        log_info("清理共享目录中的 Python 缓存...")
        run_in_container(
            BUILD_CONTAINER,
            f"find {XLITE_DIR} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true",
            check=False
        )
        run_in_container(
            BUILD_CONTAINER,
            f"find {XLITE_DIR} -name '*.pyc' -delete 2>/dev/null || true",
            check=False
        )
        run_in_container(
            BUILD_CONTAINER,
            f"rm -rf {XLITE_DIR}/build {XLITE_DIR}/dist {XLITE_DIR}/*.egg-info 2>/dev/null || true",
            check=False
        )
        
        # 在编译容器中卸载旧版本
        run_in_container(BUILD_CONTAINER, "pip uninstall --yes xlite", check=False)
        
        # 使用容器内独立目录构建 wheel 包，避免污染共享卷
        build_dir = "/tmp/xlite_build"
        
        log_info(f"使用独立构建目录: {build_dir}")
        run_in_container(
            BUILD_CONTAINER,
            f"rm -rf {build_dir} && mkdir -p {build_dir}",
            check=False
        )
        
        # 在编译容器中构建 wheel 包
        # 使用 python setup.py bdist_wheel 而不是 python -m build
        # 这样可以避免创建隔离环境，同时使用容器中已安装的 torch 和 torch-npu
        # 通过设置 PYTHONPYCACHEPREFIX 来隔离 Python 缓存
        # 通过设置 LD_LIBRARY_PATH 来让 torch-npu 找到 libhccl.so、libascend_hal.so 等系统库
        # 通过设置 PATH 来让构建过程找到 llvm-objdump 等编译工具
        ascend_lib_path="/usr/local/Ascend/ascend-toolkit/8.3.RC2/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/atc/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/runtime/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/devlib"
        ascend_bin_path="/usr/local/Ascend/ascend-toolkit/8.3.RC2/atc/bin:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/bin:/usr/local/Ascend/ascend-toolkit/8.3.RC2/compiler/bin:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/ccec_compiler/bin"
        run_in_container(
            BUILD_CONTAINER,
            f"{env_prefix}cd {XLITE_DIR} && PATH={ascend_bin_path}:$PATH LD_LIBRARY_PATH={ascend_lib_path}:$LD_LIBRARY_PATH PYTHONPYCACHEPREFIX={build_dir}/pycache python setup.py bdist_wheel --dist-dir {build_dir}/dist",
            show_output=True
        )
        
        # 查找生成的 wheel 文件
        exitcode, stdout, _ = run_in_container(
            BUILD_CONTAINER,
            f"ls {build_dir}/dist/xlite-*.whl 2>/dev/null || echo 'NOT_FOUND'"
        )
        
        if "NOT_FOUND" in stdout or not stdout.strip():
            log_error("未找到生成的 wheel 文件")
            return False, None
        
        wheel_name = stdout.strip().split("/")[-1]
        container_wheel_path = f"{build_dir}/dist/{wheel_name}"
        
        # 复制 wheel 包到本地
        local_wheel_path = REPORT_DIR / "wheels" / wheel_name
        if not copy_from_container(BUILD_CONTAINER, container_wheel_path, local_wheel_path):
            log_error("复制 wheel 包失败")
            return False, None
        
        # 清理容器内的临时构建目录
        run_in_container(
            BUILD_CONTAINER,
            f"rm -rf {build_dir}",
            check=False
        )
        
        log_success(f"项目编译成功! wheel 包: {local_wheel_path}")
        return True, local_wheel_path
    except Exception as e:
        log_error(f"项目编译失败: {e}")
        return False, None


def install_wheel(wheel_path: Path) -> bool:
    """
    在本地安装 wheel 包
    
    参数:
        wheel_path: wheel 包路径
    
    返回:
        True - 成功, False - 失败
    """
    log_info(f"安装 {wheel_path.name}...")
    try:
        # 卸载旧版本
        run_command(["pip", "uninstall", "--yes", "xlite"], check=False)
        
        # 安装新版本
        run_command(["pip", "install", str(wheel_path)])
        
        log_success("wheel 包安装成功!")
        return True
    except Exception as e:
        log_error(f"wheel 包安装失败: {e}")
        return False


# ====================== 基准测试执行函数 ======================
def detect_model_device() -> Optional[str]:
    """
    检测环境中模型所在的设备名称
    
    返回:
        设备名称 (如 'nvme0n1', 'sdb')，未找到返回 None
    """
    log_info("检测模型存储设备...")
    
    # 常见的模型存储设备名称
    possible_devices = ["nvme0n1", "nvme1n1", "sdb", "sdc", "sdd"]
    
    for device in possible_devices:
        mount_path = Path(f"/mnt/{device}/models")
        if mount_path.exists():
            # 检查是否有模型文件
            if any(mount_path.iterdir()):
                log_success(f"检测到模型设备: {device}")
                return device
    
    log_warning("未检测到模型存储设备")
    return None


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
        content = script_path.read_text(encoding='utf-8')
        
        # 替换模型路径中的设备名
        # 格式: /mnt/nvme0n1/models/... -> /mnt/{device}/models/...
        import re
        updated = re.sub(
            r'/mnt/\w+/models/',
            f'/mnt/{device}/models/',
            content
        )
        
        if updated != content:
            script_path.write_text(updated, encoding='utf-8')
            log_success(f"已更新模型路径为 /mnt/{device}/models/")
        else:
            log_info("模型路径无需更新")
        
        return True
    except Exception as e:
        log_error(f"更新模型路径失败: {e}")
        return False


def get_npu_smi_version() -> Optional[str]:
    """
    获取 NPU-SMI 版本号
    
    返回:
        版本号字符串 (如 '25.2.1')，获取失败返回 None
    """
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            # 解析版本号: Version: 25.2.1
            import re
            match = re.search(r'Version:\s*([\d.]+)', result.stdout)
            if match:
                return match.group(1)
    except Exception as e:
        log_warning(f"获取 NPU-SMI 版本失败: {e}")
    return None


def should_disable_xccl() -> bool:
    """
    根据 NPU-SMI 版本判断是否禁用 XCCL
    
    规则:
    - Version >= 25.3: 不禁用 (新版本已修复相关问题)
    - Version < 25.3: 禁用 (旧版本需要禁用 XCCL)
    - 无法获取版本: 默认禁用
    
    返回:
        True - 需要设置 XLITE_DISABLE_XCCL=True
        False - 不需要设置
    """
    version = get_npu_smi_version()
    
    if version is None:
        log_warning("无法获取 NPU-SMI 版本，默认禁用 XCCL")
        return True
    
    log_info(f"NPU-SMI 版本: {version}")
    
    # 解析主版本号和次版本号
    parts = version.split('.')
    if len(parts) >= 2:
        try:
            major = int(parts[0])
            minor = int(parts[1])
            version_num = major + minor / 10
            
            if version_num < 25.3:
                log_info(f"版本 {version} < 25.3，禁用 XCCL")
                return True
            else:
                log_info(f"版本 {version} >= 25.3，不禁用 XCCL")
                return False
        except ValueError:
            pass
    
    log_warning(f"无法解析版本号 {version}，默认禁用 XCCL")
    return True


def run_benchmark(model_type: str = "moe") -> Optional[Path]:
    """
    在测试容器中执行基准测试脚本
    
    参数:
        model_type: 模型类型 (dense/moe/dense_quant/moe_quant/origin/quant/all)
    
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
    
    # 创建当前版本报告目录 (版本号+日期)
    current_version = get_current_version()
    current_date = datetime.now().strftime("%Y%m%d")
    report_subdir = REPORT_DIR / f"xlite-{current_version}-{current_date}"
    report_subdir.mkdir(parents=True, exist_ok=True)
    
    # 将日志文件移动到报告目录
    global LOG_FILE
    if LOG_FILE and LOG_FILE.exists():
        old_log_file = LOG_FILE
        new_log_file = report_subdir / old_log_file.name
        try:
            import shutil
            shutil.move(str(old_log_file), str(new_log_file))
            LOG_FILE = new_log_file
            log_info(f"日志文件已移动到: {LOG_FILE}")
        except Exception as e:
            log_warning(f"移动日志文件失败: {e}")
    
    try:
        # 设置环境变量，用于覆盖脚本中的输出目录
        env = os.environ.copy()
        env["MAIN_OUTPUT_DIR_OVERRIDE"] = str(report_subdir)
        
        # 根据 NPU-SMI 版本决定是否禁用 XCCL
        if should_disable_xccl():
            env["XLITE_DISABLE_XCCL"] = "True"
        
        # 执行基准测试脚本
        result = subprocess.run(
            ["bash", "online_server_compare.sh", model_type],
            cwd=E2E_DIR,
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            log_error(f"基准测试执行失败: {result.stderr}")
            return None
        
        log_success("基准测试执行完成!")
        return report_subdir
        
    except Exception as e:
        log_error(f"基准测试执行异常: {e}")
        return None


# ====================== 报告解析函数 ======================
def parse_benchmark_report(report_path: Path) -> Dict:
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
    metrics = {
        "model": "",
        "timestamp": "",
        "scenarios": []
    }
    
    if not report_path.exists():
        return metrics
    
    content = report_path.read_text(encoding='utf-8')
    
    # 提取模型名称 (从标题行)
    model_match = re.search(r'## (.+?) TPS', content)
    if model_match:
        metrics["model"] = model_match.group(1).strip()
    
    # 提取报告生成时间
    time_match = re.search(r'Report Generated Time: (.+)', content)
    if time_match:
        metrics["timestamp"] = time_match.group(1).strip()
    
    # 定义表格行的正则表达式
    # 格式: | concurrency | item_type | ttft_avg | ttft_p99 | tpot_avg | tpot_p99 | qps | output_speed |
    table_pattern = re.compile(
        r'\|\s*(\d+)\s*\|\s*(baseline-aclgraph|xlite-full|xlite-decode-only|diff1|diff2)\s*\|'
        r'\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|'
        r'\s*([\d.]+|N/A)\s*\|\s*([\d.]+|N/A)\s*\|'
    )
    
    # 遍历匹配的每一行
    current_scenario = None
    for match in table_pattern.finditer(content):
        concurrency = int(match.group(1))  # 并发数
        item_type = match.group(2)          # 数据类型
        ttft_avg = match.group(3)           # TTFT 平均值
        ttft_p99 = match.group(4)           # TTFT P99值
        tpot_avg = match.group(5)           # TPOT 平均值
        tpot_p99 = match.group(6)           # TPOT P99值
        qps = match.group(7)                # 每秒请求数
        output_speed = match.group(8)       # 输出token速度
        
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
                    "xlite_decode_only": {}
                }
            
            # 安全转换为浮点数
            def safe_float(val):
                try:
                    return float(val) if val != "N/A" else None
                except:
                    return None
            
            # 构建指标数据
            data = {
                "ttft_avg": safe_float(ttft_avg),
                "ttft_p99": safe_float(ttft_p99),
                "tpot_avg": safe_float(tpot_avg),
                "tpot_p99": safe_float(tpot_p99),
                "qps": safe_float(qps),
                "output_speed": safe_float(output_speed)
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
def compare_metrics(current: Dict, baseline: Dict) -> Dict:
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
        "row_changes": []
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
                        "is_improvement": False
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
                    "is_improvement": is_improvement
                }
                
                # 如果有任何指标变化超过阈值，标记整行
                if is_degradation or is_improvement:
                    has_significant_change = True
            
            # 如果整行有显著变化，记录整行数据
            if has_significant_change:
                row_change = {
                    "concurrency": concurrency,
                    "service": service_name,
                    "changes": changes
                }
                comparison["row_changes"].append(row_change)
    
    return comparison


# ====================== 版本信息获取函数 ======================
def get_vllm_ascend_version() -> str:
    """
    获取 vllm-ascend 版本号
    
    返回:
        版本号字符串，获取失败返回 "N/A"
    """
    try:
        result = subprocess.run(
            ["pip", "show", "vllm_ascend"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
        return "N/A"
    except Exception as e:
        log_warning(f"获取 vllm-ascend 版本失败: {e}")
        return "N/A"


def get_xlite_commit() -> str:
    """
    获取 xlite 项目的 commit 号
    
    返回:
        commit 号（短格式），获取失败返回 "N/A"
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=XLITE_DIR,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "N/A"
    except Exception as e:
        log_warning(f"获取 xlite commit 失败: {e}")
        return "N/A"


# ====================== 报告生成函数 ======================
def generate_model_report(model_name: str, model_comparisons: List[Dict], report_dir: Optional[Path] = None, model_index: int = 0, total_models: int = 1) -> str:
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
    xlite_commit = get_xlite_commit()
    current_version = get_current_version()
    baseline_info = get_baseline_info()
    
    # 获取当前测试日期（从report_dir中提取）
    current_date = "unknown"
    if report_dir:
        match = re.search(r'xlite-[\w.]+-(\d{8})', str(report_dir))
        if match:
            current_date = match.group(1)
    

    lines.append(f'<font color="blue"><b>xlite {current_date} 性能测试报告 [{model_index + 1}/{total_models}]</b></font>')
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
            detail_lines.append("")
            
            if comparison.get("error"):
                detail_lines.append(f"  ⚠️  {comparison['error']}")
                detail_lines.append("")
                continue

            # 显示表格头
            detail_lines.append("  | 并发 | 服务 | TTFT Avg(ms) | TTFT P99(ms) | TPOT Avg(ms) | TPOT P99(ms) | Output Speed(tokens/s) |")
            detail_lines.append("  |------|------|-------------|-------------|-------------|-------------|----------------------|")
            
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
                metrics_order = ['ttft_avg', 'ttft_p99', 'tpot_avg', 'tpot_p99', 'output_speed']
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
    status_text = "发现性能劣化(超过5%阈值)，请关注！" if total_degradations > 0 else "性能正常，无劣化"
    
    ip_address = MACHINE_IP if MACHINE_IP else "unknown"
    report_path = f"{ip_address}:{report_dir}" if report_dir else ""
    
    # 使用HTML折叠样式（内联样式）
    lines.append(f'📊 <font color="blue"><b>{model_name}</b></font> 性能统计: <b>劣化项: {total_degradations} | 提升项: {total_improvements}</b>')
    lines.append(f"【当前版本】 vllm-ascend 版本: {vllm_ascend_version}, xlite commit: {xlite_commit}")
    lines.append(f"【对比基线】 vllm-ascend 版本: {baseline_info['vllm_ascend_version']}, xlite 版本: {baseline_info['version']}")
    lines.append(f'{status_icon} {status_text}')
    
    # 添加报告路径
    if report_path:
        lines.append(f"报告保存路径: {report_path}")
        lines.append("")
    
    # 添加详细数据（仅在性能异常或提升时显示）
    if total_degradations > 0 or total_improvements > 0:
        lines.extend(detail_lines)
    
    return "\n".join(lines)


# ====================== 数据持久化函数 ======================
def save_metrics_json(metrics: Dict, filepath: Path):
    """保存指标数据到 JSON 文件"""
    filepath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')


def load_metrics_json(filepath: Path) -> Dict:
    """从 JSON 文件加载指标数据"""
    if filepath.exists():
        return json.loads(filepath.read_text(encoding='utf-8'))
    return {}


# ====================== 历史报告查找函数 ======================
def get_current_version() -> str:
    """
    获取当前 xlite 版本号
    
    返回:
        版本号字符串 (如 '0.1.0rc4')
    """
    version_file = XLITE_DIR / "xlite" / "__init__.py"
    if version_file.exists():
        content = version_file.read_text(encoding='utf-8')
        match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", content)
        if match:
            return match.group(1)
    return "unknown"


def extract_model_info(report_path: Path) -> Optional[Dict]:
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
    match = re.search(
        r'benchmark_comparison_(.+?)_input(\d+)_output(\d+)',
        report_path.name
    )
    if match:
        return {
            "model_name": match.group(1),
            "input_len": int(match.group(2)),
            "output_len": int(match.group(3))
        }
    return None


def get_previous_version_report(model_name: str, input_len: int, output_len: int) -> Tuple[Optional[Path], Optional[str]]:
    """
    获取基线版本指定模型的测试报告文件路径
    
    参数:
        model_name: 模型名称 (如 'Qwen3-32B')
        input_len: 输入长度
        output_len: 输出长度
    
    返回:
        (报告文件路径, 版本号)，不存在则返回 (None, None)
    
    说明:
        - 基线目录: xlite-{version} (无日期，如 xlite-0.1.0rc4)
        - 当前目录: xlite-{version}-{date} (有日期，如 xlite-0.1.0rc4-20260403)
    """
    current_version = get_current_version()
    
    if current_version == "unknown":
        log_warning("无法获取当前版本号")
        return None, None
    
    # 基线目录使用当前版本号（不带日期）
    baseline_version_dir = REPORT_DIR / f"xlite-{current_version}"
    
    if not baseline_version_dir.exists():
        log_warning(f"基线目录不存在: {baseline_version_dir}")
        return None, current_version
    
    # 查找匹配模型名称、输入输出长度的报告文件
    pattern = f"benchmark_comparison_{model_name}_input{input_len}_output{output_len}_*.log"
    reports = list(baseline_version_dir.glob(pattern))
    if reports:
        return reports[0], current_version
    
    log_warning(f"基线目录中未找到模型 {model_name} (input={input_len}, output={output_len}) 的报告")
    return None, current_version


def get_baseline_info() -> Dict:
    """
    获取基线版本信息
    
    返回:
        基线版本信息字典:
        {
            "version": "版本号",
            "date": "日期",
            "commit": "commit号",
            "vllm_ascend_version": "vllm-ascend版本号"
        }
    """
    current_version = get_current_version()
    baseline_version_dir = REPORT_DIR / f"xlite-{current_version}"
    
    baseline_info = {
        "version": current_version,
        "date": "unknown",
        "commit": "unknown",
        "vllm_ascend_version": "unknown"
    }
    
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
            content = summary_file.read_text(encoding='utf-8')
            match = re.search(r'xlite commit:\s*(\S+)', content)
            if match:
                baseline_info["commit"] = match.group(1)
            match = re.search(r'vllm-ascend 版本:\s*(\S+)', content)
            if match:
                baseline_info["vllm_ascend_version"] = match.group(1)
        except Exception as e:
            log_warning(f"读取基线summary文件失败: {e}")
    
    return baseline_info


# ====================== 通知发送函数 ======================
def send_notification(message: str, receiver_id: Optional[str] = None):
    """
    发送通知到 
    
    参数:
        message: 通知消息内容（支持HTML格式）
        receiver_id: 接收者ID
    """
    log_info("发送通知到...")
    
    if not receiver_id:
        log_warning("未指定接收者ID (--receiver)，跳过通知发送")
        print("\n" + message)
        return
    
    try:
        import urllib.request
        
        # 取消代理设置
        for proxy_var in ["https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"]:
            if proxy_var in os.environ:
                del os.environ[proxy_var]
                log_info(f"已取消代理设置: {proxy_var}")
        
        # 将换行符转换为HTML换行
        html_message = message.replace("\n", "</br>")
        
        # 消息格式（支持HTML）
        data = json.dumps({
            "content": html_message,
            "receiver": receiver_id
        }).encode('utf-8')
        
        # 发送HTTP POST请求
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
        log_success("通知发送成功!")
    except Exception as e:
        log_warning(f"通知发送失败: {e}")
        print("\n" + message)


# ====================== 环境检查函数 ======================
def check_and_install_docker() -> bool:
    """
    检查并安装 docker
    
    返回:
        True - docker 可用或安装成功
        False - docker 安装失败
    """
    log_info("检查 docker 是否可用...")
    
    # 检查 docker 命令是否存在
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            log_success(f"docker 已安装: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        log_warning("docker 命令未找到，尝试安装...")
    except Exception as e:
        log_warning(f"检查 docker 失败: {e}")
    
    # 尝试安装 docker
    try:
        log_info("正在安装 docker.io...")
        result = subprocess.run(
            ["apt-get", "update"],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            log_error(f"apt-get update 失败: {result.stderr}")
            return False
        
        result = subprocess.run(
            ["apt-get", "install", "-y", "docker.io"],
            capture_output=True,
            text=True,
            timeout=600
        )
        if result.returncode != 0:
            log_error(f"docker.io 安装失败: {result.stderr}")
            return False
        
        # 验证安装
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            log_success(f"docker 安装成功: {result.stdout.strip()}")
            return True
        else:
            log_error("docker 安装后验证失败")
            return False
            
    except Exception as e:
        log_error(f"安装 docker 过程中出错: {e}")
        return False


# ====================== 主函数 ======================
def main():
    """
    主函数 - 执行完整的自动化测试流程
    
    流程:
    1. 解析命令行参数
    2. 拉取最新代码 (编译容器)
    3. 编译项目 (编译容器)
    4. 安装 wheel 包
    5. 执行基准测试
    6. 解析测试报告
    7. 与上一版本数据对比
    8. 生成并发送报告
    9. 返回退出码
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="xlite 每日自动化测试机器人")
    parser.add_argument("--model", default="moe", choices=["dense", "moe", "dense_quant", "moe_quant", "origin", "quant", "all"],
                        help="测试模型类型 (default: moe)")
    parser.add_argument("--skip-pull", action="store_true", help="跳过代码拉取")
    parser.add_argument("--skip-build", action="store_true", help="跳过编译")
    parser.add_argument("--receiver", type=str, default="927280411401503971",
                        help="接收者ID (default: 927280411401503971)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="性能劣化阈值 (default: 0.05 = 5%%)")
    parser.add_argument("--build-container", type=str, default=None,
                        help="编译容器名称")
    parser.add_argument("--report-dir", type=str, default=None,
                        help="指定已有的报告目录（调试模式，跳过拉取、编译、测试步骤）")
    parser.add_argument("--env-type", type=str, default="yellow", choices=["blue", "yellow"],
                        help="环境类型 (default: yellow)")
    parser.add_argument("--compare-ttft", action="store_true",
                        help="启用TTFT指标作为对比指标（默认不比较TTFT）")
    parser.add_argument("--compare-tpot-p99", action="store_true",
                        help="启用TPOT P99指标作为对比指标（默认不比较TPOT P99）")
    
    args = parser.parse_args()
    
    # 更新全局配置
    global DEGRADATION_THRESHOLD, BUILD_CONTAINER, ENV_TYPE, COMPARE_TTFT_METRICS, COMPARE_TPOT_P99_METRICS
    DEGRADATION_THRESHOLD = args.threshold
    ENV_TYPE = args.env_type
    COMPARE_TTFT_METRICS = args.compare_ttft
    COMPARE_TPOT_P99_METRICS = args.compare_tpot_p99
    
    # 更新容器配置
    if args.build_container:
        BUILD_CONTAINER = args.build_container
    
    # 设置日志文件到默认目录（确保所有日志都能被保存）
    setup_logging(REPORT_DIR)
    log_info(f"日志文件: {LOG_FILE}")
    
    # 打印启动信息
    log_info("=" * 60)
    log_info("xlite 每日自动化测试机器人启动")
    log_info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_info(f"编译容器: {BUILD_CONTAINER}")
    log_info(f"环境类型: {ENV_TYPE}")
    log_info("=" * 60)
    
    # 检查并安装 docker（非调试模式需要）
    if not args.report_dir:
        if not check_and_install_docker():
            log_error("docker 不可用且安装失败，终止执行")
            sys.exit(1)
    
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
            success, wheel_path = build_project()
            if not success:
                log_error("编译失败，终止执行")
                sys.exit(1)
        
        # 步骤3: 安装 wheel 包
        if wheel_path:
            if not install_wheel(wheel_path):
                log_error("wheel 包安装失败，终止执行")
                sys.exit(1)
        
        # 步骤4: 执行基准测试 (在测试容器中)
        report_dir = run_benchmark(args.model)
        if not report_dir:
            log_error("基准测试失败，终止执行")
            sys.exit(1)
    
    # 步骤5: 查找并解析当前版本报告
    current_reports = list(report_dir.glob("benchmark_comparison_*.log"))
    if not current_reports:
        log_error("未找到测试报告文件")
        sys.exit(1)
    
    # 按模型分别处理报告
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
        
        # 步骤6: 与基线版本数据对比
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
                "baseline_version": baseline_version or "unknown"
            }
        
        all_comparison_results.append(comparison_result)
    
    if not all_comparison_results:
        log_error("没有有效的对比结果")
        sys.exit(1)
    
    # 步骤7: 发送通知（按模型分开发送）
    # 按模型分组
    model_groups = {}
    for comparison in all_comparison_results:
        model_name = comparison.get("model", "Unknown")
        if not model_name:
            model_name = "Unknown"
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(comparison)
    
    # 输出模型分组信息
    log_info(f"检测到 {len(model_groups)} 个模型:")
    for model_name, comparisons in model_groups.items():
        log_info(f"  - {model_name}: {len(comparisons)} 个测试场景")
    
    # 为每个模型生成并发送独立报告
    model_names = list(model_groups.keys())
    log_info(f"准备为 {len(model_names)} 个模型发送通知")
    for idx, model_name in enumerate(model_names):
        model_comparisons = model_groups[model_name]
        log_info(f"正在生成模型 {model_name} 的报告 [{idx + 1}/{len(model_names)}]")
        model_report = generate_model_report(model_name, model_comparisons, report_dir, idx, len(model_names))
        
        # 保存模型报告到文件
        model_report_file = report_dir / f"daily_summary_{model_name}_{idx + 1}.txt"
        model_report_file.write_text(model_report, encoding='utf-8')
        log_info(f"模型 {model_name} 的报告已保存至: {model_report_file}")
        
        # 打印模型报告
        print("\n" + "=" * 60)
        print(f"模型 {model_name} 的报告 [{idx + 1}/{len(model_names)}]")
        print("=" * 60)
        print(model_report)
        print("=" * 60 + "\n")
        
        log_info(f"正在发送模型 {model_name} 的通知 [{idx + 1}/{len(model_names)}]")
        send_notification(model_report, args.receiver)
    
    # 步骤8: 设置退出码
    has_any_degradation = False
    for comparison in all_comparison_results:
        for row in comparison.get("row_changes", []):
            for change in row["changes"].values():
                if change.get("is_degradation"):
                    has_any_degradation = True
                    break
            if has_any_degradation:
                break
        if has_any_degradation:
            break
    
    if has_any_degradation:
        log_warning(f"检测到性能劣化。报告已保存至: {report_dir}")
        log_success("自动化测试完成!")
        return 2
    
    log_success("自动化测试完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
