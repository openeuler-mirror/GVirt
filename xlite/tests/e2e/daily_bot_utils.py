#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xlite 每日自动化测试机器人共享工具模块

本模块提供两个机器人脚本共享的基础功能:
- 日志输出工具
- 命令执行工具
- 容器操作工具
- 代码拉取和项目编译
- 版本信息获取
- 通知发送

使用方法:
    from daily_bot_utils import (
        Colors, log_info, log_success, log_warning, log_error,
        setup_logging, run_command, run_in_container, copy_from_container,
        pull_latest_code, build_project, install_wheel,
        get_current_version, get_current_commit,
        save_metrics_json, load_metrics_json,
        send_notification, should_disable_xccl, detect_model_device
    )
"""

import os
import sys
import re
import json
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ====================== 路径配置 ======================
# 项目根目录 (容器内路径)
PROJECT_ROOT = Path("/workspaces/code/opencode/GVirt")
# xlite 项目目录
XLITE_DIR = PROJECT_ROOT / "xlite"
# e2e 测试脚本目录 (使用容器内路径，用于运行测试)
E2E_DIR = XLITE_DIR / "tests" / "e2e"

# ====================== 容器配置 ======================
# 编译容器名称 (需要在使用时设置)
BUILD_CONTAINER = ""
# 测试容器名称 (需要在使用时设置)
TEST_CONTAINER = ""
# 容器内 wheel 包存放路径
CONTAINER_WHEEL_DIR = "/workspaces/code/opencode/GVirt/xlite/dist"

# ====================== 全局配置 ======================
# Webhook URL
WEBHOOK_URL = "http://cid-service.huawei.com/service-ldap/msg/espace"
# 机器IP地址
MACHINE_IP = ""
# 环境类型: blue 或 yellow，默认 yellow
# yellow 环境需要 source /home/env.sh，blue 环境不需要
ENV_TYPE = "yellow"


# ====================== 日志输出工具类 ======================
class Colors:
    """终端颜色常量，用于美化日志输出"""

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"


# 日志文件路径
LOG_FILE: Optional[Path] = None


def setup_logging(report_dir: Path, log_prefix: str = "daily_bot"):
    """
    设置日志文件路径

    参数:
        report_dir: 报告目录
        log_prefix: 日志文件名前缀 (如 'daily_benchmark', 'daily_aisbench')
    """
    global LOG_FILE

    # 确保报告目录存在
    report_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE = report_dir / f"{log_prefix}_{timestamp}.log"


def move_log_file(new_report_dir: Path) -> Optional[Path]:
    """
    将日志文件移动到新的报告目录

    参数:
        new_report_dir: 新的报告目录

    返回:
        新的日志文件路径
    """
    global LOG_FILE
    if LOG_FILE and LOG_FILE.exists():
        old_log_file = LOG_FILE
        new_log_file = new_report_dir / old_log_file.name
        try:
            shutil.move(str(old_log_file), str(new_log_file))
            LOG_FILE = new_log_file
            log_info(f"日志文件已移动到: {LOG_FILE}")
            return new_log_file
        except Exception as e:
            log_warning(f"移动日志文件失败: {e}")
    return None


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


# ====================== 配置设置函数 ======================
def set_build_container(container_name: str):
    """
    设置编译容器名称

    参数:
        container_name: 容器名称
    """
    global BUILD_CONTAINER
    BUILD_CONTAINER = container_name


def set_test_container(container_name: str):
    """
    设置测试容器名称

    参数:
        container_name: 容器名称
    """
    global TEST_CONTAINER
    TEST_CONTAINER = container_name


def set_env_type(env_type: str):
    """
    设置环境类型

    参数:
        env_type: 环境类型 ('blue' 或 'yellow')
    """
    global ENV_TYPE
    ENV_TYPE = env_type


def set_machine_ip(ip: str):
    """
    设置机器IP地址

    参数:
        ip: IP地址
    """
    global MACHINE_IP
    MACHINE_IP = ip


# ====================== 命令执行工具函数 ======================
def run_command(
    cmd: List[str], cwd: Optional[Path] = None, check: bool = True, show_output: bool = False
) -> Tuple[int, str, str]:
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
        result = subprocess.run(cmd, cwd=cwd, text=True)
        if check and result.returncode != 0:
            log_error(f"命令执行失败，返回码: {result.returncode}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode, result.stdout or "", result.stderr or ""
    else:
        # 捕获输出
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if check and result.returncode != 0:
            log_error(f"命令执行失败: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result.returncode, result.stdout, result.stderr


# ====================== 容器操作工具函数 ======================
def run_in_container(
    container_name: str, cmd: str, check: bool = True, show_output: bool = False
) -> Tuple[int, str, str]:
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
    if container_name:
        log_info(f"在容器 {container_name} 中执行: {cmd}")
        docker_cmd = ["docker", "exec", container_name, "bash", "-c", cmd]
        return run_command(docker_cmd, check=check, show_output=show_output)
    else:
        log_info(f"在本地执行: {cmd}")
        return run_command(["bash", "-c", cmd], check=check, show_output=show_output)


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
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if container_name:
            log_info(f"从容器 {container_name} 复制 {container_path} 到 {local_path}")
            run_command(["docker", "cp", f"{container_name}:{container_path}", str(local_path)])
        else:
            log_info(f"复制 {container_path} 到 {local_path}")
            run_command(["cp", f"{container_path}", str(local_path)])
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

    REPO_URL = os.environ.get("XLITE_REPO_URL", "https://atomgit.com/openeuler/GVirt.git")
    REPO_BRANCH = os.environ.get("XLITE_REPO_BRANCH", "master")

    # 根据环境类型决定是否 source env.sh
    env_prefix = ""
    if ENV_TYPE == "yellow" and os.path.exists("/home/env.sh"):
        env_prefix = "source /home/env.sh && "

    try:
        # 在编译容器中拉取代码
        log_info("在编译容器中拉取代码...")
        log_info(f"BUILD_CONTAINER: {BUILD_CONTAINER}")
        run_in_container(BUILD_CONTAINER, f"rm -rf {PROJECT_ROOT}")
        run_in_container(BUILD_CONTAINER, f"mkdir -p {PROJECT_ROOT.parent}")
        run_in_container(BUILD_CONTAINER, f"{env_prefix}git clone -b {REPO_BRANCH} {REPO_URL} {PROJECT_ROOT}")
        log_success("编译容器仓库克隆成功!")

        if TEST_CONTAINER == BUILD_CONTAINER:
            log_info("编译容器和测试容器相同，跳过测试容器的代码拉取")
            return True

        # 在测试容器中拉取代码
        log_info("在测试容器中拉取代码...")
        run_in_container(TEST_CONTAINER, f"rm -rf {PROJECT_ROOT}")
        run_in_container(TEST_CONTAINER, f"mkdir -p {PROJECT_ROOT.parent}")
        run_in_container(TEST_CONTAINER, f"{env_prefix}git clone -b {REPO_BRANCH} {REPO_URL} {PROJECT_ROOT}")
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
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def get_xlite_lib_path() -> Path:
    """
    获取 xlite 库路径

    返回:
        xlite 库路径
    """
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


def build_project(report_dir: Path) -> Tuple[bool, Optional[Path]]:
    """
    在编译容器中编译 xlite 项目，并将 wheel 包复制到本地

    执行步骤:
    1. 清理共享目录中的 Python 缓存
    2. 在编译容器中卸载旧版本
    3. 使用容器内独立目录构建 wheel 包
    4. 清理编译缓存
    5. 将 wheel 包复制到本地

    参数:
        report_dir: 报告目录 (用于存储 wheel 包)

    返回:
        (成功标志, wheel包路径)
    """
    log_info("开始编译项目...")
    try:
        # 根据环境类型决定是否 source env.sh
        env_prefix = ""
        if ENV_TYPE == "yellow" and os.path.exists("/home/env.sh"):
            env_prefix = "source /home/env.sh && "

        # 清理共享目录中的 Python 缓存，避免多容器环境冲突
        log_info("清理共享目录中的 Python 缓存...")
        run_in_container(
            BUILD_CONTAINER,
            f"find {XLITE_DIR} -type d -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null || true",
            check=False,
        )
        run_in_container(BUILD_CONTAINER, f"find {XLITE_DIR} -name '*.pyc' -delete 2>/dev/null || true", check=False)
        run_in_container(
            BUILD_CONTAINER,
            f"rm -rf {XLITE_DIR}/build {XLITE_DIR}/dist {XLITE_DIR}/*.egg-info 2>/dev/null || true",
            check=False,
        )

        # 在编译容器中卸载旧版本
        run_in_container(BUILD_CONTAINER, "pip uninstall --yes xlite", check=False)

        # 使用容器内独立目录构建 wheel 包，避免污染共享卷
        build_dir = "/tmp/xlite_build"

        log_info(f"使用独立构建目录: {build_dir}")
        run_in_container(BUILD_CONTAINER, f"rm -rf {build_dir} && mkdir -p {build_dir}", check=False)

        # 在编译容器中构建 wheel 包
        ascend_lib_path = "/usr/local/Ascend/ascend-toolkit/8.3.RC2/fwkacllib/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/atc/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/runtime/lib64:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/devlib"
        ascend_bin_path = "/usr/local/Ascend/ascend-toolkit/8.3.RC2/atc/bin:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/bin:/usr/local/Ascend/ascend-toolkit/8.3.RC2/compiler/bin:/usr/local/Ascend/ascend-toolkit/8.3.RC2/aarch64-linux/ccec_compiler/bin"
        run_in_container(
            BUILD_CONTAINER,
            f"{env_prefix}cd {XLITE_DIR} && PATH={ascend_bin_path}:$PATH LD_LIBRARY_PATH={ascend_lib_path}:$LD_LIBRARY_PATH PYTHONPYCACHEPREFIX={build_dir}/pycache python setup.py bdist_wheel --dist-dir {build_dir}/dist",
            show_output=True,
        )

        # 查找生成的 wheel 文件
        exitcode, stdout, _ = run_in_container(
            BUILD_CONTAINER, f"ls {build_dir}/dist/xlite-*.whl 2>/dev/null || echo 'NOT_FOUND'"
        )

        if "NOT_FOUND" in stdout or not stdout.strip():
            log_error("未找到生成的 wheel 文件")
            return False, None

        wheel_name = stdout.strip().split("/")[-1]
        container_wheel_path = f"{build_dir}/dist/{wheel_name}"

        # 复制 wheel 包到本地
        local_wheel_path = report_dir / "wheels" / wheel_name
        if not copy_from_container(BUILD_CONTAINER, container_wheel_path, local_wheel_path):
            log_error("复制 wheel 包失败")
            return False, None

        # 清理容器内的临时构建目录
        run_in_container(BUILD_CONTAINER, f"rm -rf {build_dir}", check=False)

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


# ====================== 版本信息函数 ======================
def get_current_version() -> str:
    """
    获取当前 xlite 版本号

    返回:
        版本号字符串 (如 '0.1.0rc4')
    """
    # 尝试从 xlite 包获取版本号
    try:
        # 动态添加 xlite 路径
        xlite_lib_path = get_xlite_lib_path()
        if xlite_lib_path.exists():
            sys.path.insert(0, str(xlite_lib_path.parent))

        # 尝试导入 xlite 并获取版本
        try:
            import xlite

            if hasattr(xlite, "__version__"):
                return xlite.__version__
        except ImportError:
            pass

        # 尝试从 setup.py 获取版本
        setup_py = XLITE_DIR / "setup.py"
        if setup_py.exists():
            content = setup_py.read_text(encoding="utf-8")
            match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", content)
            if match:
                return match.group(1)
    except Exception as e:
        log_warning(f"获取版本号失败: {e}")

    return "unknown"


def get_current_commit() -> str:
    """
    获取当前 xlite commit hash

    返回:
        commit hash 字符串
    """
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=XLITE_DIR, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return "unknown"


def get_vllm_ascend_version() -> str:
    """
    获取 vllm-ascend 版本号

    返回:
        版本号字符串，获取失败返回 "N/A"
    """
    try:
        result = subprocess.run(["pip", "show", "vllm_ascend"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
        return "N/A"
    except Exception as e:
        log_warning(f"获取 vllm-ascend 版本失败: {e}")
        return "N/A"


def get_xlite_commit() -> str:
    """
    获取 xlite 项目的 commit 号（短格式）

    返回:
        commit 号（短格式），获取失败返回 "N/A"
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=XLITE_DIR, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "N/A"
    except Exception as e:
        log_warning(f"获取 xlite commit 失败: {e}")
        return "N/A"


# ====================== 环境检测函数 ======================
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


def get_npu_smi_version() -> Optional[str]:
    """
    获取 NPU-SMI 版本号

    返回:
        版本号字符串 (如 '25.2.1')，获取失败返回 None
    """
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 解析版本号: Version: 25.2.1
            match = re.search(r"Version:\s*([\d.]+)", result.stdout)
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
    parts = version.split(".")
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


# ====================== 数据存储函数 ======================
def save_metrics_json(metrics: Dict, json_path: Path):
    """
    将指标数据保存为 JSON 文件

    参数:
        metrics: 指标数据字典
        json_path: JSON 文件路径
    """
    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        log_info(f"指标数据已保存至: {json_path}")
    except Exception as e:
        log_error(f"保存指标数据失败: {e}")


def load_metrics_json(json_path: Path) -> Dict:
    """
    从 JSON 文件加载指标数据

    参数:
        json_path: JSON 文件路径

    返回:
        指标数据字典
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_error(f"加载指标数据失败: {e}")
        return {}


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
        log_warning("未指定接收者ID，跳过通知发送")
        print("\n" + message)
        return

    try:
        import urllib.request

        # 构建不使用代理的 opener，避免永久修改 os.environ
        # ProxyHandler({}) 创建一个不使用任何代理的处理器
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)

        # 将换行符转换为HTML换行
        html_message = message.replace("\n", "</br>")

        # 消息格式（支持HTML）
        data = json.dumps({"content": html_message, "receiver": receiver_id}).encode("utf-8")

        # 发送HTTP POST请求 (使用无代理的 opener)
        req = urllib.request.Request(WEBHOOK_URL, data=data, headers={"Content-Type": "application/json"})
        opener.open(req, timeout=60)
        log_success("通知发送成功!")
    except Exception as e:
        log_warning(f"通知发送失败: {e}")
        print("\n" + message)
