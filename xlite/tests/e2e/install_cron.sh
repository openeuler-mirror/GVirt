#!/bin/bash
# ==============================================================================
# GVirt 每日测试机器人 - Cron 定时任务安装脚本
#
# 功能说明:
#   此脚本用于交互式地安装每日自动化测试的 cron 定时任务
#   用户可以选择执行时间、测试模型类型、配置接收者ID
#
# 使用方法:
#   bash install_cron.sh
#
# 安装后:
#   - 定时任务会自动执行 daily_benchmark_bot.py
#   - 日志保存在 /home/daily_reports/cron.log
#   - 可以使用 crontab -l 查看已安装的任务
#
# 容器启动方式:
#
# 1. 启动编译容器 (xlite-build):
#    docker run -itd --shm-size=10.24gb --net=host --privileged --cap-add=SYS_PTRACE --user root \
#      --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
#      -v /usr/local/dcmi:/usr/local/dcmi:ro \
#      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
#      -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common:ro \
#      -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
#      -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
#      -v /etc/vnpu.cfg:/etc/vnpu.cfg:ro \
#      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
#      -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
#      -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
#      --name xlite-build \
#      -v /tmp:/tmp -v /home:/home -v /mnt/sdb/:/mnt/sdb \
#      hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20260324 \
#      /bin/bash -c "while true;do echo hello;sleep 5;done"
#
# 2. 启动测试容器 (daily-test):
#    docker run -itd --shm-size=10.24gb --net=host --privileged --cap-add=SYS_PTRACE --user root \
#      --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
#      -v /usr/local/dcmi:/usr/local/dcmi:ro \
#      -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
#      -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common:ro \
#      -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
#      -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
#      -v /etc/vnpu.cfg:/etc/vnpu.cfg:ro \
#      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
#      -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
#      -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
#      --name daily-test \
#      -v /tmp:/tmp -v /home:/home -v /mnt/sdb/:/mnt/sdb \
#      -v /var/run/docker.sock:/var/run/docker.sock \
#      quay.io/ascend/vllm-ascend:v0.17.0rc1-a3 \
#      /bin/bash -c "while true;do echo hello;sleep 5;done"
#
# 注意事项:
#   - 在运行此脚本前，请确保编译容器和测试容器已启动
#   - 测试容器需要挂载 /var/run/docker.sock 以便在容器内执行 docker 命令
# ==============================================================================

set -e

# ====================== 路径配置 ======================
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 机器人脚本路径
BOT_SCRIPT="${SCRIPT_DIR}/daily_benchmark_bot.py"
# 日志目录
LOG_DIR="/home/daily_reports"
# Cron 日志文件路径
CRON_LOG="${LOG_DIR}/cron.log"

# ====================== 检测 Python 解释器 ======================
detect_python() {
    # 尝试多个可能的 Python 路径（优先使用高版本）
    local python_paths=(
        "/usr/local/python3.11.14/bin/python3"
        "/usr/local/python3.11.13/bin/python3"
        "/usr/local/python3.11.12/bin/python3"
        "/usr/local/python3.10.12/bin/python3"
        "/usr/bin/python3"
        "python3"
    )
    
    for py in "${python_paths[@]}"; do
        if command -v "$py" &> /dev/null; then
            echo "$py"
            return 0
        fi
    done
    
    # 如果都找不到，返回默认的 python3
    echo "python3"
}

PYTHON_CMD=$(detect_python)
echo "检测到 Python 解释器: ${PYTHON_CMD}"
PYTHON_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "Python 版本: ${PYTHON_VERSION}"

# ====================== 环境检查 ======================
check_environment() {
    local errors=0
    
    echo ""
    echo "检查运行环境..."
    echo ""
    
    # 检查 1: Docker socket 是否挂载
    echo "1. 检查 Docker socket..."
    if [ -S /var/run/docker.sock ]; then
        echo "   ✅ Docker socket 已挂载: /var/run/docker.sock"
    else
        echo "   ❌ Docker socket 未挂载"
        echo "      解决方法: 启动容器时添加 -v /var/run/docker.sock:/var/run/docker.sock"
        errors=$((errors + 1))
    fi
    
    # 检查 2: Docker CLI 是否安装
    echo ""
    echo "2. 检查 Docker CLI..."
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version 2>/dev/null || echo "unknown")
        echo "   ✅ Docker CLI 已安装: ${DOCKER_VERSION}"
        
        # 测试 docker 是否可以正常连接
        if docker info &> /dev/null; then
            echo "   ✅ Docker 连接正常"
        else
            echo "   ⚠️  Docker CLI 已安装但无法连接到 Docker daemon"
            echo "      请确保容器有权限访问 Docker socket"
        fi
    else
        echo "   ❌ Docker CLI 未安装"
        echo "      解决方法: "
        echo "        - Ubuntu/Debian: apt-get update && apt-get install -y docker.io"
        echo "        - CentOS/RHEL: yum install -y docker-client"
        echo "        - 或在 Dockerfile 中添加: RUN apt-get install -y docker.io"
        errors=$((errors + 1))
    fi
    
    # 检查 3: Cron 服务是否安装
    echo ""
    echo "3. 检查 Cron 服务..."
    if command -v crontab &> /dev/null; then
        echo "   ✅ Cron 已安装"
        
        # 检查 cron 服务是否运行
        if pgrep -x "cron" > /dev/null || pgrep -x "crond" > /dev/null; then
            echo "   ✅ Cron 服务正在运行"
        else
            echo "   ⚠️  Cron 已安装但服务未运行"
            echo "      启动方法: service cron start 或 service crond start"
        fi
    else
        echo "   ❌ Cron 未安装"
        echo "      解决方法: "
        echo "        - Ubuntu/Debian: apt-get update && apt-get install -y cron"
        echo "        - CentOS/RHEL: yum install -y cronie"
        echo "        - 或在 Dockerfile 中添加: RUN apt-get install -y cron"
        errors=$((errors + 1))
    fi
    
    echo ""
    echo "======================================"
    if [ ${errors} -eq 0 ]; then
        echo "环境检查通过 ✅"
        return 0
    else
        echo "环境检查失败 ❌ (发现 ${errors} 个问题)"
        echo ""
        echo "请解决上述问题后重新运行此脚本"
        return 1
    fi
}

# 执行环境检查
check_environment
if [ $? -ne 0 ]; then
    exit 1
fi

# ====================== 开始安装 ======================
echo "======================================"
echo "xlite 每日测试机器人 - Cron 任务安装"
echo "======================================"

# 显示当前系统时间，供用户确认
echo ""
echo "当前系统时间信息:"
echo "  系统时间: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""
echo "⚠️  注意: cron 任务将使用上述系统时间执行"
echo "   如果系统时间或时区不正确，定时任务将在错误的时间执行"
echo ""
read -p "系统时间是否正确? (y/n): " time_confirm

if [ "${time_confirm}" != "y" ] && [ "${time_confirm}" != "Y" ]; then
    echo ""
    echo "请先校正系统时间后再安装定时任务"
    echo ""
    echo "校正时间的方法:"
    echo "  1. 查看当前时间: date"
    echo "  2. 设置时区 (方法一): timedatectl set-timezone Asia/Shanghai"
    echo "  3. 设置时区 (方法二): ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime"
    echo "  4. 同步网络时间: ntpdate ntp.aliyun.com"
    echo "  5. 或手动设置时间: date -s 'YYYY-MM-DD HH:MM:SS'"
    echo ""
    exit 1
fi

# 检查机器人脚本是否存在
if [ ! -f "${BOT_SCRIPT}" ]; then
    echo "错误: 找不到脚本文件 ${BOT_SCRIPT}"
    exit 1
fi

# 设置脚本可执行权限
chmod +x "${BOT_SCRIPT}"

# 创建日志目录
mkdir -p "${LOG_DIR}"
chown "${USER}:${USER}" "${LOG_DIR}"

# ====================== 选择执行时间 ======================
echo ""
echo "请选择定时执行时间:"
echo "  1) 每天凌晨 2:00 执行 (推荐)"
echo "  2) 每天凌晨 4:00 执行"
echo "  3) 每天晚上 22:00 执行"
echo "  4) 自定义时间"
echo "  5) 现在立即执行"
echo ""
read -p "请输入选项 (1-5): " time_option

# 根据选择设置 cron 时间表达式
case ${time_option} in
    1) CRON_TIME="0 2 * * *" ;;      # 每天凌晨 2:00
    2) CRON_TIME="0 4 * * *" ;;      # 每天凌晨 4:00
    3) CRON_TIME="0 22 * * *" ;;     # 每天晚上 22:00
    4)
        # 自定义时间，需要用户输入完整的 cron 表达式
        echo ""
        echo "cron 时间表达式格式说明:"
        echo "  *    *    *    *    *"
        echo "  │    │    │    │    │"
        echo "  │    │    │    │    └── 星期几 (0-7, 0和7都表示周日)"
        echo "  │    │    │    └─────── 月份 (1-12)"
        echo "  │    │    └──────────── 日期 (1-31)"
        echo "  │    └───────────────── 小时 (0-23)"
        echo "  └────────────────────── 分钟 (0-59)"
        echo ""
        echo "示例:"
        echo "  0 2 * * *     - 每天凌晨 2:00"
        echo "  30 4 * * 1-5  - 周一到周五凌晨 4:30"
        echo "  0 */2 * * *   - 每两小时执行一次"
        echo ""
        read -p "请输入 cron 时间表达式: " CRON_TIME
        ;;
    5)
        # 现在立即执行，不设置 cron 任务
        CRON_TIME="now"
        ;;
    *)
        echo "无效选项，使用默认时间 (凌晨 2:00)"
        CRON_TIME="0 2 * * *"
        ;;
esac

# ====================== 选择测试模型 ======================
echo ""
echo "请选择测试模型:"
echo "  1) moe (Qwen3-30B-A3B-Instruct-2507)"
echo "  2) dense (Qwen3-32B)"
echo "  3) all (所有模型)"
echo ""
read -p "请输入选项 (1-3, 默认1): " model_option

# 根据选择设置模型类型
case ${model_option} in
    2) MODEL_TYPE="dense" ;;
    3) MODEL_TYPE="all" ;;
    *) MODEL_TYPE="moe" ;;
esac

# ====================== 配置  接收者 ======================
# 接收者ID用于发送测试结果到群组
read -p "请输入接收者ID (默认: 927280411401503971): " RECEIVER_ID
RECEIVER_ID=${RECEIVER_ID:-927280411401503971}

# ====================== 配置编译容器 ======================
echo ""
echo "请输入编译容器名称 (用于拉取代码和编译项目)"
read -p "编译容器名称 (默认: xlite-build): " BUILD_CONTAINER
BUILD_CONTAINER=${BUILD_CONTAINER:-xlite-build}

# ====================== 构建 Cron 任务命令 ======================
# 基础命令: 切换到脚本目录并执行 Python 脚本
CRON_CMD="cd ${SCRIPT_DIR} && ${PYTHON_CMD} ${BOT_SCRIPT} --model ${MODEL_TYPE} --receiver '${RECEIVER_ID}' --build-container ${BUILD_CONTAINER}"

# 添加日志重定向
CRON_CMD="${CRON_CMD} >> ${CRON_LOG} 2>&1"

# 完整的 cron 条目
CRON_ENTRY="${CRON_TIME} ${CRON_CMD}"

# ====================== 确认并安装 ======================
echo ""

# 如果选择现在立即执行
if [ "${CRON_TIME}" = "now" ]; then
    echo "将立即执行测试..."
    echo "执行命令: ${PYTHON_CMD} ${BOT_SCRIPT} --model ${MODEL_TYPE} --receiver '${RECEIVER_ID}' --build-container ${BUILD_CONTAINER}"
    echo ""
    
    read -p "确认执行? (y/n): " confirm
    
    if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
        echo "已取消"
        exit 0
    fi
    
    echo ""
    echo "======================================"
    echo "开始执行测试..."
    echo "======================================"
    cd ${SCRIPT_DIR}
    ${PYTHON_CMD} ${BOT_SCRIPT} --model ${MODEL_TYPE} --receiver "${RECEIVER_ID}" --build-container ${BUILD_CONTAINER}
    exit $?
fi

echo "将添加以下 cron 任务:"
echo "  ${CRON_ENTRY}"
echo ""

read -p "确认添加? (y/n): " confirm

if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
    echo "已取消"
    exit 0
fi

# 安装 cron 任务
# 1. 获取现有 crontab (排除旧的 daily_benchmark_bot.py 任务)
# 2. 添加新任务
# 3. 写入 crontab
(crontab -l 2>/dev/null | grep -v "daily_benchmark_bot.py" || true; echo "${CRON_ENTRY}") | crontab -

# ====================== 安装完成 ======================
echo ""
echo "======================================"
echo "Cron 任务安装成功!"
echo "======================================"
echo "日志目录: ${LOG_DIR}"
echo "Cron 日志: ${CRON_LOG}"
echo ""
echo "常用命令:"
echo "  查看定时任务: crontab -l"
echo "  编辑定时任务: crontab -e"
echo "  删除定时任务: crontab -r"
echo "  手动执行测试: ${PYTHON_CMD} ${BOT_SCRIPT} --model ${MODEL_TYPE} --receiver '${RECEIVER_ID}' --build-container ${BUILD_CONTAINER}"
echo "======================================"
