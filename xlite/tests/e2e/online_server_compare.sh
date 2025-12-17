#!/bin/bash
set -euo pipefail

# ====================== 通用函数定义======================
# 统一等待10秒并输出清晰日志
wait_10s() {
    local stage_desc=$1
    echo -e "\n======================================"
    echo "${stage_desc}"
    echo "等待 10 秒..."
    echo "======================================"
    sleep 10
}

wait_5min() {
    echo -e "\n======================================"
    echo "等待 5 min..."
    echo "======================================"
    sleep 300
}

# 启动服务（参数：服务启动脚本路径 日志文件名）
start_server() {
    local server_script=$1
    local log_file=$2
    # 清理旧日志（避免残留关键字影响匹配）
    rm -f "${log_file}"
    echo -e "\n======================================"
    echo "启动服务：${server_script} 8080 ${log_file}"
    echo "日志文件：${log_file}（独立日志）"
    echo "======================================"
    # 后台启动服务并记录PID
    bash "${server_script}" 8080 "${log_file}" &
    SERVER_PID=$!
    echo "服务进程PID：${SERVER_PID}"
}

# 等待服务启动就绪（参数：日志文件名）
wait_server_ready() {
    local log_file=$1
    TARGET_KEYWORD="Application startup complete"
    TIMEOUT=600  # 超时时间（秒），0 表示不超时
    if timeout $TIMEOUT bash -c "tail -f --pid=$$ '$log_file' 2>/dev/null | grep -q --line-buffered '$TARGET_KEYWORD'"; then
        echo -e "\n成功匹配到关键字：$TARGET_KEYWORD"
    else
        exit_code=$?
        # 区分超时和其他错误
        if [ $exit_code -eq 124 ]; then
            echo -e "\n错误：监控超时（超过 $TIMEOUT 秒未找到关键字）"
        elif [ $exit_code -eq 1 ]; then
            echo -e "\n错误：日志文件已结束，但未找到关键字"
        else
            echo -e "\n错误：监控异常退出（退出码：$exit_code）"
        fi
        exit $exit_code
    fi
}

# 执行压测脚本（参数：脚本）
run_bench_scripts_sequential() {
    # 第一个参数是压测脚本路径
    local bench_script=$1
    # 移除第一个参数，剩余的所有参数作为脚本的入参
    shift
    local script_args="$@"

    echo -e "\n======================================"
    echo "执行压测脚本：${bench_script}"
    echo "脚本参数：${script_args}"
    echo "======================================"
    
    # 执行压测脚本，并传递参数（注意引号保留参数的完整性）
    echo -e "\n启动压测：${bench_script} ${script_args}"
    bash "${bench_script}" ${script_args} || echo "${bench_script} 执行返回非零状态码（不影响后续流程）"
    echo -e "\n压测脚本执行完成！"
}

# 清理指定类型进程（避免无进程时kill报错）
cleanup_single_process() {
    local process_name=$1
    local grep_pattern=$2
    echo -e "\n正在清理 ${process_name} 进程..."
    PIDS=$(ps -ef | grep "${grep_pattern}" | grep -v grep | awk '{print $2}')
    if [ -n "${PIDS}" ]; then
        echo "找到进程PID：${PIDS}"
        echo "${PIDS}" | xargs kill -9
        echo "${process_name} 进程已终止"
    else
        echo "未找到 ${process_name} 相关进程，无需清理"
    fi
}

# 批量清理Python和VLLM进程（间隔10秒）
cleanup_all_processes() {
    cleanup_single_process "Python" "python"
    wait_10s "Python进程清理完成，准备清理VLLM进程"
    cleanup_single_process "VLLM" "VLLM"
}

# ====================== 第一轮：Xlite Decode Only 服务流程 =====================
echo "======================================"
echo "开始第一轮流程：Decode Only 服务 + 压测"
echo "======================================"

# 1. 启动Decode Only服务（日志：server_decode_only.log）
start_server "online_server_xlite_decode_only.sh" "server_decode_only.log"
wait_5min

# 2. 等待服务就绪 → 间隔10秒
wait_server_ready "server_decode_only.log"
wait_10s "服务启动稳定中，准备执行压测脚本"

# 3. 执行压测脚本
run_bench_scripts_sequential \
    "online_server_test.sh" \
    "xlite_decode_only"

# 4. 压测完成 → 间隔10秒 → 清理进程
wait_10s "压测脚本执行完成，准备清理进程"
cleanup_all_processes

# ====================== 两轮流程之间间隔10秒 ======================
wait_10s "第一轮流程全部完成，准备启动第二轮流程"

# ====================== 第二轮：Xlite Full 服务流程 =====================
echo "======================================"
echo "开始第二轮流程：Full 服务 + 压测"
echo "======================================"

# 1. 启动Full服务（日志：server_full.log）
start_server "online_server_xlite_full.sh" "server_full.log"
wait_5min

# 2. 等待服务就绪 → 间隔10秒
wait_server_ready "server_full.log"
wait_10s "服务启动稳定中，准备执行压测脚本"

# 3. 执行压测脚本
run_bench_scripts_sequential \
    "online_server_test.sh" \
    "xlite_full"

# 4. 压测完成 → 间隔10秒 → 清理进程
wait_10s "压测脚本执行完成，准备清理进程"
cleanup_all_processes

# ====================== 第三轮：Aclgraph 服务流程 =====================
echo "======================================"
echo "开始第三轮流程：Aclgraph 服务 + 压测"
echo "======================================"

# 1. 启动Full服务（日志：server_aclgraph.log）
start_server "online_server_aclgraph.sh" "server_aclgraph.log"
wait_5min

# 2. 等待服务就绪 → 间隔10秒
wait_server_ready "server_aclgraph.log"
wait_10s "服务启动稳定中，准备执行压测脚本"

# 3. 执行压测脚本
run_bench_scripts_sequential \
    "online_server_test.sh" \
    "aclgraph"

# 4. 压测完成 → 间隔10秒 → 清理进程
wait_10s "压测脚本执行完成，准备清理进程"
cleanup_all_processes

# ====================== 整体流程结束 ======================
echo -e "\n======================================"
echo "三轮流程（Xlite Decode Only + Xlite Full + Aclgraph）全部执行完成！"
echo "日志文件保留：server_decode_only.log、server_full.log、server_aclgraph.log（可后续分析）"
echo "======================================"