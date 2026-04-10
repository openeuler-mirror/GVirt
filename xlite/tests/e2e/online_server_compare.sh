#!/bin/bash
set -uo pipefail

# ====================== 参数解析 ======================
MODEL_TYPE=${1:-moe}

# 定义模型配置
declare -A MODELS
MODELS[dense]="Qwen3-32B|/mnt/nvme0n1/models/Qwen3-32B"
MODELS[moe]="Qwen3-30B-A3B-Instruct-2507|/mnt/nvme0n1/models/Qwen3-30B-A3B-Instruct-2507"

# 确定要测试的模型列表
declare -a TEST_MODELS
if [ "${MODEL_TYPE}" = "all" ]; then
    TEST_MODELS=("dense" "moe")
elif [ "${MODEL_TYPE}" = "dense" ] || [ "${MODEL_TYPE}" = "moe" ]; then
    TEST_MODELS=("${MODEL_TYPE}")
else
    echo "错误：无效的模型类型 '${MODEL_TYPE}'，请使用 'dense'、'moe' 或 'all'"
    echo "用法：$0 [dense|moe|all]"
    echo "  dense  - 测试 Qwen3-32B 模型"
    echo "  moe    - 测试 Qwen3-30B-A3B-Instruct-2507 模型（默认）"
    echo "  all    - 测试所有模型"
    exit 1
fi

echo "======================================"
echo "测试模型类型：${MODEL_TYPE}"
echo "======================================"

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

# 格式化时间显示（参数：秒数）
# 返回格式：X小时Y分钟Z秒（如果小时为0则不显示小时）
format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    
    if [ ${hours} -gt 0 ]; then
        echo "${hours}小时${minutes}分钟${seconds}秒"
    else
        echo "${minutes}分钟${seconds}秒"
    fi
}

# 启动服务（参数：服务启动脚本路径 日志文件名名 模型路径 max_num_batched_tokens max_num_seqs max_model_len tensor_parallel_size）
start_server() {
    local server_script=$1
    local log_file=$2
    local model_path=$3
    local max_num_batched_tokens=$4
    local max_num_seqs=$5
    local max_model_len=$6
    local tensor_parallel_size=$7
    # 清理旧日志（避免残留关键字影响匹配）
    rm -f "${log_file}"
    echo -e "\n======================================"
    echo "启动服务：${server_script} 8080 ${log_file} ${model_path} ${max_num_batched_tokens} ${max_num_seqs} ${max_model_len} ${tensor_parallel_size}"
    echo "日志文件：${log_file}（独立日志）"
    echo "======================================"
    # 后台启动服务并记录PID
    bash "${server_script}" 8080 "${log_file}" "${model_path}" "${max_num_batched_tokens}" "${max_num_seqs}" "${max_model_len}" "${tensor_parallel_size}" &
    SERVER_PID=$!
    echo "服务进程PID：${SERVER_PID}"
}

# 等待服务启动就绪（参数：日志文件名）
wait_server_ready() {
    local log_file=$1
    TARGET_KEYWORD="Application startup complete"
    TIMEOUT=600  # 超时时间（秒），0 表示不超时
    CHECK_INTERVAL=5  # 检查间隔（秒）
    
    echo "等待服务启动完成（关键字：${TARGET_KEYWORD}）..."
    echo "超时时间：${TIMEOUT} 秒，检查间隔：${CHECK_INTERVAL} 秒"
    
    local elapsed_time=0
    while [ $elapsed_time -lt $TIMEOUT ]; do
        # 检查日志文件是否存在
        if [ -f "$log_file" ]; then
            # 检查日志文件中是否包含关键字
            if grep -q "$TARGET_KEYWORD" "$log_file" 2>/dev/null; then
                echo -e "\n成功匹配到关键字：$TARGET_KEYWORD"
                echo "等待 10 秒让服务稳定..."
                sleep 10
                return 0
            fi
        fi

        # 等待指定间隔
        sleep $CHECK_INTERVAL
        elapsed_time=$((elapsed_time + CHECK_INTERVAL))
    done
    
    # 超时处理
    echo -e "\n错误：监控超时（超过 ${TIMEOUT} 秒未找到关键字：${TARGET_KEYWORD}）"
    if [ -f "$log_file" ]; then
        echo "日志文件最后 20 行："
        tail -n 20 "$log_file"
    else
        echo "日志文件不存在：${log_file}"
    fi
    return 1
}

# 执行压测脚本（参数：脚本）
run_bench_scripts_sequential() {
    # 第一个参数是压测脚本路径
    local bench_script=$1
    # 移除第一个参数，剩余的所有参数作为脚本的入参
    shift
    local script_args=("$@")

    echo -e "\n======================================"
    echo "执行压测脚本：${bench_script}"
    echo "脚本参数：${script_args[@]}"
    echo "======================================"
    
    # 执行压测脚本，并传递参数（使用数组保持参数的完整性）
    echo -e "\n启动压测：${bench_script} ${script_args[@]}"
    bash "${bench_script}" "${script_args[@]}" || echo "${bench_script} 执行返回非零状态码（不影响后续流程）"
    echo -e "\n压测脚本执行完成！"
}

# 执行单个场景的完整测试流程
# 参数说明：
#   $1 - scenario_desc: 场景描述
#   $2 - input_len: 输入token长度
#   $3 - output_len: 输出token长度
#   $4 - concurrency_list: 并发度列表（空格分隔）
#   $5 - num_prompts_multiplier: num_prompts_multiplier倍数（num_prompts = maxconcurrency * num_prompts_multiplier）
#   $6 - model_path: 模型路径
#   $7 - max_num_batched_tokens: 最大批处理token数
#   $8 - max_num_seqs: 最大序列数
#   $9 - max_model_len: 最大模型长度
#   $10 - tensor_parallel_size: 张量并行大小
run_scenario_test() {
    local scenario_desc=$1
    local input_len=$2
    local output_len=$3
    local concurrency_list=$4
    local num_prompts_multiplier=$5
    local model_path=$6
    local max_num_batched_tokens=$7
    local max_num_seqs=$8
    local max_model_len=$9
    local tensor_parallel_size=${10}
    local main_output_dir=${11}
    
    # 提取模型名称（从路径中获取最后一部分）
    local model_name=$(basename "${model_path}")
    
    # 记录开始时间
    local start_time=$(date +%s)
    local start_time_formatted=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "======================================"
    echo "场景：${scenario_desc}"
    echo "输入长度：${input_len}，输出长度：${output_len}"
    echo "并发列表：${concurrency_list}"
    echo "num_prompts_multiplier：${num_prompts_multiplier}"
    echo "模型路径：${model_path}"
    echo "max_num_batched_tokens：${max_num_batched_tokens}"
    echo "max_num_seqs：${max_num_seqs}"
    echo "max_model_len：${max_model_len}"
    echo "tensor_parallel_size：${tensor_parallel_size}"
    echo "开始时间：${start_time_formatted}"
    echo "======================================"
    
    # Xlite Decode Only 服务
    echo -e "\n>>> 测试 Xlite Decode Only 服务"
    local decode_only_log="${main_output_dir}/server_${model_name}_input${input_len}_output${output_len}_xlite_decode_only.log"
    start_server "online_server_xlite_decode_only.sh" "${decode_only_log}" "${model_path}" "${max_num_batched_tokens}" "${max_num_seqs}" "${max_model_len}" "${tensor_parallel_size}"
    if ! wait_server_ready "${decode_only_log}"; then
        echo "警告：Xlite Decode Only 服务启动失败，跳过此服务"
        cleanup_all_processes
    else
        run_bench_scripts_sequential "online_server_test.sh" "xlite_decode_only_${model_name}" "${input_len}" "${output_len}" "qwen" "${model_path}" "127.0.0.1" "8080" "${concurrency_list}" "${num_prompts_multiplier}" "${main_output_dir}"
        wait_10s "压测脚本执行完成，准备清理进程"
        cleanup_all_processes
    fi
    wait_10s "Xlite Decode Only 测试完成，准备下一个服务"
    
    # Xlite Full 服务
    echo -e "\n>>> 测试 Xlite Full 服务"
    local full_log="${main_output_dir}/server_${model_name}_input${input_len}_output${output_len}_xlite_full.log"
    start_server "online_server_xlite_full.sh" "${full_log}" "${model_path}" "${max_num_batched_tokens}" "${max_num_seqs}" "${max_model_len}" "${tensor_parallel_size}"
    if ! wait_server_ready "${full_log}"; then
        echo "警告：Xlite Full 服务启动失败，跳过此服务"
        cleanup_all_processes
    else
        run_bench_scripts_sequential "online_server_test.sh" "xlite_full_${model_name}" "${input_len}" "${output_len}" "qwen" "${model_path}" "127.0.0.1" "8080" "${concurrency_list}" "${num_prompts_multiplier}" "${main_output_dir}"
        wait_10s "压测脚本执行完成，准备清理进程"
        cleanup_all_processes
    fi
    wait_10s "Xlite Full 测试完成，准备下一个服务"
    
    # Aclgraph 服务
    echo -e "\n>>> 测试 Aclgraph 服务"
    local aclgraph_log="${main_output_dir}/server_${model_name}_input${input_len}_output${output_len}_aclgraph.log"
    start_server "online_server_aclgraph.sh" "${aclgraph_log}" "${model_path}" "${max_num_batched_tokens}" "${max_num_seqs}" "${max_model_len}" "${tensor_parallel_size}"
    if ! wait_server_ready "${aclgraph_log}"; then
        echo "警告：Aclgraph 服务启动失败，跳过此服务"
        cleanup_all_processes
    else
        run_bench_scripts_sequential "online_server_test.sh" "aclgraph_${model_name}" "${input_len}" "${output_len}" "qwen" "${model_path}" "127.0.0.1" "8080" "${concurrency_list}" "${num_prompts_multiplier}" "${main_output_dir}"
        wait_10s "压测脚本执行完成，准备清理进程"
        cleanup_all_processes
    fi
    wait_10s "Aclgraph 测试完成，准备下一个场景"
    
    # 计算和和显示耗时
    local end_time=$(date +%s)
    local end_time_formatted=$(date "+%Y-%m-%d %H:%M:%S")
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "\n======================================"
    echo "场景 ${scenario_desc} 全部完成！"
    echo "结束时间：${end_time_formatted}"
    echo "总耗时：${hours}小时 ${minutes}分钟 ${seconds}秒（${duration}秒）"
    echo "======================================"
    
    # 返回耗时（秒数）
    echo "${duration}"
}

# 清理指定类型进程（避免无进程时kill报错，排除当前进程）
cleanup_single_process() {
    local process_name=$1
    local grep_pattern=$2
    echo -e "\n正在清理 ${process_name} 进程..."
    # 排除当前进程 ($$) 和父进程 ($PPID)，避免杀掉 daily_benchmark_bot.py 和 run_benchmark_scheduler.py
    PIDS=$(ps -ef | grep "${grep_pattern}" | grep -v grep | grep -v "daily_benchmark_bot" | grep -v "run_benchmark_scheduler" | awk '{print $2}')
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

# 处理场景数据并生成对比报告
process_scenario_data() {
    local scenario_desc=$1
    local input_len=$2
    local output_len=$3
    local model_name=$4
    local tensor_parallel_size=$5
    local main_output_dir=$6
    
    echo -e "\n======================================"
    echo "处理场景数据：${scenario_desc}"
    echo "======================================"
    
    # 构建三个服务的结果目录路径
    local aclgraph_dir="${main_output_dir}/result_input_${input_len}_output_${output_len}_aclgraph_${model_name}"
    local xlite_full_dir="${main_output_dir}/result_input_${input_len}_output_${output_len}_xlite_full_${model_name}"
    local xlite_decode_only_dir="${main_output_dir}/result_input_${input_len}_output_${output_len}_xlite_decode_only_${model_name}"
    
    # 检查目录是否存在
    if [ ! -d "${aclgraph_dir}" ]; then
        echo "警告：目录 ${aclgraph_dir} 不存在，跳过数据处理"
        return 1
    fi
    if [ ! -d "${xlite_full_dir}" ]; then
        echo "警告：目录 ${xlite_full_dir} 不存在，跳过数据处理"
        return 1
    fi
    if [ ! -d "${xlite_decode_only_dir}" ]; then
        echo "警告：目录 ${xlite_decode_only_dir} 不存在，跳过数据处理"
        return 1
    fi
    
    # 构建输出文件名（包含年月日时间信息和tensor_parallel_size）
    local date_suffix=$(date +%Y%m%d)
    local output_file="${main_output_dir}/benchmark_comparison_${model_name}_input${input_len}_output${output_len}_tp${tensor_parallel_size}_${date_suffix}.log"
    
    echo "调用 process_data.py 处理数据..."
    echo "  baseline-aclgraph: ${aclgraph_dir}"
    echo "  xlite-full: ${xlite_full_dir}"
    echo "  xlite-decode-only: ${xlite_decode_only_dir}"
    echo "  输出文件: ${output_file}"
    echo "  模型名称: ${model_name}"
    
    # 调用 process_data.py
    python process_data.py \
        "${aclgraph_dir}" \
        "${xlite_full_dir}" \
        "${xlite_decode_only_dir}" \
        -o "${output_file}" \
        -m "${model_name}"
    
    if [ $? -eq 0 ]; then
        echo -e "\n数据处理完成！报告已生成：${output_file}"
        return 0
    else
        echo -e "\n错误：数据处理失败"
        return 1
    fi
}

# ====================== 场景测试 ======================

# 清理代理环境变量
echo -e "\n清理代理环境变量..."
unset https_proxy && unset http_proxy && unset HTTPS_PROXY && unset HTTP_PROXY
echo "代理环境变量已清理"

# 创建带日期的主输出文件夹（支持环境变量覆盖）
if [[ -n "${MAIN_OUTPUT_DIR_OVERRIDE:-}" ]]; then
    MAIN_OUTPUT_DIR="${MAIN_OUTPUT_DIR_OVERRIDE}"
else
    MAIN_OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${MAIN_OUTPUT_DIR}"
echo -e "\n======================================"
echo "主输出目录：${MAIN_OUTPUT_DIR}"
echo "======================================"

# 记录总测试开始时间
total_start_time=$(date +%s)
total_start_time_formatted=$(date "+%Y-%m-%d %H:%M:%S")

# 定义场景配置数组
# 格式：input_len|output_len|concurrency_list|num_prompts_multiplier|max_num_batched_tokens|max_num_seqs|max_model_len|tensor_parallel_size
declare -A SCENARIOS
SCENARIOS[1]="512|512|1 16 32 48 64 96|10|65536|96|7168|8"
SCENARIOS[2]="3584|1536|1 16 32 48 64|10|65536|64|7168|8"
SCENARIOS[3]="8192|1024|1 16 32 48 64|10|65536|64|11264|8"
SCENARIOS[4]="102400|10240|1 16|1|136192|16|136192|8"

# 定义每个模型的场景数量
declare -A MODEL_SCENARIO_COUNT
MODEL_SCENARIO_COUNT[dense]=3
MODEL_SCENARIO_COUNT[moe]=4

# 用于存储每个模型的场景测试结果
declare -A MODEL_TEST_RESULTS

# 遍历所有要测试的模型
for model_type in "${TEST_MODELS[@]}"; do
    IFS='|' read -r TEST_MODEL TEST_MODEL_PATH <<< "${MODELS[$model_type]}"
    
    # 获取当前模型需要测试的场景数量
    MODEL_SCENARIO_LIMIT=${MODEL_SCENARIO_COUNT[$model_type]}
    
    echo "======================================"
    echo "开始测试模型：${TEST_MODEL}"
    echo "测试场景数量：${MODEL_SCENARIO_LIMIT}"
    echo "======================================"
    
    # 初始化当前模型的测试结果
    MODEL_TEST_RESULTS[$TEST_MODEL]=""
    
    # 遍历场景执行测试
    for ((i=1; i<=MODEL_SCENARIO_LIMIT; i++)); do
        IFS='|' read -r input_len output_len concurrency_list num_prompts_multiplier max_num_batched_tokens max_num_seqs max_model_len tensor_parallel_size <<< "${SCENARIOS[$i]}"
        
        # 记录场景测试结果
        scenario_result="失败"
        scenario_duration="N/A"
        
        # 执行场景测试（捕获错误和耗时）
        scenario_output=$(run_scenario_test \
            "${TEST_MODEL}_场景${i}：输入${input_len}token,输出${output_len}token" \
            "${input_len}" \
            "${output_len}" \
            "${concurrency_list}" \
            "${num_prompts_multiplier}" \
            "${TEST_MODEL_PATH}" \
            "${max_num_batched_tokens}" \
            "${max_num_seqs}" \
            "${max_model_len}" \
            "${tensor_parallel_size}" \
            "${MAIN_OUTPUT_DIR}" | tee /dev/tty)
        scenario_exit_code=${PIPESTATUS[0]}
        
        # 检查场景测试是否成功
        if [ ${scenario_exit_code} -ne 0 ]; then
            echo -e "\n警告：场景${i}测试失败，跳过数据处理，继续下一个场景"
            continue
        fi
        
        # 场景测试成功，标记为成功并提取耗时
        scenario_result="成功"
        scenario_duration=$(echo "${scenario_output}" | tail -n 1)
        scenario_duration_formatted=$(format_duration ${scenario_duration})
        
        # 处理场景数据（捕获错误）
        if ! process_scenario_data "${TEST_MODEL}_场景${i}：输入${input_len}token,输出${output_len}token" "${input_len}" "${output_len}" "${TEST_MODEL}" "${tensor_parallel_size}" "${MAIN_OUTPUT_DIR}"; then
            echo -e "\n警告：场景${i}数据处理失败，继续下一个场景"
            scenario_result="数据处理失败"
        fi
        
        # 记录场景测试结果
        MODEL_TEST_RESULTS[$TEST_MODEL]="${MODEL_TEST_RESULTS[$TEST_MODEL]}场景${i}：输入${input_len}token,输出${output_len}token,并发数：${concurrency_list},tensor_parallel_size：${tensor_parallel_size},耗时：${scenario_duration_formatted} - ${scenario_result}\n"
    done
    
    echo -e "\n======================================"
    echo "模型 ${TEST_MODEL} 的测试场景全部执行完成！"
    echo "======================================"
done

# 计算和显示总耗时
total_end_time=$(date +%s)
total_end_time_formatted=$(date "+%Y-%m-%d %H:%M:%S")
total_duration=$((total_end_time - total_start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))
total_seconds=$((total_duration % 60))

# ====================== 整体流程结束 ======================
echo -e "\n======================================"
echo "所有测试完成！"
echo "开始时间：${total_start_time_formatted}"
echo "结束时间：${total_end_time_formatted}"
echo "总耗时：${total_hours}小时 ${total_minutes}分钟 ${total_seconds}秒（${total_duration}秒）"
echo ""
echo "======================================"
echo "测试结果汇总："
echo "======================================"
for model_type in "${TEST_MODELS[@]}"; do
    IFS='|' read -r TEST_MODEL TEST_MODEL_PATH <<< "${MODELS[$model_type]}"
    MODEL_SCENARIO_LIMIT=${MODEL_SCENARIO_COUNT[$model_type]}
    
    echo ""
    echo "模型：${TEST_MODEL}"
    echo "测试场景数量：${MODEL_SCENARIO_LIMIT}"
    echo "场景测试结果："
    echo -e "${MODEL_TEST_RESULTS[$TEST_MODEL]}"
done
echo ""
echo "======================================"
echo "每个场景都测试了三种服务：Xlite Decode Only、Xlite Full、Aclgraph"
echo "======================================"
echo ""
echo "生成的对比报告文件："
date_suffix=$(date +%Y%m%d)
for model_type in "${TEST_MODELS[@]}"; do
    IFS='|' read -r TEST_MODEL TEST_MODEL_PATH <<< "${MODELS[$model_type]}"
    MODEL_SCENARIO_LIMIT=${MODEL_SCENARIO_COUNT[$model_type]}
    for ((i=1; i<=MODEL_SCENARIO_LIMIT; i++)); do
        IFS='|' read -r input_len output_len concurrency_list num_prompts_multiplier max_num_batched_tokens max_num_seqs max_model_len tensor_parallel_size <<< "${SCENARIOS[$i]}"
        echo "  - ${MAIN_OUTPUT_DIR}/benchmark_comparison_${TEST_MODEL}_input${input_len}_output${output_len}_tp${tensor_parallel_size}_${date_suffix}.log"
    done
done
echo ""
echo "======================================"
echo "所有测试结果已保存到主输出目录：${MAIN_OUTPUT_DIR}"
echo "======================================"