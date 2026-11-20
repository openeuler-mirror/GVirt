#!/bin/bash
set -uo pipefail

# ====================== 参数解析 ======================
# 用法：$0 [model_type] [output_dir]
#   model_type - 见下方说明，默认 moe
#   output_dir - 主输出目录（可选）：
#                * 不传则新建带时间戳的目录；
#                * 传入已存在且非空的目录则进入断点续测模式：扫描其中已有产物，
#                  跳过已完成的"场景×服务×并发度"，仅补跑未完成部分；
#                * 传入不存在的目录则新建。
#                也可用环境变量 MAIN_OUTPUT_DIR_OVERRIDE 指定（优先级低于位置参数）。
MODEL_TYPE=${1:-moe}
RESUME_DIR_ARG=${2:-}

# 定义模型配置
# 格式: <served_name>|<model_path>|<n_layers>
#   n_layers（可省略，默认 0=全量加载）: >0 时用 load_format=dummy 只跑前 n_layers 层，
#   经 online_api_server.sh 的 N_LAYERS/LF 环境变量注入。
declare -A MODELS
MODELS[moe]="Qwen3-30B-A3B-Instruct-2507|/mnt/nvme0n1/models/Qwen3-30B-A3B-Instruct-2507"
MODELS[moe_quant]="Qwen3-30B-A3B-Instruct-2507-w8a8|/mnt/nvme0n1/models/Qwen3-30B-A3B-Instruct-2507-w8a8"
MODELS[moe_quant_a3]="GLM-5.2-w8a8|/mnt/nvme0n1/models/GLM-5.2-w8a8/"
# glm5_2_dummy: GLM-5.2-w8a8 的 dummy 快速冒烟变体，跑在 910B3(A2) 上、复用 DEFAULT 6 场景（tp=8），
# dummy 加载只跑前 10 层（N_LAYERS=10）；served_name 加 -dummy 后缀避免与全量混淆
MODELS[glm5_2_dummy]="GLM-5.2-w8a8-dummy|/mnt/nvme0n1/models/GLM-5.2-w8a8/|10"

# 确定要测试的模型列表
declare -a TEST_MODELS
if [ "${MODEL_TYPE}" = "all" ]; then
    TEST_MODELS=("moe" "moe_quant" "moe_quant_a3" "glm5_2_dummy")
elif [ "${MODEL_TYPE}" = "moe_mix" ]; then
    TEST_MODELS=("moe" "moe_quant")
elif [ "${MODEL_TYPE}" = "quant" ]; then
    TEST_MODELS=("moe_quant" "moe_quant_a3")
elif [ "${MODEL_TYPE}" = "moe" ] || [ "${MODEL_TYPE}" = "moe_quant" ] || [ "${MODEL_TYPE}" = "moe_quant_a3" ] || [ "${MODEL_TYPE}" = "glm5_2_dummy" ]; then
    TEST_MODELS=("${MODEL_TYPE}")
else
    echo "错误：无效的模型类型 '${MODEL_TYPE}'，请使用 'moe'、'moe_quant'、'moe_quant_a3'、'glm5_2_dummy'、'moe_mix'、'quant' 或 'all'"
    echo "用法：$0 [model_type] [output_dir]"
    echo "  model_type（可选，默认 moe）："
    echo "  moe          - 测试 Qwen3-30B-A3B-Instruct-2507 模型（默认）"
    echo "  moe_quant    - 测试 Qwen3-30B-A3B-Instruct-2507-w8a8 模型"
    echo "  moe_quant_a3 - 测试 GLM-5.2-w8a8 模型（全量加载）"
    echo "  glm5_2_dummy - 测试 GLM-5.2-w8a8 模型（dummy 加载，只跑前 10 层）"
    echo "  moe_mix      - 测试 moe + moe_quant（量化前后对比）"
    echo "  quant        - 测试所有量化模型（moe_quant + moe_quant_a3，不含 glm5_2_dummy）"
    echo "  all          - 测试所有模型（含 glm5_2_dummy）"
    echo "  output_dir（可选）：主输出目录；传入已存在且非空的目录即进入断点续测模式"
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

# 启动服务（参数：服务启动脚本路径 日志文件名名 模型路径 max_num_batched_tokens max_num_seqs max_model_len tensor_parallel_size mode）
start_server() {
    local server_script=$1
    local log_file=$2
    local model_path=$3
    local max_num_batched_tokens=$4
    local max_num_seqs=$5
    local max_model_len=$6
    local tensor_parallel_size=$7
    local mode=$8
    # 清理旧日志（避免残留关键字影响匹配）
    rm -f "${log_file}"
    echo -e "\n======================================"
    echo "启动服务：${server_script} 8080 ${log_file} ${model_path} ${max_num_batched_tokens} ${max_num_seqs} ${max_model_len} ${tensor_parallel_size} ${mode}"
    echo "日志文件：${log_file}（独立日志）"
    echo "======================================"
    # 后台启动服务并记录PID
    bash "${server_script}" 8080 "${log_file}" "${model_path}" "${max_num_batched_tokens}" "${max_num_seqs}" "${max_model_len}" "${tensor_parallel_size}" "${mode}" &
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

# ====================== 断点续测支持 ======================
# 判断单个并发度压测日志是否已完成：
# vllm bench 正常结束时会输出 "Request throughput (req/s):" 指标行，
# 掐断/异常退出时该关键字缺失，据此可靠区分"已完成"与"未完成"。
is_bench_log_complete() {
    local log_file=$1
    [ -f "$log_file" ] && grep -q "Request throughput (req/s)" "$log_file" 2>/dev/null
}

# 过滤出某服务下尚未完成的并发度（同步过滤 multiplier，保持一一对应）。
# 输入：concurrency_list multiplier_list result_dir input_len output_len
# 输出（全局变量）：FILTERED_CONCURRENCY、FILTERED_MULTIPLIER（空格分隔，已完成的被剔除）。
# 返回码：过滤后仍需补跑的并发度数量（0 表示该服务已全部完成）。
filter_incomplete_concurrencies() {
    local concurrency_list=$1
    local multiplier_list=$2
    local result_dir=$3
    local input_len=$4
    local output_len=$5
    FILTERED_CONCURRENCY=""
    FILTERED_MULTIPLIER=""
    local c_arr=($concurrency_list)
    local m_arr=($multiplier_list)
    local need_count=0
    local i=0
    for c in "${c_arr[@]}"; do
        local m=${m_arr[$i]:-}
        local log_file="${result_dir}/input_${input_len}_output_${output_len}_concurrency${c}.log"
        if is_bench_log_complete "$log_file"; then
            echo "    [续测] 复用已完成: concurrency${c}"
        else
            FILTERED_CONCURRENCY="${FILTERED_CONCURRENCY}${FILTERED_CONCURRENCY:+ }${c}"
            FILTERED_MULTIPLIER="${FILTERED_MULTIPLIER}${FILTERED_MULTIPLIER:+ }${m}"
            need_count=$((need_count+1))
        fi
        i=$((i+1))
    done
    return ${need_count}
}

# 判断某服务的所有并发度压测是否都已完成（用于决定是否跳过整个服务）。
# 参数：result_dir concurrency_list input_len output_len
is_service_bench_complete() {
    local result_dir=$1
    local concurrency_list=$2
    local input_len=$3
    local output_len=$4
    [ -d "$result_dir" ] || return 1
    for c in $concurrency_list; do
        local log_file="${result_dir}/input_${input_len}_output_${output_len}_concurrency${c}.log"
        if ! is_bench_log_complete "$log_file"; then
            return 1
        fi
    done
    return 0
}

# 判断某服务的所有场景所有并发度是否都已完成（服务外提复用结构下的服务级续测判据）。
# 参数：service_tag model_name main_output_dir
# 依赖全局：SCENARIOS（当前模型的场景表）、MODEL_SCENARIO_LIMIT（场景数）。
# 返回码：0=该服务全部场景全部并发已完成, 1=有未完成项。
is_service_all_scenarios_complete() {
    local service_tag=$1
    local model_name=$2
    local main_output_dir=$3
    for ((i=1; i<=MODEL_SCENARIO_LIMIT; i++)); do
        IFS='|' read -r input_len output_len concurrency_list _mult <<< "${SCENARIOS[$i]}"
        local result_dir="${main_output_dir}/result_input_${input_len}_output_${output_len}_${service_tag}_${model_name}"
        if ! is_service_bench_complete "${result_dir}" "${concurrency_list}" "${input_len}" "${output_len}"; then
            return 1
        fi
    done
    return 0
}

# 执行单个服务的全部场景测试（服务外提复用结构：服务起一次，跑完所有场景再 kill）
# 参数说明：
#   $1 - service_mode: online_api_server.sh 的 mode 参数 (xlite_decode_only | xlite_full_mode | aclgraph)
#   $2 - service_tag: result_input 目录命名用的服务标签 (xlite_decode_only | xlite_full | aclgraph)
#   $3 - model_path: 模型路径
#   $4 - max_num_batched_tokens: 最大批处理token数（模型级统一）
#   $5 - max_num_seqs: 最大序列数（模型级统一）
#   $6 - max_model_len: 最大模型长度（模型级统一）
#   $7 - tensor_parallel_size: 张量并行大小（模型级统一）
#   $8 - main_output_dir: 主输出目录（断点续测产物所在目录）
# 依赖全局：SCENARIOS（当前模型的场景表，4字段 in|out|并发|mult）、MODEL_SCENARIO_LIMIT（场景数）。
run_service_all_scenarios() {
    local service_mode=$1
    local service_tag=$2
    local model_path=$3
    local max_num_batched_tokens=$4
    local max_num_seqs=$5
    local max_model_len=$6
    local tensor_parallel_size=$7
    local main_output_dir=$8

    local model_name=$(basename "${model_path}")

    # service_tag -> 中文服务名 (便于日志直读)
    local service_cn="未知服务"
    case "${service_tag}" in
        xlite_decode_only) service_cn="Xlite Decode Only" ;;
        xlite_full)        service_cn="Xlite Full" ;;
        aclgraph)          service_cn="Aclgraph" ;;
    esac
    # load_format 提示: dummy 分层加载时显示跑的层数, 否则全量
    local load_info="全量加载"
    if [ "${LF:-auto}" = "dummy" ] && [ -n "${N_LAYERS:-}" ] && [ "${N_LAYERS}" -gt 0 ]; then
        load_info="dummy 加载, 只跑前 ${N_LAYERS} 层"
    fi

    echo "======================================"
    echo "服务：${service_cn}（tag=${service_tag}, mode=${service_mode}）"
    echo "模型：${model_name}（${load_info}）"
    echo "max_num_batched_tokens：${max_num_batched_tokens}"
    echo "max_num_seqs：${max_num_seqs}"
    echo "max_model_len：${max_model_len}"
    echo "tensor_parallel_size：${tensor_parallel_size}"
    echo "场景数：${MODEL_SCENARIO_LIMIT}"
    echo "======================================"

    # 服务级断点续测：该服务所有场景所有并发都已完成则跳过起服务
    if is_service_all_scenarios_complete "${service_tag}" "${model_name}" "${main_output_dir}"; then
        echo "[续测] ${service_cn} 所有场景均已完成，跳过服务启动与压测"
        return 0
    fi

    local service_log="${main_output_dir}/server_${model_name}_${service_tag}.log"
    start_server "online_api_server.sh" "${service_log}" "${model_path}" "${max_num_batched_tokens}" "${max_num_seqs}" "${max_model_len}" "${tensor_parallel_size}" "${service_mode}"
    if ! wait_server_ready "${service_log}"; then
        echo "警告：${service_cn} 服务启动失败，跳过此服务"
        cleanup_all_processes
        return 1
    fi
    echo ">>> ${service_cn} 服务已就绪，开始压测 ${MODEL_SCENARIO_LIMIT} 个场景"

    # 场景在内：逐场景压测（仅补跑未完成的并发度）
    for ((i=1; i<=MODEL_SCENARIO_LIMIT; i++)); do
        IFS='|' read -r input_len output_len concurrency_list multiplier_list <<< "${SCENARIOS[$i]}"
        local result_dir="${main_output_dir}/result_input_${input_len}_output_${output_len}_${service_tag}_${model_name}"
        echo -e "\n>>> [场景${i}/${MODEL_SCENARIO_LIMIT}] ${service_cn}：input=${input_len}, output=${output_len}, 并发=[${concurrency_list}], multiplier=[${multiplier_list}]"
        filter_incomplete_concurrencies "${concurrency_list}" "${multiplier_list}" "${result_dir}" "${input_len}" "${output_len}"
        local need=$?
        if [ ${need} -eq 0 ]; then
            echo "    [续测] 全部并发已完成，跳过压测"
            continue
        fi
        echo "    待补跑并发度(${need})：${FILTERED_CONCURRENCY}"
        run_bench_scripts_sequential "online_server_test.sh" "${service_tag}_${model_name}" \
            "${input_len}" "${output_len}" "qwen" "${model_path}" "127.0.0.1" "8080" \
            "${FILTERED_CONCURRENCY}" "${FILTERED_MULTIPLIER}" "${main_output_dir}"
    done

    wait_10s "${service_cn} 压测完成，准备清理进程"
    cleanup_all_processes
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
    local device_type=${7:-910B3(A2)}

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

    # 断点续测：若对比报告已存在且非空，则跳过重新生成（避免覆盖已有报告）
    if [ -s "${output_file}" ]; then
        echo "[续测] 对比报告已存在，跳过数据处理：${output_file}"
        return 0
    fi

    echo "调用 process_data.py 处理数据..."
    echo "  baseline-aclgraph: ${aclgraph_dir}"
    echo "  xlite-full: ${xlite_full_dir}"
    echo "  xlite-decode-only: ${xlite_decode_only_dir}"
    echo "  输出文件: ${output_file}"
    echo "  模型名称: ${model_name}"
    echo "  测试环境型号: ${device_type}"

    # 调用 process_data.py
    python process_data.py \
        "${aclgraph_dir}" \
        "${xlite_full_dir}" \
        "${xlite_decode_only_dir}" \
        -o "${output_file}" \
        -m "${model_name}" \
        -d "${device_type}"

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

# 创建主输出文件夹：
# 优先级：位置参数2 > 环境变量 MAIN_OUTPUT_DIR_OVERRIDE > 新建带时间戳目录。
# 若指定目录已存在且非空，则进入断点续测模式（复用已有产物，仅补跑未完成项）。
if [[ -n "${RESUME_DIR_ARG}" ]]; then
    MAIN_OUTPUT_DIR="${RESUME_DIR_ARG}"
elif [[ -n "${MAIN_OUTPUT_DIR_OVERRIDE:-}" ]]; then
    MAIN_OUTPUT_DIR="${MAIN_OUTPUT_DIR_OVERRIDE}"
else
    MAIN_OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${MAIN_OUTPUT_DIR}"

# 断点续测模式检测：目录已存在且非空即视为续测
RESUME_MODE=0
if [ -d "${MAIN_OUTPUT_DIR}" ] && [ -n "$(ls -A "${MAIN_OUTPUT_DIR}" 2>/dev/null)" ]; then
    RESUME_MODE=1
fi

echo -e "\n======================================"
echo "主输出目录：${MAIN_OUTPUT_DIR}"
if [ ${RESUME_MODE} -eq 1 ]; then
    echo "运行模式：断点续测（跳过已完成的 场景×服务×并发度，仅补跑未完成项）"
else
    echo "运行模式：全新测试"
fi
echo "======================================"

# 记录总测试开始时间
total_start_time=$(date +%s)
total_start_time_formatted=$(date "+%Y-%m-%d %H:%M:%S")

# 定义场景配置（按模型维度选择，all/quant 等组合模式下 moe_quant_a3 也用 A3 专属参数）
# 服务启动参数（模型级统一，覆盖该模型所有场景；服务外提后一个服务起一次跑全部场景，
# 故 max_model_len/max_num_seqs 取各场景最大值以覆盖全部）。
# vLLM 用分页 KV cache，max_model_len/max_num_seqs 仅是上限不预分配，统一为大值不影响实际 KV 占用。
# DEFAULT 模型（moe/moe_quant/glm5_2_dummy，910B3/A2）：mnbt=8192, mns=96, mml=89600, tp=8
DEFAULT_MNBT=8192
DEFAULT_MNS=96
DEFAULT_MML=89600
DEFAULT_TP=8
# A3 专属（moe_quant_a3，Ascend910(A3)）：mnbt=8192, mns=32, mml=22768, tp=16
A3_MNBT=8192
A3_MNS=32
A3_MML=22768
A3_TP=16

# 场景配置：仅 in|out|并发|mult 四字段（服务启动参数已提取为上方模型级常量）。
# 格式：input_len|output_len|concurrency_list|num_prompts_multiplier_list
# num_prompts_multiplier_list 与 concurrency_list 一一对应，每个并发数对应各自的 multiplier
declare -A SCENARIOS_DEFAULT
SCENARIOS_DEFAULT[1]="512|512|1 16 32 48 64|10 10 10 10 10"
SCENARIOS_DEFAULT[2]="3584|1536|1 16 32|10 10 10"
SCENARIOS_DEFAULT[3]="8192|1024|1 16 32|10 10 10"
SCENARIOS_DEFAULT[4]="16384|1536|1 16|10 10"
SCENARIOS_DEFAULT[5]="32768|3072|1 8|10 10"
SCENARIOS_DEFAULT[6]="65536|6144|1 4|10 10"

# moe_quant_a3（GLM-5.2-w8a8）运行于 Ascend910(A3)，用 A3 专属服务启动参数（见上方 A3_* 常量）
declare -A SCENARIOS_A3
SCENARIOS_A3[1]="3584|1536|1 16 32|8 4 4"
SCENARIOS_A3[2]="8192|1024|1 16 32|8 4 4"
SCENARIOS_A3[3]="16384|1536|1 16 32|8 4 4"

# SCENARIOS 在 for model_type 循环内按当前模型选择，使 all/quant 模式下 moe_quant_a3 也使用 A3 专属参数
declare -A SCENARIOS

# 定义每个模型的场景数量
declare -A MODEL_SCENARIO_COUNT
MODEL_SCENARIO_COUNT[moe]=6
MODEL_SCENARIO_COUNT[moe_quant]=6
MODEL_SCENARIO_COUNT[moe_quant_a3]=3
MODEL_SCENARIO_COUNT[glm5_2_dummy]=6

# 用于存储每个模型的场景测试结果
declare -A MODEL_TEST_RESULTS

# 遍历所有要测试的模型
for model_type in "${TEST_MODELS[@]}"; do
    IFS='|' read -r TEST_MODEL TEST_MODEL_PATH TEST_N_LAYERS <<< "${MODELS[$model_type]}"
    TEST_N_LAYERS=${TEST_N_LAYERS:-0}

    # 获取当前模型需要测试的场景数量
    MODEL_SCENARIO_LIMIT=${MODEL_SCENARIO_COUNT[$model_type]}

    SCENARIOS=()
    # 模型级服务启动参数与 device_type：moe_quant_a3 用 A3 专属, 其余用 DEFAULT
    device_type="910B3(A2)"
    if [ "${model_type}" = "moe_quant_a3" ]; then
        for k in "${!SCENARIOS_A3[@]}"; do SCENARIOS[$k]="${SCENARIOS_A3[$k]}"; done
        MNBT=${A3_MNBT}; MNS=${A3_MNS}; MML=${A3_MML}; TP=${A3_TP}
        device_type="Ascend910(A3)"
    else
        for k in "${!SCENARIOS_DEFAULT[@]}"; do SCENARIOS[$k]="${SCENARIOS_DEFAULT[$k]}"; done
        MNBT=${DEFAULT_MNBT}; MNS=${DEFAULT_MNS}; MML=${DEFAULT_MML}; TP=${DEFAULT_TP}
    fi

    # dummy 分层加载: 经 online_api_server.sh 的 N_LAYERS/LF 环境变量注入。
    # 在本模型循环内 export, 循环末尾恢复, 避免污染后续模型。
    SAVED_N_LAYERS="${N_LAYERS:-}"
    SAVED_LF="${LF:-}"
    if [ "${TEST_N_LAYERS}" -gt 0 ]; then
        export N_LAYERS="${TEST_N_LAYERS}"
        export LF="dummy"
        echo "[dummy] ${TEST_MODEL}: 只跑前 ${TEST_N_LAYERS} 层"
    else
        unset N_LAYERS LF
    fi

    echo "======================================"
    echo "开始测试模型：${TEST_MODEL}"
    echo "测试场景数量：${MODEL_SCENARIO_LIMIT}"
    echo "服务启动参数：mnbt=${MNBT} mns=${MNS} mml=${MML} tp=${TP}"
    echo "======================================"

    # 初始化当前模型的测试结果
    MODEL_TEST_RESULTS[$TEST_MODEL]=""

    # 服务外提：3 服务各起一次, 每个服务跑完全部场景再 kill。
    # 服务日志命名 server_<model>_<service>.log（每服务一个, 不再按场景分）。
    run_service_all_scenarios "xlite_decode_only" "xlite_decode_only" "${TEST_MODEL_PATH}" \
        "${MNBT}" "${MNS}" "${MML}" "${TP}" "${MAIN_OUTPUT_DIR}"
    wait_10s "xlite_decode_only 完成，准备 xlite_full_mode"

    run_service_all_scenarios "xlite_full_mode" "xlite_full" "${TEST_MODEL_PATH}" \
        "${MNBT}" "${MNS}" "${MML}" "${TP}" "${MAIN_OUTPUT_DIR}"
    wait_10s "xlite_full_mode 完成，准备 aclgraph"

    run_service_all_scenarios "aclgraph" "aclgraph" "${TEST_MODEL_PATH}" \
        "${MNBT}" "${MNS}" "${MML}" "${TP}" "${MAIN_OUTPUT_DIR}"

    # 最后统一对比：3 服务的 result_input_* 目录此时都已填满, 逐场景生成对比报告。
    # 此循环位于脚本顶层而非函数内，不可使用 local（set -u 下会因 unbound variable 退出）
    for ((i=1; i<=MODEL_SCENARIO_LIMIT; i++)); do
        IFS='|' read -r input_len output_len concurrency_list num_prompts_multiplier_list <<< "${SCENARIOS[$i]}"
        if ! process_scenario_data "${TEST_MODEL}_场景${i}：输入${input_len}token,输出${output_len}token" \
            "${input_len}" "${output_len}" "${TEST_MODEL}" "${TP}" "${MAIN_OUTPUT_DIR}" "${device_type}"; then
            echo -e "\n警告：场景${i}数据处理失败，继续下一个场景"
            MODEL_TEST_RESULTS[$TEST_MODEL]="${MODEL_TEST_RESULTS[$TEST_MODEL]}场景${i}：输入${input_len}token,输出${output_len}token,并发数：${concurrency_list},tensor_parallel_size：${TP} - 数据处理失败\n"
        else
            MODEL_TEST_RESULTS[$TEST_MODEL]="${MODEL_TEST_RESULTS[$TEST_MODEL]}场景${i}：输入${input_len}token,输出${output_len}token,并发数：${concurrency_list},tensor_parallel_size：${TP} - 数据对比成功\n"
        fi
    done

    echo -e "\n======================================"
    echo "模型 ${TEST_MODEL} 的测试场景全部执行完成！"
    echo "======================================"

    # 恢复 dummy 分层加载相关环境变量, 避免污染下一个模型
    if [ -n "${SAVED_N_LAYERS}" ]; then
        export N_LAYERS="${SAVED_N_LAYERS}"
    else
        unset N_LAYERS
    fi
    if [ -n "${SAVED_LF}" ]; then
        export LF="${SAVED_LF}"
    else
        unset LF
    fi
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
    # 重新选取该模型的场景表与 tp（汇总循环跨多模型，SCENARIOS 是上一轮循环的残留）
    declare -A _SUMMARY_SCENARIOS=()
    if [ "${model_type}" = "moe_quant_a3" ]; then
        for k in "${!SCENARIOS_A3[@]}"; do _SUMMARY_SCENARIOS[$k]="${SCENARIOS_A3[$k]}"; done
        _SUMMARY_TP=${A3_TP}
    else
        for k in "${!SCENARIOS_DEFAULT[@]}"; do _SUMMARY_SCENARIOS[$k]="${SCENARIOS_DEFAULT[$k]}"; done
        _SUMMARY_TP=${DEFAULT_TP}
    fi
    for ((i=1; i<=MODEL_SCENARIO_LIMIT; i++)); do
        IFS='|' read -r input_len output_len concurrency_list num_prompts_multiplier_list <<< "${_SUMMARY_SCENARIOS[$i]}"
        echo "  - ${MAIN_OUTPUT_DIR}/benchmark_comparison_${TEST_MODEL}_input${input_len}_output${output_len}_tp${_SUMMARY_TP}_${date_suffix}.log"
    done
done
echo ""
echo "======================================"
echo "所有测试结果已保存到主输出目录：${MAIN_OUTPUT_DIR}"
echo "======================================"
