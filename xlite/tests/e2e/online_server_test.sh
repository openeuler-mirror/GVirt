#!/bin/bash

# ===================== 可配置参数=====================
# type参数默认值
TYPE=${1:-default}
# 输入/输出长度
INPUT_LEN=${2:-512}
OUTPUT_LEN=${3:-512}
# 模型和tokenizer路径
MODEL_NAME=${4:-qwen}
TOKENIZER_PATH=${5:-/mnt/nvme0n1/models/Qwen3-32B}
# 服务器配置
HOST=${6:-127.0.0.1}
PORT=${7:-8080}
# 并发数列表
CONCURRENCY_LIST=${8:-"1 16 32 48 64 100"}
# NUM_PROMPTS_MULTIPLIER列表，与CONCURRENCY_LIST一一对应
NUM_PROMPTS_MULTIPLIER=${9:-"10 10 10 10 10 10"}
# 主输出目录（可选参数）
MAIN_OUTPUT_DIR=${10:-}
# 其他配置
RANDOM_RANGE_RATIO=0.2

# 构建输出目录
if [ -n "${MAIN_OUTPUT_DIR}" ]; then
    dir="${MAIN_OUTPUT_DIR}/result_input_${INPUT_LEN}_output_${OUTPUT_LEN}_${TYPE}"
else
    dir=result_input_${INPUT_LEN}_output_${OUTPUT_LEN}_${TYPE}
fi
mkdir -p ${dir}

concurrency_arr=(${CONCURRENCY_LIST})
multiplier_arr=(${NUM_PROMPTS_MULTIPLIER})
# 若两者长度不一致，报错退出
if [[ ${#concurrency_arr[@]} -ne ${#multiplier_arr[@]} ]]; then
    echo "ERROR: CONCURRENCY_LIST(${#concurrency_arr[@]}) and NUM_PROMPTS_MULTIPLIER(${#multiplier_arr[@]}) don't match"
    exit 1
fi

for i in "${!concurrency_arr[@]}"
do
    maxconcurrency=${concurrency_arr[$i]}
    multiplier=${multiplier_arr[$i]}
    num_prompts=$((maxconcurrency * multiplier))

    # 构建文件名
    file=input_${INPUT_LEN}_output_${OUTPUT_LEN}_concurrency${maxconcurrency}
    log_file="${dir}/${file}.log"

    echo "`date +'%m-%d %H:%M:%S'`: vllm bench start: maxconcurrency: ${maxconcurrency} "

    # 执行vllm bench命令
    vllm bench serve \
    --max-concurrency ${maxconcurrency} \
    --num-prompts ${num_prompts} \
    --host ${HOST} \
    --port ${PORT} \
    --model ${MODEL_NAME} \
    --dataset-name random \
    --backend openai-chat \
    --random-input-len ${INPUT_LEN} \
    --random-output-len ${OUTPUT_LEN} \
    --random-range-ratio ${RANDOM_RANGE_RATIO} \
    --tokenizer ${TOKENIZER_PATH} \
    --endpoint /v1/chat/completions \
    --ignore-eos | tee ${log_file} &
    bench_pid=$!

    # Profiling 捕获进程(通过环境变量 XLITE_ENABLE_PROFILING=1 开启)
    if [[ "${XLITE_ENABLE_PROFILING}" == "1" ]]; then
        (
            # 监控日志文件，等待 "Starting main benchmark run" 出现，同时检查 bench 进程存活
            while ! grep -q "Starting main benchmark run" "${log_file}" 2>/dev/null; do
                if ! kill -0 ${bench_pid} 2>/dev/null; then
                    echo "`date +'%m-%d %H:%M:%S'`: Bench process exited, aborting profiling."
                    exit 0
                fi
                sleep 1
            done

            echo "`date +'%m-%d %H:%M:%S'`: Detected 'Starting main benchmark run', waiting 30s before profiling..."
            sleep 30
            echo "`date +'%m-%d %H:%M:%S'`: Start profiling..."
            curl -s -X POST http://${HOST}:${PORT}/start_profile > /dev/null
            sleep 1
            curl -s -X POST http://${HOST}:${PORT}/stop_profile > /dev/null
            echo "`date +'%m-%d %H:%M:%S'`: End profiling..."
            # 数据转换方法，在 profiling 文件夹，如 vllm_profile 中执行：
            # python -c "from torch_npu.profiler.profiler import analyse; import glob; [analyse(p) for p in glob.glob('*_ascend_pt')]"
        ) &
        profile_pid=$!
    fi

    # 等待 vllm bench 完成
    wait ${bench_pid}

    # 等待 profiling 进程完成
    if [[ -n "${profile_pid}" ]]; then
        wait ${profile_pid} 2>/dev/null
    fi

    echo "`date +'%m-%d %H:%M:%S'`: vllm bench end: maxconcurrency: ${maxconcurrency} "
done