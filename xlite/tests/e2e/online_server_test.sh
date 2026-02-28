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
CONCURRENCY_LIST=${8:-1 16 32 48 64 100}
# NUM_PROMPTS_MULTIPLIER参数（默认值10）
NUM_PROMPTS_MULTIPLIER=${9:-10}
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

# 遍历并发数
for maxconcurrency in ${CONCURRENCY_LIST}
do
    # 构建文件名
    file=input_${INPUT_LEN}_output_${OUTPUT_LEN}_concurrency${maxconcurrency}
    num_prompts=$((maxconcurrency * NUM_PROMPTS_MULTIPLIER))
    
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
    --ignore-eos  | tee ${dir}/${file}.log
done