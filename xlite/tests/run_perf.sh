#!/bin/bash
export FORWARD_BACKEND=xlite
models_base_path=${1:-/mnt/nvme0n1/models}
perf_log_file=${PERF_LOG:-tests/test_output/xlite_perf.log}

mkdir -p tests/test_output

# 运行bench测试并解析结果的辅助函数
function run_bench_test()
{
    local model=$1
    local bs=$2
    local n1=$3      # prompt长度
    local n2=$4      # 输出token数
    local dtype=$5
    local model_func=$6
    local tp_str=$7
    local test_type=$8  # "decode" 或 "prefill"

    local log_file=tests/test_output/perf_${model}_${test_type}_bs${bs}.log

    RUN_MODE=bench BENCH_IT=1 BENCH_BS=$bs BENCH_N1=$n1 BENCH_N2=$n2 MODEL=$model_func \
        bash tests/run.sh 2>&1 | tee $log_file

    # 解析输出: "Iter X/Y: prefilled N | generated M tokens in Z ms. avg: A | B tokens/s @ C ms step bs: D"
    perf_line=$(grep "tokens/s @ " $log_file | tail -n 1)
    decode_tps=$(echo $perf_line | awk '{print $15}')   # tokens/s
    step_latency=$(echo $perf_line | awk '{print $18}') # ms step latency

    if [ "$test_type" = "decode" ]; then
        echo "| ${model} | ${dtype} | ${tp_str} | ${n1}+${n2} | ${bs} | ${step_latency} | ${decode_tps} |" >> ${perf_log_file}
    else
        echo "| ${model} | ${dtype} | ${tp_str} | ${n1}+${n2} | 1 | ${step_latency} |" >> ${perf_log_file}
    fi
    rm $log_file
}

# 默认执行perf测试
echo "| 模型 | 类型 | 并行策略 | 输入输出长度 | batch size | 平均decode时延 | 平均decode吞吐 |" > ${perf_log_file}
echo "|--|--|--|--|--|--|--|" >> ${perf_log_file}

# Decode测试 - llama_7B (100+1024)
for bs in 1; do
    run_bench_test llama_7B $bs 100 1024 float16 run_llama_7B TP1 decode
done

# Decode测试 - llama_13B
for bs in 1 16; do
    run_bench_test llama_13B $bs 100 1024 float16 run_llama_13B TP2 decode
done

# Decode测试 - qwen3_32B
for dtype in float16 bfloat16; do
    for bs in 1 16; do
        run_bench_test qwen3_32B $bs 100 1024 $dtype run_qwen3_32B TP8 decode
    done
done

# Decode测试 - qwen3_moe_30B
for bs in 1 16; do
    run_bench_test qwen3_moe_30B $bs 100 1024 bfloat16 run_qwen3_moe_30B TP8moeEP8 decode
done

# 16-NPU模型
npu_count=$(python -c "import torch; print(torch.npu.device_count())")
if [ $npu_count -ge 16 ]; then
    for bs in 1 16 64; do
        run_bench_test glm4_moe $bs 100 1024 bfloat16 run_glm4_moe TP16moeEP16 decode
    done

    for bs in 1 16; do
        run_bench_test deepseek_v3 $bs 100 1024 bfloat16 run_deepseek_v3 TP16moeEP16 decode
    done

    for bs in 1 16; do
        run_bench_test deepseek_v3_w8a8 $bs 100 1024 bfloat16 run_deepseek_v3_w8a8 TP16moeEP16 decode
    done

    for bs in 1 16; do
        run_bench_test glm5_w8a8 $bs 100 1024 bfloat16 run_glm5_w8a8 TP16moeEP16 decode
    done

    for bs in 1 16; do
        run_bench_test minimax_m2 $bs 100 1024 bfloat16 run_minimax_m2 TP16moeEP16 decode
    done
fi

# Prefill测试 (3456+1)
echo "" >> ${perf_log_file}
echo "| 模型 | 类型 | 并行策略 | 输入输出长度 | batch size | prefill时延 |" >> ${perf_log_file}
echo "|--|--|--|--|--|--|" >> ${perf_log_file}

run_bench_test llama_7B 1 3456 1 float16 run_llama_7B TP1 prefill
run_bench_test llama_13B 1 3456 1 float16 run_llama_13B TP2 prefill
run_bench_test qwen3_32B 1 3456 1 float16 run_qwen3_32B TP8 prefill
run_bench_test qwen3_32B 1 3456 1 bfloat16 run_qwen3_32B TP8 prefill
run_bench_test qwen3_moe_30B 1 3456 1 bfloat16 run_qwen3_moe_30B TP8moeEP8 prefill

if [ $npu_count -ge 16 ]; then
    run_bench_test glm4_moe 1 3456 1 bfloat16 run_glm4_moe TP16moeEP16 prefill
    run_bench_test deepseek_v3 1 3456 1 bfloat16 run_deepseek_v3 TP16moeEP16 prefill
    run_bench_test deepseek_v3_w8a8 1 3456 1 bfloat16 run_deepseek_v3_w8a8 TP16moeEP16 prefill
    run_bench_test glm5_w8a8 1 3456 1 bfloat16 run_glm5_w8a8 TP16moeEP16 prefill
    run_bench_test minimax_m2 1 3456 1 bfloat16 run_minimax_m2 TP16moeEP16 prefill
fi