port=$1
log=$2
model_path=${3:-/mnt/nvme0n1/models/Qwen3-32B}
max_num_batched_tokens=${4:-8192}
max_num_seqs=${5:-200}
max_model_len=${6:-6656}
tensor_parallel_size=${7:-8}

# 基础
export VLLM_USE_V1=1
# 调度优化
export TASK_QUEUE_ENABLE=1
# 通信优化
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
# 计算优化
export OMP_PROC_BIND=false
export VLLM_ASCEND_ENABLE_NZ=2
# 绑核
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl kernel.sched_migration_cost_ns=50000

# 根据 NPU-SMI 版本决定是否禁用 XCCL
# Version >= 25.3 不设置，< 25.3 设置
npu_smi_version=$(npu-smi info 2>/dev/null | grep -oP 'Version:\s*\K[\d.]+' | head -1)
if [[ -n "${npu_smi_version}" ]]; then
    # 提取主版本号、次版本号
    major=$(echo "$npu_smi_version" | cut -d. -f1)
    minor=$(echo "$npu_smi_version" | cut -d. -f2)

    # 正确比较：25.3 为分界线
    if [[ $major -lt 25 ]] || [[ $major -eq 25 && $minor -lt 3 ]]; then
        export XLITE_DISABLE_XCCL=True
        echo "NPU-SMI Version ${npu_smi_version} < 25.3, enabling XLITE_DISABLE_XCCL"
    else
        echo "NPU-SMI Version ${npu_smi_version} >= 25.3, not setting XLITE_DISABLE_XCCL"
    fi
else
    export XLITE_DISABLE_XCCL=True
    echo "Unable to detect NPU-SMI version, enabling XLITE_DISABLE_XCCL by default"
fi

ip=127.0.0.1

expert_parallel_param=""
if [[ "${model_path}" == *"Qwen3-30B-A3B-Instruct-2507"* ]]; then
	expert_parallel_param="--enable-expert-parallel"
fi

python -m vllm.entrypoints.openai.api_server \
	--model ${model_path}  \
	--tensor-parallel-size ${tensor_parallel_size} \
	--gpu-memory-utilization 0.9 \
	--max-num-batched-tokens ${max_num_batched_tokens} \
	--max-num-seqs=${max_num_seqs} \
	--block-size 128 \
	--max-model-len ${max_model_len} \
	--trust-remote-code \
	--served-model-name qwen \
	--no-enable-prefix-caching \
	--additional-config '{"xlite_graph_config": {"enabled": true, "full_mode": true}, "enable_cpu_binding": true}' \
	--compilation-config '{"cudagraph_capture_sizes":[1, 16, 32, 48, 64, 96, 152, 200], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
	--async-scheduling \
	${expert_parallel_param} \
	--host ${ip} \
	--port ${port} > ${log} 2>&1 &