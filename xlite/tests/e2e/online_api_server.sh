port=$1
log=$2
model_path=${3:-/mnt/nvme0n1/models/Qwen3-32B}
max_num_batched_tokens=${4:-8192}
max_num_seqs=${5:-200}
max_model_len=${6:-6656}
tensor_parallel_size=${7:-8}
# 模式选择: aclgraph, xlite_decode_only, xlite_full_mode
mode=${8:-aclgraph}

# 基础
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
# 调度优化
export TASK_QUEUE_ENABLE=1
# 通信优化
export HCCL_BUFFSIZE=512
export HCCL_OP_EXPANSION_MODE="AIV"
# 计算优化
export OMP_PROC_BIND=false
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

# Profiling 参数 (通过环境变量 XLITE_ENABLE_PROFILING=1 开启)
profiler_param='{}'
if [[ "${XLITE_ENABLE_PROFILING}" == "1" ]]; then
    profiler_dir="./profile_$(basename "${model_path}")_${mode}_$(date +%m%d_%H%M%S)"
    profiler_param=$(jq -n --arg dir "$profiler_dir" \
        '{"profiler": "torch", "torch_profiler_dir": $dir, "torch_profiler_with_stack": false}')
    echo "Profiling enabled, output to ${profiler_dir}"
fi

expert_parallel_param=""
config_file="${model_path}/config.json"
if [[ -f "${config_file}" ]]; then
    if grep -q "num_experts\|moe_intermediate_size" "${config_file}" 2>/dev/null; then
        expert_parallel_param="--enable-expert-parallel"
        echo "Detected MoE model, enabling expert parallel"
    fi
fi

quantization_param=""
if [[ -f "${model_path}/quant_model_description.json" ]]; then
	quantization_param="--quantization ascend"
    echo "Detected quanted model, enabling quantization"
fi

# 根据 mode 选择参数
case "${mode}" in
	aclgraph)
		additional_config_param='{"weight_nz_mode": 2, "enable_balance_scheduling": true, "enable_flashcomm1": true, "multistream_overlap_shared_expert": true}'
		;;
	xlite_decode_only)
		additional_config_param='{"xlite_graph_config": {"enabled": true}, "weight_nz_mode": 2, "enable_balance_scheduling": true, "enable_flashcomm1": true, "multistream_overlap_shared_expert": true}'
		;;
	xlite_full_mode)
		additional_config_param='{"xlite_graph_config": {"enabled": true, "full_mode": true}, "weight_nz_mode": 2, "enable_balance_scheduling": true}'
		;;
	*)
		echo "Unknown mode: ${mode}, exit"
		exit 1
		;;
esac

# load_format / n_layers: 仅跑前 n_layers 层时用 dummy 加载（默认 auto、跑全部层）
# N_LAYERS>0 时强制 dummy 并通过 hf-overrides 覆盖 num_hidden_layers
n_layers="${N_LAYERS:-0}"
load_format="${LF:-auto}"
extra_args=()

# 校验 N_LAYERS 为非负整数: 非法值 (如 abc/1.5/-1/带空格) 会令下面的 [ -gt 0 ]
# 向 stderr 报 "integer expression expected" 并静默回退全量加载, 可能使 dummy 冒烟
# 在无意间以全量权重起服务 (加载时长放大一个量级). 非法值显式报错终止.
case "${n_layers}" in
	''|*[!0-9]*)
		echo "错误: N_LAYERS 非法值 '${N_LAYERS}' (应为非负整数), 终止启动" >&2
		exit 1
		;;
esac
if [ "${n_layers}" -gt 0 ]; then
	load_format="dummy"
	extra_args+=(--hf-overrides '{"num_hidden_layers": '"${n_layers}"'}')
	echo "Using ${n_layers} layers"
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
	--additional-config "${additional_config_param}" \
	--compilation-config '{"cudagraph_capture_sizes":[1, 4, 8, 12, 16, 24, 32, 48, 64, 96], "cudagraph_mode": "FULL_DECODE_ONLY"}' \
	--async-scheduling \
	${expert_parallel_param} \
	${quantization_param} \
	--profiler-config "${profiler_param}" \
	--host ${ip} \
	--port ${port} \
	--load-format "${load_format}" \
	"${extra_args[@]}" > ${log} 2>&1 &
