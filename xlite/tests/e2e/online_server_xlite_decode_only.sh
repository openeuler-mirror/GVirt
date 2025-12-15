port=$1
log=$2
export TASK_QUEUE_ENABLE=1
export VLLM_USE_V1=1
export OMP_PROC_BIND=false
export HCCL_OP_EXPANSION_MODE="AIV"
export VLLM_ASCEND_ENABLE_TOPK_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_FLASHCOMM=1
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE=1
export VLLM_ASCEND_ENABLE_PREFETCH_MLP=1
ip=127.0.0.1
python -m vllm.entrypoints.openai.api_server \
	--model /mnt/nvme0n1/models/Qwen3-32B  \
	--tensor-parallel-size 8 \
	--gpu-memory-utilization 0.9 \
	--max-num-batched-tokens 8192 \
	--max-num-seqs=200 \
	--block-size 128 \
	--max-model-len 6656 \
	--trust-remote-code \
	--disable-log-requests \
	--served-model-name qwen \
	--no-enable-prefix-caching \
	--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
	--additional-config='{"xlite_graph_config": {"enabled": true}}' \
	--async-scheduling \
	--host ${ip} \
	--port ${port} > ${log} 2>&1 &