export FORWARD_BACKEND=${FORWARD_BACKEND:-xlite}
cd "$(dirname "$0")/.."
models_base_path=${1:-/mnt/nvme0n1/models}
test_config_path=tests/test_config.json
test_input_path=tests/test_input_default.json

if [ -n "${XLITE_NPUS:-}" ]; then
    export ASCEND_RT_VISIBLE_DEVICES=$XLITE_NPUS
fi
if [ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]; then
    export XLITE_DEVS_PER_NODE=$(echo $ASCEND_RT_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
export XLITE_DP_SIZE=${XLITE_DP_SIZE:-1}
if [ "${XLITE_GRAPH:-0}" = "1" ]; then
    export XLITE_ENABLE_GRAPH_COMM=1
else
    export XLITE_ENABLE_GRAPH_COMM=0
fi
export MASTER_PORT=${MASTER_PORT:-$((29600 + RANDOM % 399))}

# Usage (env-driven):
#   XLITE_NPUS=<devs>    NPU ids (e.g. 0,1,...,15); -> ASCEND_RT_VISIBLE_DEVICES
#   XLITE_DP_SIZE=<dp>   DP size (default 1 = pure TP)
#   XLITE_GRAPH=<0|1>    1 = in-graph comm (default 0 = eager)
#   MODEL=<func>         run.sh function; omit for the default sweep
#   MAX_NEW_TOKENS, TEMPERATURE, RUN_MODE, FORWARD_BACKEND as before
# Example: XLITE_NPUS=0,1,...,15 XLITE_GRAPH=1 XLITE_DP_SIZE=2 \
#          MAX_NEW_TOKENS=16 MODEL=run_deepseek_v3_w8a8 bash tests/run.sh /mnt/sdb/models
#
# DP layout: world_size = tp_size * dp_size; contiguous tp_size ranks form a TP
# group, same tp_rank across groups form a DP group. Dense layers shard within
# the TP group; MoE experts span the full world (moe_ep_size * moe_tp_size ==
# world_size, keep config's moe_ep_size as-is). DP only changes XLITE_DP_SIZE.
# Interactive mode rejects XLITE_DP_SIZE>1 (use single/bench).

RUN_ARGS=(--config "$test_config_path") # 通用运行参数数组
max_new_tokens=${MAX_NEW_TOKENS:-128}
RUN_ARGS+=(--max-new-tokens "$max_new_tokens")
temperature=${TEMPERATURE:-0.0}
RUN_ARGS+=(--temperature "$temperature")

# 交互模式配置开关，可选single/interactive/bench，默认为single
run_mode=${RUN_MODE:-single}
if [ "$run_mode" = "bench" ]; then
    # bench模式不需要确定性环境变量
    RUN_ARGS+=(--mode bench)
    bench_batch_size=${BENCH_BS:-16}
    bench_iters=${BENCH_IT:-4}
    bench_prompt_len=${BENCH_N1:-2048}
    bench_new_tokens=${BENCH_N2:-1024}
    RUN_ARGS+=(--bench-batch-size "$bench_batch_size" --bench-iters "$bench_iters" --bench-prompt-len "$bench_prompt_len" --bench-new-tokens "$bench_new_tokens")
elif [ "$run_mode" = "interactive" ]; then
    export HCCL_DETERMINISTIC=true
    export LCCL_DETERMINISTIC=true
    RUN_ARGS+=(--mode interactive)
else
    export HCCL_DETERMINISTIC=true
    export LCCL_DETERMINISTIC=true
    RUN_ARGS+=(--mode single)
    # DP>1 needs a multiple of dp_size IDENTICAL queries so every DP rank gets
    # the same row count (uniform m for MoE AllGather/ReduceScatter); no rank
    # gets an empty-query sentinel whose MoE forward would corrupt the real
    # query via the DP AllGather. dp_size==1 keeps the original single query.
    _dp_n=${XLITE_DP_SIZE:-1}
    if [ "$_dp_n" -gt 1 ]; then
        _nrows=$(python -c "import sys; dp=int(sys.argv[1]); print(((dp+dp-1)//dp)*dp)" "$_dp_n")
        python - "$test_input_path" "$_nrows" <<'PY'
import json, sys
path, n = sys.argv[1], int(sys.argv[2])
item = {"query": "How to sleep well at night?", "response": ""}
json.dump([item for _ in range(n)], open(path, "w"), ensure_ascii=False, indent=4)
PY
    else
        echo '[
            {
                "query": "How to sleep well at night?",
                "response": ""
            }
        ]' > $test_input_path
    fi
    RUN_ARGS+=(--input-file "$test_input_path")
fi

function run_qwen2.5_0.5B()
{
    echo '{
        "vocab_size": 151936,
        "dim": 896,
        "inter_dim": 4864,
        "n_layers": 24,
        "n_heads": 14,
        "n_kv_heads": 2,
        "norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "dtype": "bfloat16",
        "tie_word_embeddings": true,
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    python tests/generate.py --model qwen2 --ckpt-path $models_base_path/Qwen2.5-0.5B-Instruct/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_qwen2_32B()
{
    echo '{
        "vocab_size": 152064,
        "dim": 5120,
        "inter_dim": 27648,
        "n_layers": 64,
        "n_heads": 40,
        "n_kv_heads": 8,
        "norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen2 --ckpt-path $models_base_path/qwen32b/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_qwen3_32B()
{
    local _mb=$(python -c "import sys; print(max(1, int(sys.argv[1])))" "${XLITE_DP_SIZE:-1}")
    echo '{
        "vocab_size": 151936,
        "dim": 5120,
        "head_dim": 128,
        "inter_dim": 25600,
        "n_layers": 64,
        "n_heads": 64,
        "n_kv_heads": 8,
        "norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "dtype": "bfloat16",
        "max_batch_size": '"$_mb"',
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path $models_base_path/Qwen3-32B/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_qwen3_moe_30B()
{
    echo '{
        "vocab_size": 151936,
        "dim": 2048,
        "head_dim": 128,
        "inter_dim": 6144,
        "moe_inter_dim": 768,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "n_routed_experts": 128,
        "n_activated_experts": 8,
        "n_layers": 48,
        "n_heads": 32,
        "n_kv_heads": 4,
        "norm_eps": 1e-06,
        "rope_theta": 10000000.0,
        "moe_ep_size": 8,
        "moe_tp_size": 1,
        "dtype": "bfloat16",
        "max_batch_size": 64,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_moe --ckpt-path $models_base_path/Qwen3-30B-A3B-Instruct-2507/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_llama_7B()
{
    local dp=${XLITE_DP_SIZE:-1}
    echo '{
        "vocab_size": 32000,
        "dim": 4096,
        "inter_dim": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,
        "norm_eps": 1e-05,
        "dtype": "float16",
        "max_batch_size": '"$dp"',
        "max_seq_len": 1024
    }' > $test_config_path
    XLITE_DP_SIZE=$dp torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-2} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/Llama-2-7b-chat-hf/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_llama_13B()
{
    echo '{
        "vocab_size": 32000,
        "dim": 5120,
        "inter_dim": 13824,
        "n_layers": 40,
        "n_heads": 40,
        "n_kv_heads": 40,
        "norm_eps": 1e-05,
        "dtype": "float16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-2} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/Llama2-Chinese-13b-Chat/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_llama_34B()
{
    echo '{
        "vocab_size": 32000,
        "dim": 8192,
        "inter_dim": 22016,
        "n_layers": 48,
        "n_heads": 64,
        "n_kv_heads": 8,
        "norm_eps": 1e-05,
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/codellama34B/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_deepseek_v3_w8a8()
{
    echo '{
        "vocab_size": 129280,
        "dim": 7168,
        "inter_dim": 18432,
        "moe_inter_dim": 2048,
        "n_layers": 61,
        "n_dense_layers": 3,
        "n_heads": 128,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_expert_groups": 8,
        "n_limited_groups": 4,
        "route_scale": 2.5,
        "score_func": "sigmoid",
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "original_seq_len": 4096,
        "rope_theta": 10000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "dtype": "bf16",
        "quantization": "w8a8",
        "moe_ep_size": 16,
        "moe_tp_size": 1,
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-16} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v3 --ckpt-path $models_base_path/DeepSeek-V3.1-w8a8-mtp-QuaRot ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_deepseek_v4_w8a8()
{
    echo '{
        "vocab_size": 129280,
        "dim": 4096,
        "moe_inter_dim": 2048,
        "n_layers": 43,
        "n_hash_layers": 3,
        "n_mtp_layers": 0,
        "n_heads": 64,
        "norm_eps": 1e-06,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 6,
        "score_func": "sqrtsoftplus",
        "route_scale": 1.5,
        "swiglu_limit": 10.0,
        "q_lora_rank": 1024,
        "head_dim": 512,
        "rope_head_dim": 64,
        "o_groups": 8,
        "o_lora_rank": 1024,
        "window_size": 128,
        "compress_ratios": [0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0],
        "compress_rope_theta": 160000.0,
        "original_seq_len": 65536,
        "rope_theta": 10000.0,
        "rope_factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "temperature": 1.0,
        "dtype": "bf16",
        "quantization": "w8a8",
        "moe_ep_size": 8,
        "moe_tp_size": 1,
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v4 --ckpt-path $models_base_path/DeepSeek-V4-Flash-w8a8-mtp ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_glm4_moe()
{
    echo '{
        "moe_ep_size": 16,
        "moe_tp_size": 1,
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-16} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm4_moe --ckpt-path $models_base_path/GLM-4.7/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_deepseek_v32()
{
    echo '{
        "vocab_size": 129280,
        "dim": 7168,
        "inter_dim": 18432,
        "moe_inter_dim": 2048,
        "n_layers": 61,
        "n_dense_layers": 3,
        "n_heads": 128,
        "norm_eps": 1e-06,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_expert_groups": 8,
        "n_limited_groups": 4,
        "score_func": "sigmoid",
        "route_scale": 2.5,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "original_seq_len": 4096,
        "rope_theta": 10000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "quantization": "none",
        "model_type": "deepseek_v32",
        "dtype": "bf16",
        "moe_ep_size": 16,
        "moe_tp_size": 1,
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-16} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v32 --ckpt-path $models_base_path/DeepSeek-V3.2-bf16/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_glm5()
{
    echo '{
        "vocab_size": 154880,
        "dim": 6144,
        "inter_dim": 12288,
        "moe_inter_dim": 2048,
        "n_layers": 78,
        "n_dense_layers": 3,
        "n_heads": 64,
        "norm_eps": 1e-05,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "score_func": "sigmoid",
        "route_scale": 2.5,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "original_seq_len": 4096,
        "rope_theta": 1000000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "max_batch_size": 1,
        "max_seq_len": 1024,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "indexer_rope_interleave": true,
        "quantization": "none",
        "model_type": "glm5",
        "dtype": "bfloat16",
        "moe_ep_size": 16,
        "moe_tp_size": 1,
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-16} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm5 --ckpt-path $models_base_path/GLM-5/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_glm5_w8a8()
{
    # 层数 (离线 bench 可裁剪, 如 8 层只测性能); 默认 78 = 完整模型。
    local _n_layers=${XLITE_N_LAYERS:-78}
    # moe_ep_size 须等于卡数; 默认 16, 离线 8 卡场景设 8。
    local _moe_ep_size=${XLITE_MOE_EP_SIZE:-16}
    # checkpoint 目录名 (不同环境挂载名可能不同); 默认 GLM-5-w8a8。
    local _ckpt_dir=${XLITE_GLM5_W8A8_CKPT:-GLM-5-w8a8}
    echo '{
        "vocab_size": 154880,
        "dim": 6144,
        "inter_dim": 12288,
        "moe_inter_dim": 2048,
        "n_layers": '"$_n_layers"',
        "n_dense_layers": 3,
        "n_heads": 64,
        "norm_eps": 1e-05,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "score_func": "sigmoid",
        "route_scale": 2.5,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "original_seq_len": 4096,
        "rope_theta": 1000000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "max_batch_size": 1,
        "max_seq_len": 1024,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "indexer_rope_interleave": true,
        "quantization": "w8a8",
        "model_type": "glm5",
        "dtype": "bfloat16",
        "moe_ep_size": '"$_moe_ep_size"',
        "moe_tp_size": 1
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-16} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm5 --ckpt-path $models_base_path/$_ckpt_dir/ ${RUN_ARGS[@]}
    rm $test_config_path
}

# GLM-5.2 W8A8 (shared indexer: index_topk_freq=4/index_skip_topk_offset=3,
# layer 3 skips indexer and reuses layer 2 topkIndices). EP8, moe_ep_size=8.
function run_glm5_2_w8a8()
{
    echo '{
        "vocab_size": 154880,
        "dim": 6144,
        "inter_dim": 12288,
        "moe_inter_dim": 2048,
        "n_layers": 4,
        "n_dense_layers": 3,
        "n_heads": 64,
        "norm_eps": 1e-05,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_expert_groups": 1,
        "n_limited_groups": 1,
        "score_func": "sigmoid",
        "route_scale": 2.5,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "original_seq_len": 4096,
        "rope_theta": 8000000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "max_batch_size": 1,
        "max_seq_len": 1024,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": 2048,
        "indexer_rope_interleave": true,
        "index_topk_freq": 4,
        "index_skip_topk_offset": 3,
        "quantization": "w8a8",
        "model_type": "glm5",
        "dtype": "bfloat16",
        "moe_ep_size": 8,
        "moe_tp_size": 1
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm5 --ckpt-path $models_base_path/GLM-5.2-w8a8/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_minimax_m2()
{
    echo '{
        "moe_ep_size": 16,
        "moe_tp_size": 1,
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-16} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model minimax_m2 --ckpt-path $models_base_path/MiniMax-M2.7-bf16/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_qwen3_5_0.8B()
{
    echo '{
        "vocab_size": 248320,
        "dim": 1024,
        "head_dim": 256,
        "inter_dim": 3584,
        "n_layers": 24,
        "n_heads": 8,
        "n_kv_heads": 2,
        "norm_eps": 1e-06,
        "rope_theta": 10000000.0,
        "dtype": "bfloat16",
        "tie_word_embeddings": true,
        "qkv_bias": false,
        "qk_norm": true,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    python tests/generate.py --model qwen3_5 --ckpt-path $models_base_path/Qwen3.5-0.8B/ --config $test_config_path ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_qwen3_5_27B()
{
    echo '{
        "vocab_size": 248320,
        "dim": 5120,
        "head_dim": 256,
        "inter_dim": 17408,
        "n_layers": 64,
        "n_heads": 24,
        "n_kv_heads": 4,
        "norm_eps": 1e-06,
        "rope_theta": 10000000.0,
        "dtype": "bfloat16",
        "tie_word_embeddings": false,
        "qkv_bias": false,
        "qk_norm": true,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    python tests/generate.py --model qwen3_5 --ckpt-path $models_base_path/Qwen3.5-27B/ --config $test_config_path ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_qwen3_5_moe_35B()
{
    echo '{
        "vocab_size": 248320,
        "dim": 2048,
        "head_dim": 256,
        "inter_dim": 6144,
        "moe_inter_dim": 512,
        "shared_expert_inter_dim": 512,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_layers": 40,
        "n_heads": 16,
        "n_kv_heads": 2,
        "norm_eps": 1e-06,
        "rope_theta": 10000000.0,
        "full_attention_interval": 4,
        "moe_ep_size": 8,
        "moe_tp_size": 1,
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-8} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_5_moe --ckpt-path $models_base_path/Qwen3.5-35B-A3B/ ${RUN_ARGS[@]}
    rm $test_config_path
}

function run_qwen3_5_moe_122B()
{
    echo '{
        "vocab_size": 248320,
        "dim": 3072,
        "head_dim": 256,
        "inter_dim": 9216,
        "moe_inter_dim": 1024,
        "shared_expert_inter_dim": 1024,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_layers": 48,
        "n_heads": 32,
        "n_kv_heads": 2,
        "norm_eps": 1e-06,
        "rope_theta": 10000000.0,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "partial_rotary_factor": 0.25,
        "moe_ep_size": 16,
        "moe_tp_size": 1,
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=${XLITE_DEVS_PER_NODE:-16} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_5_moe --ckpt-path $models_base_path/Qwen3.5-122B-A10B/ ${RUN_ARGS[@]}
    rm $test_config_path
}

model_func=${MODEL:-}

if [ -n "$model_func" ]; then
    # 仅运行指定模型
    $model_func
else
    # 默认运行所有模型
    run_llama_7B
    run_llama_13B
    #run_llama_34B

    #run_qwen2.5_0.5B
    #run_qwen2_32B
    run_qwen3_32B
    run_qwen3_moe_30B
    #run_qwen3_5_0.8B
    #run_qwen3_5_moe_35B

    npu_count=$(python -c "import torch; print(torch.npu.device_count())")
    if [ $npu_count -ge 16 ]; then
        run_glm4_moe
        run_deepseek_v3_w8a8
        run_glm5_w8a8
        run_minimax_m2
        #run_qwen3_5_moe_122B
    fi

    # bf16 need 32 NPUs
    #run_deepseek_v32
    #run_glm5
fi

# 清理默认输入文件（如果存在）
if [ -f "$test_input_path" ]; then
    rm $test_input_path
fi
