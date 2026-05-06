export FORWARD_BACKEND=xlite
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=true
models_base_path=${1:-/mnt/nvme0n1/models}
test_config_path=tests/test_config.json

# 交互模式配置开关
# 设置为 "true" 启用交互模式，设置为其他值或不设置则使用批量输入模式
INTERACTIVE_MODE=${INTERACTIVE_MODE:-false}

# 默认输入文件路径（非交互模式使用）
DEFAULT_INPUT_FILE=tests/test_input_default.json

# 根据 INTERACTIVE_MODE 设置通用参数
if [ "$INTERACTIVE_MODE" = "true" ]; then
    INTERACTIVE_FLAG="--interactive"
    INPUT_FILE_FLAG=""
else
    INTERACTIVE_FLAG=""
    # 创建默认输入文件
    echo '[
        {
            "query": "你好，请介绍一下自己。",
            "response": ""
        }
    ]' > $DEFAULT_INPUT_FILE
    INPUT_FILE_FLAG="--input-file $DEFAULT_INPUT_FILE"
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
    python tests/generate.py --model qwen2 --ckpt-path $models_base_path/Qwen2.5-0.5B-Instruct/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG

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
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen2 --ckpt-path $models_base_path/qwen32b/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG

    # batch input
    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen --ckpt-path $models_base_path/qwen32b/ --config $test_config_path --input tests/test.json

    rm $test_config_path
}

function run_qwen3_32B()
{
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
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path $models_base_path/Qwen3-32B/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG

    # batch input
    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path $models_base_path/Qwen3-32B/ --config $test_config_path --input tests/test.json

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
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_moe --ckpt-path $models_base_path/Qwen3-30B-A3B-Instruct-2507/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
    rm $test_config_path
}

function run_llama_7B()
{
    echo '{
        "vocab_size": 32000,
        "dim": 4096,
        "inter_dim": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,
        "norm_eps": 1e-05,
        "dtype": "float16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > $test_config_path
    python tests/generate.py --model llama --ckpt-path $models_base_path/Llama-2-7b-chat-hf/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
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
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/Llama2-Chinese-13b-Chat/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
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
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/codellama34B/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
    rm $test_config_path
}

function run_deepseek_v3()
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
        "dtype": "bf16",
        "quantization": "experts_int8",
        "moe_ep_size": 16,
        "moe_tp_size": 1
    }' > $test_config_path
    # w8a8
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v3 --ckpt-path $models_base_path/DeepSeek-R1-expert-int8 --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG

    # batch input
    # torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v3 --ckpt-path $models_base_path/DeepSeek-R1-expert-int8 --config $test_config_path --input tests/test.json

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
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm4_moe --ckpt-path $models_base_path/GLM-4.7/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
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
        "moe_tp_size": 1
    }' > $test_config_path
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v32 --ckpt-path $models_base_path/DeepSeek-V3.2-bf16/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
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
        "moe_tp_size": 1
    }' > $test_config_path
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm5 --ckpt-path $models_base_path/GLM-5/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
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
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model minimax_m2 --ckpt-path $models_base_path/MiniMax-M2.5-bf16/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG
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
    python tests/generate.py --model qwen3_5 --ckpt-path $models_base_path/Qwen3.5-0.8B/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG --max-new-tokens 64
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
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_5_moe --ckpt-path $models_base_path/Qwen3.5-35B-A3B/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG --max-new-tokens 64
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
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_5_moe --ckpt-path $models_base_path/Qwen3.5-122B-A10B/ --config $test_config_path $INTERACTIVE_FLAG $INPUT_FILE_FLAG --max-new-tokens 64
    rm $test_config_path
}

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
    run_deepseek_v3
    run_minimax_m2
    #run_qwen3_5_moe_122B
    #run_deepseek_v32
    #run_glm5
fi

# 清理默认输入文件（如果存在）
if [ -f "$DEFAULT_INPUT_FILE" ]; then
    rm $DEFAULT_INPUT_FILE
fi
