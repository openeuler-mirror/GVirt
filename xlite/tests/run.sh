export FORWARD_BACKEND=xlite
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=true
models_base_path=${1:-/mnt/nvme0n1/models}
test_config_path=tests/test_config.json

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
    python tests/generate.py --model qwen2 --ckpt-path $models_base_path/Qwen2.5-0.5B-Instruct/ --config $test_config_path --interactive

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
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen2 --ckpt-path $models_base_path/qwen32b/ --config $test_config_path --interactive

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
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path $models_base_path/Qwen3-32B/ --config $test_config_path --interactive

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
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_moe --ckpt-path $models_base_path/Qwen3-30B-A3B-Instruct-2507/ --config $test_config_path --interactive
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
    python tests/generate.py --model llama --ckpt-path $models_base_path/Llama-2-7b-chat-hf/ --config $test_config_path --interactive
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
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/Llama2-Chinese-13b-Chat/ --config $test_config_path --interactive
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
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/codellama34B/ --config $test_config_path --interactive
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
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v3 --ckpt-path $models_base_path/DeepSeek-R1-expert-int8 --config $test_config_path --interactive

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
    torchrun --nproc_per_node=16 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm4_moe --ckpt-path $models_base_path/GLM-4.7/ --config $test_config_path --interactive
    rm $test_config_path
}

#run_qwen2.5_0.5B
#run_qwen2_32B
run_qwen3_32B
run_qwen3_moe_30B
run_llama_7B
run_llama_13B
#run_llama_34B

npu_count=$(python -c "import torch; print(torch.npu.device_count())")
if [ $npu_count -ge 16 ]; then
    run_glm4_moe
    run_deepseek_v3
fi
