export FORWARD_BACKEND=xlite
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=true

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
    }' > tests/test_config.json
    python tests/generate.py --model qwen2 --ckpt-path /mnt/nvme0n1/models/Qwen2.5-0.5B-Instruct/ --config tests/test_config.json --interactive
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
    }' > tests/test_config.json
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen2 --ckpt-path /mnt/nvme0n1/models/qwen32b/ --config tests/test_config.json --interactive

    # batch input
    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen --ckpt-path /mnt/nvme0n1/models/qwen32b/ --config tests/test_config.json --input tests/test.json
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
    }' > tests/test_config.json
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path /mnt/nvme0n1/models/Qwen3-32B/ --config tests/test_config.json --interactive

    # batch input
    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path /mnt/nvme0n1/models/Qwen3-32B/ --config tests/test_config.json --input tests/test.json
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
        "dtype": "bfloat16",
        "max_batch_size": 1,
        "max_seq_len": 1024
    }' > tests/test_config.json
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_moe --ckpt-path /mnt/nvme0n1/models/Qwen3-30B-A3B-Instruct-2507/ --config tests/test_config.json --interactive
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
    }' > tests/test_config.json
    python tests/generate.py --model llama --ckpt-path /mnt/nvme0n1/models/Llama-2-7b-chat-hf/ --config tests/test_config.json --interactive
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
    }' > tests/test_config.json
    torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path /mnt/nvme0n1/models/Llama2-Chinese-13b-Chat/ --config tests/test_config.json --interactive
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
    }' > tests/test_config.json
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path /mnt/nvme0n1/models/codellama34B/ --config tests/test_config.json --interactive
}

function run_deepseek()
{
    # n_layers: 8
    # quantization: experts_int8
    echo '{
        "vocab_size": 129280,
        "dim": 7168,
        "inter_dim": 18432,
        "moe_inter_dim": 2048,
        "n_layers": 8,
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
        "quantization": "experts_int8"
    }' > tests/test_config.json
    # bf16
    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --ckpt-path /mnt/nvme1n1/models/deepseek-R1-bf16-8layers-8d/ --config tests/test_config.json --interactive

    # w8a8
    torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --ckpt-path /mnt/nvme1n1/models/deepseek-R1-expert-int8-8layers-8d/ --config tests/test_config.json --interactive

    # batch input
    # torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --ckpt-path /mnt/nvme1n1/models/deepseek-R1-expert-int8-8layers-8d/ --config tests/test_config.json --input tests/test.json
}

#run_qwen2.5_0.5B
run_qwen2_32B
run_qwen3_32B
#run_qwen3_moe_30B
run_llama_7B
run_llama_13B
run_llama_34B
#run_deepseek
