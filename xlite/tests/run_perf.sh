models_base_path=${1:-/mnt/nvme0n1/models}
function run_qwen2_32B()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
    dtype=$6
    echo '{
        "vocab_size": 152064,
        "dim": 5120,
        "inter_dim": 27648,
        "n_layers": 64,
        "n_heads": 40,
        "n_kv_heads": 8,
        "norm_eps": 1e-05,
        "rope_theta": 1000000.0,
        "dtype": "'${dtype}'",
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen2 --ckpt-path $models_base_path/qwen32b/ --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function run_qwen3_32B()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
    dtype=$6
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
        "dtype": "'${dtype}'",
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path $models_base_path/Qwen3-32B/ --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function run_qwen3_moe_30B()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
    dtype=$6
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
        "moe_ep_size": '${tp}',
        "moe_tp_size": 1,
        "dtype": "'${dtype}'",
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3_moe --ckpt-path $models_base_path/Qwen3-30B-A3B-Instruct-2507/ --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function run_llama_7B()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
    dtype=$6
    echo '{
        "vocab_size": 32000,
        "dim": 4096,
        "inter_dim": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 32,
        "norm_eps": 1e-05,
        "dtype": "'${dtype}'",
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/Llama-2-7b-chat-hf/ --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function run_llama_13B()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
    dtype=$6
    echo '{
        "vocab_size": 32000,
        "dim": 5120,
        "inter_dim": 13824,
        "n_layers": 40,
        "n_heads": 40,
        "n_kv_heads": 40,
        "norm_eps": 1e-05,
        "dtype": "'${dtype}'",
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/Llama2-Chinese-13b-Chat/ --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function run_llama_34B()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
    dtype=$6
    echo '{
        "vocab_size": 32000,
        "dim": 8192,
        "inter_dim": 22016,
        "n_layers": 48,
        "n_heads": 64,
        "n_kv_heads": 8,
        "norm_eps": 1e-05,
        "dtype": "'${dtype}'",
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model llama --ckpt-path $models_base_path/codellama34B/ --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function run_glm4_moe()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
    echo '{
        "moe_ep_size": '${tp}',
        "moe_tp_size": 1,
        "dtype": "bfloat16",
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model glm4_moe --ckpt-path $models_base_path/GLM-4.7/ --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function run_deepseek_v3()
{
    tp=$1
    bs=$2
    input_tokens=$3
    output_tokens=$4
    max_seq_len=$(($input_tokens + $output_tokens))
    file=$5
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
        "moe_tp_size": 1,
        "max_seq_len": '${max_seq_len}',
        "max_batch_size": '${bs}'
    }' > tests/test_config.json

    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model deepseek_v3 --ckpt-path $models_base_path/DeepSeek-R1-expert-int8 --config tests/test_config.json --input-file tests/test_output/${file}.json --max-new-tokens ${output_tokens} | tee tests/test_output/${file}.log
}

function test_perf_decode {
    model=$1
    tp=$2
    bs=$3
    output_tokens=$4
    backend=$5
    dtype=$6
    file=${backend}_${model}_${dtype}_bs${bs}len${output_tokens}
    echo '[' > tests/test_output/${file}.json
    for ((i=1; i<=${bs}; i++)); do
        if [ $i -ne ${bs} ]; then
            echo '   {
                "query": "小明从家走到学校需要13分钟，早上上学他出发了6分钟之后，发现本子还在家里需要回去拿，请问他还需要多久才能到学校",
                "response": ""
            },' >> tests/test_output/${file}.json
        else
            echo '   {
                "query": "小明从家走到学校需要13分钟，早上上学他出发了6分钟之后，发现本子还在家里需要回去拿，请问他还需要多久才能到学校",
                "response": ""
            }' >> tests/test_output/${file}.json
        fi
    done
    echo ']' >> tests/test_output/${file}.json

    input_tokens=100
    if [[ "${model}" == "qwen2_32B" ]]; then
        run_qwen2_32B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "qwen3_32B" ]]; then
        run_qwen3_32B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "qwen3_moe_30B" ]]; then
        run_qwen3_moe_30B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "llama_7B" ]]; then
        run_llama_7B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "llama_13B" ]]; then
        run_llama_13B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "llama_34B" ]]; then
        run_llama_34B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "glm4_moe" ]]; then
        run_glm4_moe ${tp} ${bs} ${input_tokens} ${output_tokens} ${file}
    elif [[ "${model}" == "deepseek_v3" ]]; then
        run_deepseek_v3 ${tp} ${bs} ${input_tokens} ${output_tokens} ${file}
    fi
    perf_stat=$(cat "tests/test_output/${file}.log" | grep "tokens\/s @ " | grep "generate" | head -n 1)
    if [[ "${model}" == "deepseek_v3" || "${model}" == "qwen3_moe_30B" || "${model}" == "glm4_moe" ]]; then
        echo $perf_stat | awk -v bs="$bs" '{print "| '${model}' | '${dtype}' | TP'${tp}'moeEP'${tp}' | " $2 / bs " | " $14 " | " $11 " | " $8 " |"}' >> ${perf_log_file}
    else
        echo $perf_stat | awk -v bs="$bs" '{print "| '${model}' | '${dtype}' | TP'${tp}' | " $2 / bs " | " $14 " | " $11 " | " $8 " |"}' >> ${perf_log_file}
    fi
    sleep 10
}

function test_perf_prefill {
    model=$1
    tp=$2
    bs=$3
    output_tokens=$4
    backend=$5
    dtype=$6
    file=${backend}_${model}_${dtype}_bs${bs}len${output_tokens}
    cp tests/test_prefill.json tests/test_output/${file}.json

    input_tokens=3456
    if [[ "${model}" == "qwen2_32B" ]]; then
        run_qwen2_32B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "qwen3_32B" ]]; then
        run_qwen3_32B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "qwen3_moe_30B" ]]; then
        run_qwen3_moe_30B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "llama_7B" ]]; then
        run_llama_7B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "llama_13B" ]]; then
        run_llama_13B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "llama_34B" ]]; then
        run_llama_34B ${tp} ${bs} ${input_tokens} ${output_tokens} ${file} ${dtype}
    elif [[ "${model}" == "glm4_moe" ]]; then
        run_glm4_moe ${tp} ${bs} ${input_tokens} ${output_tokens} ${file}
    elif [[ "${model}" == "deepseek_v3" ]]; then
        run_deepseek_v3 ${tp} ${bs} ${input_tokens} ${output_tokens} ${file}
    fi
    perf_stat=$(cat "tests/test_output/${file}.log" | grep "tokens\/s @ " | grep "generate" | head -n 1)
    if [[ "${model}" == "deepseek_v3" || "${model}" == "qwen3_moe_30B" || "${model}" == "glm4_moe" ]]; then
        echo $perf_stat | awk -v bs="$bs" '{print "| '${model}' | '${dtype}' | TP'${tp}'moeEP'${tp}' | " $2 / bs " | " $14 " | " $11 " |"}' >> ${perf_log_file}
    else
        echo $perf_stat | awk -v bs="$bs" '{print "| '${model}' | '${dtype}' | TP'${tp}' | " $2 / bs " | " $14 " | " $11 " |"}' >> ${perf_log_file}
    fi
    sleep 10
}

export HCCL_OP_BASE_FFTS_MODE=TRUE
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
backend=xlite
export FORWARD_BACKEND=${backend}

mkdir -p tests/test_output
perf_log_file=tests/test_output/${backend}_perf.log
echo "| 模型 | 类型 | 并行策略 | 输入输出长度 | batch size | 平均decode时延 | 平均decode吞吐 |" > ${perf_log_file}
echo "|--|--|--|--|--|--|--|" >> ${perf_log_file}


model=llama_7B
tp=1
output_tokens=1024
dtype=float16
for bs in 1
do
    test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
done

model=llama_13B
tp=4
output_tokens=1024
dtype=float16
for bs in 1 16
do
    test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
done

#model=llama_34B
#tp=8
#output_tokens=1024
#dtype=bfloat16
#for bs in 1 16
#do
#    test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
#done

#model=qwen2_32B
#tp=8
#output_tokens=1024
#dtype=bfloat16
#for bs in 1 16
#do
#    test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
#done

for dtype in float16 bfloat16
do
    model=qwen3_32B
    tp=8
    output_tokens=1024
    for bs in 1 16
    do
        test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
    done
done

model=qwen3_moe_30B
tp=8
output_tokens=1024
dtype=bfloat16
for bs in 1 16
do
    test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
done

npu_count=$(python -c "import torch; print(torch.npu.device_count())")
if [ $npu_count -ge 16 ]; then
    model=glm4_moe
    tp=16
    output_tokens=1024
    dtype=bfloat16
    for bs in 1 16 64
    do
        test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
    done

    model=deepseek_v3
    tp=16
    output_tokens=1024
    dtype=bfloat16
    for bs in 1 16
    do
        test_perf_decode ${model} ${tp} ${bs} ${output_tokens} ${backend} ${dtype}
    done
fi

echo "" >> ${perf_log_file}
echo "| 模型 | 类型 | 并行策略 | 输入输出长度 | batch size | prefill时延 |" >> ${perf_log_file}
echo "|--|--|--|--|--|--|" >> ${perf_log_file}


model=llama_7B
tp=1
dtype=float16
test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}

model=llama_13B
tp=4
dtype=float16
test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}

#model=llama_34B
#tp=8
#dtype=bfloat16
#test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}

#model=qwen2_32B
#tp=8
#dtype=bfloat16
#test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}

for dtype in float16 bfloat16
do
    model=qwen3_32B
    tp=8
    test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}
done

model=qwen3_moe_30B
tp=8
dtype=bfloat16
test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}

npu_count=$(python -c "import torch; print(torch.npu.device_count())")
if [ $npu_count -ge 16 ]; then
    model=glm4_moe
    tp=16
    dtype=bfloat16
    test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}

    model=deepseek_v3
    tp=16
    dtype=bfloat16
    test_perf_prefill ${model} ${tp} 1 1 ${backend} ${dtype}
fi