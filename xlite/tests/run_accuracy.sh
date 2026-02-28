#!/bin/bash

export FORWARD_BACKEND=xlite
export HCCL_DETERMINISTIC=true
export LCCL_DETERMINISTIC=true

function run_qwen3_0.6B() {
    tp=$1
    bs=$2
    max_tokens=$3
    temperature=$4
    file=$5
    
    echo '{
        "vocab_size": 151936,
        "dim": 1024,
        "head_dim": 128,
        "inter_dim": 3072,
        "n_layers": 28,
        "n_heads": 16,
        "n_kv_heads": 8,
        "norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "dtype": "bfloat16",
        "tie_word_embeddings": true,
        "max_batch_size": '${bs}',
        "max_seq_len": 1024
    }' > tests/test_config.json
    
    torchrun --nproc_per_node=${tp} --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path /mnt/nvme0n1/models/Qwen3-0.6B/ --config tests/test_config.json --input-file tests/${file}.json --max-new-tokens ${max_tokens} --temperature ${temperature} --no-prefix | tee tests/${file}.log
}

function test_accuracy() {
    model=$1
    tp=$2
    bs=$3
    max_tokens=$4
    temperature=$5
    backend=$6
    dtype=$7
    file=${backend}_${model}_${dtype}_accuracy
    
    echo '[' > tests/${file}.json
    
    prompts=(
        "Hello, my name is"
        "The president of the United States is"
        "The capital of France is"
        "The future of AI is"
    )
    
    for ((i=0; i<${#prompts[@]}; i++)); do
        if [ $i -ne $((${#prompts[@]} - 1)) ]; then
            echo '   {
                "query": "'"${prompts[$i]}"'",
                "response": ""
            },' >> tests/${file}.json
        else
            echo '   {
                "query": "'"${prompts[$i]}"'",
                "response": ""
            }' >> tests/${file}.json
        fi
    done
    echo ']' >> tests/${file}.json
    
    if [[ "${model}" == "qwen3_0.6B" ]]; then
        run_qwen3_0.6B ${tp} ${bs} ${max_tokens} ${temperature} ${file}
    fi
    
    python3 << 'EOF'
import json

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is"
]

golden_answers = [
    " Lina. I'm a 22-year-old student from China. I'm interested in studying in the US. I'm looking for a job in the",
    ' the same as the president of the United Nations. This is because the president of the United States is the same as the president of the United Nations. The president',
    ' Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. The capital of Japan is Tokyo. The capital',
    " not just a technological challenge but a profound transformation of how we live, work, and interact with the world. As we stand at the intersection of artificial intelligence and"
]

backend = "xlite"
model = "qwen3_0.6B"
dtype = "bfloat16"
file = f"{backend}_{model}_{dtype}_accuracy"

with open(f"tests/{file}.json", 'r', encoding='utf-8') as f:
    results = json.load(f)

print("")
print("=" * 40)
print("Accuracy Test Results")
print("=" * 40)

passed = 0
failed = 0

for i in range(len(prompts)):
    query = prompts[i]
    expected = golden_answers[i]
    actual = results[i]['response']
    
    if actual == expected:
        print(f"✓ Test {i} PASSED")
        passed += 1
    else:
        print(f"✗ Test {i} FAILED")
        print(f"  Query: {query}")
        print(f"  Expected: {expected}")
        print(f"  Actual:   {actual}")
        failed += 1

print("")
print("=" * 40)
print(f"Summary: {passed} passed, {failed} failed out of {len(prompts)} tests")
print("=" * 40)

if failed == 0:
    print("All accuracy tests PASSED!")
    exit(0)
else:
    print("Some accuracy tests FAILED!")
    exit(1)
EOF
}

backend=xlite
export FORWARD_BACKEND=${backend}

model=qwen3_0.6B
tp=1
bs=4
max_tokens=32
temperature=0.0
dtype=bfloat16

echo "Running accuracy test for ${model}..."
test_accuracy ${model} ${tp} ${bs} ${max_tokens} ${temperature} ${backend} ${dtype}
