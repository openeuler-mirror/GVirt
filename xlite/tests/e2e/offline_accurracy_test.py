from vllm import LLM, SamplingParams
from vllm_ascend.batch_invariant import init_batch_invariance

import sys

# 开启xlite/开启量化
# example: python offline_accurracy_test.py --xlite --quant
has_xlite = '--xlite' in sys.argv
has_quant = '--quant' in sys.argv

# 开启 batch 不变性，消除 batch 间数值不确定性
init_batch_invariance()

base_model_path = "/mnt/nvme0n1/models/"

models = [
    ["qwen3-0.6B", f"{base_model_path}/Qwen3-0.6B", 1],
    ["qwen3-32B", f"{base_model_path}/Qwen3-32B", 8],
    ["qwen3-moe", f"{base_model_path}/Qwen3-30B-A3B-Instruct-2507", 8],
]

# 部分模型仅量化后可使用单机运行
quant_models = [
    ["qwen3-0.6B", f"{base_model_path}/Qwen3-0.6B-W8A8", 1],
    ["qwen3-32B", f"{base_model_path}/Qwen3-32B-w8a8-nopdmix", 8],
    ["qwen3-moe", f"{base_model_path}/Qwen3-30B-A3B-Instruct-2507-w8a8/", 8],
    # ["minimax-2.7", f"{base_model_path}/MiniMax-M2.7-w8a8-QuaRot/", 8],
    ["glm-4.7-w8a8", f"{base_model_path}/GLM-4.7-W8A8-floatmtp", 8],
]

prompts = [
    "The future of AI is",
    "vLLM is a high-throughput and memory-efficient inference and serving engine for LLMs.",
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "介绍一下华为",
    "13 + 27 等于多少？",
    "0.1 + 0.2 等于多少？",
    "中国的首都是哪里？",
    "如果 A>B，B>C，A 和 C 谁大？",
    "床前明月光的下一句是什么？",
    "用 Python 写一行代码计算 1 到 100 的和。",
    "请只回答是或否：地球是圆的吗？",
]

# 采样设置
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=100
)

# 模型遍历
target_models = quant_models if has_quant else models
for name, model, tp_size in target_models:
    llm = LLM(
        model=model,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.97,
        block_size=128,
        quantization="ascend" if has_quant else None,
        enforce_eager=True,
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"} if has_xlite else None,
        additional_config={"xlite_graph_config": {"enabled": True, "full_mode": True}} if has_xlite else {},
        trust_remote_code=True,
        seed=0
    )

    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{name:>15}]: Prompt: {prompt!r}, Generated text: {generated_text!r}")
    del llm
