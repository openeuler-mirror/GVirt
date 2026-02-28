### vllm_ascend + xlite 在线服务性能测试及对比

1. **快速开始**
```
# 安装vllm_ascend, 可参考https://github.com/vllm-project/vllm-ascend/blob/main/README.md

# 安装xlite
pip install xlite
```

2. **启动在线服务**
```
# xlite可选择decode only或full其中一种模式启动在线服务：
# a. 配置xlite decode only模式，设置--additional-config='{"xlite_graph_config": {"enabled": true}}'
cd ./tests/e2e/
bash online_server_xlite_decode_only.sh 8080 server.log [model_path] [max_num_batched_tokens] [max_num_seqs] [max_model_len] [tensor_parallel_size]

# b. 配置xlite full模式，设置--additional-config='{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
cd ./tests/e2e/
bash online_server_xlite_full.sh 8080 server.log [model_path] [max_num_batched_tokens] [max_num_seqs] [max_model_len] [tensor_parallel_size]

# c. 配置aclgraph模式（不使用xlite）
cd ./tests/e2e/
bash online_server_aclgraph.sh 8080 server.log [model_path] [max_num_batched_tokens] [max_num_seqs] [max_model_len] [tensor_parallel_size]

# 参数说明：
# - port: 服务端口（必需）
# - log: 日志文件路径（必需）
# - model_path: 模型路径（可选，默认/mnt/nvme0n1/models/Qwen3Qwen3-32B）
# - max_num_batched_tokens: 最大批处理token数（可选，默认8192）
# - max_num_seqs: 最大序列数（可选，默认200）
# - max_model_len: 最大模型长度（可选，默认6656）
# - tensor_parallel_size: 张量并行大小（可选，默认8）
```

3. **启动在线压测**
```
# 运行压测脚本
cd ./tests/e2e/
bash online_server_test.sh [TYPE] [INPUT_LEN] [OUTPUT_LEN] [MODEL_NAME] [TOKENIZER_PATH] [HOST] [PORT] [CONCURRENCY_LIST] [NUM_PROMPTS_MULTIPLIER] [MAIN_OUTPUT_DIR]

# 参数说明：
# - TYPE: 测试类型标识（可选，默认default）
# - INPUT_LEN: 输入token长度（可选，默认512）
# - OUTPUT_LEN: 输出token长度（可选，默认512）
# - MODEL_NAME: 模型名称（可选，默认qwen）
# - TOKENIZER_PATH: tokenizer路径（可选，默认/mnt/nvme0n1/models/Qwen3-32B）
# - HOST: 服务器地址（可选，默认127.0.0.1）
# - PORT: 服务器端口（可选，默认8080）
# - CONCURRENCY_LIST: 并发数列表，空格分隔（可选，默认"1 16 32 48 64 100"）
# - NUM_PROMPTS_MULTIPLIER: num_prompts倍数，num_prompts = maxconcurrency * NUM_PROMPTS_MULTIPLIER（可选，默认10）
# - MAIN_OUTPUT_DIR: 主输出目录（可选）

# 示例：使用默认参数
bash online_server_test.sh

# 示例：自定义参数
bash online_server_test.sh xlite_decode_only 512 512 qwen /mnt/nvme0n1/models/Qwen3-32B 127.0.0.1 8080 "1 16 32 48 64 100" 10 ./benchmark_results
```

4. **性能对比**
```
# 方式方法1：一键运行完整性能对比（推荐）
# 自动测试多个场景，包括三种服务（xlite_decode_only、xlite_full、aclgraph），并生成对比报告
cd ./tests/e2e/
bash online_server_compare.sh [MODEL_TYPE]

# 参数说明：
# - MODEL_TYPE: 模型类型（可选，默认moe）
#   - dense: 测试 Qwen3-32B 模型（3个场景）
#   - moe: 测试 Qwen3-30B-A3B-Instruct-2507 模型（4个场景，默认）
#   - all: 测试所有模型

# 测试场景说明：
# - 场景1：输入512 token，输出512 token，并发数：1 16 32 48 64 96
# - 场景2：输入3584 token，输出1536 token，并发数：1 16 32 48 64
# - 场景3：输入8192 token，输出1024 token，并发数：1 16 32 48 64
# - 场景4：输入102400 token，输出10240 token，并发数：1 16（仅moe模型）

# 输出结果：
# - 所有测试结果保存在 benchmark_results_<timestamp> 目录
# - 自动生成对比报告，文件名格式：benchmark_comparison_<model_name>_input<in>_output<out>_tp<tp>_<date>.log

# 方式方法2：手动解析性能对比数据
# 入参为aclgraph、xlite_full和xlite_decode_only的压测结果保存路径
cd ./tests/e2e/
python process_data.py <aclgraph_dir> <xlite_full_dir> <xlite_decode_only_dir> -o <output_file> -m <model_name>

# 参数说明：
# - aclgraph_dir: aclgraph结果目录（必需）
# - xlite_full_dir: xlite_full结果目录（必需）
# - xlite_decode_only_dir: xlite_decode_only结果目录（必需）
# - -o/--output: 输出文件路径（可选，默认./benchmark_comparison.log）
# - -m/--model: 模型名称（可选，默认"Qwen3 32B"）

# 示例：
python process_data.py ./result_input_512_output_512_aclgraph ./result_input_512_output_512_xlite_full ./result_input_512_output_512_xlite_decode_only -o ./benchmark_comparison.log -m "Qwen3 32B"
```