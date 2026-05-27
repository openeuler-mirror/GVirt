# vllm_ascend + xlite 在线服务性能测试及对比

## 快速开始

```bash
# 安装vllm_ascend, 可参考https://github.com/vllm-project/vllm-ascend/blob/main/README.md

# 安装xlite
pip install xlite
```

## 启动在线服务

```bash
# xlite可选择decode only或full其中一种模式启动在线服务：
# a. 配置xlite decode only模式，设置--additional-config='{"xlite_graph_config": {"enabled": true}}'
cd ./tests/e2e/
bash online_api_server.sh 8080 server.log [model_path] [max_num_batched_tokens] [max_num_seqs] [max_model_len] [tensor_parallel_size] "xlite_decode_only"

# b. 配置xlite full模式，设置--additional-config='{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
cd ./tests/e2e/
bash online_api_server.sh 8080 server.log [model_path] [max_num_batched_tokens] [max_num_seqs] [max_model_len] [tensor_parallel_size] "xlite_full_mode"

# c. 配置aclgraph模式（不使用xlite）
cd ./tests/e2e/
bash online_api_server.sh 8080 server.log [model_path] [max_num_batched_tokens] [max_num_seqs] [max_model_len] [tensor_parallel_size] "aclgraph"

# 参数说明：
# - port: 服务端口（必需）
# - log: 日志文件路径（必需）
# - model_path: 模型路径（可选，默认/mnt/nvme0n1/models/Qwen3Qwen3-32B）
# - max_num_batched_tokens: 最大批处理token数（可选，默认8192）
# - max_num_seqs: 最大序列数（可选，默认200）
# - max_model_len: 最大模型长度（可选，默认6656）
# - tensor_parallel_size: 张量并行大小（可选，默认8）
```

## 启动在线压测

```bash
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

## 性能对比

```bash
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

## 精度测试

```bash
# 运行精度测试脚本
# 以下脚本会比较（Qwen3-30B-A3B, TP4EP4, xlite decode-only）和（Qwen3-30B-A3B, TP4EP4, aclgraph）两种配置在ceval数据集上的输出差异
# 通过修改(--models, --tp-sizes, --ep-sizes, --xlite, --xlite-full)参数，可以测试不同配置的性能和输出差异
cd tests/e2e/
python batch_aisbench.py --help # 查看参数说明
python batch_aisbench.py "/workspace/benchmark" \
--num-prompts 64 \
--model-dir "/mnt/sdb/models" \
--models "Qwen3-30B-A3B" "Qwen3-30B-A3B" \
--tp-sizes 4 4 \
--ep-sizes 1 1 \
--xlite 1 0 \
--xlite-full 0 0
```

## 每日自动化测试机器人

每日自动化测试机器人用于定时执行性能测试、对比版本差异、检测性能劣化并发送通知。

### 容器准备

需要准备两个容器：编译容器和测试容器。

**启动编译容器 (xlite-build):**

```bash
docker run -itd --shm-size=10.24gb --net=host --privileged --cap-add=SYS_PTRACE --user root \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common:ro \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  --name xlite-build \
  -v /tmp:/tmp -v /home:/home -v /mnt/sdb/:/mnt/sdb \
  hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20260324 \
  /bin/bash -c "while true;do echo hello;sleep 5;done"
```

### 启动测试容器 (daily-test)

```bash
docker run -itd --shm-size=10.24gb --net=host --privileged --cap-add=SYS_PTRACE --user root \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/common:ro \
  -v /usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /etc/vnpu.cfg:/etc/vnpu.cfg:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /usr/bin/hccn_tool:/usr/bin/hccn_tool \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  --name daily-test \
  -v /tmp:/tmp -v /home:/home -v /mnt/sdb/:/mnt/sdb \
  -v /var/run/docker.sock:/var/run/docker.sock \
  quay.io/ascend/vllm-ascend:v0.17.0rc1-a3 \
  /bin/bash -c "while true;do echo hello;sleep 5;done"
```

### 安装定时任务

使用 Python 原生的 `schedule` 库实现定时任务，适合容器环境和宿主机环境：

**安装依赖：**

```bash
pip install schedule
```

**运行调度器：**

```bash
cd ./tests/e2e/

# 基本用法（每天晚上 8:00 执行）
python3 run_benchmark_scheduler.py

# 自定义执行时间（每天凌晨 2:00）
python3 run_benchmark_scheduler.py --run-time 02:00

# 立即执行一次，然后启动调度器
python3 run_benchmark_scheduler.py --run-now

# 自定义所有参数
python3 run_benchmark_scheduler.py \
    --model moe \
    --receiver "1234567890" \
    --build-container xlite-build \
    --run-time 02:00
```

**后台运行：**

```bash
# 使用 nohup 后台运行
nohup python3 run_benchmark_scheduler.py > scheduler.log 2>&1 &

# 记录 PID
echo $! > .scheduler.pid

# 停止调度器
kill $(cat .scheduler.pid)
```

### 手动执行测试

除了定时任务，也可以手动执行测试：

```bash
cd ./tests/e2e/
python3 daily_benchmark_bot.py --model moe --receiver 1234567890
```

**可选参数：**

- `--model`: 测试模型类型（dense/moe/all），默认 moe
- `--skip-pull`: 跳过代码拉取
- `--skip-build`: 跳过编译
- `--receiver`: 接收者ID，welink 群号
- `--threshold`: 性能劣化阈值，默认 0.05（即5%）
- `--build-container`: 编译容器名称
- `--env-type`: 环境类型（blue/yellow），默认 yellow
  - yellow 环境：需要执行 source /home/env.sh
  - blue 环境：不需要 source /home/env.sh

### 测试流程

自动化测试机器人执行以下流程：

1. **代码拉取**：在编译容器和测试容器中从远程仓库拉取最新代码
2. **项目编译**：在编译容器中编译项目生成 wheel 包
3. **安装部署**：在本地安装 wheel 包
4. **基准测试**：在测试容器中执行基准测试脚本
5. **报告解析**：解析测试报告并与上一版本的数据对比
6. **性能检测**：检测性能劣化（超过阈值时告警）
7. **通知发送**：发送测试结果到群组

### 性能对比规则

- **劣化判断**：
  - QPS 和 Output Speed：下降超过阈值视为劣化
  - TTFT 和 TPOT：上升超过阈值视为劣化（延迟越高越差）
- **阈值设置**：默认为 5%，可通过 `--threshold` 参数调整
- **对比基线**：自动查找同版本号的基线测试报告进行对比

### 输出结果

测试结果保存在 `/home/daily_reports/` 目录下：

- **报告目录结构**：
  - `xlite-{version}-{date}/`：当前版本测试结果（带日期）
  - `xlite-{version}/`：基线版本测试结果（不带日期）

- **报告文件**：
  - `daily_benchmark_YYYYMMDD_HHMMSS.log`：测试执行日志（包含所有输出信息）
  - `benchmark_comparison_*.log`：性能对比报告
  - `metrics_*.json`：指标数据（JSON格式）
  - `comparison_*.json`：对比结果（JSON格式）
  - `daily_summary_*.txt`：每日测试摘要

**日志说明**：

- 测试执行日志会同时输出到终端和文件
- 日志文件命名格式：`daily_benchmark_YYYYMMDD_HHMMSS.log`
- 日志文件保存在版本报告目录中：`/home/daily_reports/xlite-{version}-{date}/daily_benchmark_YYYYMMDD_HHMMSS.log`
- 日志文件与测试报告（benchmark_comparison_*.log 等）保存在同一目录

### 常用命令

**调度器管理：**
```bash
# 启动调度器（前台运行）
python3 run_benchmark_scheduler.py

# 启动调度器（后台运行）
nohup python3 run_benchmark_scheduler.py > scheduler.log 2>&1 &

# 查看调度器日志
tail -f scheduler.log

# 停止调度器
ps aux | grep run_benchmark_scheduler.py
kill <PID>
```

**手动执行测试：**

```bash
python3 daily_benchmark_bot.py --model moe --receiver "927280411401503971"
```

### NPU-SMI 版本检测

脚本会自动检测 NPU-SMI 版本并决定是否禁用 XCCL：

- **Version >= 25.3**：不禁用 XCCL（新版本已修复相关问题）
- **Version < 25.3**：禁用 XCCL（旧版本需要禁用）
- **无法获取版本**：默认禁用 XCCL

### 退出码

- `0`：成功，无性能劣化
- `1`：执行失败
- `2`：成功，但检测到性能劣化
