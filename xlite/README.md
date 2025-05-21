# Xlite轻量化推理运行时

#### 介绍
xlite (GVirt前端)：轻量级DeepSeek-R1模型运行时，支持多样性算力协同，当前支持在昇腾硬件上高效运行。
xlite公开了DeepSeek-R1运行所需的模型构图以及算子，所有算子基于昇腾AscendC开发。

#### 软件架构
![image](doc/images/architecture.png)

xlite可作为pytorch、mindspore后端，也可以作为vllm的platform plugin直接接入vllm；xlite当前支持昇腾硬件。

#### 性能
待刷新

#### 快速上手
DeepSeek-R1 671B bf16 需要4台推理服务器运行，每台服务器拥有8张Ascend910B3卡。

如果你只有一台Atlas 800I A2推理服务器，可以通过减少模型层数运行。例如仅运行8层参数，则在convert.py命令行中增加“--n-layers 8”入参，并对应修改tests/deepseek_config_671B.json中的n_layers为8。注意：这种方法只能用于快速测试验证，并不能获得完整的模型输出。

1. **创建容器**
```
docker run --name xlite -it --rm --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /var/log/npu:/usr/slog -v /mnt/nvme0n1:/mnt/nvme0n1 -v /home:/home --net=host xxx /bin/bash
```
该容器可用于编译和运行xlite。当前容器使用[openeuler_torch_ascend_arm.Dockerfile](docker/openeuler_torch_ascend_arm.Dockerfile)创建。

2. **准备模型参数** （一台服务器上运行）
```
# 从huggingface下载原始模型，并完成参数转换和切分。模型下载地址：https://huggingface.co/deepseek-ai/DeepSeek-R1

# （选择一）：模型参数从fp8转换为bf16
python tools/fp8_cast_bf16.py --input-fp8-hf-path /mnt/nvme0n1/models/deepseek-R1 --output-bf16-hf-path /mnt/nvme0n1/models/deepseek-R1-bf16

# （选择二）：模型参数从fp8转换为int8（仅量化MOE，其他参数转化为bf16）
bash tools/w8a8.sh <FP8_MODEL_PATH> <INT8_MODEL_PATH> <NUM_PROCESSES>
# 注1：<NUM_PROCESSES>表示并行使用NPU的数目。脚本会并行运行多个进程，每个进程各在一个NPU上执行量化。
# 注2: bf16与int8二选一即可。之后的例子默认用户使用bf16，用户如使用int8模型，只需按提示修改参数即可。

# 模型按照并行策略进行切分，model-parallel代表在几张卡上运行，默认MOE路由专家按照EP方式切分，其他参数按照TP方式切分
python tools/convert.py --hf-ckpt-path /mnt/nvme0n1/models/deepseek-R1-bf16 --save-path /mnt/nvme0n1/models/deepseek-R1-bf16-61layers-16d --model-parallel 16

# 拷贝参数到每个服务器
scp -r /mnt/nvme0n1/models/deepseek-R1-bf16-61layers-16d x.x.x.x:/mnt/nvme0n1/models/deepseek-R1-bf16-61layers-16d
```

3. **运行DeepSeek-R1推理**
```
# 切换为xlite运行时（待补充）

# nproc_per_node：每个服务器上的进程数; nnodes：共几个服务器; node_rank：第一台服务器的node为0，第二台服务器的node为1，以此类推; master_addr：配置为0号node的ip。
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=${node} --master_addr=x.x.x.x tests/deepseek_generate.py --ckpt-path /mnt/nvme0n1/models/deepseek-R1-bf16-61layers-16d --config tests/deepseek_config_671B.json --interactive
# 注：用户如使用int8模型，需将config参数改为"tests/deepseek_config_671B_experts_int8.json"
```

#### 编译


#### 目录结构
csrc：轻量化运行时的核心代码

doc：相关文档介绍

docker：容器镜像Dockerfile

tests：测试用例

tools：一些工具