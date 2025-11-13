环境要求：DeepSeek-R1 671B bf16 需要4台Atlas 800I A2推理服务器运行，路由专家int8量化后需要2台Atlas 800I A2推理服务器运行。（每台服务器拥有8张Ascend910B3卡）

注：如果你只有一台Atlas 800I A2推理服务器，可以通过减少模型层数运行。例如仅运行8层参数，则在convert.py命令行中增加“--n-layers 8”入参，并对应修改tests/deepseek_config_671B.json中的n_layers为8。注意：这种方法只能用于快速测试验证，并不能获得完整的模型输出。

1. **创建容器**
```
docker run --name xlite -it --rm --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /var/log/npu:/usr/slog -v /mnt/nvme0n1:/mnt/nvme0n1 -v /home:/home --net=host hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20250530 /bin/bash
# 安装依赖
pip install -r requirements.txt
```
该容器可用于编译和运行xlite。当前容器使用[openeuler_torch_ascend_arm.Dockerfile](docker/openeuler_torch_ascend_arm.Dockerfile)创建。

2. **准备模型参数** （一台服务器上运行）
```
# 1）从huggingface下载原始模型，并完成参数转换和切分。模型下载地址：https://huggingface.co/deepseek-ai/DeepSeek-R1

# 2）参数类型转换
# （选择一）：模型参数从fp8转换为bf16
export dtype=bf16
python xlite/tools/fp8_cast_bf16.py --input-fp8-hf-path /mnt/nvme0n1/models/deepseek-R1 --output-bf16-hf-path /mnt/nvme0n1/models/deepseek-R1-${dtype}

# （选择二）：模型参数从fp8转换为int8（仅量化路由专家，其他参数转化为bf16）
export dtype=expert-int8
bash xlite/tools/w8a8.sh /mnt/nvme0n1/models/deepseek-R1 /mnt/nvme0n1/models/deepseek-R1-${dtype}

# 3）模型按照并行策略进行切分，model-parallel代表在几张卡上运行，默认MOE路由专家按照EP方式切分，其他参数按照TP方式切分
python xlite/tools/convert.py --hf-ckpt-path /mnt/nvme0n1/models/deepseek-R1-${dtype} --save-path /mnt/nvme0n1/models/deepseek-R1-${dtype}-61layers-${num_npu}d --model-parallel ${num_npu}

# 4）拷贝参数到每个服务器
scp -r /mnt/nvme0n1/models/deepseek-R1-${dtype}-61layers-${num_npu}d x.x.x.x:/mnt/nvme0n1/models/
```

3. **运行DeepSeek-R1推理**
```
# 1) 切换为xlite运行时（可选）
# 1.1） 源码编译，参考"编译"章节

# 1.2） 切换xlite运行时
export FORWARD_BACKEND=xlite
export XLITE_NODE_IPS="ip0,ip1"
# 注1：XLITE_NODE_IPS用于配置多台服务器的ip列表，使用","作为分隔符，要求按node顺序排列，即与下一步命令中的node_rank对应。

# 2) 运行DeepSeek-R1推理
torchrun --nproc_per_node=8 --nnodes=${nnodes} --node_rank=${node} --master_addr=x.x.x.x tests/generate.py --ckpt-path /mnt/nvme0n1/models/deepseek-R1-${dtype}-61layers-${num_npu}d --config tests/deepseek_config_671B.json --interactive
# 注1：nproc_per_node：每个服务器上的进程数; nnodes：共几个服务器; node_rank：第一台服务器的node为0，第二台服务器的node为1，以此类推; master_addr：配置为0号node的ip。
# 注2：用户如使用int8模型，需将config文件tests/deepseek_config_671B.json中参数"quantization"的值修改为"experts_int8"。
# 注3：用户如使用8层模型，需将config文件tests/deepseek_config_671B.json中参数"n_layers"的值修改为"8"。
```