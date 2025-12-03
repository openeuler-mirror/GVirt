# Xlite轻量化推理运行时

#### 介绍
xlite (GVirt前端)：轻量级Transformer模型运行时，支持多样性算力协同，当前支持在昇腾硬件上高效运行。
xlite公开了Transformer运行所需的模型构图以及算子，所有算子基于昇腾AscendC开发。
目前支持Qwen系列、Llama系列、DeepSeek-R1模型。

本项目的核心算子初始版本和思路来自于华为终端小艺AI Infra团队的贡献，相关优化实现可参考论文：

[《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads》 [ASPLOS 2026]](https://arxiv.org/abs/2412.18106)


#### 软件架构
![image](doc/images/architecture.png)

xlite可作为pytorch、mindspore后端，也可以作为vllm的platform plugin直接接入vllm；xlite当前支持昇腾硬件。

#### 性能
待刷新

#### 快速上手（Qwen3 dense模型为例）
环境：1台Atlas 800I A2推理服务器，并从huggingface下载好模型参数
1. **创建容器**
```
docker run --name xlite -it --rm --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /var/log/npu:/usr/slog -v /mnt/nvme0n1:/mnt/nvme0n1 -v /home:/home --net=host hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20250530 /bin/bash
# 安装依赖
pip install -r requirements-dev.txt
```
该容器可用于编译和运行xlite。当前容器使用[openeuler_torch_ascend_arm.Dockerfile](docker/openeuler_torch_ascend_arm.Dockerfile)创建。

2. **运行推理**
```
export FORWARD_BACKEND=xlite
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
    "dtype": "float16",
    "max_batch_size": 8,
    "max_seq_len": 1024
}' > tests/test_config.json
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 tests/generate.py --model qwen3 --ckpt-path /mnt/nvme0n1/models/Qwen3-32B/ --config tests/test_config.json --interactive
```

#### 编译
```
# 准备
rm -rf build && mkdir -p build
# 编译
cmake -B build && cmake --build build -j
# 安装
cmake --install build
# 测试验证：可使用算子测试，也可使用完整模型测试
python tests/kernels/add.py
```

#### 构建安装包
当前支持rpm和whl，可选择合适的方式构建出包，用于不同场景的二进制发布和安装部署，开发场景可忽略。
##### 方法1：rpm
```
# 拷贝源码至/root/rpmbuild/SOURCES/xlite-1.0.tar.gz，执行以下命令
rpmbuild -bb xlite.spec --nodebuginfo
```
构建后生成的rpm包在/root/rpmbuild/RPMS/目录下
##### 方法2：whl
```
python setup.py bdist_wheel && python setup.py clean
```
或者使用下面的方法直接进行编译和whl包的构建
```
python -m build
```
构建后生成的whl包在dist目录下


#### 目录结构
csrc：轻量化运行时的核心代码

xlite：python代码，包括tools等

doc：相关文档介绍

docker：容器镜像Dockerfile

tests：测试用例
