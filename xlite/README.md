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
环境：Python 3.9~3.11，1台Atlas 800I A2推理服务器，并从huggingface下载好模型参数
1. **创建容器**
```bash
# 创建arm容器
docker run --name xlite -it --rm --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /var/log/npu:/usr/slog -v /mnt/nvme0n1:/mnt/nvme0n1 -v /home:/home --net=host hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20251219 /bin/bash

# 创建x86容器
docker run --name xlite -it --rm --privileged -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons -v /var/log/npu:/usr/slog -v /mnt/nvme0n1:/mnt/nvme0n1 -v /home:/home --net=host hub.oepkgs.net/oedeploy/openeuler/x86_64/gvirt:20251219 /bin/bash

# 进入容器后，安装xlite；可使用pip安装，也可使用源码安装（开发环境推荐后者）
pip install xlite # 安装最新发布的版本

# 或者使用源码编译安装最新的开发版本
git clone https://atomgit.com/openeuler/GVirt.git
cd GVirt/xlite
pip install . --no-build-isolation # 安装当前目录下的xlite包
# 若开发环境安装，建议使用"-v .[dev]"（py源码修改后可直接在开发环境中生效）
pip install -v -e .[dev] --no-build-isolation
```
该容器可用于编译和运行xlite，详细镜像见下表：

| 硬件型号 | CPU架构 | 容器镜像 | Dockerfile |
|---------|---------|---------|-----------|
| Atlas 800I/T A2 | aarch64 | hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20251219 | [openeuler_torch_ascend_arm.Dockerfile](docker/openeuler_torch_ascend_arm.Dockerfile) |
| Atlas 800I/T A2 | x86_64 | hub.oepkgs.net/oedeploy/openeuler/x86_64/gvirt:20251219 | [openeuler_torch_ascend_x86.Dockerfile](docker/openeuler_torch_ascend_x86.Dockerfile) |
| Atlas 800I/T A3 | aarch64 | hub.oepkgs.net/oedeploy/openeuler/aarch64/gvirt:20260324 | [openeuler_torch_ascend_a3_arm.Dockerfile](docker/openeuler_torch_ascend_a3_arm.Dockerfile) |
| Atlas 800I/T A3 | x86_64 | hub.oepkgs.net/oedeploy/openeuler/x86_64/gvirt:20260324 | [openeuler_torch_ascend_a3_x86.Dockerfile](docker/openeuler_torch_ascend_a3_x86.Dockerfile) |


2. **运行推理**
```bash
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

#### vllm + vllm_ascend + xlite 在线服务
1. **安装xlite**
```bash
# 安装vllm_ascend, 可参考https://github.com/vllm-project/vllm-ascend/blob/main/README.md

# 安装xlite
pip install xlite
```

2. **离线示例**
```bash
import os
from vllm import LLM

# xlite默认支持decode-only模式, 可通过设置 "full_mode": True 使能full模式
model = LLM(model="path/to/Qwen3-32B", tensor_parallel_size=8, additional_config={"xlite_graph_config": {"enabled": True, "full_mode": True}})
outputs = model.generate("Hello, how are you?")
```

3. **在线示例**
```bash
vllm serve path/to/Qwen3-32B --tensor-parallel-size 8 --additional-config='{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

4. **性能测试**

vllm_ascend + xlite在线服务的性能测试及性能对比分析，请参考[e2e_test.md](doc/e2e_test.md)

构建/编译/分发准备过程请参考[创建容器](#快速上手qwen3-dense模型为例)部分。需提前安装构建依赖：
```bash
...
cd GVirt/xlite
pip install -r requirements-build.txt
pip install -r requirements-dev.txt # 进一步安装开发依赖（可选）
```

#### 编译
```bash
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
```bash
# 切换到xlite目录下，执行以下命令准备rpm构建环境
mkdir -p /root/rpmbuild/{BUILD,BUILDROOT,RPMS,SOURCES,SOPES,SPECS,SRPMS}
# 拷贝源码至/root/rpmbuild/SOURCES/xlite-${VERSION}.tar.gz，执行以下命令
VERSION=0.1.0 # 替换为当前版本号
git archive --format=tar.gz --prefix=xlite-${VERSION}/ -o /root/rpmbuild/SOURCES/xlite-${VERSION}.tar.gz HEAD
cp xlite.spec /root/rpmbuild/SPECS/
cd /root/rpmbuild/SPECS
rpmbuild -bb xlite.spec --nodebuginfo
```
构建后生成的rpm包在/root/rpmbuild/RPMS/目录下
##### 方法2：whl
```bash
python -m build --wheel --no-isolation # 构建whl包；如需.tar.gz包可去掉--wheel参数
```
推荐使用上面命令，完整遵循`pyproject.toml`中的构建配置（含`[build-system]`）。
也可以使用下面的方法通过`setup.py`进行编译和whl包构建（传统方式，可能不会自动安装`[build-system]`依赖；构建后执行清理）：
```bash
python setup.py bdist_wheel && python setup.py clean # 构建whl包并清理构建产物
```
构建后生成的whl包在dist目录下

#### 代码提交
代码提交前请本地容器环境内执行格式检查
代码检查及修正方法参考：[静态检查说明](doc/static_checker.md)

#### 目录结构
csrc：轻量化运行时的核心代码

xlite：python代码，包括tools等

doc：相关文档介绍

docker：容器镜像Dockerfile

tests：测试用例
