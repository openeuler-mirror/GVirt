<div align="center">
  <img src="doc/images/xlite_logo.png" alt="xlite logo">
</div>

<h1 align="center">Xlite轻量化推理运行时</h1>

## 介绍

xlite (GVirt前端)：轻量级Transformer模型运行时，支持多样性算力协同，当前支持在昇腾硬件上高效运行。
xlite公开了Transformer运行所需的模型构图以及算子，所有算子基于昇腾AscendC/CCE开发。


## 背景与动机

在大模型推理场景中，传统的单流串行执行模式存在以下问题：

- **核间负载不均**：不同AICORE之间的任务分配不均衡，部分核心闲置
- **资源浪费多**：计算资源和传输资源利用率低，存在显著浪费
- **执行时间长**：Host CPU下发算子开销大，造成严重的host bond问题

xlite通过以下技术手段解决上述问题：

- **多流并行**：充分利用卡内资源，将单流串行改为多流并行执行
- **核间负载均衡**：核间负载均衡，提升资源利用率
- **CPU NPU协同**：C++侧完全消除Python的GC、线程等干扰，简化Host tiling计算，去除小块内存申请释放及拷贝，消除Host bond

性能效果显著：

在GLM-4.7双机推理场景（40K输入、1K输出、prefix cache命中率约90%）下：

- TPOT时延降低17%~30%
- 吞吐提升13%~41%

详细性能数据参考 [PR #7935](https://github.com/vllm-project/vllm-ascend/pull/7935)。



## 软件架构

![image](doc/images/architecture.png)

xlite已适配vllm_ascend，可通过xlite_graph_config配置快速使能xlite加速效果，使用方法参考[官方指导文档](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html)。

xlite支持模型见[模型列表](doc/models.md)。


## 快速开始

### 安装

```bash
# 安装vllm_ascend, 可参考https://github.com/vllm-project/vllm-ascend/blob/main/README.md
# 安装xlite
pip install xlite
```

### 离线推理示例

```python
import os
from vllm import LLM

# xlite默认支持decode-only模式, 可通过设置 "full_mode": True 使能full模式
model = LLM(model="path/to/Qwen3-32B", tensor_parallel_size=8, additional_config={"xlite_graph_config": {"enabled": True, "full_mode": True}})
outputs = model.generate("Hello, how are you?")
```

### 在线服务示例

```bash
vllm serve path/to/Qwen3-32B --tensor-parallel-size 8 --additional-config='{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

## 开发指南

编译构建、容器镜像、源码安装等开发相关内容，请参考 [开发指南](doc/contributing.md)

## 目录结构

```
.
├── csrc/    - 轻量化运行时的核心代码
├── xlite/   - python代码，包括tools等
├── doc/     - 相关文档介绍
├── docker/  - 容器镜像Dockerfile
└── tests/   - 测试用例
```

## 致谢

本项目的核心算子初始版本和思路来自于华为终端小艺AI Infra团队的贡献，相关优化实现可参考论文：

[《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads》 [ASPLOS 2026]](https://arxiv.org/abs/2412.18106)