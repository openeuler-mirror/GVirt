<div align="center">
  <img src="doc/images/xlite_logo.png" alt="xlite logo">
</div>

<h1 align="center">Xlite Lightweight Inference Runtime</h1>

## Introduction

xlite (GVirt Frontend): A lightweight Transformer model runtime that supports diverse computing power collaboration, currently supporting efficient execution on Ascend hardware. xlite exposes the model graph construction and operators required for Transformer runtime, with all operators developed based on Ascend AscendC/CCE.


## Background and Motivation

In large model inference scenarios, the traditional single-stream serial execution mode has the following problems:

- **Inter-core Load Imbalance**: Uneven task distribution among different AICOREs, leaving some cores idle
- **High Resource Waste**: Low utilization of computing and transmission resources, with significant waste
- **Long Execution Time**: High overhead from Host CPU dispatching operators, causing serious host bond issues

xlite addresses the above problems through the following technical approaches:

- **Multi-stream Parallelism**: Fully utilizes on-card resources by changing from single-stream serial to multi-stream parallel execution
- **Inter-core Load Balancing**: Balances load across cores, improving resource utilization
- **CPU-NPU Collaboration**: Completely eliminates Python GC and threading interference on the C++ side, simplifies Host tiling computation, removes small memory allocations/deallocations and copies, eliminating Host bond

Significant performance improvements:

In the GLM-4.7 dual-node inference scenario (40K input, 1K output, ~90% prefix cache hit rate):

- TPOT latency reduced by 17%~30%
- Throughput improved by 13%~41%

For detailed performance data, refer to [PR #7935](https://github.com/vllm-project/vllm-ascend/pull/7935).



## Software Architecture

![image](doc/images/architecture.png)

xlite has been adapted for vllm_ascend and can be quickly enabled through xlite_graph_config configuration. For usage instructions, refer to the [official guide documentation](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html).

For supported models, see [Model List](doc/models.md).


## Quick Start

### Installation

```bash
# Install vllm_ascend, refer to https://github.com/vllm-project/vllm-ascend/blob/main/README.md
# Install xlite
pip install xlite --extra-index https://download.pytorch.org/whl/cpu/
```

### Offline Inference Example

```python
import os
from vllm import LLM

# xlite supports decode-only mode by default, enable full mode by setting "full_mode": True
model = LLM(model="path/to/Qwen3-32B", tensor_parallel_size=8, additional_config={"xlite_graph_config": {"enabled": True, "full_mode": True}})
outputs = model.generate("Hello, how are you?")
```

### Online Service Example

```bash
vllm serve path/to/Qwen3-32B --tensor-parallel-size 8 --additional-config='{"xlite_graph_config": {"enabled": true, "full_mode": true}}'
```

## Development Guide

For build and compilation, container images, source installation and other development-related content, please refer to [Development Guide](doc/contributing.md)

## Directory Structure

```
.
├── csrc/    - Core code of lightweight runtime
├── xlite/   - Python code, including tools, etc.
├── doc/     - Related documentation
├── docker/  - Container image Dockerfiles
└── tests/   - Test cases
```

## Acknowledgments

The initial versions and ideas for the core operators of this project come from contributions by the Huawei Terminal Xiaoyi AI Infra team. For related optimization implementations, please refer to the paper:

["XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM Workloads" [ASPLOS 2026]](https://arxiv.org/abs/2412.18106)