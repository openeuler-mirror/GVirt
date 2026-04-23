# GVirt

#### Introduction
GVirt is a lightweight XPU virtualization frontend and backend inference runtime. It provides a minimalist and efficient heterogeneous computing environment, supporting diverse computing power collaboration.

#### Software Architecture
![image](xlite/doc/images/architecture.png)

- **Xlite (GVirt Frontend)**: A lightweight Transformer model runtime that supports diverse computing power collaboration, currently supporting efficient execution on Ascend hardware. Xlite exposes the model graph construction and operators required for Transformer runtime, with all operators developed based on Ascend AscendC/CCE. Currently supports Qwen series, Llama series, and DeepSeek-R1 models. See [Xlite README](xlite/README.md) for details.

#### Quick Start
Please refer to [Xlite Quick Start](xlite/README.md#快速开始)

#### Build and Installation
Please refer to [Xlite Development Guide](xlite/README.md#开发指南)

#### Contributing

1.  Fork this repository
2.  Create a new Feat_xxx branch
3.  Submit your code
4.  Create a Pull Request


#### Directory Structure

| Directory | Description |
|------|------|
| xlite | Lightweight inference runtime core code (Python) |
| xlite/csrc | Lightweight runtime core code (C++/AscendC) |
| xlite/doc | Related documentation |
| xlite/docker | Container image Dockerfiles |
| xlite/tests | Test cases |