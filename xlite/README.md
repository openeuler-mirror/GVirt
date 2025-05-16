# Xlite轻量化推理运行时

#### 介绍
xlite (GVirt前端)：轻量级DeepSeek-R1模型运行时，支持多样性算力协同，当前支持在昇腾硬件上高效运行。
xlite公开了DeepSeek-R1运行所需的模型构图以及算子，所有算子基于昇腾AscendC开发。

#### 软件架构
![image](doc/images/architecture.png)

xlite可作为pytorch、mindspore后端，也可以作为vllm的platform plugin直接接入vllm；xlite当前支持昇腾硬件。

#### 性能


#### 快速上手


#### 编译
```
bash build.sh
```

#### 目录结构
csrc: 轻量化运行时的核心代码
doc：相关文档介绍
docker：容器镜像Dockerfile
tests：测试用例