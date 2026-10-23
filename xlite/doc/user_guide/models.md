# 支持模型

xlite 支持以下大语言模型的推理部署：

| 模型 | 支持状态 |
|------|---------|
| Qwen3-32B | ✅ |
| Qwen3-30B-A3B | ✅ |
| Qwen3-235B-A22B | ✅ |
| Qwen3.5-35B-A3B | ✅ |
| Qwen3.5-122B-A10B | ✅ |
| Qwen3.6-35B-A3B | ✅ |
| Qwen3.8-27B | ✅ |
| DeepSeek-V3/3.1/R1 | ✅ |
| DeepSeek-V3.2 | ✅ |
| DeepSeek-V4 | (规划中) |
| DeepSeek-V4.1 | (规划中) |
| GLM-4.7 | ✅ |
| GLM-4.7-Flash | ✅ |
| GLM-5/5.1 | ✅ |
| GLM-5.2 | ✅ |
| GLM-5.3 | ✅ |
| GLM-5.3-Flash | (规划中) |
| MiniMax-M2.5/2.7 | ✅ |
| Qwen3-VL-8B | ✅ |
| Qwen3-VL-30B-A3B | ✅ |
| Qwen3-VL-235B-A22B | ✅ |


## 支持硬件

- Ascend A2
- Ascend A3
## Qwen3.5 原生分页 GDN 调用契约

Conv state 使用 `[P,W-1,C]`，SSM state 使用 `[P,Hv,V,K]`，保留页零作为 null。
AttnMeta 和 AttnMetaV2 的 `gdn_conv_indices` / `gdn_ssm_indices` 为逐层设备 INT32 页号列表；
full-attention 层放空 tensor。Conv 与 SSM 页面由调用者独立分配和映射，禁止将 batch 行号
直接当作生产环境页号。初次请求及多 token 调用还必须提供逐层 `gdn_query_starts[B+1]`、
`gdn_conv_initial[B]`、`gdn_ssm_initial[B]`。前缀和须对应 packed query 长度；
initial=0 表示从零状态开始，非零表示使用当前页，空段不写页。

旧 dense Conv、token Conv 和 recurrent 算子接口已删除。测试模型的 helper 只负责
固定 batch 的演示页面映射；生产调度器负责请求生命周期、页面恢复及 metadata 正确性。
