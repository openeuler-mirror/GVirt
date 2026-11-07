# 模型架构流图

本目录（`doc/developer_guide/models/`）按**架构系列**存放 xlite 支持的各模型 forward 流程图：同一系列（相同的 attention / FFN / MoE 结构与 C++ 路径）共用一张图，系列内模型仅在规模（层数、专家数、hiddenSize）上不同。系列以各模型 config.json 的 `architectures` 字段标识（如 GLM-5.2 为 `GlmMoeDsaForCausalLM`）。每张图一个 `.md`，含 mermaid 图源、shape 记号表、真实模型 config 取值举例与关键分支说明。

- 图示规范（配色 / 形状 / 箭头 / 记号 / 提交 checklist）：[doc/conventions/diagram_conventions.md](../../conventions/diagram_conventions.md)。
  > 原 `doc/flow/`（现 `doc/conventions/`）下的架构 forward 文档已全部迁移至本目录并删除；该目录仅存放文档规范，本链接持续有效。
- 各模型支持状态总表：[doc/user_guide/models.md](../../user_guide/models.md)。

## 架构系列索引

| architectures（config.json） | 文档 | 覆盖模型 | 架构要点 | 状态 |
|------------------------------|------|----------|----------|------|
| `DeepseekV3ForCausalLM` | [DeepseekV3ForCausalLM.md](DeepseekV3ForCausalLM.md) | DeepSeek-V3/3.1/R1 | MLA 低秩 KV 缓存 + sigmoid 分组 MoE + EPLB + shared expert | ✅ 已完成 |
| `DeepseekV4ForCausalLM` | [DeepseekV4ForCausalLM.md](DeepseekV4ForCausalLM.md) | DeepSeek-V4（Flash，w8a8 + MTP） | MHC 多流主循环（`hcMult=4`）+ CXA 压缩稀疏注意力（`compress_ratios` / indexer topk）+ sqrtsoftplus·hash MoE + shared expert + w8a8 | ✅ 已完成 |
| `DeepseekV41ForCausalLM` | —— | DeepSeek-V4.1 | —— | 🗓 未实现（规划中） |
| `Glm4MoeForCausalLM`<br>`Glm4MoeLiteForCausalLM`<br>`GlmMoeDsaForCausalLM`<br>`GlmMoeDsaMtpForCausalLM` | [GlmMoeDsaForCausalLM.md](GlmMoeDsaForCausalLM.md) | GLM-5/5.1/5.2、GLM-5.2-MTP、GLM-5.3（非 Flash；`architectures` 与 5.2 相同，repo 无 checkpoint / 测试接入）、GLM-4.7（GQA MHA）、GLM-4.7-Flash（MLA，仅测试路径覆盖） | GLM-5 系：MLA + DSA indexer + sigmoid MoE + shared expert（w8a8/w4a8/bf16，EP），5.2 多 indexer full/shared 交错；GLM-4.7 系非同构——4.7 走 `ForwardAttnMHA`（见 [Qwen3ForCausalLM.md](Qwen3ForCausalLM.md) §2），4.7-Flash 走 MLA 无 indexer，两者共用 [GlmMoeDsaForCausalLM.md](GlmMoeDsaForCausalLM.md) §3.2 的 sigmoid MoE | ✅ 已完成（4.7 / 4.7-Flash 组件级覆盖） |
| `Glm5NextForConditionalGeneration` | —— | GLM-5.3-Flash | MHC 多流主循环 + KDA 线性注意力 + 稀疏 MLA | 🗓 未实现（规划中） |
| `MiniMaxM2ForCausalLM` | [MiniMaxM2ForCausalLM.md](MiniMaxM2ForCausalLM.md) | MiniMax-M2.5/2.7 | 全注意力（GQA + full-variance QK-norm）+ sigmoid MoE（e_score_correction_bias） | ✅ 已完成 |
| `Qwen3ForCausalLM`<br>`Qwen3MoeForCausalLM`<br>`Qwen3VLForConditionalGeneration`<br>`Qwen3VLMoeForConditionalGeneration` | [Qwen3ForCausalLM.md](Qwen3ForCausalLM.md) | Qwen3-32B、Qwen3-30B-A3B、Qwen3-235B-A22B、Qwen3-VL-8B/30B-A3B/235B-A22B（text 侧） | 标准 GQA + QK-norm + 全维 RoPE；dense / 全层 MoE（softmax 打分，无 shared expert）；VL 变体加 M-RoPE + DeepStack | ✅ 已完成 |
| `Qwen3_5ForConditionalGeneration`<br>`Qwen3_5MoeForConditionalGeneration` | [Qwen3_5ForConditionalGeneration.md](Qwen3_5ForConditionalGeneration.md) | Qwen3.5-35B-A3B、Qwen3.5-122B-A10B（dense 0.8B/27B 见文档）、Qwen3.6-35B-A3B、Qwen3.8-27B | MHA 与 Gated Delta Rule 线性注意力按层交错（interval=4）；输出门控；partial RoPE | ✅ 已完成 |

> 各行按 `architectures` 字段排序分组，同一字段值下的模型共用一张图；字段值取自各 checkpoint 的 config.json。系列归属经 vllm-ascend 适配器入口（`vllm_ascend/xlite/xlite.py` 的 `_supported_architectures`）与 C++ `ModelConfig` 分支核实。

## 新增 / 迁移一张图

1. 先读 [图示规范](../../conventions/diagram_conventions.md)，确认配色、形状、箭头与记号约定。
2. 复制本目录下最接近的现有图（推荐 [Qwen3ForCausalLM.md](Qwen3ForCausalLM.md) 作为模板）改写：架构概述 → 配置依据（[tests/run.sh](../../../tests/run.sh) 真实函数）→ 记号表与复合维度 → 分节流程图（各节图后配关键点文字）。
3. **分支取舍**：对**本系列恒走一条**的分支（如 Qwen3 的 `hcMult==0`、未配置 EPLB）不画菱形判断，其条件与取值在节点标签或图后关键点文字注明即可；菱形只保留三类有实际悬念的分支——**系列内变体差异**（dense / MoE）、**部署配置相关**（TP>1、量化路径）、**运行时动态**（attention tile 长短路径、循环边界）。
4. mermaid 图源直接写在 `.md` 代码块内，保持可 diff。

## 与其他文档的关系

- 各 step 内调用的**单算子**语义（输入输出、tiling、并行策略）见 [doc/developer_guide/kernels/](../kernels/README.md)，流图只引用算子名不展开算子内部。
- 模型加载与 config 转换（HF → `ModelConfig`）在 vllm-ascend 适配器中完成，流图以转换后的 `ModelConfig` 字段为准。
