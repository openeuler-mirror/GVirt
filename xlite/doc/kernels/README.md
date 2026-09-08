# xlite 算子实现原理文档

本目录包含 [csrc/kernels](../../csrc/kernels) 下所有 NPU 算子的实现原理说明文档。

每份文档包含:

- **功能概述**:算子的数学语义与用途
- **输入输出参数**:各参数的方向、Shape、Dtype 与含义
- **支持的数据类型**:实例化的 dtype 变体及对应源文件
- **实现原理**:数据流、UB 内存布局、tiling 与多 Block 并行策略、流水线同步、关键向量指令与边界处理

## 算子列表

### 基础算子

| 算子 | 文档 | 说明 |
|---|---|---|
| add | [add.md](add.md) | 逐元素矩阵加法 `z = x + y` |
| add_bias | [add_bias.md](add_bias.md) | 逐元素加 bias |
| muls | [muls.md](muls.md) | 逐元素标量乘 |
| silu_and_mul | [silu_and_mul.md](silu_and_mul.md) | SiLU 激活与门控乘融合 |
| sigmoid_gate_mul | [sigmoid_gate_mul.md](sigmoid_gate_mul.md) | Sigmoid 门控乘融合 |
| cast | [cast.md](cast.md) | dtype 类型转换 |
| softmax | [softmax.md](softmax.md) | Softmax |
| concat | [concat.md](concat.md) | 按指定维度拼接 |
| split | [split.md](split.md) | 按指定维度切分 |
| transpose_1_2 | [transpose_1_2.md](transpose_1_2.md) | 交换维度 1 与 2 |

### 归一化与量化

| 算子 | 文档 | 说明 |
|---|---|---|
| norm | [norm.md](norm.md) | RMSNorm / LayerNorm / L2Norm |
| qk_rms_norm | [qk_rms_norm.md](qk_rms_norm.md) | Q、K 联合 RMSNorm |
| quant | [quant.md](quant.md) | 量化 |
| quant_dyn | [quant_dyn.md](quant_dyn.md) | 动态量化 |
| dequant | [dequant.md](dequant.md) | 反量化 |
| msd_merge_dequant | [msd_merge_dequant.md](msd_merge_dequant.md) | MSD 多份合并反量化 |
| unpack_activation | [unpack_activation.md](unpack_activation.md) | 激活值解包(int8) |
| beta_decay | [beta_decay.md](beta_decay.md) | beta 衰减 |

### 矩阵乘

| 算子 | 文档 | 说明 |
|---|---|---|
| matmul | [matmul.md](matmul.md) | 矩阵乘(含 int8/int4 量化变体) |
| group_matmul | [group_matmul.md](group_matmul.md) | 分组矩阵乘(MoE) |
| conv1d_and_silu | [conv1d_and_silu.md](conv1d_and_silu.md) | 1D 卷积与 SiLU 融合 |
| conv1d_and_silu_token | [conv1d_and_silu_token.md](conv1d_and_silu_token.md) | token 维 1D 卷积与 SiLU 融合 |

### 注意力

| 算子 | 文档 | 说明 |
|---|---|---|
| attention | [attention.md](attention.md) | Decode 注意力 |
| flash_attention | [flash_attention.md](flash_attention.md) | Flash Attention(Prefill) |
| mla_v2 | [mla_v2.md](mla_v2.md) | MLA V2 |
| flash_mla_v2 | [flash_mla_v2.md](flash_mla_v2.md) | Flash MLA V2 |
| mla_prepare | [mla_prepare.md](mla_prepare.md) | MLA 前处理 |
| gather_sparse_kv_cache | [gather_sparse_kv_cache.md](gather_sparse_kv_cache.md) | 稀疏 KV Cache 收集 |
| indexer_prepare | [indexer_prepare.md](indexer_prepare.md) | Indexer 前处理(DSA 稀疏注意力) |
| indexer_scores | [indexer_scores.md](indexer_scores.md) | Indexer 分数计算 |
| indexer_topk | [indexer_topk.md](indexer_topk.md) | Indexer TopK 选路 |
| cxa | [cxa.md](cxa.md) | C4A/C128A 滑窗+压缩稀疏注意力(DeepSeek-V4) |
| einsum_mht_hdt_mhd | [einsum_mht_hdt_mhd.md](einsum_mht_hdt_mhd.md) | 线性注意力 einsum |
| einsum_mht_htd_mhd | [einsum_mht_htd_mhd.md](einsum_mht_htd_mhd.md) | 线性注意力 einsum |
| recurrent_gated_delta_rule | [recurrent_gated_delta_rule.md](recurrent_gated_delta_rule.md) | 门控 Delta Rule 循环核 |

### 位置编码与 embedding

| 算子 | 文档 | 说明 |
|---|---|---|
| rope_and_cache | [rope_and_cache.md](rope_and_cache.md) | RoPE + KV Cache 写入 |
| rope_complex_and_cache | [rope_complex_and_cache.md](rope_complex_and_cache.md) | 复数形式 RoPE + KV Cache 写入 |
| embed_kernel | [embed_kernel.md](embed_kernel.md) | Embedding 查表 |

### TopK

| 算子 | 文档 | 说明 |
|---|---|---|
| topk | [topk.md](topk.md) | TopK 选取 |
| softmax_topk | [softmax_topk.md](softmax_topk.md) | Softmax + TopK 融合 |
| sigmoid_topk | [sigmoid_topk.md](sigmoid_topk.md) | Sigmoid + TopK 融合 |
| sqrtsoftplus_hash_topk | [sqrtsoftplus_hash_topk.md](sqrtsoftplus_hash_topk.md) | SqrtSoftplus + Hash TopK |

### MoE 与重排

| 算子 | 文档 | 说明 |
|---|---|---|
| permutation | [permutation.md](permutation.md) | Token 按索引重排 |
| unpermutation | [unpermutation.md](unpermutation.md) | Token 逆重排 |
| reorder_moe | [reorder_moe.md](reorder_moe.md) | MoE Token 重排 |
| repeat_interleave | [repeat_interleave.md](repeat_interleave.md) | 重复扩展 |
| experts_counts_sum | [experts_counts_sum.md](experts_counts_sum.md) | 专家计数求和 |

### 通信

| 算子 | 文档 | 说明 |
|---|---|---|
| all_gather | [all_gather.md](all_gather.md) | IPC 直通 AllGather |
| all_reduce | [all_reduce.md](all_reduce.md) | IPC 直通 AllReduce |
| reduce_scatter | [reduce_scatter.md](reduce_scatter.md) | IPC 直通 ReduceScatter |
| ring_sync | [ring_sync.md](ring_sync.md) | 通信 ring 同步辅助 |

### 模型专属融合算子

| 算子 | 文档 | 说明 |
|---|---|---|
| hc_act | [hc_act.md](hc_act.md) | HC 模型激活融合 |
| hc_post | [hc_post.md](hc_post.md) | HC 模型后处理融合 |

## 目录约定

- 算子实现在 `csrc/kernels/<name>.h`(模板)或 `csrc/kernels/<name>.cpp`
- dtype 实例化在 `csrc/kernels/<name>_<dtype>.cpp`,文件名即对应支持的 dtype
- 算子正确性测试在 [tests/kernels](../../tests/kernels)
- host 侧启动与参数封装在 `csrc/op.cpp`、`csrc/ccl.cpp`
