# embed_kernel

## 功能概述

带词表切分的 embedding 查表:对每个 token id,若其落在本 rank 的词表区间 `[emb_start_idx, emb_end_idx)` 内,则从权重矩阵中取出对应行写入输出;否则输出全零行。配合 TP 部署时,各 rank 只持有 `vocab_size / tp_size` 行切片,区间外置零后由上层 `all_reduce` 求和得到完整 embedding([tests/kernels/embed.py:46-53](../../tests/kernels/embed.py#L46-L53)、[csrc/model.cpp:306-313](../../csrc/model.cpp#L306-L313) 的 `ForwardParallelEmbed`)。

## 输入输出参数

Python 接口:`embed(rt, weight, in_, out, start, end)`(`XliteOpEmbed`,[csrc/op.cpp:499-516](../../csrc/op.cpp#L499-L516))。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x(weight) | 输入 | `[emb_end_idx - emb_start_idx, dim]` | float16 / bfloat16 | 本 rank 的词表切片权重,行 `row - emb_start_idx` 对应全局 token id `row` |
| y(in) | 输入 | `[batch_size]` | int32 | token id 序列(全局词表编号) |
| z(out) | 输出 | `[batch_size, dim]` | 同 weight | embedding 结果;区间外 token 为全零行 |
| dim | 标量 | - | uint32 | embedding 维度(即 hidden_size),要求能被 16 整除([csrc/kernels/embed_kernel.h:24](../../csrc/kernels/embed_kernel.h#L24) 注释) |
| batch_size | 标量 | - | uint32 | token 数 |
| emb_start_idx / emb_end_idx | 标量 | - | uint32 | 本 rank 词表区间 `[start, end)`,host 由 `vocab_size / world_size * rank` 推得 |
| tp_size | 标量 | - | uint32 | `tp_size > 1` 时区间外行写零;`tp_size == 1` 时所有 id 都在区间内,跳过写零路径 |

## 支持的数据类型

- `float16_t`([embed_kernel_float16_t.cpp](../../csrc/kernels/embed_kernel_float16_t.cpp))
- `bfloat16_t`([embed_kernel_bfloat16_t.cpp](../../csrc/kernels/embed_kernel_bfloat16_t.cpp))

要求 embed 权重与 out dtype 一致([csrc/op.cpp:506-509](../../csrc/op.cpp#L506-L509))。

## 实现原理

实现位于 [csrc/kernels/embed_kernel.h](../../csrc/kernels/embed_kernel.h),C220 向量核上的极简行搬运 kernel,不做任何数值计算。

### 多 Block 并行

`for (index = block_idx; index < batch_size; index += block_num)`:token 级条带化分配,每个 AIV core 处理一批不相交的 token([csrc/kernels/embed_kernel.h:30](../../csrc/kernels/embed_kernel.h#L30))。

### UB 内存布局

只占 UB 头部两小块([csrc/kernels/embed_kernel.h:20-21](../../csrc/kernels/embed_kernel.h#L20-L21)):

- `ub_addr`(偏移 0):单行 embedding 暂存,`dim` 个元素;
- `zero_ub_addr`(偏移 `dim * 2` 字节):`vector_dup` 预填充的全零行,仅 `tp_size > 1` 时初始化([csrc/kernels/embed_kernel.h:26-28](../../csrc/kernels/embed_kernel.h#L26-L28))。

### 数据流

每 token 三步([csrc/kernels/embed_kernel.h:31-40](../../csrc/kernels/embed_kernel.h#L31-L40)):

1. 标量单元直接从 GM 读 `row = y[index]`(逐元素 GM 标量读,无预取);
2. **区间判断**:若 `row >= emb_end_idx || row < emb_start_idx`,把 UB 中的全零行 `copy_ubuf_to_gm` 到 `z + index * dim`;
3. **正常路径**:`copy_gm_to_ubuf` 把权重第 `row - emb_start_idx` 行(长度 `dim`,一个 burst 序列,`len_burst = dim / 16`、`n_burst = 1`)搬入 `ub_addr`,再 `copy_ubuf_to_gm` 写到输出第 `index` 行。

搬运为 GM→UB→GM 中继(而非 GM 到 GM 直搬),两段搬运之间以 `pipe_barrier(PIPE_ALL)` 保证 MTE2 写入对 MTE3 可见([csrc/kernels/embed_kernel.h:36-38](../../csrc/kernels/embed_kernel.h#L36-L38))。整个循环体内无向量计算,同步只用 pipe_barrier,不使用事件标志。

### 测试参考

[tests/kernels/embed.py](../../tests/kernels/embed.py):BATCH=64、VOCAB=32000、DIM=4096,fp16/bf16;多卡时与 `F.embedding` + 掩码置零 + `all_reduce` 的参考实现对比,单卡直接对比。
