# embed_kernel

## 功能概述

带词表切分的 embedding 查表:对每个 token id,若其落在本 rank 的词表区间 `[emb_start_idx, emb_end_idx)` 内,则从权重矩阵中取出对应行写入输出;否则输出全零行。配合 TP 部署时,各 rank 只持有 `vocab_size / tp_size` 行切片,区间外置零后由上层 `all_reduce` 求和得到完整 embedding(见 `tests/kernels/embed.py` 的参考实现段,关键字 `F.embedding` 与 `all_reduce`;以及 `csrc/model.cpp` 的 `XModel::ForwardParallelEmbed`)。

## 输入输出参数

Python 接口:`embed(rt, weight, in_, out, start, end)`(`XliteOpEmbed`,见 `csrc/op.cpp` 的 `XliteOpEmbed`)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x(weight) | 输入 | `[emb_end_idx - emb_start_idx, dim]` | float16 / bfloat16 | 本 rank 的词表切片权重,行 `row - emb_start_idx` 对应全局 token id `row` |
| y(in) | 输入 | `[batch_size]` | int32 | token id 序列(全局词表编号) |
| z(out) | 输出 | `[batch_size, dim]` | 同 weight | embedding 结果;区间外 token 为全零行 |
| dim | 标量 | - | uint32 | embedding 维度(即 hidden_size),要求能被 16 整除(见 `csrc/kernels/embed_kernel.h` 的 `embed_kernel` 模板函数,关键字 `dim / 16` 注释) |
| batch_size | 标量 | - | uint32 | token 数 |
| emb_start_idx / emb_end_idx | 标量 | - | uint32 | 本 rank 词表区间 `[start, end)`,host 由 `vocab_size / world_size * rank` 推得 |
| tp_size | 标量 | - | uint32 | `tp_size > 1` 时区间外行写零;`tp_size == 1` 时所有 id 都在区间内,跳过写零路径 |

## 支持的数据类型

- `float16_t`([embed_kernel_float16_t.cpp](../../../csrc/kernels/embed_kernel_float16_t.cpp))
- `bfloat16_t`([embed_kernel_bfloat16_t.cpp](../../../csrc/kernels/embed_kernel_bfloat16_t.cpp))

要求 embed 权重与 out dtype 一致(见 `csrc/op.cpp` 的 `XliteOpEmbed`,关键字 `EachXDtype`)。

## 实现原理

实现位于 [csrc/kernels/embed_kernel.h](../../../csrc/kernels/embed_kernel.h),C220 向量核上的极简行搬运 kernel,不做任何数值计算。

### 多 Block 并行

`for (index = block_idx; index < batch_size; index += block_num)`:token 级条带化分配,每个 AIV core 处理一批不相交的 token(见 `csrc/kernels/embed_kernel.h` 的 `embed_kernel` 模板函数,关键字 `index += block_num`)。

### UB 内存布局

占 UB 头部三块、各 `dim` 个元素(见 `csrc/kernels/embed_kernel.h` 的 `embed_kernel` 模板函数,关键字 `zero_ub_addr`/`stage_ub_addr`):

- `zero_ub_addr`(偏移 0):`vector_dup` 预填充的全零行,仅 `tp_size > 1` 时初始化(见 `csrc/kernels/embed_kernel.h` 的 `embed_kernel` 模板函数,关键字 `vector_dup(zero_ub_addr`);
- `stage_ub_addr[2]`(偏移 `dim * 2` / `dim * 4` 字节):ping-pong 双暂存,让第 `i+1` 行的 MTE2 读入与第 `i` 行的 MTE3 写出重叠。

三块共 `3 * dim` 个元素,`dim % 16 == 0` 且容量约束由 host 侧 `XliteOpEmbed` 强制(`EMBED_UB_BUDGET` 定义于 `csrc/kernels/kernel_param.h`,host 检查与 kernel 内的 assert 共用同一常量、不会漂移;kernel 内的 assert 在 NDEBUG 下会被编译掉,host 检查是唯一守卫)。

### 数据流

每 token 三步(见 `csrc/kernels/embed_kernel.h` 的 `embed_kernel` 模板函数,关键字 `copy_gm_to_ubuf` 的 token 循环):

1. 标量单元直接从 GM 读 `row = y[index]`(逐元素 GM 标量读,无预取);
2. **区间判断**:若 `row >= emb_end_idx || row < emb_start_idx`,把 UB 中的全零行 `copy_ubuf_to_gm` 到 `z + index * dim`;
3. **正常路径**:`copy_gm_to_ubuf` 把权重第 `row - emb_start_idx` 行(长度 `dim`,一个 burst 序列,`len_burst = dim / 16`、`n_burst = 1`)搬入暂存,再 `copy_ubuf_to_gm` 写到输出第 `index` 行。

行按 token id 离散聚集,单行搬运无法合并;搬运为 GM→UB→GM 中继(而非 GM 到 GM 直搬)。kernel 按每核行数二选一(见 `csrc/kernels/embed_kernel.h` 的 `embed_kernel` 模板函数,关键字 `PIPE_MIN_ROWS_PER_CORE`):

- **小批量(每核 < 4 行,典型 decode batch)**:沿用逐行 `pipe_barrier(PIPE_ALL)` 串行循环 —— 每核行数不足以让下一行的读与当前行的写重叠,事件标志只增加每行固定开销;
- **大批量(prefill / 大 decode batch)**:MTE2/MTE3 ping-pong 事件流水(与 `concat_kernel` 相同的模式):第 `i` 行读入 `stage[curr]` 的同时,把上一行从 `stage[1-curr]` 写出;`set_flag`/`wait_flag` 的 EVENT_ID0/1 链保证一个暂存半区只在两行之前的 MTE3 写完成后才被复用。910B2 实测 bf16 大批量场景 kernel 时长 −24% ~ −53%,小批量持平。

### 测试参考

[tests/kernels/embed.py](../../../tests/kernels/embed.py):BATCH=64/512、VOCAB=32000、DIM=4096,fp16/bf16(64 行走小批量串行路径,512 行走 ping-pong 流水路径);多卡时与 `F.embedding` + 掩码置零 + `all_reduce` 的参考实现对比,单卡直接对比。msprof 逐 kernel 计时见 [tests/perf/embed.py](../../../tests/perf/embed.py)。
