# indexer_prepare

## 功能概述

DeepSeek V3.2 DSA(DeepSeek Sparse Attention)indexer 的 K/Q 预处理融合算子(核内函数 `norm_ropex_cache_muls`,见 `csrc/kernels/indexer_prepare.h:15`)。对 indexer 的 K 投影 `kw` 执行:LayerNorm → 前缀 GPT-J 交错式 RoPE → 写入 paged `index_k_cache`;当 `is_long` 为真(序列长度超过 topK 走长序列路径)时,额外对 Q 做 per-head RoPE(原地),并对 `kw` 尾部的 `index_n_heads` 个头权重列按 `scale` 缩放(仅对 position > top_k 的 token)。该算子是 `indexer_scores`/`indexer_topk` 的前置步骤,为其准备好归一化、加旋转位置编码后的 K cache 与 Q。

数学语义(见测试 `tests/kernels/indexer_prepare.py:169-205` 的参考实现):

1. `ln = LayerNorm(kw[:, :index_head_dim], kNorm, kNormBias, eps)`;
2. `ln[:, :rope_head_dim] = InterleavedRoPE(ln[:, :rope_head_dim], freqs_cis[position])`,并把结果按 `slot_mapping` scatter 到 `index_k_cache`;
3. 长序列路径(`is_long`):`q` 每个 head 的 `rope_head_dim` 前缀原地 RoPE;`kw[:, index_head_dim : index_head_dim+index_n_heads] *= scale`(仅 `position > top_k` 的 token)。

## 输入输出参数

Python 侧调用(`tests/kernels/indexer_prepare.py:209`):

```python
indexer_prepare(rt, kw, k_norm, k_norm_bias, freqs_cis, position,
                index_head_dim, index_n_heads, rope_head_dim, block_size,
                index_k_cache, slot_mapping, norm_eps, q, scale, topK, is_long)
```

host 侧 launch 见 `csrc/op.cpp:1163`(`XliteOpIndexerPrepare`)。kernel 签名(`csrc/kernels/indexer_prepare.h:392`):

```cpp
indexer_prepare_<dtype>(GM_ADDR kw, GM_ADDR kNorm, GM_ADDR kNormBias, GM_ADDR freqs,
                        GM_ADDR position, uint32_t token_num, uint32_t index_head_dim,
                        uint32_t index_n_heads, uint32_t rope_head_dim, uint32_t block_size,
                        float norm_eps, bool norm_in_fp32, GM_ADDR index_k_cache,
                        GM_ADDR slot_mapping, GM_ADDR q, float scale, uint32_t top_k,
                        bool is_long)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| kw | 输入/输出 | `[token_num, index_head_dim + index_n_heads]` | fp16 / bf16 | indexer K 投影。前 `index_head_dim` 列为 LN 输入;启用 cache 时 kw 只读,尾列在长序列路径下被原地缩放 |
| kNorm | 输入 | `[index_head_dim]` | fp32(或与 kw 同 dtype) | LayerNorm weight |
| kNormBias | 输入 | `[index_head_dim]` | fp32(或与 kw 同 dtype) | LayerNorm bias |
| freqs | 输入 | `[max_seq_len, rope_head_dim/2]` complex(即 `[max_seq_len, rope_head_dim]` float 交错 cos/sin) | fp32 | 预计算的 RoPE 频率表,按 token 绝对位置索引 |
| position | 输入 | `[token_num]` | int64 | 每个 token 的绝对序列位置(= 各 batch 的 `cached_lens[i]` 起始偏移累加) |
| index_k_cache | 输出 | `[block_num, block_size, index_head_dim]` | fp16 / bf16 | paged indexer K cache;`block_size=0` 时禁用缓存 |
| slot_mapping | 输入 | `[token_num]` | int32 | token → cache 平坦槽位映射 |
| q | 输入/输出 | `[token_num, index_n_heads * index_head_dim]` | fp16 / bf16 | indexer Q;仅 `is_long` 时被原地 RoPE |
| index_head_dim / index_n_heads / rope_head_dim / block_size / token_num | — | 标量 | uint32 | 维度参数,`total_dim = index_head_dim + index_n_heads`;所有维度要求为 64 的倍数(`indexer_prepare.h:36` 注释) |
| norm_eps | — | 标量 | float | LN epsilon |
| norm_in_fp32 | — | 标量 | bool | kNorm/kNormBias 是否为 fp32(host 侧由 dtype 推断) |
| scale | — | 标量 | float | 长序列路径的头权重缩放系数(测试中为 `1/sqrt(n_heads*head_dim)`) |
| top_k / is_long | — | 标量 | uint32 / bool | muls 门控阈值与长序列开关 |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_prepare_bfloat16_t` | `csrc/kernels/indexer_prepare_bfloat16_t.cpp` | kw 为 bf16;kNorm/kNormBias 可为 bf16 或 fp32 |
| `indexer_prepare_float16_t` | `csrc/kernels/indexer_prepare_float16_t.cpp` | kw 为 fp16;kNorm/kNormBias 可为 fp16 或 fp32 |

dtype 分派逻辑见 `csrc/op.cpp:1178-1188`。kernel 为纯向量核(`__DAV_C220_VEC__`,`KERNEL_TASK_TYPE_DEFAULT` 未设,由 `XliteOpIndexerPrepare` 以 `rt.aivNum` 个 block launch)。

## 实现原理

### 数据流与行级并行

核心实现是模板函数 `norm_ropex_cache_muls`(`csrc/kernels/indexer_prepare.h:15`)。token 行按块间切分:每行独立处理(LayerNorm 是行内归约),无块间依赖:

```cpp
uint32_t n_rows_per_core = DIV_ROUND_UP(n_rows, block_num);
uint32_t rel_block_idx = (block_idx + block_num - core_offset) % block_num;
```

`core_offset` 机制(`indexer_prepare.h:40-44`)允许同一 launch 中多个阶段串联:当前阶段的行分布相对上一个阶段的结束位置旋转,避免所有行恰好落在同一批 core 上(下游 `rope_complex_and_cache` 复用该偏移,见 `indexer_prepare.h:384-388`)。

每行的处理流水(单行内的指令序列,双缓冲乒乓):

1. **搬入**:`kw[irow]` 的前 `calc_dim = norm_dim + scale_dim` 列 GM→UB,`convert_input` 转 fp32(`indexer_prepare.h:206-212`);
2. **位置/槽位预取**:`position`、`slot_mapping` 分批(每批 `ub_pos_num` 个,S 管线标量读取)搬入 UB(`indexer_prepare.h:187-203`);
3. **freqs 拆分**:`vreducev2` 从交错表中抽偶数位(cos)与奇数位(sin)(`indexer_prepare.h:230-236`);
4. **muls**(长序列):`position > top_k` 时对尾列 `vmuls` 缩放(`indexer_prepare.h:242-246`);
5. **LayerNorm**(fp32 计算):`vmuls`(x/n)→ `reduce_sum`(均值)→ `duplicate_item` → `vsub` → `vmul` → `reduce_sum`(方差)→ `vadds`(eps)→ `vsqrt` → `vdiv` → weight/bias 仿射(`indexer_prepare.h:249-299`);
6. **RoPE**(GPT-J 交错):`vreducev2` 按步长抽偶/奇元素为 `in_even`/`in_odd`,`vmul`×4 + `vsub`/`vadd` 得到 `out[0::2] = x_even*cos - x_odd*sin`、`out[1::2] = x_even*sin + x_odd*cos`(`indexer_prepare.h:301-327`);
7. **搬出**:`convert_output` 转回 Dtype;启用 cache 时写 `index_k_cache + slot*norm_dim`(LN+RoPE 后的 `index_head_dim` 列)+ 长序列时尾列写回 `kw` 原地(`indexer_prepare.h:331-346`)。

### UB 内存布局

UB 布局针对 220 架构 bank group 手工排布(`indexer_prepare.h:59-129`):常驻区(低地址)依次为 `ub_weight`/`ub_bias`(带 `UB_BANK_CONFLICT_OFFSET` 错位)、`ub_pos`、`ub_slot_mapping`、乒乓 `in0/in1`、`out0/out1`、`ub_freqs0/1` 与 RoPE 的 cos/sin 中间缓冲;计算区(高地址,从 `UB_SIZE` 向下)依次为 `in_float`/`out_float`/`calc_float`(fp32 计算)与 RoPE 的 even/odd/cos/sin。行大小按 `UB_BANKGROUP_ROW_SIZE` 对齐并交错 bank 起点以缓解 bank conflict(`indexer_prepare.h:60-66`),最后 `assert(off <= invoff)` 保证两区不重叠。

### 流水线同步

搬运(MTE2/MTE3)与计算(V/S)通过计数事件 ID 交叉同步:`in` 缓冲用 `EVENT_ID0/1` 乒乓,`ub_freqs` 用 `EVENT_ID2/3`,`out` 缓冲用 MTE3 的 `EVENT_ID0/1`。归约结果被 S 管线标量消费前有 `set_flag(PIPE_V, PIPE_S)`/`wait_flag` 配对(`indexer_prepare.h:256-257`)。循环结束后清空所有挂起 flag 并 `pipe_barrier(PIPE_ALL)`(`indexer_prepare.h:350-359`)。

### 长序列路径(is_long)

`indexer_prepare`(`indexer_prepare.h:362-389`)先调 `norm_ropex_cache_muls` 处理 K(此时 `scale_dim = is_long ? index_n_heads : 0`),再调 `rope_complex_and_cache`(复用 `csrc/kernels/rope_complex_and_cache.h`,传入更新后的 `core_offset`)对 Q 的 `index_n_heads` 个 head 各自做 RoPE。`is_long` 的判定在模型层:当 paged 序列长度(`max_num_blocks * block_size`)超过 topK 时为真(测试 `tests/kernels/indexer_prepare.py:126-128`)。
