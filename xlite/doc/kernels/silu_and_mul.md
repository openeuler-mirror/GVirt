# silu_and_mul

## 功能概述

SwiGLU 激活：输入 x 按最后一维分成两半 gate 和 up，输出 `y = silu(gate) * up`，其中 `silu(x) = x / (1 + e^-x)`。可选 `swiglu_limit` 截断（clamp）：`up = clamp(up, -L, L)`、`gate = clamp(gate, max=L)`。本算子由小艺团队贡献（参考论文 XY-Serve, ASPLOS 2026，见 `csrc/kernels/silu_and_mul.h:15-16`）。

注意：fp16 与 bf16 的实现是两套不同的 kernel——fp16/float 走 `silu_and_mul.h` 的通用模板，bf16 走 `silu_and_mul_bfloat16_t.cpp` 中单独手写的实现。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入 | [num_tokens, 2*dim] | float / float16 / bfloat16 | gate 占前 dim 列，up 占后 dim 列 |
| y | 输出 | [num_tokens, dim] | 同 x | 激活结果 |
| pnum_tokens | 输入(可选) | [1] | uint32 | 实际 token 数指针（动态 batch 场景），取 min(*pnum_tokens, num_tokens)；可为空 |
| num_tokens | 标量 | - | uint32_t | token 数，host 侧传 `in.shape[0]` |
| dim | 标量 | - | uint32_t | 输出列宽，host 侧传 `out.shape[1]`（`csrc/op.cpp:781-782`） |
| swiglu_limit | 标量 | - | float | 截断阈值 L，<= 0 表示不启用 clamp |

Python 调用方式（`tests/kernels/silu_and_mul.py:32`）：`silu_and_mul(rt, input, output)`，带 clamp 时 `silu_and_mul(rt, input, output, swiglu_limit=L)`。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 | 计算类型 CalType |
|-------|-----------|-------------|------------------|
| float | `csrc/kernels/silu_and_mul_float.cpp` | `silu_and_mul_float` | float |
| float16_t | `csrc/kernels/silu_and_mul_float16_t.cpp` | `silu_and_mul_float16_t` | float16_t |
| bfloat16_t | `csrc/kernels/silu_and_mul_bfloat16_t.cpp` | `silu_and_mul_bfloat16_t` | 独立实现（片内 fp32 计算） |

fp16 变体的 CalType 也是 fp16，即 silu 的中间计算（exp、div 等）在 fp16 上完成；float 变体全程 fp32。bf16 变体升精度到 fp32 计算。

## 实现原理（fp16 / float 通用模板，`csrc/kernels/silu_and_mul.h`）

### 并行切分

- 先用 pnum_tokens 修正实际 token 数（`csrc/kernels/silu_and_mul.h:25-28`）；
- 按行（token）静态均分给各 block：`tokens_per_block = DIV_ROUND_UP(num_tokens, block_dims)`，当前 block 处理 `[tokens_per_block * block_idx, ...)` 范围内的行，行数不足时截断（`csrc/kernels/silu_and_mul.h:32-43`）——注意这里不是跨 block 的 stride 循环，而是连续分段。

### 数据流与 UB 布局

UB 分成 5 等份，每份 `MAX_HIDDENSIZE_PER_PIECE = 38912` 个元素（= UB_SIZE 196608 / 5，`csrc/kernels/silu_and_mul.h:12-13, 52-57`）：

| 缓冲区 | 作用 |
|--------|------|
| ubuf0 / ubuf2 | gate 半区（x 前 dim 列），ping-pong 双缓冲 |
| ubuf1 / ubuf3 | up 半区（x 后 dim 列），ping-pong 双缓冲 |
| cal_ubuf | 中间计算缓冲（e^-x、1+e^-x、silu 结果） |

每个 tile 一次搬运 `tokens_per_tile = 38912 / (dim * sizeof(CalType))` 个 token（行内 dim 列不再分块，要求整行放入一个子缓冲）。用 `__set_dmi_config` 构造 DMI 配置：`token_copy_config` 以 token 为 burst 间隔（nBurst=tokens_per_tile，lenBurst=token_len 字节数）按行搬运，`out_copy_config` 整块连续搬出（`csrc/kernels/silu_and_mul.h:77-86`）。

### 计算步骤（`csrc/kernels/silu_and_mul.h:113-141`）

按 silu(x) = x / (1 + e^-x) 分解为向量指令序列（x_ubuf=gate，y_ubuf=up，全部在 CalType 上运算）：

1. （可选 clamp）`vmins(x_ubuf, x_ubuf, L)`、`vmins(y_ubuf, y_ubuf, L)` + `vmaxs(y_ubuf, y_ubuf, -L)`：gate 只截上界，up 截上下界；
2. `vmuls(cal_ubuf, x_ubuf, -1)`：求 -x；
3. `vexp(cal_ubuf, cal_ubuf)`：求 e^-x；
4. `vadds(cal_ubuf, cal_ubuf, 1.0)`：求 1 + e^-x；
5. `vdiv(cal_ubuf, x_ubuf, cal_ubuf)`：silu = x / (1 + e^-x)；
6. `vmul(x_ubuf, y_ubuf, cal_ubuf)`：up * silu(gate)，结果写回 gate 缓冲后直接搬出到 GM。

向量指令的 repeat/stride 通过 `set_vector_xt` / `set_vector_1src_xt` 打包成 config（repeat 为 tile 总元素数按 256B repeat 折算，`csrc/kernels/silu_and_mul.h:63-75`）。

### 流水线同步

ping-pong 双缓冲 + 2 个 event id（`csrc/kernels/silu_and_mul.h:88-151`）：

- 初始 `set_flag(PIPE_MTE3, PIPE_MTE2, ID0/ID1)` 使两个缓冲初始可用；
- `wait_flag(PIPE_MTE3, PIPE_MTE2, event_id)` 等该缓冲上一轮写回完成 → 两路 `copy_gm_to_ubuf` 分别搬 gate 半区和 up 半区 → `set_flag(PIPE_MTE2, PIPE_V, event_id)`；`wait_flag(PIPE_MTE2, PIPE_V, event_id)` 后进入计算；
- 计算完成 `set_flag(PIPE_V, PIPE_MTE3, event_id)` → MTE3 `wait_flag` 后 `copy_ubuf_to_gm` 搬出 → `set_flag(PIPE_MTE3, PIPE_MTE2, event_id)` 归还缓冲；
- `ping = 1 - ping` 切换缓冲，下一 tile 的 MTE2 搬入与当前 tile 的 V 计算/MTE3 搬出重叠。

退出前 `wait_flag` 收尾 + `pipe_barrier(PIPE_ALL)`（`csrc/kernels/silu_and_mul.h:152-154`）。

### 边界处理

- 最后一个 tile 的 token 数不足 tokens_per_tile 时，重新按实际 token 数计算 lenBurst / nBurst 再构造 DMI config（`csrc/kernels/silu_and_mul.h:96-102`），避免多搬多写；
- token 数小于 block_idx 分段起点时直接 return；
- dim 上限：单行（2*dim 个 CalType）需放入 2 个子缓冲，即 `dim * sizeof(CalType) <= 38912` 字节量级。

## 实现原理（bf16 变体，`csrc/kernels/silu_and_mul_bfloat16_t.cpp`）

bf16 采用不同的切分与布局：

- 任务切分：`index = block_idx; index < n_tokens * split; index += block_num` 跨 block stride 循环；若 dim 过大（`dim > UB_SIZE / (5*4 + 5*2)` 折算），将 dim 对半拆分成 `split` 段，行 × 段作为并行任务（`row = index / split, line = index % split`）；
- UB 布局：`x32_ub0/x32_ub1`（fp32 gate+up，双缓冲，每块 2*padded_dim 个 fp32）、`tmp`（fp32 中间量）、`output_ub`（bf16 输出）、`x32_ub0_bf16/x32_ub1_bf16`（bf16 原始输入，双缓冲）；
- 计算流程与 fp16 模板相同（clamp → vmuls(-1) → vexp → vadds(1) → vdiv → vmul），但显式插入 `vconv_bf162f32` 升精度、`vconv_f322bf16r` 降精度（`silu_and_mul_bfloat16_t.cpp:80-83, 118-120`）；
- 尾部处理：`dim_split % 64 != 0` 时用 `SetMaskFromHighBit` + `vector_dup(0)` 把 fp32 缓冲尾部 padding 置零（`silu_and_mul_bfloat16_t.cpp:85-92`），随后恢复全 1 mask；搬运用 32B 取整的 burst_copy，最后一个 split 段按实际长度写回；
- 同步：与 fp16 模板同构的 set_flag/wait_flag 双缓冲流水（EVENT_ID0/ID1）。
