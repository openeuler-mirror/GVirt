# muls

## 功能概述

二维矩阵逐元素乘以标量：`output = input * scale`，可用于权重/激活缩放等场景。fp16/bf16 输入升精度到 float32 完成乘法后舍回原 dtype。host 侧还支持只计算每行 `[calcOffset, calcOffset + calcNum)` 区间的列（用于张量并行时只处理本卡负责的切片）。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| input | 输入 | [shape0, shape1] | float16 / bfloat16 | 输入矩阵；shape0 为行数（block 切分维度），shape1 为列数（一维 tensor 时按 shape1=1 处理，`csrc/op.cpp:1771`） |
| scale | 标量 | - | float | 乘法系数（kernel 端为 fp32） |
| output | 输出 | [shape0, shape1] | 同 input | 结果，可与 input 为同一 tensor |
| shape0 | 标量 | - | uint32_t | 行数，host 侧传 `input.shape[0]` |
| shape1 | 标量 | - | uint32_t | 列数，host 侧传 `input.shape[1]`（不足 2 维时为 1） |
| calcOffset | 标量 | - | uint32_t | 每行参与计算的起始列偏移（默认 0） |
| calcNum | 标量 | - | uint32_t | 每行参与计算的元素个数（默认 UINT32_MAX，即取 shape1 - calcOffset；上限 `MAX_MULS_CALC_NUM = 16320`，`csrc/kernels/kernel_param.h:46`) |

kernel 侧实际地址为 `input + process * shape1 + calcOffset`（`csrc/kernels/muls.h:55-56`），即每行只处理 `[calcOffset, calcOffset+calcNum)` 这段连续数据。Python 调用方式（`tests/kernels/muls.py:27`）：`muls(rt, x, scale, y)`（calcOffset/calcNum 走默认值，全量计算）。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/muls_float16_t.cpp` | `muls_float16_t` |
| bfloat16_t | `csrc/kernels/muls_bfloat16_t.cpp` | `muls_bfloat16_t` |

仅支持 `__DAV_C220_VEC__`，其余架构导出空实现（`csrc/kernels/muls.h:104-110`）。注意：模板里虽然按 `float` 分支保留了 off 布局逻辑，但当前 host 侧（`csrc/op.cpp:1764-1770`）与实例化文件都只提供 fp16/bf16。

## 实现原理

### 整体数据流

主循环 `for (process = first; process < shape0; process += block_num)`（`csrc/kernels/muls.h:54`），行间多 block 并行。每行处理 `calcNum` 个元素（GM 地址带 calcOffset 偏移）：

GM(行切片) --MTE2--> UB(in[curr]) --V: vconv--> UB(calc) --V: vmuls--> UB(calc) --V: vconv 舍入--> UB(out[curr]) --MTE3--> GM(行切片)

起始行 `first = (block_idx + block_num - coreOffset) % block_num`：coreOffset 默认 0，用于多处 kernel 协作时错开起始 block（`csrc/kernels/muls.h:53`）。

### UB 内存布局

UB 偏移手工累加分配（`csrc/kernels/muls.h:21-40`），`len = ROUND_UP(calcNum * sizeof(Dtype), VECTOR_MAX_BYTESIZE)`（256B 对齐）：

| 缓冲区 | 大小 | 用途 |
|--------|------|------|
| in1 / in2 | len（各） | 原始 dtype 输入，双缓冲（ping-pong） |
| calc | ROUND_UP(calcNum*4, 256) | fp32 计算缓冲 |
| out1 / out2 | len（各） | 原始 dtype 输出，双缓冲 |

双缓冲（`in[2]`、`out[2]`，`curr` 在 0/1 之间翻转）使当前行的 MTE3 写回与下一行的 MTE2 搬入可以重叠。

### 向量计算步骤（`csrc/kernels/muls.h:64-80`）

`repeat = DIV_ROUND_UP(calcNum, 64)`（64 = 256B 内的 fp32 元素数）：

1. `vconv_f162f32` / `vconv_bf162f32`：输入升精度到 fp32；
2. `vmuls(calc, calc, scale, ...)`：fp32 标量乘（scale 为 fp32，因此 fp16 输入不会因半精度标量损失精度）；
3. `vconv_f322f16r` / `vconv_f322bf16r`：舍入转回原 dtype。

每步之间 `pipe_barrier(PIPE_V)` 保证顺序。

### 流水线同步

初始置 4 个 flag（V→MTE2 的 ID0/ID1、MTE3→V 的 ID0/ID1，`csrc/kernels/muls.h:48-51`）。每轮迭代（`csrc/kernels/muls.h:58-88`）：

- `wait_flag(PIPE_V, PIPE_MTE2, ID0+curr)` 等输入缓冲空闲 → `CopyGmToUbufAligned` 搬入（按 actualLen 字节数自动选择 32B/16B/8B 对齐的 DMA 原语，见 `csrc/kernels/kernel_macro.h:813-823`）→ `set_flag(PIPE_MTE2, PIPE_V, ID0)`；
- vconv 完成后 `set_flag(PIPE_V, PIPE_MTE2, ID0+curr)` 归还输入缓冲；
- `wait_flag(PIPE_MTE3, PIPE_V, ID0+curr)` 等输出缓冲空闲 → vconv 转回 dtype → `set_flag(PIPE_V, PIPE_MTE3, ID0)` → `wait_flag` 后 `CopyUbufToGmAligned` 写回 → `set_flag(PIPE_MTE3, PIPE_V, ID0+curr)` 归还输出缓冲；
- `curr = 1 - curr` 切换 ping-pong。

退出前 `wait_flag` 收尾（`csrc/kernels/muls.h:90-93`）。

### 边界处理

- 搬运用 `CopyGmToUbufAligned`/`CopyUbufToGmAligned`（按字节数 32B/2B/1B 对齐自适应选择 DMA 原语），支持 calcNum 非 32B 整数倍的场景，写回长度为精确的 `calcNum * sizeof(Dtype)` 字节；
- host 侧校验 `calcOffset < shape1`、`calcNum <= shape1 - calcOffset` 且 `calcNum <= 16320`（`csrc/op.cpp:1772-1789`）；
- calcNum 上限 16320 = 255 repeat × 64 fp32，即单行切片需一次性放入 UB 计算缓冲（无行内二次分块）。
