# sigmoid_gate_mul

## 功能概述

门控注意力输出融合算子：`out = attn * sigmoid(gate)`。gate 支持两种形态：逐元素（gate 与 attn 同形）或按行广播（gate 每行只有一个标量，`gateDim == 1`，此时先算出每行的 sigmoid 标量再对整行 attn 做标量乘）。fp16/bf16 均升精度到 fp32 计算后舍回原 dtype。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| attn | 输入 | [numTokens, dim] | float16 / bfloat16 | 注意力输出 |
| gate | 输入 | [numTokens, dim] 或 [numTokens, 1] | 同 attn | 门控值；最后一维为 1 时按行广播 |
| out | 输出 | [numTokens, dim] | 同 attn | 结果，可与 attn 为同一 tensor（原地更新，模型 ForwardAttn 使用 in-place 路径） |
| numTokens | 标量 | - | uint32_t | token 数（行数），host 侧传 `attn.shape[0]` |
| dim | 标量 | - | uint32_t | 每行元素数，host 侧传 `attn.shape[1]` |
| gateDim | 标量 | - | uint32_t | gate 的最后一维大小（1 或 dim），host 侧传 `gate.shape[1]`（`csrc/op.cpp:2050-2053`） |

Python 调用方式（`tests/kernels/sigmoid_gate_mul.py:26`）：`sigmoid_gate_mul(rt, attn, gate, out)`。host 侧校验三者必须为 2D、行数一致、gate 最后一维为 1 或等于 dim（`csrc/op.cpp:2032-2043`）。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/sigmoid_gate_mul_float16_t.cpp` | `sigmoid_gate_mul_float16_t` |
| bfloat16_t | `csrc/kernels/sigmoid_gate_mul_bfloat16_t.cpp` | `sigmoid_gate_mul_bfloat16_t` |

仅支持 `__DAV_C220_VEC__`，其余架构导出空实现（`csrc/kernels/sigmoid_gate_mul.h:184-190`）。

## 实现原理

### 整体数据流

双层循环（`csrc/kernels/sigmoid_gate_mul.h:74, 103`）：

- 外层按行（token）跨 block 并行：`for (token = block_idx; token < numTokens; token += block_num)`；
- 内层按列 tile 分块：`tile = min(dim, 2048)` 且向下对齐到 64（`calcPad = VECTOR_MAX_NUM_OF_FP32`）；dim > 2048 时一行拆成多个 tile 依次处理。

每个 tile 的数据流：

GM(attn tile) --MTE2--> UB(attnIn[curr]) --V: vconv--> UB(attnF[curr]) \
GM(gate tile) --MTE2--> UB(gateIn[curr]) --V: vconv--> UB(gateF[curr]) \
--V: sigmoid + vmul--> UB(attnF[curr]) --V: vconv 舍入--> UB(outBuf[curr]) --MTE3--> GM(out tile)

### UB 内存布局

每个 ping-pong 槽内按顺序分配 5 段（`csrc/kernels/sigmoid_gate_mul.h:34-55`），两套槽（slot0/slot1）共 10 段，另有全局共享的 `ones` 常量缓冲：

| 缓冲区（每槽） | 大小 | 用途 |
|--------|------|------|
| attnIn | ROUND_UP(tile*2B, 32) | attn 原始 dtype |
| gateIn | 同上 | gate 原始 dtype |
| attnF | ROUND_UP(tile*4B, 32) | attn 的 fp32 |
| gateF | 同上 | gate 的 fp32（广播模式下存单元素） |
| out | 同 attnIn | 输出 dtype |
| ones（共享） | tile*4B | 常量 1.0，用于 vdiv 求 1/(1+e^-x) |

`ones` 在循环前用 `vector_dup(ones, 1.0, ...)` 一次性填充（`csrc/kernels/sigmoid_gate_mul.h:57-59`）。tile 上限 2048 是为了让两套 ping-pong 槽全部放下 UB。

### 计算步骤

sigmoid 用 `1 / (1 + exp(-x))` 分解（`csrc/kernels/sigmoid_gate_mul.h:142-150`）：

逐元素模式（gateDim == dim）：
1. `vconv_f162f32` / `vconv_bf162f32`：attn、gate 分别升精度；
2. `vmuls(gateF, gateF, -1.0)` → `vexp` → `vadds(gateF, gateF, 1.0)`：得 1 + e^-gate；
3. `vdiv(gateF, ones, gateF)`：得 sigmoid(gate) = 1 / (1 + e^-gate)；
4. `vmul(attnF, attnF, gateF)`：attn * sigmoid(gate)；
5. `vconv_f322f16r` / `vconv_f322bf16r`：舍回原 dtype。

广播模式（gateDim == 1，`csrc/kernels/sigmoid_gate_mul.h:76-101`）：每行只搬 1 个 gate 元素，用单个 repeat 的 vmuls/vexp/vadds/vdiv 算出 sigmoid 标量，再通过 Scalar 流水（`set_flag/wait_flag(PIPE_V, PIPE_S, EVENT_ID2)`）把 `gateF[curr][0]` 读入标量寄存器 `sig`；随后每 tile 只需对 attnF 做 `vmuls(attnF, attnF, sig)` 标量乘（`csrc/kernels/sigmoid_gate_mul.h:138-139`），省去整行 gate 的搬运与逐元素 sigmoid。

### 流水线同步

初始 4 个 flag：V→MTE2 的 ID0/ID1（两个输入槽空闲）、MTE3→V 的 ID0/ID1（两个输出槽空闲）（`csrc/kernels/sigmoid_gate_mul.h:67-70`）。每 tile 迭代：

- `wait_flag(PIPE_V, PIPE_MTE2, ID0+curr)` 等输入槽空闲 → `CopyGmToUbufAligned` 搬 attn（广播模式只搬 1 个 gate 元素，否则再搬 gate tile）→ `set_flag(PIPE_MTE2, PIPE_V, ID0)` → `wait_flag` 同步后进入 V 计算；
- vconv 完成后 `set_flag(PIPE_V, PIPE_MTE2, ID0+curr)` 归还输入槽；
- 计算完成 → `wait_flag(PIPE_MTE3, PIPE_V, ID0+curr)` 等输出槽空闲 → vconv 转回 dtype → `set_flag(PIPE_V, PIPE_MTE3, ID0)` → `wait_flag` 后 `CopyUbufToGmAligned` 写回 → `set_flag(PIPE_MTE3, PIPE_V, ID0+curr)` 归还输出槽；
- `curr = 1 - curr` 切换 ping-pong，当前 tile 的 MTE3 写回与下一 tile 的 MTE2 搬入重叠。

退出前 `wait_flag` 收尾（`csrc/kernels/sigmoid_gate_mul.h:170-173`）。

### 边界处理

- tile 对齐：`tile = ROUND_DOWN(min(dim, 2048), 64)`；若 dim < 64 则 tile 取 64（此时 tile > dim，按实际 calcNum 处理）；
- 行内尾 tile：`calcNum = min(dim - col, tile)`，搬运与写回均用精确字节数 `calcNum * sizeof(Dtype)`，`CopyGmToUbufAligned`/`CopyUbufToGmAligned` 按 32B/2B/1B 对齐自动选择 DMA 原语；
- repeat 数按 `DIV_ROUND_UP(calcNum, 64)` 计算，UB 中 padding 数据不参与写回。
