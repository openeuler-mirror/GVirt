# sigmoid_gate_mul

## 功能概述

门控注意力输出融合算子：`out = attn * sigmoid(gate) [+ add]`。gate 支持两种形态：逐元素（gate 与 attn 同形）或按行广播（gate 每行只有一个标量，`gateDim == 1`，此时先算出每行的 sigmoid 标量再对整行 attn 做标量乘）。fp16/bf16 均升精度到 fp32 计算后舍回原 dtype。`add` 为可选融合加数（MoE 共享专家尾部的 routed 输出残差，`csrc/model.cpp` 的 `XModel::ForwardMoE`，关键字 `Fused tail`）：与门控乘法在同一 pass 内以 fp32 求和，省去一次独立 elementwise add 的 kernel 发射与中间结果 GM 往返；传 `nullptr`（Python 传 `None`/缺省）禁用。add 以 Dtype 载入，故要求 `!attnFp32`（kernel 内 `assert(!(add != nullptr && attnFp32))`），`silu_gate_mul` 入口恒传 `nullptr`。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| attn | 输入 | [numTokens, dim] | float16 / bfloat16 | 注意力输出 |
| gate | 输入 | [numTokens, dim] 或 [numTokens, 1] | 同 attn | 门控值；最后一维为 1 时按行广播 |
| out | 输出 | [numTokens, dim] | 同 attn | 结果，可与 attn 为同一 tensor（原地更新，模型 ForwardAttn 使用 in-place 路径） |
| add | 输入（可选） | [numTokens, dim] | 同 attn | 融合加数，与 attn 同形同 dtype；host 侧传 `add->ptr`，`nullptr` 禁用（`csrc/op.cpp` 的 `XliteOpSigmoidGateMul`，关键字 `add must be 2D`） |
| numTokens | 标量 | - | uint32_t | token 数（行数），host 侧传 `attn.shape[0]` |
| dim | 标量 | - | uint32_t | 每行元素数，host 侧传 `attn.shape[1]` |
| gateDim | 标量 | - | uint32_t | gate 的最后一维大小（1 或 dim），host 侧传 `gate.shape[1]`（`csrc/op.cpp` 的 `XliteOpSigmoidGateMul`，关键字 `gate.shape[1]`） |

Python 调用方式（`tests/kernels/sigmoid_gate_mul.py` 的 `run_test`，关键字 `sigmoid_gate_mul(rt, attn, gate, out)`）：`sigmoid_gate_mul(rt, attn, gate, out, add=None)`。host 侧校验三者必须为 2D、行数一致、gate 最后一维为 1 或等于 dim（`csrc/op.cpp` 的 `XliteOpSigmoidGateMul`，关键字 `gate last dim must be 1`）；`add` 给定时必须为 2D 且与 attn 同形同 dtype。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/sigmoid_gate_mul_float16_t.cpp` | `sigmoid_gate_mul_float16_t` |
| bfloat16_t | `csrc/kernels/sigmoid_gate_mul_bfloat16_t.cpp` | `sigmoid_gate_mul_bfloat16_t` |

仅支持 `__DAV_C220_VEC__`，其余架构导出空实现（`csrc/kernels/sigmoid_gate_mul.h` 的 `#else` 分支 `SIGMOID_GATE_MUL_FUNC_DEFINE`）。

## 实现原理

### 整体数据流

双层循环（`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体，关键字 `for (uint32_t token = block_idx` 与 `for (uint32_t col = 0; col < dim`）：

- 外层按行（token）跨 block 并行：`for (token = block_idx; token < numTokens; token += block_num)`；
- 内层按列 tile 分块：`tile = min(dim, 2048)` 且向下对齐到 64（`calcPad = VECTOR_MAX_NUM_OF_FP32`）；dim > 2048 时一行拆成多个 tile 依次处理。

每个 tile 的数据流（`add` 给定时多一条 addend 搬运与求和）：

GM(attn tile) --MTE2--> UB(attnIn[curr]) --V: vconv--> UB(attnF[curr]) \
GM(gate tile) --MTE2--> UB(gateIn[curr]) --V: vconv--> UB(gateF[curr]) \
GM(add tile) --MTE2--> UB(addBuf[curr]) --V: vconv--> UB(addF[curr]) \
--V: sigmoid + vmul (+ vadd)--> UB(attnF[curr]) --V: vconv 舍入--> UB(outBuf[curr]) --MTE3--> GM(out tile)

### UB 内存布局

每个 ping-pong 槽内按顺序分配 7 段（`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体 UB 偏移分配段，关键字 `attnIn0`/`gateIn0`/`addBuf0`），两套槽（slot0/slot1）共 14 段，另有全局共享的 `ones` 常量缓冲：

| 缓冲区（每槽） | 大小 | 用途 |
|--------|------|------|
| attnIn | ROUND_UP(tile*2B, 32) | attn 原始 dtype |
| gateIn | 同上 | gate 原始 dtype |
| attnF | ROUND_UP(tile*4B, 32) | attn 的 fp32 |
| gateF | 同上 | gate 的 fp32（广播模式下存单元素） |
| out | 同 attnIn | 输出 dtype |
| addBuf | 同 attnIn | 融合加数原始 dtype（无条件预留，`add == nullptr` 时闲置） |
| addF | 同 attnF | 融合加数的 fp32（同上） |
| ones（共享） | tile*4B | 常量 1.0，用于 vdiv 求 1/(1+e^-x) |

`ones` 在循环前用 `vector_dup(ones, 1.0, ...)` 一次性填充（`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体，关键字 `vector_dup(ones`）。tile 上限 2048 是为了让两套 ping-pong 槽全部放下 UB。

### 计算步骤

sigmoid 用 `1 / (1 + exp(-x))` 分解（`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体，关键字 `sigmoid(gate) = 1 / (1 + exp(-gate))`）：

逐元素模式（gateDim == dim）：

1. `vconv_f162f32` / `vconv_bf162f32`：attn、gate 分别升精度；
2. `vmuls(gateF, gateF, -1.0)` → `vexp` → `vadds(gateF, gateF, 1.0)`：得 1 + e^-gate；
3. `vdiv(gateF, ones, gateF)`：得 sigmoid(gate) = 1 / (1 + e^-gate)；
4. `vmul(attnF, attnF, gateF)`：attn * sigmoid(gate)；
5. （`add` 给定时）`vadd(attnF, attnF, addF)`：与融合加数求和（关键字 `vadd(attnF[curr], attnF[curr], addF[curr]`）；
6. `vconv_f322f16r` / `vconv_f322bf16r`：舍回原 dtype。

广播模式（gateDim == 1，`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体，关键字 `const bool broadcast` 与 `if (broadcast)`）：每行只搬 1 个 gate 元素，用单个 repeat 的 vmuls/vexp/vadds/vdiv 算出 sigmoid 标量，再通过 Scalar 流水（`set_flag/wait_flag(PIPE_V, PIPE_S, EVENT_ID2)`）把 `gateF[curr][0]` 读入标量寄存器 `sig`；随后每 tile 只需对 attnF 做 `vmuls(attnF, attnF, sig)` 标量乘（`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体，关键字 `vmuls(attnF[curr], attnF[curr], sig`），省去整行 gate 的搬运与逐元素 sigmoid。

两种模式下 gate 激活与 attn 乘法全程保持 FP32，sigmoid/silu 结果不做中间 dtype 舍入（区别于 native MHA 在 gate dtype 域求 sigmoid 的数值路径，是有意的精度选择）。

### 流水线同步

初始 4 个 flag：V→MTE2 的 ID0/ID1（两个输入槽空闲）、MTE3→V 的 ID0/ID1（两个输出槽空闲）（`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体，关键字 `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0)`）。每 tile 迭代：

- `wait_flag(PIPE_V, PIPE_MTE2, ID0+curr)` 等输入槽空闲 → `CopyGmToUbufAligned` 搬 attn（广播模式只搬 1 个 gate 元素，否则再搬 gate tile；`add` 给定时再搬 add tile，三者共用同一对事件）→ `set_flag(PIPE_MTE2, PIPE_V, ID0)` → `wait_flag` 同步后进入 V 计算；
- vconv 完成后 `set_flag(PIPE_V, PIPE_MTE2, ID0+curr)` 归还输入槽；
- 计算完成 → `wait_flag(PIPE_MTE3, PIPE_V, ID0+curr)` 等输出槽空闲 → vconv 转回 dtype → `set_flag(PIPE_V, PIPE_MTE3, ID0)` → `wait_flag` 后 `CopyUbufToGmAligned` 写回 → `set_flag(PIPE_MTE3, PIPE_V, ID0+curr)` 归还输出槽；
- `curr = 1 - curr` 切换 ping-pong，当前 tile 的 MTE3 写回与下一 tile 的 MTE2 搬入重叠。

退出前 `wait_flag` 收尾（`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul` 函数体末尾，关键字 `wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1)`）。

### 边界处理

- tile 对齐：`tile = ROUND_DOWN(min(dim, 2048), 64)`；若 dim < 64 则 tile 取 64（此时 tile > dim，按实际 calcNum 处理）；
- 行内尾 tile：`calcNum = min(dim - col, tile)`，搬运与写回均用精确字节数 `calcNum * sizeof(Dtype)`，`CopyGmToUbufAligned`/`CopyUbufToGmAligned` 按 32B/2B/1B 对齐自动选择 DMA 原语；
- repeat 数按 `DIV_ROUND_UP(calcNum, 64)` 计算，UB 中 padding 数据不参与写回。

### 片上缓冲分配与占用汇总

`sigmoid_gate_mul`(`csrc/kernels/sigmoid_gate_mul.h` 的 `sigmoid_gate_mul`)把 UB 排成「两个乒乓槽 + 共享尾区」:每槽内 `attnIn|gateIn|attnF|gateF|out|addBuf|addF` 七段按 `UB_BUF_ALIGN_SIZE=32` 对齐累进,槽后接共享的 `rawGate` 与 `ones`;`tile = ROUND_DOWN(min(dim,2048), VECTOR_MAX_NUM_OF_FP32=64)`(dim<64 时取 64),dim ≥ 2048 后 tile 封顶 2048,占用与行宽无关。硬件容量基准(910B/dav C220):UB 192KB(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE=196608`);本算子为纯 AIV 向量算子,不使用 L1/L0A/L0B/L0C。dtype 记法:模板实参 `Dtype` 为 attn/gate/out 元素类型,fp16/bf16 两个实例均为 2B、占用不分列;片内计算全程 fp32。典型配置取 dim=4096(Qwen3.5-MoE 注意力输出门控,16 头×headDim 256,`tests/models/qwen3_5_moe.py`),tile=2048;`silu_gate_mul` 入口(见下文)以 attnFp32=true 启动同一模板,`attnIn` 槽改按 fp32 尺寸分配,两列同取该配置。`addBuf`/`addF` 槽无条件预留(`add == nullptr` 时闲置,地址偏移在编译期常量路径上,无运行时代价),两入口占用一致。

**UB(预算 `UB_SIZE=192KB`;tile=2048,两 dtype 实例占用相同)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | sigmoid 入口占用(B) | silu_gate 入口占用(B) |
|---|---|---|---|---|---|---|
| `attnIn` | `(tile,)` | Dtype(silu_gate 入口为 float) | 2(事件0/1) | `ROUND_UP(tile*4,32)`(attnFp32)/ `ROUND_UP(tile*2,32)` | 8192 | 16384 |
| `gateIn` | `(tile,)` | Dtype | 2(事件0/1) | `ROUND_UP(tile*2,32)` | 8192 | 8192 |
| `attnF` | `(tile,)` | float | 2(事件0/1) | `ROUND_UP(tile*4,32)` | 16384 | 16384 |
| `gateF` | `(tile,)` | float | 2(事件0/1) | 同上 | 16384 | 16384 |
| `outBuf`(`out0`/`out1`) | `(tile,)` | Dtype | 2(事件0/1) | `ROUND_UP(tile*2,32)` | 8192 | 8192 |
| `addBuf` | `(tile,)` | Dtype | 2(事件0/1) | 同 `gateIn` | 8192 | 8192 |
| `addF` | `(tile,)` | float | 2(事件0/1) | 同 `attnF` | 16384 | 16384 |
| `rawGate` | `(tile,)` | float | 1(共享,siluGate 通路暂存原始 gate) | 同 `attnF` | 8192 | 8192 |
| `ones` | `(tile,)` | float | 1(共享,常量 1.0) | 同上 | 8192 | 8192 |
| **合计** | | | | | **98304** | **106496** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | sigmoid 入口 | 占比 | silu_gate 入口 | 占比 |
|---|---|---|---|---|---|
| UB | 192 KB | 98304 B | 51.2% | 106496 B | 55.4% |

要点:

- **tile 封顶 2048 是 UB 预算的直接产物**:`maxTile=2048` 常量使两套乒乓槽加共享区合计 98304 B;host 侧(`csrc/op.cpp` 的 `XliteOpSigmoidGateMul`)不校验 dim,更大的行宽由内层 tile 循环按 2048 分块消化,占用不再增长——`maxTile` 常量即容量来源,无 kernel 断言。
- **fp32 计算缓冲是大头**:`attnF`/`gateF`/`rawGate`/`ones` 四类 fp32 槽共 49152 B,sigmoid 入口下占 2/3;Dtype 原始槽(`attnIn`/`gateIn`/`out`)合计仅 24576 B——全程 fp32 计算的精度选择决定了占用结构。
- 广播模式(gateDim==1)不改变布局:`gateF` 槽恒按 tile 预留(只写首元素),逐元素与广播两模式占用一致;`silu_gate_mul` 入口仅 `attnIn` 槽翻倍为 fp32 尺寸(+8192 B),其余槽位不变。

## SiLU gate 入口

`silu_gate_mul(rt, normed, gate, out)` 计算 `out = normed * silu(gate)`。
三个张量为连续同形 `[T,D]`，normed 为 FP32，gate/out 为 BF16 或 FP16。
FP32 norm 保留到 gate 乘法后再舍入，不支持 sigmoid 入口的单列广播。
实现复用本 header，实例为 `silu_gate_mul_{bfloat16_t,float16_t}.cpp`。
该入口不改变 `sigmoid_gate_mul` 的数学语义。测试见 `tests/kernels/silu_gate_mul.py`。
