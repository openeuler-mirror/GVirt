# repeat_interleave

## 功能概述

线性注意力（GDN Step 6）的头扩展（head expansion）：把 K 头个数的每个 head 复制 `expand = nVHeads/nKHeads` 份到 V 头维度，语义为 `dst[t, h*expand + e, :] = src[t, h, :]`（等价 ExpandLinearHeads / torch.repeat_interleave 在头维的扩展）。纯按字节的 GM→GM 拷贝，dtype 无关；用单次 kernel launch 取代原先 host 侧每层 `nKHeads*expand` 次小尺寸 `aclrtMemcpyAsync`（decode 场景约 96 次 256B 拷贝/层，主导 host 耗时约 1ms/层，`csrc/kernels/repeat_interleave.cpp:10-16` 注释）。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| in | 输入 | [num_tokens, nKHeads * headBytes]（按字节，token 主序、头内连续） | 任意（按字节搬运） | 输入 token×K 头矩阵 |
| out | 输出 | [num_tokens, nVHeads * headBytes] | 同 in（字节层面无需一致） | 输出 token×V 头矩阵，nVHeads == nKHeads * expand |
| numTokens | 标量 | - | uint32_t | token 数 T |
| nKHeads | 标量 | - | uint32_t | 源头数 |
| nVHeads | 标量 | - | uint32_t | 目标头数，须被 nKHeads 整除（host 侧强校验，`csrc/op.cpp:1712-1714`） |
| headBytes | 标量 | - | uint32_t | 每头字节数 = headDim * elemSize（`csrc/model.cpp:695`），上限 96KB（`csrc/op.cpp:1716-1720`） |

Python 侧无独立测试脚本；由模型层 `ForwardAttnLinear` 调用（`csrc/model.cpp:697`：`nKHeads == nVHeads` 时退化为单次 memcpy，`nVHeads % nKHeads != 0` 时报错）。

## 支持的数据类型

| 变体 | 源文件 | kernel 符号 |
|------|--------|-------------|
| 纯 cpp，dtype 无关（byte-wise 拷贝） | `csrc/kernels/repeat_interleave.cpp` | `repeat_interleave` |

host 侧不做 dtype 分派，元素大小通过 headBytes（字节数）传入；仅支持 `__DAV_C220_VEC__`。

## 实现原理

### 任务切分：按输出头段并行（`csrc/kernels/repeat_interleave.cpp:43,68`）

工作单元 = 一个输出头段 `(t, hv)`，共 `totalSegs = numTokens * nVHeads` 段。block 以步长 block_num 认领段：`for (seg = block_idx; seg < totalSegs; seg += block_num)`。每段：

- `t = seg / nVHeads`，`hv = seg % nVHeads`，源头 `kh = hv / expand`；
- 源地址 `src = t * (nKHeads*headBytes) + kh * headBytes`（同一 K 头被 expand 个输出段读取，`csrc/kernels/repeat_interleave.cpp:69-73`）；
- 目标地址 `dst = t * (nVHeads*headBytes) + hv * headBytes`。

block 数由 host 侧按段数缩放：decode（约 48 段）只起几个 block，prefill（约 24576 段）打满全部 AIV（`ConvKernelBlockNum`，tilePerCore=4096，`csrc/op.cpp:46-57,1721-1724`）。

### UB staging 与乒乓（`csrc/kernels/repeat_interleave.cpp:47-61,75-99`）

- UB 分成两半（`PINGPONG=2`，`halfBuf = UB_SIZE/2`）：`dataBuf[0]`、`dataBuf[1]`。headBytes 通常很小（≤512B），单段总能放进半个 UB；防御性地把 `segBytes` 钳到 halfBuf（`csrc/kernels/repeat_interleave.cpp:54-57`）。
- 每段数据路径：GM(src) --MTE2--> UB(dataBuf[curr]) --MTE3--> GM(dst)。段 i 读入 `dataBuf[i%2]`，同时上一段从 `dataBuf[(i-1)%2]` 写出 —— 乒乓使读写下一次段可重叠。
- 由于 MTE2 写 UB 目的地址、MTE3 读 UB 源地址都要求 block 对齐，每段固定从所在半区的偏移 0 开始（头注释 `csrc/kernels/repeat_interleave.cpp:49-52`）。

### 流水线同步

- 初始 `set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0/1)` 各一次，预置两个半区可用（`csrc/kernels/repeat_interleave.cpp:59-61`）；
- 每轮：先写上一段（`wait_flag(PIPE_MTE2, PIPE_MTE3, ID0+wbuf)` 确保 wbuf 已读完 → `CopyUbufToGmAligned` → `set_flag(PIPE_MTE3, PIPE_MTE2, ID0+wbuf)` 归还 wbuf），再读当前段（`wait_flag(PIPE_MTE3, PIPE_MTE2, ID0+curr)` 等 curr 空闲 → `CopyGmToUbufAligned` → `set_flag(PIPE_MTE2, PIPE_MTE3, ID0+curr)` 通知可写）（`csrc/kernels/repeat_interleave.cpp:75-86`）；
- 循环外 flush 最后一个 pending 段，末尾 `wait_flag(PIPE_MTE3, PIPE_MTE2, ID0/1)` + `pipe_barrier(PIPE_ALL)` 收尾（`csrc/kernels/repeat_interleave.cpp:94-104`）。
- 搬运原语 `CopyGmToUbufAligned`/`CopyUbufToGmAligned` 按字节数 32B/2B/1B 对齐自适应选择 DMA 原语（`csrc/kernels/kernel_macro.h:813-839`）。

### 边界处理

`numTokens/nKHeads/nVHeads/headBytes` 为 0、`nVHeads % nKHeads != 0`、`expand == 0` 时 kernel 直接返回（`csrc/kernels/repeat_interleave.cpp:31-40`）；host 侧对非法维度与 `headBytes > 96KB` 提前抛错（`csrc/op.cpp:1709-1720`）。
