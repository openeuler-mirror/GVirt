# reorder_moe

## 功能概述

MoE EP（专家并行）场景下的 token 行重排：在"按来源 EP rank 分组"（source-grouped）与"按专家分组"（expert-grouped）两种布局之间搬运连续的 token 行块。`forward=1` 时正向（source→expert，dispatch 后整理成本地专家连续段），`forward=0` 时逆向（expert→source，计算后还原）。搬运粒度是"来源 i × 专家 e"的连续行段，纯字节搬运用户数据，不做任何计算。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| input (in) | 输入 | [total_tokens, hidden_size] | float / float16 / bfloat16 | forward=1 时为 source-grouped 布局；forward=0 时为 expert-grouped 布局（`tests/kernels/reorder_moe.py` 的 `reorder_forward_ref`/`reorder_reverse_ref`，source-grouped 与 expert-grouped 两种布局的参考实现） |
| output (out) | 输出 | [total_tokens, hidden_size] | 同 input | 与 input 相反布局的输出；total_tokens = counts[:, localStart:localEnd].sum() |
| counts | 输入 | [moe_ep_size, n_routed_experts] | int32 | 每个来源（EP rank/DP rank）发给每个专家的 token 数；host 侧取 `counts.shape[0]`=moeEpSize、`counts.shape[1]`=nRoutedExperts（`csrc/op.cpp` 的 `XliteOpReorderMoE`，关键字 `counts.shape[0]`、`counts.shape[1]`） |
| hiddenSize | 标量 | - | uint32_t | 每行元素数（不含 dtype 字节数） |
| localStart / localEnd | 标量 | - | uint32_t | 本卡负责的专家区间 [localStart, localEnd)，只有这些专家的行参与重排 |
| forward | 标量 | - | uint32_t | 1=source→expert（scatter），0=expert→source（gather）；host 侧由 bool 转换（`csrc/op.cpp` 的 `XliteOpReorderMoE`，关键字 `forward ? 1 : 0`） |
| elemBytes | 标量 | - | uint32_t | 元素字节数，host 侧按 in.dtype 计算（`csrc/op.cpp` 的 `XliteOpReorderMoE`，关键字 `XDtypeBit(in.dtype) / 8`） |

Python 调用方式（`tests/kernels/reorder_moe.py` 中 `reorder_moe(rt, inp, out_fwd, counts, HIDDEN_SIZE, local_start, local_end, True)` 调用）。

## 支持的数据类型

| 变体 | 源文件 | kernel 符号 |
|------|--------|-------------|
| 纯 cpp，dtype 无关（按 elemBytes 字节搬运） | `csrc/kernels/reorder_moe.cpp` | `reorder_moe` |

host 侧不做 dtype 限制，dtype 通过 `elemBytes` 传入；测试覆盖 float / float16 / bfloat16（`tests/kernels/reorder_moe.py` 的 `DTYPE_LIST`，关键字 `torch.float, torch.float16, torch.bfloat16`）。仅支持 `__DAV_C220_VEC__`。

## 实现原理

### 偏移表计算（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `chunkOffsets`、`expertOffsets`）

先把整张 counts（`moe_ep_size * n_routed_experts` 个 int32）从 GM 搬入 UB（`countsBuf`，`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `copy_gm_to_ubuf(countsBuf`），然后标量累计出两张偏移表：

- `chunkOffsets[i]`：source-grouped 布局中来源 i 的行段起始（对每个 i 累加 `counts[i, localStart..localEnd)`，`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `chunkOffsets[i] = chunkAccum`）；
- `expertOffsets[e-localStart]`：expert-grouped 布局中专家 e 的行段起始（对每个 e 累加 `counts[*, e]`，`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `expertOffsets[e - localStart] = running`）。

### 任务切分（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `pairStart`、`pairEnd`、`numPairs`）

工作单元是 `(来源 i, 专家 e)` 对，共 `numPairs = moeEpSize * nLocalExperts` 个。pair 按块均分（带余数补偿）：每个 block 负责一段连续的 `[pairStart, pairEnd)`。所有 block 都遍历全部 pair 来推进偏移量，但只对自己分到的 pair 执行实际搬运（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `if (p >= pairStart && p < pairEnd)`）。

### 数据搬运（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `copy_gm_to_ubuf(dataBuf`、`copy_ubuf_to_gm_align_b32`）

对 pair (i, e)，行数 `cnt = counts[i, e]`，`tileBytes = hiddenSize * elemBytes`：

- forward=1：源偏移 = `chunkOffsets[i]`（chunkRun），目标偏移 = `expertOffsets[e]`；
- forward=0：两者互换（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `srcOff = expertOffsets[e - localStart]`、`dstOff = chunkRun`）；
- 每个行段总字节 `cnt * tileBytes`，按 `chunkBufSize`（UB 剩余空间对半、32B 对齐）分子块：GM→UB 与 UB→GM 交替走 ping-pong 双半区（`CopyGmToUbufAligned`/`CopyUbufToGmAligned`，按字节长度自动选 b32/b16 对齐变体，见 `kernel_macro.h`），循环直至搬完；
- 搬完一个 pair 后 `expertOffsets[e] += cnt; chunkRun += cnt` 推进两张游标（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `expertOffsets[e - localStart] += cnt`、`chunkRun += cnt`），语义与测试参考实现一致（`tests/kernels/reorder_moe.py` 的 `reorder_forward_ref`，关键字 `expert_offsets[e] += cnt`、`chunk_run += cnt`）。

### 流水线同步

数据搬运采用双缓冲 ping-pong 协议（与 `split.cpp` 同款）：`dataBuf` 对半分为两个暂存半区，子块 k 读入 `dataBuf[k % 2]`，上一个子块延迟一拍从 `dataBuf[(k-1) % 2]` 写回，使 MTE2（GM→UB 读入）与 MTE3（UB→GM 写回）重叠执行，而非单缓冲下的交替串行。延迟写回跨 pair 边界同样成立：input 只读、output 只写（两个不相交的 GM 张量），不同 pair 的读写之间无 hazard，事件 flag 仅用于保护 UB 暂存半区的复用（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `EVENT_ID0 + wbuf`、`EVENT_ID0 + curr`）。循环前用两个 `set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0/1)` 预置种子，收尾时冲刷最后一个待写子块并等待两个半区排空（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0)`）。GM 偏移（`srcByteOff`/`dstByteOff`/`bo`）均为 64 位，避免大张量下的截断。

### 边界与空段处理

- input 与 output 必须是两个独立张量（不支持 in-place）：流水协议让后一个行段的 GM 读入与前一个行段的写回并发，若两者指向同一 GM 区域会读到未写完的数据；所有现有调用方（`csrc/model.cpp` 的 dispatch/combine 两处）均传入独立分配的张量；
- `numPairs == 0`（本卡无本地专家）或 `pairStart >= numPairs`（block 多于 pair）时直接返回（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `if (numPairs == 0)`、`if (pairStart >= numPairs)`）；
- `cnt == 0` 的 pair 跳过搬运但照常推进游标（`csrc/kernels/reorder_moe.cpp` 的 `reorder_moe_kernel`，关键字 `if (cnt == 0)`）；
- host 侧 `in.numel == 0 || localStart >= localEnd` 时直接返回不 launch（`csrc/op.cpp` 的 `XliteOpReorderMoE`，关键字 `in.numel == 0`、`localStart >= localEnd`）。

### 片上缓冲分配与占用汇总(AIV UB)

`reorder_moe_kernel`(`csrc/kernels/reorder_moe.cpp`)无 Init,进入 kernel 后从 UB 偏移 0 顺序排布三个偏移/计数表,余下空间对半分为数据中转乒乓缓冲 `dataBuf[2]`(`chunkBufSize = ROUND_DOWN((UB_SIZE - off)/2, 32B)`,让下一子块的 MTE2 装入与上一子块的 MTE3 写回重叠)——UB 占用按定义恒为 100%。各段按 `VECTOR_MAX_BYTESIZE`(256B)向上对齐,垫片按槽位字节计。dtype 记法:kernel 以 uint8 字节视角按 `elemBytes` 搬运,UB 布局与 dtype 无关,单占用列。硬件容量基准(910B/dav C220):UB 192KB(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`)。典型模型配置:GLM-5.x 类 MoE,EP=16(moe_ep_size=16)、n_routed_experts=160、本地专家区间 10 个(nLocalExperts=10)。

**AIV UB(容量 192KB,每 block 一份)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `countsBuf` | (moe_ep_size, n_routed_experts) | int32 | 1(串行,事件 EVENT_ID0) | `ROUND_UP(moe_ep_size*n_routed_experts*4, VECTOR_MAX_BYTESIZE)` B | 10240 |
| `expertOffsets` | nLocalExperts | int32 | 1 | `ROUND_UP(nLocalExperts*4, VECTOR_MAX_BYTESIZE)` B | 256 |
| `chunkOffsets` | moe_ep_size | int32 | 1 | `ROUND_UP(moe_ep_size*4, VECTOR_MAX_BYTESIZE)` B | 256 |
| `dataBuf[2]` | 剩余 UB 的一半(uint8 流水段) | uint8 | 2(事件 EVENT_ID0/1) | `ROUND_DOWN((UB_SIZE-off)/2, 32B)` B | 185856(两半各 92928) |
| **合计** | | | | | **196608** |

**占用汇总**(字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| UB(每 block) | 192 KB | 196608 B | 100% |

要点:

- **UB 恒占满**:`dataBuf[2]` 的定义即"偏移表之后的剩余全部空间对半",始终是绝对大头(185856 B,占 94.5%);每半 `chunkBufSize` 同时是 GM↔UB 分块搬运的块长,行段总字节再大也只是多循环几轮;
- 三个偏移表槽位(`countsBuf`/`expertOffsets`/`chunkOffsets`,合计 10752 B)随 moe_ep_size×n_routed_experts 与 nLocalExperts 线性增长,kernel 以 `assert(off <= UB_SIZE)` 拦截偏移表越界(否则剩余空间下溢 uint32 使搬运越界,debug 构建生效)、`assert(chunkBufSize >= BLOCK_SIZE)` 保证暂存半区非空,host 无等价校验;典型 MoE 规模(16×160 int32 = 10KB)距 192KB 很远,靠规模兜底;
- 数据通路是双缓冲乒乓(split.cpp 协议,事件 EVENT_ID0/1):子块 k 装入 `dataBuf[k%2]`、隔一个子块后从 `dataBuf[(k-1)%2]` 写回,MTE2 装入与 MTE3 写回重叠;写回跨 pair 边界延迟(input 只读、output 只写、GM 不相交,事件标志独自守护 UB 复用,见上文「数据流」)。
