# reorder_moe

## 功能概述

MoE EP（专家并行）场景下的 token 行重排：在"按来源 EP rank 分组"（source-grouped）与"按专家分组"（expert-grouped）两种布局之间搬运连续的 token 行块。`forward=1` 时正向（source→expert，dispatch 后整理成本地专家连续段），`forward=0` 时逆向（expert→source，计算后还原）。搬运粒度是"来源 i × 专家 e"的连续行段，纯字节搬运用户数据，不做任何计算。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| input (in) | 输入 | [total_tokens, hidden_size] | float / float16 / bfloat16 | forward=1 时为 source-grouped 布局；forward=0 时为 expert-grouped 布局（`tests/kernels/reorder_moe.py:41,105`） |
| output (out) | 输出 | [total_tokens, hidden_size] | 同 input | 与 input 相反布局的输出；total_tokens = counts[:, localStart:localEnd].sum() |
| counts | 输入 | [moe_ep_size, n_routed_experts] | int32 | 每个来源（EP rank/DP rank）发给每个专家的 token 数；host 侧取 `counts.shape[0]`=moeEpSize、`counts.shape[1]`=nRoutedExperts（`csrc/op.cpp:1815-1816`） |
| hiddenSize | 标量 | - | uint32_t | 每行元素数（不含 dtype 字节数） |
| localStart / localEnd | 标量 | - | uint32_t | 本卡负责的专家区间 [localStart, localEnd)，只有这些专家的行参与重排 |
| forward | 标量 | - | uint32_t | 1=source→expert（scatter），0=expert→source（gather）；host 侧由 bool 转换（`csrc/op.cpp:1820`） |
| elemBytes | 标量 | - | uint32_t | 元素字节数，host 侧按 in.dtype 计算（`csrc/op.cpp:1817`） |

Python 调用方式（`tests/kernels/reorder_moe.py:208-209`）：`reorder_moe(rt, inp, out_fwd, counts, HIDDEN_SIZE, local_start, local_end, True)`。

## 支持的数据类型

| 变体 | 源文件 | kernel 符号 |
|------|--------|-------------|
| 纯 cpp，dtype 无关（按 elemBytes 字节搬运） | `csrc/kernels/reorder_moe.cpp` | `reorder_moe` |

host 侧不做 dtype 限制，dtype 通过 `elemBytes` 传入；测试覆盖 float / float16 / bfloat16（`tests/kernels/reorder_moe.py:31`）。仅支持 `__DAV_C220_VEC__`。

## 实现原理

### 偏移表计算（`csrc/kernels/reorder_moe.cpp:43-77`）

先把整张 counts（`moe_ep_size * n_routed_experts` 个 int32）从 GM 搬入 UB（`countsBuf`，`csrc/kernels/reorder_moe.cpp:58-59`），然后标量累计出两张偏移表：

- `chunkOffsets[i]`：source-grouped 布局中来源 i 的行段起始（对每个 i 累加 `counts[i, localStart..localEnd)`，`csrc/kernels/reorder_moe.cpp:64-70`）；
- `expertOffsets[e-localStart]`：expert-grouped 布局中专家 e 的行段起始（对每个 e 累加 `counts[*, e]`，`csrc/kernels/reorder_moe.cpp:71-77`）。

### 任务切分（`csrc/kernels/reorder_moe.cpp:23-41`）

工作单元是 `(来源 i, 专家 e)` 对，共 `numPairs = moeEpSize * nLocalExperts` 个。pair 按块均分（带余数补偿）：每个 block 负责一段连续的 `[pairStart, pairEnd)`。所有 block 都遍历全部 pair 来推进偏移量，但只对自己分到的 pair 执行实际搬运（`csrc/kernels/reorder_moe.cpp:83-127`）。

### 数据搬运（`csrc/kernels/reorder_moe.cpp:99-120`）

对 pair (i, e)，行数 `cnt = counts[i, e]`，`tileBytes = hiddenSize * elemBytes`：

- forward=1：源偏移 = `chunkOffsets[i]`（chunkRun），目标偏移 = `expertOffsets[e]`；
- forward=0：两者互换（`csrc/kernels/reorder_moe.cpp:90-97`）；
- 每个行段总字节 `cnt * tileBytes`，按 `dataBufSize`（UB 减去三个偏移表后剩余的全部空间）分块：GM→UB（`copy_gm_to_ubuf`，按 BLOCK_SIZE=32B 块数搬运）→ UB→GM（`copy_ubuf_to_gm_align_b32`，按字节长度对齐写回），循环直至搬完；
- 搬完一个 pair 后 `expertOffsets[e] += cnt; chunkRun += cnt` 推进两张游标（`csrc/kernels/reorder_moe.cpp:123-125`），语义与测试参考实现一致（`tests/kernels/reorder_moe.py:72-84`）。

### 流水线同步

分块拷贝循环内用 `PIPE_MTE3 → PIPE_MTE2` 的 EVENT_ID0 乒乓：先 `set_flag` 预置，每次 `wait_flag` 确保上一块 UB→GM 已完成后才把新数据读入同一 `dataBuf`，读入完成（`PIPE_MTE2 → PIPE_MTE3`）后再写回（`csrc/kernels/reorder_moe.cpp:104-120`）。这是单缓冲 + flag 的串行复用（同一 dataBuf 分块），保证 GM→UB→GM 顺序执行不冲突。

### 边界与空段处理

- `numPairs == 0`（本卡无本地专家）或 `pairStart >= numPairs`（block 多于 pair）时直接返回（`csrc/kernels/reorder_moe.cpp:25-26,40-41`）；
- `cnt == 0` 的 pair 跳过搬运但照常推进游标（`csrc/kernels/reorder_moe.cpp:86-88`）；
- host 侧 `in.numel == 0 || localStart >= localEnd` 时直接返回不 launch（`csrc/op.cpp:1811-1813`）。
