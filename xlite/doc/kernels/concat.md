# concat

## 功能概述

通用字节级拼接：把最多 8 个输入张量的原始字节顺序拼成一个连续输出缓冲区（一维 flat concat，即 torch.cat(dim=0) 的字节版）。kernel 按"packet（行）× 列宽"的统一模型实现，`concat` 是 `numPackets=1` 的退化情形；列拼接版本见 `concat_col`（同文件导出）。纯字节搬运，dtype 无关，不同 dtype/shape 的输入可混合拼包，只要输出字节数等于输入字节数之和。单次 kernel launch 取代逐 tensor 的 memcpy 序列（MoE packedSend 打包路径）。

## 输入输出参数

以 `concat`（flat）入口为准（`csrc/kernels/concat.cpp:140-151`）：

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| in0..in7 (inputs) | 输入 | 任意（取字节数 s_i = tensor.bytes） | 任意 | 最多 8 个（`XLITE_CONCAT_SPLIT_MAX_INPUTS = 8`，`csrc/kernels/concat.cpp:9-11`）；不足 8 个时多余指针/size 为 0 |
| out | 输出 | 一维（totalBytes 字节） | 任意 | out.bytes 必须等于 Σ s_i（host 侧校验，`csrc/_C.cpp:2157-2167`） |
| s0..s7 | 标量 | - | uint64_t | 各输入的字节数 |
| nInputs | 标量 | - | uint32_t | 实际输入个数 |
| totalBytes | 标量 | - | uint64_t | Σ s_i，即 flat 输出总字节数 |

统一模型（`concat_kernel`，`csrc/kernels/concat.cpp:25-30`）中每个输入视为 `[numPackets, sizes[i]]`（sizes[i] = 每行字节数），输出为 `[numPackets, totalSize]`（totalSize = Σ sizes[i]），语义：`out[pkt, inOff[i] : inOff[i]+sizes[i]] = inputs[i][pkt, :]`。flat concat 即 numPackets=1、pkt 恒为 0、源地址退化为平铺的 inOff（`csrc/kernels/concat.cpp:146-150` 注释）。

Python 调用方式（`tests/kernels/concat.py:62`）：`concat(rt, inputs, out)`；列拼接 `concat_col(rt, inputs, out)`（`tests/kernels/concat_col.py:20`）。

## 支持的数据类型

| 变体 | 源文件 | kernel 符号 |
|------|--------|-------------|
| 纯 cpp，dtype 无关（byte-wise 拷贝） | `csrc/kernels/concat.cpp` | `concat`（flat）、`concat_col`（按行/列） |

host 侧不做 dtype 分派；测试覆盖 float16/bfloat16/float32/int32/int8 及混合 dtype 字节拼包（`tests/kernels/concat.py:40,74-102`）。仅支持 `__DAV_C220_VEC__`。

## 实现原理

以下针对统一 kernel `concat_kernel`（`csrc/kernels/concat.cpp:25-138`），`concat`/`concat_col` 都只是它的封装。

### 列偏移表与任务切分

- `inOff[i]`：输入 i 在输出行内的起始列（字节）偏移，前缀和计算（`csrc/kernels/concat.cpp:44-47`）；
- 总工作量 = `totalBytes = numPackets * totalSize`（输出总字节数）。block 以 `segBufSize`（半个 UB）为步长跨步切分平坦的 `[0, totalBytes)` 区间：`for (bo = block_idx * segBufSize; bo < totalBytes; bo += block_num * segBufSize)`（`csrc/kernels/concat.cpp:77-78`），保证不论 packet/输入多少所有核均分数据（头注释：替代旧的按 `numPackets*nInputs` 分任务在少量 job 时闲置大量核的做法，见 split 同源说明）。
- block 数由 host 侧按总字节缩放（`CopyKernelBlockNum`，tilePerCore≈2MB：小传输只起 1 个 block 避免多核 launch+同步开销，大传输打满 AIV，`csrc/op.cpp:28-39,1498`）。

### 子段分解与乒乓搬运（`csrc/kernels/concat.cpp:79-125`）

每个 block 处理 `[bo, bo+remain)` 区间，按输入边界切成子段：

1. 把平坦输出偏移 `segDstOff` 分解为 `pkt = segDstOff / totalSize`（行号）、`inPkt = segDstOff % totalSize`（行内列偏移）；
2. 线性查找覆盖 inPkt 的输入 idx（`while (idx+1 < nInputs && inPkt >= inOff[idx+1]) idx++`）；
3. 本次取 `take = min(remain, sizes[idx] - localOff, segBufSize)` 字节，源地址为 `inputs[idx] + pkt * sizes[idx] + localOff`（跨行 strided 读，`csrc/kernels/concat.cpp:86-96,112-114`）；
4. 乒乓：先把上一子段从 `dataBuf[1-curr]` 写出到输出（连续目标地址 `out + pendDstOff`），再把本子段读入 `dataBuf[curr]` —— 两半 UB 交替，读下一段与写上一段重叠（`csrc/kernels/concat.cpp:101-115`）；
5. 循环结束 flush 最后一个 pending 子段（`csrc/kernels/concat.cpp:128-133`）。

每个子段固定从所在半区的偏移 0 开始，因为 MTE2 写 UB 目的、MTE3 读 UB 源都要求 block 对齐（`csrc/kernels/concat.cpp:49-53` 注释）。

### 流水线同步

初始各 `set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0/1)` 一次预置两个半区可用（`csrc/kernels/concat.cpp:64-66`）；每轮"写上一段（wait MTE2→MTE3 + CopyUbufToGmAligned + set MTE3→MTE2）/ 读当前段（wait MTE3→MTE2 + CopyGmToUbufAligned + set MTE2→MTE3）"用对应 EVENT_ID 握手；末尾 `wait_flag` ID0/1 + `pipe_barrier(PIPE_ALL)` 收尾（`csrc/kernels/concat.cpp:135-137`）。搬运原语按字节数 32B/2B/1B 对齐自适应（`csrc/kernels/kernel_macro.h:813-839`），因此任意奇数字节尾段（如 bf16 奇数宽度、int8 尾巴）都能正确搬运。

### 边界处理与 host 侧校验

- `nInputs == 0 || numPackets == 0 || totalSize == 0` 直接返回（`csrc/kernels/concat.cpp:36-38`）；
- 超过 8 个输入时 host 侧回退为逐 tensor 的 `aclrtMemcpyAsync`（`csrc/op.cpp:1506-1514`）；
- Python 侧校验 `out.bytes == Σ inputs.bytes`，不符抛错（`csrc/_C.cpp:2157-2167`）；`concat_col` 额外校验各输入 dtype 与前导维一致、out 为 `[times, Σ lastDim]`（`csrc/op.cpp:1521-1539`）。
