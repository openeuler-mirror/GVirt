# split

## 功能概述

通用字节级拆分（deinterleave）：把一个连续输入缓冲区按"packet（包）× 段"模型拆散到最多 8 个输出张量。输入被视为 `numPackets` 个包，每包由 nOutputs 段顺序组成、第 j 段长 `sizes[j]` 字节；对包 i 和段 j：`sizes[j]` 字节从 `in + i*totalSize + outOff[j]` 拷到 `outputs[j] + i*sizes[j]`。它是 MoE packedRecv（AllGather 后）解包路径的核心，单次 kernel launch 取代逐包逐段的 memcpy 风暴。列拆分 `split_col` 也复用同一 kernel（host 侧以 numPackets=行数调用）。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| in | 输入 | [numPackets * totalSize] 字节 | 任意 | 连续输入；totalSize = Σ sizes[j] |
| out0..out7 (outputs) | 输出 | out_j 为 [numPackets * sizes[j]] 字节 | 任意 | 最多 8 个输出；`split_col` 场景为 `[rows, widths[j]]`（每行取一段） |
| s0..s7 (sizes) | 标量 | - | uint64_t | 各段字节数。flat split 由 Python 侧显式传入（字节，`tests/kernels/split.py` 的 `split(rt, in_, outputs, sizes_bytes, num_packets)` 调用）；split_col 由 host 侧按 `outputs[i].shape.back() * elemSize` 计算（`csrc/op.cpp` 的 `XliteOpSplitCol`，关键字 `s[i] = static_cast<uint64_t>(outputs[i].shape.back()) * elemSize`） |
| nOutputs | 标量 | - | uint32_t | 输出个数 |
| numPackets | 标量 | - | uint32_t | 包数；split_col 场景 = 输入的前导维乘积（行数，`csrc/op.cpp` 的 `XliteOpSplitCol`，关键字 `height = std::accumulate` / `static_cast<uint32_t>(height)`） |
| totalSize | 标量 | - | uint64_t | 单包字节数 Σ sizes[j] |

Python 调用方式：

- flat：`split(rt, in_, outputs, sizes_bytes, num_packets)`（`tests/kernels/split.py` 的 `split(...)` 调用段），sizes 为每段字节数；
- 列拆分：`split_col(rt, x, [q, k, v])`（`tests/kernels/split_col.py` 的 `split_col` 调用段），等价 `torch.split(x, width, dim=2)`。

## 支持的数据类型

| 变体 | 源文件 | kernel 符号 |
|------|--------|-------------|
| 纯 cpp，dtype 无关（byte-wise 拷贝） | `csrc/kernels/split.cpp` | `split` |

host 侧不做 dtype 分派；测试覆盖 float16/bfloat16/float32/int32/int8（`tests/kernels/split.py` 的 `run_test` 用例段）、split_col 覆盖 bf16/fp16（`tests/kernels/split_col.py` 的 `run_case` 用例段）。仅支持 `__DAV_C220_VEC__`。`concat`（`csrc/kernels/concat.cpp`）是其镜像操作。

## 实现原理

### 段偏移表与任务切分（`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `outOff[i] = outOff[i - 1] + sizes[i - 1]` 前缀和与 `for (uint64_t gbo =` 任务切分段）

- `outOff[j]`：输出 j 的段在包内的起始字节偏移（前缀和，`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `outOff[i] = outOff[i - 1] + sizes[i - 1]` 前缀和段）；
- 总工作量 = `totalBytes = numPackets * totalSize`。block 以 `segBufSize`（半个 UB）为步长跨步切分平坦源区间 `[0, totalBytes)`：`for (gbo = block_idx * segBufSize; gbo < totalBytes; gbo += block_num * segBufSize)`。头注释明确该平坦切分替代旧的 `totalJobs = numPackets * nOutputs` 分配方式（numPackets==1 时 3 个 job 摊到 48 核大量闲置，`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `totalJobs = numPackets * nOutputs` 注释段）；
- block 数由 host 侧按 `numPackets * totalSize` 字节缩放（`CopyKernelBlockNum`，tilePerCore≈2MB，`csrc/op.cpp` 的 `CopyKernelBlockNum` 定义与 `XliteOpSplitCol` / `XliteOpSplit` 中的调用点）。

### 子段分解与乒乓搬运（`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `while (remain > 0)` 子段分解段）

每个 block 处理 `[gbo, gbo+remain)` 源区间，按"输出/包边界"切子段：

1. 包内偏移 `inPkt = segGbo % totalSize`，线性查找覆盖它的输出 o（`while (o+1 < nOutputs && inPkt >= outOff[o+1]) o++`，`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `while (o + 1 < nOutputs && inPkt >= outOff[o + 1])` 查找段）；
2. `take = min(remain, sizes[o] - localOff, segBufSize)`；take 以 segBufSize 为上限的兜底仅发生在单段超过半个 UB 时（源在单个输出内仍连续，直接再切，`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `cap to UB half` 注释段）；
3. 源段是连续的：`CopyGmToUbufAligned(dataBuf[curr], in + segGbo, take)`；
4. 写出目标需要重算（挂起的 pendGbo 是平坦偏移，写时重新分解为 `(pkt', o', localOff')`，nOutputs≤8 所以代价很低）：`CopyUbufToGmAligned(outputs[o'] + pkt'*sizes[o'] + localOff', dataBuf[1-curr], pendSize)`（`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `pendGbo maps to` / `CopyUbufToGmAligned` 写出重算段）；
5. 乒乓：读当前段入 `dataBuf[curr]` 与写上一段出 `dataBuf[1-curr]` 重叠；循环外 flush 最后一段。

与 concat 相同，每个子段固定从所在半区偏移 0 开始以满足 MTE3 读 UB 源的 block 对齐要求（`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `Ping-pong UB staging` 注释段）。

### 流水线同步

初始 `set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0/1)` 预置两半区（`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `Seed both MTE3->MTE2 flags` 段）；每轮"写上一段 / 读当前段"用 `PIPE_MTE2→PIPE_MTE3`、`PIPE_MTE3→PIPE_MTE2` 的对应 EVENT_ID 握手；末尾 `wait_flag` ID0/1 + `pipe_barrier(PIPE_ALL)` 收尾（`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `pipe_barrier(PIPE_ALL)` 收尾段）。搬运原语按字节数 32B/2B/1B 对齐自适应（`csrc/kernels/kernel_macro.h` 的 `CopyGmToUbufAligned` / `CopyUbufToGmAligned`），支持任意奇数字节段。

### 边界处理与 host 侧校验

- `numPackets == 0 || nOutputs == 0 || totalSize == 0` 直接返回（`csrc/kernels/split.cpp` 的 `split_kernel`，关键字 `if (numPackets == 0 || nOutputs == 0 || totalSize == 0)` 早返回段）；
- 超过 8 个输出时 host 侧回退为逐包逐段 `aclrtMemcpyAsync`（`csrc/op.cpp` 的 `XliteOpSplit`，关键字 `Fallback for the rare > maxOutputs case` 段）；
- Python 侧校验：`outputs.size() == sizes.size()`、`totalSize * numPackets == in.bytes`、每个 `outputs[j].bytes >= sizes[j] * numPackets`（防止越界写设备内存，`csrc/_C.cpp` 的 `Split` wrapper，关键字 `outputs count` / `input bytes` / `numPackets*sizes` 校验段，测试 `tests/kernels/split.py` 的 undersized output 拒绝用例段 专门验证 undersized output 被拒绝）；`split_col` 额外校验各输出 dtype/前导维一致且列宽之和等于输入列宽（`csrc/op.cpp` 的 `XliteOpSplitCol`，关键字 `XliteOpSplitCol: outputs must share dtype` / `sum(outputs last dim)` 校验段）。
