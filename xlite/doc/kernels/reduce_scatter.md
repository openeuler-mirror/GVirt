# reduce_scatter

## 功能概述

ReduceScatter(SUM)集合通信算子:每个 rank 持有形状相同的完整输入张量(`[rankSize * N]`),通信完成后按 rank 顺序切分为 rankSize 块,rank r 的输出(第 r 块,`[N]`)等于所有 rank 输入第 r 块的逐元素和。即"先全量求和再散射"。实现与 all_reduce 的第一阶段同构:本 rank 先写出自己块,再通过 IPC 直通内存从其他 rank 读同一块并硬件原子加累加。语义等价于 `torch.distributed.reduce_scatter(op=SUM)`。

## 输入输出参数

Python 侧调用:`reduce_scatter(rt, z, y, comm_type)`(`tests/kernels/reduce_scatter.py:55`),host 侧 launch 见 `csrc/op.cpp:241`(`XliteOpReduceScatter`)。

kernel 签名(`csrc/kernels/reduce_scatter.cpp:314`):

```cpp
reduce_scatter_<dtype>(GM_ADDR input, GM_ADDR output, uint64_t count, uint32_t rankId,
                       uint32_t rankSize, uint64_t generation, GM_ADDR param,
                       uint32_t copySize, bool fetchOffset)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input | 输入 | `[rankSize * N]`(逻辑上任意形状,N = count/rankSize) | FP16 / BF16 / INT8 / INT32 / FP32 | 本 rank 的完整输入。要求 `in.numel == out.numel * rankSize`(`csrc/op.cpp:245`) |
| output | 输出 | `[N]` | 同 input | 全 rank 求和后的第 rankId 块 |
| count | 标量 | - | uint64 | 输入总元素数 `in.numel`;kernel 内 `countPerRank = DIV_ROUND_UP(count, rankSize)`(`csrc/kernels/reduce_scatter.cpp:41`) |
| rankId / rankSize | 标量 | - | uint32 | 通信域内编号与总 rank 数(TP 取 `rankId % tpSize`,DP 取 `rankId / tpSize`,`csrc/op.cpp:257`) |
| generation | 标量 | - | uint64 | 通信代数,每次调用递增(`csrc/op.cpp:342`),用于 IPC flag 单调比较 |
| param | 输入 | - | `XcclParam` | 所有 rank 的 `ipcMems` / `ipcXTensorMems` 基址(`csrc/kernels/kernel_param.h:20`) |
| copySize | 标量 | - | uint32 | 每次 UB 搬运的目标字节数,默认 `COPY_SIZE`(32768),host 侧按核数调整(`csrc/op.cpp:310-316`) |
| fetchOffset | 标量 | - | bool | true 时通过 IPC 内存发布/获取各 rank 张量偏移;DP 域恒为 true(`csrc/op.cpp:342`) |

测试 shape 约定(`tests/kernels/reduce_scatter.py:44-55`):每 rank 输入 `y = cat(x_list, dim=0)` 形状 `[dim1*world_size, dim2]`,输出 `[dim1, dim2]`,`count = dim1*world_size*dim2`。注意 kernel 把 `count` 向上取整分块,故 `count` 最好能被 rankSize 整除;否则最后一块较短(`countCurrRank` 钳位,`csrc/kernels/reduce_scatter.cpp:124-126`)。

## 支持的数据类型

单一模板实现 `csrc/kernels/reduce_scatter.cpp`,统一实例化 6 个变体(`csrc/kernels/reduce_scatter.cpp:326-331`):

| dtype 变体 | kernel 符号 |
|---|---|
| int8_t | `reduce_scatter_int8_t` |
| int16_t | `reduce_scatter_int16_t` |
| int32_t | `reduce_scatter_int32_t` |
| float16_t | `reduce_scatter_float16_t` |
| bfloat16_t | `reduce_scatter_bfloat16_t` |
| float | `reduce_scatter_float` |

host 侧 dispatch FP16/BF16/INT8/INT32/FP32(`csrc/op.cpp:320-340`);INT16 有实例化但未启用,INT64 走 HCCL 回退。

## 实现原理

### IPC 直通通信机制

与 all_gather / all_reduce 共用同一套 IPC 基础设施(详见 [all_gather.md](all_gather.md)):

- host 侧 `XcclComm::Init`(`csrc/ccl.cpp:34`)通过 socket 交换 IPC key 并互相导入两组内存:大块张量池 `ipcXTensorMems` 与 2MB 同步/偏移区 `ipcMems`,基址打包进 `XcclParam` 传给 kernel。
- `Init`(`csrc/kernels/reduce_scatter.cpp:23`)中 block 0 在 `fetchOffset` 时把本 rank input/output 相对 `ipcXTensorMems[myRankId]` 的偏移写入 `ipcMems[myRankId]` 头部 `XcclIpcMemData`(`csrc/kernels/kernel_param.h:15`)并刷 cache,随后置起始 flag 0;各核 `WorkSplit` 均摊等待其余 rank 的 flag 0(`csrc/kernels/reduce_scatter.cpp:86-89`)。
- 就绪后每核为每个远端 rank 构造 GlobalTensor:远端地址 = `ipcXTensorMems[r] + {inputOffset 或 outputOffset}`(`csrc/kernels/reduce_scatter.cpp:95-120`)。`skipMyRank`(`input + myRankId*countPerRank*sizeof(Dtype) == output`,即输出正好是输入中间一段,in-place 切分)时跳过自身块的预写(`csrc/kernels/reduce_scatter.cpp:210`)。
- `SetIpcFlag` / `WaitIpcFlag`(`csrc/kernels/reduce_scatter.cpp:130-155`)用 `DataCopyPad` 搬 4 字节 flag 做 GM 信号量,`flagValue >= generation` 判完成;generation 单调递增防止跨调用误判。

### 数据搬运:自身块预写 + 原子累加

`Run`(`csrc/kernels/reduce_scatter.cpp:202`)与 all_reduce 阶段 1 几乎相同:

1. **自身块预写**(非 skipMyRank):本 rank 输出的第 `myRankId` 块对应本地 input 的 `[offsetCurrRank, offsetCurrRank + countCurrRank)`,每核取 `countCurrRank / coreNum` 的一片,经 UB ping-pong(`CopyGMtoUbuf` → `CopyUbufToGM`,非 32B 对齐尾巴走 `DataCopyPad` 分支,`csrc/kernels/reduce_scatter.cpp:174-200`)从本地 input 搬到本地 output,完成后 `CrossCoreSetFlag/CrossCoreWaitFlag` 全核汇合(`csrc/kernels/reduce_scatter.cpp:210-237`)。
2. **原子累加**:`SetAtomicAdd<Dtype>()` 开启 GM 原子加(`csrc/kernels/reduce_scatter.cpp:238`),各核按 `(processRankIdx, taskIdx)` 分工——rankSize > coreNum 时一核串多个 rank,否则 `corePerRank = coreNum / rankSize` 个核分摊一个对端 rank(`csrc/kernels/reduce_scatter.cpp:240-247`)。对每个对端,从其 `inputBuf[processRankIdx]` 的第 `myRankId*countCurrRank + taskOffset` 起、按 `copyCount`(= `ROUND_DOWN(copySize,32)/sizeof(Dtype)`)一块读入 UB,原子加写到本地 output 的 `[taskOffset, ...)`(`csrc/kernels/reduce_scatter.cpp:249-280`)。所有 rank 对同一块的原子加在本 rank output 上汇总出全量和。

与 all_reduce 不同,reduce_scatter 没有第二阶段:结束前只做 `SetAtomicNone()` + `PipeBarrier<PIPE_ALL>()`(`csrc/kernels/reduce_scatter.cpp:282-286`),不含置/等完成 flag 1 的跨 rank 收尾握手。

host 侧核数选择:launch 核数上限 `rank`(每 rank 数据 ≥ `DOUBLE_AIVNUM_SIZE_BOUND` 327680 字节时翻倍为 `2*rank`),并 `ROUND_DOWN(coreNum, rank)` 对齐(`csrc/op.cpp:299-316`);与 kernel 内 `corePerRank = coreNum / rankSize` 的整除假设匹配。

### 回退路径

单卡退化为 `aclrtMemcpyAsync`(`csrc/op.cpp:273-280`);无 XcclComm、INT64 或张量不在共享池时走 HCCL `HcclReduceScatter`(`csrc/op.cpp:354-357`)。
