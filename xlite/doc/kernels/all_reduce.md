# all_reduce

## 功能概述

AllReduce(SUM)集合通信算子:所有 rank 各持有一份同形状输入,通信完成后每个 rank 的输出等于所有 rank 输入的逐元素和。实现在单机多卡上采用"reduce-scatter + all-gather"两阶段的 IPC 直通方案:各 rank 通过 NPU 间 IPC 映射内存直接读写对端显存,先把总数据按 rank 分块、每 rank 把自己负责的块累加成完整和,再把各块的和中继回所有 rank。语义等价于 `torch.distributed.all_reduce(op=SUM)`。

## 输入输出参数

Python 侧调用:`all_reduce(rt, z, x, comm_type)`(`tests/kernels/all_reduce.py:53`),host 侧 launch 见 `csrc/op.cpp:360`(`XliteOpAllReduceSum`)。

kernel 签名(`csrc/kernels/all_reduce.cpp:367`):

```cpp
allreduce_<dtype>(GM_ADDR input, GM_ADDR output, uint64_t count, uint32_t rankId,
                  uint32_t rankSize, uint64_t generation, GM_ADDR param,
                  uint32_t copySize, bool fetchOffset)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input | 输入 | `[N]`(逻辑上任意形状,N = count 个元素) | FP16 / BF16 / INT8 / INT32 / FP32 | 本 rank 输入。要求 `in.numel == out.numel` 且 dtype 一致(`csrc/op.cpp:363`) |
| output | 输出 | `[N]` | 同 input | 全 rank 求和结果。`skipMyRank`(input == output,in-place)时省去自身块的先写(`csrc/kernels/all_reduce.cpp:45`) |
| count | 标量 | - | uint64 | 元素总数 `in.numel`,被切成 rankSize 块 |
| rankId / rankSize | 标量 | - | uint32 | 通信域内编号与总 rank 数(TP 取 `rankId % tpSize`,DP 取 `rankId / tpSize`,`csrc/op.cpp:374`) |
| generation | 标量 | - | uint64 | 通信代数,host 侧每次调用递增(`csrc/op.cpp:455`),用于 IPC flag 单调比较 |
| param | 输入 | - | `XcclParam` | 所有 rank 的 `ipcMems` / `ipcXTensorMems` 基址(`csrc/kernels/kernel_param.h:20`) |
| copySize | 标量 | - | uint32 | 每次 UB 搬运的目标字节数,默认 `COPY_SIZE`(32768),host 侧按核数调整(`csrc/op.cpp:426-428`) |
| fetchOffset | 标量 | - | bool | true 时通过 IPC 内存发布/获取各 rank 张量偏移;DP 域恒为 true(`csrc/op.cpp:455`) |

测试 shape 约定(`tests/kernels/all_reduce.py:44`):`x`/`z` 均为 `[dim1, dim2]`(从 `[1,1]` 到 `[512,7168]`,含非对齐 37),`count = dim1*dim2`。

## 支持的数据类型

单一模板实现 `csrc/kernels/all_reduce.cpp`,统一实例化 6 个变体(`csrc/kernels/all_reduce.cpp:379-384`):

| dtype 变体 | kernel 符号 |
|---|---|
| int8_t | `allreduce_int8_t` |
| int16_t | `allreduce_int16_t` |
| int32_t | `allreduce_int32_t` |
| float16_t | `allreduce_float16_t` |
| bfloat16_t | `allreduce_bfloat16_t` |
| float | `allreduce_float` |

host 侧 dispatch FP16/BF16/INT8/INT32/FP32(`csrc/op.cpp:433-448`);INT16 有实例化但未启用,INT64 走 HCCL 回退。

## 实现原理

### 分块约定

`count` 个元素被均分为 rankSize 块:`countPerRank = DIV_ROUND_UP(count, rankSize)`,rank r 负责第 r 块 `[r*countPerRank, (r+1)*countPerRank)`,最后一 rank 的实际长度为 `countLastRank = count - countPerRank*(rankSize-1)`(`csrc/kernels/all_reduce.cpp:41,127`)。两阶段算法:

- **阶段 1(reduce-scatter)**:每 rank 先把自己的第 r 块(用 `COPY_SIZE` 大块)原样拷到自己的 output,再遍历其他 rank,从对端 input 读出同一块,用硬件原子加 `SetAtomicAdd<Dtype>` 累加到本地 output 的第 r 块上。完成后本 rank output 的第 r 块即为该块的全量和。
- **阶段 2(all-gather)**:每 rank 从其他 rank 的 output 读回它们的第 r 块,覆盖写入本地 output 对应位置,使所有 rank 都拿到完整的 `count` 个和值。

### IPC 直通通信机制

与 all_gather 共用同一套 IPC 基础设施(详见 [all_gather.md](all_gather.md)):

- host 侧 `XcclComm::Init`(`csrc/ccl.cpp:34`)通过 socket 交换 IPC 导出 key,`aclrtIpcMemImportByKey` 互相导入权重/激活大块内存(`ipcXTensorMems`)与 2MB 同步小块(`ipcMems`),打包成 `XcclParam` 传给 kernel。
- `Init`(`csrc/kernels/all_reduce.cpp:23`)中,`fetchOffset` 为 true 时 block 0 把本 rank input/output 相对 `ipcXTensorMems[myRankId]` 的偏移写入 `ipcMems[myRankId]` 头部的 `XcclIpcMemData{inputOffset, outputOffset}`(`csrc/kernels/kernel_param.h:15`)并置起始 flag 0;否则按各 rank 布局对称的假设直接用本地偏移换算远端地址(`csrc/kernels/all_reduce.cpp:110-115`)。
- rank 间握手:各核用 `WorkSplit` 均分"等待其余 rank 的 flag 0"(`csrc/kernels/all_reduce.cpp:86-89`),`WaitIpcFlag` 以 `flagValue >= generation` 为完成条件,generation 单调递增保证跨调用正确(`csrc/kernels/all_reduce.cpp:141-155`)。

### 阶段 1:自身块预写 + 原子累加

`Run`(`csrc/kernels/all_reduce.cpp:202`)先做自身块搬运:非 in-place 时(skipMyRank == false),每核取 `countCurrRank / coreNum` 的一片中经 UB ping-pong(`CopyGMtoUbuf` → `CopyUbufToGM`,非 32B 对齐尾巴走 `DataCopyPad`)从本地 input 拷到本地 output,随后 `CrossCoreSetFlag/CrossCoreWaitFlag` 确保全核完成(`csrc/kernels/all_reduce.cpp:210-236`)。

接着 `SetAtomicAdd<Dtype>()` 开启 GM 原子加(`csrc/kernels/all_reduce.cpp:237`),遍历对端 rank:每核映射到某个 `(processRankIdx, taskIdx)` 组合——rankSize > coreNum 时一核串多个 rank 段,否则 `corePerRank` 个核分摊一个 rank 段(`csrc/kernels/all_reduce.cpp:239-254`)。从对端 `inputBuf[processRankIdx]` 的第 `myRankId*countPerRank + taskOffset` 起、按 `copyCount` 一块读入 UB,再原子加写到本地 output 同位置(`csrc/kernels/all_reduce.cpp:263-277`)。由于各核只写自己 output 的第 r 块、且是原子加,无需额外锁;ping-pong 双缓冲用 `SetFlag/WaitFlag<MTE2_MTE3/MTE3_MTE2>(EVENT_ID0+i)` 流水(`csrc/kernels/all_reduce.cpp:270-276`)。

host 侧核数选择与 all_gather 相同:launch 核数上限 `rank`(每 rank 数据 ≥ `DOUBLE_AIVNUM_SIZE_BOUND` 327680 字节时翻倍),并 `ROUND_DOWN(coreNum, rank)` 整除对齐(`csrc/op.cpp:412-429`)。

### 阶段间全局同步

原子累加完成后,`SetAtomicNone()` 关闭原子,全核 `CrossCoreSetFlag/CrossCoreWaitFlag` 汇合,block 0 置完成 flag 1 = generation,各核等待自己负责的对端 rank 的 flag 1(`csrc/kernels/all_reduce.cpp:280-296`)。这一步保证阶段 2 开始前,所有 rank 的第 r 块和值都已落盘可见。

### 阶段 2:all-gather 中继

每核从对端 `outputBuf[processRankIdx]`(对端 output 的第 processRankIdx 块,注意最后一 rank 块长用 `countLastRank`)读入 UB,普通写覆盖到本地 output 的对应块(`csrc/kernels/all_reduce.cpp:298-332`)。与 all_gather 的搬运结构相同,只是源换成对端的 output。全部核收尾 `WaitFlag` 排空 ping-pong 缓冲后 `PipeBarrier<PIPE_ALL>()` 结束(`csrc/kernels/all_reduce.cpp:334-338`)。

### 回退路径

单卡(`rank <= 1`)退化为 `aclrtMemcpyAsync`(`csrc/op.cpp:388-395`);无 XcclComm、INT64 或张量不在共享池时走 HCCL `HcclAllReduce`(`csrc/op.cpp:466-469`)。in-place(input == output)时跳过自身块预写,自身贡献直接由"output 初值即 input"保证。
