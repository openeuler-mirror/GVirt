# all_reduce

## 功能概述

AllReduce(SUM)集合通信算子:所有 rank 各持有一份同形状输入,通信完成后每个 rank 的输出等于所有 rank 输入的逐元素和。实现在单机多卡上采用"reduce-scatter + all-gather"两阶段的 IPC 直通方案:各 rank 通过 NPU 间 IPC 映射内存直接读写对端显存,先把总数据按 rank 分块、每 rank 把自己负责的块累加成完整和,再把各块的和中继回所有 rank。语义等价于 `torch.distributed.all_reduce(op=SUM)`。

## 输入输出参数

Python 侧调用:`all_reduce(rt, z, x, comm_type)`(`tests/kernels/all_reduce.py` 中 `all_reduce(rt, z, x, comm_type)` 调用),host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpAllReduceSum`。

kernel 签名(`csrc/kernels/all_reduce.cpp` 的 `allreduce_##dtype` 入口,由宏 `ALLREDUCE_FUNC_DEFINE` 定义):

```cpp
allreduce_<dtype>(GM_ADDR input, GM_ADDR output, uint64_t count, uint32_t rankId,
                  uint32_t rankSize, uint64_t generation, GM_ADDR param,
                  uint32_t copySize, bool fetchOffset)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
| --- | --- | --- | --- | --- |
| input | 输入 | `[N]`(逻辑上任意形状,N = count 个元素) | FP16 / BF16 / INT8 / INT32 / FP32 | 本 rank 输入。要求 `in.numel == out.numel` 且 dtype 一致(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中张量校验) |
| output | 输出 | `[N]` | 同 input | 全 rank 求和结果。`skipMyRank`(input == output,in-place)时省去自身块的先写(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Init`,关键字 `this->skipMyRank = input == output`) |
| count | 标量 | - | uint64 | 元素总数 `in.numel`,被切成 rankSize 块 |
| rankId / rankSize | 标量 | - | uint32 | 通信域内编号与总 rank 数(TP 取 `rankId % tpSize`,DP 取 `rankId / tpSize`,`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中 TP/DP 分支) |
| generation | 标量 | - | uint64 | 通信代数,host 侧每次调用递增(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中 `launchKernel` 调用,关键字 `xcclComm->generation++`),用于 IPC flag 单调比较 |
| param | 输入 | - | `XcclParam` | 所有 rank 的 `ipcMems` / `ipcXTensorMems` 基址(`csrc/kernels/kernel_param.h` 的 `struct XcclParam`) |
| copySize | 标量 | - | uint32 | 每次 UB 搬运的目标字节数,默认 `COPY_SIZE`(32768),host 侧按核数调整(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中 `copySize` 调整段) |
| fetchOffset | 标量 | - | bool | true 时通过 IPC 内存发布/获取各 rank 张量偏移;DP 域恒为 true(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中 `launchKernel` 调用,关键字 `fetchOffset \|\| type == DP`) |

测试 shape 约定(`tests/kernels/all_reduce.py` 中 shape 循环,关键字 `for dim1, dim2 in zip([1, 1, 2, 4, 8, 32, 512]`):`x`/`z` 均为 `[dim1, dim2]`(从 `[1,1]` 到 `[512,7168]`,含非对齐 37),`count = dim1*dim2`。

## 支持的数据类型

单一模板实现 `csrc/kernels/all_reduce.cpp`,统一实例化 6 个变体(`csrc/kernels/all_reduce.cpp` 末尾的 `ALLREDUCE_FUNC_DEFINE` 实例化列表):

| dtype 变体 | kernel 符号 |
| --- | --- |
| int8_t | `allreduce_int8_t` |
| int16_t | `allreduce_int16_t` |
| int32_t | `allreduce_int32_t` |
| float16_t | `allreduce_float16_t` |
| bfloat16_t | `allreduce_bfloat16_t` |
| float | `allreduce_float` |

host 侧 dispatch FP16/BF16/INT8/INT32/FP32(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中 dtype dispatch `switch (in.dtype)`);INT16 有实例化但未启用,INT64 走 HCCL 回退。

## 实现原理

### 分块约定

`count` 个元素被均分为 rankSize 块:`countPerRank = DIV_ROUND_UP(count, rankSize)`,rank r 负责第 r 块 `[r*countPerRank, (r+1)*countPerRank)`,最后一 rank 的实际长度为 `countLastRank = count - countPerRank*(rankSize-1)`(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Init` 中 `countPerRank` / `countLastRank` 计算)。两阶段算法:

- **阶段 1(reduce-scatter)**:每 rank 先把自己的第 r 块(用 `COPY_SIZE` 大块)原样拷到自己的 output,再遍历其他 rank,从对端 input 读出同一块,用硬件原子加 `SetAtomicAdd<Dtype>` 累加到本地 output 的第 r 块上。完成后本 rank output 的第 r 块即为该块的全量和。
- **阶段 2(all-gather)**:每 rank 从其他 rank 的 output 读回它们的第 r 块,覆盖写入本地 output 对应位置,使所有 rank 都拿到完整的 `count` 个和值。

### IPC 直通通信机制

与 all_gather 共用同一套 IPC 基础设施(详见 [all_gather.md](all_gather.md)):

- host 侧 `XcclComm::Init`(`csrc/ccl.cpp` 的 `XcclComm::Init`)通过 socket 交换 IPC 导出 key,`aclrtIpcMemImportByKey` 互相导入权重/激活大块内存(`ipcXTensorMems`)与 2MB 同步小块(`ipcMems`),打包成 `XcclParam` 传给 kernel。
- `Init`(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Init`)中,`fetchOffset` 为 true 时 block 0 把本 rank input/output 相对 `ipcXTensorMems[myRankId]` 的偏移写入 `ipcMems[myRankId]` 头部的 `XcclIpcMemData{inputOffset, outputOffset}`(`csrc/kernels/kernel_param.h` 的 `struct XcclIpcMemData`)并置起始 flag 0;否则按各 rank 布局对称的假设直接用本地偏移换算远端地址(`AllReduce::Init` 中 `else` 分支换算远端地址段)。
- rank 间握手:各核用 `WorkSplit` 均分"等待其余 rank 的 flag 0"(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Init` 中 `WorkSplit` 等待 flag 0 段),`WaitIpcFlag` 以 `flagValue >= generation` 为完成条件,generation 单调递增保证跨调用正确(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::WaitIpcFlag`)。

### 阶段 1:自身块预写 + 原子累加

`Run`(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Run`)先做自身块搬运:非 in-place 时(skipMyRank == false),每核取 `countCurrRank / coreNum` 的一片中经 UB ping-pong(`CopyGMtoUbuf` → `CopyUbufToGM`,非 32B 对齐尾巴走 `DataCopyPad`)从本地 input 拷到本地 output,随后 `CrossCoreSetFlag/CrossCoreWaitFlag` 确保全核完成(`AllReduce::Run` 中 `if (!skipMyRank)` 自身块预写段)。

接着 `SetAtomicAdd<Dtype>()` 开启 GM 原子加(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Run` 中 `SetAtomicAdd<Dtype>()` 段),遍历对端 rank:每核映射到某个 `(processRankIdx, taskIdx)` 组合——rankSize > coreNum 时一核串多个 rank 段,否则 `corePerRank` 个核分摊一个 rank 段(`AllReduce::Run` 中 `rankPerCore` / `corePerRank` 分工段)。从对端 `inputBuf[processRankIdx]` 的第 `myRankId*countPerRank + taskOffset` 起、按 `copyCount` 一块读入 UB,再原子加写到本地 output 同位置(`AllReduce::Run` 中对端 rank 原子加搬运循环)。由于各核只写自己 output 的第 r 块、且是原子加,无需额外锁;ping-pong 双缓冲用 `SetFlag/WaitFlag<MTE2_MTE3/MTE3_MTE2>(EVENT_ID0+i)` 流水(`AllReduce::Run` 中同一搬运循环的 ping-pong flag,关键字 `MTE2_MTE3`(EVENT_ID0 + curr)`)。

host 侧核数选择与 all_gather 相同:launch 核数上限 `rank`(每 rank 数据 ≥ `DOUBLE_AIVNUM_SIZE_BOUND` 327680 字节时翻倍),并 `ROUND_DOWN(coreNum, rank)` 整除对齐(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中核数选择段)。

### 阶段间全局同步

原子累加完成后,`SetAtomicNone()` 关闭原子,全核 `CrossCoreSetFlag/CrossCoreWaitFlag` 汇合,block 0 置完成 flag 1 = generation,各核等待自己负责的对端 rank 的 flag 1(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Run` 中阶段间全局同步段,关键字 `SetIpcFlag(1, generation)`)。这一步保证阶段 2 开始前,所有 rank 的第 r 块和值都已落盘可见。

### 阶段 2:all-gather 中继

每核从对端 `outputBuf[processRankIdx]`(对端 output 的第 processRankIdx 块,注意最后一 rank 块长用 `countLastRank`)读入 UB,普通写覆盖到本地 output 的对应块(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Run` 中 all-gather 中继段)。与 all_gather 的搬运结构相同,只是源换成对端的 output。全部核收尾 `WaitFlag` 排空 ping-pong 缓冲后 `PipeBarrier<PIPE_ALL>()` 结束(`AllReduce::Run` 末尾 ping-pong 排空与 `PipeBarrier<PIPE_ALL>()` 段)。

### 回退路径

单卡(`rank <= 1`)退化为 `aclrtMemcpyAsync`(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中 `rank <= 1` 分支);无 XcclComm、INT64 或张量不在共享池时走 HCCL `HcclAllReduce`(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 末尾 HCCL 回退,关键字 `HcclAllReduce`)。in-place(input == output)时跳过自身块预写,自身贡献直接由"output 初值即 input"保证。

### 片上缓冲分配与占用汇总(AIV UB)

`Init`(`csrc/kernels/all_reduce.cpp` 的 `AllReduce::Init`)手动排布全部片上缓冲:UB 上自偏移 0 起顺序累进,先给 `flagBuf` 一个 `UB_BUF_ALIGN_SIZE`(32B)对齐槽(实际只用 4 字节信号值),随后 `ubBuf[PINGPONG_BUF_NUM]` 两槽连续排布、每槽固定 `COPY_SIZE`(32768B);`inputBuf` / `outputBuf` / `ipcFlagBuf` 是 GM 侧 GlobalTensor,不占片上。硬件容量基准(910B/dav C220,NpuArch 2201):AIV UB 取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),kernel 为 `KERNEL_TYPE_MIX_AIV_1_0`(一核一 block),每个 AIV block 各持一份独立 UB、布局相同。

dtype 记法(模板实参 `Dtype`,见 `ALLREDUCE_FUNC_DEFINE` 实例化列表):`int8_t`/`int16_t`/`int32_t`/`float16_t`/`bfloat16_t`/`float`。UB 槽位大小是字节常量(`COPY_SIZE`、`UB_BUF_ALIGN_SIZE`),与 `Dtype` 无关,故单占用列;运行期入参 `copySize` 只决定每块搬运元素数 `copyCount = ROUND_DOWN(copySize, 32)/sizeof(Dtype)`,不改变槽位——host 侧默认即上限 `COPY_SIZE`,仅在 `corePerRank > 1` 且 `copySize*corePerRank > MAX_TOTAL_COPY_SIZE` 时向下钳到 `MAX_TOTAL_COPY_SIZE/corePerRank`(`csrc/op.cpp` 的 `XliteOpAllReduceSum` 中 `copySize` 调整段)。下表占用按典型配置代入:TP=8(rankSize=8)、BF16、每次通信张量 `[8192, 4096]`(count=33554432,DeepSeek-V4 TP 域 hidden 全量求和规模)——占用与 rankSize/count/copySize 均无关,数值恒定。

**AIV UB(容量 192KB,每 block 一份)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `flagBuf` | 1 个 uint32(`SetIpcFlag`/`WaitIpcFlag` 的 4B 信号中转) | uint32 | 1(无乒乓,`PipeBarrier<PIPE_ALL>` 串行复用) | `UB_BUF_ALIGN_SIZE` = 32 B | 32 |
| `ubBuf` | `COPY_SIZE/sizeof(Dtype)` 个元素(BF16 时 16384) | Dtype | 2(事件 EVENT_ID0/1,MTE3_MTE2↔MTE2_MTE3) | `COPY_SIZE` = 32768 B | 65536 |
| **合计** | | | | | **65568** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用(B) | 占比 |
|---|---|---|---|
| UB(每 AIV block) | 192 KB | 65568 | 33.3% |

要点:

- UB 是唯一使用的片上存储级,占用恒 65568 B(33.3%,余量 131040 B),与 dtype、rankSize、count、运行期 `copySize` 均无关——本算子的压力全部落在 GM 间 IPC 通道与原子加上,片上缓冲极其宽松;
- `ubBuf` 两份 32KB 乒乓(阶段 1 自身块预写、对端原子累加、阶段 2 中继共用同一对槽)是绝对大头,占 UB 占用的 99.95%;
- `flagBuf` 实际只用 4 字节信号值,却按 `UB_BUF_ALIGN_SIZE`(32B)对齐占一整槽;
- kernel 侧无容量断言;host 对 `copySize` 的钳制(`MAX_TOTAL_COPY_SIZE=12800`,`csrc/kernels/kernel_param.h` 的 `MAX_TOTAL_COPY_SIZE`)约束的是 `copySize*corePerRank` 的搬运粒度预算而非 UB 容量,且 host 从不调大 `copySize`(默认即 `COPY_SIZE` 上限),UB 槽位恒不被写溢。
