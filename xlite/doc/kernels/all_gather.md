# all_gather

## 功能概述

AllGather 集合通信算子:每个 rank 持有形状相同的输入张量,通信完成后每个 rank 的输出张量按 rank 顺序拼接了所有 rank 的输入(`out[r*count:(r+1)*count] = rank r 的输入`)。与走 HCCL 库的通用实现不同,本算子通过 NPU 间 IPC 直通内存(peer-to-peer 映射)在 kernel 内直接读写其他 rank 的显存,单机多卡(TP/DP/EP 域)场景下省去了 HCCL 框架开销,延迟更低。语义等价于 `torch.distributed.all_gather` 后按 dim 0 拼接。

## 输入输出参数

Python 侧调用:`all_gather(rt, z, x, comm_type)`(`tests/kernels/all_gather.py:54`),host 侧 launch 见 `csrc/op.cpp:85`(`XliteOpAllGather`)。

kernel 签名(`csrc/kernels/all_gather.cpp:297`):

```cpp
allgather_<dtype>(GM_ADDR input, GM_ADDR output, uint64_t count, uint32_t rankId,
                  uint32_t rankSize, uint64_t generation, GM_ADDR param,
                  uint32_t copySize, bool fetchOffset)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input | 输入 | `[N]`(逻辑上任意形状,展开后 N = count 个元素) | FP16 / BF16 / INT8 / BIT1(打包为 INT8) / INT32 / FP32 | 本 rank 的输入张量。BIT1 时 host 侧将 count 折算为打包字节数(`csrc/op.cpp:202`) |
| output | 输出 | `[rankSize * N]` | 同 input | 拼接结果,第 r 段为 rank r 的数据;要求 `out.numel == in.numel * rankSize`(`csrc/op.cpp:94`) |
| count | 标量 | - | uint64 | 单个 rank 的元素个数(`in.numel`;BIT1 为 `in.numel / 8`) |
| rankId / rankSize | 标量 | - | uint32 | 通信域内本 rank 编号与 rank 总数(TP 域取 `rankId % tpSize`,DP 域取 `rankId / tpSize`,`csrc/op.cpp:112-121`) |
| generation | 标量 | - | uint64 | 通信代数,每次调用递增(`xcclComm->generation++`,`csrc/op.cpp:223`),用于 IPC flag 的单调比较,避免跨调用残留信号 |
| param | 输入 | - | `XcclParam` | 设备侧指针,含所有 rank 的 `ipcMems` / `ipcXTensorMems` 基址(`csrc/kernels/kernel_param.h:20`) |
| copySize | 标量 | - | uint32 | 每次 UB 搬运的目标字节数,默认 `COPY_SIZE`(32768),host 侧按 `MAX_TOTAL_COPY_SIZE / corePerRank` 向下调整(`csrc/op.cpp:183-185`) |
| fetchOffset | 标量 | - | bool | 为 true 时 kernel 先把自己的 input/output 相对 `ipcXTensorMems` 的偏移发布到 IPC 内存再互相同步;host 侧在 DP 域或上层显式要求时置 true(`csrc/op.cpp:223`) |

测试中的 shape 约定(`tests/kernels/all_gather.py:44`):输入 `[dim1, dim2]`(如 `[1,1]`、`[1,37]`、`[512,7168]`,含非 32B 对齐的 37),输出 `[dim1*world_size, dim2]`,`count = dim1*dim2`,按行展开后逐 rank 拼接。

## 支持的数据类型

单一模板实现 `csrc/kernels/all_gather.cpp`,文件末尾统一实例化 6 个变体(`csrc/kernels/all_gather.cpp:309-314`):

| dtype 变体 | kernel 符号 |
|---|---|
| int8_t | `allgather_int8_t` |
| int16_t | `allgather_int16_t` |
| int32_t | `allgather_int32_t` |
| float16_t | `allgather_float16_t` |
| bfloat16_t | `allgather_bfloat16_t` |
| float | `allgather_float` |

host 侧实际 dispatch 的是 FP16/BF16/INT8/BIT1(复用 int8 变体)/INT32/FP32(`csrc/op.cpp:191-216`);INT16 有实例化但 host 侧未启用。INT64 走 HCCL 回退路径。

## 实现原理

### IPC 直通通信机制

通信建立在 host 侧 `XcclComm::Init`(`csrc/ccl.cpp:34`)准备好的两组 IPC 内存上:

1. **ipcXTensorMems**:每个 rank 通过 TCP socket(`XSock::AllGather`)交换大块共享内存(权重/激活池)的 `aclrtIpcMemGetExportKey` 导出 key,再用 `aclrtIpcMemImportByKey`(开启 `ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS`)互相导入,得到跨 rank 可直接寻址的基地址数组(`csrc/ccl.cpp:44-53`)。通信的张量落在这块内存里,各 rank 只需交换"张量相对基址的偏移"即可互访。
2. **ipcMems**:每 rank 另外分配 2MB(`XLITE_CCL_IPC_MEM_SIZE`,`csrc/ccl.cpp:14`)的小块 IPC 内存。布局约定:`[0,16)` 字节为 `XcclIpcMemData{inputOffset, outputOffset}`(`csrc/kernels/kernel_param.h:15`),`XLITE_IPC_MEM_FLAG_OFFSET`(4096)处开始是同步 flag 数组。
3. 两组基址被打包成 `struct XcclParam` 拷贝到设备,通信 kernel 通过 `param` 入参读到(`csrc/kernels/all_gather.cpp:27-35`)。

### offset 发布与 rank 间同步

`Init` 中 block 0 负责发布偏移:`fetchOffset` 为 true 时把本 rank input/output 指针相对自己 `ipcXTensorMems` 基址的偏移写入 `ipcMems[myRankId]` 的前两个 uint64,并 `DataCacheCleanAndInvalid` 刷写(`csrc/kernels/all_gather.cpp:66-74`);随后 `SetIpcFlag(0, generation)` 置起始 flag。每个核再用 `WorkSplit` 把"等待其余 rankSize-1 个 rank 的 flag 0"均分到各核(`csrc/kernels/all_gather.cpp:86-90`),避免所有核对同一 GM 地址自旋造成热点。

`SetIpcFlag` / `WaitIpcFlag`(`csrc/kernels/all_gather.cpp:131-156`)实现 GM 信号量:写侧用 `DataCopyPad` 把 4 字节 value 写到对端可见的 `ipcMems[myRankId][flagId]`;读侧在 do-while 中反复 `DataCopyPad` 回 UB 并比较 `flagValue < expectValue`——由于 generation 每次调用单调递增,即使上次调用的旧值 1 残留,也不会误判本次完成。

等待所有 rank 就绪后,每个核重建远端 GlobalTensor:从 `ipcMems[r]` 读出(或按对称布局推算)对端 input/output 偏移,加到 `ipcXTensorMems[r]` 上得到远端地址(`csrc/kernels/all_gather.cpp:95-124`)。特例:`skipMyRank`(本 rank 输入正好是输出中自己那一段,`csrc/kernels/all_gather.cpp:45`)时,远端读地址改取远端 output 基址加 `r * countPerRank`(`csrc/kernels/all_gather.cpp:116-119`),直接从对端输出缓冲读,省一次本地搬运;此时本 rank 段的复制被整体跳过(`csrc/kernels/all_gather.cpp:222`)。

### 数据搬运与多核分工

`Run`(`csrc/kernels/all_gather.cpp:203`)按 rank 与核数的关系分两种方式:

- **rankSize > coreNum**:每个核串行处理多个 rank 段(`rankPerCore = DIV_ROUND_UP(rankSize, coreNum)`);
- **rankSize <= coreNum**:每个 rank 段由 `corePerRank = coreNum / rankSize` 个核分摊,核 `coreIdx` 负责 `taskIdx = coreIdx % corePerRank` 号子任务,处理该段中 `[taskOffset, taskOffset+countPerTask)` 的元素(`csrc/kernels/all_gather.cpp:219-232`)。

host 侧对应地把 launch 核数限制为 `max(rank, rank*2)`,并 `ROUND_DOWN(coreNum, rank)` 保证整除(`csrc/op.cpp:170-186`);每 rank 数据量超过 `DOUBLE_AIVNUM_SIZE_BOUND`(327680 字节)时允许翻倍核数。

段内数据按 `copyCount = ROUND_DOWN(copySize, 32) / sizeof(Dtype)` 个元素一块,经 UB 中转:`CopyGMtoUbuf` → `CopyUbufToGM`。非 32B 对齐的尾巴(如测试中 dim2=37 的行)走 `DataCopyPad` 分支(`csrc/kernels/all_gather.cpp:175-201`)。UB 使用 `PINGPONG_BUF_NUM`(2)个 `COPY_SIZE`(32KB)缓冲做双缓冲,搬运与写出通过 `SetFlag/WaitFlag<HardEvent::MTE2_MTE3 / MTE3_MTE2>(EVENT_ID0+i)` 交替流水(`csrc/kernels/all_gather.cpp:235-250`)。

### 完成同步

数据写完后各核 `CrossCoreSetFlag/CrossCoreWaitFlag` 汇合,block 0 置 flag 1 = generation,各核再分摊等待所有对端的 flag 1(`csrc/kernels/all_gather.cpp:257-269`)。此时可保证本 rank 的 output 已全部就绪,后续算子可安全读取。

### 回退路径

无 `XcclComm`(单卡或未建立 IPC)、dtype 为 INT64、或张量不在共享池中时,host 侧先做 D2D 临时拷贝再走 kernel,或直接回退 HCCL 的 `HcclAllGather`(`csrc/op.cpp:235-238`)。`rankSize <= 1` 时退化为一次 `aclrtMemcpyAsync`(`csrc/op.cpp:143-150`)。
