# ring_sync

## 功能概述

`RingSync` 是一个跨核 ring 同步辅助类(仅头文件,`csrc/kernels/ring_sync.h`),供单卡内部的 attention 类算子使用,不属于卡间通信。它把所有 AIV block 按编号连成一个环:block i 完成自己的阶段后向 block i+1 发信号,并自旋等待 block i-1 的信号。典型用途是 online-softmax:当同一段 query 的连续 KV tile 被切到不同核上计算时,后一核必须拿到前一核产出的 `max/sum` 统计量才能正确做 softmax 合并,RingSync 保证了这种跨核的链式更新顺序(头文件注释 `csrc/kernels/ring_sync.h:11-15`)。

当前使用方:`flash_attention`(`csrc/kernels/flash_attention.h:51,302,327,354,377,382`)、`flash_mla_v2`(`csrc/kernels/flash_mla_v2.h:53,307,332,358,382,387`)。两处用法相同:每处理完一个 KV tile 的 softmax,若不是最后一个 tile 则 `SetNextCore()` 放行下一核;进入新 tile 的 update 阶段前 `WaitPrevCore()` 等上一核放行;kernel 结束时 `ResetPrevCore()` 清零信号。

## 输入输出参数

这是一个 C++ 辅助类,无独立 kernel 入参,无 dtype 实例化文件(以成员方式组合进宿主 kernel,模板参数 Dtype 仅沿用宿主类型)。

| 成员/接口 | 说明 |
|---|---|
| `Init(GM_ADDR sync)` | 在宿主 kernel 的 `Init` 中调用。`sync` 指向 GM 上的一块 int32 数组,长度至少 `block_num * 2`(每个 block 两个信号槽,按 `blockIdx*2 + subBlockIdx` 索引,支持一个 block 内两个 subblock 各自独立同步,`csrc/kernels/ring_sync.h:37-38`)。同时记录 `nextBlockIdx = (blockIdx+1) % block_num`、`prevBlockIdx = blockIdx==0 ? block_num-1 : blockIdx-1`,并把发送/等待的 generation 初值置 1(`csrc/kernels/ring_sync.h:33-36`) |
| `SetNextCore()` | 生产者侧:向 `sync[nextBlockIdx*2 + subBlockIdx]` 写入当前 `setNextGeneration`,然后自增 |
| `WaitPrevCore()` | 消费者侧:自旋读 `sync[prevBlockIdx*2 + subBlockIdx]`(即前一核的"next"槽,恰为本核的"prev"槽),直到值 ≥ `waitPrevGeneration`,然后自增 |
| `ResetPrevCore()` | 将 prev 槽清零(供下一次 kernel 启动前复位) |

## 支持的数据类型

不适用——纯同步逻辑,信号为 GM 上的 `int32_t`。模板参数 `Dtype` 只是为了嵌入任意宿主 kernel,不参与运算。

## 实现原理

### 信号量布局与 generation 机制

每个 block 在 GM 数组 `sync` 中拥有槽位 `blockIdx*2 + subBlockIdx`。约定:**block i 的槽只由 block i 写、由 block i+1 读**——`setNextSync` 指向自己的槽,`waitPrevSync` 指向 prev 的槽(`csrc/kernels/ring_sync.h:37-38`),即"我写我的,你读我的",天然一写一读无竞争。

信号值使用单调递增的 generation(`setNextGeneration` / `waitPrevGeneration` 初值均为 1,每次成功后 `++`,`csrc/kernels/ring_sync.h:35-36`):等待条件是 `*val < waitPrevGeneration` 则继续自旋(`csrc/kernels/ring_sync.h:63`)。这样同一 kernel 内多次循环同步不会因为旧值残留而误判(与 all_gather 等 ccl 算子的 IPC flag + generation 机制同理);`ResetPrevCore()` 写 0(`csrc/kernels/ring_sync.h:67-76`)则用于跨 kernel 启动的显式复位场景。

### 读写路径与流水线同步

GM 信号读写都经由 UB 中转并显式做流水线同步:

- 写(`SetNextCore`,`csrc/kernels/ring_sync.h:41-52`):先把 generation 值写到 UB 偏移 0 处,`set_flag/wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0)` 保证标量写对 MTE3 可见,再 `copy_ubuf_to_gm_align_b16` 落到 GM,最后 `PipeBarrier<PIPE_ALL>()` 确保到达全局可见点后才返回。
- 读(`WaitPrevCore`,`csrc/kernels/ring_sync.h:54-65`):do-while 循环里反复 `copy_gm_to_ubuf_align_b16` 从 GM 拉信号值到 UB,`set_flag/wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0)` 保证 MTE2 搬运完成后标量核再比较,未达标则继续轮询。

`GetSubblockId()` 区分同 block 的两个 subblock,使半核粒度的任务也能各自维护独立信号链。

### 在 flash_attention/flash_mla_v2 中的用法

两处用法相同,以 `flash_attention.h` 为例:一个 KV tile 的 softmax 计算完成后,若后续还有 KV tile(`!lastIsLastKvTile`),`SetNextCore()` 放行下一核(`csrc/kernels/flash_attention.h:327,377`);下一核进入 update(合并前一核的 `max/sum` 统计量做 online-softmax 修正)前 `WaitPrevCore()`(`csrc/kernels/flash_attention.h:302,354`);kernel 收尾若等过 prev(`resetPrevCore` 标记),`ResetPrevCore()` 复位信号(`csrc/kernels/flash_attention.h:382`)。由此,同一段 query 跨多核的 softmax 统计量传递形成一条沿核编号回绕的环形依赖链,这正是"ring"得名的由来。
