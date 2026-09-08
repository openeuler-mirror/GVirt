# add_bias

## 功能概述

对二维矩阵逐行加上偏置向量：`z[i, j] = x[i, j] + bias[j]`，用于线性层输出的 bias 相加。非 float 输入会在片内升精度到 float32 计算后舍回原 dtype；float 输入直接相加。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x（input） | 输入 | [rowNum, yNumel] | float / float16 / bfloat16 | 输入矩阵，rowNum = xNumel / yNumel |
| y（weight/bias） | 输入 | [yNumel] | float / float16 / bfloat16 | 偏置向量，每行广播相加 |
| z（output） | 输出 | [rowNum, yNumel] | 同 x | 结果，可与 x 为同一 tensor（原地更新，测试中即如此） |
| xNumel | 标量 | - | uint32_t | x 的总元素数，host 侧传 `output.shape[0] * output.shape[1]`（`csrc/op.cpp:1212-1213`） |
| yNumel | 标量 | - | uint32_t | 偏置长度/矩阵列宽，host 侧传 `output.shape[1]` |

Python 调用方式（`tests/kernels/add_bias.py:34`）：`add_bias(rt, z, bias, z)`。host 侧要求 input/weight/output 三者 dtype 一致（`csrc/op.cpp:1199-1210`）。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float | `csrc/kernels/add_bias_float.cpp` | `add_bias_float` |
| float16_t | `csrc/kernels/add_bias_float16_t.cpp` | `add_bias_float16_t` |
| bfloat16_t | `csrc/kernels/add_bias_bfloat16_t.cpp` | `add_bias_bfloat16_t` |

实现基于 Ascend C 编程模型（`KernelAddBias` 类，`csrc/kernels/add_bias.h:12`），与手写 intrinsics 的 kernel 风格不同。

## 实现原理

### 整体数据流

使用 Ascend C 的 TPipe/TQue 框架（`csrc/kernels/add_bias.h:100-106`）：

1. `Init`（`csrc/kernels/add_bias.h:19-35`）：设置 x/y/z 的 GlobalTensor；`rowNum = xNumel / yNumel`；分配输入队列 `queInX`（双缓冲 BUFFER_NUM=2）与输出队列 `queOutZ`（双缓冲）；bias 所在的 `yBuf`、以及非 float 时的 fp32 辅助缓冲 `yFp32Buf/xFp32Buf/zFp32Buf`（各 yNumel 个元素，VECCALC 位置）。
2. `Process`（`csrc/kernels/add_bias.h:36-55`）：
   - 一次性将 bias 从 GM 拷入 UB（`DataCopy`，长度按 32B 块取整：`ROUND_UP(yNumel * sizeof(T), BLOCK_SIZE) / sizeof(T)`），`PipeBarrier<PIPE_ALL>` 后把 bias 升精度为 float32（非 float 时 `Cast(..., CAST_NONE, yNumel)`），此后 bias 以 fp32 形式常驻 UB，供所有行复用；
   - 主循环 `for (loop = GetBlockIdx(); loop < rowNum; loop += GetBlockNum())`：行间由多 block 并行（block 间步长 GetBlockNum()），每行执行 CopyIn → Compute → CopyOut 三段。

### 分块与流水

- 单行（yNumel 个元素）为最小处理单元，行内不再分块；要求单行数据可放入队列 buffer；
- `queInX`/`queOutZ` 各 BUFFER_NUM=2 个 buffer，通过 EnQue/DeQue 形成 copy-in 与 compute、compute 与 copy-out 之间的双缓冲流水；
- CopyIn（`csrc/kernels/add_bias.h:58-63`）：从 `xGm[loop * yNumel]` 拷入一行，长度同样按 32B 取整；
- CopyOut（`csrc/kernels/add_bias.h:84-98`）：若 `yNumel * sizeof(T)` 是 32B 对齐则直接 `DataCopy`，否则构造 `DataCopyParams`（blockLen 为精确字节数）走 `DataCopyPad`，保证非对齐的最后一行不会被多写。

### 计算步骤（`csrc/kernels/add_bias.h:65-82`）

- float：`Add(zLocal, xLocal, yFp32Local, yNumel)` 直接在原 dtype 上逐元素加；
- fp16/bf16：`Cast` x → fp32（CAST_NONE）→ `Add` fp32 加 bias（fp32）→ `Cast` 结果回原 dtype（CAST_RINT，四舍五入到偶数），即"升精度计算、舍入写回"。

### 与 add kernel 的差异

add_bias 每行复用同一 bias 向量，因此 bias 只搬运/转换一次；且使用框架队列（TQue）做同步而非手写 set_flag/wait_flag，框架内部完成 MTE2/V/MTE3 的事件同步。

### 边界处理

- bias 拷贝长度与 x 行拷贝长度均向上取整到 32B；尾部 padding 只存在于 UB；
- 输出路径按是否 32B 对齐选择 `DataCopy` 或 `DataCopyPad`，避免非对齐 yNumel 时写越界。
