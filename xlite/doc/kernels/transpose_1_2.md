# transpose_1_2

## 功能概述

对 3 维张量的维度 1 和维度 2 进行转置,即 `[dim0, dim1, dim2] -> [dim0, dim2, dim1]`,等价于 PyTorch 的 `input.transpose(1, 2)`。主要用于注意力计算前后 Q/K 张量的布局变换(如 `[batch, seq_len, channels]` 与 `[batch, channels, seq_len]` 之间的转换)。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input | 输入 | `[dim0, dim1, dim2]` | float16 / bfloat16 | 源张量,行主序连续存储 |
| output | 输出 | `[dim0, dim2, dim1]` | float16 / bfloat16 | 转置结果,与 input 同 dtype |
| dim0 | 标量 | - | uint32_t | 批次大小(不参与转置) |
| dim1 | 标量 | - | uint32_t | 输入的第 1 维长度 |
| dim2 | 标量 | - | uint32_t | 输入的第 2 维长度 |

## 支持的数据类型

- `float16_t`([transpose_1_2_float16_t.cpp](../../csrc/kernels/transpose_1_2_float16_t.cpp))
- `bfloat16_t`([transpose_1_2_bfloat16_t.cpp](../../csrc/kernels/transpose_1_2_bfloat16_t.cpp))

## 实现原理

实现位于 [csrc/kernels/transpose_1_2.h](../../csrc/kernels/transpose_1_2.h),采用 Ascend C 类封装风格(`Transpose_1_2` 类,Init / CopyIn / Compute / CopyOut / Process 五段式)。

### 分块策略

- 以 `16×tileDim1 × 16×tileDim2`(`tileDim1 = tileDim2 = 8`,即 128×128)的二维块为处理单元,沿 dim1、dim2 双重循环切块([transpose_1_2.h:117-119](../../csrc/kernels/transpose_1_2.h#L117-L119))。
- 块间用 `aivCnt` 轮转计数将块轮流分配给各个 AIV core(`aivCnt % GetBlockNum() != GetBlockIdx()` 则跳过),实现多 Block 并行([transpose_1_2.h:120-122](../../csrc/kernels/transpose_1_2.h#L120-L122))。
- 尾部不足整块时计算 `padDim1` / `padDim2` 补齐量,拷贝时通过 `DataCopyPadExtParams` 处理边界([transpose_1_2.h:118](../../csrc/kernels/transpose_1_2.h#L118)、[transpose_1_2.h:123](../../csrc/kernels/transpose_1_2.h#L123))。

### UB 内存布局

UB 被均分为 4 份,两份输入 ping-pong 缓冲(`VECIN`)+ 两份输出 ping-pong 缓冲(`VECOUT`),每份 `UB_SIZE / 4`([transpose_1_2.h:46-57](../../csrc/kernels/transpose_1_2.h#L46-L57))。ping-pong 交替使用,实现搬运与计算的流水重叠。

### 三级流水

每个 tile 依次执行,通过事件标志同步 MTE2(搬入)→ V(计算)→ MTE3(搬出)三级流水:

1. **CopyIn**:`DataCopyPad` 从 GM 搬入 `inBuf[ping]`,按 `blockCount × blockLen` 分块描述源矩阵,`srcStride` 跳过行间不搬运部分([transpose_1_2.h:60-73](../../csrc/kernels/transpose_1_2.h#L60-L73))。
2. **Compute**:用 `TransDataTo5HD` 指令完成 16×16 矩阵的片上转置。对块内每个 `tileDim1` 行,构造 16 个源/目的 `LocalTensor` 列表,一次指令完成 16 列的同时转置;`dstRepStride = tileDim1 * 16`、`srcRepStride = 1` 控制转置后的排布([transpose_1_2.h:75-95](../../csrc/kernels/transpose_1_2.h#L75-L95))。
3. **CopyOut**:`DataCopyPad` 将转置结果从 `outBuf[ping]` 写回 GM,输出侧的 `blockCount` / `blockLen` / stride 与输入侧互换(维度 1、2 角色对调),`dstStride = dim1 * 2 - blockLen` 反映输出行距([transpose_1_2.h:97-109](../../csrc/kernels/transpose_1_2.h#L97-L109))。

同步链:`MTE3→MTE2`(缓冲空闲)、`MTE2→V`(数据就绪)、`V→MTE3`(结果就绪),各用 `EVENT_ID0 + ping` 区分 ping/pong 两组事件([transpose_1_2.h:70-72](../../csrc/kernels/transpose_1_2.h#L70-L72))。

### 边界处理

- `padDim1`、`padDim2` 分别记录 dim1、dim2 方向的补齐元素数,搬入时尾部 pad、搬出时裁剪;
- `CopyIn` 的 `dstStride = padDim2 / 16` 与 `padParams` 中的 `padDim2 % 16` 配合,保证 UB 内数据 32B 对齐的同时不引入脏数据([transpose_1_2.h:63-67](../../csrc/kernels/transpose_1_2.h#L63-L67))。

### 测试参考

[tests/kernels/transpose_1_2.py](../../tests/kernels/transpose_1_2.py):对 batch 1~8、seq_len 2^0~2^12、fp16/bf16 组合验证,与 `torch.transpose(1, 2)` 对比。
