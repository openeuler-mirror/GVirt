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

- `float16_t`([transpose_1_2_float16_t.cpp](../../../csrc/kernels/transpose_1_2_float16_t.cpp))
- `bfloat16_t`([transpose_1_2_bfloat16_t.cpp](../../../csrc/kernels/transpose_1_2_bfloat16_t.cpp))

## 实现原理

实现位于 [csrc/kernels/transpose_1_2.h](../../../csrc/kernels/transpose_1_2.h),采用 Ascend C 类封装风格(`Transpose_1_2` 类,Init / CopyIn / Compute / CopyOut / Process 五段式)。

### 分块策略

- 以 `16×tileDim1 × 16×tileDim2`(`tileDim1 = tileDim2 = 8`,即 128×128)的二维块为处理单元,沿 dim1、dim2 双重循环切块(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::Process`,关键字 `for (uint32_t i = 0; i < dim1; i += 16 * tileDim1)`)。
- 块间用 `aivCnt` 轮转计数将块轮流分配给各个 AIV core(`aivCnt % GetBlockNum() != GetBlockIdx()` 则跳过),实现多 Block 并行(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::Process`,关键字 `aivCnt`)。
- 尾部不足整块时计算 `padDim1` / `padDim2` 补齐量,拷贝时通过 `DataCopyPadExtParams` 处理边界(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::Process`,关键字 `padDim1` / `padDim2`)。

### UB 内存布局

UB 被均分为 4 份,两份输入 ping-pong 缓冲(`VECIN`)+ 两份输出 ping-pong 缓冲(`VECOUT`),每份 `UB_SIZE / 4`(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::Init`,关键字 `inBuf` / `UB_SIZE / 4`)。ping-pong 交替使用,实现搬运与计算的流水重叠。

### 三级流水

每个 tile 依次执行,通过事件标志同步 MTE2(搬入)→ V(计算)→ MTE3(搬出)三级流水:

1. **CopyIn**:`DataCopyPad` 从 GM 搬入 `inBuf[ping]`,按 `blockCount × blockLen` 分块描述源矩阵,`srcStride` 跳过行间不搬运部分(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::CopyIn`)。
2. **Compute**:用 `TransDataTo5HD` 指令完成 16×16 矩阵的片上转置。对块内每个 `tileDim1` 行,构造 16 个源/目的 `LocalTensor` 列表,一次指令完成 16 列的同时转置;`dstRepStride = tileDim1 * 16`、`srcRepStride = 1` 控制转置后的排布(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::Compute`,关键字 `TransDataTo5HD`)。
3. **CopyOut**:`DataCopyPad` 将转置结果从 `outBuf[ping]` 写回 GM,输出侧的 `blockCount` / `blockLen` / stride 与输入侧互换(维度 1、2 角色对调),`dstStride = dim1 * 2 - blockLen` 反映输出行距(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::CopyOut`)。

同步链:`MTE3→MTE2`(缓冲空闲)、`MTE2→V`(数据就绪)、`V→MTE3`(结果就绪),各用 `EVENT_ID0 + ping` 区分 ping/pong 两组事件(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::CopyIn` / `CopyOut`,关键字 `set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + ping)`)。

### 边界处理

- `padDim1`、`padDim2` 分别记录 dim1、dim2 方向的补齐元素数,搬入时尾部 pad、搬出时裁剪;
- `CopyIn` 的 `dstStride = padDim2 / 16` 与 `padParams` 中的 `padDim2 % 16` 配合,保证 UB 内数据 32B 对齐的同时不引入脏数据(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::CopyIn`,关键字 `dstStride = padDim2 / 16`)。

### 测试参考

[tests/kernels/transpose_1_2.py](../../../tests/kernels/transpose_1_2.py):对 batch 1~8、seq_len 2^0~2^12、fp16/bf16 组合验证,与 `torch.transpose(1, 2)` 对比。

### 片上缓冲分配与占用汇总(AIV UB)

`Init`(`csrc/kernels/transpose_1_2.h` 的 `Transpose_1_2::Init`)静态把 UB 四等分:输入两份 `inBuf[0]`/`inBuf[1]`(`TPosition::VECIN`)与输出两份 `outBuf[0]`/`outBuf[1]`(`TPosition::VECOUT`)各占 `UB_SIZE / 4`,自偏移 0 顺序排布,槽位编译期固定、与输入规模无关。硬件容量基准(910B/dav C220,NpuArch 2201):AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每个 AIV block 各持一份独立 UB、布局相同。dtype 记法:模板参数 `Dtype` 只决定实例化入口,缓冲与搬运均按 `half`(2B)处理(`LocalTensor<half>` 硬编码),fp16 与 bf16 两条路径布局完全相同,占用单列。典型配置取注意力 Q/K 布局变换(如 `[4, 4096, 64×128]`,nHeads=64、headDim=128,fp16/bf16 同取):处理块恒为 `16×tileDim1 × 16×tileDim2` = 128×128,单块数据 128×128×2 = 32768 B,占用不随配置变化。

**AIV UB(预算 `UB_SIZE=192KB`)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `inBuf[ping]` | `(16×tileDim1, 16×tileDim2)` = (128, 128) | half | 2(事件0/1) | `UB_SIZE / 4` | 98304 |
| `outBuf[ping]` | `(16×tileDim2, 16×tileDim1)` = (128, 128) | half | 2(事件0/1) | `UB_SIZE / 4` | 98304 |
| **合计** | | | | | **196608** |

**占用汇总**(占用 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| UB | 192 KB | 196608 B | 100% |

要点:

- **UB 静态四等分、恰好占满 192KB**:槽位由 `UB_SIZE / 4` 编译期固定,占比恒为 100%,与输入规模、dtype 均无关;
- 单块实际数据仅 32768 B,约为槽位(49152 B)的 2/3,余量来自固定划分而非数据需求,不存在运行期动态扩缩;
- 无运行期断言或 host 侧容量校验:tile 尺寸(`tileDim1 = tileDim2 = 8`)硬编码,`Init` 一次排定全部缓冲;
- 乒乓以 `EVENT_ID0 + ping` 区分两组事件,MTE2→V→MTE3 三级流水在 `inBuf`/`outBuf` 各两份之间轮转(见上文「三级流水」)。
