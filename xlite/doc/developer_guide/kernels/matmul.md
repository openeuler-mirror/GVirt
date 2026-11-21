# matmul

## 功能概述

通用矩阵乘法 `z = x * y^T`（即 `out = ((A * B) + bias) * deqScale`，bias 与 deqScale 可选），基于 Ascend Cube（Mmad）单元实现，是 xlite 推理库所有 Linear 层的核心算子。支持权重 ND / NZ（Fractal NZ）两种布局、权重转置（`[K, N]` 输入）、bias 融合（Mmad 带 bias 表）以及 int8 / int4 权重量化路径的 fixpipe 反量化（deqScale）融合。

## 输入输出参数

kernel 入口（`csrc/kernels/matmul.h` 的 `matmul_##dtype`，由宏 `MATMUL_FUNC_DEFINE` 定义）：

```cpp
matmul_##dtype(x, y, z, m, n, k, nz, transpose, m0, n0, k0, swizzl, bias, deqScale)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入 | [M, K]（ND，行主序） | float16 / bfloat16 / float / int8 / int4 | 激活矩阵 A |
| y | 输入 | [N, K]（ND）或 NZ 格式；transpose 时 [K, N] | 同 x | 权重矩阵 B。`nz=1` 时为 Fractal NZ 布局（`matrix_nd2nz` / `ACL_FORMAT_FRACTAL_NZ` 转换得到） |
| z | 输出 | [M, N]（ND） | float16 / bfloat16 / float；量化路径固定 float16 | 输出矩阵 C |
| m, n, k | 标量 | - | uint64_t | GEMM 维度。host 侧由 weight shape 推导：`n = transpose ? weight.shape[1] : weight.shape[0]`，`k = transpose ? weight.shape[0] : weight.shape[1]`（`csrc/op.cpp` 的 `XliteOpMatmul`，关键字 `n = transpose ? weight.shape[1]`） |
| nz | 标量 | - | uint64_t | y 是否为 NZ 格式 |
| transpose | 标量 | - | uint64_t | y 是否为 `[K, N]` 转置存储 |
| m0, n0, k0 | 标量 | - | uint64_t | 分块尺寸；`k0` 为单次 Mmad 的 K 量程（最小量化单位，A 的 L1 预取窗 `k8TileSize = 8*k0` 与 B 的 `k4TileSize = 4*k0` 由它派生，L0A/L0B 单片 K 深度即 `k0`）；传 `(uint64_t)-1` 时由 kernel/device 侧取默认值 |
| swizzl | 标量 | - | uint64_t | tile 遍历 swizzle 参数：高 8 位以上为 `swizzlCount`，低 8 位为 `swizzleDirection`（0=Zn，1=Nz），用于提升 L2 命中率 |
| bias | 输入（可选） | [N] | int32（量化路径）或 float（浮点路径，bf16 时 host 先 cast 成 fp32） | 加法偏置，走 Mmad bias 表（C2） |
| deqScale | 输入（可选） | [N]（以 uint64_t 视角） | uint64_t（低位 32bit 存 fp32/TF32） | fixpipe 反量化 scale。TF32 格式：1 符号位 + 8 指数位 + 10 尾数位，低 13 位不参与计算。Python 侧以 `[2N]` fp32 交错数组提供（偶数位为 scale、奇数位为 0，见 `tests/kernels/matmul_int8.py` 的 fixpipe scale 构造段，关键字 `scale[0::2] = deqScale`） |

Python 调用方式：

- `matmul(rt, x, y, z, weight_nz, transpose)`（`tests/kernels/matmul.py` 的 `matmul(rt, x, y, z, ...)` 调用示例，关键字 `matmul(rt, x, y, z`）
- `matmul_with_bias(rt, x, y, z, bias, weight_nz)`（`tests/kernels/matmul_with_bias.py` 的 `matmul_with_bias(...)` 调用示例，关键字 `matmul_with_bias(rt, x, y, z`）
- `matmul_dequant(rt, x, y, bias, scale, z, weight_nz, transpose)`（`tests/kernels/matmul_int8.py` 的 `matmul_dequant(...)` 调用示例，关键字 `matmul_dequant(rt, x, y_in`；`tests/kernels/matmul_int4.py` 的 `matmul_dequant(...)` 调用示例，关键字 `matmul_dequant(rt, x_int4_pack`）

各接口的 `weight_nz` 参数已废弃并将被移除：传入值被忽略，NZ 权重布局由 `InitXTensor` 边界自动检测的 FRACTAL_NZ 存储格式决定（`csrc/xlite.h` 的 `IsNpuNzFormat`），matmul 类 op 据此选择 kernel 路径。

int4 路径中 x/y 为 int32 容器（`npu_convert_weight_to_int4pack` 打包结果），`_C.cpp` 中 `View(INT4)` 重解释为 int4（`csrc/_C.cpp` 的 `Matmul` 与 `MatmulDeQuant` 两处 `if (_x.dtype == INT32 && _y.dtype == INT32)` 块，关键字 `View(INT4)`）。

## 支持的数据类型

| Dtype（x/y） | bias（MatDtype） | 输出（OutDtype） | 实例化文件 | kernel 符号 |
|------|------|------|------|------|
| float16_t | float | float16_t | `csrc/kernels/matmul_float16_t.cpp` | `matmul_float16_t` |
| bfloat16_t | float（host 侧 bf16 bias 先 cast 为 fp32，`csrc/op.cpp` 的 `XliteOpMatmul`，关键字 `aclrtlaunch_cast_bfloat16_t_float` + bias） | bfloat16_t | `csrc/kernels/matmul_bfloat16_t.cpp` | `matmul_bfloat16_t` |
| float | float | float | `csrc/kernels/matmul_float.cpp` | `matmul_float`（仅 `!transpose`，`csrc/op.cpp` 的 `XliteOpMatmul`，关键字 `aclrtlaunch_matmul_float`） |
| int8_t | int32_t | half | `csrc/kernels/matmul_int8_t.cpp` | `matmul_int8_t`（W8A8：`in/weight INT8 + out FP16`，`csrc/op.cpp` 的 `XliteOpMatmul`，关键字 `aclrtlaunch_matmul_int8_t`） |
| int4b_t | int32_t | half | `csrc/kernels/matmul_int4b_t.cpp` | `matmul_int4b_t`（W4A4/W4A8：`in/weight INT4 + out FP16`，`csrc/op.cpp` 的 `XliteOpMatmul`，关键字 `aclrtlaunch_matmul_int4b_t`） |

另外 host 侧还支持混合精度组合（`csrc/op.cpp` 的 `XliteOpMatmul`，BF16+FP32 混合精度分支，关键字 `in.dtype == BF16` 与 `FP32`）：`in BF16 + weight/out FP32`、`in BF16 + weight FP32 + out BF16`（均 `!transpose`），通过前置/后置 cast kernel 走 `matmul_float`。

kernel 任务类型为 `KERNEL_TYPE_AIC_ONLY`（`csrc/kernels/matmul.h` 的宏 `MATMUL_FUNC_DEFINE`，关键字 `KERNEL_TYPE_AIC_ONLY`），运行在 AI Cube 核上。

## 实现原理

### 整体数据流

单核处理一个或多个 `(m0 x n0)` 输出 tile，单个 tile 内沿 K 维循环。数据流为经典的五级 Cube 流水（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`）：

```text
A(x): GM --MTE2: Nd2Nz--> L1(A1, pingpong) --MTE1: LoadData2d--> L0A(A2, pingpong) \
B(y): GM --MTE2: Nd2Nz/CopyGmToL1--> L1(B1, pingpong) --MTE1: LoadData2d--> L0B(B2, pingpong) \
    --> M(Mmad): L0A x L0B --> L0C(CO1, int32/float累加) \
bias: GM --MTE2--> L1(C1) --MTE1--> C2(Bias Table) \
    --> FIX(fixpipe): L0C --NZ2ND + 可选反量化--> GM(z)
deqScale: GM --MTE2--> L1(C1) --FIX--> C2PIPE2GM(fixpipe buffer)
```

- A 矩阵 ND→NZ 转换由 DMA 硬件完成：`CopyGmToL1Nd2Nz` 使用 `DataCopy` 的 `Nd2NzParams`（`csrc/kernels/kernel_macro.h` 的 `CopyGmToL1Nd2Nz`），即 GM→L1 搬运时直接落成 NZ 分形布局；当 `srcDValue > 65535`（K 超过 16bit stride 上限）时退化为按行循环搬运。
- `CopyGmToL1`（`csrc/kernels/kernel_macro.h` 的 `CopyGmToL1`）的 `DataCopyParams` 字段以 32 字节块为单位计数（burst 长度 `nValue` 块、GM 行距 `nStride` 块、L1 目的行距 `nValue` 块）；`srcGap` 为 uint16，NZ 权重行距超过 65535 块（如大词表 lm_head 的 n = 151936）时会回绕错读，此时退化为每行单独 `DataCopy`（源地址按元素全宽计算，1 块 = 32*8/dtypeBits 个元素），L1 布局不变。
- B 矩阵按 `(transpose, nz)` 四种组合搬运（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，B GM→L1 四组合分支，关键字 `/* B GM -> L1 */`）：
    - `transpose=0, nz=0`：`CopyGmToL1Nd2Nz`，与 A 相同的 ND→NZ 硬件转换；
    - `transpose=0, nz=1`：`CopyGmToL1` 直接按 NZ 分形块拷贝（源已是 NZ）；
    - `transpose=1, nz=0`：`CopyGmToL1Nd2Nz` 把 `[K, N]` 的 ND 矩阵按 K 为 N 维、N 为 D 维转成 NZ；
    - `transpose=1, nz=1`：`CopyGmToL1` 按转置后的 NZ 块拷贝。
- L1→L0 搬运使用 Cube 的 `LoadData`（load2d）：`CopyToL0ACol`（`csrc/kernels/kernel_macro.h` 的 `CopyToL0ACol`）与 `CopyToL0BCol`（`csrc/kernels/kernel_macro.h` 的 `CopyToL0BCol`）；转置路径的 `CopyToL0BTCol`（`csrc/kernels/kernel_macro.h` 的 `CopyToL0BTCol`，关键字 `LoadDataWithTranspose`）对 int8/int4 使用 `LoadDataWithTranspose`（B 转置时 L1→L0 的分形单元为 32x32x1B，`csrc/kernels/matmul.h` 的 `Matmul::Init`，关键字 `nBlockSize = 32 * BYTE_BITS / dtypeBits`）。
- 矩阵计算：`CalMmad`（`csrc/kernels/kernel_macro.h` 的 `CalMmad`）在 L0C 上沿 K 累加，首轮 `cmatrixInitVal=true` 清零；带 bias 时首轮改用 `CalMmadWithBias`（`csrc/kernels/kernel_macro.h` 的 `CalMmadWithBias`），bias 从 C2 Bias Table 参与累加。
- 输出：`CopyToGmWithDequant`（`csrc/kernels/kernel_macro.h` 的 `CopyToGmWithDequant`）通过 fixpipe 把 L0C 的 NZ 数据以 `SetFixpipeNz2ndFlag` 转 ND 写回 GM；浮点路径 mode 为 `F322F16/F322BF16/NoQuant`，int32→half 量化反量化路径 mode 为 `VDEQF16` 并通过 `SetFixPipeConfig(deqScale)` 提供 per-N 反量化向量。

### 分块与多核并行

- tile 划分：`mLoop = ceil(m/m0)`，`nLoop = ceil(n/n0)`（`csrc/kernels/matmul.h` 的 `Matmul::TaskTilesInit`，关键字 `mLoop = DIV_ROUND_UP(m, m0)`），`TaskTilesInit` 返回 `mLoop * nLoop` 个 tile；多核按 `for (idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum())` 交错认领（`csrc/kernels/matmul.h` 的 `Matmul::Run`，关键字 `for (int64_t idx = GetBlockIdx()`）。
- 自持调度访问器（多批次驱动如 einsum 逐头批处理使用）：`RunTileByMN(midx, nidx)` 绕过 swizzle 重排直接执行指定 tile；`RebindGm(x, y, z)` 在保持 `TaskTilesInit` 几何的前提下换 A/B/C 基址,并因 residency 跟踪器键不含 GM 基址而同步重置全部跟踪器（`mROffset/nROffset/kRAOffset/kRBOffset/biasROffset`）；`MLoop()`/`NLoop()` 读出批几何；`BWholeKResident(k)` 判断 k 是否能整体驻留 B 的两个 L1 乒乓槽（`k ≤ 8*k0`,跨 tile B 复用的成立条件,chunked 遍历的准入门槛）（`csrc/kernels/matmul.h` 的 `Matmul::RunTileByMN` / `Matmul::RebindGm` / `Matmul::BWholeKResident`）。
- 默认分块（`m0 == -1` 时）：`m0=128`，`n0=256`，`k0 = 128*8/dtypeBits`；hasBias/hasDeqScale 时钳制 `n0 ≤ 256` 以留出 fixpipe/bias buffer 空间（`csrc/kernels/matmul.h` 的 `Matmul::Init`，关键字 `if (m0 == (uint64_t)-1)`）。host 侧默认逻辑在 `csrc/auto_tuner.cpp` 的 `PickMatmulTiling`：`m0 = min(ROUND_UP(m,32), 128)`、`n0 = 256`（needExtraSpace 即 hasBias/hasDeqScale 时钳制 `n0 ≤ 256`）、`k0 = 1024/dtypeBits`，并在 `totalLoops < 3*aicNum` 且尾核负载不均时进一步缩小 n0（64/128/256/384）并可能减半 k0、缩 m0 到 64，以保证各 AICore 均载。
- K 维两级量化：`k0` 即单次 Mmad 的 K 量程（最小单位）；L1 中 A 的预取窗 `k8TileSize = 8*k0`（整 8 次 Mmad 的重载周期，`kIdx8`），B 的预取窗 `k4TileSize = 4*k0`（`kIdx4`），L0A/L0B 乒乓单片 K 深度 = `k0`（`kIdx2`）（`csrc/kernels/matmul.h` 的 `Matmul::Init`，关键字 `k8TileSize = k0 << 3`），即 A 的 GM→L1 每 8 次 K 循环做一次，B 每 4 次一次，L0A/L0B 乒乓每 2 次一次。
- 分形块尺寸：`mBlockSize=16`；`nBlockSize=16`，transpose 时 `nBlockSize = 32*8/dtypeBits`（float 8、fp16/bf16 16、int8 32、int4 64，int8/int4 转置对应 32x32x1B 分形单元）；`kBlockSize = 32*8/dtypeBits`，取值同为 float 8、fp16/bf16 16、int8 32、int4 64，`csrc/kernels/matmul.h` 的 `Matmul::Init`，关键字 `mBlockSize = 16`）。边界 tile 用 `mActual/nActual` 与对应 `ROUND_UP(..., xBlockSize)` 的 pad 值处理（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，关键字 `mActual = m - mOffset`）。
- swizzle 重排：`GetMNBlockIdx`（`csrc/kernels/matmul.h` 的 `Matmul::GetMNBlockIdx`）支持两种方向——`swizzleDirection=0`（Zn：沿 M 方向按 `swizzlCount` 分组、组间 N 序蛇形遍历）与 `=1`（Nz：沿 N 方向分组、组间 M 序蛇形遍历）——调整 tile 遍历顺序以提升 cache 命中。host 侧 `XlitePickSwizzle` 按 (n, k) 查表（实测 910B3 上 `{6144,2048}` 与 `{2048,6144}` 均取 `0xe01`，即 count=14、direction=1，`csrc/swizzle.cpp` 的 `XlitePickSwizzle`，关键字 `BestSwizzles` 表）。

### L1/L0 缓冲布局

`Init`（`csrc/kernels/matmul.h` 的 `Matmul::Init`）按 tile 尺寸静态规划各 buffer 偏移：

| Buffer | 位置 | 份数 | 大小 |
|--------|------|------|------|
| l1aBuf | A1（L1） | 2（乒乓） | `m0 * k8TileSize * dtypeBits/8` |
| l1bBuf | B1（L1） | 2（乒乓） | `n0 * k4TileSize * dtypeBits/8` |
| l1BiasBuf | C1（L1） | 1 | `n0 * sizeof(MatDtype)` |
| l1DeqScaleBuf | C1（L1） | 1 | `n0 * sizeof(uint64_t)` |
| l0aBuf / l0bBuf | A2 / B2（L0） | 2（乒乓） | `m0/n0 * k0 * dtypeBits/8` |
| l0BiasBuf | C2（Bias Table，1KB，数据块 64B） | 1 | - |
| fixpipeBuf | C2PIPE2GM（fixpipe buffer，2KB，数据块 128B） | 1 | - |
| l0cBuf | CO1（L0C） | 1 | 累加结果 |

bias 从 L1 到 C2 的搬运以 64B 数据块为单位（`C2_DATABLOCK`），deqScale 从 L1 到 fixpipe buffer 以 128B 数据块为单位（`FIXPIPE_DATABLOCK`），见 `csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`（bias L1→C2 段关键字 `C2_DATABLOCK`；deqScale L1→fbuf 段关键字 `FIXPIPE_DATABLOCK`）。

### 流水线同步（事件旗标）

`SetFlags`/`WaitFlags`（`csrc/kernels/matmul.h` 的 `Matmul::SetFlags` 与 `Matmul::WaitFlags`）在 tile 生命周期的起止成对补齐全部事件；tile 内部使用：

- `MTE1_MTE2` EVENT_ID0/1（+乒乓位）：L1A 缓冲释放/占用，A 的 GM→L1 用；
- `MTE1_MTE2` EVENT_ID2/3（+乒乓位）：L1B 缓冲，B 的 GM→L1 用；
- `M_MTE1` EVENT_ID0/1（+`kIdx2`）：L0A/L0B 就绪，Mmad 消费后释放；
- `M_MTE1`/`MTE1_M` EVENT_ID4：bias 的 L1↔C2 链路独占；
- `FIX_MTE2`/`MTE2_FIX` EVENT_ID5：deqScale 的 L1↔fixpipe 链路（属于 fixpipe barrier，`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，关键字 `belong to fixpipe barrier` 注释）；
- `M_FIX`/`FIX_M` EVENT_ID0：L0C 就绪与 fixpipe 搬出完成。

乒乓选择器是 tile 内局部变量（每 tile 从 0 开始），因为 tile 退出时两级缓冲都已释放（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，关键字 `Ping-pong selectors are tile-local` 注释）。L1A 的乒乓在 `kIdx8==7` 或末轮翻转（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，关键字 `kIdx8 == 7`），L1B 在 `kIdx4==3` 或末轮翻转（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，关键字 `kIdx4 == 3`）。

### 量化路径（int8 / int4）

- int8（W8A8）：`Matmul<int8_t, int32_t, half>`（`csrc/kernels/matmul.h` 的宏 `MATMUL_FUNC_DEFINE`，关键字 `Matmul<int8_t, int32_t, half>`）。A/B 以 int8 进入 Mmad，L0C 累加为 int32；fixpipe 阶段以 `VDEQF16` 模式将 int32 乘以 deqScale 向量后转 fp16 写 GM（`csrc/kernels/kernel_macro.h` 的 `CopyToGmWithDequant`，关键字 `VDEQF16`）。
- int4（W4A4 / W4A8 激活拆分）：`Matmul<int4b_t, int32_t, half>`。dtypeBits 按 4 计（`GetDTypeBits`，`csrc/kernels/kernel_macro.h` 的 `GetDTypeBits` 宏），因此 `kBlockSize=64`、`nBlockSize`（转置时）=64，`CopyGmToL1Nd2Nz` 对 int4 把 `srcDValue/dValue` 折半（int4 两两打包进一个字节，`csrc/kernels/kernel_macro.h` 的 `CopyGmToL1Nd2Nz`，关键字 `srcDValue / 2`），B 转置路径的 `CopyToL0BTCol` 用 `LoadDataWithTranspose` 以 32x32x1B 分形、dstGap=3 解包搬运（`csrc/kernels/kernel_macro.h` 的 `CopyToL0BTCol`，关键字 `LoadDataWithTranspose`）。W4A8 场景下激活先由 `unpack_activation` 拆成 `[2m, k/2]` 的 int4 打包矩阵（行交错 low/high 4bit），本算子按 int4 GEMM 计算，结果再由 `msd_merge_dequant` 合并（见 group_matmul 文档与 `tests/kernels/matmul_int4.py`）。
- deqScale 的硬件格式：fixpipe 要求以 uint64_t 存储 fp32——高 32 位为 0、低 32 位为 fp32 二进制值（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，关键字 `以uint64_t存储fp32` 注释）；有效精度为 TF32（低 13 位尾数忽略，`csrc/kernels/matmul.h` 顶部注释，关键字 `TF32 格式`）。Python 侧生成 `[2N]` fp32 数组，偶数位放 scale、奇数位补 0（`tests/kernels/matmul_int8.py` 的 fixpipe scale 构造段，关键字 `scale[0::2] = deqScale`）。

### 权重 NZ 与 bias 融合

- `weightNZ=True` 时 y 已是 Fractal NZ 格式（`torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)` 或 `matrix_nd2nz`），kernel 走 `CopyGmToL1` 免去 ND→NZ 转换（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，B GM→L1 分支中 `CopyGmToL1` 的 nz 分支），节省 MTE2 带宽；float 类型不支持 NZ（测试中 `weight_nz and dtype != torch.float`）。
- bias 仅在量化路径为 int32；浮点路径 MatDtype=float，bf16 调用时 host 先用 `cast_bfloat16_t_float` 把 bias 升精度（`csrc/op.cpp` 的 `XliteOpMatmul`，关键字 `aclrtlaunch_cast_bfloat16_t_float`）。bias 在首个 K 块通过 `Mmad(c, a, b, bias, params)` 一次进入 L0C（`csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`，关键字 `CalMmadWithBias` 首轮）。

### 片上缓冲分配与占用汇总（AIC L1/L0A/L0B/L0C）

`Init`（`csrc/kernels/matmul.h` 的 `Matmul::Init`）按分块尺寸手动排布全部片上缓冲：L1 上 `l1aBuf`/`l1bBuf` 两对乒乓自偏移 0 顺序累进，`l1BiasBuf` 与 `l1DeqScaleBuf` 逻辑位为 C1、偏移重置回 `l1bBuf[0]` 起点与其共享槽位（bias/deqScale 在 B 的首次 GM→L1 之前用毕释放，时序互斥，见 `Init` 的 `shared with l1bBuf` 注释）；L0A（A2）、L0B（B2）、L0C（CO1）三级物理独立、各自从偏移 0 重新起排；bias 表与 deqScale 分别落在 CO1 之外的固定专用区 C2 Bias Table（1KB）与 C2PIPE2GM fixpipe buffer（2KB）。硬件容量（910B/dav C220，NpuArch 2201，CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`）：L0A 64KB、L0B 64KB、L1 512KB，L0C 总容量 **128KB**（`l0_c_size=131072`，16 bank × 8KB），Mmad 累加缓冲即 `TPosition::CO1` 且占满整个 L0C；本算子为 `KERNEL_TYPE_AIC_ONLY`，不使用 UB。

dtype 记法（模板实参，见宏 `MATMUL_FUNC_DEFINE`）：`Dtype` 为 x/y 元素类型，`MatDtype` 为 bias/L0C 累加类型，`OutDtype` 为 z 类型——浮点路径 `<fp16|bf16|float, float, 同 Dtype>`，int8 路径 `<int8_t, int32_t, half>`（W8A8），int4 路径 `<int4b_t, int32_t, half>`（W4A4/W4A8）。典型配置取 GLM-5.1 TP16 per-rank dense W8A8 线性（`tests/performance/perf_fused_quant_matmul_dequant.py` 的 `MODEL_SIZES` 中 `dense_gate_up`：m=4096、n=1536、k=6144，带 deqScale），host `PickMatmulTiling`（`csrc/auto_tuner.cpp` 的 `PickMatmulTiling`）产出 m0=128、n0=256、k0=128（int8）/64（fp16/bf16）/256（int4）。默认 `k0 = 128*8/dtypeBits`（单次 Mmad 的 K 量程）随 dtype 反比缩放，三条路径的字节占用完全一致（差异仅在 `l1bBuf` 单份 shape 的 K 维深度 `k4TileSize = 4*k0`：fp16 256、int8 512、int4 1024），占用列故取单列；占用只由分块决定，与问题规模 m/n/k 无关（尾部 tile 只会更小）。下表 shape 为逻辑形状（L1/L0 内实际按 Nd2Nz/NZ 分形排布）。

**AIC L1（A1/B1/C1，容量 512KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l1aBuf` | `(m0, k8TileSize)`=`(128, 8×k0)` | Dtype | 2（事件0/1） | `m0*k8TileSize*sizeof(Dtype)` | 262144 |
| `l1bBuf` | `(n0, k4TileSize)`=`(256, 4×k0)` | Dtype | 2（事件2/3） | `n0*k4TileSize*sizeof(Dtype)` | 262144 |
| `l1BiasBuf` | `(n0,)`=`(256,)` | MatDtype | 1（事件4） | `n0*sizeof(MatDtype)` | 0（与 `l1bBuf[0]` 共享槽位） |
| `l1DeqScaleBuf` | `(n0,)`=`(256,)` | uint64_t | 1（事件5） | `n0*sizeof(uint64_t)` | 0（与 `l1bBuf[0]` 共享槽位） |
| **合计** | | | | | **524288** |

`l1BiasBuf`/`l1DeqScaleBuf` 的 1024/2048 B 槽位落在 `l1bBuf[0]`（131072 B）头部，不新增 L1 占用；`n0 ≤ 256` 的钳制保证其不越出 `l1bBuf[0]`。

**AIC L0A（A2，容量 64KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l0aBuf` | `(m0, k0)`=`(128, k0)` | Dtype | 2（事件0/1，kIdx2） | `m0*k0*sizeof(Dtype)` | 32768 |
| **合计** | | | | | **32768** |

**AIC L0B（B2，容量 64KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l0bBuf` | `(n0, k0)`=`(256, k0)` | Dtype | 2（事件0/1，kIdx2） | `n0*k0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C（`TPosition::CO1`，容量 128KB）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `l0cBuf` | `(m0, n0)`=`(128, 256)` | MatDtype | 1（事件0） | `m0*n0*sizeof(MatDtype)`（Mmad 分形累加，无显式分配字节数，按满 tile 计） | 131072 |
| **合计** | | | | | **131072** |

bias 链路的 `l0BiasBuf`（C2 Bias Table，固定 1KB，64B 数据块）与 deqScale 链路的 `fixpipeBuf`（C2PIPE2GM，固定 2KB，128B 数据块）是 CO1 之外的独立专用硬件区，占用恒定、不计入上表；`n0 ≤ 256` 钳制的容量来源即这两区（deqScale `n0*8B ≤ 2KB`、bias `n0*4B ≤ 1KB`）。

**占用汇总**（各表占用列之和 vs 硬件容量；字节数）：

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| L1 | 512 KB | 524288 B | 100% |
| L0A | 64 KB | 32768 B | 50% |
| L0B | 64 KB | 65536 B | 100% |
| L0C | 128 KB | 131072 B | 100% |

要点：

- **默认分块恰好同时顶满 L1、L0B 与 L0C CO1 三级**：m0=128、n0=256、k0=128×8/dtypeBits 下三者均达容量 100%，L0A 因 m0 只有 n0 的一半而半满（50%）——任何分块维度的放大都先破这三级，默认值即容量上界点；
- `l1aBuf` 与 `l1bBuf`（各 262144 B）平分 L1、是绝对大头；bias/deqScale 共享 `l1bBuf[0]` 头部槽位，零额外 L1 开销；
- **`n0 ≤ 256` 钳制的容量来源是 CO1 之外的 C2 Bias Table（1KB）与 C2PIPE2GM fixpipe buffer（2KB）**（`csrc/kernels/matmul.h` 的 `Matmul::Init` 的 `bias buffer 1k`/`fixpipe buffer 2k` 注释），而非 L1/L0 本身；
- host `PickMatmulTiling` 的均载再平衡（`csrc/auto_tuner.cpp` 的 `PickMatmulTiling`：小 n 时 m0≤64/n0=64，特大 n 时 m0≤64/n0=384/k0 减半）只会按比例缩小占用，不改变「L1/L0B/L0C 同紧」的格局。
