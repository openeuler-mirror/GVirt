# matmul

## 功能概述

通用矩阵乘法 `z = x * y^T`（即 `out = ((A * B) + bias) * deqScale`，bias 与 deqScale 可选），基于 Ascend Cube（Mmad）单元实现，是 xlite 推理库所有 Linear 层的核心算子。支持权重 ND / NZ（Fractal NZ）两种布局、权重转置（`[K, N]` 输入）、bias 融合（Mmad 带 bias 表）以及 int8 / int4 权重量化路径的 fixpipe 反量化（deqScale）融合。

## 输入输出参数

kernel 入口（`csrc/kernels/matmul.h:461`）：

```cpp
matmul_##dtype(x, y, z, m, n, k, nz, transpose, m0, n0, k0, swizzl, bias, deqScale)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入 | [M, K]（ND，行主序） | float16 / bfloat16 / float / int8 / int4 | 激活矩阵 A |
| y | 输入 | [N, K]（ND）或 NZ 格式；transpose 时 [K, N] | 同 x | 权重矩阵 B。`nz=1` 时为 Fractal NZ 布局（`matrix_nd2nz` / `ACL_FORMAT_FRACTAL_NZ` 转换得到） |
| z | 输出 | [M, N]（ND） | float16 / bfloat16 / float；量化路径固定 float16 | 输出矩阵 C |
| m, n, k | 标量 | - | uint64_t | GEMM 维度。host 侧由 weight shape 推导：`n = transpose ? weight.shape[1] : weight.shape[0]`，`k = transpose ? weight.shape[0] : weight.shape[1]`（`csrc/op.cpp:645-647`） |
| nz | 标量 | - | uint64_t | y 是否为 NZ 格式 |
| transpose | 标量 | - | uint64_t | y 是否为 `[K, N]` 转置存储 |
| m0, n0, k0 | 标量 | - | uint64_t | 分块尺寸；传 `(uint64_t)-1` 时由 kernel/device 侧取默认值 |
| swizzl | 标量 | - | uint64_t | tile 遍历 swizzle 参数：高 8 位以上为 `swizzlCount`，低 8 位为 `swizzleDirection`（0=Zn，1=Nz），用于提升 L2 命中率 |
| bias | 输入（可选） | [N] | int32（量化路径）或 float（浮点路径，bf16 时 host 先 cast 成 fp32） | 加法偏置，走 Mmad bias 表（C2） |
| deqScale | 输入（可选） | [N]（以 uint64_t 视角） | uint64_t（低位 32bit 存 fp32/TF32） | fixpipe 反量化 scale。TF32 格式：1 符号位 + 8 指数位 + 10 尾数位，低 13 位不参与计算。Python 侧以 `[2N]` fp32 交错数组提供（偶数位为 scale、奇数位为 0，见 `tests/kernels/matmul_int8.py:102-104`） |

Python 调用方式：

- `matmul(rt, x, y, z, weightNZ, transpose)`（`tests/kernels/matmul.py:43`）
- `matmul_with_bias(rt, x, y, z, bias, weightNZ)`（`tests/kernels/matmul_with_bias.py:37`）
- `matmul_dequant(rt, x, y, bias, scale, z, weightNZ, transpose)`（`tests/kernels/matmul_int8.py:105`、`tests/kernels/matmul_int4.py:139`）

int4 路径中 x/y 为 int32 容器（`npu_convert_weight_to_int4pack` 打包结果），`_C.cpp` 中 `View(INT4)` 重解释为 int4（`csrc/_C.cpp:1273-1276`、`csrc/_C.cpp:1934-1936`）。

## 支持的数据类型

| Dtype（x/y） | bias（MatDtype） | 输出（OutDtype） | 实例化文件 | kernel 符号 |
|------|------|------|------|------|
| float16_t | float | float16_t | `csrc/kernels/matmul_float16_t.cpp` | `matmul_float16_t` |
| bfloat16_t | float（host 侧 bf16 bias 先 cast 为 fp32，`csrc/op.cpp:718-722`） | bfloat16_t | `csrc/kernels/matmul_bfloat16_t.cpp` | `matmul_bfloat16_t` |
| float | float | float | `csrc/kernels/matmul_float.cpp` | `matmul_float`（仅 `!transpose`，`csrc/op.cpp:724`） |
| int8_t | int32_t | half | `csrc/kernels/matmul_int8_t.cpp` | `matmul_int8_t`（W8A8：`in/weight INT8 + out FP16`，`csrc/op.cpp:735-736`） |
| int4b_t | int32_t | half | `csrc/kernels/matmul_int4b_t.cpp` | `matmul_int4b_t`（W4A4/W4A8：`in/weight INT4 + out FP16`，`csrc/op.cpp:737-738`） |

另外 host 侧还支持混合精度组合（`csrc/op.cpp:726-734`）：`in BF16 + weight/out FP32`、`in BF16 + weight FP32 + out BF16`（均 `!transpose`），通过前置/后置 cast kernel 走 `matmul_float`。

kernel 任务类型为 `KERNEL_TYPE_AIC_ONLY`（`csrc/kernels/matmul.h:466`），运行在 AI Cube 核上。

## 实现原理

### 整体数据流

单核处理一个或多个 `(m0 x n0)` 输出 tile，单个 tile 内沿 K 维循环。数据流为经典的五级 Cube 流水（`csrc/kernels/matmul.h:200-378` RunTileBody）：

```
A(x): GM --MTE2: Nd2Nz--> L1(A1, pingpong) --MTE1: LoadData2d--> L0A(A2, pingpong) \
B(y): GM --MTE2: Nd2Nz/CopyGmToL1--> L1(B1, pingpong) --MTE1: LoadData2d--> L0B(B2, pingpong) \
    --> M(Mmad): L0A x L0B --> L0C(CO1, int32/float累加) \
bias: GM --MTE2--> L1(C1) --MTE1--> C2(Bias Table) \
    --> FIX(fixpipe): L0C --NZ2ND + 可选反量化--> GM(z)
deqScale: GM --MTE2--> L1(C1) --FIX--> C2PIPE2GM(fixpipe buffer)
```

- A 矩阵 ND→NZ 转换由 DMA 硬件完成：`CopyGmToL1Nd2Nz` 使用 `DataCopy` 的 `Nd2NzParams`（`csrc/kernels/kernel_macro.h:138-158`），即 GM→L1 搬运时直接落成 NZ 分形布局；当 `srcDValue > 65535`（K 超过 16bit stride 上限）时退化为按行循环搬运。
- B 矩阵按 `(transpose, nz)` 四种组合搬运（`csrc/kernels/matmul.h:297-311`）：
  - `transpose=0, nz=0`：`CopyGmToL1Nd2Nz`，与 A 相同的 ND→NZ 硬件转换；
  - `transpose=0, nz=1`：`CopyGmToL1` 直接按 NZ 分形块拷贝（源已是 NZ）；
  - `transpose=1, nz=0`：`CopyGmToL1Nd2Nz` 把 `[K, N]` 的 ND 矩阵按 K 为 N 维、N 为 D 维转成 NZ；
  - `transpose=1, nz=1`：`CopyGmToL1` 按转置后的 NZ 块拷贝。
- L1→L0 搬运使用 Cube 的 `LoadData`（load2d）：`CopyToL0ACol`（`csrc/kernels/kernel_macro.h:168-179`）与 `CopyToL0BCol`（`kernel_macro.h:181-189`）；转置路径的 `CopyToL0BTCol`（`kernel_macro.h:191-212`）对 int8/int4 使用 `LoadDataWithTranspose`（B 转置时 L1→L0 的分形单元为 32x32x1B，`matmul.h:60-62`）。
- 矩阵计算：`CalMmad`（`kernel_macro.h:214-226`）在 L0C 上沿 K 累加，首轮 `cmatrixInitVal=true` 清零；带 bias 时首轮改用 `CalMmadWithBias`（`kernel_macro.h:228-239`），bias 从 C2 Bias Table 参与累加。
- 输出：`CopyToGmWithDequant`（`kernel_macro.h:313-340`）通过 fixpipe 把 L0C 的 NZ 数据以 `SetFixpipeNz2ndFlag` 转 ND 写回 GM；浮点路径 mode 为 `F322F16/F322BF16/NoQuant`，int32→half 量化反量化路径 mode 为 `VDEQF16` 并通过 `SetFixPipeConfig(deqScale)` 提供 per-N 反量化向量。

### 分块与多核并行

- tile 划分：`mLoop = ceil(m/m0)`，`nLoop = ceil(n/n0)`（`csrc/kernels/matmul.h:156-157`），`TaskTilesInit` 返回 `mLoop * nLoop` 个 tile；多核按 `for (idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum())` 交错认领（`matmul.h:193-195`）。
- 默认分块（`m0 == -1` 时）：`m0=128`，`n0 = (hasBias||hasDeqScale) ? 128 : 256`（L1 512KB 限制，bias/deqScale 需要额外 L1/fixpipe 空间时收缩 n0，`matmul.h:40-46`），`k0 = 512*8/dtypeBits`。host 侧默认逻辑在 `csrc/op.cpp:663-698`：`m0 = min(ROUND_UP(m,32), 128)`、`n0 = needExtraSpace ? 128 : 256`、`k0 = 4096/dtypeBits`，并在 `totalLoops < 3*aicNum` 且尾核负载不均时进一步缩小 n0（64/128/256/384）并可能减半 k0、缩 m0 到 64，以保证各 AICore 均载。
- K 维两级量化：L1 中 A 的 tile 覆盖 `kDtileSize = 2*k0`（8 个 L0 量化片），L0A/L0B 的单次 Mmad 片为 `kQtileSize = k0/4`（`matmul.h:68-69`），即 A 的 GM→L1 每 8 次 K 循环做一次（`kIdx8`），B 每 4 次一次（`kIdx4`），L0A/L0B 乒乓每 2 次一次（`kIdx2`）。
- 分形块尺寸：`mBlockSize=16`；`nBlockSize=16`（transpose 且量化类型时为 `32*8/dtypeBits`，即 int8 为 32、int4 为 64，对应 B 转置分形单元 32x32x1B）；`kBlockSize = 32*8/dtypeBits`（fp16 为 32、int8 为 32、int4 为 64，`matmul.h:58-66`）。边界 tile 用 `mActual/nActual` 与对应 `ROUND_UP(..., xBlockSize)` 的 pad 值处理（`matmul.h:215-227`）。
- swizzle 重排：`GetMNBlockIdx`（`matmul.h:380-419`）支持两种方向——`swizzleDirection=0`（Zn：沿 M 方向按 `swizzlCount` 分组、组间 N 序蛇形遍历）与 `=1`（Nz：沿 N 方向分组、组间 M 序蛇形遍历）——调整 tile 遍历顺序以提升 cache 命中。host 侧 `XlitePickSwizzle` 按 (n, k) 查表（实测 910B3 上 `{6144,2048}` 与 `{2048,6144}` 均取 `0xe01`，即 count=14、direction=1，`csrc/swizzle.cpp:11-18`）。

### L1/L0 缓冲布局

`Init`（`matmul.h:31-123`）按 tile 尺寸静态规划各 buffer 偏移：

| Buffer | 位置 | 份数 | 大小 |
|--------|------|------|------|
| l1aBuf | A1（L1） | 2（乒乓） | `m0 * kDtileSize * dtypeBits/8` |
| l1bBuf | B1（L1） | 2（乒乓） | `n0 * k0 * dtypeBits/8` |
| l1BiasBuf | C1（L1） | 1 | `n0 * sizeof(MatDtype)` |
| l1DeqScaleBuf | C1（L1） | 1 | `n0 * sizeof(uint64_t)` |
| l0aBuf / l0bBuf | A2 / B2（L0） | 2（乒乓） | `m0/n0 * kQtileSize * dtypeBits/8` |
| l0BiasBuf | C2（Bias Table，1KB，数据块 64B） | 1 | - |
| fixpipeBuf | C2PIPE2GM（fixpipe buffer，2KB，数据块 128B） | 1 | - |
| l0cBuf | CO1（L0C） | 1 | 累加结果 |

bias 从 L1 到 C2 的搬运以 64B 数据块为单位（`C2_DATABLOCK`），deqScale 从 L1 到 fixpipe buffer 以 128B 数据块为单位（`FIXPIPE_DATABLOCK`），见 `matmul.h:242-245`、`matmul.h:261-265`。

### 流水线同步（事件旗标）

`SetFlags`/`WaitFlags`（`matmul.h:125-137`、`161-173`）在 tile 生命周期的起止成对补齐全部事件；tile 内部使用：

- `MTE1_MTE2` EVENT_ID0/1（+乒乓位）：L1A 缓冲释放/占用，A 的 GM→L1 用；
- `MTE1_MTE2` EVENT_ID2/3（+乒乓位）：L1B 缓冲，B 的 GM→L1 用；
- `M_MTE1` EVENT_ID0/1（+`kIdx2`）：L0A/L0B 就绪，Mmad 消费后释放；
- `M_MTE1`/`MTE1_M` EVENT_ID4：bias 的 L1↔C2 链路独占；
- `FIX_MTE2`/`MTE2_FIX` EVENT_ID5：deqScale 的 L1↔fixpipe 链路（属于 fixpipe barrier，`matmul.h:259` 注释）；
- `M_FIX`/`FIX_M` EVENT_ID0：L0C 就绪与 fixpipe 搬出完成。

乒乓选择器是 tile 内局部变量（每 tile 从 0 开始），因为 tile 退出时两级缓冲都已释放（`matmul.h:207-210` 注释）。L1A 的乒乓在 `kIdx8==7` 或末轮翻转（`matmul.h:330-333`），L1B 在 `kIdx4==3` 或末轮翻转（`matmul.h:346-349`）。

### 量化路径（int8 / int4）

- int8（W8A8）：`Matmul<int8_t, int32_t, half>`（`matmul.h:467-470`）。A/B 以 int8 进入 Mmad，L0C 累加为 int32；fixpipe 阶段以 `VDEQF16` 模式将 int32 乘以 deqScale 向量后转 fp16 写 GM（`kernel_macro.h:331-336`）。
- int4（W4A4 / W4A8 激活拆分）：`Matmul<int4b_t, int32_t, half>`。dtypeBits 按 4 计（`GetDTypeBits`，`kernel_macro.h:71`），因此 `kBlockSize=64`、`nBlockSize`（转置时）=64，`CopyGmToL1Nd2Nz` 对 int4 把 `srcDValue/dValue` 折半（int4 两两打包进一个字节，`kernel_macro.h:144-145`），B 转置路径的 `CopyToL0BTCol` 用 `LoadDataWithTranspose` 以 32x32x1B 分形、dstGap=3 解包搬运（`kernel_macro.h:196-204`）。W4A8 场景下激活先由 `unpack_activation` 拆成 `[2m, k/2]` 的 int4 打包矩阵（行交错 low/high 4bit），本算子按 int4 GEMM 计算，结果再由 `msd_merge_dequant` 合并（见 group_matmul 文档与 `tests/kernels/matmul_int4.py`）。
- deqScale 的硬件格式：fixpipe 要求以 uint64_t 存储 fp32——高 32 位为 0、低 32 位为 fp32 二进制值（`matmul.h:258-259` 注释）；有效精度为 TF32（低 13 位尾数忽略，`matmul.h:19` 注释）。Python 侧生成 `[2N]` fp32 数组，偶数位放 scale、奇数位补 0（`tests/kernels/matmul_int8.py:101-104`）。

### 权重 NZ 与 bias 融合

- `weightNZ=True` 时 y 已是 Fractal NZ 格式（`torch_npu.npu_format_cast(..., ACL_FORMAT_FRACTAL_NZ)` 或 `matrix_nd2nz`），kernel 走 `CopyGmToL1` 免去 ND→NZ 转换（`matmul.h:300-303`、`307-311`），节省 MTE2 带宽；float 类型不支持 NZ（测试中 `weight_nz and dtype != torch.float`）。
- bias 仅在量化路径为 int32；浮点路径 MatDtype=float，bf16 调用时 host 先用 `cast_bfloat16_t_float` 把 bias 升精度（`csrc/op.cpp:718-722`）。bias 在首个 K 块通过 `Mmad(c, a, b, bias, params)` 一次进入 L0C（`matmul.h:356-360`）。
