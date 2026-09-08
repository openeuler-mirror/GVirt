# group_matmul

## 功能概述

MoE（Mixture of Experts）分组矩阵乘：把按专家排布（permutation 后各专家 token 连续存放）的激活 `x` 与一组专家权重 `weights[i]` 逐专家做 GEMM，结果按同样的行偏移写回输出。单专家语义同 `matmul`（`z_slice = x_slice * w_i^T`，可选 fixpipe 反量化 deqScale），是 `matmul` 类的批量薄封装。支持 BF16/FP16/FP32 浮点与 int8（W8A8）/int4（W4A8 MSD 流水）量化路径。

## 输入输出参数

kernel 入口（`csrc/kernels/group_matmul.h:67`）：

```cpp
group_matmul_##dtype(x, ws, z, deqScales, counts, n, kN, kK, m0, n0, k0,
                     startIdx, endIdx, weightNZ, transpose, swizzle)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入 | [sum(counts), kK]（ND，行主序） | float16 / bfloat16 / float / int8 / int4 | 所有专家 token 拼接的激活矩阵；第 i 个专家的行区间由 counts 前缀和决定。int4（W4A8 MSD）时为 `unpack_activation` 输出的 `[2m, kK/2]` int4 打包矩阵，逻辑上 `[2m, kK]` |
| ws | 输入 | [num]（指针数组，int64） | uint64 | 每个元素是专家 i 的权重 GM 地址。host 侧把 `std::vector<at::Tensor>` 的指针 MemcpyH2D 到 device（`csrc/_C.cpp:1758-1763`） |
| 每个权重 | 输入 | [kN, kK]（transpose=0）或 [kK, kN]（transpose=1）；weightNZ 时为 Fractal NZ 格式 | 同 x | 专家权重，第 i 个专家 `weightAddr = *((__gm__ uint64_t *)(ws + i*8))`（`group_matmul.h:43`） |
| z | 输出 | [sum(counts), kN]（ND）；int4 路径为 [2m, kN] | 同 x（浮点）；量化路径 float16 | 输出。第 i 个专家写到行偏移 `off = sum(counts[0..i))`（int4 时 `off` 按 2*kM 累加，`group_matmul.h:51-52`） |
| deqScales | 输入（可选） | [num]（指针数组）或空 | uint64（低位 32bit fp32/TF32） | 每个专家一个 deqScale 向量（`[kN]`），布局同 matmul 的 deqScale（fixpipe uint64 格式）；传空指针则不做反量化 |
| counts | 输入 | [n]（n >= endIdx） | uint32 | 每个专家的 token 数。int4 时 kernel 内部取 `kM = 2 * counts[i]`（激活已拆成 low/high 两半交错行，`group_matmul.h:37-41` 注释） |
| n | 标量 | - | uint32 | counts 长度（专家总数上限） |
| kN | 标量 | - | int64 | moe intermediate size（单专家 GEMM 的 N） |
| kK | 标量 | - | int64 | hidden size（单专家 GEMM 的 K） |
| m0, n0, k0 | 标量 | - | uint64 | 分块尺寸；host 固定传 -1（用 kernel 默认，`csrc/op.cpp:864-866`） |
| startIdx, endIdx | 标量 | - | uint32 | 本次调用参与的专家索引区间 `[startIdx, endIdx)`（支持分批启动） |
| weightNZ | 标量 | - | bool | 权重是否 NZ 格式 |
| transpose | 标量 | - | bool | 权重是否 `[kK, kN]` 转置存储 |
| swizzle | 标量 | - | uint64 | tile 遍历 swizzle（host 传 `rt.defaultMatmulSwizzle`，`csrc/op.cpp:866`） |

Python 调用方式（`tests/kernels/group_matmul.py:67`、`tests/kernels/group_matmul_int8.py:120`）：

```python
group_matmul(rt, x, weights, deq_scales, counts, start, end, out_dim, in_dim, z, weight_nz, transpose)
```

浮点路径 `deq_scales` 传 `[]`；int8/int4 路径传每专家的 fixpipe 布局 scale 列表（`[2*kN]` fp32 交错，`tests/kernels/group_matmul_int8.py:86-88`）。

## 支持的数据类型

| Dtype（x/权重） | bias（MatDtype） | 输出（OutDtype） | 实例化文件 | kernel 符号 |
|------|------|------|------|------|
| float16_t | float | float16_t | `csrc/kernels/group_matmul_float16_t.cpp` | `group_matmul_float16_t` |
| bfloat16_t | float | bfloat16_t | `csrc/kernels/group_matmul_bfloat16_t.cpp` | `group_matmul_bfloat16_t` |
| float | float | float | `csrc/kernels/group_matmul_float.cpp` | `group_matmul_float`（仅 `!transpose`，`csrc/op.cpp:850-851`） |
| int8_t | int32_t | half | `csrc/kernels/group_matmul_int8_t.cpp` | `group_matmul_int8_t`（W8A8，`in/weight INT8 + out FP16`） |
| int4b_t | int32_t | half | `csrc/kernels/group_matmul_int4b_t.cpp` | `group_matmul_int4b_t`（W4A8 MSD，`in INT4(int32 容器 View) + weight INT4 + out FP16`） |

dtype 分派在 `csrc/op.cpp:846-860`：`in.dtype == INT8 && weightDtype == INT8 && output FP16` 走 int8；`in.dtype == INT4 && weightDtype == INT4 && output FP16` 走 int4。W4A8 场景 Python 侧把 int32 容器的激活与权重 View 成 INT4（`csrc/_C.cpp:1753-1757`、`1768-1771`）。

## 实现原理

### 整体结构：Matmul 类的批量驱动

`group_matmul_kernel`（`csrc/kernels/group_matmul.h:13-64`）实例化一个 `Matmul<Dtype, MatDtype, OutDtype>`（复用 `csrc/kernels/matmul.h` 的完整 Cube 流水，包括 Mmad、fixpipe 反量化、swizzle），然后：

1. `matmul_op.Init(m0, n0, k0, /*hasBias=*/false, useDequant, transpose, weightNZ, swizzle)`——group_matmul 不支持 bias；
2. `SetFlags()` 一次性建立全部事件旗标（`matmul.h:125-137`）；
3. 遍历专家 `i ∈ [startIdx, endIdx)`：
   - 从 GM 读 `kM = counts[i]`（int4 时翻倍）；`kM == 0` 的专家直接跳过（`group_matmul.h:34-36`）；
   - 从 `ws` 指针数组取权重地址，从 `deqScales` 指针数组（若使用）取 scale 地址（`group_matmul.h:43-49`）；
   - 计算本专家激活/输出在拼接矩阵中的字节偏移：`xOffBytes = off * kK * dtypeBits/8`、`zOffBytes = off * kN * sizeof(OutDtype)`（`group_matmul.h:51-52`）；
   - `TaskTilesInit(x + xOffBytes, w, z + zOffBytes, nullptr, deqScale, kM, kN, kK)` 绑定本专家的 GM 地址与形状，返回 tile 数（`matmul.h:142-159`）；
   - 本核执行 `for (idx = (blockIdx + blockNum - coreOffset) % blockNum; idx < tiles; idx += blockNum) RunTileByIdx(idx)`；
   - `coreOffset = (coreOffset + tiles) % blockNum`，`off += kM`。
4. 所有专家完成后统一 `WaitFlags()`（`group_matmul.h:63`）。

### 多核负载均衡

关键在 `coreOffset`（`group_matmul.h:56-60`）：每个专家的 tile 从 `(blockIdx - coreOffset) mod blockNum` 起步交错认领。由于各专家 tile 数（`ceil(kM/m0) * ceil(kN/n0)`）不整除核数时会有剩余 tile，`coreOffset` 把上一专家的余数累计传递，使下一个专家的起始 tile 偏移相应挪动——避免每个专家都由同一批核处理"尾巴"，实现跨专家的近似均匀分工。单专家内部与 matmul 相同：tile 按 `idx % nLoop / idx / nLoop` 映射到 (midx, nidx)，再经 `GetMNBlockIdx` swizzle 重排（`matmul.h:177-184`、`380-419`）。

### int4（W4A8 MSD）路径

W4A8 的完整流水分三级（见 `tests/kernels/group_matmul_int4.py:41-58` 的 CPU 参考实现注释，对应 `XModel::ForwardMoEMSD`）：

1. `unpack_activation`：INT8 `[m, k]` 激活做 floor 分解为 low4（`x & 0x0F - 8`）/high4（`x >> 4`），打包成 `[2m, k/2]` int4 矩阵（行交错）；
2. 本算子（int4 变体）：per-expert `(low4|high4) [2c, k] x W_int4` → INT32 L0C → fixpipe `VDEQF16` 乘 per-channel deq_scale → FP16 `[2c, n]`（行交错）。kernel 内 `kM = 2 * counts[i]`（`group_matmul.h:39-41`）、`off` 按翻倍后的 kM 累加，保证 low/high 行与输出行一一交错对齐；
3. `msd_merge_dequant`：`(y_high*16 + y_low + scale_bias) * per_token_scale` 合并为 BF16 `[m, n]`，其中 `scale_bias = 8 * sum_k(W[k,col] * deq_scale[col])` 补偿 low4 的 -8 偏移。

int4 的搬运细节（在 matmul.h/kernel_macro.h 中）：`dtypeBits=4` 使 `kBlockSize=64`、`nBlockSize`（transpose）=64；`CopyGmToL1Nd2Nz` 对 int4 折半 srcDValue/dValue（`kernel_macro.h:143-145`）；B 转置路径 `CopyToL0BTCol` 以 `LoadDataWithTranspose` 32x32x1B 分形、dstGap=3 解包（`kernel_macro.h:196-204`）。

### 反量化（deqScale）

与 matmul 相同的 fixpipe 机制：deqScale（uint64 容器，低 32 位 fp32/TF32）先 GM→L1（C1），再以 128B 数据块搬到 C2PIPE2GM fixpipe buffer（`matmul.h:250-267`），最终在 `CopyToGmWithDequant` 中经 `SetFixPipeConfig(deqScale)` + `VDEQF16` 完成逐 N 通道反量化（`kernel_macro.h:313-340`）。每个专家的 scale 向量独立绑定（TaskTilesInit 传入该专家的 deqScale 地址）。浮点路径 `deqScales == nullptr` 时 `useDequant=false`，跳过全部 deqScale 搬运。

### 与 matmul 的差异总结

- 无 bias（`Init(..., false, useDequant, ...)`，`group_matmul.h:25`）；
- 多一层"专家循环 + counts 前缀和偏移"的批处理，权重/scale 通过 device 指针数组间接寻址；
- `coreOffset` 跨专家 tile 均衡；
- m0/n0/k0 由 kernel 默认值决定（host 传 -1，`csrc/op.cpp:865`），即 `m0=128`、`n0 = useDequant ? 128 : 256`、`k0 = 512*8/dtypeBits`（`matmul.h:40-46`）。
