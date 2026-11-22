# einsum_mht_hdt_mhd

## 功能概述

批量头独立的矩阵乘:`mhd = torch.einsum("mht,hdt->mhd", mht, hdt)`,对每个头 h 做一次 `[m,t] × [d,t]ᵀ = [m,d]` 的 GEMM(右操作数按 `[h, d, t]` 非转置布局直接给出,matmul 侧走 transpose=0 路径)。是 `einsum_mht_htd_mhd`(右操作数 `[h,t,d]` 转置布局)的姊妹算子,主要用于 MLA 类注意力的逐头吸收投影;当前模型代码中主要使用 htd 变体,本算子通过 Python 绑定暴露供测试与自定义流程使用。

## 输入输出参数

Python 接口:`einsum_mht_hdt_mhd(rt, mht, hdt, mhd, m, h, t, d, weight_nz=False)`(`XliteOpEinsumMhtHdtMhd`,见 `csrc/op.cpp` 的 `XliteOpEinsumMhtHdtMhd`)。`weight_nz` 已废弃并将被移除:传入值被忽略,`hdt` 的 FRACTAL_NZ 存储格式由 `InitXTensor` 自动检测(host 侧取 `hdt.nz` 传入 kernel,见 `csrc/op.cpp` 的 `XliteOpEinsumMhtHdtMhd`,关键字 `const bool weightNZ`)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| mht | 输入 | `[m, h, t]`(逻辑上 `[m, h*T]` 连续存储,T 为对齐后的行宽) | float16 / bfloat16 | 左操作数,按头切出 `[m, t]` 子矩阵作为 GEMM 的 A |
| hdt | 输入 | `[h, d, t]` | 同上 | 右操作数(权重),非转置布局;FRACTAL_NZ 分形布局自动检测(`weight_nz` 已废弃,传入值被忽略) |
| mhd | 输出 | `[m, h, d]`(逻辑上 `[m, h*D]`) | 同上 | 逐头 GEMM 结果 C = A·Bᵀ |
| m | 标量 | - | uint32 | token 数(GEMM 的 M 维),测试覆盖 3 ~ 5230 |
| h | 标量 | - | uint32 | 头数(GEMM 批次数) |
| t | 标量 | - | uint32 | 收缩维(GEMM 的 K 维),如 MLA 的 nopeHeadDim/kvLoraRank |
| d | 标量 | - | uint32 | 输出维(GEMM 的 N 维) |
| weight_nz | 标量 | - | bool | 已废弃:传入值被忽略(NZ 自动检测);将被移除 |
| T / D | 标量(可选) | - | int | mht 行宽 / mhd 行宽的对齐覆盖;缺省(`-1`)时取 t / d。MLA 流程中传入 `nopeHeadDim + ropeHeadDim` 以跳过 head 中不参与计算的 rope 区(见 `csrc/_C.cpp` 的 `MLAV2`,关键字 `nopeHeadDim + ropeHeadDim`) |

host 侧 swizzle 策略由 `XlitePickSwizzle(d, t, ...)` 依据 N/K 形状挑选(见 `csrc/op.cpp` 的 `XliteOpEinsumMhtHdtMhd`,关键字 `XlitePickSwizzle`);`m0/n0/k0` 取 `MATMUL_M0_N0_K0_DEFAULT_VALUE`(即 128/256/按 dtype)。

## 支持的数据类型

- `float16_t`([einsum_mht_hdt_mhd_float16_t.cpp](../../../csrc/kernels/einsum_mht_hdt_mhd_float16_t.cpp))
- `bfloat16_t`([einsum_mht_hdt_mhd_bfloat16_t.cpp](../../../csrc/kernels/einsum_mht_hdt_mhd_bfloat16_t.cpp))

实例化形态为 `Matmul<dtype, float, dtype>`(bias 为 float 类型形参但未启用,见 `csrc/kernels/einsum_mht_hdt_mhd.h` 的 `EINSUM_MHT_HDT_MHD_FUNC_DEFINE` 宏);三个张量 dtype 须一致。

## 实现原理

实现位于 [csrc/kernels/einsum_mht_hdt_mhd.h](../../../csrc/kernels/einsum_mht_hdt_mhd.h),是通用 GEMM 模板 `Matmul`([csrc/kernels/matmul.h](../../../csrc/kernels/matmul.h))外的一层逐头批处理封装。kernel 声明为 `KERNEL_TYPE_AIC_ONLY`,由 host 以 `rt.aicNum` 个 Cube 核启动(与向量核算子不同)。

### 逐头批处理与跨头负载均衡

所有头的 GEMM 几何相同,`TaskTilesInit` 只在头 0 上执行一次,后续头仅通过 `Matmul::RebindGm`(换 A/B/C 基址并重置 residency 跟踪器,见 `csrc/kernels/matmul.h` 的 `Matmul::RebindGm`)移动基址。按 `mLoop` 分两条调度路径(见 `csrc/kernels/einsum_mht_hdt_mhd.h` 的 `einsum_mht_hdt_mhd` 模板函数,关键字 `MLoop() >= 2`):

**mLoop >= 2 且 t ≤ 8*k0(prefill)**:全部 `h × mLoop × nLoop` 个 tile 按 `头 -> nidx -> midx` 内层排序后,切成每核连续的 chunk(每核 `ceil(totalTiles/blockNum)` 个 tile,`Matmul::RunTileByMN` 直接按 (midx, nidx) 寻址绕过 swizzle 重排)。`t ≤ 8*k0` 即 B 的两个 l1bBuf 乒乓槽(各 `k4TileSize = 4*k0`)能同时容纳整个 K 范围,是头内跨 tile B 复用成立的前提(`Matmul::BWholeKResident`);更大的 k 下 k 乒乓会回收槽位、覆写跟踪键,跨 tile 复用衰减到只剩最后两个窗口,chunked 分组无重载可省:

```text
tiles = TaskTilesInit(mht, hdt, mhd, ..., m, d, t, srcDStride=h*T, dstDStride=h*D)  // 头 0
totalTiles = tiles * h; chunk = ceil(totalTiles / blockNum)
for idx in blockIdx*chunk .. min((blockIdx+1)*chunk, totalTiles):
    (hIdx, nidx, midx) = 按 idx = (hIdx*nLoop + nidx)*mLoop + midx 分解
    hIdx 变化时 RebindGm(mht + hIdx*xStride, hdt + hIdx*yStride, mhd + hIdx*zStride)
    RunTileByMN(midx, nidx)   // midx 先耗尽再进位 nidx,nidx 耗尽进位 hIdx
```

- 目的:让每核只连续接触 ~2-3 个头而非全部 h 个头。B 权重的 GM→L1 重载按 (头, 核) 对发生(`RebindGm` 重置跟踪器,跨头不能复用),prefill 时该对数从 h × blockNum 缩到 ~h/chunk × blockNum,即约缩小 blockNum 倍;头内则因 midx 连续被 matmul 的 L1 residency 跟踪器持续命中。chunk 边界切开的头由相邻两核各加载一次(matmul 的 flag/跟踪器均为核内私有状态);
- 代价:chunked 路径忽略 host 传入的 swizzle(其 L2 局部性职责由组内连续遍历接管),A 流量主导且头较少的形状可能回退;
- `srcDStride = h*T`、`dstDStride = h*D` 把 A 的行距与 C 的行距从逻辑 k/n 改写为含跨头跳步的物理行距,使 matmul 搬运时直接跳到正确的头内子矩阵。

**mLoop == 1(decode)**:保留原来的头间错峰轮转(每头 tile 数不变,`coreOffset` 累计错开起始 block),但头 1..h-1 用 `RebindGm` 替代逐头完整的 `TaskTilesInit`(省去每头每核 5 次 `SetGlobalBuffer` 与形状除法标量开销;decode 时每核每头至多 1 个 tile,residency 无跨 tile 复用可言)。

### 单 tile 数据流(Matmul 类)

每个 `(m0=128) × (n0=256)` 输出 tile 在 Cube 核内执行标准五级流水(见 `csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`):

1. **A GM→L1**:`CopyGmToL1Nd2Nz` 把 `[mActual, kRem]` 的 A 转成 NZ 分形进 `l1aBuf`(K 方向双缓冲,K 预取 `k8TileSize = 8*k0`);
2. **B GM→L1**:本算子 `transpose=0` 且 `nz=0` 时走 `CopyGmToL1Nd2Nz(bGmBuf[nOffset*k + kOffset], nActual, kRemSize, k, ...)`——注意 B 逻辑上是 `[d, t]`(N×K),以 k 为行距 ND2NZ 装载,等效完成 Bᵀ 的分形化(见 `csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`,关键字 `transpose == 0 && nz == 0`);`nz=1` 时直接 `CopyGmToL1` 拷贝现成 NZ 分形;
3. **L1→L0A/L0B**:`CopyToL0ACol` / `CopyToL0BCol` 分形下搬,均带 ping-pong 双缓冲;
4. **Mmad**:K 方向按 `k0` 分段累加到 L0C(`CalMmad`,kIdx==0 时初始化累加器);
5. **L0C→GM**:`CopyToGmWithDequant` 写回 C tile(本算子无 bias/dequant)。

同步用 `MTE1_MTE2`(L1 空闲)、`M_MTE1`(L0 就绪)、`M_FIX`(Mmad 完成)事件链,L1A/L1B 各自独立 ping-pong,详见 [matmul.h](../../../csrc/kernels/matmul.h) 的 `RunTileBody`。

### tiling/swizzle

tile 遍历顺序由 `GetMNBlockIdx` 按 host 传入的 swizzle(方向 Zn/Nz + 块计数)重排,改善 L2/L1 命中率(见 `csrc/kernels/matmul.h` 的 `Matmul::GetMNBlockIdx`)。

### 测试参考

[tests/kernels/einsum_mht_mhd.py](../../../tests/kernels/einsum_mht_mhd.py):fp16/bf16 × weight_nz {False, True} × (m,h,t,d) ∈ {(3,4,192,512), (5230,4,512,128), (963,8,512,128), (257,64,128,128), (257,32,128,512), (5230,4,576,128), (257,64,384,128), ...},`hdt` 用 `htd.transpose(-1,-2).contiguous()` 生成,与 `torch.einsum("mht,htd->mhd")` 对比。m=3 覆盖 mLoop==1 解码路径;m=257/h=64 与 m=257/h=32 覆盖 chunked 路径上单核跨多头（多次 `RebindGm` 头切换）与 midx/nidx 进位回绕;t=576 > 8*k0 覆盖 chunked 门槛外的大 k round-robin 路径;t=384（`4*k0 < t < 8*k0` 且非 `4*k0` 倍数）覆盖单 tile 双不等宽 B 窗口,锚定 `Matmul` 的 per-slot 窗口块数跟踪（`k4TileActualBlockNum`,见 [matmul.md](matmul.md) 的「L1 residency」）。

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C)

`Matmul::Init`(`csrc/kernels/matmul.h` 的 `Matmul::Init`)按 tile 形状静态排布:L1 上 `l1aBuf`(`TPosition::A1`)与 `l1bBuf`(`TPosition::B1`)同处 512KB 物理存储、`bufferAddr` 连续累进,各持两份乒乓(bias/deqScale 的 `l1BiasBuf`/`l1DeqScaleBuf` 与 `l1bBuf` 区段衔接,本算子 `hasBias=hasDeqScale=false` 不占槽位);L0A/L0B/L0C 物理独立、各自从偏移 0 重新起排。host 侧 launch 固定传 `MATMUL_M0_N0_K0_DEFAULT_VALUE`(=(uint64_t)-1,`csrc/kernels/kernel_macro.h` 的 `MATMUL_M0_N0_K0_DEFAULT_VALUE`,调用见 `csrc/op.cpp` 的 `XliteOpEinsumMhtHdtMhd`),kernel 内解析为 `m0=128、n0=256、k0=128*BYTE_BITS/dtypeBits`(单次 Mmad 的 K 量程),并派生 `k8TileSize=8*k0`(A 的 K 方向预取窗)与 `k4TileSize=4*k0`(B 的 L1 tile);L1→L0 下搬粒度即 L0A/L0B 单片 K 深度 `k0`。本算子 `KERNEL_TYPE_AIC_ONLY`,无 AIV/UB。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置):L0A 64KB、L0B 64KB、L1 512KB,L0C 总容量 128KB（`l0_c_size=131072`，16 bank × 8KB），Mmad 累加缓冲即 `TPosition::CO1` 且占满整个 L0C。

dtype 记法:`Dtype` 为 mht/hdt/mhd 三张量的元素类型,`MatDtype` 为 mmad 累加/偏置类型(实例化 `Matmul<Dtype, float, Dtype>`,见 `csrc/kernels/einsum_mht_hdt_mhd.h` 的 `EINSUM_MHT_HDT_MHD_FUNC_DEFINE`);fp16 与 bf16 均为 2B、dtypeBits=16,即 **k0=64**(`k8TileSize=512`、`k4TileSize=256`),两条路径占用相同,下表只给单占用列。缓冲大小只依赖 m0/n0/k0/dtype,与运行期 (m,h,t,d) 及 weight_nz/transpose 无关,占用列对任意形状成立;典型配置取测试形状 (m,h,t,d)=(5230,4,512,128)(`tests/kernels/einsum_mht_mhd.py`,MLA 输出吸收类调用,t=512、d=128)。

**AIC L1(A1/B1,容量 512KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l1aBuf` | `(128, 512)`(m0 × k8TileSize,8 次 Mmad 预取窗) | Dtype | 2(事件0/1) | `m0*k8TileSize*dtypeBits/BYTE_BITS` | 262144 |
| `l1bBuf` | `(256, 256)`(n0 × k4TileSize=4×k0) | Dtype | 2(事件2/3) | `n0*k4TileSize*dtypeBits/BYTE_BITS` | 262144 |
| **合计** | | | | | **524288** |

**AIC L0A(A2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0aBuf` | `(128, 64)`(m0 × k0) | Dtype | 2(事件0/1) | `m0*k0*dtypeBits/BYTE_BITS` | 32768 |
| **合计** | | | | | **32768** |

**AIC L0B(B2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0bBuf` | `(256, 64)`(n0 × k0) | Dtype | 2(事件0/1) | `n0*k0*dtypeBits/BYTE_BITS` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0cBuf` | `(128, 256)`(m0 × n0) | float(MatDtype) | 1(事件0) | `m0*n0*sizeof(MatDtype)` | 131072 |
| **合计** | | | | | **131072** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| L1 | 512 KB | 524288 B | 100% |
| L0A | 64 KB | 32768 B | 50.0% |
| L0B | 64 KB | 65536 B | 100% |
| L0C(CO1) | 128 KB | 131072 B | 100% |

要点:

- **L1 恰好占满 512KB,L0B 与 L0C(CO1)同步顶满**:默认 tile 形状 (m0,n0,k0)=(128,256,64) 正是按 910B 各级容量反推的上限组合,host 固定传 `MATMUL_M0_N0_K0_DEFAULT_VALUE` 是排布成立的前提——kernel 侧无运行期断言,传入更大的 m0/n0/k0 会直接越过 L1/L0 容量(`Matmul::Init` 仅在 `hasDeqScale`/`hasBias` 时把 n0 收到 256 以保护 C2 的 fixpipe/bias 小缓冲,本算子两者均关);
- `l1aBuf` 与 `l1bBuf` 各占 L1 一半:每份 128×512(`k8TileSize=8*k0` 的 K 方向预取窗)与 256×256(`k4TileSize=4*k0`),单份各 128KB,是 L1 占用的绝对大头;
- L0A 只用一半:`k0=64` 让 L0A 单份仅 16KB,一份 `l1aBuf` 预取窗服务 8 次(kIdx8)L0A 下搬、一份 `l1bBuf` 服务 4 次(kIdx4)L0B 下搬,是 L1 预取与 L0 计算粒度解耦的关键;
- L0C 按 CO1 128KB 计:单份 m0×n0×4B=131072 恰好占满,K 方向按 `k0=64` 分段累加(`CalMmad` 的 kIdx==0 初始化),累加器无需超过一个输出 tile;`l0cBuf`/`l0bBuf` 等槽位大小与 htd 变体(transpose=1)完全一致,两变体只有 B 的装载/下搬分支不同。
