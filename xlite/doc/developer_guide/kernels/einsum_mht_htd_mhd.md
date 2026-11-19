# einsum_mht_htd_mhd

## 功能概述

批量头独立的矩阵乘:`mhd = torch.einsum("mht,htd->mhd", mht, htd)`,对每个头 h 做一次 `[m,t] × [t,d] = [m,d]` 的 GEMM(右操作数按 `[h, t, d]` 转置布局给出,matmul 侧走 transpose=1 路径)。主要用于 MLA 类注意力的逐头吸收投影:Q 吸收(`qAbsorb[m,h,d] = attnQWithQr[m,h,t] · W_UKᵀ[h,t,d]`)与输出吸收(`attnOutput[m,h,v] = oAbsorb[m,h,t] · W_UV[h,t,d]`),两处调用均在 `csrc/model.cpp` 的 `XModel::ForwardAttnMLACommonV2` 内(关键字 `XliteOpEinsumMhtHtdMhd`,分别对应 qAbsorb 与 attnOutput 两次调用)。

## 输入输出参数

Python 接口:`einsum_mht_htd_mhd(rt, mht, htd, mhd, m, h, t, d, weight_nz=False)`(`XliteOpEinsumMhtHtdMhd`,见 `csrc/op.cpp` 的 `XliteOpEinsumMhtHtdMhd`)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| mht | 输入 | `[m, h, t]`(逻辑上 `[m, h*T]` 连续存储) | float16 / bfloat16 | 左操作数,按头切出 `[m, t]` 子矩阵作为 GEMM 的 A |
| htd | 输入 | `[h, t, d]` | 同上 | 右操作数(权重),行主序转置布局;`weight_nz=True` 时为 NZ 分形布局 |
| mhd | 输出 | `[m, h, d]`(逻辑上 `[m, h*D]`) | 同上 | 逐头 GEMM 结果 |
| m | 标量 | - | uint32 | token 数(GEMM 的 M 维) |
| h | 标量 | - | uint32 | 头数(GEMM 批次数),MLA 中为 nLocalHeads |
| t | 标量 | - | uint32 | 收缩维(GEMM 的 K 维),如 kvLoraRank |
| d | 标量 | - | uint32 | 输出维(GEMM 的 N 维),如 kvLoraRank / vHeadDim |
| weight_nz | 标量 | - | bool | 右操作数是否为 ND2NZ 分形布局 |
| T / D | 标量(可选) | - | int | mht 行宽 / mhd 行宽的对齐覆盖;缺省(`-1`)时取 t / d。Q 吸收场景传 `nopeHeadDim + ropeHeadDim`(A 行距为完整 head 宽,只取前 t 列参与计算,见 `csrc/model.cpp` 的 `XModel::ForwardAttnMLACommonV2`,关键字 `_c.nopeHeadDim + _c.ropeHeadDim`) |

host 侧 swizzle 由 `XlitePickSwizzle(d, t, ...)` 依据 N/K 形状挑选(见 `csrc/op.cpp` 的 `XliteOpEinsumMhtHtdMhd`,关键字 `XlitePickSwizzle`);`m0/n0/k0` 取默认值(M0=128,N0=256,K0 按 dtype)。

## 支持的数据类型

- `float16_t`([einsum_mht_htd_mhd_float16_t.cpp](../../../csrc/kernels/einsum_mht_htd_mhd_float16_t.cpp))
- `bfloat16_t`([einsum_mht_htd_mhd_bfloat16_t.cpp](../../../csrc/kernels/einsum_mht_htd_mhd_bfloat16_t.cpp))

实例化形态为 `Matmul<dtype, float, dtype>`(见 `csrc/kernels/einsum_mht_htd_mhd.h` 的 `EINSUM_MHT_HTD_MHD_FUNC_DEFINE` 宏);三个张量 dtype 须一致。

## 实现原理

实现位于 [csrc/kernels/einsum_mht_htd_mhd.h](../../../csrc/kernels/einsum_mht_htd_mhd.h),是通用 GEMM 模板 `Matmul`([csrc/kernels/matmul.h](../../../csrc/kernels/matmul.h))外的一层逐头批处理封装。kernel 声明为 `KERNEL_TYPE_AIC_ONLY`,由 host 以 `rt.aicNum` 个 Cube 核启动。与 `einsum_mht_hdt_mhd` 仅有一处本质差异:`matmul_op.Init(..., transpose=1, ...)`(见 `csrc/kernels/einsum_mht_htd_mhd.h` 的 `einsum_mht_htd_mhd` 模板函数,关键字 `transpose=1` 的 `Init` 调用),即 B 装载走转置分支。

### 逐头批处理与跨头负载均衡

```text
for hIdx in 0..h:
    x = mht + hIdx * T * sizeof(Dtype)        // A: 头内 [m, t],行距 T
    y = htd + hIdx * t * d * sizeof(Dtype)    // B: 头内 [t, d]
    z = mhd + hIdx * D * sizeof(Dtype)        // C: 头内 [m, d],行距 D
    tiles = matmul_op.TaskTilesInit(x, y, z, ..., m, d, t, srcDStride=h*T, dstDStride=h*D)
    for idx in (blockIdx + blockNum - coreOffset) % blockNum .. tiles step blockNum:
        matmul_op.RunTileByIdx(idx)
    coreOffset = (coreOffset + tiles) % blockNum
```

(见 `csrc/kernels/einsum_mht_htd_mhd.h` 的 `einsum_mht_htd_mhd` 模板函数,关键字 `coreOffset`)

- 每头切成 `ceil(m/m0) × ceil(d/n0)` 个输出 tile;头间用累计 `coreOffset` 错开起始 block,均衡多核负载;
- `srcDStride = h*T`、`dstDStride = h*D` 把 A/C 的行距改写为含跨头跳步的物理行距;
- `yStride = d * t * sizeof(Dtype)` 与 hdt 变体的 `yStride` 相同——`[h,t,d]` 与 `[h,d,t]` 单头字节数一致,仅头内解释不同。

### 单 tile 数据流(Matmul 类,transpose=1 分支)

每个 `(m0=128) × (n0=256)` 输出 tile 在 Cube 核内执行五级流水(见 `csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`):

1. **A GM→L1**:`CopyGmToL1Nd2Nz` 把 `[mActual, kRem]` 的 A 转成 NZ 分形进 `l1aBuf`(K 方向双缓冲,`kDtileSize = 2*k0` 预取);
2. **B GM→L1(本算子专属分支)**:`transpose=1 && nz==0` 时走 `CopyGmToL1Nd2Nz(bGmBuf[kOffset*n + nOffset], kRemSize, nActual, n, ...)`——B 逻辑上是 `[t, d]`(K×N),以 n 为行距、交换 NK 角色做 ND2NZ,直接产出转置后的分形(见 `csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`,关键字 `transpose == 1 && nz == 0`);`nz==1` 时按 `kStride` 行距 `CopyGmToL1` 拷贝现成 NZ 分形(同函数内,关键字 `nz == 1` 的分支);
3. **L1→L0A/L0B**:`CopyToL0ACol` / **`CopyToL0BTCol`**(转置专用,带 k0ActualBlockNum 参数)分形下搬,各自 ping-pong 双缓冲(见 `csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`,关键字 `CopyToL0BTCol` 与 `transpose`);
4. **Mmad**:K 方向按 `kQtileSize` 分段累加到 L0C(`CalMmad`,kIdx==0 初始化);
5. **L0C→GM**:`CopyToGmWithDequant` 写回 C tile(无 bias/dequant)。

同步用 `MTE1_MTE2`(L1 空闲)、`M_MTE1`(L0 就绪)、`M_FIX`(Mmad 完成)事件链,详见 [matmul.h](../../../csrc/kernels/matmul.h) 的 `RunTileBody`。

### tiling/swizzle

tile 遍历顺序由 `GetMNBlockIdx` 按 host 传入的 swizzle(方向 Zn/Nz + 块计数)重排,改善 cache 命中率(见 `csrc/kernels/matmul.h` 的 `Matmul::GetMNBlockIdx`)。

### 测试参考

[tests/kernels/einsum_mht_mhd.py](../../../tests/kernels/einsum_mht_mhd.py):fp16/bf16 × weight_nz {False, True} × (m,h,t,d) 多组形状,`htd` 直接 `contiguous()` 生成,与 `torch.einsum("mht,htd->mhd")` 对比;[tests/kernels/linear_att_proj.py](../../../tests/kernels/linear_att_proj.py) 覆盖线性注意力 QKV/Z/B/A 投影的组合 matmul 路径(其内部经 `XliteOpMatmul` 复用同一 Matmul 模板)。

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C)

`Matmul::Init`(`csrc/kernels/matmul.h` 的 `Matmul::Init`)按 tile 形状静态排布,布局与 `einsum_mht_hdt_mhd` 完全一致:L1 上 `l1aBuf`(`TPosition::A1`)与 `l1bBuf`(`TPosition::B1`)同处 512KB 物理存储、`bufferAddr` 连续累进,各两份乒乓;L0A/L0B/L0C 物理独立、各从偏移 0 起排;本算子 `hasBias=hasDeqScale=false`,bias/deqScale 槽位不占。`transpose=1` 只改 B 的装载与下搬分支——GM→L1 走 `CopyGmToL1Nd2Nz` 的 NK 角色交换、L1→L0 走 `CopyToL0BTCol`(均见 `csrc/kernels/matmul.h` 的 `Matmul::RunTileBody`,关键字 `transpose == 1 && nz == 0`)——`l1bBuf`/`l0bBuf` 的槽位大小不变;`nBlockSize = 32*BYTE_BITS/dtypeBits`(transpose=1)与 `transpose=0` 的 16 在 fp16/bf16(dtypeBits=16)下同为 16,仅 int8 路径才翻倍,故两级布局公式与 hdt 变体一字不差。host 侧 launch 同样固定传 `MATMUL_M0_N0_K0_DEFAULT_VALUE`(=(uint64_t)-1,`csrc/kernels/kernel_macro.h` 的 `MATMUL_M0_N0_K0_DEFAULT_VALUE`,调用见 `csrc/op.cpp` 的 `XliteOpEinsumMhtHtdMhd`),kernel 内解析为 `m0=128、n0=256、k0=512*BYTE_BITS/dtypeBits`。本算子 `KERNEL_TYPE_AIC_ONLY`,无 AIV/UB。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置):L0A 64KB、L0B 64KB、L1 512KB,L0C 总容量 256KB(`l0_c_size=262144`)为 CO1/CO2 两区之和,mmad 累加缓冲(`TPosition::CO1`)可用的 CO1 区为 **128KB**,下文 L0C 占用均按 CO1 128KB 计。

dtype 记法:`Dtype` 为 mht/htd/mhd 三张量的元素类型,`MatDtype` 为 mmad 累加/偏置类型(实例化 `Matmul<Dtype, float, Dtype>`,见 `csrc/kernels/einsum_mht_htd_mhd.h` 的 `EINSUM_MHT_HTD_MHD_FUNC_DEFINE`);fp16 与 bf16 均为 2B、dtypeBits=16,即 **k0=256**(`kDtileSize=512`、`kQtileSize=64`),两条路径占用相同,下表只给单占用列。缓冲大小只依赖 m0/n0/k0/dtype,与运行期 (m,h,t,d) 及 weight_nz/transpose 无关;典型配置取 MLA 输出吸收类测试形状 (m,h,t,d)=(963,8,512,128)(`tests/kernels/einsum_mht_mhd.py`,模型路径 `csrc/model.cpp` 的 `XModel::ForwardAttnMLACommonV2` 的 qAbsorb/attnOutput 两次调用)。

**AIC L1(A1/B1,容量 512KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l1aBuf` | `(128, 512)`(m0 × kDtileSize,K 方向双倍预取) | Dtype | 2(事件0/1) | `m0*kDtileSize*dtypeBits/BYTE_BITS` | 262144 |
| `l1bBuf` | `(256, 256)`(n0 × k0) | Dtype | 2(事件2/3) | `n0*k0*dtypeBits/BYTE_BITS` | 262144 |
| **合计** | | | | | **524288** |

**AIC L0A(A2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0aBuf` | `(128, 64)`(m0 × kQtileSize) | Dtype | 2(事件0/1) | `m0*kQtileSize*dtypeBits/BYTE_BITS` | 32768 |
| **合计** | | | | | **32768** |

**AIC L0B(B2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0bBuf` | `(256, 64)`(n0 × kQtileSize) | Dtype | 2(事件0/1) | `n0*kQtileSize*dtypeBits/BYTE_BITS` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB;总容量 256KB 含 CO2)**

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

- **L1 恰好占满 512KB,L0B 与 L0C(CO1)同步顶满**,与 hdt 变体数值完全一致——默认 tile 形状 (m0,n0,k0)=(128,256,256) 按 910B 各级容量上限反推,host 固定传 `MATMUL_M0_N0_K0_DEFAULT_VALUE` 是排布成立的前提,kernel 侧无运行期断言,传入更大的 m0/n0/k0 会直接越界;
- `l1aBuf` 与 `l1bBuf` 各占 L1 一半(单份各 128KB),是 L1 大头;`kQtileSize=k0/4=64` 让 L0A 单份仅 16KB(占比 50%),一份 L1 预取窗服务 8 次 L0A 下搬、4 次 L0B 下搬;
- `transpose=1` 的转置语义全部在装载路径内消化(`CopyToL0BTCol` 带 `k0ActualBlockNum` 参数),片上槽位与 hdt 变体一字不差,`nBlockSize` 在 fp16/bf16 下也不随 transpose 变化(仅 int8 路径翻倍为 32);
- L0C 按 CO1 128KB 计:单份 m0×n0×4B=131072 恰好占满,K 方向按 `kQtileSize=64` 分段累加(`CalMmad` 的 kIdx==0 初始化),累加器无需超过一个输出 tile。
