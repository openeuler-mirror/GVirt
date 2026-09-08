# einsum_mht_htd_mhd

## 功能概述

批量头独立的矩阵乘:`mhd = torch.einsum("mht,htd->mhd", mht, htd)`,对每个头 h 做一次 `[m,t] × [t,d] = [m,d]` 的 GEMM(右操作数按 `[h, t, d]` 转置布局给出,matmul 侧走 transpose=1 路径)。主要用于 MLA 类注意力的逐头吸收投影:Q 吸收(`qAbsorb[m,h,d] = attnQWithQr[m,h,t] · W_UKᵀ[h,t,d]`)与输出吸收(`attnOutput[m,h,v] = oAbsorb[m,h,t] · W_UV[h,t,d]`),见 [csrc/model.cpp:469-471](../../csrc/model.cpp#L469-L471) 与 [csrc/model.cpp:542-543](../../csrc/model.cpp#L542-L543)。

## 输入输出参数

Python 接口:`einsum_mht_htd_mhd(rt, mht, htd, mhd, m, h, t, d, weight_nz=False)`(`XliteOpEinsumMhtHtdMhd`,[csrc/op.cpp:2145-2167](../../csrc/op.cpp#L2145-L2167))。

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
| T / D | 标量(可选) | - | int | mht 行宽 / mhd 行宽的对齐覆盖;缺省(`-1`)时取 t / d。Q 吸收场景传 `nopeHeadDim + ropeHeadDim`(A 行距为完整 head 宽,只取前 t 列参与计算,[csrc/model.cpp:469-471](../../csrc/model.cpp#L469-L471)) |

host 侧 swizzle 由 `XlitePickSwizzle(d, t, ...)` 依据 N/K 形状挑选([csrc/op.cpp:2160-2163](../../csrc/op.cpp#L2160-L2163));`m0/n0/k0` 取默认值(M0=128,N0=256,K0 按 dtype)。

## 支持的数据类型

- `float16_t`([einsum_mht_htd_mhd_float16_t.cpp](../../csrc/kernels/einsum_mht_htd_mhd_float16_t.cpp))
- `bfloat16_t`([einsum_mht_htd_mhd_bfloat16_t.cpp](../../csrc/kernels/einsum_mht_htd_mhd_bfloat16_t.cpp))

实例化形态为 `Matmul<dtype, float, dtype>`([csrc/kernels/einsum_mht_htd_mhd.h:54](../../csrc/kernels/einsum_mht_htd_mhd.h#L54));三个张量 dtype 须一致。

## 实现原理

实现位于 [csrc/kernels/einsum_mht_htd_mhd.h](../../csrc/kernels/einsum_mht_htd_mhd.h),是通用 GEMM 模板 `Matmul`([csrc/kernels/matmul.h](../../csrc/kernels/matmul.h))外的一层逐头批处理封装。kernel 声明为 `KERNEL_TYPE_AIC_ONLY`,由 host 以 `rt.aicNum` 个 Cube 核启动。与 `einsum_mht_hdt_mhd` 仅有一处本质差异:`matmul_op.Init(..., transpose=1, ...)`([csrc/kernels/einsum_mht_htd_mhd.h:22](../../csrc/kernels/einsum_mht_htd_mhd.h#L22)),即 B 装载走转置分支。

### 逐头批处理与跨头负载均衡

```
for hIdx in 0..h:
    x = mht + hIdx * T * sizeof(Dtype)        // A: 头内 [m, t],行距 T
    y = htd + hIdx * t * d * sizeof(Dtype)    // B: 头内 [t, d]
    z = mhd + hIdx * D * sizeof(Dtype)        // C: 头内 [m, d],行距 D
    tiles = matmul_op.TaskTilesInit(x, y, z, ..., m, d, t, srcDStride=h*T, dstDStride=h*D)
    for idx in (blockIdx + blockNum - coreOffset) % blockNum .. tiles step blockNum:
        matmul_op.RunTileByIdx(idx)
    coreOffset = (coreOffset + tiles) % blockNum
```

([csrc/kernels/einsum_mht_htd_mhd.h:33-44](../../csrc/kernels/einsum_mht_htd_mhd.h#L33-L44))

- 每头切成 `ceil(m/m0) × ceil(d/n0)` 个输出 tile;头间用累计 `coreOffset` 错开起始 block,均衡多核负载;
- `srcDStride = h*T`、`dstDStride = h*D` 把 A/C 的行距改写为含跨头跳步的物理行距;
- `yStride = d * t * sizeof(Dtype)` 与 hdt 变体的 `yStride` 相同——`[h,t,d]` 与 `[h,d,t]` 单头字节数一致,仅头内解释不同。

### 单 tile 数据流(Matmul 类,transpose=1 分支)

每个 `(m0=128) × (n0=256)` 输出 tile 在 Cube 核内执行五级流水([csrc/kernels/matmul.h:200-378](../../csrc/kernels/matmul.h#L200-L378)):

1. **A GM→L1**:`CopyGmToL1Nd2Nz` 把 `[mActual, kRem]` 的 A 转成 NZ 分形进 `l1aBuf`(K 方向双缓冲,`kDtileSize = 2*k0` 预取);
2. **B GM→L1(本算子专属分支)**:`transpose=1 && nz==0` 时走 `CopyGmToL1Nd2Nz(bGmBuf[kOffset*n + nOffset], kRemSize, nActual, n, ...)`——B 逻辑上是 `[t, d]`(K×N),以 n 为行距、交换 NK 角色做 ND2NZ,直接产出转置后的分形([csrc/kernels/matmul.h:304-306](../../csrc/kernels/matmul.h#L304-L306));`nz==1` 时按 `kStride` 行距 `CopyGmToL1` 拷贝现成 NZ 分形([csrc/kernels/matmul.h:307-311](../../csrc/kernels/matmul.h#L307-L311));
3. **L1→L0A/L0B**:`CopyToL0ACol` / **`CopyToL0BTCol`**(转置专用,带 k0ActualBlockNum 参数)分形下搬,各自 ping-pong 双缓冲([csrc/kernels/matmul.h:339-345](../../csrc/kernels/matmul.h#L339-L345));
4. **Mmad**:K 方向按 `kQtileSize` 分段累加到 L0C(`CalMmad`,kIdx==0 初始化);
5. **L0C→GM**:`CopyToGmWithDequant` 写回 C tile(无 bias/dequant)。

同步用 `MTE1_MTE2`(L1 空闲)、`M_MTE1`(L0 就绪)、`M_FIX`(Mmad 完成)事件链,详见 [matmul.h](../../csrc/kernels/matmul.h) 的 `RunTileBody`。

### tiling/swizzle

tile 遍历顺序由 `GetMNBlockIdx` 按 host 传入的 swizzle(方向 Zn/Nz + 块计数)重排,改善 cache 命中率([csrc/kernels/matmul.h:380-419](../../csrc/kernels/matmul.h#L380-L419))。

### 测试参考

[tests/kernels/einsum_mht_mhd.py](../../tests/kernels/einsum_mht_mhd.py):fp16/bf16 × weight_nz {False, True} × (m,h,t,d) 多组形状,`htd` 直接 `contiguous()` 生成,与 `torch.einsum("mht,htd->mhd")` 对比;[tests/kernels/linear_att_proj.py](../../tests/kernels/linear_att_proj.py) 覆盖线性注意力 QKV/Z/B/A 投影的组合 matmul 路径(其内部经 `XliteOpMatmul` 复用同一 Matmul 模板)。
