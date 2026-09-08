# einsum_mht_hdt_mhd

## 功能概述

批量头独立的矩阵乘:`mhd = torch.einsum("mht,hdt->mhd", mht, hdt)`,对每个头 h 做一次 `[m,t] × [d,t]ᵀ = [m,d]` 的 GEMM(右操作数按 `[h, d, t]` 非转置布局直接给出,matmul 侧走 transpose=0 路径)。是 `einsum_mht_htd_mhd`(右操作数 `[h,t,d]` 转置布局)的姊妹算子,主要用于 MLA 类注意力的逐头吸收投影;当前模型代码中主要使用 htd 变体,本算子通过 Python 绑定暴露供测试与自定义流程使用。

## 输入输出参数

Python 接口:`einsum_mht_hdt_mhd(rt, mht, hdt, mhd, m, h, t, d, weight_nz=False)`(`XliteOpEinsumMhtHdtMhd`,[csrc/op.cpp:2121-2143](../../csrc/op.cpp#L2121-L2143))。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| mht | 输入 | `[m, h, t]`(逻辑上 `[m, h*T]` 连续存储,T 为对齐后的行宽) | float16 / bfloat16 | 左操作数,按头切出 `[m, t]` 子矩阵作为 GEMM 的 A |
| hdt | 输入 | `[h, d, t]` | 同上 | 右操作数(权重),非转置布局;`weight_nz=True` 时为 ND2NZ 后的 NZ 分形布局 |
| mhd | 输出 | `[m, h, d]`(逻辑上 `[m, h*D]`) | 同上 | 逐头 GEMM 结果 C = A·Bᵀ |
| m | 标量 | - | uint32 | token 数(GEMM 的 M 维),测试覆盖 3 ~ 5230 |
| h | 标量 | - | uint32 | 头数(GEMM 批次数) |
| t | 标量 | - | uint32 | 收缩维(GEMM 的 K 维),如 MLA 的 nopeHeadDim/kvLoraRank |
| d | 标量 | - | uint32 | 输出维(GEMM 的 N 维) |
| weight_nz | 标量 | - | bool | 右操作数是否为 NZ 分形布局 |
| T / D | 标量(可选) | - | int | mht 行宽 / mhd 行宽的对齐覆盖;缺省(`-1`)时取 t / d。MLA 流程中传入 `nopeHeadDim + ropeHeadDim` 以跳过 head 中不参与计算的 rope 区(见 [csrc/model.cpp:469-471](../../csrc/model.cpp#L469-L471) 对 htd 变体的用法) |

host 侧 swizzle 策略由 `XlitePickSwizzle(d, t, ...)` 依据 N/K 形状挑选([csrc/op.cpp:2136-2139](../../csrc/op.cpp#L2136-L2139));`m0/n0/k0` 取 `MATMUL_M0_N0_K0_DEFAULT_VALUE`(即 128/256/按 dtype)。

## 支持的数据类型

- `float16_t`([einsum_mht_hdt_mhd_float16_t.cpp](../../csrc/kernels/einsum_mht_hdt_mhd_float16_t.cpp))
- `bfloat16_t`([einsum_mht_hdt_mhd_bfloat16_t.cpp](../../csrc/kernels/einsum_mht_hdt_mhd_bfloat16_t.cpp))

实例化形态为 `Matmul<dtype, float, dtype>`(bias 为 float 类型形参但未启用,[csrc/kernels/einsum_mht_hdt_mhd.h:54](../../csrc/kernels/einsum_mht_hdt_mhd.h#L54));三个张量 dtype 须一致。

## 实现原理

实现位于 [csrc/kernels/einsum_mht_hdt_mhd.h](../../csrc/kernels/einsum_mht_hdt_mhd.h),是通用 GEMM 模板 `Matmul`([csrc/kernels/matmul.h](../../csrc/kernels/matmul.h))外的一层逐头批处理封装。kernel 声明为 `KERNEL_TYPE_AIC_ONLY`,由 host 以 `rt.aicNum` 个 Cube 核启动(与向量核算子不同)。

### 逐头批处理与跨头负载均衡

```
for hIdx in 0..h:
    x = mht + hIdx * T * sizeof(Dtype)        // A: 头内 [m, t],行距 T
    y = hdt + hIdx * d * t * sizeof(Dtype)    // B: 头内 [d, t]
    z = mhd + hIdx * D * sizeof(Dtype)        // C: 头内 [m, d],行距 D
    tiles = matmul_op.TaskTilesInit(x, y, z, ..., m, d, t, srcDStride=h*T, dstDStride=h*D)
    for idx in (blockIdx + blockNum - coreOffset) % blockNum .. tiles step blockNum:
        matmul_op.RunTileByIdx(idx)
    coreOffset = (coreOffset + tiles) % blockNum
```

([csrc/kernels/einsum_mht_hdt_mhd.h:33-44](../../csrc/kernels/einsum_mht_hdt_mhd.h#L33-L44))

- 每头的 GEMM 被切成 `ceil(m/m0) × ceil(d/n0)` 个输出 tile,`TaskTilesInit` 返回 tile 总数;
- 头与头之间用 `coreOffset` 累计错开起始 block:每个头处理完后把该头 tile 数累加到偏移,使下一头的第一个 tile 落到"轮到"的核上,避免按头整块切分造成的小头尾负载不均;
- `srcDStride = h*T`、`dstDStride = h*D` 把 A 的行距与 C 的行距从逻辑 k/n 改写为含跨头跳步的物理行距,使 matmul 搬运时直接跳到正确的头内子矩阵。

### 单 tile 数据流(Matmul 类)

每个 `(m0=128) × (n0=256)` 输出 tile 在 Cube 核内执行标准五级流水([csrc/kernels/matmul.h:200-378](../../csrc/kernels/matmul.h#L200-L378)):

1. **A GM→L1**:`CopyGmToL1Nd2Nz` 把 `[mActual, kRem]` 的 A 转成 NZ 分形进 `l1aBuf`(K 方向双缓冲,K 预取 `kDtileSize = 2*k0`);
2. **B GM→L1**:本算子 `transpose=0` 且 `nz=0` 时走 `CopyGmToL1Nd2Nz(bGmBuf[nOffset*k + kOffset], nActual, kRemSize, k, ...)`——注意 B 逻辑上是 `[d, t]`(N×K),以 k 为行距 ND2NZ 装载,等效完成 Bᵀ 的分形化([csrc/kernels/matmul.h:297-299](../../csrc/kernels/matmul.h#L297-L299));`nz=1` 时直接 `CopyGmToL1` 拷贝现成 NZ 分形;
3. **L1→L0A/L0B**:`CopyToL0ACol` / `CopyToL0BCol` 分形下搬,均带 ping-pong 双缓冲;
4. **Mmad**:K 方向按 `kQtileSize` 分段累加到 L0C(`CalMmad`,kIdx==0 时初始化累加器);
5. **L0C→GM**:`CopyToGmWithDequant` 写回 C tile(本算子无 bias/dequant)。

同步用 `MTE1_MTE2`(L1 空闲)、`M_MTE1`(L0 就绪)、`M_FIX`(Mmad 完成)事件链,L1A/L1B 各自独立 ping-pong,详见 [matmul.h](../../csrc/kernels/matmul.h) 的 `RunTileBody`。

### tiling/swizzle

tile 遍历顺序由 `GetMNBlockIdx` 按 host 传入的 swizzle(方向 Zn/Nz + 块计数)重排,改善 L2/L1 命中率([csrc/kernels/matmul.h:380-419](../../csrc/kernels/matmul.h#L380-L419))。

### 测试参考

[tests/kernels/einsum_mht_mhd.py](../../tests/kernels/einsum_mht_mhd.py):fp16/bf16 × weight_nz {False, True} × (m,h,t,d) ∈ {(3,4,192,512), (5230,4,512,128), (963,8,512,128), ...},`hdt` 用 `htd.transpose(-1,-2).contiguous()` 生成,与 `torch.einsum("mht,htd->mhd")` 对比。
