# linear_att_proj

## 功能概述

线性注意力(Linear Attention / GDN)的 QKV 投影 host 融合入口,把四段权重沿列拼接、一次转置 matmul、再切分四段输出,合并为单次下发:

```text
W = concat_col(W_qkv[n,k], W_z[v,k], W_b[h,k], W_a[h,k])   # [n+v+h+h, k]
out = matmul(x[m,k], W, transpose=true)                     # [m, n+v+h+h]
(mix_qkv, z, b, a) = split_col(out)                         # [m,n] / [m,v] / [m,h] / [m,h]
```

无独立 kernel,由 `concat_col` → `matmul`(转置)→ `split_col` 三个已有算子组合而成([csrc/_C.cpp](../../../csrc/_C.cpp) 的 `LinearAttProj`)。中间拼接权重 `W` 与 matmul 输出 `out` 由 Runtime 内存池临时分配(`rt.GetTensor`),计算末尾 `rt.Synchronize()` 等待完成并归还临时张量。

## 输入输出参数

Python 入口 `linear_att_proj(rt, x, W_qkv, W_z, W_b, W_a, mix_qkv, z, b, a, m, n, v, h, k)`([xlite/_C.pyi](../../../xlite/_C.pyi) 的 `linear_att_proj`):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x | 输入 | `[m, k]` | float16 / bfloat16 | 输入激活(batch*seqlen 展平) |
| W_qkv | 输入 | `[n, k]` | 同 x | QKV 权重 |
| W_z | 输入 | `[v, k]` | 同 x | Z(gating)权重 |
| W_b | 输入 | `[h, k]` | 同 x | B(beta)权重 |
| W_a | 输入 | `[h, k]` | 同 x | A(decay)权重 |
| mix_qkv | 输出 | `[m, n]` | 同 x | 混合 QKV 投影结果 |
| z | 输出 | `[m, v]` | 同 x | gating 参数 |
| b | 输出 | `[m, h]` | 同 x | beta 输入 |
| a | 输出 | `[m, h]` | 同 x | decay 输入 |
| m | 标量 | - | uint32 | 行数(batch*seqlen) |
| n | 标量 | - | uint32 | QKV 权重维 |
| v | 标量 | - | uint32 | Z 权重维 |
| h | 标量 | - | uint32 | B/A 权重维 |
| k | 标量 | - | uint32 | 隐藏维(收缩维) |

## 支持的数据类型

沿用底层 `matmul` 支持的 dtype(float16 / bfloat16,以及 int8/int4 量化变体视权重而定);四段权重与输入须同 dtype。dtype 分派由 `XliteOpMatmul` 完成([csrc/op.cpp](../../../csrc/op.cpp) 的 `XliteOpMatmul`)。

## 实现原理

纯 host 侧组合,无独立 kernel 计算:

1. **权重拼接**:从 Runtime 内存池取临时张量 `W = rt.GetTensor({k, n+v+h+h})`,调用 `XliteOpConcatCol` 把 `{W_qkv, W_z, W_b, W_a}` 沿列拼成 `[n+v+h+h, k]`([csrc/_C.cpp](../../../csrc/_C.cpp) 的 `LinearAttProj`,关键字 `XliteOpConcatCol`)。
2. **matmul**:取临时输出 `out = rt.GetTensor({m, n+v+h+h})`,调用 `XliteOpMatmul(rt, x, W, out, ..., transpose=true)`,权重以转置布局参与计算([csrc/_C.cpp](../../../csrc/_C.cpp) 的 `LinearAttProj`,关键字 `XliteOpMatmul`);随后 `rt.PutTensor(W)` 归还拼接权重。
3. **输出切分**:调用 `XliteOpSplitCol` 把 `out` 沿列切回 `{mix_qkv, z, b, a}`(各列宽 `n/v/h/h`,总和须等于 `n+v+h+h`),再 `rt.PutTensor(out)` 归还。
4. `rt.Synchronize()` 等待整条链路完成。

子算子实现细节见 [concat](concat.md)、[matmul](matmul.md)、[split](split.md)。

### 片上缓冲分配与占用汇总(子算子 AIV UB / AIC L1/L0A/L0B/L0C)

本算子无独立 kernel,片上缓冲按实际调用的三个子算子 kernel 计:`concat_col` 与 `split`(`csrc/kernels/concat.cpp`、`csrc/kernels/split.cpp`,AIV,UB 两半乒乓平分)、`matmul_<dtype>`(`csrc/kernels/matmul.h` 的 `Matmul::Init`,AIC,L1/L0A/L0B/L0C 累进 `bufferAddr`)。三次 launch 在同一 stream 上顺序执行,占用并列呈现、不叠加。硬件容量基准(910B/dav C220,CANN 平台配置):UB 192KB(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`)、L1 512KB、L0A 64KB、L0B 64KB,L0C 总容量 128KB（`l0_c_size=131072`，16 bank × 8KB），Mmad 累加缓冲即 `TPosition::CO1` 且占满整个 L0C。dtype 记法:`Dtype` 为 x/权重元素类型(FP16/BF16,均 2B)、`MatDtype` 为 mmad 累加器类型(float,4B)——两条 dtype 路径元素同为 2B,占用一致,单占用列。典型模型配置(取 kernel 测试 `tests/kernels/linear_att_proj.py`):m=4096、k=1024、n+v+h+h=8224(N=6144、V=2048、H=16+16),matmul 经 `PickMatmulTiling`(`csrc/auto_tuner.cpp` 的 `PickMatmulTiling`)选定 m0=128、n0=256、k0=64,无 bias/deqScale。

**concat_col / split(AIV UB,容量 192KB,每 block 一份;两 kernel 布局相同)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `dataBuf[0]`/`dataBuf[1]` | UB_SIZE/2 字节流水段 | uint8 | 2(事件 EVENT_ID0/1,MTE3↔MTE2) | `UB_SIZE/2` B | 196608 |
| **合计** | | | | | **196608** |

**matmul(AIC L1,容量 512KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l1aBuf` | (m0, k8TileSize=8*k0) = (128, 512) | Dtype | 2(事件 EVENT_ID0/1,MTE1↔MTE2) | `m0*k8TileSize*sizeof(Dtype)` B | 262144 |
| `l1bBuf` | (n0, k4TileSize=4*k0) = (256, 256) | Dtype | 2(事件 EVENT_ID2/3,MTE1↔MTE2) | `n0*k4TileSize*sizeof(Dtype)` B | 262144 |
| **合计** | | | | | **524288** |

**matmul(AIC L0A,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0aBuf` | (m0, k0) = (128, 64) | Dtype | 2(事件 EVENT_ID0/1,M↔MTE1) | `m0*k0*sizeof(Dtype)` B | 32768 |
| **合计** | | | | | **32768** |

**matmul(AIC L0B,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0bBuf` | (n0, k0) = (256, 64) | Dtype | 2(事件 EVENT_ID0/1,M↔MTE1) | `n0*k0*sizeof(Dtype)` B | 65536 |
| **合计** | | | | | **65536** |

**matmul(AIC L0C CO1 区,容量 128KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0cBuf` | (m0, n0) = (128, 256) | float(MatDtype) | 1(事件 M↔FIX EVENT_ID0) | `m0*n0*sizeof(MatDtype)` B | 131072 |
| **合计** | | | | | **131072** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| UB(concat_col/split,每 block) | 192 KB | 196608 B | 100% |
| L1(matmul) | 512 KB | 524288 B | 100% |
| L0A(matmul) | 64 KB | 32768 B | 50% |
| L0B(matmul) | 64 KB | 65536 B | 100% |
| L0C CO1(matmul) | 128 KB | 131072 B | 100% |

要点:

- 三个子 kernel 都以"吃满所在存储级"为目标设计:concat/split 的 `dataBuf` 按定义平分 UB(100%);matmul 的 tiling 在 m0=128、n0=256、k0=64 下 L1(2×128KB A 槽 + 2×128KB B 槽)恰好 100%、L0B 2×32KB 100%、L0C 128×256 fp32 100%——L1 与 L0B/CO1 是该配置的实际卡点,任一 tile 再增大即溢出;
- 小 m(decode 单 token 等)场景下 `PickMatmulTiling` 会按形状与核数动态缩小 tile(m0 降至 ROUND_UP(m,32)),各段占用显著低于上表;上表是大 m prefill 形态的占用;
- matmul 内无容量断言,约束隐含在 tiling 选择中(`csrc/op.cpp` 的 `XliteOpMatmul` → `PickMatmulTiling` 对 hasBias/hasDeqScale 时 n0 钳到 256);本融合入口无 bias/deqScale,L1 的 `l1BiasBuf`/`l1DeqScaleBuf`(C1 区,与 `l1bBuf` 别名共享)、L0 的 `l0BiasBuf`(C2 bias 表)与 `fixpipeBuf`(C2PIPE2GM)均不分配;
- FP16 与 BF16 元素同为 2B,占用一致;占用差异只来自 tiling 随 m/n/k 的选择,而非 dtype。
