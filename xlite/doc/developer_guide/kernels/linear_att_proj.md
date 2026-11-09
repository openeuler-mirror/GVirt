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
