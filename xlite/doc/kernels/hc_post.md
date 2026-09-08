# hc_post

## 功能概述

DeepSeek-V4 Hyper-Connection 的后融合算子(post-attn/FFN merge):注意力或 FFN 子模块输出后,用 `hc_act` 算出的 `post`/`comb` 门控把被 hc_pre 压缩的 H 条残差流重新展开。数学定义(`tests/kernels/hc_post.py:14-19`):

```
y[m, k, d] = post[m, k] * x[m, d] + Σ_h comb[m, h*K + k] * residual[m, h, d]
```

即 `term1` 为 post 门控的子模块输出广播,`term2` 是一个 K=H 的小批量 GEMM(comb [H,H] × residual [H,D])。这种小 K GEMM 在 Cube 矩阵核上效率很低,因此整个算子用纯 Vector 指令(`vmuls` + K 次 `vaxpy`)一次 AIV pass 完成,同时融合 bf16↔fp32 类型转换,fp32 精度计算。模型中以 in-place 方式调用(residual == y,`model.cpp:1618`,测试 `tests/kernels/hc_post.py:82` 同样)。

## 输入输出参数

Python 侧调用:`hc_post(rt, x_n, post_n, comb_n, residual_inplace, residual_inplace, n, HC_MULT, hidden)`(`tests/kernels/hc_post.py:82`),host 侧封装 `XliteOpHcPost`(`csrc/op.cpp:2218`)。

kernel 签名(`csrc/kernels/hc_post.h:146`):

```cpp
hc_post_<dtype>(GM_ADDR x, GM_ADDR post, GM_ADDR comb, GM_ADDR residual, GM_ADDR y,
                uint32_t m, uint32_t hcMult, uint32_t hidden)
```

记 `K = hcMult`(当前模型固定 4):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x | 输入 | `[m, D]` | BF16 | 子模块(attn/FFN)输出,`D = hidden` |
| post | 输入 | `[m, K]` | FP32 | post 门(`hc_act` 输出,取值约 [0,2]) |
| comb | 输入 | `[m, K*K]` | FP32 | Sinkhorn 双随机混合矩阵,`comb[h*K+k]`:h=源流,k=输出流 |
| residual | 输入 | `[m, K, D]` | BF16 | hc 域残差(K 条流);in-place 调用时与 y 同一 tensor |
| y | 输出 | `[m, K, D]` | BF16 | 展开后的 K 条残差流;in-place 时覆写 residual |
| m | 标量 | - | uint32 | token 数(host 校验 `m == x.shape[0]` 隐含,`x.numel == 0` 时直接返回,`csrc/op.cpp:2221`) |
| hcMult | 标量 | - | uint32 | 流数 K |
| hidden | 标量 | - | uint32 | 每流特征维 D(测试覆盖 256 ~ 4096) |

host 侧 dtype 校验:x/residual/y 必须为 BF16、post/comb 必须为 FP32,否则抛错(`csrc/op.cpp:2224-2232`)。测试 shape 约定(`tests/kernels/hc_post.py:51-82`):`[b, s, ...]` 展平为 `[n=b*s, ...]`,case 从 `(1,1,256)` decode 单 token 到 `(8,1024,4096)` 真实 prefill 规模。

## 支持的数据类型

- 模板 `csrc/kernels/hc_post.h`(模板参数 Dtype 为 x/residual/y 的类型)。
- 实例化文件 `csrc/kernels/hc_post_bfloat16_t.cpp`,导出 `hc_post_bfloat16_t`,即仅 **BF16**(门控 post/comb 固定 FP32)。
- kernel 整体用 `#ifdef __DAV_C220_VEC__` 保护,非 C220 向量核平台编译为空实现(`csrc/kernels/hc_post.h:153-160`)。

## 实现原理

单 AIV 向量核 kernel。并行按两级展开:`m` 个 token 以 `tok = block_idx; tok < m; tok += block_num` 网格切分(`csrc/kernels/hc_post.h:77`);token 内 `hidden` 维再按 `dTile` 分块流式处理。

### tiling 与 UB 布局

`dTile = min(hidden, 2048)`(`csrc/kernels/hc_post.h:28`):上限由 UB 容量决定——最大的 UB 占用是整块残差的 fp32 副本 `residualFp32[K * dTile]`(K=4、dTile=2048 时 32KB fp32),加上双缓冲的 `inXDtype/inResidDtype/outDtype` 与 `xFp32/outFp32`,选 2048 恰好装进 192KB UB(代码注释 `csrc/kernels/hc_post.h:26`)。UB 依次分配:标量表 `postU[K]`、`combU[K*K]`(token 级、跨 dTile 复用)、`residualFp32[K*dTile]`、`xFp32/outFp32[dTile]`,以及 x/残差/输出的双缓冲 IO 槽(`csrc/kernels/hc_post.h:39-69`),末尾 `assert(off <= UB_SIZE)` 防溢出。

### 每 token 流程

1. **载入标量表**:MTE2 把该 token 的 `post[K]`、`comb[K*K]` 拷入 UB,`EVENT_ID5` 通知标量核(`csrc/kernels/hc_post.h:83-86`)。in-place 调用时(测试与模型即如此),全部 K 条源流先读完才写任何输出,避免覆写尚未消费的输入(代码注释 `csrc/kernels/hc_post.h:13-14`;顺序由 dTile 循环内"先载入再写出"的流水约束保证)。
2. **D-tile 循环**(`d = 0; d < hidden; d += dTile`,`csrc/kernels/hc_post.h:89`):
   - MTE2 载入 `xPtr+d` 与 K 条 `residBase + h*hidden + d` 到 IO 槽 `curr`(`csrc/kernels/hc_post.h:97-103`),V 侧 `convert_input` 全部转 fp32(`csrc/kernels/hc_post.h:105-108`);
   - 对每个输出流 k(共 K 个):`vmuls(outFp32, xFp32, postK)` 完成 term1,再对 h=0..K-1 依次 `vaxpy(outFp32, residualFp32 + h*dTile, combHK)` 完成 term2(`csrc/kernels/hc_post.h:117-125`);
   - `convert_output` 转 bf16,MTE3 写出到 `y + tok*K*hidden + k*hidden + d`(`csrc/kernels/hc_post.h:127-134`)。

term2 的总向量操作量为 `K*K` 次 `vaxpy`(每 dTile),K=4 时 16 次,远小于一次 Cube GEMM 的启动开销,这正是用纯 Vector 实现的动机(`tests/kernels/hc_post.py:20-22` 的注释)。

### 流水线同步

三段流水用事件对管理:

- `EVENT_ID0 + curr`:MTE2 载入与 V 计算的交接,双向 wait/set(`csrc/kernels/hc_post.h:97-110`);
- `EVENT_ID2 + curr`:V 计算(fp32 结果 + bf16 转换)与 MTE3 写出的交接(`csrc/kernels/hc_post.h:127-134`);
- `EVENT_ID5`:标量表(post/comb)MTE2 → 标量核读取,每个 token 只在第一个 dTile(`d == 0`)等待一次(`csrc/kernels/hc_post.h:114-116`)。

IO 槽 `curr` 在每个 dTile 末尾翻转(`curr = 1 - curr`,`csrc/kernels/hc_post.h:136`),实现载入与写出的双缓冲重叠。kernel 末尾按对称顺序 wait 掉所有已 set 的事件(`csrc/kernels/hc_post.h:139-142`)。
