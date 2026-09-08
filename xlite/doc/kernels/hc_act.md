# hc_act

## 功能概述

DeepSeek-V4 Hyper-Connection 的门控激活融合算子(per-token activation),一次 kernel 同时完成三件事:

1. 由 `mixes` 计算 Hyper-Connection 三组门控系数:`pre`(残差合并门,前向 pre-merge 用)、`post`(注意力/FFN 之后的重展开系数)、`comb`(流间混合矩阵,经 softmax + Sinkhorn 归一化为近似双随机矩阵);
2. 只把 `post`、`comb` 写回 GM(`pre` 不落盘,直接在 kernel 内消费);
3. 融合 Step-5 pre-merge:`y[m, hidden] = Σ_h pre[h] * xResid[m, h, hidden]`,输出 bf16。

数学定义(与 `tests/kernels/hc_act.py` 中参考实现一致):

```
mixHc  = (2 + hcMult) * hcMult          # hcMult=4 时为 24
pre    = sigmoid(mixes[:, :K]           * scale[0] + base[:K])         + eps   # [K]
post   = 2 * sigmoid(mixes[:, K:2K]     * scale[1] + base[K:2K])               # [K]
comb   = sinkhorn(softmax(mixes[:, 2K:] * scale[2] + base[2K:]) + eps)         # [K*K]
y      = Σ_h pre[h] * xResid[:, h, :]                                            # [hidden]
```

另有 head 模式(`headOnly`,对应 `hc_head`,模型最后的 Hyper-Connection 头):`mixes` 只有 `[m, K]`、`hcScale` 只有 1 个标量、`hcBase` 只有 `[K]`,仅计算 `pre` 并做 merge,不算 post/comb/Sinkhorn;host 侧由 `hcBase.numel() == hcMult` 自动识别(`csrc/op.cpp:2191-2193`)。上游调用链为 `rmsnorm → matmul → hc_act`(见 `tests/kernels/hc_pre.py`、`tests/kernels/hc_head.py`)。

## 输入输出参数

Python 侧调用:`hc_act(rt, mixes, hc_scale, hc_base, post, comb, HC_MULT, eps, sinkhorn_iters, x_resid=..., output=...)`(`tests/kernels/hc_act.py:123`),host 侧封装 `XliteOpHcAct`(`csrc/op.cpp:2183`)。

kernel 签名(`csrc/kernels/hc_act.h:337`):

```cpp
hc_act_float(GM_ADDR mixes, GM_ADDR hcBase, GM_ADDR post, GM_ADDR comb, GM_ADDR hcScale,
             uint32_t m, uint32_t hcMult, float eps, uint32_t sinkhornIters,
             uint32_t headOnly, GM_ADDR xResid, GM_ADDR yOut, uint32_t hidden)
```

记 `K = hcMult`(当前模型固定 4,要求 1 ≤ K ≤ 7,`csrc/kernels/hc_act.h:98`)、`mixHc = (2+K)*K`(head 模式为 K):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| mixes | 输入 | `[m, mixHc]` | FP32 | 门控混合系数(上游 matmul 输出),按 `[pre(K) | post(K) | comb(K*K)]` 分段 |
| hcScale | 输入 | `[3]`(head 模式 `[1]`) | FP32 | 每段缩放标量 `{scalePre, scalePost, scaleComb}` |
| hcBase | 输入 | `[mixHc]`(head 模式 `[K]`) | FP32 | 每段偏置;head 模式由 `numel == hcMult` 自动识别 |
| post | 输出 | `[m, K]` | FP32 | post 门(head 模式可为空 tensor,kernel 不写) |
| comb | 输出 | `[m, K*K]` | FP32 | Sinkhorn 归一化后的混合矩阵,行优先 `comb[h*K+k]`(head 模式不写) |
| xResid | 输入 | `[m, K, hidden]` | BF16 | 未归一化残差(merge 输入),host 侧强制 BF16(`csrc/op.cpp:2206`) |
| yOut | 输出 | `[m, hidden]` | BF16 | pre-merge 结果 `Σ_h pre[h]*xResid[:,h,:]` |
| m | 标量 | - | uint32 | token 数(`mixes.shape[0]`) |
| hcMult | 标量 | - | uint32 | Hyper-Connection 流数 K ∈ [1,7] |
| eps | 标量 | - | float | 数值稳定项(pre 加 eps、softmax 加 eps、Sinkhorn 分母加 eps) |
| sinkhornIters | 标量 | - | uint32 | Sinkhorn 迭代轮数(测试与模型均用 20) |
| headOnly | 标量 | - | uint32 | 1 = head 模式(仅 pre + merge) |
| hidden | 标量 | - | uint32 | 每流特征维 D(`yOut.shape[1]`);测试覆盖 256 与真实规模 4096 |

测试 shape 约定(`tests/kernels/hc_act.py`):`mixes` 由 `[b, s, MIX_HC]` 展平为 `[n=b*s, 24]`,`x_resid` 为 `[n, 4, hidden]` bf16;head 模式 `mixes` 为 `[n, 4]`、`hc_scale` 为 `[1]`、`hc_base` 为 `[4]`。

## 支持的数据类型

- 模板 `csrc/kernels/hc_act.h`(模板参数 Dtype 只影响 merge I/O 的 `xResid`/`yOut`)。
- 实例化文件 `csrc/kernels/hc_act_float.cpp`,导出 `hc_act_float`,内部固定以 `Dtype = bfloat16_t` 调用模板(`csrc/kernels/hc_act.h:342`),即 **门控计算 FP32、merge I/O BF16** 的单一组合;host 侧强制校验 mixes/hcScale/hcBase/post/comb 为 FP32、xResid/output 为 BF16(`csrc/op.cpp:2196-2208`)。
- kernel 整体用 `#ifdef __DAV_C220_VEC__` 保护,非 C220 向量核平台编译为空实现(`csrc/kernels/hc_act.h:347-355`)。

## 实现原理

单 AIV 向量核 kernel,`m` 个 token 按 `process = block_idx; process < m; process += block_num` 网格切分(`csrc/kernels/hc_act.h:211`),每 token 全流程在 UB 内完成,双缓冲(`curr = 0/1`)流水重叠 MTE2 载入与 V 计算。

### UB 布局与向量位掩码

所有段(pre/post/comb)拼在一条 `mixHc` 长度的向量里处理,用 64 位向量掩码区分段(`csrc/kernels/hc_act.h:104-108`):`maskPre` = 低 K 位,`maskPost` = 接着 K 位,`maskComb` = 其后 K*K 位。这也是限制 K ≤ 7 的原因(`1ULL << (K*K)` 在 K ≥ 8 时溢出,`csrc/kernels/hc_act.h:96-98`)。UB 中依次分配:mixes 双缓冲、calc 计算区、postOut/combOut 双缓冲、combAlign(comb 的对齐副本,K 行每行 `hcMultAlign = ROUND_UP(K, 8)` 个元素,`csrc/kernels/hc_act.h:134-140`)、baseUb、colBuf/scalarBuf/brcbBuf(Sinkhorn 归一化 scratch)、onesUb(常数 1.0,用 `vdiv` 实现 1/x)、offRamp(`vgather` 的字节偏移斜坡)、以及 merge 用的 `inDtype/outDtype` 双缓冲与 `xFp32`/`yCalc`(`csrc/kernels/hc_act.h:113-180`)。

### 门控计算步骤

每 token(`csrc/kernels/hc_act.h:216-273`):

1. MTE2 载入 `mixesUb[curr]`(mixHc 个 fp32)与 `inDtypeArr[curr]`(K×hidden bf16 残差),同时 V 侧处理上一 token;
2. `vmuls` 按掩码分别乘 `scalePre/scalePost/scaleComb`(`csrc/kernels/hc_act.h:225-230`);
3. bf16 残差整块转 fp32(`convert_input`,按 `VECTOR_MAX_REPEAT`(255)次 repeat 分块,`csrc/kernels/hc_act.h:234-238`);
4. 掩码 `maskAll` 下 `vadd baseUb` 一次性给三段加偏置(`csrc/kernels/hc_act.h:244-245`);
5. 掩码 `maskPrePost` 下算 sigmoid 链:`×(-1) → vexp → +1 → vdiv(1/x)`,即 `sigmoid(x) = 1/(1+e^{-x})`(`csrc/kernels/hc_act.h:247-255`);
6. `maskPre` 下 `+eps` 得 pre;`maskPost` 下 `×2` 得 post(`csrc/kernels/hc_act.h:257-263`);
7. comb 段(`calcUb` 中偏移 `2K` 起,按行排布)用 `vgather` + offRamp 斜坡逐行搬运到 `combAlign` 对齐布局(`csrc/kernels/hc_act.h:264-270`);post 段同理 gather 出 `postOut[curr]`(`csrc/kernels/hc_act.h:285-288`)。

### pre-merge 融合

标量核从 `calcUb[h]` 逐个读出 `preH`,对 fp32 残差做 `vaxpy(yCalc, xFp32 + h*hidden, preH, vecRep)` 累加 K 次,得 `y = Σ_h pre[h]*x[:, h, :]`(`csrc/kernels/hc_act.h:276-280`);结果 `convert_output` 转 bf16 后 MTE3 写 `yRow`(`csrc/kernels/hc_act.h:282-293`)。`postOut[curr]` 同步写回 GM(`csrc/kernels/hc_act.h:294-297`)。`pre` 因此完全不经过 GM,省一次读写往返。

### Sinkhorn 归一化(comb)

head 模式跳过整段。普通模式下(`csrc/kernels/hc_act.h:301-322`),在 `combAlign` 上依次调用三个 UB 辅助函数:

- `hc_softmax_rows`(`csrc/kernels/hc_act.h:54`):逐行 softmax——`vcmax` 取行最大、`vbrcb` 广播、减 max、`vexp`、`vcadd` 行求和、广播、`vdiv`、最后 `+eps`;
- `hc_col_normalize`(`csrc/kernels/hc_act.h:16`):列归一——跨行同列 `vadd` 累加出 `colSum`,`+eps` 作分母,`vdiv` 逐列除;先执行一次;
- `hc_row_normalize`(`csrc/kernels/hc_act.h:36`):行归一——`vcadd` 行求和、`+eps`、`vbrcb` 广播、`vdiv` 逐行除。

迭代结构为:softmax → 1 次列归一 → `(行归一, 列归一) × (sinkhornIters-1)`(`csrc/kernels/hc_act.h:303-310`),与参考实现 `tests/kernels/hc_act.py:hc_split_sinkhorn` 逐步一致,使 comb 收敛到近似双随机矩阵。结果从 `combAlign` 经 `combOut[curr]` 双缓冲逐行写回 GM 的 `comb[process*K*K]`(`csrc/kernels/hc_act.h:312-321`)。

### 流水线同步

MTE2(载入)/ V(计算)/ MTE3(写出)之间用事件对流水:`EVENT_ID0+curr` 管理 mixes+残差载入与 V 计算的交接(`csrc/kernels/hc_act.h:216-222`),`EVENT_ID6+curr` 管理 y 的 bf16 转换与写出(`csrc/kernels/hc_act.h:282-298`),`EVENT_ID2+curr` 管理 comb 写出(`csrc/kernels/hc_act.h:312-321`)。head 模式不使用 EVENT_ID2/3,收尾 wait 也相应裁剪(`csrc/kernels/hc_act.h:325-333`),避免等待永远不会被置位的事件。每轮 token 处理完 `curr = 1 - curr` 翻转缓冲。
