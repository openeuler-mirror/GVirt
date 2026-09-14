# hc_act

## 功能概述

DeepSeek-V4 Hyper-Connection 的门控激活算子(per-token activation),一次 kernel 完成 Hyper-Connection 三组门控系数的计算与写出,并按 `preSum` 选择「融合 pre sum」或「拆分写 pre」两条路径:

1. 由 `mixes` 计算 `pre`(残差合并门)、`post`(注意力/FFN 之后的重展开系数)、`comb`(流间混合矩阵,经 softmax + Sinkhorn 归一化为近似双随机矩阵);
2. **`preSum=1`(融合,对应 `hc_act` op)**:kernel 内直接消费 `pre` 做 pre sum `y[m, hidden] = Σ_h pre[h] * xResid[m, h, hidden]`,只把 `post`、`comb` 写回 GM(`pre` 不落盘),输出 bf16;
3. **`preSum=0`(拆分,对应 `hc_split_sinkhorn` op)**:把 `pre` 也写回 GM(pre sum 由独立算子 [`hc_pre`](hc_pre.md) 完成),此时 `xResid`/`yOut`/`hidden` 不用(null/0),恒为非 head 模式。

数学定义(与 `tests/kernels/hc_act.py`、`tests/kernels/hc_split_sinkhorn.py` 参考实现一致):

```
mixHc  = (2 + hcMult) * hcMult          # hcMult=4 时为 24
pre    = sigmoid(mixes[:, :K]           * scale[0] + base[:K])         + eps   # [K]
post   = 2 * sigmoid(mixes[:, K:2K]     * scale[1] + base[K:2K])               # [K]
comb   = sinkhorn(softmax(mixes[:, 2K:] * scale[2] + base[2K:]) + eps)         # [K*K]
y      = Σ_h pre[h] * xResid[:, h, :]            # 仅 preSum=1            # [hidden]
```

另有 head 模式(`headOnly`,对应 `hc_head`,模型最后的 Hyper-Connection 头,**仅 `preSum=1` 时出现**):`mixes` 只有 `[m, K]`、`hcScale` 只有 1 个标量、`hcBase` 只有 `[K]`,仅计算 `pre` 并做 pre sum,不算 post/comb/Sinkhorn;host 侧由 `hcBase.numel() == hcMult` 自动识别(`csrc/op.cpp:2231-2233`)。上游调用链为 `rmsnorm → matmul → hc_act`(见 `tests/kernels/hc_pre.py`、`tests/kernels/hc_head.py`)。

## 输入输出参数

Python 侧:
- 融合:`hc_act(rt, mixes, hc_scale, hc_base, post, comb, HC_MULT, eps, sinkhorn_iters, x_resid=..., output=...)`,host 侧封装 `XliteOpHcAct`(`csrc/op.cpp:2223`),以 `preSum=1`、`pre=nullptr` 启动(`csrc/op.cpp:2253-2255`);
- 拆分:`hc_split_sinkhorn(rt, mixes, hc_scale, hc_base, pre, post, comb, HC_MULT, eps, sinkhorn_iters)`,host 侧封装 `XliteOpHcSplitSinkhorn`(`csrc/op.cpp:2275`),以 `preSum=0`、`headOnly=0`、`xResid/yOut=nullptr`、`hidden=0` 启动(`csrc/op.cpp:2295-2298`)。

kernel 签名(模板 `csrc/kernels/hc_act.h:84-89`,入口宏 `hc_act_float` `csrc/kernels/hc_act.h:379-389`):

```cpp
hc_act_float(GM_ADDR mixes, GM_ADDR hcBase, GM_ADDR post, GM_ADDR comb, GM_ADDR hcScale,
             uint32_t m, uint32_t hcMult, float eps, uint32_t sinkhornIters,
             uint32_t headOnly, uint32_t preSum, GM_ADDR pre, GM_ADDR xResid, GM_ADDR yOut,
             uint32_t hidden)
```

记 `K = hcMult`(要求 1 ≤ K ≤ 7,`csrc/kernels/hc_act.h:100`)、`mixHc = (2+K)*K`(head 模式为 K):

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| mixes | 输入 | `[m, mixHc]` | FP32 | 门控混合系数(上游 matmul 输出),按 `[pre(K) \| post(K) \| comb(K*K)]` 分段 |
| hcScale | 输入 | `[3]`(head 模式 `[1]`) | FP32 | 每段缩放标量 `{scalePre, scalePost, scaleComb}` |
| hcBase | 输入 | `[mixHc]`(head 模式 `[K]`) | FP32 | 每段偏置;head 模式由 `numel == hcMult` 自动识别 |
| post | 输出 | `[m, K]` | FP32 | post 门(head 模式可为空 tensor,kernel 不写) |
| comb | 输出 | `[m, K*K]` | FP32 | Sinkhorn 归一化后的混合矩阵,行优先 `comb[h*K+k]`(head 模式不写) |
| preSum | 标量 | - | uint32 | 1 = 融合 pre sum(写 y,pre 不落盘);0 = 拆分(写 pre 到 GM,pre sum 由 hc_pre 完成) |
| pre | 输出 | `[m, K]` | FP32 | pre 门;仅 `preSum=0` 写回 GM,`preSum=1` 传 nullptr |
| xResid | 输入 | `[m, K, hidden]` | BF16 | 未归一化残差(pre-sum 输入);仅 `preSum=1` 使用,host 强制 BF16(`csrc/op.cpp:2246`) |
| yOut | 输出 | `[m, hidden]` | BF16 | pre-sum 结果 `Σ_h pre[h]*xResid[:,h,:]`;仅 `preSum=1` 写出 |
| m | 标量 | - | uint32 | token 数(`mixes.shape[0]`) |
| hcMult | 标量 | - | uint32 | Hyper-Connection 流数 K ∈ [1,7] |
| eps | 标量 | - | float | 数值稳定项(pre 加 eps、softmax 加 eps、Sinkhorn 分母加 eps) |
| sinkhornIters | 标量 | - | uint32 | Sinkhorn 迭代轮数(测试与模型均用 20) |
| headOnly | 标量 | - | uint32 | 1 = head 模式(仅 pre + pre sum;仅 `preSum=1`) |
| hidden | 标量 | - | uint32 | 每流特征维 D(`yOut.shape[1]`);`preSum=0` 时传 0 |

测试 shape 约定:`tests/kernels/hc_act.py` 中 `mixes` 由 `[b, s, MIX_HC]` 展平为 `[n=b*s, 24]`,`x_resid` 为 `[n, 4, hidden]` bf16;head 模式 `mixes` 为 `[n, 4]`、`hc_scale` 为 `[1]`、`hc_base` 为 `[4]`。`tests/kernels/hc_split_sinkhorn.py` 覆盖 `preSum=0` 路径(`CASES = [(2,8,20,1e-6), (1,1,20,1e-6), (8,4096,20,1e-6)]`)。

## 支持的数据类型

- 模板 `csrc/kernels/hc_act.h`(模板参数 Dtype 只影响 `preSum=1` 时 pre-sum I/O 的 `xResid`/`yOut`)。
- 实例化文件 `csrc/kernels/hc_act_float.cpp`,导出 `hc_act_float`,内部固定以 `Dtype = bfloat16_t` 调用模板(`csrc/kernels/hc_act.h:385-388`),即 **门控计算 FP32、pre-sum I/O BF16** 的单一组合;host 侧强制校验 mixes/hcScale/hcBase/post/comb/pre 为 FP32、xResid/output 为 BF16(`csrc/op.cpp:2236-2248`、`csrc/op.cpp:2285-2288`)。
- kernel 整体用 `#ifdef __DAV_C220_VEC__` 保护,非 C220 向量核平台编译为空实现(`csrc/kernels/hc_act.h:390-398`)。

## 实现原理

单 AIV 向量核 kernel,`m` 个 token 按 `process = block_idx; process < m; process += block_num` 网格切分(`csrc/kernels/hc_act.h:230`),每 token 全流程在 UB 内完成,双缓冲(`curr = 0/1`)流水重叠 MTE2 载入与 V 计算。

### UB 布局与向量位掩码

所有段(pre/post/comb)拼在一条 `mixHc` 长度的向量里处理,用 64 位向量掩码区分段(`csrc/kernels/hc_act.h:106-110`):`maskPre` = 低 K 位,`maskPost` = 接着 K 位,`maskComb` = 其后 K*K 位(head 模式置 0)。这也是限制 K ≤ 7 的原因(`1ULL << (K*K)` 在 K ≥ 8 时溢出,`csrc/kernels/hc_act.h:98-100`)。

UB 共享前缀(`csrc/kernels/hc_act.h:115-160`):mixes 双缓冲、calc 计算区、postOut/combOut 双缓冲、combAlign(comb 的对齐副本,K 行每行 `hcMultAlign = ROUND_UP(K, 8)` 个元素,`csrc/kernels/hc_act.h:136-142`)、baseUb、colBuf/scalarBuf/brcbBuf(Sinkhorn 归一化 scratch)、onesUb(常数 1.0,用 `vdiv` 实现 1/x)、scaleUb、offRamp(`vgather` 的字节偏移斜坡)。

尾部按 `preSum` 分配(`csrc/kernels/hc_act.h:174-195`):
- `preSum=1`:`inDtype0/1`(残差 bf16 双缓冲)、`outDtype0/1`(y bf16 双缓冲)、`xFp32`([K, hidden] 残差 fp32)、`yCalc`(pre-sum 累加器);
- `preSum=0`:`preOut0/1`(pre 双缓冲)、`yCalc`(pre 的 vgather 暂存,ub2ub 中转到 preOut)。两路径不重叠。

### 门控计算步骤

每 token(`csrc/kernels/hc_act.h:230-295`):

1. MTE2 载入 `mixesUb[curr]`(mixHc 个 fp32);`preSum=1` 时同时载入 `inDtypeArr[curr]`(K×hidden bf16 残差,`csrc/kernels/hc_act.h:236-241`);
2. `vmuls` 按掩码分别乘 `scalePre/scalePost/scaleComb`(`csrc/kernels/hc_act.h:245-252`);
3. `preSum=1` 时 bf16 残差整块转 fp32(`convert_input`,按 `VECTOR_MAX_REPEAT`(255)次 repeat 分块,`csrc/kernels/hc_act.h:255-261`),并 `vector_dup(yCalc, 0)` 清零累加器;
4. **`pipe_barrier(PIPE_V)`(`csrc/kernels/hc_act.h:263`)在 `if(preSum)` 之外**:保证 `vmuls(calcUb <- mixesUb[curr])` 排空后,MTE2 才能在下一轮同 `curr` 迭代重写 `mixesUb`。这条屏障对 `preSum=0` 的正确性至关重要(此前它误置于 `if(preSum)` 块内,导致拆分路径在多核、多 token 下 `mixesUb` 被提前覆盖,post/comb 出现非确定性误差);随后 `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0+curr)`(`csrc/kernels/hc_act.h:264`);
5. 掩码 `maskAll` 下 `vadd baseUb` 一次性给三段加偏置(`csrc/kernels/hc_act.h:267-268`);
6. 掩码 `maskPrePost` 下算 sigmoid 链:`×(-1) → vexp → +1 → vdiv(1/x)`,即 `sigmoid(x) = 1/(1+e^{-x})`(`csrc/kernels/hc_act.h:271-278`);
7. `maskPre` 下 `+eps` 得 pre;`maskPost` 下 `×2` 得 post(`csrc/kernels/hc_act.h:282-286`);
8. comb 段(`calcUb` 中偏移 `2K` 起,按行排布)用 `vgather` + offRamp 斜坡逐行搬运到 `combAlign` 对齐布局(`csrc/kernels/hc_act.h:289-293`);post 段同理 gather 出 `postOut[curr]`(`csrc/kernels/hc_act.h:323-327`)。

### pre sum / pre 写回(按 preSum 分支)

`preSum=1`(`csrc/kernels/hc_act.h:298-306`):标量核从 `calcUb[h]` 逐个读出 `preH`(经 `set_flag/wait_flag(PIPE_V, PIPE_S, EVENT_ID4)` 让 V→S 可见),对 fp32 残差做 `vaxpy(yCalc, xFp32 + h*hidden, preH, vecRep)` 累加 K 次,得 `y = Σ_h pre[h]*x[:, h, :]`;结果 `convert_output` 转 bf16 后 MTE3 写 `yRow`(`csrc/kernels/hc_act.h:316-333`)。

`preSum=0`(`csrc/kernels/hc_act.h:307-313`):用 `vgather` 把 `calcUb` 中的 pre 段 gather 到 `yCalc`,经 `copy_ubuf_to_ubuf` 中转到 `preOut[curr]`,MTE3 写回 GM 的 `pre[process*K]`(`csrc/kernels/hc_act.h:319-335`);pre sum 不在本算子内,由 [`hc_pre`](hc_pre.md) 消费这条 `pre`。`postOut[curr]` 同步写回 GM(`csrc/kernels/hc_act.h:337-340`)。因此融合路径下 `pre` 完全不经过 GM,省一次读写往返;拆分路径下 `pre` 显式落盘以解耦门控与 pre sum。

### Sinkhorn 归一化(comb)

head 模式跳过整段。普通模式下(`csrc/kernels/hc_act.h:344-365`),在 `combAlign` 上依次调用三个 UB 辅助函数:

- `hc_softmax_rows`(`csrc/kernels/hc_act.h:55`):逐行 softmax——`vcmax` 取行最大、`vbrcb` 广播、减 max、`vexp`、`vcadd` 行求和、广播、`vdiv`、最后 `+eps`;
- `hc_col_normalize`(`csrc/kernels/hc_act.h:17`):列归一——跨行同列 `vadd` 累加出 `colSum`,`+eps` 作分母,`vdiv` 逐列除;先执行一次;
- `hc_row_normalize`(`csrc/kernels/hc_act.h:37`):行归一——`vcadd` 行求和、`+eps`、`vbrcb` 广播、`vdiv` 逐行除。

迭代结构为:softmax → 1 次列归一 → `(行归一, 列归一) × (sinkhornIters-1)`(`csrc/kernels/hc_act.h:346-353`),与参考实现 `tests/kernels/hc_split_sinkhorn.py:hc_split_sinkhorn_ref` 逐步一致,使 comb 收敛到近似双随机矩阵。结果从 `combAlign` 经 `combOut[curr]` 双缓冲逐行写回 GM 的 `comb[process*K*K]`(`csrc/kernels/hc_act.h:355-364`)。

### 流水线同步

MTE2(载入)/ V(计算)/ MTE3(写出)之间用事件对流水:`EVENT_ID0+curr` 管理 mixes(+残差)载入与 V 计算的交接(`csrc/kernels/hc_act.h:235-243`),`EVENT_ID6+curr` 管理 y/pre 的转换与写出(`csrc/kernels/hc_act.h:316-341`),`EVENT_ID2/3+curr` 管理 comb 写出(`csrc/kernels/hc_act.h:355-364`)。启动时按需预置 `set_flag`(comb 仅非 head 模式,`csrc/kernels/hc_act.h:215-223`);`preSum=1` 的 pre sum 用 `EVENT_ID4` 做 V→S 事件往返,让标量核安全读取 `calcUb[h]`(`csrc/kernels/hc_act.h:300-301`)。head 模式不使用 EVENT_ID2/3,收尾 wait 也相应裁剪(`csrc/kernels/hc_act.h:368-376`),避免等待永远不会被置位的事件。每轮 token 处理完 `curr = 1 - curr` 翻转缓冲。

## 关键代码位置

- 模板入口:`csrc/kernels/hc_act.h:84`
- `hc_act_float` 导出宏:`csrc/kernels/hc_act.h:379`
- token 循环:`csrc/kernels/hc_act.h:230`
- preSum 分支 UB 分配:`csrc/kernels/hc_act.h:174`
- vmuls→pipe_barrier(精度关键):`csrc/kernels/hc_act.h:245`、`csrc/kernels/hc_act.h:263`
- pre sum(vaxpy)/pre gather 分支:`csrc/kernels/hc_act.h:298`、`csrc/kernels/hc_act.h:307`
- Sinkhorn 迭代:`csrc/kernels/hc_act.h:346`
- host 封装 `XliteOpHcAct`:`csrc/op.cpp:2223`;`XliteOpHcSplitSinkhorn`:`csrc/op.cpp:2275`
