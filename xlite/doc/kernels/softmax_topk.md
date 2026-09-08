# softmax_topk

## 功能概述

MoE(Mixture of Experts)路由门控算子:对每个 token 的 router 得分做 softmax,再选出 topK 个专家,输出"稀疏权重行 + 专家路由位图"两个结果(DeepSeek 系列模型 `norm_topk_prob=True` 的标准门控,`softmax → topk → 归一化` 全融合在单个向量 kernel 内)。数学语义(见测试 `tests/kernels/softmax_topk.py:30-40`):

```
p = softmax(scores)                       # [nRoutedExperts]
topkW, topkI = topk(p, topK)              # 取 topK
topkW = topkW / sum(topkW)                # normTopKProb 时
weightsMap[i, topkI[j]] = topkW[j]        # 稀疏权重(其余为 0)
routingMap[i] |= bit(topkI[j])            # 位图路由
```

## 输入输出参数

Python 侧调用(`tests/kernels/softmax_topk.py:43`):

```python
softmax_topk(rt, scores, indices, weights_out, indices_out, topK, normTopKProb)
```

host 侧 launch 见 `csrc/op.cpp:1214`(`XliteOpSoftmaxTopK`)。kernel 签名(`csrc/kernels/softmax_topk.h:269`):

```cpp
softmax_topk_<dtype>(GM_ADDR socres, GM_ADDR indices, GM_ADDR weightsMap,
                     GM_ADDR routingMap, uint32_t numTokens, uint32_t numRoutedExperts,
                     uint32_t topK, bool normTopKProb)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| scores(即 socres) | 输入 | `[numTokens, numRoutedExperts]` | fp32 / bf16 | router logits |
| indices | 输入 | `[numRoutedExperts]` | int32 | `arange` 恒等索引表(host 传 `indices.shape[0]` 作为专家数) |
| weightsMap | 输出 | `[numTokens, numRoutedExperts]` | 与 scores 同 dtype | 稀疏权重行:topK 个被选槽位为归一化后的概率,其余为 0 |
| routingMap | 输出 | `[numTokens, ceil(numRoutedExperts/32)]`(BIT1 打包) | BIT1(uint32 位图) | 每个被选专家置 1 bit。注意:BIT1 语义下每 32 个专家打包为 1 个 u32(测试 `tests/kernels/softmax_topk.py:38-39` 中 `idx = elem / 64`、`1 << (elem % 64)` 是因为 BIT1 张量在 torch 侧被视作 int64 承载) |
| numTokens / numRoutedExperts / topK / normTopKProb | — | 标量 | uint32 / bool | 测试配置:nRoutedExperts=128, topK=8, normTopKProb=True |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `softmax_topk_float` | `csrc/kernels/softmax_topk_float.cpp` | scores/weightsMap 为 fp32 |
| `softmax_topk_bfloat16_t` | `csrc/kernels/softmax_topk_bfloat16_t.cpp` | scores/weightsMap 为 bf16(核内转 fp32 计算,输出转回 bf16) |

dtype 分派见 `csrc/op.cpp:1220-1231`:要求 scores 与 weightsMap 同 dtype、indices 为 INT32、routingMap 为 BIT1。纯向量核。

## 实现原理

### 任务划分

`Run`(`csrc/kernels/softmax_topk.h:68-99`):token 维循环跨核交错分配 `for (tokenIdx = block_idx; tokenIdx < nTokens; tokenIdx += block_num)`(`softmax_topk.h:81`),每 token 独立,无核间同步。`indicesIn`(恒等索引表)在循环外一次性搬入 UB,所有 token 共享(`softmax_topk.h:74-76`)。

### 单 token 流水

每个 token 依次经过 6 个阶段(`softmax_topk.h:82-93`):

1. **`InitOutBuf`**(`softmax_topk.h:101-112`):`vector_dup` 清零 `routingMapOut` 与 `weightsOut`,经 MTE3→V 事件确保上一轮 DMA 搬出已完成;
2. **`CopyInScores`**(`softmax_topk.h:114-125`):该 token 的得分行 GM→UB(bf16 走 `socresInTmp` 中转,fp32 直入 `socresIn`);
3. **`CalcSoftmax`**(`softmax_topk.h:127-159`):bf16 先 `vconv_bf162f32`;`ReduceMax` + `vbrcb`(标量广播到整 block)求 max;`vsub` 减 max 后 `vexp`;`ReduceSum` + `vbrcb` 求和;`vdiv` 归一 —— 标准的数值稳定 softmax,全程 fp32;
4. **`SelectTopK`**(`softmax_topk.h:161-197`)—— **排序式 topk**(适用于 nRoutedExperts=128 这类小专家数,整行一次排完):
   - `vbitsort(sortTmp, calc, indicesIn, 128/32)`:4 组、每组 32 元素的位排序,输出值+索引成对的 4×64 元素;
   - `vmrgsort4(sortMrgTmp, addrArray, lengths, config)`:4 路归并(`validBits=0xF`,4 队列各长 32),一次指令把整行排成降序;
   - `vreducev2` 两次(mode=2 隔 2 取索引、mode=1 取值)分别抽出 topK 的**索引载荷** `indicesTopK` 与**值载荷** `weightsTopK`;
   - `normTopKProb` 时 `ReduceSum` + `vbrcb` + `vdiv` 对 topK 权重归一化;
5. **`FillOutMap`**(`softmax_topk.h:199-217`):S 管线标量循环 topK 次,`bitmapSet`(`csrc/kernels/kernel_macro.h:793-797`,u64 字内置位)写路由位图,并按索引 scatter 权重到 `weightsOut` 稀疏槽位;bf16 时再 `vconv_f322bf16r` 转换输出;
6. **`CopyOutMap`**(`softmax_topk.h:219-241`):位图(`nRoutedExperts/32` 个 u32)与稀疏权重行 GM 搬出。

### UB 内存布局与同步

`Init`(`softmax_topk.h:37-65`)按 `nRoutedExperts` 上界分配,全部以 `pad = ROUND_UP(nRoutedExperts, 64) * 4` 字节对齐:`socresIn`/`indicesIn`/`routingMapOut`/`weightsOut`/`calc`/`sortTmp`(2 倍)/`sortMrgTmp`(2 倍)/`weightsTopK`/`indicesTopK`,bf16 额外有 `socresInTmp`/`weightsOutTmp`;`reduceTmp` 固定一个 `VECTOR_MAX_BYTESIZE`(256B)。

同步:MTE2 搬入用 V↔MTE2 的 EVENT_ID0;搬出用 MTE3↔V 的 EVENT_ID0(位图)/EVENT_ID1(权重);S 管线消费 V 产出的 `indicesTopK` 前有 `set_flag(PIPE_V, PIPE_S)` / `wait_flag` 配对(`softmax_topk.h:196,201`),保证标量读到的位图/权重已完成。循环结束统一 `wait_flag` 清算 + `pipe_barrier(PIPE_ALL)`(`softmax_topk.h:95-98`)。
