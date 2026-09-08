# sqrtsoftplus_hash_topk

## 功能概述

V4 MoE 路由门控算子(DeepSeek V4 风格,`sqrtsoftplus` 激活 + 可选 hash 选路),见 kernel 头注释(`csrc/kernels/sqrtsoftplus_hash_topk.h:22-27`)。数学语义(测试参考 `tests/sqrtsoftplus_hash_topk.py:25-39`):

```
s0 = sqrt(softplus(scores))             # 原始得分(pre-bias)
if bias:  b = s0 + bias                 # biased 得分,仅用于选路
indices = hash ? tid2eid[input_ids] : topk(b, K)[1]   # hash 选路或 biased topk
w = s0.gather(indices)                  # 权重取 pre-bias 原值
w = w / sum(w) * scale                  # 归一化(总是执行)再乘 route_scale
```

输出两个 GM 结果:**稀疏 `[M,N]` 权重行**(topK 个非零槽位)+ **`[M, ceil(N/32)]` BIT1 路由位图**;dense topk 索引 `indicesTopK` 是核内中间量,不落盘(`sqrtsoftplus_hash_topk.h:27`)。与 `sigmoid_topk` 的区别:激活函数不同(sqrtsoftplus vs sigmoid)、无分组限制、支持 hash 查表选路(`tid2eid[input_ids]`)且归一化总是执行(`sqrtsoftplus_hash_topk.h:337` 的 `vdiv` 无条件)。

## 输入输出参数

Python 侧调用(`tests/kernels/sqrtsoftplus_hash_topk.py:102`):

```python
sqrtsoftplus_hash_topk(rt, scores, indices_helper, bias, input_ids, tid2eid,
                       out_weights, routing_map, scale, top_k, use_hash)
```

host 侧 launch 见 `csrc/op.cpp:1260`(`XliteOpSqrtsoftplusHashTopK`)。kernel 签名(`csrc/kernels/sqrtsoftplus_hash_topk.h:432`):

```cpp
sqrtsoftplus_hash_topk_<dtype>(GM_ADDR scores, GM_ADDR indices, GM_ADDR bias,
                               GM_ADDR inputIds, GM_ADDR tid2eid, GM_ADDR outWeights,
                               GM_ADDR routingMap, float scale, uint32_t numTokens,
                               uint32_t numRoutedExperts, uint32_t topK, uint8_t hash)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| scores | 输入 | `[numTokens, numRoutedExperts]` | fp32 / bf16 | router logits |
| indices | 输入 | `[numRoutedExperts]` | int32 | `arange` 恒等索引表(host 传 `indices.shape[0]` 作为专家数) |
| bias | 输入 | `[numRoutedExperts]` 或空 | fp32 | 选路 bias;hash 模式下 host 传 nullptr(dead weight,`csrc/op.cpp:1281-1283`) |
| inputIds | 输入 | `[numTokens]` 或空 | int32 | token 的 input id(词表索引);仅 hash 模式使用 |
| tid2eid | 输入 | `[vocab_size, topK]` 或空 | int32 | hash 表:input_id → topK 个专家 id;仅 hash 模式使用 |
| outWeights | 输出 | `[numTokens, numRoutedExperts]` | 与 scores 同 dtype | 稀疏权重行:被选 topK 槽位为归一化×scale 权重,其余 0 |
| routingMap | 输出 | `[numTokens, ceil(numRoutedExperts/32)]` | BIT1(u32 位图) | 每个被选专家置 1 bit |
| scale | — | 标量 | float | route_scale(V4 默认 1.5) |
| numTokens / numRoutedExperts / topK / hash | — | 标量 | uint32 / u8 | 测试配置:M ∈ {9, 4096},N ∈ {256, 384},topK=6,hash ∈ {0,1} |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `sqrtsoftplus_hash_topk_float` | `csrc/kernels/sqrtsoftplus_hash_topk_float.cpp` | scores/outWeights 为 fp32 |
| `sqrtsoftplus_hash_topk_bfloat16_t` | `csrc/kernels/sqrtsoftplus_hash_topk_bfloat16_t.cpp` | scores/outWeights 为 bf16(核内 fp32 计算) |

dtype 分派见 `csrc/op.cpp:1269-1280`(indices 须 INT32、routingMap 须 BIT1)。纯向量核。

## 实现原理

### 任务划分与 hash 输入预取

`Run`(`csrc/kernels/sqrtsoftplus_hash_topk.h:112-167`):token 维跨核交错(`tokenIdx = block_idx; ... += block_num`,`sqrtsoftplus_hash_topk.h:141`)。恒等索引与 bias 在循环外一次性搬入(`sqrtsoftplus_hash_topk.h:118-123`)。

hash 模式的 `input_ids` 采用**窗口预取**:`PrefetchInputIds`(`sqrtsoftplus_hash_topk.h:254-269`)把最多 `HASH_INPUT_IDS_BATCH = 256` 个 input_id 一次 DMA 到 UB(`inputIdsArr`),S 管线标量直接从 UB 读;窗口滑出后(`tokenIdx >= baseInputIdsIdx + 256`)再拉下一批,UB 占用恒定(常量定义 `sqrtsoftplus_hash_topk.h:20`)。

### 单 token 流水(7 阶段)

1. **`FetchInputs`**(`sqrtsoftplus_hash_topk.h:243-249`)= `PrefetchInputIds`(hash)+ `CopyIn`:
   - `CopyIn`(`sqrtsoftplus_hash_topk.h:175-212`):得分行 DMA 到乒乓 `scoresInTmp[curr]`,V 管线接力转 fp32 到 `scoresIn`(bf16 用 `vconv_bf162f32`,fp32 用 `copy_ubuf_to_ubuf`,`sqrtsoftplus_hash_topk.h:195-202`);hash 模式下 S 管线读 `inputId = inputIdsArr[tokenIdx - baseInputIdsIdx]`,计算 `tid2eidGm + inputId * topK` 行地址,经 S→MTE2 EVENT_ID1 地址栅栏后 DMA 该行(长度 topK,不保证 32B 对齐故走 `CopyGmToUbufAligned` 的 b16 回退,`sqrtsoftplus_hash_topk.h:189-192`)到乒乓 `tid2eidRow[curr]`,再由 V 拷入 `indicesTopK` 作为"外部给定的选路索引";
   - 三条独立 flag 轨道:scores 用 MTE2↔V ID0+curr、tid2eid 用 MTE2↔V ID2+curr、地址就绪用 S→MTE2 ID1(`sqrtsoftplus_hash_topk.h:169-174` 注释);
2. **`CalcScores`**(`sqrtsoftplus_hash_topk.h:217-239`):数值稳定的 softplus —— `vabs`→`vmuls(-1)`→`vexp`→`vadds(1)`→`vln` 得 `log(1+exp(-|x|))`,`vrelu` 得 `max(x,0)`,相加后 `vsqrt` 得原始得分 `calc_unbiased`;有 bias 时 `vadd` 得选路得分 `calc`;
3. **`SelectTopK`**(仅非 hash,`sqrtsoftplus_hash_topk.h:273-315`):与 `sigmoid_topk` 相同的"vbitsort 块排序 + vmrgsort4 四路归并 + 逐块二路归并"多块循环(`tailLen = N/32 - 4`,N=256 时为 4),`vreducev2` 抽出 topK 索引 `indicesTopK`;
4. **`GatherWeights`**(`sqrtsoftplus_hash_topk.h:321-328`):`vmuls(idx*4)` 索引转字节偏移(注意注释:此处标量必须传裸 int `4`,不能是 float),`vgather` 从 `calc_unbiased`(pre-bias)收集 topK 权重 —— hash 与非 hash 两条路径在此汇合(hash 的索引来自 tid2eid,权重同样查 pre-bias 值);
5. **`NormalizeAndScale`**(`sqrtsoftplus_hash_topk.h:332-342`):`ReduceSum`+`vbrcb`+`vdiv`(无条件归一化)+ `vmuls(scale)`;
6. **`FillRoutingMap`**(`sqrtsoftplus_hash_topk.h:347-361`):`vector_dup` 清零稀疏权重行 `weightsSparse[N]` 与位图 `routingMap[N/32]`,S 管线标量循环 topK 次 `bitmapSet` + 稀疏 scatter;
7. **`StageOut`/`CopyOut`**(`sqrtsoftplus_hash_topk.h:365-393`):V 管线把结果接力到乒乓输出槽 `weightsOut[curr]`(bf16 转换 / fp32 拷贝)与 `routingMapOut[curr]`,MTE3 异步 DMA 到 GM —— 计算与搬出乒乓重叠。

### UB 内存布局

`Init`(`sqrtsoftplus_hash_topk.h:55-109`)精心划分了四类区域(头注释 `sqrtsoftplus_hash_topk.h:61-62,91,101`):

- **输入 DMA 乒乓**:`scoresInTmp[2]`(Dtype)、`tid2eidRow[2]`(topK 个 u32);
- **常驻**:`scoresIn`(V 私有 fp32)、`biasIn`、`indicesIn`、`inputIdsArr`(256×u32);
- **计算区(单份,V/S 私有)**:`calc`、`calc_unbiased`、`reduceTmp`、`sortTmp`(2 倍)、`sortMrgTmp`(2 倍)、`weightsTopK`、`indicesTopK`、`weightsSparse`、`routingMap`;
- **输出 DMA 乒乓**:`weightsOut[2]`(Dtype)、`routingMapOut[2]`。

topK 相关缓冲按 256B 向量块对齐(`padK`,`sqrtsoftplus_hash_topk.h:58-59` 注释:ReduceSum/vbrcb/vreducev2 假定块对齐)。

### 流水线同步

全算子以计数事件构成多轨软件流水(头注释 `sqrtsoftplus_hash_topk.h:128-131`):scores 轨道 V↔MTE2 ID0+curr、输出轨道 V↔MTE3 ID0+curr、tid2eid 轨道 V↔MTE2 ID2+curr,`Run` 开头"预置"全部 slot-free flag 使首 token 不阻塞(`sqrtsoftplus_hash_topk.h:131-138`),循环结束后 drain 最后一次 set,防止计数 flag 泄漏到下一次 launch(`sqrtsoftplus_hash_topk.h:156-165`)。V→S 的 EVENT_ID0 用于 S 管线读 `indicesTopK` 前的栅栏(`sqrtsoftplus_hash_topk.h:341,352`);S→MTE2 的 EVENT_ID1 用于 tid2eid 行地址可见性(`sqrtsoftplus_hash_topk.h:187-188`)。
