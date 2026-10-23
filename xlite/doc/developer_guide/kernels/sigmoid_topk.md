# sigmoid_topk

## 功能概述

带分组限制(group-limited)的 MoE 路由门控算子(DeepSeek-V2/V3 风格):对 router 得分做 sigmoid,加上 bias 后先按专家分组选出 top `nTopkGroup` 个组,再在选中组的专家里选 topK,输出"稀疏权重行 + 路由位图"。数学语义(见测试 `tests/kernels/sigmoid_topk.py` 参考实现段,关键字 `scores_sigmoid`/`group_scores`):

```
s = sigmoid(scores)                    # 未加 bias 的"原始"得分
b = s + bias                           # biased 得分,仅用于选路
groupScores[g] = sum(top2(b[g*nEpg : (g+1)*nEpg]))   # 组得分 = 组内 top2 之和
groupIdx = topk(groupScores, nTopkGroup)
b 去掉未选中组的专家(masked)
topkI = topk(b_masked, topK);  topkW = s.gather(topkI)   # 权重取未加 bias 的原值!
topkW = topkW / sum(topkW) * scale     # normTopKProb 时归一,再乘 scale
```

关键细节:**bias 只引导选路,权重取 pre-bias 的 sigmoid 值**(kernel 注释 "bias steers topk only" 在 `sqrtsoftplus_hash_topk.h` 中同样出现;本算子的 `vgather(calc_unbiased)` 对应此语义,见 `csrc/kernels/sigmoid_topk.h` 的 `SigmoidTopK::SelectTopK`(`vgather(calc_unbiased)` 段))。

## 输入输出参数

Python 侧调用(`tests/kernels/sigmoid_topk.py` 中 `sigmoid_topk(rt, ...)` 调用):

```python
sigmoid_topk(rt, scores, indices, bias, scale, weights_out, indices_out,
             n_group, topk_group, topK, True)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpSigmoidTopK`。kernel 签名(`csrc/kernels/sigmoid_topk.h` 的 `SIGMOID_TOPK_FUNC_DEFINE` 宏 / `sigmoid_topk_<dtype>` 入口):

```cpp
sigmoid_topk_<dtype>(GM_ADDR scores, GM_ADDR indices, GM_ADDR bias, float scale,
                     GM_ADDR weightsMap, GM_ADDR routingMap, uint32_t numTokens,
                     uint32_t numRoutedExperts, uint32_t nGroup, uint32_t nTopkGroup,
                     uint32_t topK, bool normTopKProb)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| scores | 输入 | `[numTokens, numRoutedExperts]` | fp32 / bf16 | router logits |
| indices | 输入 | `[numRoutedExperts]` | int32 | `arange` 恒等索引表(host 传 `indices.shape[0]` 作为专家数) |
| bias | 输入(可选) | `[numRoutedExperts]` | fp32 | 专家 bias(加性校正,只影响选路)。**可传空 tensor**:host 侧 `TensorPtr()` 得到 `nullptr`,device 端跳过搬运并把 UB 的 `biasIn` 一次性清零,使 bias 加法成为精确 no-op。无 `e_score_correction_bias` 的模型(如 GLM-4.7-Flash 的 `noaux_tc`)走此路径 |
| scale | — | 标量 | float | 权重缩放(route_scale),归一化后相乘 |
| weightsMap | 输出 | `[numTokens, numRoutedExperts]` | 与 scores 同 dtype | 稀疏权重行:被选 topK 槽位为归一化×scale 的权重,其余 0 |
| routingMap | 输出 | `[numTokens, ceil(numRoutedExperts/32)]` | BIT1(u32 位图) | 每个被选专家置 1 bit |
| numTokens / numRoutedExperts / nGroup / nTopkGroup / topK / normTopKProb | — | 标量 | uint32 / bool | 测试配置:(160,1,1,8)、(256,8,4,8)、(256,1,1,8)、(64,1,1,4)(GLM-4.7-Flash,含有 bias / 无 bias 两种),即专家数 64/160/256、组数 1/8、组选数 1/4、topK=4/8;`nExpertsPerGroup = numRoutedExperts / nGroup`,候选得分长度 `nScores = nGroup==1 ? numRoutedExperts : nTopkGroup*nExpertsPerGroup`(`csrc/kernels/sigmoid_topk.h` 的 `SigmoidTopK::Init`,关键字 `nScores`),而**实际参与排序的宽度**是 `nSortScores = max(nScores, 4*SORT_BLOCK_SIZE)`(`csrc/kernels/sigmoid_topk.h` 的 `SigmoidTopK::Init`,关键字 `nSortScores`,详见下文 SelectTopK) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `sigmoid_topk_float` | `csrc/kernels/sigmoid_topk_float.cpp` | scores/weightsMap 为 fp32 |
| `sigmoid_topk_bfloat16_t` | `csrc/kernels/sigmoid_topk_bfloat16_t.cpp` | scores/weightsMap 为 bf16(核内 fp32 计算,输出 `vconv_f322bf16r` 转回) |

dtype 分派见 `csrc/op.cpp` 的 `XliteOpSigmoidTopK`(dtype 分派分支,indices 须 INT32、routingMap 须 BIT1)。纯向量核。

## 实现原理

### 任务划分

`Run`(`csrc/kernels/sigmoid_topk.h` 的 `SigmoidTopK::Run`):token 维跨核交错(`tokenIdx = block_idx; tokenIdx += block_num`,`SigmoidTopK::Run` 内 token 循环),每 token 独立。恒等索引 `indicesIn` 在循环外预取一次(`SigmoidTopK::Run` 开头 `copy_gm_to_ubuf_align_b32(indicesIn, ...)`),紧接 `SigmoidTopK::InitSortTail` 把排序补齐尾部的 lane 填成 `FLOAT_MIN`/索引 0;`biasIn` 在每 token 的 `CopyInScores` 中搬入(bias 是 fp32,见 `SigmoidTopK::CopyInScores`)——**当 bias 为空指针时跳过该次搬运**,改为在 `Run` 里用 `vector_dup` 把 `biasIn` 清零一次(`calcRepeat*64*4 == pad`,恰好覆盖整个缓冲区),使 `CalcGroupScores` 里两处无条件的 `vadd(..., biasIn, ...)` 成为加 0.0 的精确 no-op,无需为有/无 bias 复制两套分支。

### 单 token 流水(6 阶段)

1. **`InitOutBuf`**(`SigmoidTopK::InitOutBuf`):清零输出位图与权重行;
2. **`CopyInScores`**(`SigmoidTopK::CopyInScores`):得分 + bias 搬入(bf16 经 `scoresInTmp` 中转);
3. **`CalcSigmoid`**(`SigmoidTopK::CalcSigmoid`):bf16 转 fp32 后,`vector_dup(0)` + `vsub` 得 `-x`,`vexp`、`vadd`(+1)、`vdiv` 得 `1/(1+exp(-x))`,存入 `calc_unbiased`。其中在 `vsub` 后即 `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0)` 提前释放搬入缓冲(`SigmoidTopK::CalcSigmoid` 内提前释放 MTE2 段),使**下一 token 的得分搬入与本 token 的 exp/div 计算重叠**;
4. **`CalcGroupScores`**(`SigmoidTopK::CalcGroupScores`):
   - `nGroup == 1`:直接 `scores = calc_unbiased + bias`(`SigmoidTopK::CalcGroupScores` 内 `nGroup == 1` 分支),无分组;
   - `nGroup > 1`:`calc = calc_unbiased + bias`;对每组做 `vbitsort`(组内按 `SORT_BLOCK_SIZE=32` 位排序),S 管线取 `sortTmp[0] + sortTmp[2]` 即**组内 top2 之和**作为组得分(降序排序的前两个值载荷位于偏移 0 和 2,`SigmoidTopK::CalcGroupScores` 内组循环 `sortTmp[0] + sortTmp[2]`);再对 `nTopkGroup` 个组得分 `vbitsort` + `vreducev2` 抽出被选组号 `groupIndices`;最后 S 管线把被选组的 `calc`(biased 得分)与 `indicesIn`(对应索引段)按组拷贝拼接成连续候选区 `scores`/`indices`(每组 `nExpertsPerGroup` 个,`SigmoidTopK::CalcGroupScores` 内 `copy_ubuf_to_ubuf` 拼接循环),未选组专家即被剔除;
5. **`SelectTopK`**(`SigmoidTopK::SelectTopK`)—— **排序式 topk + 多块循环归并**(候选长度 `nScores` 超过 4 个 32 元素块时,如 256 专家/8 组×4 组=128 候选)。由于 `vmrgsort4` 是**无条件 4 路归并**,排序宽度必须 ≥ `4*SORT_BLOCK_SIZE = 128`;当 `nScores < 128`(如 GLM-4.7-Flash 的 64 专家)时用 `nSortScores = max(nScores, 128)` 补齐,并把 `scores`/`indices`/`indicesIn`/`sortTmp`/`sortMrgTmp` 按补齐后的宽度定尺。补出的 lane 由 `SigmoidTopK::InitSortTail` 一次性填 `FLOAT_MIN`/索引 0——必然排在最后,既抢不到 topK 名额,也不会给 `vgather` 喂越界索引。注意 `nGroup == 1` 时 `CalcGroupScores` 的 `vadd` 按 `calcRepeat` 整 repeat 写 `scores`,专家数非 64 整数倍(如 96)时写宽越过 `nRoutedExperts` 覆写补齐段:因此 `SelectTopK` 排序前先调 `SigmoidTopK::PadScoresTail` 每 token 重钉 `FLOAT_MIN`(S 标量写,两侧 `PIPE_ALL` 栅栏分别保序于 V 管线写入/读取);`nGroup > 1` 拼接循环恰好只写 `nScores` 个 lane,一次性填充仍然有效:
   - `vbitsort(sortTmp, scores, ...)` 对全部候选按 32 一组位排序;
   - `vmrgsort4` 先归并前 4 块(`validBits=0xF`),结果拷回 `sortTmp`;
   - `for i in 4..4+tailLen`:`vmrgsort4` 二路归并("累计结果" + "下一块",`validBits=0b11`,长度编码 `32*i | 32<<16`)逐块并入(`SigmoidTopK::SelectTopK` 内 `for (i = 4; ...)` 归并循环)——tailLen = `nSortScores/32 - 4`,实现为 `nSortBlocks > 4 ? nSortBlocks - 4 : 0`,避免 `nScores < 128` 时 uint64 下溢;
   - `vreducev2` 抽 topK 索引 `indicesTopK`;`vmuls(idx, 4)` 把索引转字节偏移,`vgather` 从 `calc_unbiased`(pre-bias sigmoid)按索引收集权重 `weightsTopK`(`SigmoidTopK::SelectTopK` 内 `vgather(calc_unbiased)`);
   - `normTopKProb` 时 `ReduceSum`+`vbrcb`+`vdiv` 归一;`vmuls` 乘 `scale`;
6. **`FillOutMap` / `CopyOutMap`**(`SigmoidTopK::FillOutMap` 与 `CopyOutMap`):S 管线标量循环 `bitmapSet` + 稀疏权重 scatter,bf16 转换后位图与权重行搬出到 GM。

### UB 内存布局与同步

`Init`(`SigmoidTopK::Init`)按 `nRoutedExperts` 上界(`pad = ROUND_UP(N,64)*4` 字节)分配:`scoresIn`/`biasIn`/`routingMapOut`/`weightsOut`/`calc_unbiased`/`calc`/`reduceTmp` 各一份 pad;**被排序触碰的** `indicesIn`/`scores`/`indices` 用 `idxPad = max(pad, ROUND_UP(nSortScores,64)*4)`(`SigmoidTopK::Init` 内关键字 `idxPad`),`sortTmp`/`sortMrgTmp` 用 `2*idxPad`(对应上文 SelectTopK 的 128 候选下限);`groupScores`/`groupIndices`/`weightsTopK`/`indicesTopK` 各一个 256B 向量块;bf16 额外 `scoresInTmp`/`weightsOutTmp`。对 `nScores >= 128` 的所有配置 `idxPad == pad`,布局与历史版本逐字节一致。

同步轨道:V↔MTE2 EVENT_ID0(得分搬入)、MTE3↔V EVENT_ID0/1(位图/权重搬出)、V↔S EVENT_ID0(排序结果供标量读,如组 top2、被选组拼接)。`CalcSigmoid` 中提前释放 MTE2 轨道实现 token 间搬入/计算重叠是本算子流水的一个特点(`SigmoidTopK::CalcSigmoid` 内 `set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0)` 提前释放段)。
