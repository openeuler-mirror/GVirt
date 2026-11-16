# softmax_topk

## 功能概述

MoE(Mixture of Experts)路由门控算子:对每个 token 的 router 得分做 softmax,再选出 topK 个专家,输出"稀疏权重行 + 专家路由位图"两个结果(DeepSeek 系列模型 `norm_topk_prob=True` 的标准门控,`softmax → topk → 归一化` 全融合在单个向量 kernel 内)。排序宽度随专家数自适应:`numRoutedExperts` 低于 `vmrgsort4` 无条件 4 路归并的下限 `4*SORT_BLOCK_SIZE = 128`(如 GLM-4.7-Flash 的 64 专家)时补齐到 128;高于 128 时由 `MrgSort` 按 4 叉平衡树逐层归并(见实现原理 SelectTopK 段),因此 256 专家的 Qwen3.5-MoE / 512 专家的 Qwen3.8-Max 同样适用。数学语义(见测试 `tests/kernels/softmax_topk.py` 参考实现段,关键字 `standard_softmax`/`torch.topk`):

```text
p = softmax(scores)                       # [nRoutedExperts]
topkW, topkI = topk(p, topK)              # 取 topK
topkW = topkW / sum(topkW)                # normTopKProb 时
weightsMap[i, topkI[j]] = topkW[j]        # 稀疏权重(其余为 0)
routingMap[i] |= bit(topkI[j])            # 位图路由
```

## 输入输出参数

Python 侧调用(`tests/kernels/softmax_topk.py` 中 `softmax_topk(rt, ...)` 调用):

```python
softmax_topk(rt, scores, indices, weights_out, indices_out, topK, normTopKProb)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpSoftmaxTopK`。kernel 签名(`csrc/kernels/softmax_topk.h` 的 `SOFTMAX_TOPK_FUNC_DEFINE` 宏 / `softmax_topk_<dtype>` 入口):

```cpp
softmax_topk_<dtype>(GM_ADDR scores, GM_ADDR indices, GM_ADDR weightsMap,
                     GM_ADDR routingMap, uint32_t numTokens, uint32_t numRoutedExperts,
                     uint32_t topK, bool normTopKProb)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
| --- | --- | --- | --- | --- |
| scores(即 scores) | 输入 | `[numTokens, numRoutedExperts]` | fp32 / bf16 | router logits |
| indices | 输入 | `[numRoutedExperts]` | int32 | `arange` 恒等索引表(host 传 `indices.shape[0]` 作为专家数) |
| weightsMap | 输出 | `[numTokens, numRoutedExperts]` | 与 scores 同 dtype | 稀疏权重行:topK 个被选槽位为归一化后的概率,其余为 0 |
| routingMap | 输出 | `[numTokens, ceil(numRoutedExperts/32)]`(BIT1 打包) | BIT1(uint32 位图) | 每个被选专家置 1 bit。注意:BIT1 语义下每 32 个专家打包为 1 个 u32(测试 `tests/kernels/softmax_topk.py` BIT1 打包段,关键字 `elem / 64`、`1 << (elem % 64)` 是因为 BIT1 张量在 torch 侧被视作 int64 承载) |
| numTokens / numRoutedExperts / topK / normTopKProb | — | 标量 | uint32 / bool | 测试配置:nRoutedExperts=64/128/192/256/512(64 覆盖 <128 的补齐路径,192/256/512 覆盖尾块累积归并),topK=8,normTopKProb=True。测试仅取 64 的倍数是测试侧 BIT1 位图以 int64 承载的约束:kernel 本身按 `nRoutedExperts/32` 个 u32 步进 GM、每 token DMA 精确 `nRoutedExperts/8` 字节,非 64 倍数时 int64 载体凑不成整字(`tests/kernels/sqrtsoftplus_hash_topk.py` 的 int32 载体即覆盖 96 专家) |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
| --- | --- | --- |
| `softmax_topk_float` | `csrc/kernels/softmax_topk_float.cpp` | scores/weightsMap 为 fp32 |
| `softmax_topk_bfloat16_t` | `csrc/kernels/softmax_topk_bfloat16_t.cpp` | scores/weightsMap 为 bf16(核内转 fp32 计算,输出转回 bf16) |

dtype 分派见 `csrc/op.cpp` 的 `XliteOpSoftmaxTopK`(dtype 分派分支):要求 scores 与 weightsMap 同 dtype、indices 为 INT32、routingMap 为 BIT1。纯向量核。

## 实现原理

### 任务划分

`Run`(`csrc/kernels/softmax_topk.h` 的 `SoftmaxTopK::Run`):token 维循环跨核交错分配 `for (tokenIdx = block_idx; tokenIdx < nTokens; tokenIdx += block_num)`(`SoftmaxTopK::Run` 内 token 循环),每 token 独立,无核间同步。`indicesIn`(恒等索引表)在循环外一次性搬入 UB,所有 token 共享(`SoftmaxTopK::Run` 开头 `copy_gm_to_ubuf_align_b32(indicesIn, ...)`)。

### 单 token 流水

每个 token 依次经过 6 个阶段(`SoftmaxTopK::Run` 内阶段调用序列):

1. **`InitOutBuf`**(`SoftmaxTopK::InitOutBuf`):`vector_dup` 清零 `routingMapOut` 与 `weightsOut`,经 MTE3→V 事件确保上一轮 DMA 搬出已完成;
2. **`CopyInScores`**(`SoftmaxTopK::CopyInScores`):该 token 的得分行 GM→UB(bf16 走 `socresInTmp` 中转,fp32 直入 `socresIn`);
3. **`CalcSoftmax`**(`SoftmaxTopK::CalcSoftmax`):bf16 先 `vconv_bf162f32`;`ReduceMax` + `vbrcb`(标量广播到整 block)求 max;`vsub` 减 max 后 `vexp`;`ReduceSum` + `vbrcb` 求和;`vdiv` 归一 —— 标准的数值稳定 softmax,全程 fp32;
4. **`SelectTopK`**(`SoftmaxTopK::SelectTopK`)—— **排序式 topk**(小专家数整行一次排完):
   - 排序宽度 `nSortScores = max(numRoutedExperts, 4*SORT_BLOCK_SIZE)`(`SORT_BLOCK_SIZE=32`,定义于 `csrc/kernels/kernel_macro.h`,即下限 128)。`vmrgsort4` 是无条件 4 路归并,候选不足 4 块时归并会消费未初始化 lane、静默污染 topK,故宽度补齐;
   - `vbitsort(sortTmp, calc, indicesIn, nSortScores/32)`:`nSortScores/32` 组、每组 32 元素的位排序,输出值+索引成对的元素;
   - 补齐段处理:`InitSortTail`(`SoftmaxTopK::InitSortTail`,每 Run 一次)把 `indicesIn` 的补齐 lane 索引填 0(与真实专家 0 重复,但永远不会被选中,见下)并以 `PIPE_ALL` 栅栏保序;`PadCalcTail`(`SoftmaxTopK::PadCalcTail`,每 token)把 `calc` 的补齐 lane 重钉为 `FLOAT_MIN`——`CalcSoftmax` 按 `DIV_ROUND_UP(numRoutedExperts, 64)` 个 repeat 重写 `calc`,专家数非 64 整数倍(如 96→2 repeat→128 lane)时写宽会覆写补齐段,故须逐 token 重钉;softmax 输出恒正,必须用 `FLOAT_MIN` 而非 0 才能保证排在最后、抢不走 topK 名额。两侧 `PIPE_ALL` 栅栏分别将 S 管线标量写保序于在飞的 V 管线读写;`numRoutedExperts >= 128` 时两函数直接早退,零开销;
   - 归并(`SoftmaxTopK::SelectTopK`):`vbitsort` 之后统一调用 `MrgSort`(`csrc/kernels/kernel_macro.h` 的 `MrgSort`)——它按 4 叉平衡树逐层归并 `nSortScores/SORT_BLOCK_SIZE` 个块,并在每层把队列剪枝到 `MIN(lenPerRepeat, topK)`,因此候选数超过 128 时既不需要手写折叠循环,也不受 `vmrgsort4` 单次 4 队列的限制;函数通过出参返回结果所在缓冲(`sortTmp` 或 `sortMrgTmp`,取决于归并层数),后续 `vreducev2` 从该指针抽取;
   - `vreducev2` 两次(mode=2 隔 2 取索引、mode=1 取值)分别抽出 topK 的**索引载荷** `indicesTopK` 与**值载荷** `weightsTopK`;
   - `normTopKProb` 时 `ReduceSum` + `vbrcb` + `vdiv` 对 topK 权重归一化;
5. **`FillOutMap`**(`SoftmaxTopK::FillOutMap`):S 管线标量循环 topK 次,`bitmapSet`(`csrc/kernels/kernel_macro.h` 的 `bitmapSet`,u64 字内置位)写路由位图,并按索引 scatter 权重到 `weightsOut` 稀疏槽位;bf16 时再 `vconv_f322bf16r` 转换输出;
6. **`CopyOutMap`**(`SoftmaxTopK::CopyOutMap`):位图(`nRoutedExperts/32` 个 u32)与稀疏权重行 GM 搬出。

### UB 内存布局与同步

`Init`(`SoftmaxTopK::Init`)按 `nRoutedExperts` 上界分配:`socresIn`/`routingMapOut`/`weightsOut` 以 `pad = ROUND_UP(nRoutedExperts, 64) * 4` 字节定尺,bf16 额外有 `socresInTmp`/`weightsOutTmp`;`reduceTmp` 与 `weightsTopK` 各固定一个 `VECTOR_MAX_BYTESIZE`(256B),`indicesTopK` 则是 fp32 路径的末尾分配,其后无 off 增量。被排序触碰的 `indicesIn`/`calc` 及 `sortTmp`/`sortMrgTmp`(各 2 倍)按补齐宽度定尺(`idxPad = max(pad, ROUND_UP(nSortScores,64)*4)`);`numRoutedExperts >= 128` 时 `idxPad == pad`,UB 布局与旧版逐字节一致。

同步:MTE2 搬入用 V↔MTE2 的 EVENT_ID0;搬出用 MTE3↔V 的 EVENT_ID0(位图)/EVENT_ID1(权重);S 管线消费 V 产出的 `indicesTopK` 前有 `set_flag(PIPE_V, PIPE_S)` / `wait_flag` 配对(`SelectTopK` 末尾 `set_flag(PIPE_V, PIPE_S, EVENT_ID0)` 与 `FillOutMap` 开头 `wait_flag`),保证标量读到的位图/权重已完成。循环结束统一 `wait_flag` 清算 + `pipe_barrier(PIPE_ALL)`(`SoftmaxTopK::Run` 末尾 drain 段)。

### 片上缓冲分配与占用汇总(AIV UB)

`Init`(`csrc/kernels/softmax_topk.h` 的 `SoftmaxTopK::Init`)自偏移 0 起在 UB 上顺序累进排布:普通行缓冲以 `pad = ROUND_UP(nRoutedExperts, VECTOR_MAX_NUM_OF_FP32)*4` 定尺,被排序触碰的 `indicesIn`/`calc` 及 `sortTmp`/`sortMrgTmp` 按补齐宽度 `idxPad = MAX(pad, ROUND_UP(nSortScores,64)*4)` 定尺;计算与排序全程 fp32,bf16 仅多两个 Dtype 中转槽。硬件容量基准(910B/dav C220,NpuArch 2201):UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每个 AIV subblock 各持一份独立 UB、布局相同。dtype 记法:`Dtype` 为 scores/weightsMap 元素类型(fp32/bf16),`padDtype = ROUND_UP(nRoutedExperts, VECTOR_MAX_BYTESIZE/sizeof(Dtype))*sizeof(Dtype)`;`numTokens` 不参与定尺(token 维跨核交错)。典型配置:numRoutedExperts=64、topK=8、normTopKProb=True(GLM-4.7-Flash 门控,触发 <128 的补齐路径:`nSortScores=128`,`idxPad=512 > pad=256`);两条 dtype 路径同取该配置,占用差异仅来自 bf16 中转槽。

**AIV UB(预算 `UB_SIZE=192KB`,每 subblock 一份)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | fp32 占用(B) | bf16 占用(B) |
|---|---|---|---|---|---|---|
| `socresIn` | 64 float | float | 1 | `pad` | 256 | 256 |
| `indicesIn` | 128 uint32(补齐到 nSortScores) | uint32 | 1 | `idxPad` | 512 | 512 |
| `routingMapOut` | 2 uint32 | uint32 | 1 | `pad` | 256 | 256 |
| `weightsOut` | 64 float | float | 1 | `pad` | 256 | 256 |
| `calc` | 128 float(补齐到 nSortScores) | float | 1 | `idxPad` | 512 | 512 |
| `reduceTmp` | 64 float 槽位 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 | 256 |
| `sortTmp` | 256 float(128 值+索引交错对) | float | 1 | `2*idxPad` | 1024 | 1024 |
| `sortMrgTmp` | 256 float(128 值+索引交错对) | float | 1 | `2*idxPad` | 1024 | 1024 |
| `weightsTopK` | 64 float 槽位(topK=8 载荷) | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 | 256 |
| `indicesTopK` | 64 uint32 槽位(topK=8 载荷) | uint32 | 1 | `pad`(fp32 路径为末尾槽) | 256 | 256 |
| `socresInTmp` | 64 Dtype(仅 bf16) | Dtype | 1 | `padDtype` | — | 256 |
| `weightsOutTmp` | 64 Dtype(仅 bf16,末尾槽) | Dtype | 1 | `padDtype` | — | 256 |
| **合计** | | | | | **4608** | **5120** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | fp32 路径占用 | 占比 | bf16 路径占用 | 占比 |
|---|---|---|---|---|---|
| UB(每 AIV subblock) | 192 KB | 4608 B | 2.3% | 5120 B | 2.6% |

要点:

- **UB 极其宽裕(≤2.6%),唯一算得上"大头"的是排序工作区**:`sortTmp`/`sortMrgTmp` 各 `2*idxPad`、合计 2048 B,占总占用的 40%(bf16 路径 5120 B 口径;fp32 路径 4608 B 下为 44%)——值+索引成对排序的暂存宽度;
- `idxPad > pad` 仅出现在 `numRoutedExperts < 128` 的补齐路径(64 专家时 512 vs 256);`numRoutedExperts >= 128` 时 `idxPad == pad`,布局回落到与历史版本逐字节一致;
- 无 kernel 断言、无 host 容量校验(`csrc/op.cpp` 的 `XliteOpSoftmaxTopK` 只校验 dtype/BIT1),布局随 `numRoutedExperts` 线性缩放,现实专家数下远低于 192KB,容量风险不存在。
