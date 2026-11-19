# sqrtsoftplus_hash_topk

## 功能概述

V4 MoE 路由门控算子(DeepSeek V4 风格,`sqrtsoftplus` 激活 + 可选 hash 选路),见 kernel 头注释(`csrc/kernels/sqrtsoftplus_hash_topk.h` 顶部 `SqrtsoftplusHashTopK` 类注释,关键字 `bias steers topk only`)。数学语义(测试参考 `tests/kernels/sqrtsoftplus_hash_topk.py` 的 `ref_gate`):

```text
s0 = sqrt(softplus(scores))             # 原始得分(pre-bias)
if bias:  b = s0 + bias                 # biased 得分,仅用于选路
indices = hash ? tid2eid[input_ids] : topk(b, K)[1]   # hash 选路或 biased topk
w = s0.gather(indices)                  # 权重取 pre-bias 原值
w = w / sum(w) * scale                  # 归一化(总是执行)再乘 route_scale
```

输出两个 GM 结果:**稀疏 `[M,N]` 权重行**(topK 个非零槽位)+ **`[M, ceil(N/32)]` BIT1 路由位图**;dense topk 索引 `indicesTopK` 是核内中间量,不落盘(`csrc/kernels/sqrtsoftplus_hash_topk.h` 顶部注释,关键字 `indicesTopK is INTERNAL`)。与 `sigmoid_topk` 的区别:激活函数不同(sqrtsoftplus vs sigmoid)、无分组限制、支持 hash 查表选路(`tid2eid[input_ids]`)且归一化总是执行(`SqrtsoftplusHashTopK::NormalizeAndScale` 内 `vdiv` 无条件)。`numRoutedExperts` 低于 `vmrgsort4` 归并下限 `4*SORT_BLOCK_SIZE = 128`(如 GLM-4.7-Flash 的 64/96 专家)时,`SelectTopK` 的排序宽度补齐到 128——仅影响非 hash 路径,hash 路径不排序但仍按补齐宽度分配缓冲,机制见实现原理。

## 输入输出参数

Python 侧调用(`tests/kernels/sqrtsoftplus_hash_topk.py` 中 `sqrtsoftplus_hash_topk(rt, ...)` 调用):

```python
sqrtsoftplus_hash_topk(
    rt, scores, indices_helper, bias, input_ids, tid2eid, out_weights, routing_map, scale, top_k, use_hash
)
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpSqrtsoftplusHashTopK`。kernel 签名(`csrc/kernels/sqrtsoftplus_hash_topk.h` 的 `SQRTSOFTPLUS_HASH_TOPK_FUNC_DEFINE` 宏 / `sqrtsoftplus_hash_topk_<dtype>` 入口):

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
| bias | 输入 | `[numRoutedExperts]` 或空 | fp32 | 选路 bias;hash 模式下 host 传 nullptr(dead weight,`csrc/op.cpp` 的 `XliteOpSqrtsoftplusHashTopK`,关键字 `biasPtr = hash ? nullptr`) |
| inputIds | 输入 | `[numTokens]` 或空 | int32 | token 的 input id(词表索引);仅 hash 模式使用 |
| tid2eid | 输入 | `[vocab_size, topK]` 或空 | int32 | hash 表:input_id → topK 个专家 id;仅 hash 模式使用 |
| outWeights | 输出 | `[numTokens, numRoutedExperts]` | 与 scores 同 dtype | 稀疏权重行:被选 topK 槽位为归一化×scale 权重,其余 0 |
| routingMap | 输出 | `[numTokens, ceil(numRoutedExperts/32)]` | BIT1(u32 位图) | 每个被选专家置 1 bit |
| scale | — | 标量 | float | route_scale(V4 默认 1.5) |
| numTokens / numRoutedExperts / topK / hash | — | 标量 | uint32 / u8 | 测试配置:M ∈ {9, 4096},N ∈ {64, 96, 256, 384}(64/96 覆盖 <128 的补齐路径),topK=6,hash ∈ {0,1} |

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `sqrtsoftplus_hash_topk_float` | `csrc/kernels/sqrtsoftplus_hash_topk_float.cpp` | scores/outWeights 为 fp32 |
| `sqrtsoftplus_hash_topk_bfloat16_t` | `csrc/kernels/sqrtsoftplus_hash_topk_bfloat16_t.cpp` | scores/outWeights 为 bf16(核内 fp32 计算) |

dtype 分派见 `csrc/op.cpp` 的 `XliteOpSqrtsoftplusHashTopK`(dtype 分派分支,indices 须 INT32、routingMap 须 BIT1)。纯向量核。

## 实现原理

### 任务划分与 hash 输入预取

`Run`(`csrc/kernels/sqrtsoftplus_hash_topk.h` 的 `SqrtsoftplusHashTopK::Run`):token 维跨核交错(`tokenIdx = block_idx; ... += block_num`,`SqrtsoftplusHashTopK::Run` 内 token 循环)。恒等索引与 bias 在循环外一次性搬入(`SqrtsoftplusHashTopK::Run` 开头 `copy_gm_to_ubuf_align_b32(indicesIn/biasIn, ...)`)。

hash 模式的 `input_ids` 采用**窗口预取**:`PrefetchInputIds`(`SqrtsoftplusHashTopK::PrefetchInputIds`)把最多 `HASH_INPUT_IDS_BATCH = 256` 个 input_id 一次 DMA 到 UB(`inputIdsArr`),S 管线标量直接从 UB 读;窗口滑出后(`tokenIdx >= baseInputIdsIdx + 256`)再拉下一批,UB 占用恒定(常量定义 `csrc/kernels/sqrtsoftplus_hash_topk.h` 的 `HASH_INPUT_IDS_BATCH`)。

### 单 token 流水(7 阶段)

1. **`FetchInputs`**(`SqrtsoftplusHashTopK::FetchInputs`)= `PrefetchInputIds`(hash)+ `CopyIn`:
   - `CopyIn`(`SqrtsoftplusHashTopK::CopyIn`):得分行 DMA 到乒乓 `scoresInTmp[curr]`,V 管线接力转 fp32 到 `scoresIn`(bf16 用 `vconv_bf162f32`,fp32 用 `copy_ubuf_to_ubuf`,`SqrtsoftplusHashTopK::CopyIn` 内 scores relay 段);hash 模式下 S 管线读 `inputId = inputIdsArr[tokenIdx - baseInputIdsIdx]`,计算 `tid2eidGm + inputId * topK` 行地址,经 S→MTE2 EVENT_ID1 地址栅栏后 DMA 该行(长度 topK,不保证 32B 对齐故走 `CopyGmToUbufAligned` 的 b16 回退,`SqrtsoftplusHashTopK::CopyIn` 内 tid2eid DMA 段)到乒乓 `tid2eidRow[curr]`,再由 V 拷入 `indicesTopK` 作为"外部给定的选路索引";
   - 三条独立 flag 轨道:scores 用 MTE2↔V ID0+curr、tid2eid 用 MTE2↔V ID2+curr、地址就绪用 S→MTE2 ID1(`SqrtsoftplusHashTopK::CopyIn` 顶部注释,关键字 `Flag tracks`);
2. **`CalcScores`**(`SqrtsoftplusHashTopK::CalcScores`):数值稳定的 softplus —— `vabs`→`vmuls(-1)`→`vexp`→`vadds(1)`→`vln` 得 `log(1+exp(-|x|))`,`vrelu` 得 `max(x,0)`,相加后 `vsqrt` 得原始得分 `calc_unbiased`;有 bias 时 `vadd` 得选路得分 `calc`;
3. **`SelectTopK`**(仅非 hash,`SqrtsoftplusHashTopK::SelectTopK`):与 `sigmoid_topk` 相同的"vbitsort 块排序 + vmrgsort4 四路归并 + 逐块二路归并"多块循环,`vreducev2` 抽出 topK 索引 `indicesTopK`。排序宽度 `nSortScores = max(numRoutedExperts, 4*SORT_BLOCK_SIZE)`(`SORT_BLOCK_SIZE=32`,定义于 `csrc/kernels/kernel_macro.h`):不足 4 块时 `tailLen = nBlocks - 4` 的 uint64 减法下溢、尾部归并循环走出边界,且归并会消费未初始化 lane 污染 topK,故补齐到 128。补齐段处理与 `softmax_topk` 同构:`InitSortTail`(`SqrtsoftplusHashTopK::InitSortTail`,每 Run 一次)把 `indicesIn` 补齐 lane 索引填 0 并以 `PIPE_ALL` 栅栏保序——索引 0 与真实专家重复但其得分恒为 FLOAT_MIN 排在最后,既抢不走 topK 名额,也不会把越界索引喂给 `GatherWeights` 的 `vgather`;`PadCalcTail`(`SqrtsoftplusHashTopK::PadCalcTail`,每 token)把 `calc` 补齐 lane 重钉为 `FLOAT_MIN`(`CalcScores` 按 `DIV_ROUND_UP(numRoutedExperts,64)` 个 repeat 重写 `calc`,专家数非 64 整数倍时写宽覆写补齐段,须逐 token 重钉;取有限的 `FLOAT_MIN` 而非 -inf 是为保持 vbitsort 位序良定义,sqrtsoftplus+bias 的值域远高于它),两侧 `PIPE_ALL` 栅栏将 S 标量写保序于在飞 V 读写。`numRoutedExperts >= 128` 时两函数早退、`tailLen` 与旧版完全一致(N=256 时为 4,N=384 时为 8);低于 128 时 `tailLen` 钳到 0、尾部循环整体跳过(4 路归并已覆盖全部补齐宽度);
4. **`GatherWeights`**(`SqrtsoftplusHashTopK::GatherWeights`):`vmuls(idx*4)` 索引转字节偏移(注意注释:此处标量必须传裸 int `4`,不能是 float),`vgather` 从 `calc_unbiased`(pre-bias)收集 topK 权重 —— hash 与非 hash 两条路径在此汇合(hash 的索引来自 tid2eid,权重同样查 pre-bias 值);
5. **`NormalizeAndScale`**(`SqrtsoftplusHashTopK::NormalizeAndScale`):`ReduceSum`+`vbrcb`+`vdiv`(无条件归一化)+ `vmuls(scale)`;
6. **`FillRoutingMap`**(`SqrtsoftplusHashTopK::FillRoutingMap`):`vector_dup` 清零稀疏权重行 `weightsSparse[N]` 与位图 `routingMap[N/32]`,S 管线标量循环 topK 次 `bitmapSet` + 稀疏 scatter;
7. **`StageOut`/`CopyOut`**(`SqrtsoftplusHashTopK::StageOut` 与 `CopyOut`):V 管线把结果接力到乒乓输出槽 `weightsOut[curr]`(bf16 转换 / fp32 拷贝)与 `routingMapOut[curr]`,MTE3 异步 DMA 到 GM —— 计算与搬出乒乓重叠。

### UB 内存布局

`Init`(`SqrtsoftplusHashTopK::Init`)精心划分了四类区域(头注释 `SqrtsoftplusHashTopK::Init` 内分区域注释):

- **输入 DMA 乒乓**:`scoresInTmp[2]`(Dtype)、`tid2eidRow[2]`(topK 个 u32);
- **常驻**:`scoresIn`(V 私有 fp32)、`biasIn`、`indicesIn`、`inputIdsArr`(256×u32);
- **计算区(单份,V/S 私有)**:`calc`、`calc_unbiased`、`reduceTmp`、`sortTmp`(2 倍)、`sortMrgTmp`(2 倍)、`weightsTopK`、`indicesTopK`、`weightsSparse`、`routingMap`。其中被排序触碰的 `indicesIn`(常驻区)与 `calc`、`sortTmp`、`sortMrgTmp` 按补齐宽度定尺(`padSortN = max(padN, ROUND_UP(nSortScores,64)*4)`,`SqrtsoftplusHashTopK::Init` 内关键字 `padSortN`);`numRoutedExperts >= 128` 时 `padSortN == padN`,UB 布局与旧版逐字节一致;
- **输出 DMA 乒乓**:`weightsOut[2]`(Dtype)、`routingMapOut[2]`。

topK 相关缓冲按 256B 向量块对齐(`padK`,`SqrtsoftplusHashTopK::Init` 内 `padK` 注释:ReduceSum/vbrcb/vreducev2 假定块对齐)。

### 流水线同步

全算子以计数事件构成多轨软件流水(头注释 `SqrtsoftplusHashTopK::Run` 内 flag 轨道注释):scores 轨道 V↔MTE2 ID0+curr、输出轨道 V↔MTE3 ID0+curr、tid2eid 轨道 V↔MTE2 ID2+curr,`Run` 开头"预置"全部 slot-free flag 使首 token 不阻塞(`SqrtsoftplusHashTopK::Run` 内 `Pre-arm` 预置段),循环结束后 drain 最后一次 set,防止计数 flag 泄漏到下一次 launch(`SqrtsoftplusHashTopK::Run` 末尾 drain 段)。V→S 的 EVENT_ID0 用于 S 管线读 `indicesTopK` 前的栅栏(`NormalizeAndScale` 末尾 `set_flag(PIPE_V, PIPE_S, EVENT_ID0)` 与 `FillRoutingMap` 开头 `wait_flag`);S→MTE2 的 EVENT_ID1 用于 tid2eid 行地址可见性(`SqrtsoftplusHashTopK::CopyIn` 内 `addr ready` 栅栏)。

### 片上缓冲分配与占用汇总(AIV UB)

`Init`(`csrc/kernels/sqrtsoftplus_hash_topk.h` 的 `SqrtsoftplusHashTopK::Init`)自偏移 0 起在 UB 上按"输入 DMA 乒乓 → 常驻 → 计算区 → 输出 DMA 乒乓"四类区域顺序排布:行缓冲以 `padN = ROUND_UP(numRoutedExperts, VECTOR_MAX_NUM_OF_FP32)*4` 定尺,被排序触碰的 `indicesIn`/`calc` 及 `sortTmp`/`sortMrgTmp` 按补齐宽度 `padSortN = MAX(padN, ROUND_UP(nSortScores,64)*4)` 定尺,topK 相关槽位按 256B 向量块对齐(`padK = ROUND_UP(topK,64)*4`),hash 预取窗口 `inputIdsArr` 固定 `ROUND_UP(HASH_INPUT_IDS_BATCH*4, VECTOR_MAX_BYTESIZE)`;计算全程 fp32。硬件容量基准(910B/dav C220,NpuArch 2201):UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每个 AIV subblock 各持一份独立 UB、布局相同。dtype 记法:`Dtype` 为 scores/outWeights 元素类型(fp32/bf16),`padNDtype = ROUND_UP(numRoutedExperts, VECTOR_MAX_BYTESIZE/sizeof(Dtype))*sizeof(Dtype)`;hash 与非 hash 共用同一布局(排序缓冲在 hash 路径闲置),`numTokens` 不参与定尺。典型配置:numRoutedExperts=256、topK=6、scale=1.5(V4 门控,`nSortScores=256`、`padSortN == padN`);两条 dtype 路径同取该配置,占用差异仅来自 `padNDtype` 定尺的 4 个 Dtype 槽。

**AIV UB(预算 `UB_SIZE=192KB`,每 subblock 一份)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | fp32 占用(B) | bf16 占用(B) |
|---|---|---|---|---|---|---|
| `scoresInTmp[2]` | 256 Dtype(得分行 DMA) | Dtype | 2(scores 轨道,ID0+curr) | `padNDtype` | 2048 | 1024 |
| `scoresIn` | 256 float(V 私有 fp32) | float | 1 | `padN` | 1024 | 1024 |
| `biasIn` | 256 float | float | 1 | `padN` | 1024 | 1024 |
| `indicesIn` | 256 uint32 | uint32 | 1 | `padSortN` | 1024 | 1024 |
| `tid2eidRow[2]` | 6 uint32(按 padK 槽位) | uint32 | 2(tid2eid 轨道,ID2+curr) | `padK` | 512 | 512 |
| `inputIdsArr` | 256 uint32(hash 预取窗口) | uint32 | 1 | `ROUND_UP(HASH_INPUT_IDS_BATCH*4, VECTOR_MAX_BYTESIZE)` | 1024 | 1024 |
| `calc` | 256 float | float | 1 | `padSortN` | 1024 | 1024 |
| `calc_unbiased` | 256 float | float | 1 | `padN` | 1024 | 1024 |
| `reduceTmp` | 64 float 槽位 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 | 256 |
| `sortTmp` | 512 float(排序工作区) | float | 1 | `2*padSortN` | 2048 | 2048 |
| `sortMrgTmp` | 512 float(排序工作区) | float | 1 | `2*padSortN` | 2048 | 2048 |
| `weightsTopK` | 64 float 槽位(topK=6 载荷) | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 | 256 |
| `indicesTopK` | 64 uint32 槽位(topK=6 载荷) | uint32 | 1 | `VECTOR_MAX_BYTESIZE` | 256 | 256 |
| `weightsSparse` | 256 float | float | 1 | `padN` | 1024 | 1024 |
| `routingMap` | 8 uint32(位图,按 padN 定尺) | uint32 | 1 | `padN` | 1024 | 1024 |
| `weightsOut[2]` | 256 Dtype(输出乒乓) | Dtype | 2(输出轨道,ID0+curr) | `padNDtype` | 2048 | 1024 |
| `routingMapOut[2]` | 8 uint32(输出乒乓,按 padN 定尺) | uint32 | 2(输出轨道,ID0+curr) | `padN` | 2048 | 2048 |
| **合计** | | | | | **19712** | **17664** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | fp32 路径占用 | 占比 | bf16 路径占用 | 占比 |
|---|---|---|---|---|---|
| UB(每 AIV subblock) | 192 KB | 19712 B | 10.0% | 17664 B | 9.0% |

要点:

- **fp32 路径占用反而高于 bf16**:`padNDtype` 定尺的 `scoresInTmp[2]`/`weightsOut[2]` 四个 Dtype 槽在 fp32 时各 1024 B、bf16 减半到 512 B,这是两条路径唯一的占用差异(其余槽位全为 fp32);
- 排序工作区 `sortTmp`/`sortMrgTmp`(各 `2*padSortN`,合计 4096 B)与 `padN`/`padSortN` 定尺的行缓冲共占主体;hash 路径不排序但共用同一布局,`inputIdsArr` 预取窗口恒为 1024 B(与 numTokens 无关);
- `numRoutedExperts >= 128` 时 `padSortN == padN`(N=256 即是);低于 128(GLM-4.7-Flash 的 64/96)时补齐宽度 `ROUND_UP(128,64)*4 = 512` 接管定尺;
- 无 kernel 断言,host(`csrc/op.cpp` 的 `XliteOpSqrtsoftplusHashTopK`)只校验 dtype/BIT1;布局随 `numRoutedExperts` 线性缩放,测试最大 384 专家时 fp32 占用亦仅 28416 B(14.5%),容量风险不存在。
