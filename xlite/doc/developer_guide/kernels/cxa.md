# cxa

## 功能概述

cxa 是 DeepSeek-V4 使用的统一注意力算子(C4A and C128A attention,即压缩比 4 与压缩比 128 的压缩注意力)。它将**滑动窗口段(SWA)**与**压缩稀疏段**拼接在同一行分数上,**在同一个 softmax 中联合计算**,并把每头的 `attn_sink` 偏置并入 softmax 分母:

- **滑动窗口段(SWA)**:query 位于绝对位置 `q` 时,只能看到窗口 `[max(0, q - window_size + 1), q]` 内的 SWA token;超出当前已生成长度的位置 mask 为 `-inf`。SWA cache 始终分页,两种模式均经 `swaBlockTables` 查页。
- **压缩稀疏段**:对压缩 KV(每 `compress_ratio` 个 token 压缩成 1 个),仅保留 `topk_indices` 引用的位置;`topk_indices` 中为 `-1` 的项 mask 为 `-inf`(其 exp 贡献为 0)。
- **attn_sink**:每头一个可学习的 sink 偏置(fp32),`exp(attn_sink - row_max)` 被加进 softmax 分母(见 `csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `*sink = attnSink[headIdx]`))。

两种压缩 cache 布局(`dense` 标志区分):

- **sparse 模式**(`dense=false`,默认):压缩 cache 分页,经 `compressBlockTables` 查页,可选 top-k 选路;
- **dense 模式**(`dense=true`):压缩 cache 为每 batch 连续布局,batch `b` 占用压缩 token `[b * indexTopK, (b + 1) * indexTopK)`(`maxSeqLen == indexTopK`),压缩段按因果连续参与 softmax(不做 top-k mask,`topkIndices` 被忽略),由 binding 侧先调 `gather_sparse_kv_cache` 收集 dense cache 再进入本算子;`compressBlockTables` 不使用。SWA cache 在两种模式下都保持分页。

退化形态:

- `window_size = 0`:纯压缩注意力(无 SWA 段,`swaSegWidth = 0`,scores workspace 仅含压缩段);
- `compress_ratio = 0`(或 `index_topk = 0`):纯滑窗 / 滑窗+全量压缩注意力(无 topk 选路,压缩段全保留)。

与 Python 参考实现的对应关系:tests/kernels/cxa.py 将本算子与 tests/models/deepseek_v4_kernel.py 中的 `sparse_attn`(对拼接后的 `[window_size + compressed_kv_len]` KV 矩阵做一次 gather+softmax)逐 case 对比;dense 模式下两侧复用同一份采样的 topk_indices,SWA 路径不变。

## 输入输出参数

Python 签名见 `xlite/_C.pyi` 的 `def cxa(...)`,kernel 侧入口见 `csrc/kernels/cxa.h` 的 `CXA_FUNC_DEFINE`(关键字 `cxa_##dtype`)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | [totalQ, nHeads, headDim] | bfloat16 | query |
| swa_k_cache | 输入 | [swaBlockNum, swaBlockSize, headDim] | bfloat16 | 分页滑动窗口 KV cache(两种模式均分页);支持 dim-0 跨度(内核按 `swaKCacheStride = MAX(stride[0], swaBlockSize*headDim)` 寻址) |
| compress_k_cache | 输入 | [compressBlockNum, compressBlockSize, headDim] | bfloat16 | 分页压缩 KV cache;sparse 模式下分页,`compress_ratio == 0` 时不使用(可为空);dense 模式下为已 gather 的连续 cache `[batch, indexTopK, headDim]`;sparse 模式支持 dim-0 跨度(`compressKCacheStride = MAX(stride[0], compressBlockSize*headDim)`) |
| swa_block_tables | 输入 | [rows >= batch, swaBlockTableStride](1-D 表视作单行) | int32 | SWA cache 的 block table;持久化缓冲,行跨度为对应表 `stride[0]`(分配跨度),持久化 V2 表的 `shape[1]` 已由 PrepareAttn 收窄到运行期宽度,仅每行 lens 推导的有效前缀被读取 |
| compress_block_tables | 输入 | [rows >= batch, compressBlockTableStride](1-D 表视作单行) | int32 | 压缩 cache 的 block table;持久化缓冲,行跨度/行宽语义同上;dense 模式不使用(可为空) |
| swa_block_size | 输入 | 标量 | - | SWA cache block 大小 |
| compress_block_size | 输入 | 标量 | - | 压缩 cache block 大小 |
| attn_sink | 输入 | [nHeads] | float32 | 每头注意力 sink 偏置 |
| output | 输出 | [totalQ, nHeads, headDim] | bfloat16 | 注意力输出,原位写入 |
| batch | 输入 | 标量 | - | batch 大小 |
| query_start_loc | 输入 | [batch] | int32 | query 长度前缀和 |
| lens | 输入 | [batch] | int32 | 每 batch 当前 query 长度 |
| cached_lens | 输入 | [batch] | int32 | 每 batch 已缓存 token 长度 |
| n_heads | 输入 | 标量 | - | 本地 query 头数 |
| head_dim | 输入 | 标量 | - | 头维度 |
| scale | 输入 | 标量 | - | softmax 缩放因子(`1 / sqrt(head_dim)`) |
| window_size | 输入 | 标量 | - | 滑窗宽度(SWA 段语义宽度) |
| compress_ratio | 输入 | 标量 | - | 压缩比;`0` 表示禁用压缩段(纯滑窗) |
| index_topk | 输入 | 标量 | - | 每行 top-k 压缩索引数;`0` 表示禁用 topk 稀疏路径;dense 模式下即 dense cache 长度 `maxSeqLen`(host 要求 > 0) |
| topk_indices | 输入 | [totalQ, indexTopK] | int32 | 压缩段 top-k 索引(IndexerTopK 输出);`-1` 表示 mask 掉的位置;dense 模式忽略(可为空) |
| dense | 输入 | 标量 | - | 压缩 cache 布局:`false` 分页,`true` 连续 |
| enable_flash_attention | 输入 | 标量 | bool | 置 `True` 时 host 路由到 flash_cxa(在线 softmax,见 [flash_cxa.md](flash_cxa.md));本算子恒为 `False` |
| tile_size_of_cached_kv | 输入 | 标量 | uint32 | flash_cxa 的 KV tile 宽度(默认 8192);非 flash 路径不使用 |
| zeroGm | host | GM | - | host 自动提供的零缓冲(`rt._attnZeroBuffer`),非 Python API 参数;`CxaAicHelper::Init` 用它预填零 L1B(见下文"支持的数据类型"末段) |

另有内部 scores workspace(由 binding 侧分配):

- **scores**:`[aicNum * XLITE_MAX_M0 * 2, swaSegWidth + kvSize]`,dtype 同 q;
    - `swaSegWidth = windowSize == 0 ? 0 : windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B`(`csrc/_C.cpp` 的 `CXA`(关键字 `swaSegWidth`)、`csrc/kernels/cxa.h` 的 `CXA::Init`(关键字 `swaSegWidth`)):一个 query tile 的因果窗并集宽 `windowSize + XLITE_MAX_M0`,再加 `K_BLOCK_SIZE_2B`(=16,`csrc/kernels/kernel_param.h` 的 `K_BLOCK_SIZE_2B` 宏)吸收 `windowStart` 向下取整产生的前导列;
    - `kvSize`:压缩段宽度。dense 模式取 `ROUND_UP(indexTopK, 4 * CXA_SVCK0)`(`csrc/_C.cpp` 的 `CXA`(关键字 `ROUND_UP((uint64_t)indexTopK, 4 * CXA_SVCK0)`)):单 batch 连续 cache 仅含 `indexTopK` 个压缩 token,与 batch 数无关;sparse 模型层取 `ROUND_UP(rt._maxTotalLens / compressRatio, 4 * CXA_SVCK0)`(由 lens 推导);standalone 入口取 `ROUND_UP(DeriveMaxSeqLen / compressRatio, 4 * CXA_SVCK0)`(由 lens 推导,同模型层),均按 `4 * CXA_SVCK0` 对齐(见下文 scores 布局)。

## 支持的数据类型

仅 **bfloat16_t**:`csrc/kernels/cxa_bfloat16_t.cpp` 只实例化 `cxa_bfloat16_t`,host 侧 `XliteOpCXA`(`csrc/op.cpp` 的 `XliteOpCXA`)也仅接受 BF16 分支,否则抛异常。

调度类型为 `KERNEL_TYPE_MIX_AIC_1_2`(`csrc/kernels/cxa.h` 的 `CXA::Init`(关键字 `KERNEL_TYPE_MIX_AIC_1_2`)):1 个 AIC(cube 核)配 2 个 AIV(vector 核)混合调度,AIC 负责 QK/SV 两个 cube GEMM,AIV 负责 softmax;AIC 与 AIV 之间用 `ffts_cross_core_sync` / `wait_flag_dev`(inner-group 同步,flag 0 = AIC→AIV 的 QK 完成,flag 1 = AIV→AIC 的 softmax 完成)做跨核握手。

host 侧约束(`csrc/op.cpp` 的 `XliteOpCXA`(关键字 `dense=True requires` 约束)):dense 模式要求 `compressRatio > 0` 且 `indexTopK > 0`(dense cache 长度);`windowSize > 0` 时要求 `swaSegWidth = windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B ≤ MAX_SWA_SEG_WIDTH(4096)`(关键字 `swaSegWidth > MAX_SWA_SEG_WIDTH`):topK>0 路径的 PingPong softmax 需要宽 `topK + swaSegWidth` 的 `inTopK` 缓冲(见 softmax.md 的 UB 预算)。topK>0 的行宽合同 `outN = swaSegWidth + 逐行压缩宽 ≤ MAX_SOFTMAX_PINGPONG_LEN(10496)` 由 Python 层保证(`tests/kernels/cxa.py` 的 max_row 检查,超界路由 flash),host 无法校验(逐行、依赖 lens);topK==0 行走行宽容忍的 `RunAivSoftmaxLong`。host launch 末尾把 `rt._attnZeroBuffer.ptr` 作为 `zeroGm` 传给 kernel(`CXA_FUNC_DEFINE` 签名最后一参),`CxaAicHelper::Init` 用它把两份 L1B ping-pong 缓冲预填零,避免上一 tile 残留 K 数据导致 mmad NaN(与 attention 同源 nan bug 修复)。

## 实现原理

### scores workspace 布局

每行分数的 stride 为 `qkStride = swaSegWidth + kvSize`(`csrc/kernels/cxa.h` 的 `CXA::Init`(关键字 `qkStride = swaSegWidth + kvSize`)),一段连续内存中两个段前后拼接:

```text
列 0                                              swaSegWidth        qkStride
|<----------- SWA 段(含 lead-in) ----------->|<------ 压缩段 ------>|
   [lead-in | 因果可见窗 [windowStart, windowEnd]]
```

- **SWA 段**(列 0 起):`RunAicQK` 把窗起点 `windowStart` **向下取整到 `kBlockSize`(32/sizeof(Dtype)=16)倍数**得到 `alignStart`,scores 列 0 对应绝对位置 `alignStart`(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`(关键字 `alignStart = ROUND_DOWN(windowStart, kBlockSize)`))。这样保证每个 tile 的 nSize 都是 `kBlockSize` 的倍数,没有列跨 tile 边界,SV 路径不会重复统计边界 token;lead-in 列 `[0, windowStart - alignStart)` 由 `RunAivSoftmax` mask 成 `-inf`。lead-in 宽度恒小于 `K_BLOCK_SIZE_2B = 16`,这就是 `swaSegWidth` 中 `+ K_BLOCK_SIZE_2B` 的来源。
- **压缩段**(列 `swaSegWidth` 起,无 SWA 段时从列 0 起):`RunAicQK` 把压缩段分数写到 `scores[swaSegWidthEff + nOffset]`(`swaSegWidthEff = hasSwa ? swaSegWidth : 0`,`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`(关键字 `swaSegWidthEff = hasSwa ? (int)swaSegWidth : 0`)),按 `qkcn0`(dense 模式 `MAX_N0`=128,否则 `compressBlockSize`)分 tile;host 侧把 `kvSize` pad 到 `4 * CXA_SVCK0` 的倍数,因为 `RunAicSV` 每次读 `4 * svck0` 个压缩段分数(`csrc/kernels/kernel_param.h` 的 `CXA_SVCK0` 宏注释、`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicSV`(关键字 `4 * svck0`))。

scores workspace 整体被切成两块 ping-pong 缓冲(`PINGPONG_BUF_NUM = 2`):每个 AIC 核占用 `block_idx * XLITE_MAX_M0 * qkStride` 起的第一块,以及偏移 `block_num * XLITE_MAX_M0 * qkStride` 的第二块,交替用于相邻两个 tile(`csrc/kernels/cxa.h` 的 `CXA::Init`(关键字 `scores[0].SetGlobalBuffer`)),使 QK(本 tile)与 SV(上一 tile)能同时使用不同的缓冲。

### AIC 路径(RunAic)

入口 `csrc/kernels/cxa.h` 的 `CXA::RunAic`,cube 核执行,按 batch 遍历,批内任务按 `firstCore = (blockIdx + blockNum - coreOffset) % blockNum` 起始、步长 `blockNum` 轮转分发(`coreOffset` 跨 batch 累积,避免所有 batch 都从 0 号核开始):

1. `GetOptimalM0(queryLen, cachedLen)`(`csrc/kernels/kernel_macro.h` 的 `GetOptimalM0`)按序列长度选择 M0(短序列 16,totalLen ≤12K 取 128,随后随长度增大在 112/96/80/64/48/32 间逐级递减,超长序列降到 16);`queryTileSize = m0 / nHeads`(即一个 tile 内所有头一起算,`mSize = queryLen * nHeads` 行)。
2. 每个 query tile 调 `CxaAicHelper::RunAicQK` 计算本 tile 的 QK 并把分数写入 `scores[curr]`,随后 `ffts_cross_core_sync(PIPE_FIX, config)` 通知 AIV "QK 完成"(flag 0,mode 2 = 组内 AIC/AIV 同步,`csrc/kernels/cxa.h` 的 `CXA::RunAic`(关键字 `ffts_cross_core_sync(PIPE_FIX, config)`))。
3. 当前 tile 的 QK 与上一 tile 的 `RunAicSV` 流水重叠:若 `needDoSV`,先 `wait_flag_dev(1)` 等 AIV 完成上一 tile 的 softmax,再执行 `RunAicSV(scores[last], ...)`,用上一 tile 的分数做 softmax×V,把输出原位写到 `output[lastMhOffset]`(`csrc/kernels/cxa.h` 的 `CXA::RunAic`(关键字 `RunAicSV(scores[last]`))。`last*` 系列变量缓存上一 tile 的全部上下文(batch、GM 偏移、block table 指针、`batchCompressKCache`、calcLen 等)。
4. 循环结束后补做最后一个 tile 的 SV(`csrc/kernels/cxa.h` 的 `CXA::RunAic`(关键字 `do last softmax * V`))。

dense 模式下每 batch 取 compress cache 子视图 `compressKCache[batchIdx * maxSeqLen * headDim]`(`csrc/kernels/cxa.h` 的 `CXA::RunAic`(关键字 `batchCompressKCache = compressKCache[batchIdx * maxSeqLen`)),`compressBt = nullptr`、`compressBlockTableStride = 0`;sparse 模式按表行跨度取该 batch 的 block table 指针(`csrc/kernels/cxa.h` 的 `CXA::RunAic`(关键字 `compressBt =` ... `compressBlockTables`))。

### AIV 路径(RunAiv)

入口 `csrc/kernels/cxa.h` 的 `CXA::RunAiv`,vector 核执行,任务划分与 AIC 完全一致(同样的 batch 遍历、`GetOptimalM0`、轮转起点),保证 AIC/AIV 对同一 tile 的编号一一对应。每个 tile:

1. 拆分:`nWork = queryTaskLen * nHeads` 行 softmax 工作,均分给本 AIC 配对的 2 个 AIV 子核(`get_subblockid()`),每个子核处理 `nWorkPerCore` 行,起点偏移 `qkOffset = nWorkStart * qkStride`。
2. 计算 mask 相关量:
   - `calcSoftmaxLen = cachedLen + queryTaskStart + 1`:该 tile 第一行的因果可见长度(绝对坐标);
   - `winStart = max(0, calcSoftmaxLen - windowSize)`:第一行因果窗起点;`winCalcLen = calcSoftmaxLen - ROUND_DOWN(winStart, K_BLOCK_SIZE_2B)`(`csrc/kernels/cxa.h` 的 `CXA::RunAiv`(关键字 `winCalcLen`)):scores 列 0 对应 `alignStart`,故可见 SWA 列数是 `calcSoftmaxLen - alignStart`,窗滑动时比 `windowSize` 多出的部分就是 lead-in(<16);
   - `ncCalcLen = compressRatio == 0 ? 0 : calcLen / compressRatio`(`csrc/kernels/cxa.h` 的 `CXA::RunAiv`(关键字 `ncCalcLen`));`outN = swaSegWidth + ROUND_UP(ncCalcLen, 4 * svk0)`(上限 `qkStride`(`csrc/kernels/cxa.h` 的 `CXA::RunAiv`(关键字 `outN = qkStride`)),压缩比为 0 时无压缩段):本 tile 需要处理/回写的分数列数。dense 模式下 `ncCalcLen` 不再单独 clamp 到 `maxSeqLen`,而是靠 `outN` 被 `qkStride = swaSegWidth + ROUND_UP(maxSeqLen, 4 * svk0)` 兜底,把压缩段约束到 `maxSeqLen` 个 token。
3. `wait_flag_dev(0)` 等 AIC 的 QK 完成,然后:
   - **无 topk 路径**(`indexTopK == 0`,dense 模式恒走此路径):`RunAivSoftmax`(内部转 `RunAivSoftmaxLong`,`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmax`)直接在 scores 上做在线 softmax;`m0 > XLITE_MAX_M0 - 4` 时无 expBuf 暂存(传 nullptr)。
   - **topk 路径**:调 `RunAivSoftmaxPingPong`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`),传入 `topkIndices + indexTopK * queryTaskOffset`(`ncCalcLen > indexTopK` 时,否则视为无 topk)。压缩段分数不再整段参与 softmax,而是用 `vgather` 按 topk 索引从 GM 收集到 UB(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `CopyGmToUbufAligned(indices[curr], topkIndices`)),与 SWA 段拼成 `[swaSegWidth + topK]` 行做 softmax,再按 64-slot 粒度的 hit-mask scatter 回 scores workspace 对应压缩位置(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `CopyUbufToGmAligned(buf + idx * n`)),供 AIC 的 SV 读取。索引为 `-1` 的位置通过 `vcmpvs_ge`/`vcmpvs_lt` 边界比较(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `vcmpvs_ge` / `vcmpvs_lt`))被排除在 gather/scatter 之外,等效 mask 为 `-inf`。
4. softmax 完成后 `ffts_cross_core_sync(PIPE_MTE3, config)` 通知 AIC(flag 1),并翻转 ping-pong 缓冲 `curr`。

**窗口 causal mask**(`RunAivSoftmaxPingPong` 内):分数先转 fp32(`vconv_bf162f32`,`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `vconv_bf162f32`)),再对 SWA 段 `[0, swaSegWidth)` 做双向 mask,均用 `vector_dup` 写 `-3.4028235e38`(fp32 最小值,≈-inf):

- **右 mask**:每行可见窗宽 `actualWinLen = min(winCalcLen + seqIdx, swaSegWidth)`,将 `[actualWinLen, swaSegWidth)` 填 `-inf`。按 64 元素(VECTOR_MAX_BYTESIZE/4)的 repeat 对齐处理:区间落在单个 repeat 内用 `SetMaskRange(lo, hi)` 精确置位(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `SetMaskRange(lo, hi)`)),跨 repeat 则首块用 `SetMaskFromHighBit`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `SetMaskFromHighBit`))、尾块用 `SetMask(wtail)` 位级控制(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `SetMask(wtail)`))。
- **左 mask**:lead-in 列 `[0, lEnd)` 填 `-inf`,其中 `lEnd = actualWinLen > winSize ? actualWinLen - winSize : 0`,即窗起点之前的取整前导列,整 repeat 用普通 `vector_dup`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `vector_dup(cal, min, lFull`)),余数用 `SetMask(lRem)`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `SetMask(lRem)`))。
- 该行 causally 不可见(`actualCalcLen <= 0`)或 topk 全被 mask 且无 SWA 段时,直接清零输出并跳过计算。

**attn_sink 并入分母**(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `*sink = attnSink[headIdx]`)):在线 softmax 得到 row_max 后,从 GM 读 `attnSink[headIdx]` 到 UB(同处 `*sink = attnSink[headIdx]` 一行),与分数同样地做 `sink - max`(`vsub`,`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `vsub(sink, sink, temp`))→ `vexp`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `vexp(sink, sink`)),并把 `vexp(sink)` 加到 `ReduceSumV2` 的结果上(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `vadd(temp, temp, sink`)),即分母 `s = Σ exp(sᵢ - max) + exp(sink - max)`;最后 `vdiv` 归一化(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`(关键字 `vdiv(cal, cal, temp`))只作用于分数部分,sink 不进入输出。

`RunAivSoftmaxLong`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`)与 PingPong 版逻辑相同,区别在于 UB 布局:`Long` 版按 `MAX_SUB_CONTEXT_SIZE` 分子块支持超长行(无 attnSink 时上限 subBlockNum ≤ 127 / n ≤ 2064512,启用 attnSink 时 126 / 2048256,见 `RunAivSoftmaxLong` 上方注释);PingPong 版(topK>0 路径)用双缓冲 in/out 加速常规长度,行宽须 ≤ `MAX_SOFTMAX_PINGPONG_LEN`(10496,UB 预算见 softmax.md 与 `softmax_attn_aiv.h` 的 static_assert 块)。两版此前接收 `compressCap` 参数用于 dense 模式 clamp 压缩段因果长度,现已移除:dense 模式改由调用方 `outN` 被 `qkStride` 兜底(见 AIV 路径第 2 步)约束压缩段到 `maxSeqLen`。

### CxaAicHelper(QK / SV cube 计算)

`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper` 类,封装 AIC 的两个 cube GEMM 及 L1/L0 缓冲管理。

**tiling 常量**(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::Init`):

| 常量 | 含义 | 取值 |
|---|---|---|
| `qkwn0` | QK 的 SWA 段 n 方向 tile | `swaBlockSize` |
| `qkcn0` | QK 的压缩段 n 方向 tile | dense 模式为 `MAX_N0`(128),否则 `compressBlockSize` |
| `qkk0` | QK 的 k 方向 tile | `256 / sizeof(Dtype)`(bf16 为 128) |
| `svn0` | SV 的 n 方向 tile(headDim 方向) | 256 |
| `svwk0` | SV 的 SWA 段 k 方向 tile | `swaBlockSize` |
| `svck0` | SV 的压缩段 k 方向 tile | `CXA_SVCK0 = 64`(`csrc/kernels/kernel_param.h` 的 `CXA_SVCK0` 宏) |

**L1(A1/A2)/L0 缓冲布局**(`Init` 中手动排布 bufferAddr):QK 部分有 `ql1aBuf`(A1,`XLITE_MAX_M0 * headDim`)、双缓冲 `kl1bBuf`(A1,`(max(qkwn0,qkcn0) * 4 * qkk0)`,按 `kIdx4 == 0` 时一次搬 4 个 k-tile)、双缓冲 `qkl0aBuf`/`qkl0bBuf`(A2/B2,A2 为 `XLITE_MAX_M0 * qkk0`、B2 为 `MAX_N0 * qkk0`,两值数值相等但分属 M/N 维)、单缓冲 `qkl0cBuf`(CO1,`XLITE_MAX_M0 * MAX_N0 * float`);SV 部分有双缓冲 `scoresl1aBuf`(A1,分数源,大小按 `max(svwk0, svck0)` 的 4 倍留)、双缓冲 `ktl1bBuf`(A1,K^T,SWA 段 `svwk0` / 压缩段一次搬 `2 * svck0`)、双缓冲 `svl0aBuf`/`svl0bBuf`(A2/B2)、单缓冲 `svl0cBuf`(CO1,`XLITE_MAX_M0 * svn0 * float`)。所有双缓冲通过 `HardEvent`(MTE1_MTE2 / MTE2_MTE1 / M_MTE1 / MTE1_M / M_FIX / FIX_M)的手工 Set/WaitFlag 做 GM→L1→L0→MMAD→L0C→GM 的逐级流水。

**RunAicQK**(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`,`scores = Q * K`,签名新增 `bool hasSwa`):Q 一次整拷进 L1;SWA 段仅当 `hasSwa && windowSize != 0` 时沿 `nwIdx` 遍历窗内 block(swaBlockTables 查页),否则跳过整个 SWA 段(无 SWA 时该 tile 不写 SWA 分数);压缩段沿 `nIdx` 遍历压缩 block(dense 模式 `block = nIdx + nIdxStart` 连续布局不查页,sparse 模式 `compressBlockTables[nIdx + nIdxStart]` 查页,`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`(关键字 `compressBlockTables[nIdx + nIdxStart]`)),每 tile 内沿 headDim 按 `qkk0` 分 k 循环,`kIdx4`(每 4 个 k-tile)触发 ping-pong 换 L1B 缓冲。`CalMmad` 累加到 L0C 后 `CopyToGm` 写回 scores 对应列偏移:列偏移用 `swaSegWidthEff = hasSwa ? swaSegWidth : 0`(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`(关键字 `swaSegWidthEff = hasSwa ? (int)swaSegWidth : 0`)),即无 SWA 段时压缩段分数从列 0 起写,避免 SWA 段预留空白列。压缩段长度 `ncTotalLen = kvLen / compressRatio`,dense 模式 clamp 到 `compressTotalLen`(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`(关键字 `ncTotalLen = kvLen / compressRatio`));压缩段 nIdx 起点按 `nIdxStart = (kvOffset / compressRatio) / qkcn0`(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`(关键字 `nIdxStart = (kvOffset / compressRatio) / qkcn0`))换算到压缩坐标。

**RunAicSV**(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicSV`,`output = softmax(scores) * K^T`,签名新增 `bool hasSwa`):n 方向沿 headDim 按 `svn0 = 256` 分 tile;k 方向先遍历 SWA 段(仅 `hasSwa && windowSize != 0` 时,分数 tile 与 K^T tile 都从 `alignStart` 起对齐,保证 `[kOffset, kOffset + kBlockPad)` 区间两两不相交),再遍历压缩段:分数按 `4 * svck0` 粒度搬入 L1(`kIdx4`,源列偏移同样用 `swaSegWidthEff`,`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicSV`(关键字 `scores[swaSegWidthEff + kOffset]`)),K^T 按 `2 * svck0` 粒度搬入 L1(`kIdx2`);压缩段 K^T 在 dense 模式下连续单次拷贝(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicSV`(关键字 `dense cache: contiguous layout, single copy`)),sparse 模式按 block table 逐 block 多块拷贝(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicSV`(关键字 `paged cache: per-block lookup, multi-block copy`))。若 `(!hasSwa || windowSize == 0) && kcTotalLen == 0` 直接返回(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicSV`(关键字 `(!hasSwa || windowSize == 0) && kcTotalLen == 0`))。所有 tile 的 mmad 累加到同一 `svl0cBuf`(`init` 标志控制首 tile 清零),每轮 n-tile 结束后 `CopyToGm` 写 `output[nOffset]`。

`hasSwa` 由调用方传入:cxa(非 flash)始终 `hasSwa = windowSize != 0`(`csrc/kernels/cxa.h` 的 RunAicQK/SV 调用);flash_cxa 仅在第一个 KV tile(`kvIdx == 0`)传 `true`,其余 tile 传 `false`(SWA 段只属于第一个 tile)。`dense` 成员(`csrc/kernels/cxa.h` 的 `CXA::Init`(关键字 `this->dense = dense != 0`) 由 host `dense` 参数设置)区分 paged/dense cache 布局,dense 时压缩段 block table 不参与寻址、AIC 侧压缩段长度 clamp 到 `maxSeqLen`(`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::RunAicQK`(关键字 `ncTotalLen = compressTotalLen`)与 `CxaAicHelper::RunAicSV`(关键字 `kcTotalLen = compressTotalLen`))。

### 校验

tests/kernels/cxa.py 覆盖 5 种模型配置(base-swa、swa、c128a、c4a-ntok、c4a-topk)× 多种 batch/长度组合(含 131071 超长 cached_len、多 batch 混合长度),与 `sparse_attn` 参考实现对比容差 `atol=5e-5, rtol=5e-2`;每条 case 同时覆盖 sparse 与 dense 两种模式(dense 模式要求 query_len == 1 per batch,两侧复用同一份采样的 topk_indices),topk 场景中参考实现的绝对压缩索引被换算到 op 的压缩相对坐标(压缩 token j -> 索引 j,kernel 将其置于分数列 `swaSegWidth + j`)。

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C、AIV UB)

`csrc/kernels/cxa_aic_helper.h` 的 `CxaAicHelper::Init` 手动排布 AIC 侧全部片上缓冲:每级在 `TPosition::A1/A2/B2/CO1` 上累进 `bufferAddr`,且 **QK 段与 SV 段在同级的偏移各自从 0 重新起排**——`RunAic` 中 QK 与 SV 是串行阶段(`RunAicQK` 完全结束、FIX 把分数写回 scores workspace 之后才 `wait_flag_dev(1)` 进入 `RunAicSV`),两段物理上互斥复用同一块地址空间,该级实际占用取两段布局的较大者(下文各表按段分列,汇总按 max 聚合)。AIV 侧 `RunAivSoftmaxLong` / `RunAivSoftmaxPingPong`(`csrc/kernels/softmax_attn_aiv.h`)为函数内联排布:`off` 从 0 顺序累进,每槽按 `VECTOR_MAX_BYTESIZE=256B` 向上对齐。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L1 512KB、L0A 64KB、L0B 64KB,L0C 总容量 256KB(`l0_c_size=262144`)系 CO1/CO2 两区之和,mmad 累加(`TPosition::CO1`)可用的 CO1 区为 **128KB**,下文 L0C 占用均按 CO1 128KB 计;AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`),每个 AIV subblock 各持一份独立 UB。

dtype 记法:`CXA<Dtype>` 仅以 bfloat16_t 实例化(`csrc/kernels/cxa_bfloat16_t.cpp`),`Dtype`=bf16(2B),L0C mmad 累加器为 float(4B);AIV UB 布局随 `Dtype` 与运行期行宽缩放,单 dtype 无需分列。典型配置取 `tests/kernels/cxa.py` 的 c4a-topk 模型(nHeads=16、headDim=512、windowSize=128,即 `swaSegWidth = windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B = 272`、compressRatio=4、indexTopK=512、swaBlockSize=64、compressBlockSize=128)与 decode 工作负载 `(1, [4096], [1])`(sparse 由 lens 推导 `kvSize = ROUND_UP(4097/4, 4*CXA_SVCK0) = 1024`,`qkStride = 272+1024 = 1296`);tiling 常量为 `qkwn0=swaBlockSize=64`、`qkcn0=dense?MAX_N0:compressBlockSize=128`(dense 模式 host 传 `compressBlockSize=0`,helper 内改取 `MAX_N0=128`,与 sparse 取值一致)、`qkk0=256/sizeof(Dtype)=128`、`svn0=(swaBlockSize<=CXA_SVCK0)?256:128=256`、`svwk0=64`、`svck0=CXA_SVCK0=64`、`svwkMax=max(svwk0,svck0)=64`。下表 shape 为逻辑形状(L1/L0 内实际按 Nd2Nz/NZ 分形排布),占用 = 单份大小 × 乒乓个数。

**AIC L1(A1,容量 512KB)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `ql1aBuf` | `(XLITE_MAX_M0, headDim)`=(128, 512) | bf16 | 1(事件0) | `XLITE_MAX_M0*headDim*sizeof(Dtype)` | 131072 |
| `kl1bBuf` | `(max(qkwn0,qkcn0), 4*qkk0)`=(128, 512) | bf16 | 2(事件2/3) | `max(qkwn0,qkcn0)*4*qkk0*sizeof(Dtype)` | 262144 |
| **合计** | | | | | **393216** |

**AIC L1(A1)——SV 段(与 QK 段互斥复用)**

`Init` 尾部(`__DAV_C220_CUBE__` 分支)用 host 提供的 `zeroGm` 把两份 `ktl1bBuf` 按 `kBlockPad×svn0` 预填零,避免上一 tile 残留 K 数据导致 mmad NaN。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `scoresl1aBuf` | `(XLITE_MAX_M0, 4*svwkMax)`=(128, 256) | bf16 | 2(事件0/1) | `XLITE_MAX_M0*4*svwkMax*sizeof(Dtype)` | 131072 |
| `ktl1bBuf` | `(svn0, max(svwk0,2*svck0))`=(256, 128) | bf16 | 2(事件2/3) | `max(svwk0,2*svck0)*svn0*sizeof(Dtype)` | 131072 |
| **合计** | | | | | **262144** |

**AIC L0A(A2,容量 64KB)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `qkl0aBuf` | `(XLITE_MAX_M0, qkk0)`=(128, 128) | bf16 | 2(事件0/1) | `XLITE_MAX_M0*qkk0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0A(A2)——SV 段(与 QK 段互斥复用)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svl0aBuf` | `(XLITE_MAX_M0, svwkMax)`=(128, 64) | bf16 | 2(事件0/1) | `XLITE_MAX_M0*svwkMax*sizeof(Dtype)` | 32768 |
| **合计** | | | | | **32768** |

**AIC L0B(B2,容量 64KB)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `qkl0bBuf` | `(MAX_N0, qkk0)`=(128, 128) | bf16 | 2(事件0/1) | `MAX_N0*qkk0*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0B(B2)——SV 段(与 QK 段互斥复用)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svl0bBuf` | `(svn0, svwkMax)`=(256, 64) | bf16 | 2(事件0/1) | `svn0*svwkMax*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB;总容量 256KB 含 CO2)——QK 段**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `qkl0cBuf` | `(XLITE_MAX_M0, MAX_N0)`=(128, 128) | float | 1(事件0) | `XLITE_MAX_M0*MAX_N0*sizeof(float)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区)——SV 段(与 QK 段互斥复用)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `svl0cBuf` | `(XLITE_MAX_M0, svn0)`=(128, 256) | float | 1(事件0) | `XLITE_MAX_M0*svn0*sizeof(float)` | 131072 |
| **合计** | | | | | **131072** |

**AIV UB(预算 `UB_SIZE=192KB`,每 subblock 一份)——`RunAivSoftmaxLong`(indexTopK==0 的无 topk 路径)**

槽位按行宽分块 `MAX_SUB_CONTEXT_SIZE = (VECTOR_MAX_REPEAT-1)*(VECTOR_MAX_BYTESIZE/sizeof(float)) = 254*64 = 16256` 固定分配,长行按子块循环处理(故占用不随行宽增长);`calcDst` 取 `useV2`(`totalSubBlockNum <= 65` 恒真)分支的 `(VECTOR_MAX_REPEAT-1)/2*VECTOR_MAX_BYTESIZE`;`max`/`sum` 随子块数 `totalSubBlockNum+1` 增长(典型 `outN=1296` → 1 块 → 2 槽);`expBuf` 为 GM 侧中转(scores workspace 的高位行,`m0 > XLITE_MAX_M0-4` 时传空),不占 UB。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in` | `MAX_SUB_CONTEXT_SIZE=16256` 个 Dtype | bf16 | 1(事件0,搬入) | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `out` | `MAX_SUB_CONTEXT_SIZE=16256` 个 Dtype | bf16 | 1(事件1,写出) | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `calc` | `MAX_SUB_CONTEXT_SIZE=16256` float | float | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(float)` | 65024 |
| `calcDst` | `127*VECTOR_MAX_BYTESIZE`(归并暂存) | float | 1 | `(VECTOR_MAX_REPEAT-1)/2*VECTOR_MAX_BYTESIZE` | 32512 |
| `max` | `(totalSubBlockNum+1)` 个 256B 槽 | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 512 |
| `sum` | 同上 | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 512 |
| `sink` | 1 个 fp32 标量广播槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| **合计** | | | | | **163840** |

**AIV UB——`RunAivSoftmaxPingPong`(topk 路径,`indexTopK > 0`)**

`len = MAX(outN, topK+swaSegWidth)`(典型 decode 行 `outN = qkStride = 1296`);`maxOut`/`sumOut` 仅 `saveMaxSum`(flash_cxa 传 max/sum 工作区,cxa 恒传 nullptr)时占用,此处为 0;`sink` 槽在 `attnSink` 非空时占用(cxa 恒传)。行内逐行处理,`in`/`out`/`indices` 乒乓与 GM 搬运重叠。

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `in` | `len=1296` 个 Dtype | bf16 | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 5632 |
| `out` | `len=1296` 个 Dtype | bf16 | 2(事件2/3) | `ROUND_UP(len*sizeof(Dtype),256)` | 5632 |
| `cal` | `len=1296` float | float | 1 | `ROUND_UP(len*sizeof(float),256)` | 5376 |
| `temp` | `(len/2)` float 量级的归约暂存 | float | 1 | `(DIV_ROUND_UP(len*4,256)/2)*256` | 2560 |
| `maxOut` | —(saveMaxSum=false) | float | 2 | `VECTOR_MAX_BYTESIZE` | 0 |
| `sumOut` | —(saveMaxSum=false) | float | 2 | `VECTOR_MAX_BYTESIZE` | 0 |
| `indices` | `topK=512` 个 int32 | int32 | 2(事件0) | `ROUND_UP(topK*4,256)` | 4096 |
| `inTopK` | `(topK+swaSegWidth)=784` 个 Dtype | bf16 | 1 | `ROUND_UP((topK+swaSegWidth)*sizeof(Dtype),256)` | 1792 |
| `indicesFp32` | `topK=512` float | float | 1 | `ROUND_UP(topK*4,256)` | 2048 |
| `cmp0` | 边界标量广播槽(kvOffset 基) | int32 | 1 | `VECTOR_MAX_BYTESIZE*2` | 512 |
| `cmp1` | 边界标量广播槽(因果上界) | int32 | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp320` | 边界 fp32 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `cmpFp321` | 边界 fp32 槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| `topkMask0` | `topK/8` 位 hit-mask | uint8 | 1 | `ROUND_UP(topK/8,256)` | 256 |
| `topkMask1` | `topK/8` 位 hit-mask | uint8 | 1 | `ROUND_UP(topK/8,256)` | 256 |
| `currIndices` | `topK=512` 个 int32 | int32 | 1 | `ROUND_UP(topK*4,256)` | 2048 |
| `sink` | sink 标量广播槽 | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| **合计** | | | | | **31232** |

**占用汇总**(各表占用列 vs 硬件容量;L1/L0A/L0B/L0C 按 QK/SV 两段互斥取 max;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| L1(取 QK/SV 较大者) | 512 KB | 393216 B(QK 段) | 75.0% |
| L0A(取 QK/SV 较大者) | 64 KB | 65536 B(QK 段) | 100% |
| L0B(取 QK/SV 较大者) | 64 KB | 65536 B(两段相等) | 100% |
| L0C CO1(取 QK/SV 较大者) | 128 KB | 131072 B(SV 段) | 100% |
| UB(Long 版,每 AIV subblock) | 192 KB | 163840 B | 83.3% |
| UB(PingPong 版,每 AIV subblock) | 192 KB | 31232 B | 15.9% |

要点:

- **`svl0cBuf` = XLITE_MAX_M0(128)×svn0(256)×4B = 128KB,恰好顶满 CO1 区**——这是「L0C 配置 `l0_c_size=262144` 系 CO1/CO2 两区之和、mmad 累加(`TPosition::CO1`)只能用 CO1 128KB」的直接实证(swaBlockSize≤CXA_SVCK0=64 时 svn0=256);QK 段的 `qkl0cBuf`(64KB)与之各自从 CO1 偏移 0 起排,靠 QK/SV 串行互斥复用同一地址空间、同刻只有一段活跃——若按两段相加(192KB)计将超出 CO1,互斥复用是该布局容得下的前提;
- **L0A/L0B 同样顶满 64KB**:QK 段 `qkl0aBuf`/`qkl0bBuf` 乒乓各 64KB,SV 段 `svl0bBuf` 乒乓(svn0=256、svwkMax=64)也恰 64KB;`CxaAicHelper::Init` 中 svn0 的自适应取值(swaBlockSize > CXA_SVCK0 时减半为 128,见代码注释「Halve svn0 … so it fits (64KB)」)正是为保 SV 段 L0B 不溢出;
- **host 校验 `nHeads <= XLITE_MAX_M0=128` 的容量来源**:AIC 把一个 query token 的全部头(`mSize = queryTaskLen*nHeads` 行)装进 XLITE_MAX_M0 行的 L1 A/L0A 缓冲与 qk workspace(`csrc/op.cpp` 的 `XliteOpCXA` 校验;dense 模式直接 `m0 = nHeads`,一 token 一 tile);L1 的大头是 QK 段的 `ql1aBuf`+`kl1bBuf`(headDim=512 下 Q 整拷 128KB + K 四 tile 预取双缓冲 256KB,共 75%);
- **AIV UB 双布局**:topk 路径按 `len` 动态排布(典型 decode 仅 31232B),无 topk 路径按 `MAX_SUB_CONTEXT_SIZE=16256` 固定分块(163840B、不随行宽增长、支持任意行宽);两版上限由 `csrc/kernels/softmax_attn_aiv.h` 的 `static_assert` 按 `MAX_SOFTMAX_PINGPONG_LEN=10496`/`MAX_TOPK_NUM=2048`/`MAX_SWA_SEG_WIDTH=4096` 编译期校验,行宽超 `MAX_SOFTMAX_PINGPONG_LEN` 由 Python 侧路由改走 flash_cxa(flash 的行宽 `swaSegWidth + tileSizeOfCachedKV` 另经 `csrc/op.cpp` 的 `XliteOpFlashCXA` 校验 ≤ 10496,另有 `swaSegWidth ≤ MAX_SWA_SEG_WIDTH` 前置校验)。
