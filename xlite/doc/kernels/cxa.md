# cxa

## 功能概述

cxa 是 DeepSeek-V4 使用的统一注意力算子(C4A and C128A attention,即压缩比 4 与压缩比 128 的压缩注意力)。它将**滑动窗口段(SWA)**与**压缩稀疏段**拼接在同一行分数上,**在同一个 softmax 中联合计算**,并把每头的 `attn_sink` 偏置并入 softmax 分母:

- **滑动窗口段(SWA)**:query 位于绝对位置 `q` 时,只能看到窗口 `[max(0, q - window_size + 1), q]` 内的 SWA token;超出当前已生成长度的位置 mask 为 `-inf`。
- **压缩稀疏段**:对压缩 KV(每 `compress_ratio` 个 token 压缩成 1 个),仅保留 `topk_indices` 引用的位置;`topk_indices` 中为 `-1` 的项 mask 为 `-inf`(其 exp 贡献为 0)。
- **attn_sink**:每头一个可学习的 sink 偏置(fp32),`exp(attn_sink - row_max)` 被加进 softmax 分母(见 `csrc/kernels/softmax_attn_aiv.h:394`)。

两种退化形态:

- `window_size = 0`:纯压缩注意力(无 SWA 段,`swaSegWidth = 0`,scores workspace 仅含压缩段);
- `compress_ratio = 0`(或 `index_topk = 0`):纯滑窗 / 滑窗+全量压缩注意力(无 topk 选路,压缩段全保留)。

与 Python 参考实现的对应关系:tests/kernels/cxa.py 将本算子与 tests/models/deepseek_v4_kernel.py 中的 `sparse_attn`(对拼接后的 `[window_size + compressed_kv_len]` KV 矩阵做一次 gather+softmax)逐 case 对比,两侧读取同一份 token 数据,输出直接可比。

## 输入输出参数

Python 签名见 `xlite/_C.pyi:2227`(`def cxa(...)`),kernel 侧入口见 `csrc/kernels/cxa.h:353`(`CXA_FUNC_DEFINE`)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| q | 输入 | [totalQ, nHeads, headDim] | bfloat16 | query |
| swa_k_cache | 输入 | [swaBlockNum, swaBlockSize, headDim] | bfloat16 | 分页滑动窗口 KV cache |
| compress_k_cache | 输入 | [compressBlockNum, compressBlockSize, headDim] | bfloat16 | 分页压缩 KV cache;`compress_ratio == 0` 时不使用(可为空) |
| swa_block_tables | 输入 | [batch, swaMaxNumBlocks](或展平 1-D) | int32 | SWA cache 的 block table |
| compress_block_tables | 输入 | [batch, compressMaxNumBlocks](或展平 1-D) | int32 | 压缩 cache 的 block table |
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
| index_topk | 输入 | 标量 | - | 每行 top-k 压缩索引数;`0` 表示禁用 topk 稀疏路径 |
| topk_indices | 输入 | [totalQ, indexTopK] | int32 | 压缩段 top-k 索引(IndexerTopK 输出);`-1` 表示 mask 掉的位置 |

另有内部 scores workspace(由 binding 侧分配,`csrc/_C.cpp:1633`):

- **scores**:`[2 * aicNum * XLITE_MAX_M0, swaSegWidth + kvSize]`,dtype 同 q;
  - `swaSegWidth = windowSize == 0 ? 0 : windowSize + XLITE_MAX_M0 + K_BLOCK_SIZE_2B`(`csrc/_C.cpp:1632`、`csrc/kernels/cxa.h:78`):一个 query tile 的因果窗并集宽 `windowSize + XLITE_MAX_M0`,再加 `K_BLOCK_SIZE_2B`(=16,`csrc/kernels/kernel_param.h:36`)吸收 `windowStart` 向下取整产生的前导列;
  - `kvSize`:压缩段宽度,host 侧取 `ROUND_UP(compressMaxNumBlocks * compressBlockSize, 4 * CXA_SVCK0)`(`csrc/_C.cpp:1633`),按 `4 * CXA_SVCK0` 对齐(见下文 scores 布局)。

## 支持的数据类型

仅 **bfloat16_t**:`csrc/kernels/cxa_bfloat16_t.cpp` 只实例化 `cxa_bfloat16_t`,host 侧 `XliteOpCXA`(`csrc/op.cpp:961`)也仅接受 BF16 分支,否则抛异常。

调度类型为 `KERNEL_TYPE_MIX_AIC_1_2`(`csrc/kernels/cxa.h:53`):1 个 AIC(cube 核)配 2 个 AIV(vector 核)混合调度,AIC 负责 QK/SV 两个 cube GEMM,AIV 负责 softmax;AIC 与 AIV 之间用 `ffts_cross_core_sync` / `wait_flag_dev`(inner-group 同步,flag 0 = AIC→AIV 的 QK 完成,flag 1 = AIV→AIC 的 softmax 完成)做跨核握手。

## 实现原理

### scores workspace 布局

每行分数的 stride 为 `qkStride = swaSegWidth + kvSize`(`csrc/kernels/cxa.h:79`),一段连续内存中两个段前后拼接:

```
列 0                                              swaSegWidth        qkStride
|<----------- SWA 段(含 lead-in) ----------->|<------ 压缩段 ------>|
   [lead-in | 因果可见窗 [windowStart, windowEnd]]
```

- **SWA 段**(列 0 起):`RunAicQK` 把窗起点 `windowStart` **向下取整到 `kBlockSize`(32/sizeof(Dtype)=16)倍数**得到 `alignStart`,scores 列 0 对应绝对位置 `alignStart`(`csrc/kernels/cxa_aic_helper.h:155`)。这样保证每个 tile 的 nSize 都是 `kBlockSize` 的倍数,没有列跨 tile 边界,SV 路径不会重复统计边界 token;lead-in 列 `[0, windowStart - alignStart)` 由 `RunAivSoftmax` mask 成 `-inf`。lead-in 宽度恒小于 `K_BLOCK_SIZE_2B = 16`,这就是 `swaSegWidth` 中 `+ K_BLOCK_SIZE_2B` 的来源。
- **压缩段**(列 `swaSegWidth` 起):`RunAicQK` 把压缩段分数写到 `scores[swaSegWidth + nOffset]`(`csrc/kernels/cxa_aic_helper.h:324`),按 `qkcn0`(compressBlockSize)分 tile;host 侧把 `kvSize` pad 到 `4 * CXA_SVCK0` 的倍数,因为 `RunAicSV` 每次读 `4 * svck0` 个压缩段分数(`csrc/kernels/kernel_param.h:40` 注释、`csrc/kernels/cxa_aic_helper.h:484`)。

scores workspace 整体被切成两块 ping-pong 缓冲(`PINGPONG_BUF_NUM = 2`):每个 AIC 核占用 `block_idx * XLITE_MAX_M0 * qkStride` 起的第一块,以及偏移 `block_num * XLITE_MAX_M0 * qkStride` 的第二块,交替用于相邻两个 tile(`csrc/kernels/cxa.h:81`),使 QK(本 tile)与 SV(上一 tile)能同时使用不同的缓冲。

### AIC 路径(RunAic)

入口 `csrc/kernels/cxa.h:91`,cube 核执行,按 batch 遍历,批内任务按 `firstCore = (blockIdx + blockNum - coreOffset) % blockNum` 起始、步长 `blockNum` 轮转分发(`coreOffset` 跨 batch 累积,避免所有 batch 都从 0 号核开始):

1. `GetOptimalM0(queryLen, cachedLen)`(`csrc/kernels/kernel_macro.h:883`)按序列长度选择 M0(短序列 16,≤12K 取 128,超长序列降到 16);`queryTileSize = m0 / nHeads`(即一个 tile 内所有头一起算,`mSize = queryLen * nHeads` 行)。
2. 每个 query tile 调 `CxaAicHelper::RunAicQK` 计算本 tile 的 QK 并把分数写入 `scores[curr]`,随后 `ffts_cross_core_sync(PIPE_FIX, config)` 通知 AIV "QK 完成"(flag 0,mode 2 = 组内 AIC/AIV 同步,`csrc/kernels/cxa.h:98`)。
3. 当前 tile 的 QK 与上一 tile 的 `RunAicSV` 流水重叠:若 `needDoSV`,先 `wait_flag_dev(1)` 等 AIV 完成上一 tile 的 softmax,再执行 `RunAicSV(scores[last], ...)`,用上一 tile 的分数做 softmax×V,把输出原位写到 `output[lastMhOffset]`。`last*` 系列变量缓存上一 tile 的全部上下文(batch、GM 偏移、block table 指针、calcLen 等)。
4. 循环结束后补做最后一个 tile 的 SV(`csrc/kernels/cxa.h:189`)。

### AIV 路径(RunAiv)

入口 `csrc/kernels/cxa.h:202`,vector 核执行,任务划分与 AIC 完全一致(同样的 batch 遍历、`GetOptimalM0`、轮转起点),保证 AIC/AIV 对同一 tile 的编号一一对应。每个 tile:

1. 拆分:`nWork = queryTaskLen * nHeads` 行 softmax 工作,均分给本 AIC 配对的 2 个 AIV 子核(`get_subblockid()`),每个子核处理 `nWorkPerCore` 行,起点偏移 `qkOffset = nWorkStart * qkStride`。
2. 计算 mask 相关量:
   - `calcSoftmaxLen = cachedLen + queryTaskStart + 1`:该 tile 第一行的因果可见长度(绝对坐标);
   - `winStart = max(0, calcSoftmaxLen - windowSize)`:第一行因果窗起点;`winCalcLen = calcSoftmaxLen - ROUND_DOWN(winStart, K_BLOCK_SIZE_2B)`(`csrc/kernels/cxa.h:262`):scores 列 0 对应 `alignStart`,故可见 SWA 列数是 `calcSoftmaxLen - alignStart`,窗滑动时比 `windowSize` 多出的部分就是 lead-in(<16);
   - `outN = swaSegWidth + ROUND_UP(calcLen / compressRatio, 4 * svk0)`(上限 `qkStride`,压缩比为 0 时无压缩段):本 tile 需要处理/回写的分数列数。
3. `wait_flag_dev(0)` 等 AIC 的 QK 完成,然后:
   - **无 topk 路径**(`indexTopK == 0`):`RunAivSoftmax`(内部转 `RunAivSoftmaxLong`,`csrc/kernels/softmax_attn_aiv.h:801`)直接在 scores 上做在线 softmax;`m0 > XLITE_MAX_M0 - 4` 时无 expBuf 暂存(传 nullptr)。SWA+全量压缩走此路径。
   - **topk 路径**:调 `RunAivSoftmaxPingPong`(`csrc/kernels/softmax_attn_aiv.h:65`),传入 `topkIndices + indexTopK * queryTaskOffset`(`calcLen > indexTopK` 时,否则视为无 topk)。压缩段分数不再整段参与 softmax,而是用 `vgather` 按 topk 索引从 GM 收集到 UB(`csrc/kernels/softmax_attn_aiv.h:292`),与 SWA 段拼成 `[swaSegWidth + topK]` 行做 softmax,再按 64-slot 粒度的 hit-mask scatter 回 scores workspace 对应压缩位置(`csrc/kernels/softmax_attn_aiv.h:406`),供 AIC 的 SV 读取。索引为 `-1` 的位置通过 `vcmpvs_ge`/`vcmpvs_lt` 边界比较被排除在 gather/scatter 之外,等效 mask 为 `-inf`。
4. softmax 完成后 `ffts_cross_core_sync(PIPE_MTE3, config)` 通知 AIC(flag 1),并翻转 ping-pong 缓冲 `curr`。

**窗口 causal mask**(`RunAivSoftmaxPingPong` 内,`csrc/kernels/softmax_attn_aiv.h:268`):分数先转 fp32(`vconv_bf162f32`),再对 SWA 段 `[0, swaSegWidth)` 做双向 mask,均用 `vector_dup` 写 `-3.4028235e38`(fp32 最小值,≈-inf):

- **右 mask**:每行可见窗宽 `actualWinLen = min(winCalcLen + seqIdx, swaSegWidth)`,将 `[actualWinLen, swaSegWidth)` 填 `-inf`。按 64 元素(VECTOR_MAX_BYTESIZE/4)的 repeat 对齐处理:区间落在单个 repeat 内用 `SetMaskRange(lo, hi)` 精确置位,跨 repeat 则首块用 `SetMaskFromHighBit`、尾块用 `SetMask(wtail)` 位级控制(`csrc/kernels/softmax_attn_aiv.h:276`)。
- **左 mask**:lead-in 列 `[0, lEnd)` 填 `-inf`,其中 `lEnd = actualWinLen > winSize ? actualWinLen - winSize : 0`,即窗起点之前的取整前导列(`csrc/kernels/softmax_attn_aiv.h:330`),整 repeat 用普通 `vector_dup`,余数用 `SetMask(lRem)`。
- 该行 causally 不可见(`actualCalcLen <= 0`)或 topk 全被 mask 且无 SWA 段时,直接清零输出并跳过计算。

**attn_sink 并入分母**(`csrc/kernels/softmax_attn_aiv.h:394`):在线 softmax 得到 row_max 后,从 GM 读 `attnSink[headIdx]` 到 UB,与分数同样地做 `sink - max` → `vexp`,并把 `vexp(sink)` 加到 `ReduceSumV2` 的结果上,即分母 `s = Σ exp(sᵢ - max) + exp(sink - max)`;最后 `vdiv` 归一化只作用于分数部分,sink 不进入输出。

`RunAivSoftmaxLong` 与 PingPong 版逻辑相同,区别在于 UB 布局:`Long` 版按 `MAX_SUB_CONTEXT_SIZE` 分子块支持超长行(上限见 `csrc/kernels/softmax_attn_aiv.h:40` 的 `static_assert`,`MAX_TOPK_NUM + MAX_SWA_SEG_WIDTH` 规模必须放进 UB);PingPong 版用双缓冲 in/out 加速常规长度。

### CxaAicHelper(QK / SV cube 计算)

`csrc/kernels/cxa_aic_helper.h:12`,封装 AIC 的两个 cube GEMM 及 L1/L0 缓冲管理。

**tiling 常量**(`Init`,`csrc/kernels/cxa_aic_helper.h:17`):

| 常量 | 含义 | 取值 |
|---|---|---|
| `qkwn0` | QK 的 SWA 段 n 方向 tile | `swaBlockSize` |
| `qkcn0` | QK 的压缩段 n 方向 tile | dense 模式为 `MAX_N0`(128),否则 `compressBlockSize` |
| `qkk0` | QK 的 k 方向 tile | `256 / sizeof(Dtype)`(bf16 为 128) |
| `svn0` | SV 的 n 方向 tile(headDim 方向) | 256 |
| `svwk0` | SV 的 SWA 段 k 方向 tile | `swaBlockSize` |
| `svck0` | SV 的压缩段 k 方向 tile | `CXA_SVCK0 = 64`(`csrc/kernels/kernel_param.h:40`) |

**L1(A1/A2)/L0 缓冲布局**(`Init` 中手动排布 bufferAddr):QK 部分有 `ql1aBuf`(A1,`XLITE_MAX_M0 * headDim`)、双缓冲 `kl1bBuf`(A1,`(max(qkwn0,qkcn0) * 4 * qkk0)`,按 `kIdx4 == 0` 时一次搬 4 个 k-tile)、双缓冲 `qkl0aBuf`/`qkl0bBuf`(A2/B2,`MAX_N0 * qkk0`)、单缓冲 `qkl0cBuf`(CO1,`XLITE_MAX_M0 * MAX_N0 * float`);SV 部分有双缓冲 `scoresl1aBuf`(A1,分数源,大小按 `max(svwk0, svck0)` 的 4 倍留)、双缓冲 `ktl1bBuf`(A1,K^T,SWA 段 `svwk0` / 压缩段一次搬 `2 * svck0`)、双缓冲 `svl0aBuf`/`svl0bBuf`(A2/B2)、单缓冲 `svl0cBuf`(CO1,`XLITE_MAX_M0 * svn0 * float`)。所有双缓冲通过 `HardEvent`(MTE1_MTE2 / MTE2_MTE1 / M_MTE1 / MTE1_M / M_FIX / FIX_M)的手工 Set/WaitFlag 做 GM→L1→L0→MMAD→L0C→GM 的逐级流水。

**RunAicQK**(`csrc/kernels/cxa_aic_helper.h:127`,`scores = Q * K`):Q 一次整拷进 L1;SWA 段沿 `nwIdx` 遍历窗内 block(swaBlockTables 查页),压缩段沿 `nIdx` 遍历压缩 block(compressBlockTables 查页),每 tile 内沿 headDim 按 `qkk0` 分 k 循环,`kIdx4`(每 4 个 k-tile)触发 ping-pong 换 L1B 缓冲。`CalMmad` 累加到 L0C 后 `CopyToGm` 写回 scores 对应列偏移(SWA 段 `nOffset - alignStart`,压缩段 `swaSegWidth + nOffset`)。

**RunAicSV**(`csrc/kernels/cxa_aic_helper.h:342`,`output = softmax(scores) * K^T`):n 方向沿 headDim 按 `svn0 = 256` 分 tile;k 方向先遍历 SWA 段(分数 tile 与 K^T tile 都从 `alignStart` 起对齐,保证 `[kOffset, kOffset + kBlockPad)` 区间两两不相交,`csrc/kernels/cxa_aic_helper.h:359`),再遍历压缩段:分数按 `4 * svck0` 粒度搬入 L1(`kIdx4`),K^T 按 `2 * svck0` 粒度搬入 L1(`kIdx2`);paged cache 按 block table 逐 block 拷贝,dense cache 单次连续拷贝(`csrc/kernels/cxa_aic_helper.h:499`)。所有 tile 的 mmad 累加到同一 `svl0cBuf`(`init` 标志控制首 tile 清零),每轮 n-tile 结束后 `CopyToGm` 写 `output[nOffset]`。

`dense` 模板参数(本算子传 `false`,`csrc/kernels/cxa.h:88`)区分 paged/dense cache 布局,dense 时压缩段 block table 不参与寻址。

### 校验

tests/kernels/cxa.py 覆盖 5 种模型配置(base-swa、swa、c128a、c4a-ntok、c4a-topk)× 16 种 batch/长度组合(含 131071 超长 cached_len、多 batch 混合长度),与 `sparse_attn` 参考实现对比容差 `atol=5e-5, rtol=5e-2`;topk 场景中参考实现的绝对压缩索引被换算到 op 的"分数行坐标"(列 `swaSegWidth + j`,`tests/kernels/cxa.py:308`)。
