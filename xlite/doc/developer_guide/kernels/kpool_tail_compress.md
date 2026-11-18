# kpool_tail_compress

## 功能概述

GLM-5.3-Flash DSA indexer 的 kpool 尾部压缩算子族:一个 host 入口 `XliteOpKpoolTailCompressAndSpill`(`csrc/op.cpp`)在同一 stream 上先后下发两个纯向量(AIV)kernel——`kpool_tail_compress` 与 `kpool_tail_spill`(均实现在 `csrc/kernels/kpool_tail_compress.h`,模板类 `KpoolTailCompress` / `KpoolTailSpill`),语义对齐 vllm-ascend 原生 triton kernel `_glm5_next_kpool_tail_compress_kernel` + `_store_kpool_tail_kernel`。

**compress kernel**(每个当前 batch token 一个程序,跨核跨步):当该 token 恰好落在一个 kpool 池的末位(`pos >= 0` 且 `(pos+1) % kpool == 0`、`slotMapping[ti] >= 0` 且槽在范围内)时,把窗口 `[pos-kpool+1, pos]` 共 `kpool` 个池槽的 k/gate 折叠成一行写入分页 indexer cache:

```plaintext
window_p = pos - (kpool-1-p),  p = 0..kpool-1          # 池内各槽的绝对位置
scores_p = gate_p + ape_p                              # ape: 每 kpool 槽位的均值权重, [kpool, headDim]
w = softmax_pool(scores)                               # 沿池轴逐 headDim 维 softmax
out = sum_p w_p * k_p                                  # 加权求和
indexerCache[mainSlot/kpool 池槽] = out                # 分页写入, 池槽粒度 = 主 KV 槽/kpool
```

窗口内各槽的数据来源分三种:**本 batch 内的行**直接读 k/gate 输入(`src_row = prev_query_end + eff - reqStart`,`eff = max(poolPos, 0)`,`reqStart = seqLen - queryLen`);**更早的历史行**读该请求私有的 FP32 tail ring(`physical = tailBlockTable[req]`,`offset = eff % cap`,每行按 `[k | gate]` 两段排布);**位置 0 之前**读 0。tail ring 与分页 indexer cache 均为 dim-0 页 padding 视图:内维连续,dim-0 stride(元素单位)≥ 内维跨度,kernel 以 `tailStrideT` / `idxStrideT` 参与寻址,dense 时与 `cap*2*headDim` / `indexerBlockSize*headDim` 相同。

**spill kernel**(每条保留行一个程序,共 `numReqs * cap` 个程序):把当前 batch 的每个 token 的 k/gate 行(两个 FP32 行)追加写入该请求的 tail ring 绝对槽 `physical*tailStrideT + (pos % cap)*2*headDim`(`physical = tailBlockTable[req]`),即 ring 的**最新 `cap` 个有效行**被逐条覆写。

两个 kernel 同 stream 顺序执行:compress 先读历史 ring 行、spill 后覆写它们,host 的下发顺序保证历史行在被读取前不被破坏。模型路径由 `csrc/model.cpp` 的 `XModel::ForwardAttnIndexerKpool` 在 `indexKpoolCompress && indexKpool > 1` 时调用,tail ring 与分页 indexer cache 分别为层 cache 元组的第 4/3 项。

## 输入输出参数

Python 绑定(`xlite/_C.pyi` 的 `kpool_tail_compress_and_spill`):

```python
kpool_tail_compress_and_spill(
    rt, tail_cache, indexer_cache, k, gate, ape, positions, query_start_loc,
    lens, cached_lens, slot_mapping, tail_block_table,
    head_dim, kpool, main_block_size,
)
```

host 侧派生参数(`csrc/op.cpp` 的 `XliteOpKpoolTailCompressAndSpill`):`numTokens = k.shape[0]`、`numReqs = queryStartLoc.numel - 1`、`cap = tailCache.shape[1]`、`tailBlocks = tailCache.shape[0]`、`indexerBlockSize = indexerCache.shape[1]`、`idxBlocks = indexerCache.shape[0]`、`tailStrideT = tailCache.stride[0]`、`idxStrideT = indexerCache.stride[0]`。kernel 签名见 `csrc/kernels/kpool_tail_compress.h` 的 `KPOOL_TAIL_COMPRESS_FUNC_DEFINE` / `KPOOL_TAIL_SPILL_FUNC_DEFINE`(spill 多出的派生参数只有 `numTokens`,其余同源)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| tailCache | 输入+原位输出 | `[tailBlocks, cap, 2*headDim]` | fp32 | 请求 tail ring 的分页池视图,每行一个池槽的 `[k | gate]` 两段行主序;允许 dim-0 页 padding(`stride[0] >= cap*2*headDim`,现网页 40 行),dim-0 stride 以 `tailStrideT` 传入 kernel。compress 读历史行,spill 原位覆写当前 batch 的最新行(本 batch 内已存在的行同样被 spill 重写,值不变) |
| indexerCache | 输出 | `[idxBlocks, indexerBlockSize, 1, headDim]` | bf16 / fp16 / fp32 | 分页 indexer cache(mainSlot/kpool 池槽粒度);允许 dim-0 页 padding(`stride[0] >= indexerBlockSize*headDim`),dim-0 stride 以 `idxStrideT` 传入 kernel;dtype 仅决定 store 宽度 |
| k | 输入 | `[numTokens, headDim]` | fp32 | 当前 batch token 的 k 投影(上游 `wk(x) -> k_norm`,端到端 fp32);行距 `kStrideT` 取 `k.shape[1]` |
| gate | 输入 | `[numTokens, headDim]` | fp32 | 当前 batch token 的 kpool compress gate 投影(`x.float() @ W_f32`);行距 `gateStrideT` 取 `gate.shape[1]` |
| ape | 输入 | `[kpool, headDim]` | fp32 | 每 kpool 池槽位的均值权重(加性,softmax 前) |
| positions | 输入 | `[numTokens]` | int64 | token 绝对位置;`pos < 0` 的 token 被两个 kernel 过滤 |
| queryStartLoc | 输入 | `[numReqs+1]` | int32 | 前缀和;`queryStartLoc[r]` 为请求 r 的绝对起始,最后一个 `start <= ti` 的 r 拥有 token ti |
| lens | 输入 | `[>= numReqs]` | int32 | 各请求本 batch 的 query 长度 |
| cachedLens | 输入 | `[>= numReqs]` | int32 | 各请求已缓存长度;`seqLen = cachedLens[req] + lens[req]` |
| slotMapping | 输入 | `[>= numTokens]` | int32 | 主 KV 的 token 槽表;`mainSlot < 0`(padding)的 token 被 compress 过滤 |
| tailBlockTable | 输入 | `[>= numReqs]` | int32 | 每请求 tail ring 的物理块号;负值或 `>= tailBlocks` 视为无效,compress 侧对应历史行读 0 |
| head_dim | — | 标量 | uint32 | headDim,`<= 256`(kernel UB 预算 `KPOOL_TAIL_MAX_HEAD_DIM`) |
| kpool | — | 标量 | uint32 | 池宽,`<= 8`(`KPOOL_TAIL_MAX_KPOOL`)且 `kpool >= 1`;必须满足 `cap >= kpool`、`mainBlockSize % kpool == 0` |
| main_block_size | — | 标量 | uint32 | 主 KV 的 block 大小;host 侧不直接传给 kernel,pooled 写槽 `mainSlot/kpool` 的整除性由它保证 |

workspace 类参数:无。`tailCache` 为原位读写,`indexerCache` 为纯输出(不读旧值)。

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `kpool_tail_compress_float` / `kpool_tail_spill_float` | `csrc/kernels/kpool_tail_compress_float.cpp` | indexerCache 存 fp32,accF 直接 store |
| `kpool_tail_compress_float16_t` / `kpool_tail_spill_float16_t` | `csrc/kernels/kpool_tail_compress_float16_t.cpp` | indexerCache 存 fp16,store 前 `vconv_f322f16` |
| `kpool_tail_compress_bfloat16_t` / `kpool_tail_spill_bfloat16_t` | `csrc/kernels/kpool_tail_compress_bfloat16_t.cpp` | indexerCache 存 bf16,store 前 `vconv_f322bf16r` |

FP32 契约(对齐原生实现 `sparse_attn_indexer_kpool.py`):**k/gate/ape/tailCache 恒为 fp32**,与 dtype 变体无关——模板参数只选择 indexerCache 的 store 宽度;host 校验 `k.dtype != FP32 || gate.dtype != FP32` 即抛错。spill kernel 本身无 dtype 相关存储(tailCache/k/gate 全 fp32),仍按 dtype 名各导出一份实例,使 host 侧分派逻辑保持不变。host 分派段(`csrc/op.cpp` 的 `XliteOpKpoolTailCompressAndSpill`,关键字 `aclrtlaunch_kpool_tail_compress_`)按 `indexerCache.dtype` 在 FP32/FP16/BF16 三者中选择,其余 dtype 抛错。

## 实现原理

两个 kernel 均为纯向量核(`__DAV_C220_VEC__` 分支,非 `__DAV_C220_VEC__` 下宏展开为空壳),无 AIC、无跨核接力、无外部 workspace,所有中间量都在单程序 UB 内完成。

### 分核与启动

- **compress**:`KpoolTailCompress::Process` 以 `for (ti = GetBlockIdx(); ti < numTokens; ti += GetBlockNum())` 跨核跨步,每程序处理若干 token;launch 网格为 `PickMinBlockNum(rt.aivNum, numTokens)`(`csrc/op.cpp` 的 `XliteOpKpoolTailCompressAndSpill`)。每程序在 `Init` 后先用 `LoadFloatVec` 把 `ape` 整体(`kpool*headDim` fp32)搬入 UB `apeF` 一次,之后逐 token 复用。
- **spill**:`KpoolTailSpill::Process` 以 `pid = GetBlockIdx(); pid += GetBlockNum()` 跨步,任务空间为 `numReqs * cap`(`pid / cap` 为请求号、`pid % cap` 为保留偏移 `retainedOffset`);launch 网格为 `PickMinBlockNum(rt.aivNum, numReqs * cap)`。
- 两 kernel 均以静默早退守护退化形状(`headDim == 0 || kpool == 0 || cap == 0 || numTokens == 0 || numReqs == 0` 及超预算),host 侧将这些条件全部镜像为 fail-closed 抛错(见下文),避免"跳过 launch 导致 cache 停留旧值且无任何报错"。

### compress:KpoolTailCompress

**UB 布局**(`KpoolTailCompress::Init`,自偏移 0 顺序排布,`assert(off <= UB_SIZE)` 收尾):`apeF`/`kF`/`gF` 各 `ROUND_UP(kpool*headDim*4, 256)`(ape 常驻,k/gate 为池窗口 `[kpool, headDim]` 两份工作区);`maxF`/`sumF`/`accF`/`tmpF` 各 `ROUND_UP(headDim*4, 256)`;`gTmpD` 为 `headDim*sizeof(Dtype)` 的降宽 store 暂存。headDim ≤ 256、kpool ≤ 8 的预算即由此套 UB 占用反推。

**逐 token 流程**(`KpoolTailCompress::ProcessOneToken`):

1. **有效过滤**:读 `mainSlot = slotMapping[ti]`、`pos = positions[ti]`;pooled 写槽 `slot = mainSlot / kpool`(`mainSlot < 0` 时为 -1);`valid = (slot >= 0) && (slot < numIndexerSlots) && (pos >= 0) && ((pos+1) % kpool == 0)`,不满足直接返回(padding token 由 `mainSlot < 0` 一并覆盖,`ti < numTokens == queryStartLoc[numReqs]`)。
2. **请求归属**:线性扫 `queryStartLoc`,最后一个 `start <= ti` 的 r 即所属请求;`seqLen = cachedLens[r] + lens[r]`,`reqStart = seqLen - lens[r]` 为本 batch 起始(与 `indexer_topk`/compressor 家族的 metadata 约定一致)。
3. **池窗口收集**(`KpoolTailCompress::GatherWindow`):对池内每个槽位 p,`poolPos = pos - (kpool-1-p)`、`eff = max(poolPos, 0)`,三分支:
   - `poolPos >= reqStart && poolPos < seqLen`(本 batch 行):`srcRow = prevEnd + eff - reqStart`(`prevEnd = queryStartLoc[r]`),钳制到 `[0, numTokens-1]` 后用 `LoadRowF32` 把 k 行、gate 行各搬一份进 `kF`/`gF` 的第 p 行;
   - `poolPos >= 0`(历史行):`physical = tailBlockTable[r]`,`pageOffset = eff % cap`,从 `tailCache + physical*tailStrideT + pageOffset*2*headDim` 读 k 行、`+headDim` 处读 gate 行;`physical` 越界(`< 0` 或 `>= tailBlocks`)时用 `vector_dup(0)` 填零;
   - `poolPos < 0`(位置 0 之前):`vector_dup(0)` 填零。
   每次行加载经 `pipe_barrier(PIPE_MTE2)` + `set_flag/wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0)` 保证 MTE2→V 的数据可见。
4. **压缩计算**(`KpoolTailCompress::CompressAndStore`,全向量指令):
   - 加权得分:逐池行 `vadd(gF[p], gF[p], apeF[p])` 得 `scores_p = gate_p + ape_p`;
   - 数值稳定 softmax:先 `vmax` 链跨池行取逐维最大值进 `maxF`,再逐行 `vsub` 减 max、`vexp`;`vadd` 链累加得 `sumF`;
   - 加权求和:逐行 `vmul(tmpF, kF[p], gF[p])` + `vadd(accF, accF, tmpF)` 累加,最后 `vdiv(accF, accF, sumF)` 归一;
   - 每条向量指令后 `pipe_barrier(PIPE_V)` 串行定序(单程序内无流水重叠,正确性优先)。
5. **分页 store**:池槽 `slot` 映射到 `(indexerBlock = slot / indexerBlockSize, indexerOffset = slot % indexerBlockSize)`,目标地址 `indexerCache + indexerBlock*idxStrideT + indexerOffset*headDim`(`idxStrideT` 为 indexerCache 的 dim-0 stride,dense 时等于 `indexerBlockSize*headDim`)。fp32 变体直接 `copy_ubuf_to_gm(accF)`;fp16/bf16 变体先经 `ConvFloatToHalf`(`vconv_f322f16` / `vconv_f322bf16r`,按 `VECTOR_MAX_REPEAT` 分段)降到 `gTmpD` 再搬运。每次 store 前 `set_flag/wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0)` 等 V 计算完成,`pipe_barrier(PIPE_MTE3)` 等写出排空;末尾再以 `set_flag/wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0)` 做 MTE3→MTE2 反向 fence,保证同一程序处理下一个 token 的 MTE2 装载不会越过本 token 的 MTE3 写出。

### spill:KpoolTailSpill

单程序单任务(`KpoolTailSpill::ProcessOneRow`),仅一块 `headDim*4` 的 UB 行缓冲 `rowF` 复用:

1. **任务分解**:`reqId = pid / cap`,`retainedOffset = pid % cap`;`row = queryStartLoc[reqId] + lens[reqId] - 1 - retainedOffset`,即该请求本 batch 的末尾第 `retainedOffset` 条保留行;`valid = (row >= start) && (row >= 0) && (row < numTokens)`,不满足直接返回(`retainedOffset >= lens[reqId]` 的任务自然落空)。
2. **ring 槽寻址**:`pos = positions[row]`、`physical = tailBlockTable[reqId]`;`pos < 0` 或 `physical` 越界时返回;目标地址 `physical*tailStrideT + (pos % cap)*2*headDim`(逻辑槽公式与原生 tail builder 的 `block*cap + pos%cap` 一致,dim-0 页 padding 由 `tailStrideT` 吸收)。
3. **两段写入**:k 行 `LoadRowF` 进 `rowF` → `set_flag/wait_flag(PIPE_V, PIPE_MTE3)` → `copy_ubuf_to_gm` 写 `tailCache + addr` → `pipe_barrier(PIPE_MTE3)`;随后 `set_flag/wait_flag(PIPE_MTE3, PIPE_V)` + `set_flag/wait_flag(PIPE_MTE3, PIPE_MTE2)` 双 fence(k 的 MTE3 store 排空后,才允许 `rowF` 被 gate 的 MTE2 装载/V 读取覆写);gate 行同流程写 `tailCache + addr + headDim`,再以 MTE3→MTE2 fence 收尾。

### host 侧约束(fail-closed 校验)

`csrc/op.cpp` 的 `XliteOpKpoolTailCompressAndSpill` 在任何 launch 前完成全部校验,违例一律抛错(不静默跳过):

- **dtype 契约**:`tailCache`/`ape`/`k`/`gate` 必须 FP32;`positions` INT64、`queryStartLoc`/`lens`/`cachedLens`/`slotMapping`/`tailBlockTable` INT32;`indexerCache` 限定 FP32/FP16/BF16(分派段否则抛错)。
- **stride 契约**:池视图 `tailCache`/`indexerCache` 要求内维 stride 链式致密且 `stride[0] >= 内维跨度`(dim-0 页 padding,`RequireDim0Pitched`),dim-0 stride 取元素数传入 kernel;逐步张量(`k`/`gate`/`ape`/`positions`/`queryStartLoc`/`lens`/`cachedLens`/`slotMapping`/`tailBlockTable`)要求全维连续(`RequireContiguous`)。
- **rank 检查先行**:所有 `shape.size() >= n` 检查先于任何 `shape[i]` 下标访问(坏 descriptor 必须抛错而非触发 host 侧越界);`queryStartLoc` 非空。
- **维度有效性**:`headDim`/`kpool`/`cap`/`numReqs`/`mainBlockSize` 均非 0。
- **kernel 预算镜像**:`headDim <= 256`(`KPOOL_TAIL_MAX_HEAD_DIM`)、`kpool <= 8`(`KPOOL_TAIL_MAX_KPOOL`)。
- **池/分页相容性**:`cap >= kpool`(spill ring 必须容得下一个完整 compress 窗口)、`mainBlockSize % kpool == 0`(pooled 写槽 `mainSlot/kpool` 保持整除)。
- **表容量**:`tailBlockTable.numel >= numReqs`。
- **形状一致性**:`tailCache.shape[2] == 2*headDim`、`indexerCache.shape[3] == headDim`、`k.shape[1] == gate.shape[1] == headDim`、`ape.shape == [kpool, headDim]`、`positions.numel >= numTokens`、`slotMapping.numel >= numTokens`、`lens`/`cachedLens.numel >= numReqs`。
- **短路返回**:`IsDummyRuntime(rt)` 或 `k.numel == 0` 时直接返回,不 launch。

### 两个 kernel 的顺序依赖

compress 读 ring 的历史行、spill 覆写同一批 ring 行,二者通过同 stream 顺序下发实现"先读后写"(`csrc/op.cpp` 的 `XliteOpKpoolTailCompressAndSpill`,compress launch 在前、spill launch 在后):同 stream 上的 kernel 串行执行,spill 不会在 compress 仍在读取时覆盖历史行。两 kernel 之间无显式同步标志。

## 关键代码位置

- `csrc/kernels/kpool_tail_compress.h` 的 `KpoolTailCompress` — compress 模板类(`ProcessOneToken` / `GatherWindow` / `CompressAndStore` 为核心三步)
- `csrc/kernels/kpool_tail_compress.h` 的 `KpoolTailSpill` — spill 模板类(`ProcessOneRow` 为核心)
- `csrc/kernels/kpool_tail_compress.h` 的 `KPOOL_TAIL_COMPRESS_FUNC_DEFINE` / `KPOOL_TAIL_SPILL_FUNC_DEFINE` — 两个 kernel 的导出宏
- `csrc/op.cpp` 的 `XliteOpKpoolTailCompressAndSpill` — host 封装:全部 fail-closed 校验、dtype 分派与两段 launch
- `csrc/model.cpp` 的 `XModel::ForwardAttnIndexerKpool` — 模型路径调用点(k/gate 的 fp32 投影来源与本算子的衔接)
