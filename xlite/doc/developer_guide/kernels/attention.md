# attention

## 功能概述

标准 MHA/GQA 的 prefill/decode 统一注意力算子(非 flash 路径,单趟 softmax):从融合的 QKV 输入中取 Q,与 paged KV cache 中的 K 做 QK^T,对分数做 causal-mask 的行 softmax,再与 V 相乘得到输出 `out = softmax(mask(QK^T)) · V`。当 `blockTables.shape[1] * blockSize`(运行期总 KV 长度,即表运行期行宽,不取 block table 的分配跨度)超过 host 侧 tile 阈值时,模型层改走 flash_attention 在线 softmax 版本(`csrc/model.cpp` 的 `ForwardAttnMHA`,flash 路由分支关键字 `<= rt._tileSizeOfCachedKV`);本算子假设整个 KV 范围一次算完,中间 QK 分数经 GM 中的 `qk` workspace 传递。AIC(Cube)负责两个 GEMM,QKV 分数和 SV;AIV(Vector)负责 softmax,三者通过软件流水重叠。

## 输入输出参数

Python 侧调用(`tests/kernels/attention.py` 的 `attention(rt, qkv_xlite, ...)` 调用点,关键字 `attention(rt, qkv_xlite`):

```python
attention(
    rt,
    qkv,
    k_cache,
    v_cache,
    output,
    query_start_loc,
    query_lens,
    cached_lens,
    block_tables,
    n_heads,
    n_kv_heads,
    head_dim,
    BLOCK_SIZE,
    batch,
    enable_flash=False,
)
```

host 封装 `Attention` → `XliteOpAttention`(`csrc/_C.cpp` 的 `Attention`、`csrc/op.cpp` 的 `XliteOpAttention`)。kernel 签名(`csrc/kernels/attention.h` 的 `attention_<dtype>` 入口,宏 `ATTN_FUNC_DEFINE`):

```cpp
attention_<dtype>(input, kCache, vCache, qk, output, queryStartLoc, queryLens,
                 cachedLens, blockTables, kCacheStride, vCacheStride, nHeads,
                 nKVHeads, headSize, blockSize, batch, blockTableStride, maxSeqLen, zeroGm)
```

末尾的 `zeroGm` 是 host 自动提供的零缓冲(`rt._attnZeroBuffer`,host launch 见 `csrc/op.cpp` 的 `XliteOpAttention`):`AicHelper::Init` 在 AIC 侧用 `CopyGmToL1Nd2Nz` 把两份 L1B ping-pong 缓冲预填零,避免 L1B 残留脏数据导致 mmad 产生 NaN(nan bug 修复,见实现原理)。

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| input (qkv) | 输入 | `[total_tokens, (nHeads + 2*nKVHeads)*headSize]` | FP16/BF16 | 融合 QKV,布局 [Q \| K \| V];本算子只读 Q 段(K/V 已由 rope_and_cache 写入缓存) |
| kCache / vCache | 输入 | `[numBlocks, blockSize, nKVHeads, headSize]` | FP16/BF16 | paged KV cache,物理块号由 blockTables 给出 |
| qk | workspace | `[aicNum * XLITE_MAX_M0 * 2, maxSeqLen]` | FP16/BF16 | QK^T 分数缓冲(host 分配,见 `csrc/_C.cpp` 的 `Attention`(`!enableFlashAttention` 分支);行宽 `maxSeqLen = blockTables.shape[1] * blockSize`(`shape[1]` 为运行期宽度:V2 由 PrepareAttn 收窄,standalone 由 lens 推导收窄)),按 block_idx 双缓冲 |
| output | 输出 | `[total_tokens, nHeads * headSize]` | FP16/BF16 | attention 输出,`[token, head, headDim]` 展平 |
| queryStartLoc | 输入 | `[batch]` | INT32 | 每 batch 在 total_tokens 中的起始偏移 |
| queryLens | 输入 | `[batch]` | INT32 | 每 batch 的 query token 数 |
| cachedLens | 输入 | `[batch]` | INT32 | 每 batch 已缓存的 KV 长度(causal mask 基准) |
| blockTables | 输入 | `[rows >= batch, blockTableStride]` | INT32 | 逻辑块 → 物理块映射表;行为持久化缓冲,行跨度(blockTableStride)为分配跨度 `blockTables.stride[0]`,行宽为每步运行期宽度 `shape[1]`,仅每行前 `ceil((cachedLens[i]+queryLens[i])/blockSize)` 列有效,行/列尾为残留数据,不被读取 |
| kCacheStride / vCacheStride | 标量 | - | uint32 | K/V cache 的 dim-0 跨度(元素数),host 传 `XTensor.stride[0]`;连续 cache 时等于 `blockSize * nKVHeads * headSize`,非连续 cache 时为真实跨距 |
| nHeads / nKVHeads | 标量 | - | uint32 | 本卡 query / KV 头数,支持 GQA(`headNumInGroup = nHeads/nKVHeads`)与 MHA(二者相等) |
| headSize | 标量 | - | uint32 | 头维(测试覆盖 64/128) |
| blockSize | 标量 | - | uint32 | KV cache 块大小(128) |
| batch | 标量 | - | uint32 | batch 数 |
| blockTableStride | 标量 | - | uint32 | blockTables 的行跨度(元素数),host 取 `blockTables.stride[0]`(XTensor 的 dim-0 跨距;持久化 V2 表为分配跨度,V1/standalone 紧凑表等于行宽),仅用于行寻址 |
| maxSeqLen | 标量 | - | uint32 | qk workspace 的行宽,host 取 `blockTables.shape[1] * blockSize`(`shape[1]` 为运行期宽度:V2 由 PrepareAttn 收窄,standalone 由 lens 推导收窄) |
| zeroGm | 标量 | - | GM | host 自动提供的零缓冲(`rt._attnZeroBuffer`),非 Python API 参数;`AicHelper::Init` 用它预填零 L1B(见实现原理) |

`total_tokens = sum(queryLens)`。测试中覆盖的典型配置见 `tests/kernels/attention.py` 的 `models`/`work` 配置列表(关键字 `models = [`):从 (1,1) decode 到 batch=8 混合 prefill、cached_len 最大 131071。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|---|---|---|
| float16_t | `csrc/kernels/attention_float16_t.cpp` | `attention_float16_t` |
| bfloat16_t | `csrc/kernels/attention_bfloat16_t.cpp` | `attention_bfloat16_t` |

host 侧要求 qkv/qk/kCache/vCache/output 五者 dtype 一致(`csrc/op.cpp` 的 `XliteOpAttention`,关键字 `EachXDtype`)。

## 实现原理

混合 AIC/AIV kernel(`KERNEL_TYPE_MIX_AIC_1_2` 由 Init 隐含,flash 变体显式声明;attention 与 flash_attention 共用同一模式),`Run()` 按 `__DAV_C220_CUBE__`/`__DAV_C220_VEC__` 分派到 `RunAic`/`RunAiv`(`csrc/kernels/attention.h` 的 `Attention::Run`,关键字 `__DAV_C220_CUBE__`)。整体是三级软件流水:**AIC 算 QK → AIV 算 softmax → AIC 算 SV**,相邻任务的 QK 与 SV 在 Cube 上重叠,softmax 在 Vector 上与 Cube 并行。

### 任务划分与 M0 自适应

- 每个 batch 按 `queryTileSize` 切 query 块,任务空间为 `queryNum * nKVHeads`(`csrc/kernels/attention.h` 的 `RunAic`,关键字 `taskNum = queryNum * nKVHeads`)。
- `m0 = XliteFitM0ToHeads(GetOptimalM0(queryLen, cachedLen), headNumInGroup)`(`csrc/kernels/kernel_macro.h` 的 `GetOptimalM0` + `XliteFitM0ToHeads`):queryLen≤64 取 16,否则按总长 12K/20K/24K/30K/48K/60K/96K 分档取 128/112/96/80/64/48/32/16;`XliteFitM0ToHeads` 再把 m0 向上对齐到 `ROUND_UP(headNumInGroup, MBLOCKSIZE)`,确保 `queryTileSize = m0/headNumInGroup ≥ 1`(nHeads > m0 的退化头数,如 GLM-4.7-Flash 20 头,queryLen≤64 时原始 m0=16 不够装一个 tile 会让 L1A/qk workspace 越界)。`m0` 是 Cube 单次 mmad 的 M 维(tokens × headNumInGroup),长序列时压缩 m0 以控制 qk workspace 的 GM 占用(qk 行宽为 maxSeqLen)。GQA 组内多 query 头拼成 m0 行一起算。
- 任务按 `totalIdx % block_num` 轮转分配到各核,并用 `(totalIdx/block_num)%2` 交替正反序以均衡负载(`csrc/kernels/attention.h` 的 `RunAic`,关键字 `totalIdx % block_num`)。
- AIV 侧每个核再按 `get_subblockid()`(2 个 subblock)把 `queryTaskLen * headNumInGroup` 行对半分(`csrc/kernels/attention.h` 的 `RunAiv`,关键字 `get_subblockid()`)。

### AIC 侧:QK 与 SV GEMM(AicHelper)

`AicHelper`(`csrc/kernels/attention_aic_helper.h`)是 attention/flash_attention 共用的 Cube 侧矩阵乘助手,在 `Init()` 中静态布局 L1(A1/B1)与 L0(A2/B2/CO1)的 ping-pong 缓冲:

- `cubeKvTile` 从 blockSize 起逐次减半,直到 2 份 KV tile(含 L0B 的 `tile×headSize` 与 L0A 的 `M0×tile`)同时放进 64KB L0,避免 headDim=256、blockSize=128 时 L0 溢出(`csrc/kernels/attention_aic_helper.h` 的 `AicHelper::Init`,关键字 `cubeKvTile`)。`Init` 末尾(仅 `__DAV_C220_CUBE__`)用 `CopyGmToL1Nd2Nz` 把两份 L1B ping-pong 缓冲从 host 提供的 `zeroGm` 预填零,防止上一 tile 残留的 K 数据在 mmad 累加时混入产生 NaN(nan bug 修复)。
- **RunAicQK**(`csrc/kernels/attention_aic_helper.h` 的 `AicHelper::RunAicQK`):`Q(m0×headSize) × K^T(headSize×kvLen)`。Q 经 `DataCopy` Nd2Nz 进 L1A 再 `CopyToL0ACol`;K 沿 KV 维按 `cubeKvTile` 分块,通过 blockTable 查物理块、按 `block * kCacheStride + rowInBlock * kvMemSize` 寻址(kvMemSize = nKVHeads*headSize),`CopyGmToL1Nd2Nz` + `CopyToL0BCol` 双缓冲进 L0B;`CalMmad` 累加后 `CopyToGm` 写回 qk workspace(行距 qkStride=maxSeqLen)。M/N 维按 MBLOCKSIZE/NBLOCKSIZE=16 对齐,分块边界随 blockSize 截断。
- **RunAicSV**(`csrc/kernels/attention_aic_helper.h` 的 `AicHelper::RunAicSV`):`softmax(QK)(m0×kvLen) × V(kvLen×headSize)`。此时 A 矩阵是 GM 里的 softmax 结果,V 与 K 同样按 `block * vCacheStride + rowInBlock * kvMemSize` 寻址,L1A/L1B 双 ping-pong,`CalMmad` 以 `first` 标志控制首块清零(K 维分块累加);输出按 GQA scatter 写回:`headNumInGroup==1` 时按 `nHeads*headSize` 行距直写,否则逐 token 搬 `headNumInGroup` 行(`gqaScatter=true` 分支,见 `AicHelper::RunAicSV`,关键字 `gqaScatter`)。

### AIV 侧:行 softmax(softmax_attn_aiv.h 并入)

AIV 等 QK 完成后调用 `RunAivSoftmax`(`csrc/kernels/attention.h` 的 `RunAiv`,关键字 `RunAivSoftmax`),实际转发到 `RunAivSoftmax` → `RunAivSoftmaxLong`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmax`,转发至 `RunAivSoftmaxLong`)。要点:

- **causal mask**:第 `idx` 行(`seqIdx = (idx+maskOff)/maskStride` 由 GQA 行号反解 token 序号)的有效长度 `actualCalcLen = calcLen + seqIdx`,即 `cachedLen + queryTaskStart + tokenIdx + 1`(master 上该函数为 cxa 算子新增 winSize/winCalcLen/compressRatio/attnSink/swaSegWidth 等 SWA/compress/attn_sink 扩展参数,attention 系调用时均取默认值 0/0/1/nullptr/0,行为不变);超出部分填 0(先按 VECTOR_MAX_BYTESIZE 对齐用 `SetMaskFromHighBit`+`vector_dup` 置零尾部,再整段补零,`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`,关键字 `SetMaskFromHighBit`)。整行无效时直接写全 0(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`,关键字 `actualCalcLen <= 0`)。
- **长行分块**:一行最多一次处理 `(VECTOR_MAX_REPEAT-1) * calcPad` 个元素(约 255×64 fp32),更长时按 subBlock 分段:第一遍求各段局部 max 并在线归并(段间 max 变化时对已存 exp 乘以 `exp(lastMax-totalMax)` 校正,中间 exp 值写入 GM 的 `expBuf` —— 即 qk workspace 尾部 `(m0 + subIdx*2)*maxSeqLen` 处,见 `csrc/kernels/attention.h` 的 `RunAiv`(关键字 `(m0 + subIdx * 2) * maxSeqLen`),只有 m0 ≤ XLITE_MAX_M0-4 时可用);第二遍从 expBuf 读回、除以 totalSum 写回(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`,关键字 `CopyUbufToGmAligned`)。
- **expBuf 约束**:这就是 `GetOptimalM0` 长序列档位让 m0 ≤ 124 的原因——qk workspace 尾部 4 行被借作 fp32 exp 缓冲。
- **数值精度**:bf16/fp16 先 `vconv_*2f32` 转 fp32,`vmuls`(可选 scale)、`ReduceMaxV2`/`vbrcb`(标量广播成块,避免标量核参与)、`vsub`、`vexp`、`ReduceSumV2`、`vdiv` 后 `vconv_f322*` 舍入回原 dtype(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`,关键字 `vconv_f162f32`)。
- MTE2(载入)/V(计算)/MTE3(写出)三流水线用 `set_flag/wait_flag` 事件同步,行间 ping-pong 双缓冲(`in/out` 各两份,EVETN_ID2/3 交替)。

### AIC/AIV 跨核同步

AIC 在 QK 写完后 `ffts_cross_core_sync(PIPE_FIX, config)`(flagIdx=0,inner-group 模式)通知 AIV;AIV softmax 完成后 `ffts_cross_core_sync(PIPE_MTE3, config)`(flagIdx=1)通知 AIC,AIC 用 `wait_flag_dev(1)` 等 softmax 完成才启动对应 SV(`csrc/kernels/attention.h` 的 `RunAic`,关键字 `ffts_cross_core_sync`)。任务队列上相邻任务错峰:AIC 发起第 i+1 个任务的 QK 后,才回头做第 i 个任务的 SV,使 Cube 与 Vector 始终有活干;最后一个任务的 SV 在循环尾部补做(`csrc/kernels/attention.h` 的 `RunAic`,关键字 `do last softmax * V`)。

### 关键代码位置

- 主类与任务调度:`csrc/kernels/attention.h` 的 `Attention::RunAic` / `Attention::RunAiv`
- QK/SV GEMM 与 L1/L0 布局:`csrc/kernels/attention_aic_helper.h` 的 `AicHelper::Init` / `AicHelper::RunAicQK` / `AicHelper::RunAicSV`
- softmax:`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`
- M0 分档:`csrc/kernels/kernel_macro.h` 的 `GetOptimalM0`;常量 `XLITE_MAX_M0=128`、`MAX_SOFTMAX_PINGPONG_LEN=10496`(`csrc/kernels/kernel_param.h`)
- host launch:`csrc/op.cpp` 的 `XliteOpAttention`、路由 `csrc/model.cpp` 的 `ForwardAttnMHA`

### 片上缓冲分配与占用汇总(AIC L1/L0A/L0B/L0C、AIV UB)

AIC 侧由 `AicHelper::Init`(`csrc/kernels/attention_aic_helper.h` 的 `AicHelper::Init`)静态排布:L1 上 `l1aBuf`(`TPosition::A1`)与 `l1bBuf`(`TPosition::B1`)同处 512KB 物理存储、`bufferAddr` 连续累进,各持两份乒乓;L0A/L0B/L0C 三级物理独立、各自从偏移 0 重新起排,唯 `l0aBuf[1]` 置于 `l0aSvBytes` 偏移处——QK 的 Q 固定占用 buf0,SV 阶段 buf0/buf1 乒乓,两阶段顺序执行故可复用同一 64KB。AIV 侧无 Init 静态布局:`RunAivSoftmaxLong`(`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`)在每次调用入口按运行期参数从偏移 0 排布 UB,函数退出即释放;超长行的中间 exp 走 GM 的 `expBuf`(qk workspace 尾部 4 行),不占 UB。硬件容量(910B/dav C220,NpuArch 2201,CANN 平台配置 `l0_a_size`/`l0_b_size`/`l0_c_size`/`l1_size`):L0A 64KB、L0B 64KB、L1 512KB,L0C 总容量 256KB(`l0_c_size=262144`)为 CO1/CO2 两区之和,mmad 累加缓冲(`TPosition::CO1`)可用的 CO1 区为 **128KB**,下文 L0C 占用均按 CO1 128KB 计;AIV UB 预算取 `UB_SIZE=192KB`(`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`,4KB×48 bank),每个 AIV subblock 各持一份独立 UB。

dtype 记法:`Dtype` 为 qkv/KV cache/qk 的元素类型,fp16 与 bf16 实例化(见上文)均为 2B,故 AIC 与 UB 占用均与 dtype 无关,下表只给单占用列。典型配置取 llama_7B_TP1 的 MHA:nHeads=nKVHeads=32、headSize=128、blockSize=128(`tests/kernels/attention.py` 的 `models` 列表);`AicHelper` 的 L1/L0 布局只依赖 headSize/blockSize/`XLITE_MAX_M0`/dtype,与 nHeads/nKVHeads 无关,该配置下 `cubeKvTile` 维持 blockSize=128 不收缩。AIV 侧 UB 仅 `max`/`sum` 两槽随行宽变化,下表按最长工作负载(`work` 列表的 `(2, [131071]*2, [1]*2)` decode,qk 行宽 `maxSeqLen=131072`、`outN=131072`)计。

**AIC L1(A1/B1,容量 512KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l1aBuf` | `(128, MAX(headSize, cubeKvTile))`(QK 的 Q:`mBlockPad × headSize`;SV 的 A:`mBlockPad × kBlockPad`) | Dtype | 2(事件0/1,SV 阶段;QK 阶段固定 buf0,事件0) | `XLITE_MAX_M0*MAX(headSize,cubeKvTile)*sizeof(Dtype)` | 65536 |
| `l1bBuf` | `(cubeKvTile, headSize)`(QK 的 K / SV 的 V tile) | Dtype | 2(事件2/3) | `cubeKvTile*headSize*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **131072** |

**AIC L0A(A2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0aBuf` | `(128, cubeKvTile)`(QK 的 Q 固定 buf0:`(128, headSize)`) | Dtype | 2(槽位,SV 阶段事件0/1 乒乓,`l0aBuf[1]` 置于 `l0aSvBytes` 偏移) | `XLITE_MAX_M0*cubeKvTile*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0B(B2,容量 64KB)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0bBuf` | `(cubeKvTile, headSize)` | Dtype | 2(事件2/3) | `cubeKvTile*headSize*sizeof(Dtype)` | 65536 |
| **合计** | | | | | **65536** |

**AIC L0C(CO1 区,容量 128KB;总容量 256KB 含 CO2)**

| 变量 | shape(每份) | dtype | 乒乓个数 | 单份大小 | 占用(B) |
|---|---|---|---|---|---|
| `l0cBuf` | `(mBlockPad, nBlockPad)`,两维均 ≤ 128(QK:`mBlockPad × ROUND_UP(nSize,16)`;SV:`mBlockPad × headSize`) | float | 1(事件0) | `XLITE_MAX_M0*MAX(ROUND_UP(cubeKvTile,16),ROUND_UP(headSize,16))*sizeof(float)` | 65536 |
| **合计** | | | | | **65536** |

**AIV UB(预算 `UB_SIZE=192KB`,每 subblock 一份;`RunAivSoftmaxLong` 运行期排布)**

`RunAivSoftmaxLong` 把一行切成 `totalSubBlockNum = DIV_ROUND_UP(outN, MAX_SUB_CONTEXT_SIZE)` 段逐段处理,段长 `MAX_SUB_CONTEXT_SIZE = (VECTOR_MAX_REPEAT-1)*calcPad = 254*64 = 16256`(`calcPad = VECTOR_MAX_BYTESIZE/sizeof(float) = 64`);`totalSubBlockNum ≤ 65` 时 `calcDst` 用 `ReduceMaxV2`/`ReduceSumV2` 的 127×256B 工作区,更长行退回 `ReduceMax`/`ReduceSum` 的 4×256B 工作区(最长工作负载 131072 仍在前一档)。下表占用按 outN=131072(`totalSubBlockNum=9`)计:

| 变量 | shape(每份) | dtype | 乒乓个数 | 大小(每份) | 占用(B) |
|---|---|---|---|---|---|
| `in` | 16256 个 Dtype(`MAX_SUB_CONTEXT_SIZE`) | Dtype | 1(事件0/1 流水) | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `out` | 16256 个 Dtype | Dtype | 1(事件0/1 流水) | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `calc` | 16256 个 float | float | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(float)` | 65024 |
| `calcDst` | 127×256B(useV2 归约工作区) | float | 1 | `(VECTOR_MAX_REPEAT-1)/2*VECTOR_MAX_BYTESIZE` | 32512 |
| `max` | (totalSubBlockNum+1)×256B(段 max 槽 + `totalMax` 尾槽) | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 2560 |
| `sum` | (totalSubBlockNum+1)×256B | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 2560 |
| `sink` | 256B(attnSink 标量槽,attention 系调用恒传 nullptr,槽位仍保留) | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| **合计** | | | | | **167936** |

**占用汇总**(各表占用列之和 vs 硬件容量;字节数):

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| L1 | 512 KB | 131072 B | 25.0% |
| L0A | 64 KB | 65536 B | 100% |
| L0B | 64 KB | 65536 B | 100% |
| L0C(CO1) | 128 KB | 65536 B | 50.0% |
| UB(每 AIV subblock) | 192 KB | 167936 B | 85.4% |

要点:

- **L0A 与 L0B 在 headSize=128、blockSize=128 时双双占满 64KB,是本算子最紧的两级**:`AicHelper::Init` 的 `cubeKvTile` 收缩循环(`2*cubeKvTile*headSize*sizeof(Dtype) ≤ ASCEND_L0_BYTES=64KB` 且 `2*XLITE_MAX_M0*cubeKvTile*sizeof(Dtype) ≤ 64KB`,关键字 `cubeKvTile`)是 L0 不溢出的唯一保证——headDim=256、blockSize=128 时 tile 减半到 64,两份 KV tile 才放得下 L0;
- AIV UB 最紧(85.4%),大头是 `in`/`out`/`calc`/`calcDst` 四个 `MAX_SUB_CONTEXT_SIZE` 级别的常量槽(合计 162560B,占 96.8%);UB 占用仅 `max`/`sum` 的 `(totalSubBlockNum+1)*256B` 随行宽缓增,超长行经 GM `expBuf` 分段中转,不依赖编译期断言——`MAX_SUB_CONTEXT_SIZE` 分块本身即上限控制;
- L1 仅占 25%(两份 `l1aBuf` + 两份 `l1bBuf` 各 32KB);`l0cBuf` 单份 64KB,QK 与 SV 串行复用同一 CO1 区;host 侧 `nHeads/nKvHeads ≤ XLITE_MAX_M0` 校验(`csrc/op.cpp` 的 `XliteOpAttention`)源自 `XliteFitM0ToHeads` 的 M0 tile 约束而非存储容量;
- `GetOptimalM0` 分档与 `m0` 自适应只影响任务划分与 qk workspace 的 GM 占用,`AicHelper` 一律按 `XLITE_MAX_M0=128` 上限分配,片上占用不随 m0 档位变化。
