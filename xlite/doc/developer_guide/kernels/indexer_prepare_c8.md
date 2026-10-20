# indexer_prepare_c8

## 功能概述

GLM-5.1 Lightning Indexer(LI)的 **C8(INT8 量化)** K/Q 预处理算子。Host 的 `RunIndexerPrepareC8` 启动一个 AIC/AIV 融合核,完成 LN/RoPE、BF16 Hadamard 矩阵乘和量化写回,处理 K 与(长序列时)Q:

- **K**:LayerNorm(使用传入 `norm_eps`)→ 前 64 维交错 RoPE → Hadamard 矩阵乘 → 动态量化,按 `slot_mapping` 写入 paged **INT8** `index_k_cache` 与 **FP16** `k_scale_cache`;`slot = -1`(padding)跳过。
- **Q**(`is_long`):每 head 做 RoPE → Hadamard → 量化(**不做 LayerNorm**),稠密写入 INT8 `q8`、FP16 `q_scale`,并把 Q scale 折叠进头权重 `scaled_weights`。
- **独立 K 测试入口**(`indexer_k_cache_c8`):同一 kernel、Q 阶段禁用,用于单测 K 路径。

固定配置:`dim = 128`、`rope_dim = 64`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers`),量化上限为 127。`XRuntime::InitIndexerHadamard` 生成并复用归一化 Sylvester 矩阵 `H`,元素为 `±BF16(1/√128) = ±0.08837890625`。Indexer 用 32 个头、head dim 128、RoPE dim 64,局部 `tpSize = 1`。

数学语义:

1. K:`ln = LayerNorm(kw[:, :128], kNorm, kNormBias, eps)`;`ln[:, :64] = InterleavedRoPE(ln[:, :64], freqs[position])`;`h = BF16(ln @ H)`;动态量化 `q8, scale = dyn_quant(h, 127)`,按 slot 写 cache。
2. Q(每 head):前 64 维做交错 RoPE、后 64 维保持原值,随后 `h = BF16(q_rope @ H)`;量化得 `q8`、`q_scale`;`scaled_weights = FP16(原始头权重 × FP16(q_scale))`(不使用传入的 `scale`)。

动态量化(以变换后每行 128 元素 `x` 为单位,见 `csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers::Quantize`):`amax = max(|x|)`;`amax != 0` 时 `factor = vdiv(127, amax)`,对 `x*factor` 取整转 INT8,scale 存 `FP16(amax × 1/127)`;`amax == 0` 时量化值与 scale 均写零。

## 输入输出参数

Python 侧调用(`xlite/_C.pyi`):

```python
# 独立 K 测试入口
indexer_k_cache_c8(rt, k, k_norm, k_norm_bias, freqs, position, slot_mapping,
                   k_cache, scale_cache, norm_eps=1e-6)

# 完整 K+Q 路径(经 indexer_prepare,当 index_k_cache 为 INT8 时走 C8)
indexer_prepare(rt, kw, k_norm, k_norm_bias, freqs, position, index_head_dim,
               index_n_heads, rope_head_dim, block_size, index_k_cache, slot_mapping,
               norm_eps, q, scale, top_k, is_long, k_scale_cache=, q8=, q_scale=,
               scaled_weights=)   # k_scale_cache/q8/q_scale/scaled_weights 仅 C8 必填
```

host 侧有两个算子入口:

- `XliteOpIndexerPrepare`(`csrc/op.cpp` 的 `XliteOpIndexerPrepare`,关键字 `if (indexKCache.dtype == INT8)` LI-C8 分支)在 `indexKCache.dtype == INT8` 时走 LI-C8:约束 `indexHeadDim==128 && ropeHeadDim==64 && indexNHeads==32 && kw 为 BF16 && kNorm/kNormBias 为 FP32 && q 为 BF16 && tpSize==1`,长序列时 `q/q8/qScale/scaledWeights` 必须非空;
- `XliteOpIndexerKCacheC8`(`csrc/op.cpp` 的 `XliteOpIndexerKCacheC8`)为独立 K 测试接口,禁用 Q 阶段。两者共用 `RunIndexerPrepareC8`。融合 kernel 的签名(`csrc/kernels/indexer_prepare_c8.h` 的 `INDEXER_PREPARE_C8_FUNC_DEFINE` 宏):

```cpp
indexer_prepare_c8_<dtype>(GM_ADDR kw, GM_ADDR weight, GM_ADDR bias, GM_ADDR freqs,
                           GM_ADDR positions, GM_ADDR slots, GM_ADDR k_cache,
                           GM_ADDR k_scale_cache, GM_ADDR q, GM_ADDR q8, GM_ADDR q_scale,
                           GM_ADDR scaled_weights, GM_ADDR hadamard, GM_ADDR workspace,
                           uint32_t tokens, uint32_t row_stride,
                           uint32_t heads, uint32_t max_position, uint32_t capacity,
                           bool freqs_fp32, bool is_long, float norm_eps)
```

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| kw / k | 输入(只读) | `[tokens, 160]` | bf16 | Indexer K 投影;`160 = 128(head_dim) + 32(heads)` |
| weight(kNorm) | 输入 | `[128]` | fp32 | LayerNorm weight |
| bias(kNormBias) | 输入 | `[128]` | fp32 | LayerNorm bias |
| freqs | 输入 | `[max_position, 64]` 交错 cos/sin | fp32 或 bf16 | 预计算 RoPE 频率表;`freqs_fp32` 由 host 按 dtype 推断 |
| positions | 输入 | `[tokens]` | int64 | 每个 token 的绝对序列位置 |
| slots(slot_mapping) | 输入 | `[tokens]` | int32 | token → cache 平坦槽位;`-1` 跳过(padding) |
| k_cache(index_k_cache) | 输出 | `[num_blocks, block_size, 1, 128]` | int8 | paged INT8 K cache;`capacity = num_blocks*block_size` |
| k_scale_cache | 输出 | `[num_blocks, block_size, 1, 1]` | fp16 | 每 token 一个 FP16 scale;短序列也写 |
| q | 输入(只读) | `[tokens, 4096]` | bf16 | Indexer Q(`4096 = 32*128`);仅 `is_long` 有效,否则传空 |
| q8 | 输出 | `[tokens, 4096]` | int8 | 量化 Q;仅 `is_long` |
| q_scale | 输出 | `[tokens, 32]` | fp16 | 每 Q head 一个 scale;仅 `is_long` |
| scaled_weights | 输出 | `[tokens, 32]` | fp16 | `FP16(原始头权重 × FP16(q_scale))`;仅 `is_long` |
| hadamard | 输入 | `[128,128]` | bf16 | Runtime 复用的归一化 Sylvester 矩阵 |
| workspace | 临时空间 | `[4,cores,32,128]` | bf16 | 每个 AIC 的 Matmul 输入/输出各两份;`cores=min(rt.aicNum,tasks)` |
| tokens / row_stride / heads / max_position / capacity | — | 标量 | uint32 | `heads=32`;`max_position = freqs.shape[0]`;`capacity = k_scale_cache.numel` |
| freqs_fp32 | — | 标量 | bool | freqs 是否为 fp32(host 推断) |
| is_long | — | 标量 | bool | 长序列开关,启用 Q 阶段 |
| norm_eps | — | 标量 | float | LayerNorm epsilon |

`dest < 0 || dest >= capacity || pos < 0 || pos >= max_position` 的 K 行不读取 RoPE 频率表、不写 cache。前处理将其临时行置零,量化写回阶段跳过。

## 支持的数据类型

| dtype 变体 | 源文件 | 说明 |
|---|---|---|
| `indexer_prepare_c8_bfloat16_t` | `csrc/kernels/indexer_prepare_c8_bfloat16_t.cpp` | kw/q 为 bf16;kNorm/kNormBias 为 fp32;k_cache 为 int8、scale 为 fp16 |

仅 BF16 一种实例化(配置固定为 32 头/128 维/64 RoPE)。dtype 分派见 `csrc/op.cpp` 的 `XliteOpIndexerPrepare` / `XliteOpIndexerKCacheC8`。kernel 使用 `KERNEL_TYPE_MIX_AIC_1_2`,每个 AIC 配对两个 AIV;Cube 部分复用 `Matmul<bfloat16_t,float,bfloat16_t>`。

## 实现原理

### 总体结构与并行

`RunIndexerPrepareC8` 从 Runtime 内存池分配 `[4,cores,32,128]` BF16 工作区,四个平面分别为 `prepared[0/1]`、`rotated[0/1]`。每个 AIC 在每个平面中独占一个 32 行 tile,两个 AIV 各最多处理 16 行。工作区大小由启动核数决定,24 个 AIC 时共 768 KiB。临时空间计入 DummyRun;固定矩阵独立占 32 KiB,首次使用时分配,在 Runtime 析构时释放。

顶层 `indexer_prepare_c8` 按 tile 分配任务。前 `ceil(tokens/32)` 个任务处理 K,长序列时再为每个 token 的 32 个 Q head 分配一个任务。任务按 `block_idx + i*block_num` 轮询分配给 AIC/AIV 组合;K 尾块只处理有效行,没有有效行的 AIV 仍参与跨核同步。K、Q 复用同一份 UB,GM 双缓冲允许下一 tile 的前处理与上一 tile 的矩阵乘重叠。

### IndexerC8Buffers 与 Init

`IndexerC8Buffers`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers`)持有全部 UB 缓冲(weight/bias/x/tmp/other/trig/cos/sin/bf/trig_bf/rounded/half_values/quant/scale/scale_half/position/slot 及 RoPE 的 gather 索引表),K、Q 阶段复用。每个 AIV 占 69,568 字节;position、slot、head weight 随 GM 双缓冲保留两份。

`Init`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers::Init`)在进入循环前**预计算全部 gather 索引与符号表**,使逐行变换用 `vgather` 替代标量寻址:

- `cos_indices`/`sin_indices`/`swap_indices`:RoPE 相邻对(cos 取偶位、sin 取奇位、swap 取配对元素)的 UB 字节偏移,用 `vadds` + `sign_masks`(`csrc/kernels/indexer_prepare_c8.h` 的 `IndexerC8Buffers::Init`,关键字 `sign_masks` / `vadds(rounded, rounded, int32_t((1u << bit) * sizeof(float))`)在 `rounded` 上生成,再为每行加上对应的基址偏移;
- `rope_sign` 初始化为全 1,再用 mask 在对应位写 -1。

`row_indices` 用于把各行的均值、方差和 amax 汇集成连续向量,批量计算倒数与 scale。

### RoPE、矩阵乘与量化

1. **RoPE**(前 64 维相邻对,尾部不变):`IndexerC8Buffers::RoPE` 用 `vgather` 按 `cos_indices`/`sin_indices`/`swap_indices` 取出 cos/sin/配对元素,`vmul`×2 + `vmul`(符号) + `vadd` 得旋转结果,转 BF16 后由 `indexer_c8_prepare_tile` 写入 GM。同一 token 的 Q head 共享同一行 trig。
2. **Hadamard**:AIC 用 `Matmul` 对当前 tile 与 BF16 归一化矩阵做乘法,FP32 累加、BF16 输出。分块和 L1/L0 搬运复用现有 BF16 Matmul。
3. **动态量化**:`indexer_c8_store_tile` 从 GM 读取 BF16 结果、转为 FP32,调用 `IndexerC8Buffers::Quantize`:`vabs` → `ReduceMax`(每行 amax) →(非零时)`vdiv(127, amax)` 得 factor → `vmuls`(x*factor) → `vconv_f322s32r` 直接 FP32→INT32 取整(避免 FP16→INT8 二次舍入)→ `vconv_s322f32` → `vconv_f322f16` → `vconv_f162s8` 得 INT8;scale 存为 FP16(`vconv_f322f16`),`amax==0` 时 factor 为零。转换指令使用 repeat 批量处理各行。

`indexer_c8_prepare_tile` 成组搬入 BF16 行与元数据,K 的 trig 按 position 逐行搬运,每个 Q tile 中,每个 AIV 只读一行 trig;freqs 为 fp32 时先舍入 BF16。`indexer_c8_store_tile` 对 K 按 slot 散列写 INT8 行与精确 2 字节 FP16 scale;对 Q,每个 AIV 连续写出 16 个 head 的 Q8、scale 和 scaled weights。

### K 前处理

每个 AIV 在 `Init` 中将 weight/bias 搬入 UB。K tile 搬入后调用 `IndexerC8Buffers::LayerNorm`:FP32 `vmuls`(x/n) → `reduce_sum`(均值) → `vadds`(减均值) → `vmul` → `vmuls`(除以 dim) → `reduce_sum`(方差) → `vadds`(eps) → `vsqrt` → `vdiv`(求倒数) → `vmuls`(归一化) → weight/bias 仿射,再 `convert_output/input` 经 BF16 舍入。归约沿用 `norm.h` 的 128 维路径,随后做 RoPE 并写入当前 prepared tile。

### Q 前处理与权重

每个 Q tile 对应一个 token,两个 AIV 各处理 16 个 head。**无 LayerNorm**,共享 position/trig,RoPE 后写入 prepared tile。对应的 16 个 BF16 head weight 一次搬入并保留到计算 `scaled_weights` 时使用;量化后将已舍入为 FP16 的 Q scale 和 BF16 头权重转为 FP32 相乘,最后舍入 FP16 写 `scaled_weights`。

### 流水线同步

核内 MTE2(搬入)↔ V(计算)↔ MTE3(写回)用 `set_flag/wait_flag` 同步;LN/量化归约结果由 S 管线读取前使用 V→S 事件。跨核沿用 Indexer 的 mode 2 握手:两个 AIV 完成 prepared 写入后,由 MTE3 发出事件 2/3;AIC 等待后做 Matmul,由 FIX 发出事件 0/1;AIV 等待后量化、写回。每组 AIC/AIV 初始最多准备两个 tile,消费当前 rotated tile 后才复用同一槽位,避免 Cube 覆盖尚未读取的数据。结束时排空本核流水。
