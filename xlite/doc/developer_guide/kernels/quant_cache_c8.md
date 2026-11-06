# quant_cache_c8

## 功能概述

C8 缓存写入算子(compressor 路径):把连续 BF16 行——即 compressor 归一化 + rope 之后的 `weightedSum`——按行量化后散写到分页 INT8 key cache,并同步写每 slot 一个 fp16 scale。slot 寻址为 block 分层(`slot = block * blockSize + offset`,行落在 `block * kCacheStride + offset * dim`,scale 落在 `block * kScaleCacheStride + offset`)——DSV4 的 vllm 池把 K 与 scale 按页打包进同一分配(stride[0] = blockSize * (dim + 2)),dense 自建池则退化回 flat 寻址;量化语义与 [indexer_prepare_c8](indexer_prepare_c8.md) 的 K 路径逐字节一致,两个算子写出的 C8 缓存对 [indexer_topk](indexer_topk.md) 的 C8 reader 可互换:

```text
scale_row  = abs_max(x_row) / 127        (fp16, 写 kScaleCache[block*kScaleCacheStride + offset])
factor     = abs_max == 0 ? 0 : 127 / abs_max
out        = round_nearest(x_row * factor)   (int8, 写 kCache[block*kCacheStride + offset*dim : +dim])
```

round_nearest 由 `vconv_f322s32r → vconv_s322f32 → vconv_f322f16 → vconv_f162s8` 链保证(先取整到最近整数再打包,见 `csrc/kernels/quant_cache_c8.h` 的 `quant_cache_c8`,关键字 `vconv_f322s32r`);零行量化为全零 int8 + 零 scale。slot 为负或 ≥ capacity 的行整行跳过,两个 cache 均不写。

它是 DSV4 Path B 第二阶段("compressor int8 写变体")的落点:compressor 以 `compress_kv=None` 运行时原地 rope `weightedSum`([compressor](compressor.md) 的 C8 模式),本算子随后把行量化入池,替代 bf16 substitute 池,对齐 vllm-ascend 原生 int8 indexer cache 格式(第四阶段起直接写 vllm 原生池,kv tuple 第 1/5 槽)。

## 输入输出参数

Python 入口 `quant_cache_c8(rt, x, slot_mapping, k_cache, k_scale_cache, block_size)`(`csrc/_C.cpp` 的 `QuantCacheC8`,关键字 `m.def("quant_cache_c8"`)。kernel 签名见 `csrc/kernels/quant_cache_c8.h` 的 `quant_cache_c8`:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x | 输入 | `[rows, dim]` | bfloat16 | 待量化行(compressor 的 weightedSum);dim 为 64 的倍数且 ≤ `QUANT_CACHE_C8_MAX_DIM`(4096) |
| slot_mapping | 输入 | `[rows]` | int32 | flat cache slot = block * blockSize + offset;负值或 ≥ capacity 跳过两 cache 写入 |
| k_cache | 输出 | `[num_blocks, blockSize, (1,) dim]` 或 flat `[capacity, dim]` / `[capacity*dim]` | int8 | 分页 key cache(paged 池允许 pitch,stride[0] 为 KCacheStride;flat 池连续即可) |
| k_scale_cache | 输出 | `[num_blocks, blockSize, (1,) 1]` 或 flat `[capacity]` | float16 | 每 slot 一个 scale = absmax/127(paged 池允许 pitch;flat 池连续即可) |
| rows | 标量 | - | uint32_t | host 传 `x.shape[0]` |
| dim | 标量 | - | uint32_t | host 传 `x.shape[1]` |
| capacity | 标量 | - | uint32_t | host 传 `k_scale_cache.numel` |
| blockSize | 标量 | - | uint32_t | 每 block 的 slot 数;必须整除 capacity |
| kCacheStride | 标量 | - | uint32_t | int8 元素计的每 block 跨距;dense 池 = blockSize * dim |
| kScaleCacheStride | 标量 | - | uint32_t | fp16 元素计的每 block 跨距;dense 池 = blockSize |

host 侧检查(`csrc/op.cpp` 的 `XliteOpQuantCacheC8`):x 必须 BF16 二维、dim 为非零 64 倍数且 ≤ `QUANT_CACHE_C8_MAX_DIM`、slot_mapping 与行数一致、`k_cache.numel == capacity * dim`、`blockSize != 0` 且整除 capacity;跨距按 shape 维度分派——rank > 2 的 paged 池取 `stride[0]` 并校验 `k_cache.stride[1] == dim`、`k_scale_cache.stride[1] == 1`(block 内行连续),2-D/1-D flat 池校验连续后按 blockSize 推导(`blockSize * dim` / `blockSize`,整池合成单块)。`rows == 0` 直接返回。launch 用 `PickMinBlockNum(rt.aivNum, rows)`。

测试参考:[tests/kernels/quant_cache_c8.py](../../../tests/kernels/quant_cache_c8.py)(dim 64/128/192 × 行数 0/1/17/129/257,零行/常量行/outlier、越界 slot 跳过、非对齐 cache view 双端 guard,以及 `packed-vllm-pool`:as_strided 把 K/scale 按页交错打包进一个分配的 DSV4 布局);[tests/kernels/compressor.py](../../../tests/kernels/compressor.py) 的 `c8_quant` 项验证 compressor C8 模式 + 本算子的端到端等价性(原地 rope 行与 paged 行 bit-exact,量化 cache 对齐参考)。

## 支持的数据类型

- 输入 `bfloat16_t` → 输出 `int8_t` + `half` scale([quant_cache_c8_bfloat16_t.cpp](../../../csrc/kernels/quant_cache_c8_bfloat16_t.cpp))

host 仅接受 `x.dtype == BF16`,kernel 入口符号为 `quant_cache_c8_bfloat16_t`(host 经 `aclrtlaunch_quant_cache_c8_bfloat16_t` 启动)。

## 实现原理

实现位于 [csrc/kernels/quant_cache_c8.h](../../../csrc/kernels/quant_cache_c8.h),函数 `quant_cache_c8<dtype>`(模板;BF16 实例化由 `QUANT_CACHE_C8_FUNC_DEFINE` 宏展开为入口符号 `quant_cache_c8_bfloat16_t`)。

### 分块策略

- **行间多 Block 轮转 + ping-pong 流水**:`for (row = block_idx; row < rows; row += block_num)`,槽号 `curr` 每行翻转(`curr = 1 - curr`),GM→UB 载入位于循环顶部且仅此一处(见下文"流水线同步"):当前行的 MTE3 写出与下一行的 MTE2 载入重叠。dim 小(V4 为 128),每行 V 工作仅数条向量指令,故不做 quant_dyn 式的行组批化(如后续 profiling 需要,可按其 fast path 的 rows-group 模式扩展)。
- **slot 读取**:slot 随行数据同乘循环顶部的 MTE2 载入,S 管线经 `MTE2→S` 同 ID set/wait 对从 UB 标量读——避免每行一次的 GM 标量读(发射器独占停顿,任何其它管线都盖不住);标量读完 slot 才发 `MTE2→V` 对放行 convert,覆写指令更晚发射,复用安全。越界(负或 ≥ capacity)行走纯 flag 中继:不计算、不写出,但仍归还两个缓冲槽,使每个槽无论行有效性都恰好一 set/wait 对;行数据无条件载入,slot 无效时丢弃。
- **对齐**:UB 各段按 `ROUND_UP(bytes, VECTOR_MAX_BYTESIZE)`(256B)排布;int8 行写出用 `copy_ubuf_to_gm_align_b8`(容忍非 32B 对齐的 cache view,与 indexer_prepare_c8 的 store 同款),2B scale 用 `CopyUbufToGmAligned` 的 b16 路径(源 scaleF16 256B 对齐)。

### UB 内存布局(`csrc/kernels/quant_cache_c8.h` 的 `quant_cache_c8`)

跨管线缓冲双缓冲(MTE2 写的 xRow/slotBuf、MTE3 读的 quant/scaleF16 各两份),V 内部缓冲单份(V 管线自身串行化行),总容量以 `QUANT_CACHE_C8_MAX_DIM` 为上限(20B/元素 + 5 个 256B 暂存,运行时 `assert(off <= UB_SIZE)` 守护):

| 缓冲 | dtype | 用途 |
|---|---|---|
| xRow[2] | bfloat16 | 行输入(ping-pong) |
| slotBuf[2] | int32(64) | slot 载入(ping-pong,lane 0 有效) |
| xRowF32 | float | BF16→FP32;复用为 round-trip 的 fp32 中转 |
| absValues | float | `vabs` 中间,`ReduceMax` 结果落点(absmax 在 absValues[0]) |
| rounded | int32 | `vconv_f322s32r` 最近整数中转 |
| halfValues | half | FP32→FP16 中转 |
| quant[2] | int8 | `vconv_f162s8` 结果(ping-pong) |
| scaleF32 | float(64) | `vmuls(absValues, 1/127)` 的 scale 暂存(仅 lane 0 有意义) |
| scaleF16[2] | half(64) | fp16 scale,256B 对齐供 2B MTE3 写出(ping-pong) |

### 流水线同步

[muls](muls.md) 式 ping-pong:**per-slot 事件 ID**(`EVENT_ID0 + curr`)保证归还等待与对应缓冲槽的归还配对——相邻两行用不同槽,两行后才复用,单 ID 会让下一行的等待错配两行前的归还。循环前预置四个归还 flag(`V→MTE2` × 2、`MTE3→V` × 2),`curr` 每行翻转,末尾 drain 四路 + `pipe_barrier(PIPE_ALL)`——空 launch 时循环不执行,四个预置 flag 恰好被 drain 配平,无需特判:

- **载入链**(循环顶部,GM→UB 仅此一处):等 `V→MTE2[curr]`(该槽上一位使用者两行前归还;首两行由预置 flag 满足)后把行数据 + slot 载入槽 curr;`MTE2→S` 同 ID set/wait 对让标量读 slot 判有效性,`MTE2→V` 同 ID set/wait 对放行 `convert_input`,`V→MTE2[curr]` 归还 xRow[curr](供两行后的载入复用);
- **scale/factor 链**:`ReduceMax` 后 `V→S`,S 管线读 `*absValues` 得 absmax,标量算 `factor = 127/absmax`(零行为 0)作为后续 `vmuls` 的立即数——参照 indexer_prepare_c8 的 `rowFactor` 模式,标量只进指令操作数,不经 UB 回流 V;scale 本体在 V 管线算(`vmuls` + `vconv_f322f16`);
- **写出链**:写 scaleF16[curr]/quant[curr] 前等 `MTE3→V[curr]`(两行前的写出已归还);量化链完成后 `V→MTE3` 同 ID set/wait 对,两笔写出(int8 行 + 2B scale)后 `MTE3→V[curr]` 归还——当前行的 MTE3 写出与下一行的 MTE2 载入重叠;
- **无效行中继**:slot 越界的行不计算不写出;load-done 已被循环顶部的 `MTE2→V` 对消费,中继只需归还 `V→MTE2[curr]`、消费并重置 `MTE3→V[curr]`,保持每槽每行恰好一 set/wait 对(漏归还会让两行后的等待错配更早的 flag,产生覆写竞态)。

### 关键计算步骤(`csrc/kernels/quant_cache_c8.h` 的 `quant_cache_c8`)

1. 行 BF16 搬入,`convert_input`(`vconv_bf162f32`)转 FP32(关键字 `convert_input<dtype>(xRowF32`);
2. `vabs` + `ReduceMax(absValues, absValues, dim)` 归约整行 absmax 到 absValues[0](关键字 `vabs(absValues`);
3. S 管线读 absmax 算 factor(零行 → 0);V 管线 `vmuls(scaleF32, absValues, 1/127)` + `vconv_f322f16(scaleF16, scaleF32)` 产出 fp16 scale(关键字 `float factor`);
4. `vmuls(xRowF32, xRowF32, factor)` 缩放,`vconv_f322s32r → vconv_s322f32 → vconv_f322f16 → vconv_f162s8` 四连转 INT8(关键字 `vconv_f322s32r`,与 indexer_prepare_c8 的 K 量化链逐条一致);
5. `copy_ubuf_to_gm_align_b8` 写 int8 行到 `kCache + block*kCacheStride + offset*dim`,`CopyUbufToGmAligned` 写 2B scale 到 `kScaleCache + block*kScaleCacheStride + offset`(关键字 `copy_ubuf_to_gm_align_b8`)。

### 边界处理

- slot < 0 或 ≥ capacity:整行跳过,两 cache 不写(`csrc/kernels/quant_cache_c8.h` 的 `quant_cache_c8`,关键字 `slot < 0`);
- 零行:factor=0 → int8 全零,scale = 0/127 = 0(与 indexer_prepare_c8 的 nonzero 守卫等价);
- dim 必须 64 的倍数(op 检查;`convert_input` 无 tail)且 ≤ `QUANT_CACHE_C8_MAX_DIM`(`csrc/kernels/kernel_param.h`,UB 布局上限,op 检查);
- host 侧 `x.shape[0] == 0` 直接返回(`csrc/op.cpp` 的 `XliteOpQuantCacheC8`)。
