# softmax

## 功能概述

按行（last dim）做数值稳定的 softmax：`y[i, :] = exp(x[i, :] - max) / Σ exp(x[i, :] - max)`，只对每行前 `calcLen`（可含 causal mask 偏移）个元素计算，其余位置清零，原地写回。提供两个入口：常规 `softmax`（整行一次性放入 UB，乒乓流水）与 `softmax_long`（行长超出 UB 时按子块分段 + GM expBuf 中间缓存，支持最长约 2M 元素的行）。该 kernel 亦被 attention（flash attention 的行内 softmax、稀疏 top-k gather/scatter）复用（`RunAivSoftmaxPingPong` 带 topK/topkIndices 扩展参数），本算子的入口只用到基础路径。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入/输出（原地） | [m, n] | float16 / bfloat16 | 每行 n 个元素，m 为行数；host 侧传 `x.shape[0]`=m、`x.shape[1]`=n（`csrc/op.cpp:1360,1377`） |
| expBuf | 中间缓冲（仅 long） | [1, n] | float32 | 分段子块 exp(x-max) 的 GM 暂存；Python 侧由 runtime tensor pool 分配（`csrc/_C.cpp:1828-1830`） |
| calcLen (contextLen) | 标量 | - | uint32_t | 每行实际参与 softmax 的基础长度；行 i 的有效长度为 calcLen（+可选 mask 偏移），超出部分写 0 |
| isLong | 标量（Python 参数） | - | bool | 选择 `softmax`（False）或 `softmax_long`（True）入口（`tests/kernels/softmax.py:29,52`） |

Python 调用方式（`tests/kernels/softmax.py:29`）：`softmax(rt, y, size, False)` / `softmax(rt, y, size, True)`。测试中行尾用 `-inf` 填充并与 `torch.nn.functional.softmax` 对比。

注意：入口固定单 block launch（`launchKernel(1, ...)`，`csrc/op.cpp:1360,1377`），行间串行；行内由向量指令并行。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/softmax.cpp` | `softmax_float16_t` / `softmax_long_float16_t` |
| bfloat16_t | `csrc/kernels/softmax.cpp` | `softmax_bfloat16_t` / `softmax_long_bfloat16_t` |

核心实现为模板头文件 `csrc/kernels/softmax_attn_aiv.h`（`RunAivSoftmaxPingPong` / `RunAivSoftmaxLong`），`softmax.cpp` 用宏实例化 fp16/bf16 两个变体；仅支持 `__DAV_C220_VEC__`（头文件内有 UB 容量 static_assert，`csrc/kernels/softmax_attn_aiv.h:17-63`）。

## 实现原理

### 常规路径：RunAivSoftmaxPingPong（`csrc/kernels/softmax_attn_aiv.h:64-506`）

对每行 idx（0..m）：

1. **有效长度**：`actualCalcLen = calcLen + seqIdx`（seqIdx 由 maskOff/maskStride/seqHead 推出，本入口均为默认值 0/1/true，故 = calcLen；master 上该函数另为 cxa 算子新增 winSize/winCalcLen/compressRatio/attnSink/swaSegWidth 等 SWA/compress/attn_sink 扩展参数，本入口取默认值时行为不变，仅当 swaSegWidth>0 时 actualCalcLen 改按 `swaSegWidth + (calcLen+seqIdx)/compressRatio` 计算），钳到 [0, outN]（outN=n）（`csrc/kernels/softmax_attn_aiv.h:160-179`）；`actualCalcLen <= 0` 时整行写 0 直接返回；
2. **MTE2**：搬入行前 actualCalcLen 个元素到 `in[curr]`（乒乓双缓冲，`csrc/kernels/softmax_attn_aiv.h:181-182`）；
3. **V 计算**（全部在 FP32 进行）：
   - `vconv_f162f32`/`vconv_bf162f32`：升精度到 `cal`（`csrc/kernels/softmax_attn_aiv.h:284-288`）；
   - `ReduceMaxV2`：行最大值；`vbrcb` 把标量广播成一个 block 的向量避免标量运算（`csrc/kernels/softmax_attn_aiv.h:352-357`）；
   - `vsub`（x−max）→ `vexp` → `ReduceSumV2`（和）→ `vbrcb` 广播 → `vdiv`（归一化）（`csrc/kernels/softmax_attn_aiv.h:374-405`）；
   - `vconv_f322f16r`/`vconv_f322bf16r` 舍回原 dtype 存 `out[curr]`（`csrc/kernels/softmax_attn_aiv.h:408-412`）；
4. **尾零**：outN > actualCalcLen 时用 `SetMaskFromHighBit` + `vector_dup` 把行尾补 0（`csrc/kernels/softmax_attn_aiv.h:475-487`）；
5. **MTE3**：整行 outN 个元素写回 `buf + idx*n`（`csrc/kernels/softmax_attn_aiv.h:491`）。

UB 布局：`in[2]`/`out[2]`（乒乓）+ FP32 计算缓冲 `cal`/`temp` + 可选 max/sum/topk 缓冲（本入口 saveMaxSum=false、topK=0，均不分配），长度上界由 `MAX_SOFTMAX_PINGPONG_LEN = 11776`（`csrc/kernels/kernel_param.h:43`)约束。

流水线：`PIPE_V ↔ PIPE_MTE2` 的 EVENT_ID2/3 管输入乒乓，`PIPE_V → PIPE_MTE3 → PIPE_V` 的 EVENT_ID0 管输出写回与缓冲归还，使第 idx+1 行的 MTE2 搬入与第 idx 行的 MTE3 写回、V 计算重叠（`csrc/kernels/softmax_attn_aiv.h:156-160,181,489-498`）。

### 长行路径：RunAivSoftmaxLong（`csrc/kernels/softmax_attn_aiv.h:801-1145`）

当行长大到单次放不下 UB（单子块上限 `MAX_SUB_CONTEXT_SIZE = (VECTOR_MAX_REPEAT-1) * 64` 个 fp32，即 255×64=16320；子块数 `subBlockNum = DIV_ROUND_UP(outN, MAX_SUB_CONTEXT_SIZE)`）时启用，两趟处理：

- **stage 1（max & 部分和）**（`csrc/kernels/softmax_attn_aiv.h:876-1053`）：逐子块搬入→升精度→`ReduceMaxV2/ReduceMax` 求块内 max；多子块时维护 `max[]`/`sum[]` 数组与全局 `totalMax/totalSum`，新块 max 超过历史 max 时用 `exp(lastMax-totalMax)` 修正历史部分和（在线 softmax 合并），每块的 `exp(x-max)` 写入 GM `expBuf` 暂存（`csrc/kernels/softmax_attn_aiv.h:1006-1049`）；
- **stage 2（归一化）**（`csrc/kernels/softmax_attn_aiv.h:1060-1120`）：逐子块从 expBuf 读回 fp32，`maxFlag[block]` 标记的块乘 `exp(curMax-totalMax)` 修正，`vdiv` 除以 totalSum，转回 dtype 写回原行；末尾子块的尾零与未覆盖子块的整块清零处理（`csrc/kernels/softmax_attn_aiv.h:1101-1139`）。

UB 布局只保留单份 `in`/`out`（各 MAX_SUB_CONTEXT_SIZE）+ FP32 `calc`/`calcDst` + 每子块 max/sum 槽；stage1/stage2 内部各用 EVENT_ID0/1/3/4/5 做块间 MTE2/MTE3/V 握手（`csrc/kernels/softmax_attn_aiv.h:848-851`）。测试覆盖至行长 2064512（`tests/kernels/softmax.py:41`），与头文件注释的上限一致（`csrc/kernels/softmax_attn_aiv.h:795-799`）。

### 复用扩展（本入口未启用）

`RunAivSoftmaxPingPong` 的模板参数还支持 hasScale/scale（先乘缩放）、saveMaxSum/maxBuf/sumBuf（输出行 max/sum 供 online softmax 合并）、topK/topkIndices（稀疏 attention：按命中位图 `vgather` 收集分数、softmax 后标量 scatter 回原位置，`csrc/kernels/softmax_attn_aiv.h:190-274,415-473`）；另为 cxa 算子新增 winSize/winCalcLen/compressRatio/attnSink/swaSegWidth 等 SWA/compress/attn_sink 扩展参数；同头文件的 `RunAivSoftmaxUpdate`（`csrc/kernels/softmax_attn_aiv.h:563`）实现 flash attention 的跨 KV 块在线 softmax 合并。这些路径由 attention/mla 相关 kernel 使用。
