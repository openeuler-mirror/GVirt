# softmax

## 功能概述

按行（last dim）做数值稳定的 softmax：`y[i, :] = exp(x[i, :] - max) / Σ exp(x[i, :] - max)`，只对每行前 `calcLen`（可含 causal mask 偏移）个元素计算，其余位置清零，原地写回。提供两个入口：常规 `softmax`（整行一次性放入 UB，乒乓流水）与 `softmax_long`（行长超出 UB 时按子块分段 + GM expBuf 中间缓存，支持最长约 2M 元素的行）。该 kernel 亦被 attention（flash attention 的行内 softmax、稀疏 top-k gather/scatter）复用（`RunAivSoftmaxPingPong` 带 topK/topkIndices 扩展参数），本算子的入口只用到基础路径。

## 输入输出参数

| 参数 | 方向 | Shape | Dtype | 说明 |
|------|------|-------|-------|------|
| x | 输入/输出（原地） | [m, n] | float16 / bfloat16 | 每行 n 个元素，m 为行数；host 侧传 `x.shape[0]`=m、`x.shape[1]`=n（`csrc/op.cpp` 的 `XliteOpSoftmax` / `XliteOpSoftmaxLong`，关键字 `launchKernel(1,`） |
| expBuf | 中间缓冲（仅 long） | [1, n] | float32 | 分段子块 exp(x-max) 的 GM 暂存；Python 侧由 runtime tensor pool 分配（`csrc/_C.cpp` 的 `Softmax` wrapper，关键字 `rt.GetTensor({1, _x.shape[1]}, FP32`） |
| calcLen (contextLen) | 标量 | - | uint32_t | 每行实际参与 softmax 的基础长度；行 i 的有效长度为 calcLen（+可选 mask 偏移），超出部分写 0 |
| isLong | 标量（Python 参数） | - | bool | 选择 `softmax`（False）或 `softmax_long`（True）入口（`tests/kernels/softmax.py` 的 `softmax(rt, y, size, ...)` 调用段） |

Python 调用方式（`tests/kernels/softmax.py` 的 `softmax(rt, y, size, ...)` 调用段）：`softmax(rt, y, size, False)` / `softmax(rt, y, size, True)`。测试中行尾用 `-inf` 填充并与 `torch.nn.functional.softmax` 对比。

注意：入口固定单 block launch（`launchKernel(1, ...)`，`csrc/op.cpp` 的 `XliteOpSoftmax` / `XliteOpSoftmaxLong`），行间串行；行内由向量指令并行。

## 支持的数据类型

| dtype | 实例化文件 | kernel 符号 |
|-------|-----------|-------------|
| float16_t | `csrc/kernels/softmax.cpp` | `softmax_float16_t` / `softmax_long_float16_t` |
| bfloat16_t | `csrc/kernels/softmax.cpp` | `softmax_bfloat16_t` / `softmax_long_bfloat16_t` |

核心实现为模板头文件 `csrc/kernels/softmax_attn_aiv.h`（`RunAivSoftmaxPingPong` / `RunAivSoftmaxLong`），`softmax.cpp` 用宏实例化 fp16/bf16 两个变体；仅支持 `__DAV_C220_VEC__`（头文件内有 UB 容量 static_assert，`csrc/kernels/softmax_attn_aiv.h` 的 `static_assert` 块，关键字 `MAX_SOFTMAX_PINGPONG_LEN ... <= UB_SIZE`）。

## 实现原理

### 常规路径：RunAivSoftmaxPingPong（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`）

对每行 idx（0..m）：

1. **有效长度**：`actualCalcLen = calcLen + seqIdx`（seqIdx 由 maskOff/maskStride/seqHead 推出，本入口均为默认值 0/1/true，故 = calcLen；master 上该函数另为 cxa 算子新增 winSize/winCalcLen/compressRatio/attnSink/swaSegWidth 等 SWA/compress/attn_sink 扩展参数，本入口取默认值时行为不变，仅当 swaSegWidth>0 时 actualCalcLen 改按 `swaSegWidth + (calcLen+seqIdx)/compressRatio` 计算），钳到 [0, outN]（outN=n）（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `actualCalcLen = calcLen + seqIdx`）；`actualCalcLen <= 0` 时整行写 0 直接返回；
2. **MTE2**：搬入行前 actualCalcLen 个元素到 `in[curr]`（乒乓双缓冲，`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `copy_gm_to_ubuf` 行搬入段）；
3. **V 计算**（全部在 FP32 进行）：
   - `vconv_f162f32`/`vconv_bf162f32`：升精度到 `cal`（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `vconv_f162f32` 升精度段）；
   - `ReduceMaxV2`：行最大值；`vbrcb` 把标量广播成一个 block 的向量避免标量运算（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `ReduceMaxV2` / `vbrcb`）；
   - `vsub`（x−max）→ `vexp` → `ReduceSumV2`（和）→ `vbrcb` 广播 → `vdiv`（归一化）（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `vsub` / `vexp` / `ReduceSumV2` / `vdiv`）；
   - `vconv_f322f16r`/`vconv_f322bf16r` 舍回原 dtype 存 `out[curr]`（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `vconv_f322f16r` 舍回段）；
4. **尾零**：outN > actualCalcLen 时用 `SetMaskFromHighBit` + `vector_dup` 把行尾补 0（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `SetMaskFromHighBit` / `vector_dup` 尾零段）；
5. **MTE3**：整行 outN 个元素写回 `buf + idx*n`（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `copy_ubuf_to_gm` 写回段）。

UB 布局：`in[2]`/`out[2]`（乒乓）+ FP32 计算缓冲 `cal`/`temp` + 可选 max/sum/topk 缓冲（本入口 saveMaxSum=false、topK=0，均不分配），长度上界由 `MAX_SOFTMAX_PINGPONG_LEN = 10496`（`csrc/kernels/kernel_param.h` 的 `MAX_SOFTMAX_PINGPONG_LEN` 宏)约束。该值按最坏特性组合（saveMaxSum + topK + attnSink 同时启用）校准，且以 **SWA + topK 场景为绑定约束**（行宽缓冲之外再叠加 `topK + swaSegWidth` 宽的 `inTopK`）：`MAX_TOPK_NUM=2048`、`MAX_SWA_SEG_WIDTH=4096` 时 SWA 全开布局 195072B ≤ UB_SIZE(196608B)（无 SWA 全开 186880B），static_assert 见 `csrc/kernels/softmax_attn_aiv.h` 的 UB 预算注释块。SWA 场景的行宽 `outN = swaSegWidth + 压缩段宽` 由 host（flash_cxa：`swaSegWidth + tileSizeOfCachedKV`）或 Python 层（cxa：`tests/kernels/cxa.py` 的 max_row）保证 ≤ 10496，`swaSegWidth ≤ MAX_SWA_SEG_WIDTH` 由 `XliteOpCXA`/`XliteOpFlashCXA` 保证（此前断言以 topK+swaSegWidth=6144 为基准，未覆盖 outN 主导的常态，11776/11136 旧值在 SWA+topK 组合下越界）。

流水线：`PIPE_V ↔ PIPE_MTE2` 的 EVENT_ID2/3 管输入乒乓，`PIPE_V → PIPE_MTE3 → PIPE_V` 的 EVENT_ID0 管输出写回与缓冲归还，使第 idx+1 行的 MTE2 搬入与第 idx 行的 MTE3 写回、V 计算重叠（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `EVENT_ID2` / `EVENT_ID3` / `EVENT_ID0` 同步段）。

### 长行路径：RunAivSoftmaxLong（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`）

当行长大到单次放不下 UB（单子块上限 `MAX_SUB_CONTEXT_SIZE = (VECTOR_MAX_REPEAT-1) * 64` 个 fp32，即 254×64=16256；子块数 `subBlockNum = DIV_ROUND_UP(outN, MAX_SUB_CONTEXT_SIZE)`）时启用，两趟处理：

- **stage 1（max & 部分和）**（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`，关键字 `ReduceMaxV2` / `totalMax` / `totalSum` 的 stage1 段）：逐子块搬入→升精度→`ReduceMaxV2/ReduceMax` 求块内 max；多子块时维护 `max[]`/`sum[]` 数组与全局 `totalMax/totalSum`，新块 max 超过历史 max 时用 `exp(lastMax-totalMax)` 修正历史部分和（在线 softmax 合并），每块的 `exp(x-max)` 写入 GM `expBuf` 暂存（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`，关键字 `expBuf` 写入段）；
- **stage 2（归一化）**（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`，关键字 `maxFlag` / `vdiv` 的 stage2 段）：逐子块从 expBuf 读回 fp32，`maxFlag[block]` 标记的块乘 `exp(curMax-totalMax)` 修正，`vdiv` 除以 totalSum，转回 dtype 写回原行；末尾子块的尾零与未覆盖子块的整块清零处理（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`，关键字 `vector_dup` 尾零/整块清零段）。

UB 布局只保留单份 `in`/`out`（各 MAX_SUB_CONTEXT_SIZE）+ FP32 `calc`/`calcDst` + 每子块 max/sum 槽 + 可选 sink 槽（attnSink 非空时才分配，见 `csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`，关键字 `sink`）；stage1/stage2 内部各用 EVENT_ID0/1/3/4/5 做块间 MTE2/MTE3/V 握手（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong`，关键字 `EVENT_ID0` / `EVENT_ID1` / `EVENT_ID3` / `EVENT_ID4` / `EVENT_ID5` 握手段）。测试覆盖至行长 2064512（`tests/kernels/softmax.py` 的 2064512 长行测试用例），与头文件注释的上限一致（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxLong` 上方注释，关键字 `subBlockNum <= 127` / `n <= 2064512`；启用 attnSink 的 cxa 路径少一个 sink 槽的余量，上界收紧为 126 / 2048256）。

### 复用扩展（本入口未启用）

`RunAivSoftmaxPingPong` 的模板参数还支持 hasScale/scale（先乘缩放）、saveMaxSum/maxBuf/sumBuf（输出行 max/sum 供 online softmax 合并）、topK/topkIndices（稀疏 attention：按命中位图 `vgather` 收集分数、softmax 后标量 scatter 回原位置，`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxPingPong`，关键字 `vgather` / `topkIndices` 扩展段）；另为 cxa 算子新增 winSize/winCalcLen/compressRatio/attnSink/swaSegWidth 等 SWA/compress/attn_sink 扩展参数；同头文件的 `RunAivSoftmaxUpdate`（`csrc/kernels/softmax_attn_aiv.h` 的 `RunAivSoftmaxUpdate`）实现 flash attention 的跨 KV 块在线 softmax 合并。这些路径由 attention/mla 相关 kernel 使用。

### 片上缓冲分配与占用汇总（AIV UB）

两个入口均无常驻 `Init` 排布——`RunAivSoftmaxPingPong` 与 `RunAivSoftmaxLong`（`csrc/kernels/softmax_attn_aiv.h`）在函数入口按运行期行长现场构建一次性 UB 布局（常规路径 `len = outN`；长行路径输入/输出/计算槽按 `MAX_SUB_CONTEXT_SIZE` 定宽，`max`/`sum` 随子块数缩放），launch 内所有行共用。硬件容量基准（910B/dav C220，NpuArch 2201）：UB 预算取 `UB_SIZE=192KB`（`csrc/kernels/kernel_macro.h` 的 `UB_SIZE`，4KB×48 bank），单 block launch，每个 AIV subblock 各持一份独立 UB、布局相同。dtype 记法：`Dtype` 为 x 元素类型（fp16/bf16，均 2B），UB 布局与 dtype 无关，单占用列。本算子入口仅走基础路径（`saveMaxSum=false`、`topK=0`、`attnSink=nullptr`），maxOut/sumOut/indices/inTopK/cmp/topkMask/sink 等扩展槽全部为 0 尺寸、不占槽位。典型配置：常规路径取 n=13952（m=1，测试常规路径最大 128 对齐行）；长行路径取 n=2064512（m=1，头注释行长上界，`totalSubBlockNum=127`）。

**AIV UB——常规路径 `RunAivSoftmaxPingPong`（预算 `UB_SIZE=192KB`，n=13952）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `in[2]` | 13952 Dtype | Dtype | 2（V↔MTE2 事件 ID2+curr） | `ROUND_UP(len*sizeof(Dtype), VECTOR_MAX_BYTESIZE)` | 55808 |
| `out[2]` | 13952 Dtype | Dtype | 2（事件 ID0 写出、ID2+curr 归还） | `ROUND_UP(len*sizeof(Dtype), VECTOR_MAX_BYTESIZE)` | 55808 |
| `cal` | 13952 float | float | 1 | `ROUND_UP(len*sizeof(float), VECTOR_MAX_BYTESIZE)` | 55808 |
| `temp` | 6976 float（109 个 256B 块） | float | 1 | `(DIV_ROUND_UP(len*4, 256)/2)*256` | 27904 |
| **合计** | | | | | **195328** |

**AIV UB——长行路径 `RunAivSoftmaxLong`（预算 `UB_SIZE=192KB`，n=2064512，`MAX_SUB_CONTEXT_SIZE=16256`，子块数 127）**

| 变量 | shape（每份） | dtype | 乒乓个数 | 单份大小 | 占用（B） |
|---|---|---|---|---|---|
| `in` | 16256 Dtype | Dtype | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `out` | 16256 Dtype | Dtype | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(Dtype)` | 32512 |
| `calc` | 16256 float | float | 1 | `MAX_SUB_CONTEXT_SIZE*sizeof(float)` | 65024 |
| `calcDst` | 256 float（4 个 256B 块） | float | 1 | 子块数>65：`DIV_ROUND_UP(254,64)*256`；≤65：`127*256` | 1024 |
| `max` | 128 个 256B 块（每子块 max 槽 + totalMax） | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 32768 |
| `sum` | 128 个 256B 块（每子块 sum 槽 + totalSum） | float | 1 | `(totalSubBlockNum+1)*VECTOR_MAX_BYTESIZE` | 32768 |
| `sink` | 64 float（无条件预留，仅 attnSink 复用方写入） | float | 1 | `VECTOR_MAX_BYTESIZE` | 256 |
| **合计** | | | | | **196864** |

**占用汇总**（各表占用列之和 vs 硬件容量；字节数）：

| 存储级 | 硬件容量 | 占用 | 占比 |
|---|---|---|---|
| UB（常规路径，n=13952） | 192 KB | 195328 B | 99.3% |
| UB（长行路径，n=2064512） | 192 KB | 196864 B | 100.1% |

要点：

- **常规路径贴满 UB**：`in/out` 四个 Dtype 行缓冲合计 111616 B 是大头；本入口无扩展槽时布局的行长极限为 n=14016（恰 196608 B 顶满 192KB），典型配置 n=13952 已达 99.3%，更长行必须切 `softmax_long`；
- **容量来源是头文件 `static_assert`**：`csrc/kernels/softmax_attn_aiv.h` 的 `static_assert` 块按 `MAX_SOFTMAX_PINGPONG_LEN=10496` 且计入全部 topK/saveMaxSum/sink 扩展槽验算（SWA+topK 绑定场景 195072 B、无 SWA 满配 186880 B，均 ≤ `UB_SIZE`，前者余 1536 B）作保；本入口扩展槽全空，故实际行长可放宽到 14016（恰占满 196608 B）；
- **长行路径在 127 子块的行长上界处标称越界 256 B**：`RunAivSoftmaxLong` 上方头注释的容量公式未计入无条件预留的 `sink` 槽——按该公式 127 子块恰 196608 B 顶满，含 `sink` 后布局达 196864 B；本入口 `attnSink=null` 不写 `sink`，实际访问不越界，但启用 attnSink 的复用方在该极端行长下写 `sink` 会越过 UB 标称边界；
- `temp`（常规）/`calcDst`（长行）是 `ReduceMaxV2`/`ReduceSumV2` 的部分和暂存，按 repeat 数约减半定宽。
