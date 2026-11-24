# engram_gate

## 功能概述

DeepSeek-V4.1 engram post-wkv 融合 gate + 残差 inject 算子(核内模板 `engram_gate`,见 `csrc/kernels/engram_gate.h`)。它是 `vllm_ascend/ops/triton/engram_gate.py` 的 AIV 对应版本。wkv(一个 matmul)在本算子之外运行;本算子消费其 `[T, (HC+1)*DIM]` 输出(`HC*DIM` 的 key 行 + `DIM` 的 value 行),并将结果写回残差流。

核心数学(每 token `t`、每 hc branch `b`,共 `HC` 个 branch):

```text
rstd   = rsqrt(Σh²/DIM+eps) * rsqrt(Σkey²/DIM+eps)
dot    = Σ(h * (qw*kw) * key) * rstd * DIM^-0.5
mag    = sqrt(max(|dot|, 1e-6))
gate   = sigmoid(copysign(mag, dot))   (dot≥0 → +mag,dot<0 → -mag)
gate   = mask[t] ? gate : 0
out[t, b] = hResid[t, b] + gate * value[t]
```

数学顺序严格镜像 `engram_gate.py`(`enable_fp_fusion=False`):残差 FMA `h + gate*value` 保持为独立的 `vmul`+`vadd`,不收缩为单条 FMA。

## 双基分离(rotate 路径的核心设计)

V4.1 的 engram gate 数学与残差 inject **作用在不同的基上**。这是 `engram_gate.h` 需要两个独立 hidden 输入(`hidden` 与 `hiddenResidual`)的根本原因。

gate 数学(rstd/dot)活在 **original(未旋转)基**,残差 inject 活在 **rotated 基**:

| 量 | 基 | kernel 里如何落地 |
|----|----|-------------------|
| hidden(gate math 的 rstd/dot) | original(未旋转) | `hidden` 参数 = gate-math 基 |
| key | original | kv 前半段,原样用 |
| qw, kw | original | 原样用 |
| gate 数学结果 | original | 在 original 基算完 |
| resid(inject 基) | rotated | `hiddenResidual` 参数 = residual(rotated 基) |
| value | rotated | kv 尾段,直接加到 rotated resid |

caller 负责在 kernel 之外把 hidden 派生到 original 基(对 rotated checkpoint 先 `@ R.T`)、把 residual 留在 rotated 基,再把两者喂给本 op。模型侧如何派生(rotate 总开关 `value_projection_rotated`、`engramRotation` 哨兵、HF config 的 `engram_rotation_config` 字段)属模型集成范畴,待对应 PR 落地时补充链路文档。

当调用方对两个 hidden 传同一张量(unrotated 路径,`hiddenResidual == nullptr`),kernel 退化为仅 AIV 的单 hidden 参考实现:`hiddenResidual` 置空 → `gmHResid` 别名 `gmH` → `resIn` 槽从 `hidden` 搬运(内容 == `hIn`),FMA 注入基即 `hIn` 内容。

## 输入输出参数

Python 侧调用(`xlite/_C.pyi`,测试 `tests/kernels/engram_gate.py` 的 `engram_gate(...)` 调用(关键字 `engram_gate(rt, hidden, kv=`)):

```python
engram_gate(
    rt, hidden, kv, q_weight, k_weight, mask, out, hc_mult, dim, eps, hidden_residual=...
)  # 省略 → 空 → nullptr → alias hidden
```

host 侧 launch 见 `csrc/op.cpp` 的 `XliteOpEngramGate`(以 `PickMinBlockNum(rt.aivNum, totalWork)` 个 block 启动,`totalWork = numTokens * hcMult`)。kernel 签名(`csrc/kernels/engram_gate.h` 的 `ENGRAM_GATE_FUNC_DEFINE` 宏,关键字 `engram_gate_##dtype`):

```cpp
engram_gate_<dtype>(GM_ADDR hidden, GM_ADDR hiddenResidual, GM_ADDR kv, GM_ADDR qw, GM_ADDR kw,
                    GM_ADDR mask, GM_ADDR out, uint32_t numTokens, uint32_t hcMult, uint32_t dim,
                    float eps)
```

| GM 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| hidden | 输入 | `[T, HC, DIM]` | bf16 | gate-math basis(unrotated 路径下与 resid 同基);contiguous(stride-1 on DIM) |
| hiddenResidual | 输入 | `[T, HC, DIM]` / 空 | bf16 | residual-stream basis(may == hidden);nullptr ⇒ alias hidden(`gmHResid` 别名 `gmH`,`resIn` 从 `hidden` 搬运) |
| kv | 输入 | `[T, (HC+1)*DIM]` | bf16 | **interleaved,非 `[T, HC, DIM]`**:见下文 |
| qw, kw | 输入 | `[HC, DIM]` | bf16 | per-branch 常量(依赖 branch,不依赖 token) |
| mask | 输入 | `[T]` | int32 | nonzero = active(bool/uint8 不受 XDtypeOf 支持) |
| out | 输出 | `[T, HC, DIM]` | bf16 | `hResid + gate[t,b]*value[t]`;caller 可原地写回 residual(`out == hiddenResidual`) |

**kv 的 interleaved 布局**(与 hidden 的 `[T, HC, DIM]` 不同):

```text
kv[t] = key[t,0:DIM] | key[t,1:DIM] | ... | key[t,HC-1:DIM] | value[t,:DIM]
        |<-------------- HC*DIM ------------>|  |<------ DIM ------>|
```

- key 是 per-branch(`[T, HC, DIM]`),按 branch 偏移取址。
- value 是 **shared across ALL branches**(逻辑 `[T, DIM]`,broadcast 到 `[T, HC, DIM]` 经 `gate[t,b]*value[t]`)。
- 指针算术:`gmKey = kv + t*(HC+1)*DIM + branch*DIM`(per-branch 偏移);`gmVal = kv + t*(HC+1)*DIM + HC*DIM`(固定尾部,所有 branch 共用同一行 value)。value 不随 branch 变是省带宽的关键。

对应 triton kernel(`engram_gate.py:34-48`)第 47 行 `value = load(kv + token*stride + HC*DIM + col)`,以及测试 oracle(`tests/kernels/engram_gate.py:38`)的 `kv[:, HC*DIM:]` 取 `[T, DIM]`。

## 每个 work item 的量

一个 work item = 一个 (token, branch) pair。kernel 内同时只活一个 work item。

**向量量**(全 DIM 宽,活于 UB ping-pong 槽):

| 量 | 逻辑 shape | UB buffer | dtype 流向 |
|----|-----------|-----------|-----------|
| h(hidden, gate-math basis) | `[DIM]` | `hIn[curr]` | bf16 → fp32(calcH) |
| key | `[DIM]` | `keyIn[curr]` | bf16 → fp32(calcKey) |
| value | `[DIM]` | `valIn[curr]` | bf16 → fp32(FMA 时转) |
| hResid(residual basis) | `[DIM]` | `resIn[curr]` | bf16 → fp32(resIn 恒用独立槽;nullptr 时 resIn 从 gmH 搬运) |
| qw, kw | `[DIM]`(本 branch) | `qwIn`,`kwIn`(两个独立 bf16 槽)→ `qwF`/`calcKey` | bf16 → fp32 |
| qw·kw | `[DIM]` | `qwF` | fp32(reused across all tokens of a branch) |
| out = hResid + gate·value | `[DIM]` | `outIn[curr]`(转回 bf16 后) | fp32 → bf16 |

**标量量**(单 fp32,从全 DIM 向量 reduce 出,每个 (t,b) 产生一个):

```text
hh=Σh²  kk=Σkey²  hk=Σ(h·(qw·kw)·key)  rstdH rstdK rstd dot mag gate
```

每个标量落到独立的 32B UB word(`scalarRstdH`/`scalarRstdK`/`scalarMag`/`scalarGate`),使 V 的下一次写不会与 S 的待读 race。sigmoid 用标量除法 `1.0f / *scalarGate`(engram_gate.h:296),无需 ones 向量缓存。

## UB 预算

`__DAV_C220_VEC__`(2201/A3,`__NPU_ARCH__ != 3510`)下 `UB_SIZE = 196608` (192KB)。取 `Dtype=bf16`(2B)、`DIM=5120`(V4.1 hidden_size):`dtypeLen = ROUND_UP(5120×2, 32) = 10240` B (10KB);`f32Len = ROUND_UP(5120×4, 32) = 20480` B (20KB);标量槽 = `ROUND_UP(4, 32) = 32` B。

| # | buffer | 类型 | 大小 | 用途 |
|---|--------|------|------|------|
| 1-2 | hIn0/1 | bf16 | 10KB×2 | hidden 行 ping-pong |
| 3-4 | keyIn0/1 | bf16 | 10KB×2 | key 行 ping-pong |
| 5-6 | valIn0/1 | bf16 | 10KB×2 | value 行 ping-pong |
| 7-8 | resIn0/1 | bf16 | 10KB×2 | residual 行 ping-pong(恒占位;nullptr 时从 gmH 搬运,内容 == hIn) |
| 9-10 | outIn0/1 | bf16 | 10KB×2 | 输出暂存 ping-pong |
| 11 | qwIn | bf16 | 10KB | qw dtype 暂存(ID4) |
| 12 | kwIn | bf16 | 10KB | kw dtype 暂存(ID4,与 qwIn 共享 slot 计数信号量) |
| 13 | qwF | fp32 | 20KB | qw·kw 预乘 |
| 14 | calcH | fp32 | 20KB | h 侧计算 scratch |
| 15 | calcKey | fp32 | 20KB | key 侧计算 scratch |
| 16-19 | scalarRstdH/K/Mag/Gate | fp32 | 32B×4 | V→S 标量落点 |

汇总:bf16 槽 = `12 × 10240 = 122880` B (120KB);fp32 scratch = `3 × 20480 = 61440` B (60KB);标量 = `4 × 32 = 128` B。**`off` 终值 = 184448 B (180.125KB)**;**`UB_SIZE - off = 12160 B (11.875KB)`**。`assert(off <= UB_SIZE)` 通过,余量 12160B(与 host 侧 `EngramGateUbNeed` 一致)。

几个浪费点(若后续要腾空间):4 个标量槽各占 32B 实际只用 4B(对齐必须,浪费 112B);`resIn0/1`(20KB)在 nullptr 路径搬运的内容 == hIn(多一次 10KB GM→UB DMA,结果正确但冗余)。

## rotate / model-layer 集成链路

本 PR 只含 standalone op(kernel + host op + 绑定 + 测试),不含 `xlite.py`/`_C.cpp`/`model.cpp` 的模型侧集成。kernel 无 `rotated` 分支:`hiddenResidual == nullptr` 时 `gmHResid` 别名 `gmH`、`resIn` 从 `hidden` 搬运(见上文 UB 预算与 kernel 注释),非空时按 `hiddenResidual` 取址。模型侧如何派生 un-rotated hidden / rotated residual 并调用本 op(rotate 总开关 `value_projection_rotated`、`engramRotation` 哨兵、`XliteOpEngramGate` 在 `model.cpp` 的调用点)属模型集成范畴,待对应 PR 落地时补充链路文档。

## 测试

`tests/kernels/engram_gate.py` 以纯 torch oracle(`torch_engram_gate` / `torch_engram_gate_rotated`,镜像 `engram_gate.py:34-48` 的未融合 FP32 顺序)做验收,`rtol=1e-2, atol=1e-3`,覆盖:

- **unrotated 路径**(`hidden_residual` 省略 → nullptr → alias hidden,退化为仅 AIV 单 hidden 参考实现):`run_test`,case 覆盖 decode(T=1)、prefill(T=8, 41)、小 DIM=128 reduce sanity,bf16 only。
- **rotated 路径**(distinct hidden vs hidden_residual,mirrors `value_projection_rotated`):`run_test_rotated`,构造 un-rotated hidden(`hidden_rot @ rot`)与 rotated residual 两个独立张量喂给 kernel。
- 三种 mask 模式:`all` / `partial` / `none`。
