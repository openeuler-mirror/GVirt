# fused_quant_matmul_dequant

## 功能概述

融合三段式 W8A8 dense 管线（quant → matmul → dequant）为单次 launch：AIV 侧先把 BF16 激活量化为 INT8（静态 per-channel 或动态 per-token 二选一），AIC 侧做 INT8 matmul（FixPipe 乘 deq_scale 反量化输出 FP16），AIV 侧再反量化为 BF16。相比 [fusion_operator_matmul_dequant_pipeline](fusion_operator_matmul_dequant_pipeline.md)（输入已是 INT8），本算子把上游量化 kernel 也吸收进来，省去两次独立 launch。数学语义与串行链路 `quant/quant_dynamic → matmul_dequant → dequant` 一致。

```
AIV t (全部):  quant(x 行段 [t*chunk, (t+1)*chunk)) → xQuant (+ per-token scale → outScale, dyn)
               --SyncAll<true> → V2C flag 9 (每 AIV set 一次, AIC wait 两次)-->
AIC b:         grid-stride matmul tiles → out (FP16 视图)
               --C2V flag 8 per tile (挂 PIPE_FIX, 骑 CopyToGm 之后)-->
AIV 2b/2b+1:   dequant(out FP16 → BF16 原位, dyn 乘 outScale) --V2C_DONE flag 10-->
```

## 输入输出参数

Python 入口 `fused_quant_matmul_dequant(rt, x, quant_scale_rec, quant_offset, weight, out, bias, deq_scale, x_quant, out_scale, num, dyn_quant, weight_nz, transpose)`（`csrc/_C.cpp`）。kernel 签名见 `FUSED_QUANT_MATMUL_DEQUANT_FUNC_DEFINE` 宏（[csrc/kernels/fused_quant_matmul_dequant.h](../../csrc/kernels/fused_quant_matmul_dequant.h)）:

| 参数 | 方向 | Shape | Dtype | 说明 |
|---|---|---|---|---|
| x | 输入 | `[m, k]` | bfloat16 | 原始激活 |
| quant_scale_rec / quant_offset | 输入 | `[k]` | bfloat16 | 静态模式量化 scale 倒数/偏置；dyn 模式传空 |
| weight | 输入 | `[n, k]`（或转置/NZ） | int8 | 权重 |
| x_quant | workspace | `[m, k]` | int8 | 量化输出（AIV 写、AIC 读），host 分配 |
| out | 输出 | `[m, n]` | bfloat16 | 结果（AIC 写 FP16 视图、AIV 原位覆盖 BF16） |
| bias | 输入(可选) | `[n]` | int32 | matmul bias |
| deq_scale (weight_scale) | 输入(可选) | `[2n]` | float32 | FixPipe 反量化 scale（FP32 按 uint64 交错存储，同 matmul） |
| out_scale | workspace | `[m]` | float32 | dyn 模式 per-token scale（quant 写、dequant 读）；静态模式传空 |
| num | 输入(可选) | `[1]` | uint32 | 动态 token 数，dequant 侧行收缩（同独立 dequant） |
| dyn_quant | 标量 | - | bool | true=动态 per-token；false=静态 per-channel |
| weight_nz / transpose | 标量 | - | bool | 权重布局 |
| m0/n0/k0 | 标量 | - | uint64 | tiling，默认走 `PickMatmulTiling`（与独立 matmul 一致） |

测试参考：[tests/kernels/fused_quant_matmul_dequant.py](../../tests/kernels/fused_quant_matmul_dequant.py)，golden 为串行三 kernel 链，覆盖 nz/transpose/dyn/static 交叉、quant 尾路径与长序列。perf 参考：[tests/performance/perf_fused_quant_matmul_dequant.py](../../tests/performance/perf_fused_quant_matmul_dequant.py)。

## 支持的数据类型

- x `bfloat16_t`、weight `int8_t`、out `bfloat16_t`；matmul 累加 `int32_t`、中间输出 `half`
- kernel 入口符号 `fused_quant_matmul_dequant_int8_t`（host 经 `aclrtlaunch_...` 启动，`csrc/op.cpp` 的 `XliteOpFusedQuantMatmulDequant`）

## 实现原理

实现位于 [csrc/kernels/fused_quant_matmul_dequant.h](../../csrc/kernels/fused_quant_matmul_dequant.h)，类模板 `FusedQuantMatmulDequant<int8_t, int32_t, half, bfloat16_t>`，AIC 侧复用 `Matmul`，AIV 侧复用 `Quant`/`QuantDyn`/`Dequant` 子算子原语。

- **双侧均衡**：AIC 跑与独立 matmul 相同的 grid-stride（`idx ≡ b mod N`）；AIV 量化行按全 AIV 均分连续段（`chunk = ceil(m/总AIV数)`）——量化并行度与独立 kernel 相同，matmul 负载与独立 matmul 相同。
- **V2C 粗粒度 barrier**：AIV 全部完成量化后 `AscendC::SyncAll<true>`（AIV-only 软屏障）聚合，每个 AIV 向本组 AIC set flag 9（挂 PIPE_MTE3 骑量化 store 之后——ffts 屏障是本平台唯一带数据可见性保证的跨核释放），AIC wait×2。细粒度 per-m-tile 交接在本平台不可行（无标量 atomic / MTE1 不过 L2 / ffts flag 组内路由，见 [fusion_gm_atomic_sync_experiments.md](fusion_gm_atomic_sync_experiments.md)）。
- **C2V per tile**（flag 8 挂 PIPE_FIX）与 **V2C_DONE 退出回执**（flag 10）：与现有两段融合算子一致的已验证契约；AIC 退出前必须等配对 AIV 完成（混合核退出契约）。
- **AIV 两阶段串行**：全部量化（batched 段优先，k 非 32 倍数或超 K_TILE 回退逐行）→ SyncAll → dequant 锁步，规避 quant/dequant 子算子的 UB 基址与 EVENT_ID 复用冲突。
- **ffts mode id 路由**（实测）：`0x2`（AIC→AIV）只到同组 AIV；AIV 发起必须 `0x3`（同为组内路由）。flagId 8/9/10 与核内事件号空间独立。

## 性能特征（910C 实测，2026-10-13）

对 `quant + 现有两段融合`（XLITE_FUSED_DENSE_W8A8 开启路径）：

- GLM 全部 8 个真实 size（含 attn 静态 ×4）每 forward 节省：**decode(m=128) ~13ms（35%）**、**prefill(m=2048) ~11ms（18%）**；
- 逐 shape 全区间 0.61~0.93x 无劣化（收益 ~90% 来自省 launch/sync，device 纯 kernel 时间基本持平——粗粒度 barrier 无重叠，kernel 计算量不变）；
- 测量注意：共占设备时 kernel 可能被静默丢弃，perf 前 `npu-smi` 确认空闲；`perf_xlite_ops.py` 的 `DEVICE_TIME=True` 给 wall 加 ~100us/调用，干净对比须置 False。

## 边界处理

- AIV 行段 `rowStart >= m` 跳过、`rowEnd` 向 m 截断；dequant/matmul 子算子各自有 `mActual/nActual` 边界处理。
- V2C 的 set/wait 无条件执行（AIC tiles 为零时同样等待），保证 flag 计数平衡。
- quant 按全量 m 量化（与串行链路一致）；`num` 仅作用于 dequant 侧行收缩。
