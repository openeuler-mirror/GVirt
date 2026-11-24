---
name: cube-optimization
description: How the Ascend 910B Cube works and how to make matmul/GEMM/attention QK-SV kernels fast on it — Mmad blocking, the GM→L1→L0→L0C→GM pipeline, L0/L1 budget rules, QK/SV tile sizing and adaptive M0, the tiling/swizzle/double-buffering/fusion levers, and the common correctness pitfalls (uint16 srcGap overflow on large-N NZ loads, dirty-L1B NaN, Mmad serialization thresholds, dequant/bias n0 limits). Use when writing, optimizing, reviewing, analyzing, debugging, or sizing a Matmul / GEMM / attention QK-SV / MoE kernel.
---

# Cube: how it works and how to optimize it

Reference impl: `matmul.h` (`Matmul`, reused by `group_matmul.h`), Cube helpers `attention_aic_helper.h`/`mla_aic_helper.h`/`cxa_aic_helper.h`. Docs: `matmul.md` (read first), `group_matmul.md`, `microbench.md`.

## Compute model

- `Mmad` = 16×16 multiply-accumulate: m/n in `MBLOCKSIZE`/`NBLOCKSIZE` (16) blocks, k in `kBlockSize = 32*8/dtypeBits` elements per 32-byte K block (float → 8, fp16/bf16 → 16, int8 → 32, int4 → 64).
- L0A (left) × L0B (right) → L0C accumulate. `CalMmad(c, a, b, m, n, k, init)`; `init=true` clears L0C on the first K-block.
- fp16/bf16/float in → **fp32** L0C; int8/int4 in → **int32** L0C, fp16 out via fixpipe dequant.

## Pipeline

```
A: GM --MTE2(Nd2Nz)--> L1[A1] --MTE1(LoadData)--> L0A[A2]
B: GM --MTE2--> L1[B1] --MTE1--> L0B[B2]   --Mmad--> L0C[CO1] --FixPipe(NZ2ND + bias/dequant)--> GM
```

- GM→L1 folds ND→NZ into the DMA (`Nd2NzParams`); L1→L0 is `LoadData`; L0C→GM is FixPipe.
- Each stage is double-buffered (`PINGPONG_BUF_NUM=2`) with `SetFlag`/`WaitFlag<HardEvent::…>` handshakes (event table in `matmul.md`).

## Budgets (the hard limit)

- Defaults `m0=128, n0=256, k0=1024/dtypeBits` (fp16/bf16 → 64, int8 → 128, int4 → 256) fill L1/L0B/L0C at 100% (L0A 50%).
- Byte budgets: `L0A = 2*m0*k0*dtypeBytes ≤ 64KB`, `L0B = 2*n0*k0*dtypeBytes ≤ 64KB`, `L0C = 4*m0*n0` (always fp32/int32, dtype-independent), `L1 = 8*k0*dtypeBytes*(2*m0+n0) ≤ 512KB` (plus a bias/deq tail `n0*sizeof(MatDtype)` / `n0*8B` shared in the `l1bBuf` slot). Each L0A/L0B ping-pong slot holds one `k0`; the L1 A prefetch window is `k8TileSize = 8*k0`, the B window `k4TileSize = 4*k0`.
- `n0 ≤ 256` when bias/dequant is on (BiasTable 1 KB / fixpipe 2 KB bound n0, not L1/L0).

## Optimization levers

1. **Tiling** — host `PickMatmulTiling`: `m0 = min(ROUND_UP(m,32), 128)`, `n0 = 256`, `k0 = 1024/dtypeBits`; shrinks n0 (or `m0=64, n0=384, k0/=2`) when `totalLoops < 3*aicNum` and the tail is uneven, for even core load.
2. **Swizzle (L2 reuse)** — `GetMNBlockIdx` reorders tiles so B-columns stay L2-resident; `swizzle` = `count<<8 | direction` (0=Zn, 1=Nz). `swizzle.cpp` pins measured 910B3 choices ({6144,2048}/{2048,6144} → `0xe01`).
3. **Mmad serialization** — only small tiles need `PipeBarrier<PIPE_M>`: `if (mActualBlockNum*nActualBlockNum < 10)`; large tiles pipeline themselves.
4. **Bias + dequant in FixPipe** — bias via `Mmad(..., bias, …)` (C2) on the first K-block; int8/int4 dequant via `VDEQF16` + `SetFixPipeConfig` in the L0C→GM copy (skips a separate kernel).
5. **Chain Mmads through L1** — `L0C→L1→L0B` (`CopyL0CToL1` + `CopyToL0BCol`) to skip the GM round-trip for Cube→Cube (`indexer_scores.h`).
6. **Fused attention QK/SV** (`*_aic_helper.h`):
   - shrink `cubeKvTile` until `2*tile*headSize*sz` and `2*M0*tile*sz` fit 64 KB L0;
   - adaptive M0: `GetOptimalM0` (16…128 by seq len) + `XliteFitM0ToHeads` (m0 ≥ nHeads);
   - pre-zero L1B from `zeroGm` to avoid NaN.
7. **MoE load balance** — `group_matmul` carries `coreOffset` so per-expert tail tiles rotate across cores.

## Measurement

`TFLOPS = 2*m0*n0*k0 / complete_us / 1e6` (microbench `cube_mmad_aic`, SYS_CNT `complete` time, not wall). Don't claim a win without before/after measurement.

## Pitfalls

- Unconditional `PipeBarrier<PIPE_M>` stalls large-tile throughput (use the threshold).
- Dirty L1B → NaN (pre-fill zero).
- Swizzle vs fusion: a coarse Cube→Vector handoff forces the producer off its swizzle order and can be slower than serial (`cross-core-sync`).
- `n0 > 256` + bias/dequant silently overflows BiasTable/fixpipe.
- `srcGap` of `__set_dmi_config` (uint16, `CopyGmToL1` NZ load) wraps for a row pitch > 65535 blocks, i.e. n > 65535+nValue (lm_head n=151936); split into one burst per row.

## Reference

- `matmul.md` (pipeline/budgets/events/dtypes), `group_matmul.md` (MoE), `microbench.md` (TFLOPS + budget guard).
- `data-levels-and-transfer` (levels/pipes), `cross-core-sync` (Cube↔Vector ordering).
