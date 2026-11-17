---
name: cross-core-sync
description: Implement correct, low-overhead manual AIC/AIV (Cube/Vector) cross-core synchronization in AscendC kernels using CrossCoreSetFlag/CrossCoreWaitFlag, with intra-core SetFlag/WaitFlag for pipeline ordering. Use when a manually-written MIX kernel passes data between the Cube core and Vector core (Mmad result handoff, reduce-then-read), when cross-core sync blocks, deadlocks, or fires spuriously, or when choosing a manual sync primitive (mode 0/1/2/4, SyncAll, IBSet/IBWait) and its handoff granularity.
---

# Manual Cross-Core (AIC/AIV) Synchronization

## Mechanism

In separated (MIX) mode the Cube core (AIC) and Vector cores (AIV) are distinct cores (one `block` + N `subblock`), each with its own Scalar unit and instruction streams. They rendezvous only through explicit synchronization, never through shared on-chip memory for the C result.

The manual sync primitive is **`CrossCoreSetFlag`/`CrossCoreWaitFlag`**, a hardware barrier keyed by a `flagId`:

- `CrossCoreSetFlag` reports "done" to the dispatch module after the calling core's preceding work on a given pipe completes.
- `CrossCoreWaitFlag` blocks further instruction issue (already-issued instructions still run) until all required cores have reported.

Each `flagId` has a per-core counter (init 0, capped at 15): `SetFlag` increments it, and a `WaitFlag` blocks until it is non-zero, then decrements it. For a grouped mode the wait is released only once the mode's required `SetFlag`s have landed (e.g. mode 2: the AIC's one set releases both paired AIVs). More than 15 outstanding `SetFlag`s on one counter raises an error.

Within a single core, pipeline-to-pipeline ordering is done manually with **`SetFlag`/`WaitFlag`** (keyed by a `HardEvent`, e.g. `MTE2_V`, `V_MTE3`) or **`PipeBarrier`**.

Sync does **not** move data. On Atlas A2 (910B, NPU arch 2201) the Cube→Vector C result physically travels **L0C → GM → UB**: the Cube `Fixpipe` writes L0C→GM, then the Vector `DataCopy`/`MTE2` reads GM→UB. The direct L0C→UB path exists only on 3510 (950PR/DT). Sync orders that write-then-read across the GM staging. The payoff of fusing is **cross-iteration overlap** — the Cube computes tile *k+1* while the Vector consumes tile *k* — not merely saving a kernel launch; the GM round-trip itself remains.

## When to apply

- A manually-written MIX (Basic-API) kernel hands the `Mmad` result to vector code and you must guarantee readiness without a race.
- A kernel needs all-AIV, AIC↔AIV, or AIV↔AIV synchronization beyond compute (reduce-then-read, atomic accumulation, deterministic ordering).
- Cross-core sync is suspected of deadlock, spurious release, or incorrect results.

When **not** to apply: the fusion primarily pays off through cross-iteration overlap (Cube on tile *k+1* while Vector consumes tile *k*), with launch removal and skipping the serial write-out as secondary gains. If the handoff forces the producer off its optimal tile order (see "Schedule the handoff granularity"), that overlap is lost and the fused kernel can be *slower* than the serial producer→consumer pair. Benchmark serial vs fused on the target shapes before committing.

## MIX block / sub-block id mapping

A MIX launch schedules one physical block per launched block; `KERNEL_TYPE_MIX_AIC_1_2` gives each block 1 AIC + 2 AIV, and the same kernel binary runs on every core, dispatched by `__DAV_C220_CUBE__` / `__DAV_C220_VEC__` (or `ASCEND_IS_AIC` / `ASCEND_IS_AIV` in other codebases):

- AIC: `GetBlockIdx() ∈ [0, N)`, `GetBlockNum() == N`.
- AIV: `GetBlockIdx() ∈ [0, 2N)` — it is `2 * block_idx + subblockid`; `GetBlockNum()` is still `N`; `GetTaskRatio()` returns `2`.

AIC block `b` therefore pairs with AIVs `2b` and `2b+1` (`b = t / 2`, `sub = t % 2`). Both sides must walk the **same work order** so the flag produced for unit `k` is consumed by the same unit `k`; the per-core counter (0–15) absorbs the producer running ahead, but it is **bounded** — a producer more than 15 handoffs ahead of the consumer raises an error.

## How to apply

### 1. Pick the right primitive

| Need | Use |
| --- | --- |
| All AICs or all AIVs barrier (across AI Cores) | `CrossCoreSetFlag<0,…>` + `CrossCoreWaitFlag<0>` (mode 0) |
| All AIVs within one AI Core | mode 1 |
| AIC ↔ all AIVs in one AI Core | mode 2 (one-to-many pairing) |
| AIC ↔ a single AIV in one AI Core | mode 4 (3510 only) |
| AIV↔AIV pair (software) | `IBSet`/`IBWait` (GM flag poll) |
| All-core sync for a fused operator | `SyncAll` |
| AIVs in strict blockIdx order (deterministic) | `InitDetermineComputeWorkspace` + `NotifyNextBlock` + `WaitPreBlock` |
| Pipeline ordering within one core | `SetFlag`/`WaitFlag` or `PipeBarrier` |

### 2. Manual Cube→Vector handoff pattern (910B)

```cpp
// kernel entry sets KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
#if defined(__DAV_C220_CUBE__)
    // ... Matmul writes C to L0C, then Fixpipe moves L0C -> GM ...
    CrossCoreSetFlag<0x2, PIPE_FIX>(C2V_FLAG);   // report after PIPE_FIX completes
#elif defined(__DAV_C220_VEC__)
    CrossCoreWaitFlag<0x2>(C2V_FLAG);            // blocks all pipes on 910B
    // now safe: DataCopy(C_gm -> UB), then vector compute
#endif
```

Pick `C2V_FLAG` outside the internal-interface range (see "Avoid flagId conflicts"); 8/9 are safe for ≤4 `Matmul` objects. Signatures — note the asymmetry (`SetFlag` templates have no defaults, `WaitFlag` does):

```cpp
template <uint8_t modeId, pipe_t pipe>              __aicore__ inline void CrossCoreSetFlag(uint16_t flagId);
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S> __aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);
```

`CrossCoreWaitFlag` ignores its `modeId` on the wait side (the routing is already fixed by the `SetFlag` side), so an un-templated call pairs identically to an explicit `<0x2>`. Still write the mode on both sides as a readability convention.

### 3. Schedule the handoff granularity

Sync only orders the producer and consumer; it does not make either run faster. The handoff granularity is the single biggest lever — and the most common mistake:

- A handoff point makes the consumer's input ready only after **all** producer tiles it needs have been written. If the consumer needs columns spread across several producer tiles (e.g. `silu(gate)·up` needs the matched `gate` and `up` columns, `inter` apart), a per-tile handoff is insufficient and you get pushed to a coarser barrier.
- A coarse barrier often forces the producer to **abandon its swizzle / tile order**. `Matmul` reorders tiles (swizzle) so B-columns stay L2-resident while sweeping m; walking tiles grouped by m — or disabling swizzle to make the grouping legal — destroys that reuse and can make the fused kernel *slower* than the serial matmul+activation even though the sync is correct.
- Prefer the **finest dependency-closed granularity**: hand off each unit the moment all of its producer tiles have landed, keeping the producer's swizzled order intact. For `matmul → elementwise` (each consumer tile depends on one producer tile) per-tile handoff is enough; for cross-column consumers like `silu_and_mul` the unit is the (row, column-pair) and needs the gate/up pair.

A per-tile handoff in lockstep — both sides must walk the **same full-tile index** so the flag for tile `t` is consumed by the same tile; mode 2 fires both AIVs, each taking the half its sub-block owns:

```cpp
// T = number of full producer tiles.
// AIC block b (GetBlockIdx() = b in [0, N)):
for (t = GetBlockIdx(); t < T; t += GetBlockNum()) {
    RunMatmulTile(t);                            // mmad + fixpipe -> GM
    CrossCoreSetFlag<0x2, PIPE_FIX>(C2V_FLAG);   // fires after the tile lands
}
// AIV 2b+s (GetBlockIdx() = 2b+s in [0, 2N)): walk the *same* tiles, take half s.
for (t = GetBlockIdx() / 2; t < T; t += GetBlockNum()) {   // b = idx / 2
    CrossCoreWaitFlag<0x2>(C2V_FLAG);
    DataCopy(scratch[t].half(GetBlockIdx() & 1) -> UB);    // s = idx & 1
    VectorCompute(t, GetBlockIdx() & 1);
}
```

The producer may run ahead, and every outstanding `SetFlag` occupies a slot on the flag's counter (capped at 15) — so the AIC must never get more than 15 tiles ahead of its consumer. Either keep the tiles per block ≤ 15 (`T/N ≤ 15`), or throttle the producer with a periodic reverse ack: the AIV `SetFlag`s every `V2C_SYNC_INTERVAL` tiles and the AIC `CrossCoreWaitFlag`s on it, exactly as the dequant pipeline fusion does.

- If the consumer writes to a **separate output buffer** and each handoff unit maps to a disjoint range of the producer's staging buffer, the reverse (V2C) ack is unnecessary *for buffer reuse* — but it may still be needed (as the throttle above) to bound the producer's run-ahead.

### 4. Mind the sync-mode / pipe / flagId rules

- **Modes on 910B (2201)**: 0, 1, 2 only (mode 4 is 3510-only). Kernel type must be a MIX variant (`KERNEL_TYPE_MIX_AIC_1_2` etc.).
- **`pipe` on 910B**: `CrossCoreSetFlag` requires an explicit pipe (`PIPE_V/M/MTE1/MTE2/MTE3/FIX`, not `PIPE_S`/`PIPE_ALL`). `CrossCoreWaitFlag`'s `pipe` is **ignored** on 2201 — it blocks *all* pipelines, so use the default.
- **`flagId`**: 0–15 per core for modes 0/1/2 (low 4 bits truncated). Must be paired; >15 outstanding `SetFlag`s on one counter raises an error.
- **Mode 2 pairing is one-to-many**: the AIC's single `SetFlag` releases all its paired AIVs' `WaitFlag`s (AIC→AIV), or all AIVs' `SetFlag`s release the AIC's single `WaitFlag` (AIV→AIC). On 910B a mode-2 handoff always fires **both** AIVs together (per-AIV AIC→AIV is mode 4, 3510-only) — the natural fit when both AIVs consume the same produced tile (split-output), but it cannot express independent per-AIV readiness.

### 5. Avoid flagId conflicts

Internal interfaces consume flags on 2201 (modes 0/1/2): `Matmul` (N objects) uses `[0, 2N-1]`; `SyncAll(isAIVOnly=false)` uses 11–13; `SyncAll(isAIVOnly=true)` uses 14; `SetNextTaskStart`/`WaitPreTaskEnd` use 11–14. Pick an unused flag when mixing manual sync with such interfaces — 8/9 sit outside the `Matmul` range only for ≤4 Matmul objects.

## Lower-level API: ffts_cross_core_sync / wait_flag_dev

Some kernels drive the same hardware flags through the CCE stub instead of the named API — the attention/softmax/indexer pipelines (`attention.h`, `cxa.h`, `indexer_topk.h`, `indexer_prepare_c8.h`, `permutation.cpp`) use:

```cpp
ffts_cross_core_sync(pipe, config);  // set: config packs flagId + mode
wait_flag_dev(flagId);               // wait
```

The `ffts_cross_core_sync` set side takes a packed `config = 1 | (mode << 4) | (flagId << 8)` — both `permutation.cpp` (mode 0) and `indexer_topk.h` (mode 2) follow this — while the `wait_flag_dev` side takes the bare `flagId`. Treat these as the same primitive: `mode` selects the group pairing, `flagId` selects the counter, and the same pairing/pipe/flag rules apply. Prefer the named `CrossCoreSetFlag<mode, pipe>(flagId)` for new code.

## Pitfalls

- **Pairing**: `CrossCoreSetFlag` and `CrossCoreWaitFlag` must always be paired, else undefined behavior or a permanently blocked core.
- **Qualifier**: cross-core sync needs a MIX kernel; `KERNEL_TYPE_AIC_ONLY`/`AIV_ONLY` do not enable the dispatch module, so sync silently cannot work.
- **No ordering guarantee** between consecutive `CrossCoreSetFlag` calls on one core — the pair-by-flagId discipline is what keeps them matched.
- **Mode-0 deadlock** in multi-stream: if total cores exceed physical cores and ≥2 concurrent operators use inter-core sync, some cores wait forever. Enable batchmode (`__schedmode__(mode)` or `SetScheduleMode`).
- **GM staging**: the Cube→Vector result goes through GM; forgetting the wait means reading stale GM. Use `PIPE_FIX` on the `SetFlag` so the signal fires only after `Fixpipe` finished writing GM.
- **Intra-core vs inter-core**: `SetFlag`/`WaitFlag` only order pipelines on the *same* core; they do not cross cores — use `CrossCoreSetFlag`/`CrossCoreWaitFlag` for AIC↔AIV.

## Verification

- Numerical results identical to a reference (sync correctness).
- **Determinism**: run the same shape several times and check the mismatch positions are stable. Identical positions across runs ⇒ a deterministic numerical difference (e.g. a rounding boundary), not a race. Positions that move run-to-run ⇒ the flag fires too early and the consumer reads stale GM.
- No hang/deadlock across repeated runs and multi-stream; profiling shows the barrier wait is only as long as the true dependency.
