---
name: enable-double-buffer
description: Overlap data movement with compute in AscendC kernels using Ping/Pong double buffering, hiding MTE2/MTE3 transfer latency behind Vector/Cube compute. Use when a kernel loops over data blocks with serial CopyIn/Compute/CopyOut, profiling shows the Vector or Cube unit idling while MTE transfers run, or per-block transfer time is comparable to compute time. Covers TPipe/TQue InitBuffer(que, 2, ...) and manual C-API ping/pong buffers with set_flag/wait_flag.
---

# Enable Double Buffer (开启Double Buffer)

**Priority**: medium

## Mechanism

AI Core instruction queues are independent and can run in parallel: MTE2 (GM->UB copy-in), MTE3 (UB->GM copy-out), V (vector compute), M (cube compute), S (scalar). A naive loop serializes CopyIn -> Compute -> CopyOut per block, so with each stage taking time t the Vector unit is only ~1/3 utilized. Double Buffering allocates Ping and Pong buffer sets: while Vector computes block N in Ping, MTE2 fills block N+1 into Pong and MTE3 drains block N-1 from Ping. Transfer time is hidden behind compute.

## When to apply

- Kernel processes data in a loop of blocks with CopyIn/Compute/CopyOut phases.
- Profiling shows low Vector/Cube utilization and long MTE wait gaps.
- Transfer time is a meaningful fraction of per-block compute time.

Do NOT apply when:
- Transfer time is negligible relative to compute (little to gain).
- Data is small enough for one shot: no cross-block overlap is possible, and the extra buffers/sync can make things worse.

## Prerequisites

- Per-core data splits into at least 2 blocks.
- UB can hold two input + two output buffer sets plus other temporaries (recompute tile length / loop count after enabling).
- Sync between buffer reuse across pipelines is correctly managed.
- Buffer addresses and transfer lengths satisfy alignment rules of the target product and transfer API; tail blocks handled by actual length.

## How to apply

### TPipe/TQue programming

Set the `num` parameter of `InitBuffer` to 2 in the operator class `Init`:

```cpp
    pipe.InitBuffer(inQueueSrc0, 2, sizeSrc0 * sizeof(half));  // 2 = enable double buffer
```

The queue manages buffer reuse and pipeline sync through AllocTensor/EnQue/DeQue/FreeTensor ordering.

### Pointer-based C API programming

Manually allocate Ping and Pong UB buffers, alternate by block index, and protect each set with `set_flag`/`wait_flag` event pairs:

Required sync relations: MTE2-write -> Vector-read and Vector-read -> next MTE2-write (on input); Vector-write -> MTE3-read and MTE3-read -> next Vector-write (on output).

## Pitfalls

- UB usage doubles: resize tiles and re-derive `round` from tiling; a fixed constant is wrong across shapes.
- Tail blocks that cannot satisfy main-block double-buffer conditions must be handled separately.
- Manual C-API code must pair every buffer handoff with `set_flag(srcPipe, dstPipe, EVENT_IDn)` / `wait_flag(srcPipe, dstPipe, EVENT_IDn)`, using a distinct event id per ping/pong set (see `split.cpp`, `transpose_1_2.h` under `xlite/csrc/kernels/`).
- Always drain with `pipe_barrier(PIPE_ALL)` before kernel exit.

## Verification

- Correctness: results identical to the single-buffer version.
- Performance: Vector utilization (aiv_vec_ratio) rises, Task Duration drops; compare in msprof.
