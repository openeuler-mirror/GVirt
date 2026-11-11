---
name: transfer-larger-blocks
description: Maximize MTE transfer bandwidth in AscendC kernels by moving larger data blocks per DataCopy call. Use when profiling shows bandwidth underutilization, many small consecutive copies, or tiles smaller than ~16KB; measured bandwidth typically saturates for single transfers of 16KB and above.
---

# Transfer Larger Data Blocks (尽量一次搬运较大的数据块)

**Priority**: high

## Mechanism

Bandwidth utilization (effective / theoretical) of MTE transfers depends on the size of each individual transfer. Measured curves on AI processors show that single transfers below a few KB achieve a fraction of peak bandwidth, while transfers of 16KB and above usually reach the best sustained bandwidth (exact knee varies by product and measurement jitter). Many small copies therefore waste bandwidth and add per-instruction overhead.

## When to apply

- Tiling produces per-copy sizes well under 16KB.
- Code contains loops of small `DataCopy` calls over contiguous data.
- Profiling shows low achieved GM bandwidth relative to peak.

## How to apply

1. Size tiles (per-iteration data blocks) so each `DataCopy` moves at least ~16KB where UB capacity allows.
2. Merge adjacent contiguous copies into one call covering the union.
3. If data is strided, use the bulk strided form instead of per-row loops.
4. Balance against double buffering: two buffer sets must still fit in UB — block size = UB budget / buffers / tensors.

```cpp
// Small per-iteration copies
constexpr uint32_t smallTile = 4 * 1024;             // 4KB per copy: bandwidth loss
// Prefer a tile that makes each DataCopy >= 16KB
constexpr uint32_t tile = 16 * 1024 / sizeof(half);  // elements per transfer
```

## Pitfalls

- Larger blocks reduce loop-level pipelining granularity; ensure at least 2 blocks remain per core for double buffering.
- UB capacity is the hard limit; sharing UB with high-level-API temporaries may force trade-offs.
- The 16KB figure is an empirical guideline from the docs, not a hard threshold; validate on the target product.

## Verification

- Measure achieved bandwidth (bytes moved / MTE busy time) before and after.
- Task Duration should drop when copies consolidate.
