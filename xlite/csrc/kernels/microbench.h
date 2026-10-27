/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "matmul.h"

// Micro-benchmark kernel family. Each core type ships ONE merged kernel
// (bench_aic / bench_aiv / bench_mix, one kernel per TU) that dispatches on a
// uint32_t op selector (kernel_param.h BenchOp); only the ops valid for that core type
// appear in the kernel's switch. Experiments:
//   1. empty launch-floor probes        BENCH_OP_EMPTY            both builds
//   2. scalar dependent chains          BENCH_OP_SCALAR_*         both builds
//      + timed twins                    BENCH_OP_SCALAR_*_TIMED   MICROBENCH_TIMED
//   3. MTE memory bandwidth/latency     BENCH_OP_MTE_*            MICROBENCH_TIMED
//   4. vector-op throughput             BENCH_OP_VEC_*            MICROBENCH_TIMED
//   5. cube matmul throughput           BENCH_OP_CUBE_MMAD        MICROBENCH_TIMED
// (3/4/5 exist only in XLITE_BENCH_TIMED=ON builds.) Design, data and the
// reproduction recipe: doc/developer_guide/kernels/microbench.md and
// tests/performance/micro_bench.py.
//
// ── Unified kernel signature ──
// bench_<ct>(GM_ADDR p0, p1, p2, p3, ticks, uint32_t seed,
//            uint64_t sz0, sz1, sz2, uint32_t iters, uint32_t op)
// op selects which params are meaningful: empty uses none; scalar p0=out +
// seed/iters; mte p0=src,p1=dst,sz0=bytes; vec p0=src,p1=out,sz0=elems;
// cube p0=a,p1=b,p2=c,sz0/sz1/sz2=m0/n0/k0. seed is a uint32_t bit pattern
// (integer variants read it directly, fadd reconstructs the float via a union)
// because the device has no double path and every other kernel in csrc already
// uses 32/64-bit scalar args. Core-kind routing (KERNEL_TASK_TYPE_DEFAULT) and
// numBlocks semantics: aiv/aic count their own blocks, mix counts AIC blocks
// (hardware pairs each with 2 AIV, 1:2).
//
// ── Translation unit layout ──
// One merged bench kernel per core type, one kernel per TU. ascendc auto_gen
// assigns AIC/AIV single-core kernels a global rename index (<name>_<idx>) but
// emits the .ascend.meta.<name>_<idx> type section with only the file's
// first-kernel index, so a multi-kernel AIC/AIV TU leaves every kernel after
// the first without a type section -> runtime registration fails
// (RegisterAscendBinary ret 107000). Each merged bench_<ct> is the sole kernel
// in its TU, staying on auto_gen's single-kernel path; MIX uses a different
// auto_gen path (no rename), so bench_mix is unaffected.
//
// Measurement-validity note: merging per-op bodies into one kernel changes each
// body's codegen context (the switch and sibling bodies are visible to the
// compiler), shifting per-op cycle counts by a small constant vs the
// one-kernel-per-TU baseline. Re-baseline after this change; never compare
// merged numbers with old per-TU numbers, and never compare timed with default.

// ==================== common: pasting + core-kind routing ====================
// Concatenation helpers (expand arguments before pasting).
#define BENCH_CAT(a, b) BENCH_CAT_IMPL(a, b)
#define BENCH_CAT_IMPL(a, b) a##b
#define BENCH_CAT3(a, b, c) BENCH_CAT3_IMPL(a, b, c)
#define BENCH_CAT3_IMPL(a, b, c) a##b##c

// Core-kind task types (lowercase aliases, kept for any single-TU experiment
// that still pastes a kernel name with its core suffix).
#define BENCH_TASK_aiv KERNEL_TYPE_AIV_ONLY
#define BENCH_TASK_aic KERNEL_TYPE_AIC_ONLY
#define BENCH_TASK_mix KERNEL_TYPE_MIX_AIC_1_2

// Build-level flavour switch, set to 1 by CMake XLITE_BENCH_TIMED=ON for the
// whole kernel+host build. Defaults to 0 here so the header is self-contained
// (and a plain `#define MICROBENCH_TIMED 1` before the include still
// overrides, for single-TU experiments).
#ifndef MICROBENCH_TIMED
#define MICROBENCH_TIMED 0
#endif

// ==================== experiment 1: empty launch-floor probes ====================
// No computation, no data movement, no synchronization: the device time of
// one kernel round trip (host launch -> core scheduling -> device execution
// -> return), i.e. the per-launch floor any real kernel of that core type
// cannot go below. The merged kernel's non-empty param list satisfies the
// host-stub extractor; the empty op ignores all params.
#define BENCH_EMPTY_BODY() \
    do {                   \
    } while (0)

// ==================== experiment 2: scalar dependent chains ====================
// A dependent chain of `iters` scalar ops (S pipe). Measurement-validity design
// (each item load-bearing):
// - anti-DCE: the accumulator is consumed by storing it to GM (p0) at the end.
// - anti-folding: the initial value and operand both come from the runtime
//   parameter `seed`, unknown at compile time. That alone is NOT enough for
//   `acc += x`: bisheng reassociates the invariant-operand add chain into the
//   closed form `acc + N*x` (measured: constant ~1.7us for any iters, i.e.
//   the loop did not run). The add variants therefore entangle the operand
//   with the loop counter, `acc += (x ^ i)`: the xor of an IV has no closed
//   form. mul / div / fp32 chains have no cheap closed form and need no
//   entanglement (measured linear in iters when unguarded).
// - chain semantics: each iteration consumes the previous result, so the
//   measured per-iteration cost is op latency + loop control, not throughput.
//   `iters` counts scalar ops for every variant (add_unroll4 divides by 4
//   internally), keeping per-op costs comparable.
// - result write-back is a direct scalar GM store: the raw
//   copy_ubuf_to_gm_align_b16 intrinsic is rejected by the cube compile
//   channel (each common kernel file is compiled into BOTH the aiv and aic
//   channels), while the direct store plus a final pipe_barrier compiles on
//   both and its once-per-launch cost only lands in the constant term.
// - no MIX variant: pure scalar work has no cube/vector component.
// - variant system (per core type), solving three unknowns from three slopes
//   (b = loop-control cost per trip, a = add, o = xor):
//     scalar_add          acc += (x ^ i)          s1 = b + a + o
//     scalar_xor          acc ^= (x ^ i)          sx = b + 2*o
//     scalar_add_unroll4  4x acc += (x ^ (4t+k))  s4 = b/4 + a + o
//   => b = 4*(s1 - s4)/3, o = (sx - b)/2, a = s1 - b - o.
//   scalar_mul / scalar_div / scalar_fadd measure their op + b directly.
//
// Each BENCH_SCALAR_*_BODY() is the full post-marker body (decls + loop + GM
// store + pipe_barrier), wrapped in do-while so it drops cleanly into a
// switch case. KERNEL_TASK_TYPE_DEFAULT is emitted once at the kernel top.
#define BENCH_SCALAR_ADD_BODY()                \
    do {                                       \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc + (x ^ i);               \
        }                                      \
        *(__gm__ uint32_t *)p0 = acc;          \
        pipe_barrier(PIPE_ALL);                \
    } while (0)

#define BENCH_SCALAR_XOR_BODY()                \
    do {                                       \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc ^ (x ^ i);               \
        }                                      \
        *(__gm__ uint32_t *)p0 = acc;          \
        pipe_barrier(PIPE_ALL);                \
    } while (0)

#define BENCH_SCALAR_MUL_BODY()                \
    do {                                       \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc * x;                     \
        }                                      \
        *(__gm__ uint32_t *)p0 = acc;          \
        pipe_barrier(PIPE_ALL);                \
    } while (0)

#define BENCH_SCALAR_DIV_BODY()                \
    do {                                       \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc / x;                     \
        }                                      \
        *(__gm__ uint32_t *)p0 = acc;          \
        pipe_barrier(PIPE_ALL);                \
    } while (0)

// fadd carries seed as a float bit pattern; reconstruct via a union.
#define BENCH_SCALAR_FADD_BODY()               \
    do {                                       \
        union {                                \
            uint32_t u;                        \
            float f;                           \
        } cv;                                  \
        cv.u = seed;                           \
        float x = cv.f;                        \
        float acc = x;                         \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc + x;                     \
        }                                      \
        *(__gm__ float *)p0 = acc;             \
        pipe_barrier(PIPE_ALL);                \
    } while (0)

// Unrolled-by-4: 4 dependent adds per loop trip, iters = total add count (host
// rejects iters not divisible by 4). Counter values are 4t+0..3 so each add's
// entangled operand matches the plain variant's work.
#define BENCH_SCALAR_ADD_UNROLL4_BODY()            \
    do {                                           \
        uint32_t acc = seed;                       \
        uint32_t x = seed;                         \
        for (uint32_t t = 0; t < iters / 4; t++) { \
            acc = acc + (x ^ (4 * t));             \
            acc = acc + (x ^ (4 * t + 1));         \
            acc = acc + (x ^ (4 * t + 2));         \
            acc = acc + (x ^ (4 * t + 3));         \
        }                                          \
        *(__gm__ uint32_t *)p0 = acc;              \
        pipe_barrier(PIPE_ALL);                    \
    } while (0)

#if MICROBENCH_TIMED
// Timed twins of the scalar items: the SAME bodies wrapped with in-kernel
// SYS_CNT sampling (kernel_macro.h BENCH_TICK*): slot 0 = region start, slot 1
// = loop issued, PIPE_ALL drain, slot 2 = true completion; accumulator and tick
// slots are stored only after the closing tick, so no store lands inside the
// measured window. Host entry: XliteOpBenchTimed - one dispatch for the whole
// timed family. The case labels in bench_<ct>.cpp are themselves gated by
// MICROBENCH_TIMED, so these BODY macros are only ever expanded in a timed
// build (no empty #else needed).
#define BENCH_SCALAR_ADD_TIMED_BODY()          \
    do {                                       \
        BENCH_TICKS_DECL;                      \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        BENCH_TICK(0);                         \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc + (x ^ i);               \
        }                                      \
        BENCH_TICK(1);                         \
        pipe_barrier(PIPE_ALL);                \
        BENCH_TICK(2);                         \
        *(__gm__ uint32_t *)p0 = acc;          \
        BENCH_TICKS_FLUSH(ticks, 3);           \
    } while (0)

#define BENCH_SCALAR_XOR_TIMED_BODY()          \
    do {                                       \
        BENCH_TICKS_DECL;                      \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        BENCH_TICK(0);                         \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc ^ (x ^ i);               \
        }                                      \
        BENCH_TICK(1);                         \
        pipe_barrier(PIPE_ALL);                \
        BENCH_TICK(2);                         \
        *(__gm__ uint32_t *)p0 = acc;          \
        BENCH_TICKS_FLUSH(ticks, 3);           \
    } while (0)

#define BENCH_SCALAR_MUL_TIMED_BODY()          \
    do {                                       \
        BENCH_TICKS_DECL;                      \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        BENCH_TICK(0);                         \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc * x;                     \
        }                                      \
        BENCH_TICK(1);                         \
        pipe_barrier(PIPE_ALL);                \
        BENCH_TICK(2);                         \
        *(__gm__ uint32_t *)p0 = acc;          \
        BENCH_TICKS_FLUSH(ticks, 3);           \
    } while (0)

#define BENCH_SCALAR_DIV_TIMED_BODY()          \
    do {                                       \
        BENCH_TICKS_DECL;                      \
        uint32_t acc = seed;                   \
        uint32_t x = seed;                     \
        BENCH_TICK(0);                         \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc / x;                     \
        }                                      \
        BENCH_TICK(1);                         \
        pipe_barrier(PIPE_ALL);                \
        BENCH_TICK(2);                         \
        *(__gm__ uint32_t *)p0 = acc;          \
        BENCH_TICKS_FLUSH(ticks, 3);           \
    } while (0)

#define BENCH_SCALAR_FADD_TIMED_BODY()         \
    do {                                       \
        BENCH_TICKS_DECL;                      \
        union {                                \
            uint32_t u;                        \
            float f;                           \
        } cv;                                  \
        cv.u = seed;                           \
        float x = cv.f;                        \
        float acc = x;                         \
        BENCH_TICK(0);                         \
        for (uint32_t i = 0; i < iters; i++) { \
            acc = acc + x;                     \
        }                                      \
        BENCH_TICK(1);                         \
        pipe_barrier(PIPE_ALL);                \
        BENCH_TICK(2);                         \
        *(__gm__ float *)p0 = acc;             \
        BENCH_TICKS_FLUSH(ticks, 3);           \
    } while (0)

#define BENCH_SCALAR_ADD_UNROLL4_TIMED_BODY()      \
    do {                                           \
        BENCH_TICKS_DECL;                          \
        uint32_t acc = seed;                       \
        uint32_t x = seed;                         \
        BENCH_TICK(0);                             \
        for (uint32_t t = 0; t < iters / 4; t++) { \
            acc = acc + (x ^ (4 * t));             \
            acc = acc + (x ^ (4 * t + 1));         \
            acc = acc + (x ^ (4 * t + 2));         \
            acc = acc + (x ^ (4 * t + 3));         \
        }                                          \
        BENCH_TICK(1);                             \
        pipe_barrier(PIPE_ALL);                    \
        BENCH_TICK(2);                             \
        *(__gm__ uint32_t *)p0 = acc;              \
        BENCH_TICKS_FLUSH(ticks, 3);               \
    } while (0)

// ==================== experiment 3: MTE memory micro-benchmarks ====================
// lmbench-style probes driven entirely by the MTE pipes; timing comes from
// the same SYS_CNT slots, so this family shares the timed build. lmbench
// mapping:
//   mte_read   ~ bw_mem rd     streaming GM->UB loads (MTE2) over a 2-tile UB
//                              ring, no per-tile sync - the MTE queue pipeline
//                              is the object being measured
//   mte_write  ~ bw_mem wr     UB tile filled once from src (handshake BEFORE
//                              the opening tick), then streaming UB->GM (MTE3)
//   mte_copy   ~ bw_mem bcopy  GM->UB->GM ping-pong with the split.cpp flag
//                              choreography; bandwidth counts bytes once, the
//                              engine moves them twice (read+write)
//   mte_chase  ~ lat_mem_rd    dependent chain: 64B GM->UB load, MTE2->S
//                              handshake, the loaded word IS the next offset -
//                              nothing can prefetch. Per-step cost = MTE issue
//                              + GM latency + handshake, i.e. the real
//                              load-to-use cost of consuming GM via MTE.
// Unified mte semantics: p0=src, p1=dst, ticks, sz0=bytes (host enforces %32,
// chase %64 because one step is a 64B block), iters = reps of the whole working
// set (chase: chain steps). Blocks partition [0, bytes) grid-stride in
// BENCH_MTE_TILE segments (split.cpp idiom); chase is single-block by nature
// (the host clamps blockNum to 1). The working set is re-streamed every rep
// without any flush, so the size sweep itself exposes L2 residency - that curve
// is the point, matching lmbench.
// Double-flavour guard: every common TU is preprocessed in BOTH the vec and
// cube flavours and MTE intrinsics are rejected by dav-c220-cube, so the BODY
// macros are double-defined dequant.h style: real bodies under
// __DAV_C220_VEC__, same-signature empty stubs otherwise (the cube-flavour stub
// never runs - host launches bench_aiv only on AIV blocks - but must compile).

// Staging tile = one UB half (split.cpp idiom), 98304 on 910B3. The host
// (op.cpp XliteOpBenchMte) and Python (micro_bench.py MTE_TILE) keep this
// in sync.
#define BENCH_MTE_TILE (UB_SIZE / 2)

// Per-block stride for the scalar GM outputs (dst word of read/chase, the
// ticks flush). Scalar GM stores from DIFFERENT cores that land in the same
// 64B line race line-granularly and whole lines lose their updates (measured
// on 910B3: packed strides lose ~half the per-block words nondeterministically
// once blockNum >= 2, all-or-nothing per 64B line, permanent; MTE3 stores are
// exempt, which is why production kernels store to GM via copy_ubuf_to_gm).
// Giving each block its own 64B line makes the scalar stores disjoint;
// op.cpp and micro_bench.py mirror this stride.
#define BENCH_MTE_STRIDE 64

#ifdef __DAV_C220_VEC__

// Streaming read. The post-window consume (one word of the last-loaded tile,
// after the closing tick and an MTE2->S handshake) makes the dst value
// predictable for the Python check; the loads themselves cannot be elided -
// MTE copies are opaque DMA intrinsics - and micro_bench.py guards this with
// a time-vs-iters ratio.
#define BENCH_MTE_READ_BODY()                                                    \
    do {                                                                         \
        set_atomic_none();                                                       \
        set_mask_norm();                                                         \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                             \
        BENCH_TICKS_DECL;                                                        \
        __ubuf__ uint8_t *ring[2] = {                                            \
            (__ubuf__ uint8_t *)((uintptr_t)0),                                  \
            (__ubuf__ uint8_t *)((uintptr_t)BENCH_MTE_TILE),                     \
        };                                                                       \
        uint64_t half = BENCH_MTE_TILE;                                          \
        uint32_t segCnt = 0;                                                     \
        BENCH_TICK(0);                                                           \
        for (uint32_t r = 0; r < iters; r++) {                                   \
            for (uint64_t gbo = (uint64_t)block_idx * half; gbo < sz0;           \
                 gbo += (uint64_t)block_num * half) {                            \
                uint64_t remain = sz0 - gbo;                                     \
                uint32_t len = (uint32_t)(remain < half ? remain : half);        \
                CopyGmToUbufAligned(ring[segCnt & 1], p0 + gbo, len);            \
                segCnt++;                                                        \
            }                                                                    \
        }                                                                        \
        BENCH_TICK(1);                                                           \
        pipe_barrier(PIPE_ALL);                                                  \
        BENCH_TICK(2);                                                           \
        uint32_t acc = 0;                                                        \
        if (segCnt > 0) {                                                        \
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID2);                              \
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID2);                             \
            acc = *(__ubuf__ uint32_t *)ring[(segCnt - 1) & 1];                  \
        }                                                                        \
        *(__gm__ uint32_t *)(p1 + (uint64_t)block_idx * BENCH_MTE_STRIDE) = acc; \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);    \
    } while (0)

// Streaming write. The pattern tile is filled from src once and the MTE2->MTE3
// handshake completes BEFORE the opening tick (otherwise it would pollute the
// window); afterwards UB is only read by MTE3, so no further sync is needed.
// dst ends up as the src prefix tiled along every block's segment walk.
#define BENCH_MTE_WRITE_BODY()                                                \
    do {                                                                      \
        set_atomic_none();                                                    \
        set_mask_norm();                                                      \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                          \
        BENCH_TICKS_DECL;                                                     \
        __ubuf__ uint8_t *pat = (__ubuf__ uint8_t *)((uintptr_t)0);           \
        uint64_t half = BENCH_MTE_TILE;                                       \
        uint64_t fillLen = sz0 < half ? sz0 : half;                           \
        CopyGmToUbufAligned(pat, p0, (uint32_t)fillLen);                      \
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);                            \
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);                           \
        BENCH_TICK(0);                                                        \
        for (uint32_t r = 0; r < iters; r++) {                                \
            for (uint64_t gbo = (uint64_t)block_idx * half; gbo < sz0;        \
                 gbo += (uint64_t)block_num * half) {                         \
                uint64_t remain = sz0 - gbo;                                  \
                uint32_t len = (uint32_t)(remain < half ? remain : half);     \
                CopyUbufToGmAligned(p1 + gbo, pat, len);                      \
            }                                                                 \
        }                                                                     \
        BENCH_TICK(1);                                                        \
        pipe_barrier(PIPE_ALL);                                               \
        BENCH_TICK(2);                                                        \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3); \
    } while (0)

// bcopy. Each rep is one complete split.cpp-style ping-pong execution: seed
// both MTE3->MTE2 buffer-free flags, walk the grid-stride segments loading
// into one half while storing the previous half (MTE2<->MTE3 flag pairs),
// flush the pending tail, then wait both flags dry - balanced even for an
// idle block. The per-rep tail wait drains the last stores inside the
// window, so t1 already ~ t2 for this variant.
#define BENCH_MTE_COPY_BODY()                                                 \
    do {                                                                      \
        set_atomic_none();                                                    \
        set_mask_norm();                                                      \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                          \
        BENCH_TICKS_DECL;                                                     \
        __ubuf__ uint8_t *buf[2] = {                                          \
            (__ubuf__ uint8_t *)((uintptr_t)0),                               \
            (__ubuf__ uint8_t *)((uintptr_t)BENCH_MTE_TILE),                  \
        };                                                                    \
        uint64_t half = BENCH_MTE_TILE;                                       \
        BENCH_TICK(0);                                                        \
        for (uint32_t r = 0; r < iters; r++) {                                \
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);                        \
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);                        \
            uint32_t curr = 0;                                                \
            bool hasPending = false;                                          \
            uint64_t pendGbo = 0;                                             \
            uint32_t pendLen = 0;                                             \
            for (uint64_t gbo = (uint64_t)block_idx * half; gbo < sz0;        \
                 gbo += (uint64_t)block_num * half) {                         \
                uint64_t remain = sz0 - gbo;                                  \
                uint32_t len = (uint32_t)(remain < half ? remain : half);     \
                if (hasPending) {                                             \
                    uint32_t wbuf = 1 - curr;                                 \
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);        \
                    CopyUbufToGmAligned(p1 + pendGbo, buf[wbuf], pendLen);    \
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);         \
                }                                                             \
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);            \
                CopyGmToUbufAligned(buf[curr], p0 + gbo, len);                \
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);             \
                pendGbo = gbo;                                                \
                pendLen = len;                                                \
                hasPending = true;                                            \
                curr = 1 - curr;                                              \
            }                                                                 \
            if (hasPending) {                                                 \
                uint32_t wbuf = 1 - curr;                                     \
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);            \
                CopyUbufToGmAligned(p1 + pendGbo, buf[wbuf], pendLen);        \
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);             \
            }                                                                 \
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);                       \
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);                       \
        }                                                                     \
        BENCH_TICK(1);                                                        \
        pipe_barrier(PIPE_ALL);                                               \
        BENCH_TICK(2);                                                        \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3); \
    } while (0)

// Dependent-chain latency (lat_mem_rd). One 64B block per step; the first u32
// of the block is the byte offset of the next block (host-side Python builds
// a randomized single-cycle permutation ring). t1 ~ t2 is expected: the chain
// is serial, nothing is in flight when the loop ends.
#define BENCH_MTE_CHASE_BODY()                                                   \
    do {                                                                         \
        set_atomic_none();                                                       \
        set_mask_norm();                                                         \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                             \
        BENCH_TICKS_DECL;                                                        \
        __ubuf__ uint32_t *ptr = (__ubuf__ uint32_t *)((uintptr_t)0);            \
        uint32_t acc = 0;                                                        \
        uint64_t cur = 0;                                                        \
        BENCH_TICK(0);                                                           \
        for (uint32_t i = 0; i < iters; i++) {                                   \
            CopyGmToUbufAligned((__ubuf__ uint8_t *)ptr, p0 + cur, 64);          \
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);                              \
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);                             \
            uint32_t next = ptr[0];                                              \
            acc ^= next;                                                         \
            cur = next;                                                          \
        }                                                                        \
        BENCH_TICK(1);                                                           \
        pipe_barrier(PIPE_ALL);                                                  \
        BENCH_TICK(2);                                                           \
        *(__gm__ uint32_t *)(p1 + (uint64_t)block_idx * BENCH_MTE_STRIDE) = acc; \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);    \
    } while (0)

#else  // !__DAV_C220_VEC__: cube-flavour preprocess compile sees empty stubs

#define BENCH_MTE_READ_BODY() \
    do {                      \
    } while (0)
#define BENCH_MTE_WRITE_BODY() \
    do {                       \
    } while (0)
#define BENCH_MTE_COPY_BODY() \
    do {                      \
    } while (0)
#define BENCH_MTE_CHASE_BODY() \
    do {                       \
    } while (0)

#endif  // __DAV_C220_VEC__

// ==================== experiment 4: vector-op micro-benchmarks ====================
// Pure vector-pipe throughput vs vector length: a dependent chain of vector
// instructions over UB-resident data. GM is touched only outside the measured
// window ([x0; c] staged in before t0, dst copied out after t2), so the size
// axis (sz0 = elems = elements per op application, swept from Python) measures
// the vector unit itself. Variants:
//   add_f32 / mul_f32 / add_i32   dst = dst op c    binary chains
//   add_f16                       dst = dst + c     16-bit: 128 elems/repeat
//   sqrt_f32 / exp_f32            dst = dst * op(c) transcendental carriers
// The transcendental variants must not chain op directly (an exp chain
// diverges to inf within a few steps - special-value slow path; a sqrt chain
// collapses to 1.0), so the dependency is carried by the mul while op runs on
// the CONSTANT vector: every iteration sees fresh normal inputs, and cost(op)
// is recovered by subtracting mul_f32 at the same elems/iters (same
// differential trick as the scalar family's three-slope decomposition).
// Constants are chosen with op(c) = 1 (sqrt: c = 1.0, exp: c = 0.0), so dst
// is bit-stable and the numerical check degenerates to out == x0 bitwise.
// Per-block UB layout: dst = UB[0, bytes), cst = UB[bytes, 2*bytes); the
// carriers add a scratch tile at UB[2*bytes, 2*bytes + BENCH_VEC_TMP_BYTES)
// holding op(c) for one max-length chunk (reused every chunk). Host budget
// (op.cpp XliteOpBenchVec): 2*bytes + BENCH_VEC_TMP_BYTES <= UB_SIZE, i.e.
// elems <= 16384 at 4B dtypes / 32768 at 2B.
// elems is a whole number of 256B repeats (64 elements at 4B, 128 at 2B), so
// the full mask applies and there is no tail. One intrinsic call covers at
// most VECTOR_MAX_REPEAT (255) repeats, so larger elems take several calls
// per iteration (the chunk loop below) - that multi-call amortization is
// part of what the size sweep measures. Contiguous strides throughout
// (blockStride 1 = 32B, repeatStride 8 = 256B).
//
// ⭐ RAW hazard (910B3, measured): the vector pipe OVERLAPS a subsequent
// instruction's operand reads with the previous instruction's in-flight
// writes - a barrier-free dst = dst + c chain of two iterations yields
// exactly ONE effective application (the second vadd read the pre-write
// dst). Every chunk is therefore closed with pipe_barrier(PIPE_V), the same
// load-bearing idiom as quant.h's vmul -> barrier -> vadd on the same row.
// Binary chains: [op; barrier]; carriers: [op; barrier; mul] - exactly one
// barrier per chunk in each, so the transcendental differential compares
// identical barrier structure. The carrier's leftover tmp WAR is
// value-neutral under a uniform c (micro_bench.py's fold check enforces it)
// and the cross-iteration vmul RAW is covered by the barrier before the next
// vmul. Consequence: at small elems the per-iteration cost is the
// dependent-recurrence latency (issue + streaming + barrier), not
// back-to-back issue; at large elems the barrier amortizes.
// Every block runs the SAME chain (blockNum 0 = all AIV blocks). Block b
// copies its final dst to out + b*bytes (disjoint regions, each >= 256B and
// 64B aligned - no shared-line store races) and samples its 3 SYS_CNT slots
// to ticks + b*BENCH_MTE_STRIDE (the 64B-line output contract above). Same
// double-flavour guard as the MTE family.

// Scratch tile for the transcendental carriers: one max-length chunk of
// VECTOR_MAX_REPEAT repeats (op.cpp and micro_bench.py keep the UB budget
// in sync).
#define BENCH_VEC_TMP_BYTES (VECTOR_MAX_REPEAT * VECTOR_MAX_BYTESIZE)

#ifdef __DAV_C220_VEC__

// Binary chain body: dst = dst op cst over all elems per iteration, chunked
// into <= VECTOR_MAX_REPEAT-repeat intrinsic calls. T = float / int32_t / half,
// OP = vadd / vmul. Used by add_f32 / mul_f32 / add_i32 / add_f16.
#define BENCH_VEC_BIN_BODY(T, OP)                                                              \
    do {                                                                                       \
        set_atomic_none();                                                                     \
        set_mask_norm();                                                                       \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                           \
        BENCH_TICKS_DECL;                                                                      \
        __ubuf__ T *dst = (__ubuf__ T *)((uintptr_t)0);                                        \
        __ubuf__ T *cst = (__ubuf__ T *)((uintptr_t)(sz0 * sizeof(T)));                        \
        uint32_t bytes = (uint32_t)(sz0 * sizeof(T));                                          \
        uint32_t epr = VECTOR_MAX_BYTESIZE / (uint32_t)sizeof(T);                              \
        uint32_t repTotal = (uint32_t)(sz0 / epr);                                             \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)dst, p0, bytes);                               \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)cst, p0 + bytes, bytes);                       \
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                                \
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                               \
        BENCH_TICK(0);                                                                         \
        for (uint32_t i = 0; i < iters; i++) {                                                 \
            uint32_t off = 0;                                                                  \
            uint32_t rem = repTotal;                                                           \
            while (rem > 0) {                                                                  \
                uint32_t r = rem > VECTOR_MAX_REPEAT ? VECTOR_MAX_REPEAT : rem;                \
                OP(dst + off, dst + off, cst + off, r, 1, 1, 1, 8, 8, 8);                      \
                pipe_barrier(PIPE_V);                                                          \
                off += r * epr;                                                                \
                rem -= r;                                                                      \
            }                                                                                  \
        }                                                                                      \
        BENCH_TICK(1);                                                                         \
        pipe_barrier(PIPE_ALL);                                                                \
        BENCH_TICK(2);                                                                         \
        CopyUbufToGmAligned(p1 + (uint64_t)block_idx * bytes, (__ubuf__ uint8_t *)dst, bytes); \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);                  \
    } while (0)

// Transcendental carrier body: tmp = op(cst chunk); dst chunk = dst chunk *
// tmp (UOP is vsqrt / vexp, fp32 only - see the section comment for why the
// chain is carried by the mul and how cost(op) is recovered).
#define BENCH_VEC_CARRIER_BODY(UOP)                                                            \
    do {                                                                                       \
        set_atomic_none();                                                                     \
        set_mask_norm();                                                                       \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                           \
        BENCH_TICKS_DECL;                                                                      \
        __ubuf__ float *dst = (__ubuf__ float *)((uintptr_t)0);                                \
        __ubuf__ float *cst = (__ubuf__ float *)((uintptr_t)(sz0 * sizeof(float)));            \
        __ubuf__ float *tmp = (__ubuf__ float *)((uintptr_t)(2 * sz0 * sizeof(float)));        \
        uint32_t bytes = (uint32_t)(sz0 * sizeof(float));                                      \
        uint32_t epr = VECTOR_MAX_BYTESIZE / (uint32_t)sizeof(float);                          \
        uint32_t repTotal = (uint32_t)(sz0 / epr);                                             \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)dst, p0, bytes);                               \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)cst, p0 + bytes, bytes);                       \
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                                \
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                               \
        BENCH_TICK(0);                                                                         \
        for (uint32_t i = 0; i < iters; i++) {                                                 \
            uint32_t off = 0;                                                                  \
            uint32_t rem = repTotal;                                                           \
            while (rem > 0) {                                                                  \
                uint32_t r = rem > VECTOR_MAX_REPEAT ? VECTOR_MAX_REPEAT : rem;                \
                UOP(tmp, cst + off, r, 1, 1, 8, 8);                                            \
                pipe_barrier(PIPE_V);                                                          \
                vmul(dst + off, dst + off, tmp, r, 1, 1, 1, 8, 8, 8);                          \
                off += r * epr;                                                                \
                rem -= r;                                                                      \
            }                                                                                  \
        }                                                                                      \
        BENCH_TICK(1);                                                                         \
        pipe_barrier(PIPE_ALL);                                                                \
        BENCH_TICK(2);                                                                         \
        CopyUbufToGmAligned(p1 + (uint64_t)block_idx * bytes, (__ubuf__ uint8_t *)dst, bytes); \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);                  \
    } while (0)

#define BENCH_VEC_ADD_F32_BODY() BENCH_VEC_BIN_BODY(float, vadd)
#define BENCH_VEC_MUL_F32_BODY() BENCH_VEC_BIN_BODY(float, vmul)
#define BENCH_VEC_ADD_I32_BODY() BENCH_VEC_BIN_BODY(int32_t, vadd)
#define BENCH_VEC_ADD_F16_BODY() BENCH_VEC_BIN_BODY(half, vadd)
#define BENCH_VEC_SQRT_F32_BODY() BENCH_VEC_CARRIER_BODY(vsqrt)
#define BENCH_VEC_EXP_F32_BODY() BENCH_VEC_CARRIER_BODY(vexp)

#else  // !__DAV_C220_VEC__: cube-flavour preprocess compile sees empty stubs

#define BENCH_VEC_ADD_F32_BODY() \
    do {                         \
    } while (0)
#define BENCH_VEC_MUL_F32_BODY() \
    do {                         \
    } while (0)
#define BENCH_VEC_ADD_I32_BODY() \
    do {                         \
    } while (0)
#define BENCH_VEC_ADD_F16_BODY() \
    do {                         \
    } while (0)
#define BENCH_VEC_SQRT_F32_BODY() \
    do {                          \
    } while (0)
#define BENCH_VEC_EXP_F32_BODY() \
    do {                         \
    } while (0)

#endif  // __DAV_C220_VEC__

// ==================== experiment 5: cube matmul throughput ====================
// A single A@B.T matmul per kernel launch via the proven Matmul class from
// matmul.h. The dependent-chain measurement (iters > 1) is done by the host
// issuing the kernel `iters` times (same wall-clock pattern as experiment 2b),
// NOT by looping inside the kernel — looping Matmul::Run() in-kernel causes
// L0A/L0B read/write conflicts after a few iterations because the flag
// lifecycle (SetFlags/WaitFlags) doesn't cleanly reset across iterations on
// tiles where kLoop <= 4. The kernel does one full matmul with in-kernel
// SYS_CNT sampling: tick 0 = before Run, tick 1 = after Run, tick 2 = after
// PIPE_ALL drain. L0C is re-initialized (init=true), so repeated launches
// produce the same result. Anti-DCE: A/B/C are runtime GM pointers.
// Only compiled in XLITE_BENCH_TIMED=ON builds (same gate as MTE/VEC).
//
// Double-flavour guard: cube intrinsics are rejected by the vec compile
// channel, so the real body + the cube_mmad_body template live under
// __DAV_C220_CUBE__; the vec-flavour stub is empty. KERNEL_TASK_TYPE_DEFAULT is
// emitted ONCE at the kernel top (unconditional), so the vec-flavour preprocess
// - which is the flavour CANN's auto_gen pre-scan runs on for an AIC_ONLY
// kernel - still sees the marker and emits the .ascend.meta type section
// (without it a single-kernel cube TU has no type section and runtime
// registration fails, RegisterAscendBinary ret 107000).

#ifdef __DAV_C220_CUBE__

// Cube matmul body: one Matmul::Run() (SetFlags → tile dispatch → WaitFlags),
// bracketed by SYS_CNT ticks. Only the first block (block_idx < block_num)
// runs; CANN launches AIC kernels on all cube cores, so idle cores must
// early-return to avoid SetFlags/WaitFlags on cores with no tile work.
template <typename Dtype, typename MatDtype, typename OutDtype>
__aicore__ inline void cube_mmad_body(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ticks, uint64_t m0,
                                      uint64_t n0, uint64_t k0, uint32_t blkIdx, uint64_t *tk)
{
    BENCH_TICK_PTR(tk, 0);
    using MM = Matmul<Dtype, MatDtype, OutDtype>;
    MM op;
    op.Init(m0, n0, k0, false, false, 0, 0, 0);
    op.Run(a, b, c, nullptr, nullptr, m0, n0, k0);
    BENCH_TICK_PTR(tk, 1);
    pipe_barrier(PIPE_ALL);
    BENCH_TICK_PTR(tk, 2);
    BENCH_TICKS_FLUSH_PTR(tk, ticks + (uint64_t)blkIdx * 8, 3);
}

#define BENCH_CUBE_MMAD_BODY()                                                             \
    do {                                                                                   \
        if (block_idx < block_num) {                                                       \
            BENCH_TICKS_DECL;                                                              \
            cube_mmad_body<half, float, half>(p0, p1, p2, ticks, sz0, sz1, sz2, block_idx, \
                                              benchTicks);                                 \
        }                                                                                  \
    } while (0)

#else  // !__DAV_C220_CUBE__: vec-flavour preprocess compile sees empty stub

#define BENCH_CUBE_MMAD_BODY() \
    do {                       \
    } while (0)

#endif  // __DAV_C220_CUBE__

#endif  // MICROBENCH_TIMED
