/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "matmul.h"

// Micro-benchmark kernel family. ALL implementation lives in this header;
// the bench_*.cpp files are pure per-core item lists. Experiments, in order:
//   1. empty launch-floor probes   empty_<ct>            both builds
//   2. scalar dependent chains     scalar_<op>_<ct>      both builds
//      + timed twins               scalar_<op>_<ct>_timed   MICROBENCH_TIMED
//   3. MTE memory bandwidth/latency mte_<op>_aiv        bench_aiv.cpp (timed)
//   4. vector-op throughput        vec_<op>_aiv          bench_aiv.cpp (timed)
// (3/4 exist only in XLITE_BENCH_TIMED=ON builds.) Design rationale, measured
// data and the reproduction recipe: doc/kernels/microbench.md and
// tests/performance/micro_bench.py.
//
// ── Core-kind routing ──
// KERNEL_TASK_TYPE_DEFAULT routes a kernel into a device compile channel; the
// lowercase core suffix used in kernel names maps to:
//   aiv  KERNEL_TYPE_AIV_ONLY     vector cores only; numBlocks counts AIV blocks
//   aic  KERNEL_TYPE_AIC_ONLY     cube cores only;   numBlocks counts AIC blocks
//   mix  KERNEL_TYPE_MIX_AIC_1_2  cube+vector in one binary, scheduled
//                                 1 AIC : 2 AIV (same mix shape as
//                                 indexer_topk / cxa / flash_attention);
//                                 numBlocks counts AIC blocks, each paired
//                                 with 2 AIV blocks by hardware
//
// ── TU composition rules (both are measurement-validity constraints) ──
// 1. The ascendc precompile scan extracts the task type per source file
//    (the first KERNEL_TASK_TYPE_DEFAULT marker wins for the whole file), so
//    kernels with different core types must not share a translation unit;
//    same-type kernels may (grouped this way, same as all_reduce.cpp).
// 2. The DEFAULT build's TU composition is FROZEN: the three *_ITEMS macros
//    below (BENCH_TIMED_ITEMS / BENCH_MTE_ITEMS / BENCH_VEC_ITEMS) expand
//    to nothing unless the build defines MICROBENCH_TIMED, so the default
//    TUs stay token-identical to the calibrated layout and the delivered
//    numbers carry over. A timed build ADDS the *_timed twins plus the MTE
//    and vector families to the same bench_aiv TU, which shifts codegen
//    context: 1-slot integer ops are
//    TU-sensitive (adding the twins once moved the untouched scalar_mul_aiv
//    from 1.667 to 1.112 ns/op, 3 -> 2 issue slots; latency-bound fadd/div
//    did not move). Timed-build numbers are therefore their own baseline -
//    never compare them with default-build numbers, and re-baseline after
//    adding items to either list.

// ==================== common: pasting + core-kind routing ====================
// Concatenation helpers (expand arguments before pasting).
#define BENCH_CAT(a, b) BENCH_CAT_IMPL(a, b)
#define BENCH_CAT_IMPL(a, b) a##b
#define BENCH_CAT3(a, b, c) BENCH_CAT3_IMPL(a, b, c)
#define BENCH_CAT3_IMPL(a, b, c) a##b##c

// Core-kind task types (lowercase aliases, for pasting with the kernel name
// suffix; e.g. BENCH_CAT(BENCH_TASK_, aiv) -> KERNEL_TYPE_AIV_ONLY).
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
// cannot go below. The single unused GM_ADDR parameter exists because the
// host-stub extractor requires a non-empty parameter list; the host side
// passes nullptr (optional params such as matmul bias are routinely launched
// as nullptr, so the launch command stays on the real code path).

#define EMPTY_FUNC_DEFINE(name, task_type)                    \
    extern "C" __global__ __aicore__ void name(GM_ADDR dummy) \
    {                                                         \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                  \
    }

// Empty launch-floor probe (per core kind). Unconditional: both builds need
// the launch floor and the host empty_* API.
#define BENCH_EMPTY(ct) EMPTY_FUNC_DEFINE(BENCH_CAT(empty_, ct), BENCH_CAT(BENCH_TASK_, ct))

// ==================== experiment 2: scalar dependent chains ====================
// A dependent chain of `iters` scalar ops (S pipe), loop count as a runtime
// kernel parameter so variants can be swept from Python without rebuilding.
// Measurement-validity design (each item load-bearing):
// - anti-DCE: the accumulator is consumed by storing it to GM at the end.
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
// Timed twins (build-level MICROBENCH_TIMED): the SAME bodies wrapped with
// in-kernel SYS_CNT sampling (kernel_macro.h BENCH_TICK*): slot 0 = region
// start, slot 1 = loop issued, PIPE_ALL drain, slot 2 = true completion;
// accumulator and tick slots are stored only after the closing tick, so no
// store lands inside the measured window. Host entry: XliteOpBenchTimed -
// one dispatch for the whole timed family.

#define SCALAR_KERNEL_DEFINE(name, task_type, Dtype, body)                              \
    extern "C" __global__ __aicore__ void name(GM_ADDR out, Dtype seed, uint32_t iters) \
    {                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                            \
        Dtype acc = seed;                                                               \
        Dtype x = seed;                                                                 \
        for (uint32_t i = 0; i < iters; i++) {                                          \
            body;                                                                       \
        }                                                                               \
        *(__gm__ Dtype *)out = acc;                                                     \
        pipe_barrier(PIPE_ALL);                                                         \
    }

// Unrolled-by-4 variant: 4 dependent adds per loop trip, iters = total add
// count (host side rejects iters not divisible by 4). Counter values are
// 4t+0..3 so each add's entangled operand matches the plain variant's work.
#define SCALAR_ADD_UNROLL4_KERNEL_DEFINE(name, task_type)                                  \
    extern "C" __global__ __aicore__ void name(GM_ADDR out, uint32_t seed, uint32_t iters) \
    {                                                                                      \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                               \
        uint32_t acc = seed;                                                               \
        uint32_t x = seed;                                                                 \
        for (uint32_t t = 0; t < iters / 4; t++) {                                         \
            acc = acc + (x ^ (4 * t));                                                     \
            acc = acc + (x ^ (4 * t + 1));                                                 \
            acc = acc + (x ^ (4 * t + 2));                                                 \
            acc = acc + (x ^ (4 * t + 3));                                                 \
        }                                                                                  \
        *(__gm__ uint32_t *)out = acc;                                                     \
        pipe_barrier(PIPE_ALL);                                                            \
    }

// Per-variant bodies: the measured scalar op itself. add/xor entangle the
// operand with the loop counter (anti-folding, see the section comment);
// mul/div/fadd chains are naturally unfoldable and take the plain operand.
#define BENCH_BODY_ADD acc = acc + (x ^ i)
#define BENCH_BODY_XOR acc = acc ^ (x ^ i)
#define BENCH_BODY_MUL acc = acc * x
#define BENCH_BODY_DIV acc = acc / x
#define BENCH_BODY_FADD acc = acc + x

// Scalar dependent chains (uint32, except fadd on float).
#define BENCH_SCALAR_ADD(ct)                                                               \
    SCALAR_KERNEL_DEFINE(BENCH_CAT(scalar_add_, ct), BENCH_CAT(BENCH_TASK_, ct), uint32_t, \
                         BENCH_BODY_ADD)
#define BENCH_SCALAR_XOR(ct)                                                               \
    SCALAR_KERNEL_DEFINE(BENCH_CAT(scalar_xor_, ct), BENCH_CAT(BENCH_TASK_, ct), uint32_t, \
                         BENCH_BODY_XOR)
#define BENCH_SCALAR_MUL(ct)                                                               \
    SCALAR_KERNEL_DEFINE(BENCH_CAT(scalar_mul_, ct), BENCH_CAT(BENCH_TASK_, ct), uint32_t, \
                         BENCH_BODY_MUL)
#define BENCH_SCALAR_DIV(ct)                                                               \
    SCALAR_KERNEL_DEFINE(BENCH_CAT(scalar_div_, ct), BENCH_CAT(BENCH_TASK_, ct), uint32_t, \
                         BENCH_BODY_DIV)
#define BENCH_SCALAR_FADD(ct)                                                            \
    SCALAR_KERNEL_DEFINE(BENCH_CAT(scalar_fadd_, ct), BENCH_CAT(BENCH_TASK_, ct), float, \
                         BENCH_BODY_FADD)
#define BENCH_SCALAR_ADD_UNROLL4(ct) \
    SCALAR_ADD_UNROLL4_KERNEL_DEFINE(BENCH_CAT(scalar_add_unroll4_, ct), BENCH_CAT(BENCH_TASK_, ct))

#if MICROBENCH_TIMED
// Both timed generators take the FULL kernel name (assembled by the callers
// with a single BENCH_CAT3 paste). CANN's auto_gen wrapper #defines the base
// name (scalar_add_aic -> scalar_add_aic_origin) around the #include of the
// item list, so assembling the suffix in a second step (BENCH_CAT(name,
// _timed) here) would let the rescan rename the base first and paste _timed
// onto the renamed token (scalar_add_aic_origin_timed) instead of the
// scalar_add_aic_timed_origin the generated wrapper calls. One BENCH_CAT3
// paste forms the complete name as a single token, which the rescan then
// renames wholesale - the same mechanism that renames the plain kernels.
#define SCALAR_KERNEL_DEFINE_TIMED(name, task_type, Dtype, body)                       \
    extern "C" __global__ __aicore__ void name(GM_ADDR out, GM_ADDR ticks, Dtype seed, \
                                               uint32_t iters)                         \
    {                                                                                  \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                           \
        BENCH_TICKS_DECL;                                                              \
        Dtype acc = seed;                                                              \
        Dtype x = seed;                                                                \
        BENCH_TICK(0);                                                                 \
        for (uint32_t i = 0; i < iters; i++) {                                         \
            body;                                                                      \
        }                                                                              \
        BENCH_TICK(1);                                                                 \
        pipe_barrier(PIPE_ALL);                                                        \
        BENCH_TICK(2);                                                                 \
        *(__gm__ Dtype *)out = acc;                                                    \
        BENCH_TICKS_FLUSH(ticks, 3);                                                   \
    }

#define SCALAR_ADD_UNROLL4_KERNEL_DEFINE_TIMED(name, task_type)                           \
    extern "C" __global__ __aicore__ void name(GM_ADDR out, GM_ADDR ticks, uint32_t seed, \
                                               uint32_t iters)                            \
    {                                                                                     \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                              \
        BENCH_TICKS_DECL;                                                                 \
        uint32_t acc = seed;                                                              \
        uint32_t x = seed;                                                                \
        BENCH_TICK(0);                                                                    \
        for (uint32_t t = 0; t < iters / 4; t++) {                                        \
            acc = acc + (x ^ (4 * t));                                                    \
            acc = acc + (x ^ (4 * t + 1));                                                \
            acc = acc + (x ^ (4 * t + 2));                                                \
            acc = acc + (x ^ (4 * t + 3));                                                \
        }                                                                                 \
        BENCH_TICK(1);                                                                    \
        pipe_barrier(PIPE_ALL);                                                           \
        BENCH_TICK(2);                                                                    \
        *(__gm__ uint32_t *)out = acc;                                                    \
        BENCH_TICKS_FLUSH(ticks, 3);                                                      \
    }

// Timed twins of the scalar items (same names + _timed suffix, extra GM_ADDR
// ticks parameter). One macro per variant, mirroring the plain items above;
// the name is built with a single BENCH_CAT3 paste (see the note above the
// timed generators for why the suffix must not be pasted in a second step).
#define BENCH_SCALAR_ADD_TIMED(ct)                                                              \
    SCALAR_KERNEL_DEFINE_TIMED(BENCH_CAT3(scalar_add_, ct, _timed), BENCH_CAT(BENCH_TASK_, ct), \
                               uint32_t, BENCH_BODY_ADD)
#define BENCH_SCALAR_XOR_TIMED(ct)                                                              \
    SCALAR_KERNEL_DEFINE_TIMED(BENCH_CAT3(scalar_xor_, ct, _timed), BENCH_CAT(BENCH_TASK_, ct), \
                               uint32_t, BENCH_BODY_XOR)
#define BENCH_SCALAR_MUL_TIMED(ct)                                                              \
    SCALAR_KERNEL_DEFINE_TIMED(BENCH_CAT3(scalar_mul_, ct, _timed), BENCH_CAT(BENCH_TASK_, ct), \
                               uint32_t, BENCH_BODY_MUL)
#define BENCH_SCALAR_DIV_TIMED(ct)                                                              \
    SCALAR_KERNEL_DEFINE_TIMED(BENCH_CAT3(scalar_div_, ct, _timed), BENCH_CAT(BENCH_TASK_, ct), \
                               uint32_t, BENCH_BODY_DIV)
#define BENCH_SCALAR_FADD_TIMED(ct)                                                              \
    SCALAR_KERNEL_DEFINE_TIMED(BENCH_CAT3(scalar_fadd_, ct, _timed), BENCH_CAT(BENCH_TASK_, ct), \
                               float, BENCH_BODY_FADD)
#define BENCH_SCALAR_ADD_UNROLL4_TIMED(ct)                                              \
    SCALAR_ADD_UNROLL4_KERNEL_DEFINE_TIMED(BENCH_CAT3(scalar_add_unroll4_, ct, _timed), \
                                           BENCH_CAT(BENCH_TASK_, ct))

// The whole timed family in one line per core list. Invoke WITHOUT a
// trailing semicolon: the default-build expansion is empty, so the default
// TUs stay token-identical to the frozen 7-kernel layout.
#define BENCH_TIMED_ITEMS(ct)    \
    BENCH_SCALAR_ADD_TIMED(ct);  \
    BENCH_SCALAR_XOR_TIMED(ct);  \
    BENCH_SCALAR_MUL_TIMED(ct);  \
    BENCH_SCALAR_DIV_TIMED(ct);  \
    BENCH_SCALAR_FADD_TIMED(ct); \
    BENCH_SCALAR_ADD_UNROLL4_TIMED(ct)
#else
#define BENCH_TIMED_ITEMS(ct)
#endif  // MICROBENCH_TIMED

// ==================== experiment 3: MTE memory micro-benchmarks ====================
// lmbench-style probes driven entirely by the MTE pipes; timing comes from
// the same SYS_CNT slots, so this family shares the timed build. Item list:
// BENCH_MTE_ITEMS in bench_aiv.cpp (AIV only - MTE lives on the vector
// core), which expands to nothing in the default build so the frozen TU
// stays untouched. lmbench mapping:
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
// Unified signature mte_<op>_aiv(src, dst, ticks, bytes, iters): bytes =
// working-set size swept from Python (host enforces %32, chase %64 because
// one step is a 64B block), iters = reps of the whole working set (chase:
// chain steps). Blocks partition [0, bytes) grid-stride in BENCH_MTE_TILE
// segments (split.cpp idiom); chase is single-block by nature (the host
// clamps blockNum to 1). The working set is re-streamed every rep without
// any flush, so the size sweep itself exposes L2 residency - that curve is
// the point, matching lmbench.
// Double-flavour guard: every common TU is preprocessed in BOTH the vec and
// cube flavours and MTE intrinsics are rejected by dav-c220-cube, so the
// generators are double-defined dequant.h style: real bodies under
// __DAV_C220_VEC__, same-signature empty stubs otherwise (symbol set
// identical on both channels; task-type extraction runs on the vec flavour).

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

// Cube-flavour stub: same signature, empty body (see the section comment).
#define MTE_EMPTY_KERNEL_DEFINE(name)                                                   \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR dst, GM_ADDR ticks, \
                                               uint64_t bytes, uint32_t iters)          \
    {                                                                                   \
    }

#ifdef __DAV_C220_VEC__

// Streaming read. The post-window consume (one word of the last-loaded tile,
// after the closing tick and an MTE2->S handshake) makes the dst value
// predictable for the Python check; the loads themselves cannot be elided -
// MTE copies are opaque DMA intrinsics - and micro_bench.py guards this with
// a time-vs-iters ratio.
#define MTE_READ_KERNEL_DEFINE(name, task_type)                                         \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR dst, GM_ADDR ticks, \
                                               uint64_t bytes, uint32_t iters)          \
    {                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                            \
        set_atomic_none();                                                              \
        set_mask_norm();                                                                \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                    \
        BENCH_TICKS_DECL;                                                               \
        __ubuf__ uint8_t *ring[2] = {                                                   \
            (__ubuf__ uint8_t *)((uintptr_t)0),                                         \
            (__ubuf__ uint8_t *)((uintptr_t)BENCH_MTE_TILE),                            \
        };                                                                              \
        uint64_t half = BENCH_MTE_TILE;                                                 \
        uint32_t segCnt = 0;                                                            \
        BENCH_TICK(0);                                                                  \
        for (uint32_t r = 0; r < iters; r++) {                                          \
            for (uint64_t gbo = (uint64_t)block_idx * half; gbo < bytes;                \
                 gbo += (uint64_t)block_num * half) {                                   \
                uint64_t remain = bytes - gbo;                                          \
                uint32_t len = (uint32_t)(remain < half ? remain : half);               \
                CopyGmToUbufAligned(ring[segCnt & 1], src + gbo, len);                  \
                segCnt++;                                                               \
            }                                                                           \
        }                                                                               \
        BENCH_TICK(1);                                                                  \
        pipe_barrier(PIPE_ALL);                                                         \
        BENCH_TICK(2);                                                                  \
        uint32_t acc = 0;                                                               \
        if (segCnt > 0) {                                                               \
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID2);                                     \
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID2);                                    \
            acc = *(__ubuf__ uint32_t *)ring[(segCnt - 1) & 1];                         \
        }                                                                               \
        *(__gm__ uint32_t *)(dst + (uint64_t)block_idx * BENCH_MTE_STRIDE) = acc;       \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);           \
    }

// Streaming write. The pattern tile is filled from src once and the MTE2->MTE3
// handshake completes BEFORE the opening tick (otherwise it would pollute the
// window); afterwards UB is only read by MTE3, so no further sync is needed.
// dst ends up as the src prefix tiled along every block's segment walk.
#define MTE_WRITE_KERNEL_DEFINE(name, task_type)                                        \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR dst, GM_ADDR ticks, \
                                               uint64_t bytes, uint32_t iters)          \
    {                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                            \
        set_atomic_none();                                                              \
        set_mask_norm();                                                                \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                    \
        BENCH_TICKS_DECL;                                                               \
        __ubuf__ uint8_t *pat = (__ubuf__ uint8_t *)((uintptr_t)0);                     \
        uint64_t half = BENCH_MTE_TILE;                                                 \
        uint64_t fillLen = bytes < half ? bytes : half;                                 \
        CopyGmToUbufAligned(pat, src, (uint32_t)fillLen);                               \
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);                                      \
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);                                     \
        BENCH_TICK(0);                                                                  \
        for (uint32_t r = 0; r < iters; r++) {                                          \
            for (uint64_t gbo = (uint64_t)block_idx * half; gbo < bytes;                \
                 gbo += (uint64_t)block_num * half) {                                   \
                uint64_t remain = bytes - gbo;                                          \
                uint32_t len = (uint32_t)(remain < half ? remain : half);               \
                CopyUbufToGmAligned(dst + gbo, pat, len);                               \
            }                                                                           \
        }                                                                               \
        BENCH_TICK(1);                                                                  \
        pipe_barrier(PIPE_ALL);                                                         \
        BENCH_TICK(2);                                                                  \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);           \
    }

// bcopy. Each rep is one complete split.cpp-style ping-pong execution: seed
// both MTE3->MTE2 buffer-free flags, walk the grid-stride segments loading
// into one half while storing the previous half (MTE2<->MTE3 flag pairs),
// flush the pending tail, then wait both flags dry - balanced even for an
// idle block. The per-rep tail wait drains the last stores inside the
// window, so t1 already ~ t2 for this variant.
#define MTE_COPY_KERNEL_DEFINE(name, task_type)                                         \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR dst, GM_ADDR ticks, \
                                               uint64_t bytes, uint32_t iters)          \
    {                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                            \
        set_atomic_none();                                                              \
        set_mask_norm();                                                                \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                    \
        BENCH_TICKS_DECL;                                                               \
        __ubuf__ uint8_t *buf[2] = {                                                    \
            (__ubuf__ uint8_t *)((uintptr_t)0),                                         \
            (__ubuf__ uint8_t *)((uintptr_t)BENCH_MTE_TILE),                            \
        };                                                                              \
        uint64_t half = BENCH_MTE_TILE;                                                 \
        BENCH_TICK(0);                                                                  \
        for (uint32_t r = 0; r < iters; r++) {                                          \
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);                                  \
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);                                  \
            uint32_t curr = 0;                                                          \
            bool hasPending = false;                                                    \
            uint64_t pendGbo = 0;                                                       \
            uint32_t pendLen = 0;                                                       \
            for (uint64_t gbo = (uint64_t)block_idx * half; gbo < bytes;                \
                 gbo += (uint64_t)block_num * half) {                                   \
                uint64_t remain = bytes - gbo;                                          \
                uint32_t len = (uint32_t)(remain < half ? remain : half);               \
                if (hasPending) {                                                       \
                    uint32_t wbuf = 1 - curr;                                           \
                    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);                  \
                    CopyUbufToGmAligned(dst + pendGbo, buf[wbuf], pendLen);             \
                    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);                   \
                }                                                                       \
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + curr);                      \
                CopyGmToUbufAligned(buf[curr], src + gbo, len);                         \
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + curr);                       \
                pendGbo = gbo;                                                          \
                pendLen = len;                                                          \
                hasPending = true;                                                      \
                curr = 1 - curr;                                                        \
            }                                                                           \
            if (hasPending) {                                                           \
                uint32_t wbuf = 1 - curr;                                               \
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0 + wbuf);                      \
                CopyUbufToGmAligned(dst + pendGbo, buf[wbuf], pendLen);                 \
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + wbuf);                       \
            }                                                                           \
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);                                 \
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);                                 \
        }                                                                               \
        BENCH_TICK(1);                                                                  \
        pipe_barrier(PIPE_ALL);                                                         \
        BENCH_TICK(2);                                                                  \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);           \
    }

// Dependent-chain latency (lat_mem_rd). One 64B block per step; the first u32
// of the block is the byte offset of the next block (host-side Python builds
// a randomized single-cycle permutation ring). t1 ~ t2 is expected: the chain
// is serial, nothing is in flight when the loop ends.
#define MTE_CHASE_KERNEL_DEFINE(name, task_type)                                        \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR dst, GM_ADDR ticks, \
                                               uint64_t bytes, uint32_t iters)          \
    {                                                                                   \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                            \
        set_atomic_none();                                                              \
        set_mask_norm();                                                                \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                    \
        BENCH_TICKS_DECL;                                                               \
        __ubuf__ uint32_t *ptr = (__ubuf__ uint32_t *)((uintptr_t)0);                   \
        uint32_t acc = 0;                                                               \
        uint64_t cur = 0;                                                               \
        BENCH_TICK(0);                                                                  \
        for (uint32_t i = 0; i < iters; i++) {                                          \
            CopyGmToUbufAligned((__ubuf__ uint8_t *)ptr, src + cur, 64);                \
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);                                     \
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);                                    \
            uint32_t next = ptr[0];                                                     \
            acc ^= next;                                                                \
            cur = next;                                                                 \
        }                                                                               \
        BENCH_TICK(1);                                                                  \
        pipe_barrier(PIPE_ALL);                                                         \
        BENCH_TICK(2);                                                                  \
        *(__gm__ uint32_t *)(dst + (uint64_t)block_idx * BENCH_MTE_STRIDE) = acc;       \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);           \
    }

#else  // !__DAV_C220_VEC__: cube-flavour preprocess compile sees empty stubs

#define MTE_READ_KERNEL_DEFINE(name, task_type) MTE_EMPTY_KERNEL_DEFINE(name)
#define MTE_WRITE_KERNEL_DEFINE(name, task_type) MTE_EMPTY_KERNEL_DEFINE(name)
#define MTE_COPY_KERNEL_DEFINE(name, task_type) MTE_EMPTY_KERNEL_DEFINE(name)
#define MTE_CHASE_KERNEL_DEFINE(name, task_type) MTE_EMPTY_KERNEL_DEFINE(name)

#endif  // __DAV_C220_VEC__

// MTE test items, collected by BENCH_MTE_ITEMS (bench_aiv.cpp, timed build
// only). The kernel name is
// assembled with ONE BENCH_CAT paste: CANN's auto_gen wrapper #defines the
// base name (mte_read_aiv -> mte_read_aiv_origin) around the #include of the
// item list, and a single pasted token is renamed wholesale on rescan (same
// rule as the timed twins above).

#define BENCH_MTE_READ(ct) \
    MTE_READ_KERNEL_DEFINE(BENCH_CAT(mte_read_, ct), BENCH_CAT(BENCH_TASK_, ct))
#define BENCH_MTE_WRITE(ct) \
    MTE_WRITE_KERNEL_DEFINE(BENCH_CAT(mte_write_, ct), BENCH_CAT(BENCH_TASK_, ct))
#define BENCH_MTE_COPY(ct) \
    MTE_COPY_KERNEL_DEFINE(BENCH_CAT(mte_copy_, ct), BENCH_CAT(BENCH_TASK_, ct))
#define BENCH_MTE_CHASE(ct) \
    MTE_CHASE_KERNEL_DEFINE(BENCH_CAT(mte_chase_, ct), BENCH_CAT(BENCH_TASK_, ct))

// The whole MTE family in one line (AIV only - see the section comment).
// Invoke WITHOUT a trailing semicolon: the default-build expansion is
// empty, keeping bench_aiv.cpp token-identical to the frozen layout (same
// contract as BENCH_TIMED_ITEMS above).
#if MICROBENCH_TIMED
#define BENCH_MTE_ITEMS(ct) \
    BENCH_MTE_READ(ct);     \
    BENCH_MTE_WRITE(ct);    \
    BENCH_MTE_COPY(ct);     \
    BENCH_MTE_CHASE(ct)
#else
#define BENCH_MTE_ITEMS(ct)
#endif  // MICROBENCH_TIMED

// ==================== experiment 4: vector-op micro-benchmarks ====================
// Pure vector-pipe throughput vs vector length: a dependent chain of vector
// instructions over UB-resident data. GM is touched only outside the measured
// window ([x0; c] staged in before t0, dst copied out after t2), so the size
// axis (elems = elements per op application, swept from Python) measures the
// vector unit itself. Variants:
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
// double-flavour guard as the MTE family. Item list: BENCH_VEC_ITEMS in
// bench_aiv.cpp (timed build only).

// Scratch tile for the transcendental carriers: one max-length chunk of
// VECTOR_MAX_REPEAT repeats (op.cpp and micro_bench.py keep the UB budget
// in sync).
#define BENCH_VEC_TMP_BYTES (VECTOR_MAX_REPEAT * VECTOR_MAX_BYTESIZE)

// Cube-flavour stub: same signature, empty body (see the section comment).
#define VEC_EMPTY_KERNEL_DEFINE(name)                                                   \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR out, GM_ADDR ticks, \
                                               uint64_t elems, uint32_t iters)          \
    {                                                                                   \
    }

#ifdef __DAV_C220_VEC__

// Binary chain kernel: dst = dst op cst over all elems per iteration, chunked
// into <= VECTOR_MAX_REPEAT-repeat intrinsic calls (OP is vadd / vmul; T is
// float / int32_t / half).
#define VEC_BIN_KERNEL_DEFINE(name, task_type, T, OP)                                           \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR out, GM_ADDR ticks,         \
                                               uint64_t elems, uint32_t iters)                  \
    {                                                                                           \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                                    \
        set_atomic_none();                                                                      \
        set_mask_norm();                                                                        \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                            \
        BENCH_TICKS_DECL;                                                                       \
        __ubuf__ T *dst = (__ubuf__ T *)((uintptr_t)0);                                         \
        __ubuf__ T *cst = (__ubuf__ T *)((uintptr_t)(elems * sizeof(T)));                       \
        uint32_t bytes = (uint32_t)(elems * sizeof(T));                                         \
        uint32_t epr = VECTOR_MAX_BYTESIZE / (uint32_t)sizeof(T);                               \
        uint32_t repTotal = (uint32_t)(elems / epr);                                            \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)dst, src, bytes);                               \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)cst, src + bytes, bytes);                       \
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                                 \
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                                \
        BENCH_TICK(0);                                                                          \
        for (uint32_t i = 0; i < iters; i++) {                                                  \
            uint32_t off = 0;                                                                   \
            uint32_t rem = repTotal;                                                            \
            while (rem > 0) {                                                                   \
                uint32_t r = rem > VECTOR_MAX_REPEAT ? VECTOR_MAX_REPEAT : rem;                 \
                OP(dst + off, dst + off, cst + off, r, 1, 1, 1, 8, 8, 8);                       \
                pipe_barrier(PIPE_V);                                                           \
                off += r * epr;                                                                 \
                rem -= r;                                                                       \
            }                                                                                   \
        }                                                                                       \
        BENCH_TICK(1);                                                                          \
        pipe_barrier(PIPE_ALL);                                                                 \
        BENCH_TICK(2);                                                                          \
        CopyUbufToGmAligned(out + (uint64_t)block_idx * bytes, (__ubuf__ uint8_t *)dst, bytes); \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);                   \
    }

// Transcendental carrier kernel: tmp = op(cst chunk); dst chunk = dst chunk *
// tmp (UOP is vsqrt / vexp, fp32 only - see the section comment for why the
// chain is carried by the mul and how cost(op) is recovered).
#define VEC_CARRIER_KERNEL_DEFINE(name, task_type, UOP)                                         \
    extern "C" __global__ __aicore__ void name(GM_ADDR src, GM_ADDR out, GM_ADDR ticks,         \
                                               uint64_t elems, uint32_t iters)                  \
    {                                                                                           \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                                    \
        set_atomic_none();                                                                      \
        set_mask_norm();                                                                        \
        set_vector_mask((uint64_t)-1, (uint64_t)-1);                                            \
        BENCH_TICKS_DECL;                                                                       \
        __ubuf__ float *dst = (__ubuf__ float *)((uintptr_t)0);                                 \
        __ubuf__ float *cst = (__ubuf__ float *)((uintptr_t)(elems * sizeof(float)));           \
        __ubuf__ float *tmp = (__ubuf__ float *)((uintptr_t)(2 * elems * sizeof(float)));       \
        uint32_t bytes = (uint32_t)(elems * sizeof(float));                                     \
        uint32_t epr = VECTOR_MAX_BYTESIZE / (uint32_t)sizeof(float);                           \
        uint32_t repTotal = (uint32_t)(elems / epr);                                            \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)dst, src, bytes);                               \
        CopyGmToUbufAligned((__ubuf__ uint8_t *)cst, src + bytes, bytes);                       \
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                                 \
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);                                                \
        BENCH_TICK(0);                                                                          \
        for (uint32_t i = 0; i < iters; i++) {                                                  \
            uint32_t off = 0;                                                                   \
            uint32_t rem = repTotal;                                                            \
            while (rem > 0) {                                                                   \
                uint32_t r = rem > VECTOR_MAX_REPEAT ? VECTOR_MAX_REPEAT : rem;                 \
                UOP(tmp, cst + off, r, 1, 1, 8, 8);                                             \
                pipe_barrier(PIPE_V);                                                           \
                vmul(dst + off, dst + off, tmp, r, 1, 1, 1, 8, 8, 8);                           \
                off += r * epr;                                                                 \
                rem -= r;                                                                       \
            }                                                                                   \
        }                                                                                       \
        BENCH_TICK(1);                                                                          \
        pipe_barrier(PIPE_ALL);                                                                 \
        BENCH_TICK(2);                                                                          \
        CopyUbufToGmAligned(out + (uint64_t)block_idx * bytes, (__ubuf__ uint8_t *)dst, bytes); \
        BENCH_TICKS_FLUSH(ticks + (uint64_t)block_idx * BENCH_MTE_STRIDE, 3);                   \
    }

#else  // !__DAV_C220_VEC__: cube-flavour preprocess compile sees empty stubs

#define VEC_BIN_KERNEL_DEFINE(name, task_type, T, OP) VEC_EMPTY_KERNEL_DEFINE(name)
#define VEC_CARRIER_KERNEL_DEFINE(name, task_type, UOP) VEC_EMPTY_KERNEL_DEFINE(name)

#endif  // __DAV_C220_VEC__

// Vector-op test items, collected by BENCH_VEC_ITEMS (bench_aiv.cpp, timed
// build only; AIV only - the vector pipe lives on the vector core). Single
// BENCH_CAT paste per kernel name, same auto_gen rule as the MTE items.

#define BENCH_VEC_ADD_F32(ct) \
    VEC_BIN_KERNEL_DEFINE(BENCH_CAT(vec_add_f32_, ct), BENCH_CAT(BENCH_TASK_, ct), float, vadd)
#define BENCH_VEC_MUL_F32(ct) \
    VEC_BIN_KERNEL_DEFINE(BENCH_CAT(vec_mul_f32_, ct), BENCH_CAT(BENCH_TASK_, ct), float, vmul)
#define BENCH_VEC_ADD_I32(ct) \
    VEC_BIN_KERNEL_DEFINE(BENCH_CAT(vec_add_i32_, ct), BENCH_CAT(BENCH_TASK_, ct), int32_t, vadd)
#define BENCH_VEC_ADD_F16(ct) \
    VEC_BIN_KERNEL_DEFINE(BENCH_CAT(vec_add_f16_, ct), BENCH_CAT(BENCH_TASK_, ct), half, vadd)
#define BENCH_VEC_SQRT_F32(ct) \
    VEC_CARRIER_KERNEL_DEFINE(BENCH_CAT(vec_sqrt_f32_, ct), BENCH_CAT(BENCH_TASK_, ct), vsqrt)
#define BENCH_VEC_EXP_F32(ct) \
    VEC_CARRIER_KERNEL_DEFINE(BENCH_CAT(vec_exp_f32_, ct), BENCH_CAT(BENCH_TASK_, ct), vexp)

// The whole vector family in one line (AIV only). Same no-semicolon /
// default-empty contract as BENCH_MTE_ITEMS above.
#if MICROBENCH_TIMED
#define BENCH_VEC_ITEMS(ct) \
    BENCH_VEC_ADD_F32(ct);  \
    BENCH_VEC_MUL_F32(ct);  \
    BENCH_VEC_ADD_I32(ct);  \
    BENCH_VEC_ADD_F16(ct);  \
    BENCH_VEC_SQRT_F32(ct); \
    BENCH_VEC_EXP_F32(ct)
#else
#define BENCH_VEC_ITEMS(ct)
#endif  // MICROBENCH_TIMED

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

#if MICROBENCH_TIMED

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

#define CUBE_MMAD_KERNEL_DEFINE(name, task_type, Dtype, MatDtype, OutDtype)                    \
    extern "C" __global__ __aicore__ void name(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ticks, \
                                               uint64_t m0, uint64_t n0, uint64_t k0)          \
    {                                                                                          \
        KERNEL_TASK_TYPE_DEFAULT(task_type);                                                   \
        if (block_idx >= block_num)                                                            \
            return;                                                                            \
        BENCH_TICKS_DECL;                                                                      \
        cube_mmad_body<Dtype, MatDtype, OutDtype>(a, b, c, ticks, m0, n0, k0, block_idx,       \
                                                  benchTicks);                                 \
    }

#else  // !__DAV_C220_CUBE__: vec-flavour preprocess compile sees empty stubs
// Same 7-parameter signature as the real cube kernel so CANN's auto_gen
// pre-scan (which runs on this flavour) emits a matching wrapper; the body
// is empty because cube intrinsics are rejected by the vec compile channel.
#define CUBE_MMAD_KERNEL_DEFINE(name, task_type, Dtype, MatDtype, OutDtype)                    \
    extern "C" __global__ __aicore__ void name(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ticks, \
                                               uint64_t m0, uint64_t n0, uint64_t k0)          \
    {                                                                                          \
    }
#endif  // __DAV_C220_CUBE__

#define BENCH_CUBE_MMAD(ct)                                                                     \
    CUBE_MMAD_KERNEL_DEFINE(BENCH_CAT(cube_mmad_, ct), BENCH_CAT(BENCH_TASK_, ct), half, float, \
                            half)

// The whole cube family in one line (AIC only). Same no-semicolon /
// default-empty contract as BENCH_MTE_ITEMS above.
#define BENCH_CUBE_ITEMS(ct) BENCH_CUBE_MMAD(ct)

#else
#define BENCH_CUBE_ITEMS(ct)
#endif  // MICROBENCH_TIMED
