/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_KERNEL_PARAM_H_
#define _XLITE_KERNEL_PARAM_H_

// ccl kernel param
#define XLITE_CCL_MAX_RANK_SIZE 32
#define XLITE_IPC_MEM_FLAG_OFFSET 4096
#define COPY_SIZE 32768
#define MAX_TOTAL_COPY_SIZE 12800
#define DOUBLE_AIVNUM_SIZE_BOUND 327680

// ipc mem layout
struct XcclIpcMemData {
    uint64_t inputOffset;
    uint64_t outputOffset;
};

struct XcclParam {
    uint64_t ipcMems[XLITE_CCL_MAX_RANK_SIZE];
    uint64_t ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE];
};

// norm
enum class NormKind {
    Rms,
    Layer,
    L2,
};

// hardware
#define UB_BLOCK_SIZE 32  // unified buffer block size in bytes
// embed UB budget in bytes: [zero template | ping | pong] staging, dim elements
// each. Equals the in-file UB_SIZE (192KB) for the C220 arch embed_kernel
// targets; defined here as a shared constant for host op.cpp and the kernel.
#define EMBED_UB_BUDGET 196608
#define VECTOR_MAX_BYTESIZE 256  // The maximum byte size of one repeat in vector
#define UB_BUF_ALIGN_SIZE 32     // The align size of UB buffer address; i.e., BLOCK_SIZE
#if __NPU_ARCH__ != 3510
#define UB_SIZE 196608  // 192KB, total size of unified buffer (UB), = UB_BANK_SIZE * UB_BANK_NUM
#else
#define UB_SIZE 262144  // 256 KB
#endif
// performance tuning
#define GM_UB_BURST_SIZE 16384  // 16KB, recommended burst size for GM <--> UB copy bandwidth-wise
// gather_sparse_kv_cache tiling (top-k slots per chunk task, rotating buffer count)
#define GATHER_SPARSE_KV_CACHE_CNT_ONE_LOOP 32
#define GATHER_SPARSE_KV_CACHE_C8_CNT_ONE_LOOP 16
#define GATHER_SPARSE_KV_CACHE_BUF_NUM 2
// model initialization
#define INIT_MIN_SEQ_POS 102400ull  // for the identity table 0.1.2...maxSeqLen-1
// attention/mla/cxa/indexer
// Conservative BF16 Q RoPE budget for 192 KiB UB (178.5 KiB at ropeDim=64).
#define ROPE_COMPLEX_MAX_ELEMENTS_PER_TOKEN 4096
#define XLITE_MAX_M0 128
// mmad k/n granularity in elements for a 2-byte dtype (BLOCK_SIZE / 2); the lead-in
// RunAicQK/RunAicSV round windowStart down to is always less than this.
#define K_BLOCK_SIZE_2B 16
// compress-segment n/k tile (svck0) in RunAicQK/RunAicSV (see cxa_aic_helper.h); the
// compress segment of the scores workspace is padded to a multiple of 4*svck0 because
// RunAicSV reads 4*svck0 compress-score elements per call.
#define CXA_SVCK0 64

// indexer
// 24 is the measured optimum for the tile-flat tiers (decode-only and >4096-row
// prefill): interleaved A/B on the default work set showed tile 25 regressing
// 1.6-2.8% vs 24, so the macro keeps the legacy default. The 22/17 tiers for
// chunked prefill / mixed batches are host-side overrides, still <= this macro.
#define MAX_INDEXER_Q_TILE_LEN 24
#define MAX_INDEXER_KV_TILE_LEN 4096
#define INDEXER_C8_TILE_ROWS 32
#define QUANT_CACHE_C8_MAX_DIM 4096
#define QUANT_Q_C8_MAX_DIM 4096
#if MAX_INDEXER_KV_TILE_LEN > INIT_MIN_SEQ_POS
#error "MAX_INDEXER_KV_TILE_LEN must not exceed INIT_MIN_SEQ_POS"
#endif
#define MAX_TOPK_NUM 2048
// Runtime topk row width may exceed MAX_TOPK_NUM: the GLM5 kpool path packs the PKI
// causal tail into columns [topK, topK+kpool-1) of the topk buffer (e.g. 2048+3=2051).
// The aiv softmax mask walks 128-slot blocks, so widened rows are padded to
// TOPK_RUN_ALIGN slots with -1 (GLM5: 2051 -> 2176). MAX_TOPK_NUM stays the
// compile-time bound for legacy (DSA) shapes.
#define TOPK_RUN_ALIGN 128
#define MAX_TOPK_RUN_NUM ((MAX_TOPK_NUM + 8 + TOPK_RUN_ALIGN - 1) / TOPK_RUN_ALIGN * TOPK_RUN_ALIGN)
// RunAivSoftmaxPingPong row bound: the softmax row len = MAX(outN, topK +
// swaSegWidth) must stay within this for the worst-case feature mix
// (saveMaxSum + topK + attnSink all on) to fit UB. The binding scenario is
// SWA + topK (the row-sized buffers plus an inTopK holding topK +
// MAX_SWA_SEG_WIDTH elements): at MAX_TOPK_NUM that layout is 195072B <=
// UB_SIZE(196608B), 1536B slack (static_asserts in softmax_attn_aiv.h); the
// no-SWA worst case is 186880B. Row-contract enforcement: flash_cxa checks
// swaSegWidth + tileSizeOfCachedKV at the host, cxa's row is checked by the
// Python layer (tests/kernels/cxa.py max_row), flash_mla_v2 / mla_v2 /
// flash_attention check the tile / maxSeqLen at the host.
#define MAX_SOFTMAX_PINGPONG_LEN 10496
// SWA segment width bound: swaSegWidth = windowSize + XLITE_MAX_M0 +
// K_BLOCK_SIZE_2B (cxa.h / flash_cxa.h). Host-rejects wider windows
// (XliteOpCXA / XliteOpFlashCXA): the topK path's inTopK buffer holds
// topK + swaSegWidth elements and must stay within the budget above.
#define MAX_SWA_SEG_WIDTH 4096
#define SOFTMAX_MASK_CHUNK 2048  // topkMask vand-slot granularity
#define MSD_MERGE_DEQUANT_MAX_N 8064

// kpool tail
#define KPOOL_TAIL_MAX_HEAD_DIM 256  // kpool_tail input caps
#define KPOOL_TAIL_MAX_KPOOL 8

// muls
#define MAX_MULS_CALC_NUM 16320

// bench micro-benchmark op selector. Each merged bench_<ct> kernel (bench_aic
// / bench_aiv / bench_mix) takes a uint32_t op and dispatches on it; only the
// ops valid for that core type appear in the kernel's switch. Host (op.cpp) and
// kernel (microbench.h) share this enum so the dispatch stays in sync.
enum BenchOp : uint32_t {
    // ---- AIC / AIV / MIX ----
    BENCH_OP_EMPTY = 0,  // empty launch-floor probe (all 3 core types)
    // ---- AIC / AIV (no MIX): scalar dependent chains, default build ----
    BENCH_OP_SCALAR_ADD = 1,
    BENCH_OP_SCALAR_XOR = 2,
    BENCH_OP_SCALAR_MUL = 3,
    BENCH_OP_SCALAR_DIV = 4,
    BENCH_OP_SCALAR_FADD = 5,  // float variant
    BENCH_OP_SCALAR_ADD_UNROLL4 = 6,
    // ---- Timed twins (MICROBENCH_TIMED only) ----
    BENCH_OP_SCALAR_ADD_TIMED = 7,
    BENCH_OP_SCALAR_XOR_TIMED = 8,
    BENCH_OP_SCALAR_MUL_TIMED = 9,
    BENCH_OP_SCALAR_DIV_TIMED = 10,
    BENCH_OP_SCALAR_FADD_TIMED = 11,
    BENCH_OP_SCALAR_ADD_UNROLL4_TIMED = 12,
    // ---- AIC only (MICROBENCH_TIMED) ----
    BENCH_OP_CUBE_MMAD = 13,
    // ---- AIV only (MICROBENCH_TIMED) ----
    BENCH_OP_MTE_READ = 14,
    BENCH_OP_MTE_WRITE = 15,
    BENCH_OP_MTE_COPY = 16,
    BENCH_OP_MTE_CHASE = 17,
    BENCH_OP_VEC_ADD_F32 = 18,
    BENCH_OP_VEC_MUL_F32 = 19,
    BENCH_OP_VEC_ADD_I32 = 20,
    BENCH_OP_VEC_ADD_F16 = 21,
    BENCH_OP_VEC_SQRT_F32 = 22,
    BENCH_OP_VEC_EXP_F32 = 23,
};
#endif
