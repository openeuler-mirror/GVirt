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

// harware
#define UB_BLOCK_SIZE 32  // unified buffer block size in bytes
// performance tuning
#define GM_UB_BURST_SIZE 16384  // 16KB, recommended burst size for GM <--> UB copy bandwidth-wise
// gather_sparse_kv_cache tiling (top-k slots per chunk task, rotating buffer count)
#define GATHER_SPARSE_KV_CACHE_CNT_ONE_LOOP 32
#define GATHER_SPARSE_KV_CACHE_BUF_NUM 4
// model initialization
#define INIT_MIN_SEQ_POS 102400ull  // for the identity table 0.1.2...maxSeqLen-1
// attention/mla/cxa/indexer
#define XLITE_MAX_M0 128
// mmad k/n granularity in elements for a 2-byte dtype (BLOCK_SIZE / 2); the lead-in
// RunAicQK/RunAicSV round windowStart down to is always less than this.
#define K_BLOCK_SIZE_2B 16
// compress-segment n/k tile (svck0) in RunAicQK/RunAicSV (see cxa_aic_helper.h); the
// compress segment of the scores workspace is padded to a multiple of 4*svck0 because
// RunAicSV reads 4*svck0 compress-score elements per call.
#define CXA_SVCK0 64
#define MAX_INDEXER_KV_TILE_LEN 4096
#define INDEXER_C8_TILE_ROWS 32
#if MAX_INDEXER_KV_TILE_LEN > INIT_MIN_SEQ_POS
#error "MAX_INDEXER_KV_TILE_LEN must not exceed INIT_MIN_SEQ_POS"
#endif
#define MAX_TOPK_NUM 2048
#define MAX_SOFTMAX_PINGPONG_LEN 11776

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
