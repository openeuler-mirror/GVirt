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
#define MAX_TOPK_NUM 2048
#define MAX_SOFTMAX_PINGPONG_LEN 11776

// muls
#define MAX_MULS_CALC_NUM 16320
#endif
