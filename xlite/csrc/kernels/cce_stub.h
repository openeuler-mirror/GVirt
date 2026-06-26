/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef STUB_FUN_INTELLISENSE_H
#define STUB_FUN_INTELLISENSE_H
#include <cstdint>
// IntelliSense-compatible type stubs
typedef struct {
    uint16_t val;
} half;
typedef struct {
    uint16_t val;
} bfloat16_t;

typedef enum {
    CRFMODE_NONE = 0,
    CRFMODE_DEQSCALE_VDEQ8 = 8,
    CRFMODE_DEQSCALE_DEQ8 = 9,
    CRFMODE_DEQSCALE_VDEQ16 = 10,
    CRFMODE_DEQSCALE_DEQ16 = 11,
    CRFMODE_DEQSCALE_VDEQS16 = 12,
    CRFMODE_DEQSCALE_DEQS16 = 13,
    CRFMODE_DEQSCALE_VDEQ2S16 = 14,
    CRFMODE_DEQSCALE_DEQ2S16 = 15,
} ConvReluFix_t;
typedef enum {
    CRMODE_NONE = 0,
    CRMODE_F32toF16_NONE = 1,
    CRMODE_F32toF16_RELU = 2,
    CRMODE_S32toF16_NONE = 3,
    CRMODE_F16toF32_NONE = 4,
    CRMODE_NONE_RELU = 5,
    CRMODE_F16_MUL = 6,
    CRMODE_S32toF16_DEQSCALE_SPR = 7,
    CRMODE_DEQSCALE_VDEQ8 = 8,
    CRMODE_DEQSCALE_DEQ8 = 9,
    CRMODE_DEQSCALE_VDEQ16 = 10,
    CRMODE_DEQSCALE_DEQ16 = 11,
    CRMODE_DEQSCALE_VDEQS16 = 12,
    CRMODE_DEQSCALE_DEQS16 = 13,
} ConvRelu_t;
typedef enum {
    NoConversion = 0,
    CvtMode1 = 1,
    CvtMode2 = 2,
} CvtMode_t;
typedef enum {
    DUAL_MODE0 = 0,
    DUAL_MODE1 = 1,
    DUAL_MODE2 = 2,
    DUAL_MODE3 = 3,
    DUAL_MODE4 = 4,
} DualMode_t;
typedef enum {
    VALUE_INDEX = 0,
    INDEX_VALUE = 1,
    ONLY_VALUE = 2,
    ONLY_INDEX = 3,
} Order_t;
typedef enum {
    NoPooling = 0,
    AVGPooling = 1,
    MAXPooling = 2,
    GAVGPooling = 3,
} Pool_t;
typedef enum {
    NoConv = 0,
    VQS162B8_POST = 1,
    QS162B8_POST = 2,
    VQF162B8_POST = 3,
    QF162B8_POST = 4,
    VQS162S4_POST = 5,
    QS162S4_POST = 6,
    VQF162S4_POST = 7,
    QF162S4_POST = 8,
    VQS162S16_POST = 9,
    QS162S16_POST = 10,
    VQF162S16_POST = 11,
    QF162S16_POST = 12,
} QuantMode_post;
typedef enum {
    Bin_N0 = 0,
    Bin_N1 = 1,
    Bin_N2 = 2,
    Bin_N3 = 3,
} Bin;
typedef enum {
    NoQuant = 0,
    F322F16 = 1,
    VQF322HIF8_PRE = 2,
    QF322HIF8_PRE = 3,
    VQF322HIF8_PRE_HYBRID = 4,
    QF322HIF8_PRE_HYBRID = 5,
    AttachF16Mul = 6,
    VREQ8 = 8,
    REQ8 = 9,
    VDEQF16 = 10,
    DEQF16 = 11,
    VSHIFTS322S16 = 12,
    SHIFTS322S16 = 13,
    VQF322FP8_PRE = 12,
    QF322FP8_PRE = 13,
    VQF322F32_PRE = 14,
    QF322F32_PRE = 15,
    F322BF16 = 16,
    VQF162B8_PRE = 17,
    QF162B8_PRE = 18,
    VQF162S4_PRE = 19,
    QF162S4_PRE = 20,
    VREQ4 = 21,
    REQ4 = 22,
    VQF322B8_PRE = 23,
    QF322B8_PRE = 24,
    VQF322S4_PRE = 25,
    QF322S4_PRE = 26,
    VDEQS16 = 27,
    DEQS16 = 28,
    VQF162S16_PRE = 29,
    QF162S16_PRE = 30,
    VQF322F16_PRE = 31,
    QF322F16_PRE = 32,
    VQF322BF16_PRE = 33,
    QF322BF16_PRE = 34,
    VQS322BF16_PRE = 35,
    QS322BF16_PRE = 36,
} QuantMode_t;
typedef enum {
    NoRELU = 0,
    NormalRELU = 1,
    LeakyRELU = 2,
    PRELU = 3,
} Relu_t;
typedef enum {
    NoREQ = 0,
    REQ = 1,
    VREQ = 2,
} Req_t;
typedef enum {
    DATA_EXP_0 = 48,
    DATA_EXP_1 = 49,
    DATA_EXP_2 = 50,
    DATA_EXP_3 = 51,
} VSPR_t;
typedef enum {
    inc = 0,
    dec = 1,
} addr_cal_mode_t;
typedef enum {
    ATOMIC_SUM = 0,
} atomic_op_t;
typedef enum {
    UNKNOWN_VALUE = 0,
    MERGING_VALUE = 1,
    ZEROING_VALUE = 2,
} CpuMode;
typedef enum {
    P0 = 0,
    P1 = 1,
    P2 = 2,
    P3 = 3,
} Part_T;
typedef enum {
    EVEN = 0,
    ODD = 1,
} Part;
typedef enum {
    CAST_RINT = 0,
    CAST_ROUND = 1,
    CAST_FLOOR = 2,
    CAST_CEIL = 3,
    CAST_TRUNC = 4,
    CAST_ODD = 5,
    CAST_HYBRID = 6,
} ROUND;
typedef enum {
    RS_DISABLE_VALUE = 0,
    RS_ENABLE_VALUE = 1,
} RoundingSaturation;
typedef enum {
    LOWEST = 0,
    HIGHEST = 1,
} Pos;
typedef enum {
    Lower = 0,
    Higher = 1,
} HiloPart;
typedef enum {
    NOSTORED = 0,
    STORED = 1,
} StoreMode;
typedef enum {
    NO_POST_UPDATE_VALUE = 0,
    POST_UPDATE_VALUE = 1,
} Post;
typedef enum {
    SPR_AR_VALUE = 74,
} Spr;
typedef enum {
    NO_RND_VALUE = 0,
    RND_VALUE = 1,
} Rnd;
typedef enum {
    ALL = 0,
    VL1 = 1,
    VL2 = 2,
    VL3 = 3,
    VL4 = 4,
    VL8 = 5,
    VL16 = 6,
    VL32 = 7,
    VL64 = 8,
    VL128 = 9,
    M3 = 10,
    M4 = 11,
    H = 12,
    Q = 13,
    ALLF = 15,
} Pat;
typedef enum {
    DIST_NORM_B8 = 0,
    DIST_NORM_B16 = 1,
    DIST_NORM_B32 = 2,
    DIST_ONEPT_B8 = 3,
    DIST_ONEPT_B16 = 4,
    DIST_ONEPT_B32 = 5,
    DIST_PK_B16 = 6,
    DIST_PK_B32 = 7,
    DIST_INTLV_B8 = 8,
    DIST_INTLV_B16 = 9,
    DIST_PK_B64 = 10,
    DIST_INTLV_B32 = 11,
    DIST_PK4_B32 = 12,
    DIST_MRG4CHN_B8 = 13,
    DIST_MRG2CHN_B8 = 14,
    DIST_MRG2CHN_B16 = 15,
} DistVST;
typedef enum {
    INC_ORDER_VALUE = 0,
    DEC_ORDER_VALUE_VALUE = 1,
} Order;
typedef enum {
    DIST_NORM = 0,
    DIST_BRC_B8 = 1,
    DIST_BRC_B16 = 2,
    DIST_BRC_B32 = 3,
    DIST_US_B8 = 6,
    DIST_US_B16 = 7,
    DIST_DS_B8 = 8,
    DIST_DS_B16 = 9,
    DIST_BDINTLV = 10,
    DIST_DINTLV_B8 = 11,
    DIST_DINTLV_B16 = 12,
    DIST_UNPK_B8 = 13,
    DIST_UNPK_B16 = 14,
    DIST_BLK = 15,
    DIST_E2B_B16 = 16,
    DIST_E2B_B32 = 17,
    DIST_UNPK_B32 = 18,
    DIST_DINTLV_B32 = 19,
    DIST_UNPK4_B8 = 20,
    DIST_SPLT4CHN_B8 = 21,
    DIST_SPLT2CHN_B8 = 22,
    DIST_SPLT2CHN_B16 = 23,
    DIST_US = 1,
    DIST_DS = 2,
    DIST_PK = 1,
} Dist;
typedef enum {
    MODE_ZEROING = 0,
    POST_UPDATE = 1,
    NORM = 2,
    NORM_B8 = 3,
    NORM_B16 = 4,
    NORM_B32 = 5,
    UNPK_B8 = 6,
    UNPK_B16 = 7,
    UNPK_B32 = 8,
    PK = 9,
    PK_B16 = 10,
    PK_B32 = 11,
    PK_B64 = 12,
    VLD_VST = 13,
    US = 14,
    DS = 15,
    LOWER = 16,
    HIGHER = 17,
    PART_EVEN = 18,
    PART_ODD = 19,
    ROUND_R = 20,
    ROUND_A = 21,
    ROUND_F = 22,
    ROUND_C = 23,
    ROUND_Z = 24,
    ROUND_O = 25,
    RS_ENABLE = 26,
    RS_DISABLE = 27,
    SPR_AR = 28,
    MODE_STORED = 29,
    PAT_VL8 = 30,
    PAT_VL32 = 31,
    PAT_ALL = 32,
    PAT_ALLF = 33,
    PAT_H = 34,
    VST_VST = 35,
    VST_VLD = 36,
    PAT_VL16 = 37,
    BRC_B16 = 38,
    BRC_B32 = 39,
    INC_ORDER = 40,
    DEC_ORDER = 41,
    PAT_VL64 = 42,
    ONEPT_B8 = 43,
    ONEPT_B16 = 44,
    ONEPT_B32 = 45,
    PART_P0 = 46,
    PART_P1 = 47,
    PART_P2 = 48,
    PART_P3 = 49,
    UNPK4_B8 = 50,
    PK4_B32 = 51,
    ROUND_H = 52,
    VST_LD = 53,
    VST_ST = 54,
    VLD_ST = 55,
    ST_VLD = 56,
    ST_VST = 57,
    LD_VST = 58,
    VV_ALL = 59,
    VS_ALL = 60,
    SV_ALL = 61,
    MODE_NO_STORED = 62,
    MODE_MERGING = 63,
    BRC_B8 = 64,
} Literal;
typedef enum {
    ATOMIC_NONE = 0,
    ATOMIC_F32 = 1,
    ATOMIC_F16 = 2,
    ATOMIC_S16 = 3,
    ATOMIC_S32 = 4,
    ATOMIC_S8 = 5,
    ATOMIC_BF16 = 6,
} atomic_type_t;
typedef enum {
    BM_DISABLE = 0,
    BM_ENABLE = 1,
} bm_t;
typedef enum {
    SINGLE_CACHE_LINE = 0,
    ENTIRE_DATA_CACHE,
} cache_line_t;
typedef enum {
    CSIZE0 = 0,
    CSIZE1 = 1,
} csize_t;
typedef enum {
    CACHELINE_ALL = 0,
    CACHELINE_UB,
    CACHELINE_OUT,
    CACHELINE_ATOMIC,
} dcci_dst_t;
typedef enum {
    e0 = 0,
    e2 = 2,
    e4 = 4,
    e6 = 6,
} even0_7_t;
typedef enum {
    EVENT_ID0 = 0,
    EVENT_ID1,
    EVENT_ID2,
    EVENT_ID3,
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201) || \
                              (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102))
    EVENT_ID4,
    EVENT_ID5,
    EVENT_ID6,
    EVENT_ID7,
#endif
} event_t;
typedef enum {
    DSB_ALL = 0,
    DSB_DDR,
    DSB_UB,
    DSB_SEQ,
} mem_dsb_t;
typedef enum {
    L1 = 0,
    L0A,
    L0B,
    L0C,
    UB,
    BT,
} mem_t;
typedef enum {
    PAD_NONE = 0,
    PAD_MODE1 = 1,
    PAD_MODE2 = 2,
    PAD_MODE3 = 3,
    PAD_MODE4 = 4,
    PAD_MODE5 = 5,
    PAD_MODE6 = 6,
    PAD_MODE7 = 7,
    PAD_MODE8 = 8,
} pad_t;
typedef enum {
    PIPE_S = 0,
    PIPE_V,
    PIPE_M,
    PIPE_MTE1,
    PIPE_MTE2,
    PIPE_MTE3,
    PIPE_ALL,
    PIPE_MTE4 = 7,
    PIPE_MTE5 = 8,
    PIPE_V2 = 9,
    PIPE_FIX = 10,
} pipe_t;
typedef enum {
    VA0 = 0,
    VA1,
    VA2,
    VA3,
    VA4,
    VA5,
    VA6,
    VA7,
} ub_addr8_t;
typedef enum {
    UFMode0 = 0,
    Reserved,
    UFMode2,
    UFMode3,
} unit_flag_t;
typedef enum {
    L128 = 0,
    H128,
} vpart_t;
typedef enum {
    b8 = 0,
    b16 = 1,
    b32 = 2,
    s8 = 3,
    s32 = 4,
    f16 = 5,
    fmix = 6,
    f32 = 7,
} vtype_t;
typedef enum {
    W_3 = 0,
    W_5,
} w_size_t;
typedef enum {
    No_LUT = 0,
    QMode_2to8 = 1,
    QMode_3to8 = 2,
    QMode_4to8 = 3,
    ANTIQ_S82F16 = 8,
    VANTIQ_S82F16 = 9,
    ANTIQ_S82BF16 = 10,
    VANTIQ_S82BF16 = 11,
    ANTIQ_FP82F16 = 12,
    VANTIQ_FP82F16 = 13,
    ANTIQ_HiF82F16 = 14,
    VANTIQ_HiF82F16 = 15,
} QMode_t;
int64_t bcnt0(uint64_t in);
int64_t bcnt1(uint64_t in);
void broadcast_ub_to_cc(__cc__ half *dst, __ubuf__ half *src, uint64_t config);
void broadcast_ub_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t nBurst, uint8_t lenBurst,
                        uint8_t srcGap, uint8_t dstGap);
void broadcast_ub_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t nBurst, uint8_t lenBurst,
                        uint8_t srcGap, uint8_t dstGap, bool repeat);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ float *src, uint64_t config);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t nBurst, uint8_t lenBurst,
                        uint8_t srcGap, uint8_t dstGap);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t nBurst, uint8_t lenBurst,
                        uint8_t srcGap, uint8_t dstGap, bool repeat);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ half *src, uint64_t config);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t nBurst, uint8_t lenBurst,
                        uint8_t srcGap, uint8_t dstGap);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t nBurst, uint8_t lenBurst,
                        uint8_t srcGap, uint8_t dstGap, bool repeat);
void broadcast_ub_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint64_t config);
void broadcast_ub_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t nBurst,
                        uint8_t lenBurst, uint8_t srcGap, uint8_t dstGap);
void broadcast_ub_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t nBurst,
                        uint8_t lenBurst, uint8_t srcGap, uint8_t dstGap, bool repeat);
int64_t clz(uint64_t in);
void col2img(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config0, uint64_t config1);
void col2img(__ubuf__ half *dst, __ubuf__ half *src, int16_t firstHi, uint8_t posWk, uint8_t posHk,
             int16_t firstWi, uint8_t nRepeat, uint8_t strideW, uint8_t strideH, uint8_t Wk,
             uint8_t Hk, uint8_t dilationW, uint8_t dilationH, bool repeat);
void col2img(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config0, uint64_t config1);
void col2img(__ubuf__ float *dst, __ubuf__ float *src, int16_t firstHi, uint8_t posWk,
             uint8_t posHk, int16_t firstWi, uint8_t nRepeat, uint8_t strideW, uint8_t strideH,
             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH, bool repeat);
void compress_ub_to_gm(__gm__ uint8_t *dst, __gm__ uint8_t *src0, __ubuf__ uint8_t *src1,
                       int64_t config);
void compress_ub_to_gm(__gm__ uint8_t *dst, __gm__ uint8_t *src0, __ubuf__ uint8_t *src1,
                       uint16_t nBlock, bool sBlock, uint8_t sid, uint8_t zero_value);
void compress_ub_to_gm(__gm__ half *dst, __gm__ half *src0, __ubuf__ half *src1, int64_t config);
void compress_ub_to_gm(__gm__ half *dst, __gm__ half *src0, __ubuf__ half *src1, uint16_t nBlock,
                       bool sBlock, uint8_t sid, uint8_t zero_value);
half conv_f322f16o(float in);
int64_t conv_f322s32a(float in);
int64_t conv_f322s32c(float in);
int64_t conv_f322s32f(float in);
int64_t conv_f322s32r(float in);
void dhistv2(vector_u16 &dst, vector_u8 src, vector_bool mask, uint32_t bin);
void chistv2(vector_u16 &dst, vector_u8 src, vector_bool mask, uint32_t bin);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ void *src, uint64_t config);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ void *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ float *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ float *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap, uint8_t fixVal);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ half *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ half *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap, uint8_t fixVal);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ int32_t *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ int32_t *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap, uint8_t fixVal);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ bfloat16_t *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap);
void copy_cbuf_to_bt(uint64_t dst, __cbuf__ bfloat16_t *src, uint16_t convControl, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t sourceGap, uint16_t dstGap, uint8_t fixVal);
void copy_cbuf_to_fbuf(__fbuf__ void *dst, __cbuf__ void *src, uint64_t config);
void copy_cbuf_to_fbuf(__fbuf__ void *dst, __cbuf__ void *src, uint16_t burstNum, uint16_t burstLen,
                       uint16_t srcGapSize, uint16_t dstGapSize);
void copy_cbuf_to_gm(__gm__ void *dst, __cbuf__ void *src, uint64_t config);
void copy_cbuf_to_gm(__gm__ void *dst, __cbuf__ void *src, uint8_t sid, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_cbuf_to_gm_align(__gm__ half *dst, __cbuf__ half *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_gm_align(__gm__ float *dst, __cbuf__ float *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_gm_align(__gm__ int16_t *dst, __cbuf__ int16_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_gm_align(__gm__ int32_t *dst, __cbuf__ int32_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_gm_align(__gm__ int8_t *dst, __cbuf__ int8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_gm_align(__gm__ uint16_t *dst, __cbuf__ uint16_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_gm_align(__gm__ uint32_t *dst, __cbuf__ uint32_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_gm_align(__gm__ uint8_t *dst, __cbuf__ uint8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_cbuf_to_ubuf(__ubuf__ void *dst, __cbuf__ void *src, uint64_t config);
void copy_cbuf_to_ubuf(__ubuf__ void *dst, __cbuf__ void *src, uint8_t sid, uint16_t nBurst,
                       uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_depthwise_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint64_t config,
                               ConvRelu_t crMode);
void copy_depthwise_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t nBurst,
                               uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                               ConvRelu_t crMode);
void copy_depthwise_cc_to_ubuf(__ubuf__ float *dst, __cc__ half *src, uint64_t config,
                               ConvRelu_t crMode);
void copy_depthwise_cc_to_ubuf(__ubuf__ float *dst, __cc__ half *src, uint8_t sid, uint16_t nBurst,
                               uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                               ConvRelu_t crMode);
void copy_depthwise_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint64_t config,
                               ConvRelu_t crMode);
void copy_depthwise_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t nBurst,
                               uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                               ConvRelu_t crMode);
void copy_depthwise_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint64_t config,
                               ConvRelu_t crMode);
void copy_depthwise_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t nBurst,
                               uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                               ConvRelu_t crMode);
void copy_gm_to_cbuf(__cbuf__ void *dst, __gm__ void *src, uint64_t config, pad_t padMode);
void copy_gm_to_cbuf(__cbuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride, pad_t padMode);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ half *dst, __gm__ half *src, uint64_t xm,
                                     uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ half *dst, __gm__ half *src, uint8_t sid,
                                     uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                     uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                     uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                     uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t xm,
                                     uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint8_t sid,
                                     uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                     uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                     uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                     uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint64_t xm,
                                     uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint8_t sid,
                                     uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                     uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                     uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                     uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ uint16_t *dst, __gm__ uint16_t *src, uint64_t xm,
                                     uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b16(__cbuf__ uint16_t *dst, __gm__ uint16_t *src, uint8_t sid,
                                     uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                     uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                     uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                     uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b32s(__cbuf__ float *dst, __gm__ float *src, uint64_t xm,
                                      uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b32s(__cbuf__ float *dst, __gm__ float *src, uint8_t sid,
                                      uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                      uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                      uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                      uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b32s(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint64_t xm,
                                      uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b32s(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint8_t sid,
                                      uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                      uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                      uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                      uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b32s(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint64_t xm,
                                      uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b32s(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint8_t sid,
                                      uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                      uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                      uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                      uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b8(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint64_t xm,
                                    uint64_t xt);
void copy_gm_to_cbuf_multi_nd2nz_b8(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint8_t sid,
                                    uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                    uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                    uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                    uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz_b8(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint64_t xm,
                                    uint64_t xt);
void pstu(vector_align &alignData, vector_bool src, __ubuf__ uint32_t *&base);
void pstu(vector_align &alignData, vector_bool src, __ubuf__ uint16_t *&base);
void movvp(vector_bool &dst, vector_u32 src, int16_t part);
void movvp(vector_bool &dst, vector_u16 src, int16_t part);
void vstai(vector_align data, __ubuf__ uint8_t *base, int32_t offset);
void vstai(vector_align data, __ubuf__ int8_t *base, int32_t offset);
void copy_gm_to_cbuf_multi_nd2nz_b8(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint8_t sid,
                                    uint16_t ndNum, uint16_t nValue, uint16_t dValue,
                                    uint16_t srcNdMatrixStride, uint16_t srcDValue,
                                    uint16_t dstNzC0Stride, uint16_t dstNzNStride,
                                    uint16_t dstNzMatrixStride);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ half *dst, __gm__ half *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ half *dst_addr, __gm__ half *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ bfloat16_t *dst_addr, __gm__ bfloat16_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ float *dst, __gm__ float *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ float *dst_addr, __gm__ float *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ int16_t *dst_addr, __gm__ int16_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ int8_t *dst_addr, __gm__ int8_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ int32_t *dst_addr, __gm__ int32_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ uint16_t *dst, __gm__ uint16_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ uint16_t *dst_addr, __gm__ uint16_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ uint8_t *dst_addr, __gm__ uint8_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ uint32_t *dst_addr, __gm__ uint32_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ fp8_e5m2_t *dst, __gm__ fp8_e5m2_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ fp8_e5m2_t *dst_addr, __gm__ fp8_e5m2_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ fp8_e4m3fn_t *dst, __gm__ fp8_e4m3fn_t *src,
                                 uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ fp8_e4m3fn_t *dst_addr, __gm__ fp8_e4m3fn_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ hifloat8_t *dst, __gm__ hifloat8_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_nd2nz(__cbuf__ hifloat8_t *dst_addr, __gm__ hifloat8_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ bfloat16_t *dst_addr, __gm__ bfloat16_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ half *dst, __gm__ half *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ half *dst_addr, __gm__ half *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ float *dst, __gm__ float *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ float *dst_addr, __gm__ float *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ int8_t *dst_addr, __gm__ int8_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ uint8_t *dst_addr, __gm__ uint8_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ fp8_e4m3fn_t *dst, __gm__ fp8_e4m3fn_t *src,
                                 uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ fp8_e4m3fn_t *dst_addr, __gm__ fp8_e4m3fn_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ fp8_e5m2_t *dst, __gm__ fp8_e5m2_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ fp8_e5m2_t *dst_addr, __gm__ fp8_e5m2_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ hifloat8_t *dst, __gm__ hifloat8_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ hifloat8_t *dst_addr, __gm__ hifloat8_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ int16_t *dst_addr, __gm__ int16_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ uint16_t *dst, __gm__ uint16_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ uint16_t *dst_addr, __gm__ uint16_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ int32_t *dst_addr, __gm__ int32_t *src_addr, uint8_t sid,
                                 uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,
                                 uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint64_t config0,
                                 uint64_t config1);
void copy_gm_to_cbuf_multi_dn2nz(__cbuf__ uint32_t *dst_addr, __gm__ uint32_t *src_addr,
                                 uint8_t sid, uint64_t loop1_src_stride, uint8_t l2_cache_ctl,
                                 uint16_t n_value, uint32_t d_value, uint64_t loop4_src_stride,
                                 bool smallc0_en);
void copy_gm_to_ubuf(__ubuf__ void *dst, __gm__ void *src, uint64_t config);
void copy_gm_to_ubuf(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_gm_to_ubuf_align_b16(__ubuf__ void *dst, __gm__ void *src, uint64_t config,
                               uint64_t gapConfig);
void copy_gm_to_ubuf_align_b16(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                               uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                               uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align_b32(__ubuf__ void *dst, __gm__ void *src, uint64_t config,
                               uint64_t gapConfig);
void copy_gm_to_ubuf_align_b32(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                               uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                               uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align_b8(__ubuf__ void *dst, __gm__ void *src, uint64_t config,
                              uint64_t gapConfig);
void copy_gm_to_ubuf_align_b8(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                              uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                              uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ int8_t *dst, __gm__ int8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ uint8_t *dst, __gm__ uint8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ int16_t *dst, __gm__ int16_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ uint16_t *dst, __gm__ uint16_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint8_t leftPaddingNum,
                           uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ half *dst, __gm__ half *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ int32_t *dst, __gm__ int32_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ uint32_t *dst, __gm__ uint32_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint8_t leftPaddingNum,
                           uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_align(__ubuf__ float *dst, __gm__ float *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ uint16_t *dst, __gm__ uint16_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint8_t leftPaddingNum,
                           uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ half *dst, __gm__ half *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint8_t leftPaddingNum,
                           uint8_t rightPaddingNum, uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_cbuf_align(__cbuf__ float *dst, __gm__ float *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                           uint32_t srcGap, uint32_t dstGap);
void copy_gm_to_ubuf_pad_b16(__ubuf__ void *dst, __gm__ void *src, uint64_t config,
                             uint64_t paddingConfig);
void copy_gm_to_ubuf_pad_b16(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                             uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                             uint8_t leftPaddingNum, uint8_t rightPaddingNum);
void copy_gm_to_ubuf_pad_b32(__ubuf__ void *dst, __gm__ void *src, uint64_t config,
                             uint64_t paddingConfig);
void copy_gm_to_ubuf_pad_b32(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                             uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                             uint8_t leftPaddingNum, uint8_t rightPaddingNum);
void copy_gm_to_ubuf_pad_b8(__ubuf__ void *dst, __gm__ void *src, uint64_t config,
                            uint64_t paddingConfig);
void copy_gm_to_ubuf_pad_b8(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            uint8_t leftPaddingNum, uint8_t rightPaddingNum);
void copy_matrix_cbuf_to_cc(__cc__ half *dst, __cbuf__ half *src, uint64_t config);
void copy_matrix_cbuf_to_cc(__cc__ half *dst, __cbuf__ half *src, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_matrix_cbuf_to_cc(__cc__ half *dst, __cbuf__ float *src, uint64_t config);
void copy_matrix_cbuf_to_cc(__cc__ half *dst, __cbuf__ float *src, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_matrix_cbuf_to_cc(__cc__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config);
void copy_matrix_cbuf_to_cc(__cc__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_matrix_cbuf_to_cc(__cc__ bfloat16_t *dst, __cbuf__ float *src, uint64_t config);
void copy_matrix_cbuf_to_cc(__cc__ bfloat16_t *dst, __cbuf__ float *src, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_matrix_cbuf_to_cc(__cc__ float *dst, __cbuf__ float *src, uint64_t config);
void copy_matrix_cbuf_to_cc(__cc__ float *dst, __cbuf__ float *src, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_matrix_cbuf_to_cc(__cc__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config);
void copy_matrix_cbuf_to_cc(__cc__ int32_t *dst, __cbuf__ int32_t *src, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_matrix_cbuf_to_cc(__cc__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config);
void copy_matrix_cbuf_to_cc(__cc__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                            bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ bfloat16_t *dst, __cc__ float *src, uint8_t sid,
                            uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
                            uint16_t srcStride, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ bfloat16_t *dst, __cc__ float *src, uint8_t sid,
                            uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
                            uint16_t srcStride, uint8_t clip_relu_pre, uint8_t UnitFlagMode,
                            QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ float *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                            bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int32_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                            bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                            bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                            bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                            bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf(__cbuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                            uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                            uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                            uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                            QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                            uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf_b4(__cbuf__ void *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf_b4(__cbuf__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                               bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf_b4(__cbuf__ void *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf_b4(__cbuf__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                               bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf_bf16(__cbuf__ uint16_t *dst, __cc__ float *src, uint64_t xm,
                                 uint64_t xt);
void copy_matrix_cc_to_cbuf_bf16(__cbuf__ uint16_t *dst, __cc__ float *src, uint8_t sid,
                                 uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
                                 uint16_t srcStride, uint8_t clip_relu_pre, uint8_t UnitFlagMode,
                                 QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit,
                                 bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf_bf16(__cbuf__ uint16_t *dst, __cc__ float *src, uint8_t sid,
                                 uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
                                 uint16_t srcStride, uint8_t clip_relu_pre, uint8_t UnitFlagMode,
                                 QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit,
                                 bool NZ2ND_EN, QuantMode_post quant_post, uint8_t relu_post,
                                 bool clip_relu_post, uint8_t eltwiseOP, uint8_t eltwise_antq_cfg,
                                 bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf_bf16(__cbuf__ uint16_t *dst, __cc__ float *src, uint8_t sid,
                                 uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
                                 uint16_t srcStride, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                                 uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_cbuf_s4(__cbuf__ void *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf_s4(__cbuf__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                               uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf_s4(__cbuf__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                               uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                               QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                               uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf_s4(__cbuf__ void *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_cbuf_s4(__cbuf__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                               uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_cbuf_s4(__cbuf__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                               uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                               QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                               uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ float *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ float *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ float *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ float *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ int32_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ int16_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ bfloat16_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                          bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm(__gm__ bfloat16_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ bfloat16_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ int16_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm(__gm__ bfloat16_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ half *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ float *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ half *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ int8_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ uint8_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ int32_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ bfloat16_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ fp8_e4m3fn_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm(__gm__ hifloat8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                          uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                          bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                          bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                          bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                          bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ bfloat16_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst_addr, __cc__ float *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ float *dst_addr, __cc__ float *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ half *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ int8_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ int32_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ bfloat16_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ fp8_e4m3fn_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_cbuf(__cbuf__ hifloat8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                            uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                            uint16_t loop_src_stride, uint8_t l2_cache_ctl, uint8_t clip_relu_pre,
                            uint8_t unit_flag_ctl, uint64_t quant_pre, uint8_t relu_pre,
                            bool split_en, bool NZ2ND_en, uint64_t quant_post, uint8_t relu_post,
                            bool clip_relu_post, bool loop_enhance_en, uint8_t eltwise_op,
                            bool eltwise_antq_cfg, bool loop_enhance_merge_en, bool C0_pad_en,
                            bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ bfloat16_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ half *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ float *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ half *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ int32_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ bfloat16_t *dst_addr, __cc__ int32_t *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ fp8_e4m3fn_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_ub(__ubuf__ hifloat8_t *dst_addr, __cc__ float *src_addr, uint8_t sid,
                          uint16_t n_size, uint16_t m_size, uint32_t loop_dst_stride,
                          uint16_t loop_src_stride, uint8_t dual_dst_ctl, bool sub_blockid,
                          uint8_t clip_relu_pre, uint8_t unit_flag_ctl, uint64_t quant_pre,
                          uint8_t relu_pre, bool split_en, bool NZ2ND_en, uint64_t quant_post,
                          uint8_t relu_post, bool clip_relu_post, bool loop_enhance_en,
                          uint8_t eltwise_op, bool eltwise_antq_cfg, bool loop_enhance_merge_en,
                          bool C0_pad_en, bool wino_post_en, bool broadcast_en, bool NZ2DN_en);
void copy_matrix_cc_to_gm_b4(__gm__ void *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm_b4(__gm__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                             bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm_b4(__gm__ void *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm_b4(__gm__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                             bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm_bf16(__gm__ uint16_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm_bf16(__gm__ uint16_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                               uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm_bf16(__gm__ uint16_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                               uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                               QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                               uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm_bf16(__gm__ uint16_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                               uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                               uint8_t UnitFlagMode, QuantMode_t QuantPRE, uint8_t ReLUPRE,
                               bool channelSplit, bool NZ2ND_EN);
void copy_matrix_cc_to_gm_s4(__gm__ void *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm_s4(__gm__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm_s4(__gm__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                             QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                             uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_gm_s4(__gm__ void *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_gm_s4(__gm__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_gm_s4(__gm__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                             QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                             uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ half *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ bfloat16_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ float *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ half *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                          uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                          uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                          uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                          QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                          uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub_bf16(__ubuf__ uint16_t *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub_bf16(__ubuf__ uint16_t *dst, __cc__ float *src, uint8_t sid,
                               uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
                               uint16_t srcStride, uint8_t clip_relu_pre, uint8_t UnitFlagMode,
                               QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit,
                               bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub_bf16(__ubuf__ uint16_t *dst, __cc__ float *src, uint8_t sid,
                               uint16_t NSize, uint16_t MSize, uint32_t dstStride_dst_D,
                               uint16_t srcStride, uint8_t clip_relu_pre, uint8_t UnitFlagMode,
                               QuantMode_t QuantPRE, uint8_t ReLUPRE, bool channelSplit,
                               bool NZ2ND_EN, QuantMode_post quant_post, uint8_t relu_post,
                               bool clip_relu_post, uint8_t eltwiseOP, uint8_t eltwise_antq_cfg,
                               bool C0_Pad_EN);
void copy_matrix_cc_to_ub_s4(__ubuf__ void *dst, __cc__ float *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub_s4(__ubuf__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub_s4(__ubuf__ void *dst, __cc__ float *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                             QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                             uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ub_s4(__ubuf__ void *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void copy_matrix_cc_to_ub_s4(__ubuf__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN, bool C0_Pad_EN);
void copy_matrix_cc_to_ub_s4(__ubuf__ void *dst, __cc__ int32_t *src, uint8_t sid, uint16_t NSize,
                             uint16_t MSize, uint32_t dstStride_dst_D, uint16_t srcStride,
                             uint8_t clip_relu_pre, uint8_t UnitFlagMode, QuantMode_t QuantPRE,
                             uint8_t ReLUPRE, bool channelSplit, bool NZ2ND_EN,
                             QuantMode_post quant_post, uint8_t relu_post, bool clip_relu_post,
                             uint8_t eltwiseOP, uint8_t eltwise_antq_cfg, bool C0_Pad_EN);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_matrix_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_small_matrix_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config,
                                  ConvRelu_t crMode);
void copy_small_matrix_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid,
                                  uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                                  uint16_t dstStride, ConvRelu_t crMode);
void copy_ubuf_to_cbuf(__cbuf__ void *dst, __ubuf__ void *src, uint64_t config);
void copy_ubuf_to_cbuf(__cbuf__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                       uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_ubuf_to_fbuf(__fbuf__ void *dst, __ubuf__ void *src, uint64_t config);
void copy_ubuf_to_fbuf(__fbuf__ void *dst, __ubuf__ void *src, CvtMode_t cvtMode, uint16_t burstNum,
                       uint16_t burstLen, uint16_t srcGapSize, uint16_t dstGapSize);
void copy_ubuf_to_gm(__gm__ void *dst, __ubuf__ void *src, uint64_t config);
void copy_ubuf_to_gm(__gm__ void *dst, __ubuf__ void *src, uint64_t config, bm_t byteMode);
void copy_ubuf_to_gm(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride, bm_t byteMode);
void copy_ubuf_to_gm(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                     uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_ubuf_to_gm_align_b16(__gm__ void *dst, __ubuf__ void *src, uint64_t config,
                               uint64_t gapConfig);
void copy_ubuf_to_gm_align_b16(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                               uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                               uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align_b32(__gm__ void *dst, __ubuf__ void *src, uint64_t config,
                               uint64_t gapConfig);
void copy_ubuf_to_gm_align_b32(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                               uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                               uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align_b8(__gm__ void *dst, __ubuf__ void *src, uint64_t config,
                              uint64_t gapConfig);
void copy_ubuf_to_gm_align_b8(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                              uint32_t lenBurst, uint8_t leftPaddingNum, uint8_t rightPaddingNum,
                              uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ int8_t *dst, __ubuf__ int8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ uint8_t *dst, __ubuf__ uint8_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ int16_t *dst, __ubuf__ int16_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid,
                           uint16_t nBurst, uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_align(__gm__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t nBurst,
                           uint32_t lenBurst, uint32_t srcGap, uint32_t dstGap);
void copy_ubuf_to_gm_pad_b16(__gm__ void *dst, __ubuf__ void *src, uint64_t config,
                             uint64_t paddingConfig);
void copy_ubuf_to_gm_pad_b16(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                             uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                             uint8_t leftPaddingNum, uint8_t rightPaddingNum);
void copy_ubuf_to_gm_pad_b32(__gm__ void *dst, __ubuf__ void *src, uint64_t config,
                             uint64_t paddingConfig);
void copy_ubuf_to_gm_pad_b32(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                             uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                             uint8_t leftPaddingNum, uint8_t rightPaddingNum);
void copy_ubuf_to_gm_pad_b8(__gm__ void *dst, __ubuf__ void *src, uint64_t config,
                            uint64_t paddingConfig);
void copy_ubuf_to_gm_pad_b8(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            uint8_t leftPaddingNum, uint8_t rightPaddingNum);
void copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, uint64_t config);
void copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t nBurst,
                       uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride);
void copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, uint16_t nBurst, uint16_t lenBurst,
                       uint16_t srcStride, uint16_t dstStride);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t sid, uint16_t nBurst,
                            uint16_t lenBurst, uint16_t srcStride, uint16_t dstStride,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config,
                            ConvRelu_t crMode);
void copy_vector_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid,
                            uint16_t nBurst, uint16_t lenBurst, uint16_t srcStride,
                            uint16_t dstStride, ConvRelu_t crMode);
void copy_cbuf_to_gm_align_v2(__gm__ void *dst_addr, __cbuf__ void *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_cbuf_to_gm_align_v2(__gm__ void *dst_addr, __cbuf__ void *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t l2_cache_ctl,
                              uint64_t burst_dst_stride, uint32_t burst_src_stride);
void copy_ubuf_to_gm_align_v2(__gm__ void *dst_addr, __ubuf__ void *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_ubuf_to_gm_align_v2(__gm__ void *dst_addr, __ubuf__ void *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t l2_cache_ctl,
                              uint64_t burst_dst_stride, uint32_t burst_src_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ bfloat16_t *dst_addr, __gm__ bfloat16_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ bfloat16_t *dst_addr, __gm__ bfloat16_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ half *dst_addr, __gm__ half *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ half *dst_addr, __gm__ half *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ float *dst_addr, __gm__ float *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ float *dst_addr, __gm__ float *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ int16_t *dst_addr, __gm__ int16_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ int16_t *dst_addr, __gm__ int16_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ int32_t *dst_addr, __gm__ int32_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ int32_t *dst_addr, __gm__ int32_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ int8_t *dst_addr, __gm__ int8_t *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ int8_t *dst_addr, __gm__ int8_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ uint16_t *dst_addr, __gm__ uint16_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ uint16_t *dst_addr, __gm__ uint16_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ uint32_t *dst_addr, __gm__ uint32_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ uint32_t *dst_addr, __gm__ uint32_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ uint8_t *dst_addr, __gm__ uint8_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ uint8_t *dst_addr, __gm__ uint8_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ fp8_e5m2_t *dst_addr, __gm__ fp8_e5m2_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ fp8_e5m2_t *dst_addr, __gm__ fp8_e5m2_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ fp8_e4m3fn_t *dst_addr, __gm__ fp8_e4m3fn_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ fp8_e4m3fn_t *dst_addr, __gm__ fp8_e4m3fn_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void copy_gm_to_cbuf_align_v2(__cbuf__ hifloat8_t *dst_addr, __gm__ hifloat8_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_cbuf_align_v2(__cbuf__ hifloat8_t *dst_addr, __gm__ hifloat8_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ bfloat16_t *dst_addr, __gm__ bfloat16_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ bfloat16_t *dst_addr, __gm__ bfloat16_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ half *dst_addr, __gm__ half *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ half *dst_addr, __gm__ half *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ float *dst_addr, __gm__ float *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ float *dst_addr, __gm__ float *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ int16_t *dst_addr, __gm__ int16_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ int16_t *dst_addr, __gm__ int16_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ int32_t *dst_addr, __gm__ int32_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ int32_t *dst_addr, __gm__ int32_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ int8_t *dst_addr, __gm__ int8_t *src_addr, uint64_t config0,
                              uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ int8_t *dst_addr, __gm__ int8_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ uint16_t *dst_addr, __gm__ uint16_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ uint16_t *dst_addr, __gm__ uint16_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ uint32_t *dst_addr, __gm__ uint32_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ uint32_t *dst_addr, __gm__ uint32_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ uint8_t *dst_addr, __gm__ uint8_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ uint8_t *dst_addr, __gm__ uint8_t *src_addr, uint8_t sid,
                              uint32_t burst_num, uint32_t burst_len, uint8_t left_padding_count,
                              uint8_t right_padding_count, bool constant_padding_ctl,
                              uint8_t l2_cache_ctl, uint64_t burst_src_stride,
                              uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ fp8_e5m2_t *dst_addr, __gm__ fp8_e5m2_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ fp8_e5m2_t *dst_addr, __gm__ fp8_e5m2_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ fp8_e4m3fn_t *dst_addr, __gm__ fp8_e4m3fn_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ fp8_e4m3fn_t *dst_addr, __gm__ fp8_e4m3fn_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void copy_gm_to_ubuf_align_v2(__ubuf__ hifloat8_t *dst_addr, __gm__ hifloat8_t *src_addr,
                              uint64_t config0, uint64_t config1);
void copy_gm_to_ubuf_align_v2(__ubuf__ hifloat8_t *dst_addr, __gm__ hifloat8_t *src_addr,
                              uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                              uint8_t left_padding_count, uint8_t right_padding_count,
                              bool constant_padding_ctl, uint8_t l2_cache_ctl,
                              uint64_t burst_src_stride, uint32_t burst_dst_stride);
void create_ca_matrix(__ca__ void *dst, int64_t repeat, half value);
void create_ca_matrix(__ca__ half *dst, int64_t repeat, half value);
void create_ca_matrix(__ca__ int16_t *dst, int64_t repeat, half value);
void create_ca_matrix(__ca__ uint16_t *dst, int64_t repeat, half value);
void create_ca_matrix(__ca__ half *dst, int64_t repeat, uint32_t value);
void create_ca_matrix(__ca__ float *dst, int64_t repeat, uint32_t value);
void create_ca_matrix(__ca__ float *dst, int64_t repeat, half value);
void create_ca_matrix(__ca__ int16_t *dst, int64_t repeat, uint32_t value);
void create_ca_matrix(__ca__ int32_t *dst, int64_t repeat, uint32_t value);
void create_ca_matrix(__ca__ int32_t *dst, int64_t repeat, half value);
void create_ca_matrix(__ca__ uint16_t *dst, int64_t repeat, uint32_t value);
void create_ca_matrix(__ca__ uint32_t *dst, int64_t repeat, uint32_t value);
void create_ca_matrix(__ca__ uint32_t *dst, int64_t repeat, half value);
void create_ca_matrix_bf16(__ca__ bfloat16_t *dst, int64_t repeat, bfloat16_t value);
void create_cb_matrix(__cb__ void *dst, int64_t repeat, half value);
void create_cb_matrix(__cb__ half *dst, int64_t repeat, half value);
void create_cb_matrix(__cb__ int16_t *dst, int64_t repeat, half value);
void create_cb_matrix(__cb__ uint16_t *dst, int64_t repeat, half value);
void create_cb_matrix(__cb__ half *dst, int64_t repeat, uint32_t value);
void create_cb_matrix(__cb__ float *dst, int64_t repeat, uint32_t value);
void create_cb_matrix(__cb__ float *dst, int64_t repeat, half value);
void create_cb_matrix(__cb__ int16_t *dst, int64_t repeat, uint32_t value);
void create_cb_matrix(__cb__ int32_t *dst, int64_t repeat, uint32_t value);
void create_cb_matrix(__cb__ int32_t *dst, int64_t repeat, half value);
void create_cb_matrix(__cb__ uint16_t *dst, int64_t repeat, uint32_t value);
void create_cb_matrix(__cb__ uint32_t *dst, int64_t repeat, uint32_t value);
void create_cb_matrix(__cb__ uint32_t *dst, int64_t repeat, half value);
void create_cb_matrix_bf16(__cb__ bfloat16_t *dst, int64_t repeat, bfloat16_t value);
void create_cbuf_matrix(__cbuf__ void *dst, int64_t repeat, half value);
void create_cbuf_matrix(__cbuf__ half *dst, int64_t repeat, half value);
void create_cbuf_matrix(__cbuf__ int16_t *dst, int64_t repeat, half value);
void create_cbuf_matrix(__cbuf__ uint16_t *dst, int64_t repeat, half value);
void create_cbuf_matrix(__cbuf__ half *dst, int64_t repeat, uint32_t value);
void create_cbuf_matrix(__cbuf__ float *dst, int64_t repeat, uint32_t value);
void create_cbuf_matrix(__cbuf__ float *dst, int64_t repeat, half value);
void create_cbuf_matrix(__cbuf__ int16_t *dst, int64_t repeat, uint32_t value);
void create_cbuf_matrix(__cbuf__ int32_t *dst, int64_t repeat, uint32_t value);
void create_cbuf_matrix(__cbuf__ int32_t *dst, int64_t repeat, half value);
void create_cbuf_matrix(__cbuf__ uint16_t *dst, int64_t repeat, uint32_t value);
void create_cbuf_matrix(__cbuf__ uint32_t *dst, int64_t repeat, uint32_t value);
void create_cbuf_matrix(__cbuf__ uint32_t *dst, int64_t repeat, half value);
void create_cbuf_matrix_bf16(__cbuf__ bfloat16_t *dst, int64_t repeat, bfloat16_t value);
void dc_preload(__gm__ uint64_t *address, int16_t offset);
void dc_preload(__gm__ uint64_t *address, int64_t offset);
void dci();
void dcci(__gm__ void *dst, uint64_t entire);
void dcci(__gm__ void *dst, uint64_t entire, uint64_t type);
void decompress_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, int64_t config);
void decompress_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint16_t nBlock, bool sBlock,
                           uint8_t sid, uint8_t zero_value);
void decompress_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, int64_t config);
void decompress_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint16_t nBlock, bool sBlock,
                           uint8_t sid, uint8_t zero_value);
void decompress_gm_to_ub(__ubuf__ uint8_t *dst, __gm__ uint8_t *src, int64_t config);
void decompress_gm_to_ub(__ubuf__ uint8_t *dst, __gm__ uint8_t *src, uint16_t nBlock, bool sBlock,
                         uint8_t sid, uint8_t zero_value);
void decompress_gm_to_ub(__ubuf__ half *dst, __gm__ half *src, int64_t config);
void decompress_gm_to_ub(__ubuf__ half *dst, __gm__ half *src, uint16_t nBlock, bool sBlock,
                         uint8_t sid, uint8_t zero_value);
void depthwise_conv(__cc__ half *dst, __cbuf__ half *src0, __cb__ half *src1, uint64_t config,
                    bool h);
void depthwise_conv(__cc__ half *dst, __cbuf__ half *src0, __cb__ half *src1, uint16_t inW,
                    uint16_t inH, uint8_t offset, uint8_t addr_SMASK, uint8_t weightOffset,
                    uint8_t padMode, bool h);
void depthwise_conv(__cc__ float *dst, __cbuf__ half *src0, __cb__ half *src1, uint64_t config,
                    bool h);
void depthwise_conv(__cc__ float *dst, __cbuf__ half *src0, __cb__ half *src1, uint16_t inW,
                    uint16_t inH, uint8_t offset, uint8_t addr_SMASK, uint8_t weightOffset,
                    uint8_t padMode, bool h);
void depthwise_conv(__cc__ int32_t *dst, __cbuf__ int8_t *src0, __cb__ int8_t *src1,
                    uint64_t config, bool h);
void depthwise_conv(__cc__ int32_t *dst, __cbuf__ int8_t *src0, __cb__ int8_t *src1, uint16_t inW,
                    uint16_t inH, int8_t offset, uint8_t addr_SMASK, uint8_t weightOffset,
                    uint8_t padMode, bool h);
void depthwise_conv(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ uint8_t *src1,
                    uint64_t config, bool h);
void depthwise_conv(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ uint8_t *src1, uint16_t inW,
                    uint16_t inH, uint8_t offset, uint8_t addr_SMASK, uint8_t weightOffset,
                    uint8_t padMode, bool h);
void depthwise_conv(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ int8_t *src1,
                    uint64_t config, bool h);
void depthwise_conv(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ int8_t *src1, uint16_t inW,
                    uint16_t inH, uint8_t offset, uint8_t addr_SMASK, uint8_t weightOffset,
                    uint8_t padMode, bool h);
void depthwise_conv_v2(__cc__ half *dst, __cbuf__ half *src0, __cb__ half *src1, uint64_t config);
void depthwise_conv_v2(__cc__ half *dst, __cbuf__ half *src0, __cb__ half *src1, uint16_t inW,
                       uint16_t inH, uint8_t offset, uint8_t addr_SMASK, uint8_t w_START,
                       uint16_t w_STEP, uint8_t flag, bool RELU, bool offset_EN, uint8_t PAD);
void depthwise_conv_v2(__cc__ float *dst, __cbuf__ half *src0, __cb__ half *src1, uint64_t config);
void depthwise_conv_v2(__cc__ float *dst, __cbuf__ half *src0, __cb__ half *src1, uint16_t inW,
                       uint16_t inH, uint8_t offset, uint8_t addr_SMASK, uint8_t w_START,
                       uint16_t w_STEP, uint8_t flag, bool RELU, bool offset_EN, uint8_t PAD);
void depthwise_conv_v2(__cc__ int32_t *dst, __cbuf__ int8_t *src0, __cb__ int8_t *src1,
                       uint64_t config);
void depthwise_conv_v2(__cc__ int32_t *dst, __cbuf__ int8_t *src0, __cb__ int8_t *src1,
                       uint16_t inW, uint16_t inH, uint8_t offset, uint8_t addr_SMASK,
                       uint8_t w_START, uint16_t w_STEP, uint8_t flag, bool RELU, bool offset_EN,
                       uint8_t PAD);
void depthwise_conv_v2(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ uint8_t *src1,
                       uint64_t config);
void depthwise_conv_v2(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ uint8_t *src1,
                       uint16_t inW, uint16_t inH, uint8_t offset, uint8_t addr_SMASK,
                       uint8_t w_START, uint16_t w_STEP, uint8_t flag, bool RELU, bool offset_EN,
                       uint8_t PAD);
void depthwise_conv_v2(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ int8_t *src1,
                       uint64_t config);
void depthwise_conv_v2(__cc__ int32_t *dst, __cbuf__ uint8_t *src0, __cb__ int8_t *src1,
                       uint16_t inW, uint16_t inH, uint8_t offset, uint8_t addr_SMASK,
                       uint8_t w_START, uint16_t w_STEP, uint8_t flag, bool RELU, bool offset_EN,
                       uint8_t PAD);
void dsb(mem_dsb_t arg0);
uint64_t fake_overflow_status_1();
uint64_t fake_overflow_status_2();
void ffts_cross_core_sync(pipe_t pipe, uint64_t config);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint64_t config, uint64_t arg3);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint64_t config, uint64_t rnd, bm_t arg4);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa, bm_t rnd);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, uint64_t config, uint64_t arg3);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, uint64_t config, uint64_t rnd, bm_t arg4);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa, bm_t rnd);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ int8_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint16_t *src, uint64_t config, uint64_t arg3);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint16_t *src, uint64_t config, uint64_t rnd, bm_t arg4);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint16_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa, bm_t rnd);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint16_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint8_t *src, uint64_t config, uint64_t arg3);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint8_t *src, uint64_t config, uint64_t rnd, bm_t arg4);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint8_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa, bm_t rnd);
void fifr1(__ubuf__ int16_t *dst, __ubuf__ uint8_t *src, w_size_t winsize, uint8_t srcStride,
           uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
           int8_t r1c5, int8_t r1t, int8_t r2t, int8_t r3t, int8_t r4t, int8_t r5t, uint8_t rsah,
           uint8_t rsa);
void fix_depthwisein_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm,
                                uint64_t xt);
void fix_depthwisein_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                                uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                                uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                                bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua,
                                Pool_t pool, Req_t req, DualMode_t dualMode, uint16_t Ws,
                                uint16_t WSize);
void fix_depthwisein_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm,
                                uint64_t xt);
void fix_depthwisein_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src,
                                unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize,
                                uint16_t MSize, uint16_t srcBurstGap, uint16_t dstBurstGap,
                                ConvReluFix_t crMode, bool biasEn, Relu_t relub, bool elewiseEn,
                                Relu_t relua, Pool_t pool, Req_t req, DualMode_t dualMode,
                                uint16_t Ws, uint16_t WSize);
void fix_depthwisein_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm,
                                uint64_t xt);
void fix_depthwisein_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                                uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                                uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                                bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua,
                                Pool_t pool, Req_t req, DualMode_t dualMode, uint16_t Ws,
                                uint16_t WSize);
void fix_depthwisein_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm,
                                uint64_t xt);
void fix_depthwisein_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src,
                                unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize,
                                uint16_t MSize, uint16_t srcBurstGap, uint16_t dstBurstGap,
                                ConvReluFix_t crMode, bool biasEn, Relu_t relub, bool elewiseEn,
                                Relu_t relua, Pool_t pool, Req_t req, DualMode_t dualMode,
                                uint16_t Ws, uint16_t WSize);
void fix_depthwiseout_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm,
                                 uint64_t xt);
void fix_depthwiseout_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src,
                                 unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize,
                                 uint16_t MSize, uint16_t srcBurstGap, uint16_t dstBurstGap,
                                 ConvReluFix_t crMode, bool biasEn, Relu_t relub, bool elewiseEn,
                                 Relu_t relua, Pool_t pool, Req_t req, DualMode_t dualMode,
                                 uint16_t Ws, uint16_t WSize);
void fix_depthwiseout_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm,
                                 uint64_t xt);
void fix_depthwiseout_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src,
                                 unit_flag_t unitFlagMode, uint8_t elewiseOp, uint16_t NSize,
                                 uint16_t MSize, uint16_t srcBurstGap, uint16_t dstBurstGap,
                                 ConvReluFix_t crMode, bool biasEn, Relu_t relub, bool elewiseEn,
                                 Relu_t relua, Pool_t pool, Req_t req, DualMode_t dualMode,
                                 uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ half *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ half *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ half *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ half *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ half *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ half *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ int32_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ int16_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_cbuf(__cbuf__ int32_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_matrix_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_matrix_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                           uint8_t elewiseOp, uint16_t NSize, uint16_t MSize, uint16_t srcBurstGap,
                           uint16_t dstBurstGap, ConvReluFix_t crMode, bool biasEn, Relu_t relub,
                           bool elewiseEn, Relu_t relua, Pool_t pool, Req_t req,
                           DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_cbuf(__cbuf__ int32_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_cbuf(__cbuf__ half *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_cbuf(__cbuf__ int16_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_cbuf(__cbuf__ int16_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_cbuf(__cbuf__ int8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_cbuf(__cbuf__ uint8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_cbuf(__cbuf__ int32_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_ubuf(__ubuf__ int16_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_ubuf(__ubuf__ int8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, uint64_t xm, uint64_t xt);
void fix_winograd_cc_to_ubuf(__ubuf__ uint8_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fix_winograd_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, unit_flag_t unitFlagMode,
                             uint8_t elewiseOp, uint16_t NSize, uint16_t MSize,
                             uint16_t srcBurstGap, uint16_t dstBurstGap, ConvReluFix_t crMode,
                             bool biasEn, Relu_t relub, bool elewiseEn, Relu_t relua, Pool_t pool,
                             Req_t req, DualMode_t dualMode, uint16_t Ws, uint16_t WSize);
void fmax(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void fmax(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, w_size_t winsize, uint8_t srcStride,
          uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
          int8_t r1c5);
void fmin(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void fmin(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, w_size_t winsize, uint8_t srcStride,
          uint8_t lineNum, uint8_t dstStride, int8_t r1c1, int8_t r1c2, int8_t r1c3, int8_t r1c4,
          int8_t r1c5);
int64_t get_acc_val();
int64_t get_ar();
vector_address VagCpuSim(uint16_t size, uint16_t in0, uint32_t s0);
vector_address VagCpuSim(uint16_t size, uint16_t in0, uint32_t s0, uint16_t in1, uint32_t s1);
vector_address VagCpuSim(uint16_t size, uint16_t in0, uint32_t s0, uint16_t in1, uint32_t s1,
                         uint16_t in2, uint32_t s2);
vector_address VagCpuSim(uint16_t size, uint16_t in0, uint32_t s0, uint16_t in1, uint32_t s1,
                         uint16_t in2, uint32_t s2, uint16_t in3, uint32_t s3);
void pld(vector_bool &dst, __ubuf__ uint32_t *base, vector_address offset, Literal dist);
void pld(vector_bool &dst, __ubuf__ uint32_t *base, vector_address offset, int32_t dist);
void pst(vector_bool src, __ubuf__ uint32_t *base, vector_address offset, Literal dist);
void pst(vector_bool src, __ubuf__ uint32_t *base, vector_address offset, int32_t dist);
int64_t get_arch_ver();
int64_t get_block_idx();
int64_t get_block_num();
void get_cmpmask(__ubuf__ void *dst);
int64_t get_cond();
int64_t get_condition_flag();
int64_t get_coreid();
int64_t get_ctrl();
int64_t get_data_main_base();
int64_t get_data_size();
int64_t get_data_ub_base();
int64_t get_ffts_base_addr();
int64_t get_fpc();
int64_t get_icache_prl_st();
uint64_t get_imm(uint64_t imm0_15);
uint64_t get_imm(uint64_t imm0_15, uint64_t imm16_31);
uint64_t get_imm(uint64_t imm0_15, uint64_t imm16_31, uint64_t imm32_47);
uint64_t get_imm(uint64_t imm0_15, uint64_t imm16_31, uint64_t imm32_47, uint64_t imm48_63);
int64_t get_k_num();
int64_t get_l2_in_main();
int64_t get_l2_vaddr_base();
int64_t get_lpcnt();
int64_t get_max_min_cnt();
uint64_t get_overflow_status();
int64_t get_para_base();
int64_t get_pc();
int64_t get_rpn_cor_ir();
int64_t get_rsvd_cnt();
int64_t get_safety_crc_data();
int64_t get_safety_crc_en();
int64_t get_smmu_tag_ver();
int64_t get_st_atomic_cfg();
int64_t get_stack_phy_base();
int64_t get_status();
int64_t get_subblockdim();
int64_t get_subblockid();
int64_t get_sys_cnt();
int64_t get_sys_va_base();
int64_t get_thread_dim();
int64_t get_thread_id();
int64_t get_vl();
int64_t get_vms4_sr();
void hebcd_out_to_ub(__ubuf__ void *dst, __gm__ void *src, __gm__ void *src_head, uint64_t config1);
void hebcd_out_to_ub(__ubuf__ void *dst, __gm__ void *src, __gm__ void *src_head,
                     uint8_t pictureFormat, uint8_t scrambleMode, bool readSequence, bool rawMode,
                     uint8_t smmu_tib_hit_sid, uint16_t width, uint16_t heigth, uint8_t headerLine,
                     uint16_t payload);
void hebce_l1_to_out(__gm__ void *dst_compressed, __gm__ void *dst_out, __cbuf__ void *src,
                     uint64_t config1);
void hebce_l1_to_out(__gm__ void *dst_compressed, __gm__ void *dst_out, __cbuf__ void *src,
                     bool discrete, bool tag, uint8_t pictureFormat, uint8_t scramble,
                     bool sourceReadSequence, bool raw, uint8_t smmuTlbHintSid,
                     uint16_t picutreWidth, uint16_t pictureHeight, uint8_t headerLine,
                     uint16_t payloadLine);
void hebce_ub_to_out(__gm__ void *dst_compressed, __gm__ void *dst_out, __ubuf__ void *src,
                     uint64_t config1);
void hebce_ub_to_out(__gm__ void *dst_compressed, __gm__ void *dst_out, __ubuf__ void *src,
                     bool discrete, bool tag, uint8_t pictureFormat, uint8_t scramble,
                     bool sourceReadSequence, bool raw, uint8_t smmuTlbHintSid,
                     uint16_t picutreWidth, uint16_t pictureHeight, uint8_t headerLine,
                     uint16_t payloadLine);
void hset_flag(pipe_t pipe, pipe_t tpipe, event_t eventID, mem_t memory, bool v);
void hset_flag(pipe_t pipe, pipe_t tpipe, uint64_t eventID, mem_t memory, bool v);
void hwait_flag(pipe_t pipe, pipe_t tpipe, event_t eventID, mem_t memory, bool v);
void hwait_flag(pipe_t pipe, pipe_t tpipe, uint64_t eventID, mem_t memory, bool v);
void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t config0, uint64_t config1,
                        csize_t c);
void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                        uint64_t config1, csize_t c);
void img2col_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                        uint64_t config1, csize_t c);
void img2col_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t config0, uint64_t config1,
                        csize_t c);
void img2col_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                        uint64_t config1, csize_t c);
void img2col_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                        uint64_t config1, csize_t c);
void img2col_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t config0, uint64_t config1,
                        csize_t c);
void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_ub(__ubuf__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                        uint64_t config1, csize_t c);
void img2col_cbuf_to_ub(__ubuf__ int8_t *dst, __cbuf__ int8_t *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2col_cbuf_to_ub(__ubuf__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                        uint64_t config1, csize_t c);
void img2col_cbuf_to_ub(__ubuf__ uint8_t *dst, __cbuf__ uint8_t *src, uint8_t posWk, uint8_t posHk,
                        int16_t firstWi, int16_t firstHi, uint16_t idxChannel, uint8_t strideW,
                        uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                        uint8_t dilationH, uint8_t dstJmpOffset, uint8_t enRepeat, uint8_t nRepeat,
                        csize_t c);
void img2colv2_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t config0, uint64_t config1);
void img2colv2_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t config0, uint64_t config1,
                          bm_t config2);
void img2colv2_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool filterW, bool filterH, bool transpose, bool fmatrixCtrl,
                          uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool filterW, bool filterH, bool transpose, bool fmatrixCtrl,
                          uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool filterW, bool filterH, bool transpose, bool fmatrixCtrl,
                          uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ fp8_e4m3fn_t *dst, __cbuf__ fp8_e4m3fn_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ fp8_e5m2_t *dst, __cbuf__ fp8_e5m2_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca(__ca__ hifloat8_t *dst, __cbuf__ hifloat8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint64_t config0,
                             uint64_t config1);
void img2colv2_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint64_t config0,
                             uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool filterW, bool filterH, bool transpose, bool fmatrixCtrl,
                             uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t config0, uint64_t config1);
void img2colv2_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t config0, uint64_t config1,
                          bm_t config2);
void img2colv2_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool sizeChannel, bool src3, bool src4, bool src5, uint16_t src6);
void img2colv2_cbuf_to_cb(__cb__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_cb(__cb__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_cb(__cb__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb(__cb__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool sizeChannel, bool src3, bool src4, bool src5,
                          uint16_t src6);
void img2colv2_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool sizeChannel, bool src3, bool src4, bool src5, uint16_t src6);
void img2colv2_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool sizeChannel, bool src3, bool src4, bool src5,
                          uint16_t src6);
void img2colv2_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool filterSizeW, bool filterSizeH, bool fMatrixCtrl, bool sizeChannel,
                          uint16_t src6);
void img2colv2_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool sizeChannel, bool src3, bool src4, bool src5,
                          uint16_t src6);
void img2colv2_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterSizeW, bool filterSizeH, bool fMatrixCtrl,
                          bool sizeChannel, uint16_t src6);
void img2colv2_cbuf_to_cb(__cb__ fp8_e4m3fn_t *dst, __cbuf__ fp8_e4m3fn_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ fp8_e5m2_t *dst, __cbuf__ fp8_e5m2_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb(__cb__ hifloat8_t *dst, __cbuf__ hifloat8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterW, bool filterH, bool transpose,
                          bool fmatrixCtrl, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint64_t config0,
                             uint64_t config1);
void img2colv2_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint64_t config0,
                             uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool filterSizeW, bool filterSizeH, bool fMatrixCtrl, bool sizeChannel,
                             uint16_t src6);
void img2colv2_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool filterSizeW, bool filterSizeH, bool fMatrixCtrl, bool sizeChannel,
                          uint16_t src6);
void img2colv2_cbuf_to_ub(__ubuf__ float *dst, __cbuf__ float *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ub(__ubuf__ float *dst, __cbuf__ float *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ub(__ubuf__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ub(__ubuf__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ub(__ubuf__ float *dst, __cbuf__ float *src, uint16_t stepK, uint16_t stepM,
                          uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                          uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                          bool filterSizeW, bool filterSizeH, bool fMatrixCtrl, bool sizeChannel,
                          uint16_t src6);
void img2colv2_cbuf_to_ub(__ubuf__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ub(__ubuf__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ub(__ubuf__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ub(__ubuf__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ub(__ubuf__ int32_t *dst, __cbuf__ int32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterSizeW, bool filterSizeH, bool fMatrixCtrl,
                          bool sizeChannel, uint16_t src6);
void img2colv2_cbuf_to_ub(__ubuf__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ub(__ubuf__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ub(__ubuf__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ub(__ubuf__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ub(__ubuf__ int8_t *dst, __cbuf__ int8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterSizeW, bool filterSizeH, bool fMatrixCtrl,
                          bool sizeChannel, uint16_t src6);
void img2colv2_cbuf_to_ub(__ubuf__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ub(__ubuf__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ub(__ubuf__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ub(__ubuf__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ub(__ubuf__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterSizeW, bool filterSizeH, bool fMatrixCtrl,
                          bool sizeChannel, uint16_t src6);
void img2colv2_cbuf_to_ub(__ubuf__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                          uint64_t config1);
void img2colv2_cbuf_to_ub(__ubuf__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                          uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ub(__ubuf__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ub(__ubuf__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ub(__ubuf__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t stepK,
                          uint16_t stepM, uint16_t posK, uint16_t posM, uint8_t strideW,
                          uint8_t strideH, uint8_t Wk, uint8_t Hk, uint8_t dilationW,
                          uint8_t dilationH, bool filterSizeW, bool filterSizeH, bool fMatrixCtrl,
                          bool sizeChannel, uint16_t src6);
void img2colv2_cbuf_to_ub_s4(__ubuf__ void *dst, __cbuf__ void *src, uint64_t config0,
                             uint64_t config1);
void img2colv2_cbuf_to_ub_s4(__ubuf__ void *dst, __cbuf__ void *src, uint64_t config0,
                             uint64_t config1, bm_t config2);
void img2colv2_cbuf_to_ub_s4(__ubuf__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool transpose, uint16_t sizeChannel);
void img2colv2_cbuf_to_ub_s4(__ubuf__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool transpose, uint16_t sizeChannel, bm_t enDualSrc);
void img2colv2_cbuf_to_ub_s4(__ubuf__ void *dst, __cbuf__ void *src, uint16_t stepK, uint16_t stepM,
                             uint16_t posK, uint16_t posM, uint8_t strideW, uint8_t strideH,
                             uint8_t Wk, uint8_t Hk, uint8_t dilationW, uint8_t dilationH,
                             bool filterSizeW, bool filterSizeH, bool fMatrixCtrl, bool sizeChannel,
                             uint16_t src6);
uint64_t insert_imm(uint64_t dst, uint8_t uimm8, uint8_t pos, bool ext);
uint64_t insert_reg(uint64_t dst, uint64_t src, uint8_t k, uint8_t n);
uint64_t ld_dev(uint16_t *src, int16_t offset);
uint64_t ld_dev(uint32_t *src, int16_t offset);
uint64_t ld_dev(uint64_t *src, int16_t offset);
uint64_t ld_dev(uint8_t *src, int16_t offset);
void ldva(ub_addr8_t dst, uint64_t src, bool h);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config,
                     bool transpose);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     bool hw_wait_ctrl, bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config0,
                     uint64_t config1, bool transpose);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ int16_t *dst, __cbuf__ int16_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ uint16_t *dst, __cbuf__ uint16_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t config, bool transpose);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config, bool transpose);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config, bool transpose);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca(__ca__ float *dst, __cbuf__ float *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ fp8_e5m2_t *dst, __cbuf__ fp8_e5m2_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ fp8_e4m3fn_t *dst, __cbuf__ fp8_e4m3fn_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca(__ca__ hifloat8_t *dst, __cbuf__ hifloat8_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint64_t config, bool transpose,
                        addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
                        uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                        addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
                        uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                        bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_ca_s4(__ca__ void *dst, __cbuf__ void *src, uint16_t mStartPosition,
                        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                        uint16_t dstStride, bool transpose);
void load_cbuf_to_ca_transpose(__ca__ half *dst, __cbuf__ half *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_ca_s4(__ca__ fp4x2_e1m2_t *dst, __cbuf__ fp4x2_e1m2_t *src,
                        uint16_t mStartPosition, uint16_t kStartPosition, uint8_t mStep,
                        uint8_t kStep, int16_t srcStride, uint16_t dstStride, bool transpose);
void load_cbuf_to_ca_s4(__ca__ fp4x2_e2m1_t *dst, __cbuf__ fp4x2_e2m1_t *src,
                        uint16_t mStartPosition, uint16_t kStartPosition, uint8_t mStep,
                        uint8_t kStep, int16_t srcStride, uint16_t dstStride, bool transpose);
void load_cbuf_to_ca_mx(uint64_t dst, __cbuf__ bfloat16_t *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_ca_mx(uint64_t dst, __cbuf__ half *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_ca_mx(uint64_t dst, __cbuf__ int16_t *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_ca_mx(uint64_t dst, __cbuf__ uint16_t *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_ca_mx(uint64_t dst, __cbuf__ void *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_ca_transpose(__ca__ half *dst, __cbuf__ half *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_ca_transpose(__ca__ float *dst, __cbuf__ float *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_ca_transpose(__ca__ float *dst, __cbuf__ float *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_ca_transpose(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_ca_transpose(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_ca_transpose(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_ca_transpose(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_ca_transpose(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_ca_transpose(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_ca_transpose(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_ca_transpose(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_ca_transpose(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_ca_winograd(__ca__ half *dst, __cbuf__ half *src, uint64_t config0,
                              uint64_t config1);
void load_cbuf_to_ca_winograd(__ca__ half *dst, __cbuf__ half *src, uint16_t FMWidth,
                              uint16_t FMHeight, uint16_t FMChannel, uint8_t dstGap,
                              uint8_t colIndicator, uint8_t padModeHc, uint8_t padModeV,
                              uint16_t stepK, uint16_t posK, uint16_t stepM, uint16_t posM);
void load_cbuf_to_ca_winograd(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                              uint64_t config1);
void load_cbuf_to_ca_winograd(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t FMWidth,
                              uint16_t FMHeight, uint16_t FMChannel, uint8_t dstGap,
                              uint8_t colIndicator, uint8_t padModeHc, uint8_t padModeV,
                              uint16_t stepK, uint16_t posK, uint16_t stepM, uint16_t posM);
void load_cbuf_to_ca_winograd(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                              uint64_t config1);
void load_cbuf_to_ca_winograd(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t FMWidth,
                              uint16_t FMHeight, uint16_t FMChannel, uint8_t dstGap,
                              uint8_t colIndicator, uint8_t padModeHc, uint8_t padModeV,
                              uint16_t stepK, uint16_t posK, uint16_t stepM, uint16_t posM);
void load_cbuf_to_ca_winograd_v2(__ca__ int16_t *dst, __cbuf__ int16_t *src, uint64_t config0,
                                 uint64_t config1);
void load_cbuf_to_ca_winograd_v2(__ca__ int16_t *dst, __cbuf__ int16_t *src, uint16_t FMWidth,
                                 uint16_t FMHeight, uint16_t FMChannel, uint8_t padModeH,
                                 uint8_t padModeV, uint16_t stepK, uint16_t posK, uint16_t stepM,
                                 uint16_t posM);
void load_cbuf_to_ca_winograd_v2(__ca__ int16_t *dst, __cbuf__ int16_t *src, uint16_t FMWidth,
                                 uint16_t FMHeight, uint16_t FMChannel, uint8_t padModeH,
                                 uint8_t padModeV, uint16_t stepK, uint16_t posK, uint16_t stepM,
                                 uint16_t posM, bool FMenable);
void load_cbuf_to_ca_winograd_v2(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config0,
                                 uint64_t config1);
void load_cbuf_to_ca_winograd_v2(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t FMWidth,
                                 uint16_t FMHeight, uint16_t FMChannel, uint8_t padModeH,
                                 uint8_t padModeV, uint16_t stepK, uint16_t posK, uint16_t stepM,
                                 uint16_t posM);
void load_cbuf_to_ca_winograd_v2(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t FMWidth,
                                 uint16_t FMHeight, uint16_t FMChannel, uint8_t padModeH,
                                 uint8_t padModeV, uint16_t stepK, uint16_t posK, uint16_t stepM,
                                 uint16_t posM, bool FMenable);
void load_cbuf_to_ca_winograd_v2(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config0,
                                 uint64_t config1);
void load_cbuf_to_ca_winograd_v2(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t FMWidth,
                                 uint16_t FMHeight, uint16_t FMChannel, uint8_t padModeH,
                                 uint8_t padModeV, uint16_t stepK, uint16_t posK, uint16_t stepM,
                                 uint16_t posM, bool FMenable);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config,
                     bool transpose);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     bool hw_wait_ctrl, bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint64_t config0,
                     uint64_t config1, bool transpose);
void load_cbuf_to_cb(__ca__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t config, bool transpose);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_cb(__ca__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__ca__ float *dst, __cbuf__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__ca__ int32_t *dst, __cbuf__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config, bool transpose);
void load_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_cb(__ca__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__ca__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config, bool transpose);
void load_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose);
void load_cbuf_to_cb(__ca__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, bool transpose,
                     addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ float *dst, __cbuf__ float *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ int16_t *dst, __cbuf__ int16_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ uint16_t *dst, __cbuf__ uint16_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ fp8_e5m2_t *dst, __cbuf__ fp8_e5m2_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cb__ fp8_e4m3fn_t *dst, __cbuf__ fp8_e4m3fn_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb(__cbuf__ hifloat8_t *dst, __cbuf__ hifloat8_t *src, uint16_t mStartPosition,
                     uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                     uint16_t dstStride, bool transpose);
void load_cbuf_to_cb_b2(__cb__ void *dst, __cbuf__ void *src, uint64_t config);
void load_cbuf_to_cb_b2(__cb__ half *dst, __cbuf__ half *src, uint64_t config);
void load_cbuf_to_cb_b2(__cb__ half *dst, __cbuf__ half *src, uint8_t repeat, uint8_t sid);
void load_cbuf_to_cb_b2(__cb__ half *dst, __cbuf__ half *src, uint8_t repeat);
void load_cbuf_to_cb_b2(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config);
void load_cbuf_to_cb_b2(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint8_t repeat, uint8_t sid);
void load_cbuf_to_cb_b2(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint8_t repeat);
void load_cbuf_to_cb_b2(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config);
void load_cbuf_to_cb_b2(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint8_t repeat, uint8_t sid);
void load_cbuf_to_cb_b2(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint8_t repeat);
void load_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint64_t config, bool transpose,
                        addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
                        uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool transpose,
                        addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t baseIdx, uint8_t repeat,
                        uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                        bool transpose, addr_cal_mode_t addr_cal_mode);
void load_cbuf_to_cb_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t mStartPosition,
                        uint16_t kStartPosition, uint8_t mStep, uint8_t kStep, int16_t srcStride,
                        uint16_t dstStride, bool transpose);
void load_cbuf_to_cb_s4(__cb__ fp4x2_e1m2_t *dst, __cbuf__ fp4x2_e1m2_t *src,
                        uint16_t mStartPosition, uint16_t kStartPosition, uint8_t mStep,
                        uint8_t kStep, int16_t srcStride, uint16_t dstStride, bool transpose);
void load_cbuf_to_cb_s4(__cb__ fp4x2_e2m1_t *dst, __cbuf__ fp4x2_e2m1_t *src,
                        uint16_t mStartPosition, uint16_t kStartPosition, uint8_t mStep,
                        uint8_t kStep, int16_t srcStride, uint16_t dstStride, bool transpose);
void load_gm_to_ca_2dv2(__ca__ half *dst, __gm__ half *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2(__ca__ float *dst, __gm__ float *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2(__ca__ int16_t *dst, __gm__ int16_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2(__ca__ int32_t *dst, __gm__ int32_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2(__ca__ uint32_t *dst, __gm__ uint32_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2(__ca__ int64_t *dst, __gm__ int32_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2(__ca__ int8_t *dst, __gm__ int8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_ca_2dv2_s4(__ca__ void *dst, __gm__ void *src, uint32_t mStartPosition,
                           uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                           uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ half *dst, __gm__ half *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ float *dst, __gm__ float *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ int16_t *dst, __gm__ int16_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ int32_t *dst, __gm__ int32_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ uint32_t *dst, __gm__ uint32_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ int64_t *dst, __gm__ int64_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ int8_t *dst, __gm__ int8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                        uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cb_2dv2_s4(__cb__ void *dst, __gm__ void *src, uint32_t mStartPosition,
                           uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                           uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ half *dst, __gm__ half *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ float *dst, __gm__ float *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ int16_t *dst, __gm__ int16_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ int64_t *dst, __gm__ int64_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                          uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_gm_to_cbuf_2dv2_s4(__cbuf__ void *dst, __gm__ void *src, uint32_t mStartPosition,
                             uint32_t kStartPosition, int32_t srcStride, uint8_t dstStride,
                             uint8_t mStep, uint8_t kStep, uint8_t sid);
void load_cbuf_to_cb_mx(uint64_t dst, __cbuf__ bfloat16_t *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_cb_mx(uint64_t dst, __cbuf__ half *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_cb_mx(uint64_t dst, __cbuf__ int16_t *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_cb_mx(uint64_t dst, __cbuf__ uint16_t *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_cb_mx(uint64_t dst, __cbuf__ void *src, uint16_t xStartPosition,
                        uint16_t yStartPosition, uint8_t xStep, uint8_t yStep, uint16_t srcStride,
                        uint16_t dstStride);
void load_cbuf_to_cb_sp(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config);
void load_cbuf_to_cb_sp(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t startID,
                        uint8_t repeatTime);
void load_cbuf_to_cb_transpose(__cb__ half *dst, __cbuf__ half *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_cb_transpose(__cb__ half *dst, __cbuf__ half *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose(__cb__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose(__cb__ float *dst, __cbuf__ float *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_cb_transpose(__cb__ float *dst, __cbuf__ float *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_cb_transpose(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_cb_transpose(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_cb_transpose(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint64_t config,
                               uint64_t fracStride);
void load_cbuf_to_cb_transpose(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose_s4(__cb__ void *dst, __cbuf__ void *src, uint64_t config,
                                  uint64_t fracStride);
void load_cbuf_to_cb_transpose_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t indexID,
                                  uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                                  bool addrmode, uint16_t dstFracStride);
void load_cbuf_to_cb_transpose_s4(__cb__ void *dst, __cbuf__ void *src, uint16_t indexID,
                                  uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                                  bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_transpose(__cb__ half *dst, __cbuf__ half *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_transpose(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_transpose(__cb__ uint8_t *dst, __cbuf__ uint8_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_transpose(__cb__ int32_t *dst, __cbuf__ int32_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_transpose(__cb__ uint32_t *dst, __cbuf__ uint32_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_transpose(__cb__ float *dst, __cbuf__ float *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_transpose(__cb__ bfloat16_t *dst, __cbuf__ bfloat16_t *src, uint16_t indexID,
                               uint8_t repeat, uint16_t srcStride, uint16_t dstStride,
                               bool addrmode, uint16_t dstFracStride, uint16_t srcFracStride);
void load_cbuf_to_cb_winograd(__cb__ half *dst, __cbuf__ half *src, uint64_t config);
void load_cbuf_to_cb_winograd(__cb__ half *dst, __cbuf__ half *src, uint8_t innerDstStride,
                              uint16_t srcRepeatStride, uint8_t dstRepeatStride, uint8_t addr_SMASK,
                              uint8_t weightIndicator, bool repeatIndicator,
                              bool weightMatrixOffset, uint8_t repeatStride);
void load_cbuf_to_cb_winograd(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config);
void load_cbuf_to_cb_winograd(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint8_t innerDstStride,
                              uint16_t srcRepeatStride, uint8_t dstRepeatStride, uint8_t addr_SMASK,
                              uint8_t weightIndicator, bool repeatIndicator,
                              bool weightMatrixOffset, uint8_t repeatStride);
void load_cbuf_to_cb_winograd_v2(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint64_t config);
void load_cbuf_to_cb_winograd_v2(__cb__ int8_t *dst, __cbuf__ int8_t *src, uint16_t inChannelSize,
                                 uint16_t totalInChannelSize, uint16_t outChannelSize,
                                 bool dataType);
void load_decompress_header_from_gm(__gm__ void *dst, uint64_t config);
void load_decompress_header_from_gm(__gm__ void *dst, uint16_t nBlock, uint8_t sid);
void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint64_t config);
void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_ca(__ca__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config);
void load_gm_to_ca(__ca__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_ca(__ca__ float *dst, __gm__ float *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ float *dst, __gm__ float *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ float *dst, __gm__ float *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int32_t *dst, __gm__ int32_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int32_t *dst, __gm__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int32_t *dst, __gm__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int8_t *dst, __gm__ int8_t *src, uint64_t config);
void load_gm_to_ca(__ca__ int8_t *dst, __gm__ int8_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_ca(__ca__ uint32_t *dst, __gm__ uint32_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ uint32_t *dst, __gm__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ uint32_t *dst, __gm__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config);
void load_gm_to_ca(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca_2dv2(__ca__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_2dv2(__ca__ half *dst, __gm__ half *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_2dv2(__ca__ float *dst, __gm__ float *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_2dv2(__ca__ int8_t *dst, __gm__ int8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_2dv2(__ca__ uint8_t *dst, __gm__ uint8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_2dv2(__ca__ fp8_e5m2_t *dst, __gm__ fp8_e5m2_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_2dv2(__ca__ fp8_e4m3fn_t *dst, __gm__ fp8_e4m3fn_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_2dv2(__ca__ hifloat8_t *dst, __gm__ hifloat8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint8_t dstStride, uint8_t mStep, uint8_t kStep,
                        uint8_t sid, uint8_t l2CacheCtl);
void load_gm_to_ca_s4(__ca__ void *dst, __gm__ void *src, uint64_t config,
                      addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca_s4(__ca__ void *dst, __gm__ void *src, uint16_t baseIdx, uint8_t repeat,
                      uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                      addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca_s4(__ca__ void *dst, __gm__ void *src, uint16_t baseIdx, uint8_t repeat,
                      uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                      addr_cal_mode_t addr_cal_mode);
void load_gm_to_ca_unzip(__ca__ half *dst, __gm__ half *src);
void load_gm_to_ca_unzip(__ca__ void *dst, __gm__ void *src);
void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint64_t config);
void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_cb(__cb__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config);
void load_gm_to_cb(__cb__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_cb(__cb__ float *dst, __gm__ float *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ float *dst, __gm__ float *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ float *dst, __gm__ float *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int32_t *dst, __gm__ int32_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int32_t *dst, __gm__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int32_t *dst, __gm__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int8_t *dst, __gm__ int8_t *src, uint64_t config);
void load_gm_to_cb(__cb__ int8_t *dst, __gm__ int8_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_cb(__cb__ uint32_t *dst, __gm__ uint32_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ uint32_t *dst, __gm__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ uint32_t *dst, __gm__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config);
void load_gm_to_cb(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                   addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid);
void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                   uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb_2dv2(__cb__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_2dv2(__cb__ half *dst, __gm__ half *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_2dv2(__cb__ float *dst, __gm__ float *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_2dv2(__cb__ int8_t *dst, __gm__ int8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_2dv2(__cb__ uint8_t *dst, __gm__ uint8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_2dv2(__cb__ fp8_e5m2_t *dst, __gm__ fp8_e5m2_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_2dv2(__cb__ fp8_e4m3fn_t *dst, __gm__ fp8_e4m3fn_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_2dv2(__cbuf__ hifloat8_t *dst, __gm__ hifloat8_t *src, uint32_t mStartPosition,
                        uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep, uint16_t kStep,
                        uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cb_s4(__cb__ void *dst, __gm__ void *src, uint64_t config,
                      addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb_s4(__cb__ void *dst, __gm__ void *src, uint16_t baseIdx, uint8_t repeat,
                      uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                      addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb_s4(__cb__ void *dst, __gm__ void *src, uint16_t baseIdx, uint8_t repeat,
                      uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                      addr_cal_mode_t addr_cal_mode);
void load_gm_to_cb_unzip(__cb__ half *dst, __gm__ half *src);
void load_gm_to_cb_unzip(__cb__ void *dst, __gm__ void *src);
void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint64_t config);
void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint64_t config,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid);
void load_gm_to_cbuf(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config);
void load_gm_to_cbuf(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint64_t config,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     bool hw_wait_ctrl, addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint16_t baseIdx,
                     uint8_t repeat, uint16_t srcStride, uint8_t sid);
void load_gm_to_cbuf(__cbuf__ float *dst, __gm__ float *src, uint64_t config,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ float *dst, __gm__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ float *dst, __gm__ float *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint64_t config,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int32_t *dst, __gm__ int32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint64_t config);
void load_gm_to_cbuf(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint64_t config,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid);
void load_gm_to_cbuf(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint64_t config,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ uint32_t *dst, __gm__ uint32_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config);
void load_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint64_t config,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                     addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid);
void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint16_t baseIdx, uint8_t repeat,
                     uint16_t srcStride, uint8_t sid, addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf_2dv2(__cbuf__ bfloat16_t *dst, __gm__ bfloat16_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                          uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2(__cbuf__ half *dst, __gm__ half *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                          uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2(__cbuf__ float *dst, __gm__ float *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                          uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2(__cbuf__ int8_t *dst, __gm__ int8_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                          uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2(__cbuf__ uint8_t *dst, __gm__ uint8_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                          uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2(__cbuf__ fp8_e5m2_t *dst, __gm__ fp8_e5m2_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                          uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2(__cbuf__ fp8_e4m3fn_t *dst, __gm__ fp8_e4m3fn_t *src,
                          uint32_t mStartPosition, uint32_t kStartPosition, uint16_t dstStride,
                          uint16_t mStep, uint16_t kStep, uint8_t sid, uint8_t decompMode,
                          uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2(__cbuf__ hifloat8_t *dst, __gm__ hifloat8_t *src, uint32_t mStartPosition,
                          uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                          uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_2dv2_s4(__cbuf__ void *dst, __gm__ void *src, uint32_t mStartPosition,
                             uint32_t kStartPosition, uint16_t dstStride, uint16_t mStep,
                             uint16_t kStep, uint8_t sid, uint8_t decompMode, uint8_t l2CacheCtl);
void load_gm_to_cbuf_s4(__cbuf__ void *dst, __gm__ void *src, uint64_t config,
                        addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf_s4(__cbuf__ void *dst, __gm__ void *src, uint16_t baseIdx, uint8_t repeat,
                        uint16_t srcStride, uint16_t dstStride, uint8_t sid,
                        addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf_s4(__cbuf__ void *dst, __gm__ void *src, uint16_t baseIdx, uint8_t repeat,
                        uint16_t srcStride, uint16_t dstStride, uint8_t sid, bool hw_wait_ctrl,
                        addr_cal_mode_t addr_cal_mode);
void load_gm_to_cbuf_unzip(__cbuf__ half *dst, __gm__ half *src);
void load_gm_to_cbuf_unzip(__cbuf__ void *dst, __gm__ void *src);
void load_image_to_cbuf(__cbuf__ half *dst, uint64_t xs, uint64_t xt);
void load_image_to_cbuf(__cbuf__ half *dst, uint16_t horSize, uint16_t verSize, uint16_t horStartP,
                        uint16_t verStartP, uint16_t sHorRes, uint8_t topPadSize,
                        uint8_t botPadSize, uint16_t lPadSize, uint16_t rPadSize, uint8_t sid);
void load_image_to_cbuf(__cbuf__ int16_t *dst, uint64_t xs, uint64_t xt);
void load_image_to_cbuf(__cbuf__ int16_t *dst, uint16_t horSize, uint16_t verSize,
                        uint16_t horStartP, uint16_t verStartP, uint16_t sHorRes,
                        uint8_t topPadSize, uint8_t botPadSize, uint16_t lPadSize,
                        uint16_t rPadSize, uint8_t sid);
void load_image_to_cbuf(__cbuf__ int8_t *dst, uint64_t xs, uint64_t xt);
void load_image_to_cbuf(__cbuf__ int8_t *dst, uint16_t horSize, uint16_t verSize,
                        uint16_t horStartP, uint16_t verStartP, uint16_t sHorRes,
                        uint8_t topPadSize, uint8_t botPadSize, uint16_t lPadSize,
                        uint16_t rPadSize, uint8_t sid);
void load_image_to_cbuf(__cbuf__ uint8_t *dst, uint64_t xs, uint64_t xt);
void load_image_to_cbuf(__cbuf__ uint8_t *dst, uint16_t horSize, uint16_t verSize,
                        uint16_t horStartP, uint16_t verStartP, uint16_t sHorRes,
                        uint8_t topPadSize, uint8_t botPadSize, uint16_t lPadSize,
                        uint16_t rPadSize, uint8_t sid);
void load_smask_table_from_cbuf(uint16_t dst, __cbuf__ void *src, uint64_t config);
void load_smask_table_from_gm(uint16_t dst, __gm__ void *src, uint64_t config);
void load_smask_table_from_ub(uint16_t dst, __ubuf__ void *src, uint64_t config);
void load_unzip_index_from_gm(__gm__ void *src, uint64_t config);
void load_unzip_index_from_gm(__gm__ void *src, uint32_t numOfIndexTabEntry, uint8_t sid);
void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint64_t config);
void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_controlC);
void mad(__cc__ half *dst, __ca__ half *src0, __cb__ half *src1, uint16_t m, uint16_t k, uint16_t n,
         uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA, bool ctrlMatrixB,
         bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint64_t config);
void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_controlC);
void mad(__cc__ float *dst, __ca__ half *src0, __cb__ half *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA, bool ctrlMatrixB,
         bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ float *dst, __ca__ half *src0, __cb__ half *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
         bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ bfloat16_t *a, __cb__ bfloat16_t *b, uint64_t config);
void mad(__cc__ float *c, __ca__ bfloat16_t *a, __cb__ bfloat16_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *dst, __ca__ bfloat16_t *src0, __cb__ bfloat16_t *src1, uint16_t m,
         uint16_t k, uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
         bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ float *a, __cb__ float *b, uint64_t config);
void mad(__cc__ float *c, __ca__ float *a, __cb__ float *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ float *dst, __ca__ float *src0, __cb__ float *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
         bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ int16_t *a, __cb__ int8_t *b, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ int16_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t smaskOffset, unit_flag_t unitFlagMode, bool isWeightOffset,
         bool init_val_controlC);
void mad(__cc__ int32_t *c, __ca__ int16_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t fix_shift_val, uint8_t unitFlag, bool gemv_disable, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ int16_t *a, __cb__ int16_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t fix_shift_val, uint8_t unitFlag, bool gemv_disable, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t fix_shift_val, uint8_t unitFlag, bool gemv_disable, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t fix_shift_val, uint8_t unitFlag, bool gemv_disable, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ half *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t fix_shift_val, uint8_t unitFlag, bool gemv_disable, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_controlC);
void mad(__cc__ int32_t *dst, __ca__ int8_t *src0, __cb__ int8_t *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA, bool ctrlMatrixB,
         bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ int32_t *dst, __ca__ int8_t *src0, __cb__ int8_t *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
         bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t config);
void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint16_t m, uint16_t k,
         uint16_t n, bool init_val_controlC);
void mad(__cc__ uint32_t *dst, __ca__ uint8_t *src0, __cb__ uint8_t *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA, bool ctrlMatrixB,
         bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint16_t m, uint16_t k,
         uint16_t n, bool init_val_controlC);
void mad(__cc__ int32_t *dst, __ca__ uint8_t *src0, __cb__ uint8_t *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA, bool ctrlMatrixB,
         bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *dst, __ca__ uint8_t *src0, __cb__ uint8_t *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
         bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_controlC);
void mad(__cc__ int32_t *dst, __ca__ uint8_t *src0, __cb__ int8_t *src1, uint16_t m, uint16_t k,
         uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA, bool ctrlMatrixB,
         bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal);
void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint64_t addr, uint64_t config);
void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint64_t addr, uint16_t m, uint16_t k,
         uint16_t n, bool init_val_controlC);
void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint64_t addr, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint64_t addr, uint64_t config);
void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint64_t addr, uint16_t m, uint16_t k,
         uint16_t n, bool init_val_controlC);
void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint64_t addr, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ bfloat16_t *a, __cb__ bfloat16_t *b, uint64_t addr,
         uint64_t config);
void mad(__cc__ float *c, __ca__ bfloat16_t *a, __cb__ bfloat16_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ float *a, __cb__ float *b, uint64_t addr, uint64_t config);
void mad(__cc__ float *c, __ca__ float *a, __cb__ float *b, uint64_t addr, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ int16_t *a, __cb__ int8_t *b, uint64_t addr, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ int16_t *a, __cb__ int8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, uint8_t smaskOffset, unit_flag_t unitFlagMode, bool isWeightOffset,
         bool init_val_controlC);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint64_t addr, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, bool init_val_controlC);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t addr, uint64_t config);
void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, bool init_val_controlC);
void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t addr, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, bool init_val_controlC);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint64_t addr, uint64_t config);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, bool init_val_controlC);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint64_t addr, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e5m2_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e4m3fn_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e4m3fn_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e5m2_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ hifloat8_t *a, __cbuf__ hifloat8_t *b, uint16_t m, uint16_t k,
         uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ hifloat8_t *a, __cbuf__ hifloat8_t *b, uint64_t bias, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e5m2_t *b, uint64_t bias, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e4m3fn_t *b, uint64_t bias, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e4m3fn_t *b, uint64_t bias, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e5m2_t *b, uint64_t bias, uint16_t m,
         uint16_t k, uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
         bool cmatrixInitVal);
void mad_b8u2(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ void *b, uint64_t config);
void mad_b8u2(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ void *b, uint16_t m, uint16_t k,
              uint16_t n, bool init_val_controlC);
void mad_b8u2(__cc__ int32_t *dst, __ca__ int8_t *src0, __cb__ void *src1, uint16_t m, uint16_t k,
              uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA,
              bool ctrlMatrixB, bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad_b8u2(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ void *b, uint64_t config);
void mad_b8u2(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ void *b, uint16_t m, uint16_t k,
              uint16_t n, bool init_val_controlC);
void mad_b8u2(__cc__ int32_t *dst, __ca__ uint8_t *src0, __cb__ void *src1, uint16_t m, uint16_t k,
              uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA,
              bool ctrlMatrixB, bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad_bf162f32(__cc__ float *c, __ca__ uint16_t *a, __cb__ uint16_t *b, uint64_t config);
void mad_bf162f32(__cc__ float *c, __ca__ uint16_t *a, __cb__ uint16_t *b, uint16_t m, uint16_t k,
                  uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
                  bool cmatrixInitVal);
void mad_bf162f32(__cc__ float *dst, __ca__ uint16_t *src0, __cb__ uint16_t *src1, uint16_t m,
                  uint16_t k, uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
                  bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource,
                  bool cmatrixInitVal);
void mad_f16u2(__cc__ half *c, __ca__ half *a, __cb__ void *b, uint64_t config);
void mad_f16u2(__cc__ half *c, __ca__ half *a, __cb__ void *b, uint16_t m, uint16_t k, uint16_t n,
               bool init_val_controlC);
void mad_f16u2(__cc__ half *dst, __ca__ half *src0, __cb__ void *src1, uint16_t m, uint16_t k,
               uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA,
               bool ctrlMatrixB, bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad_s4(__cc__ int32_t *c, __ca__ void *a, __cb__ void *b, uint64_t config);
void mad_s4(__cc__ int32_t *c, __ca__ void *a, __cb__ void *b, uint16_t m, uint16_t k, uint16_t n,
            bool init_val_controlC);
void mad_s4(__cc__ int32_t *dst, __ca__ void *src0, __cb__ void *src1, uint16_t m, uint16_t k,
            uint16_t n, uint8_t featOffset, uint8_t smaskOffset, bool ctrlMatrixA, bool ctrlMatrixB,
            bool isWeightOffset, bool isSparsity, bool initMatrixC);
void mad_s4(__cc__ int32_t *c, __ca__ void *a, __cb__ void *b, uint16_t m, uint16_t k, uint16_t n,
            uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource, bool cmatrixInitVal);
void mad_s4(__cc__ int32_t *dst, __ca__ void *src0, __cb__ void *src1, uint16_t m, uint16_t k,
            uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
            bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource, bool cmatrixInitVal);
void mad_s4(__cc__ int32_t *c, __ca__ void *a, __cb__ void *b, uint16_t m, uint16_t k, uint16_t n,
            uint8_t fix_shift_val, uint8_t unitFlag, bool sub_dtype, bool gemv_disable,
            bool cmatrixSource, bool cmatrixInitVal);
void mad_s4(__cc__ int32_t *c, __ca__ void *a, __cb__ void *b, uint16_t m, uint16_t k, uint16_t n,
            uint8_t fix_shift_val, uint8_t unitFlag, bool gemv_disable, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_sp(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint64_t config);
void mad_sp(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlagMode, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e1m2_t *a, __cb__ fp4x2_e1m2_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e1m2_t *a, __cb__ fp4x2_e2m1_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e2m1_t *a, __cb__ fp4x2_e2m1_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e2m1_t *a, __cb__ fp4x2_e1m2_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e4m3fn_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e5m2_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e5m2_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e4m3fn_t *b, uint16_t m, uint16_t k,
            uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource, bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e1m2_t *a, __cb__ fp4x2_e1m2_t *b, uint64_t bias,
            uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e1m2_t *a, __cb__ fp4x2_e2m1_t *b, uint64_t bias,
            uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e2m1_t *a, __cb__ fp4x2_e2m1_t *b, uint64_t bias,
            uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp4x2_e2m1_t *a, __cb__ fp4x2_e1m2_t *b, uint64_t bias,
            uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e4m3fn_t *b, uint64_t bias,
            uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e4m3fn_t *a, __cb__ fp8_e5m2_t *b, uint64_t bias,
            uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e5m2_t *b, uint64_t bias, uint16_t m,
            uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_mx(__cc__ float *c, __ca__ fp8_e5m2_t *a, __cb__ fp8_e4m3fn_t *b, uint64_t bias,
            uint16_t m, uint16_t k, uint16_t n, uint8_t unitFlag, bool gemvCtrl, bool cmatrixSource,
            bool cmatrixInitVal);
void mad_s8s4(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ void *b, uint16_t m, uint16_t k,
              uint16_t n, uint8_t fix_shift_val, uint8_t unitFlag, bool gemv_disable,
              bool cmatrixSource, bool cmatrixInitVal);
void mad_tf322f32(__cc__ float *c, __ca__ float *a, __cb__ float *b, uint64_t config);
void mad_tf322f32(__cc__ float *c, __ca__ float *a, __cb__ float *b, uint16_t m, uint16_t k,
                  uint16_t n, uint8_t unitFlag, bool kDirectionAlign, bool cmatrixSource,
                  bool cmatrixInitVal);
void mad_tf322f32(__cc__ float *dst, __ca__ float *src0, __cb__ float *src1, uint16_t m, uint16_t k,
                  uint16_t n, uint8_t featOffset, uint8_t smaskOffset, uint8_t unitFlag,
                  bool kDirectionAlign, bool isWeightOffset, bool cmatrixSource,
                  bool cmatrixInitVal);
void mov_vspr(VSPR_t spr_id, uint64_t in1);
void mte4_mte5_off();
void mte4_mte5_on();
void mvf_dci();
void mvf_gm_to_ub_b128_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint64_t config);
void mvf_gm_to_ub_b128_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b128_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint64_t config);
void mvf_gm_to_ub_b128_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b256_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint64_t config);
void mvf_gm_to_ub_b256_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b256_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint64_t config);
void mvf_gm_to_ub_b256_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b32_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint64_t config);
void mvf_gm_to_ub_b32_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b32_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint64_t config);
void mvf_gm_to_ub_b32_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b512_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint64_t config);
void mvf_gm_to_ub_b512_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b512_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint64_t config);
void mvf_gm_to_ub_b512_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                           uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b64_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint64_t config);
void mvf_gm_to_ub_b64_u16(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void mvf_gm_to_ub_b64_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint64_t config);
void mvf_gm_to_ub_b64_u32(__ubuf__ void *dst, __ubuf__ void *srcIndex, __gm__ void *src,
                          uint16_t indexNum, uint8_t dstStride, uint8_t sid);
void pc_trace_off();
void pc_trace_on();
void pipe_barrier(pipe_t pipe);
void preload(const void *addr);
void preload(const void *addr, int64_t prefetchLen);
void release_pbid(uint64_t src);
void rpn_cor(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void rpn_cor(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t strideDst,
             uint16_t strideSrc);
void rpn_cor_diag(__ubuf__ half *dst, __ubuf__ half *src);
void rpn_cor_diag2(__ubuf__ half *dst, __ubuf__ half *src);
uint64_t sbitset0(uint64_t x, int64_t idx);
uint64_t sbitset1(uint64_t x, int64_t idx);
void scatter_vabs(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vabs(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                  uint16_t srcStride);
void scatter_vabs_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vabs_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                      uint16_t srcStride);
void scatter_vabs_f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vabs_f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                      uint16_t srcStride);
void scatter_vabs_s16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vabs_s16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                      uint16_t srcStride);
void scatter_vadd(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vadd(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vadd_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vadd_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vadd_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vadd_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vadd_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vadd_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vadd_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vadd_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, half a, uint64_t config);
void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat,
                   uint16_t dstStride, uint16_t srcStride);
void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, float a, uint64_t config);
void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat,
                   uint16_t dstStride, uint16_t srcStride);
void scatter_vadds_f16(ub_addr8_t dst, ub_addr8_t src, half a, uint64_t config);
void scatter_vadds_f16(ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vadds_f32(ub_addr8_t dst, ub_addr8_t src, float a, uint64_t config);
void scatter_vadds_f32(ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vadds_s16(ub_addr8_t dst, ub_addr8_t src, int16_t a, uint64_t config);
void scatter_vadds_s16(ub_addr8_t dst, ub_addr8_t src, int16_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vadds_s32(ub_addr8_t dst, ub_addr8_t src, int32_t a, uint64_t config);
void scatter_vadds_s32(ub_addr8_t dst, ub_addr8_t src, int32_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vaxpy(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, half value, uint64_t config,
                   bool isHigh);
void scatter_vaxpy(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, half value, uint8_t repeat,
                   uint16_t dstStride, uint16_t srcStride, bool isHigh);
void scatter_vaxpy(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, float value, uint64_t config,
                   bool isHigh);
void scatter_vaxpy(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, float value, uint8_t repeat,
                   uint16_t dstStride, uint16_t srcStride, bool isHigh);
void scatter_vaxpy_f16(ub_addr8_t Y, ub_addr8_t X, half a, uint64_t config, bool storeHighHalf);
void scatter_vaxpy_f16(ub_addr8_t Y, ub_addr8_t X, half a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride, bool storeHighHalf);
void scatter_vaxpy_f32(ub_addr8_t Y, ub_addr8_t X, float a, uint64_t config, bool storeHighHalf);
void scatter_vaxpy_f32(ub_addr8_t Y, ub_addr8_t X, float a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride, bool storeHighHalf);
void scatter_vaxpy_fmix(ub_addr8_t Y, ub_addr8_t X, half a, uint64_t config, bool storeHighHalf);
void scatter_vaxpy_fmix(ub_addr8_t Y, ub_addr8_t X, half a, uint8_t repeat, uint16_t dstStride,
                        uint16_t srcStride, bool storeHighHalf);
void scatter_vcmp_eq(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_eq_f16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_eq_f32(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_eq_s16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ge(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ge_f16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ge_f32(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ge_s16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_gt(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_gt_f16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_gt_f32(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_gt_s16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_le(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_le_f16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_le_f32(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_le_s16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_lt(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_lt_f16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_lt_f32(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_lt_s16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ne(vtype_t type, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ne_f16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ne_f32(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vcmp_ne_s16(ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void vcmps_eq(vector_bool &dst, vector_u16 src1, uint16_t src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_u16 src1, uint16_t src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_u16 src1, uint16_t src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_u16 src1, uint16_t src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_u16 src1, uint16_t src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_u16 src1, uint16_t src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_s16 src1, int16_t src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_s16 src1, int16_t src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_s16 src1, int16_t src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_s16 src1, int16_t src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_s16 src1, int16_t src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_s16 src1, int16_t src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_u32 src1, uint32_t src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_u32 src1, uint32_t src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_u32 src1, uint32_t src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_u32 src1, uint32_t src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_u32 src1, uint32_t src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_u32 src1, uint32_t src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_s32 src1, int32_t src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_s32 src1, int32_t src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_s32 src1, int32_t src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_s32 src1, int32_t src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_s32 src1, int32_t src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_s32 src1, int32_t src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_f16 src1, half src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_f16 src1, half src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_f16 src1, half src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_f16 src1, half src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_f16 src1, half src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_f16 src1, half src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_f32 src1, float src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_f32 src1, float src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_f32 src1, float src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_f32 src1, float src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_f32 src1, float src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_f32 src1, float src2, vector_bool mask);
void scatter_vconcat_f16(__ubuf__ half *dst, ub_addr8_t src, uint64_t config);
void scatter_vconcat_f16(__ubuf__ half *dst, ub_addr8_t src, uint8_t repeat,
                         uint8_t regionDataControlBit);
void scatter_vconcat_f32(__ubuf__ float *dst, ub_addr8_t src, uint64_t config);
void scatter_vconcat_f32(__ubuf__ float *dst, ub_addr8_t src, uint8_t repeat,
                         uint8_t regionDataControlBit);
void scatter_vconv_deq(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_deq(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s32a(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s32a(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s32c(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s32c(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s32f(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s32f(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s32r(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s32r(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s32z(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s32z(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s8(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s8(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s8a(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s8a(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s8c(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s8c(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s8f(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s8f(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162s8z(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162s8z(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162u8(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162u8(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162u8a(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162u8a(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162u8c(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162u8c(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162u8f(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162u8f(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f162u8z(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f162u8z(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f322f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f322f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f322f16o(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f322f16o(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f322s32a(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f322s32a(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f322s32c(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f322s32c(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f322s32f(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f322s32f(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f322s32r(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f322s32r(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_f322s32z(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_f322s32z(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_s322f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_s322f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_s82f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_s82f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool storeHighHalf);
void scatter_vconv_u82f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool storeHighHalf);
void scatter_vconv_u82f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool storeHighHalf);
void scatter_vcshuffle_b16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vcshuffle_b16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, uint8_t shuffleGroup);
void scatter_vcshuffle_b32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vcshuffle_b32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride, uint8_t shuffleGroup);
void scatter_vcshuffle_b8(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vcshuffle_b8(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, uint8_t shuffleGroup);
void scatter_vdiv(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vdiv(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vdiv_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vdiv_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vdiv_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vdiv_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vector_mov(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vector_mov(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat,
                        uint16_t dstStride, uint16_t srcStride);
void scatter_vector_mov_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vector_mov_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride);
void scatter_vector_mov_s16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vector_mov_s16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride);
void scatter_vector_mov_u16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vector_mov_u16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                            uint16_t srcStride);
void scatter_vexp(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vexp(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                  uint16_t srcStride);
void scatter_vexp_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vexp_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                      uint16_t srcStride);
void scatter_vexp_f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vexp_f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                      uint16_t srcStride);
void scatter_vextract_f16(ub_addr8_t dst, __ubuf__ half *src, uint64_t config);
void scatter_vextract_f16(ub_addr8_t dst, __ubuf__ half *src, uint8_t repeat,
                          uint8_t regionDataCtrl);
void scatter_vextract_f32(ub_addr8_t dst, __ubuf__ float *src, uint64_t config);
void scatter_vextract_f32(ub_addr8_t dst, __ubuf__ float *src, uint8_t repeat,
                          uint8_t regionDataCtrl);
void scatter_vln(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vln(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                 uint16_t srcStride);
void scatter_vln_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vln_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                     uint16_t srcStride);
void scatter_vln_f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vln_f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                     uint16_t srcStride);
void scatter_vmadd(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmadd(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                   uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmadd_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmadd_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                       uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmadd_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmadd_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                       uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmaddrelu(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1,
                       uint64_t config);
void scatter_vmaddrelu(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1,
                       uint8_t repeat, uint16_t dstStride, uint16_t src0Stride,
                       uint16_t src1Stride);
void scatter_vmaddrelu_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmaddrelu_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                           uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmaddrelu_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmaddrelu_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                           uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmax(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmax(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmax_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmax_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmax_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmax_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmax_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmax_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmax_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmax_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmaxs_f16(ub_addr8_t dst, ub_addr8_t src0, half src1, uint64_t config);
void scatter_vmaxs_f16(ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vmaxs_f32(ub_addr8_t dst, ub_addr8_t src0, float src1, uint64_t config);
void scatter_vmaxs_f32(ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vmaxs_s16(ub_addr8_t dst, ub_addr8_t src0, int16_t src1, uint64_t config);
void scatter_vmaxs_s16(ub_addr8_t dst, ub_addr8_t src, int16_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vmaxs_s32(ub_addr8_t dst, ub_addr8_t src0, int32_t src1, uint64_t config);
void scatter_vmaxs_s32(ub_addr8_t dst, ub_addr8_t src, int32_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vmin(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmin(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmin_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmin_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmin_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmin_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmin_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmin_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmin_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmin_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmins_f16(ub_addr8_t dst, ub_addr8_t src0, half src1, uint64_t config);
void scatter_vmins_f16(ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vmins_f32(ub_addr8_t dst, ub_addr8_t src0, float src1, uint64_t config);
void scatter_vmins_f32(ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vmins_s16(ub_addr8_t dst, ub_addr8_t src0, int16_t src1, uint64_t config);
void scatter_vmins_s16(ub_addr8_t dst, ub_addr8_t src, int16_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vmins_s32(ub_addr8_t dst, ub_addr8_t src0, int32_t src1, uint64_t config);
void scatter_vmins_s32(ub_addr8_t dst, ub_addr8_t src, int32_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vmla(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmla(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmla_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmla_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmla_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmla_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmla_fmix(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmla_fmix(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                       uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmul(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmul(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmul_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmul_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmul_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmul_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmul_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmul_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmul_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vmul_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vmulconv_f162s8(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config,
                             bool storeHighHalf);
void scatter_vmulconv_f162s8(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                             uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride,
                             bool storeHighHalf);
void scatter_vmulconv_f162u8(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config,
                             bool storeHighHalf);
void scatter_vmulconv_f162u8(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                             uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride,
                             bool storeHighHalf);
void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, half a, uint64_t config);
void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat,
                   uint16_t dstStride, uint16_t srcStride);
void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, float a, uint64_t config);
void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat,
                   uint16_t dstStride, uint16_t srcStride);
void scatter_vmuls_f16(ub_addr8_t dst, ub_addr8_t src, half a, uint64_t config);
void scatter_vmuls_f16(ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vmuls_f32(ub_addr8_t dst, ub_addr8_t src, float a, uint64_t config);
void scatter_vmuls_f32(ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vmuls_s16(ub_addr8_t dst, ub_addr8_t src, int16_t a, uint64_t config);
void scatter_vmuls_s16(ub_addr8_t dst, ub_addr8_t src, int16_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vmuls_s32(ub_addr8_t dst, ub_addr8_t src, int32_t a, uint64_t config);
void scatter_vmuls_s32(ub_addr8_t dst, ub_addr8_t src, int32_t a, uint8_t repeat,
                       uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b16(ub_addr8_t dst, __ubuf__ int16_t *src, uint64_t config);
void scatter_vnchwconv_b16(ub_addr8_t dst, __ubuf__ int16_t *src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b16(ub_addr8_t dst, __ubuf__ uint16_t *src, uint64_t config);
void scatter_vnchwconv_b16(ub_addr8_t dst, __ubuf__ uint16_t *src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b16(__ubuf__ int16_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vnchwconv_b16(__ubuf__ int16_t *dst, ub_addr8_t src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b16(__ubuf__ uint16_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vnchwconv_b16(__ubuf__ uint16_t *dst, ub_addr8_t src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vnchwconv_b16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride);
void scatter_vnchwconv_b32(ub_addr8_t dst, __ubuf__ int32_t *src, uint64_t config);
void scatter_vnchwconv_b32(ub_addr8_t dst, __ubuf__ int32_t *src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b32(ub_addr8_t dst, __ubuf__ uint32_t *src, uint64_t config);
void scatter_vnchwconv_b32(ub_addr8_t dst, __ubuf__ uint32_t *src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b32(__ubuf__ int32_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vnchwconv_b32(__ubuf__ int32_t *dst, ub_addr8_t src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b32(__ubuf__ uint32_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vnchwconv_b32(__ubuf__ uint32_t *dst, ub_addr8_t src, uint8_t repeat,
                           uint16_t dstStride, uint16_t srcStride);
void scatter_vnchwconv_b32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vnchwconv_b32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                           uint16_t srcStride);
void scatter_vnchwconv_b8(ub_addr8_t dst, __ubuf__ int8_t *src, uint64_t config, bool dstHighHalf,
                          bool srcHighHalf);
void scatter_vnchwconv_b8(ub_addr8_t dst, __ubuf__ int8_t *src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool dstHighHalf, bool srcHighHalf);
void scatter_vnchwconv_b8(ub_addr8_t dst, __ubuf__ uint8_t *src, uint64_t config, bool dstHighHalf,
                          bool srcHighHalf);
void scatter_vnchwconv_b8(ub_addr8_t dst, __ubuf__ uint8_t *src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool dstHighHalf, bool srcHighHalf);
void scatter_vnchwconv_b8(__ubuf__ int8_t *dst, ub_addr8_t src, uint64_t config, bool dstHighHalf,
                          bool srcHighHalf);
void scatter_vnchwconv_b8(__ubuf__ int8_t *dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool dstHighHalf, bool srcHighHalf);
void scatter_vnchwconv_b8(__ubuf__ uint8_t *dst, ub_addr8_t src, uint64_t config, bool dstHighHalf,
                          bool srcHighHalf);
void scatter_vnchwconv_b8(__ubuf__ uint8_t *dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool dstHighHalf, bool srcHighHalf);
void scatter_vnchwconv_b8(ub_addr8_t dst, ub_addr8_t src, uint64_t config, bool dstHighHalf,
                          bool srcHighHalf);
void scatter_vnchwconv_b8(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                          uint16_t srcStride, bool dstHighHalf, bool srcHighHalf);
void scatter_vrec(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrec(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                  uint16_t srcStride);
void scatter_vrec_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrec_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                      uint16_t srcStride);
void scatter_vrec_f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrec_f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                      uint16_t srcStride);
void scatter_vrelu(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrelu(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                   uint16_t srcStride);
void scatter_vrelu_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrelu_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vrelu_f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrelu_f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vrelu_s32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrelu_s32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vrsqrt(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrsqrt(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat,
                    uint16_t dstStride, uint16_t srcStride);
void scatter_vrsqrt_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrsqrt_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                        uint16_t srcStride);
void scatter_vrsqrt_f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vrsqrt_f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                        uint16_t srcStride);
void scatter_vscmerge(__ubuf__ uint16_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vscmerge(__ubuf__ uint16_t *dst, ub_addr8_t src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscmerge(__ubuf__ uint8_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vscmerge(__ubuf__ uint8_t *dst, ub_addr8_t src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscmerge(__ubuf__ half *dst, ub_addr8_t src, uint64_t config);
void scatter_vscmerge(__ubuf__ half *dst, ub_addr8_t src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscmerge_b16(__ubuf__ uint16_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vscmerge_b16(__ubuf__ uint16_t *dst, ub_addr8_t src, uint16_t matDimM,
                          uint8_t matDimN);
void scatter_vscmerge_b8(__ubuf__ uint8_t *dst, ub_addr8_t src, uint64_t config);
void scatter_vscmerge_b8(__ubuf__ uint8_t *dst, ub_addr8_t src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscmerge_f16(__ubuf__ half *dst, ub_addr8_t src, uint64_t config);
void scatter_vscmerge_f16(__ubuf__ half *dst, ub_addr8_t src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscsplit(ub_addr8_t dst, __ubuf__ uint16_t *src, uint64_t config);
void scatter_vscsplit(ub_addr8_t dst, __ubuf__ uint16_t *src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscsplit(ub_addr8_t dst, __ubuf__ uint8_t *src, uint64_t config);
void scatter_vscsplit(ub_addr8_t dst, __ubuf__ uint8_t *src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscsplit(ub_addr8_t dst, __ubuf__ half *src, uint64_t config);
void scatter_vscsplit(ub_addr8_t dst, __ubuf__ half *src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscsplit_b16(ub_addr8_t dst, __ubuf__ uint16_t *src, uint64_t config);
void scatter_vscsplit_b16(ub_addr8_t dst, __ubuf__ uint16_t *src, uint16_t matDimM,
                          uint8_t matDimN);
void scatter_vscsplit_b16(ub_addr8_t dst, __ubuf__ int16_t *src, uint64_t config);
void scatter_vscsplit_b16(ub_addr8_t dst, __ubuf__ int16_t *src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscsplit_b8(ub_addr8_t dst, __ubuf__ uint8_t *src, uint64_t config);
void scatter_vscsplit_b8(ub_addr8_t dst, __ubuf__ uint8_t *src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscsplit_b8(ub_addr8_t dst, __ubuf__ int8_t *src, uint64_t config);
void scatter_vscsplit_b8(ub_addr8_t dst, __ubuf__ int8_t *src, uint16_t matDimM, uint8_t matDimN);
void scatter_vscsplit_f16(ub_addr8_t dst, __ubuf__ half *src, uint64_t config);
void scatter_vscsplit_f16(ub_addr8_t dst, __ubuf__ half *src, uint16_t matDimM, uint8_t matDimN);
void scatter_vsel(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsel(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vsel_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsel_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride,
                      uint8_t CMPMASKMode);
void scatter_vsel_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsel_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride,
                      uint8_t CMPMASKMode);
void scatter_vsqrt(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vsqrt(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                   uint16_t srcStride);
void scatter_vsqrt_f16(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vsqrt_f16(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vsqrt_f32(ub_addr8_t dst, ub_addr8_t src, uint64_t config);
void scatter_vsqrt_f32(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dstStride,
                       uint16_t srcStride);
void scatter_vsub(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsub(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                  uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vsub_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsub_f16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vsub_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsub_f32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vsub_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsub_s16(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void scatter_vsub_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void scatter_vsub_s32(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat,
                      uint16_t dstStride, uint16_t src0Stride, uint16_t src1Stride);
void set_aipp_spr_0(uint64_t config);
void set_aipp_spr_1(uint64_t config);
void set_aipp_spr_10(uint64_t config);
void set_aipp_spr_11(uint64_t config);
void set_aipp_spr_12(uint64_t config);
void set_aipp_spr_13(uint64_t config);
void set_aipp_spr_14(uint64_t config);
void set_aipp_spr_15(uint64_t config);
void set_aipp_spr_16(uint64_t config);
void set_aipp_spr_17(uint64_t config);
void set_aipp_spr_18(uint64_t config);
void set_aipp_spr_19(uint64_t config);
void set_aipp_spr_2(uint64_t config);
void set_aipp_spr_20(uint64_t config);
void set_aipp_spr_21(uint64_t config);
void set_aipp_spr_22(uint64_t config);
void set_aipp_spr_23(uint64_t config);
void set_aipp_spr_24(uint64_t config);
void set_aipp_spr_25(uint64_t config);
void set_aipp_spr_26(uint64_t config);
void set_aipp_spr_27(uint64_t config);
void set_aipp_spr_28(uint64_t config);
void set_aipp_spr_29(uint64_t config);
void set_aipp_spr_3(uint64_t config);
void set_aipp_spr_30(uint64_t config);
void set_aipp_spr_31(uint64_t config);
void set_aipp_spr_4(uint64_t config);
void set_aipp_spr_5(uint64_t config);
void set_aipp_spr_6(uint64_t config);
void set_aipp_spr_7(uint64_t config);
void set_aipp_spr_8(uint64_t config);
void set_aipp_spr_9(uint64_t config);
void set_atomic_add();
void set_atomic_bf16();
void set_atomic_f16();
void set_atomic_f32();
void set_atomic_max();
void set_atomic_min();
void set_atomic_none();
void set_atomic_s16();
void set_atomic_s32();
void set_atomic_s8();
uint32_t atomicAdd(__gm__ uint32_t *address, uint32_t value);
int32_t atomicAdd(__gm__ int32_t *address, int32_t value);
uint64_t atomicAdd(__gm__ uint64_t *address, uint64_t value);
int64_t atomicAdd(__gm__ int64_t *address, int64_t value);
float atomicAdd(__gm__ float *address, float value);
uint32_t atomicMax(__gm__ uint32_t *address, uint32_t value);
int32_t atomicMax(__gm__ int32_t *address, int32_t value);
uint64_t atomicMax(__gm__ uint64_t *address, uint64_t value);
int64_t atomicMax(__gm__ int64_t *address, int64_t value);
float atomicMax(__gm__ float *address, float value);
uint32_t atomicMin(__gm__ uint32_t *address, uint32_t value);
int32_t atomicMin(__gm__ int32_t *address, int32_t value);
uint64_t atomicMin(__gm__ uint64_t *address, uint64_t value);
int64_t atomicMin(__gm__ int64_t *address, int64_t value);
float atomicMin(__gm__ float *address, float value);
uint64_t atomicCAS(__gm__ uint64_t *address, uint64_t value1, uint64_t value2);
uint32_t atomicCAS(__gm__ uint32_t *address, uint32_t value1, uint32_t value2);
uint64_t atomicExch(__gm__ uint64_t *address, uint64_t value);
uint32_t atomicExch(__gm__ uint32_t *address, uint32_t value);
void set_cache_normread();
void set_cache_notwriteback();
void set_cache_readinv();
void set_cache_readlast();
void set_channel_para(uint64_t config);
void set_cmpmask(__ubuf__ void *src);
void set_cond(uint64_t config);
void set_condition_flag(uint64_t config);
void set_ctrl(uint64_t config);
void set_data_exp_0(uint64_t config);
void set_data_exp_1(uint64_t config);
void set_data_exp_2(uint64_t config);
void set_data_exp_3(uint64_t config);
void set_deqscale(half config);
void set_deqscale(uint64_t config);
void set_elt_antiq_para(uint64_t config);
void set_elt_src_para(uint64_t config);
void set_mte2_antiq_para(uint64_t config);
void set_fcol2img(uint64_t config);
void set_ffts_base_addr(uint64_t config);
void set_fix_clip_relu(uint64_t config);
void set_fixp_addr(uint64_t config);
void set_flag(pipe_t pipe, pipe_t tpipe, event_t pipeID);
void set_flag(pipe_t tpipe, event_t pipeID);
void set_flag(pipe_t pipe, pipe_t tpipe, uint64_t pipeID);
void set_flag(pipe_t tpipe, uint64_t pipeID);
void set_fmatrix(uint64_t config);
void set_fmatrix_b(uint64_t config);
void set_fmatrix_dual_0(uint64_t config);
void set_fmatrix_dual_1(uint64_t config);
void set_fpc(uint64_t config);
void set_fpdeq(uint64_t config);
void set_l0_set_value(half config);
void set_l0_set_value(uint32_t config);
void set_l0a_2d(__ca__ void *dst, int64_t config);
void set_l0a_2d(__ca__ half *dst, int64_t config);
void set_l0a_2d(__ca__ float *dst, int64_t config);
void set_l0a_2d(__ca__ int16_t *dst, int64_t config);
void set_l0a_2d(__ca__ int32_t *dst, int64_t config);
void set_l0a_2d(__ca__ uint16_t *dst, int64_t config);
void set_l0a_2d(__ca__ uint32_t *dst, int64_t config);
void set_l0b_2d(__cb__ void *dst, int64_t config);
void set_l0b_2d(__cb__ half *dst, int64_t config);
void set_l0b_2d(__cb__ float *dst, int64_t config);
void set_l0b_2d(__cb__ int16_t *dst, int64_t config);
void set_l0b_2d(__cb__ int32_t *dst, int64_t config);
void set_l0b_2d(__cb__ uint16_t *dst, int64_t config);
void set_l0b_2d(__cb__ uint32_t *dst, int64_t config);
void set_l1_2d(__cbuf__ void *dst, int64_t config);
void set_l1_2d(__cbuf__ half *dst, int64_t config);
void set_l1_2d(__cbuf__ float *dst, int64_t config);
void set_l1_2d(__cbuf__ int16_t *dst, int64_t config);
void set_l1_2d(__cbuf__ int32_t *dst, int64_t config);
void set_l1_2d(__cbuf__ uint16_t *dst, int64_t config);
void set_l1_2d(__cbuf__ uint32_t *dst, int64_t config);
void set_l1_3d_size(uint64_t config);
void set_l3d_rpt(uint64_t config);
void set_loop3_para(uint64_t config);
void set_low_pre_tbl(uint64_t config);
void set_lpcnt(uint64_t config);
void set_lrelu_alpha(uint64_t config);
void set_lrelu_alpha(half config);
void set_lrelu_alpha(float config);
void set_mask_count();
void set_mask_norm();
void set_mov_pad_val(uint64_t config);
void set_pad_val_outtoub(uint64_t config);
void set_pad_val_outtol1(uint64_t config);
void set_nd_para(uint64_t config);
void set_padding(uint64_t config);
void set_padding(half config);
void set_padding(int16_t config);
void set_padding(uint16_t config);
void set_pcie_rd_ctrl(uint64_t config);
void set_pcie_wr_ctrl(uint64_t config);
void set_pnt_coe(uint64_t config);
void set_quant_post(uint64_t config);
void set_quant_pre(uint64_t config);
void set_rawheader_to_gm(__gm__ void *dst, uint64_t config);
void set_rawheader_to_gm(__gm__ void *dst, uint16_t nHeader, bool N, uint8_t SID);
void set_relu_alpha(uint64_t config);
void set_reqscale(uint64_t config);
void set_rpn_cor_ir(uint64_t config);
void set_rpn_offset(half config);
void set_safety_crc_en(uint64_t config);
void set_safety_crc_excp(uint64_t config);
void set_smask_index(uint64_t config);
void set_st_atomic_cfg(uint64_t config);
void set_st_atomic_cfg(atomic_type_t type, atomic_op_t op);
void set_va_reg_sb(ub_addr8_t addr, uint64_t *array);
void set_vector_mask(uint64_t mask1, uint64_t mask0);
void set_vector_mask_dup(uint64_t mask);
void set_vpipe(uint64_t config);
void set_vsp(uint64_t config);
int64_t sff0(uint64_t in);
int64_t sff1(uint64_t in);
int64_t sflbits(int64_t in);
void st_dev(uint16_t src, __gm__ uint16_t *dst, int16_t offset);
void st_dev(uint32_t src, __gm__ uint32_t *dst, int16_t offset);
void st_dev(uint64_t src, __gm__ uint64_t *dst, int16_t offset);
void st_dev(uint8_t src, __gm__ uint8_t *dst, int16_t offset);
void store_l1_to_out_image(__gm__ half *dst, __cbuf__ half *src, uint64_t config, uint64_t para);
void store_l1_to_out_image(__gm__ half *dst, __cbuf__ half *src, uint16_t horizontalSize,
                           uint8_t outputFormat, uint16_t verticalSize, uint8_t channelTrans,
                           uint16_t desImagestride, uint8_t smmu_tlb_hint_sid,
                           bool configuration_S16_to_u12_u14, uint8_t rightShiftConfiguration,
                           uint64_t normarlization);
void store_l1_to_out_image(__gm__ int16_t *dst, __cbuf__ int16_t *src, uint64_t config,
                           uint64_t para);
void store_l1_to_out_image(__gm__ int16_t *dst, __cbuf__ int16_t *src, uint16_t horizontalSize,
                           uint8_t outputFormat, uint16_t verticalSize, uint8_t channelTrans,
                           uint16_t desImagestride, uint8_t smmu_tlb_hint_sid,
                           bool configuration_S16_to_u12_u14, uint8_t rightShiftConfiguration,
                           uint64_t normarlization);
void trap(uint64_t err_code);
vector_bool plt_b8(uint32_t &scalar, Literal post);
vector_bool plt_b16(uint32_t &scalar, Literal post);
vector_bool plt_b32(uint32_t &scalar, Literal post);
vector_bool plt_2xvl_b64(uint32_t &scalar, Literal post);
vector_bool pltm_b8(uint16_t scalar0, uint32_t scalar1);
vector_bool pltm_b16(uint16_t scalar0, uint32_t scalar1);
vector_bool pltm_b32(uint16_t scalar0, uint32_t scalar1);
void pnot(vector_bool &dst, vector_bool src, vector_bool mask);
void pand(vector_bool &dst, vector_bool src0, vector_bool src1, vector_bool mask);
void por(vector_bool &dst, vector_bool src0, vector_bool src1, vector_bool mask);
void pxor(vector_bool &dst, vector_bool src0, vector_bool src1, vector_bool mask);
void pmov(vector_bool &dst, vector_bool src);
void pslide_b8(vector_bool &dst, vector_bool src0, vector_bool src1, int16_t slideAmount);
void pslide_b16(vector_bool &dst, vector_bool src0, vector_bool src1, int16_t slideAmount);
void pslide_b32(vector_bool &dst, vector_bool src0, vector_bool src1, int16_t slideAmount);
void psel(vector_bool &dst, vector_bool src0, vector_bool src1, vector_bool mask);
void vpack(vector_u8 &dst, vector_u16 &src, Literal part);
void vpack(vector_u8 &dst, vector_s16 &src, Literal part);
void vpack(vector_u16 &dst, vector_u32 &src, Literal part);
void vpack(vector_u16 &dst, vector_s32 &src, Literal part);
void vpack(vector_u8 &dst, vector_u16 &src, int32_t part);
void vpack(vector_u8 &dst, vector_s16 &src, int32_t part);
void vpack(vector_u16 &dst, vector_u32 &src, int32_t part);
void vpack(vector_u16 &dst, vector_s32 &src, int32_t part);
void vunpack(vector_u16 &dst, vector_u8 &src, Literal part);
void vunpack(vector_s16 &dst, vector_s8 &src, Literal part);
void vunpack(vector_u32 &dst, vector_u16 &src, Literal part);
void vunpack(vector_s32 &dst, vector_s16 &src, Literal part);
void vunpack(vector_u16 &dst, vector_u8 &src, int32_t part);
void vunpack(vector_s16 &dst, vector_s8 &src, int32_t part);
void vunpack(vector_u32 &dst, vector_u16 &src, int32_t part);
void vunpack(vector_s32 &dst, vector_s16 &src, int32_t part);
void ppack(vector_bool &dst, vector_bool src, int32_t part);
void punpack(vector_bool &dst, vector_bool src, int32_t part);
vector_address vag_b32(uint32_t s0);
vector_address vag_b32(uint32_t s0, uint32_t s1);
vector_address vag_b32(uint32_t s0, uint32_t s1, uint32_t s2);
vector_address vag_b32(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3);
vector_address vag_b16(uint32_t s0);
vector_address vag_b16(uint32_t s0, uint32_t s1);
vector_address vag_b16(uint32_t s0, uint32_t s1, uint32_t s2);
vector_address vag_b16(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3);
vector_address vag_b8(uint32_t s0);
vector_address vag_b8(uint32_t s0, uint32_t s1);
vector_address vag_b8(uint32_t s0, uint32_t s1, uint32_t s2);
vector_address vag_b8(uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3);
void trap();
void mem_bar(Literal mem_type);
void vaddc(vector_bool &mask0, vector_u32 &dst, vector_u32 src0, vector_u32 src1,
           vector_bool mask1);
void vaddc(vector_bool &mask0, vector_s32 &dst, vector_s32 src0, vector_s32 src1,
           vector_bool mask1);
void vsubc(vector_bool &mask0, vector_u32 &dst, vector_u32 src0, vector_u32 src1,
           vector_bool mask1);
void vsubcs(vector_bool &mask0, vector_u32 &dst, vector_u32 src0, vector_u32 src1,
            vector_bool mask1, vector_bool mask2);
void vsubc(vector_bool &mask0, vector_s32 &dst, vector_s32 src0, vector_s32 src1,
           vector_bool mask1);
void vsubcs(vector_bool &mask0, vector_s32 &dst, vector_s32 src0, vector_s32 src1,
            vector_bool mask1, vector_bool mask2);
void vaddcs(vector_bool &mask0, vector_u32 &dst, vector_u32 src0, vector_u32 src1,
            vector_bool mask1, vector_bool mask2);
void vaddcs(vector_bool &mask0, vector_s32 &dst, vector_s32 src0, vector_s32 src1,
            vector_bool mask1, vector_bool mask2);
void vmull(vector_u32 &dst0, vector_u32 &dst1, vector_u32 src0, vector_u32 src1, vector_bool mask0);
void vmull(vector_s32 &dst0, vector_s32 &dst1, vector_s32 src0, vector_s32 src1, vector_bool mask0);
void vxor(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask0);
void vxor(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vxor(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask0);
void vxor(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask0, int32_t mode);
void vxor(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask0);
void vxor(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask0, int32_t mode);
void vxor(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask0);
void vxor(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vxor(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask0);
void vxor(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask0, int32_t mode);
void vxor(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask0);
void vxor(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask0, int32_t mode);
void vprelu(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask0, int32_t mode);
void vprelu(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask0, int32_t mode);
void vgather2(vector_s16 &dst, __ubuf__ int8_t *src0, vector_u16 src1, vector_bool mask0);
void vgather2(vector_u16 &dst, __ubuf__ uint8_t *src0, vector_u16 src1, vector_bool mask0);
void vgather2(vector_s16 &dst, __ubuf__ int16_t *src0, vector_u16 src1, vector_bool mask0);
void vgather2(vector_u16 &dst, __ubuf__ uint16_t *src0, vector_u16 src1, vector_bool mask0);
void vgather2(vector_s32 &dst, __ubuf__ int32_t *src0, vector_u32 src1, vector_bool mask0);
void vgather2(vector_u32 &dst, __ubuf__ uint32_t *src0, vector_u32 src1, vector_bool mask0);
void vgather2(vector_f16 &dst, __ubuf__ half *src0, vector_u16 src1, vector_bool mask0);
void vgather2(vector_f32 &dst, __ubuf__ float *src0, vector_u32 src1, vector_bool mask0);
void vgather2(vector_bf16 &dst, __ubuf__ bfloat16_t *src0, vector_u16 src1, vector_bool mask0);
void vgatherb(vector_s8 &dst, __ubuf__ int8_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_u8 &dst, __ubuf__ uint8_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_s16 &dst, __ubuf__ int16_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_u16 &dst, __ubuf__ uint16_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_s32 &dst, __ubuf__ int32_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_u32 &dst, __ubuf__ uint32_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_bf16 &dst, __ubuf__ bfloat16_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_f16 &dst, __ubuf__ half *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_f32 &dst, __ubuf__ float *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_s64 &dst, __ubuf__ int64_t *base, vector_u32 indexOffset, vector_bool pg);
void vgatherb(vector_s8 &dst, __ubuf__ int8_t *base, vector_u32 indexOffset);
void vgatherb(vector_u8 &dst, __ubuf__ uint8_t *base, vector_u32 indexOffset);
void vgatherb(vector_s16 &dst, __ubuf__ int16_t *base, vector_u32 indexOffset);
void vgatherb(vector_u16 &dst, __ubuf__ uint16_t *base, vector_u32 indexOffset);
void vgatherb(vector_s32 &dst, __ubuf__ int32_t *base, vector_u32 indexOffset);
void vgatherb(vector_u32 &dst, __ubuf__ uint32_t *base, vector_u32 indexOffset);
void vgatherb(vector_bf16 &dst, __ubuf__ bfloat16_t *base, vector_u32 indexOffset);
void vgatherb(vector_f16 &dst, __ubuf__ half *base, vector_u32 indexOffset);
void vgatherb(vector_f32 &dst, __ubuf__ float *base, vector_u32 indexOffset);
void vgatherb(vector_s64 &dst, __ubuf__ int64_t *base, vector_u32 indexOffset);
void vgatherb(vector_u64 &dst, __ubuf__ uint64_t *base, vector_u32 indexOffset, vector_bool pg);
void vscatter(vector_s8 &data, __ubuf__ int8_t *base, vector_u16 index, vector_bool mask);
void vscatter(vector_u8 &data, __ubuf__ uint8_t *base, vector_u16 index, vector_bool mask);
void vscatter(vector_s16 &data, __ubuf__ int16_t *base, vector_u16 index, vector_bool mask);
void vscatter(vector_u16 &data, __ubuf__ uint16_t *base, vector_u16 index, vector_bool mask);
void vscatter(vector_s32 &data, __ubuf__ int32_t *base, vector_u32 index, vector_bool mask);
void vscatter(vector_u32 &data, __ubuf__ uint32_t *base, vector_u32 index, vector_bool mask);
void vscatter(vector_bf16 &data, __ubuf__ bfloat16_t *base, vector_u16 index, vector_bool mask);
void vscatter(vector_f16 &data, __ubuf__ half *base, vector_u16 index, vector_bool mask);
void vscatter(vector_f32 &data, __ubuf__ float *base, vector_u32 index, vector_bool mask);
void vadd(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vadd(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vadd(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vadd(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vadd(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vadd(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vadd(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vadd(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vadd(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, Literal mode);
void vadd(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vadd(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vadd(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vadd(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vadd(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vadd(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vadd(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vadd(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vadd(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vadd(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
          int32_t mode);
void vadd(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
          int32_t mode);
void vci(vector_s32 &dst, int32_t offset, int32_t mode);
void vci(vector_s16 &dst, int16_t offset, int32_t mode);
void vci(vector_s8 &dst, int8_t offset, int32_t mode);
void vci(vector_f16 &dst, half offset, int32_t mode);
void vci(vector_f32 &dst, float offset, int32_t mode);
void vci(vector_s32 &dst, int32_t offset, Literal order);
void vci(vector_s16 &dst, int16_t offset, Literal order);
void vci(vector_s8 &dst, int8_t offset, Literal order);
void vci(vector_f16 &dst, half offset, Literal order);
void vci(vector_f32 &dst, float offset, Literal order);
void vci(vector_2xvl_s64 &dst, int32_t offset, int32_t mode);
void vci(vector_2xvl_s64 &dst, int32_t offset);
void vabs(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vabs(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vabs(vector_s8 &dst, vector_s8 src, vector_bool mask, Literal mode);
void vabs(vector_s16 &dst, vector_s16 src, vector_bool mask, Literal mode);
void vabs(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal mode);
void vabs(vector_2xvl_s64 &dst, vector_2xvl_s64 src, vector_bool mask, Literal mode);
void vabs(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vabs(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vabs(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t mode);
void vabs(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t mode);
void vabs(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vabs(vector_2xvl_s64 &dst, vector_2xvl_s64 src, vector_bool mask, int32_t mode);
void vrec(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vrec(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vrec(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vrec(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vrsqrt(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vrsqrt(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vrelu(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vrelu(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vrelu(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal mode);
void vrelu(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vrelu(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vrelu(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vsqrt(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vsqrt(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vsqrt(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vsqrt(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vln(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vln(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vln(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vln(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vneg(vector_s8 &dst, vector_s8 src, vector_bool mask, Literal mode);
void vneg(vector_s16 &dst, vector_s16 src, vector_bool mask, Literal mode);
void vneg(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal mode);
void vneg(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vneg(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vneg(vector_2xvl_u64 &dst, vector_2xvl_u64 src, vector_bool mask, Literal mode);
void vneg(vector_2xvl_s64 &dst, vector_2xvl_s64 src, vector_bool mask, Literal mode);
void vneg(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t mode);
void vneg(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t mode);
void vneg(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vneg(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vneg(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vneg(vector_2xvl_u64 &dst, vector_2xvl_u64 src, vector_bool mask, int32_t mode);
void vneg(vector_2xvl_s64 &dst, vector_2xvl_s64 src, vector_bool mask, int32_t mode);
void vexp(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vexp(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vexp(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vexp(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vnot(vector_u8 &dst, vector_u8 src, vector_bool mask, Literal mode);
void vnot(vector_s8 &dst, vector_s8 src, vector_bool mask, Literal mode);
void vnot(vector_u16 &dst, vector_u16 src, vector_bool mask, Literal mode);
void vnot(vector_s16 &dst, vector_s16 src, vector_bool mask, Literal mode);
void vnot(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vnot(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal mode);
void vnot(vector_u32 &dst, vector_u32 src, vector_bool mask, Literal mode);
void vnot(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vnot(vector_2xvl_u64 &dst, vector_2xvl_u64 src, vector_bool mask, Literal mode);
void vnot(vector_2xvl_s64 &dst, vector_2xvl_s64 src, vector_bool mask, Literal mode);
void vnot(vector_u8 &dst, vector_u8 src, vector_bool mask, int32_t mode);
void vnot(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t mode);
void vnot(vector_u16 &dst, vector_u16 src, vector_bool mask, int32_t mode);
void vnot(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t mode);
void vnot(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vnot(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vnot(vector_u32 &dst, vector_u32 src, vector_bool mask, int32_t mode);
void vnot(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vnot(vector_2xvl_u64 &dst, vector_2xvl_u64 src, vector_bool mask, int32_t mode);
void vnot(vector_2xvl_s64 &dst, vector_2xvl_s64 src, vector_bool mask, int32_t mode);
void plds(vector_bool &dst, __ubuf__ uint32_t *base, int32_t offset, Literal dist);
void plds(vector_bool &dst, __ubuf__ uint32_t *&base, int32_t offset, Literal dist, Literal post);
void plds(vector_bool &dst, __ubuf__ uint32_t *base, int32_t offset, int32_t dist);
void plds(vector_bool &dst, __ubuf__ uint32_t *&base, int32_t offset, int32_t dist, int32_t post);
vector_bool movp_b16();
vector_bool movp_b32();
void psts(vector_bool src, __ubuf__ uint32_t *base, int32_t offset, Literal dist);
void psts(vector_bool src, __ubuf__ uint32_t *base, int32_t offset, int32_t dist);
void psts(vector_bool src, __ubuf__ uint32_t *&base, int32_t offset, int32_t dist, int32_t post);
void ppack(vector_bool &dst, vector_bool src, Literal part);
void punpack(vector_bool &dst, vector_bool src, Literal part);
void vintlv(vector_u8 &dst0, vector_u8 &dst1, vector_u8 src0, vector_u8 src1);
void vintlv(vector_s8 &dst0, vector_s8 &dst1, vector_s8 src0, vector_s8 src1);
void vintlv(vector_f16 &dst0, vector_f16 &dst1, vector_f16 src0, vector_f16 src1);
void vintlv(vector_s16 &dst0, vector_s16 &dst1, vector_s16 src0, vector_s16 src1);
void vintlv(vector_u16 &dst0, vector_u16 &dst1, vector_u16 src0, vector_u16 src1);
void vintlv(vector_f32 &dst0, vector_f32 &dst1, vector_f32 src0, vector_f32 src1);
void vintlv(vector_s32 &dst0, vector_s32 &dst1, vector_s32 src0, vector_s32 src1);
void vintlv(vector_u32 &dst0, vector_u32 &dst1, vector_u32 src0, vector_u32 src1);
void vintlv(vector_bf16 &dst0, vector_bf16 &dst1, vector_bf16 src0, vector_bf16 src1);
void vdintlv(vector_u8 &dst0, vector_u8 &dst1, vector_u8 src0, vector_u8 src1);
void vdintlv(vector_s8 &dst0, vector_s8 &dst1, vector_s8 src0, vector_s8 src1);
void vdintlv(vector_f16 &dst0, vector_f16 &dst1, vector_f16 src0, vector_f16 src1);
void vdintlv(vector_s16 &dst0, vector_s16 &dst1, vector_s16 src0, vector_s16 src1);
void vdintlv(vector_u16 &dst0, vector_u16 &dst1, vector_u16 src0, vector_u16 src1);
void vdintlv(vector_f32 &dst0, vector_f32 &dst1, vector_f32 src0, vector_f32 src1);
void vdintlv(vector_s32 &dst0, vector_s32 &dst1, vector_s32 src0, vector_s32 src1);
void vdintlv(vector_u32 &dst0, vector_u32 &dst1, vector_u32 src0, vector_u32 src1);
void vpack(vector_u8 &dst, vector_u16 src, Literal part, Literal mode);
void vpack(vector_u8 &dst, vector_s16 src, Literal part, Literal mode);
void vpack(vector_u16 &dst, vector_u32 src, Literal part, Literal mode);
void vpack(vector_u16 &dst, vector_s32 src, Literal part, Literal mode);
void pintlv_b8(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);
void pintlv_b16(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);
void pintlv_b32(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);
void pdintlv_b8(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);
void pdintlv_b16(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);
void pdintlv_b32(vector_bool &dst0, vector_bool &dst1, vector_bool src0, vector_bool src1);
void vsstb(vector_s8 data, __ubuf__ int8_t *base, int32_t offset, vector_bool mask);
void vsstb(vector_u8 data, __ubuf__ uint8_t *base, int32_t offset, vector_bool mask);
void vsstb(vector_s16 data, __ubuf__ int16_t *base, int32_t offset, vector_bool mask);
void vsstb(vector_u16 data, __ubuf__ uint16_t *base, int32_t offset, vector_bool mask);
void vsstb(vector_s32 data, __ubuf__ int32_t *base, int32_t offset, vector_bool mask);
void vsstb(vector_u32 data, __ubuf__ uint32_t *base, int32_t offset, vector_bool mask);
void vsstb(vector_bf16 data, __ubuf__ bfloat16_t *base, int32_t offset, vector_bool mask);
void vsstb(vector_f16 data, __ubuf__ half *base, int32_t offset, vector_bool mask);
void vsstb(vector_f32 data, __ubuf__ float *base, int32_t offset, vector_bool mask);
void vsstb(vector_s8 dst, __ubuf__ int8_t *&base, int32_t offset, vector_bool mask, int32_t post);
void vsstb(vector_u8 dst, __ubuf__ uint8_t *&base, int32_t offset, vector_bool mask, int32_t post);
void vsstb(vector_s16 dst, __ubuf__ int16_t *&base, int32_t offset, vector_bool mask, int32_t post);
void vsstb(vector_u16 dst, __ubuf__ uint16_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsstb(vector_s32 dst, __ubuf__ int32_t *&base, int32_t offset, vector_bool mask, int32_t post);
void vsstb(vector_u32 dst, __ubuf__ uint32_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsstb(vector_bf16 dst, __ubuf__ bfloat16_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsstb(vector_f16 dst, __ubuf__ half *&base, int32_t offset, vector_bool mask, int32_t post);
void vsstb(vector_f32 dst, __ubuf__ float *&base, int32_t offset, vector_bool mask, int32_t post);
void vsstb(vector_s8 dst, __ubuf__ int8_t *&base, int32_t offset, vector_bool mask, Literal post);
void vsstb(vector_u8 dst, __ubuf__ uint8_t *&base, int32_t offset, vector_bool mask, Literal post);
void vsstb(vector_s16 dst, __ubuf__ int16_t *&base, int32_t offset, vector_bool mask, Literal post);
void vsstb(vector_u16 dst, __ubuf__ uint16_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsstb(vector_s32 dst, __ubuf__ int32_t *&base, int32_t offset, vector_bool mask, Literal post);
void vsstb(vector_u32 dst, __ubuf__ uint32_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsstb(vector_bf16 dst, __ubuf__ bfloat16_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsstb(vector_f16 dst, __ubuf__ half *&base, int32_t offset, vector_bool mask, Literal post);
void vsstb(vector_f32 dst, __ubuf__ float *&base, int32_t offset, vector_bool mask, Literal post);
void vsldb(vector_s8 &dst, __ubuf__ int8_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_u8 &dst, __ubuf__ uint8_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_s16 &dst, __ubuf__ int16_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_u16 &dst, __ubuf__ uint16_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_s32 &dst, __ubuf__ int32_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_u32 &dst, __ubuf__ uint32_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_s64 &dst, __ubuf__ int64_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_bf16 &dst, __ubuf__ bfloat16_t *base, int32_t offset, vector_bool mask);
void vsldb(vector_f16 &dst, __ubuf__ half *base, int32_t offset, vector_bool mask);
void vsldb(vector_f32 &dst, __ubuf__ float *base, int32_t offset, vector_bool mask);
void vsldb(vector_s8 &dst, __ubuf__ int8_t *&base, int32_t offset, vector_bool mask, int32_t post);
void vsldb(vector_u8 &dst, __ubuf__ uint8_t *&base, int32_t offset, vector_bool mask, int32_t post);
void vsldb(vector_s16 &dst, __ubuf__ int16_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsldb(vector_u16 &dst, __ubuf__ uint16_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsldb(vector_s32 &dst, __ubuf__ int32_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsldb(vector_u32 &dst, __ubuf__ uint32_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsldb(vector_s64 &dst, __ubuf__ int64_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsldb(vector_bf16 &dst, __ubuf__ bfloat16_t *&base, int32_t offset, vector_bool mask,
           int32_t post);
void vsldb(vector_f16 &dst, __ubuf__ half *&base, int32_t offset, vector_bool mask, int32_t post);
void vsldb(vector_f32 &dst, __ubuf__ float *&base, int32_t offset, vector_bool mask, int32_t post);
void vsldb(vector_s8 &dst, __ubuf__ int8_t *&base, int32_t offset, vector_bool mask, Literal post);
void vsldb(vector_u8 &dst, __ubuf__ uint8_t *&base, int32_t offset, vector_bool mask, Literal post);
void vsldb(vector_s16 &dst, __ubuf__ int16_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsldb(vector_u16 &dst, __ubuf__ uint16_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsldb(vector_s32 &dst, __ubuf__ int32_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsldb(vector_u32 &dst, __ubuf__ uint32_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsldb(vector_s64 &dst, __ubuf__ int64_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsldb(vector_bf16 &dst, __ubuf__ bfloat16_t *&base, int32_t offset, vector_bool mask,
           Literal post);
void vsldb(vector_f16 &dst, __ubuf__ half *&base, int32_t offset, vector_bool mask, Literal post);
void vsldb(vector_f32 &dst, __ubuf__ float *&base, int32_t offset, vector_bool mask, Literal post);
void vst(vector_s8 data, __ubuf__ int8_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_u8 data, __ubuf__ uint8_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_s16 data, __ubuf__ int16_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_u16 data, __ubuf__ uint16_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_s32 data, __ubuf__ int32_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_u32 data, __ubuf__ uint32_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_u64 data, __ubuf__ uint64_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_s64 data, __ubuf__ int64_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_bf16 data, __ubuf__ bfloat16_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_f16 data, __ubuf__ half *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_f32 data, __ubuf__ float *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_s8 data, __ubuf__ int8_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_u8 data, __ubuf__ uint8_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_s16 data, __ubuf__ int16_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_u16 data, __ubuf__ uint16_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_s32 data, __ubuf__ int32_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_u32 data, __ubuf__ uint32_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_u64 data, __ubuf__ uint64_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_s64 data, __ubuf__ int64_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_bf16 data, __ubuf__ bfloat16_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_f16 data, __ubuf__ half *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_f32 data, __ubuf__ float *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_s8 src0, vector_s8 src1, __ubuf__ int8_t *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_u8 src0, vector_u8 src1, __ubuf__ uint8_t *base, vector_address offset,
         Literal dist, vector_bool mask);
void vst(vector_s16 src0, vector_s16 src1, __ubuf__ int16_t *base, vector_address offset,
         Literal dist, vector_bool mask);
void vst(vector_u16 src0, vector_u16 src1, __ubuf__ uint16_t *base, vector_address offset,
         Literal dist, vector_bool mask);
void vst(vector_bf16 src0, vector_bf16 src1, __ubuf__ bfloat16_t *base, vector_address offset,
         Literal dist, vector_bool mask);
void vst(vector_f16 src0, vector_f16 src1, __ubuf__ half *base, vector_address offset, Literal dist,
         vector_bool mask);
void vst(vector_s8 src0, vector_s8 src1, __ubuf__ int8_t *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_u8 src0, vector_u8 src1, __ubuf__ uint8_t *base, vector_address offset,
         int32_t dist, vector_bool mask);
void vst(vector_s16 src0, vector_s16 src1, __ubuf__ int16_t *base, vector_address offset,
         int32_t dist, vector_bool mask);
void vst(vector_u16 src0, vector_u16 src1, __ubuf__ uint16_t *base, vector_address offset,
         int32_t dist, vector_bool mask);
void vst(vector_bf16 src0, vector_bf16 src1, __ubuf__ bfloat16_t *base, vector_address offset,
         int32_t dist, vector_bool mask);
void vst(vector_f16 src0, vector_f16 src1, __ubuf__ half *base, vector_address offset, int32_t dist,
         vector_bool mask);
void vst(vector_s32 src0, vector_s32 src1, __ubuf__ int32_t *base, vector_address offset,
         int32_t dist, vector_bool mask);
void vst(vector_u32 src0, vector_u32 src1, __ubuf__ uint32_t *base, vector_address offset,
         int32_t dist, vector_bool mask);
void vst(vector_f32 src0, vector_f32 src1, __ubuf__ float *base, vector_address offset,
         int32_t dist, vector_bool mask);
void vsts(vector_s8 data, __ubuf__ int8_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_u8 data, __ubuf__ uint8_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_s16 data, __ubuf__ int16_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_u16 data, __ubuf__ uint16_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_s32 data, __ubuf__ int32_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_u32 data, __ubuf__ uint32_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_u64 data, __ubuf__ uint64_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_s64 data, __ubuf__ int64_t *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_bf16 data, __ubuf__ bfloat16_t *base, int32_t offset, Literal dist,
          vector_bool mask);
void vsts(vector_f16 data, __ubuf__ half *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_f32 data, __ubuf__ float *base, int32_t offset, Literal dist, vector_bool mask);
void vsts(vector_s8 data, __ubuf__ int8_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_u8 data, __ubuf__ uint8_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_s16 data, __ubuf__ int16_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_u16 data, __ubuf__ uint16_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_s32 data, __ubuf__ int32_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_u32 data, __ubuf__ uint32_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_bf16 data, __ubuf__ bfloat16_t *&base, int32_t offset, int32_t dist,
          vector_bool mask, int32_t post);
void vsts(vector_f16 data, __ubuf__ half *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_f32 data, __ubuf__ float *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_s64 data, __ubuf__ int64_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_u64 data, __ubuf__ uint64_t *&base, int32_t offset, int32_t dist, vector_bool mask,
          int32_t post);
void vsts(vector_s8 data, __ubuf__ int8_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_u8 data, __ubuf__ uint8_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_s16 data, __ubuf__ int16_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_u16 data, __ubuf__ uint16_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_s32 data, __ubuf__ int32_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_u32 data, __ubuf__ uint32_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_u64 data, __ubuf__ uint64_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_s64 data, __ubuf__ int64_t *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_bf16 data, __ubuf__ bfloat16_t *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vsts(vector_f16 data, __ubuf__ half *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_f32 data, __ubuf__ float *base, int32_t offset, int32_t dist, vector_bool mask);
void vsts(vector_2xvl_u64 data, __ubuf__ uint64_t *base, int32_t offset, vector_bool mask);
void vsts(vector_2xvl_s64 data, __ubuf__ int64_t *base, int32_t offset, vector_bool mask);
void vsts(vector_s8 src0, vector_s8 src1, __ubuf__ int8_t *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vsts(vector_u8 src0, vector_u8 src1, __ubuf__ uint8_t *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vsts(vector_s16 src0, vector_s16 src1, __ubuf__ int16_t *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vsts(vector_u16 src0, vector_u16 src1, __ubuf__ uint16_t *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vsts(vector_s32 src0, vector_s32 src1, __ubuf__ int32_t *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vsts(vector_u32 src0, vector_u32 src1, __ubuf__ uint32_t *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vsts(vector_bf16 src0, vector_bf16 src1, __ubuf__ bfloat16_t *base, int32_t offset,
          int32_t dist, vector_bool mask);
void vsts(vector_f16 src0, vector_f16 src1, __ubuf__ half *base, int32_t offset, int32_t dist,
          vector_bool mask);
void vstui(vector_align &alignData, uint32_t v1, vector_f16 src, __ubuf__ half *&base);
void vstui(vector_align &alignData, uint32_t offset, vector_f16 src, __ubuf__ half *&base,
           Literal post);
void vstui(vector_align &alignData, uint32_t v1, vector_f32 src, __ubuf__ float *&base);
void vstui(vector_align &alignData, uint32_t offset, vector_f32 src, __ubuf__ float *&base,
           Literal post);
void vstu(vector_align &alignData, vector_address &offset, vector_s8 src, __ubuf__ int8_t *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_u8 src, __ubuf__ uint8_t *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_u16 src, __ubuf__ uint16_t *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_s16 src, __ubuf__ int16_t *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_f16 src, __ubuf__ half *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_bf16 src,
          __ubuf__ bfloat16_t *base, int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_u32 src, __ubuf__ uint32_t *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_s32 src, __ubuf__ int32_t *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_f32 src, __ubuf__ float *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_u64 src, __ubuf__ uint64_t *base,
          int32_t post);
void vstu(vector_align &alignData, vector_address &offset, vector_s64 src, __ubuf__ int64_t *base,
          int32_t post);
void vstus(vector_align &alignData, uint32_t v1, vector_u8 src, __ubuf__ uint8_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_u8 src, __ubuf__ uint8_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_s8 src, __ubuf__ int8_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_s8 src, __ubuf__ int8_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_u16 src, __ubuf__ uint16_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_u16 src, __ubuf__ uint16_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_s16 src, __ubuf__ int16_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_s16 src, __ubuf__ int16_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_f16 src, __ubuf__ half *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_f16 src, __ubuf__ half *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_bf16 src, __ubuf__ bfloat16_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_bf16 src, __ubuf__ bfloat16_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_u32 src, __ubuf__ uint32_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_u32 src, __ubuf__ uint32_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_s32 src, __ubuf__ int32_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_s32 src, __ubuf__ int32_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_f32 src, __ubuf__ float *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_f32 src, __ubuf__ float *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_u64 src, __ubuf__ uint64_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_u64 src, __ubuf__ uint64_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t v1, vector_s64 src, __ubuf__ int64_t *&base);
void vstus(vector_align &alignData, uint32_t offset, vector_s64 src, __ubuf__ int64_t *&base,
           Literal post);
void vstus(vector_align &alignData, uint32_t offset, vector_u8 src, __ubuf__ uint8_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_s8 src, __ubuf__ int8_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_u16 src, __ubuf__ uint16_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_s16 src, __ubuf__ int16_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_f16 src, __ubuf__ half *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_bf16 src, __ubuf__ bfloat16_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_u32 src, __ubuf__ uint32_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_s32 src, __ubuf__ int32_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_f32 src, __ubuf__ float *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_u64 src, __ubuf__ uint64_t *&base,
           int32_t post);
void vstus(vector_align &alignData, uint32_t offset, vector_s64 src, __ubuf__ int64_t *&base,
           int32_t post);
void vstai(vector_align data, __ubuf__ half *base, int32_t offset);
void vstai(vector_align data, __ubuf__ half *&base, int32_t offset, Literal post);
void vstai(vector_align data, __ubuf__ float *base, int32_t offset);
void vstai(vector_align data, __ubuf__ float *&base, int32_t offset, Literal post);
void vsta(vector_align data, __ubuf__ uint8_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ int8_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ uint16_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ int16_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ half *base, vector_address offset);
void vsta(vector_align data, __ubuf__ bfloat16_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ uint32_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ int32_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ float *base, vector_address offset);
void vsta(vector_align data, __ubuf__ int64_t *base, vector_address offset);
void vsta(vector_align data, __ubuf__ uint64_t *base, vector_address offset);
void vstas(vector_align data, __ubuf__ uint8_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ uint8_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ int8_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ int8_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ uint16_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ uint16_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ int16_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ int16_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ half *base, int32_t offset);
void vstas(vector_align data, __ubuf__ half *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ bfloat16_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ bfloat16_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ uint32_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ uint32_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ int32_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ int32_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ float *base, int32_t offset);
void vstas(vector_align data, __ubuf__ float *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ uint64_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ uint64_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ int64_t *base, int32_t offset);
void vstas(vector_align data, __ubuf__ int64_t *&base, int32_t offset, Literal post);
void vstas(vector_align data, __ubuf__ uint8_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ int8_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ uint16_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ int16_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ half *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ bfloat16_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ uint32_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ int32_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ float *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ uint64_t *&base, int32_t offset, int32_t post);
void vstas(vector_align data, __ubuf__ int64_t *&base, int32_t offset, int32_t post);
void vor(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vor(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vor(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vor(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vor(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vor(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vor(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vor(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vor(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, Literal mode);
void vor(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
         int32_t mode);
void vor(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
         int32_t mode);
void vor(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vor(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vor(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vor(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vor(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vor(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vor(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vor(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vor(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vand(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vand(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vand(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vand(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vand(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vand(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vand(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vand(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vand(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, Literal mode);
void vand(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
          int32_t mode);
void vand(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
          int32_t mode);
void vand(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vand(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vand(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vand(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vand(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vand(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vand(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vand(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vand(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vmax(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vmax(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vmax(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vmax(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vmax(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vmax(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vmax(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vmax(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vmax(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, Literal mode);
void vmax(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
          int32_t mode);
void vmax(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
          int32_t mode);
void vmax(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vmax(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vmax(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vmax(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vmax(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vmax(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vmax(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vmax(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vmax(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vmin(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vmin(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vmin(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vmin(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vmin(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vmin(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vmin(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vmin(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vmin(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, Literal mode);
void vmin(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
          int32_t mode);
void vmin(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
          int32_t mode);
void vmin(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vmin(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vmin(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vmin(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vmin(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vmin(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vmin(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vmin(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vmin(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vsub(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vsub(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vsub(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vsub(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vsub(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vsub(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vsub(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vsub(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vsub(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, Literal mode);
void vsub(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vsub(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vsub(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vsub(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vsub(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vsub(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vsub(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vsub(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vsub(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vsub(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
          int32_t mode);
void vsub(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
          int32_t mode);
void vmul(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vmul(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vmul(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vmul(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vmul(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vmul(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vmul(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vmul(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vmul(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, Literal mode);
void vmul(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vmul(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vmul(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vmul(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vmul(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vmul(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vmul(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vmul(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vmul(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vmul(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
          int32_t mode);
void vmul(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
          int32_t mode);
void vmula(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vmula(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vmula(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vmula(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vmula(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vmula(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vmula(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vshl(vector_u8 &dst, vector_u8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vshl(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vshl(vector_u16 &dst, vector_u16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vshl(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vshl(vector_u32 &dst, vector_u32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vshl(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vshl(vector_u8 &dst, vector_u8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vshl(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vshl(vector_u16 &dst, vector_u16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vshl(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vshl(vector_u32 &dst, vector_u32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vshl(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vshr(vector_u8 &dst, vector_u8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vshr(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vshr(vector_u16 &dst, vector_u16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vshr(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vshr(vector_u32 &dst, vector_u32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vshr(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vshr(vector_u8 &dst, vector_u8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vshr(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vshr(vector_u16 &dst, vector_u16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vshr(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vshr(vector_u32 &dst, vector_u32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vshr(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vdiv(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, Literal mode);
void vdiv(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, Literal mode);
void vdiv(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vdiv(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vdiv(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vdiv(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vdiv(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vdiv(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vdiv(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vdiv(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vdiv(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vdiv(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vdiv(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vdiv(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vdiv(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vdiv(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vabsdif(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vsad(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vsad(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vsad(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vsad(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vsad(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vsad(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vexpdif(vector_f32 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vexpdif(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vexpdif(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vadif(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask, int32_t mode);
void vadif(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask, int32_t mode);
void vadif(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, int32_t mode);
void vadif(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, int32_t mode);
void vadif(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, int32_t mode);
void vadif(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, int32_t mode);
void vdiv(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, vector_2xvl_s64 src1, vector_bool mask,
          int32_t mode);
void vdiv(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, vector_2xvl_u64 src1, vector_bool mask,
          int32_t mode);
void vmadd(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, int32_t mode);
void vmadd(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, int32_t mode);
void vmadd(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask, int32_t mode);
void vcmp_eq(vector_bool &dst, vector_u8 src1, vector_u8 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_u8 src1, vector_u8 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_u8 src1, vector_u8 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_u8 src1, vector_u8 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_u8 src1, vector_u8 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_u8 src1, vector_u8 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_s8 src1, vector_s8 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_s8 src1, vector_s8 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_s8 src1, vector_s8 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_s8 src1, vector_s8 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_s8 src1, vector_s8 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_s8 src1, vector_s8 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_f16 src1, vector_f16 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_f16 src1, vector_f16 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_f16 src1, vector_f16 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_f16 src1, vector_f16 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_f16 src1, vector_f16 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_f16 src1, vector_f16 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_f32 src1, vector_f32 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_f32 src1, vector_f32 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_f32 src1, vector_f32 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_f32 src1, vector_f32 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_f32 src1, vector_f32 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_f32 src1, vector_f32 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_u16 src1, vector_u16 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_u16 src1, vector_u16 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_u16 src1, vector_u16 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_u16 src1, vector_u16 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_u16 src1, vector_u16 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_u16 src1, vector_u16 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_s16 src1, vector_s16 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_s16 src1, vector_s16 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_s16 src1, vector_s16 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_s16 src1, vector_s16 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_s16 src1, vector_s16 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_s16 src1, vector_s16 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_u32 src1, vector_u32 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_u32 src1, vector_u32 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_u32 src1, vector_u32 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_u32 src1, vector_u32 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_u32 src1, vector_u32 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_u32 src1, vector_u32 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_s32 src1, vector_s32 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_s32 src1, vector_s32 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_s32 src1, vector_s32 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_s32 src1, vector_s32 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_s32 src1, vector_s32 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_s32 src1, vector_s32 src2, vector_bool mask);
void vcmp_eq(vector_bool &dst, vector_bf16 src1, vector_bf16 src2, vector_bool mask);
void vcmp_ne(vector_bool &dst, vector_bf16 src1, vector_bf16 src2, vector_bool mask);
void vcmp_gt(vector_bool &dst, vector_bf16 src1, vector_bf16 src2, vector_bool mask);
void vcmp_ge(vector_bool &dst, vector_bf16 src1, vector_bf16 src2, vector_bool mask);
void vcmp_lt(vector_bool &dst, vector_bf16 src1, vector_bf16 src2, vector_bool mask);
void vcmp_le(vector_bool &dst, vector_bf16 src1, vector_bf16 src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_u8 src1, uint8_t src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_u8 src1, uint8_t src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_u8 src1, uint8_t src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_u8 src1, uint8_t src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_u8 src1, uint8_t src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_u8 src1, uint8_t src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_s8 src1, int8_t src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_s8 src1, int8_t src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_s8 src1, int8_t src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_s8 src1, int8_t src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_s8 src1, int8_t src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_s8 src1, int8_t src2, vector_bool mask);
void vcmps_eq(vector_bool &dst, vector_bf16 src1, bfloat16_t src2, vector_bool mask);
void vcmps_ne(vector_bool &dst, vector_bf16 src1, bfloat16_t src2, vector_bool mask);
void vcmps_gt(vector_bool &dst, vector_bf16 src1, bfloat16_t src2, vector_bool mask);
void vcmps_ge(vector_bool &dst, vector_bf16 src1, bfloat16_t src2, vector_bool mask);
void vcmps_lt(vector_bool &dst, vector_bf16 src1, bfloat16_t src2, vector_bool mask);
void vcmps_le(vector_bool &dst, vector_bf16 src1, bfloat16_t src2, vector_bool mask);
void vsel(vector_u8 &dst, vector_u8 src0, vector_u8 src1, vector_bool mask);
void vsel(vector_s8 &dst, vector_s8 src0, vector_s8 src1, vector_bool mask);
void vsel(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask);
void vsel(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask);
void vsel(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask);
void vsel(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask);
void vsel(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask);
void vsel(vector_bf16 &dst, vector_bf16 src0, vector_bf16 src1, vector_bool mask);
void vsel(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask);
void vselr(vector_u8 &dst, vector_u8 src0, vector_u8 src1);
void vselr(vector_s8 &dst, vector_s8 src0, vector_s8 src1);
void vselr(vector_u16 &dst, vector_u16 src0, vector_u16 src1);
void vselr(vector_s16 &dst, vector_s16 src0, vector_s16 src1);
void vselr(vector_u32 &dst, vector_u32 src0, vector_u32 src1);
void vselr(vector_s32 &dst, vector_s32 src0, vector_s32 src1);
void vselr(vector_f16 &dst, vector_f16 src0, vector_f16 src1);
void vbr(vector_bf16 &dst, bfloat16_t val);
void vbr(vector_f16 &dst, half val);
void vbr(vector_u16 &dst, uint16_t val);
void vbr(vector_s16 &dst, int16_t val);
void vbr(vector_f32 &dst, float val);
void vbr(vector_u32 &dst, uint32_t val);
void vbr(vector_s32 &dst, int32_t val);
void vbr(vector_u8 &dst, uint8_t val);
void vbr(vector_s8 &dst, int8_t val);
void sprclr(Literal spr_id);
void pmov(vector_bool &dst, vector_bool src, vector_bool mask);
void vsqz(vector_u8 &dst, vector_u8 src, vector_bool mask, Literal store);
void vsqz(vector_s8 &dst, vector_s8 src, vector_bool mask, Literal store);
void vsqz(vector_u8 &dst, vector_u8 src, vector_bool mask, uint32_t store);
void vsqz(vector_s8 &dst, vector_s8 src, vector_bool mask, uint32_t store);
void sprclr(uint32_t spr_id);
void vsqz(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal store);
void vsqz(vector_u16 &dst, vector_u16 src, vector_bool mask, Literal store);
void vsqz(vector_s16 &dst, vector_s16 src, vector_bool mask, Literal store);
void vsqz(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal store);
void vsqz(vector_u32 &dst, vector_u32 src, vector_bool mask, Literal store);
void vsqz(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal store);
void vstur(vector_align &alignData, vector_s8 src, __ubuf__ int8_t *base, Literal post);
void vstur(vector_align &alignData, vector_u8 src, __ubuf__ uint8_t *base, Literal post);
void vstur(vector_align &alignData, vector_s8 src, __ubuf__ int8_t *base, uint32_t post);
void vstur(vector_align &alignData, vector_u8 src, __ubuf__ uint8_t *base, uint32_t post);
void vsqz(vector_f16 &dst, vector_f16 src, vector_bool mask, uint32_t store);
void vsqz(vector_u16 &dst, vector_u16 src, vector_bool mask, uint32_t store);
void vsqz(vector_s16 &dst, vector_s16 src, vector_bool mask, uint32_t store);
void vsqz(vector_f32 &dst, vector_f32 src, vector_bool mask, uint32_t store);
void vsqz(vector_u32 &dst, vector_u32 src, vector_bool mask, uint32_t store);
void vsqz(vector_s32 &dst, vector_s32 src, vector_bool mask, uint32_t store);
void vusqz(vector_s16 &dst, vector_bool mask);
void vusqz(vector_u16 &dst, vector_bool mask);
void vusqz(vector_s8 &dst, vector_bool mask);
void vusqz(vector_u8 &dst, vector_bool mask);
void vusqz(vector_u32 &dst, vector_bool mask);
void vusqz(vector_s32 &dst, vector_bool mask);
void vstur(vector_align &alignData, vector_f16 src, __ubuf__ half *base, Literal post);
void vstur(vector_align &alignData, vector_bf16 src, __ubuf__ bfloat16_t *base, Literal post);
void vstur(vector_align &alignData, vector_u16 src, __ubuf__ uint16_t *base, Literal post);
void vstur(vector_align &alignData, vector_s16 src, __ubuf__ int16_t *base, Literal post);
void vstur(vector_align &alignData, vector_f32 src, __ubuf__ float *base, Literal post);
void vstur(vector_align &alignData, vector_u32 src, __ubuf__ uint32_t *base, Literal post);
void vstur(vector_align &alignData, vector_s32 src, __ubuf__ int32_t *base, Literal post);
void vstar(vector_align data, __ubuf__ uint8_t *base);
void vstar(vector_align data, __ubuf__ int8_t *base);
void vstur(vector_align &alignData, vector_f16 src, __ubuf__ half *base, uint32_t post);
void vstur(vector_align &alignData, vector_bf16 src, __ubuf__ bfloat16_t *base, uint32_t post);
void vstur(vector_align &alignData, vector_u16 src, __ubuf__ uint16_t *base, uint32_t post);
void vstur(vector_align &alignData, vector_s16 src, __ubuf__ int16_t *base, uint32_t post);
void vstur(vector_align &alignData, vector_f32 src, __ubuf__ float *base, uint32_t post);
void vstur(vector_align &alignData, vector_u32 src, __ubuf__ uint32_t *base, uint32_t post);
void vstur(vector_align &alignData, vector_s32 src, __ubuf__ int32_t *base, uint32_t post);
void vstar(vector_align data, __ubuf__ half *base);
void vstar(vector_align data, __ubuf__ int16_t *base);
void vstar(vector_align data, __ubuf__ uint16_t *base);
void vstar(vector_align data, __ubuf__ bfloat16_t *base);
void vstar(vector_align data, __ubuf__ float *base);
void vstar(vector_align data, __ubuf__ int32_t *base);
void vstar(vector_align data, __ubuf__ uint32_t *base);
void vstar(vector_align data, __ubuf__ int64_t *base);
vector_bool pge_b8(Literal dist);
vector_bool pge_b16(Literal dist);
vector_bool pge_b32(Literal dist);
vector_bool pset_b8(Literal dist);
vector_bool pset_b16(Literal dist);
vector_bool pset_b32(Literal dist);
vector_bool pset_2xvl_b64(Literal dist);
vector_bool pset_b8(int32_t dist);
vector_bool pset_b16(int32_t dist);
vector_bool pset_b32(int32_t dist);
vector_bool pset_2xvl_b64(int32_t dist);
void vlda(vector_align &data, __ubuf__ uint8_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ int8_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ uint16_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ int16_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ half *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ bfloat16_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ uint32_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ int32_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ float *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ uint64_t *base, vector_address offset);
void vlda(vector_align &data, __ubuf__ int64_t *base, vector_address offset);
void vldu(vector_s8 &dst, vector_align &alignData, vector_address &offset, __ubuf__ int8_t *base,
          uint32_t inc);
void vldu(vector_u8 &dst, vector_align &alignData, vector_address &offset, __ubuf__ uint8_t *base,
          uint32_t inc);
void vldu(vector_s16 &dst, vector_align &alignData, vector_address &offset, __ubuf__ int16_t *base,
          uint32_t inc);
void vldu(vector_u16 &dst, vector_align &alignData, vector_address &offset, __ubuf__ uint16_t *base,
          uint32_t inc);
void vldu(vector_f16 &dst, vector_align &alignData, vector_address &offset, __ubuf__ half *base,
          uint32_t inc);
void vldu(vector_bf16 &dst, vector_align &alignData, vector_address &offset,
          __ubuf__ bfloat16_t *base, uint32_t inc);
void vldu(vector_u32 &dst, vector_align &alignData, vector_address &offset, __ubuf__ uint32_t *base,
          uint32_t inc);
void vldu(vector_s32 &dst, vector_align &alignData, vector_address &offset, __ubuf__ int32_t *base,
          uint32_t inc);
void vldu(vector_f32 &dst, vector_align &alignData, vector_address &offset, __ubuf__ float *base,
          uint32_t inc);
void vldu(vector_s64 &dst, vector_align &alignData, vector_address &offset, __ubuf__ int64_t *base,
          uint32_t inc);
void vldu(vector_u64 &dst, vector_align &alignData, vector_address &offset, __ubuf__ uint64_t *base,
          uint32_t inc);
void vldas(vector_align &data, __ubuf__ uint8_t *base);
void vldas(vector_align &data, __ubuf__ int8_t *base);
void vldas(vector_align &data, __ubuf__ half *base);
void vldas(vector_align &data, __ubuf__ bfloat16_t *base);
void vldas(vector_align &data, __ubuf__ uint16_t *base);
void vldas(vector_align &data, __ubuf__ int16_t *base);
void vldas(vector_align &data, __ubuf__ float *base);
void vldas(vector_align &data, __ubuf__ uint32_t *base);
void vldas(vector_align &data, __ubuf__ int32_t *base);
void vldas(vector_align &data, __ubuf__ int64_t *base);
void vldas(vector_align &data, __ubuf__ uint64_t *base);
void vldus(vector_u8 &dst, vector_align &alignData, __ubuf__ uint8_t *base);
void vldus(vector_s8 &dst, vector_align &alignData, __ubuf__ int8_t *base);
void vldus(vector_f16 &dst, vector_align &alignData, __ubuf__ half *base);
void vldus(vector_u16 &dst, vector_align &alignData, __ubuf__ uint16_t *base);
void vldus(vector_s16 &dst, vector_align &alignData, __ubuf__ int16_t *base);
void vldus(vector_bf16 &dst, vector_align &alignData, __ubuf__ bfloat16_t *base);
void vldus(vector_f32 &dst, vector_align &alignData, __ubuf__ float *base);
void vldus(vector_u32 &dst, vector_align &alignData, __ubuf__ uint32_t *base);
void vldus(vector_s32 &dst, vector_align &alignData, __ubuf__ int32_t *base);
void vldus(vector_u64 &dst, vector_align &alignData, __ubuf__ uint64_t *base);
void vldus(vector_s64 &dst, vector_align &alignData, __ubuf__ int64_t *base);
void vldus(vector_u8 &dst, vector_align &alignData, __ubuf__ uint8_t *&base, uint32_t inc,
           Literal post);
void vldus(vector_s8 &dst, vector_align &alignData, __ubuf__ int8_t *&base, uint32_t inc,
           Literal post);
void vldus(vector_f16 &dst, vector_align &alignData, __ubuf__ half *&base, uint32_t inc,
           Literal post);
void vldus(vector_u16 &dst, vector_align &alignData, __ubuf__ uint16_t *&base, uint32_t inc,
           Literal post);
void vldus(vector_s16 &dst, vector_align &alignData, __ubuf__ int16_t *&base, uint32_t inc,
           Literal post);
void vldus(vector_bf16 &dst, vector_align &alignData, __ubuf__ bfloat16_t *&base, uint32_t inc,
           Literal post);
void vldus(vector_f32 &dst, vector_align &alignData, __ubuf__ float *&base, uint32_t inc,
           Literal post);
void vldus(vector_u32 &dst, vector_align &alignData, __ubuf__ uint32_t *&base, uint32_t inc,
           Literal post);
void vldus(vector_s32 &dst, vector_align &alignData, __ubuf__ int32_t *&base, uint32_t inc,
           Literal post);
void vldus(vector_u8 &dst, vector_align &alignData, __ubuf__ uint8_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_s8 &dst, vector_align &alignData, __ubuf__ int8_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_f16 &dst, vector_align &alignData, __ubuf__ half *&base, uint32_t inc,
           int32_t post);
void vldus(vector_u16 &dst, vector_align &alignData, __ubuf__ uint16_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_s16 &dst, vector_align &alignData, __ubuf__ int16_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_bf16 &dst, vector_align &alignData, __ubuf__ bfloat16_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_f32 &dst, vector_align &alignData, __ubuf__ float *&base, uint32_t inc,
           int32_t post);
void vldus(vector_u32 &dst, vector_align &alignData, __ubuf__ uint32_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_s32 &dst, vector_align &alignData, __ubuf__ int32_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_u64 &dst, vector_align &alignData, __ubuf__ uint64_t *&base, uint32_t inc,
           int32_t post);
void vldus(vector_s64 &dst, vector_align &alignData, __ubuf__ int64_t *&base, uint32_t inc,
           int32_t post);
void vdup(vector_u8 &dst, uint8_t src, vector_bool mask, Literal mode);
void vdup(vector_s8 &dst, int8_t src, vector_bool mask, Literal mode);
void vdup(vector_u16 &dst, uint16_t src, vector_bool mask, Literal mode);
void vdup(vector_s16 &dst, int16_t src, vector_bool mask, Literal mode);
void vdup(vector_s32 &dst, int32_t src, vector_bool mask, Literal mode);
void vdup(vector_u32 &dst, uint32_t src, vector_bool mask, Literal mode);
void vdup(vector_f32 &dst, float src, vector_bool mask, Literal mode);
void vdup(vector_f16 &dst, half src, vector_bool mask, Literal mode);
void vdup(vector_bf16 &dst, bfloat16_t src, vector_bool mask, Literal mode);
void vdup(vector_u8 &dst, uint8_t src, vector_bool mask, int32_t mode);
void vdup(vector_s8 &dst, int8_t src, vector_bool mask, int32_t mode);
void vdup(vector_u16 &dst, uint16_t src, vector_bool mask, int32_t mode);
void vdup(vector_s16 &dst, int16_t src, vector_bool mask, int32_t mode);
void vdup(vector_s32 &dst, int32_t src, vector_bool mask, int32_t mode);
void vdup(vector_u32 &dst, uint32_t src, vector_bool mask, int32_t mode);
void vdup(vector_f32 &dst, float src, vector_bool mask, int32_t mode);
void vdup(vector_f16 &dst, half src, vector_bool mask, int32_t mode);
void vdup(vector_bf16 &dst, bfloat16_t src, vector_bool mask, int32_t mode);
void vdup(vector_2xvl_s64 &dst, int64_t src, vector_bool mask, int32_t mode);
void vdup(vector_2xvl_u64 &dst, uint64_t src, vector_bool mask, int32_t mode);
void vdup(vector_u16 &dst, vector_u16 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_u32 &dst, vector_u32 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_u8 &dst, vector_u8 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_bf16 &dst, vector_bf16 src, vector_bool mask, int32_t type, Literal mode);
void vdup(vector_u8 &dst, vector_u8 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_u16 &dst, vector_u16 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_u32 &dst, vector_u32 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t type, int32_t mode);
void vdup(vector_bf16 &dst, vector_bf16 src, vector_bool mask, int32_t type, int32_t mode);
void vmov(vector_u8 &dst, vector_u8 src, vector_bool mask, int32_t mode);
void vmov(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t mode);
void vmov(vector_u16 &dst, vector_u16 src, vector_bool mask, int32_t mode);
void vmov(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t mode);
void vmov(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vmov(vector_bf16 &dst, vector_bf16 src, vector_bool mask, int32_t mode);
void vmov(vector_u32 &dst, vector_u32 src, vector_bool mask, int32_t mode);
void vmov(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vmov(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vmov(vector_u8 &dst, vector_u8 src);
void vmov(vector_s8 &dst, vector_s8 src);
void vmov(vector_u16 &dst, vector_u16 src);
void vmov(vector_s16 &dst, vector_s16 src);
void vmov(vector_f16 &dst, vector_f16 src);
void vmov(vector_bf16 &dst, vector_bf16 src);
void vmov(vector_u32 &dst, vector_u32 src);
void vmov(vector_s32 &dst, vector_s32 src);
void vmov(vector_f32 &dst, vector_f32 src);
void vld(vector_s8 &dst, __ubuf__ int8_t *base, vector_address offset, Literal dist);
void vld(vector_u8 &dst, __ubuf__ uint8_t *base, vector_address offset, Literal dist);
void vld(vector_s16 &dst, __ubuf__ int16_t *base, vector_address offset, Literal dist);
void vld(vector_u16 &dst, __ubuf__ uint16_t *base, vector_address offset, Literal dist);
void vld(vector_s32 &dst, __ubuf__ int32_t *base, vector_address offset, Literal dist);
void vld(vector_u32 &dst, __ubuf__ uint32_t *base, vector_address offset, Literal dist);
void vld(vector_s64 &dst, __ubuf__ int64_t *base, vector_address offset, Literal dist);
void vld(vector_u64 &dst, __ubuf__ uint64_t *base, vector_address offset, Literal dist);
void vld(vector_bf16 &dst, __ubuf__ bfloat16_t *base, vector_address offset, Literal dist);
void vld(vector_f16 &dst, __ubuf__ half *base, vector_address offset, Literal dist);
void vld(vector_f32 &dst, __ubuf__ float *base, vector_address offset, Literal dist);
void vld(vector_s8 &dst, __ubuf__ int8_t *base, vector_address offset, int32_t dist);
void vld(vector_u8 &dst, __ubuf__ uint8_t *base, vector_address offset, int32_t dist);
void vld(vector_s16 &dst, __ubuf__ int16_t *base, vector_address offset, int32_t dist);
void vld(vector_u16 &dst, __ubuf__ uint16_t *base, vector_address offset, int32_t dist);
void vld(vector_s32 &dst, __ubuf__ int32_t *base, vector_address offset, int32_t dist);
void vld(vector_u32 &dst, __ubuf__ uint32_t *base, vector_address offset, int32_t dist);
void vld(vector_s64 &dst, __ubuf__ int64_t *base, vector_address offset, int32_t dist);
void vld(vector_u64 &dst, __ubuf__ uint64_t *base, vector_address offset, int32_t dist);
void vld(vector_bf16 &dst, __ubuf__ bfloat16_t *base, vector_address offset, int32_t dist);
void vld(vector_f16 &dst, __ubuf__ half *base, vector_address offset, int32_t dist);
void vld(vector_f32 &dst, __ubuf__ float *base, vector_address offset, int32_t dist);
void vld(vector_s8 &dst, __ubuf__ int8_t *base, Literal dist);
void vld(vector_u8 &dst, __ubuf__ uint8_t *base, Literal dist);
void vld(vector_s16 &dst, __ubuf__ int16_t *base, Literal dist);
void vld(vector_u16 &dst, __ubuf__ uint16_t *base, Literal dist);
void vld(vector_s32 &dst, __ubuf__ int32_t *base, Literal dist);
void vld(vector_u32 &dst, __ubuf__ uint32_t *base, Literal dist);
void vld(vector_s64 &dst, __ubuf__ int64_t *base, Literal dist);
void vld(vector_u64 &dst, __ubuf__ uint64_t *base, Literal dist);
void vld(vector_bf16 &dst, __ubuf__ bfloat16_t *base, Literal dist);
void vld(vector_f16 &dst, __ubuf__ half *base, Literal dist);
void vld(vector_f32 &dst, __ubuf__ float *base, Literal dist);
void vld(vector_s8 &dst, __ubuf__ int8_t *base, int32_t dist);
void vld(vector_u8 &dst, __ubuf__ uint8_t *base, int32_t dist);
void vld(vector_s16 &dst, __ubuf__ int16_t *base, int32_t dist);
void vld(vector_u16 &dst, __ubuf__ uint16_t *base, int32_t dist);
void vld(vector_s32 &dst, __ubuf__ int32_t *base, int32_t dist);
void vld(vector_u32 &dst, __ubuf__ uint32_t *base, int32_t dist);
void vld(vector_s64 &dst, __ubuf__ int64_t *base, int32_t dist);
void vld(vector_u64 &dst, __ubuf__ uint64_t *base, int32_t dist);
void vld(vector_bf16 &dst, __ubuf__ bfloat16_t *base, int32_t dist);
void vld(vector_f16 &dst, __ubuf__ half *base, int32_t dist);
void vld(vector_f32 &dst, __ubuf__ float *base, int32_t dist);
void vld(vector_s8 &dst, __ubuf__ int8_t *base, int32_t dist, int32_t post);
void vld(vector_u8 &dst, __ubuf__ uint8_t *base, int32_t dist, int32_t post);
void vld(vector_s16 &dst, __ubuf__ int16_t *base, int32_t dist, int32_t post);
void vld(vector_u16 &dst, __ubuf__ uint16_t *base, int32_t dist, int32_t post);
void vld(vector_s32 &dst, __ubuf__ int32_t *base, int32_t dist, int32_t post);
void vld(vector_u32 &dst, __ubuf__ uint32_t *base, int32_t dist, int32_t post);
void vld(vector_s64 &dst, __ubuf__ int64_t *base, int32_t dist, int32_t post);
void vld(vector_u64 &dst, __ubuf__ uint64_t *base, int32_t dist, int32_t post);
void vld(vector_bf16 &dst, __ubuf__ bfloat16_t *base, int32_t dist, int32_t post);
void vld(vector_f16 &dst, __ubuf__ half *base, int32_t dist, int32_t post);
void vld(vector_f32 &dst, __ubuf__ float *base, int32_t dist, int32_t post);
void vld(vector_s8 &dst0, vector_s8 &dst1, __ubuf__ int8_t *base, vector_address offset,
         Literal dist);
void vld(vector_u8 &dst0, vector_u8 &dst1, __ubuf__ uint8_t *base, vector_address offset,
         Literal dist);
void vld(vector_s16 &dst0, vector_s16 &dst1, __ubuf__ int16_t *base, vector_address offset,
         Literal dist);
void vld(vector_u16 &dst0, vector_u16 &dst1, __ubuf__ uint16_t *base, vector_address offset,
         Literal dist);
void vld(vector_s32 &dst0, vector_s32 &dst1, __ubuf__ int32_t *base, vector_address offset,
         Literal dist);
void vld(vector_u32 &dst0, vector_u32 &dst1, __ubuf__ uint32_t *base, vector_address offset,
         Literal dist);
void vld(vector_u64 &dst0, vector_u64 &dst1, __ubuf__ uint64_t *base, vector_address offset,
         Literal dist);
void vld(vector_s64 &dst0, vector_s64 &dst1, __ubuf__ int64_t *base, vector_address offset,
         Literal dist);
void vld(vector_bf16 &dst0, vector_bf16 &dst1, __ubuf__ bfloat16_t *base, vector_address offset,
         Literal dist);
void vld(vector_f16 &dst0, vector_f16 &dst1, __ubuf__ half *base, vector_address offset,
         Literal dist);
void vld(vector_f32 &dst0, vector_f32 &dst1, __ubuf__ float *base, vector_address offset,
         Literal dist);
void vld(vector_s8 &dst0, vector_s8 &dst1, __ubuf__ int8_t *base, vector_address offset,
         int32_t dist);
void vld(vector_u8 &dst0, vector_u8 &dst1, __ubuf__ uint8_t *base, vector_address offset,
         int32_t dist);
void vld(vector_s16 &dst0, vector_s16 &dst1, __ubuf__ int16_t *base, vector_address offset,
         int32_t dist);
void vld(vector_u16 &dst0, vector_u16 &dst1, __ubuf__ uint16_t *base, vector_address offset,
         int32_t dist);
void vld(vector_s32 &dst0, vector_s32 &dst1, __ubuf__ int32_t *base, vector_address offset,
         int32_t dist);
void vld(vector_u32 &dst0, vector_u32 &dst1, __ubuf__ uint32_t *base, vector_address offset,
         int32_t dist);
void vld(vector_u64 &dst0, vector_u64 &dst1, __ubuf__ uint64_t *base, vector_address offset,
         int32_t dist);
void vld(vector_s64 &dst0, vector_s64 &dst1, __ubuf__ int64_t *base, vector_address offset,
         int32_t dist);
void vld(vector_bf16 &dst0, vector_bf16 &dst1, __ubuf__ bfloat16_t *base, vector_address offset,
         int32_t dist);
void vld(vector_f16 &dst0, vector_f16 &dst1, __ubuf__ half *base, vector_address offset,
         int32_t dist);
void vld(vector_f32 &dst0, vector_f32 &dst1, __ubuf__ float *base, vector_address offset,
         int32_t dist);
void vld(vector_s8 &dst0, vector_s8 &dst1, __ubuf__ int8_t *base, Literal offset);
void vld(vector_u8 &dst0, vector_u8 &dst1, __ubuf__ uint8_t *base, Literal offset);
void vld(vector_s16 &dst0, vector_s16 &dst1, __ubuf__ int16_t *base, Literal offset);
void vld(vector_u16 &dst0, vector_u16 &dst1, __ubuf__ uint16_t *base, Literal offset);
void vld(vector_s32 &dst0, vector_s32 &dst1, __ubuf__ int32_t *base, Literal offset);
void vld(vector_u32 &dst0, vector_u32 &dst1, __ubuf__ uint32_t *base, Literal offset);
void vld(vector_u64 &dst0, vector_u64 &dst1, __ubuf__ uint64_t *base, Literal offset);
void vld(vector_s64 &dst0, vector_s64 &dst1, __ubuf__ int64_t *base, Literal offset);
void vld(vector_bf16 &dst0, vector_bf16 &dst1, __ubuf__ bfloat16_t *base, Literal offset);
void vld(vector_f16 &dst0, vector_f16 &dst1, __ubuf__ half *base, Literal offset);
void vld(vector_f32 &dst0, vector_f32 &dst1, __ubuf__ float *base, Literal offset);
void vlds(vector_s8 &dst, __ubuf__ int8_t *base, int32_t offset, Literal dist);
void vlds(vector_u8 &dst, __ubuf__ uint8_t *base, int32_t offset, Literal dist);
void vlds(vector_s16 &dst, __ubuf__ int16_t *base, int32_t offset, Literal dist);
void vlds(vector_u16 &dst, __ubuf__ uint16_t *base, int32_t offset, Literal dist);
void vlds(vector_s32 &dst, __ubuf__ int32_t *base, int32_t offset, Literal dist);
void vlds(vector_u32 &dst, __ubuf__ uint32_t *base, int32_t offset, Literal dist);
void vlds(vector_u64 &dst, __ubuf__ uint64_t *base, int32_t offset, Literal dist);
void vlds(vector_s64 &dst, __ubuf__ int64_t *base, int32_t offset, Literal dist);
void vlds(vector_bf16 &dst, __ubuf__ bfloat16_t *base, int32_t offset, Literal dist);
void vlds(vector_f16 &dst, __ubuf__ half *base, int32_t offset, Literal dist);
void vlds(vector_f32 &dst, __ubuf__ float *base, int32_t offset, Literal dist);
void vlds(vector_s8 &dst, __ubuf__ int8_t *base, int32_t offset, int32_t dist);
void vlds(vector_u8 &dst, __ubuf__ uint8_t *base, int32_t offset, int32_t dist);
void vlds(vector_s16 &dst, __ubuf__ int16_t *base, int32_t offset, int32_t dist);
void vlds(vector_u16 &dst, __ubuf__ uint16_t *base, int32_t offset, int32_t dist);
void vlds(vector_s32 &dst, __ubuf__ int32_t *base, int32_t offset, int32_t dist);
void vlds(vector_u32 &dst, __ubuf__ uint32_t *base, int32_t offset, int32_t dist);
void vlds(vector_u64 &dst, __ubuf__ uint64_t *base, int32_t offset, int32_t dist);
void vlds(vector_s64 &dst, __ubuf__ int64_t *base, int32_t offset, int32_t dist);
void vlds(vector_bf16 &dst, __ubuf__ bfloat16_t *base, int32_t offset, int32_t dist);
void vlds(vector_f16 &dst, __ubuf__ half *base, int32_t offset, int32_t dist);
void vlds(vector_f32 &dst, __ubuf__ float *base, int32_t offset, int32_t dist);
void vlds(vector_s8 &dst, __ubuf__ int8_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_u8 &dst, __ubuf__ uint8_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_s16 &dst, __ubuf__ int16_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_u16 &dst, __ubuf__ uint16_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_s32 &dst, __ubuf__ int32_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_u32 &dst, __ubuf__ uint32_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_u64 &dst, __ubuf__ uint64_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_s64 &dst, __ubuf__ int64_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_bf16 &dst, __ubuf__ bfloat16_t *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_f16 &dst, __ubuf__ half *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_f32 &dst, __ubuf__ float *&base, int32_t offset, int32_t dist, int32_t post);
void vlds(vector_2xvl_u64 &dst, __ubuf__ uint64_t *base, int32_t offset);
void vlds(vector_2xvl_s64 &dst, __ubuf__ int64_t *base, int32_t offset);
void vlds(vector_s8 &dst0, vector_s8 &dst1, __ubuf__ int8_t *base, int32_t offset, int32_t dist);
void vlds(vector_u8 &dst0, vector_u8 &dst1, __ubuf__ uint8_t *base, int32_t offset, int32_t dist);
void vlds(vector_s16 &dst0, vector_s16 &dst1, __ubuf__ int16_t *base, int32_t offset, int32_t dist);
void vlds(vector_u16 &dst0, vector_u16 &dst1, __ubuf__ uint16_t *base, int32_t offset,
          int32_t dist);
void vlds(vector_s32 &dst0, vector_s32 &dst1, __ubuf__ int32_t *base, int32_t offset, int32_t dist);
void vlds(vector_u32 &dst0, vector_u32 &dst1, __ubuf__ uint32_t *base, int32_t offset,
          int32_t dist);
void vlds(vector_u64 &dst0, vector_u64 &dst1, __ubuf__ uint64_t *base, int32_t offset,
          int32_t dist);
void vlds(vector_s64 &dst0, vector_s64 &dst1, __ubuf__ int64_t *base, int32_t offset, int32_t dist);
void vlds(vector_bf16 &dst0, vector_bf16 &dst1, __ubuf__ bfloat16_t *base, int32_t offset,
          int32_t dist);
void vlds(vector_f16 &dst0, vector_f16 &dst1, __ubuf__ half *base, int32_t offset, int32_t dist);
void vlds(vector_f32 &dst0, vector_f32 &dst1, __ubuf__ float *base, int32_t offset, int32_t dist);
void vlds(vector_s8 &dst0, vector_s8 &dst1, __ubuf__ int8_t *&base, int32_t offset, int32_t dist,
          int32_t post);
void vlds(vector_u8 &dst0, vector_u8 &dst1, __ubuf__ uint8_t *&base, int32_t offset, int32_t dist,
          int32_t post);
void vlds(vector_s16 &dst0, vector_s16 &dst1, __ubuf__ int16_t *&base, int32_t offset, int32_t dist,
          int32_t post);
void vlds(vector_u16 &dst0, vector_u16 &dst1, __ubuf__ uint16_t *&base, int32_t offset,
          int32_t dist, int32_t post);
void vlds(vector_s32 &dst0, vector_s32 &dst1, __ubuf__ int32_t *&base, int32_t offset, int32_t dist,
          int32_t post);
void vlds(vector_u32 &dst0, vector_u32 &dst1, __ubuf__ uint32_t *&base, int32_t offset,
          int32_t dist, int32_t post);
void vlds(vector_u64 &dst0, vector_u64 &dst1, __ubuf__ uint64_t *&base, int32_t offset,
          int32_t dist, int32_t post);
void vlds(vector_s64 &dst0, vector_s64 &dst1, __ubuf__ int64_t *&base, int32_t offset, int32_t dist,
          int32_t post);
void vlds(vector_bf16 &dst0, vector_bf16 &dst1, __ubuf__ bfloat16_t *&base, int32_t offset,
          int32_t dist, int32_t post);
void vlds(vector_f16 &dst0, vector_f16 &dst1, __ubuf__ half *&base, int32_t offset, int32_t dist,
          int32_t post);
void vlds(vector_f32 &dst0, vector_f32 &dst1, __ubuf__ float *&base, int32_t offset, int32_t dist,
          int32_t post);
void vaxpy(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, Literal mode);
void vaxpy(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, Literal mode);
void vaxpy(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, int32_t mode);
void vaxpy(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, int32_t mode);
void vadds(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, Literal mode);
void vadds(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask);
void vadds(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, Literal mode);
void vadds(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vadds(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, Literal mode);
void vadds(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, Literal mode);
void vadds(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, Literal mode);
void vadds(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, Literal mode);
void vadds(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, Literal mode);
void vadds(vector_bf16 &dst, vector_bf16 src0, bfloat16_t scalar, vector_bool mask, Literal mode);
void vadds(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, int32_t mode);
void vadds(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, int32_t mode);
void vadds(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vadds(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, int32_t mode);
void vadds(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, int32_t mode);
void vadds(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, int32_t mode);
void vadds(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, int32_t mode);
void vadds(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, int32_t mode);
void vadds(vector_bf16 &dst, vector_bf16 src0, bfloat16_t scalar, vector_bool mask, int32_t mode);
void vadds(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, int64_t scalar, vector_bool mask,
           int32_t mode);
void vadds(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, uint64_t scalar, vector_bool mask,
           int32_t mode);
void vmuls(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, Literal mode);
void vmuls(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, Literal mode);
void vmuls(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vmuls(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, Literal mode);
void vmuls(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, Literal mode);
void vmuls(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, Literal mode);
void vmuls(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, Literal mode);
void vmuls(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, Literal mode);
void vmuls(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, int32_t mode);
void vmuls(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, int32_t mode);
void vmuls(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vmuls(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, int32_t mode);
void vmuls(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, int32_t mode);
void vmuls(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, int32_t mode);
void vmuls(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, int32_t mode);
void vmuls(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, int32_t mode);
void vmuls(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, int64_t scalar, vector_bool mask,
           int32_t mode);
void vmuls(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, uint64_t scalar, vector_bool mask,
           int32_t mode);
void vmaxs(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, Literal mode);
void vmaxs(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, Literal mode);
void vmaxs(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vmaxs(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, Literal mode);
void vmaxs(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, Literal mode);
void vmaxs(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, Literal mode);
void vmaxs(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, Literal mode);
void vmaxs(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, Literal mode);
void vmaxs(vector_bf16 &dst, vector_bf16 src0, bfloat16_t scalar, vector_bool mask, Literal mode);
void vmaxs(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, int64_t scalar, vector_bool mask,
           int32_t mode);
void vmaxs(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, uint64_t scalar, vector_bool mask,
           int32_t mode);
void vmaxs(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, int32_t mode);
void vmaxs(vector_bf16 &dst, vector_bf16 src0, bfloat16_t scalar, vector_bool mask, int32_t mode);
void vmins(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, Literal mode);
void vmins(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, Literal mode);
void vmins(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vmins(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, Literal mode);
void vmins(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, Literal mode);
void vmins(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, Literal mode);
void vmins(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, Literal mode);
void vmins(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, Literal mode);
void vmins(vector_bf16 &dst, vector_bf16 src0, bfloat16_t scalar, vector_bool mask, Literal mode);
void vmins(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, int64_t scalar, vector_bool mask,
           int32_t mode);
void vmins(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, uint64_t scalar, vector_bool mask,
           int32_t mode);
void vmins(vector_s32 &dst, vector_s32 src0, int32_t scalar, vector_bool mask, int32_t mode);
void vmins(vector_u32 &dst, vector_u32 src0, uint32_t scalar, vector_bool mask, int32_t mode);
void vmins(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vmins(vector_u16 &dst, vector_u16 src0, uint16_t scalar, vector_bool mask, int32_t mode);
void vmins(vector_s8 &dst, vector_s8 src0, int8_t scalar, vector_bool mask, int32_t mode);
void vmins(vector_u8 &dst, vector_u8 src0, uint8_t scalar, vector_bool mask, int32_t mode);
void vmins(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, int32_t mode);
void vmins(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, int32_t mode);
void vmins(vector_bf16 &dst, vector_bf16 src0, bfloat16_t scalar, vector_bool mask, int32_t mode);
void vshls(vector_s32 &dst, vector_s32 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshls(vector_u32 &dst, vector_u32 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshls(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshls(vector_u16 &dst, vector_u16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshls(vector_s8 &dst, vector_s8 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshls(vector_u8 &dst, vector_u8 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshls(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, int64_t scalar, vector_bool mask,
           int32_t mode);
void vshls(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, uint64_t scalar, vector_bool mask,
           int32_t mode);
void vshls(vector_s32 &dst, vector_s32 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshls(vector_u32 &dst, vector_u32 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshls(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshls(vector_u16 &dst, vector_u16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshls(vector_s8 &dst, vector_s8 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshls(vector_u8 &dst, vector_u8 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshrs(vector_s32 &dst, vector_s32 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshrs(vector_u32 &dst, vector_u32 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshrs(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshrs(vector_u16 &dst, vector_u16 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshrs(vector_s8 &dst, vector_s8 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshrs(vector_u8 &dst, vector_u8 src0, int16_t scalar, vector_bool mask, Literal mode);
void vshrs(vector_2xvl_s64 &dst, vector_2xvl_s64 src0, int64_t scalar, vector_bool mask,
           int32_t mode);
void vshrs(vector_2xvl_u64 &dst, vector_2xvl_u64 src0, uint64_t scalar, vector_bool mask,
           int32_t mode);
void vshrs(vector_s32 &dst, vector_s32 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshrs(vector_u32 &dst, vector_u32 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshrs(vector_s16 &dst, vector_s16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshrs(vector_u16 &dst, vector_u16 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshrs(vector_s8 &dst, vector_s8 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vshrs(vector_u8 &dst, vector_u8 src0, int16_t scalar, vector_bool mask, int32_t mode);
void vmulscvt(vector_f16 &dst, vector_f32 src0, float scalar, vector_bool mask, int32_t mode);
void vlrelu(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, Literal mode);
void vlrelu(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, Literal mode);
void vlrelu(vector_f32 &dst, vector_f32 src0, float scalar, vector_bool mask, int32_t mode);
void vlrelu(vector_f16 &dst, vector_f16 src0, half scalar, vector_bool mask, int32_t mode);
void use_pipe_v();
void use_pipe_v2();
void v4dtrans(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void v4dtrans(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint16_t imageSize, uint16_t nChannel,
              bool conversionMode);
void v4dtrans(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void v4dtrans(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint16_t imageSize, uint16_t nChannel,
              bool conversionMode);
void v4dtrans(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void v4dtrans(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint16_t imageSize, uint16_t nChannel,
              bool conversionMode);
void vaadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vaadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vaadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t scr1RepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vaadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vaadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vaadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t scr1RepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vabs(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vabs(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vabs(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vabs(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vabs(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vabs(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vabs(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vabs(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vabs(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vabs(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vabs(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vabs(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t repeat);
void vadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vadd(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vadd(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vadd(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vadd(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint64_t config);
void vadd(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vadd(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vadddeqrelu(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
                 uint64_t config);
void vadddeqrelu(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
                 uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
                 uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vadddeqrelu(__ubuf__ half *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
                 uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
                 uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
                 bool repeatStrideMode, bool strideSizeMode);
void vaddrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vaddrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vaddrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vaddrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vaddrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vaddrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vaddrelu(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
              uint64_t config);
void vaddrelu(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vaddrelu(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vaddreluconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                         uint64_t config, bool h);
void vaddreluconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode,
                         bool h);
void vaddreluconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool h);
void vaddreluconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                          uint64_t config, bool h);
void vaddreluconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                          uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                          uint8_t src1BlockStride, uint8_t dstRepeatStride,
                          uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
                          bool strideSizeMode, bool h);
void vaddreluconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                          uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                          uint8_t src1BlockStride, uint8_t dstRepeatStride,
                          uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h);
void vaddreluconv_s162s8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                         uint64_t config, bool h);
void vaddreluconv_s162s8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode,
                         bool h);
void vaddreluconv_s162s8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool h);
void vaddreluconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint64_t config, bool h);
void vaddreluconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint64_t config, bool h);
void vaddreluconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride,
                             bool repeatStrideMode, bool strideSizeMode, bool h);
void vaddreluconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h);
void vaddreluconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride,
                             bool repeatStrideMode, bool strideSizeMode, bool h);
void vaddreluconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h);
void vadds(__ubuf__ half *dst, __ubuf__ half *src, half a, uint64_t config);
void vadds(__ubuf__ half *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vadds(__ubuf__ half *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vadds(__ubuf__ half *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vadds(__ubuf__ float *dst, __ubuf__ float *src, float a, uint64_t config);
void vadds(__ubuf__ float *dst, __ubuf__ float *src, float a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vadds(__ubuf__ float *dst, __ubuf__ float *src, float a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vadds(__ubuf__ float *dst, __ubuf__ float *src, float a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vadds(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int16_t a, uint64_t config);
void vadds(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int16_t a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vadds(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int16_t a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vadds(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int16_t a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vadds(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t a, uint64_t config);
void vadds(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vadds(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vadds(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vand(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vand(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1StrideBlock,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vand(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1StrideBlock,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vand(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1,
          uint64_t config);
void vand(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1StrideBlock,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vand(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1StrideBlock,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vaxpy(__ubuf__ half *dst, __ubuf__ half *src, half a, uint64_t config);
void vaxpy(__ubuf__ half *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vaxpy(__ubuf__ half *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vaxpy(__ubuf__ half *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vaxpy(__ubuf__ float *dst, __ubuf__ float *src, float a, uint64_t config);
void vaxpy(__ubuf__ float *dst, __ubuf__ float *src, float a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vaxpy(__ubuf__ float *dst, __ubuf__ float *src, float a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vaxpy(__ubuf__ float *dst, __ubuf__ float *src, float a, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vaxpy(__ubuf__ float *dst, __ubuf__ half *src, half a, uint64_t config);
void vaxpy(__ubuf__ float *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vaxpy(__ubuf__ float *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vaxpy(__ubuf__ float *dst, __ubuf__ half *src, half a, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vbi(__ubuf__ half *dst, __ubuf__ uint16_t *src0, __ubuf__ half *src1, uint64_t config);
void vbi(__ubuf__ half *dst, __ubuf__ uint16_t *src0, __ubuf__ half *src1, uint8_t hRepeat,
         bool repeatMode, uint16_t dstBlockStride, uint16_t vROffset, uint8_t vRepeat);
void vbitsort(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vbitsort(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vbitsort(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vbitsort(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vbitsort(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vbitsort(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vbitsort(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ uint32_t *src1, uint64_t config);
void vbitsort(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ uint32_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vbitsort(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ uint32_t *src1, uint64_t config);
void vbitsort(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ uint32_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vbitsort(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ uint32_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vbitsort(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ uint32_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vbs(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ uint32_t *src1, uint64_t config);
void vbs(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ uint32_t *src1, uint64_t config);
void vbrcb(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vbrcb(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint16_t dstBlockStride,
           uint16_t dstRepeatStride, uint8_t repeat);
void vbrcb(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vbrcb(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint16_t dstBlockStride,
           uint16_t dstRepeatStride, uint8_t repeat);
void vcadd(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vcadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride);
void vcadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode);
void vcadd(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config, bool MASK);
void vcadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool mode);
void vcadd(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vcadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride);
void vcadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode);
void vcadd(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config, bool MASK);
void vcadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool mode);
void vcadd(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vcadd(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vcadd(vector_s16 &dst, vector_s8 src, vector_bool mask, Literal mode);
void vcadd(vector_s32 &dst, vector_s16 src, vector_bool mask, Literal mode);
void vcadd(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal mode);
void vcadd(vector_u16 &dst, vector_u8 src, vector_bool mask, Literal mode);
void vcadd(vector_u32 &dst, vector_u16 src, vector_bool mask, Literal mode);
void vcadd(vector_u32 &dst, vector_u32 src, vector_bool mask, Literal mode);
void vcadd(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vcadd(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vcadd(vector_s16 &dst, vector_s8 src, vector_bool mask, int32_t mode);
void vcadd(vector_s32 &dst, vector_s16 src, vector_bool mask, int32_t mode);
void vcadd(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vcadd(vector_u16 &dst, vector_u8 src, vector_bool mask, int32_t mode);
void vcadd(vector_u32 &dst, vector_u16 src, vector_bool mask, int32_t mode);
void vcadd(vector_u32 &dst, vector_u32 src, vector_bool mask, int32_t mode);
void vcmax(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vcmax(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vcmax(vector_s16 &dst, vector_s16 src, vector_bool mask, Literal mode);
void vcmax(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal mode);
void vcmax(vector_u16 &dst, vector_u16 src, vector_bool mask, Literal mode);
void vcmax(vector_u32 &dst, vector_u32 src, vector_bool mask, Literal mode);
void vcmax(vector_u8 &dst, vector_u8 src, vector_bool mask, Literal mode);
void vcmax(vector_s8 &dst, vector_s8 src, vector_bool mask, Literal mode);
void vcmax(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vcmax(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vcmax(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t mode);
void vcmax(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vcmax(vector_u16 &dst, vector_u16 src, vector_bool mask, int32_t mode);
void vcmax(vector_u32 &dst, vector_u32 src, vector_bool mask, int32_t mode);
void vcmax(vector_u8 &dst, vector_u8 src, vector_bool mask, int32_t mode);
void vcmax(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t mode);
void vcmin(vector_f16 &dst, vector_f16 src, vector_bool mask, Literal mode);
void vcmin(vector_f32 &dst, vector_f32 src, vector_bool mask, Literal mode);
void vcmin(vector_u16 &dst, vector_u16 src, vector_bool mask, Literal mode);
void vcmin(vector_u32 &dst, vector_u32 src, vector_bool mask, Literal mode);
void vcmin(vector_s16 &dst, vector_s16 src, vector_bool mask, Literal mode);
void vcmin(vector_s32 &dst, vector_s32 src, vector_bool mask, Literal mode);
void vcmin(vector_u8 &dst, vector_u8 src, vector_bool mask, Literal mode);
void vcmin(vector_s8 &dst, vector_s8 src, vector_bool mask, Literal mode);
void vcmin(vector_f16 &dst, vector_f16 src, vector_bool mask, int32_t mode);
void vcmin(vector_f32 &dst, vector_f32 src, vector_bool mask, int32_t mode);
void vcmin(vector_u16 &dst, vector_u16 src, vector_bool mask, int32_t mode);
void vcmin(vector_u32 &dst, vector_u32 src, vector_bool mask, int32_t mode);
void vcmin(vector_s16 &dst, vector_s16 src, vector_bool mask, int32_t mode);
void vcmin(vector_s32 &dst, vector_s32 src, vector_bool mask, int32_t mode);
void vcmin(vector_u8 &dst, vector_u8 src, vector_bool mask, int32_t mode);
void vcmin(vector_s8 &dst, vector_s8 src, vector_bool mask, int32_t mode);
void vcbd_s162s32(__ubuf__ int32_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vcbd_s162s32(__ubuf__ int32_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_s162u32(__ubuf__ uint32_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vcbd_s162u32(__ubuf__ uint32_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_s162u8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vcbd_s162u8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_s322s16(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, uint64_t config);
void vcbd_s322s16(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_s322u16(__ubuf__ uint16_t *dst, __ubuf__ int32_t *src, uint64_t config);
void vcbd_s322u16(__ubuf__ uint16_t *dst, __ubuf__ int32_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_s322u8(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src, uint64_t config);
void vcbd_s322u8(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u162s32(__ubuf__ int32_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vcbd_u162s32(__ubuf__ int32_t *dst, __ubuf__ uint16_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u162u32(__ubuf__ uint32_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vcbd_u162u32(__ubuf__ uint32_t *dst, __ubuf__ uint16_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u162u8(__ubuf__ uint8_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vcbd_u162u8(__ubuf__ uint8_t *dst, __ubuf__ uint16_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u322s16(__ubuf__ int16_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vcbd_u322s16(__ubuf__ int16_t *dst, __ubuf__ uint32_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u322u16(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vcbd_u322u16(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u322u8(__ubuf__ uint8_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vcbd_u322u8(__ubuf__ uint8_t *dst, __ubuf__ uint32_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u82s16(__ubuf__ int16_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void vcbd_u82s16(__ubuf__ int16_t *dst, __ubuf__ uint8_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u82s32(__ubuf__ int32_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void vcbd_u82s32(__ubuf__ int32_t *dst, __ubuf__ uint8_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u82u16(__ubuf__ uint16_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void vcbd_u82u16(__ubuf__ uint16_t *dst, __ubuf__ uint8_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcbd_u82u32(__ubuf__ uint32_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void vcbd_u82u32(__ubuf__ uint32_t *dst, __ubuf__ uint8_t *src, uint8_t repeat,
                 uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                 uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vcgadd(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vcgadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride);
void vcgadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride, bool repeatStrideMode, bool strideSizeMode);
void vcgadd(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vcgadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride);
void vcgadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride, bool repeatStrideMode, bool strideSizeMode);
void vcgadd(vector_f16 &dst, vector_f16 src, vector_bool pg, int32_t mode);
void vcgadd(vector_f32 &dst, vector_f32 src, vector_bool pg, int32_t mode);
void vcgadd(vector_u16 &dst, vector_u16 src, vector_bool pg, int32_t mode);
void vcgadd(vector_s16 &dst, vector_s16 src, vector_bool pg, int32_t mode);
void vcgadd(vector_u32 &dst, vector_u32 src, vector_bool pg, int32_t mode);
void vcgadd(vector_s32 &dst, vector_s32 src, vector_bool pg, int32_t mode);
void vcgmax(vector_f16 &dst, vector_f16 src, vector_bool pg, int32_t mode);
void vcgmax(vector_f32 &dst, vector_f32 src, vector_bool pg, int32_t mode);
void vcgmax(vector_u16 &dst, vector_u16 src, vector_bool pg, int32_t mode);
void vcgmax(vector_s16 &dst, vector_s16 src, vector_bool pg, int32_t mode);
void vcgmax(vector_u32 &dst, vector_u32 src, vector_bool pg, int32_t mode);
void vcgmax(vector_s32 &dst, vector_s32 src, vector_bool pg, int32_t mode);
void vcgmin(vector_f16 &dst, vector_f16 src, vector_bool pg, int32_t mode);
void vcgmin(vector_f32 &dst, vector_f32 src, vector_bool pg, int32_t mode);
void vcgmin(vector_u16 &dst, vector_u16 src, vector_bool pg, int32_t mode);
void vcgmin(vector_s16 &dst, vector_s16 src, vector_bool pg, int32_t mode);
void vcgmin(vector_u32 &dst, vector_u32 src, vector_bool pg, int32_t mode);
void vcgmin(vector_s32 &dst, vector_s32 src, vector_bool pg, int32_t mode);
void vcgadd(vector_f16 &dst, vector_f16 src, vector_bool pg, Literal mode);
void vcgadd(vector_f32 &dst, vector_f32 src, vector_bool pg, Literal mode);
void vcgadd(vector_u16 &dst, vector_u16 src, vector_bool pg, Literal mode);
void vcgadd(vector_s16 &dst, vector_s16 src, vector_bool pg, Literal mode);
void vcgadd(vector_u32 &dst, vector_u32 src, vector_bool pg, Literal mode);
void vcgadd(vector_s32 &dst, vector_s32 src, vector_bool pg, Literal mode);
void vcgmax(vector_f16 &dst, vector_f16 src, vector_bool pg, Literal mode);
void vcgmax(vector_f32 &dst, vector_f32 src, vector_bool pg, Literal mode);
void vcgmax(vector_u16 &dst, vector_u16 src, vector_bool pg, Literal mode);
void vcgmax(vector_s16 &dst, vector_s16 src, vector_bool pg, Literal mode);
void vcgmax(vector_u32 &dst, vector_u32 src, vector_bool pg, Literal mode);
void vcgmax(vector_s32 &dst, vector_s32 src, vector_bool pg, Literal mode);
void vcgmin(vector_f16 &dst, vector_f16 src, vector_bool pg, Literal mode);
void vcgmin(vector_f32 &dst, vector_f32 src, vector_bool pg, Literal mode);
void vcgmin(vector_u16 &dst, vector_u16 src, vector_bool pg, Literal mode);
void vcgmin(vector_s16 &dst, vector_s16 src, vector_bool pg, Literal mode);
void vcgmin(vector_u32 &dst, vector_u32 src, vector_bool pg, Literal mode);
void vcgmin(vector_s32 &dst, vector_s32 src, vector_bool pg, Literal mode);
void vcpadd(vector_f16 &dst, vector_f16 src, vector_bool pg, Literal mode);
void vcpadd(vector_f32 &dst, vector_f32 src, vector_bool pg, Literal mode);
void vcpadd(vector_f16 &dst, vector_f16 src, vector_bool pg, int32_t mode);
void vcpadd(vector_f32 &dst, vector_f32 src, vector_bool pg, int32_t mode);
void vcgmax(__ubuf__ half *dst, __ubuf__ half *src, uint64_t confg);
void vcgmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride);
void vcgmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride, bool repeatStrideMode, bool strideSizeMode);
void vcgmax(__ubuf__ float *dst, __ubuf__ float *src, uint64_t confg);
void vcgmax(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride);
void vcgmax(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride, bool repeatStrideMode, bool strideSizeMode);
void vcgmin(__ubuf__ half *dst, __ubuf__ half *src, uint64_t confg);
void vcgmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride);
void vcgmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride, bool repeatStrideMode, bool strideSizeMode);
void vcgmin(__ubuf__ float *dst, __ubuf__ float *src, uint64_t confg);
void vcgmin(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride);
void vcgmin(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
            uint16_t src0Stride, uint16_t src1Stride, bool repeatStrideMode, bool strideSizeMode);
void vci(__ubuf__ half *dst, half src, uint64_t config);
void vci(__ubuf__ half *dst, half src, uint8_t repeat, uint16_t dstStride, uint8_t srcStride,
         bool repeatStrideMode, bool strideSizeMode);
void vci(__ubuf__ float *dst, float src, uint64_t config);
void vci(__ubuf__ float *dst, float src, uint8_t repeat, uint16_t dstStride, uint8_t srcStride,
         bool repeatStrideMode, bool strideSizeMode);
void vci(__ubuf__ int16_t *dst, int16_t src, uint64_t config);
void vci(__ubuf__ int16_t *dst, int16_t src, uint8_t repeat, uint16_t dstStride, uint8_t srcStride,
         bool repeatStrideMode, bool strideSizeMode);
void vci(__ubuf__ int32_t *dst, int32_t src, uint64_t config);
void vci(__ubuf__ int32_t *dst, int32_t src, uint8_t repeat, uint16_t dstStride, uint8_t srcStride,
         bool repeatStrideMode, bool strideSizeMode);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config, bool order);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode, bool order);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config, Order_t order);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, Order_t order);
void vcmax(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config, bool order);
void vcmax(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode, bool order);
void vcmax(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config, Order_t order);
void vcmax(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, Order_t order);
void vcmax(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint64_t config, bool order);
void vcmax(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode, bool order);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config, bool MASK);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode, bool MASK);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config, Order_t order);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, Order_t order);
void vcmin(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config, bool MASK);
void vcmin(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode, bool MASK);
void vcmin(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config, Order_t order);
void vcmin(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, Order_t order);
void vcmin(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint64_t config, bool MASK);
void vcmin(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t repeat, uint16_t dstRepeatStride,
           uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
           bool strideSizeMode, bool MASK);
void vcmp_eq(__ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmp_eq(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_eq(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_eq(__ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmp_eq(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_eq(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_eq(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint64_t config);
void vcmp_eq(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_eq(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vcmp_ge(__ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmp_ge(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_ge(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_ge(__ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmp_ge(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_ge(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_ge(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint64_t config);
void vcmp_ge(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_ge(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vcmp_gt(__ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmp_gt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_gt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_gt(__ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmp_gt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_gt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_gt(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint64_t config);
void vcmp_gt(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_gt(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vcmp_le(__ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmp_le(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_le(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_le(__ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmp_le(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_le(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_le(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint64_t config);
void vcmp_le(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_le(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vcmp_lt(__ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmp_lt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_lt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_lt(__ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmp_lt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_lt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_lt(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint64_t config);
void vcmp_lt(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_lt(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vcmp_ne(__ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmp_ne(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_ne(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_ne(__ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmp_ne(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_ne(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dstBlockStride,
             uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
             bool strideSizeMode);
void vcmp_ne(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint64_t config);
void vcmp_ne(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmp_ne(__ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
             uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
             uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1,
              uint64_t config);
void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_ge(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmpv_ge(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_ge(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmpv_ge(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_gt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmpv_gt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_gt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmpv_gt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_le(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmpv_le(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_le(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmpv_le(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_lt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmpv_lt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_lt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmpv_lt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_ne(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vcmpv_ne(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpv_ne(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vcmpv_ne(__ubuf__ uint8_t *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint64_t config);
void vcmpvs_eq(__ubuf__ uint8_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t src0BlockStride, uint16_t dstRepeatStride,
               uint16_t src0RepeatStride);
void vcmpvs_ge(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vcmpvs_ge(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_ge(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_ge(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vcmpvs_ge(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_ge(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_gt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vcmpvs_gt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_gt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_gt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vcmpvs_gt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_gt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_le(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vcmpvs_le(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_le(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_le(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vcmpvs_le(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_le(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_lt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vcmpvs_lt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_lt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_lt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vcmpvs_lt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_lt(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_ne(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vcmpvs_ne(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_ne(__ubuf__ uint8_t *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vcmpvs_ne(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vcmpvs_ne(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
               uint8_t srcRepeatStride);
void vcmpvs_ne(__ubuf__ uint8_t *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
               uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
               uint16_t srcRepeatStride);
void vconcat(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vconcat(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint8_t stride);
void vconcat(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
             uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconcat(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
             uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vconcat(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vconcat(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint8_t stride);
void vconcat(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
             uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconcat(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
             uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
             bool repeatStrideMode, bool strideSizeMode);
void vconv_bf162f32(__ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_bf162f32(__ubuf__ float *dst, __ubuf__ bfloat16_t *src, uint64_t config);
void vconv_bf162s32a(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_bf162s32a(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint64_t config);
void vconv_bf162s32c(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_bf162s32c(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint64_t config);
void vconv_bf162s32f(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_bf162s32f(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint64_t config);
void vconv_bf162s32r(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_bf162s32r(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint64_t config);
void vconv_bf162s32z(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_bf162s32z(__ubuf__ int32_t *dst, __ubuf__ bfloat16_t *src, uint64_t config);
void vconv_deq(__ubuf__ half *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_deq(__ubuf__ half *dst, __ubuf__ int32_t *src, uint8_t repeat, uint16_t dstBlockStride,
               uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconv_deq(__ubuf__ half *dst, __ubuf__ int32_t *src, uint8_t repeat, uint16_t dstBlockStride,
               uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
               bool repeatStrideMode, bool strideSizeMode);
void vconv_deqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint64_t config, bool halfBlock);
void vconv_deqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                     uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode,
                     bool halfBlock);
void vconv_deqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint64_t config, bool halfBlock);
void vconv_deqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                     uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode,
                     bool halfBlock);
void vconv_deqs162b8h(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                      uint8_t srcRepeatStride);
void vconv_deqs162b8h(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_deqs162b8h(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                      uint8_t srcRepeatStride);
void vconv_deqs162b8h(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_deqs162b8l(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                      uint8_t srcRepeatStride);
void vconv_deqs162b8l(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_deqs162b8l(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                      uint8_t srcRepeatStride);
void vconv_deqs162b8l(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_f162f32(__ubuf__ float *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162f32(__ubuf__ float *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconv_f162f32(__ubuf__ float *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                   bool repeatStrideMode, bool strideSizeMode);
void vconv_f162f32(__ubuf__ float *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s16a(__ubuf__ int16_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s16a(__ubuf__ int16_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s16c(__ubuf__ int16_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s16c(__ubuf__ int16_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s16f(__ubuf__ int16_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s16f(__ubuf__ int16_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s16r(__ubuf__ int16_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s16r(__ubuf__ int16_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s16r(__ubuf__ int16_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s16z(__ubuf__ int16_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s16z(__ubuf__ int16_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s32a(__ubuf__ int32_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s32a(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f162s32a(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s32a(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s32c(__ubuf__ int32_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s32c(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f162s32c(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s32c(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s32f(__ubuf__ int32_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s32f(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f162s32f(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s32f(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s32r(__ubuf__ int32_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s32r(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f162s32r(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s32r(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s32z(__ubuf__ int32_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s32z(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f162s32z(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s32z(__ubuf__ int32_t *dst, __ubuf__ half *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f162s4(__ubuf__ void *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s4(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                  bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s4(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s4a(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s4a(__ubuf__ void *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s4c(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s4c(__ubuf__ void *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s4f(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s4f(__ubuf__ void *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s4r(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s4r(__ubuf__ void *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s4z(__ubuf__ void *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s4z(__ubuf__ void *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                  bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f162s8a(__ubuf__ int8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s8a(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162s8a(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s8a(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162s8c(__ubuf__ int8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s8c(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162s8c(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s8c(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162s8f(__ubuf__ int8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s8f(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162s8f(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s8f(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162s8r(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162s8r(__ubuf__ int8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s8z(__ubuf__ int8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162s8z(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162s8z(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162s8z(__ubuf__ int8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162u8(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162u8(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride);
void vconv_f162u8(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162u8(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                  uint16_t srcRepeatStride);
void vconv_f162u8a(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162u8a(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162u8a(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162u8a(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162u8c(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162u8c(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162u8c(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162u8c(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162u8f(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162u8f(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162u8f(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162u8f(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162u8r(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f162u8r(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162u8z(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint64_t config);
void vconv_f162u8z(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_f162u8z(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f162u8z(__ubuf__ uint8_t *dst, __ubuf__ half *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_f322bf16a(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_f322bf16a(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322bf16c(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_f322bf16c(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322bf16f(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_f322bf16f(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322bf16r(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_f322bf16r(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322bf16z(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint8_t repeat,
                     uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                     uint16_t srcRepeatStride);
void vconv_f322bf16z(__ubuf__ bfloat16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                   bool repeatStrideMode, bool strideSizeMode);
void vconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
                   uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_f322f16a(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f16a(__ubuf__ half *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f16c(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f16c(__ubuf__ half *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f16f(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f16f(__ubuf__ half *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f16o(__ubuf__ half *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f16o(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322f16o(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322f16o(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f16r(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f16r(__ubuf__ half *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f16z(__ubuf__ half *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f16z(__ubuf__ half *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f32a(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f32a(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f32c(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f32c(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f32f(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f32f(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f32r(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f32r(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322f32z(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322f32z(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s16a(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s16a(__ubuf__ int16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s16c(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s16c(__ubuf__ int16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s16f(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s16f(__ubuf__ int16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s16r(__ubuf__ int16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s16r(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322s16r(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322s16r(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s16z(__ubuf__ int16_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s16z(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322s16z(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322s16z(__ubuf__ int16_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s32a(__ubuf__ int32_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s32a(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322s32a(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322s32a(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s32c(__ubuf__ int32_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s32c(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322s32c(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322s32c(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s32f(__ubuf__ int32_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s32f(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322s32f(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322s32f(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s32r(__ubuf__ int32_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s32r(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322s32r(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322s32r(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s32z(__ubuf__ int32_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s32z(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride);
void vconv_f322s32z(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                    uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_f322s32z(__ubuf__ int32_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s64a(__ubuf__ int64_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s64a(__ubuf__ int64_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s64c(__ubuf__ int64_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s64c(__ubuf__ int64_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s64f(__ubuf__ int64_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s64f(__ubuf__ int64_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s64r(__ubuf__ int64_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s64r(__ubuf__ int64_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_f322s64z(__ubuf__ int64_t *dst, __ubuf__ float *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_f322s64z(__ubuf__ int64_t *dst, __ubuf__ float *src, uint64_t config);
void vconv_s162f16(__ubuf__ half *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_s162f16(__ubuf__ half *dst, __ubuf__ int16_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_s162f16(__ubuf__ half *dst, __ubuf__ int16_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_s162f16a(__ubuf__ half *dst, __ubuf__ int16_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s162f16a(__ubuf__ half *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_s162f16c(__ubuf__ half *dst, __ubuf__ int16_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s162f16c(__ubuf__ half *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_s162f16f(__ubuf__ half *dst, __ubuf__ int16_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s162f16f(__ubuf__ half *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_s162f16r(__ubuf__ half *dst, __ubuf__ int16_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s162f16r(__ubuf__ half *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_s162f16z(__ubuf__ half *dst, __ubuf__ int16_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s162f16z(__ubuf__ half *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_s162f32(__ubuf__ float *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_s162f32(__ubuf__ float *dst, __ubuf__ int16_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_s162f32(__ubuf__ float *dst, __ubuf__ int16_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_s322f32(__ubuf__ float *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s322f32(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride);
void vconv_s322f32(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                   uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_s322f32(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_s322f32a(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s322f32a(__ubuf__ float *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s322f32c(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s322f32c(__ubuf__ float *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s322f32f(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s322f32f(__ubuf__ float *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s322f32r(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s322f32r(__ubuf__ float *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s322f32z(__ubuf__ float *dst, __ubuf__ int32_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s322f32z(__ubuf__ float *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s322s16(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_s322s16(__ubuf__ int16_t *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s322s64(__ubuf__ int64_t *dst, __ubuf__ int32_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_s322s64(__ubuf__ int64_t *dst, __ubuf__ int32_t *src, uint64_t config);
void vconv_s42f16(__ubuf__ half *dst, __ubuf__ void *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconv_s42f16(__ubuf__ half *dst, __ubuf__ void *src, uint64_t config);
void vconv_s642f32a(__ubuf__ float *dst, __ubuf__ int64_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s642f32a(__ubuf__ float *dst, __ubuf__ int64_t *src, uint64_t config);
void vconv_s642f32c(__ubuf__ float *dst, __ubuf__ int64_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s642f32c(__ubuf__ float *dst, __ubuf__ int64_t *src, uint64_t config);
void vconv_s642f32f(__ubuf__ float *dst, __ubuf__ int64_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s642f32f(__ubuf__ float *dst, __ubuf__ int64_t *src, uint64_t config);
void vconv_s642f32r(__ubuf__ float *dst, __ubuf__ int64_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s642f32r(__ubuf__ float *dst, __ubuf__ int64_t *src, uint64_t config);
void vconv_s642f32z(__ubuf__ float *dst, __ubuf__ int64_t *src, uint8_t repeat,
                    uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                    uint16_t srcRepeatStride);
void vconv_s642f32z(__ubuf__ float *dst, __ubuf__ int64_t *src, uint64_t config);
void vconv_s642s32(__ubuf__ int32_t *dst, __ubuf__ int64_t *src, uint8_t repeat,
                   uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                   uint16_t srcRepeatStride);
void vconv_s642s32(__ubuf__ int32_t *dst, __ubuf__ int64_t *src, uint64_t config);
void vconv_s82f16(__ubuf__ half *dst, __ubuf__ int8_t *src, uint64_t config);
void vconv_s82f16(__ubuf__ half *dst, __ubuf__ int8_t *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vconv_s82f16(__ubuf__ half *dst, __ubuf__ int8_t *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                  bool repeatStrideMode, bool strideSizeMode);
void vconv_s82f16(__ubuf__ half *dst, __ubuf__ int8_t *src, uint8_t repeat, uint16_t dstBlockStride,
                  uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vconv_u82f16(__ubuf__ half *dst, __ubuf__ uint8_t *src, uint64_t config);
void vconv_u82f16(__ubuf__ half *dst, __ubuf__ uint8_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride);
void vconv_u82f16(__ubuf__ half *dst, __ubuf__ uint8_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                  uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vconv_u82f16(__ubuf__ half *dst, __ubuf__ uint8_t *src, uint8_t repeat,
                  uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
                  uint16_t srcRepeatStride);
void vconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint64_t config, bool halfBlock);
void vconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                      uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode,
                      bool halfBlock);
void vconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint64_t config,
                      bool halfBlock);
void vconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                      uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                      uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode,
                      bool halfBlock);
void vconv_vdeqs162b8h(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                       uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                       uint8_t srcRepeatStride);
void vconv_vdeqs162b8h(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_vdeqs162b8h(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                       uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                       uint8_t srcRepeatStride);
void vconv_vdeqs162b8h(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_vdeqs162b8l(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                       uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                       uint8_t srcRepeatStride);
void vconv_vdeqs162b8l(__ubuf__ int8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vconv_vdeqs162b8l(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint8_t repeat,
                       uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
                       uint8_t srcRepeatStride);
void vconv_vdeqs162b8l(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vcvt(vector_f16 &dst, vector_u8 src, vector_bool preg, Literal tag0);
void vcvt(vector_u16 &dst, vector_u8 src, vector_bool preg, Literal tag0);
void vcvt(vector_f16 &dst, vector_s8 src, vector_bool preg, Literal tag0);
void vcvt(vector_s16 &dst, vector_s8 src, vector_bool preg, Literal tag0);
void vcvt(vector_f32 &dst, vector_f16 src, vector_bool preg, Literal tag0);
void vcvt(vector_f32 &dst, vector_f16 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_s32 &dst, vector_f16 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_f32 &dst, vector_bf16 src, vector_bool preg, Literal tag0);
void vcvt(vector_s32 &dst, vector_bf16 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_f32 &dst, vector_s16 src, vector_bool preg, Literal tag0);
void vcvt(vector_u32 &dst, vector_u16 src, vector_bool preg, Literal tag0);
void vcvt(vector_u32 &dst, vector_u8 src, vector_bool preg, Literal tag0);
void vcvt(vector_s32 &dst, vector_s8 src, vector_bool preg, Literal tag0);
void vcvt(vector_s32 &dst, vector_s16 src, vector_bool preg, Literal tag0);
void vcvt(vector_u32 &dst, vector_s16 src, vector_bool preg, Literal tag0);
void vcvt(vector_s64 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_s64 &dst, vector_s32 src, vector_bool preg, Literal tag0);
void vcvt(vector_u8 &dst, vector_f16 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_s8 &dst, vector_f16 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_f16 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_f16 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2, Literal tag3);
void vcvt(vector_bf16 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_s16 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_f8e5m2 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_f32 &dst, vector_f8e5m2 src, vector_bool preg, Literal tag0);
void vcvt(vector_f8e4m3 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_f32 &dst, vector_f8e4m3 src, vector_bool preg, Literal tag0);
void vcvt(vector_hif8 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_hif8 &dst, vector_f16 src, vector_bool preg, Literal tag0, Literal tag1,
          Literal tag2);
void vcvt(vector_f32 &dst, vector_hif8 src, vector_bool preg, Literal tag0);
void vcvt(vector_f16 &dst, vector_hif8 src, vector_bool preg, Literal tag0);
void vcvt(vector_u8 &dst, vector_u16 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_u8 &dst, vector_s16 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_u8 &dst, vector_u32 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_u8 &dst, vector_s32 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_u16 &dst, vector_u32 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_s16 &dst, vector_u32 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_u16 &dst, vector_s32 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_s16 &dst, vector_s32 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_s32 &dst, vector_s64 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_f32 &dst, vector_s64 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_s16 &dst, vector_f16 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_f16 &dst, vector_s16 src, vector_bool preg, Literal tag0);
void vcvt(vector_s32 &dst, vector_f32 src, vector_bool preg, Literal tag0, Literal tag1);
void vcvt(vector_f32 &dst, vector_s32 src, vector_bool preg, Literal tag0);
void vcvt(vector_u8 &dst, vector_u16 src, int32_t tag0, int32_t tag1);
void vcvt(vector_u8 &dst, vector_s16 src, int32_t tag0, int32_t tag1);
void vcvt(vector_u16 &dst, vector_u32 src, int32_t tag0, int32_t tag1);
void vcvt(vector_s16 &dst, vector_u32 src, int32_t tag0, int32_t tag1);
void vcvt(vector_u16 &dst, vector_s32 src, int32_t tag0, int32_t tag1);
void vcvt(vector_s16 &dst, vector_s32 src, int32_t tag0, int32_t tag1);
void vcvt(vector_u16 &dst, vector_u8 src, int32_t tag0);
void vcvt(vector_s16 &dst, vector_s8 src, int32_t tag0);
void vcvt(vector_u32 &dst, vector_u16 src, int32_t tag0);
void vcvt(vector_u32 &dst, vector_s16 src, int32_t tag0);
void vcvt(vector_s32 &dst, vector_s16 src, int32_t tag0);
void vcvt(vector_u32 &dst, vector_u8 src, int32_t tag0);
void vcvt(vector_s32 &dst, vector_s8 src, int32_t tag0);
void vcvt(vector_u8 &dst, vector_u32 src, int32_t tag0, int32_t tag1);
void vcvt(vector_u8 &dst, vector_s32 src, int32_t tag0, int32_t tag1);
void vcvt(vector_u16 &dst, vector_u8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_s16 &dst, vector_s8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_u32 &dst, vector_u16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_u32 &dst, vector_s16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_s32 &dst, vector_s16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_s64 &dst, vector_s32 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f32 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f32 &dst, vector_bf16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f16 &dst, vector_u8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f16 &dst, vector_s8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f32 &dst, vector_s16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_u8 &dst, vector_u16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_u8 &dst, vector_s16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_u16 &dst, vector_u32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_s16 &dst, vector_u32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_u16 &dst, vector_s32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_s16 &dst, vector_s32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_s32 &dst, vector_s64 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_u32 &dst, vector_u8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_s32 &dst, vector_s8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_u8 &dst, vector_u32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_u8 &dst, vector_s32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_s16 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_u8 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_s8 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_s32 &dst, vector_bf16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_s64 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_f16 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_bf16 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_s32 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_s16 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_s32 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_f32 &dst, vector_s64 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_f16 &dst, vector_s16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f32 &dst, vector_s32 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_bf16 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f16 &dst, vector_hif8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f32 &dst, vector_hif8 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_hif8 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_hif8 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_f8e4m3 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_f8e5m2 &dst, vector_f32 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2, int32_t tag3);
void vcvt(vector_f32 &dst, vector_f8e4m3 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f32 &dst, vector_f8e5m2 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f16 &dst, vector_bf16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_bf16 &dst, vector_f4e2m1 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_bf16 &dst, vector_f4e1m2 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt(vector_f4e2m1 &dst, vector_bf16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt(vector_f4e1m2 &dst, vector_bf16 src, vector_bool preg, int32_t tag0, int32_t tag1,
          int32_t tag2);
void vcvt_f162s4(vector_s4x2 &dst, vector_f16 src, vector_bool preg, int32_t tag0, int32_t tag1,
                 int32_t tag2, int32_t tag3);
void vcvt_s42f16(vector_f16 &dst, vector_s4x2 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt_s42bf16(vector_bf16 &dst, vector_s4x2 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vcvt_s42s16(vector_s16 &dst, vector_s4x2 src, vector_bool preg, int32_t tag0, int32_t tag1);
void vtrc(vector_f32 &dst, vector_f32 src, Literal mode, vector_bool preg);
void vtrc(vector_f32 &dst, vector_f32 src, int32_t mode, vector_bool tag0, int32_t preg);
void vtrc(vector_f16 &dst, vector_f16 src, int32_t mode, vector_bool tag0, int32_t preg);
void vtrc(vector_bf16 &dst, vector_bf16 src, int32_t mode, vector_bool tag0, int32_t preg);
void vcopy(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vcopy(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint8_t repeat, uint16_t dstStride,
           uint16_t srcStride, uint16_t dstRepeatSize, uint16_t srcRepeatSize);
void vcopy(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint64_t config);
void vcopy(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint8_t repeat, uint16_t dstStride,
           uint16_t srcStride, uint16_t dstRepeatSize, uint16_t srcRepeatSize);
void vcopy(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vcopy(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t repeat, uint16_t dstStride,
           uint16_t srcStride, uint16_t dstRepeatSize, uint16_t srcRepeatSize);
void vcopy(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vcopy(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t repeat, uint16_t dstStride,
           uint16_t srcStride, uint16_t dstRepeatSize, uint16_t srcRepeatSize);
void vcpadd(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vcpadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint16_t srcRepeatStride);
void vcpadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
            bool strideSizeMode);
void vcpadd(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vcpadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint16_t srcRepeatStride);
void vcpadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint16_t srcRepeatStride, bool repeatStrideMode,
            bool strideSizeMode);
void vdiv(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vdiv(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vdiv(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vdiv(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vdiv(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vdiv(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vdp(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vdp(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint16_t numPixel,
         uint8_t numMaxDisparity, uint16_t offsetPixel, uint16_t dynamicAddrRange,
         bool isBeginPixel, bool isPathReverse, bool pathMode);
void vector_dup(__ubuf__ half *dst, half src, uint64_t config);
void vector_dup(__ubuf__ half *dst, half src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vector_dup(__ubuf__ half *dst, half src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                bool repeatStrideMode, bool strideSizeMode);
void vector_dup(__ubuf__ half *dst, half src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vector_dup(__ubuf__ float *dst, float src, uint64_t config);
void vector_dup(__ubuf__ float *dst, float src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vector_dup(__ubuf__ float *dst, float src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                bool repeatStrideMode, bool strideSizeMode);
void vector_dup(__ubuf__ float *dst, float src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vector_dup(__ubuf__ int16_t *dst, int16_t src, uint64_t config);
void vector_dup(__ubuf__ int16_t *dst, int16_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vector_dup(__ubuf__ int16_t *dst, int16_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                bool repeatStrideMode, bool strideSizeMode);
void vector_dup(__ubuf__ int16_t *dst, int16_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vector_dup(__ubuf__ int32_t *dst, int32_t src, uint64_t config);
void vector_dup(__ubuf__ int32_t *dst, int32_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vector_dup(__ubuf__ int32_t *dst, int32_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                bool repeatStrideMode, bool strideSizeMode);
void vector_dup(__ubuf__ int32_t *dst, int32_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vector_dup(__ubuf__ uint16_t *dst, uint16_t src, uint64_t config);
void vector_dup(__ubuf__ uint16_t *dst, uint16_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vector_dup(__ubuf__ uint16_t *dst, uint16_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                bool repeatStrideMode, bool strideSizeMode);
void vector_dup(__ubuf__ uint16_t *dst, uint16_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vector_dup(__ubuf__ uint32_t *dst, uint32_t src, uint64_t config);
void vector_dup(__ubuf__ uint32_t *dst, uint32_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vector_dup(__ubuf__ uint32_t *dst, uint32_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
                bool repeatStrideMode, bool strideSizeMode);
void vector_dup(__ubuf__ uint32_t *dst, uint32_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vector_dup(__ubuf__ bfloat16_t *dst, bfloat16_t src, uint64_t config);
void vector_dup(__ubuf__ bfloat16_t *dst, bfloat16_t src, uint8_t repeat, uint16_t dstBlockStride,
                uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vexp(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vexp(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vexp(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vexp(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vexp(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vexp(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vexp(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vexp(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vextract(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vextract(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint8_t stride);
void vextract(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vextract(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vextract(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vextract(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint8_t stride);
void vextract(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vextract(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vgather(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vgather(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint32_t offsetAddr,
             bool repeatStrideMode);
void vgather(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint32_t offsetAddr,
             uint16_t dstRepeatStride, uint8_t repeat);
void vgather(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vgather(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t offsetAddr,
             bool repeatStrideMode);
void vgather(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t offsetAddr,
             uint16_t dstRepeatStride, uint8_t repeat);
void vgatherb(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vgatherb(__ubuf__ uint16_t *dst, __ubuf__ uint32_t *src, uint32_t offsetAddr,
              uint16_t dstRepeatStride, uint8_t dstBlockStride, uint8_t repeat);
void vgatherb(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vgatherb(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t offsetAddr,
              uint16_t dstRepeatStride, uint8_t dstBlockStride, uint8_t repeat);
void viou(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void viou(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void viou(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void viou(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void viou(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void viou(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vld_va_reg(ub_addr8_t dst, __ubuf__ uint64_t *src, vpart_t config);
void vln(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vln(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
         uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vln(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
         uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
         bool repeatStrideMode, bool strideSizeMode);
void vln(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
         uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vln(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vln(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
         uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vln(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
         uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
         bool repeatStrideMode, bool strideSizeMode);
void vln(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
         uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vlrelu(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vlrelu(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
            uint16_t dstBlockStride, uint16_t src0BlockStride, uint8_t dstRepeatStride,
            uint8_t src0RepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vlrelu(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
            uint16_t dstBlockStride, uint16_t src0BlockStride, uint16_t dstRepeatStride,
            uint16_t src0RepeatStride);
void vlrelu(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vlrelu(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
            uint16_t dstBlockStride, uint16_t src0BlockStride, uint8_t dstRepeatStride,
            uint8_t src0RepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vlrelu(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
            uint16_t dstBlockStride, uint16_t src0BlockStride, uint16_t dstRepeatStride,
            uint16_t src0RepeatStride);
void vmadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vmadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vmadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
           uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
           uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vmadd(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vmadd(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vmula(vector_s32 &dst, vector_s32 src0, vector_s32 src1, vector_bool mask, Literal mode);
void vmula(vector_u32 &dst, vector_u32 src0, vector_u32 src1, vector_bool mask, Literal mode);
void vmula(vector_f32 &dst, vector_f32 src0, vector_f32 src1, vector_bool mask, Literal mode);
void vmula(vector_s16 &dst, vector_s16 src0, vector_s16 src1, vector_bool mask, Literal mode);
void vmula(vector_u16 &dst, vector_u16 src0, vector_u16 src1, vector_bool mask, Literal mode);
void vmula(vector_f16 &dst, vector_f16 src0, vector_f16 src1, vector_bool mask, Literal mode);
void vmaddrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmaddrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
               uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
               uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmaddrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
               uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
               uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
               bool repeatStrideMode, bool strideSizeMode);
void vmaddrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vmaddrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
               uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
               uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmaddrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
               uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
               uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
               bool repeatStrideMode, bool strideSizeMode);
void vmax(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmax(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmax(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmax(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vmax(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmax(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmax(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vmax(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmax(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmax(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint64_t config);
void vmax(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmax(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmaxs(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vmaxs(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmaxs(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmaxs(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vmaxs(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmaxs(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmaxs(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint64_t config);
void vmaxs(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmaxs(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmaxs(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint64_t config);
void vmaxs(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmaxs(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmergech(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void vmergech(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vmergech(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vmergech(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vmergech(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vmergech(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
              uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vmin(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmin(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmin(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmin(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vmin(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmin(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmin(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vmin(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmin(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmin(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint64_t config);
void vmin(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmin(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmins(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vmins(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmins(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmins(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vmins(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmins(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmins(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint64_t config);
void vmins(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmins(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmins(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint64_t config);
void vmins(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmins(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmla(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmla(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmla(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmla(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vmla(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmla(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmla(__ubuf__ float *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmla(__ubuf__ float *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmla(__ubuf__ float *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmrgsort4(__ubuf__ half *const arg0, __ubuf__ half **const arg1, uint64_t arg2);
void vmrgsort4(__ubuf__ float *const arg0, __ubuf__ float **const arg1, uint64_t arg2);
void vmrgsort4(__ubuf__ half *dst, __ubuf__ half **src0, uint64_t src1, uint64_t config);
void vmrgsort4(__ubuf__ float *dst, __ubuf__ float **src0, uint64_t src1, uint64_t config);
void vmul(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmul(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmul(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmul(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vmul(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmul(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmul(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vmul(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmul(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmul(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint64_t config);
void vmul(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmul(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vmulconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                     uint64_t config);
void vmulconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
                     uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
                     uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vmulconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
                     uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
                     uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
                     bool repeatStrideMode, bool strideSizeMode);
void vmulconv_f162u8(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                     uint64_t config);
void vmulconv_f162u8(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                     uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                     uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                     uint8_t src1RepeatStride);
void vmulconv_f162u8(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                     uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                     uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                     uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmuls(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint64_t config);
void vmuls(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vmuls(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmuls(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmuls(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint64_t config);
void vmuls(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vmuls(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmuls(__ubuf__ float *dst, __ubuf__ float *src0, float src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmuls(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint64_t config);
void vmuls(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vmuls(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmuls(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, int16_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vmuls(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint64_t config);
void vmuls(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride);
void vmuls(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
           uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vmuls(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, int32_t src1, uint8_t repeat,
           uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
           uint16_t srcRepeatStride);
void vnot(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint64_t config);
void vnot(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vnot(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vnot(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vnot(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vnot(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vnot(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vnot(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vor(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vor(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
         uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
         uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vor(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
         uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
         uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
         bool repeatStrideMode, bool strideSizeMode);
void vor(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint64_t config);
void vor(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
         uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
         uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vor(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1, uint8_t repeat,
         uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
         uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
         bool repeatStrideMode, bool strideSizeMode);
void vpadding(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vpadding(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint8_t repeat,
              uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
              uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode, uint8_t padMode,
              bool padSide);
void vpadding(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vpadding(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t repeat,
              uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
              uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode, uint8_t padMode,
              bool padSide);
void vrec(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vrec(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrec(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vrec(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vrec(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vrec(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrec(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vrec(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
          uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vreduce(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1,
             uint64_t config);
void vreduce(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1,
             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
             uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
             uint8_t src1RepeatStride);
void vreduce(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1,
             uint8_t repeat, uint8_t src0BlockStride, uint8_t patternMode, uint8_t src0RepeatStride,
             uint8_t src1RepeatStride);
void vreduce(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
             uint64_t config);
void vreduce(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
             uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
             uint8_t src1RepeatStride);
void vreduce(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
             uint8_t repeat, uint8_t src0BlockStride, uint8_t patternMode, uint8_t src0RepeatStride,
             uint8_t src1RepeatStride);
void vreducev2(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1,
               uint64_t config);
void vreducev2(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src0, __ubuf__ uint16_t *src1,
               uint16_t repeat, uint8_t src0BlockStride, uint8_t patternMode,
               uint16_t src0RepeatStride, uint8_t src1RepeatStride);
void vreducev2(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
               uint64_t config);
void vreducev2(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src0, __ubuf__ uint32_t *src1,
               uint16_t repeat, uint8_t src0BlockStride, uint8_t patternMode,
               uint16_t src0RepeatStride, uint8_t src1RepeatStride);
void vrelu(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vrelu(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrelu(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vrelu(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vrelu(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vrelu(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrelu(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vrelu(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vrelu(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint64_t config);
void vrelu(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrelu(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vrelu(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vrpac(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vrpac(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrpac(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vrpac(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vrpac(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrpac(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vrsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vrsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
            bool repeatStrideMode, bool strideSizeMode);
void vrsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vrsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vrsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vrsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
            bool repeatStrideMode, bool strideSizeMode);
void vrsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
            uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vscatter(__ubuf__ uint32_t *dst, __ubuf__ uint16_t *src, uint64_t config);
void vscatter(__ubuf__ uint32_t *dst, __ubuf__ uint16_t *src, uint32_t offset, bool strideSizeMode);
void vscatter(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint64_t config);
void vscatter(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t offset, bool strideSizeMode);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ void *src1, uint64_t config);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ void *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ void *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode, bool repeatStrideMode, bool strideSizeMode);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ void *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode, bool repeatStrideMode, bool strideSizeMode);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ void *src1, uint64_t config);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ void *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ void *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode, bool repeatStrideMode, bool strideSizeMode);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ void *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode, bool repeatStrideMode, bool strideSizeMode);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          uint8_t selectMode);
void vshl(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint32_t shlDistance, uint64_t config);
void vshl(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride);
void vshl(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vshl(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride);
void vshl(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint32_t shlDistance, uint64_t config);
void vshl(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride);
void vshl(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vshl(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride);
void vshl(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint32_t shlDistance, uint64_t config);
void vshl(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride);
void vshl(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vshl(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride);
void vshl(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t shlDistance, uint64_t config);
void vshl(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride);
void vshl(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode);
void vshl(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t shlDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride);
void vshr(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int32_t shrDistance, uint64_t config,
          bool round);
void vshr(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode, bool round);
void vshr(__ubuf__ int16_t *dst, __ubuf__ int16_t *src, int32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride, bool round);
void vshr(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t shrDistance, uint64_t config,
          bool round);
void vshr(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode, bool round);
void vshr(__ubuf__ int32_t *dst, __ubuf__ int32_t *src, int32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride, bool round);
void vshr(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint32_t shrDistance, uint64_t config,
          bool round);
void vshr(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode, bool round);
void vshr(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src, uint32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride, bool round);
void vshr(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t shrDistance, uint64_t config,
          bool round);
void vshr(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride,
          uint8_t srcRepeatStride, bool repeatStrideMode, bool strideSizeMode, bool round);
void vshr(__ubuf__ uint32_t *dst, __ubuf__ uint32_t *src, uint32_t shrDistance, uint8_t repeat,
          uint16_t dstBlockStride, uint16_t srcBlockStride, uint16_t dstRepeatStride,
          uint16_t srcRepeatStride, bool round);
void vsort(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vsort(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat);
void vsort(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vsort(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat);
void vsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint64_t config);
void vsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride);
void vsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride,
           bool repeatStrideMode, bool strideSizeMode);
void vsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dstBlockStride,
           uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride);
void vsub(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vsub(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsub(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vsub(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vsub(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsub(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vsub(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint64_t config);
void vsub(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsub(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vsub(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint64_t config);
void vsub(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsub(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat,
          uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
          uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
          bool repeatStrideMode, bool strideSizeMode);
void vsubrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vsubrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vsubrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsubrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vsubrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vsubrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsubrelu(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
              uint64_t config);
void vsubrelu(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride,
              bool repeatStrideMode, bool strideSizeMode);
void vsubrelu(__ubuf__ int16_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1, uint8_t repeat,
              uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride,
              uint8_t dstRepeatStride, uint8_t src0RepeatStride, uint8_t src1RepeatStride);
void vsubreluconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                         uint64_t config, bool h);
void vsubreluconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode,
                         bool h);
void vsubreluconv_f162s8(__ubuf__ int8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool h);
void vsubreluconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                          uint64_t config, bool h);
void vsubreluconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                          uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                          uint8_t src1BlockStride, uint8_t dstRepeatStride,
                          uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool repeatStrideMode,
                          bool strideSizeMode, bool h);
void vsubreluconv_f322f16(__ubuf__ half *dst, __ubuf__ float *src0, __ubuf__ float *src1,
                          uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                          uint8_t src1BlockStride, uint8_t dstRepeatStride,
                          uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h);
void vsubreluconv_s162s8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                         uint64_t config, bool h);
void vsubreluconv_s162s8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool repeatStrideMode, bool strideSizeMode,
                         bool h);
void vsubreluconv_s162s8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                         uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                         uint8_t src1BlockStride, uint8_t dstRepeatStride, uint8_t src0RepeatStride,
                         uint8_t src1RepeatStride, bool h);
void vsubreluconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint64_t config, bool h);
void vsubreluconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint64_t config, bool h);
void vsubreluconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride,
                             bool repeatStrideMode, bool strideSizeMode, bool h);
void vsubreluconv_vdeqs162b8(__ubuf__ uint8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h);
void vsubreluconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride,
                             bool repeatStrideMode, bool strideSizeMode, bool h);
void vsubreluconv_vdeqs162b8(__ubuf__ int8_t *dst, __ubuf__ int16_t *src0, __ubuf__ int16_t *src1,
                             uint8_t repeat, uint8_t dstBlockStride, uint8_t src0BlockStride,
                             uint8_t src1BlockStride, uint8_t dstRepeatStride,
                             uint8_t src0RepeatStride, uint8_t src1RepeatStride, bool h);
void vtranspose(__ubuf__ int16_t *dst, __ubuf__ int16_t *src);
void vtranspose(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src);
void wait_flag(pipe_t pipe, pipe_t tpipe, event_t pipeID);
void wait_flag(pipe_t pipe, event_t pipeID);
void wait_flag(pipe_t pipe, pipe_t tpipe, uint64_t pipeID);
void wait_flag(pipe_t pipe, uint64_t pipeID);
void wait_flag_dev(int64_t flagID);
void wait_flag_dev(pipe_t pipe, uint16_t flagID);
void wait_flag_dev(pipe_t pipe, int64_t flagID);
void wait_intra_block(pipe_t pipe, uint64_t sync_id);
void set_mte2_nz_para(uint64_t config);
void set_mte2_src_para(uint64_t config);
void set_mte2_qtable0(uint64_t config);
void set_mte2_qtable1(uint64_t config);
void test_encoding();
void set_pad_val_nddma(uint64_t config);
void set_loop_size_outtoub(uint64_t config);
void set_loop1_stride_outtoub(uint64_t config);
void set_loop2_stride_outtoub(uint64_t config);
void set_loop_size_ubtoout(uint64_t config);
void set_loop1_stride_ubtoout(uint64_t config);
void set_loop2_stride_ubtoout(uint64_t config);
void set_loop0_stride_nddma(uint64_t config);
void set_loop1_stride_nddma(uint64_t config);
void set_loop2_stride_nddma(uint64_t config);
void set_loop3_stride_nddma(uint64_t config);
void set_loop4_stride_nddma(uint64_t config);
void set_pad_cnt_nddma(uint64_t config);
void nddma_out_to_ub_b8(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint32_t loop0_size,
                        uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size,
                        uint32_t loop4_size, uint8_t loop0_lp_size, uint8_t loop0_rp_size,
                        bool constant_padding_ctl, uint8_t l2_cache_ctl);
void nddma_out_to_ub_b16(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint32_t loop0_size,
                         uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size,
                         uint32_t loop4_size, uint8_t loop0_lp_size, uint8_t loop0_rp_size,
                         bool constant_padding_ctl, uint8_t l2_cache_ctl);
void nddma_out_to_ub_b32(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint32_t loop0_size,
                         uint32_t loop1_size, uint32_t loop2_size, uint32_t loop3_size,
                         uint32_t loop4_size, uint8_t loop0_lp_size, uint8_t loop0_rp_size,
                         bool constant_padding_ctl, uint8_t l2_cache_ctl);
void winograd_conv(__cc__ int32_t *dst, __ca__ int16_t *src0, __cb__ int8_t *src1, uint64_t config);
void winograd_conv(__cc__ int32_t *dst, __ca__ int16_t *src0, __cb__ int8_t *src1, uint16_t inM,
                   uint16_t inK, uint16_t inN, uint8_t addr_SMASK, uint8_t flag, uint8_t contrl,
                   bool acc_mode);
void winograd_conv(__cc__ int32_t *dst, __ca__ int8_t *src0, __cb__ int8_t *src1, uint64_t config);
void winograd_conv(__cc__ int32_t *dst, __ca__ int8_t *src0, __cb__ int8_t *src1, uint16_t inM,
                   uint16_t inK, uint16_t inN, uint8_t addr_SMASK, uint8_t flag, uint8_t contrl,
                   bool acc_mode);
void *get_workspace(int32_t size);
void set_core_type(int idx);
int get_process_num();
int __cce_simt_get_BLOCK_DIM_X();
int __cce_simt_get_BLOCK_DIM_Y();
int __cce_simt_get_BLOCK_DIM_Z();
int __cce_simt_get_TID_X();
int __cce_simt_get_TID_Y();
int __cce_simt_get_TID_Z();
void set_block_dim(int idx);
void set_intra_block(pipe_t pipe, uint64_t sync_id);
void get_buf(pipe_t pipe, uint8_t buf_ID, bool mode);
void rls_buf(pipe_t pipe, uint8_t buf_ID, bool mode);
uint32_t set_ub_addr_upper_bound(uint32_t bufferAddr, uint32_t dataLen);
void AscendCKernelBegin(const char *kernName, int32_t argn, uint64_t *argv);
void AscendCKernelEnd(const char *kernName, int32_t argn, uint64_t *argv);
void AscendCBlockBegin(int32_t blkidx, const char *kernName, int32_t argn, const uint64_t *argv);
void AscendCBlockEnd(int32_t blkidx, const char *kernName, int32_t argn, const uint64_t *argv);
void AscendCEnInterruptExit();
void AscendCQueCreate(uint8_t srcQuePos, uint8_t dstQuePos, uint8_t depth);
void AscendCBufInit(uint8_t bufPos, uint8_t type, uint8_t num, uint64_t addr, uint64_t len);
void AscendCBufAlloc(uint8_t quePos, uint8_t bufPos, uint64_t addr, uint64_t len);
void AscendCBufFree(uint8_t quePos, uint8_t bufPos, uint64_t addr, uint64_t len);
void AscendCBufEnque(uint8_t srcQuePos, uint8_t dstQuePos, uint8_t bufPos, uint64_t addr);
void AscendCBufDeque(uint8_t srcQuePos, uint8_t dstQuePos, uint8_t bufPos, uint64_t addr);
void AscendCBufAbsAddr(uint8_t hardware, uint64_t addr, uint64_t len);
void AscendCBufGet(uint8_t quePos, uint8_t bufPos, uint64_t addr, uint64_t len);
void AscendCTBufPoolInit(uint8_t bufPos, uint64_t addr, uint64_t len, uint64_t tbufPoolHandle);
void AscendCUpdateTbufPoolStatus(uint64_t tbufPoolHandle, bool isReset);
void AscendCRecordPoolHierarchy(uint64_t pre, uint64_t back);
void AscendCTBufPoolResetCheck(uint8_t bufPos, uint64_t addr, uint64_t len,
                               uint64_t tbufPoolHandle);
void batch_offset(uint64_t addr, uint64_t block_factor);
void wait_and_set(__gm__ void *gm_addr, __ubuf__ void *ub_addr, uint64_t wait_val,
                  uint64_t set_val);
void wait_sync(__gm__ void *local_gm_addr, __gm__ void *gm_addr, __ubuf__ void *ub_addr);
void set_tensora(void *arg0, void *arg1);
void set_tensorb(void *arg0);
void get_tensorc(void *arg0);
void __ib_set_stub(int32_t blockIdx, int32_t eventID, bool isAIVOnly);
void __ib_wait_stub(int32_t blockIdx, int32_t eventID, bool isAIVOnly);
void __sync_all_stub(int32_t usedCores, bool isAIVOnly);

#endif  // STUB_FUN_INTELLISENSE_H
