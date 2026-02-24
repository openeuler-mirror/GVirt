/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef __aicore__
#define __aicore__ [aicore]
#endif  // __aicore__

#ifndef GM_ADDR
#define GM_ADDR __gm__ uint8_t *__restrict__
#endif

#define BLOCK_SIZE 32  // UB中一个Block的单位大小

#define VECTOR_COUNT_FP16 128
#define VECTOR_COUNT_FP32 64
#define ALIGN 256

#define ROUND_DOWN(x, y) (((x) / (y)) * (y))
#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define DIV_ROUND_UP(x, y) (((x) + ((y) - 1)) / (y))

#ifdef __DAV_C220_VEC__

#define BITS_PER_DWORD 64
#define BIT_DWORD(nr) ((nr) / BITS_PER_DWORD)

inline __aicore__ void bitmap_set(__ubuf__ uint64_t *addr, uint32_t id)
{
    __ubuf__ uint64_t *p = addr + BIT_DWORD(id);
    *p |= (1ULL << (id % BITS_PER_DWORD));
}

inline __aicore__ int bitmap_test(__ubuf__ uint64_t *addr, uint32_t id)
{
    __ubuf__ uint64_t *p = addr + BIT_DWORD(id);
    return ((*p) & (1ULL << (id % BITS_PER_DWORD))) ? 1 : 0;
}

#endif

// 设置拷贝数据的config
inline __aicore__ uint64_t __set_dmi_config(uint8_t sid, uint16_t nBurst, uint16_t lenBurst,
                                            uint16_t srcGap, uint16_t dstGap)
{
    uint64_t config;
    config = ((uint64_t)(sid & 0xfULL)) | ((uint64_t)(dstGap) << 48) | ((uint64_t)(srcGap) << 32) |
             ((uint64_t)(lenBurst) << 16) | ((uint64_t)(nBurst & 0xfffULL) << 4);
    return config;
}

// 设置vector运算xt（双操作数）
inline __aicore__ uint64_t set_vector_xt(uint64_t repeatDestStride, uint64_t repeatSrc0Stride,
                                         uint64_t repeatSrc1Stride, uint64_t destStride,
                                         uint64_t src0Stride, uint64_t src1Stride, uint64_t repeat)
{
    uint64_t config = 0;
    const uint64_t _DST_STRIDE = 0;
    const uint64_t _SRC0_STRIDE = 8;
    const uint64_t _SRC1_STRIDE = 16;
    const uint64_t _REP_DST_STRIDE = 24;
    const uint64_t _REP_SRC0_STRIDE = 32;
    const uint64_t _REP_SRC1_STRIDE = 40;
    const uint64_t _REPEAT = 56;

    config |= (uint64_t)destStride << _DST_STRIDE;
    config |= (uint64_t)src0Stride << _SRC0_STRIDE;
    config |= (uint64_t)src1Stride << _SRC1_STRIDE;
    config |= (uint64_t)repeatDestStride << _REP_DST_STRIDE;
    config |= (uint64_t)repeatSrc0Stride << _REP_SRC0_STRIDE;
    config |= (uint64_t)repeatSrc1Stride << _REP_SRC1_STRIDE;
    config |= (uint64_t)repeat << _REPEAT;

    return config;
}

// 设置vector运算xt（单操作数）
inline __aicore__ uint64_t set_vector_1src_xt(uint64_t repeatDestStride, uint64_t repeatSrcStride,
                                              uint64_t destStride, uint64_t srcStride,
                                              uint64_t repeat)
{
    uint64_t config = 0;
    const uint64_t _DST_STRIDE = 0;
    const uint64_t _SRC_STRIDE = 16;
    const uint64_t _REP_DST_STRIDE = 32;
    const uint64_t _REP_SRC_STRIDE = 40;
    const uint64_t _REPEAT = 56;

    config |= (uint64_t)destStride << _DST_STRIDE;
    config |= (uint64_t)srcStride << _SRC_STRIDE;
    config |= (uint64_t)repeatDestStride << _REP_DST_STRIDE;
    config |= (uint64_t)repeatSrcStride << _REP_SRC_STRIDE;
    config |= (uint64_t)repeat << _REPEAT;

    return config;
}
