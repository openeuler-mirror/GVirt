/*
 * @file kernel_macro.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef _XLITE_KERNEL_MACRO_H_
#define _XLITE_KERNEL_MACRO_H_

#define ROUND_DOWN(x, y) (((x) / (y)) * (y))
#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define DIV_ROUND_UP(x, y) (((x) + ((y) - 1)) / (y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BLOCK_SIZE 32
#define VECTOR_MAX_BYTESIZE 256               // The maximum byte size of one repeat in vector
#define VECTOR_MAX_NUM_OF_FP32 64             // The maximum num of float32 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_FP16 128            // The maximum num of float16 dtype in one vector repeat

// 设置拷贝数据的config
inline __aicore__ uint64_t __set_dmi_config(uint8_t sid, uint16_t nBurst, uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap)
{
    uint64_t config = 0;
    const uint64_t _DST_GAP = 48;
    const uint64_t _SRC_GAP = 32;
    const uint64_t _LEN_BURST = 16;
    const uint64_t _N_BURST = 4;

    config |= (uint64_t)(sid & 0xfULL);
    config |= (uint64_t)dstGap << _DST_GAP;
    config |= (uint64_t)srcGap << _SRC_GAP;
    config |= (uint64_t)lenBurst << _LEN_BURST;
    config |= (uint64_t)(nBurst & 0xfffULL) << _N_BURST;

    return config;
}

// 设置vector运算xt（双操作数）
inline __aicore__ uint64_t set_vector_xt(uint64_t repeatDestStride, uint64_t repeatSrc0Stride, uint64_t repeatSrc1Stride, uint64_t destStride, uint64_t src0Stride, uint64_t src1Stride, uint64_t repeat)
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
inline __aicore__ uint64_t set_vector_1src_xt(uint64_t repeatDestStride, uint64_t repeatSrcStride, uint64_t destStride, uint64_t srcStride, uint64_t repeat)
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
#endif