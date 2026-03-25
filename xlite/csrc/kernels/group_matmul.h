/*
 * Copyright (C) 2025-2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "matmul.h"

template <typename Dtype, typename MatDtype, typename OutDtype>
__aicore__ void group_matmul_kernel(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR counts, uint32_t n,
                                    int64_t kN, int64_t kK, uint64_t m0, uint64_t n0, uint64_t k0,
                                    uint32_t startIdx, uint32_t endIdx, bool weightNZ,
                                    bool transpose, uint64_t swizzle)
{
    Matmul<Dtype, MatDtype, OutDtype> matmul_op;

    if (m0 == (uint64_t)-1) {
        m0 = 128;
        n0 = 256;
        k0 = 512 / sizeof(Dtype);
    }

    uint32_t nLoop = DIV_ROUND_UP(kN, n0);
    uint32_t off = 0;
    uint32_t remain = 0;
    for (uint32_t i = startIdx; i < endIdx && i < n; i++) {
        uint32_t kM = *((__gm__ uint32_t *)(counts + i * sizeof(uint32_t)));
        if (kM <= 0) {
            continue;
        }

        int mLoop = DIV_ROUND_UP(kM, m0);
        uint32_t curCount = remain + mLoop * nLoop;
        uint32_t curBlock =
            get_block_idx() >= remain ? get_block_idx() : get_block_idx() + get_block_num();
        uint64_t w_addr = *((__gm__ uint64_t *)(ws + i * sizeof(void *)));
        __gm__ uint8_t *w = (__gm__ uint8_t *)w_addr;
        matmul_op.Init(x + off * kK * sizeof(Dtype), w, z + off * kN * sizeof(Dtype), nullptr,
                       nullptr, kM, kN, kK, weightNZ, transpose, m0, n0, k0, swizzle, curBlock,
                       curCount, remain);
        matmul_op.Run();
        off += kM;
        remain = curCount % get_block_num();
    }
}

#define GROUPMATMUL_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void group_matmul_##dtype(                                   \
        GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR counts, uint32_t n, int64_t kN, int64_t kK,     \
        uint64_t m0, uint64_t n0, uint64_t k0, uint32_t startIdx, uint32_t endIdx, bool weightNZ, \
        bool transpose, uint64_t swizzle)                                                         \
    {                                                                                             \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                           \
        if constexpr (std::is_same<dtype, int8_t>::value) {                                       \
            group_matmul_kernel<dtype, int32_t, float16_t>(x, ws, z, counts, n, kN, kK, m0, n0,   \
                                                           k0, startIdx, endIdx, weightNZ,        \
                                                           transpose, swizzle);                   \
        } else {                                                                                  \
            group_matmul_kernel<dtype, float, dtype>(x, ws, z, counts, n, kN, kK, m0, n0, k0,     \
                                                     startIdx, endIdx, weightNZ, transpose,       \
                                                     swizzle);                                    \
        }                                                                                         \
    }