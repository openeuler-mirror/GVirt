/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "matmul.h"

template <typename Dtype>
__aicore__ void group_matmul_kernel(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR counts,
                                    uint32_t n, int64_t kN, int64_t kK, uint64_t m0, uint64_t n0, uint64_t k0,
                                    uint32_t startIdx, uint32_t endIdx, bool weightNZ, bool transpose)
{
    Matmul<Dtype> matmul_op;

    uint32_t off = 0;
    for (uint32_t i = startIdx; i < endIdx && i < n; i++) {
        uint32_t kM = *((__gm__ uint32_t *)(counts + i * sizeof(uint32_t)));
        if (kM <= 0) {
            continue;
        }

        if (m0 == MATMUL_M0_N0_K0_DEFAULT_VALUE ||
            n0 == MATMUL_M0_N0_K0_DEFAULT_VALUE ||
            k0 == MATMUL_M0_N0_K0_DEFAULT_VALUE) {
            m0 = ROUND_UP(kM, 32);
            if (m0 > 128) {
                m0 = 128;
            }
            n0 = 256;
            k0 = 512 / sizeof(Dtype);

            uint64_t mLoop = DIV_ROUND_UP(kM, m0);
            uint64_t nLoop = DIV_ROUND_UP(kN, n0);
            uint64_t totalLoops = mLoop * nLoop;
            uint64_t lastLoops = totalLoops % get_block_num();

            if (totalLoops < 3 * get_block_num() &&
                (lastLoops != 0 && lastLoops < get_block_num() / 2)) {
                if (n <= 32 * get_block_num()) {
                    m0 = m0 > 64 ? 64 : m0;
                    n0 = 64;
                } else if (n <= 64 * get_block_num()) {
                    n0 = 64;
                } else if (n <= 128 * get_block_num()) {
                    n0 = 128;
                } else if (n <= 256 * get_block_num()) {
                    n0 = 256;
                } else {
                    m0 = m0 > 64 ? 64 : m0;
                    n0 = 384;
                    k0 /= 2;
                }
            }
        }

        uint64_t w_addr = *((__gm__ uint64_t *)(ws + i * sizeof(void *)));
        __gm__ uint8_t *w = (__gm__ uint8_t *)w_addr;
        matmul_op.Init(x + off * kK * sizeof(Dtype), w, z + off * kN * sizeof(Dtype),
                       kM, kN, kK, weightNZ, transpose, m0, n0, k0, MATMUL_SWIZZLE_DEFAULT_VALUE);
        matmul_op.Run();
        off += kM;
    }
}

#define GROUPMATMUL_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void group_matmul_##dtype(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR counts, \
                                                           uint32_t n, int64_t kN, int64_t kK, uint64_t m0, \
                                                           uint64_t n0, uint64_t k0, uint32_t startIdx, \
                                                           uint32_t endIdx, bool weightNZ, bool transpose) \
{ \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); \
    group_matmul_kernel<dtype>(x, ws, z, counts, n, kN, kK, m0, n0, k0, startIdx, endIdx, weightNZ, transpose); \
}