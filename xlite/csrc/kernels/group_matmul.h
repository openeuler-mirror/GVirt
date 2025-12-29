/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "matmul.h"

template <typename Dtype>
__aicore__ void group_matmul_kernel(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR counts,
    uint32_t n, int64_t kN, int64_t kK, uint64_t m0, uint64_t n0, uint64_t k0, uint32_t startIdx, uint32_t endIdx)
{
    Matmul<Dtype> matmul_op;

    uint32_t off = 0;
    for (uint32_t i = startIdx; i < endIdx && i < n; i++) {
        uint32_t kM = *((__gm__ uint32_t *)(counts + i * sizeof(uint32_t)));
        if (kM <= 0) {
            continue;
        }

        uint64_t w_addr = *((__gm__ uint64_t *)(ws + i * sizeof(void *)));
        __gm__ uint8_t *w = (__gm__ uint8_t *)w_addr;
        matmul_op.Init(x + off * kK * sizeof(Dtype), w, z + off * kN * sizeof(Dtype),
                       kM, kN, kK, 0, m0, n0, k0, 257);
        matmul_op.Run();
        off += kM;
    }
}

#define GROUPMATMUL_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void group_matmul_##dtype(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR counts, \
                                                           uint32_t n, int64_t kN, int64_t kK, uint64_t m0, \
                                                           uint64_t n0, uint64_t k0, uint32_t startIdx, uint32_t endIdx) \
{ \
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY); \
    group_matmul_kernel<dtype>(x, ws, z, counts, n, kN, kK, m0, n0, k0, startIdx, endIdx); \
}