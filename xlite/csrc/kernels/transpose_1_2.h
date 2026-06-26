/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#ifdef __DAV_C220_VEC__

static __aicore__ inline void DumpBufferIndex(__ubuf__ float *buf, const __gm__ char *name,
                                              int size, int step = 1)
{
    DumpBuffer(buf, name, size, step, 1, true);
}

template <typename Dtype>
class Transpose_1_2
{
public:
    __aicore__ inline Transpose_1_2()
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t dim0, uint32_t dim1,
                                uint32_t dim2)
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        this->input.SetGlobalBuffer((__gm__ Dtype *)input);
        this->output.SetGlobalBuffer((__gm__ Dtype *)output);
        this->dim0 = dim0;
        this->dim1 = dim1;
        this->dim2 = dim2;
        uint64_t off = 0;
        ubBuf.address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        ubBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void CopyIn(uint64_t inOffset)
    {
        DataCopyExtParams copyInParams{1, 0, 0, 0, 0};
        DataCopyPadExtParams<Dtype> padParams{false, 0, 32 / sizeof(Dtype) - 1, 0};
        copyInParams.blockLen = sizeof(Dtype);
        copyInParams.blockCount = tileSize;
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        DataCopyPad(ubBuf, input[inOffset], copyInParams, padParams);
        pipe_barrier(PIPE_MTE2);
    }

    __aicore__ inline void CopyOut(uint64_t outOffset)
    {
        DataCopyExtParams copyOutParams{1, 0, 0, 0, 0};
        copyOutParams.blockLen = sizeof(Dtype);
        copyOutParams.blockCount = tileSize;
        copyOutParams.dstStride = (dim1 - 1) * sizeof(Dtype);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        DataCopyPad(output[outOffset], ubBuf, copyOutParams);
        pipe_barrier(PIPE_MTE3);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < dim0 * dim1; i++) {
            if (i % GetBlockNum() != GetBlockIdx())
                continue;
            for (uint32_t k = 0; k < dim2; k += tileSize) {
                tileSize = min((uint32_t)2048, dim2 - k);
                CopyIn(i * dim2 + k);
                CopyOut((i / dim1) * dim1 * dim2 + k * dim1 + i % dim1);
            }
        }
    }

private:
    GlobalTensor<Dtype> input;
    GlobalTensor<Dtype> output;
    LocalTensor<Dtype> ubBuf;

    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    uint32_t tileSize = 2048;
};

#define TRANSPOSE_1_2_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void transpose_1_2_##dtype(                    \
        GM_ADDR input, GM_ADDR output, uint32_t dim0, uint32_t dim1, uint32_t dim2) \
    {                                                                               \
        Transpose_1_2<dtype> op;                                                    \
        op.Init(input, output, dim0, dim1, dim2);                                   \
        op.Process();                                                               \
    }
#else
#define TRANSPOSE_1_2_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void transpose_1_2_##dtype(                    \
        GM_ADDR input, GM_ADDR output, uint32_t dim0, uint32_t dim1, uint32_t dim2) \
    {                                                                               \
    }
#endif
