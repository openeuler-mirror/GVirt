/**
 * @file embed_kernel.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 3;                                     // tensor num for each queue

template <typename T>
class KernelEmbed {
public:
    __aicore__ inline KernelEmbed() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, int dim,
                                int batchSize, int embStartIdxIn, int embEndIdxIn, int tpSize)
    {
        xGm.SetGlobalBuffer((__gm__ T *)x);
        yGm.SetGlobalBuffer((__gm__ uint32_t *)y);
        zGm.SetGlobalBuffer((__gm__ T *)z);
        pipe.InitBuffer(queBind, BUFFER_NUM, dim * sizeof(T));
        embDim = dim;
        embBatchSize = batchSize;
        embStartIdx = embStartIdxIn;
        embEndIdx = embEndIdxIn;
        embTpSize = tpSize;
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> xLocalZero = queBind.AllocTensor<T>();
        if (embTpSize > 1) {
            T scalar = 0.0;
            uint64_t mask = 128;
            // repeatTimes = embDim / 128, 128 elements one repeat, embDim elements total
            // dstBlkStride = 1, no gap between blocks in one repeat
            // dstRepStride = 8, no gap between repeats
            Duplicate(xLocalZero, scalar, mask, embDim / 128, 1, 8);
        }

        for (int index = GetBlockIdx(); index < embBatchSize; index += GetBlockNum()) {
            CopyInOut(index, xLocalZero);
        }

        queBind.FreeTensor(xLocalZero);
    }

private:
    __aicore__ inline void CopyInOut(int index, LocalTensor<T> xLocalZero)
    {
        uint32_t row = yGm.GetValue(index);
        if ((row >= embEndIdx) || (row < embStartIdx)) {
            if (((embDim * sizeof(T)) & (BLOCK_SIZE - 1)) == 0) {
                DataCopy(zGm[index * embDim], xLocalZero, embDim);
            } else {
                DataCopyParams copyParams;
                copyParams.blockLen = embDim * sizeof(T);
                copyParams.blockCount = 1;
                DataCopyPad(zGm[index * embDim], xLocalZero, copyParams);
            }
            queBind.EnQue(xLocalZero);
            xLocalZero = queBind.DeQue<T>();
        } else {
            LocalTensor<T> xLocal = queBind.AllocTensor<T>();
            DataCopy(xLocal, xGm[(row - embStartIdx) * embDim], ROUND_UP(embDim * sizeof(T), BLOCK_SIZE) / sizeof(T));
            queBind.EnQue(xLocal);
            xLocal = queBind.DeQue<T>();
            if (((embDim * sizeof(T)) & (BLOCK_SIZE - 1)) == 0) {
                DataCopy(zGm[index * embDim], xLocal, embDim);
            } else {
                DataCopyParams copyParams;
                copyParams.blockLen = embDim * sizeof(T);
                copyParams.blockCount = 1;
                DataCopyPad(zGm[index * embDim], xLocal, copyParams);
            }
            queBind.FreeTensor(xLocal);
        }
    }

private:
    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBind;
    GlobalTensor<T> xGm;
    GlobalTensor<uint32_t> yGm;
    GlobalTensor<T> zGm;
    int embDim;
    int embBatchSize;
    int embStartIdx;
    int embEndIdx;
    int embTpSize;
};

#define EMBED_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void embed_kernel_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z, int dim, int batchSize, \
                                                           int embStartIdxIn, int embEndIdxIn, int tpSize) \
{ \
    KernelEmbed<dtype> op; \
    op.Init(x, y, z, dim, batchSize, embStartIdxIn, embEndIdxIn, tpSize); \
    op.Process(); \
}

EMBED_FUNC_DEFINE(float16_t);
EMBED_FUNC_DEFINE(bfloat16_t);
