/**
 * @file silu_and_mul.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_macro.h"
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

template <typename SrcType, typename CalType>
class KernelSiluAndMul {
public:
    __aicore__ inline KernelSiluAndMul() = default;
    __aicore__ inline void Init(GM_ADDR in, GM_ADDR out, int32_t d, TPipe* pipeIn)
    {
        SetMaskNorm();
        SetAtomicNone();

        pipe = pipeIn;
        dim = d;
        copyParams = {1, dim * sizeof(SrcType), 0, 0, 0};
        padParams = {true, 0, sizeof(SrcType), 0};

        xGm.SetGlobalBuffer((__gm__ SrcType*)in);
        yGm.SetGlobalBuffer((__gm__ SrcType*)in + dim);
        zGm.SetGlobalBuffer((__gm__ SrcType*)out);
        uint32_t paddedLen = ROUND_UP(dim * sizeof(SrcType), BLOCK_SIZE);
        pipe->InitBuffer(inQueueX, BUFFER_NUM, paddedLen);
        pipe->InitBuffer(inQueueY, BUFFER_NUM, paddedLen);
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, paddedLen);

        uint32_t tmpBufPaddedLen = ROUND_UP(dim * sizeof(CalType), BLOCK_SIZE);
        pipe->InitBuffer(tmpBufZ, tmpBufPaddedLen);
        if constexpr (!std::is_same_v<SrcType, CalType>) {
            pipe->InitBuffer(tmpBufX, tmpBufPaddedLen);
            pipe->InitBuffer(tmpBufY, tmpBufPaddedLen);
        }
    }

    __aicore__ inline void Process(GM_ADDR pm, int32_t numTokens)
    {
        if (pm) {
            uint32_t pm_val = *((__gm__ uint32_t*)pm);
            numTokens = pm_val < numTokens ? pm_val : numTokens;
        }
        auto blockNum = static_cast<int32_t>(GetBlockNum());
        auto blockidx = static_cast<int32_t>(GetBlockIdx());
        for (int32_t progress = blockidx; progress < numTokens; progress += blockNum) {
            CopyIn(progress);
            if constexpr (std::is_same_v<SrcType, CalType>) {
                Compute();
            } else {
                ComputeCast();
            }
            CopyOut(progress);
        }
        SetMaskNorm();
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint32_t offset = progress * dim * 2;
        LocalTensor<SrcType> xLocal = inQueueX.AllocTensor<SrcType>();
        LocalTensor<SrcType> yLocal = inQueueY.AllocTensor<SrcType>();
        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<SrcType> xLocal = inQueueX.DeQue<SrcType>();
        LocalTensor<SrcType> yLocal = inQueueY.DeQue<SrcType>();
        LocalTensor<SrcType> zLocal = outQueueZ.AllocTensor<SrcType>();
        LocalTensor<CalType> tmpTensor = tmpBufZ.Get<CalType>();

        // -x
        auto scalar = static_cast<CalType>(-1.0);
        Muls(tmpTensor, xLocal, scalar, dim);
        // exp(-x)
        Exp(tmpTensor, tmpTensor, dim);
        // 1 + exp(-x)
        scalar = static_cast<CalType>(1.0);
        Adds(tmpTensor, tmpTensor, scalar, dim);
        // x * sigmoid(x) = x / (1 + exp(-x))
        Div(tmpTensor, xLocal, tmpTensor, dim);
        // y * x / (1 + e^-x)
        Mul(zLocal, tmpTensor, yLocal, dim);

        outQueueZ.EnQue<SrcType>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void ComputeCast()
    {
        LocalTensor<SrcType> xLocal = inQueueX.DeQue<SrcType>();
        LocalTensor<SrcType> yLocal = inQueueY.DeQue<SrcType>();
        LocalTensor<SrcType> zLocal = outQueueZ.AllocTensor<SrcType>();
        LocalTensor<CalType> xTmpTensor = tmpBufX.Get<CalType>();
        LocalTensor<CalType> yTmpTensor = tmpBufY.Get<CalType>();
        LocalTensor<CalType> zTmpTensor = tmpBufZ.Get<CalType>();

        Cast(xTmpTensor, xLocal, RoundMode::CAST_NONE, dim);
        Cast(yTmpTensor, yLocal, RoundMode::CAST_NONE, dim);

        // -x
        auto scalar = static_cast<CalType>(-1.0000000000000000e+00f);
        Muls(zTmpTensor, xTmpTensor, scalar, dim);
        // exp(-x)
        Exp(zTmpTensor, zTmpTensor, dim);
        // 1 + exp(-x)
        scalar = static_cast<CalType>(1.0);
        Adds(zTmpTensor, zTmpTensor, scalar, dim);
        // x * sigmoid(x) = x / (1 + exp(-x))
        Div(zTmpTensor, xTmpTensor, zTmpTensor, dim);
        // y * x / (1 + e^-x)
        Mul(zTmpTensor, zTmpTensor, yTmpTensor, dim);

        Cast(zLocal, zTmpTensor, RoundMode::CAST_RINT, dim);

        outQueueZ.EnQue<SrcType>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<SrcType> zLocal = outQueueZ.DeQue<SrcType>();
        DataCopyPad(zGm[progress * dim], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe* pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<TPosition::VECCALC> tmpBufX, tmpBufY, tmpBufZ;
    GlobalTensor<SrcType> xGm;
    GlobalTensor<SrcType> yGm;
    GlobalTensor<SrcType> zGm;
    DataCopyExtParams copyParams;
    DataCopyPadExtParams<SrcType> padParams;
    int32_t dim;
};

#define SILU_AND_MUL_FUNC_DEFINE(dtype, castType) \
extern "C" __global__ __aicore__ void silu_and_mul_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR pm, int nTokens, int dim) \
{ \
    KernelSiluAndMul<dtype, castType> op; \
    TPipe pipe; \
    op.Init(x, y, dim, &pipe); \
    op.Process(pm, nTokens); \
}

SILU_AND_MUL_FUNC_DEFINE(float, float);
SILU_AND_MUL_FUNC_DEFINE(float16_t, float16_t);
SILU_AND_MUL_FUNC_DEFINE(bfloat16_t, float);