/**
 * @file add.cpp
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

constexpr int32_t BUFFER_NUM = 2; //tensor num for each queue 

template <typename SrcType, typename CalType>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t xNumel, uint32_t yNumel, TPipe* pipeIn)
    {
        SetMaskNorm();

        pipe = pipeIn;
        this->xNumel = xNumel;
        this->yNumel = yNumel;
        copyParams = {1, yNumel * sizeof(SrcType), 0, 0, 0};
        padParams = {true, 0, sizeof(SrcType), 0};
        xGm.SetGlobalBuffer((__gm__ SrcType*)x);
        yGm.SetGlobalBuffer((__gm__ SrcType*)y);
        zGm.SetGlobalBuffer((__gm__ SrcType*)z);
        uint32_t paddedLen = ROUND_UP(yNumel * sizeof(SrcType), BLOCK_SIZE);
        pipe->InitBuffer(inQueueX, BUFFER_NUM, paddedLen);
        pipe->InitBuffer(inQueueY, BUFFER_NUM, paddedLen);
        pipe->InitBuffer(outQueueZ, BUFFER_NUM, paddedLen);
        if constexpr (!std::is_same_v<SrcType, CalType>) {
            uint32_t buffLen = ROUND_UP(yNumel * sizeof(CalType), BLOCK_SIZE);
            pipe->InitBuffer(tmpBuffX, buffLen);
            pipe->InitBuffer(tmpBuffY, buffLen);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t blockNum = GetBlockNum();
        for (uint32_t progress = GetBlockIdx(); progress < xNumel; progress += blockNum) {
            CopyIn(progress);
            if constexpr (std::is_same_v<SrcType, CalType>) {
                Compute();
            } else {
                ComputeCast();
            }
            CopyOut(progress);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<SrcType> xLocal = inQueueX.AllocTensor<SrcType>();
        LocalTensor<SrcType> yLocal = inQueueY.AllocTensor<SrcType>();
        DataCopyPad(xLocal, xGm[progress * yNumel], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(yLocal, yGm[progress * yNumel], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<SrcType> xLocal = inQueueX.DeQue<SrcType>();
        LocalTensor<SrcType> yLocal = inQueueY.DeQue<SrcType>();
        LocalTensor<SrcType> zLocal = outQueueZ.AllocTensor<SrcType>();

        Add(zLocal, xLocal, yLocal, yNumel);
        outQueueZ.EnQue<SrcType>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    

    __aicore__ inline void ComputeCast()
    {
        LocalTensor<SrcType> xLocal = inQueueX.DeQue<SrcType>();
        LocalTensor<SrcType> yLocal = inQueueY.DeQue<SrcType>();
        LocalTensor<SrcType> zLocal = outQueueZ.AllocTensor<SrcType>();

        LocalTensor<CalType> xTmpTensor = tmpBuffX.Get<CalType>();
        LocalTensor<CalType> yTmpTensor = tmpBuffY.Get<CalType>();
        Cast(xTmpTensor, xLocal, RoundMode::CAST_NONE, yNumel);
        Cast(yTmpTensor, yLocal, RoundMode::CAST_NONE, yNumel);
        Add(xTmpTensor, xTmpTensor, yTmpTensor, yNumel);
        Cast(zLocal, xTmpTensor, RoundMode::CAST_RINT, yNumel);

        outQueueZ.EnQue<SrcType>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<SrcType> zLocal = outQueueZ.DeQue<SrcType>();
        DataCopyPad(zGm[progress * yNumel], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe* pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<TPosition::VECCALC> tmpBuffX, tmpBuffY;
    GlobalTensor<SrcType> xGm;
    GlobalTensor<SrcType> yGm;
    GlobalTensor<SrcType> zGm;
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    DataCopyPadExtParams<SrcType> padParams{true, 0, sizeof(SrcType), 0};
    uint32_t xNumel{0};
    uint32_t yNumel{0};
};

#define ADD_FUNC_DEFINE(dtype, castType) \
extern "C" __global__ __aicore__ void add_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t xNumel, uint32_t yNumel) \
{ \
    KernelAdd<dtype, castType> op; \
    TPipe pipe; \
    op.Init(x, y, z, xNumel, yNumel, &pipe); \
    op.Process(); \
}

ADD_FUNC_DEFINE(float, float);
ADD_FUNC_DEFINE(float16_t, float);
ADD_FUNC_DEFINE(bfloat16_t, float);