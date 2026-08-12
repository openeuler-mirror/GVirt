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
        this->input.SetGlobalBuffer((__gm__ half *)input);
        this->output.SetGlobalBuffer((__gm__ half *)output);
        this->dim0 = dim0;
        this->dim1 = dim1;
        this->dim2 = dim2;

        tileDim1 = 8;
        tileDim2 = 8;

        uint64_t off = 0;
        inBuf[0].address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        inBuf[0].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += UB_SIZE / 4;
        inBuf[1].address_.logicPos = static_cast<uint8_t>(TPosition::VECIN);
        inBuf[1].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += UB_SIZE / 4;
        outBuf[0].address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
        outBuf[0].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += UB_SIZE / 4;
        outBuf[1].address_.logicPos = static_cast<uint8_t>(TPosition::VECOUT);
        outBuf[1].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void CopyIn(uint32_t b, uint32_t i, uint32_t j, uint32_t ping)
    {
        DataCopyExtParams copyParamsIn;
        copyParamsIn.blockCount = tileDim1 * 16 - padDim1;
        copyParamsIn.blockLen = tileDim2 * 32 - padDim2 * 2;
        copyParamsIn.srcStride = dim2 * 2 - copyParamsIn.blockLen;
        copyParamsIn.dstStride = padDim2 / 16;
        DataCopyPadExtParams<half> padParams{false, 0, uint8_t(padDim2 % 16), 0};

        uint32_t inOffset = b * dim1 * dim2 + i * dim2 + j;
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + ping);
        DataCopyPad(inBuf[ping], input[inOffset], copyParamsIn, padParams);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + ping);
    }

    __aicore__ inline void Compute(uint32_t ping)
    {
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + ping);
        TransDataTo5HDParams transDataParams{false, false, tileDim2, 0, 0};
        if (tileDim2 != 1) {
            transDataParams.dstRepStride = uint16_t(tileDim1 * 16);
            transDataParams.srcRepStride = 1;
        }
        for (int b = 0; b < tileDim1; b++) {
            LocalTensor<half> srcLocalList[16];
            for (int i = 0; i < 16; i++) {
                srcLocalList[i] = inBuf[ping][16 * tileDim2 * i + b * tileDim2 * 256];
            }
            LocalTensor<half> dstLocalList[16];
            for (int i = 0; i < 16; i++) {
                dstLocalList[i] = outBuf[ping][16 * tileDim1 * i + b * 16];
            }
            TransDataTo5HD(dstLocalList, srcLocalList, transDataParams);
        }
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + ping);
    }

    __aicore__ inline void CopyOut(uint32_t b, uint32_t i, uint32_t j, uint32_t ping)
    {
        DataCopyExtParams copyParamsOut;
        copyParamsOut.blockCount = tileDim2 * 16 - padDim2;
        copyParamsOut.blockLen = tileDim1 * 32 - padDim1 * 2;
        copyParamsOut.srcStride = padDim1 / 16;
        copyParamsOut.dstStride = dim1 * 2 - copyParamsOut.blockLen;

        uint32_t outOffset = b * dim1 * dim2 + j * dim1 + i;
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + ping);
        DataCopyPad(output[outOffset], outBuf[ping], copyParamsOut);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0 + ping);
    }

    __aicore__ inline void Process()
    {
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        uint32_t ping = 0, aivCnt = 0;
        for (uint32_t b = 0; b < dim0; b++) {
            for (uint32_t i = 0; i < dim1; i += 16 * tileDim1) {
                padDim1 = (i + tileDim1 * 16 > dim1) ? i + tileDim1 * 16 - dim1 : 0;
                for (uint32_t j = 0; j < dim2; j += 16 * tileDim2) {
                    aivCnt = (aivCnt + 1) % GetBlockNum();
                    if (aivCnt % GetBlockNum() != GetBlockIdx())
                        continue;
                    padDim2 = (j + tileDim2 * 16 > dim2) ? j + tileDim2 * 16 - dim2 : 0;
                    CopyIn(b, i, j, ping);
                    Compute(ping);
                    CopyOut(b, i, j, ping);
                    ping = 1 - ping;
                }
            }
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

private:
    GlobalTensor<half> input;
    GlobalTensor<half> output;
    LocalTensor<half> inBuf[2];
    LocalTensor<half> outBuf[2];

    uint8_t tileDim1, tileDim2;
    uint8_t padDim1, padDim2;
    uint32_t dim0, dim1, dim2;
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
