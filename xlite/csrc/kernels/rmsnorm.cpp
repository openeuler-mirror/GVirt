/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr int32_t HALf_INTERVAL = 2;

__aicore__ inline int32_t findPowerTwo(int32_t n)
{
    // find max power of 2 no more than n (32 bit)
    n |= n >> 1;  // Set the first digit of n's binary to 1
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

__aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float> &src_local, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
    if (g_coreType == AIV) {
        vcadd((__ubuf__ float *)src_local.GetPhyAddr(), (__ubuf__ float *)src_local.GetPhyAddr(), 1, 0, 1, 0, false);
    }
#else
    vcadd((__ubuf__ float *)src_local.GetPhyAddr(),
        (__ubuf__ float *)src_local.GetPhyAddr(),
        1,
        1,
        1,
        DEFAULT_REPEAT_STRIDE);
#endif
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    return src_local.GetValue(0);
}

template <typename T>
class KernelRMSNorm {
public:
    __aicore__ inline KernelRMSNorm() {}
    __aicore__ inline void Init(GM_ADDR inout, GM_ADDR weight, GM_ADDR out,
                                uint32_t tokenNum, uint32_t normDim, float normEps,
                                uint32_t cntPerToken, uint32_t step, uint32_t startOffset)
    {
        inoutGm.SetGlobalBuffer((__gm__ T *)inout);
        weightGm.SetGlobalBuffer((__gm__ T *)weight);
        outGm.SetGlobalBuffer((__gm__ T *)out);
        pipe.InitBuffer(queIn, BUFFER_NUM, normDim * sizeof(T));
        pipe.InitBuffer(queOut, BUFFER_NUM, normDim * sizeof(T));
        pipe.InitBuffer(weight_buf, normDim * sizeof(T));
        pipe.InitBuffer(weight_fp32_buf, normDim * sizeof(float));
        pipe.InitBuffer(inout_fp32_buf, normDim * sizeof(float));
        pipe.InitBuffer(sqx_buf, normDim * sizeof(float));
        this->tokenNum = tokenNum;
        this->normDim = normDim;
        this->normEps = normEps;
        this->cntPerToken = cntPerToken;
        this->step = step;
        this->startOffset = startOffset;
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> weightLocal = weight_buf.Get<T>();
        DataCopy(weightLocal, weightGm, ROUND_UP(normDim * sizeof(T), BLOCK_SIZE) / sizeof(T));
        PipeBarrier<PIPE_ALL>();
        // Cast weight: f16/bf16 -> f32
        LocalTensor<float> weightFp32Local = weight_fp32_buf.Get<float>();
        Cast(weightFp32Local, weightLocal, RoundMode::CAST_NONE, normDim);
        PipeBarrier<PIPE_V>();
        uint32_t loopCount = tokenNum * cntPerToken;
        for (uint32_t loop = GetBlockIdx(); loop < loopCount; loop += GetBlockNum()) {
            CopyIn(loop);
            Compute(weightFp32Local);
            CopyOut(loop);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t loop)
    {
        LocalTensor<T> xLocal = queIn.AllocTensor<T>();
        uint32_t offset = startOffset + loop / cntPerToken * step + loop % cntPerToken * normDim;
        DataCopy(xLocal, inoutGm[offset], ROUND_UP(normDim * sizeof(T), BLOCK_SIZE) / sizeof(T));
        queIn.EnQue(xLocal);
    }

    __aicore__ inline void Compute(LocalTensor<float> weightFp32)
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        LocalTensor<T> xLocal = queIn.DeQue<T>();
        LocalTensor<T> yLocal = queOut.AllocTensor<T>();
        LocalTensor<float> sqx = sqx_buf.Get<float>();
        LocalTensor<float> xBufFp32 = inout_fp32_buf.Get<float>();
        float n = (float)1.0 / normDim;

        // 1. Cast x : f16/bf16 -> f32
        Cast(xBufFp32, xLocal, RoundMode::CAST_NONE, normDim);
        PipeBarrier<PIPE_V>();
        queIn.FreeTensor(xLocal);
        // 2. Cal x^2
        Mul(sqx, xBufFp32, xBufFp32, normDim);
        PipeBarrier<PIPE_V>();
        // 3. Cal sum(x^2)
        float reduceOut = ReduceSumHalfInterval(sqx, normDim);
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        // 4. Cal rstd = 1 / sqrt(1 / sum(x^2) + eps)
        float rstdValue = 1 / sqrt(reduceOut * n + normEps);
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        // 5. Cal x * rstd
        Muls(sqx, xBufFp32, rstdValue, normDim);
        PipeBarrier<PIPE_V>();
        // 6. Cal weight * x * rstd
        Mul(sqx, sqx, weightFp32, normDim);
        PipeBarrier<PIPE_V>();
        // 7. Cast y: f32 -> f16/bf16
        Cast(yLocal, sqx, RoundMode::CAST_RINT, normDim);
        queOut.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t loop)
    {
        LocalTensor<T> yLocal = queOut.DeQue<T>();
        uint32_t offset = startOffset + loop / cntPerToken * step + loop % cntPerToken * normDim;

        if (((normDim * sizeof(T)) & (BLOCK_SIZE - 1)) == 0) {
            DataCopy<T>(outGm[offset], yLocal, normDim);
        } else {
            DataCopyParams copyParams;
            copyParams.blockLen = normDim * sizeof(T);
            copyParams.blockCount = 1;
            DataCopyPad(outGm[offset], yLocal, copyParams);
        }

        queOut.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> queIn;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queOut;
    GlobalTensor<T> inoutGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> outGm;
    TBuf<TPosition::VECCALC> inout_fp32_buf;
    TBuf<TPosition::VECCALC> sqx_buf;
    TBuf<TPosition::VECCALC> weight_buf;
    TBuf<TPosition::VECCALC> weight_fp32_buf;
    uint32_t tokenNum;
    uint32_t normDim;
    uint32_t cntPerToken;
    uint32_t step;
    uint32_t startOffset;
    float normEps;
};

#define RMSNORM_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void rmsnorm_##dtype(GM_ADDR inout, GM_ADDR weight, GM_ADDR out, \
                                                      uint32_t tokenNum, uint32_t normDim, float normEps, \
                                                      uint32_t cntPerToken, uint32_t step, uint32_t startOffset) \
{ \
    KernelRMSNorm<dtype> op; \
    op.Init(inout, weight, out, tokenNum, normDim, normEps, cntPerToken, step, startOffset); \
    op.Process(); \
}

RMSNORM_FUNC_DEFINE(float16_t);
RMSNORM_FUNC_DEFINE(bfloat16_t);