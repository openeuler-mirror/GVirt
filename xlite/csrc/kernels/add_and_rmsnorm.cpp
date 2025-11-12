/**
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
 #include "kernel_operator.h"
 #include "kernel_macro.h"

 using namespace AscendC;

 constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
 constexpr uint32_t ELEM_PER_REP_FP32 = 64;
 constexpr int32_t HALF_INTERVAL = 2;

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
            bodyCount = bodyCount / HALF_INTERVAL;
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
class KernelAddAndRMSNorm {
public:
    __aicore__ inline KernelAddAndRMSNorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR z,
                                uint32_t tokenNum, uint32_t hiddenSize, float normEps)
    {
        xGm.SetGlobalBuffer((__gm__ T *)x);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        zGm.SetGlobalBuffer((__gm__ T *)z);
        weightGm.SetGlobalBuffer((__gm__ T *)weight);
        pipe.InitBuffer(queInX, BUFFER_NUM, hiddenSize * sizeof(T));
        pipe.InitBuffer(queInY, BUFFER_NUM, hiddenSize * sizeof(T));
        pipe.InitBuffer(queOutX, BUFFER_NUM, hiddenSize * sizeof(T));
        pipe.InitBuffer(queOutZ, BUFFER_NUM, hiddenSize * sizeof(T));
        pipe.InitBuffer(weight_buf, hiddenSize * sizeof(T));
        pipe.InitBuffer(weight_fp32_buf, hiddenSize * sizeof(float));
        pipe.InitBuffer(x_fp32_buf, hiddenSize * sizeof(float));
        pipe.InitBuffer(y_fp32_buf, hiddenSize * sizeof(float));
        pipe.InitBuffer(sqx_buf, hiddenSize * sizeof(float));
        rmsnormTokenNum = tokenNum;
        rmsnormHiddenSize = hiddenSize;
        rmsnormEps = normEps;
        copyParams = {1, rmsnormHiddenSize * sizeof(T), 0, 0, 0};
        padParams = {true, 0, sizeof(T), 0};
    }
    __aicore__ inline void Process()
    {
        LocalTensor<T> weightLocal = weight_buf.Get<T>();
        DataCopy(weightLocal, weightGm, ROUND_UP(rmsnormHiddenSize * sizeof(T), BLOCK_SIZE) / sizeof(T));
        PipeBarrier<PIPE_ALL>();
        // Cast weight: f16/bf16 -> f32
        LocalTensor<float> weightFp32Local = weight_fp32_buf.Get<float>();
        Cast(weightFp32Local, weightLocal, RoundMode::CAST_NONE, rmsnormHiddenSize);
        PipeBarrier<PIPE_V>();

        for (uint32_t loop = GetBlockIdx(); loop < rmsnormTokenNum; loop += GetBlockNum()) {
            CopyIn(loop);
            Compute(loop, weightFp32Local);
            CopyOut(loop);
        }
    }

    private:
        __aicore__ inline void CopyIn(uint32_t loop)
        {
            LocalTensor<T> xLocal = queInX.AllocTensor<T>();
            LocalTensor<T> yLocal = queInY.AllocTensor<T>();
            DataCopyPad(xLocal, xGm[loop * rmsnormHiddenSize], copyParams, padParams);
            DataCopyPad(yLocal, yGm[loop * rmsnormHiddenSize], copyParams, padParams);
            queInX.EnQue(xLocal);
            queInY.EnQue(yLocal);
        }

        __aicore__ inline void Compute(uint32_t loop, LocalTensor<float> weightFp32)
        {
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            LocalTensor<T> xLocal = queInX.DeQue<T>();
            LocalTensor<T> yLocal = queInY.DeQue<T>();
            LocalTensor<T> zLocal = queOutZ.AllocTensor<T>();
            LocalTensor<T> xOutLocal = queOutX.AllocTensor<T>();
            LocalTensor<float> sqx = sqx_buf.Get<float>();
            LocalTensor<float> xBufFp32 = x_fp32_buf.Get<float>();
            LocalTensor<float> yBufFp32 = y_fp32_buf.Get<float>();
            float n = (float)1.0 / rmsnormHiddenSize;

            // 1. Cast x/y : f16/bf16 -> f32
            Cast(xBufFp32, xLocal, RoundMode::CAST_NONE, rmsnormHiddenSize);
            Cast(yBufFp32, yLocal, RoundMode::CAST_NONE, rmsnormHiddenSize);
            PipeBarrier<PIPE_V>();
            queInX.FreeTensor(xLocal);
            queInY.FreeTensor(yLocal);
            // 2. Cal x = x + y
            Add(xBufFp32, xBufFp32, yBufFp32, rmsnormHiddenSize);
            PipeBarrier<PIPE_V>();
            Cast(xOutLocal, xBufFp32, RoundMode::CAST_RINT, rmsnormHiddenSize);
            queOutX.EnQue<T>(xOutLocal);
            // 3. Cal x^2
            Mul(sqx, xBufFp32, xBufFp32, rmsnormHiddenSize);
            PipeBarrier<PIPE_V>();
            // 4. Cal sum(x^2)
            float reduceOut = ReduceSumHalfInterval(sqx, rmsnormHiddenSize);
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            // 5. Cal rstd = 1 / sqrt(1 / sum(x^2) + eps)
            float rstdValue = 1 / sqrt(reduceOut * n + rmsnormEps);
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            // 6. Cal x * rstd
            Muls(sqx, xBufFp32, rstdValue, rmsnormHiddenSize);
            PipeBarrier<PIPE_V>();
            // 7. Cal weight * x * rstd
            Mul(sqx, sqx, weightFp32, rmsnormHiddenSize);
            PipeBarrier<PIPE_V>();
            // 8. Cast y: f32 -> f16/bf16
            Cast(zLocal, sqx, RoundMode::CAST_RINT, rmsnormHiddenSize);
            queOutZ.EnQue<T>(zLocal);
        }

        __aicore__ inline void CopyOut(uint32_t loop)
        {
            LocalTensor<T> zLocal = queOutZ.DeQue<T>();
            LocalTensor<T> xOutLocal = queOutX.DeQue<T>();

            DataCopyPad(zGm[loop * rmsnormHiddenSize], zLocal, copyParams);
            DataCopyPad(xGm[loop * rmsnormHiddenSize], xOutLocal, copyParams);

            queOutZ.FreeTensor(zLocal);
            queOutX.FreeTensor(xOutLocal);
        }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> queInX, queInY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queOutX, queOutZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> weightGm;
    GlobalTensor<T> zGm;
    TBuf<TPosition::VECCALC> x_fp32_buf;
    TBuf<TPosition::VECCALC> y_fp32_buf;
    TBuf<TPosition::VECCALC> sqx_buf;
    TBuf<TPosition::VECCALC> weight_buf;
    TBuf<TPosition::VECCALC> weight_fp32_buf;
    uint32_t rmsnormTokenNum;
    uint32_t rmsnormHiddenSize;
    float rmsnormEps;
    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, sizeof(T), 0};
};

#define ADD_AND_RMSNORM_FUNC_DEFINE(dtype) \
extern "C" __global__ __aicore__ void add_and_rmsnorm_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR weight, GM_ADDR z, \
                                                              uint32_t tokenNum, uint32_t hiddenSize, float normEps) \
{ \
    KernelAddAndRMSNorm<dtype> op; \
    op.Init(x, y, weight, z, tokenNum, hiddenSize, normEps); \
    op.Process(); \
}

ADD_AND_RMSNORM_FUNC_DEFINE(float16_t);
ADD_AND_RMSNORM_FUNC_DEFINE(bfloat16_t);