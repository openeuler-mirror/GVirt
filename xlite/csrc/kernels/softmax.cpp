/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "softmax_attn_aiv.h"

#define SOFTMAX_FUNC_DEFINE(Dtype, CalcDtype) \
extern "C" __global__ __aicore__ void softmax_##Dtype(GM_ADDR x, uint32_t contextLen) \
{ \
    RunAivSoftmax<Dtype, CalcDtype>((__gm__ Dtype *)x, 1, contextLen, contextLen); \
}

SOFTMAX_FUNC_DEFINE(float16_t, float16_t);
SOFTMAX_FUNC_DEFINE(bfloat16_t, float);

#define SOFTMAX_LONG_FUNC_DEFINE(Dtype, CalcDtype) \
extern "C" __global__ __aicore__ void softmax_long_##Dtype(GM_ADDR x, uint32_t contextLen) \
{ \
    RunAivSoftmaxLong<Dtype, CalcDtype>((__gm__ Dtype *)x, 1, contextLen, contextLen); \
}

SOFTMAX_LONG_FUNC_DEFINE(float16_t, float16_t);
SOFTMAX_LONG_FUNC_DEFINE(bfloat16_t, float);