/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_OP_H_
#define _XLITE_OP_H_

#include "base.h"
#include "runtime.h"

void XliteOpAllGather(XRuntime &rt, XTensor *in, XTensor *out, enum commType type);
void XliteOpReduceScatter(XRuntime &rt, XTensor *in, XTensor *out, enum commType type);
void XliteOpAllReduceSum(XRuntime &rt, XTensor *in, XTensor *out, enum commType type);

void XliteOpEmbed(XRuntime &rt, XTensor *in, XTensor *embed, uint32_t start, uint32_t end, XTensor *out);
void XliteOpRmsNorm(XRuntime &rt, XTensor *in, XTensor *norm, float normEps, XTensor *out);
void XliteOpAdd(XRuntime &rt, XTensor *in1, XTensor *in2, XTensor *out);
void XliteOpMatmul(XRuntime &rt, XTensor *in1, XTensor *weight, XTensor *out);

void XliteOpSiluAndMul(XRuntime &rt, XTensor *in, XTensor *out);
void XliteOpCastDown(XRuntime &rt, XTensor *in, XTensor *out, XTensor *outScale);
void XliteOpCastUp(XRuntime &rt, XTensor *in, XTensor *inScale, XTensor *out);
void XliteOpSigmoidTopK(XRuntime &rt, XTensor *in, XTensor *inbias, XTensor *indicts,
                        uint32_t nGroups, uint32_t nTopkGroups, uint32_t nTopk, float scale,
                        XTensor *outWeights, XTensor *outRouting);
void XliteOpPermutation(XRuntime &rt, XTensor *in, XTensor *routing, uint32_t start, uint32_t end,
                        XTensor *out, XTensor *unpIdx, XTensor *counts);
void XliteOpUnpermutation(XRuntime &rt, XTensor *in, XTensor *unpIdx, XTensor *routing, XTensor *weights,
                          uint32_t start, uint32_t end, XTensor *out);
void XliteOpGroupMatmul(XRuntime &rt, XTensor *in, XTensor *weights, XTensor *scales,
                        XTensor *counts, long outDim, long inDim, XTensor *output);

#endif