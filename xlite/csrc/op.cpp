/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "acl.h"
#include "runtime.h"
#include "op.h"
#include "kernels/kernel_entry.h"

void XliteOpAllGather(XRuntime &rt, XTensor *in, XTensor *out, enum commType type)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpReduceScatter(XRuntime &rt, XTensor *in, XTensor *out, enum commType type)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpAllReduceSum(XRuntime &rt, XTensor *in, XTensor *out, enum commType type)
{
    std::cout << __func__ << ": TODO" << std::endl;
}


void XliteOpEmbed(XRuntime &rt, XTensor *in, XTensor *embed, uint32_t start, uint32_t end, XTensor *out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpRmsNorm(XRuntime &rt, XTensor *in, XTensor *norm, float normEps, XTensor *out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpAdd(XRuntime &rt, XTensor *in1, XTensor *in2, XTensor *out)
{
    uint32_t blockDim = 8;
    std::vector<long> shape = {8, 2048};

    if (in1->dtype != FP16 || in2->dtype != FP16 || out->dtype != FP16) {
        std::cerr << __FILE__ << ":" << __LINE__ << "unsupport dtype" << std::endl;
        return;
    }
    if (in1->shape != shape || in2->shape != shape || out->shape != shape) {
        std::cerr << __FILE__ << ":" << __LINE__ << "unsupport shape" << std::endl;
        return;
    }
    add_do(blockDim, rt.stream, in1->ptr, in2->ptr, out->ptr);
}

void XliteOpMatmul(XRuntime &rt, XTensor *in1, XTensor *weight, XTensor *out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpSiluAndMul(XRuntime &rt, XTensor *in, XTensor *out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpCastDown(XRuntime &rt, XTensor *in, XTensor *out, XTensor *outScale)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpCastUp(XRuntime &rt, XTensor *in, XTensor *inScale, XTensor *out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpSigmoidTopK(XRuntime &rt, XTensor *in, XTensor *inbias, XTensor *indicts,
                        uint32_t nGroups, uint32_t nTopkGroups, uint32_t nTopk, float scale,
                        XTensor *outWeights, XTensor *outRouting)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpPermutation(XRuntime &rt, XTensor *in, XTensor *routing, uint32_t start, uint32_t end,
                        XTensor *out, XTensor *unpIdx, XTensor *counts)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpUnpermutation(XRuntime &rt, XTensor *in, XTensor *unpIdx, XTensor *routing, XTensor *weights,
                          uint32_t start, uint32_t end, XTensor *out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpGroupMatmul(XRuntime &rt, XTensor *in, XTensor *weights, XTensor *scales,
                        XTensor *counts, long outDim, long inDim, XTensor *output)
{
    std::cout << __func__ << ": TODO" << std::endl;
}