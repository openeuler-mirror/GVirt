/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "ascend.h"
#include "runtime.h"
#include "op.h"
#include "kernels/kernel_entry.h"

static HcclDataType XDtype2HcclDtype(enum XDtype dtype)
{
    switch (dtype) {
        case INT8:
            return HCCL_DATA_TYPE_INT8;
        case INT32:
            return HCCL_DATA_TYPE_INT32;
        case INT64:
            return HCCL_DATA_TYPE_INT64;
        case FP16:
            return HCCL_DATA_TYPE_FP16;
        case BF16:
            return HCCL_DATA_TYPE_BFP16;
        case FP32:
            return HCCL_DATA_TYPE_FP32;
        default:
            std::cerr << "unknown data type " << XDtypeStr(dtype) << std::endl;
            return HCCL_DATA_TYPE_RESERVED;
    }
}

void XliteOpAllGather(XRuntime &rt, XTensor &in, XTensor &out, enum commType type)
{
    uint32_t rankSize = type == TP ? rt.tpSize() : rt.dpSize();
    if (in.dtype != out.dtype || in.numel * rankSize != out.numel) {
        std::cerr << __func__ << ": check tensor failed! input: " << in << " output: " << out << std::endl;
        return;
    }
    if ((in.numel * XDtypeBit(in.dtype)) % XDtypeBit(INT8)) {
        std::cerr << __func__ << ": all gather 8bit align check failed!" << std::endl;
        return;
    }
    CHECK_HCCL(HcclAllGather(in.ptr, out.ptr, in.numel * XDtypeBit(in.dtype) / XDtypeBit(INT8), HCCL_DATA_TYPE_INT8,
                   type == TP ? rt._tpComm : rt._dpComm, rt.stream));
}

void XliteOpReduceScatter(XRuntime &rt, XTensor &in, XTensor &out, enum commType type)
{
    uint32_t rankSize = type == TP ? rt.tpSize() : rt.dpSize();
    if (in.dtype != out.dtype || in.numel != out.numel * rankSize ) {
        std::cerr << __func__ << ": check tensor failed! input: " << in << " output: " << out << std::endl;
        return;
    }
    CHECK_HCCL(HcclReduceScatter(in.ptr, out.ptr, out.numel, XDtype2HcclDtype(in.dtype), HCCL_REDUCE_SUM,
                   type == TP ? rt._tpComm : rt._dpComm, rt.stream));
}

void XliteOpAllReduceSum(XRuntime &rt, XTensor &in, XTensor &out, enum commType type)
{
    if (in.dtype != out.dtype || in.numel != out.numel) {
        std::cerr << __func__ << ": check tensor failed! input: " << in << " output: " << out << std::endl;
        return;
    }
    CHECK_HCCL(HcclAllReduce(in.ptr, out.ptr, in.numel, XDtype2HcclDtype(in.dtype), HCCL_REDUCE_SUM,
                   type == TP ? rt._tpComm : rt._dpComm, rt.stream));
}


void XliteOpEmbed(XRuntime &rt, XTensor &in, XTensor &embed, uint32_t start, uint32_t end, XTensor &out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpRmsNorm(XRuntime &rt, XTensor &in, XTensor &norm, float normEps, XTensor &out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out)
{
    uint32_t blockDim = 8;
    std::vector<long> shape = {8, 2048};

    if (in1.dtype != FP16 || in2.dtype != FP16 || out.dtype != FP16) {
        std::cerr << __FILE__ << ":" << __LINE__ << "unsupport dtype" << std::endl;
        return;
    }
    if (in1.shape != shape || in2.shape != shape || out.shape != shape) {
        std::cerr << __FILE__ << ":" << __LINE__ << "unsupport shape" << std::endl;
        return;
    }
    add_do(blockDim, rt.stream, in1.ptr, in2.ptr, out.ptr);
}

void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpSigmoidTopK(XRuntime &rt, XTensor &in, XTensor &inbias, XTensor &indicts,
                        uint32_t nGroups, uint32_t nTopkGroups, uint32_t nTopk, float scale,
                        XTensor &outWeights, XTensor &outRouting)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpPermutation(XRuntime &rt, XTensor &in, XTensor &routing, uint32_t start, uint32_t end,
                        XTensor &out, XTensor &unpIdx, XTensor &counts)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing, XTensor &weights,
                          uint32_t start, uint32_t end, XTensor &out)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &scales,
                        XTensor &counts, uint32_t start, uint32_t end,
                        long outDim, long inDim, XTensor &output)
{
    std::cout << __func__ << ": TODO" << std::endl;
}