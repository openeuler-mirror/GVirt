/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "ascend.h"
#include "runtime.h"
#include "op.h"
#include "kernels/kernel_entry.h"
#include "aclrtlaunch_embed_kernel_float16_t.h"
#include "aclrtlaunch_embed_kernel_bfloat16_t.h"
#include "aclrtlaunch_rmsnorm_float16_t.h"
#include "aclrtlaunch_rmsnorm_bfloat16_t.h"
#include "aclrtlaunch_matmul_float16_t.h"
#include "aclrtlaunch_matmul_bfloat16_t.h"
#include "aclrtlaunch_matmul_float.h"
#include "aclrtlaunch_add_bias_float.h"
#include "aclrtlaunch_add_bias_float16_t.h"
#include "aclrtlaunch_add_bias_bfloat16_t.h"

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
    if (embed.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_embed_kernel_float16_t(rt.aivNum, rt.stream, embed.ptr, in.ptr, out.ptr, embed.shape[1], in.shape[0],
                                           start, end, rt.tpSize());
    } else if (embed.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_embed_kernel_bfloat16_t(rt.aivNum, rt.stream, embed.ptr, in.ptr, out.ptr, embed.shape[1], in.shape[0],
                                            start, end, rt.tpSize());
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpRmsNorm(XRuntime &rt, XTensor &in, XTensor &norm, float normEps, XTensor &out)
{
    if (in.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_rmsnorm_float16_t(rt.aivNum, rt.stream, in.ptr, norm.ptr, out.ptr, in.shape[0], in.shape[1], normEps);
    } else if (in.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_rmsnorm_bfloat16_t(rt.aivNum, rt.stream, in.ptr, norm.ptr, out.ptr, in.shape[0], in.shape[1], normEps);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out)
{
    uint32_t blockDim = 8;
    std::vector<long> shape = {8, 2048};

    if (in1.dtype != FP16 || in2.dtype != FP16 || out.dtype != FP16) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": unsupport dtype" << std::endl;
        return;
    }
    if (in1.shape != shape || in2.shape != shape || out.shape != shape) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": unsupport shape" << std::endl;
        return;
    }
    add_do(blockDim, rt.stream, in1.ptr, in2.ptr, out.ptr);
}

void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ)
{
    if (in.dtype == FP16 && weight.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_matmul_float16_t(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, in.shape[0], weight.shape[0], weight.shape[1], weightNZ);
    } else if (in.dtype == BF16 && weight.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_matmul_bfloat16_t(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, in.shape[0], weight.shape[0], weight.shape[1], weightNZ);
    } else if (in.dtype == FP32 && weight.dtype == FP32 && out.dtype == FP32) {
        aclrtlaunch_matmul_float(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, in.shape[0], weight.shape[0], weight.shape[1], weightNZ);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
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
                        XDtype weightDtype, long outDim, long inDim, XTensor &output)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpRopeBatch(XRuntime &rt, uint32_t numTokens, uint32_t nLocalHeads,
                        uint32_t stepDim, uint32_t ropeDim, XTensor &inputWithR, XTensor &freqs,
                        XTensor &position, XTensor &vGather, XTensor &outputPe, enum XRopeType ropeType)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpStridedRmsnorm(XRuntime &rt, XTensor &input, XTensor &w, XTensor &output,
                            uint32_t numTokens, uint32_t normDim, uint32_t stepDim, float normEps)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpReshapeAndCache(XRuntime &rt, XTensor &key, XTensor &value, XTensor &kCache, XTensor &vCache,
                              XTensor &slotMapping, int32_t numTokens, int32_t keyStride, int32_t valueStride,
                              int32_t numKvHeads, int32_t kHeadSize, int32_t vHeadSize, int32_t blockSize, int32_t blockNum)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpKvMatmul(XRuntime &rt, XTensor &input, XTensor &w, XTensor &output, int m, int n, int k,
                       XTensor &blockTable, bool nt, int blockSize, int headSize)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpPrefillKvSplit(XRuntime &rt, XTensor &kv, XTensor &kPe, XTensor &cache,
                             XTensor &blockTable, XTensor &kvFull, XTensor &v, int nTokens, int nTokensPad,
                             int nLocalHeads, int kvLoraRank, int rotDim, int headSize, int vDim, uint32_t blockSize)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpPrefillMix(XRuntime &rt, XTensor &out, XTensor &alpha, XTensor &max, XTensor &sum,
                         XTensor &q, XTensor &k, XTensor &qk, XTensor &blockTables, XTensor &paddingN, XTensor &cachedLens,
                         XTensor &v, XTensor &mixOut, XTensor &mixOutFinal, XTensor &promptLens,
                         XTensor &attnMask, XTensor &attnMaskAddr, XTensor &speculateLens, XTensor &prefillIndex,
                         XTensor &cumPromptLens, uint32_t headSize, uint32_t numHeads, uint32_t numKVHeads,
                         uint32_t blockSize, uint32_t batchSize, uint32_t mappingLen, uint32_t doTreeAttnMask,
                         uint32_t offsetM, uint32_t mSlice, float scale)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpEinsumShdHdcShc(XRuntime &rt, int numTokens, int headSize,
                              int nLocalHeads, int qStepDim, int kvUpWeightStepDim,
                              int kvLoraRank, XTensor &qWithQr, XTensor &kvUpWeight, XTensor &qAbsorb)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpDecodeAttn(XRuntime &rt, XTensor &q, XTensor &k, XTensor &o, XTensor &cachedLens,
                         XTensor &mapping, XTensor &promptLens, XTensor &promptLensCum, uint32_t numTokens,
                         uint32_t numHeads, uint32_t numKvHeads, uint32_t headSize, uint32_t blockSize,
                         uint32_t mappingLen, uint32_t maxContextLen, bool add)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpSoftmax(XRuntime &rt, XTensor &qk, XTensor &cachedLens, XTensor &promptLens,
                      XTensor &promptLensCum, float scale, uint32_t numTokens, uint32_t numHeads,
                      uint32_t blockSize, uint32_t maxContextLen)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpEinsumShtTcShc(XRuntime &rt, int numTokens, int nLocalHeads, int maxTokens,
                             int maxBlocksPerQuery, int numBlocks, int blockSize, int kvLoraRank, XTensor &scores,
                             XTensor &cachedLens, XTensor &promptLens, XTensor &promptLensCum, XTensor &blockTables,
                             XTensor &cCache, XTensor &result)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpEinsumShcHdcShd(XRuntime &rt, int numTokens, int nLocalHeads, int kvLoraRank,
                              int wkvbStep, int vDim, XTensor &scores, XTensor &kvUpWeight, XTensor &result)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping,
                      uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t maxM,
                      uint32_t rotDim, uint32_t blockSize, bool isNeox)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpPrefillAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &qk,
                             XTensor &blockTables, XTensor &paddingN, XTensor &cachedLens,
                             XTensor &vCache, XTensor &output, XTensor &lens,
                             XTensor &prefillIndex, XTensor &cumPromptLens, uint32_t headDim,
                             uint32_t nHeads, uint32_t nKvHeads, uint32_t blockSize,
                             uint32_t batch, uint32_t maxNumBlock)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpDecodeAttention(XRuntime &rt, XTensor &a2v, XTensor &v2a, XTensor &qkv,
                            XTensor &kCache, XTensor &vCache, XTensor &cachedLens, XTensor &lens,
                            XTensor &blockTables, XTensor &qk, XTensor &output, XTensor &decodeIdx,
                            XTensor &cumPromptLens, uint32_t batch, uint32_t nHeads,
                            uint32_t headDim, uint32_t blockSize, uint32_t maxNumBlock,
                            uint32_t nKvHeads, uint32_t maxM)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output)
{
    if (input.dtype == FP32 && weight.dtype == FP32 && output.dtype == FP32) {
        aclrtlaunch_add_bias_float(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr, output.shape[0] * output.shape[1], output.shape[1]);
    } else if (input.dtype == FP16 && weight.dtype == FP16 && output.dtype == FP16) {
        aclrtlaunch_add_bias_float16_t(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr, output.shape[0] * output.shape[1], output.shape[1]);
    } else if (input.dtype == BF16 && weight.dtype == BF16 && output.dtype == BF16) {
        aclrtlaunch_add_bias_bfloat16_t(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr, output.shape[0] * output.shape[1], output.shape[1]);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}