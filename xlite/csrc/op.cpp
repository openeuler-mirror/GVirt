/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "ascend.h"
#include "runtime.h"
#include "op.h"
#include "aclrtlaunch_add_float16_t.h"
#include "aclrtlaunch_add_bfloat16_t.h"
#include "aclrtlaunch_embed_kernel_float16_t.h"
#include "aclrtlaunch_embed_kernel_bfloat16_t.h"
#include "aclrtlaunch_rmsnorm_float16_t.h"
#include "aclrtlaunch_rmsnorm_bfloat16_t.h"
#include "aclrtlaunch_silu_and_mul_float.h"
#include "aclrtlaunch_silu_and_mul_float16_t.h"
#include "aclrtlaunch_silu_and_mul_bfloat16_t.h"
#include "aclrtlaunch_rope_and_cache_float16_t.h"
#include "aclrtlaunch_rope_and_cache_bfloat16_t.h"
#include "aclrtlaunch_prefill_att_float16_t.h"
#include "aclrtlaunch_prefill_att_bfloat16_t.h"
#include "aclrtlaunch_decode_att_float16_t.h"
#include "aclrtlaunch_decode_att_bfloat16_t.h"
#include "aclrtlaunch_matmul_float16_t.h"
#include "aclrtlaunch_matmul_bfloat16_t.h"
#include "aclrtlaunch_matmul_float.h"
#include "aclrtlaunch_add_bias_float.h"
#include "aclrtlaunch_add_bias_float16_t.h"
#include "aclrtlaunch_add_bias_bfloat16_t.h"
#include "aclrtlaunch_softmax_topk_float.h"
#include "aclrtlaunch_cast_bfloat16_t_float.h"
#include "aclrtlaunch_group_matmul_bfloat16_t.h"
#include "aclrtlaunch_group_matmul_float16_t.h"
#include "aclrtlaunch_group_matmul_float.h"
#include "aclrtlaunch_permutation.h"
#include "aclrtlaunch_unpermutation.h"
#include "aclrtlaunch_softmax_float16_t.h"
#include "aclrtlaunch_softmax_bfloat16_t.h"
#include "aclrtlaunch_softmax_long_float16_t.h"
#include "aclrtlaunch_softmax_long_bfloat16_t.h"

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

void XliteOpRmsNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &out, float normEps,
                    uint32_t normDim, uint32_t cntPerToken, uint32_t startOffset)
{

    if (in.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_rmsnorm_float16_t(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, out.ptr,
                                      in.shape[0], normDim, normEps, cntPerToken, in.shape[1], startOffset);
    } else if (in.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_rmsnorm_bfloat16_t(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, out.ptr,
                                      in.shape[0], normDim, normEps, cntPerToken, in.shape[1], startOffset);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out)
{
    if (in1.dtype == FP16 && in2.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_add_float16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, out.ptr, in1.shape[0], in1.shape[1]);
    } else if (in1.dtype == BF16 && in2.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_add_bfloat16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, out.ptr, in1.shape[0], in1.shape[1]);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &norm, float normEps, XTensor &out)
{
    if (in1.dtype == FP16 && in2.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_rmsnorm_float16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, norm.ptr, out.ptr,
                                      in1.shape[0], in1.shape[1], normEps, 1, in1.shape[1], 0);
    } else if (in1.dtype == BF16 && in2.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_rmsnorm_bfloat16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, norm.ptr, out.ptr,
                                       in1.shape[0], in1.shape[1], normEps, 1, in1.shape[1], 0);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ, bool transpose,
                   uint64_t m0, uint64_t n0, uint64_t k0, uint64_t swizzle)
{
    uint64_t m = in.shape[0];
    uint64_t n = transpose ? weight.shape[1] : weight.shape[0];
    uint64_t k = transpose ? weight.shape[0] : weight.shape[1];
    if (m0 == MATMUL_M0_N0_K0_DEFAULT_VALUE ||
        n0 == MATMUL_M0_N0_K0_DEFAULT_VALUE ||
        k0 == MATMUL_M0_N0_K0_DEFAULT_VALUE) {
        m0 = ROUND_UP(m, 32);
        if (m0 > 128) {
            m0 = 128;
        }
        n0 = 256;
        k0 = 512 / (XDtypeBit(weight.dtype) / 8);

        uint64_t mLoop = DIV_ROUND_UP(m, m0);
        uint64_t nLoop = DIV_ROUND_UP(n, n0);
        uint64_t totalLoops = mLoop * nLoop;
        uint64_t lastLoops = totalLoops % rt.aicNum;

        if (totalLoops < 3 * rt.aicNum &&
            (lastLoops != 0 && lastLoops < rt.aicNum / 2)) {
            if (n <= 32 * rt.aicNum) {
                m0 = m0 > 64 ? 64 : m0;
                n0 = 64;
            } else if (n <= 64 * rt.aicNum) {
                n0 = 64;
            } else if (n <= 128 * rt.aicNum) {
                n0 = 128;
            } else if (n <= 256 * rt.aicNum) {
                n0 = 256;
            } else {
                m0 = m0 > 64 ? 64 : m0;
                n0 = 384;
                k0 /= 2;
            }
        }
    }

    if (in.dtype == FP16 && weight.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_matmul_float16_t(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k,
                                     weightNZ, transpose, m0, n0, k0, swizzle);
    } else if (in.dtype == BF16 && weight.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_matmul_bfloat16_t(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k,
                                      weightNZ, transpose, m0, n0, k0, swizzle);
    } else if (in.dtype == FP32 && weight.dtype == FP32 && out.dtype == FP32 && transpose == false) {
        aclrtlaunch_matmul_float(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k,
                                 weightNZ, transpose, m0, n0, k0, swizzle);
    } else if (in.dtype == BF16 && weight.dtype == FP32 && out.dtype == FP32 && transpose == false) {
        XTensor &tmp = rt.pool->GetTensor(in.shape, FP32);
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, tmp.ptr, in.numel);
        aclrtlaunch_matmul_float(rt.aicNum, rt.stream, tmp.ptr, weight.ptr, out.ptr, m, n, k,
                                 weightNZ, transpose, m0, n0, k0, swizzle);
        rt.pool->PutTensor(tmp);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out)
{
    if (in.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_silu_and_mul_float16_t(rt.aivNum, rt.stream, in.ptr, out.ptr, nullptr, in.shape[0], out.shape[1]);
    } else if (in.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_silu_and_mul_bfloat16_t(rt.aivNum, rt.stream, in.ptr, out.ptr, nullptr, in.shape[0], out.shape[1]);
    } else if (in.dtype == FP32 && out.dtype == FP32) {
        aclrtlaunch_silu_and_mul_float(rt.aivNum, rt.stream, in.ptr, out.ptr, nullptr, in.shape[0], out.shape[1]);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out)
{
    if (in.dtype == BF16 && out.dtype == FP32) {
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, out.ptr, in.numel);
    }
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
    aclrtlaunch_permutation(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr, unpIdx.ptr, counts.ptr, in.shape[0],
                            in.shape[1], counts.shape[0], start, end);
}

void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing, XTensor &weights,
                          uint32_t start, uint32_t end, XTensor &out)
{
    if (in.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_unpermutation(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr, unpIdx.ptr, weights.ptr,
                                  out.shape[0], in.shape[1], weights.shape[1], start, end);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &scales,
                        XTensor &counts, uint32_t start, uint32_t end,
                        XDtype weightDtype, long outDim, long inDim, XTensor &output,
                        bool weightNZ, bool transpose)
{
    if (in.dtype == BF16 && weightDtype == BF16 && output.dtype == BF16) {
        aclrtlaunch_group_matmul_bfloat16_t(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr, counts.ptr,
                                            counts.shape[0], outDim, inDim, -1, -1, -1, start, end, weightNZ, transpose);
    } else if (in.dtype == FP16 && weightDtype == FP16 && output.dtype == FP16) {
        aclrtlaunch_group_matmul_float16_t(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr, counts.ptr,
                                           counts.shape[0], outDim, inDim, -1, -1, -1, start, end, weightNZ, transpose);
    } else if (in.dtype == FP32 && weightDtype == FP32 && output.dtype == FP32 && transpose == false) {
        aclrtlaunch_group_matmul_float(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr, counts.ptr,
                                       counts.shape[0], outDim, inDim, -1, -1, -1, start, end, weightNZ, transpose);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping,
                      uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                      uint32_t rotDim, uint32_t blockSize, bool isNeox)
{
    uint32_t localHeads = nHeads / rt.tpSize();
    uint32_t localKvHeads = nKvHeads / rt.tpSize();
    localKvHeads = localKvHeads == 0 ? 1 : localKvHeads;
    uintptr_t qPtr = reinterpret_cast<uintptr_t>(inout.ptr);
    uintptr_t kPtr = qPtr + localHeads * headDim * XDtypeBit(inout.dtype) / 8;
    uintptr_t vPtr = kPtr + localKvHeads * headDim * XDtypeBit(inout.dtype) / 8;
    void *k = reinterpret_cast<void *>(kPtr);
    void *v = reinterpret_cast<void *>(vPtr);
    float scale = 1 / sqrt(headDim);

    if (!isNeox) {
        std::cerr << __func__ << ": unsupported rope type gptj" << std::endl;
        return;
    }

    if (inout.dtype == FP16 && kCache.dtype == FP16 && vCache.dtype == FP16 && cossin.dtype == FP16) {
        aclrtlaunch_rope_and_cache_float16_t(rt.aivNum, rt.stream, position.ptr, inout.ptr, k, v, cossin.ptr,
                                             kCache.ptr, vCache.ptr, slotMapping.ptr, inout.shape[0], rotDim,
                                             inout.shape[1], inout.shape[1], inout.shape[1], localHeads,
                                             localKvHeads, headDim, blockSize, scale);
    } else if (inout.dtype == BF16 && kCache.dtype == BF16 && vCache.dtype == BF16 && cossin.dtype == BF16) {
        aclrtlaunch_rope_and_cache_bfloat16_t(rt.aivNum, rt.stream, position.ptr, inout.ptr, k, v, cossin.ptr,
                                             kCache.ptr, vCache.ptr, slotMapping.ptr, inout.shape[0], rotDim,
                                             inout.shape[1], inout.shape[1], inout.shape[1], localHeads,
                                             localKvHeads, headDim, blockSize, scale);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpPrefillAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &qk,
                             XTensor &blockTables, XTensor &cachedLens,
                             XTensor &vCache, XTensor &output, XTensor &lens,
                             XTensor &prefillIndex, XTensor &cumPromptLens, uint32_t headDim,
                             uint32_t nHeads, uint32_t nKvHeads, uint32_t blockSize,
                             uint32_t batch, uint32_t maxNumBlock)
{
    uint32_t localHeads = nHeads / rt.tpSize();
    uint32_t localKvHeads = nKvHeads / rt.tpSize();
    localKvHeads = localKvHeads == 0 ? 1 : localKvHeads;
    if (qkv.dtype == FP16 && qk.dtype == FP16 && kCache.dtype == FP16 && vCache.dtype == FP16 && output.dtype == FP16) {
        aclrtlaunch_prefill_att_float16_t(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, qk.ptr, blockTables.ptr,
                                          cachedLens.ptr, vCache.ptr, output.ptr, lens.ptr, prefillIndex.ptr, cumPromptLens.ptr,
                                          headDim, localHeads, localKvHeads, blockSize, batch, maxNumBlock);
    } else if (qkv.dtype == BF16 && qk.dtype == BF16 && kCache.dtype == BF16 && vCache.dtype == BF16 && output.dtype == BF16) {
        aclrtlaunch_prefill_att_bfloat16_t(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, qk.ptr, blockTables.ptr,
                                          cachedLens.ptr, vCache.ptr, output.ptr, lens.ptr, prefillIndex.ptr, cumPromptLens.ptr,
                                          headDim, localHeads, localKvHeads, blockSize, batch, maxNumBlock);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpDecodeAttention(XRuntime &rt, XTensor &a2v, XTensor &v2a, XTensor &qkv,
                            XTensor &kCache, XTensor &vCache, XTensor &cachedLens,
                            XTensor &blockTables, XTensor &qk, XTensor &output, XTensor &decodeIdx,
                            XTensor &cumPromptLens, uint32_t batch, uint32_t nHeads,
                            uint32_t headDim, uint32_t blockSize, uint32_t maxNumBlock,
                            uint32_t nKvHeads, uint32_t maxM)
{
    uint32_t localHeads = nHeads / rt.tpSize();
    uint32_t localKvHeads = nKvHeads / rt.tpSize();
    localKvHeads = localKvHeads == 0 ? 1 : localKvHeads;
    if (qkv.dtype == FP16 && qk.dtype == FP16 && kCache.dtype == FP16 && vCache.dtype == FP16 && output.dtype == FP16) {
        aclrtlaunch_decode_att_float16_t(rt.aicNum, rt.stream, a2v.ptr, v2a.ptr, qkv.ptr, kCache.ptr,
                                         vCache.ptr, cachedLens.ptr, blockTables.ptr, qk.ptr, output.ptr,
                                         decodeIdx.ptr, cumPromptLens.ptr, batch, localHeads, headDim,
                                         blockSize, maxNumBlock, localKvHeads, maxM, localHeads + 2 * localKvHeads,
                                         0, 0);
    } else if (qkv.dtype == BF16 && qk.dtype == BF16 && kCache.dtype == BF16 && vCache.dtype == BF16 && output.dtype == BF16) {
        aclrtlaunch_decode_att_bfloat16_t(rt.aicNum, rt.stream, a2v.ptr, v2a.ptr, qkv.ptr, kCache.ptr,
                                          vCache.ptr, cachedLens.ptr, blockTables.ptr, qk.ptr, output.ptr,
                                          decodeIdx.ptr, cumPromptLens.ptr, batch, localHeads, headDim,
                                          blockSize, maxNumBlock, localKvHeads, maxM, localHeads + 2 * localKvHeads,
                                          0, 0);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
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
                             int nLocalHeads, int kvLoraRank, int rotDim, int headSize, int vDim,
                             uint32_t blockSize)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XliteDsOpPrefillMix(XRuntime &rt, XTensor &out, XTensor &alpha, XTensor &max, XTensor &sum,
                         XTensor &q, XTensor &k, XTensor &qk, XTensor &blockTables, XTensor &cachedLens,
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
                              int wkvbStep, int vdim, XTensor &scores, XTensor &kvUpWeight, XTensor &result)
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

void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &socres, XTensor &indices,
                        XTensor &outWeights, XTensor &outRouting, uint32_t topK, bool normTopKProb)
{
    if (socres.dtype == FP32 && indices.dtype == INT32 && outWeights.dtype == FP32 && outRouting.dtype == BIT1) {
        aclrtlaunch_softmax_topk_float(rt.aivNum, rt.stream, socres.ptr, indices.ptr, outWeights.ptr, outRouting.ptr,
                                       socres.shape[0], indices.shape[0], topK, normTopKProb);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpSoftmax(XRuntime &rt, uint32_t calcLen, XTensor &x)
{
    if (x.dtype == FP16) {
        aclrtlaunch_softmax_float16_t(1, rt.stream, x.ptr, x.shape[0], x.shape[1], calcLen);
    } else if (x.dtype == BF16) {
        aclrtlaunch_softmax_bfloat16_t(1, rt.stream, x.ptr, x.shape[0], x.shape[1], calcLen);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}

void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x)
{
    if (x.dtype == FP16) {
        aclrtlaunch_softmax_long_float16_t(1, rt.stream, x.ptr, x.shape[0], x.shape[1], calcLen);
    } else if (x.dtype == BF16) {
        aclrtlaunch_softmax_long_bfloat16_t(1, rt.stream, x.ptr, x.shape[0], x.shape[1], calcLen);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}