/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "ascend.h"
#include "runtime.h"
#include "op.h"
#include "swizzle.h"
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
#include "aclrtlaunch_matmul_float16_t.h"
#include "aclrtlaunch_matmul_bfloat16_t.h"
#include "aclrtlaunch_matmul_float.h"
#include "aclrtlaunch_add_bias_float.h"
#include "aclrtlaunch_add_bias_float16_t.h"
#include "aclrtlaunch_add_bias_bfloat16_t.h"
#include "aclrtlaunch_softmax_topk_float.h"
#include "aclrtlaunch_softmax_topk_bfloat16_t.h"
#include "aclrtlaunch_cast_bfloat16_t_float.h"
#include "aclrtlaunch_group_matmul_bfloat16_t.h"
#include "aclrtlaunch_group_matmul_float16_t.h"
#include "aclrtlaunch_group_matmul_float.h"
#include "aclrtlaunch_permutation.h"
#include "aclrtlaunch_unpermutation_bfloat16_t.h"
#include "aclrtlaunch_unpermutation_float.h"
#include "aclrtlaunch_softmax_float16_t.h"
#include "aclrtlaunch_softmax_bfloat16_t.h"
#include "aclrtlaunch_softmax_long_float16_t.h"
#include "aclrtlaunch_softmax_long_bfloat16_t.h"
#include "aclrtlaunch_allreduce_int8_t.h"
#include "aclrtlaunch_allreduce_int32_t.h"
#include "aclrtlaunch_allreduce_float16_t.h"
#include "aclrtlaunch_allreduce_bfloat16_t.h"
#include "aclrtlaunch_allreduce_float.h"
#include "aclrtlaunch_reduce_scatter_int8_t.h"
#include "aclrtlaunch_reduce_scatter_int32_t.h"
#include "aclrtlaunch_reduce_scatter_float16_t.h"
#include "aclrtlaunch_reduce_scatter_bfloat16_t.h"
#include "aclrtlaunch_reduce_scatter_float.h"
#include "aclrtlaunch_allgather_int8_t.h"
#include "aclrtlaunch_allgather_int32_t.h"
#include "aclrtlaunch_allgather_float16_t.h"
#include "aclrtlaunch_allgather_bfloat16_t.h"
#include "aclrtlaunch_allgather_float.h"
#include "aclrtlaunch_attention_float16_t.h"
#include "aclrtlaunch_attention_bfloat16_t.h"
#include "aclrtlaunch_sigmoid_topk_float.h"
#include "aclrtlaunch_sigmoid_topk_bfloat16_t.h"
#include "aclrtlaunch_rope_complex_float16_t.h"
#include "aclrtlaunch_rope_complex_bfloat16_t.h"
#include "kernels/ccl_param.h"

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
            throw std::runtime_error(std::string("unknown data type ") + XDtypeStr(dtype));
    }
}

void XliteOpAllGather(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                      uint32_t copySize)
{
    uint32_t rankSize = type == TP ? rt.tpSize() : rt.dpSize();
    if (in.dtype != out.dtype || in.numel * rankSize != out.numel) {
        throw std::runtime_error(std::string(__func__) +
                                 ": check tensor failed! input: " + std::to_string(in.dtype) +
                                 " output: " + std::to_string(out.dtype));
    }
    if ((in.numel * XDtypeBit(in.dtype)) % XDtypeBit(INT8)) {
        throw std::runtime_error(std::string(__func__) + ": all gather 8bit align check failed!");
    }

    auto xcclComm = (type == TP) ? rt._tpXcclComm : rt._dpXcclComm;
    auto hcclComm = (type == TP) ? rt._tpComm : rt._dpComm;
    uint32_t rank = (type == TP) ? rt.tpSize() : rt.dpSize();
    uint32_t localRank = (type == TP) ? (rt.rankId() % rt.tpSize()) : (rt.rankId() / rt.tpSize());
    size_t inBytes = in.numel * XDtypeBit(in.dtype) / 8;
    size_t outBytes = out.numel * XDtypeBit(out.dtype) / 8;

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.pool->TensorInPool(in) || !rt.pool->TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpIn = nullptr;
        XTensor *tmpOut = nullptr;

        if (needCopy) {
            tmpIn =
                &rt.pool->GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            tmpOut = &rt.pool->GetTensor(out.shape, out.dtype, DBG_LOC);
            CHECK_ACL(aclrtMemcpyAsync(tmpIn->ptr, inBytes, in.ptr, inBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            inPtr = tmpIn->ptr;
            outPtr = tmpOut->ptr;
        }

        // prevents the size of each copy from being too small.
        uint64_t sizePerRank = DIV_ROUND_UP(in.numel * XDtypeBit(in.dtype) / 8, rank);
        uint64_t maxCoreNum = rank;
        if (sizePerRank >= DOUBLE_AIVNUM_SIZE_BOUND) {
            maxCoreNum = static_cast<uint64_t>(rank) * 2;
        }
        uint32_t coreNum = rt.aivNum;
        if (coreNum > maxCoreNum) {
            coreNum = maxCoreNum;
        }

        if (coreNum >= rank) {
            coreNum = ROUND_DOWN(coreNum, rank);
            uint32_t corePerRank = coreNum / rank;
            if (corePerRank > 1 && copySize * corePerRank > MAX_TOTAL_COPY_SIZE) {
                copySize = MAX_TOTAL_COPY_SIZE / corePerRank;
            }
        }

        // call correct allreduce kernel
        switch (in.dtype) {
            case FP16:
                aclrtlaunch_allgather_float16_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                localRank, rank, xcclComm->generation++,
                                                xcclComm->dParam, copySize);
                break;
            case BF16:
                aclrtlaunch_allgather_bfloat16_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                 localRank, rank, xcclComm->generation++,
                                                 xcclComm->dParam, copySize);
                break;
            // BIT1 is packed, so count is in.numel / 8.
            case BIT1:
                aclrtlaunch_allgather_int8_t(coreNum, rt.stream, inPtr, outPtr,
                                             in.numel * XDtypeBit(in.dtype) / XDtypeBit(INT8),
                                             localRank, rank, xcclComm->generation++,
                                             xcclComm->dParam, copySize);
                break;
            case INT8:
                aclrtlaunch_allgather_int8_t(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank,
                                             rank, xcclComm->generation++, xcclComm->dParam,
                                             copySize);
                break;
            case INT32:
                aclrtlaunch_allgather_int32_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                              localRank, rank, xcclComm->generation++,
                                              xcclComm->dParam, copySize);
                break;
            case FP32:
                aclrtlaunch_allgather_float(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank,
                                            rank, xcclComm->generation++, xcclComm->dParam,
                                            copySize);
                break;
            default:
                std::cerr << __func__ << ": unsupported dtype for xccl func" << std::endl;
                break;
        }

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, outBytes, outPtr, outBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.pool->PutTensor(*tmpIn);
            rt.pool->PutTensor(*tmpOut);
        }
        return;
    }

    // fallback to HCCL path
    CHECK_HCCL(HcclAllGather(in.ptr, out.ptr, in.numel * XDtypeBit(in.dtype) / XDtypeBit(INT8),
                             HCCL_DATA_TYPE_INT8, hcclComm, rt.stream));
}

void XliteOpReduceScatter(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                          uint32_t copySize)
{
    uint32_t rankSize = type == TP ? rt.tpSize() : rt.dpSize();
    if (in.dtype != out.dtype || in.numel != out.numel * rankSize) {
        throw std::runtime_error(std::string(__func__) + ": check tensor failed!");
    }

    auto xcclComm = (type == TP) ? rt._tpXcclComm : rt._dpXcclComm;
    auto hcclComm = (type == TP) ? rt._tpComm : rt._dpComm;
    uint32_t rank = (type == TP) ? rt.tpSize() : rt.dpSize();
    uint32_t localRank = (type == TP) ? (rt.rankId() % rt.tpSize()) : (rt.rankId() / rt.tpSize());
    size_t inBytes = in.numel * XDtypeBit(in.dtype) / 8;
    size_t outBytes = out.numel * XDtypeBit(out.dtype) / 8;

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.pool->TensorInPool(in) || !rt.pool->TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpIn = nullptr;
        XTensor *tmpOut = nullptr;

        if (needCopy) {
            tmpIn =
                &rt.pool->GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            tmpOut = &rt.pool->GetTensor(out.shape, out.dtype, DBG_LOC);
            CHECK_ACL(aclrtMemcpyAsync(tmpIn->ptr, inBytes, in.ptr, inBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            inPtr = tmpIn->ptr;
            outPtr = tmpOut->ptr;
        }

        // prevents the size of each copy from being too small.
        uint64_t sizePerRank = DIV_ROUND_UP(in.numel * XDtypeBit(in.dtype) / 8, rank);
        uint64_t maxCoreNum = rank;
        if (sizePerRank >= DOUBLE_AIVNUM_SIZE_BOUND) {
            maxCoreNum = static_cast<uint64_t>(rank) * 2;
        }
        uint32_t coreNum = rt.aivNum;
        if (coreNum > maxCoreNum) {
            coreNum = maxCoreNum;
        }

        if (coreNum >= rank) {
            coreNum = ROUND_DOWN(coreNum, rank);
            uint32_t corePerRank = coreNum / rank;
            if (corePerRank > 1 && copySize * corePerRank > MAX_TOTAL_COPY_SIZE) {
                copySize = MAX_TOTAL_COPY_SIZE / corePerRank;
            }
        }

        // call correct allreduce kernel
        switch (in.dtype) {
            case FP16:
                aclrtlaunch_reduce_scatter_float16_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                     localRank, rank, xcclComm->generation++,
                                                     xcclComm->dParam, copySize);
                break;
            case BF16:
                aclrtlaunch_reduce_scatter_bfloat16_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                      localRank, rank, xcclComm->generation++,
                                                      xcclComm->dParam, copySize);
                break;
            case INT8:
                aclrtlaunch_reduce_scatter_int8_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                  localRank, rank, xcclComm->generation++,
                                                  xcclComm->dParam, copySize);
                break;
            case INT32:
                aclrtlaunch_reduce_scatter_int32_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                   localRank, rank, xcclComm->generation++,
                                                   xcclComm->dParam, copySize);
                break;
            case FP32:
                aclrtlaunch_reduce_scatter_float(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                 localRank, rank, xcclComm->generation++,
                                                 xcclComm->dParam, copySize);
                break;
            default:
                std::cerr << __func__ << ": unsupported dtype for xccl func" << std::endl;
                break;
        }

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, outBytes, outPtr, outBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.pool->PutTensor(*tmpIn);
            rt.pool->PutTensor(*tmpOut);
        }
        return;
    }

    // fallback to HCCL path
    CHECK_HCCL(HcclReduceScatter(in.ptr, out.ptr, out.numel, XDtype2HcclDtype(in.dtype),
                                 HCCL_REDUCE_SUM, hcclComm, rt.stream));
}

void XliteOpAllReduceSum(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                         uint32_t copySize)
{
    if (in.dtype != out.dtype || in.numel != out.numel) {
        throw std::runtime_error(std::string(__func__) + ": check tensor failed!");
    }

    auto xcclComm = (type == TP) ? rt._tpXcclComm : rt._dpXcclComm;
    auto hcclComm = (type == TP) ? rt._tpComm : rt._dpComm;
    uint32_t rank = (type == TP) ? rt.tpSize() : rt.dpSize();
    uint32_t localRank = (type == TP) ? (rt.rankId() % rt.tpSize()) : (rt.rankId() / rt.tpSize());
    size_t bytes = in.numel * XDtypeBit(in.dtype) / 8;

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.pool->TensorInPool(in) || !rt.pool->TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpBuff = nullptr;

        if (needCopy) {
            tmpBuff =
                &rt.pool->GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            CHECK_ACL(aclrtMemcpyAsync(tmpBuff->ptr, bytes, in.ptr, bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            inPtr = tmpBuff->ptr;
            outPtr = tmpBuff->ptr;
        }

        // prevents the size of each copy from being too small.
        uint64_t sizePerRank = DIV_ROUND_UP(in.numel * XDtypeBit(in.dtype) / 8, rank);
        uint64_t maxCoreNum = rank;
        if (sizePerRank >= DOUBLE_AIVNUM_SIZE_BOUND) {
            maxCoreNum = static_cast<uint64_t>(rank) * 2;
        }
        uint32_t coreNum = rt.aivNum;
        if (coreNum > maxCoreNum) {
            coreNum = maxCoreNum;
        }

        if (coreNum >= rank) {
            coreNum = ROUND_DOWN(coreNum, rank);
            uint32_t corePerRank = coreNum / rank;
            if (corePerRank > 1 && copySize * corePerRank > MAX_TOTAL_COPY_SIZE) {
                copySize = MAX_TOTAL_COPY_SIZE / corePerRank;
            }
        }

        // call correct allreduce kernel
        switch (in.dtype) {
            case FP16:
                aclrtlaunch_allreduce_float16_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                localRank, rank, xcclComm->generation++,
                                                xcclComm->dParam, copySize);
                break;
            case BF16:
                aclrtlaunch_allreduce_bfloat16_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                                 localRank, rank, xcclComm->generation++,
                                                 xcclComm->dParam, copySize);
                break;
            case INT8:
                aclrtlaunch_allreduce_int8_t(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank,
                                             rank, xcclComm->generation++, xcclComm->dParam,
                                             copySize);
                break;
            case INT32:
                aclrtlaunch_allreduce_int32_t(coreNum, rt.stream, inPtr, outPtr, in.numel,
                                              localRank, rank, xcclComm->generation++,
                                              xcclComm->dParam, copySize);
                break;
            case FP32:
                aclrtlaunch_allreduce_float(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank,
                                            rank, xcclComm->generation++, xcclComm->dParam,
                                            copySize);
                break;
            default:
                std::cerr << __func__ << ": unsupported dtype for xccl func" << std::endl;
                break;
        }

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, bytes, outPtr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                       rt.stream));
            rt.pool->PutTensor(*tmpBuff);
        }
        return;
    }

    // fallback to HCCL path
    CHECK_HCCL(HcclAllReduce(in.ptr, out.ptr, in.numel, XDtype2HcclDtype(in.dtype), HCCL_REDUCE_SUM,
                             hcclComm, rt.stream));
}

void XliteOpEmbed(XRuntime &rt, XTensor &in, XTensor &embed, uint32_t start, uint32_t end,
                  XTensor &out)
{
    if (embed.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_embed_kernel_float16_t(rt.aivNum, rt.stream, embed.ptr, in.ptr, out.ptr,
                                           embed.shape[1], in.shape[0], start, end, rt.tpSize());
    } else if (embed.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_embed_kernel_bfloat16_t(rt.aivNum, rt.stream, embed.ptr, in.ptr, out.ptr,
                                            embed.shape[1], in.shape[0], start, end, rt.tpSize());
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpRmsNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &out, float normEps,
                    uint32_t normDim, uint32_t cntPerToken, uint32_t startOffset)
{
    if (in.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_rmsnorm_float16_t(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, out.ptr,
                                      in.shape[0], normDim, normEps, cntPerToken, in.shape[1],
                                      startOffset);
    } else if (in.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_rmsnorm_bfloat16_t(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, out.ptr,
                                       in.shape[0], normDim, normEps, cntPerToken, in.shape[1],
                                       startOffset);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out)
{
    if (in1.dtype == FP16 && in2.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_add_float16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, out.ptr, in1.shape[0],
                                  in1.shape[1]);
    } else if (in1.dtype == BF16 && in2.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_add_bfloat16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, out.ptr, in1.shape[0],
                                   in1.shape[1]);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &norm, float normEps,
                          XTensor &out)
{
    if (in1.dtype == FP16 && in2.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_rmsnorm_float16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, norm.ptr, out.ptr,
                                      in1.shape[0], in1.shape[1], normEps, 1, in1.shape[1], 0);
    } else if (in1.dtype == BF16 && in2.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_rmsnorm_bfloat16_t(rt.aivNum, rt.stream, in1.ptr, in2.ptr, norm.ptr, out.ptr,
                                       in1.shape[0], in1.shape[1], normEps, 1, in1.shape[1], 0);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ,
                   const XTensor &bias, bool transpose, uint64_t m0, uint64_t n0, uint64_t k0,
                   uint64_t swizzle)
{
    uint64_t m = in.shape[0];
    uint64_t n = transpose ? weight.shape[1] : weight.shape[0];
    uint64_t k = transpose ? weight.shape[0] : weight.shape[1];
    if (m0 == MATMUL_M0_N0_K0_DEFAULT_VALUE || n0 == MATMUL_M0_N0_K0_DEFAULT_VALUE ||
        k0 == MATMUL_M0_N0_K0_DEFAULT_VALUE) {
        m0 = ROUND_UP(m, 32);
        if (m0 > 128) {
            m0 = 128;
        }
        n0 = (bias.ptr != nullptr) ? 128 : 256;
        k0 = 4096 / XDtypeBit(weight.dtype);

        uint64_t mLoop = DIV_ROUND_UP(m, m0);
        uint64_t nLoop = DIV_ROUND_UP(n, n0);
        uint64_t totalLoops = mLoop * nLoop;
        uint64_t lastLoops = totalLoops % rt.aicNum;

        if (totalLoops < static_cast<uint64_t>(3) * rt.aicNum &&
            (lastLoops != 0 && lastLoops < rt.aicNum / 2)) {
            if (n <= static_cast<uint64_t>(32) * rt.aicNum) {
                m0 = m0 > 64 ? 64 : m0;
                n0 = 64;
            } else if (n <= static_cast<uint64_t>(64) * rt.aicNum) {
                n0 = 64;
            } else if (n <= static_cast<uint64_t>(128) * rt.aicNum) {
                n0 = 128;
            } else if (n <= static_cast<uint64_t>(256) * rt.aicNum) {
                n0 = 256;
            } else {
                m0 = m0 > 64 ? 64 : m0;
                n0 = 384;
                k0 /= 2;
            }
        }
    }

    XlitePickSwizzle(m, n, k, &swizzle);

    if (in.dtype == FP16 && weight.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_matmul_float16_t(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k,
                                     weightNZ, transpose, m0, n0, k0, swizzle, bias.ptr);
    } else if (in.dtype == BF16 && weight.dtype == BF16 && out.dtype == BF16) {
        if (bias.ptr != nullptr) {
            XTensor &biasFp32 = rt.pool->GetTensor(bias.shape, FP32, DBG_LOC);
            aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, bias.ptr, biasFp32.ptr,
                                              bias.numel);
            aclrtlaunch_matmul_bfloat16_t(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, m, n,
                                          k, weightNZ, transpose, m0, n0, k0, swizzle,
                                          biasFp32.ptr);
            rt.pool->PutTensor(biasFp32);
        } else {
            aclrtlaunch_matmul_bfloat16_t(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, m, n,
                                          k, weightNZ, transpose, m0, n0, k0, swizzle, bias.ptr);
        }
    } else if (in.dtype == FP32 && weight.dtype == FP32 && out.dtype == FP32 && !transpose) {
        aclrtlaunch_matmul_float(rt.aicNum, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k,
                                 weightNZ, transpose, m0, n0, k0, swizzle, bias.ptr);
    } else if (in.dtype == BF16 && weight.dtype == FP32 && out.dtype == FP32 && !transpose) {
        XTensor &tmp = rt.pool->GetTensor(in.shape, FP32, DBG_LOC);
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, tmp.ptr, in.numel);
        aclrtlaunch_matmul_float(rt.aicNum, rt.stream, tmp.ptr, weight.ptr, out.ptr, m, n, k,
                                 weightNZ, transpose, m0, n0, k0, swizzle, bias.ptr);
        rt.pool->PutTensor(tmp);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out)
{
    if (in.dtype == FP16 && out.dtype == FP16) {
        aclrtlaunch_silu_and_mul_float16_t(rt.aivNum, rt.stream, in.ptr, out.ptr, nullptr,
                                           in.shape[0], out.shape[1]);
    } else if (in.dtype == BF16 && out.dtype == BF16) {
        aclrtlaunch_silu_and_mul_bfloat16_t(rt.aivNum, rt.stream, in.ptr, out.ptr, nullptr,
                                            in.shape[0], out.shape[1]);
    } else if (in.dtype == FP32 && out.dtype == FP32) {
        aclrtlaunch_silu_and_mul_float(rt.aivNum, rt.stream, in.ptr, out.ptr, nullptr, in.shape[0],
                                       out.shape[1]);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out)
{
    if (in.dtype == BF16 && out.dtype == FP32) {
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, out.ptr, in.numel);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpPermutation(XRuntime &rt, XTensor &in, XTensor &routing, uint32_t start, uint32_t end,
                        XTensor &out, XTensor &unpIdx, XTensor &counts)
{
    aclrtlaunch_permutation(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr, unpIdx.ptr,
                            counts.ptr, in.shape[0], in.shape[1], counts.shape[0], start, end);
}

void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing,
                          XTensor &weights, uint32_t start, uint32_t end, XTensor &out)
{
    if (in.dtype == BF16 && out.dtype == BF16 && weights.dtype == BF16) {
        aclrtlaunch_unpermutation_bfloat16_t(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr,
                                             unpIdx.ptr, weights.ptr, out.shape[0], in.shape[1],
                                             weights.shape[1], start, end);
    } else if (in.dtype == BF16 && out.dtype == BF16 && weights.dtype == FP32) {
        aclrtlaunch_unpermutation_float(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr,
                                        unpIdx.ptr, weights.ptr, out.shape[0], in.shape[1],
                                        weights.shape[1], start, end);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &scales,
                        XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                        long outDim, long inDim, XTensor &output, bool weightNZ, bool transpose)
{
    if (in.dtype == BF16 && weightDtype == BF16 && output.dtype == BF16) {
        aclrtlaunch_group_matmul_bfloat16_t(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr,
                                            counts.ptr, counts.shape[0], outDim, inDim, -1, -1, -1,
                                            start, end, weightNZ, transpose,
                                            MATMUL_SWIZZLE_DEFAULT_VALUE);
    } else if (in.dtype == FP16 && weightDtype == FP16 && output.dtype == FP16) {
        aclrtlaunch_group_matmul_float16_t(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr,
                                           counts.ptr, counts.shape[0], outDim, inDim, -1, -1, -1,
                                           start, end, weightNZ, transpose,
                                           MATMUL_SWIZZLE_DEFAULT_VALUE);
    } else if (in.dtype == FP32 && weightDtype == FP32 && output.dtype == FP32 && !transpose) {
        aclrtlaunch_group_matmul_float(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr,
                                       counts.ptr, counts.shape[0], outDim, inDim, -1, -1, -1,
                                       start, end, weightNZ, transpose,
                                       MATMUL_SWIZZLE_DEFAULT_VALUE);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping, uint32_t nHeads,
                      uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim, uint32_t blockSize,
                      bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW)
{
    uint32_t localHeads = nHeads / rt.tpSize();
    uint32_t localKvHeads = nKvHeads / rt.tpSize();
    localKvHeads = localKvHeads == 0 ? 1 : localKvHeads;
    uintptr_t qPtr = reinterpret_cast<uintptr_t>(inout.ptr);
    uintptr_t kPtr =
        qPtr + static_cast<uint64_t>(localHeads) * headDim * XDtypeBit(inout.dtype) / 8;
    uintptr_t vPtr =
        kPtr + static_cast<uint64_t>(localKvHeads) * headDim * XDtypeBit(inout.dtype) / 8;
    void *k = reinterpret_cast<void *>(kPtr);
    void *v = reinterpret_cast<void *>(vPtr);
    float scale = 1.0f / sqrtf(static_cast<float>(headDim));

    if (!isNeox) {
        throw std::runtime_error(std::string(__func__) + ": unsupported rope type gptj");
    }

    if (inout.dtype == FP16 && kCache.dtype == FP16 && vCache.dtype == FP16 &&
        cossin.dtype == FP16) {
        aclrtlaunch_rope_and_cache_float16_t(
            rt.aivNum, rt.stream, position.ptr, inout.ptr, k, v, cossin.ptr, kCache.ptr, vCache.ptr,
            slotMapping.ptr, inout.shape[0], rotDim, inout.shape[1], inout.shape[1], inout.shape[1],
            localHeads, localKvHeads, headDim, blockSize, scale, mropeMaskH, mropeMaskW);
    } else if (inout.dtype == BF16 && kCache.dtype == BF16 && vCache.dtype == BF16 &&
               cossin.dtype == BF16) {
        aclrtlaunch_rope_and_cache_bfloat16_t(
            rt.aivNum, rt.stream, position.ptr, inout.ptr, k, v, cossin.ptr, kCache.ptr, vCache.ptr,
            slotMapping.ptr, inout.shape[0], rotDim, inout.shape[1], inout.shape[1], inout.shape[1],
            localHeads, localKvHeads, headDim, blockSize, scale, mropeMaskH, mropeMaskW);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &qk,
                      XTensor &output, XTensor &cumPromptLens, XTensor &lens, XTensor &cachedLens,
                      XTensor &blockTables, uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                      uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
{
    if (qkv.dtype == FP16 && qk.dtype == FP16 && kCache.dtype == FP16 && vCache.dtype == FP16 &&
        output.dtype == FP16) {
        aclrtlaunch_attention_float16_t(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, vCache.ptr,
                                        qk.ptr, output.ptr, cumPromptLens.ptr, lens.ptr,
                                        cachedLens.ptr, blockTables.ptr, nHeads, nKvHeads, headDim,
                                        blockSize, batch, maxNumBlock);
    } else if (qkv.dtype == BF16 && qk.dtype == BF16 && kCache.dtype == BF16 &&
               vCache.dtype == BF16 && output.dtype == BF16) {
        aclrtlaunch_attention_bfloat16_t(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, vCache.ptr,
                                         qk.ptr, output.ptr, cumPromptLens.ptr, lens.ptr,
                                         cachedLens.ptr, blockTables.ptr, nHeads, nKvHeads, headDim,
                                         blockSize, batch, maxNumBlock);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpRopeComplex(XRuntime &rt, uint32_t numTokens, uint32_t nLocalHeads, uint32_t stepDim,
                        uint32_t ropeDim, XTensor &inputWithR, XTensor &freqs, XTensor &position,
                        XTensor &vGather, XTensor &outputPe, enum XRopeType ropeType)
{
    uint32_t type = 0;
    switch (ropeType) {
        case NORMAL:
            type = 0x1;
            break;
        case INPLACE:
            type = 0x2;
            break;
        case MIX:
            type = 0x3;
            break;
        default:
            throw std::runtime_error(std::string(__func__) + ": unknown rope type");
    }

    if (inputWithR.dtype == FP16) {
        aclrtlaunch_rope_complex_float16_t(rt.aivNum, rt.stream, numTokens, nLocalHeads, stepDim,
                                           ropeDim, inputWithR.ptr, freqs.ptr, position.ptr,
                                           outputPe.ptr, vGather.ptr, type);
    } else if (inputWithR.dtype == BF16) {
        aclrtlaunch_rope_complex_bfloat16_t(rt.aivNum, rt.stream, numTokens, nLocalHeads, stepDim,
                                            ropeDim, inputWithR.ptr, freqs.ptr, position.ptr,
                                            outputPe.ptr, vGather.ptr, type);
    } else {
        throw std::runtime_error(std::string(__func__) + ": TODO");
    }
}

void XliteDsOpRopeBatch(XRuntime &rt, uint32_t numTokens, uint32_t nLocalHeads, uint32_t stepDim,
                        uint32_t ropeDim, XTensor &inputWithR, XTensor &freqs, XTensor &position,
                        XTensor &vGather, XTensor &outputPe, enum XRopeType ropeType)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpStridedRmsnorm(XRuntime &rt, XTensor &input, XTensor &w, XTensor &output,
                             uint32_t numTokens, uint32_t normDim, uint32_t stepDim, float normEps)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpReshapeAndCache(XRuntime &rt, XTensor &key, XTensor &value, XTensor &kCache,
                              XTensor &vCache, XTensor &slotMapping, int32_t numTokens,
                              int32_t keyStride, int32_t valueStride, int32_t numKvHeads,
                              int32_t kHeadSize, int32_t vHeadSize, int32_t blockSize,
                              int32_t blockNum)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpKvMatmul(XRuntime &rt, XTensor &input, XTensor &w, XTensor &output, int m, int n,
                       int k, XTensor &blockTable, bool nt, int blockSize, int headSize)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpPrefillKvSplit(XRuntime &rt, XTensor &kv, XTensor &kPe, XTensor &cache,
                             XTensor &blockTable, XTensor &kvFull, XTensor &v, int nTokens,
                             int nTokensPad, int nLocalHeads, int kvLoraRank, int rotDim,
                             int headSize, int vDim, uint32_t blockSize)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpPrefillMix(XRuntime &rt, XTensor &out, XTensor &alpha, XTensor &max, XTensor &sum,
                         XTensor &q, XTensor &k, XTensor &qk, XTensor &blockTables,
                         XTensor &cachedLens, XTensor &v, XTensor &mixOut, XTensor &mixOutFinal,
                         XTensor &promptLens, XTensor &attnMask, XTensor &attnMaskAddr,
                         XTensor &speculateLens, XTensor &prefillIndex, XTensor &cumPromptLens,
                         uint32_t headSize, uint32_t numHeads, uint32_t numKVHeads,
                         uint32_t blockSize, uint32_t batchSize, uint32_t mappingLen,
                         uint32_t doTreeAttnMask, uint32_t offsetM, uint32_t mSlice, float scale)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpEinsumShdHdcShc(XRuntime &rt, int numTokens, int headSize, int nLocalHeads,
                              int qStepDim, int kvUpWeightStepDim, int kvLoraRank, XTensor &qWithQr,
                              XTensor &kvUpWeight, XTensor &qAbsorb)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpDecodeAttn(XRuntime &rt, XTensor &q, XTensor &k, XTensor &o, XTensor &cachedLens,
                         XTensor &mapping, XTensor &promptLens, XTensor &promptLensCum,
                         uint32_t numTokens, uint32_t numHeads, uint32_t numKvHeads,
                         uint32_t headSize, uint32_t blockSize, uint32_t mappingLen,
                         uint32_t maxContextLen, bool add)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpSoftmax(XRuntime &rt, XTensor &qk, XTensor &cachedLens, XTensor &promptLens,
                      XTensor &promptLensCum, float scale, uint32_t numTokens, uint32_t numHeads,
                      uint32_t blockSize, uint32_t maxContextLen)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpEinsumShtTcShc(XRuntime &rt, int numTokens, int nLocalHeads, int maxTokens,
                             int maxBlocksPerQuery, int numBlocks, int blockSize, int kvLoraRank,
                             XTensor &scores, XTensor &cachedLens, XTensor &promptLens,
                             XTensor &promptLensCum, XTensor &blockTables, XTensor &cCache,
                             XTensor &result)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteDsOpEinsumShcHdcShd(XRuntime &rt, int numTokens, int nLocalHeads, int kvLoraRank,
                              int wkvbStep, int vdim, XTensor &scores, XTensor &kvUpWeight,
                              XTensor &result)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output)
{
    if (input.dtype == FP32 && weight.dtype == FP32 && output.dtype == FP32) {
        aclrtlaunch_add_bias_float(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr,
                                   output.shape[0] * output.shape[1], output.shape[1]);
    } else if (input.dtype == FP16 && weight.dtype == FP16 && output.dtype == FP16) {
        aclrtlaunch_add_bias_float16_t(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr,
                                       output.shape[0] * output.shape[1], output.shape[1]);
    } else if (input.dtype == BF16 && weight.dtype == BF16 && output.dtype == BF16) {
        aclrtlaunch_add_bias_bfloat16_t(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr,
                                        output.shape[0] * output.shape[1], output.shape[1]);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outWeights,
                        XTensor &outRouting, uint32_t topK, bool normTopKProb)
{
    if (scores.dtype == FP32 && indices.dtype == INT32 && outWeights.dtype == FP32 &&
        outRouting.dtype == BIT1) {
        aclrtlaunch_softmax_topk_float(rt.aivNum, rt.stream, scores.ptr, indices.ptr,
                                       outWeights.ptr, outRouting.ptr, scores.shape[0],
                                       indices.shape[0], topK, normTopKProb);
    } else if (scores.dtype == BF16 && indices.dtype == INT32 && outWeights.dtype == BF16 &&
               outRouting.dtype == BIT1) {
        aclrtlaunch_softmax_topk_bfloat16_t(rt.aivNum, rt.stream, scores.ptr, indices.ptr,
                                            outWeights.ptr, outRouting.ptr, scores.shape[0],
                                            indices.shape[0], topK, normTopKProb);
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpSigmoidTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias, float scale,
                        XTensor &outWeights, XTensor &outRouting, uint32_t nGroup,
                        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)
{
    if (scores.dtype == FP32 && indices.dtype == INT32 && outWeights.dtype == FP32 &&
        outRouting.dtype == BIT1) {
        aclrtlaunch_sigmoid_topk_float(rt.aivNum, rt.stream, scores.ptr, indices.ptr, bias.ptr,
                                       scale, outWeights.ptr, outRouting.ptr, scores.shape[0],
                                       indices.shape[0], nGroup, nTopkGroup, topK, normTopKProb);
    } else if (scores.dtype == BF16 && indices.dtype == INT32 && outWeights.dtype == BF16 &&
               outRouting.dtype == BIT1) {
        aclrtlaunch_sigmoid_topk_bfloat16_t(rt.aivNum, rt.stream, scores.ptr, indices.ptr, bias.ptr,
                                            scale, outWeights.ptr, outRouting.ptr, scores.shape[0],
                                            indices.shape[0], nGroup, nTopkGroup, topK,
                                            normTopKProb);
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
        throw std::runtime_error(std::string(__func__) + ": unsupported!");
    }
}

void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x, XTensor &expBuf)
{
    if (x.dtype == FP16) {
        aclrtlaunch_softmax_long_float16_t(1, rt.stream, x.ptr, expBuf.ptr, x.shape[0], x.shape[1],
                                           calcLen);
    } else if (x.dtype == BF16) {
        aclrtlaunch_softmax_long_bfloat16_t(1, rt.stream, x.ptr, expBuf.ptr, x.shape[0], x.shape[1],
                                            calcLen);
    } else {
        std::cerr << __func__ << ": unsupported!" << std::endl;
    }
}
