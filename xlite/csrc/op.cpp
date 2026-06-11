/*
 * Copyright (C) 2025 - 2026. Huawei Technologies Co., Ltd. All rights reserved.
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
#include "aclrtlaunch_group_matmul_int8_t.h"
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
#include "aclrtlaunch_quant_bf16_to_i8_static.h"
#include "aclrtlaunch_quant_bf16_to_i8_dynamic.h"
#include "aclrtlaunch_matmul_int8_t.h"
#include "aclrtlaunch_dequant_float16_t.h"
#include "aclrtlaunch_attention_float16_t.h"
#include "aclrtlaunch_attention_bfloat16_t.h"
#include "aclrtlaunch_sigmoid_topk_float.h"
#include "aclrtlaunch_sigmoid_topk_bfloat16_t.h"
#include "aclrtlaunch_rope_complex_and_cache_float16_t.h"
#include "aclrtlaunch_rope_complex_and_cache_bfloat16_t.h"
#include "aclrtlaunch_flash_attention_float16_t.h"
#include "aclrtlaunch_flash_attention_bfloat16_t.h"
#include "aclrtlaunch_flash_mla_bfloat16_t.h"
#include "aclrtlaunch_norm_float16_t.h"
#include "aclrtlaunch_norm_bfloat16_t.h"
#include "aclrtlaunch_indexer_scores_float16_t.h"
#include "aclrtlaunch_indexer_scores_bfloat16_t.h"
#include "aclrtlaunch_muls_float16_t.h"
#include "aclrtlaunch_muls_bfloat16_t.h"
#include "aclrtlaunch_topk_float.h"
#include "aclrtlaunch_topk_bfloat16_t.h"
#include "aclrtlaunch_mla_bfloat16_t.h"
#include "aclrtlaunch_experts_counts_sum.h"
#include "aclrtlaunch_reorder_moe.h"
#include "aclrtlaunch_conv1d_and_silu_float.h"
#include "aclrtlaunch_conv1d_and_silu_float16_t.h"
#include "aclrtlaunch_conv1d_and_silu_bfloat16_t.h"
#include "aclrtlaunch_beta_decay_float.h"
#include "aclrtlaunch_beta_decay_float16_t.h"
#include "aclrtlaunch_beta_decay_bfloat16_t.h"
#include "aclrtlaunch_transpose_1_2_float.h"
#include "aclrtlaunch_transpose_1_2_float16_t.h"
#include "aclrtlaunch_transpose_1_2_bfloat16_t.h"

#define KERNEL_PTR_TYPE(name) decltype(aclrtlaunch_##name##_bfloat16_t)

static inline bool IsDummyRuntime(const XRuntime &rt)
{
    return rt.IsDummyRuntime();
}

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

template <typename... Args>
static bool EachXDtype(enum XDtype dtype, Args &&...args)
{
    return (... && (std::forward<Args>(args).dtype == dtype));
}

void XliteOpAllGather(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                      uint32_t copySize)
{
    uint32_t rankSize = rt.tpSize();
    if (type == DP) {
        rankSize = rt.dpSize();
    } else if (type == EP) {
        rankSize = rt.moeEpSize();
    }
    if (in.dtype != out.dtype || in.numel * rankSize != out.numel) {
        std::stringstream ss;
        ss << __func__ << ": check tensor failed!"
           << " in.dtype=" << XDtypeStr(in.dtype) << "(" << in.dtype << ")"
           << " out.dtype=" << XDtypeStr(out.dtype) << "(" << out.dtype << ")"
           << " in.numel=" << in.numel << " rankSize=" << rankSize
           << " expected out.numel=" << (in.numel * rankSize) << " actual out.numel=" << out.numel;
        throw std::runtime_error(ss.str());
    }
    if ((in.numel * XDtypeBit(in.dtype)) % XDtypeBit(INT8)) {
        throw std::runtime_error(std::string(__func__) + ": all gather 8bit align check failed!");
    }

    XcclComm *xcclComm = nullptr;
    HcclComm hcclComm = nullptr;
    uint32_t rank = 0;
    uint32_t localRank = 0;
    if (type == TP) {
        xcclComm = rt._tpXcclComm;
        hcclComm = rt._tpComm;
        rank = rt.tpSize();
        localRank = rt.rankId() % rt.tpSize();
    } else if (type == DP) {
        xcclComm = rt._dpXcclComm;
        hcclComm = rt._dpComm;
        rank = rt.dpSize();
        localRank = rt.rankId() / rt.tpSize();
    } else if (type == EP) {
        xcclComm = rt._epXcclComm;
        hcclComm = rt._epComm;
        rank = rt.moeEpSize();
        localRank = rt.rankId() / rt.moeTpSize();
    }
    size_t inBytes = in.numel * XDtypeBit(in.dtype) / 8;
    size_t outBytes = out.numel * XDtypeBit(out.dtype) / 8;

    if (IsDummyRuntime(rt)) {
        if (xcclComm && in.dtype != INT64 && rankSize > 1) {
            bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
            if (needCopy) {
                XTensor &tmpIn =
                    rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
                XTensor &tmpOut = rt.GetTensor(out.shape, out.dtype, DBG_LOC);
                rt.PutTensor(tmpIn);
                rt.PutTensor(tmpOut);
            }
        }
        return;
    }

    if (rankSize <= 1) {
        if (in.ptr != out.ptr) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, outBytes, in.ptr, inBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }
        return;
    }

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpIn = nullptr;
        XTensor *tmpOut = nullptr;

        if (needCopy) {
            tmpIn = &rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            tmpOut = &rt.GetTensor(out.shape, out.dtype, DBG_LOC);
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
        uintptr_t count;
        KERNEL_PTR_TYPE(allgather) * launchKernel;
        switch (in.dtype) {
            case FP16:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_float16_t;
                break;
            case BF16:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_bfloat16_t;
                break;
            // BIT1 is packed, so count is in.numel / 8.
            case BIT1:
                count = in.numel * XDtypeBit(in.dtype) / XDtypeBit(INT8);
                launchKernel = aclrtlaunch_allgather_int8_t;
                break;
            case INT8:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_int8_t;
                break;
            case INT32:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_int32_t;
                break;
            case FP32:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_float;
                break;
            default:
                std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
                throw std::runtime_error(err_str + " unsupported dtype for xccl func");
                break;
        }
        launchKernel(coreNum, rt.stream, inPtr, outPtr, count, localRank, rank,
                     xcclComm->generation++, xcclComm->dParam, copySize);

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, outBytes, outPtr, outBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.PutTensor(*tmpIn);
            rt.PutTensor(*tmpOut);
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

    if (IsDummyRuntime(rt)) {
        if (xcclComm && in.dtype != INT64 && rankSize > 1) {
            bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
            if (needCopy) {
                XTensor &tmpIn =
                    rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
                XTensor &tmpOut = rt.GetTensor(out.shape, out.dtype, DBG_LOC);
                rt.PutTensor(tmpIn);
                rt.PutTensor(tmpOut);
            }
        }
        return;
    }

    if (rankSize <= 1) {
        if (in.ptr != out.ptr) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, outBytes, in.ptr, inBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }
        return;
    }

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpIn = nullptr;
        XTensor *tmpOut = nullptr;

        if (needCopy) {
            tmpIn = &rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            tmpOut = &rt.GetTensor(out.shape, out.dtype, DBG_LOC);
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

        KERNEL_PTR_TYPE(reduce_scatter) * launchKernel;
        // call correct allreduce kernel
        switch (in.dtype) {
            case FP16:
                launchKernel = aclrtlaunch_reduce_scatter_float16_t;
                break;
            case BF16:
                launchKernel = aclrtlaunch_reduce_scatter_bfloat16_t;
                break;
            case INT8:
                launchKernel = aclrtlaunch_reduce_scatter_int8_t;
                break;
            case INT32:
                launchKernel = aclrtlaunch_reduce_scatter_int32_t;
                break;
            case FP32:
                launchKernel = aclrtlaunch_reduce_scatter_float;
                break;
            default:
                std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
                throw std::runtime_error(err_str + " unsupported dtype for xccl func");
                break;
        }
        launchKernel(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank, rank,
                     xcclComm->generation++, xcclComm->dParam, copySize);

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, outBytes, outPtr, outBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.PutTensor(*tmpIn);
            rt.PutTensor(*tmpOut);
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

    if (IsDummyRuntime(rt)) {
        if (xcclComm && in.dtype != INT64 && rank > 1) {
            bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
            if (needCopy) {
                XTensor &tmpBuff =
                    rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
                rt.PutTensor(tmpBuff);
            }
        }
        return;
    }

    if (rank <= 1) {
        if (in.ptr != out.ptr) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, bytes, in.ptr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                       rt.stream));
        }
        return;
    }

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpBuff = nullptr;

        if (needCopy) {
            tmpBuff = &rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
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
        KERNEL_PTR_TYPE(allreduce) * launchKernel;
        switch (in.dtype) {
            case FP16:
                launchKernel = aclrtlaunch_allreduce_float16_t;
                break;
            case BF16:
                launchKernel = aclrtlaunch_allreduce_bfloat16_t;
                break;
            case INT8:
                launchKernel = aclrtlaunch_allreduce_int8_t;
                break;
            case INT32:
                launchKernel = aclrtlaunch_allreduce_int32_t;
                break;
            case FP32:
                launchKernel = aclrtlaunch_allreduce_float;
                break;
            default:
                std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
                throw std::runtime_error(err_str + " unsupported dtype for xccl func");
                break;
        }
        launchKernel(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank, rank,
                     xcclComm->generation++, xcclComm->dParam, copySize);

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, bytes, outPtr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                       rt.stream));
            rt.PutTensor(*tmpBuff);
        }
        return;
    }

    // fallback to HCCL path
    CHECK_HCCL(HcclAllReduce(in.ptr, out.ptr, in.numel, XDtype2HcclDtype(in.dtype), HCCL_REDUCE_SUM,
                             hcclComm, rt.stream));
}

void XliteOpAlltoAllV(XRuntime &rt, XTensor &in, XTensor &out, XTensor &sendCounts,
                      XTensor &recvCounts, XTensor &sdispls, XTensor &rdispls, enum commType type)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    if (in.dtype != out.dtype) {
        throw std::runtime_error(std::string(__func__) +
                                 ": check tensor failed! input: " + std::to_string(in.dtype) +
                                 " output: " + std::to_string(out.dtype));
    }

    HcclComm hcclComm = rt._tpComm;
    if (type == DP) {
        hcclComm = rt._dpComm;
    } else if (type == EP) {
        hcclComm = rt._epComm;
    }

    CHECK_HCCL(HcclAlltoAllV(in.ptr, sendCounts.ptr, sdispls.ptr, XDtype2HcclDtype(in.dtype),
                             out.ptr, recvCounts.ptr, rdispls.ptr, XDtype2HcclDtype(out.dtype),
                             hcclComm, rt.stream));
}

void XliteOpEmbed(XRuntime &rt, XTensor &in, XTensor &embed, uint32_t start, uint32_t end,
                  XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(embed_kernel) * launchKernel;
    if (EachXDtype(FP16, embed, out)) {
        launchKernel = aclrtlaunch_embed_kernel_float16_t;
    } else if (EachXDtype(BF16, embed, out)) {
        launchKernel = aclrtlaunch_embed_kernel_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(embed) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, embed.ptr, in.ptr, out.ptr, embed.shape[1], in.shape[0],
                 start, end, rt.tpSize());
}

void XliteOpRmsNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &out, float normEps,
                    uint32_t normDim, bool useNorm, const XTensor &normBias, uint32_t cntPerToken,
                    uint32_t inStartOffset, uint32_t outStartOffset, const XTensor &variance)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (in.dtype == FP16 && (out.dtype == FP16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (in.dtype == BF16 && (out.dtype == BF16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(in) + XT_STR(norm) + XT_STR(out) + XT_STR(normBias);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, normBias.ptr, out.ptr,
                 in.shape[0], normDim, normEps, false, cntPerToken, in.shape[1], out.shape[1],
                 inStartOffset, outStartOffset, useNorm, variance.ptr, rt.tpSize(), false);
}

void XliteOpLayerNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &normBias, XTensor &out,
                      float normEps, uint32_t normDim, uint32_t cntPerToken, uint32_t inStartOffset,
                      uint32_t outStartOffset)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (EachXDtype(FP16, in, out)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (EachXDtype(BF16, in, out)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(in) + XT_STR(norm) + XT_STR(normBias) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, normBias.ptr, out.ptr,
                 in.shape[0], normDim, normEps, true, cntPerToken, in.shape[1], out.shape[1],
                 inStartOffset, outStartOffset, true, nullptr, rt.tpSize(), false);
}

void XliteOpL2Norm(XRuntime &rt, XTensor &in, XTensor &out, float normEps, uint32_t normDim,
                   uint32_t cntPerToken, uint32_t inStartOffset, uint32_t outStartOffset)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (in.dtype == FP16 && (out.dtype == FP16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (in.dtype == BF16 && (out.dtype == BF16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in.ptr, nullptr, nullptr, nullptr, out.ptr, in.shape[0],
                 normDim, normEps, false, cntPerToken, in.shape[1], out.shape[1], inStartOffset,
                 outStartOffset, true, nullptr, rt.tpSize(), true);
}

void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(add) * launchKernel;
    if (EachXDtype(FP16, in1, in2, out)) {
        launchKernel = aclrtlaunch_add_float16_t;
    } else if (EachXDtype(BF16, in1, in2, out)) {
        launchKernel = aclrtlaunch_add_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in1) + XT_STR(in2) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in1.ptr, in2.ptr, out.ptr, in1.shape[0], in1.shape[1]);
}

void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in, XTensor &addInOut, XTensor &norm,
                          float normEps, XTensor &out, const XTensor &normBias)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (EachXDtype(FP16, in, addInOut, out)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (EachXDtype(BF16, in, addInOut, out)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(in) + XT_STR(addInOut) + XT_STR(norm) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in.ptr, addInOut.ptr, norm.ptr, normBias.ptr, out.ptr,
                 in.shape[0], in.shape[1], normEps, false, 1, in.shape[1], out.shape[1], 0, 0, true,
                 nullptr, rt.tpSize(), false);
}

void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ,
                   const XTensor &bias, const XTensor &deqScale, bool transpose, uint64_t m0,
                   uint64_t n0, uint64_t k0)
{
    if (IsDummyRuntime(rt)) {
        if (EachXDtype(BF16, in, weight, out) && bias.ptr != nullptr) {
            XTensor &biasFp32 = rt.GetTensor(bias.shape, FP32, DBG_LOC);
            rt.PutTensor(biasFp32);
        } else if (in.dtype == BF16 && weight.dtype == FP32 && out.dtype == FP32 && !transpose) {
            XTensor &tmp = rt.GetTensor(in.shape, FP32, DBG_LOC);
            rt.PutTensor(tmp);
        }
        return;
    }

    uint64_t m = in.shape[0];
    uint64_t n = transpose ? weight.shape[1] : weight.shape[0];
    uint64_t k = transpose ? weight.shape[0] : weight.shape[1];
    bool needExtraSpace = (bias.ptr != nullptr || deqScale.ptr != nullptr);
    uint64_t mLoop;
    uint64_t nLoop;
    uint64_t totalLoops;
    uint64_t swizzle = rt.defaultMatmulSwizzle;

    // Notice: Ensure that no overflow occurs
    // L1(512K): PINGPONG * (sizeof(x) * m0 * 2k0 + sizeof(y) * n0 * k0) + BiasSize(Optional)] +
    // FixPipe(Optional)
    //         = 4 * sizeof(x) * m0 * k0 + 2 * k0 * sizeof(y) * n0 + [4 * n0] + [8 * n0]
    //         = 2 * k0 * (2 * sizeof(x) * m0 + sizeof(y) * n0) + [12 * n0]
    // L0A(64K): PINGPONG * sizeof(x) * m0 * k0 / 4 = sizeof(x) * m0 * k0 / 2
    // L0B(64K): PINGPONG * sizeof(y) * m0 * k0 / 4 = sizeof(y) * m0 * k0 / 2
    // BiasTable(1K): n0 * sizeof(float/int32_t) = 4 * n0
    // FixPipe(2K): n0 * sizeof(uint64_t) = 8 * n0
    if (m0 == MATMUL_M0_N0_K0_DEFAULT_VALUE || n0 == MATMUL_M0_N0_K0_DEFAULT_VALUE ||
        k0 == MATMUL_M0_N0_K0_DEFAULT_VALUE) {
        m0 = ROUND_UP(m, 32);
        if (m0 > 128) {
            m0 = 128;
        }
        // if matmul has bias or dequant scale, L1 buffer will overflow!
        n0 = needExtraSpace ? 128 : 256;
        k0 = 4096 / XDtypeBit(weight.dtype);

        mLoop = DIV_ROUND_UP(m, m0);
        nLoop = DIV_ROUND_UP(n, n0);
        totalLoops = mLoop * nLoop;
        uint64_t lastLoops = totalLoops % rt.aicNum;

        // If the data size is small, we should make a data tiling mode
        // to ensure even loads on each AICore.
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
                n0 = needExtraSpace ? 128 : 256;
            } else {
                m0 = m0 > 64 ? 64 : m0;
                // BiasTable(1K): 4 * n0 <= 1K, so that n0 <= 256
                n0 = needExtraSpace ? 256 : 384;
                k0 /= 2;
            }
        }
    }
    mLoop = DIV_ROUND_UP(m, m0);
    nLoop = DIV_ROUND_UP(n, n0);
    totalLoops = mLoop * nLoop;
    uint32_t aicNum = totalLoops > rt.aicNum ? rt.aicNum : totalLoops;
    if (aicNum == 0) {
        aicNum = 1;
    }

    if (!rt.disableSwizzleTable) {
        XlitePickSwizzle(n, k, &swizzle);
    }

    KERNEL_PTR_TYPE(matmul) * launchKernel;
    XTensor *castedIn = nullptr;
    XTensor *castedBias = nullptr;
    if (EachXDtype(FP16, in, weight, out)) {
        launchKernel = aclrtlaunch_matmul_float16_t;
    } else if (EachXDtype(BF16, in, weight, out)) {
        if (bias.ptr != nullptr) {
            castedBias = &rt.GetTensor(bias.shape, FP32, DBG_LOC);
            aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, bias.ptr, castedBias->ptr,
                                              bias.numel);
        }
        launchKernel = aclrtlaunch_matmul_bfloat16_t;
    } else if (EachXDtype(FP32, in, weight, out) && !transpose) {
        launchKernel = aclrtlaunch_matmul_float;
    } else if (in.dtype == BF16 && EachXDtype(FP32, weight, out) && !transpose) {
        castedIn = &rt.GetTensor(in.shape, FP32, DBG_LOC);
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, castedIn->ptr, in.numel);
        launchKernel = aclrtlaunch_matmul_float;
    } else if (EachXDtype(INT8, in, weight) && out.dtype == FP16) {
        launchKernel = aclrtlaunch_matmul_int8_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(weight) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    auto inPtr = castedIn ? castedIn->ptr : in.ptr;
    auto biasPtr = castedBias ? castedBias->ptr : bias.ptr;

    launchKernel(aicNum, rt.stream, inPtr, weight.ptr, out.ptr, m, n, k, weightNZ, transpose, m0,
                 n0, k0, swizzle, biasPtr, deqScale.ptr);
    if (castedIn) {
        rt.PutTensor(*castedIn);
    }
    if (castedBias) {
        rt.PutTensor(*castedBias);
    }
}

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &num)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    KERNEL_PTR_TYPE(silu_and_mul) * launchKernel;
    if (EachXDtype(FP16, in, out)) {
        launchKernel = aclrtlaunch_silu_and_mul_float16_t;
    } else if (EachXDtype(BF16, in, out)) {
        launchKernel = aclrtlaunch_silu_and_mul_bfloat16_t;
    } else if (EachXDtype(FP32, in, out)) {
        launchKernel = aclrtlaunch_silu_and_mul_float;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in.ptr, out.ptr, num.ptr, in.shape[0], out.shape[1]);
}

void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (in.dtype == BF16 && out.dtype == FP32) {
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, out.ptr, in.numel);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(inScale) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpPermutation(XRuntime &rt, XTensor &in, XTensor &routing, uint32_t start, uint32_t end,
                        XTensor &out, XTensor &unpIdx, XTensor &counts)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    aclrtlaunch_permutation(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr, unpIdx.ptr,
                            counts.ptr, in.shape[0], in.shape[1], counts.shape[0], start, end);
}

void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing,
                          XTensor &weights, uint32_t start, uint32_t end, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(unpermutation) * launchKernel;
    if (EachXDtype(BF16, in, out, weights)) {
        launchKernel = aclrtlaunch_unpermutation_bfloat16_t;
    } else if (in.dtype == BF16 && out.dtype == BF16 && weights.dtype == FP32) {
        launchKernel = aclrtlaunch_unpermutation_float;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(unpIdx) + XT_STR(routing) +
                              XT_STR(weights) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr, unpIdx.ptr, weights.ptr,
                 out.shape[0], in.shape[1], weights.shape[1], start, end);
}

// deqScales: uint64_t, 低 32 位 TF32 格式有效, 1 符号位, 8 指数位, 10 尾数位, 后 13 位不参与计算
void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                        XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                        long outDim, long inDim, XTensor &output, bool weightNZ, bool transpose)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    KERNEL_PTR_TYPE(group_matmul) * launchKernel;
    if (in.dtype == BF16 && weightDtype == BF16 && output.dtype == BF16) {
        launchKernel = aclrtlaunch_group_matmul_bfloat16_t;
    } else if (in.dtype == FP16 && weightDtype == FP16 && output.dtype == FP16) {
        launchKernel = aclrtlaunch_group_matmul_float16_t;
    } else if (in.dtype == FP32 && weightDtype == FP32 && output.dtype == FP32 && !transpose) {
        launchKernel = aclrtlaunch_group_matmul_float;
    } else if (in.dtype == INT8 && weightDtype == INT8 && output.dtype == FP16) {
        launchKernel = aclrtlaunch_group_matmul_int8_t;
    } else {
        std::string err_str = DBG_PREFIX;
        err_str += XT_STR(in) + XT_STR(output) + ", weight dtype:" + XDtypeStr(weightDtype);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr, deqScales.ptr, counts.ptr,
                 counts.shape[0], outDim, inDim, -1, -1, -1, start, end, weightNZ, transpose,
                 rt.defaultMatmulSwizzle);
}

void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping, uint32_t nHeads,
                      uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim, uint32_t blockSize,
                      bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
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

    KERNEL_PTR_TYPE(rope_and_cache) * launchKernel;
    if (EachXDtype(FP16, inout, kCache, vCache, cossin)) {
        launchKernel = aclrtlaunch_rope_and_cache_float16_t;
    } else if (EachXDtype(BF16, inout, kCache, vCache, cossin)) {
        launchKernel = aclrtlaunch_rope_and_cache_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(inout) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(position) + XT_STR(cossin) + XT_STR(slotMapping);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, position.ptr, inout.ptr, k, v, cossin.ptr, kCache.ptr,
                 vCache.ptr, slotMapping.ptr, inout.shape[0], rotDim, inout.shape[1],
                 inout.shape[1], inout.shape[1], localHeads, localKvHeads, headDim, blockSize,
                 scale, mropeMaskH, mropeMaskW);
}

void XliteOpAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &qk,
                      XTensor &output, XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                      XTensor &blockTables, uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                      uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(attention) * launchKernel;
    if (EachXDtype(FP16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_attention_float16_t;
    } else if (EachXDtype(BF16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_attention_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qkv) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(qk) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, vCache.ptr, qk.ptr, output.ptr,
                 queryStartLoc.ptr, lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, nKvHeads,
                 headDim, blockSize, batch, maxNumBlock);
}

void XliteOpFlashAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache,
                           XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum, XTensor &lastMax,
                           XTensor &lastSum, XTensor &sync, XTensor &output, XTensor &queryStartLoc,
                           XTensor &lens, XTensor &cachedLens, XTensor &blockTables,
                           uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t blockSize,
                           uint32_t batch, uint32_t maxNumBlock, uint32_t tileSizeOfCachedKV)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    KERNEL_PTR_TYPE(flash_attention) * launchKernel;
    if (EachXDtype(FP16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_flash_attention_float16_t;
    } else if (EachXDtype(BF16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_flash_attention_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qkv) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(qk) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, vCache.ptr, qk.ptr, sv.ptr, max.ptr,
                 sum.ptr, lastMax.ptr, lastSum.ptr, sync.ptr, output.ptr, queryStartLoc.ptr,
                 lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, nKvHeads, headDim, blockSize,
                 batch, maxNumBlock, tileSizeOfCachedKV);
}

void XliteOpFlashMLA(XRuntime &rt, XTensor &qWithQr, XTensor &kCache, XTensor &vCache,
                     XTensor &wkvb, XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum,
                     XTensor &lastMax, XTensor &lastSum, XTensor &sync, XTensor &output,
                     XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                     XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                     uint32_t nopeHeadDim, uint32_t vHeadDim, uint32_t kvLoraRank,
                     uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock, float scale,
                     bool weightNZ, uint32_t tileSizeOfCachedKV)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (EachXDtype(BF16, qWithQr, kCache, vCache, wkvb, output)) {
        aclrtlaunch_flash_mla_bfloat16_t(
            rt.aicNum, rt.stream, qWithQr.ptr, kCache.ptr, vCache.ptr, wkvb.ptr, qk.ptr, sv.ptr,
            max.ptr, sum.ptr, lastMax.ptr, lastSum.ptr, sync.ptr, output.ptr, queryStartLoc.ptr,
            lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, ropeHeadDim, nopeHeadDim, vHeadDim,
            kvLoraRank, blockSize, batch, maxNumBlock, scale, weightNZ, tileSizeOfCachedKV);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qWithQr) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(wkvb) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpMLA(XRuntime &rt, XTensor &qWithQr, XTensor &kCache, XTensor &vCache, XTensor &wkvb,
                XTensor &qk, XTensor &output, XTensor &queryStartLoc, XTensor &lens,
                XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                uint32_t nopeHeadDim, uint32_t vHeadDim, uint32_t kvLoraRank, uint32_t blockSize,
                uint32_t batch, uint32_t maxNumBlock, float scale, bool weightNZ, uint32_t topK,
                const XTensor &topkIndices)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (EachXDtype(BF16, qWithQr, kCache, vCache, wkvb, output)) {
        aclrtlaunch_mla_bfloat16_t(rt.aicNum, rt.stream, qWithQr.ptr, kCache.ptr, vCache.ptr,
                                   wkvb.ptr, topkIndices.ptr, qk.ptr, output.ptr, queryStartLoc.ptr,
                                   lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, ropeHeadDim,
                                   nopeHeadDim, vHeadDim, kvLoraRank, blockSize, batch, maxNumBlock,
                                   scale, weightNZ, topK);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qWithQr) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(wkvb) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpRopeComplex(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t ropeDim,
                        uint32_t offset, XTensor &inputWithR, XTensor &freqs, XTensor &position)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    KERNEL_PTR_TYPE(rope_complex_and_cache) * launchKernel;
    if (inputWithR.dtype == FP16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_float16_t;
    } else if (inputWithR.dtype == BF16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(inputWithR);
        throw std::runtime_error(err_str + " TODO");
    }
    launchKernel(rt.aivNum, rt.stream, inputWithR.shape[0], nLocalHeads, stepDim, ropeDim, offset,
                 0, 0, inputWithR.ptr, freqs.ptr, position.ptr, 0, nullptr, nullptr, nullptr,
                 nullptr);
}

void XliteOpRopeComplexAndCache(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim,
                                uint32_t ropeDim, uint32_t offset, uint32_t kdim, uint32_t vdim,
                                XTensor &inputWithR, XTensor &freqs, XTensor &position,
                                uint32_t blockSize, XTensor &key, XTensor &kCache, XTensor &vCache,
                                XTensor &slotMapping)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    KERNEL_PTR_TYPE(rope_complex_and_cache) * launchKernel;
    if (inputWithR.dtype == FP16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_float16_t;
    } else if (inputWithR.dtype == BF16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(inputWithR);
        throw std::runtime_error(err_str + " TODO");
    }
    launchKernel(rt.aivNum, rt.stream, inputWithR.shape[0], nLocalHeads, stepDim, ropeDim, offset,
                 kdim, vdim, inputWithR.ptr, freqs.ptr, position.ptr, blockSize, key.ptr,
                 kCache.ptr, vCache.ptr, slotMapping.ptr);
}

void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(add_bias) * launchKernel;
    if (EachXDtype(FP32, input, weight, output)) {
        launchKernel = aclrtlaunch_add_bias_float;
    } else if (EachXDtype(FP16, input, weight, output)) {
        launchKernel = aclrtlaunch_add_bias_float16_t;
    } else if (EachXDtype(BF16, input, weight, output)) {
        launchKernel = aclrtlaunch_add_bias_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(input) + XT_STR(weight) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr,
                 output.shape[0] * output.shape[1], output.shape[1]);
}

void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outWeights,
                        XTensor &outRouting, uint32_t topK, bool normTopKProb)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(softmax_topk) * launchKernel;
    if (scores.dtype == FP32 && indices.dtype == INT32 && outWeights.dtype == FP32 &&
        outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_softmax_topk_float;
    } else if (scores.dtype == BF16 && indices.dtype == INT32 && outWeights.dtype == BF16 &&
               outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_softmax_topk_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(scores) + XT_STR(indices) + XT_STR(outWeights) + XT_STR(outRouting);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, scores.ptr, indices.ptr, outWeights.ptr, outRouting.ptr,
                 scores.shape[0], indices.shape[0], topK, normTopKProb);
}

void XliteOpSigmoidTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias, float scale,
                        XTensor &outWeights, XTensor &outRouting, uint32_t nGroup,
                        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(sigmoid_topk) * launchKernel;
    if (scores.dtype == FP32 && indices.dtype == INT32 && outWeights.dtype == FP32 &&
        outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_sigmoid_topk_float;
    } else if (scores.dtype == BF16 && indices.dtype == INT32 && outWeights.dtype == BF16 &&
               outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_sigmoid_topk_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(scores) + XT_STR(indices) + XT_STR(outWeights) + XT_STR(outRouting);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, scores.ptr, indices.ptr, bias.ptr, scale, outWeights.ptr,
                 outRouting.ptr, scores.shape[0], indices.shape[0], nGroup, nTopkGroup, topK,
                 normTopKProb);
}

void XliteOpTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outIndices,
                 XTensor &queryLens, XTensor &cachedLens, size_t k)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    uint32_t batchNum = queryLens.shape[0];
    uint32_t maxSeqLen = scores.shape[1];
    if (maxSeqLen <= k) {
        return;
    }

    if (k != 2048) {
        throw std::runtime_error(std::string(__func__) + ": only K=2048 is supported, got " +
                                 std::to_string(k));
    }

    KERNEL_PTR_TYPE(topk) * launchKernel;
    if (scores.dtype == BF16 && indices.dtype == INT32) {
        launchKernel = aclrtlaunch_topk_bfloat16_t;
    } else if (scores.dtype == FP32 && indices.dtype == INT32) {
        launchKernel = aclrtlaunch_topk_float;
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!" +
                                 std::to_string(scores.dtype));
    }
    launchKernel(rt.aivNum, rt.stream, scores.ptr, indices.ptr, outIndices.ptr, queryLens.ptr,
                 cachedLens.ptr, maxSeqLen, batchNum, k);
}

void XliteOpSoftmax(XRuntime &rt, uint32_t calcLen, XTensor &x)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(softmax) * launchKernel;
    if (x.dtype == FP16) {
        launchKernel = aclrtlaunch_softmax_float16_t;
    } else if (x.dtype == BF16) {
        launchKernel = aclrtlaunch_softmax_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(1, rt.stream, x.ptr, x.shape[0], x.shape[1], calcLen);
}

void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x, XTensor &expBuf)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(softmax_long) * launchKernel;
    if (x.dtype == FP16) {
        launchKernel = aclrtlaunch_softmax_long_float16_t;
    } else if (x.dtype == BF16) {
        launchKernel = aclrtlaunch_softmax_long_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(1, rt.stream, x.ptr, expBuf.ptr, x.shape[0], x.shape[1], calcLen);
}

// out = int8(x / scale + offset), turn scale to 1/scale before calculation
void XliteOpQuant(XRuntime &rt, XTensor &x, XTensor &scale_reciprocal, XTensor &offset,
                  XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (x.ptr == nullptr || scale_reciprocal.ptr == nullptr || offset.ptr == nullptr ||
        out.ptr == nullptr) {
        std::string err_str =
            DBG_PREFIX + XT_STR(x) + XT_STR(scale_reciprocal) + XT_STR(offset) + XT_STR(out);
        throw std::runtime_error(err_str + " null pointer!");
    }
    size_t m = x.shape[0];
    size_t n = x.shape[1];
    if (x.dtype == BF16) {
        aclrtlaunch_quant_bf16_to_i8_static(rt.aivNum, rt.stream, x.ptr, scale_reciprocal.ptr,
                                            offset.ptr, out.ptr, m, n);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpQuantDyn(XRuntime &rt, XTensor &x, XTensor &scale, XTensor &out, const XTensor &num)
{
    if (IsDummyRuntime(rt) || x.numel == 0) {
        return;
    }
    if (x.ptr == nullptr || scale.ptr == nullptr || out.ptr == nullptr) {
        std::string err_str = DBG_PREFIX + XT_STR(x) + XT_STR(scale) + XT_STR(out);
        throw std::runtime_error(err_str + " null pointer!");
    }
    size_t m = x.shape[0];
    size_t n = x.shape[1];
    if (x.dtype == BF16) {
        aclrtlaunch_quant_bf16_to_i8_dynamic(rt.aivNum, rt.stream, x.ptr, scale.ptr, out.ptr,
                                             num.ptr, m, n);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpDeQuant(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &scale,
                    const XTensor &num)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    size_t m = in.shape[0];
    size_t n = in.numel / in.shape[0];
    if (in.dtype == FP16) {
        aclrtlaunch_dequant_float16_t(rt.aivNum, rt.stream, in.ptr, scale.ptr, out.ptr, num.ptr, m,
                                      n);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out,
                          const XTensor &quantBias, const XTensor &weightScale, bool weightNZ,
                          bool transpose, const XTensor &outScale, const XTensor &num)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    if (in.dtype == INT8 && weight.dtype == INT8 && out.dtype == BF16) {
        out.View(FP16);
        XliteOpMatmul(rt, in, weight, out, weightNZ, quantBias, weightScale, transpose);
        XliteOpDeQuant(rt, out, out, outScale, num);
        out.View(BF16);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(weight) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpGroupMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                               XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                               long outDim, long inDim, XTensor &output, XTensor &outScale,
                               XTensor &num, bool weightNZ, bool transpose)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    if (in.dtype == INT8 && weightDtype == INT8 && output.dtype == BF16) {
        output.View(FP16);
        XliteOpGroupMatmul(rt, in, weights, deqScales, counts, start, end, weightDtype, outDim,
                           inDim, output, weightNZ, transpose);
        XliteOpDeQuant(rt, output, output, outScale, num);
        output.View(BF16);
    } else {
        std::string err_str = DBG_PREFIX;
        err_str += XT_STR(in) + XT_STR(output) + "weight dtype:" + XDtypeStr(weightDtype);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpConcat(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    size_t offset = 0;
    for (const auto &tensor : inputs) {
        size_t bytes = tensor.numel * XDtypeBit(tensor.dtype) / 8;
        CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(out.ptr) + offset, bytes, tensor.ptr,
                                   bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        offset += bytes;
    }
}
void XliteOpConcatCol(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    size_t times =
        std::accumulate(inputs[0].shape.begin(), inputs[0].shape.end() - 1, 1, std::multiplies());

    size_t totalLastDim = 0;
    for (const auto &tensor : inputs) {
        totalLastDim += tensor.shape.back();
    }
    size_t elemSize = XDtypeBit(inputs[0].dtype) / 8;
    size_t outRowStride = totalLastDim * elemSize;
    for (size_t h = 0; h < times; ++h) {
        size_t dstOffset = h * outRowStride;
        for (const auto &tensor : inputs) {
            size_t rowBytes = tensor.shape.back() * elemSize;
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(out.ptr) + dstOffset, rowBytes,
                                       static_cast<uint8_t *>(tensor.ptr) + h * rowBytes, rowBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            dstOffset += rowBytes;
        }
    }
}

void XliteOpSplitCol(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    size_t height = std::accumulate(in.shape.begin(), in.shape.end() - 1, 1, std::multiplies());
    size_t elemSize = XDtypeBit(in.dtype) / 8;
    for (size_t h = 0; h < height; ++h) {
        size_t srcOffset = h * in.shape.back() * elemSize;
        for (auto cur : outputs) {
            size_t rowBytes = cur.shape.back() * elemSize;
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(cur.ptr) + h * rowBytes, rowBytes,
                                       static_cast<uint8_t *>(in.ptr) + srcOffset, rowBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            srcOffset += rowBytes;
        }
    }
}

void XliteOpSplit(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs,
                  const std::vector<size_t> &sizes, uint32_t numPackets)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    // 计算总大小
    size_t totalSize = 0;
    for (size_t size : sizes) {
        totalSize += size;
    }

    // 从输入缓冲区解开为多个tensor
    for (uint32_t i = 0; i < numPackets; i++) {
        uint8_t *srcBase = static_cast<uint8_t *>(in.ptr) + i * totalSize;
        size_t srcOffset = 0;

        for (size_t j = 0; j < outputs.size(); j++) {
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(outputs[j].ptr) + i * sizes[j],
                                       sizes[j], srcBase + srcOffset, sizes[j],
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            srcOffset += sizes[j];
        }
    }
}

void XliteOpIndexerScores(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight,
                          XTensor &scores, XTensor &queryStartLoc, XTensor &lens,
                          XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads,
                          uint32_t headDim, uint32_t blockSize, uint32_t batch,
                          uint32_t maxNumBlock)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(indexer_scores) * launchKernel;
    if (EachXDtype(FP16, q, kCache, weight, scores)) {
        launchKernel = aclrtlaunch_indexer_scores_float16_t;
    } else if (EachXDtype(BF16, q, kCache, weight, scores)) {
        launchKernel = aclrtlaunch_indexer_scores_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(q) + XT_STR(kCache) + XT_STR(weight) + XT_STR(scores);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, q.ptr, kCache.ptr, weight.ptr, scores.ptr, queryStartLoc.ptr,
                 lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, headDim, blockSize, batch,
                 maxNumBlock);
}

void XliteOpMuls(XRuntime &rt, XTensor &input, float scale, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(muls) * launchKernel;
    if (EachXDtype(FP16, input, output)) {
        launchKernel = aclrtlaunch_muls_float16_t;
    } else if (EachXDtype(BF16, input, output)) {
        launchKernel = aclrtlaunch_muls_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(input) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, input.ptr, scale, output.ptr, input.numel);
}

void XliteOpExpertsCountsSum(XRuntime &rt, XTensor &expertsCountsInput, XTensor &tokensPerEpgroup,
                             XTensor &expertsCountsOutput, uint32_t nRoutedExperts)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    aclrtlaunch_experts_counts_sum(rt.aivNum, rt.stream, expertsCountsInput.ptr,
                                   tokensPerEpgroup.ptr, expertsCountsOutput.ptr, nRoutedExperts,
                                   tokensPerEpgroup.shape[0]);
}

void XliteOpReorderMoE(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &counts,
                       uint32_t hiddenSize, uint32_t localStart, uint32_t localEnd, bool forward)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    if (in.numel == 0 || localStart >= localEnd) {
        return;
    }

    uint32_t moeEpSize = counts.shape[0];
    uint32_t nRoutedExperts = counts.shape[1];
    uint32_t elemBytes = XDtypeBit(in.dtype) / 8;

    aclrtlaunch_reorder_moe(rt.aivNum, rt.stream, in.ptr, out.ptr, counts.ptr, moeEpSize,
                            nRoutedExperts, hiddenSize, localStart, localEnd, forward ? 1 : 0,
                            elemBytes);
}
void XliteOpTranspose_1_2(XRuntime &rt, XTensor &input, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(transpose_1_2) * launchKernel;
    if (EachXDtype(FP32, input, output)) {
        launchKernel = aclrtlaunch_transpose_1_2_float;
    } else if (EachXDtype(FP16, input, output)) {
        launchKernel = aclrtlaunch_transpose_1_2_float16_t;
    } else if (EachXDtype(BF16, input, output)) {
        launchKernel = aclrtlaunch_transpose_1_2_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(input) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, input.ptr, output.ptr, input.shape[0], input.shape[1],
                 input.shape[2]);
}

void XliteOpConv1dAndSiLU(XRuntime &rt, XTensor &x, XTensor &weight, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (x.shape[2] <= weight.shape[2]) {
        throw std::runtime_error("Last dimension of input is too small!");
    }
    KERNEL_PTR_TYPE(conv1d_and_silu) * launchKernel;
    if (EachXDtype(FP32, x, weight, output)) {
        launchKernel = aclrtlaunch_conv1d_and_silu_float;
    } else if (EachXDtype(FP16, x, weight, output)) {
        launchKernel = aclrtlaunch_conv1d_and_silu_float16_t;
    } else if (EachXDtype(BF16, x, weight, output)) {
        launchKernel = aclrtlaunch_conv1d_and_silu_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x) + XT_STR(weight) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, x.ptr, weight.ptr, output.ptr, x.shape[0], x.shape[1],
                 x.shape[2] - weight.shape[2], weight.shape[2]);
}

void XliteOpBetaDecay(XRuntime &rt, XTensor &b, XTensor &a, XTensor &A_log, XTensor &dt_bias,
                      XTensor &beta, XTensor &g, uint32_t bsz, uint32_t seqlen,
                      uint32_t num_v_heads)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(beta_decay) * launchKernel;
    if (EachXDtype(FP32, b, a)) {
        launchKernel = aclrtlaunch_beta_decay_float;
    } else if (EachXDtype(FP16, b, a)) {
        launchKernel = aclrtlaunch_beta_decay_float16_t;
    } else if (EachXDtype(BF16, b, a)) {
        launchKernel = aclrtlaunch_beta_decay_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(b) + XT_STR(a);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, b.ptr, a.ptr, A_log.ptr, dt_bias.ptr, beta.ptr, g.ptr, bsz,
                 seqlen, num_v_heads);
}
