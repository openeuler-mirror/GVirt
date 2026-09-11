/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include "kernel_macro.h"
#include "kernel_operator.h"
#include "matmul.h"
#include "dequant.h"

using namespace AscendC;

#define C2V_CROSS_CORE_FLAG 8
#define FUSION_MATMUL_DQ_LINEAR_SWIZZLE (1ULL << 8)

template <typename Dtype, typename MatDtype, typename OutDtype>
class FusionMatmulDequantPipeline
{
public:
    __aicore__ inline void Init(uint64_t m0, uint64_t n0, uint64_t k0, bool hasBias,
                                bool hasDeqScale, uint64_t transpose, uint64_t nz, bool hasOutScale)
    {
#if defined(__DAV_C220_CUBE__)
        this->subOp.Init(m0, n0, k0, hasBias, hasDeqScale, transpose, nz,
                         FUSION_MATMUL_DQ_LINEAR_SWIZZLE);
#elif defined(__DAV_C220_VEC__)
        // AIC:AIV = 1:2, so m0 shrinked to m0 / 2, wish m0 to be even
        this->subOp.Init(hasOutScale, m0 / GetTaskRatio() == 0 ? 1 : m0 / GetTaskRatio(), n0);
#endif
    }

    __aicore__ inline void SetFlags()
    {
        this->subOp.SetFlags();
    }

    __aicore__ inline void WaitFlags()
    {
        this->subOp.WaitFlags();
    }

    // The return value of AIV is double compared to AIC
    __aicore__ inline int64_t TaskTilesInit(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR bias,
                                            GM_ADDR deqScale, GM_ADDR outScale, GM_ADDR num,
                                            uint64_t m, uint64_t n, uint64_t k, int xSrcDValue = -1,
                                            int zDstDValue = -1)
    {
#if defined(__DAV_C220_CUBE__)
        this->tileCount =
            this->subOp.TaskTilesInit(x, y, z, bias, deqScale, m, n, k, xSrcDValue, zDstDValue);
#elif defined(__DAV_C220_VEC__)
        // when tileCount is odd, pad one AIV core
        this->tileCount = this->subOp.TaskTilesInit(z, outScale, z, num, m, n, true);
#endif
        return this->tileCount;
    }

    __aicore__ inline void RunTileByIdx(int64_t tileIdx)
    {
#if defined(__DAV_C220_CUBE__)
        this->subOp.RunTileByIdx(tileIdx);
        // The notification rides the FIX pipe queue (PIPE_FIX), so it only
        // fires after this tile's CopyToGm has retired -> GM data is visible
        // to the paired AIVs when they wake up.
        CrossCoreSetFlag<0x2, PIPE_FIX>(C2V_CROSS_CORE_FLAG);
#elif defined(__DAV_C220_VEC__)
        CrossCoreWaitFlag(C2V_CROSS_CORE_FLAG);
        this->subOp.RunTileByIdx(tileIdx);
#endif
    }

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR bias, GM_ADDR deqScale,
                               GM_ADDR outScale, GM_ADDR num, uint64_t m, uint64_t n, uint64_t k,
                               int xSrcDValue = -1, int zDstDValue = -1)
    {
        SetFlags();
        int64_t tileCount =
            TaskTilesInit(x, y, z, bias, deqScale, outScale, num, m, n, k, xSrcDValue, zDstDValue);
        for (int64_t tileIdx = GetBlockIdx(); tileIdx < tileCount;
             tileIdx += GetBlockNum() * GetTaskRatio()) {
            RunTileByIdx(tileIdx);
        }
        WaitFlags();
    }

private:
    int64_t tileCount = 0;

#if defined(__DAV_C220_CUBE__)
    Matmul<Dtype, MatDtype, OutDtype> subOp;
#elif defined(__DAV_C220_VEC__)
    Dequant<half> subOp;
#endif
};

#define FUSION_OPERATOR_MATMUL_DEQUANT_PIPELINE_FUNC_DEFINE(dtype)                              \
    extern "C" __global__ __aicore__ void fusion_operator_matmul_dequant_pipeline_##dtype(      \
        GM_ADDR x, GM_ADDR weight, GM_ADDR out, GM_ADDR bias, GM_ADDR weightScale,              \
        GM_ADDR outScale, GM_ADDR num, uint64_t m, uint64_t n, uint64_t k, uint64_t weightNz,   \
        uint64_t transposeWeight, uint64_t m0, uint64_t n0, uint64_t k0)                        \
    {                                                                                           \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                      \
        FusionMatmulDequantPipeline<dtype, int32_t, half> op;                                   \
        op.Init(m0, n0, k0, bias != nullptr, weightScale != nullptr, transposeWeight, weightNz, \
                outScale != nullptr);                                                           \
        op.Run(x, weight, out, bias, weightScale, outScale, num, m, n, k);                      \
    }
