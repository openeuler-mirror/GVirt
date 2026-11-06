/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include "kernel_macro.h"
#include "kernel_operator.h"
#include "group_tile_walker.h"
#include "group_matmul.h"
#include "dequant.h"

using namespace AscendC;

// Tile-walk code of this kernel: 0 = linear (carries the weight-strip L2
// refetch penalty W); nonzero = blocked serpentine on both arms (>>8 = count,
// &0xFF = direction 0 Zn / 1 Nz, odd blocks reversed). Global 0x601 (nz6):
// removes W, neutral on W=0/small-m (max +0.10%), bitwise-safe (pure
// bijection).
#define FUSION_GROUP_MATMUL_DQ_LINEAR_SWIZZLE 0x601ULL

// group_matmul + dequant fused into one MIX launch. The quant stage lives
// OUTSIDE this kernel: in is already INT8 (produced by QuantDyn or by the MoE
// dispatch permutation-quant hoist), so the AICs may start the matmul walk
// immediately — the only in-kernel hand-off left is the per-tile C2V flag the
// AIC raises after each tile's CopyToGm for the paired AIVs to dequant.
template <typename Dtype, typename MatDtype, typename OutDtype>
class FusedGroupMatmulDequant
{
public:
    __aicore__ inline void Init(uint64_t m0, uint64_t n0, uint64_t k0, bool hasDeqScale,
                                uint64_t transpose, uint64_t nz)
    {
        // Both arms must walk the same grid: the CUBE branch resolves the
        // (uint64_t)-1 tiling sentinel and clamps n0<=256 under deqScale, the
        // VEC branch consumes m0/n0 raw — the host must pass resolved values
        // (XliteOpFusedGroupMatmulDeQuant fixes m0=128/n0=256).
        assert(m0 != (uint64_t)-1 && (!hasDeqScale || n0 <= 256));
        this->parentM0 = m0;
        this->aivN0 = n0;
#if defined(__DAV_C220_CUBE__)
        this->groupMatmulOp.Init(m0, n0, k0, hasDeqScale, transpose, nz,
                                 FUSION_GROUP_MATMUL_DQ_LINEAR_SWIZZLE);
#elif defined(__DAV_C220_VEC__)
        // AIC:AIV = 1:2, so the dequant row-halves are m0 / GetTaskRatio().
        this->dequantOp.Init(m0 / GetTaskRatio() == 0 ? 1 : m0 / GetTaskRatio(), n0);
#endif
    }

    __aicore__ inline void Run(GM_ADDR xQuant, GM_ADDR ws, GM_ADDR deqScales, GM_ADDR counts,
                               GM_ADDR out, GM_ADDR outScale, uint32_t n, int64_t kN, int64_t kK,
                               uint32_t startIdx, uint32_t endIdx)
    {
        // GetTaskRatio() is 1 on the AIC and 2 on the AIV, so walkBlockIdx is
        // the owning AIC core index on both arms (AIV t -> t/2).
        GroupTileWalker w;
        w.Begin(counts, n, startIdx, endIdx, GetBlockIdx() / GetTaskRatio(),
                static_cast<int64_t>(this->parentM0), static_cast<int64_t>(this->aivN0), kN,
                std::is_same<Dtype, int4b_t>::value, FUSION_GROUP_MATMUL_DQ_LINEAR_SWIZZLE);
#if defined(__DAV_C220_CUBE__)
        this->groupMatmulOp.SetFlags();
        while (w.NextExpert()) {
            this->groupMatmulOp.BeginExpert(w, xQuant, ws, out, deqScales, kN, kK);
            while (w.NextTile()) {
                this->groupMatmulOp.RunTileByIdx(w.idx);
                SetC2VFlag();
            }
        }
        this->groupMatmulOp.WaitFlags();
#elif defined(__DAV_C220_VEC__)
        // m is unused by the group mirror (rows come from the walker); only
        // the GM bindings and the row stride (n = kN) matter.
        this->dequantOp.TaskTilesInit(out, outScale, out, nullptr, 0, static_cast<uint64_t>(kN),
                                      false);
        this->dequantOp.SetFlags();
        while (w.NextExpert()) {
            while (w.NextTile()) {
                WaitC2VFlag();
                this->dequantOp.RunGroupExpertTile(w);
            }
        }
        this->dequantOp.WaitFlags();
#endif
    }

private:
#if defined(__DAV_C220_CUBE__)
    GroupMatmul<Dtype, MatDtype, OutDtype> groupMatmulOp;
#elif defined(__DAV_C220_VEC__)
    Dequant<OutDtype> dequantOp;
#endif
    uint64_t parentM0 = 0;
    uint64_t aivN0 = 0;
};

#define FUSED_GROUP_MATMUL_DEQUANT_FUNC_DEFINE(dtype)                                              \
    extern "C" __global__ __aicore__ void fused_group_matmul_dequant_##dtype(                      \
        GM_ADDR x_quant, GM_ADDR ws, GM_ADDR deq_scales, GM_ADDR counts, GM_ADDR out,              \
        GM_ADDR out_scale, uint32_t n, int64_t kN, int64_t kK, uint64_t m0, uint64_t n0,           \
        uint64_t k0, uint32_t start_idx, uint32_t end_idx, uint64_t weight_nz, uint64_t transpose) \
    {                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                         \
        FusedGroupMatmulDequant<dtype, int32_t, half> op;                                          \
        op.Init(m0, n0, k0, deq_scales != nullptr, transpose, weight_nz);                          \
        op.Run(x_quant, ws, deq_scales, counts, out, out_scale, n, kN, kK, start_idx, end_idx);    \
    }
