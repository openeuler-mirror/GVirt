/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include "kernel_macro.h"
#include "kernel_operator.h"
#include "matmul.h"
#include "dequant.h"
#include "quant.h"
#include "quant_dyn.h"

using namespace AscendC;

// Cross-core flag ids live in a dedicated 16-entry space (independent of the
// intra-core set_flag/wait_flag ids used by matmul/dequant/quant).
#define C2V_CROSS_CORE_FLAG 8  // AIC tile done -> paired AIVs run one dequant tile (per tile)
#define V2C_CROSS_CORE_FLAG 9  // AIV quant done -> paired AIC starts matmul (once per AIV)
#define FUSION_MATMUL_DQ_LINEAR_SWIZZLE 0ULL

// Fused bf16-activation quant -> int8 matmul (+fixpipe deq_scale) -> dequant,
// the three-kernel W8A8 dense pipeline in one launch:
//
//   AIV t (all): quant rows [t*chunk, (t+1)*chunk) -> GM xQuant
//                [+ per-token scales -> GM outScale, dyn mode]
//                -> SyncAll<true> -> V2C flag 9 (rides PIPE_MTE3)
//   AIC b:       wait flag 9 x2 -> grid-stride matmul tiles -> out (fp16 view)
//                -> C2V flag 8 per tile (rides PIPE_FIX, after CopyToGm)
//   AIV 2b/2b+1: dequant(out fp16 -> bf16 in place, per-row scale in dyn mode)
//                -> V2C_DONE flag 10
//
// The quant hand-off is a single coarse barrier: SyncAll<true> (AIV-only soft
// barrier) retires every AIV's quant stores, then each AIV notifies its
// group's AIC; the ffts set rides PIPE_MTE3 behind the stores, the only
// cross-core release with data-visibility guarantees on this platform (a
// fine-grained per-m-tile variant is not possible here — see
// doc/kernels/fusion_gm_atomic_sync_experiments.md). Both sides stay
// balanced: quant is split evenly across ALL AIVs (contiguous chunk each)
// and matmul keeps the standalone grid-stride.
//
// m0 must be even (PickMatmulTiling rounds it to a multiple of 32).
template <typename Dtype, typename MatDtype, typename OutDtype, typename XDtype>
class FusedQuantMatmulDequant
{
public:
    __aicore__ inline void Init(uint64_t m0, uint64_t n0, uint64_t k0, bool hasBias,
                                bool hasDeqScale, uint64_t transpose, uint64_t nz)
    {
#if defined(__DAV_C220_CUBE__)
        this->matmulOp.Init(m0, n0, k0, hasBias, hasDeqScale, transpose, nz,
                            FUSION_MATMUL_DQ_LINEAR_SWIZZLE);
#elif defined(__DAV_C220_VEC__)
        // AIC:AIV = 1:2, so the dequant row-halves are m0 / GetTaskRatio()
        // Both quant variants are initialized; the active one is selected in
        // Run based on whether quantScaleRec is null (dyn) or not (static).
        this->quantDynOp.Init();
        this->quantOp.Init();
        this->dequantOp.Init(m0 / GetTaskRatio() == 0 ? 1 : m0 / GetTaskRatio(), n0);
#endif
    }

    __aicore__ inline void Run(GM_ADDR x, GM_ADDR quantScaleRec, GM_ADDR quantOffset,
                               GM_ADDR weight, GM_ADDR xQuant, GM_ADDR out, GM_ADDR bias,
                               GM_ADDR deqScale, GM_ADDR outScale, GM_ADDR num, uint64_t m,
                               uint64_t n, uint64_t k, uint64_t m0, uint64_t n0)
    {
#if defined(__DAV_C220_CUBE__)
        // Coarse V2C barrier: the paired AIVs each set flag 9 once after their
        // quant stage, so waiting twice means every row this core will read
        // has been quantized with its stores visible.
        CrossCoreWaitFlag(V2C_CROSS_CORE_FLAG);
        CrossCoreWaitFlag(V2C_CROSS_CORE_FLAG);

        this->matmulOp.TaskTilesInit(xQuant, weight, out, bias, deqScale, m, n, k);
        this->matmulOp.SetFlags();
        int64_t tiles = DIV_ROUND_UP(m, m0) * DIV_ROUND_UP(n, n0);
        for (int64_t idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum()) {
            this->matmulOp.RunTileByIdx(idx);
            // Rides the FIX pipe queue: fires only after this tile's CopyToGm
            // retired, so the waking AIVs see landed GM data.
            CrossCoreSetFlag<0x2, PIPE_FIX>(C2V_CROSS_CORE_FLAG);
        }
        this->matmulOp.WaitFlags();
#elif defined(__DAV_C220_VEC__)
        // AIV block id = get_block_idx()*2 + subblockid: t in [0, 2N), paired
        // AIC = t/2, sub selects which row-half of each m-tile this core
        // dequantizes.
        int64_t t = GetBlockIdx();
        bool isDyn = (quantScaleRec == nullptr);

        // ---- stage 1: quantize this AIV's contiguous chunk of rows — even
        // split over all AIVs (ceil(m / aiv_count) each), same quant
        // parallelism as the standalone kernel. Both stages run tile t whole
        // (TaskTilesInit derives tileRows from the same block_num *
        // task_ratio denominator, so tile t IS this AIV's chunk): both run
        // the batched engine only (both reject ragged k on the host; the
        // engine uses 2D DMA in 32B units).
        if (isDyn) {
            this->quantDynOp.TaskTilesInit(x, outScale, xQuant, nullptr, m, k, GetTaskRatio());
            this->quantDynOp.SetFlags();
            this->quantDynOp.RunTileByIdx(t);
            this->quantDynOp.WaitFlags();

        } else {
            this->quantOp.TaskTilesInit(x, quantScaleRec, quantOffset, xQuant, m, k,
                                        GetTaskRatio());
            this->quantOp.SetFlags();
            this->quantOp.RunTileByIdx(t);
            this->quantOp.WaitFlags();
        }

        // Aggregate release: SyncAll<true> is an AIV-only soft barrier — every
        // AIV's quant stores have retired when it returns; each AIV then
        // notifies its own group's AIC (the AIC waits for both of its AIVs).
        AscendC::SyncAll<true>();
        CrossCoreSetFlag<0x3, PIPE_MTE3>(V2C_CROSS_CORE_FLAG);

        // ---- stage 2: dequant — same loop structure as
        // fusion_operator_matmul_dequant_pipeline. TaskTilesInit returns the
        // AIV tile count in fused mode; RunTileByIdx decodes idx = 2*aicIdx +
        // sub internally.
        int64_t tiles = this->dequantOp.TaskTilesInit(out, outScale, out, num, m, n, true);
        this->dequantOp.SetFlags();
        for (int64_t tileIdx = t; tileIdx < tiles; tileIdx += GetBlockNum() * GetTaskRatio()) {
            CrossCoreWaitFlag(C2V_CROSS_CORE_FLAG);
            this->dequantOp.RunTileByIdx(tileIdx);
        }
        this->dequantOp.WaitFlags();
#endif
    }

private:
#if defined(__DAV_C220_CUBE__)
    Matmul<Dtype, MatDtype, OutDtype> matmulOp;
#elif defined(__DAV_C220_VEC__)
    Quant<XDtype> quantOp;
    QuantDyn<XDtype> quantDynOp;
    Dequant<half> dequantOp;
#endif
};

#define FUSED_QUANT_MATMUL_DEQUANT_FUNC_DEFINE(dtype)                                              \
    extern "C" __global__ __aicore__ void fused_quant_matmul_dequant_##dtype(                      \
        GM_ADDR x, GM_ADDR quant_scale_rec, GM_ADDR quant_offset, GM_ADDR weight, GM_ADDR x_quant, \
        GM_ADDR out, GM_ADDR bias, GM_ADDR weight_scale, GM_ADDR out_scale, GM_ADDR num,           \
        uint64_t m, uint64_t n, uint64_t k, uint64_t weight_nz, uint64_t transpose_weight,         \
        uint64_t m0, uint64_t n0, uint64_t k0)                                                     \
    {                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);                                         \
        FusedQuantMatmulDequant<dtype, int32_t, half, bfloat16_t> op;                              \
        op.Init(m0, n0, k0, bias != nullptr, weight_scale != nullptr, transpose_weight,            \
                weight_nz);                                                                        \
        op.Run(x, quant_scale_rec, quant_offset, weight, x_quant, out, bias, weight_scale,         \
               out_scale, num, m, n, k, m0, n0);                                                   \
    }
