/*
 * Copyright (C) 2025-2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "group_tile_walker.h"
#include "matmul.h"

// The per-tile AIC -> AIV hand-off flag (C2V_CROSS_CORE_FLAG) and the shared
// expert-walk skeleton (GroupTileWalker) live in group_tile_walker.h — the
// AIV mirror of fused shells (Dequant::RunGroupExpertTile) consumes both.

// kM: tokens of one expert
// kN: moe intermediate size
// kK: hidden size
// [start_idx, end_idx): Range of expert index involved this time
// deqScale: uint64_t, 低 32 位 TF32 格式有效, 1 符号位, 8 指数位, 10 尾数位, 后 13 位不参与计算
//
// GroupMatmul walks the experts in [startIdx, endIdx) through the shared
// GroupTileWalker skeleton (group_tile_walker.h), locating each expert's row
// segment of x/z by the per-expert counts (read from GM at runtime) and each
// expert's weight/deqScale by indirect GM pointer arrays. Tile dispatch keeps
// the phase continuous across experts (the walker's coreOffset), so the global
// tile index gidx = coreOffset_e + idx_local enumerates every (expert, m-tile,
// n-tile) exactly once and core `b` owns {gidx ≡ b (mod GetBlockNum())}.
//
// The per-tile C2V raise for fused MIX shells is an explicit SetC2VFlag()
// call the shell makes after each tile (group_tile_walker.h): the fused shell
// drives BeginExpert plus its own tile loop, while the standalone kernel uses
// the built-in loop in RunTiles with no cross-core calls. The AIV mirror
// drains with the matching WaitC2VFlag().
template <typename Dtype, typename MatDtype, typename OutDtype>
class GroupMatmul
{
public:
    __aicore__ inline void Init(uint64_t m0, uint64_t n0, uint64_t k0, bool hasDeqScale,
                                bool transpose, bool nz, uint64_t swizzle)
    {
        // Resolve the MATMUL_M0_N0_K0_DEFAULT_VALUE sentinel and the n0 clamp
        // locally — keep in sync with Matmul::Init's resolution. The walker
        // computes tile counts AHEAD of TaskTilesInit, so members must hold the
        // effective grid, not the raw sentinel; subOp.Init then receives the
        // resolved values explicitly, so Matmul's own resolution never fires
        // for the group path and both sides decode the same grid by
        // construction. (group_matmul passes hasBias=false, so Matmul's hasBias
        // clamp can never fire and is not mirrored here.)
        int dtypeBits = GetDTypeBits(Dtype);
        if (m0 == (uint64_t)-1) {
            m0 = 128;
            n0 = 256;
            k0 = 128 * BYTE_BITS / dtypeBits;
        }
        if (hasDeqScale && n0 > 256) {
            n0 = 256;  // A2/A3: fixpipe buffer 2k (256 * 8B)
        }
        subOp.Init(m0, n0, k0, false, hasDeqScale, transpose, nz, swizzle);
        this->m0 = m0;
        this->n0 = n0;
    }

    __aicore__ inline void SetFlags()
    {
        subOp.SetFlags();
    }

    __aicore__ inline void WaitFlags()
    {
        subOp.WaitFlags();
    }

    // One expert of the walk: bind the sub-op to expert w.i (indirect
    // weight/deqScale pointers, x/z row offsets from w.off). Call once per
    // NextExpert() before this core's tile loop of that expert.
    __aicore__ inline void BeginExpert(GroupTileWalker &w, GM_ADDR x, GM_ADDR ws, GM_ADDR z,
                                       GM_ADDR deqScales, int64_t kN, int64_t kK)
    {
        int dtypeBits = std::is_same<Dtype, int4b_t>::value ? 4 : (sizeof(Dtype) * BYTE_BITS);
        uint64_t weightAddr = *((__gm__ uint64_t *)(ws + w.i * sizeof(void *)));
        __gm__ uint8_t *weight = (__gm__ uint8_t *)weightAddr;
        __gm__ uint8_t *deqScale = nullptr;
        if (deqScales != nullptr) {
            uint64_t deqScaleAddr = *((__gm__ uint64_t *)(deqScales + w.i * sizeof(void *)));
            deqScale = (__gm__ uint8_t *)deqScaleAddr;
        }

        int64_t xOffBytes = w.off * kK * dtypeBits / BYTE_BITS;
        int64_t zOffBytes = w.off * kN * sizeof(OutDtype);

        int64_t tiles = subOp.TaskTilesInit(x + xOffBytes, weight, z + zOffBytes, nullptr, deqScale,
                                            w.kM, kN, kK);
        assert(tiles == w.tiles);  // walker and sub-op must agree on the tile grid
    }

    // Execute one output tile (forwarded so fused shells can drive their own
    // tile loop and bracket each tile with SetC2VFlag()).
    __aicore__ inline void RunTileByIdx(int64_t idx)
    {
        subOp.RunTileByIdx(idx);
    }

    // Full walk: seed the shared skeleton and dispatch every expert's tiles
    // (standalone path; fused shells drive BeginExpert + the tile loop
    // themselves).
    __aicore__ inline void RunTiles(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR deqScales,
                                    GM_ADDR counts, uint32_t n, int64_t kN, int64_t kK,
                                    uint32_t startIdx, uint32_t endIdx)
    {
        GroupTileWalker w;
        w.Begin(counts, n, startIdx, endIdx, GetBlockIdx(), static_cast<int64_t>(this->m0),
                static_cast<int64_t>(this->n0), kN, std::is_same<Dtype, int4b_t>::value,
                /*swizzle=*/0);
        while (w.NextExpert()) {
            BeginExpert(w, x, ws, z, deqScales, kN, kK);
            while (w.NextTile()) {
                subOp.RunTileByIdx(w.idx);
            }
        }
    }

    // Convenience: SetFlags + full walk + WaitFlags (mirrors Matmul::Run).
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR deqScales, GM_ADDR counts,
                               uint32_t n, int64_t kN, int64_t kK, uint32_t startIdx,
                               uint32_t endIdx)
    {
        SetFlags();
        RunTiles(x, ws, z, deqScales, counts, n, kN, kK, startIdx, endIdx);
        WaitFlags();
    }

private:
    Matmul<Dtype, MatDtype, OutDtype> subOp;
    uint64_t m0 = 0;
    uint64_t n0 = 0;
};

template <typename Dtype, typename MatDtype, typename OutDtype>
__aicore__ void group_matmul_kernel(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR deqScales,
                                    GM_ADDR counts, uint32_t n, int64_t kN, int64_t kK, uint64_t m0,
                                    uint64_t n0, uint64_t k0, uint32_t startIdx, uint32_t endIdx,
                                    bool weightNZ, bool transpose, uint64_t swizzle)
{
    GroupMatmul<Dtype, MatDtype, OutDtype> op;

    op.Init(m0, n0, k0, deqScales != nullptr, transpose, weightNZ, swizzle);
    op.Run(x, ws, z, deqScales, counts, n, kN, kK, startIdx, endIdx);
}

#define GROUPMATMUL_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void group_matmul_##dtype(                                    \
        GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR deqScales, GM_ADDR counts, uint32_t n,           \
        int64_t kN, int64_t kK, uint64_t m0, uint64_t n0, uint64_t k0, uint32_t startIdx,          \
        uint32_t endIdx, bool weightNZ, bool transpose, uint64_t swizzle)                          \
    {                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                            \
        if constexpr (std::is_same<dtype, int8_t>::value || std::is_same<dtype, int4b_t>::value) { \
            group_matmul_kernel<dtype, int32_t, half>(x, ws, z, deqScales, counts, n, kN, kK, m0,  \
                                                      n0, k0, startIdx, endIdx, weightNZ,          \
                                                      transpose, swizzle);                         \
        } else {                                                                                   \
            group_matmul_kernel<dtype, float, dtype>(x, ws, z, deqScales, counts, n, kN, kK, m0,   \
                                                     n0, k0, startIdx, endIdx, weightNZ,           \
                                                     transpose, swizzle);                          \
        }                                                                                          \
    }
