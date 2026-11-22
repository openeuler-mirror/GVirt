/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "matmul.h"

// mhd = torch.einsum("mht,hdt->mhd", mht, hdt)
template <typename Dtype, typename MatDtype, typename OutDtype>
__aicore__ void einsum_mht_hdt_mhd(GM_ADDR mht, GM_ADDR hdt, GM_ADDR mhd, uint32_t m, uint32_t h,
                                   uint32_t t, uint32_t d, uint64_t m0, uint64_t n0, uint64_t k0,
                                   bool weightNZ, uint64_t swizzle, int T = -1, int D = -1)
{
    Matmul<Dtype, MatDtype, OutDtype> matmul_op;

    if (T == -1) {
        T = t;
    }
    if (D == -1) {
        D = d;
    }
    matmul_op.Init(m0, n0, k0, false, false, 0, weightNZ, swizzle);
    matmul_op.SetFlags();

    int xStride = T * sizeof(Dtype);
    int yStride = d * t * sizeof(Dtype);
    int zStride = D * sizeof(Dtype);
    int srcDStride = h * T;
    int dstDStride = h * D;

    // All heads share the GEMM geometry, so TaskTilesInit runs once (head 0)
    // and only the GM bases move per head via RebindGm.  With mLoop >= 2 and
    // the whole K range fitting B's two L1 ping-pong slots (k = t <= 8 * k0,
    // see Matmul::BWholeKResident), the tiles are ordered group-first (head
    // -> nidx -> midx inner) and split into contiguous per-core chunks: each
    // core then touches only ~2-3 heads instead of every head, so the B
    // reloads that the residency tracker cannot dedup across (head, core)
    // pairs shrink by ~blockNum.  A larger k recycles the B slots under the
    // k ping-pong and the reuse decays to the last two windows, so it stays
    // on the legacy round-robin path where chunking has no reload to save.
    // A group cut by a chunk boundary is loaded by both cores (Matmul flags
    // and trackers are per-core state).  mLoop == 1 (decode) keeps the legacy
    // head-staggered split, with RebindGm as the per-head scalar hoist.
    int64_t tiles =
        matmul_op.TaskTilesInit(mht, hdt, mhd, nullptr, nullptr, m, d, t, srcDStride, dstDStride);
    if (matmul_op.MLoop() >= 2 && matmul_op.BWholeKResident(t)) {
        int mLoop = matmul_op.MLoop();
        int nLoop = matmul_op.NLoop();
        int blockNum = GetBlockNum();
        int64_t totalTiles = tiles * h;
        int64_t chunk = DIV_ROUND_UP(totalTiles, blockNum);
        int64_t start = static_cast<int64_t>(GetBlockIdx()) * chunk;
        int64_t end = start + chunk < totalTiles ? start + chunk : totalTiles;
        if (start < end) {
            // idx = (hIdx * nLoop + nidx) * mLoop + midx: three divisions at
            // range entry, then the indices advance by increment-and-wrap.
            int64_t group = start / mLoop;
            int64_t midx = start - group * mLoop;
            int64_t hIdx = group / nLoop;
            int64_t nidx = group - hIdx * nLoop;
            int64_t boundHIdx = 0;  // TaskTilesInit bound head 0
            for (int64_t idx = start; idx < end; idx++) {
                if (hIdx != boundHIdx) {
                    matmul_op.RebindGm(mht + hIdx * xStride, hdt + hIdx * yStride,
                                       mhd + hIdx * zStride);
                    boundHIdx = hIdx;
                }
                matmul_op.RunTileByMN(midx, nidx);
                if (++midx == mLoop) {
                    midx = 0;
                    if (++nidx == nLoop) {
                        nidx = 0;
                        hIdx++;
                    }
                }
            }
        }
    } else {
        int blockNum = GetBlockNum();
        int blockIdx = GetBlockIdx();
        uint32_t coreOffset = 0;
        for (int hIdx = 0; hIdx < (int)h; hIdx++) {
            if (hIdx != 0) {
                matmul_op.RebindGm(mht + hIdx * xStride, hdt + hIdx * yStride,
                                   mhd + hIdx * zStride);
            }
            int64_t first = (blockIdx + blockNum - coreOffset) % blockNum;
            for (int64_t idx = first; idx < tiles; idx += blockNum) {
                matmul_op.RunTileByIdx(idx);
            }
            coreOffset = (coreOffset + tiles) % blockNum;
        }
    }
    matmul_op.WaitFlags();
}

#define EINSUM_MHT_HDT_MHD_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void einsum_mht_hdt_mhd_##dtype(                            \
        GM_ADDR mht, GM_ADDR hdt, GM_ADDR mhd, uint32_t m, uint32_t h, uint32_t t, uint32_t d,   \
        uint64_t m0, uint64_t n0, uint64_t k0, bool weightNZ, uint64_t swizzle, int T, int D)    \
    {                                                                                            \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                          \
        einsum_mht_hdt_mhd<dtype, float, dtype>(mht, hdt, mhd, m, h, t, d, m0, n0, k0, weightNZ, \
                                                swizzle, T, D);                                  \
    }
