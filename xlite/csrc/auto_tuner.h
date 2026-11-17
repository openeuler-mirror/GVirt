/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef XLITE_AUTO_TUNER_H
#define XLITE_AUTO_TUNER_H

#include <vector>
#include <cstdint>
#include <cmath>

#define SINGLE_BATCH_MIN_KV_TILE_SIZE 1024
#define MIN_KV_TILE_SIZE 4096
#define MAX_KV_TILE_SIZE 8192

// Sentinel for "use the auto tiling policy"; m0/n0/k0 passed as this value
// (or left 0) make PickMatmulTiling derive the tiling itself.
#define MATMUL_M0_N0_K0_DEFAULT_VALUE ((uint64_t)(-1))

uint32_t GetTileSizeOfCachedKV(std::vector<uint32_t> &cachedLens, std::vector<uint32_t> &queryLens,
                               uint32_t headNumInGroup, uint32_t nKVHeads, uint32_t blockSize,
                               uint32_t aicNum);

// Pick the indexer_topk query-tile length, tiered by the largest sparse
// prefill qlen (decode-only batches are tile-invariant): <= 2048 -> 22,
// <= 4096 -> 17, else MAX_INDEXER_Q_TILE_LEN. Measured optima; see the
// implementation for the measurement basis. Called once per forward from
// XRuntime::PrepareAttn on the host lens.
uint32_t PickIndexerQTile(const std::vector<uint32_t> &cachedLens,
                          const std::vector<uint32_t> &queryLens, uint32_t topK,
                          uint32_t compressRatio);

// Pick the indexer_topk partitioning mode: kvSplit=0 schedules one (batch,
// query tile) task per block (serial KV-chunk walk), so it fills the blocks
// only when the sparse tiles (last-token causal boundary p0End >= topK, with
// p0End in token space for compressRatio<=1 and completed-block space
// otherwise) reach blockNum - counting chunks instead would wrongly keep
// few-tile/many-chunk decode in the serial mode. Otherwise return 1: (tile,
// chunk) tasks with the cross-core topK relay. Called once per forward from
// XRuntime::PrepareAttn on the host lens (no D2H); queryTileSize should match
// the kernel's tile (PickIndexerQTile) to count the real task granularity -
// 0 or values above MAX_INDEXER_Q_TILE_LEN fall back to the macro default
// (mirroring the kernel's Init clamp).
uint8_t PickIndexerKvSplit(const std::vector<uint32_t> &cachedLens,
                           const std::vector<uint32_t> &queryLens, uint32_t blockNum, uint32_t topK,
                           uint32_t compressRatio, uint32_t queryTileSize);

// Pick the AIC matmul tiling (m0, n0, k0) and the number of AIC blocks to
// launch, shared by XliteOpMatmul and the fused matmul+dequant pipeline so
// both paths always use the same tiling policy. Pass m0/n0/k0 as
// MATMUL_M0_N0_K0_DEFAULT_VALUE (or leave them 0) to use the auto policy.
// aicNum is the hardware AIC count (e.g. rt.aicNum); launchAicNum receives the
// actual number of AIC blocks to launch for this matmul.
void PickMatmulTiling(uint32_t aicNum, uint64_t m, uint64_t n, uint64_t k, uint64_t weightDtypeBits,
                      bool hasBias, bool hasDeqScale, uint64_t &m0, uint64_t &n0, uint64_t &k0,
                      uint32_t &launchAicNum);

// Since the per-core overhead grows with the number of cores used, for operators whose overall
// execution time is on the order of microseconds and whose single-core computation time is
// relatively small, performance can be improved by reducing the number of launched cores and
// increasing the per-core workload.
uint32_t PickMinBlockNum(uint32_t coreNum, uint64_t totalRows);
#endif
