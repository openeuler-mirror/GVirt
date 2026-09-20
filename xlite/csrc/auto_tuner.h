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