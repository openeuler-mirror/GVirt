/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef XLITE_AUTO_TUNER_H
#define XLITE_AUTO_TUNER_H

#include <vector>
#include <cstdint>

#define MAX_KV_TILE_SIZE 8192

uint32_t GetTileSizeOfCachedKV(std::vector<uint32_t> &cachedLens, std::vector<uint32_t> &queryLens,
                               uint32_t headNumInGroup, uint32_t nKVHeads, uint32_t blockSize,
                               uint32_t aicNum);

#endif