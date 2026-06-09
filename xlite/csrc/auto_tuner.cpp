/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "auto_tuner.h"
#include "kernels/kernel_param.h"

#define MIN_KV_TILE_SIZE 4096

// same logic with flash attention/mla kernels
static uint32_t CalcTaskNum(uint32_t cachedLen, uint32_t queryLen, uint32_t headNumInGroup,
                            uint32_t nKVHeads, uint32_t tileSize)
{
    uint32_t queryTileSize = XLITE_ATTENTION_MAX_M0 / headNumInGroup;
    if (queryTileSize == 0) {
        queryTileSize = XLITE_ATTENTION_MAX_M0;
    }

    uint32_t queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
    uint32_t totalLen = cachedLen + queryLen;
    uint32_t kvNum = DIV_ROUND_UP(totalLen, tileSize);

    uint32_t taskNum = queryNum * nKVHeads * kvNum;
    uint32_t validTaskNum = 0;

    for (uint32_t idx = 0; idx < taskNum; idx++) {
        uint32_t kvIdx = idx % kvNum;
        uint32_t queryIdx = idx / (kvNum * nKVHeads);

        uint32_t queryTaskStart = queryIdx * queryTileSize;
        uint32_t queryTaskLen = queryTileSize;
        if (queryTaskStart + queryTaskLen > queryLen) {
            queryTaskLen = queryLen - queryTaskStart;
        }

        uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
        uint32_t kvOffset = kvIdx * tileSize;

        if (calcLen <= kvOffset) {
            continue;
        }
        validTaskNum++;
    }
    return validTaskNum;
}

static float CalcTotalTaskNumRemainderScore(std::vector<uint32_t> &cachedLens,
                                            std::vector<uint32_t> &queryLens,
                                            uint32_t headNumInGroup, uint32_t nKVHeads,
                                            uint32_t aicNum, uint32_t tileSize)
{
    uint32_t totalTaskNum = 0;
    for (size_t i = 0; i < cachedLens.size(); i++) {
        totalTaskNum +=
            CalcTaskNum(cachedLens[i], queryLens[i], headNumInGroup, nKVHeads, tileSize);
    }

    uint32_t remainder = totalTaskNum % aicNum;
    if (remainder == 0) {
        return 1.0f;
    }
    return static_cast<float>(remainder) / static_cast<float>(aicNum);
}

static float CalcKvRemainderScore(std::vector<uint32_t> &cachedLens,
                                  std::vector<uint32_t> &queryLens, uint32_t tileSize)
{
    float totalScore = 0.0f;
    for (size_t i = 0; i < cachedLens.size(); i++) {
        uint32_t kvLen = cachedLens[i] + queryLens[i];
        uint32_t remainder = kvLen % tileSize;
        float score;
        if (remainder == 0) {
            score = 1.0f;
        } else {
            score = static_cast<float>(remainder) / static_cast<float>(tileSize);
        }
        totalScore += score;
    }
    return cachedLens.empty() ? 0.0f : totalScore / static_cast<float>(cachedLens.size());
}

static float CalcTileScore(std::vector<uint32_t> &cachedLens, std::vector<uint32_t> &queryLens,
                           uint32_t headNumInGroup, uint32_t nKVHeads, uint32_t aicNum,
                           uint32_t tileSize)
{
    // 仅用于全decode场景，评分权重固定
    float kvScore = CalcKvRemainderScore(cachedLens, queryLens, tileSize);
    float taskScore = CalcTotalTaskNumRemainderScore(cachedLens, queryLens, headNumInGroup,
                                                     nKVHeads, aicNum, tileSize);
    float tileSizeScore = static_cast<float>(tileSize) / static_cast<float>(MAX_KV_TILE_SIZE);

    // decode场景：KvScore权重0.7，taskScore权重0.2，TileSizeScore权重0.1
    return kvScore * 0.7f + taskScore * 0.2f + tileSizeScore * 0.1f;
}

uint32_t GetTileSizeOfCachedKV(std::vector<uint32_t> &cachedLens, std::vector<uint32_t> &queryLens,
                               uint32_t headNumInGroup, uint32_t nKVHeads, uint32_t blockSize,
                               uint32_t aicNum)
{
    bool isDecode = true;
    for (size_t i = 0; i < queryLens.size(); i++) {
        if (queryLens[i] != 1) {
            isDecode = false;
            break;
        }
    }

    // TODO prefill case
    if (!isDecode) {
        if (aicNum == 20) {
            return 8192;
        } else if (aicNum == 24) {
            return 6016;
        }
        return 8192;
    }

    float maxScore = -1.0f;
    uint32_t bestTileSize = MAX_KV_TILE_SIZE;

    for (uint32_t tileSize = MAX_KV_TILE_SIZE; tileSize >= MIN_KV_TILE_SIZE;
         tileSize -= blockSize) {
        float score =
            CalcTileScore(cachedLens, queryLens, headNumInGroup, nKVHeads, aicNum, tileSize);
        if (score > maxScore) {
            maxScore = score;
            bestTileSize = tileSize;
        }
    }
    return bestTileSize;
}