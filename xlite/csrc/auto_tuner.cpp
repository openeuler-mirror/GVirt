/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "auto_tuner.h"
#include "kernels/kernel_param.h"

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

static uint32_t CalcTotalTaskNum(std::vector<uint32_t> &cachedLens,
                                 std::vector<uint32_t> &queryLens, uint32_t headNumInGroup,
                                 uint32_t nKVHeads, uint32_t tileSize)
{
    uint32_t totalTaskNum = 0;
    for (size_t i = 0; i < cachedLens.size(); i++) {
        uint32_t kvLen = cachedLens[i] + queryLens[i];
        // ignore vllm pad seq
        if (kvLen == 1) {
            continue;
        }
        totalTaskNum +=
            CalcTaskNum(cachedLens[i], queryLens[i], headNumInGroup, nKVHeads, tileSize);
    }
    return totalTaskNum;
}

static float CalcTotalTaskNumRemainderScore(uint32_t totalTaskNum, uint32_t aicNum)
{
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
    int batch = 0;
    for (size_t i = 0; i < cachedLens.size(); i++) {
        uint32_t kvLen = cachedLens[i] + queryLens[i];
        // ignore vllm pad seq
        if (kvLen == 1) {
            continue;
        }
        uint32_t remainder = kvLen % tileSize;
        float score;
        if (remainder == 0) {
            score = 1.0f;
        } else {
            score = static_cast<float>(remainder) / static_cast<float>(tileSize);
        }
        totalScore += score;
        batch++;
    }
    return batch == 0 ? 0.0f : totalScore / static_cast<float>(batch);
}

static float CalcTileScore(std::vector<uint32_t> &cachedLens, std::vector<uint32_t> &queryLens,
                           uint32_t headNumInGroup, uint32_t nKVHeads, uint32_t aicNum,
                           uint32_t tileSize)
{
    float kvScore = CalcKvRemainderScore(cachedLens, queryLens, tileSize);
    uint32_t totalTaskNum =
        CalcTotalTaskNum(cachedLens, queryLens, headNumInGroup, nKVHeads, tileSize);
    float taskScore = CalcTotalTaskNumRemainderScore(totalTaskNum, aicNum);
    float tileSizeScore = static_cast<float>(tileSize) / static_cast<float>(MAX_KV_TILE_SIZE);

    if (totalTaskNum <= aicNum) {
        // 全是decode请求：kv_score权重0.3，remainder_score权重0.6，tile_size权重0.1
        return kvScore * 0.3f + std::sqrt(taskScore) * 0.6f + tileSizeScore * 0.1f;
    }

    // decode场景：KvScore权重0.7，taskScore权重0.2，TileSizeScore权重0.1
    return kvScore * 0.7f + taskScore * 0.2f + tileSizeScore * 0.1f;
}

uint32_t GetBestTileSizeByFitting(const std::vector<uint32_t> &cachedLens, size_t batch,
                                  uint32_t aicNum, uint32_t blockSize)
{
    // 单 batch 下，最优 tasks 与 cachedLen 开根号为正相关，用拟合的公式计算
    if (cachedLens.size() != batch || batch != 1) {
        throw std::runtime_error(std::string(__func__) + ":" + std::to_string(__LINE__) +
                                 ": invalid batch: " + std::to_string(batch) +
                                 ", cachedLens: " + std::to_string(cachedLens.size()));
    }
    double cachedLen = static_cast<double>(cachedLens[0]);

    /*
     * 拟合单 batch 最优 tasks 数量
     * 拟合方式：单独运行 mla 算子，并用 msprof 采集算子耗时拆解，其中 8~128K 单轮下发均优于多轮任务
     * 算子耗时大致以下内容：
     * 1. scalar 每次执行均会有耗时，假设为固定值
     * 2. cube 计算耗时在单轮内包括 QK、SM、SV 三个阶段，并行后计算、通信量均与 tile 正相关
     * 3. vector 耗时主要在核间等待，假设与 taskNum 正相关
     * 综上，latency = scalar_time + cube_time * kvTile + cachedLen / kvTile * vector_time
     * 根据表达式可知，最优 taskNum 及最优 tile 与 cachedLen 呈正相关
     */
    double bestTasks = std::round(1.5 * std::sqrt(cachedLen / 1024.0) + 3);

    // tasks 尽量小于 aicNum 避免多轮开销
    uint32_t bestTasksInt = static_cast<uint32_t>(bestTasks);
    bestTasksInt = bestTasksInt > aicNum ? aicNum : bestTasksInt;
    bestTasksInt = bestTasksInt < 1 ? 1 : bestTasksInt;

    // 选取小 tile 有利于避免 tile 内的浪费，并保证在界内 blockSize 对齐
    uint32_t cachedLenInt = static_cast<uint32_t>(std::ceil(cachedLen));
    uint32_t tile = DIV_ROUND_UP(cachedLenInt, bestTasksInt);
    tile = ROUND_UP(tile, blockSize);
    tile = tile > MAX_KV_TILE_SIZE ? MAX_KV_TILE_SIZE : tile;
    tile = tile < MIN_KV_TILE_SIZE ? MIN_KV_TILE_SIZE : tile;
    return tile;
}

uint32_t GetBestTileSizeByScores(std::vector<uint32_t> &cachedLens,
                                 std::vector<uint32_t> &queryLens, uint32_t headNumInGroup,
                                 uint32_t nKVHeads, uint32_t blockSize, uint32_t aicNum)
{
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

uint32_t GetTileSizeOfCachedKV(std::vector<uint32_t> &cachedLens, std::vector<uint32_t> &queryLens,
                               uint32_t headNumInGroup, uint32_t nKVHeads, uint32_t blockSize,
                               uint32_t aicNum)
{
    bool isDecode = true;
    uint32_t bestTileSize = MAX_KV_TILE_SIZE;
    size_t batch = queryLens.size();
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

    if (batch == 1) {
        bestTileSize = GetBestTileSizeByFitting(cachedLens, batch, aicNum, blockSize);
    } else {
        bestTileSize = GetBestTileSizeByScores(cachedLens, queryLens, headNumInGroup, nKVHeads,
                                               blockSize, aicNum);
    }
#ifdef XLITE_DEBUG_ON
    std::cout << "cachedLens[" << batch << "]: [";
    for (uint32_t i = 0; i < batch; i++) {
        std::cout << cachedLens[i] << ((i == batch - 1) ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    std::cout << "queryLens[" << batch << "]: [";
    for (uint32_t i = 0; i < batch; i++) {
        std::cout << queryLens[i] << ((i == batch - 1) ? "" : ", ");
    }
    std::cout << "]" << std::endl;
    std::cout << "selected kvTileSize: " << bestTileSize << std::endl;
#endif
    return bestTileSize;
}