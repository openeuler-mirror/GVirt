/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <tuple>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"

XModel::XModel(struct XModelConfig &c, uint32_t rankId) : _c(c), _rankId(rankId)
{
    attnNorm.resize(c.nLayers);
    attnOut.resize(c.nLayers);
    mhaQKV.resize(c.nLayers);
    mlaQA.resize(c.nLayers);
    mlaQB.resize(c.nLayers);
    mlaQNorm.resize(c.nLayers);
    mlaKVA.resize(c.nLayers);
    mlaKVB.resize(c.nLayers);
    mlaKVNorm.resize(c.nLayers);
    mlpNorm.resize(c.nLayers);
    mlpUpGate.resize(c.nDenseLayers);
    mlpDown.resize(c.nDenseLayers);
    moeGate.resize(c.nLayers);
    moeGateBias.resize(c.nLayers);
    moeSEUpGate.resize(c.nLayers);
    moeSEDown.resize(c.nLayers);
    moeREUpGate.resize(c.nLayers);
    moeREUpGateScale.resize(c.nLayers);
    moeREDown.resize(c.nLayers);
    moeREDownScale.resize(c.nLayers);
    _moeREUpGate.resize(c.nLayers);
    _moeREUpGateScale.resize(c.nLayers);
    _moeREDown.resize(c.nLayers);
    _moeREDownScale.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        moeREUpGate[i].resize(c.nRoutedExperts);
        moeREUpGateScale[i].resize(c.nRoutedExperts);
        moeREDown[i].resize(c.nRoutedExperts);
        moeREDownScale[i].resize(c.nRoutedExperts);
    }
}

void XModel::Init(void)
{
    std::vector<uint32_t> gateIdx;
    std::vector<uint64_t> weights;
    long size;
    void *ptr;

    if (_c.nDenseLayers != _c.nLayers) {
        gateIdx.resize(_c.nRoutedExperts);
        size = _c.nRoutedExperts * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t i = 0; i < _c.nRoutedExperts; i++) {
            gateIdx[i] = i;
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, gateIdx.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _gateIndicts.Init({_c.nRoutedExperts}, INT32, ptr);
    }

    size = _c.nRoutedExperts * XDtypeBit(INT64) / 8;
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        weights.resize(_c.nRoutedExperts);
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREUpGate[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREUpGate[i].Init({_c.nRoutedExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREUpGateScale[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREUpGateScale[i].Init({_c.nRoutedExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREDown[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREDown[i].Init({_c.nRoutedExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREDownScale[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREDownScale[i].Init({_c.nRoutedExperts}, INT64, ptr);
    }
}

XModel::~XModel(void)
{
    if (_c.nDenseLayers != _c.nLayers) {
        CHECK_ACL(aclrtFree(_gateIndicts.ptr));
    }
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        CHECK_ACL(aclrtFree(_moeREUpGate[i].ptr));
        CHECK_ACL(aclrtFree(_moeREUpGateScale[i].ptr));
        CHECK_ACL(aclrtFree(_moeREDown[i].ptr));
        CHECK_ACL(aclrtFree(_moeREDownScale[i].ptr));
    }
}

void XModel::ForwardParallelEmbed(XRuntime &rt, XTensor &input, XTensor &embed, XTensor &output)
{
    uint32_t vocabPerTp = DIV_ROUND_UP(_c.vocabSize, _c.defTpSize);
    uint32_t id = _rankId % _c.defTpSize;
    uint32_t start = id * vocabPerTp;
    uint32_t end = start + vocabPerTp;

    XliteOpEmbed(rt, input, embed, start, end, output);
    if (_c.defTpSize > 1) {
        XliteOpAllReduceSum(rt, output, output, TP);
    }
}

void XModel::prepareAttn(XRuntime &rt, XModelAttnMeta& attnMeta)
{
}

void XModel::ForwardAttnMLA(XRuntime &rt, uint32_t layer,
                            XModelAttnMeta& attnMeta,
                            std::vector<std::pair<XTensor, XTensor>>& kvCache,
                            XTensor &freqsCis, XTensor &hiddenState)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XModel::ForwardAttnMHA(XRuntime &rt, uint32_t layer,
                            XModelAttnMeta& attnMeta,
                            std::vector<std::pair<XTensor, XTensor>>& kvCache,
                            XTensor &freqsCis, XTensor &hiddenState)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XModel::ForwardAttn(XRuntime &rt, uint32_t layer,
                         XModelAttnMeta& attnMeta,
                         std::vector<std::pair<XTensor, XTensor>>& kvCache,
                         XTensor &freqsCis, XTensor &hiddenState)
{
    if (_aType == XMODEL_ATTN_MLA) {
        ForwardAttnMLA(rt, layer, attnMeta, kvCache, freqsCis, hiddenState);
    } else if (_aType == XMODEL_ATTN_MHA) {
        ForwardAttnMHA(rt, layer, attnMeta, kvCache, freqsCis, hiddenState);
    } else {
        std::cout << __func__ << ": TODO" << std::endl;
    }
}

void XModel::ForwardMLP(XRuntime &rt, XTensor &upGate, XTensor &down, XTensor &hiddenState, bool withAllReduce)
{
    uint32_t m = hiddenState.shape[0];
    XTensor &h13 = rt.pool->GetTensor({m, upGate.shape[0]}, hiddenState.dtype);
    XTensor &h2 = rt.pool->GetTensor({m, down.shape[1]}, hiddenState.dtype);

    XliteOpMatmul(rt, hiddenState, upGate, h13);
    XliteOpSiluAndMul(rt, h13, h2);
    XliteOpMatmul(rt, h2, down, hiddenState);

    if (withAllReduce && _c.defTpSize > 1) {
        XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
    }
    rt.pool->PutTensor(h2);
    rt.pool->PutTensor(h13);
}

std::tuple<XTensor &, XTensor &> XModel::ForwardMoEGate(XRuntime &rt, uint32_t layer, XTensor &input)
{
    uint32_t m = input.shape[0];
    XTensor &weights = rt.pool->GetTensor({m, _c.nRoutedExperts}, moeGate[layer].dtype);
    XTensor &routing = rt.pool->GetTensor({m, _c.nRoutedExperts}, BIT1);
    XTensor &scores = rt.pool->GetTensor({input.shape[0], _c.nRoutedExperts}, moeGate[layer].dtype);

    XliteOpMatmul(rt, input, moeGate[layer], scores);
    XliteOpSigmoidTopK(rt, scores, moeGateBias[layer], _gateIndicts, _c.nExpertGroups, _c.nLimitedGroups,
                       _c.nActExperts, _c.routeScale, weights, routing);

    rt.pool->PutTensor(scores);
    return {weights, routing};
}

std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &> XModel::ForwardMoEDispatch(XRuntime &rt,
    XTensor &tokenSorted, XTensor &weights, XTensor &routing)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;
    XTensor &inputPerDp = tokenSorted, &weightsPerDp = weights, &routingPerDp = routing;

    if (_c.defDpSize > 1) {
        tokenSorted = rt.pool->GetTensor({mAllDp, _c.hiddenSize}, inputPerDp.dtype);
        weights = rt.pool->GetTensor({mAllDp, _c.nRoutedExperts}, weightsPerDp.dtype);
        routing = rt.pool->GetTensor({mAllDp, _c.nRoutedExperts}, routingPerDp.dtype);
        XliteOpAllGather(rt, inputPerDp, tokenSorted, DP);
        XliteOpAllGather(rt, weightsPerDp, weights, DP);
        XliteOpAllGather(rt, routingPerDp, routing, DP);
        rt.pool->PutTensor(weightsPerDp);
        rt.pool->PutTensor(routingPerDp);
    }

    XTensor &unpIdx = rt.pool->GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32);
    XTensor &expertsSorted = rt.pool->GetTensor({mAllDp * _c.nActExperts, _c.hiddenSize}, tokenSorted.dtype);
    XTensor &expertsCounts = rt.pool->GetTensor({_c.nRoutedExperts, 1}, INT32);
    XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx, expertsCounts);

    if (_c.defDpSize > 1) {
        rt.pool->PutTensor(tokenSorted);
    }

    return {weights, routing, unpIdx, expertsSorted, expertsCounts};
}

void XModel::ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing, XTensor &unpIdx,
                               XTensor &expertsSorted, XTensor &expertsCounts)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;
    XTensor &tokenSortedAllDp = tokenSorted;

    if (_c.defDpSize > 1) {
        tokenSortedAllDp = rt.pool->GetTensor({mAllDp, _c.hiddenSize}, tokenSorted.dtype);
    }
    XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSortedAllDp);
    if (_c.defDpSize > 1) {
        XliteOpReduceScatter(rt, tokenSortedAllDp, tokenSorted, DP);
        rt.pool->PutTensor(tokenSortedAllDp);
    }
    rt.pool->PutTensor(expertsCounts);
    rt.pool->PutTensor(expertsSorted);
    rt.pool->PutTensor(unpIdx);
    rt.pool->PutTensor(routing);
    rt.pool->PutTensor(weights);
}

void XModel::ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    uint32_t m = hiddenState.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t intermediateSize = _c.moeIntermediateSize / _c.moeTPSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;

    auto [w, r] = ForwardMoEGate(rt, layer, hiddenState);
    auto [weights, routing, unpIdx, expertsSorted, expertsCounts] = ForwardMoEDispatch(rt, hiddenState, w, r);

    // routed experts
    XTensor &h13 = rt.pool->GetTensor({mAllDp * _c.nActExperts, intermediateSize * 2}, hiddenState.dtype);
    XTensor &h2 = rt.pool->GetTensor({mAllDp * _c.nActExperts, intermediateSize}, hiddenState.dtype);
    XliteOpGroupMatmul(rt, expertsSorted, _moeREUpGate[layer], _moeREUpGateScale[layer], expertsCounts,
        start, end, moeREUpGate[layer][start].dtype, intermediateSize * 2, _c.hiddenSize, h13);
    XliteOpSiluAndMul(rt, h13, h2);
    XliteOpGroupMatmul(rt, h2, _moeREDown[layer], _moeREDownScale[layer], expertsCounts,
        start, end, moeREDown[layer][start].dtype, _c.hiddenSize, intermediateSize, expertsSorted);
    rt.pool->PutTensor(h13);
    rt.pool->PutTensor(h2);

    XTensor &h = rt.pool->GetTensor({m, _c.hiddenSize}, hiddenState.dtype);
    ForwardMOECombine(rt, h, weights, routing, unpIdx, expertsSorted, expertsCounts);

    // share experts
    ForwardMLP(rt, moeSEUpGate[layer], moeSEDown[layer], hiddenState, false);
    XliteOpAdd(rt, hiddenState, h, hiddenState);
    rt.pool->PutTensor(h);

    if (_c.defTpSize > 1) {
        XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP);
    }
}

void XModel::ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    if (layer < _c.nDenseLayers) {
        ForwardMLP(rt, mlpUpGate[layer], mlpDown[layer], hiddenState, true);
    } else {
        ForwardMoE(rt, layer, hiddenState);
    }
}

void XModel::ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &output)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XModel::Forward(XRuntime &rt, XTensor &input,
                     XModelAttnMeta& attnMeta,
                     std::vector<std::pair<XTensor, XTensor>>& kvCache,
                     XTensor &freqsCis, XTensor &output)
{
    uint32_t batch = input.shape[0];
    uint32_t seqLen = input.shape[1];
    uint32_t m = batch * seqLen;

    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        std::cerr << __FILE__ << ":" << __LINE__ << "check runtime communication setting failed" << std::endl;
        return;
    }

    XTensor &x = rt.pool->GetTensor({m, _c.hiddenSize}, embed.dtype);
    XTensor &h = rt.pool->GetTensor({m, _c.hiddenSize}, embed.dtype);

    ForwardParallelEmbed(rt, input, embed, x);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        XliteOpRmsNorm(rt, x, attnNorm[i], _c.normEps, h);
        ForwardAttn(rt, i, attnMeta, kvCache, freqsCis, h);
        XliteOpAdd(rt, x, h, x);
        XliteOpRmsNorm(rt, x, mlpNorm[i], _c.normEps, h);
        ForwardFFN(rt, i, h);
        XliteOpAdd(rt, x, h, x);
    }
    XliteOpRmsNorm(rt, x, norm, _c.normEps, h);
    ForwardGetLogits(rt, h, output);

    rt.pool->PutTensor(h);
    rt.pool->PutTensor(x);
}