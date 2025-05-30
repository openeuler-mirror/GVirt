/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "acl.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"

XModel::XModel(struct XModelConfig &c, uint32_t rankId) : _c(c), _rankId(rankId)
{
    attnNorm.resize(c.nLayers);
    attnOut.resize(c.nLayers);
    mlaQA.resize(c.nLayers);
    mlaQB.resize(c.nLayers);
    mlaQNorm.resize(c.nLayers);
    mlaKVA.resize(c.nLayers);
    mlaKVB.resize(c.nLayers);
    mlaKVNorm.resize(c.nLayers);
    mlpNorm.resize(c.nLayers);
    mlpUpGate.resize(c.nDenseLayers);
    mlpDown.resize(c.nDenseLayers);
    Gate.resize(c.nLayers);
    GateBias.resize(c.nLayers);
    SEUpGate.resize(c.nLayers);
    SEDown.resize(c.nLayers);
    REUpGate.resize(c.nLayers);
    REUpGateScale.resize(c.nLayers);
    REDown.resize(c.nLayers);
    REDownScale.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        REUpGate[i].resize(c.nRoutedExperts);
        REUpGateScale[i].resize(c.nRoutedExperts);
        REDown[i].resize(c.nRoutedExperts);
        REDownScale[i].resize(c.nRoutedExperts);
    }
}

void XModel::ForwardParallelEmbed(XRuntime &rt, XTensor *input, XTensor *embed, XTensor *output)
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

void XModel::ForwardAttn(XRuntime &rt, uint32_t layer, XTensor *hiddenState)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XModel::ForwardFFN(XRuntime &rt, uint32_t layer, XTensor *hiddenState)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XModel::ForwardGetLogits(XRuntime &rt, XTensor *input, XTensor *output)
{
    std::cout << __func__ << ": TODO" << std::endl;
}

void XModel::Forward(XRuntime &rt, XTensor *input, XTensor *output)
{
    uint32_t batch = input->shape[0];
    uint32_t seqLen = input->shape[1];
    uint32_t m = batch * seqLen;
    XTensor *x, *h;

    if (rt.rankId != _rankId) {
        std::cerr << __FILE__ << ":" << __LINE__ << "check rank id failed" << std::endl;
        return;
    }

    x = rt.pool->GetTensor({m, _c.hiddenSize}, embed.dtype);
    h = rt.pool->GetTensor({m, _c.hiddenSize}, embed.dtype);

    ForwardParallelEmbed(rt, input, &embed, x);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        XliteOpRmsNorm(rt, x, &attnNorm[i], _c.normEps, h);
        ForwardAttn(rt, i, h);
        XliteOpAdd(rt, x, h, x);
        XliteOpRmsNorm(rt, x, &mlpNorm[i], _c.normEps, h);
        ForwardFFN(rt, i, h);
        XliteOpAdd(rt, x, h, x);
    }
    XliteOpRmsNorm(rt, x, &norm, _c.normEps, h);
    ForwardGetLogits(rt, h, output);

    rt.pool->PutTensor(h);
    rt.pool->PutTensor(x);
}