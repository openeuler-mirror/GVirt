/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"

XModel::XModel(struct XModelConfig &c, uint32_t rankId, enum XModelAttnType aType) : _c(c), _rankId(rankId), _aType(aType)
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
    for (uint32_t i = 0; i < c.nLayers; i++) {
        moeREUpGate[i].resize(c.nRoutedExperts);
        moeREUpGateScale[i].resize(c.nRoutedExperts);
        moeREDown[i].resize(c.nRoutedExperts);
        moeREDownScale[i].resize(c.nRoutedExperts);
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

void XModel::ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    std::cout << __func__ << ": TODO" << std::endl;
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