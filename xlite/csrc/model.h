/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_MODEL_H_
#define _XLITE_MODEL_H_

#include "base.h"

enum XModelAttnType {
    XMODEL_ATTN_MHA,
    XMODEL_ATTN_MLA,
    XMODEL_ATTN_MAX_TYPE,
};

struct XModelConfig {
    // global config
    uint32_t vocabSize;
    uint32_t hiddenSize;
    uint32_t nLayers;

    // attention config
    enum XModelAttnType attnType = XMODEL_ATTN_MHA;
    uint32_t nHeads;
    uint32_t nKvHeads;
    uint32_t headDim;
    uint32_t ropeHeadDim;
    uint32_t nopeHeadDim;
    uint32_t vHeadDim;
    uint32_t qLoraRank;
    uint32_t kvLoraRank;
    uint32_t maxM;
    uint32_t blockSize;
    uint32_t maxBatch;
    uint32_t maxSeqLen;
    float normEps;
    float ropeTheta;
    float softmaxScale;

    // mlp
    uint32_t nDenseLayers;
    uint32_t nRoutedExperts;
    uint32_t nSharedExperts;
    uint32_t nExpertGroups;
    uint32_t nLimitedGroups;
    uint32_t nActExperts;
    uint32_t intermediateSize;
    uint32_t moeIntermediateSize;
    float routeScale;

    // parallel config
    uint32_t defTpSize;
    uint32_t defDpSize;
    uint32_t moeEpSize;
    uint32_t moeTPSize;
};

struct XModelAttnMeta {
    std::vector<uint32_t> lens;
    std::vector<uint32_t> cachedLens;
    std::vector<bool> isPrefills;
    std::vector<std::vector<uint32_t>> blockTables;
};

class XModel {
public:
    XModel(struct XModelConfig &c, uint32_t rankId);
    void Init(void);
    ~XModel(void);
    void Forward(XRuntime &rt, XTensor &input,
                 XModelAttnMeta& attnMeta,
                 std::vector<std::pair<XTensor, XTensor>>& kvCache,
                 XTensor &freqsCis, XTensor &output);

    // weights
    XTensor embed;
    XTensor norm;
    XTensor head;

    std::vector<XTensor> attnNorm;
    std::vector<XTensor> attnOut;
    std::vector<XTensor> mhaQKV;
    std::vector<XTensor> mlaQA;
    std::vector<XTensor> mlaQB;
    std::vector<XTensor> mlaQNorm;
    std::vector<XTensor> mlaKVA;
    std::vector<XTensor> mlaKVB;
    std::vector<XTensor> mlaKVNorm;

    std::vector<XTensor> mlpNorm;
    std::vector<XTensor> mlpUpGate;
    std::vector<XTensor> mlpDown;

    std::vector<XTensor> moeGate;
    std::vector<XTensor> moeGateBias;
    std::vector<XTensor> moeSEUpGate;
    std::vector<XTensor> moeSEDown;
    std::vector<std::vector<XTensor>> moeREUpGate;
    std::vector<std::vector<XTensor>> moeREUpGateScale;
    std::vector<std::vector<XTensor>> moeREDown;
    std::vector<std::vector<XTensor>> moeREDownScale;

private:
    void ForwardParallelEmbed(XRuntime &rt, XTensor &input, XTensor &embed, XTensor &output);
    void prepareAttn(XRuntime &rt, XModelAttnMeta& attnMeta);
    void ForwardAttnMLA(XRuntime &rt, uint32_t layer,
                        XModelAttnMeta& attnMeta,
                        std::vector<std::pair<XTensor, XTensor>>& kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void ForwardAttnMHA(XRuntime &rt, uint32_t layer,
                        XModelAttnMeta& attnMeta,
                        std::vector<std::pair<XTensor, XTensor>>& kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void ForwardAttn(XRuntime &rt, uint32_t layer,
                     XModelAttnMeta& attnMeta,
                     std::vector<std::pair<XTensor, XTensor>>& kvCache,
                     XTensor &freqsCis, XTensor &hiddenState);
    void ForwardMLP(XRuntime &rt, XTensor &upGate, XTensor &down, XTensor &hiddenState, bool withAllReduce);
    std::tuple<XTensor &, XTensor &> ForwardMoEGate(XRuntime &rt, uint32_t layer, XTensor &input);
    std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &> ForwardMoEDispatch(XRuntime &rt,
                                                                                         XTensor &tokenSorted,
                                                                                         XTensor &weights,
                                                                                         XTensor &routing);
    void ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing, XTensor &unpIdx,
                           XTensor &expertsSorted, XTensor &expertsCounts);
    void ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &output);
    struct XModelConfig _c;
    uint32_t _rankId;

    // FFN
    XTensor _gateIndicts;
    std::vector<XTensor> _moeREUpGate;
    std::vector<XTensor> _moeREUpGateScale;
    std::vector<XTensor> _moeREDown;
    std::vector<XTensor> _moeREDownScale;
};

#endif