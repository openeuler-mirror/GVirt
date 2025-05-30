/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "xlite_acl.h"
#include "xlite_base.h"
#include "xlite_runtime.h"
#include "xlite_op.h"
#include "xlite_model.h"

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

void XModel::Forward(XRuntime &rt, XTensor *input, XTensor *output)
{
    std::cout << "rank" << _rankId << " : xlite model forward, layers: " << _c.nLayers << std::endl;
}