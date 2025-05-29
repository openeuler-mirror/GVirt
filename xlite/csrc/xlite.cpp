/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "xlite_base.h"
#include "xlite_acl.h"
#include "xlite.h"
#include "kernels/kernel_entry.h"

XRuntime::XRuntime(uint32_t devid, size_t sizeMB) : devid(devid)
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(devid));
    CHECK_ACL(aclrtCreateStream(&stream));

    pool = new XTensorPool(sizeMB << MB_BIT);
    if (pool->Init()) {
        return;
    }
}

XRuntime::~XRuntime(void)
{
    delete pool;
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(devid));
    CHECK_ACL(aclFinalize());
}

void XliteOpAdd(XRuntime &rt, XTensor *x, XTensor *y, XTensor *z)
{
    uint32_t blockDim = 8;
    std::vector<long> shape = {8, 2048};

    if (x->dtype != FP16 || y->dtype != FP16 || z->dtype != FP16) {
        std::cerr << __FILE__ << ":" << __LINE__ << "unsupport dtype" << std::endl;
        return;
    }
    if (x->shape != shape || y->shape != shape || z->shape != shape) {
        std::cerr << __FILE__ << ":" << __LINE__ << "unsupport shape" << std::endl;
        return;
    }
    add_do(blockDim, rt.stream, x->ptr, y->ptr, z->ptr);
    CHECK_ACL(aclrtSynchronizeStream(rt.stream));

    XTensor *a = rt.pool->GetTensor(shape, FP16);
    XTensor *b = rt.pool->GetTensor(shape, FP16);
    XTensor *c = rt.pool->GetTensor(shape, FP16);
    add_do(blockDim, rt.stream, a->ptr, b->ptr, c->ptr);
    CHECK_ACL(aclrtSynchronizeStream(rt.stream));
    rt.pool->PutTensor(a);
    rt.pool->PutTensor(b);
    rt.pool->PutTensor(c);
}

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