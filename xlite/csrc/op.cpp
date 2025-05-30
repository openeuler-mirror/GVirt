/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "acl.h"
#include "runtime.h"
#include "op.h"
#include "kernels/kernel_entry.h"

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