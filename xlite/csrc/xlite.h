/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_H_
#define _XLITE_H_

#include "xlite_base.h"

typedef void *aclrtStream;
class XRuntime {
public:
    XRuntime(uint32_t devid, size_t sizeMB);
    ~XRuntime();
    uint32_t devid;
    aclrtStream stream;
    XTensorPool *pool;
};

void XliteOpAdd(XRuntime &rt, XTensor *x, XTensor *y, XTensor *z);

#endif