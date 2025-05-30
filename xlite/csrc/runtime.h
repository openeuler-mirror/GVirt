/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <cstdint>

typedef void *aclrtStream;
class XTensorPool;
class XRuntime {
public:
    XRuntime(uint32_t devid, uint32_t rankId, size_t sizeMB);
    ~XRuntime();
    uint32_t devid;
    uint32_t rankId;
    aclrtStream stream;
    XTensorPool *pool;
};

#endif