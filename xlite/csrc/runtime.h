/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <cstdint>

typedef void *aclrtStream;
class XTensorPool;

enum commType {
    TP,
    DP,
    MAX_COMM_TYPE,
};

class XRuntime {
public:
    XRuntime(uint32_t devid, uint32_t rankId, size_t sizeMB);
    ~XRuntime(void);
    void Synchronize(void);
    uint32_t rankId;
    aclrtStream stream;
    XTensorPool *pool;
private:
    uint32_t devid;
    bool _init_outside = false;
};

#endif