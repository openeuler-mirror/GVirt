/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "acl.h"
#include "base.h"
#include "runtime.h"

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