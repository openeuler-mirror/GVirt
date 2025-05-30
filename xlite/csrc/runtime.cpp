/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "acl.h"
#include "base.h"
#include "runtime.h"

XRuntime::XRuntime(uint32_t devid, uint32_t rankId, size_t sizeMB) : rankId(rankId), devid(devid)
{
    aclError init_ret = aclInit(nullptr);
    if (init_ret == ACL_ERROR_REPEAT_INITIALIZE) {
        _init_outside = true;
    } else {
        CHECK_ACL(init_ret);
    }
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
    if (!_init_outside) {
        CHECK_ACL(aclFinalize());
    }
}

void XRuntime::Synchronize(void)
{
    CHECK_ACL(aclrtSynchronizeStream(stream));
}