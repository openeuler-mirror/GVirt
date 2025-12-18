/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "ascend.h"
#include "ccl.h"
#include "sock.h"

#define XLITE_CCL_IPC_MEM_SIZE 2465792 // 2M

XcclComm::~XcclComm(void)
{
    for (uint32_t rank = 0; rank < _rankSize; rank++) {
        if (ipcXTensorMems[rank] == nullptr) {
            continue;
        }
        (void)aclrtIpcMemClose(_ipcKeys[rank]);
    }
    (void)aclrtFree(ipcMems[_rankId]);
}

int XcclComm::Init(const std::string &ip, uint32_t port)
{
    int32_t pids[XLITE_CCL_MAX_RANK_SIZE];

    if (_rankSize == 1 || _rankSize > XLITE_CCL_MAX_RANK_SIZE) {
        return 0;
    }

    XSock *sock = new XSock(_rankId, _rankSize, ip, port);

    CHECK_ACL_RET(aclrtMalloc(&ipcMems[_rankId], XLITE_CCL_IPC_MEM_SIZE, ACL_MEM_MALLOC_NORMAL_ONLY), -ENOMEM);
    CHECK_ACL_RET(aclrtIpcMemGetExportKey(ipcMems[_rankId], XLITE_CCL_IPC_MEM_SIZE, _ipcKeys[_rankId],
        EXPORT_KEY_LEN, ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT), -EFAULT);

    CHECK_ACL_RET(aclrtDeviceGetBareTgid(&pids[_rankId]), -EFAULT);
    sock->AllGather(&pids[_rankId], sizeof(int32_t), pids);
    CHECK_ACL_RET(aclrtIpcMemSetImportPid(_ipcKeys[_rankId], pids, _rankSize), -EFAULT);

    sock->AllGather(&_ipcKeys[_rankId], EXPORT_KEY_LEN, _ipcKeys);
    for (uint32_t rank = 0; rank < _rankSize; rank++) {
        if (rank == _rankId) {
            continue;
        }
        CHECK_ACL_RET(aclrtIpcMemImportByKey(&ipcMems[rank], _ipcKeys[rank],
            ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS), -EFAULT);
    }
    delete sock;
    return 0;
}
