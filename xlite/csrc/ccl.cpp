/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <cstdio>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <memory>
#include "ascend.h"
#include "ccl.h"
#include "kernels/kernel_param.h"
#include "sock.h"

#define XLITE_CCL_IPC_MEM_SIZE 2465792  // 2M

XcclComm::~XcclComm(void)
{
    for (uint32_t rank = 0; rank < _rankSize; rank++) {
        if (_ipcMems[rank] == nullptr) {
            continue;
        }
        (void)aclrtIpcMemClose(_ipcKeys[rank]);
    }
    (void)aclrtFree(_ipcMems[_rankId]);
    (void)aclrtFree(dParam);
}

int XcclComm::Init(const std::string &ip, uint32_t port,
                   void *ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE])
{
    int32_t pids[XLITE_CCL_MAX_RANK_SIZE];

    if (_rankSize == 1 || _rankSize > XLITE_CCL_MAX_RANK_SIZE) {
        return 0;
    }

    auto sock = std::make_unique<XSock>(_rankId, _rankSize, ip, port);

    CHECK_ACL(aclrtMalloc(&_ipcMems[_rankId], XLITE_CCL_IPC_MEM_SIZE, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMemset(_ipcMems[_rankId], XLITE_CCL_IPC_MEM_SIZE, 0, XLITE_CCL_IPC_MEM_SIZE));
    CHECK_ACL(aclrtIpcMemGetExportKey(_ipcMems[_rankId], XLITE_CCL_IPC_MEM_SIZE, _ipcKeys[_rankId],
                                      EXPORT_KEY_LEN, ACL_RT_IPC_MEM_EXPORT_FLAG_DEFAULT));

    CHECK_ACL(aclrtDeviceGetBareTgid(&pids[_rankId]));
    sock->AllGather(&pids[_rankId], sizeof(int32_t), pids);
    CHECK_ACL(aclrtIpcMemSetImportPid(_ipcKeys[_rankId], pids, _rankSize));

    sock->AllGather(&_ipcKeys[_rankId], EXPORT_KEY_LEN, _ipcKeys);
    for (uint32_t rank = 0; rank < _rankSize; rank++) {
        if (rank == _rankId) {
            continue;
        }
        CHECK_ACL(aclrtIpcMemImportByKey(&_ipcMems[rank], _ipcKeys[rank],
                                         ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS));
    }

    struct XcclParam hParam;
    for (uint32_t rank = 0; rank < _rankSize; rank++) {
        hParam.ipcMems[rank] = reinterpret_cast<uint64_t>(_ipcMems[rank]);
        hParam.ipcXTensorMems[rank] = reinterpret_cast<uint64_t>(ipcXTensorMems[rank]);
    }
    CHECK_ACL(aclrtMalloc((void **)&dParam, sizeof(struct XcclParam), ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMemcpy(dParam, sizeof(struct XcclParam), &hParam, sizeof(struct XcclParam),
                          ACL_MEMCPY_HOST_TO_DEVICE));
    return 0;
}
