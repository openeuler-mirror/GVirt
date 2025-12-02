/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <sstream>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "sock.h"

#define XLITE_DEFAULT_IP "127.0.0.1"
#define XLITE_DP_PORT_OFFSET 200

XRuntime::XRuntime(uint32_t devid, size_t sizeMB, uint32_t rankId, uint32_t tpSize, uint32_t dpSize)
    : _devid(devid), _rankId(rankId), _tpSize(tpSize), _dpSize(dpSize)
{
    aclError init_ret = aclInit(nullptr);
    if (init_ret == ACL_ERROR_REPEAT_INITIALIZE) {
        _init_outside = true;
    } else {
        CHECK_ACL(init_ret);
    }
    CHECK_ACL(aclrtSetDevice(devid));
    CHECK_ACL(aclrtCreateStream(&stream));

    if (sizeMB != 0) {
        pool = new XTensorPool(sizeMB << MB_BIT);
        if (pool->Init()) {
            return;
        }
    }

    _rankSize = tpSize * dpSize;
    if (InitComm()) {
        delete pool;
        return;
    }

    int64_t val;
    CHECK_ACL(aclGetDeviceCapability(devid, ACL_DEVICE_INFO_AI_CORE_NUM, &val));
    aicNum = (uint32_t)val;
    CHECK_ACL(aclGetDeviceCapability(devid, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &val));
    aivNum = (uint32_t)val;
    originAicNum = aicNum;
    originAivNum = aivNum;

    CHECK_ACL(aclrtCreateEvent(&_event));
}

XRuntime::~XRuntime(void)
{
    if (_tpSize > 1 && _tpComm) {
        HcclCommDestroy(_tpComm);
    }
    if (_dpSize > 1 && _dpComm) {
        HcclCommDestroy(_dpComm);
    }

    delete pool;
    CHECK_ACL(aclrtDestroyEvent(_event));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(_devid));
    if (!_init_outside) {
        CHECK_ACL(aclFinalize());
    }
}

int XRuntime::GetNodeIps(void)
{
    const char* envDevs = std::getenv("XLITE_DEVS_PER_NODE");
    const char* envIps = std::getenv("XLITE_NODE_IPS");
    const char* envPort = std::getenv("XLITE_PORT");

    if (envDevs) {
        _nDevPerNode = atoi(envDevs);
    }

    if (envPort) {
        _port = atoi(envPort);
    }

    if (_rankSize <= _nDevPerNode) {
        _ips.push_back(std::string(XLITE_DEFAULT_IP));
        return 0;
    }

    if (!envIps) {
        std::cerr << __func__ << ": please set XLITE_NODE_IPS in multi-node environment." << std::endl;
        return -EINVAL;
    }

    std::string ipsStr(envIps);
    std::istringstream iss(ipsStr);
    std::string ip;
    while (std::getline(iss, ip, ',')) {
        _ips.push_back(ip);
    }

    if (_ips.size() != DIV_ROUND_UP(_rankSize, _nDevPerNode)) {
        std::cerr << __func__ << ": XLITE_NODE_IPS not match " << _rankSize << " / " << _nDevPerNode << std::endl;
        return -EFAULT;
    }
    return 0;
}

int XRuntime::InitComm(void)
{
    std::string ip;
    uint32_t port;
    HcclRootInfo rootInfo;

    int ret = GetNodeIps();
    if (ret) {
        return ret;
    }

    if (_tpSize > 1) {
        ip = _ips[ROUND_DOWN(_rankId, _tpSize) / _nDevPerNode];
        port = _port + _rankId / _tpSize;

        if (_rankId % _tpSize == 0) {
            CHECK_HCCL_RET(HcclGetRootInfo(&rootInfo), -EFAULT);
        }
        XSock *sock = new XSock(_rankId % _tpSize, _tpSize, ip, port);
        sock->Broadcast(&rootInfo, sizeof(rootInfo));
        delete sock;
        CHECK_HCCL_RET(HcclCommInitRootInfo(_tpSize, &rootInfo, _rankId % _tpSize, &_tpComm), -EFAULT);
    }

    if (_dpSize > 1) {
        ip = _ips[_rankId % _tpSize / _nDevPerNode];
        port = _port + XLITE_DP_PORT_OFFSET + _rankId % _tpSize;

        if (_rankId / _tpSize == 0) {
            CHECK_HCCL_RET(HcclGetRootInfo(&rootInfo), -EFAULT);
        }
        XSock *sock = new XSock(_rankId / _tpSize, _dpSize, ip, port);
        sock->Broadcast(&rootInfo, sizeof(rootInfo));
        delete sock;
        CHECK_HCCL_RET(HcclCommInitRootInfo(_dpSize, &rootInfo, _rankId / _tpSize, &_dpComm), -EFAULT);
    }

    return 0;
}

void XRuntime::Synchronize(void)
{
    CHECK_ACL(aclrtSynchronizeStream(stream));
}

void XRuntime::EventWaitCurrStream(aclrtStream currStream)
{
    CHECK_ACL(aclrtRecordEvent(_event, currStream));
    CHECK_ACL(aclrtStreamWaitEvent(stream, _event));
    CHECK_ACL(aclrtResetEvent(_event, stream));
}

void XRuntime::EventRecordCurrStream(aclrtStream currStream)
{
    CHECK_ACL(aclrtRecordEvent(_event, stream));
    CHECK_ACL(aclrtStreamWaitEvent(currStream, _event));
    CHECK_ACL(aclrtResetEvent(_event, currStream));
}

void XRuntime::MemcpyH2D(void *dst, void *src, size_t size)
{
    CHECK_ACL(aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE));
}

void XRuntime::UpdateCoreNum(float blockDimUtilization)
{
    aicNum = static_cast<uint32_t>(std::round(static_cast<float>(originAicNum) * blockDimUtilization));
    aivNum = static_cast<uint32_t>(std::round(static_cast<float>(originAivNum) * blockDimUtilization));
}

int XRuntime::InitTensorPool(size_t sizeMB)
{
    if (pool) {
        std::cout << "pool already inited!" << std::endl;
        return 0;
    }
    pool = new XTensorPool(sizeMB << MB_BIT);
    return pool->Init();
}
