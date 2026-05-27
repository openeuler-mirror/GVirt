/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <sstream>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "sock.h"
#include "ccl.h"

#define XLITE_DEFAULT_IP "127.0.0.1"
#define XLITE_DP_PORT_OFFSET 200
#define XLITE_CCL_PORT_OFFSET 400

bool isEnvironmentVariableTrue(const char *env_value_cstr)
{
    if (env_value_cstr == nullptr) {
        return false;
    }

    std::string env_value = env_value_cstr;
    for (size_t i = 0; i < env_value.size(); ++i) {
        env_value[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(env_value[i])));
    }

    return env_value == "true" || env_value == "1" || env_value == "yes" || env_value == "on";
}

XRuntime::XRuntime(uint32_t devid, size_t sizeMB, uint32_t rankId, uint32_t tpSize, uint32_t dpSize)
    : _devid(devid), _rankId(rankId), _tpSize(tpSize), _dpSize(dpSize)
{
    if (sizeMB != 0) {
        Init(sizeMB);
    }
}

void XRuntime::Init(size_t sizeMB)
{
    if (_inited) {
        return;
    }
    aclError initRet = aclInit(nullptr);
    uint32_t count;
    if (initRet == ACL_ERROR_REPEAT_INITIALIZE) {
        _initOutside = true;
    } else {
        CHECK_ACL(initRet);
    }
    CHECK_ACL(aclrtSetDevice(_devid));
    CHECK_ACL(aclrtCreateStream(&stream));
    CHECK_ACL(aclrtGetDeviceCount(&count));
    _nDevPerNode = count;

    if (sizeMB != 0) {
        _pool = new XTensorPool(sizeMB << MB_BIT, _rankId);
        if (_pool->Init()) {
            throw std::runtime_error("XRuntime: tensor pool initialization failed");
        }
    }

    _rankSize = _tpSize * _dpSize;
    if (InitHcclComm()) {
        delete _pool;
        throw std::runtime_error("XRuntime: HCCL initialization failed");
    }

    if (sizeMB != 0) {
        if (InitXcclComm()) {
            delete _pool;
            throw std::runtime_error("XRuntime: XCCL initialization failed");
        }
    }

    int64_t val;
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_AI_CORE_NUM, &val));
    aicNum = static_cast<uint32_t>(val);
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &val));
    aivNum = static_cast<uint32_t>(val);
    originAicNum = aicNum;
    originAivNum = aivNum;

    CHECK_ACL(aclrtCreateEvent(&_event));
    CHECK_ACL(aclrtCreateNotify(&notify, 0));
    CHECK_ACL(aclrtGetCurrentContext(&context));

    const char *envCommOptimizeLen = std::getenv("XLITE_COMM_OPTIMIZE_LEN");
    if (envCommOptimizeLen) {
        char *endPtr = nullptr;
        long val = strtol(envCommOptimizeLen, &endPtr, 10);
        if (endPtr != envCommOptimizeLen && *endPtr == '\0' && val >= 0) {
            commOptimizeLen = static_cast<uint32_t>(val);
        }
    }

    _inited = true;
}

XRuntime::~XRuntime(void)
{
    FiniXcclComm();

    if (_tpSize > 1 && _tpComm) {
        HcclCommDestroy(_tpComm);
    }
    if (_dpSize > 1 && _dpComm) {
        HcclCommDestroy(_dpComm);
    }

    delete _pool;
    if (_event) {
        (void)aclrtDestroyEvent(_event);
    }
    if (notify) {
        (void)aclrtDestroyNotify(notify);
    }
    if (stream) {
        (void)aclrtDestroyStream(stream);
    }
    (void)aclrtResetDevice(static_cast<int>(_devid));

    if (_attnInitialized) {
        (void)aclrtFree(_position.ptr);
        (void)aclrtFree(_slotMapping.ptr);
        (void)aclrtFree(_cachedLens.ptr);
        (void)aclrtFree(_lens.ptr);
        (void)aclrtFree(_queryStartLoc.ptr);
        (void)aclrtFree(_blockTables.ptr);
    }

    if (!_initOutside) {
        (void)aclFinalize();
    }
}

int XRuntime::GetNodeIps(void)
{
    const char *envDevs = std::getenv("XLITE_DEVS_PER_NODE");
    const char *envIps = std::getenv("XLITE_NODE_IPS");
    const char *envPort = std::getenv("XLITE_PORT");

    if (envDevs) {
        char *endPtr = nullptr;
        long val = strtol(envDevs, &endPtr, 10);
        if (endPtr != envDevs && *endPtr == '\0' && val >= 0) {
            _nDevPerNode = static_cast<uint32_t>(val);
        }
    }

    if (envPort) {
        char *endPtr = nullptr;
        long val = strtol(envPort, &endPtr, 10);
        if (endPtr != envPort && *endPtr == '\0' && val >= 0) {
            _port = static_cast<uint32_t>(val);
        }
    }

    if (_rankSize <= _nDevPerNode) {
        _ips.push_back(std::string(XLITE_DEFAULT_IP));
        return 0;
    }

    if (!envIps) {
        throw std::runtime_error(std::string(__func__) +
                                 ": please set XLITE_NODE_IPS in multi-node environment.");
    }

    std::string ipsStr(envIps);
    std::istringstream iss(ipsStr);
    std::string ip;
    while (std::getline(iss, ip, ',')) {
        _ips.push_back(ip);
    }

    if (_ips.size() != DIV_ROUND_UP(_rankSize, _nDevPerNode)) {
        throw std::runtime_error(std::string(__func__) + ": XLITE_NODE_IPS not match " +
                                 std::to_string(_rankSize) + " / " + std::to_string(_nDevPerNode));
    }
    return 0;
}

void XRuntime::FiniXcclComm(void)
{
    delete _tpXcclComm;
    delete _dpXcclComm;
    for (uint32_t rank = 0; rank < _rankSize; rank++) {
        if (_ipcXTensorMems[rank] == nullptr) {
            continue;
        }
        (void)aclrtIpcMemClose(_ipcXTensorKeys[rank]);
    }
}

int XRuntime::InitXcclComm(void)
{
    std::string ip;
    uint32_t port;
    const char *envDisableXccl = std::getenv("XLITE_DISABLE_XCCL");
    const char *envDeterministic = std::getenv("HCCL_DETERMINISTIC");
    void *ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE];

    if (_rankSize == 1 || _rankSize > _nDevPerNode || _rankSize > XLITE_CCL_MAX_RANK_SIZE) {
        return 0;
    }

    if (isEnvironmentVariableTrue(envDisableXccl) || isEnvironmentVariableTrue(envDeterministic)) {
        return 0;
    }

    XSock *sock = new XSock(_rankId, _rankSize, _ips[0], _port + XLITE_CCL_PORT_OFFSET);

    _ipcXTensorMems[_rankId] = _pool->Ptr();
    CHECK_ACL(aclrtIpcMemGetExportKey(_pool->Ptr(), _pool->Size(), _ipcXTensorKeys[_rankId],
                                      EXPORT_KEY_LEN,
                                      ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION));
    sock->AllGather(&_ipcXTensorKeys[_rankId], EXPORT_KEY_LEN, _ipcXTensorKeys);
    for (uint32_t rank = 0; rank < _rankSize; rank++) {
        if (rank == _rankId) {
            continue;
        }
        CHECK_ACL(aclrtIpcMemImportByKey(&_ipcXTensorMems[rank], _ipcXTensorKeys[rank],
                                         ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS));
    }
    delete sock;

    static uint32_t portOffset = 0;
    if (_tpSize > 1) {
        ip = _ips[ROUND_DOWN(_rankId, _tpSize) / _nDevPerNode];
        port = _port + _rankId / _tpSize + portOffset;
        _tpXcclComm = new XcclComm(_rankId % _tpSize, _tpSize);
        for (uint32_t rank = 0; rank < _tpSize; rank++) {
            ipcXTensorMems[rank] = _ipcXTensorMems[ROUND_DOWN(_rankId, _tpSize) + rank];
        }
        if (_tpXcclComm->Init(ip, port, ipcXTensorMems)) {
            return -EFAULT;
        }
    }

    if (_dpSize > 1) {
        ip = _ips[_rankId % _tpSize / _nDevPerNode];
        port = _port + XLITE_DP_PORT_OFFSET + _rankId % _tpSize + portOffset;
        _dpXcclComm = new XcclComm(_rankId / _tpSize, _dpSize);
        for (uint32_t rank = 0; rank < _dpSize; rank++) {
            ipcXTensorMems[rank] = _ipcXTensorMems[rank * _tpSize + _rankId % _tpSize];
        }
        if (_dpXcclComm->Init(ip, port, ipcXTensorMems)) {
            return -EFAULT;
        }
    }
    portOffset += 500;

    return 0;
}

int XRuntime::InitHcclComm(void)
{
    std::string ip;
    uint32_t port;
    HcclRootInfo rootInfo;

    int ret = GetNodeIps();
    if (ret) {
        return ret;
    }

    static uint32_t portOffset = 0;
    if (_tpSize > 1) {
        ip = _ips[ROUND_DOWN(_rankId, _tpSize) / _nDevPerNode];
        port = _port + _rankId / _tpSize + portOffset;

        if (_rankId % _tpSize == 0) {
            CHECK_HCCL(HcclGetRootInfo(&rootInfo));
        }
        XSock *sock = new XSock(_rankId % _tpSize, _tpSize, ip, port);
        sock->Broadcast(&rootInfo, sizeof(rootInfo));
        delete sock;
        CHECK_HCCL(HcclCommInitRootInfo(_tpSize, &rootInfo, _rankId % _tpSize, &_tpComm));
    }

    if (_dpSize > 1) {
        ip = _ips[_rankId % _tpSize / _nDevPerNode];
        port = _port + XLITE_DP_PORT_OFFSET + _rankId % _tpSize + portOffset;

        if (_rankId / _tpSize == 0) {
            CHECK_HCCL(HcclGetRootInfo(&rootInfo));
        }
        XSock *sock = new XSock(_rankId / _tpSize, _dpSize, ip, port);
        sock->Broadcast(&rootInfo, sizeof(rootInfo));
        delete sock;
        CHECK_HCCL(HcclCommInitRootInfo(_dpSize, &rootInfo, _rankId / _tpSize, &_dpComm));
    }
    portOffset += 500;

    return 0;
}

void XRuntime::InitAttn(uint64_t maxBatchedTokens, uint64_t maxBatch, uint64_t maxSeqLen,
                        uint32_t blockSize)
{
    std::vector<uint32_t> vgatherIndices;
    size_t size;
    void *ptr;

    size = maxBatchedTokens * XDtypeBit(INT64) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _position.Init({maxBatchedTokens}, INT64, ptr);

    size = maxBatchedTokens * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _slotMapping.Init({maxBatchedTokens}, INT32, ptr);

    size = maxBatch * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _cachedLens.Init({maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _lens.Init({maxBatch}, INT32, ptr);

    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _queryStartLoc.Init({maxBatch}, INT32, ptr);

    size = maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize) * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    _blockTables.Init({maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize)}, INT32, ptr);
}

void XRuntime::PrepareAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                           uint64_t maxSeqLen, uint32_t blockSize)
{
    if (!_attnInitialized) {
        InitAttn(maxBatchedTokens, maxBatch, maxSeqLen, blockSize);
        _attnInitialized = true;
    }
    uint32_t batch = attnMeta.lens.size();
    std::vector<uint32_t> lens(batch);
    std::vector<uint32_t> cachedLens(batch);
    std::vector<uint32_t> queryStartLoc(batch);
    std::vector<uint32_t> slotMapping, blockTables;
    std::vector<uint64_t> position;
    uint32_t blockNum, queryStart, blockId, id, k;
    size_t size;

    batchedTokens = 0;
    _maxNumBlocks = 0;
    _batch = static_cast<int>(batch);
    queryStart = 0;
    for (uint32_t i = 0; i < batch; i++) {
        lens[i] = attnMeta.lens[i];
        cachedLens[i] = attnMeta.cachedLens[i];
        queryStartLoc[i] = queryStart;
        queryStart += lens[i];
        blockNum = DIV_ROUND_UP(lens[i] + cachedLens[i], blockSize);
        _maxNumBlocks = blockNum > _maxNumBlocks ? blockNum : _maxNumBlocks;
        batchedTokens += lens[i];
    }

    if (batchedTokens == 0 || batchedTokens > maxBatchedTokens) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": invalid attnMeta batched tokens(" +
                                 std::to_string(batchedTokens) + ") > maxBatchedTokens(" +
                                 std::to_string(maxBatchedTokens) + ")");
    }

    size = batch * XDtypeBit(INT32) / 8;
    CHECK_ACL(aclrtMemcpy(_lens.ptr, size, lens.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(
        aclrtMemcpy(_cachedLens.ptr, size, cachedLens.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(_queryStartLoc.ptr, size, queryStartLoc.data(), size,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    position.resize(batchedTokens);
    slotMapping.resize(batchedTokens);
    k = 0;
    for (uint32_t i = 0; i < batch; i++) {
        for (uint32_t j = 0; j < lens[i]; j++) {
            position[k] = cachedLens[i] + j;
            blockId = position[k] / blockSize;
            id = position[k] % blockSize;
            slotMapping[k++] = attnMeta.blockTables[i][blockId] * blockSize + id;
        }
    }
    size = batchedTokens * XDtypeBit(INT32) / 8;
    CHECK_ACL(
        aclrtMemcpy(_slotMapping.ptr, size, slotMapping.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    _attnSlotMapping = _slotMapping;

    blockTables.resize(batch * _maxNumBlocks);
    for (uint32_t i = 0; i < batch; i++) {
        blockNum = DIV_ROUND_UP(lens[i] + cachedLens[i], blockSize);
        for (uint32_t j = 0; j < blockNum; j++) {
            blockTables[i * _maxNumBlocks + j] = attnMeta.blockTables[i][j];
        }
    }
    size = batch * _maxNumBlocks * XDtypeBit(INT32) / 8;
    CHECK_ACL(
        aclrtMemcpy(_blockTables.ptr, size, blockTables.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
    _attnBlockTables = _blockTables;
    switch (attnMeta.version) {
        case 0:
            size = batchedTokens * XDtypeBit(INT64) / 8;
            CHECK_ACL(
                aclrtMemcpy(_position.ptr, size, position.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _attnPosition = _position;
            break;
        case 1:
            _attnPosition = attnMeta.vllmPosition;
            break;
        default:
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": invalid attnMeta version: " + std::to_string(attnMeta.version));
    }
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
    aicNum =
        static_cast<uint32_t>(std::round(static_cast<float>(originAicNum) * blockDimUtilization));
    aivNum =
        static_cast<uint32_t>(std::round(static_cast<float>(originAivNum) * blockDimUtilization));
}

void XRuntime::SetCurrentContext()
{
    CHECK_ACL(aclrtSetCurrentContext(context));
}

void XRuntime::NotifyWaitPeerStream()
{
    CHECK_ACL(aclrtWaitAndResetNotify(notify, stream, 0));
}

void XRuntime::NotifyRecordPeerStream()
{
    CHECK_ACL(aclrtRecordNotify(peerNotify, stream));
}

int XRuntime::InitTensorPool(size_t sizeMB)
{
    if (sizeMB != 0) {
        Init(sizeMB);
    }
    return 0;
}

XTensor &XRuntime::GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc)
{
    return _pool->GetTensor(std::move(shape), dtype, loc);
}

void XRuntime::PutTensor(XTensor &t)
{
    _pool->PutTensor(t);
}

bool XRuntime::TensorInPool(XTensor &t)
{
    return _pool->TensorInPool(t);
}

void XRuntime::ConfigureSwizzle(uint32_t swizzle, bool useSwizzleTable)
{
    defaultMatmulSwizzle = swizzle;
    disableSwizzleTable = !useSwizzleTable;
}

void XDummyRuntime::InitDummyRuntime(size_t sizeMB)
{
    if (_inited) {
        return;
    }
    aclError initRet = aclInit(nullptr);
    uint32_t count;
    if (initRet == ACL_ERROR_REPEAT_INITIALIZE) {
        _initOutside = true;
    } else {
        CHECK_ACL(initRet);
    }
    CHECK_ACL(aclrtGetDeviceCount(&count));
    _nDevPerNode = count;

    _pool = new XDummyTensorPool(sizeMB << MB_BIT, _rankId);
    if (_pool->Init()) {
        throw std::runtime_error("XDummyRuntime: tensor pool initialization failed");
    }
    _rankSize = _tpSize * _dpSize;

    (void)InitDummyXcclComm();

    int64_t val;
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_AI_CORE_NUM, &val));
    aicNum = static_cast<uint32_t>(val);
    CHECK_ACL(aclGetDeviceCapability(_devid, ACL_DEVICE_INFO_VECTOR_CORE_NUM, &val));
    aivNum = static_cast<uint32_t>(val);
    originAicNum = aicNum;
    originAivNum = aivNum;

    const char *envCommOptimizeLen = std::getenv("XLITE_COMM_OPTIMIZE_LEN");
    if (envCommOptimizeLen) {
        char *endPtr = nullptr;
        long val = strtol(envCommOptimizeLen, &endPtr, 10);
        if (endPtr != envCommOptimizeLen && *endPtr == '\0' && val >= 0) {
            commOptimizeLen = static_cast<uint32_t>(val);
        }
    }

    _inited = true;
}

int XDummyRuntime::InitDummyXcclComm(void)
{
    const char *envDisableXccl = std::getenv("XLITE_DISABLE_XCCL");
    const char *envDeterministic = std::getenv("HCCL_DETERMINISTIC");

    if (_rankSize == 1 || _rankSize > _nDevPerNode || _rankSize > XLITE_CCL_MAX_RANK_SIZE) {
        return 0;
    }

    if (isEnvironmentVariableTrue(envDisableXccl) || isEnvironmentVariableTrue(envDeterministic)) {
        return 0;
    }

    if (_tpSize > 1) {
        _tpXcclComm = new XcclComm(_rankId % _tpSize, _tpSize);
    }

    if (_dpSize > 1) {
        _dpXcclComm = new XcclComm(_rankId / _tpSize, _dpSize);
    }
    return 0;
}

size_t XDummyRuntime::maxUsedSize(void)
{
    auto *dummyPool = dynamic_cast<XDummyTensorPool *>(_pool);
    return dummyPool ? dummyPool->maxUsedSize : 0;
}
