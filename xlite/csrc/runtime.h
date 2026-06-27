/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <cstdint>
#include "base.h"
#include "ccl.h"

#define XLITE_DEFAULT_PORT 10266
#define XLITE_DEFAULT_COMM_OPTIMIZE_LEN 6144

#ifdef XLITE_DEBUG_ON
#define XLITE_DEBUG_POINT(condition, rt, h, str) \
    if ((condition) && !(rt).IsDummyRuntime()) { \
        (rt).Synchronize();                      \
        (h).Print((str));                        \
    }

#define XLITE_DEBUG_POINT_ROWS_COLS(condition, rt, h, str, rows, cols) \
    if ((condition) && !(rt).IsDummyRuntime()) {                       \
        (rt).Synchronize();                                            \
        (h).Print(str, rows, cols);                                    \
    }

#define XLITE_DEBUG_PTR_POINT(condition, rt, h, str, subShape, subDtype) \
    if ((condition) && !(rt).IsDummyRuntime()) {                         \
        (rt).Synchronize();                                              \
        (h).PrintPtr(str, subShape, subDtype);                           \
    }

#define XLITE_DEBUG_DUMP_XTENSOR(condition, rt, h, path) \
    if ((condition) && !(rt).IsDummyRuntime()) {         \
        (rt).Synchronize();                              \
        (h).Save(path);                                  \
    }

#else
#define XLITE_DEBUG_POINT(condition, rt, h, str)
#define XLITE_DEBUG_POINT_ROWS_COLS(condition, rt, h, str, rows, cols)
#define XLITE_DEBUG_PTR_POINT(condition, rt, h, str, subShape, subDtype)
#define XLITE_DEBUG_DUMP_XTENSOR(condition, rt, h, path)
#endif

typedef void *aclrtContext;
typedef void *aclrtNotify;
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *HcclComm;
class XTensorPool;
class XcclComm;

enum XModelAttnType {
    XMODEL_ATTN_MHA,
    XMODEL_ATTN_MLA,
    XMODEL_ATTN_DSA,
    XMODEL_ATTN_MAX_TYPE,
};

struct XModelAttnMeta {
    int version = 0;

    std::vector<uint32_t> lens;
    std::vector<uint32_t> cachedLens;
    std::vector<bool> isPrefills;

    /* only for version 0 */
    std::vector<std::vector<uint32_t>> blockTables;

    /* only for version 1 */
    XTensor vllmBlockTables;
    XTensor vllmSlotMapping;
    XTensor vllmPosition;
};

enum commType {
    TP,
    EP,
    DP,
    MAX_COMM_TYPE,
};

class XRuntime
{
public:
    XRuntime(uint32_t devid, size_t sizeMB = 0, uint32_t rankId = 0, uint32_t tpSize = 1,
             uint32_t dpSize = 1, uint32_t moeTpSize = 1, uint32_t moeEpSize = 1);
    virtual ~XRuntime(void);
    void Init(size_t sizeMB);
    void InitAttn(uint64_t maxBatchedTokens, uint64_t maxBatch, uint64_t maxSeqLen,
                  uint32_t blockSize);
    void PrepareAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                     uint64_t maxSeqLen, uint32_t nHeads, uint32_t nKVheads, uint32_t blockSize);
    void Synchronize(void);
    void EventWaitCurrStream(aclrtStream currStream);
    void EventRecordCurrStream(aclrtStream currStream);
    void MemcpyH2D(void *dst, void *src, size_t size);
    void MemcpyD2H(void *dst, void *src, size_t size);
    void MemcpyD2HAsync(void *dst, void *src, size_t size);
    void UpdateCoreNum(float blockDimUtilization);

    void SetCurrentContext();
    void NotifyWaitPeerStream();
    void NotifyRecordPeerStream();

    int InitTensorPool(size_t sizeMB);
    XTensor &GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc);
    void PutTensor(XTensor &t);
    bool TensorInPool(XTensor &t);

    void ConfigureSwizzle(uint32_t swizzle, bool useSwizzleTable);

    [[nodiscard]] virtual bool IsDummyRuntime() const
    {
        return false;
    }
    bool Inited(void)
    {
        return _inited;
    };
    uint32_t rankId(void)
    {
        return _rankId;
    };
    uint32_t tpSize(void)
    {
        return _tpSize;
    };
    uint32_t dpSize(void)
    {
        return _dpSize;
    };
    uint32_t moeTpSize(void)
    {
        return _moeTpSize;
    };
    uint32_t moeEpSize(void)
    {
        return _moeEpSize;
    };
    aclrtStream stream = nullptr;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t originAicNum;
    uint32_t originAivNum;
    HcclComm _tpComm = nullptr;
    HcclComm _dpComm = nullptr;
    HcclComm _epComm = nullptr;
    uint32_t commOptimizeLen = XLITE_DEFAULT_COMM_OPTIMIZE_LEN;
    bool enableCommOptimize;
    XTensor hiddenStatePad;
    XTensor hiddenStateSlice;
    uint32_t batchedTokens;
    uint32_t defaultMatmulSwizzle = 0x600;
    bool disableSwizzleTable = false;
    bool enableMoEAllToAll = false;

    XcclComm *_tpXcclComm = nullptr;
    XcclComm *_dpXcclComm = nullptr;
    XcclComm *_epXcclComm = nullptr;

    // for multi-task parallel
    bool multiTaskParallel = false;
    uint32_t taskId = 0;
    aclrtNotify peerNotify = nullptr;
    aclrtNotify notify = nullptr;

    // ATTN
    bool _attnInitialized = false;
    uint32_t _maxNumBlocks;
    int _batch;
    uint32_t _tileSizeOfCachedKV;
    XTensor _attnPosition;
    XTensor _attnBlockTables;
    XTensor _attnSlotMapping;
    XTensor _position;
    XTensor _blockTables;
    XTensor _slotMapping;
    XTensor _cachedLens;
    XTensor _lens;
    XTensor _queryStartLoc;

    // for MoE
    XTensor _tokensPerEpGroupAllEpHost;

protected:
    int GetNodeIps(void);
    int InitHcclComm(void);
    int InitXcclComm(void);
    void FiniXcclComm(void);
    uint32_t _devid;
    aclrtEvent _event = nullptr;
    aclrtContext context = nullptr;
    bool _initOutside = false;
    bool _inited = false;
    XTensorPool *_pool = nullptr;
    uint32_t _rankId;
    uint32_t _tpSize;
    uint32_t _dpSize;
    uint32_t _moeTpSize;
    uint32_t _moeEpSize;
    uint32_t _rankSize;
    uint32_t _nDevPerNode = 0;
    uint32_t _port = XLITE_DEFAULT_PORT;
    std::vector<std::string> _ips;

    char _ipcXTensorKeys[XLITE_CCL_MAX_RANK_SIZE][EXPORT_KEY_LEN];
    void *_ipcXTensorMems[XLITE_CCL_MAX_RANK_SIZE] = {nullptr};
};

class XDummyRuntime : public XRuntime
{
public:
    using XRuntime::XRuntime;

    [[nodiscard]] bool IsDummyRuntime() const override
    {
        return true;
    }
    void InitDummyRuntime(size_t sizeMB);
    size_t maxUsedSize(void);

private:
    int InitDummyXcclComm(void);
};
#endif
