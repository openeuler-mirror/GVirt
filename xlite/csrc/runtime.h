/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_RUNTIME_H_
#define _XLITE_RUNTIME_H_

#include <cstdint>
#include <atomic>
#include <array>
#include <functional>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "base.h"

#define XLITE_DEFAULT_PORT 10266
#define XLITE_DEFAULT_COMM_OPTIMIZE_LEN 6144
#define XLITE_ACTIVE_TOKENS_RATIO_PER_EP_THRESHOLD 1024
#define XLITE_FWD_POOL_SIZE 8

typedef void *aclrtContext;
typedef void *aclrtNotify;
typedef void *aclrtStream;
typedef void *aclrtEvent;
typedef void *aclmdlRI;
typedef void *HcclComm;
class XTensorPool;
class XcclComm;

enum XModelAttnType {
    XMODEL_ATTN_MHA,
    XMODEL_ATTN_MLA,
    XMODEL_ATTN_DSA,
    XMODEL_ATTN_HYBRID_GDN,  // GDN linear attention + full attention
    XMODEL_ATTN_HYBRID_KDA,  // KDA linear attention + sparse MLA
    XMODEL_ATTN_CXA,
    XMODEL_ATTN_MAX_TYPE,
};

inline bool IsHybridAttn(XModelAttnType type)
{
    return type == XMODEL_ATTN_HYBRID_GDN || type == XMODEL_ATTN_HYBRID_KDA;
}

enum XModelLayerAttnType {
    XMODEL_LAYER_ATTN_FULL = 0,
    XMODEL_LAYER_ATTN_LINEAR = 1,
};

// CXA 5-tuple block-size indices: (indexer_state, indexer_k, compress_kv, state, swa_kv).
enum CxaCacheIndex {
    CXA_INDEXER_STATE = 0,
    CXA_INDEXER_K = 1,
    CXA_COMPRESS_KV = 2,
    CXA_STATE = 3,
    CXA_SWA_KV = 4,
};

struct XModelAttnMeta {
    int version = 0;
    enum XModelAttnType attnType = XMODEL_ATTN_MHA;

    std::vector<uint32_t> lensCpu;
    std::vector<uint32_t> cachedLensCpu;

    /* for version 0/1 */
    std::vector<std::vector<uint32_t>> blockTablesCpu;

    /* for version 1/2 */
    XTensor position;

    /* only for version 2 */
    std::vector<XTensor> blockTables;  // family slots: [0]=main KV, [1]=mamba state indices
                                       // (hybrid), [2]=tail ring (GLM only)
    std::vector<XTensor> slotMapping;  // single main-KV token slot table
    XTensor queryStartLoc;
    XTensor lens;
    XTensor cachedLens;
};

struct ForwardTask {
    enum Kind : uint8_t {
        FORWARD,
        FORWARD_AND_GET_LOGITS,
        FORWARD_WITH_INPUTS_EMBEDS,
        FORWARD_GET_LOGITS,
    } kind;
    XTensor input;
    XModelAttnMeta attnMeta;
    std::vector<std::vector<XTensor>> kvCache;
    std::vector<XTensor> deepstackInputEmbeds;
    std::vector<XTensor> freqsCis;
    XTensor freqsCisSingle;
    XTensor inputIds;
    XTensor indices;
    XTensor output;
    std::vector<XTensor> auxOutputs;
    uint64_t seq = 0;
};

class XRuntime;
using ForwardTaskHandler = std::function<void(XRuntime &, ForwardTask &)>;

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
    void InitAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                  uint64_t maxSeqLen, const std::vector<uint32_t> &blockSizes, uint32_t indexTopK);
    void PrepareAttn(XModelAttnMeta &attnMeta, uint64_t maxBatchedTokens, uint64_t maxBatch,
                     uint64_t maxSeqLen, uint32_t nHeads, uint32_t nKVheads,
                     std::vector<uint32_t> blockSizes, uint32_t hiddenSize, uint32_t nRoutedExperts,
                     uint32_t defDpSize, int inputDtype, int weightsDtype, uint32_t indexTopK,
                     const std::vector<uint32_t> &layerTypes);
    void Synchronize(void);
    void EventWaitCurrStream(aclrtStream currStream);
    void EventRecordCurrStream(aclrtStream currStream);

    uint64_t AllocFwdSeq();
    void RecordInputEvent(aclrtStream currStream, uint64_t seq);
    void WaitInputEvent(uint64_t seq);
    void RecordOutputEvent(uint64_t seq);
    void WaitOutputEvent(aclrtStream currStream, uint64_t seq);
    void MemcpyH2D(void *dst, void *src, size_t size);
    void MemcpyD2H(void *dst, void *src, size_t size);
    void MemcpyD2HAsync(void *dst, void *src, size_t size);
    void UpdateCoreNum(float blockDimUtilization);
#ifdef XLITE_DEBUG_ON
    void VerifyAttnMetaV2(const XModelAttnMeta &attnMeta, std::vector<uint32_t> blockSizes);
#endif

    void SetCurrentContext();
    void NotifyWaitPeerStream();
    void NotifyRecordPeerStream();

    void StartForwardThread();
    void StopForwardThread();
    [[nodiscard]] bool ForwardThreadEnabled() const
    {
        return _fwdThread.joinable();
    }
    void SubmitForward(ForwardTask &&task);

    void RegisterForwardTaskHandler(ForwardTask::Kind kind, ForwardTaskHandler handler);
    [[nodiscard]] bool ForwardTaskHandlersReady() const;

    void PrepareCommBuffers(uint64_t maxBatch, uint32_t hiddenSize, uint32_t nRoutedExperts,
                            uint32_t defDpSize, int inputDtype, int weightsDtype);
    void CaptureHcclAllGather(void *send, void *recv, uint32_t m, int hcclDtype,
                              enum commType type);
    void RunHcclAllGatherInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                 enum commType type);
    void AllGatherInGraph(void *send, void *recv, uint32_t m, int hcclDtype, enum commType type);
    [[nodiscard]] bool AllGatherInGraphActive(enum commType type) const;

    void CaptureHcclAllReduce(void *send, void *recv, uint32_t m, int hcclDtype,
                              enum commType type);
    void RunHcclAllReduceInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                 enum commType type);
    void AllReduceSumInGraph(void *in, void *out, uint32_t m, int hcclDtype, enum commType type);
    [[nodiscard]] bool AllReduceSumInGraphActive(enum commType type) const;
    [[nodiscard]] bool AllReduceSumInGraphMatch(enum commType type, enum XDtype dtype,
                                                uint64_t numel, uint32_t *mOut) const;

    void CaptureHcclReduceScatter(void *send, void *recv, uint32_t m, int hcclDtype,
                                  enum commType type);
    void RunHcclReduceScatterInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                                     enum commType type);
    void ReduceScatterInGraph(void *send, void *recv, uint32_t m, int hcclDtype,
                              enum commType type);
    [[nodiscard]] bool ReduceScatterInGraphActive(enum commType type) const;

    int InitTensorPool(size_t sizeMB);
    XTensor &GetTensor(std::vector<size_t> shape, enum XDtype dtype, DebugSrcLoc loc);
    void InitIndexerHadamard();
    void PutTensor(XTensor &t);
    bool TensorInPool(XTensor &t);
    int64_t GetTensorOffset(XTensor &t);

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
    aclrtStream hcclSideStream = nullptr;
    aclrtEvent hcclStartEv = nullptr;
    aclrtEvent hcclEndEv = nullptr;
    struct GraphCaptureEntry {
        aclmdlRI modelRI = nullptr;
        void *sendAddr = nullptr;
        void *recvAddr = nullptr;
    };
    std::vector<GraphCaptureEntry> _agGraphs;
    XTensor _agSendBuf;
    XTensor _agRecvBuf;
    uint64_t _agPerTokenBytes = 0;
    std::vector<GraphCaptureEntry> _rsGraphs;
    XTensor _rsSendBuf;
    XTensor _rsRecvBuf;
    uint64_t _rsPerTokenBytes = 0;
    uint32_t _rsHiddenSize = 0;
    std::vector<GraphCaptureEntry> _arGraphs;
    XTensor _arBuf;  // single in-place buffer: HCCL AllReduce supports send==recv (capture焊死同址)
    uint64_t _arPerTokenBytes = 0;  // hiddenSize * dtypeBytes(_arDtype)
    uint32_t _arHiddenSize = 0;     // per-token element count captured
    enum XDtype _arDtype = BF16;    // captured dtype (== inputDtype)
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
    double activeTokensRatioPerEp = 1.0f;

    // cross-layer buffer
    XTensor _indexerHadamard;
    XTensor _dsaTopkBuffer;  // int32, buffer for storing topk results, allocated via aclrtMalloc
    XTensor *dsaPerLayerTopk = nullptr;  // ptr to the **current** layer's topk result, either a
                                         // nullptr (no topk needed) or a ptr to _dsaTopkBuffer
    XTensor _attnZeroBuffer;

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
    // Host-side: true when this step is decode (seqlen==1 and all cached_lens>0).
    // Avoids D2H sync via GetFirstAttnPosition in every linear layer.
    bool _linearDecodeStep = false;
    // Per-layer GDN kernel operands, derived zero-copy in XRuntime::PrepareAttn from the
    // family metadata (blockTables[1..] state pages, queryStartLoc, cachedLens); full-attention
    // layer entries stay default-constructed. Indexed by model layer.
    std::vector<XTensor> _gdnConvIndices;
    std::vector<XTensor> _gdnSsmIndices;
    std::vector<XTensor> _gdnQueryStarts;
    std::vector<XTensor> _gdnConvInitial;
    std::vector<XTensor> _gdnSsmInitial;

    // Host copy from PrepareAttn, used to distinguish initialized decode queries.
    std::vector<uint32_t> _cachedLensHost;
    uint32_t _batch;
    uint32_t _maxTotalLens;
    uint32_t _maxQueryLen;  // max query length across the batch this step (1 = pure decode)
    uint32_t _tileSizeOfCachedKV;
    XTensor _attnPosition;  // [batchedTokens] or [num_dims, batchedTokens] (mrope) int64, ref:
                            // v0/1 -> _position, v2 -> attnMeta.position
    std::vector<XTensor> _attnBlockTables;  // int32 2-D, per-kv-cache; v0/1 -> compact pack,
                                            // v2 -> persistent (narrowed, pitched) view
    std::vector<XTensor> _attnSlotMapping;  // [batchedTokens] int32, per-kv-cache, ref: v0/1 ->
                                            // {_slotMapping}, v2 -> attnMeta.slotMapping
    XTensor _attnLens;        // [batch] int32, ref: v0/1 -> _lens, v2 -> attnMeta.lens
    XTensor _attnCachedLens;  // [batch] int32, ref: v0/1 -> _cachedLens, v2 -> attnMeta.cachedLens
    XTensor _attnQueryStartLoc;  // [batch] (v0/1 exclusive prefix) / [batch+1] (v2, 0-prefixed)
                                 // int32, ref: v0/1 -> _queryStartLoc, v2 -> attnMeta.queryStartLoc
    XTensor _position;           // [maxBatchedTokens] int64, internal buffer (malloc+free in dtor)
    XTensor _blockTables;  // [maxBatch * DIV_ROUND_UP(maxSeqLen, blockSize)] int32, internal buffer
    XTensor _slotMapping;  // [maxBatchedTokens] int32, internal buffer
    XTensor _cachedLens;   // [maxBatch] int32, internal buffer
    XTensor _lens;         // [maxBatch] int32, internal buffer
    XTensor _queryStartLoc;  // [maxBatch] int32, internal buffer
    // Host copy of per-request query lengths from PrepareAttn (mixed-length linear attn).
    std::vector<uint32_t> _hostLens;

    // for MoE
    XTensor _tokensPerEpGroupAllEpHost;

    // DP metadata
    uint64_t currTokens = 1;
    // maxTokensDp: initialized as max tokens across DP ranks; updated to the token number
    // to pad to for MoE when _dpSize > 1 and _nRoutedExperts > 0 and !enableMoEAllToAll
    uint64_t maxTokensDp = 1;

protected:
    int GetNodeIps(void);
    int InitHcclComm(void);
    int InitXcclComm(void);
    void FiniXcclComm(void);
    void PrepareHybridMetadata(const std::vector<uint32_t> &layerTypes,
                               const XModelAttnMeta &attnMeta);
    uint32_t _devid;
    std::vector<aclrtEvent> _inEvents;
    std::vector<aclrtNotify> _outNotifies;
    std::atomic<uint64_t> _fwdSeq{0};
    aclrtContext context = nullptr;
    bool _initOutside = false;
    bool _inited = false;
    bool _graphCommEnabled = false;
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

    void ForwardThreadMain();
    void ForwardThreadLoop();
    std::thread _fwdThread;
    std::mutex _fwdMtx;
    std::condition_variable _fwdHasWork;
    enum { FWD_IDLE, FWD_RUNNING, FWD_STOP } _fwdState = FWD_IDLE;
    std::queue<ForwardTask> _fwdQueue;
    std::array<ForwardTaskHandler, 4> _fwdHandlers;  // indexed by ForwardTask::Kind
    std::exception_ptr _fwdEx;
    bool _fwdEnabled = false;
    bool _fwdSubscribed = false;
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
