/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"
using namespace AscendC;

#define TILESIZE_16 16
#define TILESIZE_32 32
#define TILESIZE_48 48
#define TILESIZE_64 64
#define TILESIZE_128 128
#define TILESIZE_256 256
#define SEQLEN_64 64
#define SEQLEN_8K 8192
#define SEQLEN_19K 19456

#define PINGPONG_BUF_NUM 2
#define CUBE_BLOCK_SIZE 16

template <typename Dtype, typename CalcDtype>
class PrefillAttn {
public:
    __aicore__ inline PrefillAttn() {}
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR qk, GM_ADDR blockTable,
                                GM_ADDR prefixLens, GM_ADDR v, GM_ADDR out,
                                GM_ADDR promptLens, GM_ADDR prefillIndex, GM_ADDR cumPromptLen,
                                uint32_t headSize, uint32_t nHeads, uint32_t nKVHeads,
                                uint32_t blockSize, uint32_t batchSize, uint32_t maxNumBlocks)
    {
        qGmBuf.SetGlobalBuffer((__gm__ Dtype *)q);
        kGmBuf.SetGlobalBuffer((__gm__ Dtype *)k);
        vGmBuf.SetGlobalBuffer((__gm__ Dtype *)v);
        qkGmBuf.SetGlobalBuffer((__gm__ Dtype *)qk);
        outGmBuf.SetGlobalBuffer((__gm__ Dtype *)out);

        this->cumPromptLen = (__gm__ int32_t *)cumPromptLen;
        this->promptLens = (__gm__ int32_t *)promptLens;
        this->prefixLens = (__gm__ int32_t *)prefixLens;
        this->blockTable = (__gm__ int32_t *)blockTable;
        this->prefillIndex = (__gm__ int32_t *)prefillIndex;

        this->headSize = headSize;
        this->nHeads = nHeads;
        this->nKVHeads = nKVHeads;
        this->blockSize = blockSize;
        this->batchSize = batchSize;
        this->maxNumBlocks = maxNumBlocks;
    }

    /*
     * m: tokens
     * n: cachedTokens
     * k: headSize
     */
    inline __aicore__ void RunAicQK(GlobalTensor<Dtype> aGmBuf, GlobalTensor<Dtype> bGmBuf, __gm__ int32_t *mapping,
                                    GlobalTensor<Dtype> cGmBuf,
                                    int m, int padN, int headSize, int blockSize,
                                    int nHeads, int nKVHeads, int headIdx, int maskLen, int nQKVHeads)
    {
        int kK = headSize;
        int k0 = headSize;
        int k_tilefactor = k0 / CUBE_BLOCK_SIZE;
        
        int n0 = blockSize;
        int n_tilefactor = n0 / CUBE_BLOCK_SIZE;
        int n_iters = padN / n0;

        int m0 = ROUND_UP(m, CUBE_BLOCK_SIZE);
        int m_tilefactor = m0 / CUBE_BLOCK_SIZE;

        LocalTensor<Dtype> l1aBuf;
        LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
        LocalTensor<Dtype> l0aBuf;
        LocalTensor<Dtype> l0bBuf;
        LocalTensor<float> l0cBuf;

        int l1BTileBytes = n0 * k0 * sizeof(Dtype);

        uint64_t off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BTileBytes;
        }
        off = ROUND_UP(off, 1024);
        l1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        l1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off = 0;
        l0aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        l0bBuf.address_.logicPos = static_cast<uint8_t>(TPosition::B2);
        l0bBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);

        uint32_t block_memsize = nKVHeads * blockSize * headSize;
        int n_offset = 0;
        int k_offset = 0;
        int m_offset = 0;
        int pingpong_N = 0;

        CopyGmToL1Nd2Nz(l1aBuf, aGmBuf, m, k0, kK * nQKVHeads, m0);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        CopyToL0ACol(l0aBuf, l1aBuf, m_tilefactor, 0, k_tilefactor);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);

        int kv_headIdx = headIdx / (nHeads / nKVHeads);
        uint32_t head_offset_len = kv_headIdx * headSize;
        uint32_t blockTable_id = (uint32_t)(*mapping);
        CopyGmToL1Nd2Nz(l1bBuf[0], bGmBuf[blockTable_id * block_memsize + head_offset_len],
                        n0, k0, nKVHeads * headSize, n0);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1);

        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1 + (1 - pingpong_N) * 2);
        for (int nidx = 0; nidx < n_iters; ++nidx) {
            if (nidx * n0 >= maskLen) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + pingpong_N * 2);
                break;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1 + (1 - pingpong_N) * 2);
            if (nidx + 1 < n_iters) {
                uint32_t blockTable_id = (uint32_t)(*(mapping + nidx + 1));
                CopyGmToL1Nd2Nz(l1bBuf[1 - pingpong_N], bGmBuf[blockTable_id * block_memsize + head_offset_len],
                                n0, k0, nKVHeads * headSize, n0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + (1 - pingpong_N) * 2);
            }

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + pingpong_N * 2);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
            CopyToL0BCol(l0bBuf, l1bBuf[nidx & 0x1], n_tilefactor, 0, k_tilefactor);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1 + pingpong_N * 2);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID1 + pingpong_N * 2);

            if (nidx == 0)
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID1 + pingpong_N * 2);
            CalMmad(l0cBuf, l0aBuf, l0bBuf, m0, n0, k0, true, 3);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);

            CopyToGm(cGmBuf[n_offset], l0cBuf, m0, n0, m0, padN, 3);

            n_offset += n0;
            pingpong_N ^= 1;
        }

        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1 + (1 - pingpong_N) * 2);
        pipe_barrier(PIPE_ALL);
    }

    /*
     * m: tokens
     * n: headSize
     * k: cachedTokens
     */
    inline __aicore__ void RunAicSV(GlobalTensor<Dtype> aGmBuf, GlobalTensor<Dtype> bGmBuf, __gm__ int32_t *mapping,
                                    GlobalTensor<Dtype> cGmBuf,
                                    int m, int headSize, int blockSize, int kK,
                                    int nHeads, int nKVHeads, int headIdx, int maskLen)
    {
        set_atomic_none();
        set_mask_norm();

        int k0 = blockSize;
        int k_tilefactor = k0 / CUBE_BLOCK_SIZE;
        int k_iters = kK / k0;

        int padN = headSize;
        int n0 = headSize;
        int n_tilefactor = n0 / CUBE_BLOCK_SIZE;

        int m0 = ROUND_UP(m, CUBE_BLOCK_SIZE);
        int m_tilefactor = m0 / CUBE_BLOCK_SIZE;

        LocalTensor<Dtype> l1aBuf[PINGPONG_BUF_NUM];
        LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
        LocalTensor<Dtype> l0aBuf;
        LocalTensor<Dtype> l0bBuf;
        LocalTensor<float> l0cBuf;

        int l1ATileBytes = m0 * k0 * sizeof(Dtype);
        int l1BTileBytes = n0 * k0 * sizeof(Dtype);

        uint64_t off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BTileBytes;
        }
        off = ROUND_UP(off, 1024);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            l1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1ATileBytes;
        }
        off = 0;
        l0aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        l0bBuf.address_.logicPos = static_cast<uint8_t>(TPosition::B2);
        l0bBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);

        int m_offset = 0;
        int n_offset = 0;
        uint32_t block_memsize = nKVHeads * blockSize * headSize;
        int pingpong_K = 0;
        int k_offset = 0;

        CopyGmToL1Nd2Nz(l1aBuf[pingpong_K], aGmBuf, m, k0, kK, m0);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpong_K * 2);

        int kv_headIdx = headIdx / (nHeads / nKVHeads);
        uint32_t head_offset_len = kv_headIdx * headSize;

        uint32_t blockTable_id = (uint32_t)(*(mapping));
        CopyGmToL1Nd2Nz(l1bBuf[pingpong_K], bGmBuf[blockTable_id * block_memsize + head_offset_len],
                        n0, k0, nKVHeads * headSize, n0);

        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + pingpong_K * 2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + (1 - pingpong_K) * 2);
        for (int kidx = 0; kidx < k_iters; ++kidx) {
            if (kidx * k0 >= maskLen) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpong_K * 2);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + pingpong_K * 2);
                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                break;
            }

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + (1 - pingpong_K) * 2);
            if (kidx + 1 < k_iters) {
                CopyGmToL1Nd2Nz(l1aBuf[1 - pingpong_K], aGmBuf[m_offset * kK + k_offset + k0], m0, k0, kK, m0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + (1 - pingpong_K) * 2);

                uint32_t blockTable_id = (uint32_t)(*(mapping + kidx + 1));
                CopyGmToL1Nd2Nz(l1bBuf[1 - pingpong_K], bGmBuf[blockTable_id * block_memsize + head_offset_len],
                                      n0, k0, nKVHeads * headSize, n0);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + (1 - pingpong_K) * 2);
            }

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpong_K * 2);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
            CopyToL0ACol(l0aBuf, l1aBuf[pingpong_K], m_tilefactor, 0, k_tilefactor);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + pingpong_K * 2);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID1 + pingpong_K * 2);
            CopyToL0BTCol(l0bBuf, l1bBuf[pingpong_K], n_tilefactor, 0, k_tilefactor);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpong_K * 2);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID1 + pingpong_K * 2);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + pingpong_K * 2);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID1 + pingpong_K * 2);
            CalMmad(l0cBuf, l0aBuf, l0bBuf, m0, n0, k0, kidx == 0);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID1);

            if (kidx == k_iters - 1)
                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            k_offset += k0;
            pingpong_K ^= 1;
        }

        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        CopyToGm(cGmBuf, l0cBuf, m, n0, m0, padN * nHeads);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + (1 - pingpong_K) * 2);
        pipe_barrier(PIPE_ALL);
    }

    inline __aicore__ void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);
        int qkIdx = 0;
        uint32_t nQKVHeads = nHeads + nKVHeads * 2;

        for (int batch = 0; batch < batchSize; batch++) {
            uint32_t realBatch = (uint32_t)(*(prefillIndex + batch));
            uint32_t cumM = (uint32_t)(*(cumPromptLen + realBatch));
            uint32_t m = (uint32_t)(*(promptLens + realBatch));
            uint32_t padM = ROUND_UP(m, m <= SEQLEN_64 ? TILESIZE_16 : TILESIZE_128);
            uint32_t cachedTokens = (uint32_t)(*(prefixLens + realBatch));
            uint32_t padN = ROUND_UP(m + cachedTokens, blockSize);

            uint32_t m0 = TILESIZE_128;

            // 根据序列长度动态调整query的切分粒度m0，保证数据通过L2传递
            if (padN > SEQLEN_19K) {
                m0 = TILESIZE_32;
                padM = ROUND_UP(m, TILESIZE_32);
            } else if (padN > SEQLEN_8K) {
                m0 = TILESIZE_64;
                padM = ROUND_UP(m, TILESIZE_64);
            }
            if (padM <= SEQLEN_64) {
                m0 = TILESIZE_16;
            }
            uint32_t qkOffset = block_num * m0 * padN;
            uint32_t seqNum = padM / m0;
            __gm__ int32_t *curBlockTable = blockTable + maxNumBlocks * realBatch;
            int taskNum = nHeads * seqNum;
            uint32_t qOffset = cumM * headSize * nQKVHeads;
            pipe_barrier(PIPE_ALL);

            for (int i = block_idx; i < taskNum; i += block_num) {
                int seqIdx = i % seqNum;
                int headIdx = i / seqNum;
                int mActual = m0;
                int mOffset = seqIdx * m0;
                if (mOffset + mActual > m) {
                    mActual = m - mOffset;
                }
                int maskLen = mOffset + cachedTokens + mActual;

                // do 1st QK
                if (i == block_idx) {
                    RunAicQK(qGmBuf[qOffset + seqIdx * m0 * headSize * nQKVHeads + headIdx * headSize],
                             kGmBuf, curBlockTable, qkGmBuf[block_idx * m0 * padN + (qkIdx % 2 == 0 ? qkOffset : 0)],
                             mActual, padN, headSize, blockSize,
                             nHeads, nKVHeads, headIdx, maskLen, nQKVHeads);

                    uint64_t flagIdx = 0;
                    uint64_t mode = 2; // inner-group aic/aiv sync
                    uint64_t config = 1 | (mode << 4) | (flagIdx << 8);
                    ffts_cross_core_sync(PIPE_FIX, config);
                }

                // 利用类似double buffer的思想来优化，在vector计算上一部分的softmax时，cube计算QK，准备下一块数据
                int next_i = i + block_num;
                if (next_i < taskNum) {
                    int seqIdx = next_i % seqNum;
                    int headIdx = next_i / seqNum;
                    int mActual = m0;
                    int mOffset = seqIdx * m0;
                    if (mOffset + mActual > m) {
                        mActual = m - mOffset;
                    }
                    int maskLen = mOffset + cachedTokens + mActual;

                    RunAicQK(qGmBuf[qOffset + mOffset * headSize * nQKVHeads + headIdx * headSize],
                             kGmBuf, curBlockTable, qkGmBuf[block_idx * m0 * padN + ((qkIdx + 1) % 2 == 0 ? qkOffset : 0)],
                             mActual, padN, headSize, blockSize,
                             nHeads, nKVHeads, headIdx, maskLen, nQKVHeads);

                    uint64_t flagIdx = 0;
                    uint64_t mode = 2; // inner-group aic/aiv sync
                    uint64_t config = 1 | (mode << 4) | (flagIdx << 8);
                    ffts_cross_core_sync(PIPE_FIX, config);
                }

                // 等待vector的softmax计算完成后，cube计算softmax*V
                uint64_t flagIdx = 1;
                wait_flag_dev(flagIdx);

                RunAicSV(qkGmBuf[block_idx * m0 * padN + (qkIdx % 2 == 0 ? qkOffset : 0)],
                         vGmBuf, curBlockTable, outGmBuf[(cumM + mOffset) * nHeads * headSize + headIdx * headSize],
                         mActual, headSize, blockSize, padN,
                         nHeads, nKVHeads, headIdx, maskLen);
                qkIdx++;
            }
            pipe_barrier(PIPE_ALL);
        }
    }

#if __DAV_C220_VEC__
    inline __aicore__ void RunAivSoftmax(__gm__ Dtype *qk, uint16_t padN, uint16_t tokens,
                                         uint32_t cachedTokens, uint32_t curSeq, uint32_t curPromptLen)
    {
        CalcDtype min;
        if constexpr (std::is_same<CalcDtype, half>::value) {
            min = half(-65504);
        } else {
            min = -3.4028235e+38;
        }

        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        uint64_t off = 0;
        __ubuf__ Dtype *in = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += padN * sizeof(Dtype);
        __ubuf__ Dtype *out = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)off);
        off += padN * sizeof(Dtype);
        __ubuf__ CalcDtype *cal = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t)off);
        off += padN * sizeof(CalcDtype);
        __ubuf__ CalcDtype *temp = reinterpret_cast<__ubuf__ CalcDtype *>((uintptr_t)off);
        __ubuf__ CalcDtype *ptr;

        int calPad = 256 / sizeof(CalcDtype);
        int pad = 256 / sizeof(Dtype);

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        for (int idx = 0; idx < tokens; ++idx) {
            // softmax计算跳过padding的部分
            if (curSeq + idx >= curPromptLen) {
                break;
            }

            int actualCachedTokens = 1 + cachedTokens + idx; // 每一行开始mask的位置
            int padCachedTokens = ROUND_UP(actualCachedTokens, calPad);
            int repeat = padCachedTokens / calPad;

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            copy_gm_to_ubuf(in, qk + idx * padN, 0, 1,
                            DIV_ROUND_UP(actualCachedTokens * sizeof(Dtype), 32), 0, 0);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            if constexpr (std::is_same<CalcDtype, half>::value) {
                ptr = in;
            } else {
                vconv_bf162f32((__ubuf__ float *)cal, (__ubuf__ Dtype *)in, repeat, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
                ptr = cal;
            }

            // dup calPad
            if (actualCachedTokens % calPad != 0) {
                SetMaskFromHighBit(calPad, calPad - actualCachedTokens % calPad);
                vector_dup(ptr + ROUND_DOWN(actualCachedTokens, calPad), min, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }

            // max
            ReduceMax(temp, ptr, actualCachedTokens);

            // broadcast一个max标量为一个block大小的向量，避免使用scalar运算
            if constexpr (std::is_same<CalcDtype, half>::value) {
                vbrcb((__ubuf__ uint16_t*)temp, (__ubuf__ uint16_t*)temp, 0, 0, 1);
            } else {
                vbrcb((__ubuf__ uint32_t*)temp, (__ubuf__ uint32_t*)temp, 0, 0, 1);
            }
            pipe_barrier(PIPE_V);

            // QK - max
            vsub(cal, ptr, temp, repeat, 1, 1, 0, 8, 8, 0);
            pipe_barrier(PIPE_V);
            if constexpr (std::is_same<CalcDtype, half>::value) {
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
            }

            // EXP = exp(QK-max)
            vexp(cal, cal, repeat, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);

            // s = Reduce_sum(EXP)
            ReduceSum(temp, cal, actualCachedTokens);

            // broadcast一个s = Reduce_sum(EXP)标量为一个block大小的向量，避免使用scalar运算
            if constexpr (std::is_same<CalcDtype, half>::value) {
                vbrcb((__ubuf__ uint16_t*)temp, (__ubuf__ uint16_t*)temp, 0, 0, 1);
            } else {
                vbrcb((__ubuf__ uint32_t*)temp, (__ubuf__ uint32_t*)temp, 0, 0, 1);
            }
            pipe_barrier(PIPE_V);

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            if constexpr (std::is_same<CalcDtype, half>::value) {
                vdiv(out, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
            } else {
                vdiv(cal, cal, temp, repeat, 1, 1, 0, 8, 8, 0);
                pipe_barrier(PIPE_V);
                vconv_f322bf16r(out, cal, repeat, 1, 1, 4, 8);
                pipe_barrier(PIPE_V);
            }

            if (padN > actualCachedTokens) {
                if (actualCachedTokens % pad != 0) {
                    SetMaskFromHighBit(pad, pad - actualCachedTokens % pad);
                    vector_dup(out + ROUND_DOWN(actualCachedTokens, pad), Dtype(0), 1, 1, 1, 8, 0);
                    pipe_barrier(PIPE_V);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                int last = ROUND_UP(actualCachedTokens, pad);
                if (padN > last) {
                    vector_dup(out + last, Dtype(0), (padN - last) / pad, 1, 1, 8, 0);
                }
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            copy_ubuf_to_gm(qk + idx * padN, out, 0, 1, padN / 16, 0, 0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        pipe_barrier(PIPE_ALL);
    }
#endif

    inline __aicore__ void RunAiv()
    {
        set_atomic_none();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        int qkIdx = 0;

        for (int batch = 0; batch < batchSize; batch++) {
            uint32_t realBatch = (uint32_t)(*(prefillIndex + batch));
            uint32_t m = (uint32_t)(*(promptLens + realBatch));
            uint32_t padM = ROUND_UP(m, m <= SEQLEN_64 ? TILESIZE_16 : TILESIZE_128);
            uint32_t cachedTokens = (uint32_t)(*(prefixLens + realBatch));
            uint32_t padN = ROUND_UP(m + cachedTokens, blockSize);

            uint32_t m0 = TILESIZE_128;

            // 根据序列长度动态调整query的切分粒度m0，保证数据通过L2传递
            if (padN > SEQLEN_19K) {
                m0 = TILESIZE_32;
                padM = ROUND_UP(m, TILESIZE_32);
            } else if (padN > SEQLEN_8K) {
                m0 = TILESIZE_64;
                padM = ROUND_UP(m, TILESIZE_64);
            }
            if (padM <= SEQLEN_64) {
                m0 = TILESIZE_16;
            }
            uint32_t qkOffset = block_num * m0 * padN;

            int seqNum = padM / m0;
            int taskNum = nHeads * seqNum;
            pipe_barrier(PIPE_ALL);

            for (int i = block_idx; i < taskNum; i += block_num) {
                int seqIdx = i % seqNum;
                int headIdx = i / seqNum;

                uint64_t flagIdx = 0;
                wait_flag_dev(flagIdx);

                uint32_t subIdx = get_subblockid();
                int newCachedTokens = seqIdx * m0 + subIdx * m0 / 2 + cachedTokens;
                uint32_t curSeq = seqIdx * m0 + subIdx * m0 / 2;
                __gm__ Dtype *qk = ((__gm__ Dtype *)qkGmBuf.GetPhyAddr()) + block_idx * m0 * padN + subIdx * m0 / 2 * padN;
                if (qkIdx % 2 == 0) {
                    qk = qk + qkOffset;
                }

                RunAivSoftmax(qk, padN, m0 / 2, newCachedTokens, curSeq, m);
                flagIdx = 1;
                uint64_t mode = 2; // inner-group aic/aiv sync
                uint64_t config = 1 | (mode << 4) | (flagIdx << 8);
                ffts_cross_core_sync(PIPE_MTE3, config);
                qkIdx++;
            }
            pipe_barrier(PIPE_ALL);
        }
    }

    inline __aicore__ void Run()
    {
#ifdef __DAV_C220_CUBE__
        RunAic();
#elif __DAV_C220_VEC__
        RunAiv();
#endif
    }

private:
    GlobalTensor<Dtype> qGmBuf;
    GlobalTensor<Dtype> kGmBuf;
    GlobalTensor<Dtype> vGmBuf;
    GlobalTensor<Dtype> qkGmBuf;
    GlobalTensor<Dtype> outGmBuf;
    __gm__ int32_t *cumPromptLen;
    __gm__ int32_t *promptLens;
    __gm__ int32_t *prefixLens;
    __gm__ int32_t *blockTable;
    __gm__ int32_t *prefillIndex;
    uint32_t headSize;
    uint32_t nHeads;
    uint32_t nKVHeads;
    uint32_t blockSize;
    uint32_t batchSize;
    uint32_t maxNumBlocks;
};

#define PREFILL_ATTN_FUNC_DEFINE(dtype, calcDtype) \
extern "C" __global__ __aicore__ void prefill_att_##dtype( \
    GM_ADDR q, GM_ADDR k, GM_ADDR qk, GM_ADDR block_table, \
    GM_ADDR prefix_lens, \
    GM_ADDR v, GM_ADDR out, GM_ADDR prompt_lens, \
    GM_ADDR __restrict__ prefill_index, GM_ADDR __restrict__ cum_prompt_len, \
    uint32_t head_size, uint32_t num_heads, uint32_t num_kv_heads, uint32_t block_size, uint32_t batchSize, uint32_t max_num_blocks) \
{ \
    PrefillAttn<dtype, calcDtype> op; \
    op.Init(q, k, qk, block_table, \
            prefix_lens, v, out, \
            prompt_lens, prefill_index, cum_prompt_len, \
            head_size, num_heads, num_kv_heads, \
            block_size, batchSize, max_num_blocks); \
    op.Run(); \
}

PREFILL_ATTN_FUNC_DEFINE(float16_t, float16_t);
PREFILL_ATTN_FUNC_DEFINE(bfloat16_t, float);
