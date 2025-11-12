#include "kernel_operator.h"
#include "kernel_macro.h"

#define TILESIZE_16 16
#define TILESIZE_32 32
#define TILESIZE_48 48
#define TILESIZE_64 64
#define TILESIZE_128 128
#define TILESIZE_256 256
#define SEQLEN_64 64
#define SEQLEN_8K 8192
#define SEQLEN_19K 19456

#ifdef __DAV_C220_CUBE__
#define PINGPONG_BUF_NUM 2
#define CUBE_SIZE 256
#define CUBE_BLOCK_SIZE 16

inline __aicore__ void copy_to_l12d(__cbuf__ half *L1, __gm__ half *gm, int row, int col, int height, int width, int step,
    int num_heads)
{
    // Copy L1
    step = step * num_heads;
    auto offset = (row * step + col);
    copy_gm_to_cbuf_multi_nd2nz_b16(L1, gm + offset, 0, 1 /* nd2nzParams.ndNum */, height /* nd2nzParams.nValue */,
        width /* nd2nzParams.dValue */, 0 /* nd2nzParams.srcNdMatrixStride */, step /* nd2nzParams.srcDValue */,
        height /* nd2nzParams.dstNzC0Stride */, 1 /* nd2nzParams.dstNzNStride */,
        0 /* nd2nzParams.dstNzMatrixStride */);
}

// num_row可以是任意值，不用padding
inline __aicore__ void copy_to_l12d(__cbuf__ __fp16 *L1, __gm__ __fp16 *gm, int row, int col, int height, int width, int step, int num_heads, int num_row)
{
    // Copy L1
    step = step * num_heads;
    auto offset = (row * step + col);
    copy_gm_to_cbuf_multi_nd2nz_b16(L1, gm + offset, 0, 1 /* nd2nzParams.ndNum */, num_row /* nd2nzParams.nValue */,
        width /* nd2nzParams.dValue */, 0 /* nd2nzParams.srcNdMatrixStride */, step /* nd2nzParams.srcDValue */,
        height /* nd2nzParams.dstNzC0Stride */, 1 /* nd2nzParams.dstNzNStride */,
        0 /* nd2nzParams.dstNzMatrixStride */);
}

inline __aicore__ void copy_to_l0a(__ca__ half *l0, __cbuf__ half *cbuf, int m_factor, int k_factor)
{
    for (int k = 0; k < m_factor; ++k) {
        int dst_offset = CUBE_SIZE * k_factor;
        int src_offset = CUBE_SIZE;
        load_cbuf_to_ca(l0 + k * dst_offset, cbuf + k * src_offset, 0 /* loadDataParams.startIndex */,
            k_factor /* loadDataParams.repeatTimes */, m_factor /* loadDataParams.srcStride */, 0 /* loadDataParams.dstGap */, 0 /* loadDataParams.sid */,
            0 /* transpose */, inc);
    }
}

inline __aicore__ void copy_to_l0b(__cb__ half *l0, __cbuf__ half *cbuf, int n_factor, int k_factor)
{
    for (int k = 0; k < k_factor; ++k) {
        int dst_offset = CUBE_SIZE * n_factor;
        int src_offset = CUBE_SIZE * n_factor;
        load_cbuf_to_cb(l0 + k * dst_offset, cbuf + k * src_offset, 0 /* loadDataParams.startIndex */,
            n_factor /* loadDataParams.repeatTimes */, 1 /* loadDataParams.srcStride */, 0 /* loadDataParams.dstGap */, 0 /* loadDataParams.sid */,
            0 /* transpose */, inc);
    }
}

inline __aicore__ void copy_to_l0b_transpose(__cb__ half *l0, __cbuf__ half *cbuf, int n_factor, int k_factor)
{
    for (int k = 0; k < k_factor; ++k) {
        int dst_offset = n_factor * CUBE_SIZE;
        constexpr int src_offset = CUBE_SIZE;
        load_cbuf_to_cb(l0 + k * dst_offset, cbuf + k * src_offset, 0 /* loadDataParams.startIndex */,
            n_factor /* loadDataParams.repeatTimes */, k_factor /* loadDataParams.srcStride */, 0 /* loadDataParams.dstGap */, 0 /* loadDataParams.sid */,
            1 /* transpose */, inc);
    }
}
inline __aicore__ void mmad(__cc__ float *cc, __ca__ half *ca, __cb__ half *cb, int m, int n, int k, bool init_val,
    uint8_t unit_flag = 0)
{
    mad(cc, ca, cb, m, k, n, unit_flag /* unit flag */, false /* kDirection Align */, 0, init_val);
}

inline __aicore__ void copy_to_gm(__gm__ half *z, __cc__ float *cc, int m_offset, int n_offset, int m, int n, int kN, uint8_t unit_flag = 0)
{
    uint64_t config = 0x1;
    set_nd_para(config);
    int c_offset = m_offset * kN + n_offset;
    copy_matrix_cc_to_gm(z + c_offset, cc, 0 /* fixpipeInfo.sid */, n, m, kN /* fixpipeInfo.dstStride */,
        m /* fixpipeInfo.srcStride */, unit_flag /* fixpipeInfo.unit_flag */, QuantMode_t::F322F16,
        0 /* static_cast<uint8_t>(fixpipeInfo.reluEn) */, 0 /* fixpipeInfo.channelSplit */,
        1 /* fixpipeInfo.nz2ndEn */);
}

inline __aicore__ void copy_to_gm(__gm__ half *z, __cc__ float *cc, int m_offset, int n_offset, int m, int n, int kN, int m_size, int num_heads)
{
    uint64_t config = 0x1;
    set_nd_para(config);
    kN = kN * num_heads;
    int c_offset = m_offset * kN + n_offset;
    copy_matrix_cc_to_gm(z + c_offset, cc, 0 /* fixpipeInfo.sid */, n, m_size, kN /* fixpipeInfo.dstStride */,
        m /* fixpipeInfo.srcStride */, 0 /* fixpipeInfo.unit_flag */, QuantMode_t::F322F16,
        0 /* static_cast<uint8_t>(fixpipeInfo.reluEn) */, 0 /* fixpipeInfo.channelSplit */,
        1 /* fixpipeInfo.nz2ndEn */);
}

inline __aicore__ void att_QK(__gm__ half * x, __gm__ uint8_t * k, __gm__ int32_t * mapping, __gm__ half * z, int kM, int real_m, int kN,
    int head_size, int block_size, int num_heads, int num_kv_heads, int head_idx, int offsetM, int offsetM_without_cachedtokens, int num_qkv_heads)
{
    int kK = head_size;
    constexpr int k_tilesize = TILESIZE_128; // kTileSize必须小于等于headSize
    constexpr int k_tilefactor = k_tilesize / CUBE_BLOCK_SIZE;
    int k_iters = kK / k_tilesize;

    int n_tilesize = block_size;
    int n_tilefactor = n_tilesize / CUBE_BLOCK_SIZE;
    int n_iters = kN / n_tilesize;

    int m_tilesize = block_size;
    if (m_tilesize == TILESIZE_256 && n_tilesize == TILESIZE_256) {
        m_tilesize = TILESIZE_128;
    }
    if (kM <= TILESIZE_16) {
        m_tilesize = TILESIZE_16;
    } else if (kM <= TILESIZE_32) {
        m_tilesize = TILESIZE_32;
    } else if (kM <= TILESIZE_48) {
        m_tilesize = TILESIZE_48;
    } else if (kM <= TILESIZE_64) {
        m_tilesize = TILESIZE_64;
    }
    int m_tilefactor = m_tilesize / CUBE_BLOCK_SIZE;
    int m_iters = kM / m_tilesize;

    int num_row = m_tilesize;
    if (m_tilesize + offsetM_without_cachedtokens > real_m) {
        num_row = real_m - offsetM_without_cachedtokens;
    }

    __cb__ half *cb_addr = reinterpret_cast<__cb__ half *>((uintptr_t)0);
    __ca__ half *ca_addr = reinterpret_cast<__ca__ half *>((uintptr_t)0);
    __cc__ float *cc_addr = reinterpret_cast<__cc__ float *>((uintptr_t)0);

    int cb_tilesize = n_tilesize * k_tilesize;
    int cb_tilebytes = cb_tilesize * sizeof(half);
    __cbuf__ half *B_cbuf_addr0 = reinterpret_cast<__cbuf__ half *>((uintptr_t)0);
    __cbuf__ half *B_cbuf_addr1 = reinterpret_cast<__cbuf__ half *>((uintptr_t)cb_tilebytes);
    __cbuf__ half *B_cbuf_addr[PINGPONG_BUF_NUM] = {B_cbuf_addr0, B_cbuf_addr1};

    int ca_tilestart = ROUND_UP(cb_tilebytes * PINGPONG_BUF_NUM, 1024); // 最小访问粒度 512B/128B, 选择1024B对齐
    int ca_tilesize = m_tilesize * k_tilesize;
    int ca_tilebytes = ca_tilesize * sizeof(half);
    __cbuf__ half *A_cbuf_addr0 = reinterpret_cast<__cbuf__ half *>((uintptr_t)ca_tilestart);
    __cbuf__ half *A_cbuf_addr1 = reinterpret_cast<__cbuf__ half *>((uintptr_t)(ca_tilestart + ca_tilebytes));
    __cbuf__ half *A_cbuf_addr[PINGPONG_BUF_NUM] = {A_cbuf_addr0, A_cbuf_addr1};

    uint32_t block_memsize = num_kv_heads * block_size * head_size;
    bool init_val = true;
    int n_start = 0;
    int m_start = 0;
    int n_stride = n_tilesize;
    int m_stride = m_tilesize;
    int k_offset = 0;
    int m_offset = m_start;
    int pingpong_M = 0;
    int pingpong_N = 0;
    __gm__ half *x_gm_addr = ((__gm__ half *)x) + head_idx * kK;
    __gm__ half *z_gm_addr = ((__gm__ half *)z);

    copy_to_l12d(A_cbuf_addr[pingpong_M], x_gm_addr, 0, k_offset, m_tilesize, k_tilesize, kK, num_qkv_heads, num_row);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + pingpong_M * 2);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + (1 - pingpong_M) * 2);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    for (int midx = 0; midx < m_iters; ++midx) {
        int n_offset = n_start;
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + (1 - pingpong_M) * 2);
        if (midx + 1 < m_iters) {
            copy_to_l12d(A_cbuf_addr[1 - pingpong_M], x_gm_addr, m_offset + m_tilesize, k_offset, m_tilesize, k_tilesize, kK, num_qkv_heads);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + (1 - pingpong_M) * 2);
        }

        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + pingpong_M * 2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        copy_to_l0a(ca_addr, A_cbuf_addr[pingpong_M], m_tilefactor, k_tilefactor);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + pingpong_M * 2);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0 + pingpong_M * 2);

        int kv_head_idx = head_idx / (num_heads / num_kv_heads);
        uint32_t head_offset_len = kv_head_idx * block_size * head_size;
        uint32_t block_table_id = (uint32_t)(*((__gm__ int32_t *)mapping));
        __gm__ half *K_gm_addr = ((__gm__ half *)k) + block_table_id * block_memsize + head_offset_len;

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1 + pingpong_N * 2);
        copy_to_l12d(B_cbuf_addr[pingpong_N], K_gm_addr, 0, 0, block_size, head_size, head_size, 1);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + pingpong_N * 2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1 + (1 - pingpong_N) * 2);
        for (int nidx = 0; nidx < n_iters; ++nidx) {
            if (nidx * n_tilesize >= (midx + 1) * m_tilesize + offsetM) {
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + pingpong_N * 2);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                break;
            }
        
            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1 + (1 - pingpong_N) * 2);
            if (nidx + 1 < n_iters) {
                uint32_t block_table_id = (uint32_t)(*((__gm__ int32_t *)mapping + nidx + 1));
                __gm__ half *K_gm_addr = ((__gm__ half *)k) + block_table_id * block_memsize + head_offset_len;
                copy_to_l12d(B_cbuf_addr[1 - pingpong_N], K_gm_addr, 0, 0, block_size, head_size, head_size, 1);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + (1 - pingpong_N) * 2);
            }

            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + pingpong_N * 2);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            copy_to_l0b(cb_addr, B_cbuf_addr[nidx & 0x1], n_tilefactor, k_tilefactor);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1 + pingpong_N * 2);
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1 + pingpong_N * 2);

            if (nidx == 0)
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0 + pingpong_M * 2); // wait for L0A
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1 + pingpong_N * 2); // wait for L0B

            mmad(cc_addr, ca_addr, cb_addr, m_tilesize, n_tilesize, k_tilesize, init_val, 3);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (nidx == n_iters - 1)
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

            copy_to_gm(z_gm_addr, cc_addr, m_offset, n_offset, m_tilesize, n_tilesize, kN, 3);

            n_offset += n_stride;
            pingpong_N ^= 1;
        }

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        m_offset += m_stride;
        pingpong_M ^= 1;
    }
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1 + (1 - pingpong_N) * 2);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + (1 - pingpong_M) * 2);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    pipe_barrier(PIPE_ALL);
}

inline __aicore__ void att_SV(__gm__ half *x, __gm__ uint8_t * v, __gm__ int32_t *mapping, __gm__ half *z, int kM, int ori_kM, int head_size, int block_size, int kK,
    int num_heads, int num_kv_heads, int head_idx, int M, int realK, int offsetM)
{
    set_atomic_none();
    set_mask_norm();

    if (offsetM >= M) // att的M切得粒度可能小于外部M切分粒度，这里保证进来的数据M维度还在有效范围内
        return;

    int k_tilesize = block_size; // 256
    int k_tilefactor = k_tilesize / CUBE_BLOCK_SIZE;
    int k_iters = kK / k_tilesize;

    int kN = head_size;
    constexpr int n_tilesize = TILESIZE_128; // 这里nTileSize必须小于等于headSize
    constexpr int n_tilefactor = n_tilesize / CUBE_BLOCK_SIZE;
    int n_iters = kN / n_tilesize;

    int m_tilesize = block_size;
    if (m_tilesize == TILESIZE_256 && k_tilesize == TILESIZE_256) {
        m_tilesize = TILESIZE_128;
    }
    if (kM <= TILESIZE_16) {
        m_tilesize = TILESIZE_16;
    } else if (kM <= TILESIZE_32) {
        m_tilesize = TILESIZE_32;
    } else if (kM <= TILESIZE_48) {
        m_tilesize = TILESIZE_48;
    } else if (kM <= TILESIZE_64) {
        m_tilesize = TILESIZE_64;
    }
    int m_tilefactor = m_tilesize / CUBE_BLOCK_SIZE;
    int m_iters = kM / m_tilesize;

    __cb__ half *cb_addr = reinterpret_cast<__cb__ half *>((uintptr_t)0);
    __ca__ half *ca_addr = reinterpret_cast<__ca__ half *>((uintptr_t)0);
    __cc__ float *cc_addr = reinterpret_cast<__cc__ float *>((uintptr_t)0);

    int cb_tilesize = n_tilesize * k_tilesize;
    int cb_tilebytes = cb_tilesize * sizeof(half);
    __cbuf__ half *B_cbuf_addr0 = reinterpret_cast<__cbuf__ half *>((uintptr_t)0);
    __cbuf__ half *B_cbuf_addr1 = reinterpret_cast<__cbuf__ half *>((uintptr_t)cb_tilebytes);
    __cbuf__ half *B_cbuf_addr[PINGPONG_BUF_NUM] = {B_cbuf_addr0, B_cbuf_addr1};

    int ca_tilestart = ROUND_UP(cb_tilebytes * PINGPONG_BUF_NUM, 1024);
    int ca_tilesize = m_tilesize * k_tilesize;
    int ca_tilebytes = ca_tilesize * sizeof(half);
    __cbuf__ half *A_cbuf_addr0 = reinterpret_cast<__cbuf__ half *>((uintptr_t)ca_tilestart);
    __cbuf__ half *A_cbuf_addr1 = reinterpret_cast<__cbuf__ half *>((uintptr_t)(ca_tilestart + ca_tilebytes));
    __cbuf__ half *A_cbuf_addr[PINGPONG_BUF_NUM] = {A_cbuf_addr0, A_cbuf_addr1};

    int n_start = 0;
    int m_start = 0;
    int n_stride = n_tilesize;
    int m_stride = m_tilesize;
    int m_offset = m_start;
    uint32_t block_memsize = num_kv_heads * block_size * head_size;
    __gm__ half *x_gm_addr = ((__gm__ half *)x);
    __gm__ half *z_gm_addr = ((__gm__ half *)z);
    for (int midx = 0; midx < m_iters; ++midx) {
        int n_offset = n_start;
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        int pingpong_K = 0;
        for (int nidx = 0; nidx < n_iters; ++nidx) {
            int k_offset = 0;
            bool init_val = true;

            wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + pingpong_K * 2);
            copy_to_l12d(A_cbuf_addr[pingpong_K], x_gm_addr, m_offset, k_offset, m_tilesize, k_tilesize, kK, 1);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + pingpong_K * 2);

            int kv_head_idx = head_idx / (num_heads / num_kv_heads);
            uint32_t head_offset_len = kv_head_idx * block_size * head_size;

            uint32_t block_table_id = (uint32_t)(*((__gm__ int32_t *)mapping));
            __gm__ half *K_gm_addr = ((__gm__ half *)v) + block_table_id * block_memsize + head_offset_len;
            copy_to_l12d(B_cbuf_addr[pingpong_K], K_gm_addr, 0, 0, block_size, head_size, head_size, 1);

            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + pingpong_K * 2);
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + (1 - pingpong_K) * 2);
            for (int kidx = 0; kidx < k_iters; ++kidx) {
                if (kidx * k_tilesize >= realK) {
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + pingpong_K * 2);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + pingpong_K * 2);
                    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                    break;
                }
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + (1 - pingpong_K) * 2);
                if (kidx + 1 < k_iters) {
                    copy_to_l12d(A_cbuf_addr[1 - pingpong_K], x_gm_addr, m_offset, k_offset + k_tilesize, m_tilesize,
                        k_tilesize, kK, 1);
                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + (1 - pingpong_K) * 2);

                    uint32_t block_table_id = (uint32_t)(*((__gm__ int32_t *)mapping + kidx + 1));
                    __gm__ half *K_gm_addr = ((__gm__ half *)v) + block_table_id * block_memsize + head_offset_len;
                    copy_to_l12d(B_cbuf_addr[1 - pingpong_K], K_gm_addr, 0, 0, block_size, head_size, head_size, 1);

                    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + (1 - pingpong_K) * 2);
                }
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0 + pingpong_K * 2);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                copy_to_l0a(ca_addr, A_cbuf_addr[pingpong_K], m_tilefactor, k_tilefactor);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0 + pingpong_K * 2);

                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1 + pingpong_K * 2);
                copy_to_l0b_transpose(cb_addr, B_cbuf_addr[pingpong_K], n_tilefactor, k_tilefactor);
                set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + pingpong_K * 2);
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1 + pingpong_K * 2);

                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0 + pingpong_K * 2); // wait for L0A
                if (kidx == k_iters - 1)
                    wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1 + pingpong_K * 2); // wait for L0B
                mmad(cc_addr, ca_addr, cb_addr, m_tilesize, n_tilesize, k_tilesize, init_val);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

                if (kidx == k_iters - 1)
                    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
                init_val = false;
                k_offset += k_tilesize;
                pingpong_K ^= 1;
            }

            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            if (m_offset + m_tilesize + offsetM <= M) {
                copy_to_gm(z_gm_addr, cc_addr, m_offset, n_offset, m_tilesize, n_tilesize, kN, m_tilesize, num_heads);
            } else {
                copy_to_gm(z_gm_addr, cc_addr, m_offset, n_offset, m_tilesize, n_tilesize, kN, M - m_offset - offsetM, num_heads);
            }
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            n_offset += n_stride;
        }
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0 + (1 - pingpong_K) * 2);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        m_offset += m_stride;
    }
    pipe_barrier(PIPE_ALL);
}

inline __aicore__ void prefill_att_mix_aic(
    __gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* qk, __gm__ uint8_t* block_table,
    __gm__ uint8_t* padding_N, __gm__ uint8_t* prefix_lens,
    __gm__ uint8_t* v, __gm__ uint8_t* out, __gm__ uint8_t* prompt_lens,
    __gm__ uint8_t* __restrict__ cum_prompt_len,
    uint32_t head_size, uint32_t num_heads, uint32_t num_kv_heads, uint32_t block_size, uint32_t batchSize, uint32_t max_num_blocks)
{
    set_padding(0);
    set_atomic_none();
    set_nd_para((uint64_t)1);

    uint32_t num_qkv_heads = num_heads + num_kv_heads * 2;
    pipe_barrier(PIPE_ALL);

    int qk_idx = 0;
    for (int batch_idx = 0; batch_idx < batchSize; batch_idx++) {
        pipe_barrier(PIPE_ALL);
        uint32_t cum_M = (uint32_t)(*((__gm__ int32_t *)cum_prompt_len + batch_idx));
        uint32_t M = (uint32_t)(*((__gm__ int32_t *)prompt_lens + batch_idx));

        uint32_t kM = ROUND_UP(M, M <= SEQLEN_64 ? TILESIZE_16 : TILESIZE_128);
        uint32_t kN = (uint32_t)(*((__gm__ int32_t *)padding_N + batch_idx));

        uint32_t pM = TILESIZE_128;
        // 根据序列长度动态调整query的切分粒度pM，保证数据通过L2传递
        if (kN > SEQLEN_19K) {
            pM = TILESIZE_32;
            kM = ROUND_UP(M, TILESIZE_32);
        } else if (kN > SEQLEN_8K) {
            pM = TILESIZE_64;
            kM = ROUND_UP(M, TILESIZE_64);
        }
        if (kM <= SEQLEN_64) {
            pM = TILESIZE_16;  // 避免kM很小的情况（小于128），切的粒度是16，也保证可以整除
        }
        // 偏移根据pM动态生成，不用外部传递了，外部可以申请一个大的buffer，但是实际使用的保证小于L2
        uint32_t qk_offset = block_num * pM * kN;
        uint32_t seq_num = kM / pM;
        __gm__ int32_t *blockTable_addr = ((__gm__ int32_t *)block_table) + max_num_blocks * batch_idx;
        uint32_t cached_tokens = (uint32_t)(*((__gm__ int32_t *)prefix_lens + batch_idx));
        uint32_t kK = (uint32_t)(*((__gm__ int32_t *)padding_N + batch_idx));
        int total_task_num = num_heads * seq_num;
        uint32_t query_addr_offset = cum_M * head_size * num_qkv_heads;
        pipe_barrier(PIPE_ALL);
        // 第一轮，事先准备好QK，只计算一部分QK。cube计算完该部分QK后，通过核间同步通知vector计算该部分的softmax
        for (int i = block_idx; i < total_task_num && i < block_num; i += block_num) {
            int seq_idx = i % seq_num;
            int head_idx = i / seq_num;

            __gm__ half *q_gm_addr = ((__gm__ half *)q) + query_addr_offset + seq_idx * pM * head_size * num_qkv_heads;
            // do 1st QK
            __gm__ half *qk_gm_addr = ((__gm__ half *)qk) + block_idx * pM * kN;
            if (qk_idx % 2 == 0) {
                qk_gm_addr += qk_offset;
            }
            att_QK(q_gm_addr, k, blockTable_addr, qk_gm_addr, pM, M, kN, head_size, block_size, num_heads, num_kv_heads, head_idx, seq_idx * pM + cached_tokens, seq_idx * pM, num_qkv_heads);
            
            uint64_t flag_id = 0;
            uint64_t mode = 2; // inner-group aic/aiv sync
            uint64_t config = 1 | (mode << 4) | (flag_id << 8);
            ffts_cross_core_sync(PIPE_FIX, config);
        }

        for (int i = block_idx; i < total_task_num; i += block_num) {
            // 利用类似double buffer的思想来优化，在vector计算上一部分的softmax时，cube计算QK，准备下一块数据
            int next_i = i + block_num;
            if (next_i < total_task_num) {
                int seq_idx = next_i % seq_num;
                int head_idx = next_i / seq_num;

                __gm__ half *q_gm_addr = ((__gm__ half *)q) + query_addr_offset + seq_idx * pM * head_size * num_qkv_heads;
                __gm__ half *qk_gm_addr = ((__gm__ half *)qk) + block_idx * pM * kN;
                if ((qk_idx + 1) % 2 == 0) {
                    qk_gm_addr = qk_gm_addr + qk_offset;
                }
                att_QK(q_gm_addr, k, blockTable_addr, qk_gm_addr, pM, M, kN, head_size, block_size, num_heads, num_kv_heads, head_idx, seq_idx * pM + cached_tokens, seq_idx * pM, num_qkv_heads);

                uint64_t flag_id = 0;
                uint64_t mode = 2; // inner-group aic/aiv sync
                uint64_t config = 1 | (mode << 4) | (flag_id << 8);
                ffts_cross_core_sync(PIPE_FIX, config);
            }

            // 正常的处理逻辑
            int seq_idx = i % seq_num;
            int head_idx = i / seq_num;

            // 等待vector的softmax计算完成后，cube计算softmax*V
            uint64_t flag_id = 1;
            wait_flag_dev(flag_id);

            __gm__ half *out_gm_addr = ((__gm__ half *)out) + (cum_M + seq_idx * pM) * num_heads * head_size + head_idx * head_size;
            __gm__ half *qk_gm_addr = ((__gm__ half *)qk) + block_idx * pM * kN;
            if (qk_idx % 2 == 0) {
                qk_gm_addr = qk_gm_addr + qk_offset;
            }
            att_SV(qk_gm_addr, v, blockTable_addr, out_gm_addr, pM, kM, head_size, block_size, kK, num_heads, num_kv_heads, head_idx, M, seq_idx * pM + cached_tokens + pM, seq_idx * pM);
            qk_idx++;
        }
        pipe_barrier(PIPE_ALL);
    }

}

#elif __DAV_C220_VEC__
#define NUMELEMENT_OF_TEMPBUF 256
#define ONE_IN_HIGHBIT (1L << 63)

inline __aicore__ void __set_mask(int32_t len)
{
    uint64_t mask = 0;
    uint64_t one = 1;
    uint64_t temp = len % 64;
    for (int64_t i = 0; i < temp; i++) {
        mask |= one << i;
    }

    if (len == VECTOR_MAX_NUM_OF_FP16) {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    } else if (len >= (VECTOR_MAX_NUM_OF_FP16 / 2)) {
        set_vector_mask(mask, (uint64_t)-1);
    } else {
        set_vector_mask(0x0, mask);
    }
}

inline __aicore__ void __set_mask_from_highbit(int32_t len)
{
    uint64_t mask = 0;
    uint64_t temp = len % 64;
    for (int64_t i = 0; i < temp; i++) {
        mask |= ONE_IN_HIGHBIT >> i;
    }

    if (len == VECTOR_MAX_NUM_OF_FP16) {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    } else if (len >= (VECTOR_MAX_NUM_OF_FP16 / 2)) {
        set_vector_mask((uint64_t)-1, mask);
    } else {
        set_vector_mask(mask, 0x0);
    }
}

// 最多处理32000长度的片段
inline __aicore__ void att_softmax_vector_kernel(__gm__ half *attn_weight_in, __gm__ half *attn_weight_out,
                                                 uint16_t num_tokens, uint16_t len_chunk, uint16_t new_tokens,
                                                 uint32_t cached_tokens, uint32_t cur_seq, uint32_t cur_prompt_len)
{
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    int weight_size = len_chunk * 2;
    int tempInUbAddrSize = NUMELEMENT_OF_TEMPBUF * sizeof(half);
    auto *weight_in_ub_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t)0);
    auto *weight_out_ub_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t)(weight_size * 1));
    auto *temp_in_ub_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t)weight_size * 2); // 256个half，该buf决定最多处理256*128 = 32k的片段
    auto *cal_weight_in_ub_addr = reinterpret_cast<__ubuf__ half *>((uintptr_t)weight_size * 2 + tempInUbAddrSize);

    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    for (int idx = 0; idx < new_tokens; ++idx) {
        // softmax计算跳过padding的部分
        if(cur_seq + idx >= cur_prompt_len) {
            break;
        }
        int effective_tokens = 1 + cached_tokens + idx; // 每一行开始mask的位置
        int padding_effective_tokens = ROUND_UP(effective_tokens, VECTOR_MAX_NUM_OF_FP16);

        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);

        copy_gm_to_ubuf(weight_in_ub_addr, ((__gm__ half *)attn_weight_in) + idx * num_tokens, 0, 1, padding_effective_tokens / 16, 0, 0);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);


        if (effective_tokens % VECTOR_MAX_NUM_OF_FP16 != 0) {
            pipe_barrier(PIPE_V);
            // 计算最后一段的起始位置和padding的大小
            int last_padding_start = int(effective_tokens / VECTOR_MAX_NUM_OF_FP16) * VECTOR_MAX_NUM_OF_FP16;
            int padding_size = padding_effective_tokens - effective_tokens;
            __set_mask_from_highbit(padding_size);
            vector_dup(weight_in_ub_addr + last_padding_start, half(-65504), 1, 1, 1, 8, 0);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }

        pipe_barrier(PIPE_V);

        // max = MAX(QK)
        // weightInUbAddr存储的是paddingEffectiveTokens个token的其中一个head的QK计算结果
        // tempInUbAddr存储的是paddingEffectiveTokens / VECTOR_MAX_NUM_OF_FP16 个最大值
        vcmax(temp_in_ub_addr, weight_in_ub_addr, padding_effective_tokens / VECTOR_MAX_NUM_OF_FP16, 1, 1, 8, ONLY_VALUE);
        pipe_barrier(PIPE_V);
        // numRepeat个最大值再比较一次，得到最大值
        int num_blocks = padding_effective_tokens / VECTOR_MAX_NUM_OF_FP16;
        if (num_blocks <= VECTOR_MAX_NUM_OF_FP16) {
            __set_mask(num_blocks);
            vcmax(temp_in_ub_addr, temp_in_ub_addr, 1, 1, 1, 8, ONLY_VALUE);
        } else {
            __set_mask_from_highbit(NUMELEMENT_OF_TEMPBUF - num_blocks);
            vector_dup(temp_in_ub_addr + VECTOR_MAX_NUM_OF_FP16, half(-65504), 1, 1, 1, 8, 0);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            pipe_barrier(PIPE_V);
            vcmax(temp_in_ub_addr, temp_in_ub_addr, 2, 1, 1, 8, ONLY_VALUE);
            pipe_barrier(PIPE_V);
            __set_mask(2);
            vcmax(temp_in_ub_addr, temp_in_ub_addr, 1, 1, 1, 8, ONLY_VALUE);
            pipe_barrier(PIPE_V);
        }

        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
        // broadcast一个max标量为一个block大小的向量，避免使用scalar运算
        vbrcb((__ubuf__ uint16_t*)temp_in_ub_addr, (__ubuf__ uint16_t*)temp_in_ub_addr, 0, 0, 1);
        pipe_barrier(PIPE_V);

        // QK - max
        vsub(cal_weight_in_ub_addr, weight_in_ub_addr, temp_in_ub_addr, padding_effective_tokens / VECTOR_MAX_NUM_OF_FP16, 1, 1, 0, 8, 8, 0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        pipe_barrier(PIPE_V);
        // EXP = exp(QK-max)
        vexp(cal_weight_in_ub_addr, cal_weight_in_ub_addr, padding_effective_tokens / VECTOR_MAX_NUM_OF_FP16, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // s = reduce_sum(EXP)
        vcadd(temp_in_ub_addr, cal_weight_in_ub_addr, padding_effective_tokens / VECTOR_MAX_NUM_OF_FP16, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
        if (num_blocks <= VECTOR_MAX_NUM_OF_FP16) {
            __set_mask(num_blocks);
            vcadd(temp_in_ub_addr, temp_in_ub_addr, 1, 1, 1, 8, 0);
        } else {
            __set_mask_from_highbit(NUMELEMENT_OF_TEMPBUF - num_blocks);
            vector_dup(temp_in_ub_addr + VECTOR_MAX_NUM_OF_FP16, half(0), 1, 1, 1, 8, 0);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            pipe_barrier(PIPE_V);
            vcadd(temp_in_ub_addr, temp_in_ub_addr, 2, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            __set_mask(2);
            vcadd(temp_in_ub_addr, temp_in_ub_addr, 1, 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
        // broadcast一个s = reduce_sum(EXP)标量为一个block大小的向量，避免使用scalar运算
        vbrcb((__ubuf__ uint16_t*)temp_in_ub_addr, (__ubuf__ uint16_t*)temp_in_ub_addr, 0, 0, 1);
        pipe_barrier(PIPE_V);

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        vdiv(weight_out_ub_addr, cal_weight_in_ub_addr, temp_in_ub_addr, padding_effective_tokens / VECTOR_MAX_NUM_OF_FP16, 1, 1, 0, 8, 8, 0);
        if (len_chunk > padding_effective_tokens) {
            pipe_barrier(PIPE_V);
            vector_dup(weight_out_ub_addr + padding_effective_tokens, half(0),
                (len_chunk - padding_effective_tokens) / VECTOR_MAX_NUM_OF_FP16, 1, 1, 8, 0);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        copy_ubuf_to_gm(((__gm__ half *)attn_weight_out) + idx * num_tokens, weight_out_ub_addr, 0, 1, len_chunk / 16, 0, 0);

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);

    pipe_barrier(PIPE_ALL);
}

inline __aicore__ void prefill_att_mix_aiv(
    __gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* qk, __gm__ uint8_t* block_table,
    __gm__ uint8_t* padding_N, __gm__ uint8_t* prefix_lens,
    __gm__ uint8_t* v, __gm__ uint8_t* out, __gm__ uint8_t* prompt_lens,
    __gm__ uint8_t* __restrict__ cum_prompt_len,
    uint32_t head_size, uint32_t num_heads, uint32_t num_kv_heads, uint32_t block_size, uint32_t batchSize, uint32_t max_num_blocks)
{
    set_atomic_none();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    uint32_t num_qkv_heads = num_heads + num_kv_heads * 2;
    pipe_barrier(PIPE_ALL);

    int qk_idx = 0;
    for (int batch_idx = 0; batch_idx < batchSize; batch_idx++) {
        pipe_barrier(PIPE_ALL);
        uint32_t cur_prompt_len = (uint32_t)(*((__gm__ int32_t *)prompt_lens + batch_idx));

        uint32_t new_tokens = ROUND_UP(cur_prompt_len, cur_prompt_len <= SEQLEN_64 ? TILESIZE_16 : TILESIZE_128);
        uint32_t num_tokens = (uint32_t)(*((__gm__ int32_t *)padding_N + batch_idx));
        uint32_t cached_tokens = (uint32_t)(*((__gm__ int32_t *)prefix_lens + batch_idx));

        uint32_t pM = TILESIZE_128;
        // 根据序列长度动态调整query的切分粒度pM，保证数据通过L2传递
        if (num_tokens > SEQLEN_19K) {
            pM = TILESIZE_32;
            new_tokens = ROUND_UP(cur_prompt_len, TILESIZE_32);
        } else if (num_tokens > SEQLEN_8K) {
            pM = TILESIZE_64;
            new_tokens = ROUND_UP(cur_prompt_len, TILESIZE_64);
        }
        if (new_tokens <= SEQLEN_64) {
            pM = TILESIZE_16; // 避免new_tokens很小的情况（小于128），切分粒度是16，也保证可以整除
        }
        // qk为外部申请的一块GM内存，用于存储cube计算QK的输出，作为vector计算softmax的输入，并存储vector计算softmax的输出
        // qk用于double buffer，该处计算的是两个buffer之间的offset
        uint32_t qk_offset = block_num * pM * num_tokens;

        // 单个batch的计算总量：每个token计算numHeads个不同head的attention，共newTokens个token，即newTokens * numHeads次attention
        // 单个core的计算总量：每个token计算其中一个head的attention，共pM个token，即pM次attention
        // 单个batch的总迭代次数为 newTokens * numHeads / pM 次attention
        int seq_num = new_tokens / pM;
        int total_task_num = num_heads * seq_num;
        pipe_barrier(PIPE_ALL);
        for (int i = block_idx; i < total_task_num; i += block_num) {
            int seq_idx = i % seq_num;
            int head_idx = i / seq_num;

            // 等待cube的QK计算完成
            uint64_t flag_id = 0;
            wait_flag_dev(flag_id);

            // 一个AI Core分组可包含1个AIC和2个AIV
            uint32_t sub_id = get_subblockid();
            int new_cached_tokens = seq_idx * pM + sub_id * pM / 2 + cached_tokens;
            uint32_t cur_seq = seq_idx * pM + sub_id * pM / 2;
            __gm__ half *attn_weight_in_one_iter = ((__gm__ half *)qk) + block_idx * pM * num_tokens + sub_id * pM / 2*num_tokens;
            if (qk_idx % 2 == 0) {
                attn_weight_in_one_iter = attn_weight_in_one_iter + qk_offset;
            }

            att_softmax_vector_kernel(attn_weight_in_one_iter, attn_weight_in_one_iter,
                                      num_tokens, num_tokens, pM / 2, new_cached_tokens, cur_seq, cur_prompt_len);
            flag_id = 1;
            uint64_t mode = 2; // inner-group aic/aiv sync
            uint64_t config = 1 | (mode << 4) | (flag_id << 8);
            ffts_cross_core_sync(PIPE_MTE3, config);
            qk_idx++;
        }
        pipe_barrier(PIPE_ALL);
    }
}

#endif

extern "C" __global__ __aicore__ void prefill_att(
    __gm__ uint8_t* q, __gm__ uint8_t* k, __gm__ uint8_t* qk, __gm__ uint8_t* block_table,
    __gm__ uint8_t* padding_N, __gm__ uint8_t* prefix_lens,
    __gm__ uint8_t* v, __gm__ uint8_t* out, __gm__ uint8_t* prompt_lens,
    __gm__ uint8_t* __restrict__ cum_prompt_len,
    uint32_t head_size, uint32_t num_heads, uint32_t num_kv_heads, uint32_t block_size, uint32_t batchSize, uint32_t max_num_blocks)
{
#ifdef __DAV_C220_CUBE__
    prefill_att_mix_aic(q, k, qk, block_table, padding_N, prefix_lens, v, out, prompt_lens,
                        cum_prompt_len, head_size, num_heads, num_kv_heads, block_size, batchSize, max_num_blocks);
#elif __DAV_C220_VEC__
    prefill_att_mix_aiv(q, k, qk, block_table, padding_N, prefix_lens, v, out, prompt_lens,
                        cum_prompt_len, head_size, num_heads, num_kv_heads, block_size, batchSize, max_num_blocks);
#endif
}
