import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, kpool_tail_compress_and_spill

torch.manual_seed(0)
DEV = "npu"
POOL, D, CAP = 4, 128, 4
IBS = 32
rt = Runtime(0, size=500)
torch.npu.set_device(0)
torch.set_default_device(DEV)

# Reference = the production CANN path semantics (indexer_kpool.py _forward_cann):
# the tail pool is the runner's paged view [blocks, cap, 2*head_dim] fp32, each
# row holding one pool slot's [k | gate] pair, read/written row-major.

seq_lens = [9, 6]
positions64 = torch.cat([torch.arange(s) for s in seq_lens])
T = positions64.shape[0]
req_id = torch.cat([torch.full((s,), r) for r, s in enumerate(seq_lens)])
query_start_loc = torch.cat(
    [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(seq_lens), 0).to(torch.int32)]
)
cum_query_lens = query_start_loc[1:]
seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)

k32 = torch.randn(T, D, dtype=torch.float32)
gate32 = torch.randn(T, D, dtype=torch.float32)
ape = torch.randn(POOL, D, dtype=torch.float32)

MAIN_BS = 128
main_slot = (req_id * MAIN_BS + positions64 % MAIN_BS).to(torch.int32)
idx_slot = torch.where(
    (positions64 + 1) % POOL == 0,
    main_slot // POOL,
    torch.tensor(-1),
).to(torch.int32)
tail_block_table = torch.arange(len(seq_lens), dtype=torch.int32).unsqueeze(1)
tail_slot = (req_id * CAP + positions64 % CAP).to(torch.int32)
cached_lens_t = torch.zeros(len(seq_lens), dtype=torch.int32)

NB_IDX = 4
indexer_cache0 = torch.zeros(NB_IDX, IBS, 1, D, dtype=torch.bfloat16)
tail_cache0 = torch.zeros(2, CAP, 2 * D, dtype=torch.float32)

# Production tail views are dim-0 pitched: as_strided over page-padded storage
# (live shape [B, CAP, 2*D] with stride [PAGE_ROWS*2*D, 2*D, 1], PAGE_ROWS=40).
PITCH_ROWS = 40


def make_pitched_tail():
    storage = torch.zeros(2, PITCH_ROWS, 2 * D, dtype=torch.float32)
    return storage.as_strided((2, CAP, 2 * D), (PITCH_ROWS * 2 * D, 2 * D, 1))


def make_pitched_indexer(dtype):
    storage = torch.zeros(NB_IDX, 2, IBS, 1, D, dtype=dtype)
    return storage.as_strided((NB_IDX, IBS, 1, D), (2 * IBS * D, D, D, 1))


def torch_ref_a(tail0):
    q_ends = query_start_loc.tolist()
    tail_new = tail0.clone()
    comp = {}
    q_prev = 0
    for r, s in enumerate(seq_lens):
        q_end = q_ends[r + 1]
        for t in range(q_prev, q_end):
            pos = int(positions64[t])
            slot = int(idx_slot[t])
            if slot < 0 or (pos + 1) % POOL:
                continue
            pool_pos = [pos - (POOL - 1 - j) for j in range(POOL)]
            pk = torch.zeros(POOL, D, dtype=torch.float32)
            pg = torch.zeros(POOL, D, dtype=torch.float32)
            for j, pp in enumerate(pool_pos):
                eff = max(pp, 0)
                in_win = (pp >= 0) and (pp < s)
                if in_win:
                    pk[j] = k32[q_prev + pp]
                    pg[j] = gate32[q_prev + pp]
                elif pp >= 0:
                    off = eff % CAP
                    pk[j] = tail0[r, off, :D]
                    pg[j] = tail0[r, off, D:]
            w = torch.softmax(pg + ape, dim=0)
            comp[slot] = (w * pk).sum(0)
        for off in range(CAP):
            row = q_end - 1 - off
            if row < q_prev or row < 0:
                continue
            slot = int(tail_slot[row])
            if slot < 0:
                continue
            b, o = slot // CAP, slot % CAP
            tail_new[b, o, :D] = k32[row]
            tail_new[b, o, D:] = gate32[row]
        q_prev = q_end
    return comp, tail_new


def check(name, idx_got, tail_got, comp_ref, tail_ref, dtype):
    tol_idx = 1e-2 if dtype == torch.float32 else 2e-2
    tol_tail = 1e-5 if dtype == torch.float32 else 2e-2
    err_idx, n_cmp = 0.0, 0
    for slot, ref in comp_ref.items():
        got = idx_got[slot // IBS, slot % IBS, 0].float()
        err_idx = max(err_idx, (got - ref).abs().max().item())
        n_cmp += 1
    err_tail = (tail_got - tail_ref).abs().max().item()
    ok = err_idx < tol_idx and err_tail < tol_tail
    print(
        f"[{name}] slots={n_cmp}, idx err={err_idx:.3e} (<{tol_idx}), "
        f"tail err={err_tail:.3e} (<{tol_tail}) -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_scenario_a(dtype, pitched=False):
    # dtype = indexerCache store width; k/gate stay FP32 (native contract).
    # pitched=True exercises both dim-0 page-pitch strides: tailCache with the
    # production 40-row page, indexerCache with a 2x block pitch (scale_dim layout).
    tail0 = make_pitched_tail() if pitched else tail_cache0
    comp_ref, tail_ref = torch_ref_a(tail0)
    idx_c = make_pitched_indexer(dtype) if pitched else torch.zeros(NB_IDX, IBS, 1, D, dtype=dtype)
    tail_c = make_pitched_tail() if pitched else tail0.clone()
    kpool_tail_compress_and_spill(
        rt,
        tail_c,
        idx_c,
        k32,
        gate32,
        ape,
        positions64,
        query_start_loc,
        seq_lens_t,
        cached_lens_t,
        main_slot,
        tail_block_table,
        D,
        POOL,
        MAIN_BS,
    )
    return check(f"S3A{'-PITCHED' if pitched else ''}/{dtype}", idx_c, tail_c, comp_ref, tail_ref, dtype)


CAP_B = 4
pos_b64 = torch.tensor([6, 7])
q_start_loc_b = torch.tensor([0, 2], dtype=torch.int32)
lens_b = torch.tensor([2], dtype=torch.int32)
seqlen_b = torch.tensor([8], dtype=torch.int32)
cached_b = torch.tensor([6], dtype=torch.int32)
k_b32 = torch.randn(2, D, dtype=torch.float32)
gate_b32 = torch.randn(2, D, dtype=torch.float32)
tail_block_b = torch.zeros(1, 1, dtype=torch.int32)
idx_slot_b = torch.tensor([-1, 1], dtype=torch.int32)
tail_slot_b = torch.tensor([2, 3], dtype=torch.int32)
k_hist = torch.randn(CAP_B, D, dtype=torch.float32)
g_hist = torch.randn(CAP_B, D, dtype=torch.float32)
tail0_b = torch.zeros(1, CAP_B, 2 * D, dtype=torch.float32)
tail0_b[0, :, :D] = k_hist
tail0_b[0, :, D:] = g_hist


def torch_ref_b():
    pk = torch.zeros(POOL, D, dtype=torch.float32)
    pg = torch.zeros(POOL, D, dtype=torch.float32)
    for j, pp in enumerate([4, 5, 6, 7]):
        if pp < 6:
            off = pp % CAP_B
            pk[j] = tail0_b[0, off, :D]
            pg[j] = tail0_b[0, off, D:]
        else:
            pk[j] = k_b32[pp - 6]
            pg[j] = gate_b32[pp - 6]
    w = torch.softmax(pg + ape, dim=0)
    compB = (w * pk).sum(0)
    tailB = tail0_b.clone()
    for r in range(2):
        off = int(tail_slot_b[r])
        tailB[0, off, :D] = k_b32[r]
        tailB[0, off, D:] = gate_b32[r]
    return compB, tailB


def run_scenario_b(dtype):
    # dtype = indexerCache store width; k/gate stay FP32 (native contract).
    compB_ref, tailB_ref = torch_ref_b()
    idx_c = torch.zeros(NB_IDX, IBS, 1, D, dtype=dtype)
    tail_c = tail0_b.clone()
    kpool_tail_compress_and_spill(
        rt,
        tail_c,
        idx_c,
        k_b32,
        gate_b32,
        ape,
        pos_b64,
        q_start_loc_b,
        lens_b,
        cached_b,
        pos_b64.to(torch.int32),
        tail_block_b,
        D,
        POOL,
        MAIN_BS,
    )
    e_c = (idx_c[0, 1, 0].float() - compB_ref).abs().max().item()
    et_c = (tail_c - tailB_ref).abs().max().item()
    tol_idx = 1e-2 if dtype == torch.float32 else 2e-2
    tol_tail = 1e-5 if dtype == torch.float32 else 2e-2
    ok = e_c < tol_idx and et_c < tol_tail
    print(
        f"[S3B/{dtype}] idx err={e_c:.3e} (<{tol_idx}), tail err={et_c:.3e} (<{tol_tail}) -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


if __name__ == "__main__":
    ok = True
    for dtype in [torch.bfloat16, torch.float32]:
        ok &= run_scenario_a(dtype)
        ok &= run_scenario_a(dtype, pitched=True)
        ok &= run_scenario_b(dtype)
    print(f"S3 RESULT: {'PASS' if ok else 'FAIL'}")
