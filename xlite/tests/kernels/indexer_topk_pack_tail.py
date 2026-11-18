import argparse
import sys
from collections import Counter

import torch
import torch_npu  # noqa: F401

from xlite._C import Runtime, indexer_topk


def local_ref_pack_tail(query, cache, weights, block_table, positions, lens, pool_lens, max_pool, kpool, topk_pool):
    """Self-contained packTail reference: select pools, expand, then patch the causal tail."""
    T = query.shape[0]
    blocks_per_req = block_table.shape[1]
    req = torch.repeat_interleave(torch.arange(lens.numel(), device=query.device), lens)
    pools = torch.arange(max_pool, device=query.device)
    pages = (pools // cache.shape[1]).clamp(0, blocks_per_req - 1)
    blocks = block_table[req][:, pages].clamp(0, cache.shape[0] - 1)
    rows = pools % cache.shape[1]
    k = cache[blocks, rows, 0, :].float()
    per_head = torch.einsum("thd,tpd->thp", query.float(), k).to(query.dtype).float()
    scores = torch.einsum("thp,th->tp", per_head, weights.float())
    visible = torch.minimum((positions + 1) // kpool, pool_lens[req])
    scores = scores.masked_fill(pools.unsqueeze(0) >= visible.unsqueeze(1), torch.finfo(torch.float32).min)

    selected = min(topk_pool, max_pool)
    _, pool_ids = torch.topk(scores, selected, dim=1)
    pool_ids = pool_ids.to(torch.int32)
    width = topk_pool * kpool + kpool - 1
    out = torch.full((T, width), -1, dtype=torch.int32, device=query.device)
    slot = torch.arange(kpool, device=query.device, dtype=torch.int32)
    for rank in range(selected):
        out[:, rank * kpool : (rank + 1) * kpool] = pool_ids[:, rank : rank + 1] * kpool + slot

    tail_start = ((positions + 1) // kpool) * kpool
    tail_col = torch.minimum(tail_start, torch.full_like(tail_start, topk_pool * kpool))
    tail_count = positions + 1 - tail_start
    row = torch.arange(T, device=query.device)
    for tail_offset in range(kpool - 1):
        col = tail_col + tail_offset
        mask = col < width
        value = torch.where(tail_offset < tail_count, tail_start + tail_offset, torch.full_like(tail_start, -1))
        out[row[mask], col[mask]] = value[mask].to(torch.int32)
    return out, per_head, scores


torch.manual_seed(0)
DEV = "npu"
rt = Runtime(0, size=500)
torch.npu.set_device(0)
torch.set_default_device(DEV)


def run_case(
    name, query, cache, weights, q_start_loc, lens, cached_lens, block_table, positions, topk_pool, kpool, kv_split=0
):
    # indexer_topk works in pool space: topK = number of compressed rows; the
    # output width after packTail expansion is topk_pool*kpool + kpool - 1,
    # matching the local reference called with token-space index_topk.
    T = query.shape[0]
    width = topk_pool * kpool + kpool - 1
    # rows are padded to a 128-slot stride; the kernel writes the full stride
    # (padding = -1) and the caller reads the first `width` columns
    stride = (width + 127) // 128 * 128
    got = torch.full((T, stride), -12345, dtype=torch.int32)
    torch.npu.synchronize()
    indexer_topk(
        rt,
        query,
        cache,
        weights,
        got,
        q_start_loc,
        lens,
        cached_lens,
        block_table,
        query.shape[1],
        query.shape[2],
        cache.shape[1],
        len(lens),
        topk_pool,
        compress_ratio=kpool,
        kv_split=kv_split,
        pack_tail=1,
        out_stride=stride,
        relu_qk=0,
    )
    torch.npu.synchronize()
    snap1 = got.clone()
    total = int(lens.sum() + cached_lens.sum())
    pool_lens = (cached_lens + lens) // kpool
    max_pool = max((total + kpool - 1) // kpool, topk_pool)
    if block_table.ndim == 1:
        block_table = block_table.reshape(len(lens), -1)
    ref, per_head, scores = local_ref_pack_tail(
        query, cache, weights, block_table, positions, lens, pool_lens, max_pool, kpool, topk_pool
    )
    torch.npu.synchronize()
    snap2 = got.clone()
    chg = snap1 != snap2
    if bool(chg.any()):
        rows = chg.any(dim=1).nonzero().flatten().tolist()
        print(f"  [SNAP] got changed after reference: rows={rows[:8]}")
        t0 = rows[0]
        cols = chg[t0].nonzero().flatten().tolist()
        print(f"  [SNAP] row{t0} cols={cols[:20]} before={snap1[t0][cols[0]].item()} after={snap2[t0][cols[0]].item()}")
    # Set semantics: SFA ignores slot order, but the kernel emits main slots in
    # index order while the reference emits score order. Normalize future/invalid
    # entries to -1, then compare per-row sorted multisets (slot-order agnostic).
    causal = positions.unsqueeze(1)

    def norm(x):
        return x.masked_fill((x < 0) | (x > causal), -1)

    # The reference mirrors the kernel's two-stage per-head rounding; pool picks
    # may still differ inside that envelope, but template/tail lanes are strict.
    mant = 7 if query.dtype == torch.bfloat16 else 10

    def ulp(x):
        return torch.exp2(torch.floor(torch.log2(x.abs().clamp(min=1e-30))) - mant)

    env = torch.einsum("thp,th->tp", ulp(per_head), weights.float().abs()) + ulp(scores)

    pad_bad = int((got[:, width:] != -1).sum())
    if pad_bad:
        print(f"  [PAD] {pad_bad} non -1 lanes beyond width={width} (stride={stride})")

    got = got[:, :width]
    got_n = norm(got)
    ref_n = norm(ref)
    got_sorted = torch.sort(got_n, dim=1).values
    ref_sorted = torch.sort(ref_n, dim=1).values
    n_mismatch = int((got_sorted != ref_sorted).sum())
    ok = n_mismatch == 0 and pad_bad == 0
    if not ok:
        tail_lo = topk_pool * kpool
        template_row = ((positions + 1) // kpool - 1) < topk_pool

        def row_ok(t):
            # Template rows and tail/sentinel lanes must match exactly; only
            # sparse-row main slots may swap pools inside the rounding envelope.
            if template_row[t]:
                return False
            if not torch.equal(got_n[t, tail_lo:], ref_n[t, tail_lo:]):
                return False
            pg = Counter(v // kpool for v in got_n[t][:tail_lo].tolist() if v >= 0)
            pr = Counter(v // kpool for v in ref_n[t][:tail_lo].tolist() if v >= 0)
            if pg == pr:
                return True
            extra_g = pg - pr
            extra_r = pr - pg
            if sum(extra_g.values()) != sum(extra_r.values()):
                return False
            for pool, need in extra_r.items():
                for gp in [g for g in extra_g if extra_g[g] > 0]:
                    gap = abs(float(scores[t, gp]) - float(scores[t, pool]))
                    if gap <= float(env[t, gp] + env[t, pool]):
                        take = min(need, extra_g[gp])
                        extra_g[gp] -= take
                        need -= take
                        if need == 0:
                            break
                if need > 0:
                    return False
            return True

        per_row = (got_sorted != ref_sorted).sum(dim=1)
        bad = (per_row > 0).nonzero().flatten().tolist()
        real = []
        for t in bad:
            if row_ok(t):
                print(f"  t={t}: pool swap within rounding envelope -> excused")
            else:
                real.append(t)
        ok = not real and pad_bad == 0
        for t in real:
            g = got_n[t]
            r = ref_n[t]
            extra = sorted(set(g.tolist()) - set(r.tolist()))[:12]
            missing = sorted(set(r.tolist()) - set(g.tolist()))[:12]
            print(f"  t={t}: got[:12]={got[t].tolist()[:12]}")
            print(f"        ref[:12]={ref[t].tolist()[:12]}")
            print(f"        got-only(extra)={extra}")
            print(f"        ref-only(missing)={missing}")
    print(f"[{name}] kv_split={kv_split}: mismatch={n_mismatch} -> {'PASS' if ok else 'FAIL'}")
    import os

    save = os.environ.get("XLITE_SAVE_CASE")
    if save:
        # per-row pool-level sets for offline diff against kernel dumps
        pools_got = sorted(set(int(v) for v in got[got <= (int(causal.max()) // 4) * 4 - 1] // 4))
        pools_ref = sorted(set(int(v) for v in ref[ref <= (int(causal.max()) // 4) * 4 - 1] // 4))
        with open(save, "a", encoding="utf-8") as fh:
            fh.write(f"CASE {name} kv_split={kv_split}\n")
            for t in range(got.shape[0]):
                ng = norm(got)[t].tolist()
                nr = norm(ref)[t].tolist()
                # dump raw rows: sorted-multiset diff must cover -1 padding and duplicates
                fh.write(f"RAW {t} got={'[' + ','.join(str(int(v)) for v in ng) + ']'}\n")
                fh.write(f"RAW {t} ref={'[' + ','.join(str(int(v)) for v in nr) + ']'}\n")
        print(f"  saved token sets to {save} (pools got={len(pools_got)} ref={len(pools_ref)})")
    return ok


def flat_block_table(seq_lens, cached_lens, kpool, ibs=32):
    # batch-major flat table: each request owns ceil(visible_pools/ibs) pool
    # blocks, ids contiguous across requests over one shared cache.
    blocks_per_req = [max(1, (cl + ln + kpool - 1) // kpool // ibs) for cl, ln in zip(cached_lens, seq_lens)]
    blocks = max(blocks_per_req)
    table = []
    nxt = 0
    for bpr in blocks_per_req:
        row = list(range(nxt, nxt + bpr)) + [0] * (blocks - bpr)
        nxt += bpr
        table.append(row)
    return torch.tensor(table, dtype=torch.int32), blocks


def scenario_a(dtype):
    # small multi-request prefill; both rows take the dense template (fewer
    # visible pools than topK) plus the causal tail.
    D, NH, TOPK_POOL, KPOOL = 32, 16, 4, 4
    seq_lens = [9, 6]
    cached = [0, 0]
    positions = torch.cat([torch.arange(c, c + s) for c, s in zip(cached, seq_lens)])
    T = positions.shape[0]
    q_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(seq_lens), 0).to(torch.int32)]
    )
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    cached_lens = torch.tensor(cached, dtype=torch.int32)
    query = torch.randn(T, NH, D, dtype=torch.bfloat16).to(dtype)
    weights = torch.randn(T, NH, dtype=torch.bfloat16).to(dtype)
    ibs = 32
    nblocks = 8
    cache = torch.randn(nblocks, ibs, 1, D, dtype=torch.bfloat16).to(dtype)
    table, _ = flat_block_table(seq_lens, cached, KPOOL, ibs)
    ok = run_case(
        f"S1-prefill/{dtype}", query, cache, weights, q_start_loc, lens, cached_lens, table, positions, TOPK_POOL, KPOOL
    )
    return ok


def scenario_b(dtype):
    # decode shape: 1 query token over a long cached history; covers a
    # tail-full row (cached multiple of kpool) and a tail-partial row.
    D, NH, TOPK_POOL, KPOOL, IBS = 64, 16, 16, 4, 32
    ok = True
    for cached_extra in (0, 2):
        cached = 97 + cached_extra
        query = torch.randn(1, NH, D, dtype=torch.bfloat16).to(dtype)
        weights = torch.randn(1, NH, dtype=torch.bfloat16).to(dtype)
        pool_len = (cached + 1 + KPOOL - 1) // KPOOL
        nblocks = max(1, (pool_len + IBS - 1) // IBS)
        cache = torch.randn(nblocks, IBS, 1, D, dtype=torch.bfloat16).to(dtype)
        table = torch.arange(nblocks, dtype=torch.int32).unsqueeze(0)
        lens = torch.tensor([1], dtype=torch.int32)
        cached_lens = torch.tensor([cached], dtype=torch.int32)
        q_start_loc = torch.tensor([0, 1], dtype=torch.int32)
        positions = torch.arange(cached, cached + 1)
        ok &= run_case(
            f"S2-decode-tail{cached % KPOOL}/{dtype}",
            query,
            cache,
            weights,
            q_start_loc,
            lens,
            cached_lens,
            table,
            positions,
            TOPK_POOL,
            KPOOL,
        )
    return ok


def scenario_c(dtype):
    # long single-request prefill spanning several pool blocks.
    D, NH, TOPK_POOL, KPOOL, IBS = 64, 16, 16, 4, 32
    seq = 400
    positions = torch.arange(seq)
    q_start_loc = torch.tensor([0, seq], dtype=torch.int32)
    lens = torch.tensor([seq], dtype=torch.int32)
    cached_lens = torch.zeros(1, dtype=torch.int32)
    query = torch.randn(seq, NH, D, dtype=torch.bfloat16).to(dtype)
    weights = torch.randn(seq, NH, dtype=torch.bfloat16).to(dtype)
    nblocks = (seq + KPOOL * IBS - 1) // (KPOOL * IBS)
    cache = torch.randn(nblocks, IBS, 1, D, dtype=torch.bfloat16).to(dtype)
    table = torch.arange(nblocks, dtype=torch.int32).unsqueeze(0)
    return run_case(
        f"S3-longprefill/{dtype}",
        query,
        cache,
        weights,
        q_start_loc,
        lens,
        cached_lens,
        table,
        positions,
        TOPK_POOL,
        KPOOL,
    )


def scenario_d(topk_pool=512, tag="S4-glm53-8k/bf16", kv_split=0):
    # GLM53 production geometry (bf16): 8k cached tokens + 4 query tokens,
    # 4096 pools = 128 pool blocks, topK = 512 pools = 2048 token slots.
    D, NH, TOPK_POOL, KPOOL, IBS = 32, 32, topk_pool, 4, 32
    cached, qlen = 8192, 4
    query = torch.randn(qlen, NH, D, dtype=torch.bfloat16)
    weights = torch.randn(qlen, NH, dtype=torch.bfloat16)
    pool_len = (cached + qlen) // KPOOL
    nblocks = (pool_len + IBS - 1) // IBS
    cache = torch.randn(nblocks, IBS, 1, D, dtype=torch.bfloat16)
    table = torch.arange(nblocks, dtype=torch.int32).unsqueeze(0)
    lens = torch.tensor([qlen], dtype=torch.int32)
    cached_lens = torch.tensor([cached], dtype=torch.int32)
    q_start_loc = torch.tensor([0, qlen], dtype=torch.int32)
    positions = torch.arange(cached, cached + qlen)
    name = tag if kv_split == 0 else tag.replace("/bf16", "-kvsplit/bf16")
    return run_case(
        name,
        query,
        cache,
        weights,
        q_start_loc,
        lens,
        cached_lens,
        table,
        positions,
        TOPK_POOL,
        KPOOL,
        kv_split=kv_split,
    )


def scenario_d_kvsplit():
    # same geometry through the cross-core kvSplit relay path.
    D, NH, TOPK_POOL, KPOOL, IBS = 32, 32, 512, 4, 32
    cached, qlen = 8192, 4
    query = torch.randn(qlen, NH, D, dtype=torch.bfloat16)
    weights = torch.randn(qlen, NH, dtype=torch.bfloat16)
    pool_len = (cached + qlen) // KPOOL
    nblocks = (pool_len + IBS - 1) // IBS
    cache = torch.randn(nblocks, IBS, 1, D, dtype=torch.bfloat16)
    table = torch.arange(nblocks, dtype=torch.int32).unsqueeze(0)
    lens = torch.tensor([qlen], dtype=torch.int32)
    cached_lens = torch.tensor([cached], dtype=torch.int32)
    q_start_loc = torch.tensor([0, qlen], dtype=torch.int32)
    positions = torch.arange(cached, cached + qlen)
    return run_case(
        "S4b-glm53-8k-kvsplit/bf16",
        query,
        cache,
        weights,
        q_start_loc,
        lens,
        cached_lens,
        table,
        positions,
        TOPK_POOL,
        KPOOL,
        kv_split=1,
    )


def scenario_e(dtype):
    # boundary rows: query token exactly on a pool boundary (tail count 0 ->
    # all tail slots sentinel) and a sub-kpool first chunk (tail count < kpool).
    D, NH, TOPK_POOL, KPOOL, IBS = 32, 16, 4, 4, 32
    ok = True
    for seq in (4, 3, 1):
        positions = torch.arange(seq)
        q_start_loc = torch.tensor([0, seq], dtype=torch.int32)
        lens = torch.tensor([seq], dtype=torch.int32)
        cached_lens = torch.zeros(1, dtype=torch.int32)
        query = torch.randn(seq, NH, D, dtype=torch.bfloat16).to(dtype)
        weights = torch.randn(seq, NH, dtype=torch.bfloat16).to(dtype)
        cache = torch.randn(2, IBS, 1, D, dtype=torch.bfloat16).to(dtype)
        table = torch.arange(2, dtype=torch.int32).unsqueeze(0)
        ok &= run_case(
            f"S5-boundary-len{seq}/{dtype}",
            query,
            cache,
            weights,
            q_start_loc,
            lens,
            cached_lens,
            table,
            positions,
            TOPK_POOL,
            KPOOL,
        )
    return ok


def scenario_f(dtype):
    # multi-request mixed lens with packed layout (both cached and cold starts).
    D, NH, TOPK_POOL, KPOOL, IBS = 32, 16, 8, 4, 32
    seq_lens = [5, 17, 3]
    cached = [64, 0, 33]
    positions = torch.cat([torch.arange(c, c + s) for c, s in zip(cached, seq_lens)])
    T = positions.shape[0]
    q_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(torch.tensor(seq_lens), 0).to(torch.int32)]
    )
    lens = torch.tensor(seq_lens, dtype=torch.int32)
    cached_lens = torch.tensor(cached, dtype=torch.int32)
    query = torch.randn(T, NH, D, dtype=torch.bfloat16).to(dtype)
    weights = torch.randn(T, NH, dtype=torch.bfloat16).to(dtype)
    nblocks = 32
    cache = torch.randn(nblocks, IBS, 1, D, dtype=torch.bfloat16).to(dtype)
    table, _ = flat_block_table(seq_lens, cached, KPOOL, IBS)
    return run_case(
        f"S6-packed/{dtype}", query, cache, weights, q_start_loc, lens, cached_lens, table, positions, TOPK_POOL, KPOOL
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1, help="repeat the full matrix to expose intermittent races")
    parser.add_argument("--topksweep", action="store_true")
    args = parser.parse_args()
    ok = True
    for round_idx in range(args.rounds):
        print(f"ROUND {round_idx + 1}/{args.rounds}")
        round_ok = True
        for dtype in [torch.bfloat16, torch.float16]:
            round_ok &= scenario_a(dtype)
            round_ok &= scenario_b(dtype)
            round_ok &= scenario_c(dtype)
            round_ok &= scenario_e(dtype)
            round_ok &= scenario_f(dtype)
        round_ok &= scenario_d()
        round_ok &= scenario_d(kv_split=1)
        ok &= round_ok
    if args.topksweep:
        for tp in (64, 100, 128, 256):
            ok &= scenario_d(topk_pool=tp, tag=f"S4c-topk{tp}/bf16")
    print(f"indexer_topk pack_tail RESULT: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
