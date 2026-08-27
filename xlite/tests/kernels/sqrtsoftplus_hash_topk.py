#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in hope that it will be useful,
# but WITHOUT ANY WARRANTY; even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
# Correctness harness for the V4 MoE sqrtsoftplus gate. Kernel emits TWO GM outputs: a
# sparse [M,N] weight row (K nonzero slots) + a [M,ceil(N/32)] BIT1 routing bitmap; the
# dense top-k indices are internal. hash=True gathers indices from tid2eid[input_ids] and
# passes bias as null (dead weight there). See ref_gate below.
import torch
import numpy as np

from xlite._C import Runtime, sqrtsoftplus_hash_topk

RUN_KERNEL = True  # both hash=False and hash=True device paths are implemented.

rt = Runtime(0, 500)
torch.npu.set_device(0)


def ref_gate(scores, bias, input_ids, tid2eid, top_k, scale, use_hash):
    """Pure-torch reimplementation of V4 Gate.forward (deepseek_v4.py:572-589), sqrtsoftplus."""
    scores = scores.float()
    scores = torch.nn.functional.softplus(scores).sqrt()
    original_scores = scores
    if bias is not None:
        scores = scores + bias
    if use_hash:
        indices = tid2eid[input_ids]
    else:
        indices = scores.topk(top_k, dim=-1)[1]
    weights = original_scores.gather(1, indices)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * scale
    return weights, indices


# Coverage: (n_tokens, n_routed_experts, top_k, use_hash). hash=False steers topk by biased
# scores; hash=True gathers indices from tid2eid[input_ids].
configs = [
    (9, 256, 6, False),
    (9, 256, 6, True),
    (9, 384, 6, False),
    (9, 384, 6, True),
    (4096, 256, 6, False),
    (4096, 256, 6, True),
    (4096, 384, 6, False),
    (4096, 384, 6, True),
]

# vocab size for the tid2eid table; tokens drawn from [0, vocab).
vocab_size = 129280
scale = 1.5  # V4 route_scale default

n_covered = 0
for dtype in [torch.bfloat16, torch.float32]:
    for n_tokens, n_routed_experts, top_k, use_hash in configs:
        # RNG on CPU then .npu(): torch.randn/randint on NPU can yield ~0/all-zero after prior
        # ops (memory: ascendc-torch-randn-npu-rng-pollution).
        scores = torch.randn(n_tokens, n_routed_experts, dtype=dtype, device="cpu").npu()
        indices_helper = torch.arange(n_routed_experts, dtype=torch.int32, device="cpu").npu()
        bias = (torch.randn(n_routed_experts, dtype=torch.float32, device="cpu").npu()
                if not use_hash else torch.empty(0, dtype=torch.float32, device="npu:0"))
        input_ids = (torch.randint(0, vocab_size, (n_tokens,), dtype=torch.int32,
                                   device="cpu").npu() if use_hash else torch.empty(
                                       0, dtype=torch.int32, device="npu:0"))
        tid2eid = (torch.randint(0, n_routed_experts, (vocab_size, top_k),
                                 dtype=torch.int32, device="cpu").npu() if use_hash else
                   torch.empty(0, 0, dtype=torch.int32, device="npu:0"))

        # Kernel outputs: sparse [M,N] weights (K nonzero slots, rest 0) + [M,ceil(N/32)] BIT1 bitmap.
        out_weights = torch.zeros(n_tokens, n_routed_experts, dtype=dtype, device="npu:0")
        n_bitmap_words = (n_routed_experts + 31) // 32
        routing_map = torch.zeros(n_tokens, n_bitmap_words, dtype=torch.int32,
                                  device="npu:0")

        # Reference recompute on CPU (same data, unpolluted).
        ref_scores = scores.float().cpu()
        ref_bias = bias.cpu() if bias.numel() else None
        ref_input_ids = input_ids.cpu() if use_hash else None
        ref_tid2eid = tid2eid.cpu() if use_hash else None
        ref_w, ref_idx = ref_gate(ref_scores, ref_bias, ref_input_ids, ref_tid2eid, top_k,
                                  scale, use_hash)
        # Scatter ref's dense [M,K] into the kernel's sparse [M,N] layout + matching bitmap.
        ref_sparse = torch.zeros(n_tokens, n_routed_experts, dtype=ref_w.dtype)
        ref_bitmap = torch.zeros(n_tokens, n_bitmap_words, dtype=torch.int32)
        for m in range(n_tokens):
            for k in range(top_k):
                e = int(ref_idx[m, k].item())
                ref_sparse[m, e] = ref_w[m, k]
                ref_bitmap[m, e // 32] |= (1 << (e % 32))

        n_covered += 1
        if not RUN_KERNEL:
            continue

        torch.npu.synchronize()
        sqrtsoftplus_hash_topk(rt, scores, indices_helper, bias, input_ids, tid2eid,
                               out_weights, routing_map, scale, top_k, use_hash)
        torch.npu.synchronize()
        print(f'sqrtsoftplus_hash_topk (n_experts={n_routed_experts}, topk={top_k}, hash={use_hash}, '
              f'dtype={dtype}) executed!')

        close_err = None
        try:
            torch.testing.assert_close(ref_sparse.to(dtype).npu(), out_weights,
                                        atol=1e-5, rtol=1e-3)
            torch.testing.assert_close(ref_bitmap.npu(), routing_map)
        except AssertionError as e:
            close_err = e
            print(f'{e}')
        # Per-token nonzero-slot diff; only mismatched tokens print (torch folds the wide row).
        diff_tol = 1e-5
        n_mismatch = 0
        for m in range(n_tokens):
            rw = ref_sparse[m].to(out_weights.dtype).cpu()
            xw = out_weights[m].cpu()
            rnz = [(j, round(rw[j].item(), 6)) for j in range(n_routed_experts) if rw[j].item() != 0]
            xnz = [(j, round(xw[j].item(), 6)) for j in range(n_routed_experts) if xw[j].item() != 0]
            rpos = set(j for j, _ in rnz)
            xpos = set(j for j, _ in xnz)
            allpos = sorted(rpos | xpos)
            maxdiff = max((abs(rw[j].item() - xw[j].item()) for j in allpos), default=0.0)
            if rpos == xpos and maxdiff <= diff_tol:
                continue
            n_mismatch += 1
            print(f'[m={m}] ref_nonzero({len(rnz)}): {rnz}')
            print(f'[m={m}] xlt_nonzero({len(xnz)}): {xnz}')
            print(f'[m={m}] max_abs_diff_over_union = {maxdiff:.2e} (tol={diff_tol:.0e})')
        if n_mismatch == 0:
            print(f'  all {n_tokens} tokens match (within tol {diff_tol:.0e})')
        if close_err is not None or n_mismatch > 0:
            raise AssertionError(
                f'sqrtsoftplus_hash_topk mismatch (n_experts={n_routed_experts}, topk={top_k}, '
                f'hash={use_hash}, dtype={dtype}): close_err={close_err}, n_mismatch={n_mismatch}'
            ) from close_err
