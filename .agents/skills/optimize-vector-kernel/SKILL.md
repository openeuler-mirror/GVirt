---
name: optimize-vector-kernel
description: Optimize Ascend 910B3 (arch 2201) vector-core kernels for UB bank conflicts with constexpr layouts, multi-row batching, and device-verified A/B bit-identity testing. Use when profiling or rewriting kernels under csrc/kernels/ for memory-access performance.
---

# Optimize arch-2201 vector-core kernels (UB bank conflicts, constexpr layouts)

## Bank model (arch 2201)

UB = 192KB = 48 banks = 16 groups x 3 banks (A/B/C fragments, 64KB each). A bank =
16*floor(block/2048) + block%16 where block = addr/32; one 256B vector repeat touches
8 consecutive bank groups. Conflict granularity depends on the access pair (CANN
avoid_ub_bank_conflict docs): read+write conflicts only on the same *bank*; read+read
and write+write conflict on the same *bank group*. A group's 3 banks are the three
fragments (bank = 16*frag + group), so inside one fragment bank and group coincide and
every pair type needs ±256B phase opposition; across fragments banks always differ —
that settles read+write pairs for free, but never read+read / write+write (groups span
fragments; only phasing fixes those). Triage from addresses alone: group phase =
(addr/32)%16, so same-cycle operands whose base/stride deltas are multiples of 512B are
in phase and collide every row; one tensor read at two in-phase offsets conflicts with
itself (windowed/shifted reads — and multi-row-batched rows of one region — count as
pairs too).

Two rules, per validated practice (indexer_prepare.h):

1. Align every region to whole 512B rows (bank-group sweeps repeat cleanly), and give
   src/dst operands of the same op ±256B phase offsets so their group sweeps are disjoint.
   A +256B head pad on the *phased* operand does it; size the region as
   `ROUND_UP(bytes + (phase?256:0), 512)` so the pad never leaks into the next region.
   A src/dst pair already living in different 64KB fragments has disjoint banks by
   construction — skip the pad there and measure. For rows of one region read in the
   same cycle (multi-row batching), a +256B row stride (leading-dim pad) rotates them
   across group halves. Keep every tensor 32B-aligned regardless (below that,
   misalignment stalls issue on its own), and re-check double-buffer tile fits after
   adding pads.
2. Where a 3-operand pattern cannot be phase-opposed (e.g. a half-offset vadd whose src0
   and src1 are one segment apart — their group windows tile the 16-group cycle, leaving
   no in-fragment phase for the dst), put the *write* operand in a different 64KB
   fragment via top-down + bottom-up allocation instead of forcing a phase: cross-
   fragment banks differ, so the read+write pair is settled without a pad. The fragment
   move never fixes read+read or write+write pairs — those only yield to phasing or,
   for gather/transpose patterns where phases cannot be controlled at all, to staging
   through a scratch UB buffer with a rotated (non-512B-multiple) row stride.

Express layouts as `constexpr` constants from cap macros in `kernel_param.h`
(static_assert the fits; the geometry is already macro'd in `kernel_macro.h` as
`UB_BANKGROUP_*` / `UB_BANK_CONFLICT_OFFSET`), and enforce the caps host-side in
`op.cpp` — the in-kernel
`assert` compiles out under NDEBUG, so a host check that throws is the only real guard.
When the kernel picks between layouts by feature (e.g. affine vs full), the host guard
must encode the *same* selection condition.

## DMI copy rules (device-verified, expensive to rediscover)

- Multi-burst row batching: `nBurst = rows` bursts, **per-burst length = ONE row**
  (`len_burst`), src/dst gaps in 32B blocks = (stride_between_rows − row_bytes)/32.
  Passing a whole-batch length as lenBurst makes every burst over-copy onto wrong
  offsets (first rows look fine, later rows corrupt).
- UB→UB self-copies must never overlap: a single `copy_ubuf_to_ubuf(dst = src + nd,
  lenBurst = (segs-1)*nd)` broadcast *should* tile a repeating pattern, but the copy
  engine's chunked read-ahead reads the un-clobbered source region and leaves stale
  bytes in tail segments. Loop one non-overlapping burst per segment instead.
- Tolerance probe: exactly-adjacent copies (lenBurst == gap) are safe.

## Verification discipline (placement-only changes must be bit-identical)

1. Build the old binary and the new one; run a **seeded A/B probe** (both builds, same
   script, `torch.save` all outputs) and require bit-identity. Do not trust
   `torch.allclose` or unseeded unit tests for this.
2. Fill outputs with a **sentinel** (`torch.full_like(x, -33333)`) — distinguishes
   "never wrote" (launch flake / control-flow gap) from "wrote wrong values" (compute
   bug). Low-precision dtypes round the fill *per dtype*: bf16 stores -33280,
   fp16 -33344, fp32 -33333 — never hardcode one rounded constant (most kernels here
   are bf16). Compare against the dtype-cast expectation instead, e.g.
   `unwritten = out == torch.full_like(out, -33333)` recomputes the stored value
   per dtype; equivalently `out == -33333` (torch casts the scalar to out.dtype).
3. When A/B disagrees, run an **fp32 oracle** (rebuild inputs from the seed, compare
   with tolerance > 1 ulp) to see *which* side is wrong before touching code. On a
   shared NPU, whole-launch drops happen to both builds — the oracle attributes them.
4. Per-segment / per-row diff maps (which row, which segment, diff counts) pinpoint
   arithmetic bugs far faster than aggregate comparisons.

## msprof harness pattern

Model on `tests/perf/indexer_topk.py`: build all inputs up front
on device, sleep past `msprof --delay` so only kernel launches land in the window
(`--delay=5 --duration=15` with `-T 10` is comfortable), fire N launches per case with
one `rt.Synchronize()` per call, then parse per-launch rows of
`op_summary_<timestamp>.csv` (column `Task Duration(us)`) back onto the case plan by
launch order. When the export carries vector-pipe columns (`aiv_vec_ratio`), track
them besides Task Duration: short kernels can show flat duration while issue stalls
drop, and stalls disproportionate to instruction count are the tell that bank
conflicts are the bottleneck in the first place. `import torch_npu` before any
`torch.npu` use; pybind `at::Tensor&`
args reject `None` — pass `torch.empty(0)` for null weights.

## Workflow

read the kernel + its callers → triage same-cycle operand pairs by access type and
group phase → derive the layout math and static_assert it → host caps → rebuild →
seeded sentinel A/B probe → oracle on any diff → msprof before/after per case →
record the trade-offs honestly (a layout win on batched shapes can cost fixed launch
overhead on 1-row decode; measure both). msprof measurements may fluctuate with time,
so run old and new versions together for precise measurements.
