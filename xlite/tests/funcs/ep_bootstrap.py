#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""EP bootstrap with a non-trivial MoE tensor-parallel factor.

`InitHcclComm` derives each group's bootstrap node as `_ips[node_index]`, where
`_ips` holds one entry per node (single node -> exactly one entry) and the EP
group id is `rank % moe_tp_size`. With `moe_tp_size == 1` — the value every other
test and the shipped configurations use — that index is always 0, so the EP path
is never exercised with an index that can leave the vector. A `moe_tp_size > 1`
layout used to read past it (silent out-of-bounds read -> SIGSEGV on odd ranks).

Usage (no arguments, no dependency on the working directory; the script
re-executes itself once per rank):

    python3 <repo>/xlite/tests/funcs/ep_bootstrap.py

It runs on the cards visible to this process (up to 4) and covers two layouts:
`moe_tp=1` as the control and `moe_tp=2` as the regression case. Fewer than two
visible cards prints SKIP and exits 0.

Exit code: 0 = every rank bootstrapped (or skipped), 1 = at least one failure,
2 = harness problem (xlite bindings not importable).
"""

import os
import subprocess
import sys
import time

# <repo>/xlite/tests/funcs/<this file> -> <repo>/xlite. Everything below is derived
# from this path, so the script behaves the same from any working directory.
REPO_XLITE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _normalize_sys_path():
    """Make imports independent of the launch directory.

    A launcher (or a PYTHONPATH entry) can put the checkout ahead of site-packages;
    the checkout has no compiled `_C` module, so that entry would shadow the
    installed one. Drop the cwd entries and move the repo root behind the installed
    packages, keeping it importable for `tests.*` helpers.
    """
    cwd = os.path.abspath(os.getcwd())
    keep = []
    for entry in sys.path:
        resolved = os.path.abspath(entry or os.getcwd())
        if resolved not in (cwd, REPO_XLITE):
            keep.append(entry)
    sys.path[:] = keep
    if REPO_XLITE not in sys.path:
        sys.path.append(REPO_XLITE)


_normalize_sys_path()

try:
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401

    from xlite._C import Runtime  # noqa: E402
except ImportError as exc:  # pragma: no cover - environment problem, not a test failure
    print(
        f"HARNESS xlite bindings not importable ({exc}); build and install the package "
        f"first (cd {REPO_XLITE} && pip install --no-deps .)",
        flush=True,
    )
    sys.exit(2)

CHILD_ENV = "EP_BOOTSTRAP_CHILD_RANK"
MAX_RANKS = 4
CONFIG_TIMEOUT_S = 240


def run_rank():
    """Child body: bootstrap every communicator this rank participates in."""
    rank = int(os.environ[CHILD_ENV])
    world = int(os.environ["WORLD_SIZE"])
    moe_tp = int(os.environ["EP_BOOTSTRAP_MOE_TP"])
    moe_ep = int(os.environ["EP_BOOTSTRAP_MOE_EP"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        # tp_size spans the whole world so the EP groups are carved out of a single
        # TP group, which is how a single-node MoE deployment looks.
        rt = Runtime(rank, size=64, rank=rank, tp_size=world, dp_size=1, moe_tp_size=moe_tp, moe_ep_size=moe_ep)
        print(f"PASS EP bootstrap rank={rank}/{world} moe_tp={moe_tp} moe_ep={moe_ep}", flush=True)
        dist.barrier()
        del rt
    finally:
        dist.destroy_process_group()
    return 0


def run_config(world, moe_tp, moe_ep, base_port):
    """Spawn one process per rank and report the outcome of this layout."""
    env = {**os.environ}
    parent_pythonpath = os.environ.get("PYTHONPATH")
    env.update(
        {
            "WORLD_SIZE": str(world),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(base_port),
            # Runtime's own bootstrap port range; must not collide with MASTER_PORT.
            "XLITE_PORT": str(base_port + 50),
            "EP_BOOTSTRAP_MOE_TP": str(moe_tp),
            "EP_BOOTSTRAP_MOE_EP": str(moe_ep),
            # Children re-run this file; hand them the repo root the same way the other
            # funcs tests do (the script moves it behind site-packages on startup).
            "PYTHONPATH": os.pathsep.join([REPO_XLITE] + ([parent_pythonpath] if parent_pythonpath else [])),
        }
    )
    procs = []
    for rank in range(world):
        # env:// rendezvous reads RANK/WORLD_SIZE/MASTER_*; keep them explicit.
        child_env = {**env, CHILD_ENV: str(rank), "RANK": str(rank)}
        procs.append(
            subprocess.Popen(
                [sys.executable, os.path.abspath(__file__)],
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    deadline = time.time() + CONFIG_TIMEOUT_S
    ok = 0
    for rank, proc in enumerate(procs):
        try:
            out, _ = proc.communicate(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
            out = (out or "") + f"\n[TIMEOUT after {CONFIG_TIMEOUT_S}s]"
        passed = proc.returncode == 0 and f"PASS EP bootstrap rank={rank}/" in out
        ok += int(passed)
        if not passed:
            detail = " | ".join(line for line in out.splitlines() if line.strip() and "Warning" not in line)[:400]
            print(f"FAIL moe_tp={moe_tp} moe_ep={moe_ep} rank={rank} rc={proc.returncode} :: {detail}", flush=True)
    print(f"{'PASS' if ok == world else 'FAIL'} layout moe_tp={moe_tp} moe_ep={moe_ep} ranks={ok}/{world}", flush=True)
    return ok == world


def main():
    if CHILD_ENV in os.environ:
        return run_rank()

    world = min(MAX_RANKS, int(torch.npu.device_count()))
    if world < 2:
        print(f"SKIP EP bootstrap: needs >= 2 visible cards, found {world}", flush=True)
        return 0

    base_port = 29600 + (os.getpid() % 200) * 2
    layouts = [(1, world)]
    if world >= 4:
        layouts.append((2, world // 2))  # regression case: moe_tp_size > 1

    failures = 0
    for index, (moe_tp, moe_ep) in enumerate(layouts):
        failures += int(not run_config(world, moe_tp, moe_ep, base_port + index * 100))
    print(
        f"{'PASS' if failures == 0 else 'FAIL'} EP bootstrap ({len(layouts)} layout(s) x {world} rank(s))", flush=True
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
