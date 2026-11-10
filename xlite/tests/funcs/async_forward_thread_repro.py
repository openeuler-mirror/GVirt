#!/usr/bin/python3
#
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Regression test for the async forward-thread error and teardown paths.

Covers the scenarios fixed by "xlite: bugfix: signal forward-task completion on
failure and drain before teardown":

  b  a forward task that throws must still signal the output notify the caller
     queued before submitting; otherwise that stream waits forever.
  c  destroying the runtime with work in flight must drain the queued task and
     the device before freeing the tensor pool and the sync primitives.
  d  a submit that hits the error captured by a previous failed task must
     rethrow and still backfill the notify it queued, or the caller hangs.

Run from any directory (the script locates the source tree itself):

    python tests/funcs/async_forward_thread_repro.py

If the xlite bindings are not importable, set PYTHONPATH to the source-tree
xlite directory explicitly.

Exit code: 0 = every scenario passes (defects absent), 1 = at least one defect
reproduced, 2 = harness/setup problem. XLITE_FORWARD_THREAD must not be
disabled: the test targets the async path and refuses to run without it.
"""

import os
import subprocess
import sys
import textwrap

# Per-scenario timeout. A hang is the expected pre-fix behaviour, so it is
# reported as a failure instead of blocking the caller.
CASE_TIMEOUT_S = 120

# <repo>/xlite/tests/funcs/<this file> -> <repo>/xlite. Launching as a script
# puts the script directory (not the cwd) on sys.path, and the wheel excludes
# the tests package, so both `tests.models` and the source-tree `xlite` need
# this root explicitly.
REPO_XLITE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_XLITE not in sys.path:
    sys.path.insert(0, REPO_XLITE)

CHILD_ENV = {**os.environ}
_parent_pythonpath = os.environ.get("PYTHONPATH")
CHILD_ENV["PYTHONPATH"] = os.pathsep.join([REPO_XLITE] + ([_parent_pythonpath] if _parent_pythonpath else []))

# --------------------------------------------------------------------------- #
# Minimal model scaffold (reuses the random hybrid model, no checkpoint needed)
# --------------------------------------------------------------------------- #
SETUP_TEMPLATE = textwrap.dedent(
    """
    import gc
    import os

    # qwen3_5.py imports the xlite bindings only when this backend is selected
    # (same as tests/models/gdn_paged.py).
    os.environ["FORWARD_BACKEND"] = "xlite"

    import torch
    import torch_npu  # noqa: F401
    from xlite._C import Runtime
    from tests.models.qwen3_5 import ModelArgs, Qwen3_5

    def build(pool_mb=None):
        # Weight init below uses in-place ops on leaf parameters; disable autograd
        # globally for the child (mirrors the @torch.inference_mode() used by the
        # model tests).
        torch.set_grad_enabled(False)
        torch.npu.set_device(0)
        torch.manual_seed(565)
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device("npu")
        args = ModelArgs(
            dim=128, inter_dim=256, vocab_size=128, n_layers=2,
            n_heads=2, n_kv_heads=1, head_dim=256,
            linear_num_key_heads=2, linear_num_value_heads=16,
            linear_key_head_dim=64, linear_value_head_dim=64,
            full_attention_interval=2, max_batch_size=2, max_seq_len=128,
        )
        model = Qwen3_5(args).eval()
        for name, value in model.named_parameters():
            if name.endswith("norm.weight") or "layernorm.weight" in name:
                value.zero_()
            elif name.endswith("A_log"):
                value.fill_(-1.0)
            else:
                value.normal_(0, 0.04)
        model.layers[0].linear_attn.norm.weight.fill_(1.0)
        model.xlite_weight_nz = False
        model.xlite_rt = Runtime(0, size=0)
        model.init_xlite_model(args)
        model.init_xlite_kvcache(args)
        model.xlite_kv_cache[0][1] = model.xlite_kv_cache[0][1].float()
        # pool_mb lets a case force in-forward pool exhaustion.
        size = model.xlite_model.get_tensor_pool_size() if pool_mb is None else pool_mb
        model.xlite_rt.init_tensor_pool(size)
        return model, args

    def submit(model, args, tokens, out=None):
        \"\"\"Submit one forward without waiting for the forward thread.\"\"\"
        part = tokens.contiguous()
        out = torch.empty(part.numel(), args.dim) if out is None else out
        meta = model.prepare_xlite_attnmeta_v2(part, 0)
        stream = torch.npu.current_stream().npu_stream
        model.xlite_model.forward_v2(
            rt=model.xlite_rt, input=part.flatten(), attn_meta=meta,
            kv_cache=model.xlite_kv_cache, freqs_cis=[model.freqs_cis],
            output=out, curr_stream=stream,
        )
        return out
    """
)

# Scenario b: a failing forward task must still signal its output notify.
CASE_B = SETUP_TEMPLATE + textwrap.dedent(
    """
    # 1 MB pool: the first GetTensor inside _model->Forward throws on the forward
    # thread. Before the fix the catch path skipped RecordOutputEvent, so the
    # notify the caller queued before submitting was never recorded and its
    # stream blocked forever.
    model, args = build(pool_mb=1)
    tokens = torch.randint(0, args.vocab_size, (2, 9), dtype=torch.int32)
    submit(model, args, tokens)
    print("submitted; waiting", flush=True)
    torch.npu.synchronize()          # PRE-FIX: hang
    print("CASE-B-PASS", flush=True)
    """
)

# Scenario c: destroying the runtime must drain tasks and the device first.
CASE_C = SETUP_TEMPLATE + textwrap.dedent(
    """
    # One executable prefill (64 tokens) so the device has real work pending, then
    # drop ONLY the runtime: the model and its input tensors stay alive, so any
    # failure here is the runtime freeing its pool/events under the running kernel
    # rather than the caller releasing its inputs too early.
    model, args = build()
    tokens = torch.randint(0, args.vocab_size, (2, 64), dtype=torch.int32)
    submit(model, args, tokens)
    print("submitted; dropping the runtime without syncing", flush=True)
    rt = model.xlite_rt
    del model.xlite_rt
    del rt                           # Runtime dtor: drain, then free pool/events
    gc.collect()
    torch.npu.synchronize()          # inputs are still alive and valid
    print("CASE-C-PASS", flush=True)
    """
)

# Scenario d: a rejected submit must rethrow and still backfill its notify.
CASE_D = SETUP_TEMPLATE + textwrap.dedent(
    """
    import time

    # 1 MB pool makes the first task fail on the forward thread and records the
    # error for the next submit. That submit must rethrow, and the output notify
    # it already queued on the caller's stream must be backfilled; otherwise the
    # stream blocks forever even though the exception is raised.
    model, args = build(pool_mb=1)
    tokens = torch.randint(0, args.vocab_size, (2, 9), dtype=torch.int32)
    submit(model, args, tokens)
    torch.npu.synchronize()          # first task failed; backfill keeps this from hanging
    time.sleep(0.2)                  # let the worker publish the captured error
    try:
        submit(model, args, tokens)  # second submit sees the captured error
    except Exception as exc:         # noqa: BLE001
        print("RESUBMIT-RAISED", str(exc)[:80], flush=True)
    else:
        # The worker may not have published _fwdEx yet; the resubmit then took the
        # normal queue path and never exercised the rejection path.
        print("RESUBMIT-NOT-RAISED", flush=True)
        raise SystemExit(3)
    torch.npu.synchronize()          # PRE-FIX (no backfill): hang
    print("CASE-D-PASS", flush=True)
    """
)

CASES = (("b", CASE_B), ("c", CASE_C), ("d", CASE_D))


def run_case(name: str, code: str) -> str:
    """Run one scenario in a fresh process and classify the outcome."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code], env=CHILD_ENV, capture_output=True, text=True, timeout=CASE_TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        print(f"  [{name}] TIMEOUT after {CASE_TIMEOUT_S}s")
        return "hang"
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
    print(f"  [{name}] exit={proc.returncode} last output:")
    for line in tail:
        print(f"        {line}")
    blob = proc.stdout + proc.stderr
    # Distinguish a broken test harness from a reproduced defect.
    for marker in ("a leaf Variable that requires grad", "ModuleNotFoundError", "No module named", "AttributeError"):
        if marker in blob:
            return "setup-error"
    if "RESUBMIT-NOT-RAISED" in blob:
        return "setup-error"
    if proc.returncode != 0:
        return "crash"
    if f"CASE-{name.upper()}-PASS" in proc.stdout:
        return "ok"
    return "unknown"


def main() -> int:
    async_on = os.environ.get("XLITE_FORWARD_THREAD", "true").lower() not in ("0", "false", "no", "off")
    if not async_on:
        print("XLITE_FORWARD_THREAD is disabled; this test targets the async path.")
        print("Unset it (or set it to true) and rerun.")
        return 2

    print("=== async forward-thread error/teardown regression ===")
    results = [(name, run_case(name, code)) for name, code in CASES]

    print("\n=== summary ===")
    failed = []
    for name, result in results:
        if result == "ok":
            print(f"  case {name}: PASS")
        elif result in ("hang", "crash"):
            print(f"  case {name}: FAIL (defect reproduced: {result})")
            failed.append(name)
        else:
            print(f"  case {name}: HARNESS ERROR ({result})")
            failed.append(name)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
