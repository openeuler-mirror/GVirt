#!/usr/bin/env python3
"""Binding-contract probe against the most local built xlite.

Runs xlite/tests/funcs/binding_contract.py from this hook's own tree. The
probes never reach device work — contract violations reject at the binding
layer's argument parser and valid shapes die on None at the dispatcher's
type check — so no NPU is needed; a built module is. The checked tree's own
build is preferred (pre-commit runs hooks with the checked tree as cwd);
otherwise the environment's installed copy is probed. When nothing is
importable — the CI static stage has no build; the device phase runs the
probe against the freshly built PR package — the probe is skipped.

torch is imported before xlite._C: the extension's DT_RUNPATH is baked at
build time from the build environment's torch lib dir and goes stale when
that path moves, so a direct import only works where the baked path still
resolves. A torch-first import preloads the sonames and makes the search
path irrelevant (see xlite/.claude/notes/2026-11-19-binding-contract-
ld-library-path.md).
"""

import os
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "xlite"
    / "tests"
    / "funcs"
    / "binding_contract.py"
)
DEVICE = "0"
BOOTSTRAP = (
    "import torch; import runpy, sys; "
    "sys.argv = [{script!r}, '--device', {device!r}]; "
    "runpy.run_path({script!r}, run_name='__main__')"
)


def local_build_env() -> dict | None:
    """Environment resolving `import xlite` to the checked tree's own build;
    None when the checked tree has no build (the installed copy serves)."""
    pkg = Path.cwd() / "xlite" / "xlite"
    if not any(pkg.glob("_C*.so")) and not (pkg / "xlite_sabi.abi3.so").exists():
        return None
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{pkg.parent}{os.pathsep}{env.get('PYTHONPATH', '')}"
    return env


def main() -> int:
    env = local_build_env() or dict(os.environ)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; import xlite._C"],
        cwd=Path.cwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        print(
            "no importable built xlite here; skipping (the CI device phase "
            "probes the freshly built PR package)"
        )
        return 0
    return subprocess.run(
        [sys.executable, "-c", BOOTSTRAP.format(script=str(SCRIPT), device=DEVICE)],
        cwd=Path.cwd(),
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    sys.exit(main())
