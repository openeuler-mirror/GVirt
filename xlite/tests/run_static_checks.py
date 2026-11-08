#!/usr/bin/env python3
"""Unified static check entry point for xlite.

All lint checks are pre-commit hooks (see xlite/.pre-commit-config.yaml; a
symlink at the repo root lets plain pre-commit commands work). This script is
the CI entry point: it runs every hook over the whole tree, including the
CI-only (manual stage) checks, and auto-installs pre-commit when missing so
existing CI images keep working. Commit-time hooks run automatically via
`pre-commit install`.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# pre-commit prints this marker for environment/bootstrap failures (network
# flake, partial env), which are transient — unlike actual lint findings.
INFRA_FAILURE = re.compile(r"An unexpected error has occurred")
# Hook-status lines, for the legacy CPP/PYTHON markers below.
HOOK_LINE = re.compile(r"^(?P<name>.+?)\.{2,}(?P<status>Passed|Failed|Skipped)\s*$", re.MULTILINE)
# pre-commit unsets npm's user config while building hook environments, so a
# registry mirror configured in ~/.npmrc is ignored; the retry injects one.
NPM_MIRROR_ENV = {"npm_config_registry": "https://mirrors.huaweicloud.com/repository/npm/"}

CPP_HOOKS = ("clang-format", "check-cpp-copyright", "check-cpp-header-guard", "clang-tidy")
CPP_HOOK_NAMES = {"clang-format", "check-cpp-copyright", "check-cpp-header-guard", "clang-tidy"}
PYTHON_HOOK_NAMES = {
    "check python ast",
    "debug statements (python)",
    "ruff check",
    "ruff format",
    "check-torch-npu-imports",
    "mypy",
}
PYTHON_HOOKS = ("check-ast", "debug-statements", "ruff-check", "ruff-format", "check-torch-npu-imports", "mypy")


def run(args: list[str], retry_env: dict | None = None) -> tuple[bool, str]:
    output = ""
    for attempt in (1, 2):
        env = {**os.environ, **retry_env} if attempt == 2 and retry_env else None
        process = subprocess.Popen(
            args,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        chunks = []
        for line in process.stdout:
            sys.stdout.write(line)
            chunks.append(line)
        process.wait()
        output = "".join(chunks)
        if process.returncode == 0 or not INFRA_FAILURE.search(output):
            return process.returncode == 0, output
        if attempt == 1:
            print("transient pre-commit environment failure detected; retrying once ...", flush=True)
    return False, output


def category_markers(output: str) -> list[str]:
    """Legacy CPP/PYTHON status markers: the deployed pre-!640 CI bot parses
    these to gate its UT stage; drop them once that bot is replaced. Any hook
    failure (including hooks outside these categories, e.g. whitespace
    fixers) marks both FAILED so the stale parser's UT gate stays closed."""
    statuses = {m.group("name").strip(): m.group("status") for m in HOOK_LINE.finditer(output)}
    overall_failed = any(status == "Failed" for status in statuses.values())

    def passed(names: set[str]) -> bool:
        return not overall_failed and all(statuses.get(name, "Passed") != "Failed" for name in names)

    markers = []
    if any(name in statuses for name in CPP_HOOK_NAMES):
        markers.append(f"CPP: {'PASSED' if passed(CPP_HOOK_NAMES) else 'FAILED'}")
    if any(name in statuses for name in PYTHON_HOOK_NAMES):
        markers.append(f"PYTHON: {'PASSED' if passed(PYTHON_HOOK_NAMES) else 'FAILED'}")
    return markers


def ensure_pre_commit() -> str:
    """Locate pre-commit, auto-installing it so environments predating the
    dev-dependency change (existing CI images) keep working."""
    pre_commit = shutil.which("pre-commit")
    if pre_commit is not None:
        return pre_commit
    print("pre-commit not found; auto-installing it via pip ...", flush=True)
    for attempt in (1, 2):
        if subprocess.run([sys.executable, "-m", "pip", "install", "pre-commit"]).returncode == 0:
            break
        if attempt == 1:
            print("pip install failed; retrying once ...", flush=True)
    else:
        print("auto-install failed; install it manually with `pip install pre-commit` (or `pip install -e '.[dev]'`)")
        sys.exit(1)
    pre_commit = shutil.which("pre-commit") or str(Path(sys.executable).with_name("pre-commit"))
    if not Path(pre_commit).exists():
        print("pre-commit was installed but its launcher is not reachable on PATH")
        sys.exit(1)
    return pre_commit


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all xlite static checks via pre-commit")
    parser.add_argument("--cpp-only", action="store_true", help="Run only C++ checks")
    parser.add_argument("--python-only", action="store_true", help="Run only Python checks")
    parser.add_argument(
        "--skip-tidy",
        action="store_true",
        default=os.environ.get("XLITE_STATIC_CHECKS_SKIP_TIDY", "") in ("1", "true", "yes"),
        help="Skip clang-tidy (it needs a CANN build tree; the CI bot's per-PR worktrees "
        "cannot rely on one). Also enabled via XLITE_STATIC_CHECKS_SKIP_TIDY=1 so existing "
        "bot orchestrators that invoke this script without flags can opt out",
    )
    args = parser.parse_args()

    pre_commit = ensure_pre_commit()

    def run_hooks(hooks: tuple[str, ...]) -> bool:
        passed = True
        for hook in hooks:
            ok, _ = run([pre_commit, "run", hook, "--all-files", "--hook-stage", "manual"])
            passed = passed and ok
        return passed

    if args.cpp_only or args.python_only:
        category = "CPP" if args.cpp_only else "PYTHON"
        hooks = CPP_HOOKS if args.cpp_only else PYTHON_HOOKS
        passed = run_hooks(hooks)
        print(f"\n{category}: {'PASSED' if passed else 'FAILED'}")
    elif args.skip_tidy:
        cpp_hooks = tuple(h for h in CPP_HOOKS if h != "clang-tidy")
        cpp_passed = run_hooks(cpp_hooks)
        print(f"\nCPP: {'PASSED' if cpp_passed else 'FAILED'}")
        py_passed = run_hooks(PYTHON_HOOKS)
        print(f"PYTHON: {'PASSED' if py_passed else 'FAILED'}")
        passed = cpp_passed and py_passed
    else:
        passed, output = run(
            [pre_commit, "run", "--all-files", "--hook-stage", "manual", "--show-diff-on-failure"],
            retry_env=NPM_MIRROR_ENV,
        )
        for marker in category_markers(output):
            print(marker)

    print(f"\nstatic checks: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
