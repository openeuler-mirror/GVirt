#!/usr/bin/env python3
"""Unified static check entry point for xlite.

All lint checks are pre-commit hooks (see xlite/.pre-commit-config.yaml; a
symlink at the repo root lets plain pre-commit commands work). This
script is the CI entry point: it runs every hook over the whole tree,
including the CI-only (manual stage) checks, and auto-installs pre-commit
when missing so existing CI images keep working. Commit-time hooks run
automatically via `pre-commit install --install-hooks`.

The CI harness (.gitcode/run_ci_checks.py) invokes this from its trusted
.gitcode copy with --worktree set to the PR worktree and --config pointing
at the harness copy of the pre-commit config, so PR code never controls
the checks or their definitions. With --config, every relative reference in
the config's hook entries/args that resolves to a file under the config's
tree is rewritten to an absolute trusted path (pre-commit runs hooks with
the checked worktree as cwd, so a relative reference would otherwise run
the checked tree's copy). Without those flags the script checks the
checkout it lives in, via the repo-root config symlink.
"""

from __future__ import annotations

import argparse
import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# pre-commit prints this marker for environment/bootstrap failures (network
# flake, partial env), which are transient — unlike actual lint findings.
INFRA_FAILURE = re.compile(r"An unexpected error has occurred")
# Hook-status lines, for the legacy CPP/PYTHON markers below.
HOOK_LINE = re.compile(
    r"^(?P<name>.+?)\.{2,}(?P<status>Passed|Failed|Skipped)\s*$", re.MULTILINE
)
# pre-commit unsets npm's user config while building hook environments, so a
# registry mirror configured in ~/.npmrc is ignored; the retry injects one.
NPM_MIRROR_ENV = {
    "npm_config_registry": "https://mirrors.huaweicloud.com/repository/npm/"
}
# A relative reference inside a hook entry or arg (script paths, rule files).
REL_REF = re.compile(r"(?<![-\w.])([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.@{}-]+)+)")

CPP_HOOKS = (
    "clang-format",
    "check-cpp-copyright",
    "check-cpp-header-guard",
    "clang-tidy",
)
CPP_HOOK_NAMES = {
    "clang-format",
    "check-cpp-copyright",
    "check-cpp-header-guard",
    "clang-tidy",
}
PYTHON_HOOK_NAMES = {
    "check python ast",
    "debug statements (python)",
    "ruff check",
    "ruff format",
    "check-torch-npu-imports",
    "mypy",
}
PYTHON_HOOKS = (
    "check-ast",
    "debug-statements",
    "ruff-check",
    "ruff-format",
    "check-torch-npu-imports",
    "mypy",
)


def run(args: list[str], root: Path, retry_env: dict | None = None) -> tuple[bool, str]:
    output = ""
    for attempt in (1, 2):
        env = {**os.environ, **retry_env} if attempt == 2 and retry_env else None
        process = subprocess.Popen(
            args,
            cwd=root,
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
            print(
                "transient pre-commit environment failure detected; retrying once ...",
                flush=True,
            )
    return False, output


def category_markers(output: str) -> list[str]:
    """Legacy CPP/PYTHON status markers: the deployed pre-!640 CI bot parses
    these to gate its UT stage; drop them once that bot is replaced. Any hook
    failure (including hooks outside these categories, e.g. whitespace
    fixers) marks both FAILED so the stale parser's UT gate stays closed."""
    statuses = {
        m.group("name").strip(): m.group("status") for m in HOOK_LINE.finditer(output)
    }
    overall_failed = any(status == "Failed" for status in statuses.values())

    def passed(names: set[str]) -> bool:
        return not overall_failed and all(
            statuses.get(name, "Passed") != "Failed" for name in names
        )

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
        if (
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pre-commit"], check=False
            ).returncode
            == 0
        ):
            break
        if attempt == 1:
            print("pip install failed; retrying once ...", flush=True)
    else:
        print(
            "auto-install failed; install it manually with `pip install pre-commit` (or `pip install -e '.[dev]'`)"
        )
        sys.exit(1)
    pre_commit = shutil.which("pre-commit") or str(
        Path(sys.executable).with_name("pre-commit")
    )
    if not Path(pre_commit).exists():
        print("pre-commit was installed but its launcher is not reachable on PATH")
        sys.exit(1)
    return pre_commit


def derive_config(config_path: Path) -> Path:
    """A copy of the pre-commit config with every relative reference in hook
    entries/args that resolves to a file under the config's tree rewritten to
    an absolute path there: pre-commit runs hooks with the checked worktree
    as cwd, so a relative reference would execute or read the checked tree's
    copy — the very thing a trusted --config must prevent. Flags, regexes,
    URLs and shell fragments do not resolve to files there and pass through
    untouched."""
    # PyYAML ships with pre-commit; imported lazily so ensure_pre_commit can
    # install it first.
    import yaml

    tree = config_path.parents[1]

    def absolutize(text: str) -> str:
        def replace(match: re.Match) -> str:
            candidate = tree / match.group(1)
            return str(candidate.resolve()) if candidate.is_file() else match.group(1)

        return REL_REF.sub(replace, text)

    doc = yaml.safe_load(config_path.read_text())
    for repo in doc.get("repos", []):
        for hook in repo.get("hooks", []):
            if "entry" in hook:
                hook["entry"] = absolutize(hook["entry"])
            if hook.get("args"):
                hook["args"] = [absolutize(arg) for arg in hook["args"]]
    derived = Path(tempfile.mkdtemp(prefix="precommit-config-")) / config_path.name
    derived.write_text(yaml.safe_dump(doc, sort_keys=False))
    atexit.register(shutil.rmtree, derived.parent, ignore_errors=True)
    return derived


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all xlite static checks via pre-commit"
    )
    parser.add_argument(
        "--worktree",
        default=str(REPO_ROOT),
        help="Worktree (repo root) to check (default: this checkout)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="pre-commit config to use (default: discovered from the repository root)",
    )
    parser.add_argument("--cpp-only", action="store_true", help="Run only C++ checks")
    parser.add_argument(
        "--python-only", action="store_true", help="Run only Python checks"
    )
    parser.add_argument(
        "--skip-tidy",
        action="store_true",
        default=os.environ.get("XLITE_STATIC_CHECKS_SKIP_TIDY", "")
        in ("1", "true", "yes"),
        help="Skip clang-tidy (it needs a CANN build tree; kept for local development. "
        "Also enabled via XLITE_STATIC_CHECKS_SKIP_TIDY=1",
    )
    args = parser.parse_args()

    root = Path(args.worktree).resolve()
    pre_commit = ensure_pre_commit()
    config = (
        ["--config", str(derive_config(Path(args.config).resolve()))]
        if args.config
        else []
    )

    def run_hooks(hooks: tuple[str, ...]) -> bool:
        passed = True
        for hook in hooks:
            ok, _ = run(
                [
                    pre_commit,
                    "run",
                    hook,
                    "--all-files",
                    "--hook-stage",
                    "manual",
                    *config,
                ],
                root,
            )
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
            [
                pre_commit,
                "run",
                "--all-files",
                "--hook-stage",
                "manual",
                "--show-diff-on-failure",
                *config,
            ],
            root,
            retry_env=NPM_MIRROR_ENV,
        )
        for marker in category_markers(output):
            print(marker)

    print(f"\nstatic checks: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
