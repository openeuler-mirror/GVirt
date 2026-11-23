#!/usr/bin/env python3
"""xlite CI check harness.

Runs the xlite checks against a checked-out worktree and reports one JSON
object {"checks": [{name, status, output, note}]} (to --json-out, or
stdout when omitted; progress logs go to stderr). Checks: the static
pre-commit set including clang-tidy and the manual-stage binding contract
probe, the vllm-ascend integration mypy check, and the device tests — the
binding contract against the freshly built PR package plus the accuracy
suite — against one AGENTS.md cmake build of the worktree.

pr_ci_bot.py invokes this from a trusted .gitcode copy; every check runs
that copy's sibling scripts (run_static_checks.py, run_accuracy.sh)
against the PR worktree, and pre-commit runs with --config pointing at
the trusted tree's xlite/.pre-commit-config.yaml — whose relative hook
references run_static_checks.py rewrites to absolute trusted paths, so
PR code never controls how it is checked. Runner-local values (weights path, local
vllm/vllm-ascend checkouts, timeouts) come from .gitcode/ci.toml beside
this script. All subprocesses stay on this interpreter's venv, with the
venv's pybind11/torch cmake configs on CMAKE_PREFIX_PATH (CANN's
set_env.sh alone does not expose them).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG: dict = {
    "accuracy": {"weights": "/mnt/nvme0n1/models/Qwen3-0.6B"},
    "integration": {
        "vllm_ascend_dir": "/workspaces/code/opencode/vllm-ascend",
        "vllm_dir": "/workspaces/code/opencode/vllm",
    },
    "timeouts": {"static": 1800, "mypy": 1800, "build": 1800, "accuracy": 1200},
}

ALL_CHECKS = ("static", "integration", "accuracy")
HOOK_LINE = re.compile(
    r"^(?P<name>.+?)\.{2,}(?P<status>Passed|Failed|Skipped)\s*$", re.MULTILINE
)
# torch-first launch of the binding contract: the extension's DT_RUNPATH is
# baked from the build environment's torch path and goes stale when that
# path moves, so a direct `import xlite._C` only resolves where the baked
# path still exists; importing torch first preloads the sonames instead.
BINDING_BOOTSTRAP = (
    "import torch; import runpy, sys; "
    "sys.argv = [{script!r}, '--device', '0']; "
    "runpy.run_path({script!r}, run_name='__main__')"
)
# The bot scrubs secret-shaped variables from its environment before
# invoking this script; the harness keeps them out of every check
# subprocess too, so direct invocations are covered as well.
SECRET_ENV_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY)", re.IGNORECASE
)


def load_config() -> dict:
    config_file = HARNESS_DIR / "ci.toml"
    if not config_file.exists():
        return DEFAULT_CONFIG
    # Top-level sections merge shallowly, but every reader falls back per
    # key to DEFAULT_CONFIG, so a partial ci.toml (e.g. one timeout) is safe.
    with open(config_file, "rb") as f:
        return {**DEFAULT_CONFIG, **tomllib.load(f)}


def timeout(config: dict, phase: str) -> int:
    return int(config.get("timeouts", {}).get(phase, DEFAULT_CONFIG["timeouts"][phase]))


def run_cmd(
    args: list[str], cwd: Path, timeout: int, env: dict | None = None
) -> tuple[str, int]:
    # start_new_session + killpg: grandchildren of pre-commit → hooks →
    # cmake/clang-tidy and run_accuracy.sh → torchrun → NPU workers survive
    # a plain child-kill — after an accuracy timeout they would keep holding
    # the NPU and starve the next PR's device tests.
    try:
        proc = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as e:
        return f"failed to run {args[0]}: {e}", 1
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, err = proc.communicate()
        return f"timed out after {timeout}s\n{out or ''}", 1
    output = out or ""
    if err:
        output += "\n" + err
    return output, proc.returncode


def hook_sections(output: str) -> list[tuple[str, str, str]]:
    matches = list(HOOK_LINE.finditer(output))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        # Hook names come from the pre-commit output, which the PR diff
        # influences — strip markdown-table breakers before a name lands in
        # a posted note.
        name = " ".join(re.sub(r"[`|]", " ", match.group("name")).split())
        sections.append(
            (name, match.group("status").lower(), output[match.end() : end].strip())
        )
    return sections


def static_output(output: str) -> str:
    """Only the failing hooks' sections of a pre-commit run; plain output otherwise."""
    failed = [body for _, status, body in hook_sections(output) if status == "failed"]
    return "\n\n".join(failed) if failed else output


def check_env() -> dict:
    # sys.prefix, not sys.executable: venv pythons are symlinks, and
    # .resolve() would anchor PATH to the symlink target's bin instead.
    env = {k: v for k, v in os.environ.items() if not SECRET_ENV_RE.search(k)}
    env["PATH"] = f"{Path(sys.prefix) / 'bin'}{os.pathsep}{env.get('PATH', '')}"
    # The cmake dirs come from the packages themselves: a venv migration can
    # re-point imports at a system tree while sysconfig still reports the
    # venv's own site-packages, and a guessed prefix fails find_package(Torch).
    import pybind11
    import torch

    extra = f"{pybind11.get_cmake_dir()}{os.pathsep}{torch.utils.cmake_prefix_path}"
    env["CMAKE_PREFIX_PATH"] = (
        f"{extra}{os.pathsep}{env['CMAKE_PREFIX_PATH']}"
        if env.get("CMAKE_PREFIX_PATH")
        else extra
    )
    # Replay unchanged host-C++ objects through the compiler cache when it
    # is installed (the bot points CCACHE_DIR at a bot-side directory shared
    # across PR runs): content-keyed, so a re-run or a rebased PR with
    # unchanged csrc skips recompiling them. The AscendC kernel objects go
    # through CANN's recompile step either way. Absent ccache: no launcher.
    if shutil.which("ccache"):
        env["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
    # The extension modules carry a build-time RUNPATH on torch's lib dir;
    # when that path moved (venv migrations) only torch-first imports keep
    # working — the binding contract runs torch-first (see run_device_tests),
    # so no search-path surgery is needed here.
    return env


def result(name: str, status: str, output: str = "", note: str = "") -> dict:
    return {"name": name, "status": status, "output": output, "note": note}


def run_static(wt: Path, env: dict, config: dict) -> dict:
    # The trusted tree is the one this harness copy was executed from; its
    # xlite/.pre-commit-config.yaml alone defines the checks — the PR
    # worktree's own copy is never consulted, not even to decide anything.
    # run_static_checks.py rewrites the config's relative hook references
    # to trusted absolute paths, so even those cannot resolve into the PR.
    trusted_config = HARNESS_DIR.parent / "xlite" / ".pre-commit-config.yaml"
    if not trusted_config.exists():
        return result(
            "xlite static checks",
            "failed",
            "",
            "the harness tree has no xlite/.pre-commit-config.yaml",
        )
    cmd = [
        sys.executable,
        str(HARNESS_DIR / "run_static_checks.py"),
        "--worktree",
        str(wt),
        "--config",
        str(trusted_config),
    ]
    output, returncode = run_cmd(cmd, wt, timeout(config, "static"), env=env)
    check = result(
        "xlite static checks",
        "passed" if returncode == 0 else "failed",
        static_output(output) if returncode else output,
    )
    if returncode:
        failed = [
            name for name, status, _ in hook_sections(output) if status == "failed"
        ]
        if failed:
            check["note"] = "failed hooks: " + ", ".join(f"`{name}`" for name in failed)
    return check


def run_integration(
    wt: Path, vllm_ascend_dir: str, vllm_dir: str, env: dict, config: dict
) -> dict:
    env = {**env, "MYPYPATH": f"{vllm_dir}:{wt / 'xlite'}"}
    cmd = [
        "mypy",
        "--follow-imports",
        "skip",
        "--check-untyped-defs",
        "--python-version",
        "3.11",
        f"--config-file={vllm_ascend_dir}/mypy.ini",
        f"{vllm_ascend_dir}/vllm_ascend/xlite/xlite.py",
    ]
    output, returncode = run_cmd(cmd, wt, timeout(config, "mypy"), env=env)
    return result(
        "vllm-ascend integration check",
        "passed" if returncode == 0 else "failed",
        output,
    )


def build_xlite(wt: Path, env: dict, config: dict) -> tuple[str, int]:
    """AGENTS.md worktree build. Numbered -j, env-overridable with setup.py's
    128 default: bare -j leaks unbounded MAKEFLAGS into the kernel sub-makes
    and dies with EMFILE (PR !678/!686). The install prefix is the worktree's
    xlite/ so the package dir gains the built module and lib/ for
    PYTHONPATH-based resolution. Reuses the tree the static phase's clang-tidy
    hook already built (incremental build no-ops; the launch-header build and
    the device build are the same cmake project), and configures from scratch
    only when no tree exists (e.g. --skip-tidy runs)."""
    jobs = (
        os.environ.get("MAX_JOBS")
        or os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        or "128"
    )
    build_timeout = timeout(config, "build")
    if not (wt / "xlite" / "build" / "CMakeCache.txt").exists():
        configure = run_cmd(
            ["cmake", "-B", "build"], wt / "xlite", build_timeout, env=env
        )
        if configure[1] != 0:
            return configure
    build = run_cmd(
        ["cmake", "--build", "build", "-j", jobs], wt / "xlite", build_timeout, env=env
    )
    if build[1] != 0:
        return build
    return run_cmd(
        ["cmake", "--install", "build", "--prefix", str(wt / "xlite")],
        wt / "xlite",
        build_timeout,
        env=env,
    )


def run_device_tests(wt: Path, env: dict, config: dict) -> list[dict]:
    """Build the PR worktree once and run the device-dependent checks against
    the freshly built package: the binding contract (catches _C.pyi vs
    TORCH_LIBRARY name drift and enum/value drift that would silently bind
    None) and the accuracy tests."""
    build_output, build_rc = build_xlite(wt, env, config)
    if build_rc != 0:
        return [
            result("binding contract", "failed", build_output, note="PR build failed"),
            result("accuracy tests", "failed", build_output, note="PR build failed"),
        ]
    env = {
        **env,
        "XLITE_ACCURACY_CKPT": config.get("accuracy", {}).get(
            "weights", DEFAULT_CONFIG["accuracy"]["weights"]
        ),
        "PYTHONPATH": f"{wt / 'xlite'}{os.pathsep}{env.get('PYTHONPATH', '')}",
        "LD_LIBRARY_PATH": f"{wt / 'xlite' / 'xlite' / 'lib'}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}",
    }
    contract_output, contract_rc = run_cmd(
        [
            sys.executable,
            "-c",
            BINDING_BOOTSTRAP.format(
                script=str(wt / "xlite" / "tests" / "funcs" / "binding_contract.py")
            ),
        ],
        wt / "xlite",
        timeout(config, "accuracy"),
        env={**env, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    )
    checks = [
        result(
            "binding contract",
            "passed"
            if contract_rc == 0 and "contract wrappers probed" in contract_output
            else "failed",
            contract_output,
        )
    ]
    output, returncode = run_cmd(
        ["bash", str(HARNESS_DIR / "run_accuracy.sh")],
        wt / "xlite",
        timeout(config, "accuracy"),
        env=env,
    )
    checks.append(
        result(
            "accuracy tests",
            "passed"
            if returncode == 0 and "All accuracy tests PASSED!" in output
            else "failed",
            output,
        )
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the xlite CI checks against a worktree"
    )
    parser.add_argument(
        "--worktree", "-w", required=True, help="Worktree (repo root) to check"
    )
    parser.add_argument(
        "--vllm-ascend-dir",
        "-a",
        default=None,
        help="Path to the vllm-ascend checkout (default: .gitcode/ci.toml)",
    )
    parser.add_argument(
        "--vllm-dir",
        "-v",
        default=None,
        help="Path to the vllm checkout (default: .gitcode/ci.toml)",
    )
    parser.add_argument(
        "--checks",
        "-c",
        default=",".join(ALL_CHECKS),
        help=f"Comma-separated subset of {ALL_CHECKS}",
    )
    parser.add_argument(
        "--json-out",
        "-j",
        default=None,
        help="Write the JSON result to this path instead of stdout",
    )
    args = parser.parse_args()

    selected = [c for c in args.checks.split(",") if c]
    unknown = [c for c in selected if c not in ALL_CHECKS]
    if unknown:
        parser.error(f"unknown checks: {', '.join(unknown)}")
    wt = Path(args.worktree).resolve()
    config = load_config()
    integration_cfg = config.get("integration", DEFAULT_CONFIG["integration"])
    vllm_ascend_dir = args.vllm_ascend_dir or integration_cfg.get("vllm_ascend_dir", "")
    vllm_dir = args.vllm_dir or integration_cfg.get("vllm_dir", "")
    env = check_env()
    # Cross-tree cache hits: normalize the worktree path ccache hashes so
    # identical content in different PR worktrees (or a re-run after a
    # rebase) shares objects — without BASEDIR each tree's absolute -I paths
    # and debug paths hash differently and every worktree misses.
    if "CMAKE_CXX_COMPILER_LAUNCHER" in env:
        env["CCACHE_BASEDIR"] = str(wt)
        env["CCACHE_NOHASHDIR"] = "1"
    # Parallel clang-tidy workers for the static phase's tidy hook
    # (ci.toml [tidy] jobs; the hook's own default applies when unset).
    tidy_jobs = config.get("tidy", {}).get("jobs")
    if tidy_jobs:
        env["XLITE_CI_TIDY_JOBS"] = str(int(tidy_jobs))

    checks = []
    static = run_static(wt, env, config) if "static" in selected else None
    if static:
        checks.append(static)
    integration = (
        run_integration(wt, vllm_ascend_dir, vllm_dir, env, config)
        if "integration" in selected
        else None
    )
    if integration:
        checks.append(integration)
    gate = (static is None or static["status"] == "passed") and (
        integration is None or integration["status"] == "passed"
    )
    if "accuracy" in selected:
        if gate:
            checks.extend(run_device_tests(wt, env, config))
        else:
            checks.extend(
                [
                    result(
                        "binding contract",
                        "skipped",
                        note="static/integration checks failed",
                    ),
                    result(
                        "accuracy tests",
                        "skipped",
                        note="static/integration checks failed",
                    ),
                ]
            )

    payload = json.dumps({"checks": checks}, ensure_ascii=False)
    if args.json_out:
        Path(args.json_out).write_text(payload)
    else:
        print(payload)
    failed = [c["name"] for c in checks if c["status"] == "failed"]
    print(
        "checks: "
        + (", ".join(f"{c['name']}: {c['status']}" for c in checks) or "none"),
        file=sys.stderr,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
