#!/usr/bin/env python3
"""clang-tidy hook (CI / manual stage).

Requires a build tree with compile_commands.json; configures the project when
the tree is missing and always brings it current (kernel launch headers are
build-time artifacts). Kernel sources under csrc/kernels and csrc/ascend.h are
excluded (CCE intrinsics produce intentional warnings).

Analysis runs one clang-tidy process per file in parallel (XLITE_CI_TIDY_JOBS
workers, default min(16, cpus)): a serial loop makes a full pass cost the sum
of all per-file analyses instead of the slowest few. Analysis verdicts are
cached when XLITE_CI_TIDY_CACHE names a writable directory: the key fingerprints
every analysis input (the analyzed sources, all of csrc/ they include from,
the generated launch headers, the compile commands with the worktree path
normalized away, the tidy binary, the config and this hook), so identical
content re-checked in another PR, a rebase or a /rerun skips the re-analysis
and reuses the stored verdict. The CI harness points both variables at
bot-side values (jobs from .gitcode/ci.toml); unset, no caching happens and
the worker default applies (local development).
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from pathlib import Path

# Checked content anchors to cwd — pre-commit runs hooks with the checked
# worktree as cwd, so the trusted copy still builds and analyzes the checked
# tree. The tidy config anchors to this script's own tree: definitions
# follow the executing hook, not the tree being checked.
XLITE_ROOT = Path.cwd() / "xlite"
TIDY_CONFIG = Path(__file__).resolve().parents[2] / "xlite" / ".clang-tidy"
SKIP_PATHS = ("csrc/kernels/", "csrc/ascend.h")
# The static analyzer (clang-analyzer-*) explores every registration lambda in
# the TORCH_LIBRARY table — 428 s on the current _C.cpp, 96% of the file's
# analysis cost and most of a full pass — while never having reported a
# finding there; these files keep every other check family.
ANALYZER_EXEMPT_FILES = ("_C.cpp",)
ANALYSIS_SUFFIXES = (".c", ".cc", ".cpp", ".h", ".hpp")
TIDY_FINDING = re.compile(r":\d+:\d+: (?:warning|error): ")
TIDY_CACHE_ENV = "XLITE_CI_TIDY_CACHE"
CACHE_CAP = 64


def is_analysis_file(path: Path) -> bool:
    return path.suffix in ANALYSIS_SUFFIXES and not any(
        s in path.as_posix() for s in SKIP_PATHS
    )


def analysis_files() -> list[Path]:
    """The full analysis set — the same file filter pre-commit applies, so a
    run over exactly this set (pre-commit --all-files) can consult the
    cached full-verdict."""
    return sorted(
        p
        for p in (XLITE_ROOT / "csrc").rglob("*")
        if p.is_file() and is_analysis_file(p)
    )


def ensure_build_tree() -> bool:
    build_dir = XLITE_ROOT / "build"
    if not (build_dir / "compile_commands.json").exists():
        # A first configure can fail transiently (compiler ABI try_compile flake)
        # and leave a partial tree; wipe and retry once before giving up.
        for attempt in (1, 2):
            configure = subprocess.run(
                [
                    "cmake",
                    "-S",
                    str(XLITE_ROOT),
                    "-B",
                    str(build_dir),
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if configure.returncode == 0:
                break
            shutil.rmtree(build_dir, ignore_errors=True)
            if attempt == 2:
                print(f"cmake configure failed:\n{configure.stderr.strip()}")
                return False
            print(
                "cmake configure failed transiently; retrying with a clean build tree ..."
            )

    # Build even when the tree already exists: the kernel launch headers
    # op.cpp includes are build-time artifacts, so a tree predating a kernel
    # change parses stale stubs and tidy dies on aclrtlaunch_* signature
    # errors. cmake no-ops here when everything is current. Numbered -j,
    # env-overridable with setup.py's 128 default: bare -j leaks unbounded
    # MAKEFLAGS into the kernel sub-makes (EMFILE; PR !678/!686).
    jobs = (
        os.environ.get("MAX_JOBS")
        or os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL")
        or "128"
    )
    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", jobs],
        capture_output=True,
        text=True,
        check=False,
    )
    if build.returncode != 0:
        print(f"cmake build failed:\n{build.stderr.strip()[-4000:]}")
        return False

    return (build_dir / "compile_commands.json").exists()


def tidy_version() -> str:
    result = subprocess.run(
        ["clang-tidy", "--version"], capture_output=True, text=True, check=False
    )
    return (result.stdout or result.stderr).strip()


def cache_key(version: str) -> str | None:
    """Content fingerprint of every input the analysis reads. None while the
    build tree is incomplete (the caller builds first). Worktree-absolute
    paths in the compile database are normalized away, so two trees with
    identical content share one verdict."""
    build_dir = XLITE_ROOT / "build"
    compile_commands = build_dir / "compile_commands.json"
    if not compile_commands.exists():
        return None
    repo_root = Path.cwd()
    h = hashlib.sha256()
    for blob in (
        version.encode(),
        TIDY_CONFIG.read_bytes(),
        Path(__file__).read_bytes(),
        compile_commands.read_text().replace(str(repo_root), "<WT>").encode(),
    ):
        h.update(blob)
        h.update(b"\0")
    # Everything the analyzed files can include: all of csrc, plus the
    # generated launch headers the build produces from it.
    inputs = sorted(XLITE_ROOT.glob("csrc/**/*"))
    inputs += sorted(build_dir.glob("*.h"))
    inputs += sorted((build_dir / "include").rglob("*.h"))
    for path in inputs:
        if not path.is_file():
            continue
        h.update(path.relative_to(repo_root).as_posix().encode())
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def cache_dir() -> Path | None:
    raw = os.environ.get(TIDY_CACHE_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_verdict(directory: Path, key: str) -> dict | None:
    with suppress(OSError, json.JSONDecodeError):
        entry = json.loads((directory / f"{key}.json").read_text())
        if entry.get("key") == key:
            return entry
    return None


def store_verdict(directory: Path, key: str, clean: bool, output: str) -> None:
    # Atomic (temp + rename): two PRs checking identical content may write
    # the same entry concurrently.
    tmp = directory / f".{key}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"key": key, "clean": clean, "output": output}))
    os.replace(tmp, directory / f"{key}.json")
    entries = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
    for stale in entries[:-CACHE_CAP]:
        with suppress(OSError):
            stale.unlink()


def analyze(files: list[Path]) -> tuple[bool, str]:
    """Run tidy over files in parallel; returns (clean, findings text)."""

    def analyze_one(path: Path) -> tuple[bool, str]:
        rel = path.relative_to(XLITE_ROOT)
        cmd = ["clang-tidy", rel.as_posix(), f"--config-file={TIDY_CONFIG}", "-p=build"]
        if path.name in ANALYZER_EXEMPT_FILES:
            cmd.append("--checks=-clang-analyzer-*")
        result = subprocess.run(
            cmd,
            cwd=XLITE_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        # clang-tidy prints the compile database's absolute paths; verdicts are
        # cached across worktrees, so strip this worktree's prefix — else a
        # cache hit posts the populating worktree's paths under another head.
        output = (result.stdout + result.stderr).replace(str(XLITE_ROOT) + "/", "")
        hits = [line for line in output.splitlines() if TIDY_FINDING.search(line)]
        if result.returncode != 0:
            text = f"clang-tidy failed for {rel} (exit {result.returncode}):\n" + (
                "\n".join(hits) if hits else output.strip()[-2000:]
            )
        elif hits:
            text = f"clang-tidy issues in {rel}:\n" + "\n".join(hits)
        else:
            text = ""
        return result.returncode == 0 and not hits, text

    jobs = int(os.environ.get("XLITE_CI_TIDY_JOBS") or min(16, os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        # map keeps the findings in input (sorted) order: deterministic logs.
        results = list(executor.map(analyze_one, sorted(files)))
    clean = all(ok for ok, _ in results)
    return clean, "\n".join(text for _, text in results if text)


def main() -> int:
    if not (XLITE_ROOT / "CMakeLists.txt").is_file():
        print(
            f"no xlite checkout at {XLITE_ROOT} (run via pre-commit from the repo root)"
        )
        return 1

    def should_check(raw: str) -> bool:
        path = Path(raw).resolve()
        return path.is_relative_to(XLITE_ROOT) and is_analysis_file(path)

    files = sorted(Path(raw).resolve() for raw in sys.argv[1:] if should_check(raw))

    if not files:
        return 0
    if not ensure_build_tree():
        return 1

    # The cached verdict covers the full analysis set: consult it only when
    # this run asks for exactly that set (pre-commit --all-files does), and
    # store only then too — a subset run's verdict must not speak for files
    # it never analyzed.
    directory = cache_dir()
    key = cache_key(tidy_version()) if directory else None
    full_set = files == analysis_files()
    if directory and key and full_set:
        entry = load_verdict(directory, key)
        if entry is not None:
            if entry["clean"]:
                print("clang-tidy: clean (cached verdict for identical content)")
                return 0
            print(entry["output"])
            print(
                "Run `clang-tidy -p=build --fix <file>` under xlite/ to apply suggested fixes."
            )
            print("(cached verdict for identical content)")
            return 1

    clean, findings = analyze(files)
    if directory and key and full_set:
        store_verdict(directory, key, clean, findings)
    if not clean:
        print(findings)
        print(
            "Run `clang-tidy -p=build --fix <file>` under xlite/ to apply suggested fixes."
        )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
