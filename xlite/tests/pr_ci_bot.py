#!/usr/bin/env python3
"""GitCode PR CI bot for GVirt.

For every open non-draft PR whose head commit and bot version are not both
covered by an existing bot comment (so new commits and updates to this script
on master both trigger a re-check): fetch the PR head into an isolated git
worktree, run the xlite static checks, the vllm-ascend integration check, and
(when those pass) the accuracy tests, then post one concise comment with
per-check statuses. Only failing checks include output, capped to a tail of
lines.

Usage:
  python tests/pr_ci_bot.py [--token <token>] [--pr <number>] [--dry-run]
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import requests

GITCODE_API_BASE = "https://gitcode.com/api/v5"
REPO_OWNER = "openeuler"
REPO_NAME = "GVirt"
# Versioned marker: comments record the bot's content hash and the checked
# commit, so updating pr_ci_bot.py on master re-checks every open PR. Older
# comments (unversioned or legacy orchestrator) never match and stay stale.
BOT_TAG_RE = re.compile(r"<!-- gvirt-ci-bot:(?P<version>[0-9a-f]{12}):(?P<sha>[0-9a-f]{40}) -->")
BOT_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]

DEFAULT_XLITE_DIR = "/workspaces/code/opencode/GVirt"
DEFAULT_VLLM_ASCEND_DIR = "/workspaces/code/opencode/vllm-ascend"
DEFAULT_VLLM_DIR = "/workspaces/code/opencode/vllm"

API_TIMEOUT = 30
STATIC_CHECK_TIMEOUT = 1800
MYPY_TIMEOUT = 1800
INSTALL_TIMEOUT = 1800
UT_TIMEOUT = 1200
COMMENT_MAX_LINES = 120

HOOK_LINE = re.compile(r"^(?P<name>.+?)\.{2,}(?P<status>Passed|Failed|Skipped)\s*$", re.MULTILINE)

STATUS_EMOJI = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}


def log(message: str) -> None:
    print(message, flush=True)


class Check:
    def __init__(self, name: str, status: str, output: str = "", note: str = ""):
        self.name = name
        self.status = status  # passed | failed | skipped
        self.output = output
        self.note = note


def run_cmd(args: list[str], cwd: Path, timeout: int, env: dict | None = None) -> tuple[str, int]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        return output, result.returncode
    except subprocess.TimeoutExpired as e:
        partial = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return f"timed out after {timeout}s\n{partial}", 1
    except OSError as e:
        return f"failed to run {args[0]}: {e}", 1


def split_hook_sections(output: str) -> list[tuple[str, str, str]]:
    """Split pre-commit output into (hook name, status, section output) triples."""
    matches = list(HOOK_LINE.finditer(output))
    sections = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(output)
        sections.append((match.group("name").strip(), match.group("status").lower(), output[match.end() : end].strip()))
    return sections


def tail(text: str, max_lines: int = COMMENT_MAX_LINES) -> str:
    lines = text.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return f"... (truncated, showing last {max_lines} lines) ...\n" + "\n".join(lines[-max_lines:])


class GitCodeClient:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"

    def get(self, path: str, params: dict | None = None):
        response = self.session.get(f"{GITCODE_API_BASE}{path}", params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        return response.json()

    def get_open_prs(self) -> list:
        return self.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls", params={"state": "open", "per_page": 100})

    def get_pr_comments(self, pr_number: int) -> list:
        comments, page = [], 1
        while True:
            batch = self.get(
                f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments",
                params={"page": page, "per_page": 50},
            )
            if not isinstance(batch, list) or not batch:
                return comments
            comments.extend(batch)
            if len(batch) < 50:
                return comments
            page += 1

    def get_pr_head(self, pr_number: int) -> dict | None:
        commits = self.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/commits")
        if isinstance(commits, list) and commits:
            return commits[0]
        return None

    def post_pr_comment(self, pr_number: int, body: str) -> bool:
        try:
            self.session.post(
                f"{GITCODE_API_BASE}/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments",
                json={"body": body},
                timeout=API_TIMEOUT,
            ).raise_for_status()
            return True
        except requests.RequestException as e:
            log(f"Error posting comment to PR #{pr_number}: {e}")
            return False


def has_fresh_bot_comment(comments: list, head_sha: str) -> bool:
    """A PR is already checked when a bot comment carries this bot version and
    the exact head sha; anything older (new commits, bot updates, legacy
    comments) counts as stale and triggers a re-check."""
    for comment in comments:
        m = BOT_TAG_RE.search(comment.get("body", ""))
        if m and m.group("version") == BOT_VERSION and m.group("sha") == head_sha:
            return True
    return False


def git(repo_dir: Path, args: list[str], timeout: int = 300) -> str:
    result = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def fetch_pr_head(repo_dir: Path, repo_path: str, branch: str, pr_number: int) -> str:
    """Fetch the PR head into a bot-owned ref (race-free) and return its sha."""
    url = f"https://gitcode.com/{repo_path}.git"
    ref = f"refs/prbot/pr-{pr_number}"
    git(repo_dir, ["fetch", "--no-tags", "--force", url, f"+{branch}:{ref}"])
    return git(repo_dir, ["rev-parse", ref])


def prepare_worktree(repo_dir: Path, sha: str, worktree_root: Path, pr_number: int) -> Path:
    worktree_root.mkdir(parents=True, exist_ok=True)
    wt = worktree_root / f"pr-{pr_number}-{sha[:8]}"
    if wt.exists():
        shutil.rmtree(wt)
        git(repo_dir, ["worktree", "prune"])
    git(repo_dir, ["worktree", "add", "--detach", str(wt), sha])
    return wt


def remove_worktree(repo_dir: Path, wt: Path) -> None:
    try:
        git(repo_dir, ["worktree", "remove", "--force", str(wt)], timeout=60)
    except (RuntimeError, subprocess.TimeoutExpired):
        shutil.rmtree(wt, ignore_errors=True)
        with contextlib.suppress(RuntimeError, subprocess.TimeoutExpired):
            git(repo_dir, ["worktree", "prune"], timeout=60)


def run_static_checks(wt: Path) -> Check:
    if not (wt / "xlite" / ".pre-commit-config.yaml").exists():
        return Check(
            "xlite static checks",
            "skipped",
            note="PR base predates the pre-commit setup; rebase onto latest master to enable them",
        )
    output, returncode = run_cmd(
        [sys.executable, "tests/run_static_checks.py", "--skip-tidy"], wt / "xlite", STATIC_CHECK_TIMEOUT
    )
    check = Check("xlite static checks", "passed" if returncode == 0 else "failed", output)
    if returncode != 0:
        failed = [name for name, status, _ in split_hook_sections(output) if status == "failed"]
        if failed:
            check.note = "failed hooks: " + ", ".join(f"`{name}`" for name in failed)
    return check


def run_vllm_ascend_check(wt: Path, vllm_ascend_dir: str, vllm_dir: str) -> Check:
    env = os.environ.copy()
    env["MYPYPATH"] = f"{vllm_dir}:{wt / 'xlite'}"
    target = f"{vllm_ascend_dir}/vllm_ascend/xlite/xlite.py"
    cmd = [
        "mypy",
        "--follow-imports",
        "skip",
        "--check-untyped-defs",
        "--python-version",
        "3.11",
        f"--config-file={vllm_ascend_dir}/mypy.ini",
        target,
    ]
    output, returncode = run_cmd(cmd, wt, MYPY_TIMEOUT, env=env)
    return Check("vllm-ascend integration check", "passed" if returncode == 0 else "failed", output)


def build_xlite(wt: Path) -> tuple[str, int]:
    """Build the PR worktree's extension following the documented worktree
    build (AGENTS.md): rm -rf build && cmake -B build && cmake --build build
    -j && cmake --install build. The parallelism is capped at the
    verified-good 8 — the ascendc host ExternalProject installs race under
    unbounded -j, dying in recompile_binary.py TooFewObj with no compiler
    error — and the install prefix is the worktree's xlite/ so its package
    dir gains _C.so and lib/ for PYTHONPATH-based resolution."""
    build_dir = wt / "xlite" / "build"
    shutil.rmtree(build_dir, ignore_errors=True)
    configure = run_cmd(["cmake", "-B", "build"], wt / "xlite", INSTALL_TIMEOUT)
    if configure[1] != 0:
        return configure
    build = run_cmd(["cmake", "--build", "build", "-j8"], wt / "xlite", INSTALL_TIMEOUT)
    if build[1] != 0:
        return build
    return run_cmd(["cmake", "--install", "build", "--prefix", str(wt / "xlite")], wt / "xlite", INSTALL_TIMEOUT)


def worktree_test_env(wt: Path) -> dict:
    """Env that resolves `import xlite` to the PR worktree's freshly built
    package: PYTHONPATH shadows the venv's installed (main checkout) copy,
    and the cmake install placed _C.so plus lib/ inside the package dir (the
    _C rpath $ORIGIN/../xlite/lib resolves there). The venv itself is never
    touched."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(wt / "xlite") + os.pathsep + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = str(wt / "xlite" / "xlite" / "lib") + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    return env


def run_accuracy_tests(wt: Path) -> Check:
    """Build the PR worktree and run the accuracy tests against it, so they
    exercise the PR code rather than the venv's installed (main checkout)
    copy."""
    build_output, build_rc = build_xlite(wt)
    if build_rc != 0:
        return Check("accuracy tests", "failed", build_output, note="PR build failed")
    env = worktree_test_env(wt)
    output, returncode = run_cmd(["bash", "tests/run_accuracy.sh"], wt / "xlite", UT_TIMEOUT, env=env)
    status = "passed" if returncode == 0 and "All accuracy tests PASSED!" in output else "failed"
    return Check("accuracy tests", status, output)


def failed_sections_output(check: Check) -> str:
    """Only the failing hooks' sections of a pre-commit run; plain tail otherwise."""
    sections = split_hook_sections(check.output)
    failed = [section_body for _, status, section_body in sections if status == "failed"]
    if not failed:
        return tail(check.output)
    return tail("\n\n".join(failed))


def build_comment(sha: str, checks: list[Check]) -> str:
    lines = [
        f"### CI results for `{sha[:8]}`",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for check in checks:
        result = f"{STATUS_EMOJI[check.status]} {check.status.capitalize()}"
        if check.note:
            result += f" — {check.note}"
        lines.append(f"| {check.name} | {result} |")
    body = "\n".join(lines)

    for check in checks:
        if check.status != "failed":
            continue
        out = failed_sections_output(check)
        # The output may itself contain triple-backtick fences (e.g. diffs of
        # markdown files); out-fence it so the comment stays well-formed.
        longest_run = max((len(m.group(0)) for m in re.finditer(r"`+", out)), default=0)
        fence = "`" * max(3, longest_run + 1)
        body += f"\n\n<details open>\n<summary>❌ {check.name}</summary>\n\n{fence}text\n{out}\n{fence}\n\n</details>"

    body += f"\n\n<!-- gvirt-ci-bot:{BOT_VERSION}:{sha} -->\n*Posted automatically by the GVirt CI bot; it reruns on every new commit or bot update.*\n"
    return body


def process_pr(client: GitCodeClient, pr: dict, args: argparse.Namespace) -> None:
    pr_number = pr["number"]
    log(f"--- PR #{pr_number}: {pr['title']}")

    if pr.get("draft", False):
        log(f"PR #{pr_number} is a draft, skipping")
        return

    head = client.get_pr_head(pr_number)
    if head is None:
        log(f"PR #{pr_number}: no commits found, skipping")
        return
    sha = head.get("sha", "")

    comments = client.get_pr_comments(pr_number)
    if has_fresh_bot_comment(comments, sha):
        log(f"PR #{pr_number}: CI comment already covers this commit and bot version, skipping")
        return

    head_repo = pr["head"]["repo"] or {}
    repo_path = head_repo.get("full_name")  # "owner/repo" on GitCode (no owner.login)
    if not repo_path:
        log(f"PR #{pr_number}: head repository unavailable, skipping")
        return

    repo_dir = Path(args.xlite_dir)
    try:
        sha = fetch_pr_head(repo_dir, repo_path, pr["head"]["ref"], pr_number)
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        log(f"PR #{pr_number}: failed to fetch head: {e}")
        return

    wt = prepare_worktree(repo_dir, sha, Path(args.worktree_root), pr_number)
    log(f"PR #{pr_number}: checking out {sha[:8]} in {wt}")
    try:
        static_check = run_static_checks(wt)
        static_passed = static_check.status == "passed"
        checks = [static_check]

        if static_check.status == "skipped":
            checks.append(
                Check(
                    "vllm-ascend integration check",
                    "skipped",
                    note="requires the pre-commit setup; rebase onto master",
                )
            )
            checks.append(Check("accuracy tests", "skipped", note="static checks did not run"))
        else:
            vllm_check = run_vllm_ascend_check(wt, args.vllm_ascend_dir, args.vllm_dir)
            checks.append(vllm_check)
            if static_passed and vllm_check.status == "passed":
                checks.append(run_accuracy_tests(wt))
            else:
                checks.append(Check("accuracy tests", "skipped", note="static checks failed"))

        body = build_comment(sha, checks)
        if args.dry_run:
            log(f"PR #{pr_number}: dry run, comment would be:\n{body}")
        elif client.post_pr_comment(pr_number, body):
            log(f"PR #{pr_number}: posted CI comment")
    finally:
        if args.keep_worktrees:
            log(f"PR #{pr_number}: keeping worktree {wt}")
        else:
            remove_worktree(repo_dir, wt)


def main() -> int:
    parser = argparse.ArgumentParser(description="GVirt PR CI bot")
    parser.add_argument(
        "--token", "-t", default=os.environ.get("GITCODE_TOKEN"), help="GitCode API token (or GITCODE_TOKEN)"
    )
    parser.add_argument("--xlite-dir", "-x", default=DEFAULT_XLITE_DIR, help="Path to the GVirt monorepo checkout")
    parser.add_argument(
        "--vllm-ascend-dir", "-a", default=DEFAULT_VLLM_ASCEND_DIR, help="Path to the vllm-ascend checkout"
    )
    parser.add_argument("--vllm-dir", "-v", default=DEFAULT_VLLM_DIR, help="Path to the vllm checkout")
    parser.add_argument("--pr", "-p", type=int, nargs="+", help="Process specific PR number(s) instead of all open PRs")
    parser.add_argument("--dry-run", "-d", action="store_true", help="Print comments instead of posting them")
    parser.add_argument(
        "--keep-worktrees", "-k", action="store_true", help="Keep PR worktrees after processing (debugging)"
    )
    parser.add_argument(
        "--worktree-root",
        "-w",
        default=None,
        help="Directory for PR worktrees (default: <xlite-dir>/../.gvirt-ci-worktrees)",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error("--token or GITCODE_TOKEN is required")
    if args.worktree_root is None:
        # NOT a dot-directory: CANN's recompile_binary.py installs objs mirrored
        # under the build path and globs them with glob("**/*.o"), which skips
        # dot-prefixed components — a hidden worktree path (e.g.
        # .gvirt-ci-worktrees, .cache) makes every kernel-library link fail
        # with TooFewObj and no compiler error.
        args.worktree_root = str(Path(args.xlite_dir).parent / "gvirt-ci-worktrees")
    if any(part.startswith(".") for part in Path(args.worktree_root).resolve().parts[1:]):
        parser.error(
            "--worktree-root must not contain dot-prefixed path components "
            f"(CANN's recompile_binary.py glob skips hidden dirs): {args.worktree_root}"
        )

    client = GitCodeClient(args.token)
    log(f"Repository: {REPO_OWNER}/{REPO_NAME}")

    if args.pr is not None:
        for number in args.pr:
            pr = client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{number}")
            if pr:
                process_pr(client, pr, args)
            else:
                log(f"PR #{number}: not found, skipping")
        return 0

    prs = client.get_open_prs()
    log(f"Found {len(prs)} open PR(s)")
    for pr in prs:
        try:
            process_pr(client, pr, args)
        except Exception as e:  # keep the loop alive for the remaining PRs
            log(f"PR #{pr.get('number')}: unexpected error: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
