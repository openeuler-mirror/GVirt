#!/usr/bin/env python3
"""GitCode PR CI bot for GVirt.

For every open non-draft PR: choose the CI harness copy to trust — the PR's
own `.gitcode/run_ci_checks.py` when the PR author is a `.gitcode` owner
or an owner commented `/ci` on the PR, otherwise the latest master copy —
run it against a per-PR worktree, and post one comment with per-check
statuses and every check's log in collapsible sections — failed checks
open, passed collapsed — capped to a tail of lines. Posted output is
vetted: proxy settings, URL credentials, and secret-shaped variable
assignments are redacted.

The bot never writes into a developer checkout: every fetch and worktree
lives in a dedicated bare repository (ci-refs.git) under the worktree
root. PR-head fetches from contributor forks are shallow (--depth 1 —
nothing in the pipeline walks history) and their grafts stay confined to
that throwaway repo; the trusted master ref is a full fetch.

The API token never leaves this process: every secret-shaped environment
variable (including GITCODE_TOKEN) is removed before any subprocess runs,
so git fetches, the harness, and the PR's build and test steps cannot see
or print it — even when rc startup files export it for every shell.

Version tracking: a comment records the checked head sha, the SHA-256
digest of the harness file set (every text file under `.gitcode`
recursively — nested directories included — except the `[harness]
digest_exclude` list, plus the `digest_include` entries kept outside
`.gitcode` such as xlite/.pre-commit-config.yaml; paths and names
matter, the combination is order-independent, symlinked entries resolve
to the real file, and a binary in the set is an error), and the commit
the harness came from. A comment is fresh when head sha and digest match
the current target, whatever bot account posted it — a second instance
acknowledges the first's verdict instead of duplicating it; when only the
recorded commit moved (identical content), the results still hold and the
next real change posts a new comment. The digest is content-based, so freshness never needs to
resolve a commit from the git graph (rebased or vanished commits still
compare); the recorded commit stays as a hyperlinked graph reference.
pr_ci_bot.py itself is excluded from the digest: it is the entry script
and the deployed copy defines the behavior, so its hash is the marker's
version field (re-checks fire on deploy, not on master merges).

Forced re-checks: a `/rerun` comment from a harness owner or the PR
author re-checks the PR even
when the existing comment is fresh — exactly once, because the posted
comment records the triggering comment id in its marker and only a
newer `/rerun` triggers again.

Check suite: every run checks the static pre-commit set, the vllm-ascend
integration mypy check (fast: one bridge file with --follow-imports=skip)
and the accuracy tests. `/ci all` remains reserved for slow extras that may
join later; its marker still records the wider suite, and a whole-suite run
covers a default request. All check caches live bot-side under the worktree
root and are shared across PR runs — clang-tidy verdicts (XLITE_CI_TIDY_CACHE;
identical content — a doc-only PR on current master, a rebase, a re-run —
skips the re-analysis; the key fingerprints every analysis input, so any
csrc, config or tidy change misses by content), the compiler cache
(CCACHE_DIR; replays unchanged host-C++ objects) and mypy's cache
(MYPY_CACHE_DIR) — one bounded directory each, no per-PR storage.

Comment delivery: one comment per head sha by default (advance) — the
first check for a head posts a comment, and every later re-check of the
same head (a harness or bot update, a /rerun) overwrites that comment in
place, so a thread shows one advancing verdict per commit instead of a
new comment per re-check; a new head posts a new comment. In-place edits
notify nobody, so the /rerun ack comment links the verdict it updated.
--comment-mode post (env XLITE_CI_COMMENT_MODE) posts a fresh comment
per check instead; replace always edits this account's own newest
comment — for the placeholder flow, where a watcher posts a placeholder
immediately on PR creation so the CI comment sits right under the PR
description, ahead of every other bot's and reviewer's comments.

Usage:
  python .gitcode/pr_ci_bot.py [--token <token>] [--pr <number>] [--dry-run]
  # local harness + drafts, before .gitcode is on master:
  python .gitcode/pr_ci_bot.py --local-test --dry-run --pr <number>

Prefer GITCODE_TOKEN over --token: a command-line argument is readable from
/proc/<pid>/cmdline by anything running as the same user — including the
PR code that the harness executes during device tests. Deployment should
export the token in the (scrubbed) service environment instead, and keep
/proc hardened (hidepid) where the token-bearing process runs.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import tomllib
import traceback
from datetime import datetime
from pathlib import Path

import requests

GITCODE_API_BASE = "https://gitcode.com/api/v5"
REPO_OWNER = "openeuler"
REPO_NAME = "GVirt"
COMMIT_URL = f"https://atomgit.com/{REPO_OWNER}/{REPO_NAME}/commit/"
TREE_URL = f"https://atomgit.com/{REPO_OWNER}/{REPO_NAME}/tree/"
DEFAULT_MASTER_URL = f"https://gitcode.com/{REPO_OWNER}/{REPO_NAME}.git"
HARNESS_RELATIVE_PATH = ".gitcode/run_ci_checks.py"
HARNESS_TIMEOUT = 7200  # static + mypy + build + accuracy, with slack
# Bot comments record bot version + head sha + harness digest + harness
# commit + the /rerun comment id consumed by that check (0 = none) + the
# suite that ran (ci | all). A bot or harness update (version or digest
# change) re-checks every open PR; comments with older or legacy markers
# never match.
BOT_TAG_RE = re.compile(
    r"<!-- gvirt-ci-bot:(?P<version>[0-9a-f]{12}):(?P<sha>[0-9a-f]{40}):"
    r"(?P<digest>[0-9a-f]{64}):(?P<source>[0-9a-f]{40}):(?P<request>[0-9]+):(?P<suite>ci|all) -->"
)
BOT_VERSION = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
# The bot removes every secret-shaped variable from its own environment
# before spawning anything, so no subprocess — git fetch, harness, build or
# test step — can read or print them, even when rc startup files export the
# token for every shell.
SECRET_ENV_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_?KEY)", re.IGNORECASE
)

DEFAULT_XLITE_DIR = "/workspaces/code/opencode/GVirt"
# Neutral default root for the CI machine; any writable root is accepted.
DEFAULT_WORKTREE_ROOT = "/workspaces/gvirt-ci-worktrees"

API_TIMEOUT = 30
COMMENT_MAX_LINES = 120

STATUS_EMOJI = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}


def log(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def acquire_instance_lock(root: Path):
    """One scan at a time. Two concurrent scans both judge a PR not-fresh
    before either posts (the comment-listing propagation race that
    burst-posted duplicates with the old bot) — the lock makes overlapping
    instances structurally impossible. flock releases on process death, so a
    killed scan never leaves a stale lock. Returns the held file handle, or
    None when another instance holds it."""
    root.mkdir(parents=True, exist_ok=True)
    lock_file = (root / "bot.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        return None
    return lock_file


class Check:
    def __init__(self, name: str, status: str, output: str = "", note: str = ""):
        self.name = name
        self.status = status  # passed | failed | skipped
        self.output = output
        self.note = note


def run_cmd(
    args: list[str], cwd: Path, timeout: int, env: dict | None = None
) -> tuple[str, int]:
    # start_new_session + killpg: the harness tree spawns grandchildren
    # (pre-commit → hooks → cmake/clang-tidy, torchrun → NPU workers) that
    # a plain child-kill would orphan — a timed-out accuracy run would leave
    # workers holding the NPU long after the check gave up.
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


def tail(text: str, max_lines: int = COMMENT_MAX_LINES) -> str:
    lines = text.rstrip().splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return f"... (truncated, showing last {max_lines} lines) ...\n" + "\n".join(
        lines[-max_lines:]
    )


class GitCodeClient:
    def __init__(self, token: str, login: str | None = None):
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        # Freshness acknowledges any bot account's marker comment
        # (cross-instance convergence), so the login is only needed to
        # target this account's own comments in replace mode. --login
        # supplies it for tokens whose GET /user is scope-blocked
        # (fine-grained PATs often exclude the user-info scope);
        # login_error records why resolution failed for the startup log.
        self.login_error = None
        if login is not None:
            self.login = login
            return
        try:
            self.login = self.get("/user").get("login")
        except requests.RequestException as e:
            self.login = None
            self.login_error = str(e)

    def request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        json_body: dict | None = None,
    ):
        response = self.session.request(
            method,
            f"{GITCODE_API_BASE}{path}",
            params=params,
            json=json_body,
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()
        return response

    def get(self, path: str, params: dict | None = None):
        return self.request("GET", path, params=params).json()

    def get_open_prs(self) -> list:
        return self.get_paged(
            f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls", params={"state": "open"}
        )

    def get_paged(
        self, path: str, params: dict | None = None, per_page: int = 50
    ) -> list:
        items, page = [], 1
        while True:
            batch = self.get(
                path, params={**(params or {}), "page": page, "per_page": per_page}
            )
            if not isinstance(batch, list) or not batch:
                return items
            items.extend(batch)
            if len(batch) < per_page:
                return items
            page += 1

    def get_pr_comments(self, pr_number: int) -> list:
        return self.get_paged(
            f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments"
        )

    def get_pr_files(self, pr_number: int) -> list:
        return self.get_paged(
            f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/files", per_page=100
        )

    def get_pr_head(self, pr_number: int) -> dict | None:
        commits = self.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/commits")
        if isinstance(commits, list) and commits:
            return commits[0]
        return None

    def post_pr_comment(self, pr_number: int, body: str) -> dict | None:
        # The created comment's id feeds the command acks, so the anchor
        # knows what this run delivered.
        try:
            return self.request(
                "POST",
                f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr_number}/comments",
                json_body={"body": body},
            ).json()
        except (requests.RequestException, ValueError) as e:
            log(f"Error posting comment to PR #{pr_number}: {e}")
            return None

    def patch_pr_comment(self, comment_id: int, body: str) -> bool:
        # The replace mode's target: the platform lets only a comment's
        # author edit it, so in-place updates stay within this account's own
        # comments — the placeholder flow's comment qualifies.
        try:
            self.request(
                "PATCH",
                f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/comments/{comment_id}",
                json_body={"body": body},
            )
            return True
        except requests.RequestException as e:
            log(f"Error editing comment {comment_id}: {e}")
            return False


def commit_link(sha: str) -> str:
    return f"[{sha[:8]}]({COMMIT_URL}{sha})"


def tree_link(sha: str, path: str = ".gitcode") -> str:
    """Link into the tree view (directory content at that exact state) —
    the right reference for what the harness ran, while commit links
    stay for commit/diff references."""
    return f"[{sha[:8]}]({TREE_URL}{sha}/{path})"


# What check logs must never post: URL credentials (proxies, mirrors,
# private indexes) and secret-shaped or proxy variable assignments.
URL_USERINFO_RE = re.compile(
    r"([a-z][a-z0-9+.-]*://)[^/@\s:]+:[^/@\s]+@", re.IGNORECASE
)
ASSIGNMENT_RE = re.compile(
    r"\b((?:https?|all|no)_proxy|proxy|[a-z0-9_]*(?:token|secret|password|passwd|credential|api_?key)[a-z0-9_]*)\s*[:=]\s*\S+",
    re.IGNORECASE,
)


def vet_output(text: str) -> str:
    """Redact proxy settings, URL credentials, and secret-shaped variable
    assignments from check output before it lands in a posted comment."""
    text = URL_USERINFO_RE.sub(r"\1***:***@", text)
    return ASSIGNMENT_RE.sub(r"\1=***", text)


def find_fresh_comment(
    comments: list,
    head_sha: str,
    digest: str,
    request: int,
    suite: str,
):
    """The bot comment that already covers this head sha, harness digest,
    /rerun request and suite — an `all` run also covers a `ci` request,
    never the other way around — with its parsed marker; None when the PR
    needs a re-check. Any bot account's comment counts, so a second instance
    (another deployment or token) acknowledges the first's verdict instead
    of duplicating it; the tradeoff is that a forged marker would also
    suppress the check — every field is publicly derivable — accepted for
    cross-instance convergence (a reviewer can see which account posted the
    verdict, and /rerun re-verifies on demand). Comments from earlier bot
    generations (no marker or a marker without the newer fields) never
    match, so the next check posts a new comment rather than treating
    them as current."""
    for comment in comments:
        m = BOT_TAG_RE.search(comment.get("body", ""))
        if (
            m
            and m.group("version") == BOT_VERSION
            and m.group("sha") == head_sha
            and m.group("digest") == digest
            and m.group("request") == str(request)
            and (m.group("suite") == suite or m.group("suite") == "all")
        ):
            return comment, m
    return None


def latest_rerun_request(comments: list, allowed: set[str]):
    """The (comment id, author) of the newest `/rerun` comment from a
    harness owner or the PR author, if any — one re-check per such
    comment: the posted CI comment records the id it consumed, and only a
    newer /rerun counts again, so the request can never loop. Newest is
    the greatest comment id — the API's ordering is not relied upon."""
    latest = None
    for comment in comments:
        author = (comment.get("user") or {}).get("login")
        is_rerun = author in allowed and (comment.get("body") or "").strip().startswith(
            "/rerun"
        )
        if is_rerun and (latest is None or comment["id"] > latest[0]):
            latest = (comment["id"], author)
    return latest


def request_consumed(comments: list, request: int) -> bool:
    """Whether a posted CI comment already covers `request`: the greatest
    request id recorded in any account's marker — cross-instance
    convergence, same as freshness. Markers from earlier generations
    never parse, so they record nothing. A request at or below the
    recorded id was served by a check that already ran; the request note
    and the acks then leave it alone instead of re-announcing it on every
    later run."""
    consumed = 0
    for c in comments:
        m = BOT_TAG_RE.search(c.get("body") or "")
        if m:
            consumed = max(consumed, int(m.group("request")))
    return request <= consumed


ACK_TAG_RE = re.compile(r"<!-- gvirt-ci-ack:(?P<id>\d+) -->")


def comment_time(comment: dict, field: str):
    """A comment's created/updated timestamp as a datetime, or None —
    advance-mode delivery overwrites results comments in place, so their
    id never advances: whether a delivery FOLLOWED a command is judged by
    time, not id (a patch moves updated_at while the id stands still)."""
    raw = comment.get(field)
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def fresh_ci_request(comments: list, owners: set[str]) -> int | None:
    """The newest owner `/ci` or `/ci all` comment that no delivery has
    answered yet — posted after every CI results comment was last
    delivered (created or in-place updated). The one-shot escape that
    re-approves this PR's own harness even when its harness base is not
    stacked on master's; a consumed grant stays consumed until a fresh
    command. Id ordering cannot decide this: an advance-mode re-check of
    an unchanged head patches the verdict in place without moving its
    id — only the comment's updated_at records that a delivery followed.
    Comments without parsable times never count as fresh."""
    last_delivery = None
    for c in find_bot_comments(comments):
        when = comment_time(c, "updated_at") or comment_time(c, "created_at")
        if when and (last_delivery is None or when > last_delivery):
            last_delivery = when
    latest = None
    for c in comments:
        body = (c.get("body") or "").strip()
        posted = comment_time(c, "created_at")
        if (
            (c.get("user") or {}).get("login") in owners
            and (ci_requested(body) or re.match(r"/ci\s+all\b", body))
            and posted is not None
            and (last_delivery is None or posted > last_delivery)
        ):
            latest = c["id"]
    return latest


def command_label(body: str) -> str | None:
    """The CI command a comment starts with — most specific first, so
    `/ci all` is not also counted as plain `/ci` — or None for a
    non-command. Mirrors the detectors the bot honors
    (latest_rerun_request, ci_requested, all_suite_requested)."""
    text = (body or "").strip()
    if re.match(r"/ci\s+all\b", text):
        return "/ci all"
    if CI_REQUEST_RE.match(text):
        return "/ci"
    if text.startswith("/rerun"):
        return "/rerun"
    return None


def ack_commands(
    client: GitCodeClient,
    pr_number: int,
    comments: list,
    owners: set[str],
    author: str,
    posted_id: int = 0,
    edited: bool = False,
    dry_run: bool = False,
    delivered: bool = False,
    posted_discussion: str | None = None,
) -> None:
    """Ack the command comments the checks already served, so a thread
    reader can tell a consumed `/rerun` (or `/ci`, `/ci all`) from one
    still waiting — under advance delivery the ack is also the only
    notification, since an in-place overwrite notifies nobody. Served has
    two readings. Right after this run delivers (delivered=True): every
    un-acked command in the fetched comments predates the check that
    just ran, so all of it is served regardless of ids — an in-place
    overwrite never moves the verdict comment's id, so ordering alone
    cannot tell. On a back-fill scan (delivered=False): a command is
    served when a results comment with a greater id exists, or when it
    is a /rerun at or below the greatest request any marker recorded
    (an overwritten verdict keeps its id but records the request it
    consumed). Each ack is one short comment linking the verdict it
    served — saying the results were updated in place when this run's
    delivery overwrote the same-sha comment — and carrying an HTML
    marker naming the served command's id. The link uses the platform's
    comment URL form (atomgit pull page, ?did=<discussion>#tid-<id>;
    the just-delivered comment's discussion comes from the posting
    response, a back-fill anchor's from the fetched comment object).
    Any account's ack counts, so
    two instances converge instead of duplicating — and a forged ack
    marker only suppresses an ack comment, never a check. Best-effort:
    a posting failure logs and moves on. Runs on every scan — fresh or
    skipped — so a missing ack back-fills on the next pass."""
    results_ids = [c["id"] for c in find_bot_comments(comments)]
    anchor = posted_id or (max(results_ids) if results_ids else 0)
    if not anchor:
        return
    max_request = 0
    for c in comments:
        m = BOT_TAG_RE.search(c.get("body") or "")
        if m:
            max_request = max(max_request, int(m.group("request")))
    acked = set()
    for c in comments:
        m = ACK_TAG_RE.search(c.get("body") or "")
        if m:
            acked.add(int(m.group("id")))
    for c in sorted(comments, key=lambda c: c["id"]):
        cid = c["id"]
        label = command_label(c.get("body") or "")
        login = (c.get("user") or {}).get("login")
        authorized = login in owners or (label == "/rerun" and login == author)
        served = delivered or cid < anchor or cid <= max_request
        if cid in acked or not served or not label or not authorized:
            continue
        anchor_comment = next((c for c in comments if c["id"] == anchor), {})
        discussion = posted_discussion or anchor_comment.get("discussion_id")
        note_url = f"https://atomgit.com/{REPO_OWNER}/{REPO_NAME}/pull/{pr_number}"
        if discussion:
            note_url += f"?did={discussion}"
        note_url += f"#tid-{anchor}"
        where = "updated in place in" if edited else "in"
        body = (
            f"✅ `{label}` (comment {cid}) consumed — CI results {where} "
            f"[comment {anchor}]({note_url})."
            f"\n<!-- gvirt-ci-ack:{cid} -->"
        )
        if dry_run:
            log(f"PR #{pr_number}: dry run, would ack `{label}` comment {cid}")
        elif client.post_pr_comment(pr_number, body):
            log(f"PR #{pr_number}: acked `{label}` comment {cid}")


def stale_reason(
    comments: list,
    sha: str,
    version: str,
    digest: str,
    request: int,
    suite: str,
) -> str | None:
    """The operator-facing reason the newest bot comment on this very head
    does not count as fresh — None when there is no comment to explain.
    Answers 'it re-checked despite its comment': a differing version is a
    new bot build re-verifying by design, a differing digest a harness
    update, a differing request a fresh /rerun."""
    newest = None
    for c in comments:
        m = BOT_TAG_RE.search(c.get("body") or "")
        if (
            m
            and m.group("sha") == sha
            and (newest is None or c.get("id", 0) > newest[0])
        ):
            newest = (c.get("id", 0), m)
    if newest is None:
        return None
    m = newest[1]
    fields = []
    if m.group("version") != version:
        fields.append(f"version {m.group('version')}!={version}")
    if m.group("digest") != digest:
        fields.append("harness digest")
    if m.group("request") != str(request):
        fields.append(f"request {m.group('request')}!={request}")
    if not (m.group("suite") == suite or m.group("suite") == "all"):
        fields.append(f"suite {m.group('suite')}!={suite}")
    return ", ".join(fields) if fields else None


CI_REQUEST_RE = re.compile(r"/ci\b")


def ci_requested(body: str) -> bool:
    """A `/ci` command at the comment's start — word-bounded, so `/cider`
    or `/closing` never grants harness trust."""
    return bool(CI_REQUEST_RE.match(body.strip()))


def all_suite_requested(comments: list, owners: set[str]) -> bool:
    """Whether a harness owner commented `/ci all`: mark the PR for the
    whole suite — slow extras beyond the default set, when any exist.
    Plain `/ci` and `/ci all` both grant PR-harness trust via ci_requested
    in process_pr."""
    return any(
        (c.get("user") or {}).get("login") in owners
        and re.match(r"/ci\s+all\b", (c.get("body") or "").strip())
        for c in comments
    )


def touches_gitcode(files: list) -> bool:
    """Whether the PR's cumulative diff changes anything under .gitcode/."""
    return any(f.get("filename", "").startswith(".gitcode/") for f in files)


def find_bot_comments(comments: list) -> list:
    """All comments authored by any bot generation, in API order."""
    return [c for c in comments if "<!-- gvirt-ci-bot:" in (c.get("body") or "")]


def find_sha_comment(comments: list, sha: str, login: str):
    """This account's newest comment whose marker records `sha` — the
    advance mode's overwrite target. Another account's same-sha comment is
    never a target: the platform allows only a comment's author to edit
    it, so a re-check by a second account falls back to posting."""
    own = []
    for c in comments:
        m = BOT_TAG_RE.search(c.get("body") or "")
        if m and m.group("sha") == sha and (c.get("user") or {}).get("login") == login:
            own.append(c)
    return max(own, key=lambda c: c["id"]) if own else None


def post_comment(
    client: GitCodeClient,
    pr_number: int,
    bot_comments: list,
    body: str,
    mode: str,
    sha: str,
) -> tuple[int, bool, str | None] | None:
    """Deliver the CI results, returning (comment id, updated in place,
    discussion id) —
    the id anchors the command acks; None when delivery failed. The
    default (advance) keeps one comment per head sha: the first check for
    a head posts, and every later re-check of the SAME head — a harness or
    bot update, a /rerun — overwrites that comment in place, so the thread
    shows one advancing verdict per commit instead of a new comment per
    re-check. A new head posts a new comment. The post mode posts a fresh
    comment per check (the old default: in-place edits notify nobody and
    get folded in long threads — the ack comment is what notifies under
    advance). The replace mode is the placeholder flow: a watcher posts a
    placeholder immediately on PR creation — before any other bot or
    reviewer comments — so it sits right under the PR description, and
    every check edits that account's own newest comment (falls back to
    posting when the account owns none or the edit fails). An advance-mode
    edit that fails also falls back to posting: a second same-sha comment
    beats a lost verdict, and the next same-sha delivery targets the
    newest one, so the invariant self-heals. Duplicate delivery for an
    unchanged, still-fresh head is prevented by find_fresh_comment before
    any check runs, in every mode."""
    target = None
    if mode == "replace":
        own = [
            c
            for c in bot_comments
            if (c.get("user") or {}).get("login") == client.login
        ]
        if own:
            target = max(own, key=lambda c: c["id"])
    elif mode == "advance":
        target = find_sha_comment(bot_comments, sha, client.login)
    if target and client.patch_pr_comment(target["id"], body):
        log(f"PR #{pr_number}: updated CI comment {target['id']} in place")
        return target["id"], True, target.get("discussion_id")
    created = client.post_pr_comment(pr_number, body)
    if not created:
        return None
    cid = created.get("id") if isinstance(created, dict) else None
    if cid:
        log(f"PR #{pr_number}: posted CI comment {cid}")
    else:
        log(f"PR #{pr_number}: posted CI comment")
    return (
        cid,
        False,
        created.get("discussion_id") if isinstance(created, dict) else None,
    )


def load_digest_config(master_wt: Path) -> tuple[set[str], list[str]]:
    """Digest sets — read from the master copy only, so a PR cannot edit the
    exclusion or inclusion lists to hide its own harness edits."""
    config = master_wt / ".gitcode" / "ci.toml"
    if not config.exists():
        return {"CODEOWNERS"}, []
    with open(config, "rb") as f:
        harness = tomllib.load(f).get("harness", {})
        return set(harness.get("digest_exclude", ["CODEOWNERS"])), list(
            harness.get("digest_include", [])
        )


def _digest_bytes(path: Path, rel: str) -> bytes:
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        raise RuntimeError(
            f"binary file in the harness digest set: {rel} — only text scripts/configs belong there"
        )
    return data


def gitcode_digest(tree: Path, exclude: set[str], include_outside: list[str]) -> str:
    """SHA-256 over the harness files' (path, content) pairs — the
    content-based harness version key. The set is every file under
    <tree>/.gitcode (recursively — nested directories included) except the
    exclude list, plus the root-relative include_outside entries (harness
    files kept outside .gitcode, e.g. xlite/.pre-commit-config.yaml;
    symlinked entries resolve to the real file). Paths and names matter;
    the combination is order-independent; identical content stays fresh
    across rebases and compares without resolving any commit."""

    def entry_hash(rel: str, data: bytes) -> str:
        return hashlib.sha256(
            rel.encode() + b"\0" + hashlib.sha256(data).digest()
        ).hexdigest()

    entries: dict[str, str] = {}
    root = tree / ".gitcode"
    if root.is_dir():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel = path.relative_to(root).as_posix()
            if rel in exclude or path.name in exclude:
                continue
            if "__pycache__" in path.parts:
                continue  # bytecode caches are derived, not harness content
            entries[rel] = entry_hash(rel, _digest_bytes(path, rel))
    for rel in include_outside:
        resolved = (tree / rel).resolve()  # symlinks: hash the real file
        entries[rel] = entry_hash(rel, _digest_bytes(resolved, rel))
    return hashlib.sha256("".join(sorted(entries.values())).encode()).hexdigest()


def gitcode_owners(master_wt: Path) -> set[str]:
    """Owners of the CI harness — read from the master worktree only, never
    from a PR tree (a PR must not grant its own harness trust)."""
    owners_file = master_wt / ".gitcode" / "CODEOWNERS"
    if not owners_file.exists():
        return set()
    for line in owners_file.read_text().splitlines():
        if line.strip().startswith("/.gitcode"):
            return set(re.findall(r"@([\w.-]+)", line))
    return set()


def configure_shared_caches(root: Path) -> dict[str, str]:
    """Bot-side cache locations, shared across PR runs (env overridable):
    one content-keyed directory each for the tidy verdicts, the compiler
    cache and mypy — no per-PR storage to cap or evict. Set in this
    process's environment so the harness and every check inherit them."""
    caches = {
        "XLITE_CI_TIDY_CACHE": str(root / "tidy-cache"),
        "CCACHE_DIR": str(root / "ccache"),
        "MYPY_CACHE_DIR": str(root / "mypy-cache"),
    }
    for var, value in caches.items():
        os.environ.setdefault(var, value)
    return caches


def scrub_env() -> int:
    """Remove every secret-shaped variable from this process's environment
    so the whole subprocess tree — git fetches, the harness, the PR's build
    and test steps — inherits a clean environment. The token is visible only
    to this process, via the arg/env parsing above."""
    secrets = [k for k in os.environ if SECRET_ENV_RE.search(k)]
    for key in secrets:
        os.environ.pop(key, None)
    return len(secrets)


def newest_harness_commit(
    wt: Path, master_sha: str, include_outside: list[str]
) -> str | None:
    """The newest commit at or below master_sha that changed the harness
    files — `.gitcode` plus the digest's outside includes — the harness
    anchor: everything after it on master is harness-clean, so a PR head
    containing it carries the current harness and one that does not is
    stacked on an outdated harness base. Excluded digest files (e.g.
    CODEOWNERS) count here but not in the digest: a rarer, milder
    over-trigger — the PR falls back to master's identical-content
    harness copy plus the rebase note. None when no harness commit exists
    (nothing to be stacked on)."""
    out = git(
        wt, ["log", "-1", "--format=%H", master_sha, "--", ".gitcode", *include_outside]
    )
    return out or None


def git(repo_dir: Path, args: list[str], timeout: int = 300) -> str:
    result = subprocess.run(
        ["git", *args],
        check=False,
        cwd=repo_dir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def ensure_refs_repo(worktree_root: Path) -> Path:
    """Dedicated bare repository for every fetch and worktree the bot
    performs — the bot never writes into any developer checkout. PR-head
    fetches are shallow and their grafts stay confined here."""
    worktree_root.mkdir(parents=True, exist_ok=True)
    refs_repo = worktree_root / "ci-refs.git"
    if not refs_repo.is_dir():
        git(worktree_root, ["init", "--bare", str(refs_repo)])
    return refs_repo


def fetch_ref(
    repo_dir: Path, url: str, ref: str, local_ref: str, shallow: bool = False
) -> str:
    """Fetch a remote ref into a bot-owned ref of the dedicated refs repo
    (race-free) and return its sha. PR-head fetches use --depth 1: nothing
    in the pipeline walks history (freshness and the digest are
    content-based), so this bounds the transfer and the object growth in
    the throwaway repo; the trusted master ref stays a full fetch."""
    args = ["fetch", "--no-tags", "--force"]
    if shallow:
        args += ["--depth", "1"]
    git(repo_dir, [*args, url, f"+{ref}:{local_ref}"])
    return git(repo_dir, ["rev-parse", local_ref])


def ref_contains(
    repo_dir: Path, url: str, ref: str, local_ref: str, commit: str
) -> bool | None:
    """Whether the remote ref's history contains `commit`, from a bounded
    deeper re-fetch of the bot-owned ref — the depth-1 head fetch grafts
    the tip parentless, so ancestry needs history. True/False is the
    answer within the fetched window; None when it could not be
    established (fetch or git failure) — callers treat None
    conservatively, like base_is_stale's 'old beyond the bound'."""
    try:
        git(
            repo_dir,
            [
                "fetch",
                "--no-tags",
                "--force",
                "--depth",
                "301",
                url,
                f"+{ref}:{local_ref}",
            ],
            timeout=120,
        )
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, local_ref],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode == 0:
            return True
        if result.returncode == 1:
            return False
        raise RuntimeError(result.stderr.strip())
    except (RuntimeError, subprocess.TimeoutExpired):
        return None


def base_is_stale(
    repo_dir: Path, url: str, ref: str, local_ref: str, master_sha: str
) -> bool:
    """Whether the PR branched from a master older than the current tip.
    Failed PRs only, the extra objects ride the rebase nudge, which never
    blocks; an unestablishable answer (or a history too shallow after the
    fetch) counts as stale by any measure."""
    return ref_contains(repo_dir, url, ref, local_ref, master_sha) is not True


def prepare_worktree(repo_dir: Path, sha: str, worktree_root: Path, name: str) -> Path:
    worktree_root.mkdir(parents=True, exist_ok=True)
    wt = worktree_root / name
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


def ensure_master_worktree(repo_dir: Path, worktree_root: Path, sha: str) -> Path:
    """Persistent worktree at the latest master head: the trusted source of
    the harness and of CODEOWNERS."""
    wt = worktree_root / "master"
    if not wt.exists():
        worktree_root.mkdir(parents=True, exist_ok=True)
        git(repo_dir, ["worktree", "add", "--detach", str(wt), sha])
    elif git(wt, ["rev-parse", "HEAD"]) != sha:
        try:
            git(wt, ["checkout", "--detach", sha])
        except RuntimeError:
            remove_worktree(repo_dir, wt)
            git(repo_dir, ["worktree", "add", "--detach", str(wt), sha])
    return wt


def run_harness(script: Path, wt: Path, args: argparse.Namespace) -> list[Check]:
    # Bot-side results path, never inside the PR-controlled worktree: a PR
    # could pre-plant a forged all-passed file that a harness run dying
    # before its own write would then leave behind. Unlinked first, so the
    # file can only hold this run's results.
    json_out = Path(args.worktree_root) / "ci-results" / f"{wt.name}.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.unlink(missing_ok=True)
    # Runner-local vllm paths live in the harness's ci.toml [integration]
    # section; explicit flags (or CI-machine CLI overrides) win over it.
    cmd = [
        sys.executable,
        str(script),
        "--worktree",
        str(wt),
        "--checks",
        "static,integration,accuracy",
    ]
    if args.vllm_ascend_dir:
        cmd += ["--vllm-ascend-dir", args.vllm_ascend_dir]
    if args.vllm_dir:
        cmd += ["--vllm-dir", args.vllm_dir]
    cmd += ["--json-out", str(json_out)]
    output, returncode = run_cmd(cmd, wt, HARNESS_TIMEOUT)
    with contextlib.suppress(OSError, json.JSONDecodeError, KeyError, TypeError):
        results = json.loads(json_out.read_text())["checks"]
        json_out.unlink(missing_ok=True)  # consumed; don't accumulate per-sha files
        return [
            Check(c["name"], c["status"], c.get("output", ""), c.get("note", ""))
            for c in results
        ]
    log(f"harness produced no results (exit {returncode})")
    return [
        Check("ci harness", "failed", output, note="harness did not produce results")
    ]


def build_comment(
    sha: str,
    checks: list[Check],
    harness: str,
    notes: list[str],
    digest: str,
    source: str,
    request: int,
    suite: str,
) -> str:
    # A status outside the known trio can only come from a corrupted harness
    # result; failing closed beats a KeyError on the emoji lookup below.
    for check in checks:
        if check.status not in STATUS_EMOJI:
            check.status = "failed"
    lines = [
        f"### CI results for [`{sha[:8]}`]({COMMIT_URL}{sha})",
        "",
        "| Check | Result |",
        "| --- | --- |",
    ]
    for check in checks:
        # Notes come from harness output (pre-commit hook names included),
        # so they are vetted like any other posted text, flattened to one
        # line, and stripped of pipes to keep the table well-formed.
        result = f"{STATUS_EMOJI[check.status]} {check.status.capitalize()}"
        if check.note:
            note = " ".join(vet_output(check.note).replace("|", " ").split())
            result += f" — {note}"
        lines.append(f"| {check.name} | {result} |")
    lines.append("")
    lines.append(f"Check harness: {harness}")
    # Standing reminder of what the bot reacts to, collapsed to stay out of
    # the way: the table and check sections above are what reviewers read.
    lines.append("")
    lines.append(
        "<details>\n<summary>CI commands</summary>\n\n"
        "- `/ci` — a `.gitcode` owner: check this PR with its own harness copy\n"
        "- `/ci all` — additionally run the slow vllm-ascend integration mypy check (sticky for the PR)\n"
        "- `/rerun` — a `.gitcode` owner or the PR author: force one re-check even when this comment is fresh\n"
        "\n</details>"
    )
    if any(check.status == "failed" for check in checks):
        notes = [
            "Fix the findings, then push a new commit or comment `/rerun` to re-check.",
            *notes,
        ]
    for note in notes:
        lines.append(f"\n> {note}")
    body = "\n".join(lines)

    # Verbose per-check output: every check gets a collapsible section with
    # its (vetted) log — failed checks first and open, passed ones collapsed.
    ordered = [c for c in checks if c.status == "failed"] + [
        c for c in checks if c.status != "failed"
    ]
    for check in ordered:
        out = tail(vet_output(check.output)) if check.output else ""
        if not out:
            continue
        # The output may itself contain triple-backtick fences (e.g. diffs of
        # markdown files); out-fence it so the comment stays well-formed.
        longest_run = max((len(m.group(0)) for m in re.finditer(r"`+", out)), default=0)
        fence = "`" * max(3, longest_run + 1)
        open_attr = " open" if check.status == "failed" else ""
        summary = f"{STATUS_EMOJI[check.status]} {check.name}"
        body += f"\n\n<details{open_attr}>\n<summary>{summary}</summary>\n\n{fence}text\n{out}\n{fence}\n\n</details>"

    body += f"\n\n<!-- gvirt-ci-bot:{BOT_VERSION}:{sha}:{digest}:{source}:{request}:{suite} -->\n*Posted automatically by the GVirt CI bot; it reruns on every new commit, bot update, CI harness update, or a `/rerun` comment from a codeowner or the PR author.*\n"
    return body


def process_pr(
    client: GitCodeClient, pr: dict, args: argparse.Namespace, master: dict
) -> None:
    pr_number = pr["number"]
    log(f"--- PR #{pr_number}: {pr['title']}")

    if pr.get("draft", False) and not args.local_test:
        log(f"PR #{pr_number}: draft, skipping")
        return

    head = client.get_pr_head(pr_number)
    if head is None:
        log(f"PR #{pr_number}: no commits found, skipping")
        return
    head_sha = head.get("sha", "")

    comments = client.get_pr_comments(pr_number)
    author = (pr.get("user") or {}).get("login") or ""
    ci_approved = any(
        (c.get("user") or {}).get("login") in master["owners"]
        and ci_requested(c.get("body") or "")
        for c in comments
    )
    rerun = latest_rerun_request(comments, master["owners"] | {author})
    request = rerun[0] if rerun else 0
    notes = []
    if rerun and not request_consumed(comments, request):
        notes.append(f"Re-check requested by @{rerun[1]} (comment {request}).")
    trusted = author in master["owners"] or ci_approved
    fresh_ci = fresh_ci_request(comments, master["owners"])
    suite = "all" if all_suite_requested(comments, master["owners"]) else "ci"
    log(
        f"PR #{pr_number}: author {author!r}, harness trust: {'PR copy' if trusted else 'master copy'}"
        + (f", re-check requested in comment {request}" if notes else "")
        + (f", fresh /ci grant in comment {fresh_ci}" if fresh_ci else "")
        + (", whole suite requested" if suite == "all" else "")
    )

    if not trusted and find_fresh_comment(
        comments, head_sha, master["digest"], request, suite
    ):
        log(
            f"PR #{pr_number}: CI comment already covers this commit and harness, skipping"
        )
        ack_commands(
            client,
            pr_number,
            comments,
            master["owners"],
            author,
            0,
            False,
            args.dry_run,
        )
        return

    head_repo = pr["head"]["repo"] or {}
    repo_path = head_repo.get("full_name")  # "owner/repo" on GitCode (no owner.login)
    if not repo_path:
        log(f"PR #{pr_number}: head repository unavailable, skipping")
        return

    refs_repo = Path(args.refs_repo)
    try:
        sha = fetch_ref(
            refs_repo,
            f"https://gitcode.com/{repo_path}.git",
            pr["head"]["ref"],
            f"refs/prbot/pr-{pr_number}",
            shallow=True,
        )
    except (RuntimeError, subprocess.TimeoutExpired) as e:
        log(f"PR #{pr_number}: failed to fetch head: {e}")
        return

    wt = prepare_worktree(
        refs_repo, sha, Path(args.worktree_root), f"pr-{pr_number}-{sha[:8]}"
    )
    log(f"PR #{pr_number}: checking out {sha[:8]} in {wt}")
    try:
        pr_script = wt / HARNESS_RELATIVE_PATH
        # The PR's own harness only counts when stacked on master's: an
        # identical digest (same content), or the head contains master's
        # newest harness commit (the PR's own harness changes, if any, sit
        # on top of master's current harness). A stale-based harness PR
        # would otherwise test its changes on an outdated base. An
        # unconsumed /ci or /ci all re-approves the PR's copy anyway.
        pr_digest = None
        stacked = None
        if pr_script.exists():
            pr_digest = gitcode_digest(
                wt, master["digest_exclude"], master["digest_include"]
            )
            stacked = (
                True
                if not master["harness_anchor"]
                else pr_digest == master["digest"]
                or bool(
                    ref_contains(
                        Path(args.refs_repo),
                        f"https://gitcode.com/{repo_path}.git",
                        pr["head"]["ref"],
                        f"refs/prbot/pr-{pr_number}",
                        master["harness_anchor"],
                    )
                )
            )
        if pr_script.exists() and trusted and (stacked or fresh_ci):
            script, source, digest = pr_script, sha, pr_digest
            harness = f"this PR's `.gitcode` @ {tree_link(sha)} (codeowner-approved)"
        else:
            script, source, digest = master["script"], master["sha"], master["digest"]
            harness = f"master `.gitcode` @ {tree_link(master['sha'])}"
        if stacked is False and not (trusted and fresh_ci):
            notes.append(
                f"This PR's harness base predates the newest harness change on master "
                f"({commit_link(master['harness_anchor'])}); the checks ran master's current harness. "
                "Rebase onto the latest master to pick it up, or have a `.gitcode` owner "
                "comment `/ci` to re-check with this PR's harness."
            )
        log(
            f"PR #{pr_number}: harness: {'PR copy' if script == pr_script else 'master copy'}"
            + (
                f" (base behind anchor {master['harness_anchor'][:8]})"
                if stacked is False
                else ""
            )
        )
        if find_fresh_comment(comments, sha, digest, request, suite):
            log(
                f"PR #{pr_number}: CI comment already covers this commit and harness, skipping"
            )
            ack_commands(
                client,
                pr_number,
                comments,
                master["owners"],
                author,
                0,
                False,
                args.dry_run,
            )
            return
        reason = stale_reason(comments, sha, BOT_VERSION, digest, request, suite)
        if reason:
            log(f"PR #{pr_number}: re-checking {sha[:8]} (stale comment: {reason})")

        checks = run_harness(script, wt, args)
        if script == master["script"] and touches_gitcode(
            client.get_pr_files(pr_number)
        ):
            notes.append(
                f"This PR updates `.gitcode`, but the checks ran the trusted master copy at {tree_link(master['sha'])}. "
                "Ask a `.gitcode` owner to comment `/ci` to re-check with this PR's harness."
            )
        if any(check.status == "failed" for check in checks) and base_is_stale(
            Path(args.refs_repo),
            f"https://gitcode.com/{repo_path}.git",
            pr["head"]["ref"],
            f"refs/prbot/pr-{pr_number}",
            master["sha"],
        ):
            notes.append(
                f"This PR branched from a master older than the current tip ({master['sha'][:8]}) — "
                "the failing findings may come from the base rather than this PR's changes. "
                "Rebase onto the latest master and push to clear them."
            )

        body = build_comment(
            sha, checks, harness, notes, digest, source, request, suite
        )
        if args.dry_run:
            log(f"PR #{pr_number}: dry run, comment would be:\n{body}")
            ack_commands(
                client, pr_number, comments, master["owners"], author, 0, False, True
            )
        else:
            posted = post_comment(
                client,
                pr_number,
                find_bot_comments(comments),
                body,
                args.comment_mode,
                sha,
            )
            ack_commands(
                client,
                pr_number,
                comments,
                master["owners"],
                author,
                posted[0] if posted else 0,
                bool(posted and posted[1]),
                False,
                posted is not None,
                posted[2] if posted else None,
            )
    finally:
        if args.keep_worktrees:
            log(f"PR #{pr_number}: keeping worktree {wt}")
        else:
            remove_worktree(refs_repo, wt)


def main() -> int:
    parser = argparse.ArgumentParser(description="GVirt PR CI bot")
    parser.add_argument(
        "--token",
        "-t",
        default=os.environ.get("GITCODE_TOKEN"),
        help="GitCode API token (or GITCODE_TOKEN)",
    )
    parser.add_argument(
        "--login",
        default=None,
        help="The account this token posts as, for tokens whose GET /user is "
        "blocked by scope (fine-grained PATs often exclude the user-info "
        "scope) — used only to target this account's own comments in "
        "replace mode",
    )
    parser.add_argument(
        "--comment-mode",
        choices=("advance", "post", "replace"),
        default=os.environ.get("XLITE_CI_COMMENT_MODE", "advance"),
        help="advance: one comment per head sha — the first check for a head "
        "posts, later re-checks of the same head (harness update, /rerun) "
        "overwrite it in place (default). post: a fresh comment per check. "
        "replace: always edit this account's own newest comment in place — "
        "for the placeholder flow, where a watcher posts a placeholder on "
        "PR creation so the results sit right under the PR description "
        "(or XLITE_CI_COMMENT_MODE)",
    )
    parser.add_argument(
        "--xlite-dir",
        "-x",
        default=DEFAULT_XLITE_DIR,
        help="Local GVirt checkout, read-only and only used by --local-test",
    )
    parser.add_argument(
        "--vllm-ascend-dir",
        "-a",
        default=None,
        help="Path to the vllm-ascend checkout (default: the harness's ci.toml [integration])",
    )
    parser.add_argument(
        "--vllm-dir",
        "-v",
        default=None,
        help="Path to the vllm checkout (default: the harness's ci.toml [integration])",
    )
    parser.add_argument(
        "--master-url",
        "-m",
        default=DEFAULT_MASTER_URL,
        help="Repository URL to fetch the trusted master head from",
    )
    parser.add_argument(
        "--master-ref", default="master", help="Ref to track as the trusted master head"
    )
    parser.add_argument(
        "--local-test",
        "-l",
        action="store_true",
        help="Use the local checkout's .gitcode as the trusted harness (no master "
        "fetch — reads the working tree, so uncommitted harness edits are tested "
        "too) and include draft PRs. For testing and pre-merge operation; the "
        "worktree-root guards still apply",
    )
    parser.add_argument(
        "--pr",
        "-p",
        type=int,
        nargs="+",
        help="Process specific PR number(s) instead of all open PRs",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Print comments instead of posting them",
    )
    parser.add_argument(
        "--keep-worktrees",
        "-k",
        action="store_true",
        help="Keep PR worktrees after processing (debugging)",
    )
    parser.add_argument(
        "--worktree-root",
        "-w",
        default=None,
        help="Directory for bot worktrees and the shared check caches (tidy verdicts, "
        "ccache, mypy). Keep it on local disk: the caches are small-file IO-heavy "
        "and per-PR build trees go through it (default: <xlite-dir>/../gvirt-ci-worktrees)",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error("--token or GITCODE_TOKEN is required")
    log(f"Scrubbed {scrub_env()} secret-shaped environment variable(s)")
    if args.worktree_root is None:
        if args.local_test:
            # Local scratch next to the checkout; the neutral-root default
            # (/workspaces) is a CI-machine assumption.
            args.worktree_root = str(Path(args.xlite_dir).parent / "gvirt-ci-worktrees")
        else:
            # Neutral default for the CI machine; any writable root works.
            # Also NOT a dot-directory: CANN's recompile_binary.py installs
            # objs mirrored under the build path and globs them with
            # glob("**/*.o"), which skips dot-prefixed components — a hidden
            # worktree path makes every kernel-library link fail with
            # TooFewObj and no compiler error.
            args.worktree_root = DEFAULT_WORKTREE_ROOT
    root = Path(args.worktree_root).resolve()
    if any(part.startswith(".") for part in root.parts[1:]):
        parser.error(
            "--worktree-root must not contain dot-prefixed path components "
            f"(CANN's recompile_binary.py glob skips hidden dirs): {args.worktree_root}"
        )
    # Absolute from here on: the bot runs from any working directory, and a
    # relative --worktree-root would otherwise be mkdir'd against the shell
    # cwd but resolved by git against the repo dir.
    args.worktree_root = str(root)
    lock = acquire_instance_lock(root)
    if lock is None:
        log(f"another scan holds {root / 'bot.lock'}; exiting")
        return 0
    configure_shared_caches(root)

    client = GitCodeClient(args.token, login=args.login)
    if client.login is None:
        log(
            "WARNING: cannot resolve this token's own login "
            f"({client.login_error or 'no login in the GET /user response'}) — "
            "freshness acknowledges any bot account's comment, but replace-mode "
            "delivery degrades to posting; pass --login <account> to name it"
        )
    log(f"Repository: {REPO_OWNER}/{REPO_NAME}")

    # Every fetch and worktree happens in a dedicated bare repository under
    # the worktree root — the bot never writes into a developer checkout.
    args.refs_repo = str(ensure_refs_repo(Path(args.worktree_root)))
    if args.local_test:
        # The local checkout IS the trusted harness source: reads the
        # working tree (rev-parse below is its only git use, read-only), so
        # uncommitted harness edits are exercised too.
        master_sha = git(Path(args.xlite_dir), ["rev-parse", "HEAD"])
        master_wt = Path(args.xlite_dir)
    else:
        master_sha = fetch_ref(
            Path(args.refs_repo), args.master_url, args.master_ref, "refs/prbot/master"
        )
        master_wt = ensure_master_worktree(
            Path(args.refs_repo), Path(args.worktree_root), master_sha
        )
    digest_exclude, digest_include = load_digest_config(master_wt)
    master = {
        "sha": master_sha,
        "wt": master_wt,
        "digest": gitcode_digest(master_wt, digest_exclude, digest_include),
        "harness_anchor": newest_harness_commit(master_wt, master_sha, digest_include),
        "digest_exclude": digest_exclude,
        "digest_include": digest_include,
        "owners": gitcode_owners(master_wt),
        "script": master_wt / HARNESS_RELATIVE_PATH,
    }
    if not master["script"].exists():
        log(
            f"the trusted source ({master_wt}) has no {HARNESS_RELATIVE_PATH} — "
            "merge and deploy the .gitcode CI harness first, or pass --local-test "
            "to use this checkout's copy"
        )
        return 1
    log(
        f"Trusted master: {commit_link(master_sha)}, harness owners: {len(master['owners'])}"
    )

    if args.pr is not None:
        for number in args.pr:
            started = time.monotonic()
            try:
                pr = client.get(f"/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{number}")
            except requests.HTTPError as e:
                log(
                    f"PR #{number}: not found or inaccessible ({e.response.status_code}), skipping"
                )
                continue
            try:
                process_pr(client, pr, args, master)
            except Exception as e:  # noqa: BLE001 — one bad PR must not stop the rest
                log(f"PR #{number}: unexpected error: {e}\n{traceback.format_exc()}")
            log(f"PR #{number}: done in {time.monotonic() - started:.1f}s")
        return 0

    prs = client.get_open_prs()
    log(f"Found {len(prs)} open PR(s)")
    for pr in prs:
        started = time.monotonic()
        try:
            process_pr(client, pr, args, master)
        except Exception as e:  # noqa: BLE001 — keep the loop alive for the remaining PRs
            log(
                f"PR #{pr.get('number')}: unexpected error: {e}\n{traceback.format_exc()}"
            )
        log(f"PR #{pr.get('number')}: done in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
