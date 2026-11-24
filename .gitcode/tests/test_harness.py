"""Logic tests for the .gitcode CI harness.

Pure functions plus local scratch git repositories where the behavior under
test is git itself — no network, no NPU — covering the digest,
marker/freshness/suite, comment lifecycle, env scrubbing, output vetting,
derived-config rewriting and definition-drift detection. Run with:
pytest .gitcode/tests
"""

import argparse
import os
import sys
from pathlib import Path

import pr_ci_bot
import pytest
import run_ci_checks
import run_static_checks
from hooks import check_definition_drift

EXC = {"CODEOWNERS", "pr_ci_bot.py"}
INC = ["xlite/.pre-commit-config.yaml"]


def digest_tree(tmp_path, name="a", config="repos: []\n"):
    tree = tmp_path / name
    (tree / ".gitcode/hooks").mkdir(parents=True)
    (tree / ".gitcode/CODEOWNERS").write_text("/.gitcode @x\n")
    (tree / "xlite").mkdir()
    (tree / "xlite/.pre-commit-config.yaml").write_text(config)
    return tree


def sample_checks():
    return [
        pr_ci_bot.Check(
            "xlite static checks", "passed", "hook-a....Passed\nhook-b....Passed"
        ),
        pr_ci_bot.Check("accuracy tests", "failed", "boom\n```diff\n-x\n```", "note"),
    ]


class FakeClient:
    login = "bot"

    def __init__(self, patch_ok=True):
        self.patched, self.posted = [], []
        self.patch_ok = patch_ok

    def patch_pr_comment(self, comment_id, body):
        self.patched.append((comment_id, body))
        return self.patch_ok

    def post_pr_comment(self, pr_number, body):
        self.posted.append((pr_number, body))
        return True


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append(params)
        return FakeResponse(self.pages[int(params["page"]) - 1])


# --- digest ---------------------------------------------------------------


def test_digest_bumps_on_gitcode_change(tmp_path):
    tree = digest_tree(tmp_path)
    before = pr_ci_bot.gitcode_digest(tree, EXC, INC)
    (tree / ".gitcode/run_ci_checks.py").write_text("print(1)\n")
    (tree / ".gitcode/hooks/x.py").write_text("h\n")
    assert pr_ci_bot.gitcode_digest(tree, EXC, INC) != before


def test_digest_excluded_file_change_does_not_bump(tmp_path):
    tree = digest_tree(tmp_path)
    (tree / ".gitcode/run_ci_checks.py").write_text("print(1)\n")
    (tree / ".gitcode/hooks/x.py").write_text("h\n")
    before = pr_ci_bot.gitcode_digest(tree, EXC, INC)
    (tree / ".gitcode/CODEOWNERS").write_text("/.gitcode @y @z\n")
    assert pr_ci_bot.gitcode_digest(tree, EXC, INC) == before


def test_digest_outside_include_change_bumps(tmp_path):
    tree = digest_tree(tmp_path)
    (tree / ".gitcode/run_ci_checks.py").write_text("print(1)\n")
    (tree / ".gitcode/hooks/x.py").write_text("h\n")
    before = pr_ci_bot.gitcode_digest(tree, EXC, INC)
    (tree / "xlite/.pre-commit-config.yaml").write_text("repos: [x]\n")
    assert pr_ci_bot.gitcode_digest(tree, EXC, INC) != before
    (tree / "xlite/.pre-commit-config.yaml").write_text("repos: []\n")
    assert pr_ci_bot.gitcode_digest(tree, EXC, INC) == before


def test_digest_order_independent(tmp_path):
    tree = digest_tree(tmp_path)
    (tree / ".gitcode/run_ci_checks.py").write_text("print(1)\n")
    (tree / ".gitcode/hooks/x.py").write_text("h\n")
    same = digest_tree(tmp_path, "b")
    (same / ".gitcode/run_ci_checks.py").write_text("print(1)\n")
    (same / ".gitcode/hooks/x.py").write_text("h\n")
    (same / ".gitcode/CODEOWNERS").write_text("whatever\n")
    assert pr_ci_bot.gitcode_digest(same, EXC, INC) == pr_ci_bot.gitcode_digest(
        tree, EXC, INC
    )


def test_digest_symlink_include_resolves_to_real_content(tmp_path):
    linked = tmp_path / "c"
    (linked / "xlite").mkdir(parents=True)
    (linked / ".real-config.yaml").write_text("repos: []\n")
    (linked / "xlite/.pre-commit-config.yaml").symlink_to("../.real-config.yaml")
    plain = tmp_path / "d"
    (plain / "xlite").mkdir(parents=True)
    (plain / ".gitcode").mkdir()
    (plain / "xlite/.pre-commit-config.yaml").write_text("repos: []\n")
    assert pr_ci_bot.gitcode_digest(linked, EXC, INC) == pr_ci_bot.gitcode_digest(
        plain, EXC, INC
    )


def test_digest_binary_file_raises(tmp_path):
    tree = tmp_path / "e"
    (tree / ".gitcode").mkdir(parents=True)
    (tree / ".gitcode/logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
    with pytest.raises(RuntimeError, match="binary"):
        pr_ci_bot.gitcode_digest(tree, EXC, [])


def test_digest_entry_script_excluded(tmp_path):
    tree = tmp_path / "f"
    (tree / ".gitcode").mkdir(parents=True)
    (tree / ".gitcode/run_ci_checks.py").write_text("v1\n")
    before = pr_ci_bot.gitcode_digest(tree, EXC, [])
    (tree / ".gitcode/pr_ci_bot.py").write_text("print(2)\n")
    assert pr_ci_bot.gitcode_digest(tree, EXC, []) == before


def test_digest_config_sets_parsed(tmp_path):
    tree = tmp_path / "m"
    (tree / ".gitcode").mkdir(parents=True)
    (tree / ".gitcode/ci.toml").write_text(
        '[harness]\ndigest_exclude = ["CODEOWNERS", "NOTES.md"]\ndigest_include = ["xlite/.pre-commit-config.yaml"]\n'
    )
    assert pr_ci_bot.load_digest_config(tree) == (
        {"CODEOWNERS", "NOTES.md"},
        ["xlite/.pre-commit-config.yaml"],
    )


def test_digest_config_defaults_without_toml(tmp_path):
    tree = tmp_path / "m2"
    tree.mkdir()
    assert pr_ci_bot.load_digest_config(tree) == ({"CODEOWNERS"}, [])


# --- environment hygiene ---------------------------------------------------


def test_shared_cache_locations_defaults_and_overrides(monkeypatch, tmp_path):
    """One bot-side directory per cache, shared across PR runs; an explicit
    environment value wins (CI machine layout override)."""
    for var in ("XLITE_CI_TIDY_CACHE", "CCACHE_DIR", "MYPY_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    caches = pr_ci_bot.configure_shared_caches(tmp_path)
    assert caches == {
        "XLITE_CI_TIDY_CACHE": str(tmp_path / "tidy-cache"),
        "CCACHE_DIR": str(tmp_path / "ccache"),
        "MYPY_CACHE_DIR": str(tmp_path / "mypy-cache"),
    }
    assert os.environ["CCACHE_DIR"] == str(tmp_path / "ccache")
    monkeypatch.setenv("MYPY_CACHE_DIR", "/custom-mypy")
    pr_ci_bot.configure_shared_caches(tmp_path)
    assert os.environ["MYPY_CACHE_DIR"] == "/custom-mypy"


def test_harness_env_sets_ccache_launcher_only_when_installed(monkeypatch):
    """The compiler cache is presence-gated: a machine without ccache must
    not get a launcher cmake cannot resolve."""
    monkeypatch.setattr(
        run_ci_checks.shutil,
        "which",
        lambda name: "/usr/bin/ccache" if name == "ccache" else None,
    )
    assert run_ci_checks.check_env()["CMAKE_CXX_COMPILER_LAUNCHER"] == "ccache"
    monkeypatch.setattr(run_ci_checks.shutil, "which", lambda name: None)
    assert "CMAKE_CXX_COMPILER_LAUNCHER" not in run_ci_checks.check_env()


def test_scrub_env_removes_secret_shaped_vars(monkeypatch):
    monkeypatch.setenv("GITCODE_TOKEN", "secret")
    monkeypatch.setenv("FOO_API_KEY", "secret")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/lib")
    removed = pr_ci_bot.scrub_env()
    assert "GITCODE_TOKEN" not in os.environ
    assert "FOO_API_KEY" not in os.environ
    assert removed >= 2
    assert os.environ.get("LD_LIBRARY_PATH") == "/lib"


def test_harness_env_filters_secrets_and_anchors_paths(monkeypatch):
    monkeypatch.setenv("GITCODE_TOKEN", "secret")
    env = run_ci_checks.check_env()
    assert "GITCODE_TOKEN" not in env
    assert env["PATH"].split(os.pathsep)[0] == str(Path(sys.prefix) / "bin")
    assert "pybind11" in env["CMAKE_PREFIX_PATH"]


# --- owners, trust, suite ---------------------------------------------------


def test_owners_parsed(tmp_path):
    tree = tmp_path / "m"
    (tree / ".gitcode").mkdir(parents=True)
    (tree / ".gitcode/CODEOWNERS").write_text(
        "# comment\n/.gitcode @fusijie @duyiwei7w @sinever\n"
    )
    assert pr_ci_bot.gitcode_owners(tree) == {"fusijie", "duyiwei7w", "sinever"}


def test_ci_request_word_boundary():
    assert pr_ci_bot.ci_requested("/ci")
    assert pr_ci_bot.ci_requested("/ci all")
    assert pr_ci_bot.ci_requested("  /ci  ")
    # A word boundary only: lookalikes never grant harness trust.
    assert not pr_ci_bot.ci_requested("/cider")
    assert not pr_ci_bot.ci_requested("/closing the PR")
    assert not pr_ci_bot.ci_requested("please /ci")
    assert not pr_ci_bot.ci_requested("nice PR")


def test_owner_ci_grants_trust():
    comments = [
        {"user": {"login": "someone"}, "body": "nice PR"},
        {"user": {"login": "fusijie"}, "body": "/ci"},
        {"user": {"login": "rando"}, "body": "/ci please"},
    ]
    owners = {"fusijie"}
    assert any(
        (c.get("user") or {}).get("login") in owners
        and pr_ci_bot.ci_requested(c.get("body") or "")
        for c in comments
    )
    assert not any(
        (c.get("user") or {}).get("login") in owners
        and pr_ci_bot.ci_requested(c.get("body") or "")
        for c in [comments[0], comments[2]]
    )


def test_all_suite_requested_owner_only():
    comments = [
        {"user": {"login": "someone"}, "body": "/ci all"},
        {"user": {"login": "fusijie"}, "body": "/ci all please"},
        {"user": {"login": "fusijie"}, "body": "/ci"},
        {"user": {"login": "fusijie"}, "body": "/rerun"},
    ]
    assert pr_ci_bot.all_suite_requested(comments[1:], {"fusijie"})
    assert not pr_ci_bot.all_suite_requested(comments[2:], {"fusijie"})
    assert not pr_ci_bot.all_suite_requested(comments[:1], {"fusijie"})
    assert not pr_ci_bot.all_suite_requested(comments[3:], {"fusijie"})


def test_rerun_latest_request_wins():
    comments = [
        {"id": 50, "user": {"login": "someone"}, "body": "/rerun"},
        {"id": 51, "user": {"login": "fusijie"}, "body": "/rerun"},
        {"id": 52, "user": {"login": "duyiwei7w"}, "body": "/rerun please"},
    ]
    owners = {"fusijie", "duyiwei7w"}
    assert pr_ci_bot.latest_rerun_request(comments, owners) == (52, "duyiwei7w")
    assert pr_ci_bot.latest_rerun_request(comments[:1], owners) is None
    assert pr_ci_bot.latest_rerun_request(comments[:1], owners | {"someone"}) == (
        50,
        "someone",
    )


def test_rerun_newest_is_greatest_id_not_list_position():
    """The comments API's ordering must not decide which /rerun counts —
    descending listings (newest first) pick the same id as ascending ones."""
    descending = [
        {"id": 52, "user": {"login": "duyiwei7w"}, "body": "/rerun"},
        {"id": 51, "user": {"login": "fusijie"}, "body": "/rerun"},
    ]
    ascending = list(reversed(descending))
    owners = {"fusijie", "duyiwei7w"}
    assert pr_ci_bot.latest_rerun_request(descending, owners) == (52, "duyiwei7w")
    assert pr_ci_bot.latest_rerun_request(ascending, owners) == (52, "duyiwei7w")


def test_touches_gitcode():
    files = [
        {"filename": "xlite/csrc/op.cpp"},
        {"filename": ".gitcode/run_ci_checks.py"},
        {"filename": ".gitcode/hooks/run_clang_tidy.py"},
    ]
    assert pr_ci_bot.touches_gitcode(files)
    assert not pr_ci_bot.touches_gitcode(files[:1])


# --- marker, freshness, suite ----------------------------------------------


def test_marker_roundtrip_and_staleness():
    body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    own = [{"id": 1, "user": {"login": "bot"}, "body": body}]
    assert pr_ci_bot.find_fresh_comment(own, "a" * 40, "b" * 64, 0, "ci")
    assert not pr_ci_bot.find_fresh_comment(own, "d" * 40, "b" * 64, 0, "ci")
    legacy = [
        {"id": 1, "user": {"login": "bot"}, "body": "old <!-- gvirt-ci-bot:abc:sha -->"}
    ]
    assert not pr_ci_bot.find_fresh_comment(legacy, "a" * 40, "b" * 64, 0, "ci")


def test_freshness_tradeoff_forged_markers_count():
    """The six marker fields are all publicly derivable, and freshness counts
    any account's comment for cross-instance convergence — so a forged
    marker suppresses the check. Deliberate tradeoff: a reviewer can see
    which account posted the verdict, and /rerun re-verifies on demand."""
    body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    forged = [{"id": 7, "user": {"login": "pr-author"}, "body": body}]
    assert pr_ci_bot.find_fresh_comment(forged, "a" * 40, "b" * 64, 0, "ci")


def test_marker_records_suite_and_all_covers_ci():
    ci_body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    all_body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "all"
    )
    assert ":ci -->" in ci_body and ":all -->" in all_body
    all_comment = [{"id": 1, "user": {"login": "bot"}, "body": all_body}]
    assert pr_ci_bot.find_fresh_comment(all_comment, "a" * 40, "b" * 64, 0, "ci")
    assert pr_ci_bot.find_fresh_comment(all_comment, "a" * 40, "b" * 64, 0, "all")
    assert not pr_ci_bot.find_fresh_comment(
        [{"id": 1, "user": {"login": "bot"}, "body": ci_body}],
        "a" * 40,
        "b" * 64,
        0,
        "all",
    )


def test_rerun_one_shot_semantics():
    body_r0 = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    body_r1 = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 51, "ci"
    )
    cs0 = [{"id": 1, "user": {"login": "bot"}, "body": body_r0}]
    cs1 = [{"id": 1, "user": {"login": "bot"}, "body": body_r1}]
    assert pr_ci_bot.find_fresh_comment(cs0, "a" * 40, "b" * 64, 0, "ci")
    assert not pr_ci_bot.find_fresh_comment(cs0, "a" * 40, "b" * 64, 51, "ci")
    assert pr_ci_bot.find_fresh_comment(cs1, "a" * 40, "b" * 64, 51, "ci")
    assert not pr_ci_bot.find_fresh_comment(cs1, "a" * 40, "b" * 64, 52, "ci")


# --- comment shaping, vetting, links ----------------------------------------


def test_comment_layout():
    body = pr_ci_bot.build_comment(
        "a" * 40,
        sample_checks(),
        "master `.gitcode` @ [11111111](u)",
        [],
        "b" * 64,
        "c" * 40,
        0,
        "ci",
    )
    assert f"### CI results for [`{'a' * 8}`]({pr_ci_bot.COMMIT_URL}{'a' * 40})" in body
    assert "comment `/rerun` to re-check" in body
    failed_pos, passed_pos = body.find("<details open>"), body.find("<summary>✅")
    assert 0 < failed_pos < passed_pos
    assert "hook-a....Passed" in body and "````text" in body
    assert "Check harness: master `.gitcode` @ [11111111](u)" in body


def test_comment_gitcode_note_uses_tree_link():
    note = (
        f"This PR updates `.gitcode`, but the checks ran the trusted master copy at "
        f"{pr_ci_bot.tree_link('f' * 40)}. "
        "Ask a `.gitcode` owner to comment `/ci` to re-check with this PR's harness."
    )
    body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [note], "b" * 64, "c" * 40, 0, "ci"
    )
    assert note in body and f"{pr_ci_bot.TREE_URL}{'f' * 40}/.gitcode" in body


def test_vet_output_redactions():
    vetted = pr_ci_bot.vet_output(
        "pip install x\n"
        "http_proxy=http://proxy.corp.example:8080\n"
        "MY_API_KEY=supersecret\n"
        "fetch http://user:pass@mirror.internal/repo.git\n"
        "clean line stays"
    )
    assert "proxy.corp.example" not in vetted and "http_proxy=***" in vetted
    assert "supersecret" not in vetted and "MY_API_KEY=***" in vetted
    assert "user:pass" not in vetted and "***:***@mirror.internal" in vetted
    assert "pip install x" in vetted and "clean line stays" in vetted


def test_tree_vs_commit_link_forms():
    assert (
        pr_ci_bot.tree_link("d" * 40)
        == f"[{'d' * 8}]({pr_ci_bot.TREE_URL}{'d' * 40}/.gitcode)"
    )
    assert (
        pr_ci_bot.commit_link("e" * 40)
        == f"[{'e' * 8}]({pr_ci_bot.COMMIT_URL}{'e' * 40})"
    )


# --- comment lifecycle --------------------------------------------------------


def bot_comment(cid, login, body):
    return {"id": cid, "user": {"login": login}, "body": body}


def test_ci_results_post_fresh_by_default():
    """The default mode posts a fresh comment and never edits: the platform
    offers no comment pinning, so an in-place update would leave the results
    folded away in a long thread with no notification."""
    client = FakeClient()
    own = bot_comment(11, "bot", "<!-- gvirt-ci-bot:v:sha -->")
    pr_ci_bot.post_comment(client, 653, [own], "RESULTS", "post")
    assert client.posted == [(653, "RESULTS")] and not client.patched


def test_replace_mode_edits_own_newest_comment_in_place():
    """The placeholder flow: the placeholder (posted on PR creation, so it
    sits right under the description) is this account's own newest comment;
    each check edits the results into it. Greatest id wins, and another
    account's newer comment never becomes the target (author-only edits)."""
    client = FakeClient()
    older = bot_comment(11, "bot", "<!-- gvirt-ci-bot:v:sha -->")
    newer = bot_comment(12, "bot", "<!-- gvirt-ci-bot:w:sha -->")
    alien = bot_comment(44, "svc-bot", "<!-- gvirt-ci-bot:x:sha -->")
    pr_ci_bot.post_comment(client, 653, [newer, alien, older], "NEW", "replace")
    assert client.patched == [(12, "NEW")] and not client.posted


def test_replace_mode_posts_when_account_owns_no_comment():
    client = FakeClient()
    alien = bot_comment(44, "svc-bot", "<!-- gvirt-ci-bot:x:sha -->")
    pr_ci_bot.post_comment(client, 653, [alien], "NEW", "replace")
    assert client.posted == [(653, "NEW")] and not client.patched


def test_replace_mode_falls_back_to_posting_when_edit_fails():
    """The platform lets only a comment's author edit it, and the edit can
    fail (a race with a concurrent delete, say): the results must still be
    delivered, not lost."""
    client = FakeClient(patch_ok=False)
    own = bot_comment(11, "bot", "<!-- gvirt-ci-bot:v:sha -->")
    pr_ci_bot.post_comment(client, 653, [own], "NEW", "replace")
    assert client.posted == [(653, "NEW")]


def test_find_bot_comments_across_generations():
    """The replace target is picked only from marker-bearing comments, so
    human comments never enter the candidate set and any bot generation's
    marker comment does."""
    plain = {"id": 33, "user": {"login": "human"}, "body": "human review comment"}
    legacy = bot_comment(
        11, "bot", "### CI results ...\n<!-- gvirt-ci-bot:oldver:oldsha -->"
    )
    newest = bot_comment(22, "bot", "<!-- gvirt-ci-bot:another:sha -->")
    assert pr_ci_bot.find_bot_comments([plain, legacy, newest]) == [legacy, newest]


def test_instance_lock_admits_one_scan_at_a_time(tmp_path):
    """Two overlapping scans both judge a PR not-fresh before either posts
    (the comment-listing race that burst-posted duplicates with the old
    bot) — the lock must admit exactly one holder and release on close."""
    first = pr_ci_bot.acquire_instance_lock(tmp_path)
    assert first is not None
    assert pr_ci_bot.acquire_instance_lock(tmp_path) is None
    first.close()
    third = pr_ci_bot.acquire_instance_lock(tmp_path)
    assert third is not None
    third.close()


def test_base_is_stale(tmp_path):
    """A failed check plus an old base earns the author a rebase nudge: the
    depth-1 head fetch grafts the tip parentless, so the helper re-fetches
    bounded-deeper and asks ancestry — old bases are stale, bases carrying
    the current master tip are not."""
    import subprocess

    def sh(*cmd, cwd=None):
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)

    def rev(cwd):
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    fork = tmp_path / "fork"
    fork.mkdir()
    sh("git", "init", "-q", "-b", "master", str(fork))
    sh("git", "config", "user.email", "t@t", cwd=fork)
    sh("git", "config", "user.name", "t", cwd=fork)
    (fork / "f").write_text("base\n")
    sh("git", "add", ".", cwd=fork)
    sh("git", "commit", "-qm", "base", cwd=fork)
    sh("git", "checkout", "-qb", "old", cwd=fork)
    (fork / "g").write_text("pr\n")
    sh("git", "add", ".", cwd=fork)
    sh("git", "commit", "-qm", "pr", cwd=fork)
    sh("git", "checkout", "-q", "master", cwd=fork)
    (fork / "f").write_text("master-moved\n")
    sh("git", "commit", "-qam", "master moved", cwd=fork)
    master_sha = rev(fork)
    sh("git", "checkout", "-qb", "fresh", "master", cwd=fork)
    (fork / "h").write_text("pr2\n")
    sh("git", "add", ".", cwd=fork)
    sh("git", "commit", "-qm", "pr2", cwd=fork)

    refs = tmp_path / "ci-refs.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(refs)], check=True, capture_output=True
    )
    for ref in ("old", "fresh"):
        pr_ci_bot.fetch_ref(refs, str(fork), ref, f"refs/prbot/pr-{ref}", shallow=True)
    assert (
        pr_ci_bot.base_is_stale(refs, str(fork), "old", "refs/prbot/pr-old", master_sha)
        is True
    )
    assert (
        pr_ci_bot.base_is_stale(
            refs, str(fork), "fresh", "refs/prbot/pr-fresh", master_sha
        )
        is False
    )


def test_stale_reason_names_the_mismatched_marker_fields():
    """The re-check diagnostic: a same-head comment whose version differs
    (a new bot build re-verifying by design) must be named as such, not
    left to look like a fresh PR head."""
    body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    own = bot_comment(11, "bot", body)
    version = pr_ci_bot.BOT_TAG_RE.search(body).group("version")
    assert version != "deadbeef1234"
    reason = pr_ci_bot.stale_reason([own], "a" * 40, "deadbeef1234", "b" * 64, 0, "ci")
    assert reason and f"version {version}!=deadbeef1234" in reason
    assert pr_ci_bot.stale_reason([own], "z" * 40, version, "b" * 64, 0, "ci") is None
    assert pr_ci_bot.stale_reason([], "a" * 40, version, "b" * 64, 0, "ci") is None


def test_freshness_acknowledges_another_bot_account():
    """Cross-instance convergence: a fresh verdict posted by one bot account
    is acknowledged by any other instance, instead of every account
    re-checking and duplicating the results."""
    body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    other = bot_comment(11, "cibot1", body)
    assert (
        pr_ci_bot.find_fresh_comment([other], "a" * 40, "b" * 64, 0, "ci") is not None
    )
    # A differing version still re-verifies: the other instance runs a
    # different build, and the marker records which one checked.
    stale = bot_comment(12, "cibot1", body.replace(pr_ci_bot.BOT_VERSION, "a" * 12))
    assert pr_ci_bot.find_fresh_comment([stale], "a" * 40, "b" * 64, 0, "ci") is None


def test_client_login_override_and_resolution_failure(monkeypatch):
    """--login supplies the account for tokens whose GET /user is
    scope-blocked; a failed resolution records why, so the startup warning
    can tell the operator exactly what to fix instead of silently degrading
    replace mode."""
    monkeypatch.setattr(
        pr_ci_bot.GitCodeClient, "get", lambda self, path: {"login": "from-api"}
    )
    assert pr_ci_bot.GitCodeClient("t").login == "from-api"
    assert pr_ci_bot.GitCodeClient("t", login="manual").login == "manual"

    def boom(self, path):
        raise pr_ci_bot.requests.HTTPError("403 Forbidden")

    monkeypatch.setattr(pr_ci_bot.GitCodeClient, "get", boom)
    failed = pr_ci_bot.GitCodeClient("t")
    assert failed.login is None and "403" in failed.login_error


def test_freshness_matches_the_recorded_715_marker(monkeypatch):
    """Regression pin with the exact marker the deployed bot recorded on
    !715 (head 2064002b, trusted run, harness digest f78dcfe7): a run with
    the same version, digest, request and suite must skip — if a manual
    rerun re-checks, one of those inputs differs, not the matching logic."""
    sha = "2064002bb2b477c95232f30c14d5a7d2e3b80a40"
    digest = "f78dcfe7ea910545f06a754dd7a0b787ca528b0d3cedd8c712d7daaee0f56e6a"
    marker = f"abffe439ca84:{sha}:{digest}:{sha}:0:ci"
    comment = {
        "id": 203164004,
        "user": {"login": "lulina2"},
        "body": f"### CI results for [`{sha[:8]}`]\n<!-- gvirt-ci-bot:{marker} -->",
    }
    monkeypatch.setattr(pr_ci_bot, "BOT_VERSION", "abffe439ca84")
    assert pr_ci_bot.find_fresh_comment([comment], sha, digest, 0, "ci") is not None
    assert (
        pr_ci_bot.stale_reason([comment], sha, "abffe439ca84", digest, 0, "ci") is None
    )


# --- derived config and definition drift -------------------------------------


def test_derive_config_absolutizes_relative_references(tmp_path):
    import yaml

    tree = tmp_path / "cfgtree"
    (tree / ".gitcode/hooks").mkdir(parents=True)
    (tree / "xlite").mkdir()
    (tree / ".gitcode/hooks/h.py").write_text("x\n")
    (tree / "xlite/pyproject.toml").write_text("[tool.ruff]\n")
    config = tree / "xlite/.pre-commit-config.yaml"
    config.write_text(
        "repos:\n"
        "  - repo: local\n"
        "    hooks:\n"
        "      - id: a\n"
        "        name: a\n"
        "        entry: python3 .gitcode/hooks/h.py --flag\n"
        "        args: [--config-file=xlite/pyproject.toml]\n"
        "        files: ^xlite/xlite/.+\\.py$\n"
        "  - repo: https://github.com/pre-commit/pre-commit-hooks\n"
        "    rev: v6.0.0\n"
        "    hooks:\n"
        "      - id: trailing-whitespace\n"
    )
    derived = yaml.safe_load(run_static_checks.derive_config(config).read_text())
    hook = derived["repos"][0]["hooks"][0]
    assert str((tree / ".gitcode/hooks/h.py").resolve()) in hook["entry"]
    assert "--flag" in hook["entry"]
    assert str((tree / "xlite/pyproject.toml").resolve()) in hook["args"][0]
    assert hook["files"] == "^xlite/xlite/.+\\.py$"
    assert derived["repos"][1]["repo"].endswith("pre-commit-hooks")


def definition_tree(
    tmp_path,
    name,
    pyproject="[project]\nname = 'x'\n\n[tool.ruff]\nline-length = 120\n\n[tool.mypy]\nignore_missing_imports = true\n",
):
    root = tmp_path / name
    (root / "xlite").mkdir(parents=True)
    (root / ".gitcode/hooks").mkdir(parents=True)
    (root / "xlite/.clang-format").write_text("base\n")
    (root / "xlite/pyproject.toml").write_text(pyproject)
    return root


def test_drift_identical_trees_clean(tmp_path):
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    assert check_definition_drift.definition_drift(checked, own) == []


def test_drift_rule_file_change_detected(tmp_path):
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (checked / "xlite/.clang-format").write_text("relaxed\n")
    assert "xlite/.clang-format" in check_definition_drift.definition_drift(
        checked, own
    )


def test_drift_tool_section_change_detected(tmp_path):
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (checked / "xlite/pyproject.toml").write_text(
        "[project]\nname = 'x'\n\n[tool.ruff]\nline-length = 200\n\n[tool.mypy]\nignore_missing_imports = true\n"
    )
    assert (
        "xlite/pyproject.toml [tool.ruff]"
        in check_definition_drift.definition_drift(checked, own)
    )


NESTED_PYPROJECT = (
    "[project]\nname = 'x'\n\n[tool.ruff]\nline-length = 120\n\n"
    "[tool.ruff.lint]\nselect = ['E', 'F']\n\n"
    "[tool.ruff.lint.per-file-ignores]\n'tests/*' = ['F401']\n\n"
    "[tool.mypy]\nignore_missing_imports = true\n"
)


def test_drift_nested_subtable_change_detected(tmp_path):
    """The real rules live in [tool.ruff.lint] / [tool.ruff.lint.per-file-ignores];
    a nested header must not close the section collection."""
    own = definition_tree(tmp_path, "own", pyproject=NESTED_PYPROJECT)
    checked = definition_tree(tmp_path, "checked", pyproject=NESTED_PYPROJECT)
    (checked / "xlite/pyproject.toml").write_text(
        NESTED_PYPROJECT.replace(
            "select = ['E', 'F']", "select = ['E', 'F']\nignore = ['F401', 'E7']"
        )
    )
    assert check_definition_drift.definition_drift(checked, own) == [
        "xlite/pyproject.toml [tool.ruff]"
    ]


def test_drift_nested_subtable_identical_clean(tmp_path):
    own = definition_tree(tmp_path, "own", pyproject=NESTED_PYPROJECT)
    checked = definition_tree(tmp_path, "checked", pyproject=NESTED_PYPROJECT)
    assert check_definition_drift.definition_drift(checked, own) == []


def test_drift_foreign_section_unaffected_by_nested_collection(tmp_path):
    """Collecting nested subtables must not drag [tool.mypy] into the
    [tool.ruff] slice or vice versa."""
    own = definition_tree(tmp_path, "own", pyproject=NESTED_PYPROJECT)
    checked = definition_tree(tmp_path, "checked", pyproject=NESTED_PYPROJECT)
    (checked / "xlite/pyproject.toml").write_text(
        NESTED_PYPROJECT.replace(
            "ignore_missing_imports = true", "ignore_missing_imports = false"
        )
    )
    drift = check_definition_drift.definition_drift(checked, own)
    assert "xlite/pyproject.toml [tool.mypy]" in drift
    assert "xlite/pyproject.toml [tool.ruff]" not in drift


def test_drift_array_of_tables_header_collected(tmp_path):
    """[[tool.mypy.overrides]] array-of-tables headers are headers too —
    bracket-stripped matching keeps them inside the slice even when they
    sit flush against the previous section."""
    aot = (
        NESTED_PYPROJECT
        + "[[tool.mypy.overrides]]\nmodule = 'a.*'\nwarn_unused_ignores = true\n"
    )
    own = definition_tree(tmp_path, "own", pyproject=aot)
    checked = definition_tree(tmp_path, "checked", pyproject=aot)
    (checked / "xlite/pyproject.toml").write_text(
        aot.replace("warn_unused_ignores = true", "warn_unused_ignores = false")
    )
    assert (
        "xlite/pyproject.toml [tool.mypy]"
        in check_definition_drift.definition_drift(checked, own)
    )


def test_drift_array_of_tables_sibling_not_collected(tmp_path):
    """Content under a sibling [[tool.mypy_pack]] table lies outside the
    [tool.mypy] slice — editing it does not trip the [tool.mypy] freeze."""
    aot = NESTED_PYPROJECT + "[[tool.mypy_pack]]\nmodule = 'a.*'\n"
    own = definition_tree(tmp_path, "own", pyproject=aot)
    checked = definition_tree(tmp_path, "checked", pyproject=aot)
    (checked / "xlite/pyproject.toml").write_text(
        aot.replace("module = 'a.*'", "module = 'b.*'")
    )
    assert check_definition_drift.definition_drift(checked, own) == []


def test_drift_packaging_only_pyproject_edit_allowed(tmp_path):
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (checked / "xlite/pyproject.toml").write_text(
        "[project]\nname = 'y'\n\n[tool.ruff]\nline-length = 120\n\n[tool.mypy]\nignore_missing_imports = true\n"
    )
    assert (
        "xlite/pyproject.toml [tool.ruff]"
        not in check_definition_drift.definition_drift(checked, own)
    )


def test_drift_planted_ruff_toml_detected(tmp_path):
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (checked / "xlite/.ruff.toml").write_text("line-length = 80\n")
    assert "xlite/.ruff.toml" in check_definition_drift.definition_drift(checked, own)


def test_drift_planted_nested_configs_detected(tmp_path):
    """Discovery reads rule files nearest-up from each checked file, so any
    plantable path matters — not just the top-level ones."""
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (checked / "xlite/csrc").mkdir()
    (checked / "xlite/csrc/.clang-format").write_text("relaxed\n")
    (checked / "xlite/tests").mkdir()
    (checked / "xlite/tests/.ruff.toml").write_text("line-length = 80\n")
    (checked / "xlite/tests/pyproject.toml").write_text("[tool.ruff]\n")
    drift = check_definition_drift.definition_drift(checked, own)
    assert "xlite/csrc/.clang-format" in drift
    assert "xlite/tests/.ruff.toml" in drift
    assert "xlite/tests/pyproject.toml" in drift


def test_drift_nested_config_matching_trusted_copy_clean(tmp_path):
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (own / "xlite/csrc").mkdir()
    (checked / "xlite/csrc").mkdir()
    (own / "xlite/csrc/.clang-format").write_text("owner-approved\n")
    (checked / "xlite/csrc/.clang-format").write_text("owner-approved\n")
    assert check_definition_drift.definition_drift(checked, own) == []


def test_drift_nested_config_mismatched_with_trusted_copy_detected(tmp_path):
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (own / "xlite/csrc").mkdir()
    (checked / "xlite/csrc").mkdir()
    (own / "xlite/csrc/.clang-format").write_text("owner-approved\n")
    (checked / "xlite/csrc/.clang-format").write_text("relaxed\n")
    assert "xlite/csrc/.clang-format" in check_definition_drift.definition_drift(
        checked, own
    )


def test_drift_planted_clang_format_alt_name_and_codespellignore_detected(tmp_path):
    """clang-format also discovers _clang-format, and codespell gains
    per-directory codespellignore discovery in releases newer than the
    pinned 2.4.3 — both names are frozen so a tool bump cannot open the
    nearest-up hole."""
    own = definition_tree(tmp_path, "own")
    checked = definition_tree(tmp_path, "checked")
    (checked / "xlite/csrc").mkdir()
    (checked / "xlite/csrc/_clang-format").write_text("relaxed\n")
    (checked / "xlite/csrc/codespellignore").write_text("exampleword\n")
    drift = check_definition_drift.definition_drift(checked, own)
    assert "xlite/csrc/_clang-format" in drift
    assert "xlite/csrc/codespellignore" in drift


# --- client pagination -------------------------------------------------------


def test_pagination_pulls_every_page_and_preserves_params():
    client = pr_ci_bot.GitCodeClient.__new__(pr_ci_bot.GitCodeClient)
    client.session = FakeSession([[{"n": 1}, {"n": 2}], [{"n": 3}]])
    paged = client.get_paged("/pulls", params={"state": "open"}, per_page=2)
    assert [g["n"] for g in paged] == [1, 2, 3]
    assert all(c and c.get("state") == "open" for c in client.session.calls)


# --- harness output shaping ---------------------------------------------------


def test_static_output_shows_failed_sections_only():
    out = "trim trailing whitespace....Passed\nruff check....Failed\nb1\ngit check....Passed\n"
    assert run_ci_checks.static_output(out) == "b1"


def test_hook_names_sanitized_for_comment_tables():
    """A PR can shape pre-commit output lines; the hook names built from them
    must not carry backticks or pipes into a posted table row."""
    out = "evil` | forged | row....Failed\nboom\n"
    names = [name for name, _, _ in run_ci_checks.hook_sections(out)]
    assert names == ["evil forged row"]


def test_default_timeouts():
    assert run_ci_checks.timeout(run_ci_checks.DEFAULT_CONFIG, "static") == 1800


# --- comment hardening --------------------------------------------------------


def test_comment_note_vetted_and_single_line():
    checks = [
        pr_ci_bot.Check(
            "xlite static checks",
            "failed",
            "",
            "failed hooks: `x` | piped\nGITCODE_TOKEN=leaked\nmore",
        ),
    ]
    body = pr_ci_bot.build_comment(
        "a" * 40, checks, "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    assert "leaked" not in body
    assert (
        "| xlite static checks | ❌ Failed — failed hooks: `x` piped GITCODE_TOKEN=*** more |"
        in body
    )


def test_comment_unknown_status_fails_closed():
    checks = [pr_ci_bot.Check("mystery", "corrupted-json-status", "out", "")]
    body = pr_ci_bot.build_comment(
        "a" * 40, checks, "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    assert "❌ Failed" in body  # rendered as failed, no KeyError
    assert "comment `/rerun` to re-check" in body  # counted as a failure for the tip


def test_comment_carries_collapsed_ci_commands_reminder():
    body = pr_ci_bot.build_comment(
        "a" * 40, sample_checks(), "h", [], "b" * 64, "c" * 40, 0, "ci"
    )
    start = body.find("<summary>CI commands</summary>")
    assert 0 < start < body.find("<details open>")  # after the harness line
    block = body[start : body.find("</details>", start)]
    assert "`/ci`" in block and "`/ci all`" in block and "`/rerun`" in block
    assert "<details open>" not in block  # the reminder stays collapsed


# --- clang-tidy verdict cache ------------------------------------------------


def tidy_hook(monkeypatch, root):
    """The hook bound to a fake checked tree; XLITE_ROOT/TIDY_CONFIG are
    module constants computed from cwd at import, so rebind them."""
    import hooks.run_clang_tidy as tidy

    monkeypatch.setattr(tidy, "XLITE_ROOT", root / "xlite")
    monkeypatch.setattr(tidy, "TIDY_CONFIG", root / "xlite" / ".clang-tidy")
    return tidy


def tidy_tree(tmp_path, name, csrc_body="int main() {}\n", config="Checks: -*\n"):
    root = tmp_path / name
    (root / "xlite" / "csrc" / "kernels").mkdir(parents=True)
    (root / "xlite" / "csrc" / "op.cpp").write_text(csrc_body)
    (root / "xlite" / "csrc" / "kernels" / "k.h").write_text("// skipped\n")
    (root / "xlite" / ".clang-tidy").write_text(config)
    (root / "xlite" / "build").mkdir()
    # Compile database as the build generates it: absolute per-tree paths.
    (root / "xlite" / "build" / "compile_commands.json").write_text(
        f'[{{"directory": "{root}/xlite", "command": "clang++ -I{root}/xlite/csrc op.cpp"}}]'
    )
    (root / "xlite" / "build" / "aclrtlaunch_all.h").write_text("// generated\n")
    return root


def test_tidy_cache_key_is_content_addressed(monkeypatch, tmp_path):
    """Identical content in two different worktrees (the cross-PR case) must
    share one verdict: absolute per-tree paths normalize away."""
    root_a = tidy_tree(tmp_path, "a")
    root_b = tidy_tree(tmp_path, "b")

    monkeypatch.chdir(root_a)
    tidy = tidy_hook(monkeypatch, root_a)
    key_a = tidy.cache_key("tidy 22.1.8")
    monkeypatch.chdir(root_b)
    tidy_hook(monkeypatch, root_b)
    key_b = tidy.cache_key("tidy 22.1.8")
    assert key_a and key_a == key_b


def test_tidy_cache_key_busts_on_content_config_and_version(monkeypatch, tmp_path):
    root = tidy_tree(tmp_path, "a")
    monkeypatch.chdir(root)
    tidy = tidy_hook(monkeypatch, root)
    base = tidy.cache_key("tidy 22.1.8")
    assert base

    (root / "xlite" / "csrc" / "op.cpp").write_text("int changed() {}\n")
    assert tidy.cache_key("tidy 22.1.8") != base
    assert tidy.cache_key("tidy 22.1.8") == tidy.cache_key("tidy 22.1.8")

    (root / "xlite" / "csrc" / "op.cpp").write_text("int main() {}\n")
    (root / "xlite" / ".clang-tidy").write_text("Checks: bugprone-*\n")
    assert tidy.cache_key("tidy 22.1.8") != base
    (root / "xlite" / ".clang-tidy").write_text("Checks: -*\n")
    assert tidy.cache_key("tidy 22.2.0") != base
    assert tidy.cache_key("tidy 22.1.8") == base


def test_tidy_analysis_files_skip_kernels(monkeypatch, tmp_path):
    root = tidy_tree(tmp_path, "a")
    tidy = tidy_hook(monkeypatch, root)
    names = [p.name for p in tidy.analysis_files()]
    assert "op.cpp" in names and "k.h" not in names


def test_tidy_analysis_files_match_hook_filter(monkeypatch, tmp_path):
    """The verdict-cache gate needs the exact file set pre-commit passes:
    every non-kernel csrc C/C++ file, headers included (a glob that missed
    .h files made every full run miss the cache)."""
    root = tidy_tree(tmp_path, "a")
    (root / "xlite" / "csrc" / "base.h").write_text("#pragma once\n")
    tidy = tidy_hook(monkeypatch, root)
    names = sorted(p.name for p in tidy.analysis_files())
    assert names == ["base.h", "op.cpp"]


def test_tidy_verdict_cache_roundtrip_and_lru(monkeypatch, tmp_path):
    import os

    tidy = tidy_hook(monkeypatch, tmp_path)
    store = tmp_path / "cache"
    store.mkdir()
    key = "k" * 64
    tidy.store_verdict(store, key, False, "findings text")
    entry = tidy.load_verdict(store, key)
    assert entry and entry["clean"] is False and entry["output"] == "findings text"

    tidy.store_verdict(store, "c" * 64, True, "")
    assert tidy.load_verdict(store, "c" * 64)["clean"] is True
    assert tidy.load_verdict(store, "x" * 64) is None

    # Distinct mtimes: the tmpfs the suite runs on ties same-second writes.
    for i in range(tidy.CACHE_CAP + 2):
        name = f"{i:064d}"
        tidy.store_verdict(store, name, True, "")
        os.utime(store / f"{name}.json", (1000 + i, 1000 + i))
    os.utime(store / f"{key}.json", (1, 1))  # oldest
    tidy.store_verdict(store, "z" * 64, True, "")
    assert len(list(store.glob("*.json"))) == tidy.CACHE_CAP
    assert tidy.load_verdict(store, key) is None  # oldest evicted


def test_tidy_verdict_store_gated_on_full_analysis_set(monkeypatch, tmp_path):
    """A run over a file subset must not store a verdict under the full-set
    key: it would answer for files it never analyzed on the next full run
    (round-2 review: the store lacked the load path's full-set gate)."""
    root = tidy_tree(tmp_path, "a")
    (root / "xlite" / "csrc" / "base.h").write_text("#pragma once\n")
    (root / "xlite" / "CMakeLists.txt").write_text("")
    monkeypatch.chdir(root)
    tidy = tidy_hook(monkeypatch, root)
    store = tmp_path / "cache"
    monkeypatch.setenv(tidy.TIDY_CACHE_ENV, str(store))
    monkeypatch.setattr(tidy, "ensure_build_tree", lambda: True)
    monkeypatch.setattr(tidy, "tidy_version", lambda: "tidy 22.1.8")
    monkeypatch.setattr(tidy, "cache_key", lambda version: "k" * 64)
    monkeypatch.setattr(tidy, "analyze", lambda files: (True, ""))
    full = [str(p) for p in tidy.analysis_files()]

    monkeypatch.setattr(sys, "argv", ["run_clang_tidy.py", *full[:1]])
    assert tidy.main() == 0
    assert tidy.load_verdict(store, "k" * 64) is None

    monkeypatch.setattr(sys, "argv", ["run_clang_tidy.py", *full])
    assert tidy.main() == 0
    assert tidy.load_verdict(store, "k" * 64)["clean"] is True


def test_tidy_findings_are_worktree_relative(monkeypatch, tmp_path):
    """clang-tidy prints the compile database's absolute paths; the verdict
    cache is shared across worktrees, so the findings text must never embed
    this worktree's path (a cache hit would post the populating worktree's
    paths under another head — seen live on PR !705)."""

    class FakeProc:
        returncode = 0
        stderr = ""

        def __init__(self, stdout):
            self.stdout = stdout

    root = tidy_tree(tmp_path, "a")
    monkeypatch.chdir(root)
    tidy = tidy_hook(monkeypatch, root)

    def fake_run(cmd, cwd=None, capture_output=True, text=True, check=False):
        # The real clang-tidy prints paths under this worktree's xlite root.
        return FakeProc(
            f"{tidy.XLITE_ROOT}/csrc/op.cpp:41:23: warning: C-style casts are "
            "discouraged; use static_cast [modernize-avoid-c-style-cast]\n"
        )

    monkeypatch.setattr(tidy.subprocess, "run", fake_run)
    clean_a, findings_a = tidy.analyze(tidy.analysis_files())
    assert not clean_a
    assert "csrc/op.cpp:41:23: warning" in findings_a
    assert f"{root}/xlite/" not in findings_a

    # Identical content in a second worktree produces byte-identical text:
    # the cache entry it stores carries no trace of either path.
    root_b = tidy_tree(tmp_path, "b")
    monkeypatch.chdir(root_b)
    monkeypatch.setattr(tidy, "XLITE_ROOT", root_b / "xlite")
    monkeypatch.setattr(tidy, "TIDY_CONFIG", root_b / "xlite" / ".clang-tidy")
    clean_b, findings_b = tidy.analyze(tidy.analysis_files())
    assert not clean_b and findings_b == findings_a


# --- process_pr integration -------------------------------------------------


def run_process_pr(monkeypatch, tmp_path, comments, checks, stale_base=False):
    """Drive process_pr end-to-end with every heavyweight collaborator
    faked — no git, no network, no harness — for an untrusted author so
    the master-copy path runs. Returns the captured log lines."""
    logs = []
    monkeypatch.setattr(pr_ci_bot, "log", lambda msg: logs.append(msg))
    monkeypatch.setattr(pr_ci_bot, "fetch_ref", lambda *a, **k: "b" * 40)
    monkeypatch.setattr(pr_ci_bot, "prepare_worktree", lambda *a, **k: tmp_path / "wt")
    monkeypatch.setattr(pr_ci_bot, "remove_worktree", lambda *a, **k: None)
    monkeypatch.setattr(pr_ci_bot, "run_harness", lambda script, wt, args: checks)
    monkeypatch.setattr(pr_ci_bot, "base_is_stale", lambda *a, **k: stale_base)
    client = FakeClient()
    client.get_pr_head = lambda n: {"sha": "b" * 40}
    client.get_pr_comments = lambda n: comments
    client.get_pr_files = lambda n: []
    args = argparse.Namespace(
        local_test=True,
        dry_run=True,
        refs_repo=str(tmp_path / "refs.git"),
        worktree_root=str(tmp_path),
        comment_mode="post",
        keep_worktrees=False,
    )
    pr = {
        "number": 712,
        "title": "t",
        "draft": False,
        "user": {"login": "someone"},
        "head": {"repo": {"full_name": "fork/repo"}, "ref": "branch"},
    }
    master = {
        "owners": {"lulina1"},
        "digest": "d" * 64,
        "script": tmp_path / "master-script.py",
        "sha": "m" * 40,
        "digest_exclude": set(),
        "digest_include": [],
    }
    pr_ci_bot.process_pr(client, pr, args, master)
    return logs


def test_process_pr_nudge_survives_check_objects(tmp_path, monkeypatch):
    """The rebase nudge must read Check objects as attributes: any()
    evaluates on every run — green or failed — so a subscript crashed
    every entry pass after the harness completed, eating the results
    comment (the live !712 local-test crash)."""
    checks = [pr_ci_bot.Check("xlite static checks", "failed", "boom")]
    logs = run_process_pr(monkeypatch, tmp_path, [], checks, stale_base=True)
    assert any("Rebase onto the latest master" in line for line in logs)
    assert any("dry run, comment would be:" in line for line in logs)
