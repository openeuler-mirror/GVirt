"""
This script uses `ais_bench` to benchmark the accuracy (correctness on predefined datasets) of served models.

Every per-model parameter (--models, --tps, --eps, --dps, --specs, --datasets, -MNS, -MML, -MNT, -GMU, -X, ...)
is a list aligned by position and padded with its last entry; entries may repeat via the `N~<value>` prefix
(e.g. `2~ceval:52` benchmarks two models on ceval with 52 prompts each).

Only works for *nix systems with `bash` available, as it relies on `source` to activate the Python virtual environment.
Additionally, the script is designed to run within a container with `git`, `vllm`, `vllm_ascend`, and `xlite` installed.
Using `vllm-ascend`'s official container image is recommended.
"""

import os
import re
import shlex
import signal
import json
import socket
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from daily_bot_utils import expand_values, pad_to_last, tile_each

TMP_DIR = Path(tempfile.gettempdir())  # per-OS temp base path
CWD = Path(__file__).parent.resolve()  # current working directory

AIS_BENCH_DIR = TMP_DIR / "benchmark"  # path to clone ais_bench repo and store datasets
VENV_ACTIVATE_CMD: str = ""
# CPU-only torch wheel index: ais_bench requires an unpinned `torch`, which PyPI resolves
# to the multi-GB CUDA build (plus its nvidia-* dependencies) on Linux
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu/"

_PROCESS_GROUPS: set[subprocess.Popen] = set()  # to keep track of started subprocesses for cleanup on exit


@dataclass(kw_only=True, slots=True)
class BenchResult:
    model_name: str
    dataset: str
    tp_size: int
    ep: bool
    dp_size: int
    spec_decode: tuple[str, int, str]  # (method, num_spec_tokens, draft dir); num_spec_tokens == 0 disables
    xlite: bool
    xlite_full_mode: bool
    metric: str
    accuracy: float

    _n_chars: int = field(default=400, init=False, repr=False)
    _error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.xlite = self.xlite or self.xlite_full_mode
        self.xlite_full_mode = self.xlite_full_mode and self.xlite

    @property
    def error(self) -> str | None:
        return getattr(self, "_error", None)

    @error.setter
    def error(self, value: str | None):
        value = value and value.replace("\n", " ").strip()
        value = re.sub(r"\s+", " ", value) if value else ""
        if value and len(value) > self._n_chars + 3:
            value = f"{value[: self._n_chars]}..."
        self._error = value

    @property
    def spec_label(self) -> str:
        """Cell representation of the spec decode config, e.g. `mtp:3` (disabled = `n/a`)."""
        return format_spec_decode(self.spec_decode)

    @staticmethod
    def markdown_header() -> str:
        fields = ["Model", "Dataset", "TP", "EP", "DP", "Spec", "Backend", "Metric", "Accuracy", "Error"]
        return f"| {' | '.join(fields)} |\n| {' | '.join(['---'] * len(fields))} |"

    def markdown_row(self) -> str:
        if self.xlite:
            if self.xlite_full_mode:
                backend_str = "xlite full"
            else:
                backend_str = "xlite decode-only"
        else:
            backend_str = "aclgraph"

        values: list[str] = [
            self.model_name,
            self.dataset,
            str(self.tp_size),
            "Y" if self.ep else "N",
            str(self.dp_size),
            self.spec_label,
            backend_str,
            self.metric,
            f"{self.accuracy:.2f}",
            self.error or "",
        ]
        return f"| {' | '.join(values)} |"


def stop_process_group(process: subprocess.Popen, sleep_time: int = 120):
    pid = process.pid
    print(f"Stopping the process group containing PID {pid}...")
    t1 = datetime.now()
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        print(f"Process with PID {pid} not found, it may have already exited.")
        _PROCESS_GROUPS.discard(process)
        _close_stream_handles(process)
        return

    while True:
        # check if the process is still alive
        try:
            os.killpg(os.getpgid(pid), 0)  # check if process group is alive
            seconds_passed = (datetime.now() - t1).total_seconds()
            if seconds_passed > sleep_time:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                break
            time.sleep(1)
        except ProcessLookupError:
            print(f"Process with PID {pid} has been stopped successfully.")
            break

    _PROCESS_GROUPS.discard(process)
    _close_stream_handles(process)


def _close_stream_handles(process: subprocess.Popen) -> None:
    """Close any file-backed stdout/stderr streams on the given process.

    Set when :func:`start_vllm_server` redirected the server logs to files.
    ``DEVNULL``, ``None``, and ``PIPE`` have nothing to close.
    """
    for stream in (process.stdout, process.stderr):
        if stream is None or isinstance(stream, (int, bytes)):
            # subprocess.DEVNULL is an int (not a file object); skip so we never
            # call .close() on something that doesn't have one.
            continue
        stream.close()


def check_subprocess_result(res: subprocess.CompletedProcess, error_message: str):
    if res.returncode != 0:
        raise RuntimeError(f"{error_message}: {res.stderr.decode('utf-8') if res.stderr else 'No error message'}")


def _read_tail(handle: object, max_bytes: int = 16 * 1024) -> str:
    """Read the trailing bytes of a still-open log file handle opened for writing.

    Used to surface vLLM's error output when the server process dies prematurely,
    in the same role that ``res.stderr`` plays in :func:`check_subprocess_result`.
    Returns the decoded tail, or an empty string if the handle is ``None``.
    """
    try:
        handle.flush()  # ensure buffered writes are visible to the read
        pos = handle.tell()
        size = handle.seek(0, 2)  # SEEK_END
        handle.seek(max(0, size - max_bytes))
        tail = handle.read()
        handle.seek(pos)  # restore the write position so the server keeps appending cleanly
        return tail.decode("utf-8", errors="replace") if isinstance(tail, bytes) else tail
    except Exception:
        return "No error message available"


def format_spec_decode(spec_decode: tuple[str, int, str]) -> str:
    """Label of a spec decode config for reports and notifications.

    `method:num_spec_tokens` (e.g. `mtp:3`); spec decode disabled renders as `n/a` —
    with no tokens requested there is no method to name. The draft weights dir is
    deliberately not part of the label.
    """
    method, num_spec_tokens, _ = spec_decode
    return f"{method}:{num_spec_tokens}" if num_spec_tokens > 0 else "n/a"


# `<method>:<num>[:<draft_dir>]` with every part but <num> optional; a bare `<num>` means mtp
_RE_SPEC_DECODE = re.compile(r"^(?:(?P<method>\w+):)?(?P<num>\d+)(?::(?P<draft_dir>[\w.-]+))?$")


def parse_spec_decode(entry: str) -> tuple[str, int, str]:
    """Parse a ``--specs`` entry into ``(method, num_speculative_tokens, draft_weights_dir)``.

    A bare number ``N`` means ``mtp:N``; zero tokens disables spec decode (no method or
    draft dir applies); an omitted dir uses the main model's own weights, otherwise it is
    a single directory name resolved at server start as a sibling of the main model dir.
    """
    match = _RE_SPEC_DECODE.match(entry.strip())
    if match:
        num = int(match.group("num"))
        if num == 0:
            return "", 0, ""
        return (match.group("method") or "mtp").lower(), num, match.group("draft_dir") or ""
    raise ValueError(
        f"Invalid spec decode entry `{entry}`; expected `<method>:<num_spec_tokens>[:<draft_weights_dir>]` "
        "or a bare number of MTP tokens (e.g. `eagle3:2:Qwen3-a3B_eagle3` or `3`)"
    )


# `<dataset>[:<num_prompts>]` — the prompt count is optional
_RE_DATASET_ENTRY = re.compile(r"^(?P<dataset>\w+)(?::(?P<num_prompts>\d+))?$")


def parse_dataset_entry(entry: str, default_num_prompts: int) -> tuple[str, int]:
    """Parse a ``--datasets`` entry; the prompt count defaults to ``--num-prompts``."""
    match = _RE_DATASET_ENTRY.match(entry.strip())
    if match:
        return match.group("dataset"), int(match.group("num_prompts") or default_num_prompts)
    raise ValueError(
        f"Invalid dataset entry `{entry}`; expected `<n_dups>~<dataset>[:<num_prompts>]` (e.g. `2~ceval:52`)"
    )


@lru_cache(maxsize=1)
def get_xlite_version() -> str:
    """Get the version of the installed xlite package."""

    xlite_version = "unknown"
    try:
        xlite_version = (
            subprocess.run(
                ["python", "-c", "import xlite; print(xlite.__version__)"], capture_output=True, text=True, check=True
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        pass
    return xlite_version


def _gitignore_added_lines(ais_bench_dir: Path) -> list[str]:
    """Collect the lines locally added to the ais_bench repo's .gitignore (vs HEAD)."""
    res = subprocess.run(
        ["git", "-C", str(ais_bench_dir), "diff", "HEAD", "--", ".gitignore"],
        capture_output=True,
        text=True,
    )
    return [line[1:] for line in res.stdout.splitlines() if line.startswith("+") and not line.startswith("+++")]


def _sync_aisbench_repo(ais_bench_dir: Path, tag: str) -> None:
    """Hard-reset an existing ais_bench checkout to ``tag``.

    Tracked modifications are discarded (the harness rewrites the model config files
    with ``sed`` on every run), untracked files (outputs, reports, datasets) are kept.
    The tag is fetched from origin only when still missing locally: tag names embed
    their date, so an existing tag is final and hitting the network for it is wasted.
    Local ``.gitignore`` additions (output dirs, dataset entries) survive the reset:
    they are captured as added lines before the reset and re-appended afterwards — a
    line-level re-apply that cannot conflict, unlike a stash pop, when a tag rewrites
    .gitignore. Line removals are not preserved.
    """

    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(ais_bench_dir), *args], capture_output=True)

    added_gitignore_lines = _gitignore_added_lines(ais_bench_dir)

    # tag names embed their date (e.g. v3.1-20261013-master), so a locally existing tag is
    # final — only hit the network when it is still missing
    if git("rev-parse", "--quiet", "--verify", f"refs/tags/{tag}^{{commit}}").returncode == 0:
        print(f"Tag {tag} already exists locally, skipping fetch...")
    else:
        res = git("fetch", "--depth", "1", "--force", "origin", f"refs/tags/{tag}:refs/tags/{tag}")
        check_subprocess_result(res, f"Failed to fetch ais_bench tag {tag}")

    check_subprocess_result(git("reset", "--hard", tag), f"Failed to reset ais_bench repo to {tag}")

    if added_gitignore_lines:
        gitignore = ais_bench_dir / ".gitignore"
        current = gitignore.read_text().splitlines() if gitignore.exists() else []
        missing = [line for line in added_gitignore_lines if line not in current]
        if missing:
            with open(gitignore, "a") as f:
                if current and current[-1].strip():
                    f.write("\n")
                f.write("\n".join(missing) + "\n")
            print(f"Re-applied {len(missing)} local .gitignore line(s) after reset to {tag}")


def _require_git_lfs() -> None:
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        raise RuntimeError("git-lfs is required for this dataset; install it (e.g. `apt install git-lfs`) and rerun")


def _wget(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    check_subprocess_result(subprocess.run(["wget", "-O", str(target), url]), f"Failed to download {url}")


def _git_clone_dataset(url: str, target: Path, lfs_exclude: str = "") -> None:
    """Clone (or refresh an existing clone of) a dataset repo and materialize its LFS assets.

    A plain clone can leave LFS files as pointer stubs (missing smudge filter, or the
    mirror dropping downloads mid-checkout) — the stubs sit on disk like real data until
    eval crashes on them (e.g. "Parquet magic bytes not found"), so ``git lfs pull``
    re-fetches explicitly and any stub remaining in the pulled scope fails setup loudly.
    An already-existing clone (e.g. from an older setup revision) is refreshed in place.
    """
    _require_git_lfs()
    pull_args = ["git", "-C", str(target), "lfs", "pull"]
    ls_args = ["git", "-C", str(target), "lfs", "ls-files"]
    if lfs_exclude:
        pull_args += ["--exclude", lfs_exclude]
        ls_args += ["--exclude", lfs_exclude]
    if target.exists():
        print(f"Refreshing LFS assets of the existing dataset repo at {target}...")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        # skip the automatic LFS smudge during clone — it fetches every LFS object, while the
        # explicit `git lfs pull` below fetches only the scope this dataset actually needs
        env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
        check_subprocess_result(
            subprocess.run(["git", "clone", "--depth", "1", url, str(target)], env=env),
            f"Failed to clone dataset from {url}",
        )
    check_subprocess_result(subprocess.run(pull_args), f"Failed to pull LFS assets from {url}")
    res = subprocess.run(ls_args, capture_output=True, text=True)
    check_subprocess_result(res, f"Failed to list LFS assets of {target}")
    # `lfs ls-files` flags a `-` entry both for pointer stubs and for locally modified files
    # (e.g. textvqa's path-rewritten jsonl) — only a file actually holding pointer text
    # (or missing outright) is a defect
    pointer_magic = b"version https://git-lfs"
    stubs = []
    for line in res.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) > 2 and parts[1] == "-":
            stub_path = target / parts[2]
            try:
                with open(stub_path, "rb") as f:
                    is_stub = f.read(len(pointer_magic)) == pointer_magic
            except OSError:  # deleted from the worktree = not materialized either
                is_stub = True
            if is_stub:
                stubs.append(parts[2])
    if stubs:
        raise RuntimeError(
            f"{len(stubs)} LFS file(s) in {target} are still pointer stubs (e.g. {stubs[:3]}); "
            "the LFS download failed — fix access to the mirror and rerun dataset setup"
        )


def _download_ceval(ais_bench_dir: Path, download_url: str) -> None:
    dataset_dir = ais_bench_dir / "ais_bench" / "datasets" / "ceval" / "formal_ceval"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir / "ceval-exam.zip"
    _wget(download_url, zip_path)
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(dataset_dir)
    zip_path.unlink(missing_ok=True)
    if not (dataset_dir / "test").exists():
        raise RuntimeError(f"Unexpected ceval archive layout; `{dataset_dir}/test` not found")


def _download_gsm8k(ais_bench_dir: Path, download_url: str) -> None:
    datasets_root = ais_bench_dir / "ais_bench" / "datasets"
    _wget(download_url, datasets_root / "gsm8k.zip")
    with zipfile.ZipFile(datasets_root / "gsm8k.zip") as zip_file:
        zip_file.extractall(datasets_root)
    (datasets_root / "gsm8k.zip").unlink(missing_ok=True)
    if not (datasets_root / "gsm8k" / "test.jsonl").exists():
        raise RuntimeError("Unexpected gsm8k archive layout; `ais_bench/datasets/gsm8k/test.jsonl` not found")


def _download_textvqa(ais_bench_dir: Path, download_url: str) -> None:
    # the source must ship the json/jsonl files under `textvqa_json/` (the config's layout)
    target = ais_bench_dir / "ais_bench" / "datasets" / "textvqa"
    _git_clone_dataset(download_url, target)
    # the shipped jsonl refers to images via a checkout-relative `benchmark/...` prefix that
    # only resolves when the run cwd is the clone's parent — rewrite to absolute paths so
    # the base64 loader finds the images from any cwd
    stale_prefix = "benchmark/ais_bench/datasets/textvqa/"
    for jsonl in (target / "textvqa_json").glob("*.jsonl"):
        text = jsonl.read_text()
        if stale_prefix in text:
            jsonl.write_text(text.replace(f'"{stale_prefix}', f'"{target}/'))
    # the subset mirror can lack a handful of val images; a missing image would kill a whole
    # inference run, so drop those records instead — but a mass of missing images means the
    # mirror is broken and must fail the setup
    val_jsonl = target / "textvqa_json" / "textvqa_val.jsonl"
    kept, dropped = [], 0
    with open(val_jsonl) as f:
        for line in f:
            if line.strip() and Path(json.loads(line)["image"]).exists():
                kept.append(line)
            else:
                dropped += 1
    if dropped > len(kept) * 0.05:
        raise RuntimeError(f"{dropped} of {len(kept) + dropped} textvqa val records reference missing images")
    if dropped:
        val_jsonl.write_text("".join(kept))
        print(f"textvqa: dropped {dropped} val record(s) with missing images, {len(kept)} kept")


def _download_mmmu(ais_bench_dir: Path, download_url: str) -> None:
    # mmmu_gen evaluates the validation split only; excluding the test split skips
    # several GB of LFS download that would never be read
    _git_clone_dataset(download_url, ais_bench_dir / "ais_bench" / "datasets" / "mmmu", lfs_exclude="*test*")


def _download_mmstar(ais_bench_dir: Path, download_url: str) -> None:
    _wget(download_url, ais_bench_dir / "ais_bench" / "datasets" / "mmstar" / "MMStar.tsv")


def _download_realworldqa(ais_bench_dir: Path, download_url: str) -> None:
    _git_clone_dataset(download_url, ais_bench_dir / "ais_bench" / "datasets" / "RealworldQA")


def _download_bfcl(ais_bench_dir: Path, download_url: str) -> None:
    # the BFCL data ships inside the `bfcl-eval` dependency package; nothing to download
    del download_url  # unused: the data comes from the pip package, not a URL
    res = subprocess.run(
        f"{VENV_ACTIVATE_CMD} && pip install -r requirements/datasets/bfcl_dependencies.txt --no-deps",
        shell=True,
        executable="/bin/bash",
        cwd=ais_bench_dir,
        capture_output=True,
    )
    check_subprocess_result(res, "Failed to install bfcl-eval dependencies")


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the harness needs to set up, run and parse one ais_bench dataset."""

    task: str  # value forwarded to `ais_bench --datasets`
    model_config: str  # ais_bench model config (file stem under configs/models/vllm_api)
    score_column: str  # summary CSV column holding the score (the model config's `abbr`)
    row_filter: tuple[str, str]  # (dataset, metric) regexes selecting the summary rows to average
    metric_name: str  # metric label used in reports
    downloader: Callable[[Path, str], None]  # populates the dataset under ais_bench_dir
    merge_ds: bool = True  # whether `--merge-ds` may be passed to ais_bench
    ready_path: str | None = None  # repo-relative file/dir whose existence means "already downloaded"
    # bumped whenever this dataset's downloader behavior changes, so previously downloaded
    # data is refreshed to the current standard instead of being skipped as "ready"
    setup_revision: int = 1
    # All sources are China-accessible (ModelScope / Aliyun OSS) — huggingface.co is not
    # reliably reachable from where this harness runs.
    download_url: str = ""  # primary source URL (zip/TSV/git repo); "" = nothing to download (bfcl ships via pip)


DATASETS: dict[str, DatasetSpec] = {
    # text-only datasets
    "ceval": DatasetSpec(
        task="ceval_gen_0_shot_cot_chat_prompt",
        model_config="vllm_api_general_chat",
        score_column="vllm-api-general-chat",
        row_filter=(r"^ceval-weighted$", r"^weighted_average$"),
        metric_name="weighted_average",
        downloader=_download_ceval,
        ready_path="ais_bench/datasets/ceval/formal_ceval/test",
        download_url="https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip",
    ),
    "gsm8k": DatasetSpec(
        task="gsm8k_gen_0_shot_cot_chat_prompt",
        model_config="vllm_api_general_chat",
        score_column="vllm-api-general-chat",
        row_filter=(r"^gsm8k$", r"^accuracy$"),
        metric_name="accuracy",
        downloader=_download_gsm8k,
        ready_path="ais_bench/datasets/gsm8k/test.jsonl",
        download_url="http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip",
    ),
    # multimodal datasets (need a multimodal-capable served model)
    "textvqa": DatasetSpec(
        task="textvqa_gen_base64",
        model_config="vllm_api_general_chat",
        score_column="vllm-api-general-chat",
        row_filter=(r"^textvqa$", r"^accuracy$"),
        metric_name="accuracy",
        downloader=_download_textvqa,
        ready_path="ais_bench/datasets/textvqa/textvqa_json/textvqa_val.jsonl",
        setup_revision=2,  # r2: absolute image paths + dropping records whose image is missing
        # vllm-ascend's mirror already ships the `textvqa_json/` layout the config reads
        download_url="https://www.modelscope.cn/datasets/vllm-ascend/textvqa_subset.git",
    ),
    "mmmu": DatasetSpec(
        task="mmmu_gen",
        model_config="vllm_api_general_chat",
        score_column="vllm-api-general-chat",
        row_filter=(r"^mmmu$", r"^\[validation\]: Overall$"),
        metric_name="accuracy",
        downloader=_download_mmmu,
        ready_path="ais_bench/datasets/mmmu",
        setup_revision=2,  # r2: LFS materialization (pointer stubs crashed eval) + test-split exclusion
        download_url="https://www.modelscope.cn/datasets/AI-ModelScope/MMMU.git",
    ),
    "mmstar": DatasetSpec(
        task="mmstar_gen",
        model_config="vllm_api_general_chat",
        score_column="vllm-api-general-chat",
        row_filter=(r"^mmstar$", r"^Overall$"),
        metric_name="accuracy",
        downloader=_download_mmstar,
        ready_path="ais_bench/datasets/mmstar/MMStar.tsv",
        download_url="https://www.modelscope.cn/datasets/evalscope/MMStar/resolve/master/MMStar.tsv",
    ),
    "realworldqa": DatasetSpec(
        task="realworldqa_gen_base64",
        model_config="vllm_api_general_chat",
        score_column="vllm-api-general-chat",
        row_filter=(r"^realworldqa$", r"^accuracy$"),
        metric_name="accuracy",
        downloader=_download_realworldqa,
        ready_path="ais_bench/datasets/RealworldQA/data/test-00000-of-00002.parquet",
        setup_revision=2,  # r2: LFS refresh via the shared git-clone helper
        # mirror of huggingface.co/datasets/xai-community/realworldqa with the same layout
        download_url="https://www.modelscope.cn/datasets/xai-org/RealworldQA.git",
    ),
    # tool/function-calling dataset (needs the vllm_api_function_call_chat model config)
    "bfcl": DatasetSpec(
        task="BFCL_gen_all",
        model_config="vllm_api_function_call_chat",
        score_column="vllm-api-function-call-chat",
        row_filter=(r"^BFCL", r"^accuracy$"),
        metric_name="accuracy",
        downloader=_download_bfcl,
        merge_ds=False,  # BFCL sub-datasets use different evaluators and cannot be merged
    ),
}


def _ensure_dataset(key: str, ais_bench_dir: Path) -> None:
    """Download and set up one dataset, refreshing older downloads to the current standard.

    A dataset counts as ready only when its ``ready_path`` exists *and* its recorded setup
    revision (stamp under ``ais_bench/datasets/.setup_revisions/``, gitignored like the
    data itself) matches the spec's ``setup_revision``. A missing stamp reads as revision
    1 (downloaded before revisions existed), so bumping a downloader's revision re-processes
    existing data instead of skipping it as ready.
    """
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset `{key}`; available datasets: {sorted(DATASETS)}")
    spec = DATASETS[key]
    # the ais_bench .gitignore already excludes everything under ais_bench/datasets/
    # except the tracked .py sources, so downloaded data needs no extra ignore entries
    revision_file = ais_bench_dir / "ais_bench" / "datasets" / ".setup_revisions" / key
    installed_revision = int(revision_file.read_text().strip()) if revision_file.exists() else 1
    if spec.ready_path and (ais_bench_dir / spec.ready_path).exists() and installed_revision >= spec.setup_revision:
        print(f"Dataset `{key}` is already set up (revision {installed_revision}), skipping download...")
        return
    print(f"Setting up dataset `{key}` (revision {spec.setup_revision}, found {installed_revision})...")
    spec.downloader(ais_bench_dir, spec.download_url)
    revision_file.parent.mkdir(parents=True, exist_ok=True)
    revision_file.write_text(str(spec.setup_revision))


def _aisbench_pkg_version(venv_python: Path) -> str | None:
    """Version of `ais_bench_benchmark` recorded in the venv, or None if not installed.

    Editable installs record the version at install time only — it does not follow later
    re-syncs of the source tree, which is exactly the staleness the reinstall check below
    needs to detect.
    """
    try:
        res = subprocess.run(
            [str(venv_python), "-c", "from importlib.metadata import version; print(version('ais_bench_benchmark'))"],
            capture_output=True,
            text=True,
        )
    except OSError:  # broken venv (interpreter missing) counts as not installed
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _tag_to_pkg_version(tag: str) -> str:
    """Mirror ais_bench's setup.py ``get_latest_version()``: 'v3.1-20261013-master' -> '3.1.20261013'.

    A tag without the ``-master`` suffix maps to a ``.dev`` version, like setup.py.
    Returns '' when nothing can be derived (e.g. AISBENCH_GIT_TAG overridden to a branch
    name); the caller then falls back to a presence-only check.
    """
    parts = tag.removeprefix("v").replace("-", ".").replace("_", ".").split(".")
    if len(parts) < 2:
        return ""
    return ".".join(parts[:-1]) if parts[-1] == "master" else ".".join(parts[:-1]) + ".dev"


def prepare_env(ais_bench_dir: Path | None = None, datasets: list[str] | None = None):
    aisbench_git_url = os.environ.get("AISBENCH_GIT_URL", "https://github.com/AISBench/benchmark.git")
    aisbench_git_tag = os.environ.get("AISBENCH_GIT_TAG", "v3.1-20261013-master")
    global VENV_ACTIVATE_CMD
    ais_bench_dir = ais_bench_dir or AIS_BENCH_DIR
    # dataset *names* only — `--datasets` entries (with their `N~`/`:<num_prompts>`
    # parts) are resolved by the caller before reaching here
    datasets = list(datasets or ["ceval"])

    # check if benchmark repo exists; if so, hard-reset it to the wanted tag
    if not ais_bench_dir.exists():
        print(f"Cloning ais_bench repo ({aisbench_git_tag}) to {ais_bench_dir}...")
        res = subprocess.run(["git", "clone", "--depth", "1", "-b", aisbench_git_tag, aisbench_git_url, ais_bench_dir])
        check_subprocess_result(res, "Failed to clone ais_bench repo")
    else:
        print(f"Syncing existing ais_bench repo at {ais_bench_dir} to tag {aisbench_git_tag}...")
        _sync_aisbench_repo(ais_bench_dir, aisbench_git_tag)

    # make a `.venv` in the benchmark repo for ais_bench, and install requirements.
    # the venv is namespaced by Python version (e.g. `.venv_py39`, `.venv_py310`) so that
    # different interpreters don't share — or clobber — each other's installed packages.
    venv_path: str = f".venv_py{sys.version_info.major}{sys.version_info.minor}"
    venv_abspath = (ais_bench_dir / venv_path).resolve()
    if not venv_abspath.exists():
        print(f"Creating virtual environment for ais_bench at {venv_abspath}...")
        res = subprocess.run([sys.executable, "-m", "venv", str(venv_abspath)])
        check_subprocess_result(res, "Failed to create virtual environment")

    # activate the virtual environment, then (re)install ais_bench when the installed
    # package version does not match the synced tag: editable installs keep the version
    # recorded at install time, so re-syncing the repo alone never refreshes it
    activate_script = venv_abspath / "bin" / "activate"
    VENV_ACTIVATE_CMD = f"source {activate_script}"

    installed_version = _aisbench_pkg_version(venv_abspath / "bin" / "python")
    expected_version = _tag_to_pkg_version(aisbench_git_tag)
    if installed_version is not None and (not expected_version or installed_version == expected_version):
        print(f"ais_bench {installed_version} matches tag {aisbench_git_tag}, skipping installation...")
    else:
        reason = (
            "not installed"
            if installed_version is None
            else f"version {installed_version} != {expected_version or '<unknown>'} expected for tag {aisbench_git_tag}"
        )
        print(f"Installing ais_bench to {venv_abspath} ({reason})...")
        # TORCH_INDEX_URL steers the unpinned `torch` in runtime.txt (pulled in by the
        # editable install's install_requires) to the CPU build instead of the CUDA one
        res = subprocess.run(
            f"{VENV_ACTIVATE_CMD} && export PIP_RETRIES=30 PIP_TIMEOUT=120 PIP_RESUME_RETRIES=60 && "
            f'pip install --extra-index-url "{TORCH_INDEX_URL}" -r requirements/api.txt '
            f'&& pip install --extra-index-url "{TORCH_INDEX_URL}" -r requirements/extra.txt '
            f'&& pip install --extra-index-url "{TORCH_INDEX_URL}" -v -e ./ --use-pep517',
            shell=True,
            executable="/bin/bash",
            cwd=ais_bench_dir,
        )
        check_subprocess_result(res, "Failed to install ais_bench requirements")

    # download / set up every dataset the run needs (idempotent)
    for dataset_key in dict.fromkeys(datasets):
        _ensure_dataset(dataset_key, ais_bench_dir)


def cleanup_on_exit():
    """Terminate all subprocesses in the global process group set on exit."""
    for process in list(_PROCESS_GROUPS):
        try:
            stop_process_group(process, 0)
        except:
            continue


def check_port_available(port: int, timeout: int = 30) -> bool:
    """Check if the given port is available for binding."""
    t1 = datetime.now()
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex(("localhost", port))
            available = result != 0
        time_delta = (datetime.now() - t1).total_seconds()
        if available or time_delta > timeout:
            return available
        time.sleep(1)  # wait a bit before retrying


def find_available_port(start_port: int = 8241, n: int = 20000) -> int:
    """Find an available port in the given range."""
    end_port = start_port + max(1, n)
    for port in range(start_port, end_port):
        if check_port_available(port, 0):
            return port
    raise RuntimeError(f"No available port found in range {start_port}-{end_port}")


@lru_cache(maxsize=1)
def detect_hccl_network() -> tuple[str, str]:
    """Return the ``(nic, ipv4)`` of the interface owning the default route.

    Gloo and HCCL pick their bind address from ``GLOO_SOCKET_IFNAME`` /
    ``HCCL_SOCKET_IFNAME`` / ``HCCL_IF_IP``. Without them both resolve the local
    hostname instead, which is not guaranteed to be resolvable inside a
    container: Gloo then degrades to loopback with a ~10s DNS stall per rank,
    and HCCL fails outright with ``hcclGetRootInfo ... error code is 19
    (call network api fail)``. ``vllm_xlite.sh`` pins these variables for the
    same reason; this mirrors it so the harness is self-sufficient.
    """
    try:
        with open("/proc/net/route") as fh:
            rows = [line.split() for line in fh.read().splitlines()[1:]]
    except OSError:
        return "", ""
    for row in rows:
        # little-endian hex fields; the default route has Destination == 0.0.0.0
        if len(row) > 7 and row[1] == "00000000":
            nic, gateway = row[0], row[2]
            try:
                peer = socket.inet_ntoa(struct.pack("<I", int(gateway, 16)))
            except (ValueError, struct.error, OSError):
                peer = ""
            if peer in ("", "0.0.0.0"):
                peer = "223.5.5.5"  # any routed address works; nothing is sent
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.connect((peer, 9))  # route lookup only, no packet
                    return nic, sock.getsockname()[0]
            except OSError:
                return nic, ""
    return "", ""


def start_vllm_server(
    model_path: str,
    *,
    model_name: str | None = None,
    port: int | None = None,
    tp_size: int = 1,
    ep: bool = True,
    dp_size: int = 1,
    spec_decode: tuple[str, int, str] = ("", 0, ""),
    max_num_seqs: int = 512,
    max_model_len: int = 8192,
    max_num_batched_tokens: int = 8192,
    seed: int = 42,
    gpu_memory_utilization: float = 0.92,
    block_size: int = 128,
    use_mrv2: bool = False,
    xlite: bool = False,
    xlite_full_mode: bool = False,
    device_ids: list[int] | None = None,
    timeout: int = 1800,
    print_to: Path | None = None,
) -> subprocess.Popen:
    if not model_path or not Path(model_path).exists():
        raise ValueError(f"Model path {model_path} does not exist")

    device_ids = device_ids or list(range(dp_size * tp_size))
    if len(device_ids) < dp_size * tp_size:
        raise ValueError(
            f"Not enough device IDs provided for the required devices. Required devices: {dp_size * tp_size}, provided "
            f"device IDs: {device_ids} (N = {len(device_ids)})"
        )

    # start vllm server with the specified model and settings
    if port is None:
        port = find_available_port()
    if not check_port_available(port):
        raise RuntimeError(f"Port {port} is already in use, cannot start vLLM server")
    print(f"Starting vLLM server with model {model_path} on port {port}...")

    envs: dict[str, str] = {
        "VLLM_USE_V1": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1" if use_mrv2 else "0",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "HCCL_OP_EXPANSION_MODE": "AIV",
        "VLLM_ENGINE_READY_TIMEOUT_S": str(timeout),
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": "1",
        "HCCL_BUFFSIZE": "1024",
        "TASK_QUEUE_ENABLE": "1",
        "XLITE_FLASH_ATTENTION_ENABLE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(i) for i in device_ids),
    }

    env_copy = os.environ.copy()
    env_copy.update(envs)

    # Pin the NICs Gloo/HCCL bind to (mirrors vllm_xlite.sh); see detect_hccl_network().
    # Variables already present in the caller's environment take precedence.
    nic, nic_ip = detect_hccl_network()
    if nic:
        env_copy.setdefault("GLOO_SOCKET_IFNAME", nic)
        env_copy.setdefault("TP_SOCKET_IFNAME", nic)
        env_copy.setdefault("HCCL_SOCKET_IFNAME", nic)
    env_copy.setdefault("HCCL_IF_IP", nic_ip or "127.0.0.1")
    print(f"Pinning Gloo/HCCL to NIC {nic or '<unset>'} ({nic_ip or '127.0.0.1'})")

    model_name = model_name or Path(model_path).stem or "model"
    quantization = (Path(model_path) / "quant_model_description.json").exists()
    xlite = xlite or xlite_full_mode  # if full mode is enabled, xlite must be enabled as well
    xlite_full_mode = xlite_full_mode and xlite  # full mode can only be enabled when xlite is enabled
    vllm_cmd_lst: list[str] = [
        f'vllm serve "{model_path}"',
        f'--served-model-name "{model_name}"',
        f'--port "{port}"',
        "--nnodes 1",
        f'--tensor-parallel-size "{tp_size}"',
        f"{'--enable-expert-parallel' if ep else ''}",
        f'--data-parallel-size "{dp_size}"',
        f'--data-parallel-size-local "{dp_size}"',
        "--data-parallel-address 127.0.0.1",
        "--data-parallel-rpc-port 15973",
        "--async-scheduling",
        "--trust-remote-code",
        f'--allowed-local-media-path "{os.environ.get("AISBENCH_ALLOW_PATH", "/")}"',
        f'--max-num-seqs "{max_num_seqs}"',
        f'--max-model-len "{max_model_len}"',
        f'--max-num-batched-tokens "{max_num_batched_tokens}"',
        f'--seed "{seed}"',
        f'--gpu-memory-utilization "{gpu_memory_utilization}"',
        f'--block-size "{block_size}"',
        '--compilation-config \'{"cudagraph_capture_sizes":[4,8,16,32,48,64], "cudagraph_mode": "FULL_DECODE_ONLY"}\'',
        f'--additional-config \'{{"xlite_graph_config": {{"enabled": {str(xlite).lower()}, "full_mode": {str(xlite_full_mode).lower()}}}}}\'',
    ]
    if quantization:
        vllm_cmd_lst.append("--quantization ascend")
    # draft weights default to the main model's own (integrated draft heads, e.g. MTP — the
    # same config `vllm_xlite.sh` uses); otherwise the entry names a sibling dir of the main
    # model dir, and methods like eagle3/dspark/dflash (see bd5020ad6) need it passed as
    # `model` in the speculative config
    spec_method, num_spec_tokens, draft_dir = spec_decode
    if num_spec_tokens > 0:
        if draft_dir:
            draft_path = Path(model_path).parent / draft_dir
            if not draft_path.exists():
                raise ValueError(f"Speculative draft weights not found at {draft_path}")
            vllm_cmd_lst.append(
                f'--speculative-config \'{{"method": "{spec_method}","num_speculative_tokens": {num_spec_tokens},'
                f'"model": "{draft_path}"}}\''
            )
        else:
            vllm_cmd_lst.append(
                f'--speculative-config \'{{"method": "{spec_method}","num_speculative_tokens": {num_spec_tokens}}}\''
            )
    vllm_cmd = " ".join(vllm_cmd_lst)

    env_copy["XLITE_NODE_IPS"] = ",".join(["127.0.0.1"] * dp_size)
    env_copy["XLITE_LOCAL_IP"] = "127.0.0.1"
    env_copy["XLITE_DEVS_PER_NODE"] = str(dp_size * tp_size)
    env_copy["XLITE_FLASH_ATTENTION_ENABLE"] = "1"

    t1 = datetime.now()
    # by default, redirect the vllm server logs to /dev/null; otherwise, redirect them to a `print_to` file.
    # The log file handles are kept open for the lifetime of the server process and closed
    # only after the readiness check below succeeds, so vLLM keeps writing until we're done.
    stdprint_target = subprocess.DEVNULL
    if print_to is not None:
        print_to.parent.mkdir(parents=True, exist_ok=True)
        log_path = print_to.parent / f"{print_to.name}.out.log"
        stdprint_target = open(log_path, "w")
        print(f"Redirecting vLLM server logs to {log_path}...")
    process = subprocess.Popen(
        vllm_cmd,
        shell=True,
        executable="/bin/bash",
        env=env_copy,
        start_new_session=True,  # to allow killing the whole process group later
        stdout=stdprint_target,
        stderr=stdprint_target,
    )

    # check if the server is ready by checking port availability
    try:
        while (datetime.now() - t1).total_seconds() < timeout:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process exited unexpectedly with code {process.returncode}:\n"
                    f"{_read_tail(stdprint_target)}"
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("localhost", port)) == 0:
                    break  # successfully connected to the port, server is ready
            time.sleep(1)
        else:  # if `break` was not hit and loop exhausted
            raise TimeoutError(f"vLLM server did not start within {timeout} seconds")
    except Exception as e:
        stop_process_group(process, 0)
        raise e
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt received, stopping vLLM server...")
        stop_process_group(process, 0)
        raise e
    finally:
        if stdprint_target and not isinstance(stdprint_target, int):  # don't close DEVNULL or None
            stdprint_target.flush()

    _PROCESS_GROUPS.add(process)
    print(f"vLLM server started successfully with PID {process.pid}")
    return process


def run_ais_bench(
    *,
    ais_bench_dir: Path | None = None,
    ais_bench_output_dir: Path | None = None,
    ais_bench_num_prompts: int = 128,
    model_path: str = "",
    model_name: str | None = None,
    dataset: str = "ceval",
    port: int = 8384,
    tp_size: int = 1,
    ep: bool = True,
    dp_size: int = 1,
    spec_decode: tuple[str, int, str] = ("", 0, ""),
    max_num_seqs: int = 512,
    max_model_len: int = 8192,
    max_num_batched_tokens: int = 8192,
    seed: int = 42,
    gpu_memory_utilization: float = 0.92,
    block_size: int = 128,
    use_mrv2: bool = False,
    xlite: bool = False,
    xlite_full_mode: bool = False,
    extra_args: str = "",
    device_ids: list[int] | None = None,
    timeout: int = 2 * 3600,  # 2 hours
    retry: int = 3,
    vllm_timeout: int = 1800,
) -> BenchResult:
    spec = DATASETS[dataset]
    model_name = model_name or Path(model_path).stem or "model"
    quantization = (Path(model_path) / "quant_model_description.json").exists()
    xlite = xlite or xlite_full_mode  # if full mode is enabled, xlite must be enabled as well
    xlite_full_mode = xlite_full_mode and xlite  # full mode can only be enabled when xlite is enabled
    port = find_available_port(port)
    # failed/absent runs keep accuracy=NaN (never overwritten on the failure paths) so the
    # reports never show a misleading real-looking 0.00% for a run that produced no result
    bench_result = BenchResult(
        model_name=model_name,
        dataset=dataset,
        tp_size=tp_size,
        ep=ep,
        dp_size=dp_size,
        spec_decode=spec_decode,
        xlite=xlite,
        xlite_full_mode=xlite_full_mode,
        metric=spec.metric_name,
        accuracy=float("nan"),
    )

    if not model_path or not Path(model_path).exists():
        print(f"Model path {model_path} does not exist, skipping benchmark for model {model_name}...")
        bench_result.error = "Model weights not found"
        return bench_result

    SEC_SEP_1, SEC_SEP_2 = "=" * 50, "-" * 50
    print()
    print(SEC_SEP_1)
    print(
        f"Running ais_bench for model {model_name} on port {port}, using the `{dataset}` dataset with the first "
        f"{ais_bench_num_prompts} prompts from each subset..."
    )
    print(SEC_SEP_2)
    print(f"Model path: {model_path}")
    print(f"Model name: {model_name}")
    print(f"Dataset: {dataset} (ais_bench task: {spec.task})")
    print(f"ASCEND_RT_VISIBLE_DEVICES: {','.join(map(str, device_ids)) if device_ids else 'Not set'}")
    print(f"Quantization: {'Yes' if quantization else 'No'}")
    print(f"Tensor parallel size: {tp_size}")
    print(f"Expert parallel: {'Yes' if ep else 'No'}")
    print(f"Data parallel size: {dp_size}")
    print(f"Speculative decoding: {format_spec_decode(spec_decode)}")
    print(f"Max number of sequences: {max_num_seqs}")
    print(f"Max model length: {max_model_len}")
    print(f"Max number of batched tokens: {max_num_batched_tokens}")
    print(f"Seed: {seed}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Block size: {block_size}")
    print(f"Using xlite: {'Yes' if xlite else 'No'}")
    print(f"Using xlite full mode: {'Yes' if xlite_full_mode else 'No'}")
    print(SEC_SEP_2)

    ais_bench_dir = ais_bench_dir or AIS_BENCH_DIR
    ais_bench_script = (
        ais_bench_dir / "ais_bench" / "benchmark" / "configs" / "models" / "vllm_api" / f"{spec.model_config}.py"
    )
    if not ais_bench_script.exists():
        raise FileNotFoundError(f"ais_bench script not found at {ais_bench_script}")

    if xlite:
        suffix_backend = "xlite-full" if xlite_full_mode else "xlite-decode-only"
    else:
        suffix_backend = "aclgraph"
    spec_method, num_spec_tokens, _ = spec_decode
    spec_suffix = ""
    if num_spec_tokens > 0:
        spec_suffix = f"-{spec_method.upper()}.{num_spec_tokens}"
    suffix_deployment = f"TP{tp_size}DP{dp_size}{'EP' if ep else ''}{spec_suffix}-{dataset}"
    suffix = f"-{suffix_deployment}-{suffix_backend}"
    ais_bench_output_parent_dir = ais_bench_output_dir or ais_bench_dir / "outputs" / get_xlite_version()
    # the dataset rides in `suffix_deployment` so different datasets never share (and confuse) work dirs
    ais_bench_output_dir = ais_bench_output_parent_dir / model_name / suffix_backend / suffix_deployment
    if not ais_bench_output_dir.exists():
        ais_bench_output_dir.mkdir(parents=True)
    print(f"ais_bench output directory: {ais_bench_output_dir}")

    # find the line `host_port=xxx` in the script and replace it with the actual port
    res = subprocess.run(
        f'sed -i -E "s/(host_port[[:space:]]*=[[:space:]]*)[0-9]+/\\1{port}/" "{ais_bench_script}"',
        shell=True,
        executable="/bin/bash",
    )
    check_subprocess_result(res, "Failed to update port in ais_bench script")
    # find the line `model=xxx` in the script and replace it with the actual model
    res = subprocess.run(
        f'sed -i -E "s/(model[[:space:]]*=[[:space:]]*)\\"[^\\"]*\\"/\\1\\"{model_name}\\"/" "{ais_bench_script}"',
        shell=True,
        executable="/bin/bash",
    )
    check_subprocess_result(res, "Failed to update model in ais_bench script")
    # find the line `max_out_len=xxx` in the script and replace it with the actual model max length
    res = subprocess.run(
        f'sed -i -E "s/(max_out_len[[:space:]]*=[[:space:]]*)[0-9]+/\\1{max_model_len}/" "{ais_bench_script}"',
        shell=True,
        executable="/bin/bash",
    )
    check_subprocess_result(res, "Failed to update max_out_len in ais_bench script")
    # find the line `batch_size=xxx` in the script and replace it with the actual batch size (max_num_seqs)
    res = subprocess.run(
        f'sed -i -E "s/(batch_size[[:space:]]*=[[:space:]]*)[0-9]+/\\1{max_num_seqs}/" "{ais_bench_script}"',
        shell=True,
        executable="/bin/bash",
    )
    check_subprocess_result(res, "Failed to update batch_size in ais_bench script")

    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_vllm_server(
        model_path=model_path,
        model_name=model_name,
        port=port,
        tp_size=tp_size,
        ep=ep,
        dp_size=dp_size,
        spec_decode=spec_decode,
        max_num_seqs=max_num_seqs,
        max_model_len=int(max_model_len * 1.5),
        max_num_batched_tokens=max_num_batched_tokens,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        block_size=block_size,
        use_mrv2=use_mrv2,
        xlite=xlite,
        xlite_full_mode=xlite_full_mode,
        device_ids=device_ids,
        timeout=vllm_timeout,
        # redirect the server logs to files next to the benchmark output so they're preserved for inspection.
        # must use dot prefix to avoid --reuse flag in ais_bench picking the wrong working directory
        print_to=ais_bench_output_dir / ".vllm_logs" / f"{time_str}_{model_name}{suffix}",
    )

    try:
        max_cpu_workers = int(os.environ.get("XLITE_AISBENCH_WORKERS", 16))
        aisbench_cmd_lst: list[str] = [
            f"{VENV_ACTIVATE_CMD} &&",
            f"ais_bench --models {spec.model_config}",
            f"--datasets {spec.task}",
            "--mode all --dump-eval-details" + (" --merge-ds" if spec.merge_ds else ""),
            f'--max-num-workers "{min(os.cpu_count() or 4, max_cpu_workers, max_num_seqs)}"',
            f'--work-dir "{ais_bench_output_dir}"',
            f'--num-prompts "{ais_bench_num_prompts}"',
            "--num-warmups 0",
            extra_args,
        ]
        aisbench_cmd = " ".join(aisbench_cmd_lst)

        for i in range(1, retry + 1):
            retry_str = f"[attempt {i}/{retry}] " if retry > 1 else ""
            print(f"\n{retry_str}Running ais_bench for model {model_name}...")
            # if the process failed before the timeout, retry
            if i > 1 and " --reuse" not in aisbench_cmd and " -r" not in aisbench_cmd:
                aisbench_cmd += " --reuse"  # add --reuse flag for subsequent attempts
                print(f"{retry_str}ais_bench failed on previous attempt(s), retrying with --reuse...")
            process = subprocess.Popen(aisbench_cmd, shell=True, start_new_session=True, executable="/bin/bash")
            _PROCESS_GROUPS.add(process)
            print(f"{retry_str}ais_bench started with PID {process.pid} (completion timeout={timeout}s)...")
            try:
                process.wait(timeout=timeout)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"ais_bench exited with code {process.returncode}: {process.stderr.read() if process.stderr else 'no error message'}"
                    )
                break
            except KeyboardInterrupt as e:
                print(f"\n{retry_str}KeyboardInterrupt received, cancelling current attempt of ais_bench...")
                process.kill()
            except subprocess.TimeoutExpired as e:
                process.kill()
                if i == retry:
                    raise RuntimeError(f"aisbench timed out after {timeout}s") from e
            finally:
                process.kill()
                stop_process_group(process, 5)  # ensure the process is stopped before the next attempt

        # from the output directory, find the latest ais_bench_output_dir/*datetime*/summary/summary_*.csv
        summary_path = max(ais_bench_output_dir.glob("*/summary/summary_*.csv"), key=os.path.getmtime)
        df = pd.read_csv(summary_path, index_col=None)
        dataset_re, metric_re = spec.row_filter
        mask = df["dataset"].astype(str).str.match(dataset_re) & df["metric"].astype(str).str.match(metric_re)
        if not mask.any():
            raise ValueError(
                f"Could not find rows with dataset~{dataset_re!r} and metric~{metric_re!r} in summary csv at "
                f"{summary_path} (columns: {list(df.columns)})"
            )
        # average the matched rows: a single row for most datasets, one per subset for e.g. BFCL
        accuracy = float(df.loc[mask, spec.score_column].astype(float).mean())  # type: ignore[index]

        print(f"Finished ais_bench for model {model_name} with {spec.metric_name} accuracy {accuracy:.2f}")

        bench_result.accuracy = accuracy
        bench_result.error = None
    except KeyboardInterrupt as e:
        print(f"KeyboardInterrupt received, stopping vLLM server for model {model_name}...")
        cleanup_on_exit()
        raise e
    except Exception as e:
        print(f"Error occurred while running ais_bench for model {model_name}: {str(e)}")
        bench_result.error = str(e)
    finally:
        print(SEC_SEP_1)
        cleanup_on_exit()

    return bench_result


if __name__ == "__main__":
    epilog = """
Example usage::

    python batch_aisbench.py /path/to/benchmark --num-prompts 64 --model-dir /dir/to/models --models model1 model2 --tps 2 4 --eps 0 1 --xlite 1 0 --datasets ceval gsm8k

This will run the benchmark for (model1, TP2DP1, xlite decode-only, ceval), and (model2, TP4DP1+EP, aclgraph, gsm8k). If\
`--output-dir` is not specified, the outputs will be stored in `/path/to/benchmark/outputs/model1/xlite-decode-only/TP2DP1-ceval`\
and `/path/to/ais_bench_dir/outputs/model2/aclgraph/TP4DP1EP-gsm8k` respectively.\
    """
    __doc__ = f"{__doc__}\n\n{epilog}"

    import argparse

    parser = argparse.ArgumentParser(
        description="Run batch AISBench accuracy benchmark for vLLM served models",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ais_bench_dir",
        type=Path,
        nargs="?",
        default=AIS_BENCH_DIR,
        help="Path to clone ais_bench repo and store datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path to store ais_bench outputs, default to a subdirectory in ais_bench_dir",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to store the final benchmark results log in markdown format, default to a timestamped file in ais_bench_dir/reports",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default="/mnt/sdb/models",
        help="Directory containing models to benchmark, each subdirectory is a model",
    )
    parser.add_argument("--port", type=int, default=8241, help="Port for vLLM server to listen on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for benchmarking")
    parser.add_argument(
        "-GMU",
        "--gpu-memory-utilization",
        type=str,
        nargs="+",
        default=["0.92"],
        help=(
            "GPU memory utilization for vLLM server, between 0 and 1, for each model (e.g., 0.95 for 95%% "
            "utilization; last value used for padding)"
        ),
    )
    parser.add_argument(
        "-N",
        "--num-prompts",
        type=int,
        default=128,
        help=(
            "Default number of prompts within each subset of a dataset for ais_bench evaluation; "
            "a `--datasets` entry overrides it per dataset via `:<num_prompts>`"
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of model subdirectory names to benchmark. Supports N~<model> format (e.g. 3~Qwen3-30B-A3B expands to 3 copies)",
    )
    parser.add_argument(
        "-TP",
        "--tps",
        "--tp-sizes",
        dest="tps",
        type=str,
        nargs="+",
        default=["1"],
        help="Tensor parallel sizes for each model, one of 1/2/4/8/16 (last value used for padding)",
    )
    parser.add_argument(
        "-EP",
        "--eps",
        dest="eps",
        type=str,
        nargs="+",
        default=["1"],
        help="Whether to enable expert parallelism for each model, 1 for yes and 0 for no (last value used for padding)",
    )
    parser.add_argument(
        "-DP",
        "--dps",
        "--dp-sizes",
        dest="dps",
        type=str,
        nargs="+",
        default=["1"],
        help="Data parallel sizes for each model (last value used for padding)",
    )
    speculative_flags: tuple[str, ...] = ("-MTP", "--mtps", "-SPEC", "--specs")
    parser.add_argument(
        *speculative_flags,
        dest="specs",
        type=str,
        nargs="+",
        default=["0"],
        help=(
            "List of spec decode configs for each model, formatted "
            "`<spec_decode_method>:<num_spec_tokens>[:<draft_weights_dir>]` (e.g. `mtp:3`); a bare number N "
            "means `mtp:N` (`0` disables spec decode); an omitted <draft_weights_dir> uses the main model's "
            "own weights, otherwise it is a single directory name resolved as `<main_model_dir>/../<dir>` "
            "(needed by draft-based methods like eagle3/dspark/dflash); entries also support `N~`/`N*` "
            "repeat expansion like --models (last value used for padding)"
        ),
    )
    parser.add_argument(
        "-DS",
        "--datasets",
        dest="datasets",
        type=str,
        nargs="+",
        default=["ceval"],
        help=(
            "List of dataset entries for each model, formatted `<n_dups>~<dataset>:<num_prompts>`: the "
            "`<n_dups>~` repeat prefix and the `:<num_prompts>` suffix are optional (prompts default to "
            "--num-prompts; last entry used for padding). num_prompts is not part of the benchmark identity — "
            "entries differing only in their prompt count count as duplicates. Datasets are downloaded and set "
            "up automatically; multimodal datasets (textvqa/mmmu/mmstar/realworldqa) require a "
            "multimodal-capable served model, bfcl requires the vllm_api_function_call_chat model config. *all* is only"
            "valid for `--env-only` and will download all datasets, but will not run any benchmarks."
        ),
    )
    parser.add_argument(
        "-MNS",
        "--max-num-seqs",
        type=str,
        nargs="+",
        default=["512"],
        help="Max number of sequences for vLLM server and ais_bench (last value used for padding)",
    )
    parser.add_argument(
        "-MML",
        "--max-model-len",
        type=str,
        nargs="+",
        default=["8192"],
        help="Max model length for vLLM server and ais_bench (last value used for padding)",
    )
    parser.add_argument(
        "-MNT",
        "--max-num-batched-tokens",
        type=str,
        nargs="+",
        default=["8192"],
        help="Max number of batched tokens for vLLM server (last value used for padding)",
    )
    parser.set_defaults(use_mrv2=False)
    parser.add_argument(
        "-MRV2",
        "--use-mrv2",
        action="store_true",
        help="Whether to use the vLLM Model Runner V2 for model execution (default: False)",
    )
    parser.add_argument(
        "--no-use-mrv2",
        action="store_false",
        dest="use_mrv2",
        help="Whether to NOT use the vLLM Model Runner V2 for model execution (default: False)",
    )
    parser.add_argument(
        "-xargs",
        "--extra-aisbench-args",
        type=str,
        nargs="+",
        default=[""],
        help=(
            "Extra arguments to pass to the ais_bench command, for each model (last value used for padding). "
            "Values starting with `-` (e.g. `--reuse`) must use the `=` form: `--extra-aisbench-args=\"--reuse\"`"
        ),
    )
    parser.add_argument(
        "-X",
        "--xlite",
        type=str,
        nargs="+",
        default=["2"],
        help=(
            "Whether to use xlite full mode (2), xlite decode-only mode (1), or aclgraph (0) for each model (last "
            "value used for padding when `--broadcast-xlite` is not set; otherwise, values must be unique)"
        ),
    )
    parser.add_argument(
        "-BX",
        "--broadcast-xlite",
        action="store_true",
        help=(
            "This will broadcast the `--xlite` settings for all `--models`, resulting in `n_models x n_xlite` "
            "combinations of benchmarks. `--xlite` values must be unique in this case."
        ),
    )
    parser.add_argument(
        "-ds", "--device-ids", type=int, nargs="*", default=None, help="List of device IDs to use for benchmarking"
    )
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode with more verbose logging")
    parser.add_argument(
        "--dry-run", action="store_true", help="Whether to do a dry run without actually running the benchmarks"
    )
    parser.add_argument("--env-only", action="store_true", help="Whether to only prepare the environment and exit")
    parser.add_argument(
        "--timeout",
        type=int,
        default=2 * 3600,
        help="Timeout in seconds for ais_bench subprocess (default: 2h = 7200s)",
    )
    parser.add_argument(
        "--retry", type=int, default=3, help="Number of retry attempts for ais_bench on failure (default: 3)"
    )
    parser.add_argument(
        "--vllm-timeout", type=int, default=1800, help="Timeout in seconds for vLLM server startup (default: 1800s)"
    )
    args = parser.parse_args()

    print(f"Running batch AISBench with arguments: {args}")

    # dataset entries `<n_dups>~<dataset>:<num_prompts>` (both suffixes optional); `all`
    # stays the env-only token meaning "every dataset"
    raw_dataset_entries = expand_values(args.datasets)
    if "all" in raw_dataset_entries:
        if not args.env_only:
            raise ValueError("`all` is only valid for --env-only and will not run any benchmarks")
        dataset_entries: list[tuple[str, int]] = [(key, args.num_prompts) for key in DATASETS]
    else:
        dataset_entries = [parse_dataset_entry(e, args.num_prompts) for e in raw_dataset_entries]
        invalid = sorted({name for name, _ in dataset_entries if name not in DATASETS})
        if invalid:
            raise ValueError(f"Invalid dataset(s) specified: {invalid}; available datasets: {sorted(DATASETS)}")

    if not args.dry_run or args.env_only:
        # prepare the environment: clone ais_bench repo, create virtual environment, install requirements, download datasets
        prepare_env(args.ais_bench_dir, datasets=[name for name, _ in dataset_entries])
    if args.env_only:
        print("Environment preparation complete, exiting as --env-only was specified.")
        sys.exit(0)

    if not args.models:
        raise ValueError("No models specified for benchmarking, please provide model subdirectory names using --models")

    # every per-model parameter accepts `N~<value>` expansion and pads to n_models
    args_models: list[str] = expand_values(args.models)
    n_models = len(args_models)
    args_tps: list[int] = pad_to_last(expand_values(args.tps, int, "--tps"), n_models)
    args_eps: list[int] = pad_to_last(expand_values(args.eps, int, "--eps"), n_models)
    args_dps: list[int] = pad_to_last(expand_values(args.dps, int, "--dps"), n_models)
    args_specs: list[tuple[str, int, str]] = pad_to_last(
        [parse_spec_decode(s) for s in expand_values(args.specs)], n_models
    )
    dataset_entries = pad_to_last(dataset_entries, n_models)
    args_datasets: list[str] = [d for d, _ in dataset_entries]
    args_num_prompts: list[int] = [n for _, n in dataset_entries]
    args_max_num_seqs: list[int] = pad_to_last(expand_values(args.max_num_seqs, int, "--max-num-seqs"), n_models)
    args_max_model_len: list[int] = pad_to_last(expand_values(args.max_model_len, int, "--max-model-len"), n_models)
    args_max_num_batched_tokens: list[int] = pad_to_last(
        expand_values(args.max_num_batched_tokens, int, "--max-num-batched-tokens"), n_models
    )
    args_gpu_memory_utilization: list[float] = pad_to_last(
        expand_values(args.gpu_memory_utilization, float, "--gpu-memory-utilization"), n_models
    )
    args_extra_aisbench_args: list[str] = pad_to_last(expand_values(args.extra_aisbench_args), n_models)
    if bad := sorted(set(args_tps) - {1, 2, 4, 8, 16}):
        raise ValueError(f"--tps values must be one of 1/2/4/8/16, got {bad}")
    if bad := sorted(set(args_eps) - {0, 1}):
        raise ValueError(f"--eps values must be 0 or 1, got {bad}")
    args_xlite: list[int] = expand_values(args.xlite, int, "--xlite")
    if bad := sorted(set(args_xlite) - {0, 1, 2}):
        raise ValueError(f"--xlite values must be one of 0/1/2, got {bad}")
    if not args.broadcast_xlite:
        args_xlite = pad_to_last(args_xlite, n_models)
    elif len(set(args_xlite)) != len(args_xlite):
        raise ValueError("When using --broadcast-xlite, the values in --xlite must be unique to avoid duplications")
    else:
        n_xlite = len(args_xlite)
        # tile xlite values for each model: [2,1,0] * 3 models = [2,1,0,2,1,0,2,1,0]
        args_xlite = args_xlite * n_models
        # repeat every per-model list n_xlite times: [A,B] with n_xlite=2 -> [A,A,B,B]
        args_models = tile_each(args_models, n_xlite)
        args_tps = tile_each(args_tps, n_xlite)
        args_eps = tile_each(args_eps, n_xlite)
        args_dps = tile_each(args_dps, n_xlite)
        args_specs = tile_each(args_specs, n_xlite)
        args_datasets = tile_each(args_datasets, n_xlite)
        args_num_prompts = tile_each(args_num_prompts, n_xlite)
        args_max_num_seqs = tile_each(args_max_num_seqs, n_xlite)
        args_max_model_len = tile_each(args_max_model_len, n_xlite)
        args_max_num_batched_tokens = tile_each(args_max_num_batched_tokens, n_xlite)
        args_gpu_memory_utilization = tile_each(args_gpu_memory_utilization, n_xlite)
        args_extra_aisbench_args = tile_each(args_extra_aisbench_args, n_xlite)

    combinations = [
        (model, int(tp), ep, int(dp), spec, xlite > 0, xlite > 1, dataset)
        for model, tp, ep, dp, spec, xlite, dataset in zip(
            args_models, args_tps, args_eps, args_dps, args_specs, args_xlite, args_datasets
        )
    ]
    if len(set(combinations)) != len(combinations):
        for dup in {c for c in combinations if combinations.count(c) > 1}:
            print(
                f"Duplicate combination found: Model={dup[0]}, TP={dup[1]}, EP={'Y' if dup[2] else 'N'}, DP={dup[3]}, "
                f"Spec={format_spec_decode(dup[4])}, Xlite={'full' if dup[6] else ('decode-only' if dup[5] else 'aclgraph')}, "
                f"Dataset={dup[7]}"
            )
        raise ValueError("The combination of models, tps, eps, dps, specs, etc., must be unique for each model")

    max_num_devices = max(tp * dp for tp, dp in zip(args_tps, args_dps))
    device_ids_env = filter(None, os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").split(","))
    device_ids = args.device_ids or list(map(int, device_ids_env))
    device_ids = device_ids or list(range(max_num_devices))
    if max_num_devices > len(device_ids):
        raise ValueError(
            f"Not enough device IDs provided for the maximum required devices across all models. "
            f"Max required devices: {max_num_devices}, provided device IDs: {device_ids}"
        )
    print(f"Using device IDs {','.join(map(str, device_ids))} for benchmarking (max required: {max_num_devices})")

    if args.dry_run:
        print("Dry run enabled, not actually running benchmarks. The following combinations would be benchmarked:")
        for i, (model, tp, ep, dp, spec, xlite, xlite_full, dataset) in enumerate(combinations):
            backend_str = "xlite full" if xlite_full else ("xlite decode-only" if xlite else "aclgraph")
            print(
                f"  {i + 1}. Model: {model}, TP: {tp}, EP: {'Y' if ep else 'N'}, DP: {dp}, Spec: {format_spec_decode(spec)}, "
                f"Backend: {backend_str}, Dataset: {dataset}, Num Prompts: {args_num_prompts[i]}"
            )
        sys.exit(0)

    default_suffix = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{get_xlite_version()}"
    log_file_path: Path = args.log_file or args.ais_bench_dir / "reports" / f"benchmark_report_{default_suffix}.md"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    log_csv_path: Path = log_file_path.with_suffix(".csv")

    # Write headers first
    with open(log_file_path, "w") as f:
        f.write(f"# Benchmark Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        # Record the Xlite version for reference
        f.write(f"## Version Information\n\nxlite: `{get_xlite_version()}`\npython: `{sys.version}`\n\n")
        # Record the shell command used to run this script for reference
        command = shlex.join(["python", str(Path(sys.argv[0]).resolve()), *sys.argv[1:]])
        f.write(f"## Command Used\n\n```bash\n{command}\n```\n\n")
        f.write("## Final Benchmark Results:\n\n")
        f.write(f"{BenchResult.markdown_header()}\n")
    with open(log_csv_path, "w") as f:
        f.write("model_name,dataset,tp_size,ep,dp_size,spec_decode,xlite,xlite_full_mode,metric,accuracy,error\n")

    for i, (model, tp, ep, dp, spec, xlite, xlite_full, dataset) in enumerate(combinations):
        print(f"\n>>> Running benchmark task {i + 1}/{len(combinations)}")
        model_path: Path = args.model_dir / model
        try:
            result = run_ais_bench(
                ais_bench_dir=args.ais_bench_dir,
                ais_bench_output_dir=args.output_dir,
                ais_bench_num_prompts=args_num_prompts[i],
                model_path=str(model_path),
                model_name=model,
                dataset=dataset,
                port=args.port,
                tp_size=tp,
                ep=ep,
                dp_size=dp,
                spec_decode=spec,
                max_num_seqs=args_max_num_seqs[i],
                max_model_len=args_max_model_len[i],
                max_num_batched_tokens=args_max_num_batched_tokens[i],
                seed=args.seed,
                gpu_memory_utilization=args_gpu_memory_utilization[i],
                use_mrv2=args.use_mrv2,
                xlite=xlite,
                xlite_full_mode=xlite_full,
                extra_args=args_extra_aisbench_args[i],
                device_ids=device_ids[-tp * dp :],
                timeout=args.timeout,
                retry=args.retry,
                vllm_timeout=args.vllm_timeout,
            )
        except Exception as e:
            result = BenchResult(
                model_name=model,
                dataset=dataset,
                tp_size=tp,
                ep=ep,
                dp_size=dp,
                spec_decode=spec,
                xlite=xlite,
                xlite_full_mode=xlite_full,
                metric=DATASETS[dataset].metric_name,
                accuracy=float("nan"),
            )
            result.error = str(e)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping benchmark...")
            cleanup_on_exit()
            break
        finally:
            # ensure all started vLLM server processes are stopped
            cleanup_on_exit()

        # Append this result to the log files immediately
        with open(log_file_path, "a") as f:
            f.write(f"{result.markdown_row()}\n")
        with open(log_csv_path, "a") as f:
            f.write(
                f"{result.model_name},{result.dataset},{result.tp_size},{result.ep},{result.dp_size},"
                f"{result.spec_label},{result.xlite},{result.xlite_full_mode},{result.metric},{result.accuracy},"
                f"{result.error or ''}\n"
            )
        print(result.markdown_row())
        print(f"Completed benchmark task {i + 1}/{len(combinations)}")

    print(f"\nBenchmark results written to log file at {log_file_path}")
    print(f"Benchmark results also written to CSV file at {log_csv_path}")
