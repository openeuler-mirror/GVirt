#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AISBench dataset registry shared by batch_aisbench.py and daily_aisbench_bot.py.

Holds one :class:`DatasetSpec` per supported dataset: how to download it, which
ais_bench task/model-config/score-column to use, and the accuracy thresholds the
accuracy bot applies to the xlite backends.
"""

import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from daily_bot_utils import check_subprocess_result


def venv_activate_cmd(ais_bench_dir: Path) -> str:
    """bash command activating the ais_bench venv (namespaced by Python version)."""
    activate = ais_bench_dir / f".venv_py{sys.version_info.major}{sys.version_info.minor}" / "bin" / "activate"
    return f"source {activate.resolve()}"


def _require_git_lfs() -> None:
    if subprocess.run(["git", "lfs", "version"], capture_output=True).returncode != 0:
        raise RuntimeError("git-lfs is required for this dataset; install it (e.g. `apt install git-lfs`) and rerun")


def _wget(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    args: list[str] = ["wget", "-O", str(target), url]
    if os.environ.get("XLITE_AISBENCH_NO_CHECK_CERT", "0") == "1":
        args.insert(1, "--no-check-certificate")
    check_subprocess_result(subprocess.run(args), f"Failed to download {url}")


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
        f"{venv_activate_cmd(ais_bench_dir)} && pip install -r requirements/datasets/bfcl_dependencies.txt --no-deps",
        shell=True,
        executable="/bin/bash",
        cwd=ais_bench_dir,
        capture_output=True,
    )
    check_subprocess_result(res, "Failed to install bfcl-eval dependencies")


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the harness needs to set up, run, parse and judge one ais_bench dataset."""

    task: str  # value forwarded to `ais_bench --datasets`
    model_config: str  # ais_bench model config (file stem under configs/models/vllm_api)
    score_column: str  # summary CSV column holding the score (the model config's `abbr`)
    row_filter: tuple[str, str]  # (dataset, metric) regexes selecting the summary rows to average
    metric_name: str  # metric label used in reports
    downloader: Callable[[Path, str], None]  # populates the dataset under ais_bench_dir
    # accuracy judgment for the nightly report: `min_accuracy` is the absolute floor for the
    # xlite backends, `gap_threshold` the max tolerated drop vs the aclgraph baseline (which
    # itself is never threshold-checked). Floors sit well below typical published open-model
    # scores — the nightly is a reference signal, and aclgraph is the real regression baseline.
    min_accuracy: float
    gap_threshold: float
    merge_ds: bool = True  # whether `--merge-ds` may be passed to ais_bench
    ready_path: str | None = None  # repo-relative file/dir whose existence means "already downloaded"
    # bumped whenever this dataset's downloader behavior changes, so previously downloaded
    # data is refreshed to the current standard instead of being skipped as "ready"
    setup_revision: int = 1
    # All sources are China-accessible (ModelScope / Aliyun OSS) — huggingface.co is not
    # reliably reachable from where this harness runs.
    download_url: str = ""  # primary source URL (zip/TSV/git repo); "" = nothing to download (bfcl ships via pip)


# Threshold rationale: ceval 80/4 measured stable across nightlies and matches the
# published low-80s of the GLM/Qwen class served here; gsm8k floors 10 points under the
# ~95.5 the Qwen3 models report; the multimodal floors trail the Qwen3-VL numbers
# (MMMU ~75, MMStar ~72, TextVQA ~84, RealWorldQA ~74) by ~20-30 points because these
# run on prompt subsets (mmmu:256) where sampling noise alone is ~±5; BFCL v3 spans
# ~69-78 on recent open models but its multi-turn categories are volatile over an API
# harness. Gap thresholds get 6 on the noisy subsets vs 4 on ceval/gsm8k.
DATASETS: dict[str, DatasetSpec] = {
    # text-only datasets
    "ceval": DatasetSpec(
        task="ceval_gen_0_shot_cot_chat_prompt",
        model_config="vllm_api_general_chat",
        score_column="vllm-api-general-chat",
        row_filter=(r"^ceval-weighted$", r"^weighted_average$"),
        metric_name="weighted_average",
        downloader=_download_ceval,
        min_accuracy=80,
        gap_threshold=4,
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
        min_accuracy=85,
        gap_threshold=4,
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
        min_accuracy=60,
        gap_threshold=4,
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
        min_accuracy=60,
        gap_threshold=4,
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
        min_accuracy=60,
        gap_threshold=4,
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
        min_accuracy=60,
        gap_threshold=4,
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
        min_accuracy=60,
        gap_threshold=4,
        merge_ds=False,  # BFCL sub-datasets use different evaluators and cannot be merged
    ),
}


def ensure_dataset(key: str, ais_bench_dir: Path) -> None:
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
