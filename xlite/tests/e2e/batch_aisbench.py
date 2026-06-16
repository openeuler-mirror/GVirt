"""
This script uses `ais_bench` to benchmark the accuracy (correctness on predefined datasets) of served models.

Only works for *nix systems with `bash` available, as it relies on `source` to activate the Python virtual environment.
Additionally, the script is designed to run within a container with `git`, `vllm`, `vllm_ascend`, and `xlite` installed.
Using `vllm-ascend`'s official container image is recommended.
"""

from dataclasses import dataclass, field
from datetime import datetime
import os
import signal
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import zipfile

import pandas as pd

TMP_DIR = Path(tempfile.gettempdir())  # per-OS temp base path
CWD = Path(__file__).parent.resolve()  # current working directory

AIS_BENCH_DIR = TMP_DIR / "benchmark"  # path to clone ais_bench repo and store datasets
VENV_ACTIVATE_CMD: str = ""

VLLM_PROCESSES: set[subprocess.Popen] = set()  # to keep track of started vLLM server processes for cleanup


@dataclass(kw_only=True, slots=True)
class BenchResult:
    model_name: str
    dataset: str
    tp_size: int
    ep: bool
    dp_size: int
    xlite: bool
    xlite_full_mode: bool
    metric: str
    accuracy: float

    _error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.xlite = self.xlite or self.xlite_full_mode
        self.xlite_full_mode = self.xlite_full_mode and self.xlite

    @property
    def error(self) -> str | None:
        return getattr(self, "_error", None)

    @error.setter
    def error(self, value: str | None):
        value = value and value.replace("\n", " ")
        if value and len(value) > 140:
            value = f"{value[:137]}..."
        self._error = value

    @staticmethod
    def markdown_header() -> str:
        fields = ["Model", "Dataset", "TP", "EP", "DP", "Backend", "Metric", "Accuracy", "Error"]
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
            backend_str,
            self.metric,
            f"{self.accuracy:.2f}",
            self.error or "",
        ]
        return f"| {' | '.join(values)} |"


def stop_vllm_server(process: subprocess.Popen, sleep_time: int = 120):
    pid = process.pid
    print(f"Stopping vLLM server with PID {pid}...")
    t1 = datetime.now()
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        print(f"Process with PID {pid} not found, it may have already exited.")
        VLLM_PROCESSES.discard(process)
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

    VLLM_PROCESSES.discard(process)


def check_subprocess_result(res: subprocess.CompletedProcess, error_message: str):
    if res.returncode != 0:
        raise RuntimeError(f"{error_message}: {res.stderr.decode('utf-8') if res.stderr else 'No error message'}")


def prepare_env(
    ais_bench_dir: Path | None = None,
    aisbench_git_url: str = "https://github.com/AISBench/benchmark.git",
    aisbench_git_branch: str = "v3.1-20260330-master",
    ceval_dataset_url: str = "https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip",
):
    global VENV_ACTIVATE_CMD
    ais_bench_dir = ais_bench_dir or AIS_BENCH_DIR

    # check if benchmark repo exists, if not, clone it
    if not ais_bench_dir.exists():
        print(f"Cloning ais_bench repo to {ais_bench_dir}...")
        res = subprocess.run(
            ["git", "clone", "--depth", "1", "-b", aisbench_git_branch, aisbench_git_url, ais_bench_dir]
        )
        check_subprocess_result(res, "Failed to clone ais_bench repo")

    # make a `.venv` in the benchmark repo for ais_bench, and install requirements
    os.chdir(ais_bench_dir)
    venv_path = ais_bench_dir / ".venv"
    if not venv_path.exists():
        print(f"Creating virtual environment for ais_bench at {venv_path}...")
        res = subprocess.run([sys.executable, "-m", "venv", str(venv_path)])
        check_subprocess_result(res, "Failed to create virtual environment")

    # activate the virtual environment and install requirements
    activate_script = venv_path / "bin" / "activate"
    VENV_ACTIVATE_CMD = f"source {activate_script}"
    print("Activating virtual environment and installing ais_bench...")

    res = subprocess.run(f"{VENV_ACTIVATE_CMD} && command -v ais_bench", shell=True, executable="/bin/bash")
    if res.returncode != 0:
        print("ais_bench not found, installing requirements...")
        res = subprocess.run(
            f"{VENV_ACTIVATE_CMD} && pip install -r requirements/api.txt && pip install -r requirements/extra.txt "
            "&& pip install -e ./ --use-pep517",
            shell=True,
            executable="/bin/bash",
        )
        check_subprocess_result(res, "Failed to install ais_bench requirements")
    else:
        print("ais_bench already installed, skipping installation...")

    # check if the ceval dataset is downloaded
    dataset_dir = ais_bench_dir / "ais_bench" / "datasets" / "ceval" / "formal_ceval"
    if not dataset_dir.exists() or not (dataset_dir / "test").exists():
        print(f"Downloading ceval dataset to {dataset_dir}...")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dataset_dir / "ceval-exam.zip"
        res = subprocess.run(["wget", "-O", str(zip_path), ceval_dataset_url])
        if res.returncode == 0:
            with zipfile.ZipFile(zip_path) as zip_file:
                zip_file.extractall(dataset_dir)
            zip_path.unlink(missing_ok=True)
        check_subprocess_result(res, "Failed to download ceval dataset")


def start_vllm_server(
    model_path: str,
    *,
    model_name: str | None = None,
    quantization: bool = False,
    port: int = 8384,
    tp_size: int = 1,
    ep: bool = True,
    dp_size: int = 1,
    max_num_seqs: int = 128,
    max_model_len: int = 8192,
    max_num_batched_tokens: int = 8192,
    seed: int = 42,
    gpu_memory_utilization: float = 0.9,
    block_size: int = 128,
    xlite: bool = False,
    xlite_full_mode: bool = False,
    device_ids: list[int] | None = None,
    timeout: int = 1800,
    debug: bool = False,
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
    print(f"Starting vLLM server with model {model_path} on port {port}...")

    # check if port is available
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        result = sock.connect_ex(("localhost", port))
        if result == 0:
            raise RuntimeError(f"Port {port} is already in use, please specify a different port")

    envs: dict[str, str] = {
        "VLLM_USE_V1": "1",
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
    os.environ.update(envs)

    model_name = model_name or Path(model_path).stem or "model"
    xlite = xlite or xlite_full_mode  # if full mode is enabled, xlite must be enabled as well
    xlite_full_mode = xlite_full_mode and xlite  # full mode can only be enabled when xlite is enabled
    vllm_cmd_lst: list[str] = [
        f'vllm serve "{model_path}" --async-scheduling --trust-remote-code',
        f'--served-model-name "{model_name}"',
        f'--port "{port}"',
        "--nnodes 1",
        f'--tensor-parallel-size "{tp_size}"',
        f"{'--enable-expert-parallel' if ep else ''}",
        f'--data-parallel-size "{dp_size}"',
        f'--data-parallel-size-local "{dp_size}"',
        f'--max-num-seqs "{max_num_seqs}"',
        f'--max-model-len "{max_model_len}"',
        f'--max-num-batched-tokens "{max_num_batched_tokens}"',
        f'--seed "{seed}"',
        f'--gpu-memory-utilization "{gpu_memory_utilization}"',
        f'--block-size "{block_size}"',
        '--compilation-config \'{"cudagraph_capture_sizes":[1,2,4,8,16,32,64], "cudagraph_mode": "FULL_DECODE_ONLY"}\'',
        f'--additional-config \'{{"xlite_graph_config": {{"enabled": {str(xlite).lower()}, "full_mode": {str(xlite_full_mode).lower()}}}}}\'',
    ]
    if quantization:
        vllm_cmd_lst.append("--quantization ascend")
    vllm_cmd = " ".join(vllm_cmd_lst)

    os.environ["XLITE_NODE_IPS"] = ",".join(["127.0.0.1"] * dp_size)
    os.environ["XLITE_LOCAL_IP"] = "127.0.0.1"
    os.environ["XLITE_DEVS_PER_NODE"] = str(dp_size * tp_size)

    t1 = datetime.now()
    # ignore stdout and stderr of the server process, as it may contain a lot of logs
    process = subprocess.Popen(
        vllm_cmd,
        shell=True,
        executable="/bin/bash",
        env=os.environ.copy(),
        start_new_session=True,  # to allow killing the whole process group later
        stdout=None if debug else subprocess.PIPE,
        stderr=None if debug else subprocess.PIPE,
    )

    # check if the server is ready by checking port availability
    try:
        while (datetime.now() - t1).total_seconds() < timeout:
            if process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server process exited unexpectedly with code {process.returncode}:\n"
                    f"{process.stderr.read().decode('utf-8') if process.stderr else 'No error message'}"
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(("localhost", port)) == 0:
                    break  # successfully connected to the port, server is ready
            time.sleep(1)
        else:  # if `break` was not hit and loop exhausted
            raise TimeoutError(f"vLLM server did not start within {timeout} seconds")
    except Exception as e:
        process.terminate()
        raise e

    VLLM_PROCESSES.add(process)
    print(f"vLLM server started successfully with PID {process.pid}")
    return process


def run_ais_bench(
    *,
    ais_bench_dir: Path | None = None,
    ais_bench_output_dir: Path | None = None,
    ais_bench_num_prompts: int = 128,
    model_path: str = "",
    model_name: str | None = None,
    quantization: bool = False,
    port: int = 8384,
    tp_size: int = 1,
    ep: bool = True,
    dp_size: int = 1,
    max_num_seqs: int = 128,
    max_model_len: int = 8192,
    max_num_batched_tokens: int = 8192,
    seed: int = 42,
    gpu_memory_utilization: float = 0.9,
    block_size: int = 128,
    xlite: bool = False,
    xlite_full_mode: bool = False,
    extra_args: str = "",
    device_ids: list[int] | None = None,
    timeout: int = 1800,
    debug: bool = False,
) -> BenchResult:
    model_name = model_name or Path(model_path).stem or "model"
    xlite = xlite or xlite_full_mode  # if full mode is enabled, xlite must be enabled as well
    xlite_full_mode = xlite_full_mode and xlite  # full mode can only be enabled when xlite is enabled
    bench_result = BenchResult(
        model_name=model_name,
        dataset="ceval-weighted",
        tp_size=tp_size,
        ep=ep,
        dp_size=dp_size,
        xlite=xlite,
        xlite_full_mode=xlite_full_mode,
        metric="weighted_average",
        accuracy=0.0,
    )

    if not model_path or not Path(model_path).exists():
        print(f"Model path {model_path} does not exist, skipping benchmark for model {model_name}...")
        bench_result.error = "Model weights not found"
        return bench_result

    SEC_SEP_1, SEC_SEP_2 = "=" * 50, "-" * 50
    print()
    print(SEC_SEP_1)
    print(
        f"Running ais_bench for model {model_name} on port {port}, using the `ceval` dataset with the first "
        f"{ais_bench_num_prompts} prompts from each subset..."
    )
    print(SEC_SEP_2)
    print(f"Model path: {model_path}")
    print(f"Model name: {model_name}")
    print(f"ASCEND_RT_VISIBLE_DEVICES: {','.join(map(str, device_ids)) if device_ids else 'Not set'}")
    print(f"Quantization: {'Yes' if quantization else 'No'}")
    print(f"Tensor parallel size: {tp_size}")
    print(f"Expert parallel: {'Yes' if ep else 'No'}")
    print(f"Data parallel size: {dp_size}")
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
        ais_bench_dir / "ais_bench" / "benchmark" / "configs" / "models" / "vllm_api" / "vllm_api_general_chat.py"
    )
    if not ais_bench_script.exists():
        raise FileNotFoundError(f"ais_bench script not found at {ais_bench_script}")

    if xlite:
        suffix = "-xlite-full" if xlite_full_mode else "-xlite-decode-only"
    else:
        suffix = "-aclgraph"
    ais_bench_output_dir = (ais_bench_output_dir or ais_bench_dir / "outputs") / f"{model_name}{suffix}"
    if not ais_bench_output_dir.exists():
        ais_bench_output_dir.mkdir(parents=True)

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

    vllm_process = start_vllm_server(
        model_path=model_path,
        model_name=model_name,
        quantization=quantization,
        port=port,
        tp_size=tp_size,
        ep=ep,
        dp_size=dp_size,
        max_num_seqs=max_num_seqs,
        max_model_len=int(max_model_len * 1.5),
        max_num_batched_tokens=max_num_batched_tokens,
        seed=seed,
        gpu_memory_utilization=gpu_memory_utilization,
        block_size=block_size,
        xlite=xlite,
        xlite_full_mode=xlite_full_mode,
        device_ids=device_ids,
        timeout=timeout,
        debug=debug,
    )
    if vllm_process.stdout:
        vllm_process.stdout.close()
    if vllm_process.stderr:
        vllm_process.stderr.close()

    try:
        aisbench_cmd_lst: list[str] = [
            f"{VENV_ACTIVATE_CMD} &&",
            "ais_bench --models vllm_api_general_chat",
            "--datasets ceval_gen_0_shot_cot_chat_prompt",
            "--mode all --dump-eval-details --merge-ds",
            f'--max-num-workers "{min(os.cpu_count() or 4, 64, max_num_seqs)}"',
            f'--work-dir "{ais_bench_output_dir}"',
            f'--num-prompts "{ais_bench_num_prompts}"',
            extra_args,
        ]
        aisbench_cmd = " ".join(aisbench_cmd_lst)
        res = subprocess.run(aisbench_cmd, shell=True, capture_output=not debug, executable="/bin/bash")
        check_subprocess_result(res, "Failed to run ais_bench")

        # from the output directory, find the latest ais_bench_output_dir/*datetime*/summary/summary_*.csv
        summary_path = max(ais_bench_output_dir.glob("*/summary/summary_*.csv"), key=os.path.getmtime)
        df = pd.read_csv(summary_path, index_col=None)
        mask = (df["dataset"] == "ceval-weighted") & (df["metric"] == "weighted_average")
        if not mask.any():
            raise ValueError(f"Could not find weighted average accuracy in summary csv at {summary_path}")
        weighted_accuracy = float(df.loc[mask, "vllm-api-general-chat"].values[-1])

        print(f"Finished ais_bench for model {model_name} with weighted accuracy {weighted_accuracy:.2f}")

        bench_result.accuracy = weighted_accuracy
        bench_result.error = None
    except Exception as e:
        print(f"Error occurred while running ais_bench for model {model_name}: {str(e)}")
        bench_result.error = str(e)
    finally:
        print(SEC_SEP_1)
        stop_vllm_server(vllm_process)

    return bench_result


if __name__ == "__main__":
    epilog = """
Example usage::

    python batch_aisbench.py /path/to/benchmark --num-prompts 64 --model-dir /dir/to/models --models model1 model2 --tps 2 4 --eps 0 1 --xlite 1 0

This will run the benchmark for (model1, TP2EP0, xlite decode-only), and (model2, TP4EP4, aclgraph). If `--output-dir` is\
not specified, the outputs will be stored in `/path/to/benchmark/outputs/model1-xlite-decode-only` and\
`/path/to/ais_bench_dir/outputs/model2-aclgraph` respectively.\
    """
    __doc__ = f"{__doc__}\n\n{epilog}"

    import argparse

    parser = argparse.ArgumentParser(
        description="Run batch AISBench accuracy benchmark for vLLM served models",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "ais_bench_dir", type=Path, default=AIS_BENCH_DIR, help="Path to clone ais_bench repo and store datasets"
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
        required=True,
        help="Directory containing models to benchmark, each subdirectory is a model",
    )
    parser.add_argument("--port", type=int, default=8384, help="Port for vLLM server to listen on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for benchmarking")
    parser.add_argument(
        "-GMU",
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM server, between 0 and 1 (e.g., 0.9 for 90%% utilization)",
    )
    parser.add_argument(
        "-N",
        "--num-prompts",
        type=int,
        default=128,
        help="Number of prompts within each subset of datasets to use for ais_bench evaluation",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None, help="List of model subdirectory names to benchmark"
    )
    parser.add_argument(
        "-Q",
        "--quantization",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=[0],
        help="Whether the model is quantized (1 for yes, 0 for no; last value used for padding)",
    )
    parser.add_argument(
        "-TP",
        "--tps",
        "--tp-sizes",
        dest="tps",
        type=int,
        nargs="+",
        choices=[1, 2, 4, 8, 16],
        default=[1],
        help="List of tensor parallel sizes for each model (last value used for padding)",
    )
    parser.add_argument(
        "-EP",
        "--eps",
        dest="eps",
        type=int,
        nargs="+",
        choices=[0, 1],
        default=[1],
        help="Whether to enable expert parallelism for each model (1 for yes, 0 for no; last value used for padding)",
    )
    parser.add_argument(
        "-DP",
        "--dps",
        "--dp-sizes",
        dest="dps",
        type=int,
        nargs="+",
        default=[1],
        help="List of data parallel sizes for each model (last value used for padding)",
    )
    parser.add_argument(
        "-MNS",
        "--max-num-seqs",
        type=int,
        nargs="+",
        default=[128],
        help="Max number of sequences for vLLM server and ais_bench (last value used for padding)",
    )
    parser.add_argument(
        "-MML",
        "--max-model-len",
        type=int,
        nargs="+",
        default=[8192],
        help="Max model length for vLLM server and ais_bench (last value used for padding)",
    )
    parser.add_argument(
        "-MNT",
        "--max-num-batched-tokens",
        type=int,
        nargs="+",
        default=[8192],
        help="Max number of batched tokens for vLLM server (last value used for padding)",
    )
    parser.add_argument(
        "--extra-aisbench-args",
        type=str,
        nargs="+",
        default=[""],
        help="Extra arguments to pass to ais_bench command (last value used for padding)",
    )
    parser.add_argument(
        "-X",
        "--xlite",
        type=int,
        nargs="+",
        choices=[0, 1, 2],
        default=[2],
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
    args = parser.parse_args()

    print(f"Running batch AISBench with arguments: {args}")

    prepare_env(args.ais_bench_dir)

    bench_results: list[BenchResult] = []

    if not args.models:
        raise ValueError("No models specified for benchmarking, please provide model subdirectory names using --models")

    args_models: list[str] = args.models
    n_models = len(args_models)
    args_quantization: list[int] = args.quantization + [args.quantization[-1]] * (n_models - len(args.quantization))
    args_tps: list[int] = args.tps + [args.tps[-1]] * (n_models - len(args.tps))
    args_eps: list[bool] = args.eps + [args.eps[-1]] * (n_models - len(args.eps))
    args_dps: list[int] = args.dps + [args.dps[-1]] * (n_models - len(args.dps))
    args_max_num_seqs: list[int] = args.max_num_seqs + [args.max_num_seqs[-1]] * (n_models - len(args.max_num_seqs))
    args_max_model_len: list[int] = args.max_model_len + [args.max_model_len[-1]] * (n_models - len(args.max_model_len))
    args_max_num_batched_tokens: list[int] = args.max_num_batched_tokens + [args.max_num_batched_tokens[-1]] * (
        n_models - len(args.max_num_batched_tokens)
    )
    args_extra_aisbench_args: list[str] = args.extra_aisbench_args + [args.extra_aisbench_args[-1]] * (
        n_models - len(args.extra_aisbench_args)
    )

    if not args.broadcast_xlite:
        args_xlite = args.xlite + [args.xlite[-1]] * (n_models - len(args.xlite))
    elif len(set(args.xlite)) != len(args.xlite):
        raise ValueError("When using --broadcast-xlite, the values in --xlite must be unique to avoid duplications")
    else:
        n_xlite = len(args.xlite)
        # Tile xlite values for each model: [1,0] * 3 = [1,0,1,0,1,0]
        args_xlite = args.xlite * n_models
        # Repeat each model's args n_xlite times: [A,B,C] with n_xlite=2 -> [A,A,B,B,C,C]
        args_models = [m for m in args_models for _ in range(n_xlite)]
        args_quantization = [q for q in args_quantization for _ in range(n_xlite)]
        args_tps = [tp for tp in args_tps for _ in range(n_xlite)]
        args_eps = [ep for ep in args_eps for _ in range(n_xlite)]
        args_dps = [dp for dp in args_dps for _ in range(n_xlite)]
        args_max_num_seqs = [m for m in args_max_num_seqs for _ in range(n_xlite)]
        args_max_model_len = [m for m in args_max_model_len for _ in range(n_xlite)]
        args_max_num_batched_tokens = [m for m in args_max_num_batched_tokens for _ in range(n_xlite)]
        args_extra_aisbench_args = [a for a in args_extra_aisbench_args for _ in range(n_xlite)]

    combinations = [
        (model, int(tp), ep, int(dp), xlite > 0, xlite > 1)
        for model, tp, ep, dp, xlite in zip(args_models, args_tps, args_eps, args_dps, args_xlite)
    ]
    if len(set(combinations)) != len(combinations):
        raise ValueError("The combination of models, tp_sizes, and ep_sizes must be unique for each model")

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
        for i, (model, tp, ep, dp, xlite, xlite_full) in enumerate(combinations):
            backend_str = "xlite full" if xlite_full else ("xlite decode-only" if xlite else "aclgraph")
            print(f"  {i + 1}. Model: {model}, TP: {tp}, EP: {'Y' if ep else 'N'}, DP: {dp}, Backend: {backend_str}")
        sys.exit(0)

    for i, (model, tp, ep, dp, xlite, xlite_full) in enumerate(combinations):
        print(f"\n>>> Running benchmark task {i + 1}/{len(combinations)}")
        model_path = args.model_dir / model
        try:
            result = run_ais_bench(
                ais_bench_dir=args.ais_bench_dir,
                ais_bench_output_dir=args.output_dir,
                ais_bench_num_prompts=args.num_prompts,
                model_path=str(model_path),
                model_name=model,
                quantization=bool(args_quantization[i]),
                port=args.port,
                tp_size=tp,
                ep=ep,
                dp_size=dp,
                max_num_seqs=args_max_num_seqs[i],
                max_model_len=args_max_model_len[i],
                max_num_batched_tokens=args_max_num_batched_tokens[i],
                seed=args.seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
                xlite=xlite,
                xlite_full_mode=xlite_full,
                extra_args=args_extra_aisbench_args[i],
                device_ids=device_ids[-tp * dp :],
                debug=args.debug,
            )
            bench_results.append(result)
        except Exception as e:
            result = BenchResult(
                model_name=model,
                dataset="ceval-weighted",
                tp_size=tp,
                ep=ep,
                dp_size=dp,
                xlite=xlite,
                xlite_full_mode=xlite_full,
                metric="weighted_average",
                accuracy=0.0,
            )
            result.error = str(e)
            bench_results.append(result)
        finally:
            # ensure all started vLLM server processes are stopped
            for process in list(VLLM_PROCESSES):
                stop_vllm_server(process)
        print(f"Completed benchmark task {i + 1}/{len(combinations)}")

    log_text = ""
    if bench_results:
        log_text += "\n\nFinal Benchmark Results:\n\n"
        log_text += f"{BenchResult.markdown_header()}\n"
        for result in bench_results:
            log_text += f"{result.markdown_row()}\n"
    else:
        log_text += "No benchmark results to display.\n"

    print(log_text, end="")

    log_file_path: Path = (
        args.log_file
        or args.ais_bench_dir / "reports" / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_path, "w") as f:
        f.write(log_text)
    print(f"\nBenchmark results written to log file at {log_file_path}")

    # save as csv as well for easier parsing
    log_csv_path: Path = log_file_path.with_suffix(".csv")
    df = pd.DataFrame(
        [
            {
                "model_name": result.model_name,
                "dataset": result.dataset,
                "tp_size": result.tp_size,
                "ep": result.ep,
                "dp_size": result.dp_size,
                "xlite": result.xlite,
                "xlite_full_mode": result.xlite_full_mode,
                "metric": result.metric,
                "accuracy": result.accuracy,
                "error": result.error,
            }
            for result in bench_results
        ]
    )
    df.to_csv(log_csv_path, index=False)
    print(f"Benchmark results also written to CSV file at {log_csv_path}")
