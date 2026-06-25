# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.

"""Utilities for downloading and initializing model weights."""
import glob
import json
import os
import stat
import sys
import logging
from collections import defaultdict
from typing import Any, Iterator, List, Optional, Tuple

import filelock
import numpy as np
import torch
import torch_npu
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file, safe_open
from tqdm.auto import tqdm


ACL_FORMAT_FRACTAL_NZ = 29


def rearrange_matrix(matrix, n0, k_block_size=16):
    """nd2nz"""
    assert matrix.size(0) % n0 == 0, f"matrix.size0: {matrix.size(0)} n0: {n0}"
    assert matrix.size(1) % k_block_size == 0, f"matrix.size1: {matrix.size(1)} k_block_size: {k_block_size}"

    reshaped = matrix.view(matrix.size(0) // n0, n0, matrix.size(1) // k_block_size, k_block_size)
    final_reshaped = reshaped.permute(0, 2, 1, 3).contiguous()
    return final_reshaped.view(matrix.size(0), matrix.size(1))


def matrix_nd2nz(matrix):
    """nd2nz"""
    return torch_npu.npu_format_cast(matrix, ACL_FORMAT_FRACTAL_NZ)

def load_tensor(path: str):
    data = torch.jit.load(path)
    return data.state_dict()['0'].to("npu")

def setup_logger():
    """ init logger """
    _logger = logging.getLogger(__name__)

    _logger.setLevel(logging.DEBUG)

    logfmt = "%(levelname)s %(asctime)s.%(msecs)03d %(filename)s:%(lineno)d] %(message)s"
    datefmt = "%m-%d %H:%M:%S"

    log_format = logging.Formatter(logfmt, datefmt=datefmt)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.flush = sys.stdout.flush
    console_handler.setFormatter(log_format)

    _logger.addHandler(console_handler)

    return _logger

logger = setup_logger()


class Disabledtqdm(tqdm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, disable=True)


def get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir if cache_dir is not None else "/tmp"
    lock_file_name = model_name_or_path.replace("/", "-") + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name))
    return lock


def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing


def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu")
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """)

    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")


def prepare_hf_model_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_safetensors: bool = False,
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    if use_safetensors:
        allow_patterns = ["*.safetensors"]
    else:
        # Some quantized models use .pt files for storing the weights.
        allow_patterns = ["*.bin", "*.pt"]
    hf_folder = model_name_or_path
    if not is_local:
        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model_name_or_path, cache_dir):
            hf_folder = snapshot_download(model_name_or_path,
                                          allow_patterns=allow_patterns,
                                          cache_dir=cache_dir,
                                          tqdm_class=Disabledtqdm,
                                          revision=revision)
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
    if not use_safetensors:
        hf_weights_files = [
            x for x in hf_weights_files if not x.endswith("training_args.bin")
        ]

    if len(hf_weights_files) == 0 and use_safetensors and fall_back_to_pt:
        out = prepare_hf_model_weights(model_name_or_path,
                                        cache_dir=cache_dir,
                                        use_safetensors=False,
                                        fall_back_to_pt=False,
                                        revision=revision)
    else:
        out = (hf_folder, hf_weights_files, use_safetensors)
    return out


def hf_model_weights_iterator(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    revision: Optional[str] = None,
) -> Iterator[Tuple[str, torch.Tensor]]:
    use_safetensors = False
    use_np_cache = False
    fall_back_to_pt = False
    if load_format == "auto":
        use_safetensors = True
        fall_back_to_pt = True
    elif load_format == "safetensors":
        use_safetensors = True
    elif load_format == "pt":
        pass
    elif load_format == "npcache":
        use_np_cache = True
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    hf_folder, hf_weights_files, use_safetensors = prepare_hf_model_weights(
        model_name_or_path,
        cache_dir=cache_dir,
        use_safetensors=use_safetensors,
        fall_back_to_pt=fall_back_to_pt,
        revision=revision)

    if use_np_cache:
        # Currently np_cache only support *.bin checkpoints
        assert use_safetensors is False

        # Convert the model weights from torch tensors to numpy arrays for
        # faster loading.
        np_folder = os.path.join(hf_folder, "np")
        os.makedirs(np_folder, exist_ok=True)
        weight_names_file = os.path.join(np_folder, "weight_names.json")
        # Use file lock to prevent multiple processes from
        # dumping the same model weights to numpy at the same time.
        with get_lock(model_name_or_path, cache_dir):
            if not os.path.exists(weight_names_file):
                weight_names = []
                for bin_file in hf_weights_files:
                    state = torch.load(bin_file, map_location="cpu")
                    for name, param in state.items():
                        param_path = os.path.join(np_folder, name)

                        with os.fdopen(os.open(param_path, os.O_WRONLY, stat.S_IWUSR), 'w') as f:
                            np.save(f, param.cpu().detach().numpy())
                        weight_names.append(name)
                with os.fdopen(os.open(weight_names_file, os.O_WRONLY, stat.S_IWUSR), 'w') as f:
                    json.dump(weight_names, f)

        with open(weight_names_file, "r") as f:
            weight_names = json.load(f)

        for name in weight_names:
            param_path = os.path.join(np_folder, name)
            with open(param_path, "rb") as f:
                param = np.load(f)
            yield name, torch.from_numpy(param)
    elif use_safetensors:
        for st_file in hf_weights_files:
            with safe_open(st_file, framework="pt") as f:
                for name in f.keys():
                    param = f.get_slice(name)
                    yield name, param
    else:
        for bin_file in hf_weights_files:
            state = torch.load(bin_file, map_location="cpu")
            for name, param in state.items():
                yield name, param
            del state


def convert_pyslice_to_tensor(x: Any) -> torch.Tensor:
    """convert PySafeSlice object from safetensors to torch.Tensor

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    """
    if not isinstance(x, torch.Tensor):
        x = x[:]
    return x


def load_tensor_parallel_weights(
    param: torch.Tensor,
    loaded_weight: Any,  # `torch.Tensor` or `PySafeSlice`
    m: int,
    n: int,
    param_name: str,
    is_row_parallel: bool,
    is_transpose: bool,
    tp_rank: int,
    tp_size: int,
) -> None:
    """load tensor parallel weights"""
    m_shard = m
    n_shard = n
    if is_row_parallel:
        m_shard = m // tp_size
    else:
        n_shard = n // tp_size

    if is_row_parallel:
        start_idx = tp_rank * m_shard
        end_idx = (tp_rank + 1) * m_shard
        if is_transpose:
            loaded_weight_slice = loaded_weight[:, start_idx:end_idx]
        else:
            loaded_weight_slice = loaded_weight[start_idx:end_idx]
    else:
        start_idx = tp_rank * n_shard
        end_idx = (tp_rank + 1) * n_shard
        if is_transpose:
            loaded_weight_slice = loaded_weight[start_idx:end_idx]
        else:
            loaded_weight_slice = loaded_weight[:, start_idx:end_idx]

    loaded_weight_slice = convert_pyslice_to_tensor(loaded_weight_slice)
    if param.shape != loaded_weight_slice.shape:
        logger.warning(f"%s model shape(%s) mismatch checkpoint shape(%s)",
                       param_name,
                       param.shape,
                       loaded_weight_slice.shape)

    if is_row_parallel:
        if is_transpose:
            param.data[:, :loaded_weight_slice.shape[1]].copy_(loaded_weight_slice)
        else:
            param.data[:loaded_weight_slice.shape[0]].copy_(loaded_weight_slice)
    else:
        if is_transpose:
            param.data[:loaded_weight_slice.shape[0]].copy_(loaded_weight_slice)
        else:
            param.data[:, :loaded_weight_slice.shape[1]].copy_(loaded_weight_slice)
