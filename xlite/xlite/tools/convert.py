#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def main(hf_ckpt_path, save_path, n_layers, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        
    Returns:
        None
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    for file_path in tqdm(glob(os.path.join(hf_ckpt_path, "*.safetensors"))):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if name.startswith("model.layers."):
                    layer_num = int(name.split(".")[2])
                    if layer_num >= n_layers:
                        continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                name = name.replace("qweight", "weight")
                name = name.replace("scales", "scale")
                key = name.split(".")[-2]
                assert key in mapping, f"Key {key} not found in mapping"
                new_key, dim = mapping[key]
                # ===== fix the tp for MOE quantization scales =====
                if name.endswith(".scale"):
                    if "gate_proj" in name or "up_proj" in name:
                        dim = 0
                    elif "down_proj" in name:
                        dim = None
                    else:
                        raise ValueError(f"Unknown name {name}")
                name = name.replace(key, new_key)
                for i in range(mp):
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

                    if "ffn" in name and "w1" in name:
                        name_w13 = name.replace("w1", "w13")
                        if name_w13 not in state_dicts[i].keys():
                            shape = list(new_param.size())
                            shape[dim] *= 2
                            state_dicts[i][name_w13] = torch.zeros(*shape, dtype=new_param.dtype, device=new_param.device)
                        state_dicts[i][name_w13].narrow(dim, 0, new_param.size(dim)).copy_(new_param).contiguous()
                        del state_dicts[i][name]
                    elif "ffn" in name and "w3" in name:
                        name_w13 = name.replace("w3", "w13")
                        if name_w13 not in state_dicts[i].keys():
                            shape = list(new_param.size())
                            shape[dim] *= 2
                            state_dicts[i][name_w13] = torch.zeros(*shape, dtype=new_param.dtype, device=new_param.device)
                        state_dicts[i][name_w13].narrow(dim, new_param.size(dim), new_param.size(dim)).copy_(new_param).contiguous()
                        del state_dicts[i][name]

    os.makedirs(save_path, exist_ok=True)

    for i in trange(mp):
        save_file(state_dicts[i], os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-layers", type=int, default=61)
    parser.add_argument("--n-experts", type=int, default=256)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0, "Number of experts must be divisible by model parallelism"
    main(args.hf_ckpt_path, args.save_path, args.n_layers, args.n_experts, args.model_parallel)
