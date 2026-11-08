#!/usr/bin/python3
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
"""Offline per-output-channel INT8 quantization (W8A8_DYNAMIC) for MoE routed experts.

Produces a *mixed quantization* checkpoint from a bf16 safetensors checkpoint:

  * routed experts (`model.layers.<i>.mlp.experts.<j>.{gate,up,down}_proj.weight`)
    are converted to int8 weights plus a per-output-channel fp32 scale stored as
    ``<name>.weight_scale`` (shape ``[out, 1]``) — the same tensor convention as
    ModelSlim W8A8_DYNAMIC checkpoints;
  * everything else (attention, dense MLP, shared experts, norms, router gate)
    is copied verbatim in bf16.

Quantization is symmetric per-row absmax: ``scale = absmax / 127``,
``q = clamp(round(w / scale), -127, 127)`` (rows with absmax == 0 get scale 1.0).

All layers matching the routed-expert name pattern are quantized, including the
trailing MTP layer some checkpoints ship (e.g. GLM-4.7-Flash layer 47): the repo
loader skips layers >= n_layers, so those tensors are inert — quantizing them
keeps every shard's dtype layout self-consistent.

The output is consumed by the repo test models with
``"quantization": "w8a8_moe"`` in the test config (see
``tests/models/glm4_moe_lite.py`` and ``run_glm4_moe_flash_w8a8moe`` in
``tests/run.sh``). Serving the checkpoint with vllm-ascend additionally needs a
``quant_model_description.json`` marker in the model dir, which this tool
intentionally does NOT write (the mixed bf16-attention layout is not covered by
vllm-ascend's "ascend" quantization path).

Example:
    python xlite/tools/quantize_moe_w8a8.py \
        --input  /path/to/GLM-4.7-Flash \
        --output /path/to/GLM-4.7-Flash-w8a8-moe
"""

import argparse
import json
import os
import re
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

# Routed-expert weight names as laid out by HF GLM/DeepSeek MoE checkpoints.
EXPERT_WEIGHT_RE = re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")

INT8_MAX = 127


def quantize_rowwise(weight: torch.Tensor):
    """Symmetric per-output-channel (row) int8 quantization.

    Args:
        weight: [out, in] bf16/fp16/fp32 tensor.

    Returns:
        (q, scale, rel_err): q is int8 [out, in], scale is fp32 [out, 1]
        (dequant scale, i.e. antiquant scale), rel_err is the mean relative
        reconstruction error of this tensor.
    """
    w = weight.float()
    absmax = w.abs().amax(dim=1)  # [out]
    scale = (absmax / INT8_MAX).clamp(min=0.0)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(w / scale.unsqueeze(1)).clamp_(-INT8_MAX, INT8_MAX).to(torch.int8)
    # Reconstruction error relative to the per-row dynamic range.
    err = (w - q.float() * scale.unsqueeze(1)).abs().mean(dim=1)
    rel_err = (err / (absmax + 1e-8)).mean().item()
    return q, scale.reshape(-1, 1).contiguous(), rel_err


def iter_shards(input_dir: str):
    """Yield the path of every safetensors shard, in index order."""
    index_path = os.path.join(input_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(f for f in os.listdir(input_dir) if f.endswith(".safetensors"))
    if not shards:
        sys.exit(f"error: no .safetensors shards found in {input_dir}")
    for shard in shards:
        yield os.path.join(input_dir, shard)


def scan_tensor_names(input_dir: str):
    """Return {shard_path: [tensor names]} reading only safetensors headers."""
    from safetensors import safe_open

    names = {}
    for shard_path in iter_shards(input_dir):
        with safe_open(shard_path, framework="pt") as f:
            names[shard_path] = list(f.keys())
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Quantize MoE routed experts of a bf16 checkpoint to W8A8_DYNAMIC int8."
    )
    parser.add_argument("--input", required=True, help="bf16 checkpoint dir (safetensors)")
    parser.add_argument("--output", required=True, help="output checkpoint dir (created)")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="only print the quantization error report, do not write the output",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        sys.exit(f"error: --input {args.input} is not a directory")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        sys.exit("error: --output must differ from --input")

    # Pre-scan (headers only): fail fast before writing anything.
    shard_names = scan_tensor_names(args.input)
    matched = [n for names in shard_names.values() for n in names if EXPERT_WEIGHT_RE.match(n)]
    if not matched:
        sys.exit("error: no routed-expert weights matched; is this a MoE checkpoint?")
    # A bf16 expert weight must not carry quantization siblings (weight_scale,
    # weight_scale_inv, deq_scale, scale_bias, ...). Any extra tensor sharing an
    # expert weight's prefix means the input is already (partly) quantized.
    weight_prefixes = {n[: -len(".weight")] + "." for n in matched}
    siblings = [
        n
        for names in shard_names.values()
        for n in names
        if any(n.startswith(p) and not n.endswith(".weight") for p in weight_prefixes)
        and re.match(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.", n)
    ]
    if siblings:
        sys.exit(
            "error: input already carries quantized expert tensors (e.g. "
            f"{siblings[0]}); --input must be a bf16 checkpoint"
        )

    if not args.report_only:
        os.makedirs(args.output, exist_ok=True)

    index_path = os.path.join(args.input, "model.safetensors.index.json")
    raw_index = None
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            raw_index = json.load(f)

    stats = []  # (name, numel, rel_err)
    new_keys = {}  # scale key -> shard file
    for shard_path in shard_names:
        shard_name = os.path.basename(shard_path)
        tensors = load_file(shard_path)
        out_tensors = {}
        for name, tensor in tensors.items():
            if EXPERT_WEIGHT_RE.match(name):
                q, scale, rel_err = quantize_rowwise(tensor)
                stats.append((name, tensor.numel(), rel_err))
                out_tensors[name] = q
                scale_key = name + "_scale"
                out_tensors[scale_key] = scale
                new_keys[scale_key] = shard_name
            else:
                out_tensors[name] = tensor
        if not args.report_only:
            out_path = os.path.join(args.output, shard_name)
            save_file(out_tensors, out_path, metadata={"format": "pt"})
            print(f"[quantized] {shard_name} ({len(stats)} expert tensors so far)")
        del tensors, out_tensors

    # ---- report ----
    total = sum(n for _, n, _ in stats)
    mean_rel = sum(e * n for _, n, e in stats) / total
    worst = sorted(stats, key=lambda s: -s[2])[:5]
    print(f"\nquantized tensors : {len(stats)}")
    print(f"quantized elements: {total / 1e6:.1f} M")
    print(f"mean rel error    : {mean_rel:.4%}")
    print("worst tensors:")
    for name, _, rel_err in worst:
        print(f"  {rel_err:.4%}  {name}")

    if args.report_only:
        return

    # Copy non-weight files verbatim (config, tokenizer, generation config, ...).
    for fname in os.listdir(args.input):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(args.input, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.output, fname))

    # Keep model.safetensors.index.json consistent: add the new scale keys
    # (shard filenames are unchanged, so existing entries stay valid).
    if raw_index is not None:
        raw_index.setdefault("weight_map", {}).update(new_keys)
        raw_index["weight_map"] = dict(sorted(raw_index["weight_map"].items()))
        with open(os.path.join(args.output, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
            json.dump(raw_index, f, indent=2, sort_keys=True)

    print(f"\ndone: {args.output} (routed experts int8, rest bf16)")


if __name__ == "__main__":
    main()
