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
import time
import json
import random
from argparse import ArgumentParser
from typing import List, Literal, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import AutoTokenizer


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: nn.Module, prompt_tokens: List[List[int]], max_new_tokens: int, eos_id: int, temperature: float = 1.0
) -> Tuple[List[List[int]], int]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (nn.Module): The model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, (
        f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    )
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens_cpu = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.int32, device="cpu")
    for i, t in enumerate(prompt_tokens):
        tokens_cpu[i, : len(t)] = torch.tensor(t, dtype=torch.int32, device="cpu")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="npu")
    prompt_mask = tokens_cpu != -1
    prompt_mask = prompt_mask.to("npu")
    tokens = tokens_cpu.to("npu")

    step = 0
    start = min(prompt_lens)
    for cur_pos in range(start, total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        step = step + 1
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)
    return (completion_tokens, step)


def main(
    model_type: str,
    ckpt_path: str,
    config: str,
    input_file: str = "",
    mode: Literal["single", "interactive", "bench"] = "single",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    no_prefix: bool = False,
    bench_batch_size: int = 16,
    bench_iters: int = 4,
    bench_prompt_len: int = 2048,
    bench_new_tokens: int = 1024,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        mode (Literal["single", "interactive", "bench"], optional): Generation mode. Defaults to "single".
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
        no_prefix (bool, optional): Whether to skip adding prefix/suffix to prompts. Defaults to False.
        bench_batch_size (int, optional): Batch size for bench serve. Defaults to 16.
        bench_iters (int, optional): Number of iterations for bench serve. Defaults to 4.
        bench_prompt_len (int, optional): Length of prompts for bench serve. Defaults to 2048.
        bench_new_tokens (int, optional): Number of new tokens to generate for bench serve. Defaults to 1024.
    """
    if model_type == "deepseek_v3" or model_type == "deepseek_v32" or model_type == "glm5":
        from tests.models.deepseek_v3 import ModelArgs
        from tests.models.deepseek_v3 import DeepSeek_V3 as Transformer
    elif model_type == "llama":
        from tests.models.llama import ModelArgs
        from tests.models.llama import Llama as Transformer
    elif model_type == "qwen2":
        from tests.models.qwen2 import Qwen2ModelArgs as ModelArgs
        from tests.models.llama import Llama as Transformer
    elif model_type == "qwen3":
        from tests.models.qwen3 import Qwen3ModelArgs as ModelArgs
        from tests.models.llama import Llama as Transformer
    elif model_type == "qwen3_moe":
        from tests.models.qwen3_moe import ModelArgs
        from tests.models.qwen3_moe import Qwen3MoE as Transformer
    elif model_type == "qwen3_5":
        from tests.models.qwen3_5 import ModelArgs
        from tests.models.qwen3_5 import Qwen3_5 as Transformer
    elif model_type == "qwen3_5_moe":
        from tests.models.qwen3_5_moe import ModelArgs
        from tests.models.qwen3_5_moe import Qwen3_5MoE as Transformer
    elif model_type == "glm4_moe":
        from tests.models.glm4_moe import ModelArgs
        from tests.models.glm4_moe import GLM4MoE as Transformer
    elif model_type == "minimax_m2":
        from tests.models.minimax_m2 import ModelArgs
        from tests.models.glm4_moe import GLM4MoE as Transformer
    else:
        return

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("hccl")

    # DP deployment: world_size = tp_size * dp_size. tp_rank = rank % tp_size,
    # dp_rank = rank // tp_size (matches C++ XRuntime). Weights replicated across
    # DP groups, sharded within TP; each DP rank serves a disjoint prompt subset.
    # The C++ engine handles MoE DP AllGather/ReduceScatter; dense layers per-DP.
    dp_size = int(os.getenv("XLITE_DP_SIZE", "1"))
    assert world_size % dp_size == 0, (
        f"WORLD_SIZE ({world_size}) must be divisible by XLITE_DP_SIZE ({dp_size})"
    )
    # DP is supported only on the xlite backend (its C++ engine handles the MoE
    # DP AllGather/ReduceScatter natively); the torch_npu reference path does not.
    _fb = os.getenv("FORWARD_BACKEND", "torch_npu")
    if dp_size > 1 and _fb != "xlite":
        raise NotImplementedError(
            f"XLITE_DP_SIZE>1 requires FORWARD_BACKEND=xlite (got {_fb!r}); "
            "torch_npu does not support DP."
        )
    tp_size = world_size // dp_size
    os.environ["XLITE_TP_SIZE"] = str(tp_size)
    dp_rank = rank // tp_size
    explicit_tp = os.getenv("XLITE_TP_SIZE")  # sanity-check if explicitly overridden
    if explicit_tp is not None and int(explicit_tp) != tp_size:
        raise ValueError(f"XLITE_TP_SIZE ({explicit_tp}) != world_size/dp_size ({tp_size})")
    if dp_size > 1 and mode == "interactive":
        raise NotImplementedError(
            "DP (XLITE_DP_SIZE>1) is not supported in interactive mode; "
            "use single/bench mode, or run DP groups as separate processes."
        )

    # DP subgroup: ranks sharing tp_rank across DP groups (dst=0 = dp_rank=0).
    # Kept narrow so gather_object runs only over the dp_size real DP ranks,
    # not the full world_size default group.
    dp_group = None
    if dp_size > 1 and world_size > 1:
        tp_rank = rank % tp_size
        dp_group_ranks = [g * tp_size + tp_rank for g in range(dp_size)]
        dp_group = dist.new_group(ranks=dp_group_ranks)

    # DP: each DP group's root (tp_rank==0) prints its shard, prefixed [rankN]
    # so concurrent output stays readable; global summaries use _raw_print.
    # Non-DP: only rank 0 prints, as before.
    global print
    _raw_print = print  # unprefixed, for global summaries
    _is_dp_printer = (dp_size == 1 and rank == 0) or (dp_size > 1 and rank % tp_size == 0)
    if not _is_dp_printer:
        print = lambda *_, **__: None
    elif dp_size > 1:
        _rk = rank
        def _print(*a, **kw):
            _raw_print(f"[rank{_rk}]", *a, **kw)
        print = _print
    torch.npu.set_device(local_rank)
    torch.npu.config.allow_internal_format = True
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        config = json.load(f)
        if mode == "bench":
            config["max_batch_size"] = max(config.get("max_batch_size", 1), bench_batch_size)
            config["max_seq_len"] = max(config.get("max_seq_len", 1024), bench_prompt_len + bench_new_tokens)
        args = ModelArgs(**config)
    print(args)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.set_default_dtype(dtype)
    with torch.device("npu"):
        model = Transformer(args)
    model.load_weights(ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    completion_tokens, _ = generate(model, [tokenizer.encode("Warn up")], 2, -1, 1.0)
    tokenizer.decode(completion_tokens[0])

    if mode == "bench":
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            try:
                vocab_size = len(tokenizer.get_vocab())
            except Exception:
                vocab_size = 200000

        eos_id = -1  # no eos token for bench, generate until bench_new_tokens

        if bench_prompt_len > model.max_seq_len:
            raise ValueError(f"bench_prompt_len ({bench_prompt_len}) > model.max_seq_len ({model.max_seq_len})")

        if bench_prompt_len + bench_new_tokens > model.max_seq_len:
            print(
                f"Warning: model.max_seq_len ({model.max_seq_len}) < prompt_len+new_tokens ("
                f"{bench_prompt_len + bench_new_tokens}), truncating max_new_tokens to fit"
            )
            effective_max_new_tokens = max(0, model.max_seq_len - bench_prompt_len)
        else:
            effective_max_new_tokens = bench_new_tokens

        def make_prompt():
            return [random.randrange(vocab_size) for _ in range(bench_prompt_len)]

        # DP: split bench_batch_size evenly so every DP rank gets the same local_bs
        # (uniform m for MoE DP AllGather); rank 0 reports aggregate (× dp_size).
        if dp_size > 1 and bench_batch_size % dp_size != 0:
            print(
                f"Warning: bench_batch_size ({bench_batch_size}) not divisible by dp_size "
                f"({dp_size}); rounding local batch down to {bench_batch_size // dp_size}"
            )
        local_bs = max(1, bench_batch_size // dp_size)

        print(
            f"Running bench: batch_size={bench_batch_size}"
            + (f" (={local_bs}/rank × {dp_size} DP)" if dp_size > 1 else "")
            + f", iters={bench_iters}, prompt_len={bench_prompt_len}, "
            f"decode={effective_max_new_tokens}"
        )
        for it in range(bench_iters):
            prompts = [make_prompt() for _ in range(local_bs)]
            s = time.monotonic_ns()
            completion_tokens_batch, step = generate(model, prompts, effective_max_new_tokens, eos_id, temperature)
            c = time.monotonic_ns()
            d = (c - s) / 1e6
            num_prefill = sum(len(p) for p in prompts)
            num_completion = sum(len(completion_tokens_batch[i]) for i in range(len(prompts)))
            agg_prefill = num_prefill * dp_size
            agg_completion = num_completion * dp_size
            print(
                f"Iter {it + 1}/{bench_iters}: prefilled {num_prefill}"
                + (f" (≈{agg_prefill} agg)" if dp_size > 1 else "")
                + f" | generated {num_completion}"
                + (f" (≈{agg_completion} agg)" if dp_size > 1 else "")
                + f" tokens in {d:.2f} ms. avg: {num_prefill / d * 1e3:.2f} | "
                f"{num_completion / d * 1e3:.2f} tokens/s"
                + (f" (≈{agg_completion / d * 1e3:.2f} agg tokens/s)" if dp_size > 1 else "")
                + f" @ {d / step:.2f} ms step bs: {len(prompts)}"
            )
    elif mode == "interactive":
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            if model_type in {"deepseek_v3", "glm4_moe", "glm5", "minimax_m2"}:
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
            elif model_type == "llama":
                formatted_prompt = ""
                for message in messages:
                    if message["role"] == "user":
                        formatted_prompt += f"<s>[INST] {message['content']} [/INST]"
                    elif message["role"] == "assistant":
                        formatted_prompt += f" {message['content']} </s><s>[INST] "
                if formatted_prompt.endswith("<s>[INST] "):
                    formatted_prompt = formatted_prompt[:-8]
                formatted_prompt += " "
                prompt_tokens = tokenizer.encode(formatted_prompt)
            elif model_type in {"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}:
                # Use apply_chat_template for correct format
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
            else:
                prompt_tokens = tokenizer.encode(prompt)
            completion_tokens, _ = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            completion = completion.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        if rank == 0:
            _raw_print("start to run")
        with open(input_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        def process_batch(batch, tokenizer, model, max_new_tokens, eos_token_id, temperature, no_prefix):
            if no_prefix:
                prompts_tokens_batch = [tokenizer.encode(item["query"]) for item in batch]
            elif model_type in {"deepseek_v3", "glm4_moe", "glm5", "minimax_m2"}:
                prompts_tokens_batch = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["query"]}], add_generation_prompt=True, return_dict=False
                    )
                    for item in batch
                ]
            elif model_type == "llama":
                prompts_tokens_batch = [tokenizer.encode(f"<s>[INST] {item['query']} [/INST] ") for item in batch]
            elif model_type in {"qwen2", "qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe"}:
                # Use apply_chat_template for correct format with proper newlines
                prompts_tokens_batch = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": item["query"]}], add_generation_prompt=True, return_dict=False
                    )
                    for item in batch
                ]
            else:
                prompts_tokens_batch = [tokenizer.encode(item["query"]) for item in batch]

            s = time.monotonic_ns()
            completion_tokens_batch, step = generate(
                model, prompts_tokens_batch, max_new_tokens, tokenizer.eos_token_id, temperature
            )
            c = time.monotonic_ns()
            d = (c - s) / 1e6

            completions_batch = tokenizer.batch_decode(completion_tokens_batch, skip_special_tokens=True)
            num = 0
            for idx, item in enumerate(batch):
                item["response"] = completions_batch[idx].replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ")
                num += len(completion_tokens_batch[idx]) + len(prompts_tokens_batch[idx]) - 1
            if dp_size > 1:
                stats = {"num": num, "d": d, "step": step, "bs": len(batch)}
                for item in batch:
                    item["_stats"] = stats
            else:
                for item in batch:
                    print(f"Query: {item['query']}")
                    print(f"Completion: {item['response']}")
                print(
                    f"generate {num} tokens take {d:.2f} ms. avg: {num / d * 1e3:.2f} tokens/s @ {d / step:.2f} ms bs: {len(batch)}"
                )
            return num, d, step

        # DP shard: round-robin striping so every DP rank serves a disjoint,
        # equal-sized slice (uniform m for MoE DP AllGather); the last rank is
        # padded with sentinels whose responses are dropped on gather.
        local_batch_size = args.max_batch_size
        if dp_size > 1:
            n = len(data)
            per_rank = (n + dp_size - 1) // dp_size
            local_data = []
            for i in range(per_rank):
                idx = i * dp_size + dp_rank
                if idx < n:
                    local_data.append(data[idx])
                else:
                    local_data.append({"query": "", "_pad": True})  # sentinel, dropped on gather
            dist.barrier()  # keep DP ranks lock-stepped per batch (identical m)
        else:
            local_data = data

        local_stats = []
        for i in range(0, len(local_data), local_batch_size):
            batch = local_data[i : i + local_batch_size]
            local_stats.append(process_batch(batch, tokenizer, model, max_new_tokens, tokenizer.eos_token_id, temperature, no_prefix))
            if dp_size > 1:
                dist.barrier()

        if dp_size > 1:
            # Gather local_data back to rank 0 on the default group (torch_npu's
            # gather_object needs dst as a global rank, no group= on non-dst).
            # rank 0 picks the dp_rank=0 member of each DP group (global ranks
            # 0, tp_size, 2*tp_size, ...) — the real shards; TP ranks are replicas.
            gathered = [None] * world_size if rank == 0 else None
            dist.gather_object(local_data,
                               object_gather_list=gathered if rank == 0 else None,
                               dst=0)
            if rank == 0:
                merged = [None] * n
                for src_dp in range(dp_size):
                    src_global_rank = src_dp * tp_size  # dp_rank=0 of DP group src_dp
                    shard = gathered[src_global_rank]
                    for i, item in enumerate(shard):
                        idx = i * dp_size + src_dp
                        if idx < n:
                            merged[idx] = item
                data = merged
                for src_dp in range(dp_size):
                    shard = [it for it in merged[src_dp::dp_size] if it and not it.get("_pad")]
                    if not shard:
                        continue
                    _raw_print(f"----- [DP{src_dp}] (global rank {src_dp * tp_size}) -----")
                    for item in shard:
                        _raw_print(f"[DP{src_dp}] Query: {item['query']}")
                        _raw_print(f"[DP{src_dp}] Completion: {item['response']}")
                    seen = []
                    for item in shard:
                        s = item.get("_stats")
                        if s is not None and s not in seen:
                            seen.append(s)
                    for s in seen:
                        _raw_print(
                            f"[DP{src_dp}] generate {s['num']} tokens take {s['d']:.2f} ms. "
                            f"avg: {s['num'] / s['d'] * 1e3:.2f} tokens/s @ {s['d'] / s['step']:.2f} ms bs: {s['bs']}"
                        )
                    tot = sum(s["num"] for s in seen)
                    agg_d = max((s["d"] for s in seen), default=0.0)
                    agg_step = max((s["step"] for s in seen), default=1)
                    _raw_print(
                        f"[DP{src_dp}] total {tot} tokens, {len(shard)} queries "
                        f"(parallel {agg_d:.2f} ms, {tot / agg_d * 1e3:.2f} tokens/s @ {agg_d / agg_step:.2f} ms)"
                    )
            else:
                data = local_data
            if rank == 0:
                total_tokens = sum(s[0] for s in local_stats) * dp_size
                _raw_print(f"[DP] aggregate {total_tokens} tokens across {dp_size} DP ranks")
        else:
            data = local_data

        if rank == 0:
            for item in data:
                if isinstance(item, dict):
                    item.pop("_stats", None)
            with open(input_file, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            _raw_print("The results have been written into the input_file.")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek_v3")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--mode", type=str, default="single", choices=["single", "interactive", "bench"])
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-prefix", action="store_true")
    parser.add_argument("--bench-batch-size", type=int, default=16, help="Batch size/concurrency for bench serve")
    parser.add_argument("--bench-iters", type=int, default=4, help="Number of iterations for bench serve")
    parser.add_argument("--bench-prompt-len", type=int, default=2048, help="Prompt length (tokens) for bench serve")
    parser.add_argument("--bench-new-tokens", type=int, default=1024, help="New tokens to generate for bench serve")

    args = parser.parse_args()
    assert args.model in [
        "deepseek_v3",
        "deepseek_v32",
        "glm5",
        "minimax_m2",
        "llama",
        "qwen2",
        "qwen3",
        "qwen3_moe",
        "qwen3_5",
        "qwen3_5_moe",
        "glm4_moe",
    ], f"{args.model} not supported!"
    assert args.input_file or args.mode != "single", "Either input-file or interactive mode must be specified"
    main(
        model_type=args.model,
        ckpt_path=args.ckpt_path,
        config=args.config,
        input_file=args.input_file,
        mode=args.mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        no_prefix=args.no_prefix,
        bench_batch_size=args.bench_batch_size,
        bench_iters=args.bench_iters,
        bench_prompt_len=args.bench_prompt_len,
        bench_new_tokens=args.bench_new_tokens,
    )
