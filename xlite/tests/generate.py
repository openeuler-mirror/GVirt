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
import json
from argparse import ArgumentParser
from typing import List

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
    model: nn.Module,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
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
    assert max(prompt_lens) <= model.max_seq_len, f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long, device="npu")
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="npu")
    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens), device="npu")
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    model_type: str,
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    if model_type == "deepseek":
        from tests.models.deepseek_v3 import ModelArgs
        from tests.models.deepseek_v3 import DeepSeek_V3 as Transformer
    elif model_type == "llama":
        from tests.models.llama import ModelArgs
        from tests.models.llama import Llama as Transformer
    elif model_type == "qwen":
        from tests.models.qwen2 import ModelArgs
        from tests.models.llama import Llama as Transformer
    else:
        return

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("hccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.npu.set_device(local_rank)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    torch.set_default_dtype(dtype)
    with torch.device("npu"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    tokenizer.decode(generate(model, [tokenizer.encode("  ")], 2, -1, 1.)[0])
    model.load_weights(ckpt_path)

    if interactive:
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
            if model_type == "deepseek":
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
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
            elif model_type == "qwen":
                formatted_prompt = ""
                for message in messages:
                    if message["role"] == "user":
                        formatted_prompt += f"<|im_start|>user {message['content']}<|im_end|>"
                    elif message["role"] == "assistant":
                        formatted_prompt += f"<|im_start|>assistant {message['content']}<|im_end|>"
                prompt_tokens = tokenizer.encode(formatted_prompt)
            else:
                prompt_tokens = tokenizer.encode(prompt)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            messages.append({"role": "assistant", "content": completion})
    else:
        print("start to run")
        with open(input_file, 'r', encoding='utf-8') as file:
            data = json.load(file)

        def process_batch(batch, tokenizer, model, max_new_tokens, eos_token_id, temperature):
            if model_type == "deepseek":
                prompts_tokens_batch = [
                    tokenizer.apply_chat_template([{"role": "user", "content": item['query']}], add_generation_prompt=True)
                    for item in batch
                ]
            elif model_type == "llama":
                prompts_tokens_batch = [
                    tokenizer.encode(f"<s>[INST] {item['query']} [/INST] ")
                    for item in batch
                ]
            elif model_type == "qwen":
                prompts_tokens_batch = [
                    tokenizer.encode(f"<|im_start|>user {item['query']}<|im_end|>")
                    for item in batch
                ]
            else:
                prompts_tokens_batch = [tokenizer.encode(item['query']) for item in batch]
            completion_tokens_batch = generate(model, prompts_tokens_batch, max_new_tokens, tokenizer.eos_token_id, temperature)
            completions_batch = tokenizer.batch_decode(completion_tokens_batch, skip_special_tokens=True)

            for idx, item in enumerate(batch):
                item['response'] = completions_batch[idx]
                print(f"Query: {item['query']}")
                print(f"Completion: {item['response']}")

        batch_size = args.max_batch_size

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            process_batch(batch, tokenizer, model, max_new_tokens, tokenizer.eos_token_id, temperature)
        
        with open(input_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print("The results have been written into the input_file.")

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
    parser.add_argument("--model", type=str, default="deepseek")
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    assert args.model in ["deepseek", "llama", "qwen"], f"{args.model} not supported!"
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(args.model, args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)
