# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
"""Small random hybrid model: V2 metadata and Prefill/continuation/Decode.

Run from xlite with FORWARD_BACKEND=xlite and the newly built extension on
PYTHONPATH. No checkpoint, tokenizer or serving process is required.
"""
import os

os.environ["FORWARD_BACKEND"] = "xlite"

import torch
import torch_npu  # noqa: F401
from xlite import _C
from xlite._C import Runtime
from tests.models.qwen3_5 import ModelArgs, Qwen3_5


@torch.inference_mode()
def main():
    for name in ("linear_att_conv_and_silu", "linear_att_conv_and_silu_token",
                 "recurrent_gated_delta_rule"):
        assert not hasattr(_C, name), f"obsolete operator still exported: {name}"
    torch.npu.set_device(0)
    torch.manual_seed(565)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("npu")
    args = ModelArgs(
        dim=128, inter_dim=256, vocab_size=128, n_layers=2,
        n_heads=2, n_kv_heads=1, head_dim=256,
        # The existing RoPE kernel supports rotary widths 64/128.
        linear_num_key_heads=2, linear_num_value_heads=16,
        linear_key_head_dim=64, linear_value_head_dim=64,
        full_attention_interval=2, max_batch_size=2, max_seq_len=128,
    )
    model = Qwen3_5(args).eval()
    for name, value in model.named_parameters():
        if name.endswith("norm.weight") or "layernorm.weight" in name:
            value.zero_()
        elif name.endswith("A_log"):
            value.fill_(-1.0)
        else:
            value.normal_(0, 0.04)
    model.layers[0].linear_attn.norm.weight.fill_(1.0)
    model.xlite_weight_nz = False
    model.xlite_rt = Runtime(0, 0)
    model.init_xlite_model(args)
    model.init_xlite_kvcache(args)
    # Preserve the recurrent state across scheduler splits without BF16 cache rounding.
    model.xlite_kv_cache[0][1] = model.xlite_kv_cache[0][1].float()
    model.xlite_rt.init_tensor_pool(model.xlite_model.get_tensor_pool_size())

    # K and V may have independent physical page pitches and storage offsets.
    for index, padding in enumerate((64, 128)):
        old = model.xlite_kv_cache[1][index]
        pitch = old[0].numel() + padding
        raw = torch.full((32 + old.shape[0] * pitch + 32,), float("nan"))
        view = raw.as_strided(old.shape, (pitch, *old.stride()[1:]), 32)
        view.copy_(old)
        model.xlite_kv_cache[1][index] = view

    tokens = torch.randint(0, args.vocab_size, (2, 9), dtype=torch.int32)

    def run(part, start):
        part = part.contiguous()
        out = torch.empty(part.numel(), args.dim)
        meta = model.prepare_xlite_attnmeta_v2(part, start)
        # Fetch the raw stream after metadata work has entered TorchNPU's queue.
        stream = torch.npu.current_stream().npu_stream
        model.xlite_model.forward_v2(
            model.xlite_rt, part.flatten(), meta, model.xlite_kv_cache,
            [model.freqs_cis], out, stream,
        )
        torch.npu.synchronize()
        return out.view(2, part.shape[1], args.dim).cpu()

    full = run(tokens, 0)
    expected = [x.cpu().clone() for x in model.xlite_kv_cache[0]]
    for cache in model.xlite_kv_cache[0]:
        cache.zero_()
    first = run(tokens[:, :3], 0)
    continuation = run(tokens[:, 3:8], 3)
    decode = run(tokens[:, 8:], 8)
    actual = torch.cat((first, continuation, decode), 1)
    assert torch.isfinite(actual).all()
    # Prefill rounds Conv before SiLU; Decode keeps the fused rounding
    # contract. Compare the identical Prefill rows elementwise, and bound
    # the final Decode row's relative error across the complete hidden vector.
    torch.testing.assert_close(actual[:, :-1], full[:, :-1], rtol=0.03, atol=0.02)
    decode_error = (decode.float() - full[:, -1:].float()).norm()
    decode_error /= full[:, -1:].float().norm().clamp_min(1e-6)
    assert decode_error < 0.03, decode_error
    for cache, reference in zip(model.xlite_kv_cache[0], expected):
        torch.testing.assert_close(cache.cpu(), reference, rtol=0.03, atol=0.002)
        assert torch.count_nonzero(cache[0]).item() == 0, "null page changed"
    print(f"PASS hybrid V2: Prefill -> continuation -> Decode "
          f"relative_l2={decode_error.item():.6g}", flush=True)


if __name__ == "__main__":
    main()
