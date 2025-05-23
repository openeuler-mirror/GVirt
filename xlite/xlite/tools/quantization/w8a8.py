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
from glob import glob
from tqdm import tqdm
from safetensors.torch import load_file, save_file

import shutil

import torch
import transformers

import utils
from algorithms.rotation import (
    get_orthogonal_matrix,
)

from algorithms.quantizers import (
    WeightQuantizer, sym_quant, sym_dequant, asym_quant, asym_dequant
)


def post_opt_scale(w, w_quant, zero, H=None, b_bias=None):
    """
    Due to the implementation in the current quantization implementation
    the zero is not optimized as it is also quantized. Hence, there is
    no difference between sym and asym quant.
    We are solving a problem of x^T H x - 2(b + b_bias)x
    """
    if w_quant.dtype != w.dtype:
        w_quant = w_quant.to(w.dtype)
    if zero.dtype != w.dtype:
        zero = zero.to(w.dtype)
    if b_bias is not None and b_bias.dtype != w.dtype:
        b_bias = b_bias.to(w.dtype)
    if H is not None and H.dtype != w.dtype:
        H = H.to(w.dtype)
    
    # Scale to avoid the overflow when fp16 is used.
    # this can also iumport the numerical accuracy as well
    scale_temp = w_quant.max()
    # Formulated as standard qp
    temp = (w_quant - zero) / scale_temp
    if H is not None:
        H_qp = torch.matmul(temp, H)
        b_qp = torch.matmul(w / scale_temp, H)
    else:
        H_qp = temp
        b_qp = w / scale_temp
    b_qp = torch.sum(b_qp * temp, 1, keepdim=True)
    if b_bias is not None:
        b_qp += b_bias / scale_temp
    H_qp = torch.sum(H_qp * temp, 1, keepdim=True)

    scale_opt = b_qp / H_qp
    return scale_opt


def kron_matmul(x, hadL, hadR):
    init_shape = x.shape
    x = x.view(-1, hadL.shape[0], hadR.shape[0])
    x = torch.matmul(x, hadR)
    x = torch.matmul(hadL.T, x)
    return x.view(init_shape)


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor,
                   device, block_size: int = 128) -> torch.Tensor:
    # Get the original dimensions of weight
    M, N = weight.shape

    # Compute the effective block dimensions for scale
    scale_m, scale_n = scale.shape
    assert scale_m == (M + block_size - 1) // block_size, "Mismatch in scale rows and weight rows."
    assert scale_n == (N + block_size - 1) // block_size, "Mismatch in scale columns and weight columns."

    # Convert weight to float32 for calculations
    weight = weight.to(device=device, dtype=torch.float32)

    # Expand scale to match the weight tensor's shape
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:M, :N]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded.to(device)

    return dequantized_weight


def rtn_quant(W, w_bits=8, num_groups=1, sym=True, post_opt=False, **kwargs):
    '''Minimal implementation of the rtn quant'''
    quantizer = WeightQuantizer()
    quantizer.configure(w_bits, perchannel=True, sym=sym, mse=True)
    if num_groups > 1:
        assert W.shape[1] % num_groups == 0, \
            f"number of infeatures {W.shape[1]} should be multiple of the number of groups {num_groups}"
        stride = W.shape[1] // num_groups
        scale = []
        zero = []
        Q_all = []
        Q_dequant_all = []
        for idx_g in range(num_groups):
            idx_start = idx_g * stride
            idx_end = (idx_g + 1) * stride
            quantizer.find_params(W[:, idx_start:idx_end])
            Q = quantizer.quantize(W[:, idx_start:idx_end])
            if sym:
                W_quant, _ = sym_quant(Q, quantizer.scale, quantizer.maxq.to(W.device))
            else:
                W_quant, _, _ = asym_quant(Q, scale, quantizer.zero, quantizer.maxq.to(W.device))
            if post_opt:
                scale_temp = post_opt_scale(W[:, idx_start:idx_end], W_quant, quantizer.zero, H=None)
                quantizer.scale = scale_temp
            if sym:
                Q_dequant = sym_dequant(W_quant, scale_temp)
            else:
                Q_dequant = asym_dequant(W_quant, scale_temp, quantizer.zero)
            Q_all.append(W_quant)
            Q_dequant_all.append(Q_dequant)
            scale.append(quantizer.scale.cpu())
            zero.append(quantizer.zero.cpu())
        Q = torch.cat(Q_all, dim=1)
        Q_dequant = torch.cat(Q_dequant_all, dim=1)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
    else:
        quantizer.find_params(W)
        Q = quantizer.quantize(W)
        if sym:
            W_quant, _ = sym_quant(Q, quantizer.scale, quantizer.maxq.to(W.device))
        else:
            W_quant, _, _ = asym_quant(Q, scale, quantizer.zero, quantizer.maxq.to(W.device))
        if post_opt:
            scale = post_opt_scale(W, W_quant, quantizer.zero, H=None)
            quantizer.scale = scale
        Q = W_quant
        if sym:
            Q_dequant = sym_dequant(W_quant, scale)
        else:
            Q_dequant = asym_dequant(W_quant, scale, quantizer.zero)
        scale = quantizer.scale.cpu()
        zero = quantizer.zero.cpu()
    return Q_dequant, Q, scale, zero


def rot_linear(weight, name, state_dict, device, dtype, scale_inv=None, input_trans=None,
               output_trans=None, ln=None, verbose=False, save_dequant=False, **quantize_args):
    if scale_inv is not None:
        W_ = weight_dequant(weight, scale_inv, device=device).to(dtype)
    else:
        W_ = weight.to(device=device, dtype=dtype)
    W_shape = weight.shape
    if ln is not None:
        W_ = (W_ * ln.to(W_).unsqueeze(0))
    if input_trans is not None:
        if isinstance(input_trans, list):
            W_ = kron_matmul(W_, input_trans[0], input_trans[1])
        else:
            W_ = W_.reshape(-1, input_trans.shape[0])
            W_ = torch.matmul(W_, input_trans.to(W_)).reshape(W_shape)
    if output_trans is not None:
        W_ = W_.t().contiguous().reshape(-1, output_trans.shape[0])
        W_ = torch.matmul(W_, output_trans.to(W_))
        W_ = W_.reshape(W_shape[1], W_shape[0]).t().contiguous()
    utils.cleanup_memory(device)

    if quantize_args.get("quantized", False):
        if verbose:
            print(f"quantizing {name}: {quantize_args}")
        W_, qweight, scale, _ = rtn_quant(W_, **quantize_args)
        if save_dequant:
            state_dict[name] = W_.to(weight)
        else:
            assert name.endswith(".weight")
            basename = name.rsplit(".", 1)[0]
            assert qweight.min() >= -128 and qweight.max() <= 127, f"overflow in quant range: {name}"
            state_dict[basename + ".qweight"] = qweight.to(torch.int8).cpu()
            state_dict[basename + ".scales"] = scale.cpu()
    else:
        if verbose:
            print(f"skip quantization {name}")
        state_dict[name] = W_.cpu().to(torch.bfloat16)


def is_run_quant(name, run_quant, quant_shared, quant_mtp=False):
    layer_idx = int(name.split(".")[2])
    if not quant_mtp and layer_idx > 60:
        # MTP is not quantized
        return False
    if not run_quant:
        return False
    elif "down_proj" in name or "up_proj" in name\
        or "gate_proj" in name:
        if "experts" in name:
            if "shared_experts" in name:
                return quant_shared
            else:
                return True
    else:
        if "q_a_proj" in name or "q_b_proj" in name or "o_proj" in name\
            or "kv_a_proj_with_mqa" in name or "kv_b_proj" in name:
            return False


def main(model_path, rot_model_path,
         device, dtype, rotate_dict,
         down_kron=False,
         file_idx_start=None, file_idx_end=None,
         head_rotate=False, run_quant=False,
         num_tp=1, post_opt=False,
         quant_shared=False, quant_mtp=False,
         verbose=False, save_dequant=False):
    
    os.makedirs(rot_model_path, exist_ok=True)
    model_index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    # The list of name saved for the model
    ori_weight_names = []
    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(model_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    # Define all the transformation we need
    config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    Q = rotate_dict.get("Q", None)
    if Q is not None:
        Q = Q.to(device=device, dtype=dtype)
    if head_rotate:
        Q_head = get_orthogonal_matrix(config.v_head_dim,
                    mode="hadamard", device=device)
        kv_out_trans = torch.kron(torch.eye(config.num_attention_heads).to(Q_head),
                                  torch.block_diag(torch.eye(config.qk_nope_head_dim).to(Q_head),
                                                   Q_head))
    else:
        Q_head = None
        kv_out_trans = None
    
    if down_kron:
        Q_kron = rotate_dict["Q_kron"]
        Q_kron[0] = Q_kron[0].to(device=device, dtype=dtype)
        Q_kron[1] = Q_kron[1].to(device=device, dtype=dtype)
    else:
        Q_kron = None
    
    Q_q = None
    Q_kv = None
    Q_kv_with_mqa = None

    safetensor_files = list(glob(os.path.join(model_path, "*.safetensors")))
    safetensor_files.sort()
    if down_kron:
        SAVE_KRON = True
    with tqdm(safetensor_files, leave=True) as pbar:
        for safetensor_file in safetensor_files:
            pbar.set_description(f"{os.path.basename(safetensor_file)}")
            pbar.update(1)
            file_name = os.path.basename(safetensor_file)
            file_idx = int(file_name.split("-")[1])

            if file_idx_start is not None and file_idx_end is not None:
                if file_idx < file_idx_start or file_idx > file_idx_end:
                    continue
            current_state_dict = load_file(safetensor_file, device="cpu")
            loaded_files[file_name] = current_state_dict

            new_state_dict = {}

            for name, weight in current_state_dict.items():
                # reset params
                num_groups = 1

                if name.endswith("_scale_inv"):
                    continue
                elif weight.element_size() == 1: #FP8 weight
                    scale_inv_name = f"{name}_scale_inv"
                    try:
                        # Get scale_inv from the correct file
                        scale_inv = get_tensor(scale_inv_name)
                    except KeyError:
                        print(f"Warning: Missing scale_inv tensor for {name}, skip fp8 dequant")
                        scale_inv = None
                else:
                    scale_inv = None
                
                if "weight" not in name:
                    # REMARK: this part will be buggy for other models
                    # especially when there are bias, dsv3 only has bias in gate
                    if "e_score_correction_bias" not in name:
                        raise ValueError(f"Unknown element {name}")
                    new_state_dict[name] = weight.to(torch.bfloat16)
                    ori_weight_names.append(name)
                    continue
                else:
                    # ====== related to lm_heads ======
                    if "lm_head" in name:
                        assert scale_inv is None, "the weight is not dequanted"
                        input_trans = Q
                        output_trans = None
                        ln = get_tensor("model.norm.weight")
                        ln = ln.float() / torch.mean(ln.float().abs())
                        rot_linear(weight=weight, name=name,
                            state_dict=new_state_dict, device=device, dtype=dtype,
                            input_trans=input_trans, output_trans=output_trans, ln=ln,
                            verbose=verbose)
                        ori_weight_names.append(name)
                    elif "model.norm" in name or "layernorm" in name or \
                        "61.enorm" in name or "61.hnorm" in name or \
                        "shared_head.norm" in name:
                        assert scale_inv is None, "the weight is not dequanted"
                        # Reset prehead norm and layernorm
                        if "a_layernorm" in name:
                            new_state_dict[name] = weight.to(torch.bfloat16)
                        else:
                            new_state_dict[name] = (torch.ones_like(weight).to(weight) * torch.mean(weight.float().abs())).to(torch.bfloat16)
                        ori_weight_names.append(name)
                    # ===== related to embedding =====
                    elif "embed_tokens" in name:
                        assert scale_inv is None, "the weight is not dequanted"
                        if Q is not None:
                            W_ = weight.to(device=device, dtype=dtype)
                            new_state_dict[name] = torch.matmul(W_, Q).to(torch.bfloat16)
                        else:
                            new_state_dict[name] = weight.to(torch.bfloat16)
                        ori_weight_names.append(name)
                    # ===== related to router =====
                    elif "mlp.gate.weight" in name:
                        assert scale_inv is None, "the weight is not dequanted"
                        layer_idx = int(name.split(".")[2])
                        ln = get_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
                        # ===== remove mean =====
                        ln = ln.float() / torch.mean(ln.float().abs())
                        W_ = weight.to(device=device, dtype=dtype)
                        W_ = (W_ * ln.to(W_).unsqueeze(0))
                        W_ = torch.matmul(W_, Q.to(W_))
                        new_state_dict[name] = W_  # keep the fp32 to maximize the accuracy
                        ori_weight_names.append(name)
                    # ===== related transformer layers =====
                    elif "layers" in name:
                        layer_idx = int(name.split(".")[2])
                        if "shared_head.head" in name:
                            # skip the shared head as it's redundant
                            continue
                        else:
                            if not save_dequant and is_run_quant(name, run_quant, quant_shared, quant_mtp):
                                basename = name.rsplit(".", 1)[0]
                                ori_weight_names.append(basename + ".qweight")
                                ori_weight_names.append(basename + ".scales")
                                weight_map[basename + ".qweight"] = file_name
                                weight_map[basename + ".scales"] = file_name
                            else:
                                ori_weight_names.append(name)
                        # rotate q
                        if "q_a_proj" in name:
                            input_trans = Q
                            output_trans = Q_q
                            ln = get_tensor(f"model.layers.{layer_idx}.input_layernorm.weight")
                            # ===== remove mean =====
                            ln = ln.float() / torch.mean(ln.float().abs())
                        elif "q_b_proj" in name:
                            input_trans = Q_q
                            output_trans = None
                            ln = None
                        # rotate kv
                        elif "kv_a_proj_with_mqa" in name:
                            input_trans = Q
                            output_trans = Q_kv_with_mqa
                            ln = get_tensor(f"model.layers.{layer_idx}.input_layernorm.weight")
                            # ===== remove mean =====
                            ln = ln.float() / torch.mean(ln.float().abs())
                        elif "kv_b_proj" in name:
                            input_trans = Q_kv
                            output_trans = kv_out_trans
                            ln = None
                        # rotate o
                        elif "o_proj" in name:
                            input_trans = Q_head
                            output_trans = Q
                            ln = None
                            num_groups = num_tp
                        # rotate mlp gate up
                        elif "gate_proj" in name or "up_proj" in name:
                            input_trans = Q
                            output_trans = None
                            ln = get_tensor(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
                            # ===== remove mean =====
                            ln = ln.float() / torch.mean(ln.float().abs())
                        # rotate mlp down
                        elif "down_proj" in name:
                            if "experts" in name and "shared_experts" not in name:
                                input_trans = Q_kron
                            else:
                                input_trans = None
                            output_trans = Q
                            ln = None
                            num_groups = num_tp
                        # ========= special complenents for MTP ========
                        elif "eh_proj" in name:
                            # cannot ignore the output_trans here as we need to
                            # use the lm_head in the original model which is rotated
                            input_trans = torch.block_diag(Q, Q)
                            output_trans = Q
                            ln = torch.cat([
                                    get_tensor(f"model.layers.{layer_idx}.enorm.weight"),
                                    get_tensor(f"model.layers.{layer_idx}.hnorm.weight")
                                ]).flatten()
                            # ===== remove mean =====
                            ln = ln.float() / torch.mean(ln.float().abs())
                        else:
                            raise ValueError(f"Unknown object{name}")
                        rot_linear(weight=weight, name=name, state_dict=new_state_dict,
                            device=device, dtype=dtype, scale_inv=scale_inv,
                            input_trans=input_trans,
                            output_trans=output_trans, ln=ln, verbose=verbose,
                            **{"quantized": is_run_quant(name, run_quant, quant_shared, quant_mtp),
                            "w_bits": 8, "sym": True, "num_groups": num_groups,
                            "post_opt": post_opt, "save_dequant": save_dequant})

            if down_kron and SAVE_KRON:
                # save the kron down matrices in the first file
                new_state_dict["model.down_kron.kron_left"] = rotate_dict["Q_kron"][0].to(torch.bfloat16)
                new_state_dict["model.down_kron.kron_right"] = rotate_dict["Q_kron"][1].to(torch.bfloat16)
                # modification to update index.json
                weight_map["model.down_kron.kron_right"] = file_name
                weight_map["model.down_kron.kron_left"] = file_name
                ori_weight_names.append("model.down_kron.kron_right")
                ori_weight_names.append("model.down_kron.kron_left")
                SAVE_KRON = False
            
            new_safetensor_file = os.path.join(rot_model_path, file_name)
            save_file(new_state_dict, new_safetensor_file)

            # Memory management: keep only the 2 most recently used files
            if len(loaded_files) > 2:
                oldest_file = next(iter(loaded_files))
                del loaded_files[oldest_file]
                utils.cleanup_memory(device)
    
    # Update model index
    print("saving the new index file")
    new_model_index_file = os.path.join(rot_model_path, "model.safetensors.index.json")
    for weight_name in ori_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    temp = list(weight_map.keys())
    for weight_name in temp:
        if weight_name not in ori_weight_names:
            weight_map.pop(weight_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)
    
    # copy the necessary tokenizer
    for file_path in glob(os.path.join(model_path, "*token*")):
        new_file_path = os.path.join(rot_model_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)
    # copy config file
    file_path = os.path.join(model_path, "config.json")
    new_file_path = os.path.join(rot_model_path, "config.json")
    shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--rot_model_path", type=str, required=True)
    parser.add_argument("--num_tp", type=int, choices=[1,2,4,8], default=1,
                        help="number of cars to deploy tensor parallelalization, mainly related o_proj and down_proj.")
    parser.add_argument("--post_opt", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether run post_opt in RTN algorithm (default: False)")
    parser.add_argument("--run_quant", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether run quantization (default: False)")
    parser.add_argument("--save_dequant", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether save the fake bf16 weight")
    parser.add_argument("--quant_shared", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether quantize the shared expert")
    parser.add_argument("--quant_mtp", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether quantize the MTP part")
    parser.add_argument("--deactivate_head_rotate", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether do head rotation")
    parser.add_argument("--down_kron", action=argparse.BooleanOptionalAction, default=False,
                        help="Whether apply the kronecker rotation to the down projection layer activation")
    parser.add_argument("--id_npu", type=int, default=0)
    parser.add_argument("--file_idx_start", type=int)
    parser.add_argument("--file_idx_end", type=int)

    assert os.path.exists("./rotate_dict.bin"), \
        "You have to prepare the rotation matrices on the backbone, otherwise the parallel quantization will be buggy!"
    rotate_dict = torch.load("./rotate_dict.bin")
    args = parser.parse_args()
    DEV = torch.device(f"npu:{args.id_npu}")
    dtype = torch.float32
    print(f"running code on {DEV}")

    main(model_path=args.model_path, rot_model_path=args.rot_model_path,
         file_idx_start=args.file_idx_start, file_idx_end=args.file_idx_end,
         rotate_dict=rotate_dict, down_kron=args.down_kron,
         device=DEV, dtype=dtype, head_rotate=not args.deactivate_head_rotate,
         quant_shared=args.quant_shared, quant_mtp=args.quant_mtp,
         run_quant=args.run_quant, num_tp=args.num_tp, post_opt=args.post_opt,
         save_dequant=args.save_dequant, verbose=False)
