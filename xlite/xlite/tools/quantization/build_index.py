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
from safetensors.torch import load_file

def main(model_path):
    safetensor_files = list(glob(os.path.join(model_path, "*.safetensors")))
    safetensor_files.sort()
    weight_map = {}
    for file in tqdm(safetensor_files):
        temp = load_file(file)
        for name in temp.keys():
            weight_map[name] = file
    new_model_index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    main(args.model_path)
