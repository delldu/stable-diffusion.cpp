#!/usr/bin/env python
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Fri 19 Jan 2024 05:51:13 PM CST
# ***
# ************************************************************************************/
#
import os
import torch
import copy
import argparse
from safetensors.torch import load_file, save_file
import pdb

def load_weight(checkpoint, prefix=None):
    print(f"Loading weight from {checkpoint} ...")

    _, extension = os.path.splitext(checkpoint)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = load_file(checkpoint)
    else:
        state_dict = torch.load(checkpoint)

    if prefix is None:
        return state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k.replace(prefix, "")] = v
    return new_state_dict


def convert_ctrlnet(unet_model, ctrl_model, output_model):
    '''Only support sdxl 1.0'''

    sd1 = load_weight(unet_model, "model.diffusion_model.")
    sd2 = load_weight(ctrl_model)

    new_sd = {}
    sd2_keys = sd2.keys()

    for k in sd2_keys:
        if k.startswith("lora_controlnet"):
            continue
            
        if k.endswith('.down'):
            continue

        # weight = up * down
        if k.endswith(".up"):
            prefix_k = ".".join(k.split('.')[:-1])
            down_k = prefix_k + ".down"
            if not down_k in sd2:
                assert False, f"{down_k} not exists in {ctrl_model}"
            up = sd2[k]
            down = sd2[down_k]
            # new k, v

            k = prefix_k + ".weight"
            v = torch.mm(up.flatten(start_dim=1).float(), down.flatten(start_dim=1).float()).to(sd1[k].dtype).reshape(sd1[k].size())
        else:
            v = sd2[k]

        # Save
        new_sd[k] = copy.deepcopy(v) # we need deep copy for zero_convs_*, input_hint_block.* tensors are shared
    torch.save(new_sd, output_model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Stable Diffusion SDXL Control Net Models')
    parser.add_argument('--unet_input', type=str, required=True, help="unet model file name")
    parser.add_argument('--ctrl_input', type=str, required=True, help="control net model file name")
    parser.add_argument('--ctrl_output', type=str, required=True, help="control net output file name")
    args = parser.parse_args()

    convert_ctrlnet(args.unet_input, args.ctrl_input, args.ctrl_output)
