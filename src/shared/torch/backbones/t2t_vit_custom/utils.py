# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.init as init


def load_state_dict(checkpoint_path, use_ema=False, num_classes=1000, del_posemb=False):
    if not (checkpoint_path and os.path.isfile(checkpoint_path)):
        print("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()

    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict_key = "state_dict"
    if isinstance(checkpoint, dict):
        if use_ema and "state_dict_ema" in checkpoint:
            state_dict_key = "state_dict_ema"
    if state_dict_key and state_dict_key in checkpoint:
        new_state_dict = OrderedDict()
        for k, v in checkpoint[state_dict_key].items():
            # strip `module.` prefix
            name = k[7:] if k.startswith("module") else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    else:
        state_dict = checkpoint
    if num_classes != 1000:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict["head" + ".weight"]
        del state_dict["head" + ".bias"]

    if del_posemb:
        del state_dict["pos_embed"]

    return state_dict


def load_for_transfer_learning(
    model, checkpoint_path, use_ema=False, strict=True, num_classes=1000
):
    state_dict = load_state_dict(checkpoint_path, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print("==> Computing mean and std..")
    for inputs, _ in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    """Init layer parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode="fan_out")
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
