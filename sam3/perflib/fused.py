# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import torch

addmm_act_op = torch.ops.aten._addmm_activation


def addmm_act(activation, linear, mat1):
    if torch.is_grad_enabled():
        raise ValueError("Expected grad to be disabled.")

    # _addmm_activation is a CUDA-only fused kernel; fall back on CPU
    if mat1.device.type == "cpu":
        y = torch.nn.functional.linear(mat1, linear.weight, linear.bias)
        if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
            return torch.nn.functional.relu(y)
        if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
            return torch.nn.functional.gelu(y)
        raise ValueError(f"Unexpected activation {activation}")

    orig_dtype = mat1.dtype
    bias = linear.bias.detach().to(torch.bfloat16)
    mat2 = linear.weight.detach().to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(bias, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(orig_dtype)
    if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(bias, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],)).to(orig_dtype)
    raise ValueError(f"Unexpected activation {activation}")
