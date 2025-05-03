import copy
from collections import OrderedDict
from typing import Any

import torch.nn.modules.conv
from torch import nn

from .layers import *
from .layers import _LoRAConvNd

__all__ = [
    "STD_TO_LORA",
    "LORA_TO_STD",
    "create_lora",
    "remove_lora",
    "merge_lora",
]


def embedding_std_to_lora(embedding: nn.Embedding, rank: int, alpha: float) -> LoRAEmbedding:
    """
    Convert a standard embedding layer to a LoRA embedding layer.
    Args:
        embedding (nn.Embedding): The standard embedding layer.
        rank (int): The rank for the LoRA layer.
        alpha (float): The scaling factor for the LoRA layer.
    Returns:
        LoRAEmbedding: The converted LoRA embedding layer.
    """
    lora_embedding = LoRAEmbedding(
        num_embeddings=embedding.num_embeddings,
        embedding_dim=embedding.embedding_dim,
        padding_idx=embedding.padding_idx,
        max_norm=embedding.max_norm,
        norm_type=embedding.norm_type,
        scale_grad_by_freq=embedding.scale_grad_by_freq,
        sparse=embedding.sparse,
        r=rank,
        lora_alpha=alpha,
        merge_weights=False
    ).to(embedding.weight.device, dtype=embedding.weight.dtype)
    lora_embedding.weight.data.copy_(embedding.weight.data)
    return lora_embedding


def embedding_lora_to_std(lora_embedding: LoRAEmbedding) -> nn.Embedding:
    """
    Convert a LoRA embedding layer back to a standard embedding layer.
    Args:
        lora_embedding (LoRAEmbedding): The LoRA embedding layer.
    Returns:
        nn.Embedding: The converted standard embedding layer.
    """
    std_embedding = nn.Embedding(
        num_embeddings=lora_embedding.num_embeddings,
        embedding_dim=lora_embedding.embedding_dim,
        padding_idx=lora_embedding.padding_idx,
        max_norm=lora_embedding.max_norm,
        norm_type=lora_embedding.norm_type,
        scale_grad_by_freq=lora_embedding.scale_grad_by_freq,
        sparse=lora_embedding.sparse
    ).to(lora_embedding.weight.device, dtype=lora_embedding.weight.dtype)
    std_embedding.weight.data.copy_(lora_embedding.weight.data)
    return std_embedding


def linear_std_to_lora(linear: nn.Linear, rank: int, alpha: float) -> LoRALinear:
    """
    Convert a standard linear layer to a LoRA linear layer.
    Args:
        linear (nn.Linear): The standard linear layer.
        rank (int): The rank for the LoRA layer.
        alpha (float): The scaling factor for the LoRA layer.
    Returns:
        LoRALinear: The converted LoRA linear layer.
    """
    bias = linear.bias is not None
    lora_linear = LoRALinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=bias,
        r=rank,
        lora_alpha=alpha,
        merge_weights=False
    )
    lora_linear.weight.data.copy_(linear.weight.data)
    if bias:
        lora_linear.bias.data.copy_(linear.bias.data)
    return lora_linear


def linear_lora_to_std(lora_linear: LoRALinear) -> nn.Linear:
    """
    Convert a LoRA linear layer back to a standard linear layer.
    Args:
        lora_linear (LoRALinear): The LoRA linear layer.
    Returns:
        nn.Linear: The converted standard linear layer.
    """
    bias = lora_linear.bias is not None
    std_linear = nn.Linear(
        in_features=lora_linear.in_features,
        out_features=lora_linear.out_features,
        bias=bias
    )
    std_linear.weight.data.copy_(lora_linear.weight.data)
    if bias:
        std_linear.bias.data.copy_(lora_linear.bias.data)
    return std_linear


def conv_std_to_lora(conv: torch.nn.modules.conv._ConvNd, rank: int, alpha: float) -> _LoRAConvNd:
    if isinstance(conv, nn.Conv1d):
        conv_class = LoRAConv1d
    elif isinstance(conv, nn.Conv2d):
        conv_class = LoRAConv2d
    elif isinstance(conv, nn.Conv3d):
        conv_class = LoRAConv3d
    else:
        raise ValueError(f"Unsupported convolution type: {type(conv)}")
    bias = conv.bias is not None
    lora_conv = conv_class(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=bias,
        padding_mode=conv.padding_mode,
        r=rank,
        lora_alpha=alpha,
        merge_weights=False
    )
    lora_conv.weight.data.copy_(conv.weight.data)
    if bias:
        lora_conv.bias.data.copy_(conv.bias.data)
    return lora_conv


def conv_lora_to_std(lora_conv: _LoRAConvNd) -> torch.nn.modules.conv._ConvNd:
    if isinstance(lora_conv, LoRAConv1d):
        conv_class = nn.Conv1d
    elif isinstance(lora_conv, LoRAConv2d):
        conv_class = nn.Conv2d
    elif isinstance(lora_conv, LoRAConv3d):
        conv_class = nn.Conv3d
    else:
        raise ValueError(f"Unsupported convolution type: {type(lora_conv)}")
    bias = lora_conv.bias is not None
    std_conv = conv_class(
        in_channels=lora_conv.in_channels,
        out_channels=lora_conv.out_channels,
        kernel_size=lora_conv.kernel_size,
        stride=lora_conv.stride,
        padding=lora_conv.padding,
        dilation=lora_conv.dilation,
        groups=lora_conv.groups,
        bias=bias,
        padding_mode=lora_conv.padding_mode
    )
    std_conv.weight.data.copy_(lora_conv.weight.data)
    if bias:
        std_conv.bias.data.copy_(lora_conv.bias.data)
    return std_conv


STD_TO_LORA = OrderedDict({
    nn.Embedding: embedding_std_to_lora,
    nn.Linear: linear_std_to_lora,
    torch.nn.modules.conv._ConvNd: conv_std_to_lora,
})
LORA_TO_STD = OrderedDict({
    LoRAEmbedding: embedding_lora_to_std,
    LoRALinear: linear_lora_to_std,
    _LoRAConvNd: conv_lora_to_std,
})


def create_lora(
        model: nn.Module,
        param_dict: dict[str, dict[str, Any]],  # map to named modules
):  # filtered param_dict
    """
    Create LoRA layers for the given model based on the provided parameter dictionary.
    Args:
        model (nn.Module): The model to modify.
        param_dict (dict[str, dict[str, Any]]): A dictionary containing the parameters for LoRA layers.
            The keys are the names of the layers, and the values are dictionaries with the following keys:
                - 'r': The rank for the LoRA layer.
                - 'alpha': The scaling factor for the LoRA layer.
    """
    children = dict(model.named_children())
    for name, child in children.items():
        convert_fn = None
        for module_type, fn in STD_TO_LORA.items():
            if isinstance(child, module_type):
                convert_fn = fn
                break
        if name in param_dict and convert_fn is not None:
            rank = param_dict[name]["rank"]
            alpha = param_dict[name]["alpha"]
            model.add_module(name, convert_fn(child, rank=rank, alpha=alpha))
            continue
        create_lora(model=child, param_dict={
            k[len(name) + 1:]: v for k, v in param_dict.items() if k.startswith(name + ".")
        })


def remove_lora(
        model: nn.Module
):
    """
    Remove LoRA layers from the given model.
    Args:
        model (nn.Module): The model to modify.
    """
    for name, module in list(model.named_modules()):
        if type(module) not in LORA_TO_STD:
            continue
        std_module = LORA_TO_STD[type(module)](module)
        model.add_module(name, std_module)


def merge_lora(
        model: nn.Module
):
    """
    Merge LoRA layers into the original layers in the given model.
    Args:
        model (nn.Module): The model to modify.
    """
    for name, module in list(model.named_modules()):
        if type(module) not in LORA_TO_STD:
            continue
        if not module.merged:
            module = copy.deepcopy(module)
            module.merge_weights = True
            module.eval()  # trigger merge
        std_module = LORA_TO_STD[type(module)](module)
        model.add_module(name, std_module)
