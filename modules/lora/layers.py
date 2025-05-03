import math

import loralib
import torch
import torch.nn.modules.conv
from loralib import LoRALayer
from torch import nn

__all__ = [
    "LoRALayer",
    "LoRAEmbedding",
    "LoRALinear",
    "LoRAConv1d",
    "LoRAConv2d",
    "LoRAConv3d",
]


class LoRAEmbedding(loralib.Embedding):
    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.normal_(self.lora_A)
            nn.init.zeros_(self.lora_B)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", rank={self.r}"


class LoRALinear(loralib.Linear):
    def extra_repr(self) -> str:
        return super().extra_repr() + f", rank={self.r}"


class _LoRAConvNd(torch.nn.modules.conv._ConvNd, LoRALayer):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs
    ):
        super(_LoRAConvNd, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * (self.weight.dim() - 2)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_channels // self.groups * torch.tensor(kernel_size).prod()))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels, r))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        super(_LoRAConvNd, self).reset_parameters()
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(_LoRAConvNd, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.weight.data -= (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.weight.data += (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self._conv_forward(
                x,
                self.weight + (self.lora_B @ self.lora_A).view(self.weight.shape) * self.scaling,
                self.bias
            )
        return super(_LoRAConvNd, self).forward(x)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", rank={self.r}"


class LoRAConv1d(_LoRAConvNd, nn.Conv1d):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs
    ):
        _LoRAConvNd.__init__(
            self, in_channels, out_channels, kernel_size,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=merge_weights, **kwargs
        )


class LoRAConv2d(_LoRAConvNd, nn.Conv2d):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs
    ):
        _LoRAConvNd.__init__(
            self, in_channels, out_channels, kernel_size,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=merge_weights, **kwargs
        )


class LoRAConv3d(_LoRAConvNd, nn.Conv3d):
    def __init__(
            self, in_channels, out_channels, kernel_size,
            r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, **kwargs
    ):
        _LoRAConvNd.__init__(
            self, in_channels, out_channels, kernel_size,
            r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            merge_weights=merge_weights, **kwargs
        )
