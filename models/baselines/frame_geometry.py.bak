# coding=utf-8
# Copyright 2025 Ioannis Ziogas <ziogioan@ieee.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frame geometry for the baselines that learn no encoder of their own.

The post-analysis collators derive the frame grid from the model they are given. A baseline like
an eigenprojection has no encoder, so this stands in for one: it carries the conv kernels and
strides and nothing else, and computes the same frame counts and attention masks the VAE and
DecVAE models do.
"""

from typing import Union

import torch
from torch import nn


class FrameGeometry(nn.Module):
    """
    Args:
        conv_kernel (list): Kernel size of each conv layer of the frame grid.
        conv_stride (list): Stride of each conv layer of the frame grid.
    """

    def __init__(self, conv_kernel, conv_stride):
        super().__init__()
        if len(conv_kernel) != len(conv_stride):
            raise ValueError(
                f"conv_kernel and conv_stride must have the same length, got "
                f"{len(conv_kernel)} and {len(conv_stride)}."
            )
        self.kernels = list(conv_kernel)
        self.strides = list(conv_stride)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int]
    ):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.kernels, self.strides):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor
    ):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "FrameGeometry carries the frame grid only - it has no forward pass."
        )
