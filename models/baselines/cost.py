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

"""CoST, contrastive learning of disentangled seasonal-trend representations.

Ported from https://github.com/salesforce/CoST (BSD 3-Clause) so it runs inside the DecVAE
pre-training loop. The encoder, the banded Fourier layer, the momentum contrast and the two losses
follow the reference implementation; the einops calls are rewritten with plain torch, the
augmentations moved from the reference Dataset into the forward pass so the collator stays
method-agnostic, and the hardcoded .cuda() call replaced by the device of the batch.

CoST consumes a multivariate series. Here that series is one utterance read on the DecVAE frame
grid, so the representation comes out as (batch, frames, dim) and lines up with the frame labels.
"""

import copy
import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from .frame_geometry import FrameGeometry


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1)
            )
            for i in range(len(channels))
        ])

    def forward(self, x):
        return self.net(x)


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class BandedFourierLayer(nn.Module):
    """
    Learned complex-valued filter over one band of the rfft of the backbone output, which is what
    CoST reads the seasonal component with.

    Args:
        in_channels (int): Width of the backbone output.
        out_channels (int): Width of the seasonal component.
        band (int): Zero-indexed band this layer covers.
        num_bands (int): Number of bands the spectrum is split into.
        length (int): Number of frames the layer is built for. The forward pass checks the input
            against it, since a different length would silently write into the wrong bins.

    The reference keeps the filter as a complex parameter. safetensors, which the project checkpoints
    with, has no complex dtype, so the real and the imaginary halves are stored separately and
    recombined in the forward pass. The initialization is the reference one, taken apart afterwards.
    """

    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()

        self.length = length
        self.total_freqs = (self.length // 2) + 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.band = band
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0
        )

        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        weight = torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat)
        bias = torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat)
        weight, bias = self.reset_parameters(weight, bias)
        self.weight_real = nn.Parameter(weight.real.contiguous())
        self.weight_imag = nn.Parameter(weight.imag.contiguous())
        self.bias_real = nn.Parameter(bias.real.contiguous())
        self.bias_imag = nn.Parameter(bias.imag.contiguous())

    @property
    def weight(self):
        return torch.complex(self.weight_real, self.weight_imag)

    @property
    def bias(self):
        return torch.complex(self.bias_real, self.bias_imag)

    def forward(self, input):
        b, t, _ = input.shape
        if t != self.length:
            raise ValueError(
                f"BandedFourierLayer was built for {self.length} frames but received {t}. The frame "
                "count is fixed by max_duration_in_seconds and the decomposition grid, so a "
                "mismatch means the model and the cache disagree."
            )
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = self._forward(input_fft)
        return fft.irfft(output_fft, n=input.size(1), dim=1)

    def _forward(self, input):
        output = torch.einsum('bti,tio->bto', input[:, self.start:self.end], self.weight)
        return output + self.bias

    def reset_parameters(self, weight, bias):
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)
        return weight, bias


class CoSTEncoder(nn.Module):
    """
    Args:
        input_dims (int): Width of one frame of the input series.
        output_dims (int): Width of the representation, split in half between trend and seasonal.
        kernels (list): Kernel sizes of the mixture of autoregressive experts reading the trend.
        length (int): Number of frames per utterance.
        hidden_dims (int): Width of the dilated convolution backbone.
        depth (int): Number of dilated convolution blocks.
        mask_mode (str): Masking applied to the backbone input.
    """

    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k - 1) for k in kernels]
        )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )

    def forward(self, x, mask='all_true'):
        "x: (batch, frames, input_dims). The reference writes into x in place, so work on a copy"
        x = x.clone()
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)

        if mask is None:
            mask = self.mask_mode if self.training else 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        else:
            raise ValueError(f"Unknown mask mode {mask}")

        mask &= nan_mask
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        x = x.transpose(1, 2)
        x = self.feature_extractor(x)

        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        trend = torch.stack(trend, dim=0).mean(dim=0)

        x = x.transpose(1, 2)

        season = self.sfd[0](x)

        return trend, self.repr_dropout(season)


class CoSTForPreTraining(nn.Module):
    """
    CoST with the momentum contrast machinery attached, exposing the interfaces the DecVAE
    pre-training loop and the post-analysis collators expect.

    Args:
        input_dims (int): Width of one frame of the input series.
        seq_length (int): Number of frames per utterance.
        cost_args (:class:`~args_configs.cost_args.CoSTArguments`): The CoST hyperparameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the conv geometry
            of the frame grid, which the attention mask helpers and the collators read.
    """

    "CoST asserts the queue size is a multiple of the batch size, so batches must not vary"
    requires_fixed_batch_size = True

    def __init__(self, input_dims, seq_length, cost_args, config):
        super().__init__()

        self.input_dims = input_dims
        self.seq_length = seq_length
        self.output_dims = cost_args.cost_output_dims
        self.component_dims = cost_args.cost_output_dims // 2
        self.representation = cost_args.cost_representation
        self.mask_mode = cost_args.cost_mask_mode

        self.K = cost_args.cost_queue_size
        self.m = cost_args.cost_momentum
        self.T = cost_args.cost_temperature
        self.alpha = cost_args.cost_alpha
        self.sigma = cost_args.cost_augmentation_sigma
        self.aug_p = cost_args.cost_augmentation_prob

        self.encoder_q = CoSTEncoder(
            input_dims=input_dims,
            output_dims=cost_args.cost_output_dims,
            kernels=list(cost_args.cost_kernels),
            length=seq_length,
            hidden_dims=cost_args.cost_hidden_dims,
            depth=cost_args.cost_depth,
            mask_mode=cost_args.cost_mask_mode,
        )
        self.encoder_k = copy.deepcopy(self.encoder_q)

        dim = self.component_dims
        self.head_q = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.head_k = copy.deepcopy(self.head_q)

        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False
        for param_k in self.head_k.parameters():
            param_k.requires_grad = False

        self.register_buffer('queue', F.normalize(torch.randn(dim, self.K), dim=0))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        "The post-analysis collators derive the frame grid and the negatives count off the model"
        self.geometry = FrameGeometry(config.conv_kernel, config.conv_stride, config=config)
        self.config = self.geometry.config

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        return self.geometry._get_feat_extract_output_lengths(input_lengths)

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        return self.geometry._get_feature_vector_attention_mask(feature_vector_length, attention_mask)

    def _augment(self, x):
        "Jitter, scale and shift, applied per dimension as the reference PretrainDataset does"
        if torch.rand(1).item() <= self.aug_p:
            x = x * (torch.randn(x.size(-1), device=x.device) * self.sigma + 1)
        if torch.rand(1).item() <= self.aug_p:
            x = x + (torch.randn(x.size(-1), device=x.device) * self.sigma)
        if torch.rand(1).item() <= self.aug_p:
            x = x + (torch.randn(x.shape, device=x.device) * self.sigma)
        return x

    def compute_loss(self, q, k, k_negs):
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, k_negs])

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.T

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def convert_coeff(self, x, eps=1e-6):
        amp = torch.sqrt((x.real + eps).pow(2) + (x.imag + eps).pow(2))
        phase = torch.atan2(x.imag, x.real + eps)
        return amp, phase

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        z = torch.cat([z1, z2], dim=0)
        z = z.transpose(0, 1)
        sim = torch.matmul(z, z.transpose(1, 2))
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)
        return (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)
        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        if self.K % batch_size != 0:
            raise ValueError(
                f"cost_queue_size ({self.K}) must be a multiple of the batch size ({batch_size}). "
                "Both dataloaders drop their last incomplete batch, so this means the configured "
                "batch size itself does not divide the queue."
            )

        ptr = int(self.queue_ptr)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        self.queue_ptr[0] = (ptr + batch_size) % self.K

    def encode(self, input_values, sub_attention_mask=None):
        """
        Representation of every frame, for the evaluation path.

        Args:
            input_values: (batch, frames, input_dims) series on the DecVAE frame grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None. Padded frames are zeroed
                so they cannot carry a representation of the padding.
        Returns:
            A (representation, trend, season) tuple, each (batch, frames, dim). The representation
            follows cost_representation.
        """
        trend, season = self.encoder_q(input_values, mask='all_true')
        if self.representation == "trend":
            representation = trend
        elif self.representation == "seasonal":
            representation = season
        elif self.representation == "both":
            representation = torch.cat([trend, season], dim=-1)
        else:
            raise ValueError(
                f"Unknown cost_representation {self.representation}, expected 'both', 'trend' or 'seasonal'"
            )

        if sub_attention_mask is not None:
            keep = sub_attention_mask.to(torch.bool).unsqueeze(-1)
            representation = representation * keep
            trend = trend * keep
            season = season * keep

        return representation, trend, season

    def forward(self, input_values, sub_attention_mask=None):
        """
        Args:
            input_values: (batch, frames, input_dims) series on the DecVAE frame grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None. Padded frames are zeroed
                before the two views are built, so the augmentations do not turn padding into signal.
        Returns:
            A dict holding the total loss and its two parts.
        """
        if input_values.dim() != 3:
            raise ValueError(
                f"CoST expects a (batch, frames, features) series, got shape {tuple(input_values.shape)}."
            )

        if sub_attention_mask is not None:
            input_values = input_values * sub_attention_mask.to(input_values.dtype).unsqueeze(-1)

        x_q = self._augment(input_values)
        x_k = self._augment(input_values)

        "Trend: momentum contrast between the two views at one random frame"
        rand_idx = int(torch.randint(0, x_q.shape[1], (1,)).item())
        if sub_attention_mask is not None:
            "Pick a frame every utterance in the batch actually has, or the key is padding"
            valid = sub_attention_mask.to(torch.bool).all(dim=0).nonzero().flatten()
            if valid.numel() > 0:
                rand_idx = int(valid[torch.randint(0, valid.numel(), (1,))].item())

        q_t, q_s = self.encoder_q(x_q, mask=None)
        q_t = F.normalize(self.head_q(q_t[:, rand_idx]), dim=-1)

        with torch.no_grad():
            "The reference has no validation pass. Validation must read the queue and the key"
            "encoder without advancing either, or it would train the model it is measuring"
            if self.training:
                self._momentum_update_key_encoder()
            k_t, _ = self.encoder_k(x_k, mask=None)
            k_t = F.normalize(self.head_k(k_t[:, rand_idx]), dim=-1)

        trend_loss = self.compute_loss(q_t, k_t, self.queue.clone().detach())
        if self.training:
            self._dequeue_and_enqueue(k_t)

        "Seasonal: instance contrast on the amplitude and the phase of the two views"
        q_s = F.normalize(q_s, dim=-1)
        _, k_s = self.encoder_q(x_k, mask=None)
        k_s = F.normalize(k_s, dim=-1)

        q_s_amp, q_s_phase = self.convert_coeff(fft.rfft(q_s, dim=1))
        k_s_amp, k_s_phase = self.convert_coeff(fft.rfft(k_s, dim=1))

        seasonal_loss = self.instance_contrastive_loss(q_s_amp, k_s_amp) + \
                        self.instance_contrastive_loss(q_s_phase, k_s_phase)
        seasonal_loss = seasonal_loss / 2

        return {
            "loss": trend_loss + self.alpha * seasonal_loss,
            "trend_loss": trend_loss,
            "seasonal_loss": seasonal_loss,
        }


def build_cost(cost_args, input_dims, seq_length, config):
    """
    Build a CoST model for pre-training.

    Args:
        cost_args (:class:`~args_configs.cost_args.CoSTArguments`)
        input_dims (int): Width of one frame of the input series, read off the cached features.
        seq_length (int): Number of frames per utterance.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the conv geometry
            of the frame grid.
    Returns:
        CoSTForPreTraining
    """
    return CoSTForPreTraining(
        input_dims=input_dims,
        seq_length=seq_length,
        cost_args=cost_args,
        config=config,
    )
