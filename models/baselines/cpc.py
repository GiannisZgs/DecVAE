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

"""CPC, Contrastive Predictive Coding of van den Oord et al. (2018).

There is no single reference implementation, so two are followed:
https://github.com/facebookresearch/CPC_audio for the architecture, the negative sampling and the
optimizer, and https://github.com/Spijkervet/contrastive-predictive-coding for the paper's GRU and
its log-bilinear predictor. The scoring, the ordering that puts the positive first and the
cross-entropy against label 0 are CPC_audio's CPCUnsupervisedCriterion.

The encoder emits z_t per frame and the autoregressive network emits c_t at the same rate, so the
frame axis survives end to end. CPC_audio's default encoder is five strided convolutions that build
its own 10 ms grid, but the reference also replaces that stack with a filterbank front-end under
encoder_type, 'mfcc' or 'lfb'. This port follows those variants: the filterbank is DecVAE's log-mel,
already extracted per frame at preprocessing, and g_enc maps one mel frame to z. CPC is handed the
same cached tensor DecVAE's own encoder reads, so its embeddings sit on DecVAE's frame grid without
any pooling or realignment.

Negatives are drawn from the same utterance rather than from anywhere in the batch, which is a
deliberate departure documented in cpc_args.py.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .frame_geometry import FrameGeometry


class ChannelNorm(nn.Module):
    """
    The normalization CPC_audio's encoder uses under its default normMode of 'layerNorm': the
    statistics are taken over the channel axis at each position, with a per-channel affine.

    Args:
        num_features (int): Channels of the layer.
        epsilon (float): Added to the variance before the reciprocal square root.
    """

    def __init__(self, num_features, epsilon=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.epsilon) * self.weight + self.bias


class CPCEncoder(nn.Module):
    """
    g_enc, the strided convolutional stack that reads one mel frame of the grid and emits z.

    The encoder stays convolutional as in the reference, but it strides over the frame the
    filterbank already produced rather than over raw samples, which is what the reference's 'mfcc'
    and 'lfb' front-ends do. The kernels and strides are DecVAE's seven-layer stack, which takes a
    400-value frame down to exactly one position - 79, 39, 19, 9, 4, 2, 1. Nothing else is taken
    from DecVAE: the ReLU after every layer and the ChannelNorm are CPC's own. One frame therefore
    yields one z, the frame grid is whatever the cached features already are, and nothing has to be
    realigned for the post-analysis.

    Args:
        input_dims (int): Width of one frame.
        conv_kernel (list): Kernel size of each layer.
        conv_stride (list): Stride of each layer.
        output_dim (int): Channels of the last layer, which is the width of z.
        hidden (int or None): Channels of every earlier layer. None keeps one width throughout, as
            the reference's sizeHidden does.
        norm (str): 'layer' for the reference's ChannelNorm, 'batch', or 'none'.
        bias (bool): Whether the convolutions carry a bias. The reference leaves nn.Conv1d's
            default, which is True.
    """

    def __init__(self, input_dims, conv_kernel, conv_stride, output_dim, hidden=None,
                 norm="layer", bias=True):
        super().__init__()

        if norm not in ("layer", "batch", "none"):
            raise ValueError(f"Unknown cpc_encoder_norm {norm}, expected 'layer', 'batch' or 'none'.")
        if len(conv_kernel) != len(conv_stride):
            raise ValueError(
                f"conv_kernel and conv_stride must have the same length, got {len(conv_kernel)} "
                f"and {len(conv_stride)}."
            )

        self.output_dim = output_dim
        self.kernels = list(conv_kernel)
        self.strides = list(conv_stride)

        "Walk the stack over the frame width, so a frame that it cannot reduce is caught here"
        length = input_dims
        for layer, (kernel, stride) in enumerate(zip(self.kernels, self.strides)):
            if length < kernel:
                raise ValueError(
                    f"g_enc strides its conv stack over one frame, but layer {layer} has kernel "
                    f"{kernel} and only {length} positions are left of the {input_dims} a frame "
                    "holds. Either the frame was narrowed, in which case set cpc_pool_mel_bins to "
                    "false and feed the mel frame whole, or cpc_conv_kernel and cpc_conv_stride "
                    "need to be sized for this frame width."
                )
            length = (length - kernel) // stride + 1
        self.output_length = length

        widths = [hidden or output_dim] * (len(self.kernels) - 1) + [output_dim]
        convs, norms = [], []
        in_channels = 1
        for width, kernel, stride in zip(widths, self.kernels, self.strides):
            convs.append(nn.Conv1d(in_channels, width, kernel_size=kernel, stride=stride, bias=bias))
            if norm == "layer":
                norms.append(ChannelNorm(width))
            elif norm == "batch":
                norms.append(nn.BatchNorm1d(width))
            else:
                norms.append(nn.Identity())
            in_channels = width
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.norm_mode = norm

    def forward(self, x):
        """
        Args:
            x: (batch, frames, input_dims) frames.
        Returns:
            (batch, frames, output_dim) z, after the nonlinearity the reference applies to every
            layer of its encoder including the last.
        """
        batch, frames, _ = x.shape

        "Every frame is convolved on its own, as one channel over the feature axis"
        h = x.reshape(batch * frames, 1, -1)
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h)))

        "The stack leaves one position per frame; anything left over is averaged into it"
        h = h.mean(dim=-1) if h.shape[-1] > 1 else h.squeeze(-1)
        return h.reshape(batch, frames, -1)


class CPCAutoregressive(nn.Module):
    """
    g_ar, summarizing the frames up to t into the context c_t.

    Args:
        input_dim (int): Width of z.
        hidden_dim (int): Width of c.
        layers (int): Recurrent layers.
        mode (str): 'gru', 'lstm' or 'rnn'.
    """

    def __init__(self, input_dim, hidden_dim, layers=1, mode="gru"):
        super().__init__()

        families = {"gru": nn.GRU, "lstm": nn.LSTM, "rnn": nn.RNN}
        if mode not in families:
            raise ValueError(f"Unknown cpc_ar_mode {mode}, expected 'gru', 'lstm' or 'rnn'.")
        self.hidden_dim = hidden_dim
        self.net = families[mode](input_dim, hidden_dim, num_layers=layers, batch_first=True)

    def forward(self, z, lengths=None):
        """
        Args:
            z: (batch, frames, input_dim).
            lengths: (batch,) count of non-padded frames, or None.
        Returns:
            (batch, frames, hidden_dim) context.
        """
        frames = z.shape[1]
        try:
            self.net.flatten_parameters()
        except RuntimeError:
            pass

        if lengths is None:
            return self.net(z)[0]

        "Packing keeps the context from running over the padding at the end of an utterance"
        packed = pack_padded_sequence(z, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, _ = self.net(packed)
        return pad_packed_sequence(out, batch_first=True, total_length=frames)[0]


class CPCForPreTraining(nn.Module):
    """
    CPC with its InfoNCE pretext task, exposing the interfaces the DecVAE pre-training loop and the
    post-analysis collators expect.

    Args:
        input_dims (int): Width of one frame.
        seq_length (int): Number of frames per utterance. Kept for the interface; nothing in CPC is
            sized by it, so a checkpoint transfers between datasets of different duration.
        cpc_args (:class:`~args_configs.cpc_args.CPCArguments`): The CPC hyperparameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the conv geometry
            of the frame grid, which the attention mask helpers and the collators read.
    """

    "Negatives are drawn per utterance, so the batch may vary"
    requires_fixed_batch_size = False

    def __init__(self, input_dims, seq_length, cpc_args, config):
        super().__init__()

        self.input_dims = input_dims
        self.seq_length = seq_length
        self.representation = cpc_args.cpc_representation
        self.n_predicts = cpc_args.cpc_prediction_steps
        self.n_negatives = cpc_args.cpc_negative_samples
        self.negative_sampling = cpc_args.cpc_negative_sampling

        if self.representation not in ("context", "z", "both"):
            raise ValueError(
                f"Unknown cpc_representation {self.representation}, expected 'context', 'z' or 'both'."
            )
        if self.negative_sampling not in ("same_sequence", "batch"):
            raise ValueError(
                f"Unknown cpc_negative_sampling {self.negative_sampling}, expected 'same_sequence' or 'batch'."
            )
        if self.n_predicts >= seq_length:
            raise ValueError(
                f"cpc_prediction_steps is {self.n_predicts}, but an utterance holds only {seq_length} "
                "frames, so no position has a target to predict. Shorten the horizon."
            )

        self.encoder = CPCEncoder(
            input_dims=input_dims,
            conv_kernel=cpc_args.cpc_conv_kernel,
            conv_stride=cpc_args.cpc_conv_stride,
            output_dim=cpc_args.cpc_encoder_dim,
            hidden=cpc_args.cpc_encoder_hidden,
            norm=cpc_args.cpc_encoder_norm,
        )
        self.ar = CPCAutoregressive(
            input_dim=cpc_args.cpc_encoder_dim,
            hidden_dim=cpc_args.cpc_ar_dim,
            layers=cpc_args.cpc_ar_layers,
            mode=cpc_args.cpc_ar_mode,
        )

        "One log-bilinear W_k per step, as the paper writes it and as the reference builds it when"
        "its prediction network is the plain linear one"
        self.predictors = nn.ModuleList(
            [nn.Linear(cpc_args.cpc_ar_dim, cpc_args.cpc_encoder_dim, bias=False)
             for _ in range(self.n_predicts)]
        )

        self.encoder_dim = cpc_args.cpc_encoder_dim
        self.ar_dim = cpc_args.cpc_ar_dim
        self.feature_dim = {"context": self.ar_dim, "z": self.encoder_dim,
                            "both": self.ar_dim + self.encoder_dim}[self.representation]

        "The post-analysis collators derive the frame grid and the negatives count off the model"
        self.geometry = FrameGeometry(config.conv_kernel, config.conv_stride, config=config)
        self.config = self.geometry.config

    def state_dict(self, *args, **kwargs):
        """
        The model's tensors, with the recurrent weights copied out of cuDNN's flat buffer.

        On the GPU torch flattens a GRU's weights into one buffer and leaves every parameter a view
        of it, and safetensors refuses to serialize tensors that share storage. Copying those views
        gives one tensor per parameter again, which is what the checkpoint should hold anyway.
        """
        state = super().state_dict(*args, **kwargs)
        for key, tensor in state.items():
            if type(tensor) is torch.Tensor and (
                tensor.untyped_storage().nbytes() != tensor.numel() * tensor.element_size()
            ):
                state[key] = tensor.clone()
        return state

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        return self.geometry._get_feat_extract_output_lengths(input_lengths)

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        return self.geometry._get_feature_vector_attention_mask(feature_vector_length, attention_mask)

    @staticmethod
    def _lengths(batch, frames, sub_attention_mask, device):
        if sub_attention_mask is None:
            return torch.full((batch,), frames, dtype=torch.long, device=device)
        return sub_attention_mask.to(torch.long).sum(dim=-1)

    def _sample_negatives(self, z, lengths, window):
        """
        Negatives for every position of the window, shared across the prediction steps as the
        reference shares them.

        'same_sequence' offsets the position by a random step and wraps inside the utterance, which
        is the reference's construction restricted to one utterance: it never reaches the padding.
        The offset starts past the prediction horizon rather than at 1, so a negative can never be
        the true future of any step. The reference lets that collision happen, since it draws from
        the whole batch and the odds are negligible; inside a single utterance of a few hundred
        frames they are not. 'batch' picks an utterance first and then a non-padded frame of it,
        which is what the reference does.

        Returns:
            (batch, n_negatives, window, encoder_dim)
        """
        batch, frames, dim = z.shape
        device = z.device
        shape = (batch, self.n_negatives, window)
        positions = torch.arange(window, device=device).view(1, 1, window)

        if self.negative_sampling == "same_sequence":
            span = lengths.clamp(min=2).view(batch, 1, 1)
            low = self.n_predicts + 1
            reach = (span - low).clamp(min=1)
            offset = (torch.rand(shape, device=device) * reach).long() + low
            frame_index = (positions + offset) % span
            entry_index = torch.arange(batch, device=device).view(batch, 1, 1).expand(shape)
        else:
            entry_index = torch.randint(0, batch, shape, device=device)
            span = lengths.clamp(min=1)[entry_index]
            frame_index = (torch.rand(shape, device=device) * span).long().clamp(max=frames - 1)

        flat = (entry_index * frames + frame_index).reshape(-1)
        return z.reshape(batch * frames, dim)[flat].reshape(batch, self.n_negatives, window, dim)

    def encode(self, input_values, sub_attention_mask=None):
        """
        Representation of every frame, for the evaluation path.

        Args:
            input_values: (batch, frames, input_dims) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None.
        Returns:
            A (representation, z, context) tuple. z is (batch, frames, cpc_encoder_dim) and the
            context is (batch, frames, cpc_ar_dim). The representation follows cpc_representation.
        """
        batch, frames, _ = input_values.shape
        lengths = self._lengths(batch, frames, sub_attention_mask, input_values.device)

        z = self.encoder(input_values)
        context = self.ar(z, lengths if sub_attention_mask is not None else None)

        if self.representation == "context":
            representation = context
        elif self.representation == "z":
            representation = z
        else:
            representation = torch.cat([z, context], dim=-1)

        if sub_attention_mask is not None:
            keep = sub_attention_mask.to(torch.bool).unsqueeze(-1)
            representation = representation * keep
            z = z * keep
            context = context * keep

        return representation, z, context

    def forward(self, input_values, sub_attention_mask=None):
        """
        Args:
            input_values: (batch, frames, input_dims) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None. Padded frames carry no
                context, are never drawn as negatives and are never scored as targets.
        Returns:
            A dict holding the InfoNCE loss summed over the prediction steps, as the reference sums
            it, and the accuracy of picking the true future out of the candidates.
        """
        if input_values.dim() != 3:
            raise ValueError(
                f"CPC expects a (batch, frames, features) input, got shape {tuple(input_values.shape)}."
            )
        batch, frames, _ = input_values.shape
        device = input_values.device

        window = frames - self.n_predicts
        if window < 1:
            raise ValueError(
                f"CPC predicts {self.n_predicts} steps ahead, which leaves no position in a batch of "
                f"{frames} frames."
            )

        lengths = self._lengths(batch, frames, sub_attention_mask, device)
        if int(lengths.max()) <= self.n_predicts:
            raise ValueError(
                "CPC received a batch whose longest utterance is not longer than its prediction "
                f"horizon: {int(lengths.max())} non-padded frames against {self.n_predicts} steps."
            )

        z = self.encoder(input_values)
        context = self.ar(z, lengths if sub_attention_mask is not None else None)
        negatives = self._sample_negatives(z, lengths, window)

        context_window = context[:, :window]
        positions = torch.arange(window, device=device).view(1, window)
        labels = torch.zeros(batch * window, dtype=torch.long, device=device)

        losses, accuracies = [], []
        for step in range(1, self.n_predicts + 1):
            prediction = self.predictors[step - 1](context_window)
            target = z[:, step:step + window]

            "The reference scores with the mean over the feature axis, positive first. The negatives"
            "are contracted rather than broadcast, which avoids holding the full outer product"
            positive = (prediction * target).mean(dim=-1).unsqueeze(1)
            negative = torch.einsum("bwd,bkwd->bkw", prediction, negatives) / target.shape[-1]
            logits = torch.cat([positive, negative], dim=1).permute(0, 2, 1).reshape(batch * window, -1)

            "A position counts only when its target is a frame the utterance actually holds"
            keep = ((positions + step) < lengths.view(batch, 1)).reshape(-1).to(logits.dtype)
            total = keep.sum().clamp(min=1.0)

            loss = (F.cross_entropy(logits, labels, reduction="none") * keep).sum() / total
            accuracy = ((logits.argmax(dim=-1) == labels).to(logits.dtype) * keep).sum() / total
            losses.append(loss)
            accuracies.append(accuracy)

        return {
            "loss": torch.stack(losses).sum(),
            "prediction_accuracy": torch.stack(accuracies).mean(),
            "step_1_accuracy": accuracies[0],
        }


def build_cpc(cpc_args, input_dims, seq_length, config):
    """
    Build a CPC model for pre-training.

    Args:
        cpc_args (:class:`~args_configs.cpc_args.CPCArguments`)
        input_dims (int): Width of one frame, read off the cached features.
        seq_length (int): Number of frames per utterance.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the frame grid.
    Returns:
        CPCForPreTraining
    """
    return CPCForPreTraining(
        input_dims=input_dims,
        seq_length=seq_length,
        cpc_args=cpc_args,
        config=config,
    )
