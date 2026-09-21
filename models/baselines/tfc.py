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

"""TF-C, Time-Frequency Consistency.

Ported from https://github.com/mims-harvard/TFC-pretraining so it runs inside the DecVAE
pre-training loop. The two transformer encoders, the cross-space projectors, the poly NT-Xent loss
and the augmentations follow the reference implementation; the hardcoded cuda tensors are replaced
by the device of the batch, and the contrastive losses read a sampled set of frames.

TF-C contrasts a time window against its own magnitude spectrum. Here a window is one frame of the
DecVAE grid, so the representation comes out as (batch, frames, dim) and lines up with the frame
labels. Every frame is encoded on its own, as a sequence of length one, so a frame's embedding does
not depend on the other frames or on how the batch was formed.
"""

from typing import Optional, Union

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .frame_geometry import FrameGeometry


def ntxent_poly_loss(zis, zjs, temperature, use_cosine_similarity):
    """
    The reference NTXentLoss_poly, with the batch size read off the inputs.

    Args:
        zis, zjs (torch.Tensor): (samples, dim) embeddings of the two views.
        temperature (float): Temperature of the logits.
        use_cosine_similarity (bool): Cosine similarity rather than a dot product.
    Returns:
        torch.Tensor: The scalar loss.
    """
    batch_size = zis.shape[0]
    device = zis.device
    representations = torch.cat([zjs, zis], dim=0)

    if use_cosine_similarity:
        "Normalizing first gives the same matrix as the reference's pairwise call without"
        "materializing a (2N, 2N, dim) tensor, which the sampled frames can make large"
        normalized = F.normalize(representations, dim=-1)
        similarity_matrix = normalized @ normalized.T
    else:
        similarity_matrix = torch.tensordot(
            representations.unsqueeze(1), representations.T.unsqueeze(0), dims=2
        )

    "Mask out a sample against itself and against its own other view"
    diag = torch.eye(2 * batch_size, device=device)
    l1 = torch.diag(torch.ones(batch_size, device=device), -batch_size)
    l2 = torch.diag(torch.ones(batch_size, device=device), batch_size)
    mask = (1 - (diag + l1 + l2)).bool()

    l_pos = torch.diag(similarity_matrix, batch_size)
    r_pos = torch.diag(similarity_matrix, -batch_size)
    positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)
    negatives = similarity_matrix[mask].view(2 * batch_size, -1)

    logits = torch.cat((positives, negatives), dim=1) / temperature
    labels = torch.zeros(2 * batch_size, device=device).long()
    CE = F.cross_entropy(logits, labels, reduction="sum")

    onehot_label = torch.cat(
        (torch.ones(2 * batch_size, 1, device=device), torch.zeros(2 * batch_size, negatives.shape[-1], device=device)),
        dim=-1,
    ).long()
    pt = torch.mean(onehot_label * F.softmax(logits, dim=-1))

    epsilon = batch_size
    return CE / (2 * batch_size) + epsilon * (1 / batch_size - pt)


def jitter(x, sigma):
    "Second time-domain view, as DataTransform_TD builds it"
    return x + torch.randn_like(x) * sigma


def remove_frequency(x, pertub_ratio):
    "Drop a fraction of the spectrum bins"
    mask = torch.rand_like(x) > pertub_ratio
    return x * mask


def add_frequency(x, pertub_ratio):
    "Add a random component to a fraction of the spectrum bins"
    mask = torch.rand_like(x) > (1 - pertub_ratio)
    max_amplitude = x.max()
    random_am = torch.rand_like(x) * (max_amplitude * 0.1)
    return x + mask * random_am


class TFC(nn.Module):
    """
    The two contrastive encoders and their cross-space projectors.

    Args:
        ts_length (int): Width of one frame, which is the model dimension of both encoders.
        layers (int): Transformer layers per encoder.
        heads (int): Attention heads per layer.
        feedforward_multiplier (int): Feed-forward width as a multiple of ts_length.
        projector_hidden (int): Hidden width of the projectors.
        projector_dim (int): Output width of the projectors.
    """

    def __init__(self, ts_length, layers=2, heads=2, feedforward_multiplier=2,
                 projector_hidden=256, projector_dim=128):
        super().__init__()

        if ts_length % heads != 0:
            raise ValueError(
                f"TF-C makes the frame width the model dimension, so it must be divisible by "
                f"tfc_attention_heads: {ts_length} frame features and {heads} heads."
            )
        self.ts_length = ts_length

        encoder_layers_t = TransformerEncoderLayer(
            ts_length, dim_feedforward=feedforward_multiplier * ts_length, nhead=heads
        )
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, layers)
        self.projector_t = nn.Sequential(
            nn.Linear(ts_length, projector_hidden),
            nn.BatchNorm1d(projector_hidden),
            nn.ReLU(),
            nn.Linear(projector_hidden, projector_dim),
        )

        encoder_layers_f = TransformerEncoderLayer(
            ts_length, dim_feedforward=feedforward_multiplier * ts_length, nhead=heads
        )
        self.transformer_encoder_f = TransformerEncoder(encoder_layers_f, layers)
        self.projector_f = nn.Sequential(
            nn.Linear(ts_length, projector_hidden),
            nn.BatchNorm1d(projector_hidden),
            nn.ReLU(),
            nn.Linear(projector_hidden, projector_dim),
        )

    def forward(self, x_in_t, x_in_f):
        """
        Args:
            x_in_t: (batch, frames, ts_length) frames.
            x_in_f: (batch, frames, ts_length) magnitude spectra of those frames.
        Returns:
            h_time, z_time, h_freq, z_freq. The h_* are (batch, frames, ts_length), the z_* are
            (batch, frames, projector_dim).
        """
        batch, frames, _ = x_in_t.shape

        "Each frame is its own sequence of length one, as the reference encodes each sample alone"
        t = x_in_t.reshape(batch * frames, -1).unsqueeze(0)
        h_time = self.transformer_encoder_t(t).squeeze(0)
        z_time = self.projector_t(h_time)

        f = x_in_f.reshape(batch * frames, -1).unsqueeze(0)
        h_freq = self.transformer_encoder_f(f).squeeze(0)
        z_freq = self.projector_f(h_freq)

        return (h_time.reshape(batch, frames, -1), z_time.reshape(batch, frames, -1),
                h_freq.reshape(batch, frames, -1), z_freq.reshape(batch, frames, -1))


class TFCForPreTraining(nn.Module):
    """
    TF-C with its pre-training losses, exposing the interfaces the DecVAE pre-training loop and the
    post-analysis collators expect.

    Args:
        input_dims (int): Width of one frame.
        seq_length (int): Number of frames per utterance. Kept for the interface; TF-C reads each
            frame on its own, so it does not size any parameter.
        tfc_args (:class:`~args_configs.tfc_args.TFCArguments`): The TF-C hyperparameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the conv geometry
            of the frame grid, which the attention mask helpers and the collators read.
    """

    "The contrastive losses read a sampled set of frames, so the batch may vary"
    requires_fixed_batch_size = False

    def __init__(self, input_dims, seq_length, tfc_args, config):
        super().__init__()

        self.input_dims = input_dims
        self.seq_length = seq_length
        self.representation = tfc_args.tfc_representation
        self.temperature = tfc_args.tfc_temperature
        self.use_cosine_similarity = tfc_args.tfc_use_cosine_similarity
        self.lam = tfc_args.tfc_lam
        self.jitter_ratio = tfc_args.tfc_jitter_ratio
        self.pertub_ratio = tfc_args.tfc_frequency_pertub_ratio
        self.frames_per_batch_entry = tfc_args.tfc_frames_per_batch_entry

        self.tfc = TFC(
            ts_length=input_dims,
            layers=tfc_args.tfc_transformer_layers,
            heads=tfc_args.tfc_attention_heads,
            feedforward_multiplier=tfc_args.tfc_feedforward_multiplier,
            projector_hidden=tfc_args.tfc_projector_hidden,
            projector_dim=tfc_args.tfc_projector_dim,
        )

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

    @staticmethod
    def spectrum(x):
        "Magnitude spectrum of every frame, as the reference dataloader builds it"
        return fft.fft(x, dim=-1).abs()

    def _sample_frames(self, batch, frames, sub_attention_mask, device):
        """
        Indices into the flattened (batch*frames) axis that the contrastive losses read.

        Padded frames are left out, and the count is tfc_frames_per_batch_entry per utterance of
        the batch, so the contrastive batch grows with the batch rather than staying fixed.
        """
        if sub_attention_mask is not None:
            valid = sub_attention_mask.reshape(-1).to(torch.bool).nonzero().flatten()
        else:
            valid = torch.arange(batch * frames, device=device)

        if valid.numel() < 2:
            raise ValueError(
                "TF-C needs at least two frames for its contrastive losses, but the batch holds "
                f"{valid.numel()} non-padded frames."
            )
        sampled = self.frames_per_batch_entry * batch
        if valid.numel() <= sampled:
            return valid
        return valid[torch.randperm(valid.numel(), device=device)[:sampled]]

    def encode(self, input_values, sub_attention_mask=None):
        """
        Representation of every frame, for the evaluation path.

        Args:
            input_values: (batch, frames, input_dims) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None.
        Returns:
            A (representation, z_time, z_freq) tuple of the cross-space projector outputs, each
            (batch, frames, tfc_projector_dim). The representation follows tfc_representation.
        """
        _, z_time, _ , z_freq = self.tfc(input_values, self.spectrum(input_values))

        if self.representation == "time":
            representation = z_time
        elif self.representation == "freq":
            representation = z_freq
        elif self.representation == "both":
            representation = torch.cat([z_time, z_freq], dim=-1)
        else:
            raise ValueError(
                f"Unknown tfc_representation {self.representation}, expected 'both', 'time' or 'freq'"
            )

        if sub_attention_mask is not None:
            keep = sub_attention_mask.to(torch.bool).unsqueeze(-1)
            representation = representation * keep
            z_time = z_time * keep
            z_freq = z_freq * keep

        return representation, z_time, z_freq

    def forward(self, input_values, sub_attention_mask=None):
        """
        Args:
            input_values: (batch, frames, input_dims) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None. Padded frames are zeroed
                before the views are built and left out of the losses.
        Returns:
            A dict holding the total loss and its parts.
        """
        if input_values.dim() != 3:
            raise ValueError(
                f"TF-C expects a (batch, frames, features) input, got shape {tuple(input_values.shape)}."
            )
        batch, frames, _ = input_values.shape

        if sub_attention_mask is not None:
            input_values = input_values * sub_attention_mask.to(input_values.dtype).unsqueeze(-1)

        "The frames are sampled before they are encoded: every frame is encoded on its own, so this"
        "gives what encoding the whole batch and selecting afterwards would, and it leaves the"
        "projector's batch norm reading exactly the windows being contrasted, as the reference does"
        index = self._sample_frames(batch, frames, sub_attention_mask, input_values.device)
        x_t = input_values.reshape(batch * frames, -1)[index]
        x_f = self.spectrum(x_t)
        aug_t = jitter(x_t, self.jitter_ratio)
        aug_f = remove_frequency(x_f, self.pertub_ratio) + add_frequency(x_f, self.pertub_ratio)

        squeeze = lambda outputs: tuple(out.squeeze(0) for out in outputs)
        h_t, z_t, h_f, z_f = squeeze(self.tfc(x_t.unsqueeze(0), x_f.unsqueeze(0)))
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = squeeze(self.tfc(aug_t.unsqueeze(0), aug_f.unsqueeze(0)))

        ntxent = lambda a, b: ntxent_poly_loss(a, b, self.temperature, self.use_cosine_similarity)
        time_loss = ntxent(h_t, h_t_aug)
        freq_loss = ntxent(h_f, h_f_aug)
        l_TF = ntxent(z_t, z_f)

        "The paper's consistency term, logged as the reference computes it. The released trainer"
        "optimizes l_TF rather than this, and that is what is followed here"
        l_1, l_2, l_3 = ntxent(z_t, z_f_aug), ntxent(z_t_aug, z_f), ntxent(z_t_aug, z_f_aug)
        consistency_margin = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        return {
            "loss": self.lam * (time_loss + freq_loss) + l_TF,
            "time_loss": time_loss,
            "freq_loss": freq_loss,
            "time_frequency_loss": l_TF,
            "consistency_margin": consistency_margin,
        }


def build_tfc(tfc_args, input_dims, seq_length, config):
    """
    Build a TF-C model for pre-training.

    Args:
        tfc_args (:class:`~args_configs.tfc_args.TFCArguments`)
        input_dims (int): Width of one frame, read off the cached features.
        seq_length (int): Number of frames per utterance.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the frame grid.
    Returns:
        TFCForPreTraining
    """
    return TFCForPreTraining(
        input_dims=input_dims,
        seq_length=seq_length,
        tfc_args=tfc_args,
        config=config,
    )
