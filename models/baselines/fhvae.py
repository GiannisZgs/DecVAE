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

"""FHVAE, the Factorized Hierarchical VAE of Hsu, Zhang and Glass (NeurIPS 2017).

Ported to PyTorch from https://github.com/wnhsu/FactorizedHierarchicalVAE, which is TensorFlow 1.0
and Python 2.7 and cannot be run as it stands. The architecture follows rec_fhvae.py and the
lstm_1L_256_lat_32_32 config; the objective, the mu table and the discriminative term follow
base_fhvae.py; the closed-form mu estimate and the frame-level readout follow fhvae_runner.py and
datasets_loaders.py.

The reference swaps the paper's indices, so the names here are neither:

    paper z1, the segment latent   -> reference z2, qz2_x       -> here z_seg
    paper z2, the sequence latent  -> reference z1, qz1_x       -> here z_seq
    paper mu2, the prior mean      -> reference mu1_table       -> here mu_seq

An utterance is cut into segments of fhvae_seg_len frames. The sequence latent is inferred first
and the segment latent is conditioned on it, and the decoder reads both at every step. What ties an
utterance together is a trainable prior mean per training utterance; a held-out utterance has no
row, so its mean is estimated in closed form, which is also the s-vector the evaluation reads.
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .frame_geometry import FrameGeometry

LOG_2PI = math.log(2 * math.pi)


def log_gauss(x, mu, logvar):
    "Point-wise log N(x; mu, exp(logvar)), as the reference's log_gauss"
    return -0.5 * (LOG_2PI + logvar + (x - mu).pow(2) / logvar.exp())


def log_normal(x):
    "Point-wise log N(x; 0, I), as the reference's log_normal"
    return -0.5 * (LOG_2PI + x.pow(2))


def kld(mu, logvar, p_mu=None, p_logvar=None):
    """
    Dimension-wise KL( N(mu, exp(logvar)) || N(p_mu, exp(p_logvar)) ), as the reference's kld.

    A None prior mean or log-variance is the standard normal's.
    """
    p_mu = torch.zeros_like(mu) if p_mu is None else p_mu
    p_logvar = torch.zeros_like(logvar) if p_logvar is None else p_logvar
    return -0.5 * (1 + logvar - p_logvar - (logvar.exp() + (mu - p_mu).pow(2)) / p_logvar.exp())


class GaussianHead(nn.Module):
    "Two linear heads giving a mean and a log-variance, with no nonlinearity on either"

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.logvar = nn.Linear(in_dim, out_dim)

    def forward(self, h):
        return self.mu(h), self.logvar(h)


def reparameterize(mu, logvar, sample):
    "mu + exp(0.5 logvar) * eps while training, the mean otherwise"
    if not sample:
        return mu
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


class FHVAEForPreTraining(nn.Module):
    """
    FHVAE with its lower bound and discriminative term, exposing the interfaces the DecVAE
    pre-training loop and the post-analysis collators expect.

    Args:
        input_dims (int): Width of one frame.
        seq_length (int): Number of frames per utterance. Nothing is sized by it; segments are cut
            from whatever the batch holds.
        fhvae_args (:class:`~args_configs.fhvae_args.FHVAEArguments`): The FHVAE hyperparameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the conv geometry
            of the frame grid, which the attention mask helpers and the collators read.
        n_train_utts (int): Rows of the prior-mean table, one per training utterance. The
            discriminative term is a classification over them.
    """

    "Segments are cut per utterance, so the batch may vary"
    requires_fixed_batch_size = False

    "Windows are slid over the valid frames only, so encode has to be told where they end. The"
    "other baselines read each frame on its own and are handed no mask, which is left as it was"
    encode_requires_mask = True

    def __init__(self, input_dims, seq_length, fhvae_args, config, n_train_utts):
        super().__init__()

        self.input_dims = input_dims
        self.seq_length = seq_length
        self.seg_len = fhvae_args.fhvae_seg_len
        self.d_seg = fhvae_args.fhvae_d_seg
        self.d_seq = fhvae_args.fhvae_d_seq
        self.alpha = fhvae_args.fhvae_alpha
        self.representation = fhvae_args.fhvae_representation
        self.n_train_utts = int(n_train_utts)

        if self.representation not in ("both", "seg", "seq"):
            raise ValueError(
                f"Unknown fhvae_representation {self.representation}, expected 'both', 'seg' or 'seq'."
            )
        if self.seg_len < 1:
            raise ValueError(f"fhvae_seg_len must be at least 1, got {self.seg_len}.")
        if self.seg_len > seq_length:
            raise ValueError(
                f"fhvae_seg_len is {self.seg_len}, but an utterance holds only {seq_length} frames."
            )
        if self.n_train_utts < 1:
            raise ValueError(
                f"The prior-mean table needs one row per training utterance, got {n_train_utts}."
            )

        "p(z_seq | mu_seq) has a fixed standard deviation, so its log-variance is a constant"
        self.seq_var = float(fhvae_args.fhvae_seq_std) ** 2
        self.register_buffer("seq_logvar", torch.tensor(math.log(self.seq_var)))

        hidden = fhvae_args.fhvae_hidden
        self.z_seq_encoder = nn.LSTM(input_dims, hidden, num_layers=1, batch_first=True)
        self.z_seq_head = GaussianHead(hidden, self.d_seq)
        self.z_seg_encoder = nn.LSTM(input_dims + self.d_seq, hidden, num_layers=1, batch_first=True)
        self.z_seg_head = GaussianHead(hidden, self.d_seg)
        self.decoder = nn.LSTM(self.d_seq + self.d_seg, hidden, num_layers=1, batch_first=True)
        self.x_head = GaussianHead(hidden, input_dims)

        "One trainable prior mean per training utterance, the reference's mu1_table"
        self.mu_seq_table = nn.Parameter(torch.randn(self.n_train_utts, self.d_seq))

        self.feature_dim = {"both": self.d_seg + self.d_seq, "seg": self.d_seg,
                            "seq": self.d_seq}[self.representation]

        "The post-analysis collators derive the frame grid and the negatives count off the model"
        self.geometry = FrameGeometry(config.conv_kernel, config.conv_stride, config=config)
        self.config = self.geometry.config

    def state_dict(self, *args, **kwargs):
        """
        The model's tensors, with the recurrent weights copied out of cuDNN's flat buffer.

        On the GPU torch flattens an LSTM's weights into one buffer and leaves every parameter a
        view of it, and safetensors refuses to serialize tensors that share storage.
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

    def _lengths(self, batch, frames, sub_attention_mask, device):
        if sub_attention_mask is None:
            return torch.full((batch,), frames, dtype=torch.long, device=device)
        return sub_attention_mask.to(torch.long).sum(dim=-1)

    def _segment(self, x, lengths, random_starts):
        """
        Cut every utterance into segments of seg_len frames.

        Training draws the starts uniformly from the valid frames, as the reference's seg_rand does;
        evaluation cuts them non-overlapping from zero. The count per utterance is the reference's
        (L - seg_len) // seg_shift + 1 with seg_shift = seg_len, which is also the N of the prior
        term. An utterance shorter than a segment yields none and is masked out of the loss.

        Returns:
            segments (batch, max_segs, seg_len, features), valid (batch, max_segs) bool,
            n_segs (batch,)
        """
        batch, frames, dims = x.shape
        device = x.device

        n_segs = ((lengths - self.seg_len) // self.seg_len + 1).clamp(min=0)
        max_segs = int(n_segs.max())
        if max_segs == 0:
            return None, None, n_segs

        index = torch.arange(max_segs, device=device).view(1, max_segs)
        valid = index < n_segs.view(batch, 1)

        if random_starts:
            span = (lengths - self.seg_len + 1).clamp(min=1).view(batch, 1)
            starts = (torch.rand(batch, max_segs, device=device) * span).long().clamp(max=frames - self.seg_len)
        else:
            starts = index * self.seg_len
            starts = starts.expand(batch, max_segs).clamp(max=max(frames - self.seg_len, 0))
        starts = starts * valid

        offsets = torch.arange(self.seg_len, device=device).view(1, 1, self.seg_len)
        frame_index = starts.unsqueeze(-1) + offsets
        flat = (torch.arange(batch, device=device).view(batch, 1, 1) * frames + frame_index).reshape(-1)
        segments = x.reshape(batch * frames, dims)[flat].reshape(batch, max_segs, self.seg_len, dims)
        return segments, valid, n_segs

    def _encode_segments(self, segments, sample):
        """
        Run both encoders over a flat batch of segments.

        Args:
            segments: (n, seg_len, features).
            sample: Whether the latents are reparameterised or taken at their mean.
        Returns:
            A dict of the two posteriors and the samples fed onward.
        """
        h_seq, _ = self.z_seq_encoder(segments)
        "The reference reads all_h and keeps the last hidden state, with no cell state"
        mu_seq, logvar_seq = self.z_seq_head(h_seq[:, -1])
        z_seq = reparameterize(mu_seq, logvar_seq, sample)

        "The sequence latent is tiled over the segment and concatenated to every frame"
        tiled = z_seq.unsqueeze(1).expand(-1, segments.shape[1], -1)
        h_seg, _ = self.z_seg_encoder(torch.cat([segments, tiled], dim=-1))
        mu_seg, logvar_seg = self.z_seg_head(h_seg[:, -1])
        z_seg = reparameterize(mu_seg, logvar_seg, sample)

        return {"mu_seq": mu_seq, "logvar_seq": logvar_seq, "z_seq": z_seq,
                "mu_seg": mu_seg, "logvar_seg": logvar_seg, "z_seg": z_seg}

    def _decode(self, z_seq, z_seg, seg_len):
        """
        p(x | z_seg, z_seq). The input at every step is the two latents concatenated, and nothing
        is teacher-forced: the reference sets rec_dec_inp_train and rec_dec_inp_test to None.
        """
        latent = torch.cat([z_seq, z_seg], dim=-1).unsqueeze(1).expand(-1, seg_len, -1)
        h, _ = self.decoder(latent)
        return self.x_head(h)

    def _discriminative_logits(self, mu_seq):
        """
        Distance of the posterior mean from every training utterance's prior mean, as the
        reference's qy1_logits. Expanded into norms so the cost is (segments, utterances) rather
        than (segments, utterances, latent).
        """
        table = self.mu_seq_table
        squared = (mu_seq.pow(2).sum(dim=-1, keepdim=True)
                   - 2.0 * mu_seq @ table.t()
                   + table.pow(2).sum(dim=-1).view(1, -1))
        return -squared / (2 * self.seq_var)

    def estimate_mu_seq(self, mu_seq, valid, n_segs):
        """
        Closed-form estimate of an utterance's prior mean, the reference's _est_mu1 and the
        s-vector its dump_repr writes: sum over the utterance's segments of E[z_seq], divided by
        the segment count plus the prior variance.

        Args:
            mu_seq: (batch, max_segs, d_seq) posterior means.
            valid: (batch, max_segs) which segments exist.
            n_segs: (batch,) segment count per utterance.
        Returns:
            (batch, d_seq)
        """
        accumulated = (mu_seq * valid.unsqueeze(-1)).sum(dim=1)
        return accumulated / (n_segs.to(mu_seq.dtype) + self.seq_var).unsqueeze(-1)

    def forward(self, input_values, sub_attention_mask=None, utt_index=None):
        """
        Args:
            input_values: (batch, frames, features) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None. Segments are cut from the
                valid frames only.
            utt_index: (batch,) row of the prior-mean table for each utterance, for the training
                split. None, which is what a held-out split hands over, switches to the closed-form
                estimate and drops the discriminative term, as the reference's validation does.
        Returns:
            A dict holding the loss and the parts of the lower bound.
        """
        if input_values.dim() != 3:
            raise ValueError(
                f"FHVAE expects a (batch, frames, features) input, got shape {tuple(input_values.shape)}."
            )
        batch, frames, _ = input_values.shape
        device = input_values.device
        lengths = self._lengths(batch, frames, sub_attention_mask, device)

        "The reference randomizes segment starts while training and cuts them non-overlapping"
        "otherwise, which is also what the closed-form estimate is defined over"
        training = self.training
        segments, valid, n_segs = self._segment(input_values, lengths, random_starts=training)
        if segments is None:
            raise ValueError(
                f"No utterance in the batch holds a whole segment of {self.seg_len} frames; the "
                f"longest has {int(lengths.max())} valid frames."
            )

        max_segs = segments.shape[1]
        flat = segments.reshape(batch * max_segs, self.seg_len, -1)
        posterior = self._encode_segments(flat, sample=training)

        mu_seq = posterior["mu_seq"].reshape(batch, max_segs, -1)
        logvar_seq = posterior["logvar_seq"].reshape(batch, max_segs, -1)
        mu_seg = posterior["mu_seg"].reshape(batch, max_segs, -1)
        logvar_seg = posterior["logvar_seg"].reshape(batch, max_segs, -1)

        "The prior mean is the utterance's table row while training, and the closed-form estimate"
        "for a held-out utterance, which has no row"
        if utt_index is not None:
            if not training:
                raise ValueError(
                    "utt_index was passed in evaluation mode, but a held-out utterance has no row "
                    "in the prior-mean table. The estimate is computed in closed form instead."
                )
            labels = utt_index.to(torch.long).to(device)
            if int(labels.max()) >= self.n_train_utts:
                raise ValueError(
                    f"utt_index reaches {int(labels.max())} but the prior-mean table holds "
                    f"{self.n_train_utts} rows. It is built from the training split, so the "
                    "indices must be contiguous over that split."
                )
            mu_prior = self.mu_seq_table[labels].unsqueeze(1).expand(-1, max_segs, -1)
        else:
            labels = None
            mu_prior = self.estimate_mu_seq(mu_seq, valid, n_segs).unsqueeze(1).expand(-1, max_segs, -1)

        px_mu, px_logvar = self._decode(posterior["z_seq"], posterior["z_seg"], self.seg_len)
        px_mu = px_mu.reshape(batch, max_segs, self.seg_len, -1)
        px_logvar = px_logvar.reshape(batch, max_segs, self.seg_len, -1)

        "Every term is per segment, summed over the latent or feature dimensions as the reference does"
        logpx_z = log_gauss(segments, px_mu, px_logvar).sum(dim=(2, 3))
        neg_kld_seg = -kld(mu_seg, logvar_seg).sum(dim=-1)
        prior_logvar = self.seq_logvar.expand_as(logvar_seq)
        neg_kld_seq = -kld(mu_seq, logvar_seq, mu_prior, prior_logvar).sum(dim=-1)

        "log p(mu) is spread over the utterance's segments, so a full pass counts it once"
        log_pmu = log_normal(mu_prior).sum(dim=-1) / n_segs.clamp(min=1).to(mu_seq.dtype).view(batch, 1)

        lb = logpx_z + neg_kld_seg + neg_kld_seq + log_pmu

        weights = valid.to(lb.dtype)
        total = weights.sum().clamp(min=1.0)
        mean_lb = (lb * weights).sum() / total

        outputs = {
            "log_px_z": (logpx_z * weights).sum() / total,
            "kld_seg": -(neg_kld_seg * weights).sum() / total,
            "kld_seq": -(neg_kld_seq * weights).sum() / total,
            "segments_per_batch": total.detach(),
        }

        if labels is not None and self.alpha != 0.0:
            "The reference scores the posterior mean, not the sample"
            logits = self._discriminative_logits(mu_seq.reshape(batch * max_segs, -1))
            targets = labels.view(batch, 1).expand(-1, max_segs).reshape(-1)
            log_qy = -F.cross_entropy(logits, targets, reduction="none").reshape(batch, max_segs)
            mean_log_qy = (log_qy * weights).sum() / total
            loss = -(mean_lb + self.alpha * mean_log_qy)
            outputs["log_qy"] = mean_log_qy
        else:
            "Validation monitors the plain lower bound, which is what the reference selects on"
            loss = -mean_lb
            outputs["log_qy"] = torch.zeros((), device=device)

        outputs["loss"] = loss
        outputs["neg_lower_bound"] = -mean_lb
        return outputs

    def _windows(self, x, lengths):
        """
        Every window of seg_len frames at shift one, over the valid frames only, as the reference's
        get_frame_ra_dataset_conf sets up.

        An utterance shorter than a segment has its frames edge-replicated up to seg_len and yields
        a single window, so it still leaves one latent per frame rather than dropping out of the
        evaluation.

        Returns:
            windows (batch, max_windows, seg_len, features), counts (batch,)
        """
        batch, frames, dims = x.shape
        device = x.device
        counts = (lengths - self.seg_len + 1).clamp(min=1)
        max_windows = int(counts.max())

        start = torch.arange(max_windows, device=device).view(1, max_windows, 1)
        offset = torch.arange(self.seg_len, device=device).view(1, 1, self.seg_len)
        frame_index = (start + offset).expand(batch, max_windows, self.seg_len)

        "A short utterance repeats its last valid frame, and every window stays inside the valid part"
        ceiling = (lengths - 1).clamp(min=0).view(batch, 1, 1)
        frame_index = torch.minimum(frame_index, ceiling)

        flat = (torch.arange(batch, device=device).view(batch, 1, 1) * frames + frame_index).reshape(-1)
        windows = x.reshape(batch * frames, dims)[flat].reshape(batch, max_windows, self.seg_len, dims)
        return windows, counts

    def encode(self, input_values, sub_attention_mask=None):
        """
        One latent per frame, for the evaluation path.

        The reference extracts frame-level latents by sliding a segment one frame at a time and then
        padding the latent sequence out to the utterance length by replicating the first and the
        last latent, floor((seg_len-1)/2) on the left and ceil((seg_len-1)/2) on the right. The
        latents are replicated, not the frames, so the result lines up with the frame labels with no
        realignment. Both posteriors are taken at their means, and the sequence mean is what the
        segment encoder reads, as the reference's evaluation does.

        Args:
            input_values: (batch, frames, features) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None.
        Returns:
            A (representation, z_seg, z_seq) tuple, each (batch, frames, dim), plus the sequence
            estimate on the '_seq' attribute of the module for the pooled readout.
        """
        batch, frames, _ = input_values.shape
        device = input_values.device
        lengths = self._lengths(batch, frames, sub_attention_mask, device)

        windows, counts = self._windows(input_values, lengths)
        max_windows = windows.shape[1]
        posterior = self._encode_segments(windows.reshape(batch * max_windows, self.seg_len, -1),
                                          sample=False)
        mu_seg = posterior["mu_seg"].reshape(batch, max_windows, -1)
        mu_seq = posterior["mu_seq"].reshape(batch, max_windows, -1)

        left = int(math.floor((self.seg_len - 1) / 2.0))
        right = int(math.ceil((self.seg_len - 1) / 2.0))

        seg = input_values.new_zeros(batch, frames, self.d_seg)
        seq = input_values.new_zeros(batch, frames, self.d_seq)
        for b in range(batch):
            length = int(lengths[b])
            if length < 1:
                continue
            count = int(counts[b])
            if length < self.seg_len:
                "One window, replicated across the frames the utterance does have"
                seg[b, :length] = mu_seg[b, 0].unsqueeze(0).expand(length, -1)
                seq[b, :length] = mu_seq[b, 0].unsqueeze(0).expand(length, -1)
                continue
            for source, destination, width in ((mu_seg, seg, self.d_seg), (mu_seq, seq, self.d_seq)):
                centre = source[b, :count]
                padded = torch.cat([centre[:1].expand(left, width), centre,
                                    centre[-1:].expand(right, width)], dim=0)
                destination[b, :length] = padded[:length]

        if self.representation == "both":
            representation = torch.cat([seg, seq], dim=-1)
        elif self.representation == "seg":
            representation = seg
        else:
            representation = seq

        if sub_attention_mask is not None:
            keep = sub_attention_mask.to(torch.bool).unsqueeze(-1)
            representation = representation * keep
            seg = seg * keep
            seq = seq * keep

        return representation, seg, seq

    def pooled_sequence_embedding(self, input_values, sub_attention_mask=None):
        """
        The reference's s-vector, one vector per utterance: the closed-form prior-mean estimate over
        the non-overlapping segmentation, which is mean pooling of the sequence latent up to the
        factor N / (N + fhvae_seq_std^2).

        Returns:
            (batch, d_seq)
        """
        batch, frames, _ = input_values.shape
        lengths = self._lengths(batch, frames, sub_attention_mask, input_values.device)
        segments, valid, n_segs = self._segment(input_values, lengths, random_starts=False)

        if segments is None:
            "No utterance holds a whole segment, so fall back to the windowed means"
            windows, counts = self._windows(input_values, lengths)
            posterior = self._encode_segments(
                windows.reshape(batch * windows.shape[1], self.seg_len, -1), sample=False)
            mu_seq = posterior["mu_seq"].reshape(batch, windows.shape[1], -1)
            index = torch.arange(windows.shape[1], device=input_values.device).view(1, -1)
            valid = index < counts.view(batch, 1)
            return self.estimate_mu_seq(mu_seq, valid, counts)

        max_segs = segments.shape[1]
        posterior = self._encode_segments(segments.reshape(batch * max_segs, self.seg_len, -1),
                                          sample=False)
        mu_seq = posterior["mu_seq"].reshape(batch, max_segs, -1)
        return self.estimate_mu_seq(mu_seq, valid, n_segs)


def build_fhvae(fhvae_args, input_dims, seq_length, config, n_train_utts):
    """
    Build an FHVAE model for pre-training.

    Args:
        fhvae_args (:class:`~args_configs.fhvae_args.FHVAEArguments`)
        input_dims (int): Width of one frame, read off the cached features.
        seq_length (int): Number of frames per utterance.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the frame grid.
        n_train_utts (int): Rows of the prior-mean table, one per training utterance.
    Returns:
        FHVAEForPreTraining
    """
    return FHVAEForPreTraining(
        input_dims=input_dims,
        seq_length=seq_length,
        fhvae_args=fhvae_args,
        config=config,
        n_train_utts=n_train_utts,
    )
