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

"""TCL, Time-Contrastive Learning of Hyvarinen and Morioka.

Ported to PyTorch from https://github.com/hmorioka/TCL, which is TensorFlow 1.x and cannot be run as
it stands. The maxout feature extractor, the absolute-value feature layer, the multinomial logistic
regression over segment indices and the two-stage training all follow tcl/tcl.py and tcl_training.py.

A data point is one frame of the DecVAE grid, and the segments are stretches of fixed duration, so
the pretext label of a frame is its absolute position in the utterance and the same label means the
same elapsed time in every utterance. The representation is the last hidden layer, as in the
reference: the segment classifier is a training device and is dropped when the embedding is
evaluated.
"""

import math
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .frame_geometry import FrameGeometry


def fit_pca_whitening(mean, covariance, num_comp=None, zerotolerance=1e-7):
    """
    PCA whitening as subfunc/preprocessing.py fits it, from the statistics of the training frames.

    Args:
        mean (np.ndarray): (features,) mean of the training frames.
        covariance (np.ndarray): (features, features) covariance of the training frames.
        num_comp (int or None): Components kept. None keeps them all, which whitens without reducing.
        zerotolerance (float): Smallest eigenvalue, relative to the largest, that is accepted.
    Returns:
        dict: 'mean', 'W' the whitening matrix, 'A' the de-whitening matrix, and the contribution ratio.
    """
    if num_comp is None:
        num_comp = covariance.shape[0]

    d, V = np.linalg.eigh(covariance)
    "eigh returns ascending order; the reference works in descending order"
    d, V = d[::-1], V[:, ::-1]

    if np.sum((d[:num_comp] / d[0]) < zerotolerance) > 0:
        raise ValueError(
            f"The {num_comp} leading eigenvalues of the training frames are not all above "
            f"{zerotolerance} of the largest, so the whitening would divide by ~0. Lower "
            "tcl_pca_components."
        )

    contribution_ratio = float(np.sum(d[:num_comp]) / np.sum(d))
    dsqrt = np.sqrt(d[:num_comp])
    V = V[:, :num_comp]
    return {
        "mean": mean,
        "W": np.dot(np.diag(1 / dsqrt), V.transpose()),
        "A": np.dot(V, np.diag(dsqrt)),
        "contribution_ratio": contribution_ratio,
    }


def maxout(x, k):
    "Maximum over k consecutive affine maps, as the reference's maxout does"
    if x.shape[-1] % k != 0:
        raise ValueError(f"A maxout layer of {x.shape[-1]} units cannot be grouped into {k}.")
    return x.reshape(*x.shape[:-1], x.shape[-1] // k, k).max(dim=-1).values


class TCLNetwork(nn.Module):
    """
    The feature extractor and the segment classifier on top of it.

    Args:
        input_dims (int): Width of one frame.
        hidden_nodes (list): Width of each hidden layer; the last is the representation width.
        num_class (int): Number of segments, which is the number of classes of the pretext task.
        maxout_k (int): Affine maps per maxout unit.
        feature_nonlinearity (str): Nonlinearity of the last hidden layer.
    """

    def __init__(self, input_dims, hidden_nodes, num_class, maxout_k=2, feature_nonlinearity="abs"):
        super().__init__()

        if feature_nonlinearity != "abs":
            raise ValueError(
                f"Unknown tcl_feature_nonlinearity {feature_nonlinearity}, the reference only defines 'abs'."
            )
        self.hidden_nodes = list(hidden_nodes)
        self.maxout_k = maxout_k
        self.feature_nonlinearity = feature_nonlinearity
        self.feature_dim = self.hidden_nodes[-1]

        layers = []
        in_dim = input_dims
        for layer, out_dim in enumerate(self.hidden_nodes):
            "Every layer but the last is a maxout layer, so it is built maxout_k times wider"
            width = out_dim * maxout_k if layer < len(self.hidden_nodes) - 1 else out_dim
            layers.append(nn.Linear(in_dim, width))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)

        "Multinomial logistic regression over the segment indices"
        self.mlr = nn.Linear(self.feature_dim, num_class)

        for linear in list(self.layers) + [self.mlr]:
            "The reference initializes with variance scaling on fan-in and zero biases"
            nn.init.kaiming_normal_(linear.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(linear.bias)

    def features(self, x):
        """
        Args:
            x: (points, input_dims) frames.
        Returns:
            (points, feature_dim) representation.
        """
        for layer, linear in enumerate(self.layers):
            x = linear(x)
            if layer < len(self.layers) - 1:
                x = maxout(x, self.maxout_k)
            else:
                x = torch.abs(x)
        return x

    def forward(self, x):
        feats = self.features(x)
        return feats, self.mlr(feats)


class TCLForPreTraining(nn.Module):
    """
    TCL with its pretext task, exposing the interfaces the DecVAE pre-training loop and the
    post-analysis collators expect.

    Args:
        input_dims (int): Width of one frame.
        seq_length (int): Number of frames per utterance. Kept for the interface; TCL reads each
            frame on its own, so it does not size any parameter.
        tcl_args (:class:`~args_configs.tcl_args.TCLArguments`): The TCL hyperparameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the conv geometry
            of the frame grid, which the attention mask helpers and the collators read.
    """

    "Every frame is a data point of its own, so the batch may vary"
    requires_fixed_batch_size = False

    def __init__(self, input_dims, seq_length, tcl_args, config):
        super().__init__()

        self.input_dims = input_dims
        self.seq_length = seq_length
        self.mlr_init_steps = tcl_args.tcl_mlr_init_steps

        "Segments have a fixed duration, so a label means the same elapsed time in every utterance."
        "The class count follows from the longest utterance the grid holds"
        self.frames_per_segment = max(1, int(round(tcl_args.tcl_segment_duration_in_seconds / config.stride)))
        self.num_segments = int(math.ceil(seq_length / self.frames_per_segment))

        if self.num_segments < 2:
            raise ValueError(
                f"tcl_segment_duration_in_seconds {tcl_args.tcl_segment_duration_in_seconds} gives "
                f"{self.frames_per_segment} frames per segment, so an utterance of {seq_length} "
                "frames holds fewer than two segments. Shorten the segment."
            )

        "The reference whitens the data before the network. The transform is fitted on the training"
        "split and kept here, so it travels with the checkpoint and the evaluation reads the same one"
        "The reference keeps as many components as there are sources, which is also the width of"
        "its feature layer, so that relationship is what an unset count falls back to"
        self.pca_components = tcl_args.tcl_pca_components or tcl_args.tcl_hidden_nodes[-1]
        if self.pca_components > input_dims:
            raise ValueError(
                f"tcl_pca_components is {self.pca_components}, more than the {input_dims} features a frame holds."
            )
        self.register_buffer("pca_mean", torch.zeros(input_dims))
        self.register_buffer("pca_matrix", torch.eye(self.pca_components, input_dims))
        self.register_buffer("pca_fitted", torch.zeros(1, dtype=torch.long))

        self.network = TCLNetwork(
            input_dims=self.pca_components,
            hidden_nodes=tcl_args.tcl_hidden_nodes,
            num_class=self.num_segments,
            maxout_k=tcl_args.tcl_maxout_k,
            feature_nonlinearity=tcl_args.tcl_feature_nonlinearity,
        )
        self.feature_dim = self.network.feature_dim

        "Counts the optimization steps taken, so the classifier-only stage ends on its own"
        self.register_buffer("train_steps", torch.zeros(1, dtype=torch.long))

        "The post-analysis collators derive the frame grid and the negatives count off the model"
        self.geometry = FrameGeometry(config.conv_kernel, config.conv_stride, config=config)
        self.config = self.geometry.config

    def set_whitening(self, params):
        "Install the PCA whitening fitted on the training split"
        mean = torch.as_tensor(np.asarray(params["mean"]).reshape(-1), dtype=self.pca_mean.dtype)
        matrix = torch.as_tensor(np.asarray(params["W"]), dtype=self.pca_matrix.dtype)
        if mean.shape != self.pca_mean.shape or matrix.shape != self.pca_matrix.shape:
            raise ValueError(
                f"The whitening does not fit this model: mean {tuple(mean.shape)} and matrix "
                f"{tuple(matrix.shape)} against {tuple(self.pca_mean.shape)} and {tuple(self.pca_matrix.shape)}."
            )
        self.pca_mean.copy_(mean.to(self.pca_mean.device))
        self.pca_matrix.copy_(matrix.to(self.pca_matrix.device))
        self.pca_fitted.fill_(1)

    def whiten(self, x):
        "Centre and whiten the frames, as the reference's pca() does before the network"
        return (x - self.pca_mean) @ self.pca_matrix.t()

    def on_optimizer_step(self):
        "Called by the training loop once per optimization step, which is what the reference counts"
        self.train_steps += 1

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        return self.geometry._get_feat_extract_output_lengths(input_lengths)

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        return self.geometry._get_feature_vector_attention_mask(feature_vector_length, attention_mask)

    def segment_labels(self, batch, frames, sub_attention_mask, device):
        """
        Segment index of every frame, which is the label of the pretext task.

        Segments are frames_per_segment long, so a frame's label is its absolute position in the
        utterance and the same label is the same elapsed time everywhere. An utterance shorter than
        the longest one simply never reaches the later classes.
        """
        position = torch.arange(frames, device=device).unsqueeze(0).expand(batch, -1)
        labels = position // self.frames_per_segment
        return labels.clamp(max=self.num_segments - 1)

    def encode(self, input_values, sub_attention_mask=None):
        """
        Representation of every frame, for the evaluation path.

        Args:
            input_values: (batch, frames, input_dims) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None.
        Returns:
            A (representation, features, segment_logits) tuple. The representation is the last
            hidden layer, (batch, frames, feature_dim).
        """
        batch, frames, _ = input_values.shape
        feats, logits = self.network(self.whiten(input_values.reshape(batch * frames, -1)))
        feats = feats.reshape(batch, frames, -1)
        logits = logits.reshape(batch, frames, -1)

        if sub_attention_mask is not None:
            keep = sub_attention_mask.to(torch.bool).unsqueeze(-1)
            feats = feats * keep
            logits = logits * keep

        return feats, feats, logits

    def forward(self, input_values, sub_attention_mask=None):
        """
        Args:
            input_values: (batch, frames, input_dims) frames on the DecVAE grid.
            sub_attention_mask: (batch, frames) frame-level mask, or None. Padded frames are left
                out of the pretext task.
        Returns:
            A dict holding the loss and the accuracy of the segment classification.
        """
        if input_values.dim() != 3:
            raise ValueError(
                f"TCL expects a (batch, frames, features) input, got shape {tuple(input_values.shape)}."
            )
        batch, frames, _ = input_values.shape
        device = input_values.device

        labels = self.segment_labels(batch, frames, sub_attention_mask, device).reshape(-1)
        points = input_values.reshape(batch * frames, -1)
        if sub_attention_mask is not None:
            keep = sub_attention_mask.reshape(-1).to(torch.bool)
            points, labels = points[keep], labels[keep]

        if points.shape[0] < 1:
            raise ValueError("TCL received a batch with no non-padded frames.")

        feats = self.network.features(self.whiten(points))

        "The reference trains the classifier alone first, with the feature extractor held still."
        "train_steps counts optimization steps, advanced by the training loop, not forward passes"
        classifier_only = self.training and int(self.train_steps) < self.mlr_init_steps
        logits = self.network.mlr(feats.detach() if classifier_only else feats)

        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=-1) == labels).to(loss.dtype).mean()

        return {
            "loss": loss,
            "segment_accuracy": accuracy,
            "classifier_only": torch.tensor(float(classifier_only), device=device),
        }


def build_tcl(tcl_args, input_dims, seq_length, config):
    """
    Build a TCL model for pre-training.

    Args:
        tcl_args (:class:`~args_configs.tcl_args.TCLArguments`)
        input_dims (int): Width of one frame, read off the cached features.
        seq_length (int): Number of frames per utterance.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the frame grid.
    Returns:
        TCLForPreTraining
    """
    return TCLForPreTraining(
        input_dims=input_dims,
        seq_length=seq_length,
        tcl_args=tcl_args,
        config=config,
    )
