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

"""Arguments for the eigenprojection baselines (PCA, ICA, kernel PCA, SFA)."""

from dataclasses import dataclass, field


@dataclass
class EigenprojectionArguments:
    comment_eigenprojection_args: str = field(
        default="Eigenprojection Baseline Arguments",
        metadata={"help": "A comment to add to the eigenprojection arguments."},
    )
    projection_method: str = field(
        default=None,
        metadata={"help": "Which projection to learn: 'pca', 'ica', 'kpca-rbf', 'kpca-poly', "
                          "'kpca-sigmoid' or 'sfa'. These are standalone representation learners, "
                          "fitted on the train split and applied to the evaluation splits."},
    )
    projection_components: int = field(
        default=None,
        metadata={"help": "Dimensionality of the learned representation. None falls back to the "
                          "per-input-type defaults the VAE script used, kept in DEFAULT_COMPONENTS."},
    )
    projection_input_type: str = field(
        default="waveform",
        metadata={"help": "Which decomposition components feed the projection: 'waveform' or 'mel' "
                          "take the original signal alone, '*_ocs' every component but the original, "
                          "'*_all' all of them."},
    )
    projection_sfa_fit_grid: str = field(
        default="frames",
        metadata={"help": "Which time axis SFA accumulates its covariances on. 'sequence' reads the "
                          "sequence cut into frames that tile the utterance, so a one-step difference "
                          "is taken between disjoint windows - preprocessing extracts the 'mel*' "
                          "features on that same grid. 'frames' uses the strided grid the labels sit "
                          "on, whose consecutive frames overlap. Either way the map is applied to the "
                          "label grid. Ignored by the other methods, which do not read the time axis."},
    )
    projection_standardize: bool = field(
        default=True,
        metadata={"help": "Standardize the features before the projection, with the mean and standard "
                          "deviation fitted on the train split alone. Mel inputs are standardized per "
                          "mel channel, waveform inputs per frame position. Without it the high-energy "
                          "low mel bands dominate the covariance."},
    )
    projection_expansion: str = field(
        default=None,
        metadata={"help": "Function class the linear projection is learned over. None leaves the frames "
                          "as they are. 'context' concatenates projection_context_frames neighbours on "
                          "either side of each frame. 'quadratic' appends every pairwise product. Linear "
                          "SFA on plain log-mel returns overall energy and spectral tilt, which are slow "
                          "but uninformative, so the expanded run is the headline and the plain one a "
                          "secondary row."},
    )
    projection_context_frames: int = field(
        default=5,
        metadata={"help": "Neighbours taken on either side for the 'context' expansion, so the width "
                          "grows by a factor of 2*projection_context_frames + 1. Frames near the edges "
                          "of an utterance repeat the edge, which keeps one output frame per input frame "
                          "and so keeps the labels aligned."},
    )
    projection_expansion_components: int = field(
        default=500,
        metadata={"help": "Dimensionality the expanded features are PCA-reduced to before the projection "
                          "is fitted. Needed for 'quadratic', which squares the width. None or 0 skips "
                          "the reduction."},
    )
    projection_pool_mel_bins: bool = field(
        default=False,
        metadata={"help": "Average the time bins preprocessing extracted inside each frame, leaving one "
                          "value per mel channel, so a frame carries projection_n_mels features instead "
                          "of n_mels times bins. Off by default, which keeps the front-end identical to "
                          "the other baselines."},
    )
    projection_seq_pooling: str = field(
        default=None,
        metadata={"help": "How the frames of an utterance are pooled into one sequence-level embedding, "
                          "for the '*_seq' classification and disentanglement targets. None skips them. "
                          "'mean' averages over the frames of the utterance."},
    )
    projection_n_mels: int = field(
        default=80,
        metadata={"help": "Number of mel bands, for the 'mel*' input types."},
    )
    projection_mel_norm: str = field(
        default="global",
        metadata={"help": "Mel normalization applied in the collator, for the 'mel*' input types."},
    )
    projection_fit_fraction: float = field(
        default=None,
        metadata={"help": "Fraction of the training frames the projection is fitted on, sampled with "
                          "projection_seed. SFA samples whole utterances instead, as it needs the time "
                          "axis. None falls back to the fractions the VAE script used: 0.01 for kernel "
                          "PCA, 0.2 for the '*_ocs' and '*_all' waveform types, 1.0 otherwise."},
    )
    projection_seed: int = field(
        default=42,
        metadata={"help": "Seed of the generator that samples what the projection is fitted on."},
    )
    projection_kernel_gamma: float = field(
        default=0.1,
        metadata={"help": "Kernel coefficient of the 'kpca-*' methods."},
    )
