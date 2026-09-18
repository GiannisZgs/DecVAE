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
