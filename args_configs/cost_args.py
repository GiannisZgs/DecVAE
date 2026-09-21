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

"""Arguments for the CoST baseline (https://github.com/salesforce/CoST).

The defaults are the ones the reference implementation uses, except for the front-end, which CoST
has none of - it consumes a multivariate series. Here that series is the utterance read on the
DecVAE frame grid, one vector per frame.

The reference training recipe, which the shipped configs carry in the data arguments:
SGD with momentum 0.9 and weight decay 1e-4, learning rate 1e-3, a half-cosine decay to zero with
no warmup, and batch size 128. The batch size is the one every script under CoST/scripts uses;
train.py's own argparse default of 8 was never used for a published run.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CoSTArguments:
    comment_cost_args: str = field(
        default="CoST Baseline Arguments",
        metadata={"help": "A comment to add to the CoST arguments."},
    )
    cost_input_type: str = field(
        default="mel",
        metadata={"help": "What each frame of the series holds: 'mel' takes the log-mel features "
                          "extracted per frame at preprocessing, 'waveform' takes the frame samples "
                          "themselves. CoST normalizes and augments per dimension, which mel bands "
                          "suit and raw samples do not, so 'mel' is the default."},
    )
    cost_n_mels: int = field(
        default=80,
        metadata={"help": "Number of mel bands, for the 'mel' input type. Must match the value the "
                          "cache was written with, since the cache name records only the feature "
                          "type and not the band count."},
    )
    cost_mel_norm: str = field(
        default="global",
        metadata={"help": "Mel normalization applied in the collator, for the 'mel' input type."},
    )
    cost_pool_mel_bins: bool = field(
        default=False,
        metadata={"help": "Average the time bins preprocessing extracted inside each frame, leaving "
                          "one value per mel channel. Off by default, which keeps the frame "
                          "features identical to DecVAE's: n_mels times the bins mel_hops yields."},
    )
    cost_optimizer: str = field(
        default="sgd",
        metadata={"help": "Optimizer family, taken from the reference implementation: 'sgd' is "
                          "CoST's own, with cost_sgd_momentum and the weight decay and learning "
                          "rate of the data arguments. 'adamw' uses the optimizer the rest of the "
                          "project trains with, and is an ablation rather than the baseline."},
    )
    cost_sgd_momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum of the SGD optimizer, as the reference implementation sets it."},
    )
    cost_output_dims: int = field(
        default=320,
        metadata={"help": "Dimensionality of the representation. Split in half between the trend "
                          "and the seasonal component, which are concatenated back together."},
    )
    cost_hidden_dims: int = field(
        default=64,
        metadata={"help": "Width of the dilated convolution backbone."},
    )
    cost_depth: int = field(
        default=10,
        metadata={"help": "Number of dilated convolution blocks in the backbone."},
    )
    cost_kernels: List[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128],
        metadata={"help": "Kernel sizes of the mixture of autoregressive experts that reads the "
                          "trend. Every kernel is applied causally and its output trimmed back to "
                          "the frame count, so kernels wider than the utterance are still valid."},
    )
    cost_alpha: float = field(
        default=0.0005,
        metadata={"help": "Weight of the seasonal (frequency domain) loss against the trend "
                          "(time domain) loss."},
    )
    cost_queue_size: int = field(
        default=256,
        metadata={"help": "Size of the momentum contrast queue of negatives. Must be a multiple of "
                          "the train and eval batch sizes, which is why both dataloaders drop their "
                          "last incomplete batch."},
    )
    cost_momentum: float = field(
        default=0.999,
        metadata={"help": "Decay of the momentum update of the key encoder."},
    )
    cost_temperature: float = field(
        default=0.07,
        metadata={"help": "Temperature of the momentum contrast logits."},
    )
    cost_augmentation_sigma: float = field(
        default=0.5,
        metadata={"help": "Scale of the jitter, scaling and shift augmentations that build the two "
                          "views of an utterance."},
    )
    cost_augmentation_prob: float = field(
        default=0.5,
        metadata={"help": "Probability that each of the three augmentations is applied to a view."},
    )
    cost_mask_mode: str = field(
        default="all_true",
        metadata={"help": "Masking applied to the backbone input: 'all_true' masks nothing, which "
                          "is what the reference pre-training path uses, 'binomial' drops frames at "
                          "random."},
    )
    cost_transfer_reinit_fourier: bool = field(
        default=False,
        metadata={"help": "Fine-tuning only. The banded Fourier layer that reads the seasonal "
                          "component is sized by the frame count, so a source and a target dataset "
                          "with different max_duration_in_seconds cannot share it. Off, the transfer "
                          "stops with the two frame counts named. On, everything else is transferred "
                          "and that layer starts from a random initialization."},
    )
    cost_representation: str = field(
        default="both",
        metadata={"help": "Which representation the evaluation reads: 'both' concatenates the trend "
                          "and the seasonal halves, as the paper does, 'trend' or 'seasonal' take "
                          "one of them alone."},
    )
    cost_seq_pooling: str = field(
        default=None,
        metadata={"help": "How the frames of an utterance are pooled into one sequence-level "
                          "embedding, for the '*_seq' targets. None skips them. 'mean' averages "
                          "over the frames of the utterance, as DecVAE's SequenceAggregator does."},
    )
