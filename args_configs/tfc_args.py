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

"""Arguments for the TF-C baseline, Time-Frequency Consistency
(https://github.com/mims-harvard/TFC-pretraining).

The defaults are the reference implementation's, read off code/config_files/SleepEEG_Configs.py,
main.py and trainer.py - SleepEEG is the dataset TF-C pre-trains on in the paper.

The reference training recipe, which the shipped configs carry in the data arguments:
Adam with betas (0.9, 0.99) and weight decay 3e-4, learning rate 3e-4, a constant learning rate,
and batch size 128. The learning rate is constant because trainer.py builds a ReduceLROnPlateau but
steps it only in the fine-tuning branch, never while pre-training.
"""

from dataclasses import dataclass, field


@dataclass
class TFCArguments:
    comment_tfc_args: str = field(
        default="TF-C Baseline Arguments",
        metadata={"help": "A comment to add to the TF-C arguments."},
    )
    tfc_input_type: str = field(
        default="waveform",
        metadata={"help": "What each frame holds: 'waveform' takes the frame samples themselves, "
                          "'mel' takes the log-mel features extracted per frame at preprocessing. "
                          "TF-C reads a time window and its magnitude spectrum, which raw samples "
                          "suit, so 'waveform' is the default."},
    )
    tfc_n_mels: int = field(
        default=80,
        metadata={"help": "Number of mel bands, for the 'mel' input type."},
    )
    tfc_mel_norm: str = field(
        default="global",
        metadata={"help": "Mel normalization applied in the collator, for the 'mel' input type."},
    )
    tfc_pool_mel_bins: bool = field(
        default=False,
        metadata={"help": "Average the time bins preprocessing extracted inside each frame, leaving "
                          "one value per mel channel."},
    )
    tfc_optimizer: str = field(
        default="adam",
        metadata={"help": "Optimizer family, taken from the reference implementation: 'adam' is "
                          "TF-C's own, with the betas, weight decay and learning rate of the data "
                          "arguments. 'adamw' uses the optimizer the rest of the project trains "
                          "with, and is an ablation rather than the baseline."},
    )
    tfc_transformer_layers: int = field(
        default=2,
        metadata={"help": "Number of layers in each of the two transformer encoders."},
    )
    tfc_attention_heads: int = field(
        default=2,
        metadata={"help": "Attention heads per layer. The frame width must be divisible by it, "
                          "since the frame is the model dimension."},
    )
    tfc_feedforward_multiplier: int = field(
        default=2,
        metadata={"help": "Feed-forward width as a multiple of the frame width."},
    )
    tfc_projector_hidden: int = field(
        default=256,
        metadata={"help": "Hidden width of the cross-space projectors."},
    )
    tfc_projector_dim: int = field(
        default=128,
        metadata={"help": "Output width of the cross-space projectors, where the time and the "
                          "frequency embeddings are compared against each other."},
    )
    tfc_temperature: float = field(
        default=0.2,
        metadata={"help": "Temperature of the NT-Xent losses."},
    )
    tfc_use_cosine_similarity: bool = field(
        default=True,
        metadata={"help": "Use cosine similarity in the NT-Xent losses rather than a dot product."},
    )
    tfc_lam: float = field(
        default=0.2,
        metadata={"help": "Weight of the two within-domain losses against the time-frequency loss: "
                          "loss = lam * (time + frequency) + time_frequency."},
    )
    tfc_jitter_ratio: float = field(
        default=2.0,
        metadata={"help": "Standard deviation of the jitter that builds the second time-domain "
                          "view. The reference applies it to standardized inputs."},
    )
    tfc_frequency_pertub_ratio: float = field(
        default=0.1,
        metadata={"help": "Fraction of spectrum bins removed, and the fraction perturbed by an "
                          "added component, when building the frequency-domain view."},
    )
    tfc_frames_per_batch_entry: int = field(
        default=128,
        metadata={"help": "Number of frames sampled per utterance of the batch for the contrastive "
                          "losses, so the contrastive batch is this times per_device_train_batch_size. "
                          "TF-C contrasts windows, and here every frame is a window, so an utterance "
                          "holds far more windows than the reference's batch - this bounds how many "
                          "of them a step reads. The frames are drawn from the non-padded ones of the "
                          "whole batch. The evaluation embeds every frame, this only bounds the loss."},
    )
    tfc_representation: str = field(
        default="both",
        metadata={"help": "Which representation the evaluation reads, off the cross-space "
                          "projectors: 'both' concatenates the time and the frequency projections, "
                          "so the width is twice tfc_projector_dim, 'time' or 'freq' take one of "
                          "them alone."},
    )
    tfc_seq_pooling: str = field(
        default=None,
        metadata={"help": "How the frames of an utterance are pooled into one sequence-level "
                          "embedding, for the '*_seq' targets. None skips them. 'mean' averages "
                          "over the frames of the utterance, as DecVAE's SequenceAggregator does."},
    )
