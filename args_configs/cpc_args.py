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

"""Arguments for the CPC baseline, Contrastive Predictive Coding of van den Oord et al. (2018).

CPC has no single reference implementation, so the defaults here are read off the two that are
followed: https://github.com/facebookresearch/CPC_audio, whose cpc_default_config.py sets the
architecture and the optimizer, and https://github.com/Spijkervet/contrastive-predictive-coding,
whose config/audio carries the paper's GRU and its single log-bilinear predictor.

An encoder emits one vector z_t per frame and an autoregressive network summarizes the past into a
context c_t at the same rate, and the pretext task is to tell the true z_{t+k} from negatives, for
k up to cpc_prediction_steps. Both vectors come out as (batch, frames, dim) on the DecVAE grid.

Where this port departs from the references, and why:

    g_enc   CPC_audio's default encoder is five strided convolutions over the raw waveform, which
            build its own 10 ms grid. The reference also offers two front-ends that replace that
            stack outright, selected with encoder_type: 'mfcc', a fixed filterbank whose
            coefficients become z directly, and 'lfb', a learned filterbank that analyses a
            400-sample window and pools it with a Hann window. g_enc here follows those variants.
            The filterbank is DecVAE's own log-mel, already extracted at preprocessing over the
            same 400-sample window at a 20 ms hop, and the encoder stays convolutional: the stack
            strides over that mel frame instead of over raw samples, collapsing it to one z. Its
            kernels and strides are CPC's own, near enough the reference's, and are not tied to
            DecVAE's conv geometry. Nothing about the grid has to be arranged either, since CPC is
            fed the very tensor DecVAE's encoder reads, with no pooling and no realignment, so the
            frames its embeddings sit on are the frames the labels sit on.
    k       cpc_prediction_steps is 6 rather than the reference's 12, since the reference predicts
            over a 10 ms grid and DecVAE's stride is 20 ms. The lookahead is the same 120 ms.
    negs    The references draw negatives from anywhere in the batch. The default here draws them
            from the same utterance: on a corpus like SimVowels a negative from another utterance
            is a negative from another speaker, which would let the model separate them on speaker
            identity alone and confound the axis being measured. 'batch' restores the reference's
            sampling. Two consequences follow from the smaller pool: the count is the second
            reference's 10 rather than CPC_audio's 128, and the offset is drawn past the prediction
            horizon so that a negative is never the true future of some step. The references let
            that collision happen because their pool makes it negligible; inside one utterance of a
            few hundred frames it is not.

The reference training recipe, which the shipped configs carry in the data arguments:
Adam with betas (0.9, 0.999), epsilon 1e-8 and no weight decay, learning rate 2e-4, a constant
learning rate (CPC_audio's schedulerStep defaults to -1, which disables its step decay, and the
second reference builds no scheduler at all), and batch 8, which both references default to.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CPCArguments:
    comment_cpc_args: str = field(
        default="CPC Baseline Arguments",
        metadata={"help": "A comment to add to the CPC arguments."},
    )
    cpc_input_type: str = field(
        default="mel",
        metadata={"help": "What each frame holds: 'mel' takes the log-mel features extracted per "
                          "frame at preprocessing, 'waveform' takes the frame samples themselves. "
                          "'mel' is the default and is the same cached tensor DecVAE's own encoder "
                          "reads, which is what puts CPC's embeddings on DecVAE's frame grid. The "
                          "reference's default encoder convolves the waveform, but it also defines "
                          "filterbank front-ends under encoder_type, which is the variant followed "
                          "here."},
    )
    cpc_n_mels: int = field(
        default=80,
        metadata={"help": "Number of mel bands, for the 'mel' input type."},
    )
    cpc_mel_norm: str = field(
        default="global",
        metadata={"help": "Mel normalization applied in the collator, for the 'mel' input type."},
    )
    cpc_pool_mel_bins: bool = field(
        default=False,
        metadata={"help": "Average the time bins preprocessing extracted inside each frame, leaving "
                          "one value per mel channel."},
    )
    cpc_conv_kernel: List[int] = field(
        default_factory=lambda: [10, 8, 4, 4, 3],
        metadata={"help": "Kernel of each layer of g_enc, the convolutional encoder. These are "
                          "CPC_audio's own kernels with only the last shortened, from 4 to 3, so "
                          "that the stack reduces a 400-value frame to exactly one position: 79, "
                          "18, 8, 3, 1. They are CPC's hyperparameters, not DecVAE's - the frame "
                          "grid comes from the cached features, so the stack does not have to "
                          "reproduce any geometry, only to collapse one frame to one z."},
    )
    cpc_conv_stride: List[int] = field(
        default_factory=lambda: [5, 4, 2, 2, 1],
        metadata={"help": "Stride of each layer of g_enc. CPC_audio's strides with the last dropped "
                          "to 1, for the reason given under cpc_conv_kernel. Must have the same "
                          "length as cpc_conv_kernel."},
    )
    cpc_encoder_hidden: int = field(
        default=256,
        metadata={"help": "Channels of every layer of g_enc but the last, which emits "
                          "cpc_encoder_dim. The reference's encoder is five strided convolutions of "
                          "width 256, and null keeps that single width throughout, which is what "
                          "the reference does. Here the convolutions stride over the frame the "
                          "filterbank already produced, as in its 'mfcc' and 'lfb' front-ends, "
                          "rather than over raw samples."},
    )
    cpc_encoder_dim: int = field(
        default=256,
        metadata={"help": "Width of z, the frame-local representation, which is hiddenEncoder in "
                          "the reference. The configs lower it to DecVAE's latent width so the "
                          "embeddings being compared have the same size."},
    )
    cpc_ar_dim: int = field(
        default=256,
        metadata={"help": "Width of c, the context the autoregressive network emits, which is "
                          "hiddenGar in the reference. The reference keeps it equal to "
                          "cpc_encoder_dim."},
    )
    cpc_ar_layers: int = field(
        default=1,
        metadata={"help": "Layers of the autoregressive network, nLevelsGRU in the reference."},
    )
    cpc_ar_mode: str = field(
        default="gru",
        metadata={"help": "Autoregressive network: 'gru' is the paper's and the second reference's, "
                          "'lstm' is CPC_audio's default, 'rnn' is the third choice it offers."},
    )
    cpc_encoder_norm: str = field(
        default="layer",
        metadata={"help": "Normalization inside g_enc: 'layer' is the reference's default normMode, "
                          "'batch' is its batchNorm, 'none' its ID."},
    )
    cpc_prediction_steps: int = field(
        default=6,
        metadata={"help": "How many steps ahead the context predicts, nPredicts in the reference. "
                          "The reference predicts 12 steps on a 10 ms grid; 6 steps on DecVAE's "
                          "20 ms stride is the same 120 ms of lookahead."},
    )
    cpc_negative_samples: int = field(
        default=10,
        metadata={"help": "Negatives drawn per position, shared across the prediction steps as the "
                          "references share them. This is the second reference's negative_samples. "
                          "CPC_audio's negativeSamplingExt of 128 is a count for a pool of "
                          "batch times sequence candidates; the default here draws inside one "
                          "utterance, where a few hundred frames cannot support that many distinct "
                          "negatives."},
    )
    cpc_negative_sampling: str = field(
        default="same_sequence",
        metadata={"help": "Where negatives come from: 'same_sequence' draws them from other frames "
                          "of the same utterance, 'batch' from anywhere in the batch, which is what "
                          "the references do. The default is deliberately not the reference's: "
                          "cross-utterance negatives hand the model speaker identity as a free "
                          "discriminator, which confounds the axis the evaluation measures. Padded "
                          "frames are never drawn under either setting."},
    )
    cpc_optimizer: str = field(
        default="adam",
        metadata={"help": "Optimizer family, taken from the reference implementations: 'adam' is "
                          "what both of them build, with the betas and epsilon of the data "
                          "arguments. 'adamw' uses the optimizer the rest of the project trains "
                          "with, and is an ablation rather than the baseline."},
    )
    cpc_representation: str = field(
        default="context",
        metadata={"help": "Which vector the evaluation reads: 'context' is c, the autoregressive "
                          "output, which is what the paper classifies on and what the reference "
                          "evaluates unless onEncoder is set; 'z' is the frame-local encoder output, "
                          "which lines up with DecVAE's Z the way TF-C's and TCL's do; 'both' "
                          "concatenates them. Both are returned whichever is chosen."},
    )
    cpc_seq_pooling: str = field(
        default=None,
        metadata={"help": "How the frames of an utterance are pooled into one sequence-level "
                          "embedding, for the '*_seq' targets. None skips them. 'mean' averages over "
                          "the frames of the utterance, as DecVAE's SequenceAggregator does, and is "
                          "what the other baselines use. 'last' takes the context at the last "
                          "non-padded frame, which CPC has and they do not, since c_t has summarized "
                          "the whole utterance by then."},
    )
    cpc_transfer_reinit_predictors: bool = field(
        default=False,
        metadata={"help": "Fine-tuning only. Nothing in CPC is sized by the frame count, so a "
                          "checkpoint transfers whole and this is off. On, the log-bilinear "
                          "predictors start over while the encoder and the autoregressive network "
                          "transfer, which is an ablation rather than a requirement."},
    )
