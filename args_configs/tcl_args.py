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

"""Arguments for the TCL baseline, Time-Contrastive Learning of Hyvarinen and Morioka
(https://github.com/hmorioka/TCL).

TCL is self-supervised: it splits the signal into time segments and trains a feature extractor to
tell which segment a data point came from. The last hidden layer is the representation, and the
segment classifier on top of it is discarded afterwards. Here a data point is one frame of the
DecVAE grid and the segments are equal stretches of an utterance, so the representation comes out
as (batch, frames, dim) and lines up with the frame labels.

The defaults are the reference implementation's, read off tcl_training.py and tcl/tcl.py.

The reference training recipe, which the shipped configs carry in the data arguments:
SGD with momentum 0.9 and weight decay 1e-4 on the weights but not the biases, learning rate 1e-2,
and a staircase exponential decay by tcl_decay_factor every tcl_decay_steps. The reference first
trains the segment classifier alone, with the feature extractor frozen, which tcl_mlr_init_steps
reproduces.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TCLArguments:
    comment_tcl_args: str = field(
        default="TCL Baseline Arguments",
        metadata={"help": "A comment to add to the TCL arguments."},
    )
    tcl_input_type: str = field(
        default="mel",
        metadata={"help": "What each frame holds: 'mel' takes the log-mel features extracted per "
                          "frame at preprocessing, 'waveform' takes the frame samples themselves. "
                          "TCL reads a data point as a plain feature vector, which mel bands suit, "
                          "so 'mel' is the default."},
    )
    tcl_n_mels: int = field(
        default=80,
        metadata={"help": "Number of mel bands, for the 'mel' input type."},
    )
    tcl_mel_norm: str = field(
        default="global",
        metadata={"help": "Mel normalization applied in the collator, for the 'mel' input type."},
    )
    tcl_pool_mel_bins: bool = field(
        default=False,
        metadata={"help": "Average the time bins preprocessing extracted inside each frame, leaving "
                          "one value per mel channel."},
    )
    tcl_hidden_nodes: List[int] = field(
        default_factory=lambda: [40, 40, 40, 40, 20],
        metadata={"help": "Width of each hidden layer of the feature extractor. The last entry is "
                          "the width of the representation, since the last hidden layer is what is "
                          "kept and evaluated."},
    )
    tcl_maxout_k: int = field(
        default=2,
        metadata={"help": "Number of affine feature maps each maxout unit takes the maximum over. "
                          "Every hidden layer but the last one is a maxout layer, so it is built "
                          "tcl_maxout_k times wider and then reduced."},
    )
    tcl_feature_nonlinearity: str = field(
        default="abs",
        metadata={"help": "Nonlinearity of the last hidden layer, which is the representation. "
                          "'abs' is the reference's, and is what makes the features recover the "
                          "independent components up to a point-wise transform."},
    )
    tcl_segment_duration_in_seconds: float = field(
        default=0.25,
        metadata={"help": "Duration of one time segment of the pretext task, from which the number "
                          "of frames per segment follows through the frame stride. Segments have a "
                          "fixed duration rather than a fixed count, so a frame's label is its "
                          "absolute position in the utterance and the same label means the same "
                          "elapsed time in every utterance. The number of classes follows from the "
                          "longest utterance, so utterances shorter than that leave the later "
                          "classes unpopulated. Between 0.1 and 0.5 seconds is the sensible range."},
    )
    tcl_pca_components: int = field(
        default=None,
        metadata={"help": "Components kept by the PCA whitening the reference applies before the "
                          "network. None falls back to the last entry of tcl_hidden_nodes, keeping "
                          "the reference's relationship, where the count of components, of sources "
                          "and of feature units are the same number. Keeping every dimension is not "
                          "generally possible here: a frame's features are redundant, so the small "
                          "eigenvalues trip the same guard the reference raises. The transform is "
                          "fitted on the training split alone and stored with the model, so the "
                          "evaluation reads the same one."},
    )
    tcl_pca_fit_fraction: float = field(
        default=1.0,
        metadata={"help": "Fraction of the training frames the whitening is fitted on, sampled with "
                          "tcl_pca_seed. The reference fits on all of them, which is the default; "
                          "lower it for a corpus where that pass is expensive."},
    )
    tcl_pca_seed: int = field(
        default=42,
        metadata={"help": "Seed of the generator that samples the frames the whitening is fitted on."},
    )
    tcl_optimizer: str = field(
        default="sgd",
        metadata={"help": "Optimizer family, taken from the reference implementation: 'sgd' is "
                          "TCL's own momentum optimizer, with tcl_sgd_momentum and the learning "
                          "rate of the data arguments. 'adamw' uses the optimizer the rest of the "
                          "project trains with, and is an ablation rather than the baseline."},
    )
    tcl_sgd_momentum: float = field(
        default=0.9,
        metadata={"help": "Momentum of the SGD optimizer, as the reference implementation sets it."},
    )
    tcl_decay_steps: int = field(
        default=500000,
        metadata={"help": "Steps between two staircase drops of the learning rate, matching the "
                          "reference's exponential decay with staircase set. Applied only when the "
                          "config asks for the 'tcl_staircase' schedule."},
    )
    tcl_decay_factor: float = field(
        default=0.1,
        metadata={"help": "Factor the learning rate is multiplied by at every drop."},
    )
    tcl_mlr_init_steps: int = field(
        default=70000,
        metadata={"help": "Steps at the start of training during which only the segment classifier "
                          "learns and the feature extractor is held still, as the reference does in "
                          "its initialization run. 0 trains everything from the first step."},
    )
    tcl_transfer_reinit_classifier: bool = field(
        default=True,
        metadata={"help": "Fine-tuning only. The segment classifier has one class per segment of the "
                          "longest utterance, so a source and a target dataset of different duration "
                          "cannot share it. On, the feature extractor transfers and the classifier "
                          "starts over, which costs nothing because the evaluation reads the features "
                          "and discards the classifier. Off, the transfer stops instead."},
    )
    tcl_seq_pooling: str = field(
        default=None,
        metadata={"help": "How the frames of an utterance are pooled into one sequence-level "
                          "embedding, for the '*_seq' targets. None skips them. 'mean' averages "
                          "over the frames of the utterance, as DecVAE's SequenceAggregator does."},
    )
