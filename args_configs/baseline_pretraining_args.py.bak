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

"""Arguments shared by the pre-trained time-series baselines (CoST, TF-C, TCL, CPC).

Each method carries its own argument class, since their front-ends and hyperparameters differ.
This group holds only what selects between them and what the shared pre-training script needs
before the method is known.

Every method is trained with the optimizer, learning rate, schedule shape, batch size and
architecture hyperparameters of its own paper or reference implementation, not with the ones the
rest of the project uses. The optimizer family lives in the method's argument group, since it
varies; the learning rate, weight decay, schedule and batch size are carried in the data arguments
of each config. Read off the reference sources:

    CoST   SGD, momentum 0.9, weight decay 1e-4, lr 1e-3, half-cosine decay to zero, no warmup,
           batch 128. The batch size is the one every script under CoST/scripts uses - train.py's
           argparse default of 8 was never used for a published run.
    TF-C   Adam, betas (0.9, 0.99), weight decay 3e-4, lr 3e-4, constant learning rate, batch 128.
           trainer.py builds a ReduceLROnPlateau but steps it only in the fine-tuning branch, so
           pre-training never decays.
    TCL    SGD with momentum 0.9, lr 1e-2, staircase exponential decay by 0.1 every 5e5 steps over
           7e5 steps, batch 512, weight decay 1e-4. The reference first trains the logistic
           regression head alone for 7e4 steps, decaying every 5e4, before training end to end.
    CPC    Adam, betas (0.9, 0.999), epsilon 1e-8, no weight decay, lr 2e-4, constant learning
           rate, batch 8. CPC has no single reference, so this is read off both the ones followed:
           CPC_audio's cpc_default_config.py, whose schedulerStep of -1 disables its step decay and
           whose batchSizeGPU is 8, and contrastive-predictive-coding's config/audio, which builds
           no scheduler and also batches 8.

Where the reference and DecVAE disagree about an *evaluation* metric, DecVAE still wins, so the
comparison stays fair - that applies to the evaluation scripts, not to the recipes above.
"""

from dataclasses import dataclass, field


@dataclass
class BaselinePretrainingArguments:
    comment_baseline_args: str = field(
        default="Baseline Pre-Training Arguments",
        metadata={"help": "A comment to add to the baseline pre-training arguments."},
    )
    baseline_method: str = field(
        default=None,
        metadata={"help": "Which baseline to pre-train: 'cost', 'tfc', 'tcl' or 'cpc'. The method's "
                          "own argument group is read, the other groups are ignored."},
    )
    baseline_component: int = field(
        default=0,
        metadata={"help": "Which decomposition component feeds the encoder. 0 is the original "
                          "signal, which is what the VAE baselines train on, so the baselines stay "
                          "comparable. The other components are the OCs, in the order the "
                          "decomposition wrote them."},
    )
