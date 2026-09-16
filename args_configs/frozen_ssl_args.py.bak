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

"""Arguments for the frozen speech SSL baselines (wav2vec2, HuBERT, WavLM)."""

from dataclasses import dataclass, field


@dataclass
class FrozenSSLArguments:
    comment_frozen_ssl_args: str = field(
        default="Frozen SSL Baseline Arguments",
        metadata={"help": "A comment to add to the frozen SSL arguments."},
    )
    ssl_model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "HuggingFace checkpoint of the frozen SSL encoder to evaluate. Use a pre-trained "
                          "only checkpoint, never an ASR fine-tuned one, or the baseline stops being unsupervised."},
    )
    ssl_hidden_layer: int = field(
        default=-1,
        metadata={"help": "Which entry of hidden_states to read as the representation. -1 is the last transformer layer."},
    )
    ssl_use_attention_mask: bool = field(
        default=True,
        metadata={"help": "Whether to pass the sample-level attention mask to the frozen encoder."},
    )
    ssl_pca_components: int = field(
        default=None,
        metadata={"help": "Number of PCA components to reduce the collected embeddings to. None or 0 keeps "
                          "the native encoder dimensionality. Fitted on the train split, applied to eval and test."},
    )
    ssl_pca_fit_fraction: float = field(
        default=0.1,
        metadata={"help": "Fraction of the collected training frames sampled to fit the PCA. The encoders emit "
                          "768 dimensions, so fitting on every frame is needlessly expensive."},
    )
    ssl_pca_seed: int = field(
        default=42,
        metadata={"help": "Seed of the generator that samples the frames the PCA is fitted on."},
    )
