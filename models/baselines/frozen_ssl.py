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

"""Frozen speech SSL encoders (wav2vec2, HuBERT, WavLM) used as representation baselines.

These consume the un-framed utterance waveform and emit one embedding per frame on the same grid
DecVAE uses: the conv stack of every one of these checkpoints has receptive field 400 and total
stride 320, which is the grid the labels are interpolated onto at preprocessing.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModel

"Checkpoints the baselines were set up with. All are pre-trained only, never ASR fine-tuned,"
"so the baseline stays unsupervised. Any other HuggingFace speech SSL checkpoint also works."
FROZEN_SSL_CHECKPOINTS = {
    "wav2vec2": "facebook/wav2vec2-base",
    "hubert": "facebook/hubert-base-ls960",
    "wavlm": "microsoft/wavlm-base-plus",
}


class FrozenSSLEncoder(nn.Module):
    """
    Wraps a frozen HuggingFace speech SSL encoder behind the interface the VAE evaluation path
    expects, so it can be dropped in as a representation_function.

    Args:
        hf_name (str): HuggingFace checkpoint id.
        layer (int): Which entry of hidden_states to read. -1 is the last transformer layer.
        use_attention_mask (bool): Whether to pass the sample-level attention mask to the backbone.
    """

    def __init__(self, hf_name: str, layer: int = -1, use_attention_mask: bool = True):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_name)
        self.hf_name = hf_name
        self.layer = layer
        self.use_attention_mask = use_attention_mask

        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        "The post-analysis collator samples negatives for the DecVAE contrastive loss and reads"
        "num_negatives off the model config. A frozen encoder never uses them, and HuBERT's config"
        "has no such field, so supply one for the collator to read"
        self.config = self.backbone.config
        if not hasattr(self.config, "num_negatives"):
            self.config.num_negatives = 100

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        return self.backbone._get_feat_extract_output_lengths(input_lengths)

    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        return self.backbone._get_feature_vector_attention_mask(feature_vector_length, attention_mask)

    def forward(self, input_values, attention_mask=None):
        """
        Args:
            input_values: (batch, samples) un-framed waveform, already normalized by the feature extractor.
            attention_mask: (batch, samples) sample-level mask, or None.
        Returns:
            A 1-tuple whose only entry is the (batch, frames, hidden) embedding.
        """
        if input_values.dim() != 2:
            raise ValueError(
                f"{type(self).__name__} expects an un-framed (batch, samples) waveform, got shape "
                f"{tuple(input_values.shape)}. It is read from input_seq_values, which the cache only "
                "holds when the decomposition ran with seq_decomp=true."
            )

        outputs = self.backbone(
            input_values,
            attention_mask=attention_mask if self.use_attention_mask else None,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[self.layer]

        expected = int(self._get_feat_extract_output_lengths(input_values.shape[-1]))
        if hidden.shape[1] != expected:
            raise ValueError(
                f"{self.hf_name} returned {hidden.shape[1]} frames but the label grid has {expected}. "
                "Frame alignment would be silently wrong."
            )
        return (hidden,)


def build_frozen_ssl(frozen_ssl_args):
    """
    Build a frozen SSL encoder from the frozen SSL arguments.

    Args:
        frozen_ssl_args (:class:`~args_configs.frozen_ssl_args.FrozenSSLArguments`)
    Returns:
        FrozenSSLEncoder
    """
    return FrozenSSLEncoder(
        hf_name=frozen_ssl_args.ssl_model_name_or_path,
        layer=frozen_ssl_args.ssl_hidden_layer,
        use_attention_mask=frozen_ssl_args.ssl_use_attention_mask,
    )
