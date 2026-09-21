#!/usr/bin/env python
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
"""
Data collation for the pre-trained time-series baselines (CoST, TF-C, TCL).

These methods read one utterance as a series of frames on the DecVAE grid. The collator is kept
method-agnostic: it pads, normalizes the mel features, selects the decomposition component and
returns the frame series with its frame-level mask. The augmentations and the losses are the
method's own and live in its model.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor

from feature_extraction import normalize_mel_spectrogram, rereference_mel_db

"Pre-training reads no labels, so everything else the preprocessing wrote is dropped here"
KEPT_COLUMNS = ("input_values", "attention_mask", "mel_spec_max")


def normalize_baseline_mel_batch(values, spec_max, mel_norm, n_mels, device):
    """
    Normalize mel features that were extracted per utterance at the preprocessing step. Each
    decomposition component is re-referenced against the batch dB peak and normalized on its own,
    before one of them is selected, which is what the VAE collators do.

    Args:
        values (torch.Tensor): (batch, components, frames, n_mels*bins) features
        spec_max (list): Per-utterance reference peaks stored at preprocessing, one per component
        mel_norm (str or None): Normalization method, or None to leave the dB values as they are
        n_mels (int): Number of mel bands
        device (str or torch.device): Device of the returned tensors
    Returns:
        torch.Tensor: The normalized features, same shape as the input
    """
    if not spec_max or not all(s is not None for s in spec_max):
        raise ValueError(
            "The cache holds no mel_spec_max, so it was not written with mel features. Point the "
            "cache file names at a '*_mel.arrow' cache, or set the input type to 'waveform'."
        )

    stored_spec_max = torch.tensor(np.asarray(spec_max, dtype=np.float32), dtype=values.dtype)
    feature_length = values.shape[-1]

    components = []
    for o in range(values.shape[1]):
        component = rereference_mel_db(values[:, o, ...], stored_spec_max[:, o])
        if mel_norm is not None:
            component = normalize_mel_spectrogram(
                component,
                normalize=mel_norm,
                feature_length=feature_length,
                device=device,
                n_mels=n_mels,
            )
        components.append(component)

    return torch.stack(components, dim=1)


@dataclass
class DataCollatorForBaselinePretraining_NoFeatureExtraction:
    """
    Data collator to be used when the features have already been extracted at the preprocessing step
    by prepare_extract_features_pretraining_dataset. Pads the inputs, normalizes the mel
    features here where the whole batch is visible, selects the decomposition component and derives
    the frame-level attention mask.

    Args:
        model (:obj:`torch.nn.Module`):
            The baseline being pre-trained. The collator needs its ``_get_feature_vector_attention_mask``
            to derive the frame-level mask.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        input_type (:obj:`str`): The input type of the baseline - 'mel' or 'waveform'.
        n_mels (:obj:`int`): The number of mel bands, for the 'mel' input type.
        mel_norm (:obj:`str`, `optional`): The mel normalization, for the 'mel' input type.
        pool_mel_bins (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Average the time bins extracted inside each frame, leaving one value per mel channel.
        component (:obj:`int`, `optional`, defaults to :obj:`0`):
            Which decomposition component feeds the encoder. 0 is the original signal.
        padding (:obj:`bool` or :obj:`str`, `optional`, defaults to :obj:`"longest"`):
            Select a strategy to pad the returned sequences.
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
    """

    model: torch.nn.Module
    feature_extractor: Wav2Vec2FeatureExtractor
    input_type: str
    n_mels: int
    mel_norm: Optional[str] = None
    pool_mel_bins: bool = False
    component: int = 0
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        features = [{k: v for k, v in feature.items() if k in KEPT_COLUMNS} for feature in features]
        spec_max = [feature.pop("mel_spec_max", None) for feature in features]

        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device

        "DecVAE's preprocessing stores a leading singleton axis"
        if batch["input_values"].dim() == 5 and batch["input_values"].shape[1] == 1:
            batch["input_values"] = batch["input_values"].squeeze(1)

        if batch["input_values"].dim() != 4:
            raise ValueError(
                f"Expected (batch, components, frames, features) inputs, got shape "
                f"{tuple(batch['input_values'].shape)}. The baselines read the framed decomposition, "
                "which the cache only holds when it ran with frame_decomp=true."
            )

        values = batch["input_values"]
        if self.input_type.startswith("mel"):
            values = normalize_baseline_mel_batch(values, spec_max, self.mel_norm, self.n_mels, device)

        if self.component >= values.shape[1]:
            raise ValueError(
                f"baseline_component {self.component} was asked for, but the decomposition wrote "
                f"only {values.shape[1]} components."
            )
        values = values[:, self.component, ...]

        if self.input_type.startswith("mel") and self.pool_mel_bins:
            if values.shape[-1] % self.n_mels != 0:
                raise ValueError(
                    f"Cannot pool the mel bins: {values.shape[-1]} features do not divide into "
                    f"{self.n_mels} bands. The cache was written with a different band count."
                )
            values = values.reshape(*values.shape[:-1], self.n_mels, -1).mean(dim=-1)

        batch["input_values"] = values
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)

        "Frame-level mask, on the same grid the frame labels sit on"
        mask_indices_seq_length = int(values.shape[-2])
        batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
            mask_indices_seq_length, batch["attention_mask"]
        )

        return batch
