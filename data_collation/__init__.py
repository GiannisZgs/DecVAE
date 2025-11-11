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
Data collation classes for DecVAE and VAE models.

This module contains data collator classes that handle batch preparation, 
dynamic padding, feature extraction, label alignment, and other preprocessing steps required for training
DecVAE and VAE models.
"""

from .decvae_collators import DataCollatorForDecVAEPretraining, DataCollatorForDecVAE_SSL_FineTuning, DataCollatorForDecVAELatentPostAnalysis, DataCollatorForDecVAELatentDisentanglement, DataCollatorForDecVAELatentTraversals, DataCollatorForDecVAELatentVisualization
from .vae_collators import DataCollatorForVAE1DPreTraining, DataCollatorForVAE1D_SSL_FineTuning, DataCollatorForVAE1DLatentPostAnalysis, DataCollatorForVAE1DLatentTraversals
from .input_vis_collators import DataCollatorForInputVisualization

__all__ = [
    "DataCollatorForDecVAEPretraining",
    "DataCollatorForDecVAE_SSL_FineTuning",
    "DataCollatorForDecVAELatentPostAnalysis",
    "DataCollatorForDecVAELatentDisentanglement",
    "DataCollatorForDecVAELatentTraversals",
    "DataCollatorForDecVAELatentVisualization",
    "DataCollatorForVAE1DPreTraining",
    "DataCollatorForVAE1D_SSL_FineTuning",
    "DataCollatorForVAE1DLatentPostAnalysis",
    "DataCollatorForVAE1DLatentTraversals",
    "DataCollatorForInputVisualization"
]
