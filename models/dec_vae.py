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
🤗 DecVAE: Variational Decomposition Autoencoder Model.

This module implements Variational Decomposition Autoencoding by building on 
the 🤗 Wav2Vec2-encoder architecture. It extends the Wav2Vec2 model to include 
VAE functionality with latent space decomposition modeling for disentangled 
representation learning. The model supports two different time scales with 
short-term and long-term branches and utilities for feature extraction, projection, 
and decomposition.

The model architecture allows dual branched latent spaces, various normalization techniques,
and flexible component configurations through configuration options.

Adapted from the FairSeq and HuggingFace Transformers Wav2Vec2 implementation in accordance to the Apache License 2.0.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from .decomposition_masking import CustomLayerNorm, CustomBatchNorm
from safetensors.torch import load_file
from transformers.activations import ACT2FN
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Adapter,
    Wav2Vec2PositionalConvEmbedding,
    Wav2Vec2EncoderLayerStableLayerNorm,
    Wav2Vec2Encoder,
    Wav2Vec2GroupNormConvLayer,
    Wav2Vec2LayerNormConvLayer,
    Wav2Vec2NoLayerNormConvLayer,
    _compute_mask_indices
)
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from config_files.configuration_decVAE import DecVAEConfig
from itertools import combinations


logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "DecVAEConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

@dataclass
class DecVAEForPreTrainingOutput(ModelOutput):
    """
    Output type of [`DecVAEForPreTraining`].
    
    This class defines the structure of the output from the DecVAEForPreTraining model,
    encapsulating loss values, hidden states, and attention values from the model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss=True`):
            Total loss for the model.
        loss_z (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss value for the Z branch.
        loss_s (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss value for the S branch.
        hidden_states_z (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from the Z branch of the model.
        hidden_states_s (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from the S branch of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights from the model.
        unmasked_frames_features_z (`torch.FloatTensor`):
            Unmasked frame features from the Z branch.
        unmasked_frames_features_s (`torch.FloatTensor`):
            Unmasked frame features from the S branch.
        decomposition_features_z (`torch.FloatTensor`):
            Decomposition features from the Z branch.
        decomposition_features_s (`torch.FloatTensor`):
            Decomposition features from the S branch.
        used_decomposition_features_z (`torch.FloatTensor`):
            Used decomposition features from the Z branch.
        used_decomposition_features_s (`torch.FloatTensor`):
            Used decomposition features from the S branch.
        used_unmasked_features_z (`torch.FloatTensor`):
            Used unmasked features from the Z branch.
        used_unmasked_features_s (`torch.FloatTensor`):
            Used unmasked features from the S branch.
        used_projected_components_z (`torch.FloatTensor`):
            Used projected components from the Z branch.
        used_projected_components_s (`torch.FloatTensor`):
            Used projected components from the S branch.
        mu_originals_z (`torch.FloatTensor`):
            Mean vectors for original features from the Z branch.
        mu_originals_s (`torch.FloatTensor`):
            Mean vectors for original features from the S branch.
        logvar_originals_z (`torch.FloatTensor`):
            Log variance vectors for original features from the Z branch.
        logvar_originals_s (`torch.FloatTensor`):
            Log variance vectors for original features from the S branch.
        mu_projections_z (`torch.FloatTensor`):
            Mean vectors for projections from the Z branch.
        mu_projections_s (`torch.FloatTensor`):
            Mean vectors for projections from the S branch.
        logvar_projections_z (`torch.FloatTensor`):
            Log variance vectors for projections from the Z branch.
        logvar_projections_s (`torch.FloatTensor`):
            Log variance vectors for projections from the S branch.
        mu_components_z (`torch.FloatTensor`):
            Mean vectors for components from the Z branch.
        mu_components_s (`torch.FloatTensor`):
            Mean vectors for components from the S branch.
        logvar_components_z (`torch.FloatTensor`):
            Log variance vectors for components from the Z branch.
        logvar_components_s (`torch.FloatTensor`):
            Log variance vectors for components from the S branch.
        used_indices_z (`list`):
            Indices of used features from the Z branch.
        used_indices_s (`list`):
            Indices of used features from the S branch.
        decomposition_loss_z (`torch.FloatTensor`):
            Decomposition loss from the Z branch.
        decomposition_loss_s (`torch.FloatTensor`):
            Decomposition loss from the S branch.
        prior_loss_z (`torch.FloatTensor`):
            Prior regularization loss from the Z branch.
        prior_loss_s (`torch.FloatTensor`):
            Prior regularization loss from the S branch.
        divergence_dict_z (`dict`):
            Dictionary of divergence values from the Z branch.
        divergence_dict_s (`dict`):
            Dictionary of divergence values from the S branch.
        div_pos_z (`torch.FloatTensor`):
            Positive divergence from the Z branch.
        div_pos_s (`torch.FloatTensor`):
            Positive divergence from the S branch.
        div_neg_z (`torch.FloatTensor`):
            Negative divergence from the Z branch.
        div_neg_s (`torch.FloatTensor`):
            Negative divergence from the S branch.
        ce_pos_z (`torch.FloatTensor`):
            Positive cross entropy from the Z branch.
        ce_pos_s (`torch.FloatTensor`):
            Positive cross entropy from the S branch.
        ce_neg_z (`torch.FloatTensor`):
            Negative cross entropy from the Z branch.
        ce_neg_s (`torch.FloatTensor`):
            Negative cross entropy from the S branch.
        avg_correlogram_z (`torch.FloatTensor`):
            Average correlogram from the Z branch.
        avg_correlogram_s (`torch.FloatTensor`):
            Average correlogram from the S branch.
        avg_correlogram_td_z (`torch.FloatTensor`):
            Average time-domain correlogram from the Z branch.
        avg_correlogram_td_s (`torch.FloatTensor`):
            Average time-domain correlogram from the S branch.
        orthogonality_dict_z (`dict`):
            Dictionary of orthogonality metrics from the Z branch.
        orthogonality_dict_s (`dict`):
            Dictionary of orthogonality metrics from the S branch.
        orthogonality_dict_td_z (`dict`):
            Dictionary of time-domain orthogonality metrics from the Z branch.
        orthogonality_dict_td_s (`dict`):
            Dictionary of time-domain orthogonality metrics from the S branch.
        N_div_pos_z (`torch.LongTensor`):
            Number of positive divergence samples from the Z branch.
        N_div_pos_s (`torch.LongTensor`):
            Number of positive divergence samples from the S branch.
        N_div_neg_z (`torch.LongTensor`):
            Number of negative divergence samples from the Z branch.
        N_div_neg_s (`torch.LongTensor`):
            Number of negative divergence samples from the S branch.
        mask_time_indices (`torch.BoolTensor`):
            Indices of masked time frames.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_z: Optional[torch.FloatTensor] = None
    loss_s: Optional[torch.FloatTensor] = None
    hidden_states_z: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_s: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    unmasked_frames_features_z: torch.FloatTensor = None
    unmasked_frames_features_s: torch.FloatTensor = None
    decomposition_features_z: torch.FloatTensor = None
    decomposition_features_s: torch.FloatTensor = None
    used_decomposition_features_z: torch.FloatTensor = None
    used_decomposition_features_s: torch.FloatTensor = None
    used_unmasked_features_z: torch.FloatTensor = None
    used_unmasked_features_s: torch.FloatTensor = None
    used_projected_components_z: torch.FloatTensor = None
    used_projected_components_s: torch.FloatTensor = None
    mu_originals_z: torch.FloatTensor = None
    mu_originals_s: torch.FloatTensor = None
    logvar_originals_z: torch.FloatTensor = None
    logvar_originals_s: torch.FloatTensor = None
    mu_projections_z: torch.FloatTensor = None
    mu_projections_s: torch.FloatTensor = None
    logvar_projections_z: torch.FloatTensor = None
    logvar_projections_s: torch.FloatTensor = None
    mu_components_z: torch.FloatTensor = None
    mu_components_s: torch.FloatTensor = None
    logvar_components_z: torch.FloatTensor = None
    logvar_components_s: torch.FloatTensor = None
    used_indices_z: Optional[list] = None
    used_indices_s: Optional[list] = None
    decomposition_loss_z: Optional[torch.FloatTensor] = None
    decomposition_loss_s: Optional[torch.FloatTensor] = None
    prior_loss_z: Optional[torch.FloatTensor] = None
    prior_loss_s: Optional[torch.FloatTensor] = None
    divergence_dict_z: Optional[dict] = None
    divergence_dict_s: Optional[dict] = None
    div_pos_z: Optional[torch.FloatTensor] = None
    div_pos_s: Optional[torch.FloatTensor] = None
    div_neg_z: Optional[torch.FloatTensor] = None
    div_neg_s: Optional[torch.FloatTensor] = None
    ce_pos_z: Optional[torch.FloatTensor] = None
    ce_pos_s: Optional[torch.FloatTensor] = None
    ce_neg_z: Optional[torch.FloatTensor] = None
    ce_neg_s: Optional[torch.FloatTensor] = None
    avg_correlogram_z: Optional[torch.FloatTensor] = None
    avg_correlogram_s: Optional[torch.FloatTensor] = None
    avg_correlogram_td_z: Optional[torch.FloatTensor] = None
    avg_correlogram_td_s: Optional[torch.FloatTensor] = None
    orthogonality_dict_z: Optional[dict] = None
    orthogonality_dict_s: Optional[dict] = None
    orthogonality_dict_td_z: Optional[dict] = None
    orthogonality_dict_td_s: Optional[dict] = None
    N_div_pos_z: Optional[torch.LongTensor] = None
    N_div_pos_s: Optional[torch.LongTensor] = None
    N_div_neg_z: Optional[torch.LongTensor] = None
    N_div_neg_s: Optional[torch.LongTensor] = None
    mask_time_indices: Optional[torch.BoolTensor] = None


@dataclass
class DecVAEBaseModelOutput(ModelOutput):
    """
    Base class for the output of DecVAE models.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        extract_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, conv_dim[-1])`):
            Sequence of extracted feature vectors of the last convolutional layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        reconstruction_NRMSEs (`torch.FloatTensor`, *optional*):
            Normalized Root Mean Square Error values for the reconstruction of the input signals.
            Used to evaluate the quality of the decomposition/reconstruction process.
        ortho_dict (`dict`, *optional*):
            Dictionary containing orthogonality metrics for the decomposed components.
            Useful for assessing the independence between different decomposed components.
        avg_correlogram (`torch.FloatTensor`, *optional*):
            Average correlogram representing the correlation structure between decomposed components.
            Higher values indicate stronger correlations between components.
    """

    last_hidden_state: torch.FloatTensor = None
    extract_features: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    reconstruction_NRMSEs: Optional[torch.FloatTensor] = None
    ortho_dict: Optional[dict] = None
    avg_correlogram: Optional[torch.FloatTensor] = None


class DecVAELayerNormConvLayer(nn.Module):
    """
    Convolutional layer with layer normalization and activation function for the DecVAE model.
    
    This class implements a 1D convolutional layer followed by normalization (either layer norm
    or batch norm based on configuration) and an activation function. It's used as a building 
    block in the feature extraction layers of the DecVAE model.

    It slightly modifies the way convolutions are performed to ensure that masked frames are not processed. 
    
    Args:
        config (`DecVAEConfig`): 
            The configuration object with parameters for the layer.
        layer_id (`int`, defaults to 0): 
            The identifier for the layer, used to access appropriate configuration values.
    """
    
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim_z[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim_z[layer_id]
        self.layer_id = layer_id
        self.config = config

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        if config.stride == config.receptive_field:
            if self.config.feat_extract_norm == "layer":
                self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
            elif self.config.feat_extract_norm == "batch":
                self.layer_norm = nn.BatchNorm1d(self.out_conv_dim, elementwise_affine=True)
        else:
            self.orig_layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
            if self.config.feat_extract_norm == "layer":
                self.layer_norm = CustomLayerNorm(self.out_conv_dim, config, elementwise_affine=True,
                        device = self.orig_layer_norm.weight.device,dtype=self.orig_layer_norm.weight.dtype)
            elif self.config.feat_extract_norm == "batch":
                self.layer_norm = CustomBatchNorm(self.out_conv_dim, config, affine=True,
                        device = self.orig_layer_norm.weight.device,dtype=self.orig_layer_norm.weight.dtype)
        self.activation = ACT2FN[config.feat_extract_activation]

        if config.add_skip_connection:# and layer_id < 6:
            # Residual connection: match input-output dimensions if needed
            #if self.in_conv_dim != self.out_conv_dim:
            self.residual_connection = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size=1,stride = config.conv_stride[layer_id])
            #else:
            #    self.residual_connection = nn.Identity()


    def forward(self, hidden_states, mask_time_indices=None):
        """
        Forward pass for the convolutional layer with normalization and activation.
        
        This method applies a 1D convolution followed by normalization and activation to the input.
        It supports masking for specific time indices and implements skip connections if configured.
        
        Args:
            hidden_states (`torch.Tensor`): 
                Input tensor of shape [batch_size, sequence_length, in_conv_dim].
            mask_time_indices (`torch.Tensor` or `numpy.ndarray`, *optional*): 
                Boolean mask indicating which time indices to process. If None, all indices are processed.
                
        Returns:
            `torch.Tensor`: Processed features after convolution, normalization, and activation.
        """
        if mask_time_indices is not None: 
            if type(mask_time_indices) == np.ndarray:
                mask_time_indices = torch.from_numpy(mask_time_indices)
            mask_time_indices = mask_time_indices.to(hidden_states.device, dtype=torch.bool)
        else:
            # In case of original frames
            mask_time_indices = torch.ones((hidden_states.shape[0], hidden_states.shape[1]), dtype=torch.bool)
            mask_time_indices = mask_time_indices.to(hidden_states.device, dtype=torch.bool)

        output_length = int(np.floor((hidden_states.shape[-1] - self.conv.kernel_size[0]) / self.conv.stride[0]) + 1)
        conv_features = torch.zeros((hidden_states.shape[0], hidden_states.shape[1], self.out_conv_dim, output_length), device=hidden_states.device, dtype=hidden_states.dtype)

        # Create a mask for the segments to be convolved
        if self.layer_id == 0:
            mask = mask_time_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
            # Apply the mask to the hidden states
            masked_hidden_states = hidden_states.masked_select(mask).view(-1, hidden_states.shape[-1]).unsqueeze(1)
            # Perform convolution on the masked segments - Unsqueeze for input channels = 1
            if (masked_hidden_states != masked_hidden_states).any():
                print("Nan in convolution input")
            convolved_segments = self.conv(masked_hidden_states) #.squeeze(2)
        else:
            mask = mask_time_indices.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,hidden_states.shape[-2],hidden_states.shape[-1])
            masked_hidden_states = hidden_states.masked_select(mask).view(-1,hidden_states.shape[-2],hidden_states.shape[-1])
            if (masked_hidden_states != masked_hidden_states).any():
                print("Nan in convolution input")
            convolved_segments = self.conv(masked_hidden_states) #.squeeze(2)

        # Reshape the convolved segments to match the conv_features shape
        conv_features[mask_time_indices] = convolved_segments.view(-1, self.out_conv_dim, output_length)

        conv_features = conv_features.transpose(-2, -1)
        
        if (conv_features != conv_features).any():
            print("Nan in layer norm input")

        conv_features = self.layer_norm(conv_features,mask_time_indices)
    
        if (conv_features != conv_features).any():
            print("Nan inside layer norm")
        conv_features = conv_features.transpose(-2, -1)
        
        if hasattr(self,"residual_connection"):
            # Apply the residual connection (skip connection)
            residual = self.residual_connection(masked_hidden_states)
            if (residual != residual).any():
                print("Nan in skip connection")
            if residual.shape[-1] != conv_features.shape[-1]:
                #Avg pool last two features of each frame to match dimensions
                residual[...,-2] = (residual[...,-2] + residual[..., -1]) / 2
                residual = residual[...,:-1]
            # Reshape back to (B, S, C, L)    
            conv_features[mask_time_indices] += residual.view(-1, self.out_conv_dim, output_length)

        conv_features = self.activation(conv_features)

        return conv_features


#This does not exist in original modeling_wav2vec2 classes
class DecVAEInstanceNormConvLayer(nn.Module):
    """
    Convolutional layer with instance normalization and activation function for the DecVAE model.
    
    This layer implements a 1D convolution followed by instance normalization (instead of layer norm)
    and an activation function.
    
    Args:
        config (`DecVAEConfig`): 
            The configuration object with parameters for the layer.
        layer_id (`int`, defaults to 0): 
            The identifier for the layer, used to access appropriate configuration values.
    """
    
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim_z[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim_z[layer_id]

        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        #self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.instance_norm = nn.InstanceNorm1d(self.out_conv_dim, affine=True)
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        """
        Forward pass for the convolutional layer with instance normalization and activation.
        
        Args:
            hidden_states (`torch.Tensor`): 
                Input tensor of shape [batch_size, in_conv_dim, sequence_length].
                
        Returns:
            `torch.Tensor`: Processed features after convolution, instance normalization, and activation.
        """
        hidden_states = self.conv(hidden_states)

        #hidden_states = hidden_states.transpose(-2, -1)
        #hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.instance_norm(hidden_states)
        #hidden_states = hidden_states.transpose(-2, -1)

        hidden_states = self.activation(hidden_states)
        return hidden_states


class DecVAEFeatureEncoder(nn.Module):
    """
    Uses the custom convolutional layers with layer normalization and activation function for the DecVAE model.

    This class implements a stack of 1D convolutional layers followed by normalization (either layer norm
    or batch norm based on configuration) and an activation function. 

    The convolutions are performed in a slightly different way to ensure that masked frames are not processed. 
    (see DecVAELayerNormConvLayer).
    """

    def __init__(self, config):
        super().__init__()
        
        if config.feat_extract_norm == "group":
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer" or config.feat_extract_norm == "batch":
            if config.stride == config.receptive_field:
                conv_layers = [
                    Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
                ]
            else:
                conv_layers = [
                    DecVAELayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
                ]

        elif config.feat_extract_norm == "instance":
            conv_layers = [
                DecVAEInstanceNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]

        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer','batch]"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.config = config
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values,mask_time_indices=None):
        if self.config.stride == self.config.receptive_field:
            hidden_states = input_values[:, :, None].float()
        else:
            hidden_states = input_values[:, None].float() 
        
        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training and hidden_states.requires_grad == False:
            hidden_states.requires_grad = True

        levels = hidden_states.shape[0]
        
        for n in range(levels):
            if self.config.stride == self.config.receptive_field:
                decomp_level = hidden_states[n,:,0,...][:,None]
            else:
                decomp_level = hidden_states[n,0,...]
            
            for c,conv_layer in enumerate(self.conv_layers):
                if n > 0:                   
                    if self._requires_grad and self.gradient_checkpointing and self.training:
                        if self.config.stride == self.config.receptive_field:
                            decomp_level = self._gradient_checkpointing_func(
                                conv_layer.__call__,
                                decomp_level,
                            )
                        else:
                            decomp_level = self._gradient_checkpointing_func(
                                conv_layer.__call__,
                                decomp_level,
                                mask_time_indices,
                            )
                    else:
                        if self.config.stride == self.config.receptive_field:
                            decomp_level = conv_layer(decomp_level)
                        else:
                            decomp_level = conv_layer(decomp_level,mask_time_indices) #sf
                else:
                    if self._requires_grad and self.gradient_checkpointing and self.training:
                        if self.config.stride == self.config.receptive_field:
                            decomp_level = self._gradient_checkpointing_func(
                                conv_layer.__call__,
                                decomp_level,
                            )
                        else:
                            decomp_level = self._gradient_checkpointing_func(
                                conv_layer.__call__,
                                decomp_level,
                                None,
                            )
                    else:
                        #In case of original signal: mask_time_indices = None
                        if self.config.stride == self.config.receptive_field:
                            decomp_level = conv_layer(decomp_level)
                        else:
                            decomp_level = conv_layer(decomp_level,None) #was conv_layer(decomp_level,mask_time_indices) in previous version


            if 'new_hidden_states' not in locals():     
                new_hidden_states = torch.zeros((hidden_states.shape[0],decomp_level.shape[0],decomp_level.shape[1],decomp_level.shape[2],1),dtype=hidden_states.dtype,device=hidden_states.device)

            if self.config.stride == self.config.receptive_field:
                decomp_level = decomp_level.unsqueeze(-1)
            new_hidden_states[n,...] = decomp_level

        return new_hidden_states 


class DecVAEFeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.orig_layer_norm = nn.LayerNorm(config.conv_dim_z[-1], eps=config.layer_norm_eps)
        self.layer_norm = CustomLayerNorm(config.conv_dim_z[-1], config, elementwise_affine=True,
                        device = self.orig_layer_norm.weight.device,dtype=self.orig_layer_norm.weight.dtype)
        self.projection = nn.Linear(config.conv_dim_z[-1], config.hidden_size)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, hidden_states,mask_time_indices=None):
        norm_hidden_states = self.layer_norm(hidden_states,mask_time_indices)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


class Dec2VecEncoderStableLayerNorm(nn.Module):
    """
    Wav2Vec2-based Transformer Encoder consisting of *config.num_hidden_layers* layers. Each layer
    is a [`Wav2Vec2EncoderLayerStableLayerNorm`].
    Args:
        config: Wav2Vec2Config
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        if self.config.agg_norm == "layer":
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        elif self.config.agg_norm == "batch":
            self.layer_norm = nn.BatchNorm1d(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.encoder_dropout)
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # make sure padded tokens are not attended to
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0
            if self._use_flash_attention_2:
                # 2d mask is passed through the layers
                attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            else:
                # extend attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                # XXX: could optimize this like synced_gpus in generate_utils but not sure if it's worth the code complication
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if self.config.agg_norm == "batch":
            if hidden_states.shape[-2] != self.layer_norm.num_features:
                #Input to BatchNorm should be of size (N,C,L)
                hidden_states = hidden_states.transpose(-2, -1)
                hidden_states = self.layer_norm(hidden_states)
                hidden_states = hidden_states.transpose(-2, -1)
        else:
            hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )



DECVAE_START_DOCSTRING = r"""
    DecVAE is a custom model extending the Wav2Vec2 architecture with decomposition capabilities and
    variational auto-encoder components for disentangled representation learning.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`DecVAEConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DECVAE_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, components, sequence_length)`):
            Float values of input raw speech waveform or decomposed components. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, and optionally passing through a decomposition process.
        input_seq_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Sequence information that serves as additional input, used with dual-branched models.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

        mask_time_indices (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices to mask extracted features for contrastive loss. When in training mode, model learns to predict
            masked extracted features in *config.proj_codevector_dim* space.
        reconstruction_NRMSEs (`torch.FloatTensor`, *optional*):
            Normalized root mean square errors for reconstruction.
        avg_correlogram (`torch.FloatTensor`, *optional*):
            Average correlogram metrics for component analysis.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Dec2vec Model encoder outputting raw hidden-states without any specific head on top.",
    DECVAE_START_DOCSTRING,
)
class Dec2VecModel(Wav2Vec2PreTrainedModel):
    """
    Dec2Vec is the base or backbone encoder that implements the
    Variational Decomposition Autoencoder model class by extending the Wav2Vec2 architecture.

    This model enhances the Wav2Vec2 architecture with:
    1. Signal decomposition capabilities - breaking down speech into components
    2. Variational autoencoder mechanisms - enabling disentangled representation learning
    3. Multiple training modes - supporting different learning objectives
    
    Key components include:
    - Feature Extractors: Both standard and custom versions
    - Component-specific embeddings: For handling decomposed signal components
    
    This class handles the core decomposition encoding operations, while specialized heads like
    DecVAEForPreTraining implement specific objectives.

    Supports up to 15 oscillatory components (OCs) for decomposition.
    """
    
    def __init__(self, config: DecVAEConfig):
        """
        Initialize the DecVAE model.
        
        This constructor sets up the model architecture based on the provided configuration.
        
        For decomposition modes:
        - DecompositionModule for signal component separation
        - Custom DecVAEFeatureEncoder and DecVAEFeatureProjection
        - Component-specific embeddings based on NoC (Number of Components)
        
        Args:
            config (DecVAEConfig): Configuration object containing model parameters
        """
        super().__init__(config)
        self.config = config
        self.training_flag = "pretraining"

        #self.decomposition_module = DecompositionModule(self.config)
        self.custom_feature_extractor = DecVAEFeatureEncoder(config)
        self.custom_feature_projection = DecVAEFeatureProjection(config)
        if config.use_learnable_embed_OC1:
            self.comp_1_embed = nn.Parameter(torch.rand(config.hidden_size), requires_grad=False)  # nn.Parameter(torch.Tensor(config.hidden_size).uniform_())
        if config.use_learnable_embed_OC2:
            self.comp_2_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
        if config.use_learnable_embed_OC3_or_more:
            if config.NoC > 2:
                self.comp_3_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 3:
                self.comp_4_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 4:
                self.comp_5_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 5:
                self.comp_6_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 6:
                self.comp_7_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 7:
                self.comp_8_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 8:
                self.comp_9_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 9:
                self.comp_10_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 10:
                self.comp_11_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 11:
                self.comp_12_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 12:
                self.comp_13_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 13:
                self.comp_14_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 14:
                self.comp_15_embed = nn.Parameter(torch.rand(config.hidden_size),requires_grad=False)
            if config.NoC > 15:
                raise ValueError("DecVAE currently supports up to 15 components only.")

        if config.do_stable_layer_norm:
            self.encoder = Dec2VecEncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.custom_feature_extractor._freeze_parameters()


    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:       
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @add_start_docstrings_to_model_forward(DECVAE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=DecVAEBaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        reconstruction_NRMSEs: Optional[torch.Tensor] = None,
        avg_correlogram: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DecVAEBaseModelOutput]:
        """
        Main forward pass for the Dec2Vec encoder model.
        
        This method processes input audio features (or decomposed components) through the model's encoder
        architecture. 
        
        For decomposition-based approaches, it processes multiple input components separately and handles
        component masking, orthogonality metrics, and special processing for fine-tuning vs. pretraining.

        It also supports a Transformer encoder that is though not used in the current DecVAE implementation
        
        Returns:
            A DecVAEBaseModelOutput object or tuple containing:
            - last_hidden_state: Final encoder hidden states
            - extract_features: Features from the feature extraction layer
            - hidden_states: All hidden states (if output_hidden_states=True)
            - attentions: Attention weights (if output_attentions=True)
        """

        if (input_values != input_values).any():
            print("NaNs in input_values")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, levels, mask_indices_seq_length, frame_size = input_values.size()
        assert mask_indices_seq_length == mask_time_indices.size(1)
    
        #component indices = False when component was not found there
        component_indices = torch.ones_like(mask_time_indices).unsqueeze(0).expand(self.config.NoC,-1,-1).clone()    
        for b in range(batch_size):
            for f in range(mask_indices_seq_length):
                if self.config.remove_silence and (input_values[b,1:,f,:] == 0).sum() == input_values[b,1:,f,:].size(0)*input_values[b,1:,f,:].size(1):
                    #This frame is silent because all components are 0
                    #Some frames might have components but are set to false by the masking process
                    #So we are looking for frames == silent & True
                    mask_time_indices[b,f] = False
                for c in range(self.config.NoC):
                    if (input_values[b,c+1,f,:] == 0).sum() == input_values[b,c+1,f,:].size(0):
                        #This component was found empty
                        component_indices[c,b,f] = torch.tensor([0])    
        
        #Average correlogram over the batch dimension
        if avg_correlogram is not None:
            avg_correlogram = avg_correlogram.mean(dim=0)

            #Orthogonality dictionary
            for n in range(self.config.NoC):
                ortho_dict = {"avg_ortho_time_domain_{}_{}".format(i+1,j+1): avg_correlogram[i,j] for i in range(self.config.NoC) for j in range(self.config.NoC)}
        else:
            ortho_dict = None         

        input_values = input_values.transpose(0,1) 
        
        extract_features = self.custom_feature_extractor(input_values,mask_time_indices)
        if (extract_features != extract_features).any():
            print("NaNs in extract_features")
        extract_features = extract_features.squeeze(-1)
                     
        #extract_features: z
        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[2], attention_mask, add_adapter=False
            )
        
        if self.training_flag == "ft" and not self.training:
            #Only in eval mode of fine-tuning stage
            all_ones_mask = torch.ones((extract_features.shape[1],extract_features.shape[2]),dtype=torch.bool,device=mask_time_indices.device) 
            selected_hidden_states, extract_features = self.custom_feature_projection(extract_features.squeeze(0),all_ones_mask)        
            hidden_states = None
        else:
            #Linear projection for each decomposition level
            total_levels = extract_features.shape[0] #NoC + 1

            hidden_states = torch.zeros((extract_features.shape[0],extract_features.shape[1],extract_features.shape[2],
                    self.encoder.pos_conv_embed.conv.in_channels),dtype=extract_features.dtype,device=extract_features.device)
        
            #If a single decomposition level is chosen for the transformer encoder randomly, propagate that

            for i in range(total_levels):
                if i == 0:
                    all_ones_mask = torch.ones((extract_features.shape[1],extract_features.shape[2]),dtype=torch.bool,device=mask_time_indices.device) 
                    hidden_states[i,...], extract_features[i,...] = self.custom_feature_projection(extract_features[i,...],all_ones_mask)        
                else:
                    hidden_states[i,...], extract_features[i,...] = self.custom_feature_projection(extract_features[i,...],mask_time_indices)        
            if (extract_features != extract_features).any():
                print("NaNs in extract_features - line 2353")
            if (hidden_states != hidden_states).any():
                print("NaNs in hidden_states - line 2355")
            #hidden states: linear projection of z - transformer input        
            
            #Replace components that were found as 0 with the centroid of all batch components in the same level
            for n in range(self.config.NoC):
                if n == 0 and self.config.use_learnable_embed_OC1:
                    hidden_states[n+1][~component_indices[n]] = self.comp_1_embed.to(hidden_states.dtype).to(hidden_states.device)
                    #Find new centroid from this batch's non-zero components
                    self.comp_1_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 1 and self.config.use_learnable_embed_OC2:
                    hidden_states[n+1][~component_indices[n]] = self.comp_2_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_2_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 2 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_3_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_3_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 3 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_4_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_4_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 4 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_5_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_5_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 5 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_6_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_6_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 6 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_7_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_7_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 7 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_8_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_8_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 8 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_9_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_9_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 9 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_10_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_10_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 10 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_11_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_11_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 11 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_12_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_12_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 12 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_13_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_13_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 13 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_14_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_14_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)
                elif n == 14 and self.config.use_learnable_embed_OC3_or_more:
                    hidden_states[n+1][~component_indices[n]] = self.comp_15_embed.to(hidden_states.dtype).to(hidden_states.device)
                    self.comp_15_embed = nn.Parameter(torch.mean(hidden_states[n+1][component_indices[n]],dim = 0),requires_grad=False)



            #Decomposition loss needs components at the extract features stage
            #Transformer takes the original frames - In pretraining, these are not used anyway
            #In fine-tuning, they will be used for the CTC loss
            selected_hidden_states = hidden_states[0,...]

        encoder_outputs = self.encoder(
            selected_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        #context representations c - after the transformer encoder
        last_hidden_states = encoder_outputs[0]
        
        if self.adapter is not None:
            last_hidden_states = self.adapter(last_hidden_states)

        if not return_dict:
            return (last_hidden_states, extract_features) + encoder_outputs[1:]

        #last_hidden_states: c - after transformer encoder
        #extract_features: z before projection
        #hidden_states: z before transformer encoder
        return DecVAEBaseModelOutput(
            last_hidden_state=last_hidden_states,
            extract_features=extract_features,
            hidden_states=hidden_states,
            attentions=attention_mask,
            reconstruction_NRMSEs = reconstruction_NRMSEs,
            ortho_dict = ortho_dict,
            avg_correlogram = avg_correlogram,
        )


class LinearStack(nn.Module):
    """
    Fully-connected processing support.
    This fully-connected encoder that serves as the S-branch sequence-level variable encoder in multivariate DecVAE. In contrast to the 
    frame-level Z branch, S-branch does not require a Wav2vec2-style feature extractor as it does not keep the frame-level information.
    An aggregator (Sequence Aggregator) with pooling and attention capabilities
    is used at the end of the stack to aggregate features into a sequence-level representation.
    """
    def __init__(
        self,
        input_dim=3000,
        layer_sizes=[2048,1024,512,512,512,256,256],  # These will be used for linear layer dimensions
        activation="gelu",
        layer_norm_eps=1.0e-05,
        elementwise_affine=True
    ):
        super().__init__()
        
        self.linear_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        current_dim = input_dim
        
        for i, size in enumerate(layer_sizes):
            # Create a linear layer that transforms from current_dim to hidden_dim
            self.linear_layers.append(
                nn.Linear(
                    current_dim,
                    size,
                    bias=True
                )
            )
            self.norms.append(
                nn.LayerNorm(size, eps=layer_norm_eps, elementwise_affine=elementwise_affine)
            )
            self.activations.append(ACT2FN[activation])
            current_dim = size

    def forward(self, x):
        # x shape: [batch_size, components+1, sequence_length]
        # Transform to [components+1, batch_size, sequence_length]
        x = x.transpose(0, 1)
        
        # Initialize output tensor
        hidden_states = torch.zeros(
            (x.size(0), x.size(1), self.linear_layers[-1].out_features), 
            dtype=x.dtype, 
            device=x.device
        )
        
        # Process each level separately
        for l, level in enumerate(x):
            # Start with the current level
            features = level
            
            # Apply each linear layer, normalization, and activation
            for linear, norm, activation in zip(self.linear_layers, self.norms, self.activations):
                features = linear(features)
                features = norm(features)
                features = activation(features)
            
            # Store the processed features
            hidden_states[l, ...] = features.clone()
            
        return hidden_states


class ConvStack(nn.Module):
    """
    Convolutional encoder that serves as the S-branch sequence-level variable encoder in multivariate DecVAE. In contrast to the 
    frame-level Z branch, S-branch does not require a Wav2vec2-style feature extractor as it does not keep the frame-level information.
    An aggregator (Sequence Aggregator) with pooling and attention capabilities
    is used at the end of the stack to aggregate features into a sequence-level representation.
    """

    def __init__(
        self,
        input_dim =  1,
        hidden_dim = 512,
        kernel_sizes=[10, 5, 3, 3, 3],
        strides=[5, 4, 2, 2, 2],
        activation="gelu",
        layer_norm_eps= 1.0e-05,
        elementwise_affine=True
    ):
        super().__init__()
        
        self.conv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()
        current_dim = input_dim
        
        for i, (kernel, stride) in enumerate(zip(kernel_sizes, strides)):
            self.conv_layers.append(
                nn.Conv1d(
                    current_dim,
                    hidden_dim,
                    kernel,
                    stride=stride,
                    padding=0,
                    bias=True
                )
            )
            self.norms.append(
                nn.LayerNorm(hidden_dim, eps=layer_norm_eps, elementwise_affine=elementwise_affine)
            )
            self.activations.append(ACT2FN[activation])
            current_dim = hidden_dim

    def calculate_rf_stride(self,input_size, kernels, strides):
        size = input_size
        cum_stride = 1
        rf_size = 1
        
        for i, (k, s) in enumerate(zip(kernels, strides)):
            # Calculate output size
            output_size = (size - k) // s + 1
            
            # Calculate receptive field
            rf_size = rf_size + (k - 1) * cum_stride
            
            # Update cumulative stride
            cum_stride *= s            
            size = output_size
        
        return size, cum_stride, rf_size
    
    def forward(self, x):
        # x shape: [batch_size, components+1, sequence_length]
        x = x.transpose(0,1).unsqueeze(-2)
        output_size = self.calculate_rf_stride(x.size(-1), [conv.kernel_size[0] for conv in self.conv_layers], [conv.stride[0] for conv in self.conv_layers])[0]
        hidden_states = torch.zeros((x.size(0),x.size(1),self.conv_layers[-1].out_channels,output_size),dtype=x.dtype,device=x.device)
        for l,level in enumerate(x):
            for conv, norm, activation in zip(self.conv_layers, self.norms, self.activations):
                level = conv(level)
                level = level.transpose(1, 2)
                level = norm(level)
                level = level.transpose(1, 2)
                level = activation(level)            
            hidden_states[l,...] = level.clone()
        return hidden_states


class SequenceAggregator(nn.Module):
    def __init__(
        self, 
        hidden_dim,
        first_stage=None,     # None, "attention" or "lstm"
        second_stage="avg",   # "max" or "avg"
        num_heads=2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.first_stage = first_stage
        self.second_stage = second_stage
        
        if first_stage == "attention":
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)
        elif first_stage == "lstm":
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
            self.out_proj = nn.Linear(2*hidden_dim, hidden_dim)
            
        # Second stage aggregation (required)
        if second_stage == "max":
            self.second_aggregate = nn.AdaptiveMaxPool1d(1)
        elif second_stage == "avg":
            self.second_aggregate = nn.AdaptiveAvgPool1d(1)
                        
    def forward(self, x):
        # x shape: [batch_size, sequence_length, hidden_dim]
        
        if self.first_stage == "attention":
            query = self.query.expand(x.shape[0], -1, -1)
            attn_output, _ = self.attention(query, x, x)
            x = self.layer_norm(attn_output)
        elif self.first_stage == "lstm":
            output, (h_n, _) = self.lstm(x)
            h_n = torch.cat([h_n[0], h_n[1]], dim=-1)
            x = self.out_proj(h_n).unsqueeze(1)
            
        # Second stage: pooling
        x = x.transpose(1, 2)  # [batch_size, hidden_dim, sequence_length]
        x = self.second_aggregate(x)  # [batch_size, hidden_dim, 1]

            
        return x.squeeze(-1)  # [batch_size, hidden_dim]


@add_start_docstrings("""DecVAE Model with decomposition and VAE disentangled representation modeling capabilities.""", DECVAE_START_DOCSTRING)
class DecVAEForPreTraining(Wav2Vec2PreTrainedModel):
    """
    DecVAE model for self-supervised pretraining with advanced decomposition and variational autoencoder features.
    
    This model extends the base DecVAEModel with additional capabilities for pretraining:
    
    1. Dual-branched architecture:
       - Z branch: Processes content/phonetic information
       - S branch: Processes style/speaker/sequence information
    
    2. Variational components:
       - Mean and logvar projection layers for each branch
       - Component-specific priors for disentangled representation
       - Optional projection layers for orthogonal components
    
    The model supports multiple training objectives:
    - Decomposition loss (reconstruction and orthogonality) between original and component representations
    - Prior regularization (KL divergence from a Gaussian prior) for variational components
    
    This architecture is designed for learning disentangled time series representations
    that separate information within and across different time-scales (e.g. segment/sequence in audio).

    Supports up to 15 oscillatory components (OCs) for decomposition.
    """
    
    def __init__(self, config: DecVAEConfig):
        """
        Initialize the DecVAEForPreTraining model.
        
        Sets up a complex architecture with multiple components based on the configuration:
        
        1. Base encoder models:
           - dec2vec_z: Dec2Vec model for the z (content) branch based on the wav2vec2 architecture
           - encoder_s: Either convolutional or linear stack for the s (style) branch
        
        2. Projection networks:
           - project_z/project_s: Project features to the contrastive embedding space

           - project_components_z/project_components_s: For component projections
        
        3. Variational components (when use_prior_regularization=True):
           - mean_layer_*/logvar_layer_*: Variational parameter estimators for different branches
           - Component-specific variational layers for orthogonal components (OC1, OC2, etc.)
        
        4. Optional sequence aggregation for the s branch
        
        Args:
            config (DecVAEConfig): Configuration object containing model parameters
        """
        super().__init__(config)
        self.config = config
        "Initialize z latent branch"
        if config.dual_branched_latent or config.only_z_branch:
            self.wav2vec2_z = Dec2VecModel(config)
            self.dropout_z_features = nn.Dropout(config.hidden_dropout)
        "Initialize s latent branch"
        if config.dual_branched_latent or config.only_s_branch:
            if self.config.fc_or_conv_s == "conv":
                self.encoder_s = ConvStack(
                    hidden_dim=config.conv_dim_s[0],
                    kernel_sizes=config.conv_kernel,
                    strides=config.conv_stride,
                    activation = config.feat_extract_activation)
            elif self.config.fc_or_conv_s == "fc":
                self.encoder_s = LinearStack(
                    input_dim=config.fc_input_size,
                    layer_sizes=config.fc_kernels,  # Reusing the same config parameters
                    activation=config.feat_extract_activation
                )
            
            self.dropout_s_features = nn.Dropout(config.hidden_dropout)
            if config.use_first_agg or config.use_second_agg:
                if not config.use_first_agg:
                    config.first_agg = None
                if not config.use_second_agg:
                    config.second_agg = None
                self.aggregator_s = SequenceAggregator(hidden_dim=config.proj_codevector_dim_s, first_stage=config.first_agg, second_stage=config.second_agg, num_heads=config.attention_heads_s)

        if config.dual_branched_latent or config.only_z_branch:
            self.project_z = nn.Sequential(
                nn.Linear(config.z_proj_input_dim, config.proj_intermediate_dim),
                nn.Linear(config.proj_intermediate_dim, config.proj_codevector_dim_z),
                ACT2FN[config.feat_extract_activation]
            ) #nn.LayerNorm(config.proj_codevector_dim_z),
        if config.dual_branched_latent or config.only_s_branch:
            #self.project_s = nn.Linear(config.s_proj_input_dim, config.proj_codevector_dim_s)
            self.project_s = nn.Sequential(
                nn.Linear(config.s_proj_input_dim, config.proj_intermediate_dim),
                nn.Linear(config.proj_intermediate_dim, config.proj_codevector_dim_s),
                ACT2FN[config.feat_extract_activation]
            ) #nn.LayerNorm(config.proj_codevector_dim_s),
                
        if config.project_OCs:
            if config.dual_branched_latent or config.only_z_branch:
                self.project_components_z = nn.Sequential(
                    nn.Linear(config.NoC*config.proj_codevector_dim_z, config.proj_codevector_dim_z),
                    nn.Linear(config.proj_codevector_dim_z, config.proj_codevector_dim_z),
                    ACT2FN[config.feat_extract_activation]
                )
            if config.dual_branched_latent or config.only_s_branch:
                self.project_components_s = nn.Sequential(
                    nn.Linear(config.NoC_seq*config.proj_codevector_dim_s, config.proj_codevector_dim_s),
                    nn.Linear(config.proj_codevector_dim_s, config.proj_codevector_dim_s),
                    ACT2FN[config.feat_extract_activation]
                )

        if self.config.use_prior_regularization:
            if config.dual_branched_latent or config.only_z_branch:
                self.mean_layer_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
            if config.dual_branched_latent or config.only_s_branch:
                self.mean_layer_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)

            if config.dual_branched_latent or config.only_z_branch:
                "Component projection prior - z"
                self.mean_layer_proj_OCs_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_proj_OCs_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
            if config.dual_branched_latent or config.only_s_branch:
                "Component projection prior - s"
                self.mean_layer_proj_OCs_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_proj_OCs_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
            "Z latent variable branch - prior regularization layers"
            if config.dual_branched_latent or config.only_z_branch:
                "Component #1 prior - z"
                self.mean_layer_OC1_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_OC1_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #2 prior - z"
                self.mean_layer_OC2_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_OC2_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #3 prior - z"
                if self.config.NoC >= 3:
                    self.mean_layer_OC3_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC3_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #4 prior - z"
                if self.config.NoC >= 4:
                    self.mean_layer_OC4_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC4_z= nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #5 prior - z"
                if self.config.NoC >= 5:
                    self.mean_layer_OC5_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC5_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #6 prior - z"
                if self.config.NoC >= 6:
                    self.mean_layer_OC6_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC6_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #7 prior - z"
                if self.config.NoC >= 7:
                    self.mean_layer_OC7_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC7_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #8 prior - z"
                if self.config.NoC >= 8:
                    self.mean_layer_OC8_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC8_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #9 prior - z"
                if self.config.NoC >= 9:
                    self.mean_layer_OC9_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC9_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #10 prior - z"
                if self.config.NoC >= 10:
                    self.mean_layer_OC10_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC10_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #11 prior - z"
                if self.config.NoC >= 11:
                    self.mean_layer_OC11_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC11_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #12 prior - z"
                if self.config.NoC >= 12:
                    self.mean_layer_OC12_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC12_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #13 prior - z"
                if self.config.NoC >= 13:
                    self.mean_layer_OC13_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC13_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #14 prior - z"
                if self.config.NoC >= 14:
                    self.mean_layer_OC14_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC14_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #15 prior - z"
                if self.config.NoC >= 15:
                    self.mean_layer_OC15_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC15_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                if self.config.NoC >= 16:
                    print("Warning: NoC >= 16 - This is not supported in the current implementation of DecVAE")


            
            "S latent variable branch - prior regularization layers"
            if config.dual_branched_latent or config.only_s_branch:
                "Component #1 prior - s"
                self.mean_layer_OC1_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_OC1_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #2 prior - s"
                self.mean_layer_OC2_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_OC2_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #3 prior - s"
                if self.config.NoC_seq >= 3:
                    self.mean_layer_OC3_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC3_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #4 prior - s"
                if self.config.NoC_seq >= 4:
                    self.mean_layer_OC4_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC4_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #5 prior - s"            
                if self.config.NoC_seq >= 5:
                    self.mean_layer_OC5_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC5_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #6 prior - s"
                if self.config.NoC_seq >= 6:
                    self.mean_layer_OC6_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC6_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #7 prior - s"
                if self.config.NoC_seq >= 7:
                    self.mean_layer_OC7_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC7_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #8 prior - s"
                if self.config.NoC_seq >= 8:
                    self.mean_layer_OC8_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC8_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #9 prior - s"
                if self.config.NoC_seq >= 9:
                    self.mean_layer_OC9_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC9_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #10 prior - s"
                if self.config.NoC_seq >= 10:
                    self.mean_layer_OC10_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC10_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #11 prior - s"
                if self.config.NoC_seq >= 11:
                    self.mean_layer_OC11_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC11_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #12 prior - s"
                if self.config.NoC_seq >= 12:
                    self.mean_layer_OC12_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC12_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #13 prior - s"
                if self.config.NoC_seq >= 13:
                    self.mean_layer_OC13_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC13_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #14 prior - s"
                if self.config.NoC_seq >= 14:
                    self.mean_layer_OC14_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC14_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #15 prior - s"
                if self.config.NoC_seq >= 15:
                    self.mean_layer_OC15_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC15_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                if self.config.NoC_seq >= 16:
                    print("Warning: NoC_seq >= 16 - This is not supported in the current implementation of DecVAE")

        # Initialize weights and apply final processing
        self.post_init()


    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2_z.custom_feature_extractor._freeze_parameters()

    @staticmethod
    def compute_prior_reg_loss(
        mu: torch.FloatTensor,
        logvar: torch.FloatTensor,
        prior_type: str = "gaussian",
    ):
        """
        Compute the KL divergence regularization loss between encoded distributions and a prior.
        
        This static method calculates the KL divergence between a parameterized diagonal Gaussian
        distribution (defined by mu and logvar) and a prior distribution (standard
        normal distribution supported).
        """
        if prior_type == "gaussian":
            prior_reg_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)    #dim=1
        else:
            raise ValueError("Prior type not yet supported")
    
        return prior_reg_loss.sum()

    @staticmethod
    def compute_decomposition_loss(
        z_embeddings: torch.FloatTensor,
        z_t_embeddings: torch.FloatTensor,
        negative_indices: list,
        N: Union[int,str] = 100,
        temperature: int = 0.1,
        divergence_type: str = "js",
        reduction: str = "sum",
    ):
        """
        Computes the decomposition loss between original embeddings and their component embeddings.
        
        This method calculates divergence measures between the original embedding distributions
        and their decomposed components, encouraging component orthogonality and fidelity to the original.
        
        Args:
            z_embeddings (torch.FloatTensor): Original/unmasked feature embeddings.
            z_t_embeddings (torch.FloatTensor): Component/decomposed feature embeddings.
            negative_indices (list): List of indices indicating which frames to use in the loss calculation.
            N (Union[int, str], optional): Number of samples to use or "all". Defaults to 100.
            temperature (int, optional): Temperature parameter for softmax scaling. Defaults to 0.1. (Not used in this implementation)
            divergence_type (str, optional): Type of divergence measure to use: "kl" or "js". Defaults to "js".
            reduction (str, optional): Reduction method for the loss: "sum" or "mean". Defaults to "sum".
        
        Returns:
            Multiple return values including:
            - N_div_pos, N_div_neg: Counts of positive and negative divergences
            - div_dict: Dictionary of component-wise divergence measures
            - div_pos, div_neg: Aggregated positive and negative divergences
            - Additional metrics like cross-entropy values, correlogram, and orthogonality measures
            - used_decomposition_features, used_unmasked_features, used_indices: The processed features used in the calculation
        """
    
        NoC = z_t_embeddings.shape[0]
        z_embeddings = z_embeddings.unsqueeze(0)
        decomp_matrix = torch.cat([z_embeddings, z_t_embeddings], dim=0)
        i_indices, j_indices = zip(*negative_indices)
        i_indices = torch.tensor(i_indices,device=z_embeddings.device,dtype=torch.long)
        j_indices = torch.tensor(j_indices,device=z_embeddings.device,dtype=torch.long)
        batches = torch.unique(i_indices)

        N_div_pos = torch.tensor([0],device=z_embeddings.device,dtype=torch.long)
        N_div_neg = torch.tensor([0],device=z_embeddings.device,dtype=torch.long)
        div_dict = {"div_0_{}".format(i+1):[] for i in range(NoC)}
        div_dict = {**div_dict, **{"div_{}_{}".format(i+1,j+1):[] for i in range(NoC) for j in range(i+1) if i != j}}
        div_neg = 0
        div_pos = 0

        used_decomposition_features = []
        used_unmasked_features = []
        used_indices = []
        for i in batches:
            if len(z_t_embeddings.shape) == 4 and len(z_embeddings.shape) == 4:
                batch_i = i_indices[i_indices == i]
                items = len(batch_i)
                #Sample D negatives from the i/j indices
                perm = torch.randperm(batch_i.shape[0])   
                if type(N) == int:
                    if N >= items:
                        negs_i = batch_i[perm[:items]]
                        negs_j = j_indices[perm[:items]]
                    else:
                        negs_i = batch_i[perm[:N]]
                        negs_j = j_indices[perm[:N]]
                elif type(N) == str:
                    if N == "all":
                        negs_i = batch_i[perm[:items]]
                        negs_j = j_indices[perm[:items]]    

                #Hard-negative cross-correlation criterion selection - Pairs with high cross-correlation that are negatives should be excluded
                #arrange as (C+1)*H - X1,X11,X12,X13,X2,X21,X22,X23,X3,X31,X32,X33,...
                negs_j = torch.sort(negs_j)[0]
                used_indices.append(negs_j)
                
                batch_decomp_matrix = decomp_matrix[:,negs_i,negs_j,:].transpose(0,1).reshape(-1,decomp_matrix.shape[-1])
                "Compute these for the KL-divergence loss"
                masked_originals = batch_decomp_matrix[::(NoC+1),:]
                masked_components = torch.stack([k for j,k in enumerate(batch_decomp_matrix) if j%(NoC+1) != 0]).view(NoC,-1,decomp_matrix.shape[-1])

            elif len(z_t_embeddings.shape) == 3 and len(z_embeddings.shape) == 3:
                batch_decomp_matrix = decomp_matrix[:,i,:].clone()
                masked_originals = batch_decomp_matrix[0,:].unsqueeze(0)
                masked_components = batch_decomp_matrix[1:,:].unsqueeze(1)

            used_decomposition_features.append(masked_components)
            used_unmasked_features.append(masked_originals)
            
            #Calculate Kullback-Leibler divergence to use in subsequent calculations
            kl = nn.KLDivLoss(reduction='sum', log_target=False)
            eps = 1e-8 #for numerical stability in KLDiv (in the derivative calculation)
            bce = nn.BCELoss(reduction='sum')
            #Positives 
            N_div_pos += masked_originals.shape[0]*NoC
            #Negatives
            n_neg = len(list(combinations(range(0,NoC),2)))
            N_div_neg += masked_components.shape[1]*n_neg
            #Distribution P: original frames
            p = torch.nn.functional.softmax(masked_originals,dim=1) 
            

            if (masked_components != masked_components).any():
                print("NaNs in masked_components")

            #Calculate divergence-based Decomposition Loss for Positives and Negatives
            #Jensen-Shannon, Kullback-Leibler, Wasserstein Distance are supported
            for l,n in enumerate(masked_components):
                #Distribution q: components A
                q = torch.nn.functional.softmax(n,dim=-1)
                if divergence_type == "js":
                    #Intermediate mixture distribution M for Jensen-Shannon
                    m = (0.5 * (p + q)).log()    
                    JS_div_pos_pair = torch.clamp(0.5 * (kl(m,p+eps) + kl(m,q+eps)),min = 0,max = 1.0)
                    
                    if (JS_div_pos_pair != JS_div_pos_pair).any():
                        q = torch.clamp(q.clone(),min=1/q.shape[-1])
                        #q = torch.clamp(q,min=1/q.shape[-1])
                        p = torch.clamp(p.clone(),min=1/p.shape[-1])
                        #p = torch.clamp(p,min=1/p.shape[-1])
                        JS_div_pos_pair = torch.clamp(0.5 * (kl(m,p+eps) + kl(m,q+eps)),min = 0,max = 1.0)
                        if (JS_div_pos_pair != JS_div_pos_pair).any():
                            #However this way there is no gradient
                            JS_div_pos_pair = torch.clamp(0.5 * (kl(p,m) + kl(q,m)),min = 0,max = 1.0)
                            print("P distribution: ", p)
                            print("Q distribution: ", q)
                            print("Mixture : ", m)
                            print("JS_div_pos_pair: ",JS_div_pos_pair)
                            print("Zero probabilities detected in original and/or component distributions - KL divergence of original and component is not defined in this case (will assign JSD of 1)")

                    div_dict["div_0_{}".format(l+1)].append(JS_div_pos_pair)
                    div_pos += JS_div_pos_pair
                   
                elif divergence_type == "kl":
                    KL_div_pos_pair = kl(q.log(),p)
                    div_dict["div_0_{}".format(l+1)].append(KL_div_pos_pair)
                    div_pos += KL_div_pos_pair

                for k in range(l+1,masked_components.shape[0]):
                    #Distribution R: components B
                    r = torch.nn.functional.softmax(masked_components[k,...],dim=-1)
                    if divergence_type == "js":
                        #Intermediate mixture distribution M for Jensen-Shannon
                        m = (0.5 * (r + q)).log()        
                        JS_div_neg_pair = torch.clamp(0.5 * (kl(m,q+eps) + kl(m,r+eps)),min = 0,max = 1.0)
                        
                        if (JS_div_neg_pair != JS_div_neg_pair).any():
                            q = torch.clamp(q.clone(),min=1/q.shape[-1])
                            r = torch.clamp(r.clone(),min=1/r.shape[-1])
                            JS_div_neg_pair = torch.clamp(0.5 * (kl(m,q+eps) + kl(m,r+eps)),min = 0,max = 1.0)
                            if (JS_div_neg_pair != JS_div_neg_pair).any():
                                JS_div_neg_pair = torch.clamp(0.5 * (kl(q,m) + kl(r,m)),min = 0,max = 1.0)
                                print("R distribution: ", r)
                                print("Q distribution: ", q)
                                print("Mixture : ", m)
                                print("JS_div_neg_pair: ",JS_div_neg_pair)
                                print("Zero probabilities detected in components' distributions - KL divergence of component A and component B is not defined in this case (will assign JSD of 1)")

                        div_dict["div_{}_{}".format(k+1,l+1)].append(JS_div_neg_pair)
                        div_neg += JS_div_neg_pair
                        
                    elif divergence_type == "kl":
                        KL_div_neg_pair = kl(r.log(),q)
                        div_dict["div_{}_{}".format(k+1,l+1)].append(KL_div_neg_pair)
                        div_neg += KL_div_neg_pair

            
        #Cross-entropy penalty for the positive and negative pairs
        div_dict_positives = torch.stack([torch.stack(values) for key, values in div_dict.items() if key.startswith("div_0")])
        div_dict_negatives = torch.stack([torch.stack(values) for key, values in div_dict.items() if not key.startswith("div_0")])
        
        ce_pos = {"ce_0_{}".format(i+1):[] for i in range(NoC)}
        #For JS and KL calculate cross-entropy, for WS calculate log-distance
        if divergence_type == "js":
            ce_neg = 0    
            for j,neg in enumerate(div_dict_negatives):
                neg_target = torch.ones_like(neg)
                ce_neg += bce(neg,neg_target) / (len(batches))
            for j,pos in enumerate(div_dict_positives):
                pos_target = torch.zeros_like(pos)
                ce_pos["ce_0_{}".format(j+1)] = bce(pos,pos_target) / (len(batches))
        

        div_pos = div_dict_positives.sum() / (len(batches)*NoC)
        div_neg = div_dict_negatives.sum() / (len(batches)*n_neg)
        
        #div_positives and negatives are likelihoods; we want to maximize the likelihood in both cases (make close to 1)
        if reduction == "sum":
            div_dict_pos = {key: torch.sum(torch.stack(values)) / len(batches) for key, values in div_dict.items() if key.startswith("div_0")}
            div_dict_neg = {key: torch.sum(torch.stack(values)) / len(batches) for key, values in div_dict.items() if not key.startswith("div_0")}
            div_dict = {**div_dict_pos, **div_dict_neg}

        elif reduction == "mean":
            #Don't use if dividing by num_losses outside
            div_dict_pos = {key: torch.sum(torch.stack(values)) / (N_div_pos/NoC) for key, values in div_dict.items() if key.startswith("div_0")}
            div_dict_neg = {key: torch.sum(torch.stack(values)) / (N_div_neg/n_neg) for key, values in div_dict.items() if not key.startswith("div_0")}

            div_dict = {**div_dict_pos, **div_dict_neg}
            div_pos = div_pos / N_div_pos
            div_neg = div_neg / N_div_neg
            if divergence_type == "js":
                #JS_entropy_reg = JS_entropy_reg / (N_div_pos+N_div_neg)
                ce_pos = {key: value / (N_div_pos/NoC) for key, value in ce_pos.items()}
                #ce_pos = ce_pos / N_div_pos
                ce_neg = ce_neg / N_div_neg

        #For compatibility with current version
        avg_correlogram = []
        ortho_dict = {} 

        used_decomposition_features = torch.cat(used_decomposition_features,dim=1)
        used_unmasked_features = torch.cat(used_unmasked_features,dim=0)
        assert used_decomposition_features.shape[-1] == used_unmasked_features.shape[-1] == z_embeddings.shape[-1] 

        if divergence_type == "js":
            return N_div_pos, N_div_neg, div_dict, div_pos, div_neg, ce_pos, ce_neg, avg_correlogram, ortho_dict, used_decomposition_features, used_unmasked_features, used_indices
        else:
            return N_div_pos, N_div_neg, div_dict, div_pos, div_neg, avg_correlogram, ortho_dict, used_decomposition_features, used_unmasked_features, used_indices     

    @add_start_docstrings_to_model_forward(DECVAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DecVAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        input_seq_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        reconstruction_NRMSEs: Optional[torch.Tensor] = None,
        avg_correlogram: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DecVAEForPreTrainingOutput]:
        """
        Forward pass for the DecVAEForPreTraining model.
        
        This method handles the full pretraining workflow of the DecVAE model:
        
        1. Process inputs through the model branches:
           - Z branch: Processes segment-level information e.g. content/phonetic information from audio features
           - S branch: Processes sequence-level information e.g. style/speaker information (optional, when dual_branched_latent=True)
        
        2. Apply projections and transformations:
           - Project features to appropriate embedding spaces
           - Generate variational parameters (mu, logvar) for each branch and component
           - Sample from variational distributions when use_prior_regularization=True
        
        3. Calculate various loss components:
           - Decomposition loss between original and component representations
           - Prior regularization (KL divergence) for variational components
        
        4. Handle both single and dual-branch architectures:
           - Z-only: Focus on content/phonetic information
           - S-only: Focus on style/speaker information
           - Dual-branch: Process both types of information in separate paths
        
        The method adapts its behavior based on configuration parameters like loss_mode,
        divergence_type, and dual_branched_latent.
        
        Returns:
            DecVAEForPreTrainingOutput or tuple: Model outputs including loss values,
            hidden states, attention values, and other metrics depending on configuration
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)
        
        if self.config.dual_branched_latent or self.config.only_z_branch:
        #In case of decomposition, input values is now already a tensor of shape (batch_size, components,sequence_length,frame_size)
            outputs_z = self.wav2vec2_z(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                mask_time_indices=mask_time_indices,
                reconstruction_NRMSEs = reconstruction_NRMSEs,  
                avg_correlogram = avg_correlogram,
                return_dict=return_dict,
            )
        if self.config.dual_branched_latent or self.config.only_s_branch:
            hidden_states_s = self.encoder_s(input_seq_values)
            #[NoC+1,B,L,C] -> [NoC+1,B,C,L]
            if self.config.fc_or_conv_s == "conv":
                hidden_states_s = hidden_states_s.transpose(-1,-2)          

        if self.config.dual_branched_latent or self.config.only_z_branch:
            extract_features = outputs_z[1]
            extract_features = extract_features[0,...] #keep only original frames
            extract_features = self.dropout_z_features(extract_features)
        else:
            extract_features = hidden_states_s[0,...] #extract_features
            extract_features = self.dropout_s_features(extract_features)
            #Use for decomposition_loss: Hidden states are the projected form of extract features (LayerNorm + linear projection)

        if self.config.dual_branched_latent or self.config.only_z_branch:
            NoC,batch_size, sequence_length, hidden_size = outputs_z.hidden_states[1:].shape
            assert NoC == self.config.NoC
            if self.config.dual_branched_latent:
                NoC_seq,_, _, _ = hidden_states_s[1:].shape
                assert NoC_seq == self.config.NoC_seq
        else:
            if self.config.fc_or_conv_s == "conv":
                NoC_seq,batch_size, sequence_length, hidden_size = hidden_states_s[1:].shape
            elif self.config.fc_or_conv_s == "fc":
                NoC_seq,batch_size, hidden_size = hidden_states_s[1:].shape
            assert NoC_seq == self.config.NoC_seq

        if self.config.dual_branched_latent or self.config.only_z_branch:
            component_features_z = self.project_z(self.dropout_z_features(outputs_z.hidden_states[1:])) #outputs.hidden_states[i+1,...]
            if self.config.use_self_attention_z:
                raise NotImplementedError("Self-attention for z branch is not supported")
                component_features_z_attn = torch.zeros_like(component_features_z)
                for i in range(NoC):
                    component_features_z_attn[i] ,_ = self.mha_z(component_features_z[i],component_features_z[i],component_features_z[i])
            original_features_z = self.project_z(self.dropout_z_features(outputs_z.hidden_states[0,...]))
        if self.config.dual_branched_latent or self.config.only_s_branch:
            component_features_s = self.project_s(self.dropout_s_features(hidden_states_s[1:]))
            original_features_s = self.project_s(self.dropout_s_features(hidden_states_s[0,...]))

            if self.config.use_first_agg or self.config.use_second_agg:
                component_features_s_agg = torch.zeros((component_features_s.shape[0],component_features_s.shape[1],component_features_s.shape[-1]),
                        device=hidden_states_s.device,dtype=hidden_states_s.dtype)
                original_features_s_agg = self.aggregator_s(original_features_s)
                for i in range(self.config.NoC_seq):
                    component_features_s_agg[i] = self.aggregator_s(component_features_s[i])
            else:
                component_features_s_agg = component_features_s.clone()
                original_features_s_agg = original_features_s.clone()

        if self.config.use_prior_regularization:
            "Prior regularization for components"
            for i in range(NoC):     
                if self.config.dual_branched_latent or self.config.only_z_branch:  
                    if self.config.use_self_attention_z:
                        raise NotImplementedError("Self-attention for z branch is not supported")
                        #OC_z = component_features_z_attn[i,...].clone()
                    else:
                        OC_z = component_features_z[i,...].clone()
                if self.config.dual_branched_latent or self.config.only_s_branch:
                    OC_s = component_features_s_agg[i,...].clone()
                if i == 0:                            
                    if self.config.dual_branched_latent or self.config.only_z_branch:        
                        mean_OC1_z = self.mean_layer_OC1_z(OC_z) 
                        logvar_OC1_z = self.logvar_layer_OC1_z(OC_z) 
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC1_s = self.mean_layer_OC1_s(OC_s)
                        logvar_OC1_s = self.logvar_layer_OC1_s(OC_s)
                elif i == 1:
                    if self.config.dual_branched_latent or self.config.only_z_branch:     
                        mean_OC2_z = self.mean_layer_OC2_z(OC_z) 
                        logvar_OC2_z = self.logvar_layer_OC2_z(OC_z) 
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC2_s = self.mean_layer_OC2_s(OC_s)
                        logvar_OC2_s = self.logvar_layer_OC2_s(OC_s)
                elif i == 2:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC3_z = self.mean_layer_OC3_z(OC_z)
                        logvar_OC3_z = self.logvar_layer_OC3_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC3_s = self.mean_layer_OC3_s(OC_s)
                        logvar_OC3_s = self.logvar_layer_OC3_s(OC_s)
                elif i == 3:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC4_z = self.mean_layer_OC4_z(OC_z)
                        logvar_OC4_z = self.logvar_layer_OC4_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC4_s = self.mean_layer_OC4_s(OC_s)
                        logvar_OC4_s = self.logvar_layer_OC4_s(OC_s)
                elif i == 4:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC5_z = self.mean_layer_OC5_z(OC_z)
                        logvar_OC5_z = self.logvar_layer_OC5_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC5_s = self.mean_layer_OC5_s(OC_s)
                        logvar_OC5_s = self.logvar_layer_OC5_s(OC_s)
                elif i == 5:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC6_z = self.mean_layer_OC6_z(OC_z)
                        logvar_OC6_z = self.logvar_layer_OC6_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC6_s = self.mean_layer_OC6_s(OC_s)
                        logvar_OC6_s = self.logvar_layer_OC6_s(OC_s)
                elif i == 6:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC7_z = self.mean_layer_OC7_z(OC_z)
                        logvar_OC7_z = self.logvar_layer_OC7_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC7_s = self.mean_layer_OC7_s(OC_s)
                        logvar_OC7_s = self.logvar_layer_OC7_s(OC_s)
                elif i == 7:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC8_z = self.mean_layer_OC8_z(OC_z)
                        logvar_OC8_z = self.logvar_layer_OC8_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC8_s = self.mean_layer_OC8_s(OC_s)
                        logvar_OC8_s = self.logvar_layer_OC8_s(OC_s)
                elif i == 8:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC9_z = self.mean_layer_OC9_z(OC_z)
                        logvar_OC9_z = self.logvar_layer_OC9_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC9_s = self.mean_layer_OC9_s(OC_s)
                        logvar_OC9_s = self.logvar_layer_OC9_s(OC_s)
                elif i == 9:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC10_z = self.mean_layer_OC10_z(OC_z)
                        logvar_OC10_z = self.logvar_layer_OC10_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC10_s = self.mean_layer_OC10_s(OC_s)
                        logvar_OC10_s = self.logvar_layer_OC10_s(OC_s)
                elif i == 10:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC11_z = self.mean_layer_OC11_z(OC_z)
                        logvar_OC11_z = self.logvar_layer_OC11_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC11_s = self.mean_layer_OC11_s(OC_s)
                        logvar_OC11_s = self.logvar_layer_OC11_s(OC_s)
                elif i == 11:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC12_z = self.mean_layer_OC12_z(OC_z)
                        logvar_OC12_z = self.logvar_layer_OC12_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC12_s = self.mean_layer_OC12_s(OC_s)
                        logvar_OC12_s = self.logvar_layer_OC12_s(OC_s)
                elif i == 12:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC13_z = self.mean_layer_OC13_z(OC_z)
                        logvar_OC13_z = self.logvar_layer_OC13_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC13_s = self.mean_layer_OC13_s(OC_s)
                        logvar_OC13_s = self.logvar_layer_OC13_s(OC_s)
                elif i == 13:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC14_z = self.mean_layer_OC14_z(OC_z)
                        logvar_OC14_z = self.logvar_layer_OC14_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC14_s = self.mean_layer_OC14_s(OC_s)
                        logvar_OC14_s = self.logvar_layer_OC14_s(OC_s)
                elif i == 14:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC15_z = self.mean_layer_OC15_z(OC_z)
                        logvar_OC15_z = self.logvar_layer_OC15_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC15_s = self.mean_layer_OC15_s(OC_s)
                        logvar_OC15_s = self.logvar_layer_OC15_s(OC_s)
                    
                    
                            
            "Prior regularization for X signals"
            if self.config.use_prior_regularization:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    mean_X_z = self.mean_layer_z(original_features_z)
                    logvar_X_z = self.logvar_layer_z(original_features_z)
                if self.config.dual_branched_latent or self.config.only_s_branch:
                    mean_X_s = self.mean_layer_s(original_features_s_agg)
                    logvar_X_s = self.logvar_layer_s(original_features_s_agg)

            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    combined_components_z = component_features_z.permute(1,2,0,3).reshape(batch_size,sequence_length,NoC*self.config.proj_codevector_dim_z)
                    projected_components_z = self.project_components_z(combined_components_z)
                if self.config.dual_branched_latent or self.config.only_s_branch:   
                    combined_components_s = component_features_s_agg.permute(1,2,0).reshape(batch_size,NoC_seq*self.config.proj_codevector_dim_s)
                    projected_components_s = self.project_components_s(combined_components_s)
                if self.config.use_prior_regularization:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_proj_OCs_z = self.mean_layer_proj_OCs_z(projected_components_z)
                        logvar_proj_OCs_z = self.logvar_layer_proj_OCs_z(projected_components_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_proj_OCs_s = self.mean_layer_proj_OCs_s(projected_components_s)
                        logvar_proj_OCs_s = self.logvar_layer_proj_OCs_s(projected_components_s)

        if attention_mask is not None and self.config.fc_or_conv_s == "conv":
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )


        if self.config.dual_branched_latent or self.config.only_z_branch:   
            _,batch_size, sequence_length, hidden_size = component_features_z.shape
        else:
            if self.config.fc_or_conv_s == "conv":
                _,batch_size, sequence_length, hidden_size = component_features_s.shape
            elif self.config.fc_or_conv_s == "fc":
                _,batch_size, hidden_size = component_features_s.shape

        if self.config.fc_or_conv_s == "fc":
            sequence_length = mask_time_indices.shape[-1]
        #4b. Negative feature vectors according to sample_negative_indices - For decomposition loss
        eligible_indices = np.array([[i,j] for i in range(batch_size) for j in range(sequence_length) if mask_time_indices[i,j]])
        
        if (eligible_indices != eligible_indices).any():
            print("NaNs in eligible_indices - line 3238")

        "Compute decomposition loss and logits"
        "First latent variable"
        if self.config.dual_branched_latent or self.config.only_z_branch:   
            N_div_pos_z, N_div_neg_z, div_dict_z, div_pos_z, div_neg_z, ce_pos_z, ce_neg_z, avg_correlogram_z, ortho_dict_z, used_decomposition_features_z, used_unmasked_features_z,used_indices_z = self.compute_decomposition_loss(
                original_features_z,
                component_features_z,
                eligible_indices,
                N = self.config.max_frames_per_batch,
                temperature=self.config.contrastive_logits_temperature,
                divergence_type=self.config.divergence_type,
                reduction=self.config.decomp_loss_reduction,
            )
        "Second latent variable"
        if self.config.dual_branched_latent or self.config.only_s_branch:   
            N_div_pos_s, N_div_neg_s, div_dict_s, div_pos_s, div_neg_s, ce_pos_s, ce_neg_s, avg_correlogram_s, ortho_dict_s, used_decomposition_features_s, used_unmasked_features_s,used_indices_s = self.compute_decomposition_loss(
                original_features_s_agg,
                component_features_s_agg,
                eligible_indices,
                N = self.config.max_frames_per_batch,
                temperature=self.config.contrastive_logits_temperature,
                divergence_type=self.config.divergence_type,
                reduction=self.config.decomp_loss_reduction,
            )

        #Keep only projections that were used in the decomposition loss calculations
        #Careful not to break the grad graph here
        if self.config.project_OCs:
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                used_projected_components_z = torch.cat([projected_components_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
            if self.config.dual_branched_latent or self.config.only_s_branch:   
                used_projected_components_s = projected_components_s.clone()


        if self.config.dual_branched_latent or self.config.only_z_branch:   
            if type(ce_pos_z) == dict:
                decomp_loss_z = (
                    self.config.div_neg_weight * ce_neg_z +
                    self.config.weight_0_1 * ce_pos_z["ce_0_1"] + 
                    self.config.weight_0_2 * ce_pos_z["ce_0_2"] +
                    self.config.weight_0_3_and_above * sum([val for key,val in ce_pos_z.items() if int(key[-1]) > 2])
                )
                ce_pos_z = torch.sum(torch.stack([val for key,val in ce_pos_z.items()]))
            else:
                decomp_loss_z = self.config.div_pos_weight * ce_pos_z + self.config.div_neg_weight * ce_neg_z
        if self.config.dual_branched_latent or self.config.only_s_branch:
            if type(ce_pos_s) == dict:
                decomp_loss_s = (
                    self.config.div_neg_weight * ce_neg_s +
                    self.config.weight_0_1 * ce_pos_s["ce_0_1"] + 
                    self.config.weight_0_2 * ce_pos_s["ce_0_2"] +
                    self.config.weight_0_3_and_above * sum([val for key,val in ce_pos_s.items() if int(key[-1]) > 2])
                ) 
                ce_pos_s = torch.sum(torch.stack([val for key,val in ce_pos_s.items()]))
            else:
                decomp_loss_s = self.config.decomp_loss_s_weight*(self.config.div_pos_weight * ce_pos_s + self.config.div_neg_weight * ce_neg_s) #= torch.log((torch.exp(

        if self.config.use_prior_regularization:
            "Calculate loss with mu and logvar vectors only on 'masked' samples (same as used in decomposition loss)"                    
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                used_mean_X_z = torch.cat([mean_X_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
                used_logvar_X_z = torch.cat([logvar_X_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
            if self.config.dual_branched_latent or self.config.only_s_branch:
                if len(used_indices_s) == 0:
                    used_mean_X_s = mean_X_s.clone()
                    used_logvar_X_s = logvar_X_s.clone()
                else:
                    used_mean_X_s = torch.cat([mean_X_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)
                    used_logvar_X_s = torch.cat([logvar_X_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)
            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    used_mean_proj_OCs_z = torch.cat([mean_proj_OCs_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
                    used_logvar_proj_OCs_z = torch.cat([logvar_proj_OCs_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
                if self.config.dual_branched_latent or self.config.only_s_branch:   
                    if len(used_indices_s) == 0:
                        used_mean_proj_OCs_s = mean_proj_OCs_s.clone()
                        used_logvar_proj_OCs_s = logvar_proj_OCs_s.clone()
                    else:
                        used_mean_proj_OCs_s = torch.cat([mean_proj_OCs_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)
                        used_logvar_proj_OCs_s = torch.cat([logvar_proj_OCs_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)

            "Prior regularization for Z - Use masked samples only"
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                # Create dictionaries to store means and logvars
                used_mean_OCs_z = {}
                used_logvar_OCs_z = {}
                
                # Process each OC based on the NoC value
                for oc_num in range(1, min(16, NoC + 1)):
                    mean_var_name = f"mean_OC{oc_num}_z"
                    logvar_var_name = f"logvar_OC{oc_num}_z"
                    
                    # Get the original variables dynamically using locals()
                    if mean_var_name in locals() and logvar_var_name in locals():
                        mean_val = locals()[mean_var_name]
                        logvar_val = locals()[logvar_var_name]
                        
                        # Process and store in dictionaries
                        used_mean_OCs_z[oc_num] = torch.cat([mean_val[i,inds,:] for i,inds in enumerate(used_indices_z)], dim=0)
                        used_logvar_OCs_z[oc_num] = torch.cat([logvar_val[i,inds,:] for i,inds in enumerate(used_indices_z)], dim=0)
                
                if NoC >= 16:
                    print("NoC >= 16 - Not supported, components 16 and above will not be regularized")

            "Prior regularization for S - Use all samples as they have been averaged"
            if self.config.dual_branched_latent or self.config.only_s_branch:
                # Create dictionaries to store means and logvars for s branch
                used_mean_OCs_s = {}
                used_logvar_OCs_s = {}
                
                # Process each OC based on the NoC value
                for oc_num in range(1, min(16, self.config.NoC_seq + 1)):
                    mean_var_name = f"mean_OC{oc_num}_s"
                    logvar_var_name = f"logvar_OC{oc_num}_s"
                    
                    # Get the original variables dynamically using locals()
                    if mean_var_name in locals() and logvar_var_name in locals():
                        # For s branch, we just clone the values
                        used_mean_OCs_s[oc_num] = locals()[mean_var_name].clone()
                        used_logvar_OCs_s[oc_num] = locals()[logvar_var_name].clone()
                
                if self.config.NoC_seq >= 16:
                    print("NoC_seq >= 16 - Not supported, components 16 and above will not be regularized")

            "Prepare mu and logvar vectors to be returned in the output"
            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    mu_projections_z = used_mean_proj_OCs_z.clone()
                    logvar_projections_z = used_logvar_proj_OCs_z.clone()
                if self.config.dual_branched_latent or self.config.only_s_branch:   
                    mu_projections_s = used_mean_proj_OCs_s.clone()
                    logvar_projections_s = used_logvar_proj_OCs_s.clone()
            
            if self.config.dual_branched_latent or self.config.only_z_branch:
                # Initialize with the first two components
                mu_components_z = torch.stack([used_mean_OCs_z[1], used_mean_OCs_z[2]], dim=0)
                logvar_components_z = torch.stack([used_logvar_OCs_z[1], used_logvar_OCs_z[2]], dim=0)
                
                # Add remaining components if available
                for oc_num in range(3, min(16, NoC + 1)):
                    if oc_num in used_mean_OCs_z:
                        mu_components_z = torch.cat([mu_components_z, used_mean_OCs_z[oc_num].unsqueeze(0)], dim=0)
                        logvar_components_z = torch.cat([logvar_components_z, used_logvar_OCs_z[oc_num].unsqueeze(0)], dim=0)

            if self.config.dual_branched_latent or self.config.only_s_branch:
                # Initialize with the first two components
                mu_components_s = torch.stack([used_mean_OCs_s[1], used_mean_OCs_s[2]], dim=0)
                logvar_components_s = torch.stack([used_logvar_OCs_s[1], used_logvar_OCs_s[2]], dim=0)
                
                # Add remaining components if available
                for oc_num in range(3, min(16, self.config.NoC_seq + 1)):
                    if oc_num in used_mean_OCs_s:
                        mu_components_s = torch.cat([mu_components_s, used_mean_OCs_s[oc_num].unsqueeze(0)], dim=0)
                        logvar_components_s = torch.cat([logvar_components_s, used_logvar_OCs_s[oc_num].unsqueeze(0)], dim=0)

            "Compute prior regularization loss based on KL divergence between prior distribution and unit Gaussian"
            #prior_loss_X = self.compute_prior_reg_loss(used_mean_X,used_logvar_X,prior_type="gaussian")
            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:
                    prior_loss_proj_OCs_z = self.compute_prior_reg_loss(used_mean_proj_OCs_z,used_logvar_proj_OCs_z) 
                if self.config.dual_branched_latent or self.config.only_s_branch:
                    prior_loss_proj_OCs_s = self.compute_prior_reg_loss(used_mean_proj_OCs_s,used_logvar_proj_OCs_s) 
    
            if self.config.dual_branched_latent or self.config.only_z_branch:
                # Initialize prior loss dictionary for z branch
                prior_loss_OCs_z = {}
                
                # Calculate prior loss for each component
                for oc_num in range(1, min(16, NoC + 1)):
                    if oc_num in used_mean_OCs_z:
                        prior_loss_OCs_z[oc_num] = self.compute_prior_reg_loss(
                            used_mean_OCs_z[oc_num], 
                            used_logvar_OCs_z[oc_num], 
                            prior_type="gaussian"
                        )
                    else:
                        prior_loss_OCs_z[oc_num] = torch.tensor(0, device=extract_features.device)


            if self.config.dual_branched_latent or self.config.only_s_branch:
                # Initialize prior loss dictionary for s branch
                prior_loss_OCs_s = {}
                
                # Calculate prior loss for each component
                for oc_num in range(1, min(16, self.config.NoC_seq + 1)):
                    if oc_num in used_mean_OCs_s:
                        prior_loss_OCs_s[oc_num] = self.compute_prior_reg_loss(
                            used_mean_OCs_s[oc_num], 
                            used_logvar_OCs_s[oc_num], 
                            prior_type="gaussian"
                        )
                    else:
                        prior_loss_OCs_s[oc_num] = torch.tensor(0, device=extract_features.device)

                
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                if not self.config.project_OCs:
                    # Use sum on dictionary values instead of individual variables
                    prior_loss_z = self.config.prior_reg_weighting_z * self.config.beta_kl_prior_z * (
                        sum(prior_loss_OCs_z.values()) / NoC
                    )
                else:
                    # Include projected OCs in the calculation
                    prior_loss_z = self.config.prior_reg_weighting_z * self.config.beta_kl_prior_z * (
                        prior_loss_proj_OCs_z + sum(prior_loss_OCs_z.values())
                    ) / (NoC + 1)
                loss_z = decomp_loss_z + prior_loss_z

            if self.config.dual_branched_latent or self.config.only_s_branch:   
                if not self.config.project_OCs:
                    # Use sum on dictionary values instead of individual variables
                    prior_loss_s = self.config.prior_reg_weighting_s * self.config.beta_kl_prior_s * (
                        sum(prior_loss_OCs_s.values()) / self.config.NoC_seq
                    )
                else:
                    # Include projected OCs in the calculation
                    prior_loss_s = self.config.prior_reg_weighting_s * self.config.beta_kl_prior_s * (
                        prior_loss_proj_OCs_s + sum(prior_loss_OCs_s.values())
                    ) / (self.config.NoC_seq + 1)
                loss_s = decomp_loss_s + prior_loss_s
            
            if self.config.dual_branched_latent:                        
                loss = loss_z + loss_s  
            elif self.config.only_z_branch:
                loss_s = torch.tensor(0,device=extract_features.device)
                loss = loss_z.clone()
            elif self.config.only_s_branch:
                loss_z = torch.tensor(0,device=extract_features.device)
                loss = loss_s.clone()
                
        else:
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                loss_z = decomp_loss_z.clone()
            if self.config.dual_branched_latent or self.config.only_s_branch:
                loss_s = decomp_loss_s.clone()
            if self.config.dual_branched_latent:
                loss = loss_z + loss_s
            elif self.config.only_z_branch:
                loss_s = torch.tensor(0,device=extract_features.device)
                loss = loss_z.clone()
            elif self.config.only_s_branch:
                loss_z = torch.tensor(0,device=extract_features.device)
                loss = loss_s.clone()
        
        if self.config.dual_branched_latent or self.config.only_z_branch: 
            mu_originals_z = used_mean_X_z.clone()
            logvar_originals_z = used_logvar_X_z.clone()
        if self.config.dual_branched_latent or self.config.only_s_branch: 
            mu_originals_s = used_mean_X_s.clone()
            logvar_originals_s = used_logvar_X_s.clone()
        
        if not self.config.project_OCs:
            mu_projections_z = None
            logvar_projections_z = None
            used_projected_components_z = None
            mu_projections_s = None
            logvar_projections_s = None
            used_projected_components_s = None   
        
        if self.config.only_z_branch:   
            return DecVAEForPreTrainingOutput(
                loss=loss,
                loss_z = loss_z,
                hidden_states_z = outputs_z.hidden_states,
                unmasked_frames_features_z=original_features_z,
                decomposition_features_z=component_features_z,
                used_decomposition_features_z = used_decomposition_features_z, 
                used_unmasked_features_z = used_unmasked_features_z,
                used_projected_components_z = used_projected_components_z,
                mu_originals_z = mu_originals_z,
                logvar_originals_z = logvar_originals_z,
                mu_projections_z = mu_projections_z,
                logvar_projections_z = logvar_projections_z,
                mu_components_z = mu_components_z,
                logvar_components_z = logvar_components_z,
                used_indices_z = used_indices_z,
                attentions=outputs_z.attentions,     
                decomposition_loss_z = decomp_loss_z,
                prior_loss_z = prior_loss_z,      
                divergence_dict_z = div_dict_z,
                div_pos_z = div_pos_z,
                div_neg_z = div_neg_z,
                ce_pos_z = ce_pos_z,
                ce_neg_z = ce_neg_z,
                avg_correlogram_z = avg_correlogram_z,
                avg_correlogram_td_z = outputs_z.avg_correlogram,
                orthogonality_dict_z = ortho_dict_z,
                orthogonality_dict_td_z = outputs_z.ortho_dict,
                N_div_pos_z = N_div_pos_z,
                N_div_neg_z = N_div_neg_z,
                mask_time_indices = mask_time_indices,
            )
        elif self.config.only_s_branch:
            return DecVAEForPreTrainingOutput(
                loss=loss,
                loss_s = loss_s,
                hidden_states_s = hidden_states_s,
                unmasked_frames_features_s=original_features_s_agg,
                decomposition_features_s=component_features_s_agg,
                used_decomposition_features_s = used_decomposition_features_s,
                used_unmasked_features_s = used_unmasked_features_s,
                used_projected_components_s = used_projected_components_s,
                mu_originals_s = mu_originals_s,
                logvar_originals_s = logvar_originals_s,
                mu_projections_s = mu_projections_s,
                logvar_projections_s = logvar_projections_s,
                mu_components_s = mu_components_s,
                logvar_components_s = logvar_components_s,
                used_indices_s = used_indices_s,
                attentions=None,  
                decomposition_loss_s = decomp_loss_s,   
                prior_loss_s = prior_loss_s,
                divergence_dict_s = div_dict_s,
                div_pos_s = div_pos_s,
                div_neg_s = div_neg_s,
                ce_pos_s = ce_pos_s,
                ce_neg_s = ce_neg_s,
                avg_correlogram_s = avg_correlogram_s,
                avg_correlogram_td_s = None,
                orthogonality_dict_s = ortho_dict_s,
                orthogonality_dict_td_s = None,
                N_div_pos_s = N_div_pos_s,
                N_div_neg_s = N_div_neg_s,
                mask_time_indices = mask_time_indices,
            )
        elif self.config.dual_branched_latent:   
            return DecVAEForPreTrainingOutput(
                loss=loss,
                loss_z = loss_z,
                loss_s = loss_s,
                hidden_states_z = outputs_z.hidden_states,
                hidden_states_s = hidden_states_s,
                unmasked_frames_features_z=original_features_z,
                unmasked_frames_features_s=original_features_s_agg,
                decomposition_features_z=component_features_z,
                decomposition_features_s=component_features_s_agg,
                used_decomposition_features_z = used_decomposition_features_z, 
                used_decomposition_features_s = used_decomposition_features_s,
                used_unmasked_features_z = used_unmasked_features_z,
                used_unmasked_features_s = used_unmasked_features_s,
                used_projected_components_z = used_projected_components_z,
                used_projected_components_s = used_projected_components_s,
                mu_originals_z = mu_originals_z,
                mu_originals_s = mu_originals_s,
                logvar_originals_z = logvar_originals_z,
                logvar_originals_s = logvar_originals_s,
                mu_projections_z = mu_projections_z,
                mu_projections_s = mu_projections_s,
                logvar_projections_z = logvar_projections_z,
                logvar_projections_s = logvar_projections_s,
                mu_components_z = mu_components_z,
                mu_components_s = mu_components_s,
                logvar_components_z = logvar_components_z,
                logvar_components_s = logvar_components_s,
                used_indices_z = used_indices_z,
                used_indices_s = used_indices_s,
                attentions=outputs_z.attentions,   
                decomposition_loss_z = decomp_loss_z,
                decomposition_loss_s = decomp_loss_s,
                prior_loss_z = prior_loss_z,      
                prior_loss_s = prior_loss_s,
                divergence_dict_z = div_dict_z,
                divergence_dict_s = div_dict_s,
                div_pos_z = div_pos_z,
                div_pos_s = div_pos_s,
                div_neg_z = div_neg_z,
                div_neg_s = div_neg_s,
                ce_pos_z = ce_pos_z,
                ce_pos_s = ce_pos_s,
                ce_neg_z = ce_neg_z,
                ce_neg_s = ce_neg_s,
                avg_correlogram_z = avg_correlogram_z,
                avg_correlogram_s = avg_correlogram_s,
                avg_correlogram_td_z = outputs_z.avg_correlogram,
                avg_correlogram_td_s = None,
                orthogonality_dict_z = ortho_dict_z,
                orthogonality_dict_s = ortho_dict_s,
                orthogonality_dict_td_z = outputs_z.ortho_dict,
                orthogonality_dict_td_s = None,
                N_div_pos_z = N_div_pos_z,
                N_div_pos_s = N_div_pos_s,
                N_div_neg_z = N_div_neg_z,
                N_div_neg_s = N_div_neg_s,
                mask_time_indices = mask_time_indices,
            )


@add_start_docstrings("""DecVAE Model with supervised fine-tuning capabilities for classification tasks.""", DECVAE_START_DOCSTRING)
class DecVAEForSupervisedFineTuning(Wav2Vec2PreTrainedModel):
    """
    DecVAE model for supervised fine-tuning on classification tasks.
    
    This class extends DecVAEForPreTraining with additional supervised learning capabilities,
    allowing the model to be fine-tuned on labeled data using both the z-branch (segment-level)
    and s-branch (sequence-level) features.
    """
    
    def __init__(self, config: DecVAEConfig):
        """
        Initialize the DecVAEForSupervisedFineTuning model.
        
        Sets up an architecture with multiple components based on the configuration:
        
        1. Base encoder models:
           - dec2vec_z: Dec2Vec model for the z (content) branch based on the wav2vec2 architecture
           - encoder_s: Either convolutional or linear stack for the s (style) branch
        
        2. Projection networks:
           - project_z/project_s: Project features to the contrastive embedding space

           - project_components_z/project_components_s: For component projections
        
        3. Variational components (when use_prior_regularization=True):
           - mean_layer_*/logvar_layer_*: Variational parameter estimators for different branches
           - Component-specific variational layers for orthogonal components (OC1, OC2, etc.)
        
        4. Optional sequence aggregation for the s branch

        5. Classification head for supervised fine-tuning

        Args:
            config (DecVAEConfig): Configuration object containing model parameters
        """
        super().__init__(config)
        self.config = config
        "Initialize z latent branch"
        if config.dual_branched_latent or config.only_z_branch:
            self.wav2vec2_z = Dec2VecModel(config)
            self.dropout_z_features = nn.Dropout(config.hidden_dropout)
        "Initialize s latent branch"
        if config.dual_branched_latent or config.only_s_branch:
            if self.config.fc_or_conv_s == "conv":
                self.encoder_s = ConvStack(
                    hidden_dim=config.conv_dim_s[0],
                    kernel_sizes=config.conv_kernel,
                    strides=config.conv_stride,
                    activation = config.feat_extract_activation)
            elif self.config.fc_or_conv_s == "fc":
                self.encoder_s = LinearStack(
                    input_dim=config.fc_input_size,
                    layer_sizes=config.fc_kernels,  # Reusing the same config parameters
                    activation=config.feat_extract_activation
                )
            
            self.dropout_s_features = nn.Dropout(config.hidden_dropout)
            if config.use_first_agg or config.use_second_agg:
                if not config.use_first_agg:
                    config.first_agg = None
                if not config.use_second_agg:
                    config.second_agg = None
                self.aggregator_s = SequenceAggregator(hidden_dim=config.proj_codevector_dim_s, first_stage=config.first_agg, second_stage=config.second_agg, num_heads=config.attention_heads_s)

        if config.dual_branched_latent or config.only_z_branch:
            self.project_z = nn.Sequential(
                nn.Linear(config.z_proj_input_dim, config.proj_intermediate_dim),
                nn.Linear(config.proj_intermediate_dim, config.proj_codevector_dim_z),
                ACT2FN[config.feat_extract_activation]
            ) #nn.LayerNorm(config.proj_codevector_dim_z),
        if config.dual_branched_latent or config.only_s_branch:
            #self.project_s = nn.Linear(config.s_proj_input_dim, config.proj_codevector_dim_s)
            self.project_s = nn.Sequential(
                nn.Linear(config.s_proj_input_dim, config.proj_intermediate_dim),
                nn.Linear(config.proj_intermediate_dim, config.proj_codevector_dim_s),
                ACT2FN[config.feat_extract_activation]
            ) #nn.LayerNorm(config.proj_codevector_dim_s),
        
        if config.project_OCs:
            if config.dual_branched_latent or config.only_z_branch:
                self.project_components_z = nn.Sequential(
                    nn.Linear(config.NoC*config.proj_codevector_dim_z, config.proj_codevector_dim_z),
                    nn.Linear(config.proj_codevector_dim_z, config.proj_codevector_dim_z),
                    ACT2FN[config.feat_extract_activation]
                )
            if config.dual_branched_latent or config.only_s_branch:
                self.project_components_s = nn.Sequential(
                    nn.Linear(config.NoC_seq*config.proj_codevector_dim_s, config.proj_codevector_dim_s),
                    nn.Linear(config.proj_codevector_dim_s, config.proj_codevector_dim_s),
                    ACT2FN[config.feat_extract_activation]
                )

        if self.config.use_prior_regularization:
            #self.sampling = Sampling() #this only useful if a decoder is added
            if config.dual_branched_latent or config.only_z_branch:
                self.mean_layer_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
            if config.dual_branched_latent or config.only_s_branch:
                self.mean_layer_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)

            #if config.project_OCs:
            if config.dual_branched_latent or config.only_z_branch:
                "Component projection prior - z"
                self.mean_layer_proj_OCs_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_proj_OCs_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
            if config.dual_branched_latent or config.only_s_branch:
                "Component projection prior - s"
                self.mean_layer_proj_OCs_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_proj_OCs_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
            "Z latent variable branch - prior regularization layers"
            if config.dual_branched_latent or config.only_z_branch:
                "Component #1 prior - z"
                self.mean_layer_OC1_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_OC1_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #2 prior - z"
                self.mean_layer_OC2_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                self.logvar_layer_OC2_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #3 prior - z"
                if self.config.NoC >= 3:
                    self.mean_layer_OC3_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC3_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #4 prior - z"
                if self.config.NoC >= 4:
                    self.mean_layer_OC4_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC4_z= nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #5 prior - z"
                if self.config.NoC >= 5:
                    self.mean_layer_OC5_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC5_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #6 prior - z"
                if self.config.NoC >= 6:
                    self.mean_layer_OC6_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC6_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #7 prior - z"
                if self.config.NoC >= 7:
                    self.mean_layer_OC7_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC7_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #8 prior - z"
                if self.config.NoC >= 8:
                    self.mean_layer_OC8_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC8_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #9 prior - z"
                if self.config.NoC >= 9:
                    self.mean_layer_OC9_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC9_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #10 prior - z"
                if self.config.NoC >= 10:
                    self.mean_layer_OC10_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC10_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #11 prior - z"
                if self.config.NoC >= 11:
                    self.mean_layer_OC11_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC11_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #12 prior - z"
                if self.config.NoC >= 12:
                    self.mean_layer_OC12_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC12_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #13 prior - z"
                if self.config.NoC >= 13:
                    self.mean_layer_OC13_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC13_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #14 prior - z"
                if self.config.NoC >= 14:
                    self.mean_layer_OC14_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC14_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                "Component #15 prior - z"
                if self.config.NoC >= 15:
                    self.mean_layer_OC15_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                    self.logvar_layer_OC15_z = nn.Linear(config.proj_codevector_dim_z, config.z_latent_dim)
                if self.config.NoC >= 16:
                    print("Warning: NoC >= 16 - This is not supported in the current implementation of DecVAE")


            
            "S latent variable branch - prior regularization layers"
            if config.dual_branched_latent or config.only_s_branch:
                "Component #1 prior - s"
                self.mean_layer_OC1_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_OC1_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #2 prior - s"
                self.mean_layer_OC2_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                self.logvar_layer_OC2_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #3 prior - s"
                if self.config.NoC_seq >= 3:
                    self.mean_layer_OC3_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC3_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #4 prior - s"
                if self.config.NoC_seq >= 4:
                    self.mean_layer_OC4_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC4_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #5 prior - s"            
                if self.config.NoC_seq >= 5:
                    self.mean_layer_OC5_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC5_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #6 prior - s"
                if self.config.NoC_seq >= 6:
                    self.mean_layer_OC6_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC6_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #7 prior - s"
                if self.config.NoC_seq >= 7:
                    self.mean_layer_OC7_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC7_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #8 prior - s"
                if self.config.NoC_seq >= 8:
                    self.mean_layer_OC8_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC8_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #9 prior - s"
                if self.config.NoC_seq >= 9:
                    self.mean_layer_OC9_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC9_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #10 prior - s"
                if self.config.NoC_seq >= 10:
                    self.mean_layer_OC10_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC10_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #11 prior - s"
                if self.config.NoC_seq >= 11:
                    self.mean_layer_OC11_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC11_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #12 prior - s"
                if self.config.NoC_seq >= 12:
                    self.mean_layer_OC12_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC12_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #13 prior - s"
                if self.config.NoC_seq >= 13:
                    self.mean_layer_OC13_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC13_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #14 prior - s"
                if self.config.NoC_seq >= 14:
                    self.mean_layer_OC14_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC14_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                "Component #15 prior - s"
                if self.config.NoC_seq >= 15:
                    self.mean_layer_OC15_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                    self.logvar_layer_OC15_s = nn.Linear(config.proj_codevector_dim_s, config.s_latent_dim)
                if self.config.NoC_seq >= 16:
                    print("Warning: NoC_seq >= 16 - This is not supported in the current implementation of DecVAE")

        # Initialize the classification head for supervised fine-tuning
        if config.fine_tuning_classifier_z is not None or config.fine_tuning_classifier_s is not None:
            if config.only_z_branch or (config.dual_branched_latent and not config.aggregate_branch_features):
                self.classification_head_z = self._create_classifier(
                    config.fine_tuning_classifier_z,
                    config.z_latent_dim*(config.NoC+1),
                    config.proj_codevector_dim_z,
                    config.proj_codevector_dim_z,
                    config.fine_tuning_output_classes,
                    config.final_dropout 
                )
            if config.only_s_branch or (config.dual_branched_latent and not config.aggregate_branch_features):
                self.classification_head_s = self._create_classifier(
                    config.fine_tuning_classifier_s,
                    config.s_latent_dim*(config.NoC_seq+1), #config.NoC_seq+1
                    config.proj_codevector_dim_s,
                    config.proj_codevector_dim_s, 
                    config.fine_tuning_output_classes,
                    config.final_dropout
                )
            if config.dual_branched_latent and config.aggregate_branch_features:
                "Aggregate latent spaces and use both types of features"
                "In this case sequence features will be constant w.r.t segment features"
                self.classification_head_z = self._create_classifier(
                    config.fine_tuning_classifier_z,
                    config.z_latent_dim*(config.NoC+1) + config.s_latent_dim*(config.NoC_seq+1),
                    config.proj_codevector_dim_z,
                    config.proj_codevector_dim_z, 
                    config.fine_tuning_output_classes,
                    config.final_dropout
                )
            

        # Initialize weights and apply final processing
        self.post_init()
        
    def _create_classifier(self, classifier_type, input_dim, hidden_dim, proj_size, num_classes, dropout_prob=0.1):
        """
        Create a classifier head based on the specified type.
        
        Args:
            classifier_type (str): Type of classifier to create ('mlp', 'cnn', 'lstm', 'transformer', etc.)
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden layers
            proj_size (int): Projection size for some classifiers
            num_classes (int): Number of output classes
            dropout_prob (float): Dropout probability
            
        Returns:
            nn.Module: The classifier module
        """
        if classifier_type == "mlp":
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim // 2, num_classes)
            )
            
        elif classifier_type == "cnn":
            return nn.Sequential(
                nn.Unflatten(1, (1, input_dim)),  # [B, 1, input_dim]
                nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),  # Global average pooling
                nn.Flatten(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_dim*2, num_classes)
            )
            
        elif classifier_type == "lstm":
            class LSTMClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(
                        input_size=input_dim,
                        hidden_size=hidden_dim,
                        num_layers=2,
                        batch_first=True,
                        dropout=dropout_prob,
                        bidirectional=True
                    )
                    self.dropout = nn.Dropout(dropout_prob)
                    self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional
                    
                def forward(self, x):
                    # Reshape if input is just [batch, features]
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # [batch, 1, features]
                    
                    # LSTM expects [batch, seq_len, features]
                    output, (h_n, _) = self.lstm(x)
                    
                    # Concatenate the final hidden states from both directions
                    h_n = torch.cat([h_n[-2], h_n[-1]], dim=1)
                    
                    # Apply dropout and final classification
                    h_n = self.dropout(h_n)
                    logits = self.fc(h_n)
                    return logits
                    
            return LSTMClassifier()
            
        elif classifier_type == "transformer":
            # Transformer classifier - attends to different parts of the input with different weights
            class TransformerClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Project input to model dimension if needed
                    self.input_projection = nn.Linear(input_dim, proj_size) if input_dim != proj_size else nn.Identity()
                    
                    # Transformer encoder layer
                    self.encoder_layer = nn.TransformerEncoderLayer(
                        d_model=proj_size,
                        nhead=8,  # Number of attention heads
                        dim_feedforward=hidden_dim,
                        dropout=dropout_prob,
                        activation="gelu",
                        batch_first=True
                    )
                    
                    # Transformer encoder
                    self.transformer_encoder = nn.TransformerEncoder(
                        self.encoder_layer,
                        num_layers=2
                    )
                    
                    # Classification head
                    self.classifier = nn.Linear(proj_size, num_classes)
                    
                def forward(self, x):
                    # Reshape if input is just [batch, features]
                    if len(x.shape) == 2:
                        x = x.unsqueeze(1)  # [batch, 1, features]
                        
                    # Project input if needed
                    x = self.input_projection(x)
                    
                    # Apply transformer
                    x = self.transformer_encoder(x)
                    
                    # Average pooling over sequence dimension
                    x = x.mean(dim=1)
                    
                    # Classification
                    return self.classifier(x)
                    
            return TransformerClassifier()
            
    def load_pretrained_weights(self, pretrained_path, verbose=True, strict=False):
        """
        Load weights from a pretrained DecVAE model into this supervised fine-tuning model.
        
        This method properly handles loading weights from a pretrained model that doesn't have
        classifier heads, by:
        1. Loading the pretrained weights with strict=False
        2. Reporting any unexpected and missing keys
        3. Verifying that only classifier-related keys are missing
        
        Args:
            pretrained_path (str): Path to the pretrained model weights file (.safetensors, .bin, etc.)
            verbose (bool, optional): Whether to print detailed messages about loaded weights. Default: True
            strict (bool, optional): Whether to strictly enforce that the keys in state_dict match. Default: False
        
        Returns:
            Tuple[List[str], List[str]]: Lists of unexpected and missing keys
        
        Raises:
            ValueError: If critical non-classifier keys are missing from the pretrained weights
        """
        
        if verbose:
            print(f"Loading pretrained weights from: {pretrained_path}")
        
        # Determine file type and load accordingly
        if pretrained_path.endswith(".safetensors"):
            if os.path.exists(pretrained_path):
                state_dict = load_file(pretrained_path)
            else:
                raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
        elif pretrained_path.endswith(".bin"):
            if os.path.exists(pretrained_path):
                state_dict = torch.load(pretrained_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"Pretrained weights file not found: {pretrained_path}")
        else:
            raise ValueError(f"Unsupported weights file format: {pretrained_path}")
        
        # Get current model state dict
        model_state_dict = self.state_dict()
        
        # Load weights with strict=False to allow missing keys (like classifier heads)
        load_result = self.load_state_dict(state_dict, strict=strict)
        
        # Get lists of missing and unexpected keys
        missing_keys = load_result.missing_keys
        unexpected_keys = load_result.unexpected_keys
        
        # Identify keys related to classifier heads
        classifier_related_keys = [k for k in model_state_dict.keys() if 
                                "classification_head" in k or "classifier" in k]
        
        # Check if all missing keys are classifier-related
        non_classifier_missing_keys = [k for k in missing_keys if k not in classifier_related_keys]
        
        if verbose:
            print(f"Loaded pretrained weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
            
            if classifier_related_keys:
                print(f"Found {len(classifier_related_keys)} classifier-related keys in the model")

            if missing_keys:
                print(f"Missing keys: {missing_keys[:5] + ['...'] if len(missing_keys) > 5 else missing_keys}")
            
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5] + ['...'] if len(unexpected_keys) > 5 else unexpected_keys}")
        
        # Warn if there are missing keys that aren't classifier-related
        if non_classifier_missing_keys and verbose:
            print(f"Warning: {len(non_classifier_missing_keys)} missing keys are not classifier-related")
            print(f"Non-classifier missing keys: {non_classifier_missing_keys[:5] + ['...'] if len(non_classifier_missing_keys) > 5 else non_classifier_missing_keys}")
        
        return load_result, classifier_related_keys

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2_z.custom_feature_extractor._freeze_parameters()

    @staticmethod
    def compute_prior_reg_loss(
        mu: torch.FloatTensor,
        logvar: torch.FloatTensor,
        prior_type: str = "gaussian",
    ):
        """
        Compute the KL divergence regularization loss between encoded distributions and a prior.
        
        This static method calculates the KL divergence between a parameterized diagonal Gaussian
        distribution (defined by mu and logvar) and a prior distribution (standard
        normal distribution supported).
        """
        if prior_type == "gaussian":
            prior_reg_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1)    #dim=1
        else:
            raise ValueError("Prior type not yet supported")
    
        return prior_reg_loss.sum()

    @staticmethod
    def compute_supervised_loss(
        logits: torch.FloatTensor,
        labels: torch.FloatTensor,
        loss_type: str = "cross_entropy",
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.FloatTensor] = None,
        reduction: str = "sum",
    ):
        """
        Compute supervised classification loss between logits and labels.
        
        This method handles different input shapes for both logits and labels:
        - logits: [B, S, N] (batch_size, num_segments, num_classes) or [B, N] (batch_size, num_classes)
        - labels: [B, S] (batch_size, num_segments) or [B] (batch_size) or [B, S, N] (one-hot encoded)
        
        Args:
            logits (torch.FloatTensor): The predicted logits 
                Shape: [B, S, N] or [B, N] where:
                B = batch size
                S = number of segments
                N = number of classes
            labels (torch.FloatTensor): The target labels
                Shape: [B, S], [B] or [B, S, N] (one-hot encoded)
            loss_type (str, optional): Type of loss to compute. 
                Options: "cross_entropy", "focal", "label_smoothed_ce". Default: "cross_entropy"
            label_smoothing (float, optional): Label smoothing factor (0-1). Default: 0.0
            class_weights (torch.FloatTensor, optional): Weights for each class for weighted loss. Default: None
            reduction (str, optional): Specifies the reduction to apply to the output. 
                Options: 'none', 'mean', 'sum'. Default: 'mean'
                
        Returns:
            torch.FloatTensor: Computed loss
        """
        device = logits.device
        
        # Handle different shapes
        if len(logits.shape) == 3:  # [B, S, N]
            B, S, N = logits.shape
            # Reshape to [B*S, N]
            logits_flat = logits.reshape(-1, N)
            
            # Handle different label shapes
            if len(labels.shape) == 2:  # [B, S]
                labels_flat = labels.reshape(-1)
            elif len(labels.shape) == 1:  # [B]
                # Repeat labels for each segment
                labels_flat = labels.unsqueeze(1).repeat(1, S).reshape(-1)
            elif len(labels.shape) == 3:  # [B, S, N] (one-hot)
                labels_flat = labels.reshape(-1, N)
            else:
                raise ValueError(f"Unsupported label shape: {labels.shape}")
                
        elif len(logits.shape) == 2:  # [B, N]
            B, N = logits.shape
            logits_flat = logits
            
            # Handle different label shapes
            if len(labels.shape) == 1:  # [B]
                labels_flat = labels
            elif len(labels.shape) == 2:  # [B, N] (one-hot)
                if labels.shape[1] == N:  # One-hot encoded
                    labels_flat = labels
                else:  # [B, S] but S != N
                    raise ValueError(f"Label shape {labels.shape} incompatible with logits shape {logits.shape}")
            else:
                raise ValueError(f"Unsupported label shape: {labels.shape}")
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")
        
        # Check if labels are one-hot encoded
        if len(labels_flat.shape) == 2 and labels_flat.shape[1] > 1:
            is_one_hot = True
            # Convert one-hot to class indices if needed for some loss functions
            class_indices = torch.argmax(labels_flat, dim=1)
        else:
            is_one_hot = False
            class_indices = labels_flat
        
        # Compute the appropriate loss
        if loss_type == "cross_entropy":
            if is_one_hot:
                # For one-hot labels, use binary cross entropy with logits
                if class_weights is not None:
                    # Apply class weights to each sample
                    sample_weights = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1)
                    loss_fn = nn.BCEWithLogitsLoss(weight=sample_weights, reduction=reduction)
                else:
                    loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)
                
                loss = loss_fn(logits_flat, labels_flat.float())
            else:
                # For class indices, use cross entropy
                if label_smoothing > 0:
                    loss_fn = nn.CrossEntropyLoss(
                        weight=class_weights,
                        label_smoothing=label_smoothing,
                        reduction=reduction
                    )
                else:
                    loss_fn = nn.CrossEntropyLoss(
                        weight=class_weights,
                        reduction=reduction
                    )
                
                loss = loss_fn(logits_flat, class_indices.long())
                
        elif loss_type == "focal":
            # Focal Loss implementation (helps with class imbalance)
            gamma = 2.0  # Focusing parameter
            
            if is_one_hot:
                # Convert logits to probabilities
                probs = torch.sigmoid(logits_flat)
                # Calculate focal weights
                pt = labels_flat * probs + (1 - labels_flat) * (1 - probs)
                focal_weights = (1 - pt) ** gamma
                
                # Standard BCE loss
                bce_loss = nn.functional.binary_cross_entropy_with_logits(
                    logits_flat, 
                    labels_flat.float(), 
                    reduction='none'
                )
                
                # Apply focal weights
                loss = focal_weights * bce_loss
                
                # Apply class weights if provided
                if class_weights is not None:
                    class_weight_per_sample = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1, keepdim=True)
                    loss = loss * class_weight_per_sample
                    
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
            else:
                # One-hot encode class_indices for focal loss calculation
                labels_one_hot = nn.functional.one_hot(class_indices.long(), num_classes=N).float()
                
                # Convert logits to probabilities
                probs = nn.functional.softmax(logits_flat, dim=1)
                
                # Calculate focal weights
                pt = (labels_one_hot * probs).sum(dim=1)
                focal_weights = (1 - pt) ** gamma
                
                # Standard CE loss
                ce_loss = nn.functional.cross_entropy(
                    logits_flat, 
                    class_indices.long(), 
                    weight=class_weights,
                    reduction='none'
                )
                
                # Apply focal weights
                loss = focal_weights * ce_loss
                
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
                    
        elif loss_type == "label_smoothed_ce":
            # Custom implementation of label smoothing for both one-hot and class indices
            if is_one_hot:
                # For one-hot labels, smooth the labels directly
                smooth_labels = labels_flat * (1 - label_smoothing) + label_smoothing / N
                
                # Compute KL divergence loss
                log_probs = nn.functional.log_softmax(logits_flat, dim=1)
                loss = -(smooth_labels * log_probs).sum(dim=1)
                
                # Apply class weights if provided
                if class_weights is not None:
                    class_weight_per_sample = (labels_flat * class_weights.unsqueeze(0)).sum(dim=1)
                    loss = loss * class_weight_per_sample
                    
                # Apply reduction
                if reduction == 'mean':
                    loss = loss.mean()
                elif reduction == 'sum':
                    loss = loss.sum()
                elif reduction == 'none':
                    pass  # No reduction needed
            else:
                # For class indices, use built-in CrossEntropyLoss with label_smoothing
                loss_fn = nn.CrossEntropyLoss(
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                    reduction=reduction
                )
                loss = loss_fn(logits_flat, class_indices.long())
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
            
        return loss, logits_flat, class_indices

    @staticmethod
    def compute_decomposition_loss(
        z_embeddings: torch.FloatTensor,
        z_t_embeddings: torch.FloatTensor,
        negative_indices: list,
        N: Union[int,str] = 100,
        temperature: int = 0.1,
        divergence_type: str = "js",
        reduction: str = "sum",
    ):
        """
        Computes the decomposition loss between original embeddings and their component embeddings.
        
        This method calculates divergence measures between the original embedding distributions
        and their decomposed components, encouraging component orthogonality and fidelity to the original.
        
        Args:
            z_embeddings (torch.FloatTensor): Original/unmasked feature embeddings.
            z_t_embeddings (torch.FloatTensor): Component/decomposed feature embeddings.
            negative_indices (list): List of indices indicating which frames to use in the loss calculation.
            N (Union[int, str], optional): Number of samples to use or "all". Defaults to 100.
            temperature (int, optional): Temperature parameter for softmax scaling. Defaults to 0.1. (Not used in this implementation)
            divergence_type (str, optional): Type of divergence measure to use: "kl" or "js". Defaults to "js".
            reduction (str, optional): Reduction method for the loss: "sum" or "mean". Defaults to "sum".
        
        Returns:
            Multiple return values including:
            - N_div_pos, N_div_neg: Counts of positive and negative divergences
            - div_dict: Dictionary of component-wise divergence measures
            - div_pos, div_neg: Aggregated positive and negative divergences
            - Additional metrics like cross-entropy values, correlogram, and orthogonality measures
            - used_decomposition_features, used_unmasked_features, used_indices: The processed features used in the calculation
        """
    
        NoC = z_t_embeddings.shape[0]
        z_embeddings = z_embeddings.unsqueeze(0)
        decomp_matrix = torch.cat([z_embeddings, z_t_embeddings], dim=0)
        i_indices, j_indices = zip(*negative_indices)
        i_indices = torch.tensor(i_indices,device=z_embeddings.device,dtype=torch.long)
        j_indices = torch.tensor(j_indices,device=z_embeddings.device,dtype=torch.long)
        batches = torch.unique(i_indices)

        N_div_pos = torch.tensor([0],device=z_embeddings.device,dtype=torch.long)
        N_div_neg = torch.tensor([0],device=z_embeddings.device,dtype=torch.long)
        div_dict = {"div_0_{}".format(i+1):[] for i in range(NoC)}
        div_dict = {**div_dict, **{"div_{}_{}".format(i+1,j+1):[] for i in range(NoC) for j in range(i+1) if i != j}}
        div_neg = 0
        div_pos = 0

        used_decomposition_features = []
        used_unmasked_features = []
        used_indices = []
        for i in batches:
            if len(z_t_embeddings.shape) == 4 and len(z_embeddings.shape) == 4:
                batch_i = i_indices[i_indices == i]
                items = len(batch_i)
                #Sample D negatives from the i/j indices
                perm = torch.randperm(batch_i.shape[0])   
                if type(N) == int:
                    if N >= items:
                        negs_i = batch_i[perm[:items]]
                        negs_j = j_indices[perm[:items]]
                    else:
                        negs_i = batch_i[perm[:N]]
                        negs_j = j_indices[perm[:N]]
                elif type(N) == str:
                    if N == "all":
                        negs_i = batch_i[perm[:items]]
                        negs_j = j_indices[perm[:items]]    

                #Hard-negative cross-correlation criterion selection - Pairs with high cross-correlation that are negatives should be excluded
                #arrange as (C+1)*H - X1,X11,X12,X13,X2,X21,X22,X23,X3,X31,X32,X33,...
                negs_j = torch.sort(negs_j)[0]
                used_indices.append(negs_j)
                
                batch_decomp_matrix = decomp_matrix[:,negs_i,negs_j,:].transpose(0,1).reshape(-1,decomp_matrix.shape[-1])
                "Compute these for the KL-divergence loss"
                masked_originals = batch_decomp_matrix[::(NoC+1),:]
                masked_components = torch.stack([k for j,k in enumerate(batch_decomp_matrix) if j%(NoC+1) != 0]).view(NoC,-1,decomp_matrix.shape[-1])

            elif len(z_t_embeddings.shape) == 3 and len(z_embeddings.shape) == 3:
                batch_decomp_matrix = decomp_matrix[:,i,:].clone()
                masked_originals = batch_decomp_matrix[0,:].unsqueeze(0)
                masked_components = batch_decomp_matrix[1:,:].unsqueeze(1)

            used_decomposition_features.append(masked_components)
            used_unmasked_features.append(masked_originals)
            
            #Calculate Kullback-Leibler divergence to use in subsequent calculations
            kl = nn.KLDivLoss(reduction='sum', log_target=False)
            eps = 1e-8 #for numerical stability in KLDiv (in the derivative calculation)
            bce = nn.BCELoss(reduction='sum')
            #Positives 
            N_div_pos += masked_originals.shape[0]*NoC
            #Negatives
            n_neg = len(list(combinations(range(0,NoC),2)))
            N_div_neg += masked_components.shape[1]*n_neg
            #Distribution P: original frames
            p = torch.nn.functional.softmax(masked_originals,dim=1) 
            

            if (masked_components != masked_components).any():
                print("NaNs in masked_components")

            #Calculate divergence-based Decomposition Loss for Positives and Negatives
            #Jensen-Shannon, Kullback-Leibler, Wasserstein Distance are supported
            for l,n in enumerate(masked_components):
                #Distribution q: components A
                q = torch.nn.functional.softmax(n,dim=-1)
                if divergence_type == "js":
                    #Intermediate mixture distribution M for Jensen-Shannon
                    m = (0.5 * (p + q)).log()    
                    JS_div_pos_pair = torch.clamp(0.5 * (kl(m,p+eps) + kl(m,q+eps)),min = 0,max = 1.0)
                    
                    if (JS_div_pos_pair != JS_div_pos_pair).any():
                        q = torch.clamp(q.clone(),min=1/q.shape[-1])
                        #q = torch.clamp(q,min=1/q.shape[-1])
                        p = torch.clamp(p.clone(),min=1/p.shape[-1])
                        #p = torch.clamp(p,min=1/p.shape[-1])
                        JS_div_pos_pair = torch.clamp(0.5 * (kl(m,p+eps) + kl(m,q+eps)),min = 0,max = 1.0)
                        if (JS_div_pos_pair != JS_div_pos_pair).any():
                            #However this way there is no gradient
                            JS_div_pos_pair = torch.clamp(0.5 * (kl(p,m) + kl(q,m)),min = 0,max = 1.0)
                            print("P distribution: ", p)
                            print("Q distribution: ", q)
                            print("Mixture : ", m)
                            print("JS_div_pos_pair: ",JS_div_pos_pair)
                            print("Zero probabilities detected in original and/or component distributions - KL divergence of original and component is not defined in this case (will assign JSD of 1)")
                    """
                        if (JS_div_pos_pair != JS_div_pos_pair).any():
                            print("Could not assign JSD of 1 because both the original and the component have zero probabilities")
                            print("Assigning uniform distribution to component and re-calculating...")
                            q = torch.ones_like(q) / q.shape[-1]
                            m = (0.5 * (p + q)).log()     
                            JS_div_pos_pair = torch.clamp(0.5 * (kl(p.log(),m) + kl(q.log(),m)),min = 0,max = 1.0)
                            print("JS_div_pos_pair after second correction: ",JS_div_pos_pair)
                    """
                    div_dict["div_0_{}".format(l+1)].append(JS_div_pos_pair)
                    div_pos += JS_div_pos_pair
                   
                elif divergence_type == "kl":
                    KL_div_pos_pair = kl(q.log(),p)
                    div_dict["div_0_{}".format(l+1)].append(KL_div_pos_pair)
                    div_pos += KL_div_pos_pair

                for k in range(l+1,masked_components.shape[0]):
                    #Distribution R: components B
                    r = torch.nn.functional.softmax(masked_components[k,...],dim=-1)
                    if divergence_type == "js":
                        #Intermediate mixture distribution M for Jensen-Shannon
                        m = (0.5 * (r + q)).log()        
                        JS_div_neg_pair = torch.clamp(0.5 * (kl(m,q+eps) + kl(m,r+eps)),min = 0,max = 1.0)
                        
                        if (JS_div_neg_pair != JS_div_neg_pair).any():
                            q = torch.clamp(q.clone(),min=1/q.shape[-1])
                            r = torch.clamp(r.clone(),min=1/r.shape[-1])
                            JS_div_neg_pair = torch.clamp(0.5 * (kl(m,q+eps) + kl(m,r+eps)),min = 0,max = 1.0)
                            if (JS_div_neg_pair != JS_div_neg_pair).any():
                                JS_div_neg_pair = torch.clamp(0.5 * (kl(q,m) + kl(r,m)),min = 0,max = 1.0)
                                print("R distribution: ", r)
                                print("Q distribution: ", q)
                                print("Mixture : ", m)
                                print("JS_div_neg_pair: ",JS_div_neg_pair)
                                print("Zero probabilities detected in components' distributions - KL divergence of component A and component B is not defined in this case (will assign JSD of 1)")
                        """
                            if (JS_div_neg_pair != JS_div_neg_pair).any():
                                print("Could not assign JSD of 1 because both components have zero probabilities")
                                print("Assigning uniform distribution to component B and re-calculating...")
                                r = torch.ones_like(r) / r.shape[-1]
                                m = (0.5 * (r + q)).log()     
                                JS_div_neg_pair = torch.clamp(0.5 * (kl(q.log(),m) + kl(r.log(),m)),min = 0,max = 1.0)
                                print("JS_div_neg_pair after second correction: ",JS_div_neg_pair)
                        """
                        div_dict["div_{}_{}".format(k+1,l+1)].append(JS_div_neg_pair)
                        div_neg += JS_div_neg_pair
                        
                    elif divergence_type == "kl":
                        KL_div_neg_pair = kl(r.log(),q)
                        div_dict["div_{}_{}".format(k+1,l+1)].append(KL_div_neg_pair)
                        div_neg += KL_div_neg_pair

            
        #Cross-entropy penalty for the positive and negative pairs
        div_dict_positives = torch.stack([torch.stack(values) for key, values in div_dict.items() if key.startswith("div_0")])
        div_dict_negatives = torch.stack([torch.stack(values) for key, values in div_dict.items() if not key.startswith("div_0")])
        
        ce_pos = {"ce_0_{}".format(i+1):[] for i in range(NoC)}
        #For JS and KL calculate cross-entropy, for WS calculate log-distance
        if divergence_type == "js":
            ce_neg = 0    
            for j,neg in enumerate(div_dict_negatives):
                neg_target = torch.ones_like(neg)
                ce_neg += bce(neg,neg_target) / (len(batches))
            for j,pos in enumerate(div_dict_positives):
                pos_target = torch.zeros_like(pos)
                ce_pos["ce_0_{}".format(j+1)] = bce(pos,pos_target) / (len(batches))
        

        div_pos = div_dict_positives.sum() / (len(batches)*NoC)
        div_neg = div_dict_negatives.sum() / (len(batches)*n_neg)
        
        #div_positives and negatives are likelihoods; we want to maximize the likelihood in both cases (make close to 1)
        if reduction == "sum":
            div_dict_pos = {key: torch.sum(torch.stack(values)) / len(batches) for key, values in div_dict.items() if key.startswith("div_0")}
            div_dict_neg = {key: torch.sum(torch.stack(values)) / len(batches) for key, values in div_dict.items() if not key.startswith("div_0")}
            div_dict = {**div_dict_pos, **div_dict_neg}

        elif reduction == "mean":
            #Don't use if dividing by num_losses outside
            div_dict_pos = {key: torch.sum(torch.stack(values)) / (N_div_pos/NoC) for key, values in div_dict.items() if key.startswith("div_0")}
            div_dict_neg = {key: torch.sum(torch.stack(values)) / (N_div_neg/n_neg) for key, values in div_dict.items() if not key.startswith("div_0")}

            div_dict = {**div_dict_pos, **div_dict_neg}
            div_pos = div_pos / N_div_pos
            div_neg = div_neg / N_div_neg
            if divergence_type == "js":
                #JS_entropy_reg = JS_entropy_reg / (N_div_pos+N_div_neg)
                ce_pos = {key: value / (N_div_pos/NoC) for key, value in ce_pos.items()}
                #ce_pos = ce_pos / N_div_pos
                ce_neg = ce_neg / N_div_neg

        #For compatibility with current version
        avg_correlogram = []
        ortho_dict = {} 

        used_decomposition_features = torch.cat(used_decomposition_features,dim=1)
        used_unmasked_features = torch.cat(used_unmasked_features,dim=0)
        assert used_decomposition_features.shape[-1] == used_unmasked_features.shape[-1] == z_embeddings.shape[-1] 

        if divergence_type == "js":
            return N_div_pos, N_div_neg, div_dict, div_pos, div_neg, ce_pos, ce_neg, avg_correlogram, ortho_dict, used_decomposition_features, used_unmasked_features, used_indices
        else:
            return N_div_pos, N_div_neg, div_dict, div_pos, div_neg, avg_correlogram, ortho_dict, used_decomposition_features, used_unmasked_features, used_indices     

    @add_start_docstrings_to_model_forward(DECVAE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DecVAEForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        input_seq_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.BoolTensor] = None,
        labels_z: Optional[torch.Tensor] = None,
        labels_s: Optional[torch.Tensor] = None,
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        reconstruction_NRMSEs: Optional[torch.Tensor] = None,
        avg_correlogram: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, DecVAEForPreTrainingOutput]:
        """
        Forward pass for the DecVAEForPreTraining model.
        
        This complex method handles the full pretraining workflow of the DecVAE model:
        
        1. Process inputs through the model branches:
           - Z branch: Processes segment-level information e.g. content/phonetic information from audio features
           - S branch: Processes sequence-level information e.g. style/speaker information (optional, when dual_branched_latent=True)
        
        2. Apply projections and transformations:
           - Project features to appropriate embedding spaces
           - Generate variational parameters (mu, logvar) for each branch and component
           - Sample from variational distributions when use_prior_regularization=True
        
        3. Calculate various loss components:
           - Decomposition loss between original and component representations
           - Prior regularization (KL divergence) for variational components
        
        4. Handle both single and dual-branch architectures:
           - Z-only: Focus on content/phonetic information
           - S-only: Focus on style/speaker information
           - Dual-branch: Process both types of information in separate paths
        
        The method adapts its behavior based on configuration parameters like loss_mode,
        divergence_type, and dual_branched_latent.
        
        Returns:
            DecVAEForPreTrainingOutput or tuple: Model outputs including loss values,
            hidden states, attention values, and other metrics depending on configuration
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if mask_time_indices is not None:
            mask_time_indices = mask_time_indices.to(torch.bool)
        
        if self.config.dual_branched_latent or self.config.only_z_branch:
        #In case of decomposition, input values is now already a tensor of shape (batch_size, components,sequence_length,frame_size)
            outputs_z = self.wav2vec2_z(
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                mask_time_indices=mask_time_indices,
                reconstruction_NRMSEs = reconstruction_NRMSEs,  
                avg_correlogram = avg_correlogram,
                return_dict=return_dict,
            )
        if self.config.dual_branched_latent or self.config.only_s_branch:
            hidden_states_s = self.encoder_s(input_seq_values)
            #[NoC+1,B,L,C] -> [NoC+1,B,C,L]
            if self.config.fc_or_conv_s == "conv":
                hidden_states_s = hidden_states_s.transpose(-1,-2)          

        if self.config.dual_branched_latent or self.config.only_z_branch:
            extract_features = outputs_z[1]
            extract_features = extract_features[0,...] #keep only original frames
            extract_features = self.dropout_z_features(extract_features)
        else:
            extract_features = hidden_states_s[0,...] #extract_features
            extract_features = self.dropout_s_features(extract_features)
            #Use for decomposition_loss: Hidden states are the projected form of extract features (LayerNorm + linear projection)

        if self.config.dual_branched_latent or self.config.only_z_branch:
            NoC,batch_size, sequence_length, hidden_size = outputs_z.hidden_states[1:].shape
            assert NoC == self.config.NoC
            if self.config.dual_branched_latent:
                NoC_seq, _, _, _ = hidden_states_s[1:].shape
                assert NoC_seq == self.config.NoC_seq
        else:
            if self.config.fc_or_conv_s == "conv":
                NoC_seq,batch_size, sequence_length, hidden_size = hidden_states_s[1:].shape
            elif self.config.fc_or_conv_s == "fc":
                NoC_seq,batch_size, hidden_size = hidden_states_s[1:].shape
            assert NoC_seq == self.config.NoC_seq
        
        if self.config.dual_branched_latent or self.config.only_z_branch:
            component_features_z = self.project_z(self.dropout_z_features(outputs_z.hidden_states[1:])) #outputs.hidden_states[i+1,...]
            if self.config.use_self_attention_z:
                component_features_z_attn = torch.zeros_like(component_features_z)
                for i in range(NoC):
                    component_features_z_attn[i] ,_ = self.mha_z(component_features_z[i],component_features_z[i],component_features_z[i])
            original_features_z = self.project_z(self.dropout_z_features(outputs_z.hidden_states[0,...]))
        if self.config.dual_branched_latent or self.config.only_s_branch:
            component_features_s = self.project_s(self.dropout_s_features(hidden_states_s[1:]))
            original_features_s = self.project_s(self.dropout_s_features(hidden_states_s[0,...]))

            if self.config.use_first_agg or self.config.use_second_agg:
                component_features_s_agg = torch.zeros((component_features_s.shape[0],component_features_s.shape[1],component_features_s.shape[-1]),
                        device=hidden_states_s.device,dtype=hidden_states_s.dtype)
                original_features_s_agg = self.aggregator_s(original_features_s)
                for i in range(self.config.NoC_seq):
                    component_features_s_agg[i] = self.aggregator_s(component_features_s[i])
            else:
                component_features_s_agg = component_features_s.clone()
                original_features_s_agg = original_features_s.clone()

        if self.config.use_prior_regularization:
            "Prior regularization for components"
            for i in range(NoC):     
                if self.config.dual_branched_latent or self.config.only_z_branch:  
                    if self.config.use_self_attention_z:
                        OC_z = component_features_z_attn[i,...].clone()
                    else:
                        OC_z = component_features_z[i,...].clone()
                if self.config.dual_branched_latent or self.config.only_s_branch:
                    OC_s = component_features_s_agg[i,...].clone()
                if i == 0:                            
                    if self.config.dual_branched_latent or self.config.only_z_branch:        
                        mean_OC1_z = self.mean_layer_OC1_z(OC_z) 
                        logvar_OC1_z = self.logvar_layer_OC1_z(OC_z) 
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC1_s = self.mean_layer_OC1_s(OC_s)
                        logvar_OC1_s = self.logvar_layer_OC1_s(OC_s)
                elif i == 1:
                    if self.config.dual_branched_latent or self.config.only_z_branch:     
                        mean_OC2_z = self.mean_layer_OC2_z(OC_z) 
                        logvar_OC2_z = self.logvar_layer_OC2_z(OC_z) 
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC2_s = self.mean_layer_OC2_s(OC_s)
                        logvar_OC2_s = self.logvar_layer_OC2_s(OC_s)
                elif i == 2:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC3_z = self.mean_layer_OC3_z(OC_z)
                        logvar_OC3_z = self.logvar_layer_OC3_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC3_s = self.mean_layer_OC3_s(OC_s)
                        logvar_OC3_s = self.logvar_layer_OC3_s(OC_s)
                elif i == 3:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC4_z = self.mean_layer_OC4_z(OC_z)
                        logvar_OC4_z = self.logvar_layer_OC4_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC4_s = self.mean_layer_OC4_s(OC_s)
                        logvar_OC4_s = self.logvar_layer_OC4_s(OC_s)
                elif i == 4:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC5_z = self.mean_layer_OC5_z(OC_z)
                        logvar_OC5_z = self.logvar_layer_OC5_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC5_s = self.mean_layer_OC5_s(OC_s)
                        logvar_OC5_s = self.logvar_layer_OC5_s(OC_s)
                elif i == 5:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC6_z = self.mean_layer_OC6_z(OC_z)
                        logvar_OC6_z = self.logvar_layer_OC6_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC6_s = self.mean_layer_OC6_s(OC_s)
                        logvar_OC6_s = self.logvar_layer_OC6_s(OC_s)
                elif i == 6:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC7_z = self.mean_layer_OC7_z(OC_z)
                        logvar_OC7_z = self.logvar_layer_OC7_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC7_s = self.mean_layer_OC7_s(OC_s)
                        logvar_OC7_s = self.logvar_layer_OC7_s(OC_s)
                elif i == 7:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC8_z = self.mean_layer_OC8_z(OC_z)
                        logvar_OC8_z = self.logvar_layer_OC8_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC8_s = self.mean_layer_OC8_s(OC_s)
                        logvar_OC8_s = self.logvar_layer_OC8_s(OC_s)
                elif i == 8:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC9_z = self.mean_layer_OC9_z(OC_z)
                        logvar_OC9_z = self.logvar_layer_OC9_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC9_s = self.mean_layer_OC9_s(OC_s)
                        logvar_OC9_s = self.logvar_layer_OC9_s(OC_s)
                elif i == 9:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC10_z = self.mean_layer_OC10_z(OC_z)
                        logvar_OC10_z = self.logvar_layer_OC10_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC10_s = self.mean_layer_OC10_s(OC_s)
                        logvar_OC10_s = self.logvar_layer_OC10_s(OC_s)
                elif i == 10:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC11_z = self.mean_layer_OC11_z(OC_z)
                        logvar_OC11_z = self.logvar_layer_OC11_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC11_s = self.mean_layer_OC11_s(OC_s)
                        logvar_OC11_s = self.logvar_layer_OC11_s(OC_s)
                elif i == 11:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC12_z = self.mean_layer_OC12_z(OC_z)
                        logvar_OC12_z = self.logvar_layer_OC12_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC12_s = self.mean_layer_OC12_s(OC_s)
                        logvar_OC12_s = self.logvar_layer_OC12_s(OC_s)
                elif i == 12:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC13_z = self.mean_layer_OC13_z(OC_z)
                        logvar_OC13_z = self.logvar_layer_OC13_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC13_s = self.mean_layer_OC13_s(OC_s)
                        logvar_OC13_s = self.logvar_layer_OC13_s(OC_s)
                elif i == 13:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC14_z = self.mean_layer_OC14_z(OC_z)
                        logvar_OC14_z = self.logvar_layer_OC14_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC14_s = self.mean_layer_OC14_s(OC_s)
                        logvar_OC14_s = self.logvar_layer_OC14_s(OC_s)
                elif i == 14:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_OC15_z = self.mean_layer_OC15_z(OC_z)
                        logvar_OC15_z = self.logvar_layer_OC15_z(OC_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_OC15_s = self.mean_layer_OC15_s(OC_s)
                        logvar_OC15_s = self.logvar_layer_OC15_s(OC_s)
                    
                    
                            
            "Prior regularization for X signals"
            if self.config.use_prior_regularization:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    mean_X_z = self.mean_layer_z(original_features_z)
                    logvar_X_z = self.logvar_layer_z(original_features_z)
                if self.config.dual_branched_latent or self.config.only_s_branch:
                    mean_X_s = self.mean_layer_s(original_features_s_agg)
                    logvar_X_s = self.logvar_layer_s(original_features_s_agg)

            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    combined_components_z = component_features_z.permute(1,2,0,3).reshape(batch_size,sequence_length,NoC*self.config.proj_codevector_dim_z)
                    projected_components_z = self.project_components_z(combined_components_z)
                if self.config.dual_branched_latent or self.config.only_s_branch:   
                    combined_components_s = component_features_s_agg.permute(1,2,0).reshape(batch_size,NoC_seq*self.config.proj_codevector_dim_s)
                    projected_components_s = self.project_components_s(combined_components_s)
                if self.config.use_prior_regularization:
                    if self.config.dual_branched_latent or self.config.only_z_branch:   
                        mean_proj_OCs_z = self.mean_layer_proj_OCs_z(projected_components_z)
                        logvar_proj_OCs_z = self.logvar_layer_proj_OCs_z(projected_components_z)
                    if self.config.dual_branched_latent or self.config.only_s_branch:
                        mean_proj_OCs_s = self.mean_layer_proj_OCs_s(projected_components_s)
                        logvar_proj_OCs_s = self.logvar_layer_proj_OCs_s(projected_components_s)

        if attention_mask is not None and self.config.fc_or_conv_s == "conv":
            # compute reduced attention_mask correponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )


        if self.config.dual_branched_latent or self.config.only_z_branch:   
            _,batch_size, sequence_length, hidden_size = component_features_z.shape
        else:
            if self.config.fc_or_conv_s == "conv":
                _,batch_size, sequence_length, hidden_size = component_features_s.shape
            elif self.config.fc_or_conv_s == "fc":
                _,batch_size, hidden_size = component_features_s.shape

        if self.config.fc_or_conv_s == "fc":
            sequence_length = mask_time_indices.shape[-1]
        #4b. Negative feature vectors according to sample_negative_indices - For decomposition loss
        eligible_indices = np.array([[i,j] for i in range(batch_size) for j in range(sequence_length) if mask_time_indices[i,j]])
        
        if (eligible_indices != eligible_indices).any():
            print("NaNs in eligible_indices - line 3238")

        "Compute decomposition loss and logits"
        "First latent variable"
        if self.config.dual_branched_latent or self.config.only_z_branch:   
            N_div_pos_z, N_div_neg_z, div_dict_z, div_pos_z, div_neg_z, ce_pos_z, ce_neg_z, avg_correlogram_z, ortho_dict_z, used_decomposition_features_z, used_unmasked_features_z,used_indices_z = self.compute_decomposition_loss(
                original_features_z,
                component_features_z,
                eligible_indices,
                N = self.config.max_frames_per_batch,
                temperature=self.config.contrastive_logits_temperature,
                divergence_type=self.config.divergence_type,
                reduction=self.config.decomp_loss_reduction,
            )
        "Second latent variable"
        if self.config.dual_branched_latent or self.config.only_s_branch:   
            N_div_pos_s, N_div_neg_s, div_dict_s, div_pos_s, div_neg_s, ce_pos_s, ce_neg_s, avg_correlogram_s, ortho_dict_s, used_decomposition_features_s, used_unmasked_features_s,used_indices_s = self.compute_decomposition_loss(
                original_features_s_agg,
                component_features_s_agg,
                eligible_indices,
                N = self.config.max_frames_per_batch,
                temperature=self.config.contrastive_logits_temperature,
                divergence_type=self.config.divergence_type,
                reduction=self.config.decomp_loss_reduction,
            )

        #Keep only projections that were used in the decomposition loss calculations
        #Careful not to break the grad graph here
        if self.config.project_OCs:
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                used_projected_components_z = torch.cat([projected_components_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
            if self.config.dual_branched_latent or self.config.only_s_branch:   
                used_projected_components_s = projected_components_s.clone()


        if self.config.dual_branched_latent or self.config.only_z_branch:   
            if type(ce_pos_z) == dict:
                decomp_loss_z = (
                    self.config.div_neg_weight * ce_neg_z +
                    self.config.weight_0_1 * ce_pos_z["ce_0_1"] + 
                    self.config.weight_0_2 * ce_pos_z["ce_0_2"] +
                    self.config.weight_0_3_and_above * sum([val for key,val in ce_pos_z.items() if int(key[-1]) > 2])
                )
                ce_pos_z = torch.sum(torch.stack([val for key,val in ce_pos_z.items()]))
            else:
                decomp_loss_z = self.config.div_pos_weight * ce_pos_z + self.config.div_neg_weight * ce_neg_z
        if self.config.dual_branched_latent or self.config.only_s_branch:
            if type(ce_pos_s) == dict:
                decomp_loss_s = (
                    self.config.div_neg_weight * ce_neg_s +
                    self.config.weight_0_1 * ce_pos_s["ce_0_1"] + 
                    self.config.weight_0_2 * ce_pos_s["ce_0_2"] +
                    self.config.weight_0_3_and_above * sum([val for key,val in ce_pos_s.items() if int(key[-1]) > 2])
                ) 
                ce_pos_s = torch.sum(torch.stack([val for key,val in ce_pos_s.items()]))
            else:
                decomp_loss_s = self.config.decomp_loss_s_weight*(self.config.div_pos_weight * ce_pos_s + self.config.div_neg_weight * ce_neg_s) #= torch.log((torch.exp(

        if self.config.use_prior_regularization:
            "Calculate loss with mu and logvar vectors only on 'masked' samples (same as used in decomposition loss)"                    
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                used_mean_X_z = torch.cat([mean_X_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
                used_logvar_X_z = torch.cat([logvar_X_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
            if self.config.dual_branched_latent or self.config.only_s_branch:
                if len(used_indices_s) == 0:
                    used_mean_X_s = mean_X_s.clone()
                    used_logvar_X_s = logvar_X_s.clone()
                else:
                    used_mean_X_s = torch.cat([mean_X_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)
                    used_logvar_X_s = torch.cat([logvar_X_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)
            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    used_mean_proj_OCs_z = torch.cat([mean_proj_OCs_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
                    used_logvar_proj_OCs_z = torch.cat([logvar_proj_OCs_z[i,inds,:] for i,inds in enumerate(used_indices_z)],dim = 0)
                if self.config.dual_branched_latent or self.config.only_s_branch:   
                    if len(used_indices_s) == 0:
                        used_mean_proj_OCs_s = mean_proj_OCs_s.clone()
                        used_logvar_proj_OCs_s = logvar_proj_OCs_s.clone()
                    else:
                        used_mean_proj_OCs_s = torch.cat([mean_proj_OCs_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)
                        used_logvar_proj_OCs_s = torch.cat([logvar_proj_OCs_s[i,inds,:] for i,inds in enumerate(used_indices_s)],dim = 0)

            "Prior regularization for Z - Use masked samples only"
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                # Create dictionaries to store means and logvars
                used_mean_OCs_z = {}
                used_logvar_OCs_z = {}
                
                # Process each OC based on the NoC value
                for oc_num in range(1, min(16, NoC + 1)):
                    # Variable names in the original code
                    mean_var_name = f"mean_OC{oc_num}_z"
                    logvar_var_name = f"logvar_OC{oc_num}_z"
                    
                    # Get the original variables dynamically using locals()
                    if mean_var_name in locals() and logvar_var_name in locals():
                        mean_val = locals()[mean_var_name]
                        logvar_val = locals()[logvar_var_name]
                        
                        # Process and store in dictionaries
                        used_mean_OCs_z[oc_num] = torch.cat([mean_val[i,inds,:] for i,inds in enumerate(used_indices_z)], dim=0)
                        used_logvar_OCs_z[oc_num] = torch.cat([logvar_val[i,inds,:] for i,inds in enumerate(used_indices_z)], dim=0)
                
                if NoC >= 16:
                    print("NoC >= 16 - Not supported, components 16 and above will not be regularized")

            "Prior regularization for S - Use all samples as they have been averaged"
            if self.config.dual_branched_latent or self.config.only_s_branch:
                # Create dictionaries to store means and logvars for s branch
                used_mean_OCs_s = {}
                used_logvar_OCs_s = {}
                
                # Process each OC based on the NoC value
                for oc_num in range(1, min(16, self.config.NoC_seq + 1)):
                    mean_var_name = f"mean_OC{oc_num}_s"
                    logvar_var_name = f"logvar_OC{oc_num}_s"
                    
                    # Get the original variables dynamically using locals()
                    if mean_var_name in locals() and logvar_var_name in locals():
                        # For s branch, we just clone the values
                        used_mean_OCs_s[oc_num] = locals()[mean_var_name].clone()
                        used_logvar_OCs_s[oc_num] = locals()[logvar_var_name].clone()
                
                if self.config.NoC_seq >= 16:
                    print("NoC_seq >= 16 - Not supported, components 16 and above will not be regularized")

            "Prepare mu and logvar vectors to be returned in the output"
            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:   
                    mu_projections_z = used_mean_proj_OCs_z.clone()
                    logvar_projections_z = used_logvar_proj_OCs_z.clone()
                if self.config.dual_branched_latent or self.config.only_s_branch:   
                    mu_projections_s = used_mean_proj_OCs_s.clone()
                    logvar_projections_s = used_logvar_proj_OCs_s.clone()
            
            if self.config.dual_branched_latent or self.config.only_z_branch:
                # Initialize with the first two components
                mu_components_z = torch.stack([used_mean_OCs_z[1], used_mean_OCs_z[2]], dim=0)
                logvar_components_z = torch.stack([used_logvar_OCs_z[1], used_logvar_OCs_z[2]], dim=0)
                
                # Add remaining components if available
                for oc_num in range(3, min(16, NoC + 1)):
                    if oc_num in used_mean_OCs_z:
                        mu_components_z = torch.cat([mu_components_z, used_mean_OCs_z[oc_num].unsqueeze(0)], dim=0)
                        logvar_components_z = torch.cat([logvar_components_z, used_logvar_OCs_z[oc_num].unsqueeze(0)], dim=0)

            if self.config.dual_branched_latent or self.config.only_s_branch:
                # Initialize with the first two components
                mu_components_s = torch.stack([used_mean_OCs_s[1], used_mean_OCs_s[2]], dim=0)
                logvar_components_s = torch.stack([used_logvar_OCs_s[1], used_logvar_OCs_s[2]], dim=0)
                
                # Add remaining components if available
                for oc_num in range(3, min(16, self.config.NoC_seq + 1)):
                    if oc_num in used_mean_OCs_s:
                        mu_components_s = torch.cat([mu_components_s, used_mean_OCs_s[oc_num].unsqueeze(0)], dim=0)
                        logvar_components_s = torch.cat([logvar_components_s, used_logvar_OCs_s[oc_num].unsqueeze(0)], dim=0)

            "Compute prior regularization loss based on KL divergence between prior distribution and unit Gaussian"
            #prior_loss_X = self.compute_prior_reg_loss(used_mean_X,used_logvar_X,prior_type="gaussian")
            if self.config.project_OCs:
                if self.config.dual_branched_latent or self.config.only_z_branch:
                    prior_loss_proj_OCs_z = self.compute_prior_reg_loss(used_mean_proj_OCs_z,used_logvar_proj_OCs_z) 
                if self.config.dual_branched_latent or self.config.only_s_branch:
                    prior_loss_proj_OCs_s = self.compute_prior_reg_loss(used_mean_proj_OCs_s,used_logvar_proj_OCs_s) 
    
            if self.config.dual_branched_latent or self.config.only_z_branch:
                # Initialize prior loss dictionary for z branch
                prior_loss_OCs_z = {}
                
                # Calculate prior loss for each component
                for oc_num in range(1, min(16, NoC + 1)):
                    if oc_num in used_mean_OCs_z:
                        prior_loss_OCs_z[oc_num] = self.compute_prior_reg_loss(
                            used_mean_OCs_z[oc_num], 
                            used_logvar_OCs_z[oc_num], 
                            prior_type="gaussian"
                        )
                    else:
                        prior_loss_OCs_z[oc_num] = torch.tensor(0, device=extract_features.device)


            if self.config.dual_branched_latent or self.config.only_s_branch:
                # Initialize prior loss dictionary for s branch
                prior_loss_OCs_s = {}
                
                # Calculate prior loss for each component
                for oc_num in range(1, min(16, self.config.NoC_seq + 1)):
                    if oc_num in used_mean_OCs_s:
                        prior_loss_OCs_s[oc_num] = self.compute_prior_reg_loss(
                            used_mean_OCs_s[oc_num], 
                            used_logvar_OCs_s[oc_num], 
                            prior_type="gaussian"
                        )
                    else:
                        prior_loss_OCs_s[oc_num] = torch.tensor(0, device=extract_features.device)
                
        else:
            if self.config.dual_branched_latent or self.config.only_z_branch:   
                loss_z = decomp_loss_z.clone()
            if self.config.dual_branched_latent or self.config.only_s_branch:
                loss_s = decomp_loss_s.clone()
            if self.config.dual_branched_latent:
                loss = loss_z + loss_s
            elif self.config.only_z_branch:
                loss_s = torch.tensor(0,device=extract_features.device)
                loss = loss_z.clone()
            elif self.config.only_s_branch:
                loss_z = torch.tensor(0,device=extract_features.device)
                loss = loss_s.clone()
            
        
        if self.config.dual_branched_latent or self.config.only_z_branch: 
            mu_originals_z = used_mean_X_z.clone()
            logvar_originals_z = used_logvar_X_z.clone()
        if self.config.dual_branched_latent or self.config.only_s_branch: 
            mu_originals_s = used_mean_X_s.clone()
            logvar_originals_s = used_logvar_X_s.clone()

        "Supervised Loss"

        if self.config.only_z_branch or (self.config.dual_branched_latent and not self.config.aggregate_branch_features):
            "Use only indices used in other losses"
            labels_z = torch.cat([labels_z[i,inds] for i,inds in enumerate(used_indices_z)],dim = 0)

            "Take as input concatenated latent spaces of OCs and X"
            input_features_z = torch.cat((mu_originals_z.unsqueeze(0),mu_components_z), dim=0)
            input_features_z = input_features_z.reshape(-1,input_features_z.shape[0]*input_features_z.shape[-1])
            "Feed features to the classifier"
            logits_z = self.classification_head_z(input_features_z)

            supervised_loss_z, logits_z, labels_z = self.compute_supervised_loss(
                logits=logits_z,
                labels=labels_z,
                loss_type=self.config.supervised_loss_type,
                label_smoothing=0.0,
                class_weights=None,
                reduction=self.config.decomp_loss_reduction,
            )

            "bring the supervised loss to the same scale as the other losses"
            supervised_loss_z = self.config.supervised_loss_rel_weight * supervised_loss_z

        if self.config.only_s_branch or (self.config.dual_branched_latent and not self.config.aggregate_branch_features):
            input_features_s = torch.cat((mu_originals_s.unsqueeze(0),mu_components_s), dim=0)
            input_features_s = input_features_s.reshape(-1,input_features_s.shape[0]*input_features_s.shape[-1])
            "Feed features to the classifier"
            logits_s = self.classification_head_s(input_features_s)

            supervised_loss_s, logits_s, labels_s = self.compute_supervised_loss(
                logits=logits_s,
                labels=labels_s,
                loss_type=self.config.supervised_loss_type,
                label_smoothing=0.0,
                class_weights=None,
                reduction=self.config.decomp_loss_reduction,
            )

            "bring the supervised loss to the same scale as the other losses"
            supervised_loss_s = self.config.supervised_loss_rel_weight * supervised_loss_s

        if self.config.dual_branched_latent and self.config.aggregate_branch_features:
            "Use only indices used in other losses"
            labels_z = torch.cat([labels_z[i,inds] for i,inds in enumerate(used_indices_z)],dim = 0)

            if mu_originals_s.shape[0] == 1:
                "batch size of 1 will fail the below repeat operations"
                mu_originals_s = mu_originals_s.unsqueeze(0)
                mu_components_s = mu_components_s.unsqueeze(1)

            "Concatenate the context to entries of the batch"         
            orig_features = torch.cat([torch.cat([mu_originals_z[inds,:],mu_originals_s[i].repeat(len(inds),1)],dim=-1) for i,inds in enumerate(used_indices_z)],dim = 0)
            dec_features = torch.cat([torch.cat([mu_components_z[:,inds,:],mu_components_s[:,0,:].repeat(1,len(inds),1).reshape(mu_components_s.shape[0],len(inds),-1)],dim=-1) for i,inds in enumerate(used_indices_z)],dim = 1)
            input_features_z = torch.cat((orig_features.unsqueeze(0),dec_features), dim=0)
            "Make the aggregate latent by concatenating components and original"
            input_features_z = input_features_z.reshape(-1,input_features_z.shape[0]*input_features_z.shape[-1])

            if mu_originals_s.shape[0] == 1:
                "resqueeze back to original shape"
                mu_originals_s = mu_originals_s.squeeze(0)
                mu_components_s = mu_components_s.squeeze(1)

            "Feed features to the classifier"
            logits_z = self.classification_head_z(input_features_z)

            supervised_loss_z, logits_z, labels_z = self.compute_supervised_loss(
                logits=logits_z,
                labels=labels_z,
                loss_type=self.config.supervised_loss_type,
                label_smoothing=0.0,
                class_weights=None,
                reduction=self.config.decomp_loss_reduction,
            )

            "bring the supervised loss to the same scale as the other losses"
            supervised_loss_z = self.config.supervised_loss_rel_weight * supervised_loss_z

        "Final total loss calculation: supervised + prior + decomposition"
                
        if self.config.dual_branched_latent or self.config.only_z_branch:   
            if not self.config.project_OCs:
                # Use sum on dictionary values instead of individual variables
                prior_loss_z = self.config.prior_reg_weighting_z * self.config.beta_kl_prior_z * (
                    sum(prior_loss_OCs_z.values()) / NoC
                )
            else:
                # Include projected OCs in the calculation
                prior_loss_z = self.config.prior_reg_weighting_z * self.config.beta_kl_prior_z * (
                    prior_loss_proj_OCs_z + sum(prior_loss_OCs_z.values())
                ) / (NoC + 1)
        
            "Weight the sum of losses"
            loss_z = self.config.div_loss_total_weight * decomp_loss_z + self.config.prior_reg_loss_total_weight * prior_loss_z + self.config.supervised_loss_weight * supervised_loss_z

        if self.config.dual_branched_latent or self.config.only_s_branch:   
            if not self.config.project_OCs:
                # Use sum on dictionary values instead of individual variables
                prior_loss_s = self.config.prior_reg_weighting_s * self.config.beta_kl_prior_s * (
                    sum(prior_loss_OCs_s.values()) / self.config.NoC_seq
                )
            else:
                # Include projected OCs in the calculation
                prior_loss_s = self.config.prior_reg_weighting_s * self.config.beta_kl_prior_s * (
                    prior_loss_proj_OCs_s + sum(prior_loss_OCs_s.values())
                ) / (self.config.NoC_seq + 1)

            "Weight the sum of losses"
            if self.config.aggregate_branch_features:
                "The total supervised loss from the aggregated features propagates back to both branches - No need to add it again here"
                loss_s = self.config.div_loss_total_weight * decomp_loss_s + self.config.prior_reg_loss_total_weight * prior_loss_s 
            else:
                loss_s = self.config.div_loss_total_weight * decomp_loss_s + self.config.prior_reg_loss_total_weight * prior_loss_s + self.config.supervised_loss_weight * supervised_loss_s
        
        if self.config.dual_branched_latent:                        
            loss = loss_z + loss_s  
        elif self.config.only_z_branch:
            loss_s = torch.tensor(0,device=extract_features.device)
            loss = loss_z.clone()
        elif self.config.only_s_branch:
            loss_z = torch.tensor(0,device=extract_features.device)
            loss = loss_s.clone()

        "When decoder is added, we are going to need the below - Reparameterization trick"
        #z_decoder_input = self.sampling(z_mean, z_log_var)
        
        if not self.config.project_OCs:
            mu_projections_z = None
            logvar_projections_z = None
            used_projected_components_z = None
            mu_projections_s = None
            logvar_projections_s = None
            used_projected_components_s = None   

        "Return"
        if self.config.only_z_branch:   
            return DecVAEForSupervisedFineTuningOutput(
                loss=loss,
                loss_z = loss_z,
                hidden_states_z = outputs_z.hidden_states,
                unmasked_frames_features_z=original_features_z,
                decomposition_features_z=component_features_z,
                used_decomposition_features_z = used_decomposition_features_z, 
                used_unmasked_features_z = used_unmasked_features_z,
                used_projected_components_z = used_projected_components_z,
                mu_originals_z = mu_originals_z,
                logvar_originals_z = logvar_originals_z,
                mu_projections_z = mu_projections_z,
                logvar_projections_z = logvar_projections_z,
                mu_components_z = mu_components_z,
                logvar_components_z = logvar_components_z,
                used_indices_z = used_indices_z,
                attentions=outputs_z.attentions,     
                decomposition_loss_z = decomp_loss_z,
                prior_loss_z = prior_loss_z,      
                supervised_loss_z= supervised_loss_z,
                input_features_to_classifier_z=input_features_z,
                logits_z = logits_z,
                labels_z = labels_z,
                divergence_dict_z = div_dict_z,
                div_pos_z = div_pos_z,
                div_neg_z = div_neg_z,
                ce_pos_z = ce_pos_z,
                ce_neg_z = ce_neg_z,
                avg_correlogram_z = avg_correlogram_z,
                avg_correlogram_td_z = outputs_z.avg_correlogram,
                orthogonality_dict_z = ortho_dict_z,
                orthogonality_dict_td_z = outputs_z.ortho_dict,
                N_div_pos_z = N_div_pos_z,
                N_div_neg_z = N_div_neg_z,
                mask_time_indices = mask_time_indices,
            )
        elif self.config.only_s_branch:
            return DecVAEForSupervisedFineTuningOutput(
                loss=loss,
                loss_s = loss_s,
                hidden_states_s = hidden_states_s,
                unmasked_frames_features_s=original_features_s_agg,
                decomposition_features_s=component_features_s_agg,
                used_decomposition_features_s = used_decomposition_features_s,
                used_unmasked_features_s = used_unmasked_features_s,
                used_projected_components_s = used_projected_components_s,
                mu_originals_s = mu_originals_s,
                logvar_originals_s = logvar_originals_s,
                mu_projections_s = mu_projections_s,
                logvar_projections_s = logvar_projections_s,
                mu_components_s = mu_components_s,
                logvar_components_s = logvar_components_s,
                used_indices_s = used_indices_s,
                attentions=None,  
                decomposition_loss_s = decomp_loss_s,   
                prior_loss_s = prior_loss_s,
                supervised_loss_s= supervised_loss_s,
                input_features_to_classifier_s=input_features_s,
                logits_s = logits_s,
                labels_s = labels_s,
                divergence_dict_s = div_dict_s,
                div_pos_s = div_pos_s,
                div_neg_s = div_neg_s,
                ce_pos_s = ce_pos_s,
                ce_neg_s = ce_neg_s,
                avg_correlogram_s = avg_correlogram_s,
                avg_correlogram_td_s = None,
                orthogonality_dict_s = ortho_dict_s,
                orthogonality_dict_td_s = None,
                N_div_pos_s = N_div_pos_s,
                N_div_neg_s = N_div_neg_s,
                mask_time_indices = mask_time_indices,
            )
        elif self.config.dual_branched_latent and not self.config.aggregate_branch_features:   
            return DecVAEForSupervisedFineTuningOutput(
                    loss=loss,
                    loss_z = loss_z,
                    loss_s = loss_s,
                    hidden_states_z = outputs_z.hidden_states,
                    hidden_states_s = hidden_states_s,
                    unmasked_frames_features_z=original_features_z,
                    unmasked_frames_features_s=original_features_s_agg,
                    decomposition_features_z=component_features_z,
                    decomposition_features_s=component_features_s_agg,
                    used_decomposition_features_z = used_decomposition_features_z, 
                    used_decomposition_features_s = used_decomposition_features_s,
                    used_unmasked_features_z = used_unmasked_features_z,
                    used_unmasked_features_s = used_unmasked_features_s,
                    used_projected_components_z = used_projected_components_z,
                    used_projected_components_s = used_projected_components_s,
                    mu_originals_z = mu_originals_z,
                    mu_originals_s = mu_originals_s,
                    logvar_originals_z = logvar_originals_z,
                    logvar_originals_s = logvar_originals_s,
                    mu_projections_z = mu_projections_z,
                    mu_projections_s = mu_projections_s,
                    logvar_projections_z = logvar_projections_z,
                    logvar_projections_s = logvar_projections_s,
                    mu_components_z = mu_components_z,
                    mu_components_s = mu_components_s,
                    logvar_components_z = logvar_components_z,
                    logvar_components_s = logvar_components_s,
                    used_indices_z = used_indices_z,
                    used_indices_s = used_indices_s,
                    attentions=outputs_z.attentions,   
                    decomposition_loss_z = decomp_loss_z,
                    decomposition_loss_s = decomp_loss_s,
                    supervised_loss_z = supervised_loss_z,
                    supervised_loss_s = supervised_loss_s,
                    input_features_to_classifier_z = input_features_z,
                    input_features_to_classifier_s = input_features_s,
                    logits_z = logits_z,
                    logits_s = logits_s,
                    labels_z = labels_z,
                    labels_s = labels_s,
                    prior_loss_z = prior_loss_z,
                    prior_loss_s = prior_loss_s,
                    divergence_dict_z = div_dict_z,
                    divergence_dict_s = div_dict_s,
                    div_pos_z = div_pos_z,
                    div_pos_s = div_pos_s,
                    div_neg_z = div_neg_z,
                    div_neg_s = div_neg_s,
                    ce_pos_z = ce_pos_z,
                    ce_pos_s = ce_pos_s,
                    ce_neg_z = ce_neg_z,
                    ce_neg_s = ce_neg_s,
                    avg_correlogram_z = avg_correlogram_z,
                    avg_correlogram_s = avg_correlogram_s,
                    avg_correlogram_td_z = outputs_z.avg_correlogram,
                    avg_correlogram_td_s = None,
                    orthogonality_dict_z = ortho_dict_z,
                    orthogonality_dict_s = ortho_dict_s,
                    orthogonality_dict_td_z = outputs_z.ortho_dict,
                    orthogonality_dict_td_s = None,
                    N_div_pos_z = N_div_pos_z,
                    N_div_pos_s = N_div_pos_s,
                    N_div_neg_z = N_div_neg_z,
                    N_div_neg_s = N_div_neg_s,
                    mask_time_indices = mask_time_indices,
                )
        elif self.config.dual_branched_latent and self.config.aggregate_branch_features:
            return DecVAEForSupervisedFineTuningOutput(
                    loss=loss,
                    loss_z = loss_z,
                    loss_s = loss_s,
                    hidden_states_z = outputs_z.hidden_states,
                    hidden_states_s = hidden_states_s,
                    unmasked_frames_features_z=original_features_z,
                    unmasked_frames_features_s=original_features_s_agg,
                    decomposition_features_z=component_features_z,
                    decomposition_features_s=component_features_s_agg,
                    used_decomposition_features_z = used_decomposition_features_z, 
                    used_decomposition_features_s = used_decomposition_features_s,
                    used_unmasked_features_z = used_unmasked_features_z,
                    used_unmasked_features_s = used_unmasked_features_s,
                    used_projected_components_z = used_projected_components_z,
                    used_projected_components_s = used_projected_components_s,
                    mu_originals_z = mu_originals_z,
                    mu_originals_s = mu_originals_s,
                    logvar_originals_z = logvar_originals_z,
                    logvar_originals_s = logvar_originals_s,
                    mu_projections_z = mu_projections_z,
                    mu_projections_s = mu_projections_s,
                    logvar_projections_z = logvar_projections_z,
                    logvar_projections_s = logvar_projections_s,
                    mu_components_z = mu_components_z,
                    mu_components_s = mu_components_s,
                    logvar_components_z = logvar_components_z,
                    logvar_components_s = logvar_components_s,
                    used_indices_z = used_indices_z,
                    used_indices_s = used_indices_s,
                    attentions=outputs_z.attentions,   
                    decomposition_loss_z = decomp_loss_z,
                    decomposition_loss_s = decomp_loss_s,
                    supervised_loss_z = supervised_loss_z,
                    input_features_to_classifier_z = input_features_z,
                    logits_z = logits_z,
                    labels_z = labels_z,
                    prior_loss_z = prior_loss_z,
                    prior_loss_s = prior_loss_s,
                    divergence_dict_z = div_dict_z,
                    divergence_dict_s = div_dict_s,
                    div_pos_z = div_pos_z,
                    div_pos_s = div_pos_s,
                    div_neg_z = div_neg_z,
                    div_neg_s = div_neg_s,
                    ce_pos_z = ce_pos_z,
                    ce_pos_s = ce_pos_s,
                    ce_neg_z = ce_neg_z,
                    ce_neg_s = ce_neg_s,
                    avg_correlogram_z = avg_correlogram_z,
                    avg_correlogram_s = avg_correlogram_s,
                    avg_correlogram_td_z = outputs_z.avg_correlogram,
                    avg_correlogram_td_s = None,
                    orthogonality_dict_z = ortho_dict_z,
                    orthogonality_dict_s = ortho_dict_s,
                    orthogonality_dict_td_z = outputs_z.ortho_dict,
                    orthogonality_dict_td_s = None,
                    N_div_pos_z = N_div_pos_z,
                    N_div_pos_s = N_div_pos_s,
                    N_div_neg_z = N_div_neg_z,
                    N_div_neg_s = N_div_neg_s,
                    mask_time_indices = mask_time_indices,
                )

@dataclass
class DecVAEForSupervisedFineTuningOutput(ModelOutput):
    """
    Output type of [`DecVAEForSupervisedFineTuning`].

    This class defines the structure of the output from the DecVAEForSupervisedFineTuning model,
    encapsulating loss values, hidden states, and attention values from the model.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss=True`):
            Total loss for the model.
        loss_z (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss value for the Z branch.
        loss_s (`torch.FloatTensor` of shape `(1,)`, *optional*):
            Loss value for the S branch.
        hidden_states_z (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from the Z branch of the model.
        hidden_states_s (`tuple(torch.FloatTensor)`, *optional*):
            Hidden states from the S branch of the model.
        attentions (`tuple(torch.FloatTensor)`, *optional*):
            Attention weights from the model.
        unmasked_frames_features_z (`torch.FloatTensor`):
            Unmasked frame features from the Z branch.
        unmasked_frames_features_s (`torch.FloatTensor`):
            Unmasked frame features from the S branch.
        decomposition_features_z (`torch.FloatTensor`):
            Decomposition features from the Z branch.
        decomposition_features_s (`torch.FloatTensor`):
            Decomposition features from the S branch.
        used_decomposition_features_z (`torch.FloatTensor`):
            Used decomposition features from the Z branch.
        used_decomposition_features_s (`torch.FloatTensor`):
            Used decomposition features from the S branch.
        used_unmasked_features_z (`torch.FloatTensor`):
            Used unmasked features from the Z branch.
        used_unmasked_features_s (`torch.FloatTensor`):
            Used unmasked features from the S branch.
        used_projected_components_z (`torch.FloatTensor`):
            Used projected components from the Z branch.
        used_projected_components_s (`torch.FloatTensor`):
            Used projected components from the S branch.
        mu_originals_z (`torch.FloatTensor`):
            Mean vectors for original features from the Z branch.
        mu_originals_s (`torch.FloatTensor`):
            Mean vectors for original features from the S branch.
        logvar_originals_z (`torch.FloatTensor`):
            Log variance vectors for original features from the Z branch.
        logvar_originals_s (`torch.FloatTensor`):
            Log variance vectors for original features from the S branch.
        mu_projections_z (`torch.FloatTensor`):
            Mean vectors for projections from the Z branch.
        mu_projections_s (`torch.FloatTensor`):
            Mean vectors for projections from the S branch.
        logvar_projections_z (`torch.FloatTensor`):
            Log variance vectors for projections from the Z branch.
        logvar_projections_s (`torch.FloatTensor`):
            Log variance vectors for projections from the S branch.
        mu_components_z (`torch.FloatTensor`):
            Mean vectors for components from the Z branch.
        mu_components_s (`torch.FloatTensor`):
            Mean vectors for components from the S branch.
        logvar_components_z (`torch.FloatTensor`):
            Log variance vectors for components from the Z branch.
        logvar_components_s (`torch.FloatTensor`):
            Log variance vectors for components from the S branch.
        used_indices_z (`list`):
            Indices of used features from the Z branch.
        used_indices_s (`list`):
            Indices of used features from the S branch.
        decomposition_loss_z (`torch.FloatTensor`):
            Decomposition loss from the Z branch.
        decomposition_loss_s (`torch.FloatTensor`):
            Decomposition loss from the S branch.
        prior_loss_z (`torch.FloatTensor`):
            Prior regularization loss from the Z branch.
        prior_loss_s (`torch.FloatTensor`):
            Prior regularization loss from the S branch.
        supervised_loss_z (`torch.FloatTensor`):
            Supervised loss from the Z branch.
        supervised_loss_s (`torch.FloatTensor`):
            Supervised loss from the S branch.
        input_features_to_classifier_z (`torch.FloatTensor`):
            Input features for the classifier head of the Z branch - These can be aggregated features from both branches.
        input_features_to_classifier_s (`torch.FloatTensor`):
            Input features for the classifier head of the S branch.
        logits_z (`torch.FloatTensor`):
            Output classifier logits from the Z branch.
        logits_s (`torch.FloatTensor`):
            Output classifier logits from the S branch.
        labels_z (`torch.FloatTensor`):
            Ground truth labels in the format used for calculating the supervised loss for the Z branch.
        labels_s (`torch.FloatTensor`):
            Ground truth labels in the format used for calculating the supervised loss for the S branch.
        divergence_dict_z (`dict`):
            Dictionary of divergence values from the Z branch.
        divergence_dict_s (`dict`):
            Dictionary of divergence values from the S branch.
        div_pos_z (`torch.FloatTensor`):
            Positive divergence from the Z branch.
        div_pos_s (`torch.FloatTensor`):
            Positive divergence from the S branch.
        div_neg_z (`torch.FloatTensor`):
            Negative divergence from the Z branch.
        div_neg_s (`torch.FloatTensor`):
            Negative divergence from the S branch.
        ce_pos_z (`torch.FloatTensor`):
            Positive cross entropy from the Z branch.
        ce_pos_s (`torch.FloatTensor`):
            Positive cross entropy from the S branch.
        ce_neg_z (`torch.FloatTensor`):
            Negative cross entropy from the Z branch.
        ce_neg_s (`torch.FloatTensor`):
            Negative cross entropy from the S branch.
        avg_correlogram_z (`torch.FloatTensor`):
            Average correlogram from the Z branch.
        avg_correlogram_s (`torch.FloatTensor`):
            Average correlogram from the S branch.
        avg_correlogram_td_z (`torch.FloatTensor`):
            Average time-domain correlogram from the Z branch.
        avg_correlogram_td_s (`torch.FloatTensor`):
            Average time-domain correlogram from the S branch.
        orthogonality_dict_z (`dict`):
            Dictionary of orthogonality metrics from the Z branch.
        orthogonality_dict_s (`dict`):
            Dictionary of orthogonality metrics from the S branch.
        orthogonality_dict_td_z (`dict`):
            Dictionary of time-domain orthogonality metrics from the Z branch.
        orthogonality_dict_td_s (`dict`):
            Dictionary of time-domain orthogonality metrics from the S branch.
        N_div_pos_z (`torch.LongTensor`):
            Number of positive divergence samples from the Z branch.
        N_div_pos_s (`torch.LongTensor`):
            Number of positive divergence samples from the S branch.
        N_div_neg_z (`torch.LongTensor`):
            Number of negative divergence samples from the Z branch.
        N_div_neg_s (`torch.LongTensor`):
            Number of negative divergence samples from the S branch.
        mask_time_indices (`torch.BoolTensor`):
            Indices of masked time frames.
    """

    loss: Optional[torch.FloatTensor] = None
    loss_z: Optional[torch.FloatTensor] = None
    loss_s: Optional[torch.FloatTensor] = None
    hidden_states_z: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states_s: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    unmasked_frames_features_z: torch.FloatTensor = None
    unmasked_frames_features_s: torch.FloatTensor = None
    decomposition_features_z: torch.FloatTensor = None
    decomposition_features_s: torch.FloatTensor = None
    used_decomposition_features_z: torch.FloatTensor = None
    used_decomposition_features_s: torch.FloatTensor = None
    used_unmasked_features_z: torch.FloatTensor = None
    used_unmasked_features_s: torch.FloatTensor = None
    used_projected_components_z: torch.FloatTensor = None
    used_projected_components_s: torch.FloatTensor = None
    mu_originals_z: torch.FloatTensor = None
    mu_originals_s: torch.FloatTensor = None
    logvar_originals_z: torch.FloatTensor = None
    logvar_originals_s: torch.FloatTensor = None
    mu_projections_z: torch.FloatTensor = None
    mu_projections_s: torch.FloatTensor = None
    logvar_projections_z: torch.FloatTensor = None
    logvar_projections_s: torch.FloatTensor = None
    mu_components_z: torch.FloatTensor = None
    mu_components_s: torch.FloatTensor = None
    logvar_components_z: torch.FloatTensor = None
    logvar_components_s: torch.FloatTensor = None
    used_indices_z: Optional[list] = None
    used_indices_s: Optional[list] = None
    decomposition_loss_z: Optional[torch.FloatTensor] = None
    decomposition_loss_s: Optional[torch.FloatTensor] = None
    prior_loss_z: Optional[torch.FloatTensor] = None
    prior_loss_s: Optional[torch.FloatTensor] = None
    supervised_loss_z: Optional[torch.FloatTensor] = None
    supervised_loss_s: Optional[torch.FloatTensor] = None
    input_features_to_classifier_z: Optional[torch.FloatTensor] = None
    input_features_to_classifier_s: Optional[torch.FloatTensor] = None
    logits_z: Optional[torch.FloatTensor] = None
    logits_s: Optional[torch.FloatTensor] = None
    labels_z: Optional[torch.FloatTensor] = None
    labels_s: Optional[torch.FloatTensor] = None
    divergence_dict_z: Optional[dict] = None
    divergence_dict_s: Optional[dict] = None
    div_pos_z: Optional[torch.FloatTensor] = None
    div_pos_s: Optional[torch.FloatTensor] = None
    div_neg_z: Optional[torch.FloatTensor] = None
    div_neg_s: Optional[torch.FloatTensor] = None
    ce_pos_z: Optional[torch.FloatTensor] = None
    ce_pos_s: Optional[torch.FloatTensor] = None
    ce_neg_z: Optional[torch.FloatTensor] = None
    ce_neg_s: Optional[torch.FloatTensor] = None
    avg_correlogram_z: Optional[torch.FloatTensor] = None
    avg_correlogram_s: Optional[torch.FloatTensor] = None
    avg_correlogram_td_z: Optional[torch.FloatTensor] = None
    avg_correlogram_td_s: Optional[torch.FloatTensor] = None
    orthogonality_dict_z: Optional[dict] = None
    orthogonality_dict_s: Optional[dict] = None
    orthogonality_dict_td_z: Optional[dict] = None
    orthogonality_dict_td_s: Optional[dict] = None
    N_div_pos_z: Optional[torch.LongTensor] = None
    N_div_pos_s: Optional[torch.LongTensor] = None
    N_div_neg_z: Optional[torch.LongTensor] = None
    N_div_neg_s: Optional[torch.LongTensor] = None
    mask_time_indices: Optional[torch.BoolTensor] = None



__all__ = [
    "DecVAEForPreTraining",
    "Dec2VecModel",
    "DecVAEForSupervisedFineTuning"
]