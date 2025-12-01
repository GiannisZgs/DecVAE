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
"""DecVAE model configuration - Adapted from HuggingFace library - transformers/configuration_utils/PretrainedConfig"""
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.

import functools
import operator
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DecVAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DecVAEForPretrainingModel`]
    implemented in the style of Wav2Vec2Config. It is used to instantiate a
    DecVAE model according to the specified arguments, defining the model architecture. 
    
    This configuration class contains arguments for instantiating a wav2vec2 configuration
    with the defaults. This yields a similar configuration to that of the Wav2Vec2
    [facebook/wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h) architecture.
    The wav2vec2 configuration is used to instantiate the DecVAE backbone encoder.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
                
    `"Decomposition Model"' D_w^C Args:
        decomp_to_perform (`str`, *optional*, defaults to `"filter"`):
            Type of decomposition to perform in the DecompositionModule D_w^C. 
            Options are `"filter"`, `"vmd"`, `"emd"`, `"ewt"`.
        fs (`int`, *optional*, defaults to 16000):
            Sampling frequency of the input audio signals.
        freq_groups (`List[List[int]]`, *optional*):
            List of frequency bands (groups) to consider for decomposition; used to identify missing components.
        receptive_field (`float`, *optional*, defaults to 0.025):
            Receptive field (in seconds) of the DecVAE encoder.
        stride (`float`, *optional*, defaults to 0.02):
            Stride (in seconds) of the DecVAE encoder.
        lower_speech_freq (`int`, *optional*, defaults to 50):
            Lower frequency (in Hz) to consider speech components.
        higher_speech_freq (`int`, *optional*, defaults to 7500):
            Higher frequency (in Hz) to consider speech components.
        max_silence_freq (`int`, *optional*, defaults to 250):
            Maximum frequency (in Hz) to consider a frame silent and mark as empty.
        notch_band_low (`int`, *optional*, defaults to 55):
            Lower frequency (in Hz) of the notch band to apply when detecting spectral peaks.
        notch_band_high (`int`, *optional*, defaults to 70):
            Higher frequency (in Hz) of the notch band to apply when detecting spectral peaks.
        detection_intervals (`int`, *optional*, defaults to 6):
            Number of intervals to split the frequency axis into when detecting spectral peaks.
        spec_amp_tolerance (`float`, *optional*, defaults to 5e-6):
            Amplitude tolerance to consider a spectral peak as valid when detecting observed components in each frame.
        spec_amp_tolerance_seq (`float`, *optional*, defaults to 1e-6):
            Amplitude tolerance to consider a spectral peak as valid when detecting observed components in the entire sequence
        global_thres (`float`, *optional*, defaults to 0.01):
            Global threshold to consider a spectral peak as valid when detecting observed components.
        power_law (`float`, *optional*, defaults to 1.7):
            Power law exponent to apply to the spectrum when detecting observed components.
        nfft (`int`, *optional*, defaults to 512):
            Number of FFT points to consider in spectral operations inside the DecompositionModule.
        min_distance (`int`, *optional*, defaults to 300):
            Minimum distance (in Hz) between detected spectral peaks.
        peak_bandwidth (`int`, *optional*, defaults to 500):
            Minimum bandwidth (in Hz) of detected spectral peaks.
        prom_thres (`float`, *optional*, defaults to 0.05):
            Prominence threshold to consider a spectral peak as valid when detecting observed components.
        N_peaks_to_select (`int`, *optional*, defaults to 2):
            Number of spectral peaks to select as observed components in each frequency interval.
        buttord (`int`, *optional*, defaults to 4):
            Order of the Butterworth filter used in the FilterDecomposition (FD) to apply when detecting spectral peaks.
        remove_silence (`bool`, *optional*, defaults to `False`):
            Whether to perform a silence check and mark silent frames in the input audio.
        NoC (`int`, *optional*, defaults to 3):
            Number of oscillatory components to decompose each frame in the decomposition.
        NoC_seq (`int`, *optional*, defaults to 3):
            Number of oscillatory components to decompose the entire sequence in the decomposition.
        group_OCs_by_frame (`str`, *optional*, defaults to `"equally_distribute"`):
            Strategy to group observed components (OCs) when decomposing each frame. Options are `"high_freqs_first"`, `"low_freqs_first"`, `"equally_distribute"`.
        group_OCs_by_seq (`str`, *optional*, defaults to `"equally_distribute"`):
            Strategy to group observed components (OCs) when decomposing the entire sequence. Options are `"high_freqs_first"`, `"low_freqs_first"`, `"equally_distribute"`, `"top_k"`.
        
        
    ''Training Objective'' args:
        max_frames_per_batch (`int`, *optional*, defaults to 100):
            Maximum number of frames to use in each batch when computing the SSL decomposition loss.
        decomp_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Reduction method to apply to the SSL decomposition loss. Options are `"sum"` and `"mean"`.
        div_pos_weight (`float`, *optional*, defaults to 0.3333):
            Weight for all positive pairs in the divergence loss when computing the SSL decomposition loss.
        div_neg_weight (`float`, *optional*, defaults to 0.3333):
            Weight for all negative pairs in the divergence loss when computing the SSL decomposition loss.
        weight_0_1 (`float`, *optional*, defaults to 0.3333):
            Weight to apply to the first positive pair (original-1st component) in the SSL decomposition loss.
        weight_0_2 (`float`, *optional*, defaults to 0.3333):
            Weight to apply to the second positive pair (original-2nd component) in the SSL decomposition loss.
        weight_0_3_and_above (`float`, *optional*, defaults to 0.3333):
            Weight to apply to the third and above positive pairs (original-3rd and above components) in the SSL decomposition loss.
        decomp_loss_s_weight (`float`, *optional*, defaults to 1.0):
            Multiplier on the SSL decomposition loss component for the latent sequence branch S to bring in same scale with other losses.
        use_prior_regularization (`bool`, *optional*, defaults to `False`):
            Whether to use prior regularization in the SSL decomposition loss.
        beta_kl_prior_z (`float`, *optional*, defaults to 1.0):
            Weight of the KL divergence prior regularization term for the latent space Z in the Gaussian prior approximation loss.
        beta_kl_prior_s (`float`, *optional*, defaults to 1.0):
            Weight of the KL divergence prior regularization term for the latent space S in the Gaussian prior approximation loss.
        prior_reg_weighting_z (`int`, *optional*, defaults to 10):
            Multiplier on the prior approximation loss of the latent space Z to bring in same scale with other losses.
        prior_reg_weighting_s (`int`, *optional*, defaults to 400):
            Multiplier on the prior approximation loss of the latent space S to bring in same scale with other losses.
        divergence_type (`str`, *optional*, defaults to `"js"`):
            Type of divergence to use when computing the SSL decomposition loss. Only supported option at the moment is `"js"` (Jensen-Shannon divergence).

    ''DecVAE'' and ''Dec2Vec'' encoder args:
        dual_branched_latent (`bool`, *optional*, defaults to `False`):
            Whether to use a dual-branched latent space (Z and S) in the DecVAE and Dec2Vec encoders.
        only_z_branch (`bool`, *optional*, defaults to `False`):
            Whether to create a single-branched (Z) DecVAE and Dec2Vec encoder.
        only_s_branch (`bool`, *optional*, defaults to `False`):
            Whether to create a single-branched (S) DecVAE and Dec2Vec encoder.
        project_OCs (`bool`, *optional*, defaults to `False`):
            Whether to use the learnable latent subspace aggregation function of projection (f(.)) for the oscillatory components (OCs).
        use_self_attention_z (`bool`, *optional*, defaults to `False`):
            Whether to use self-attention in the Z branch of the dual-branched latent space.
        use_self_attention_s (`bool`, *optional*, defaults to `False`):
            Whether to use self-attention in the S branch of the dual-branched latent space as a pooling operator.
        use_first_agg (`bool`, *optional*, defaults to `False`):
            Whether to use the first aggregation method (attention or LSTM) in the S latent space.
        use_second_agg (`bool`, *optional*, defaults to `False`):
            Whether to use the second aggregation method (max or avg pooling) in the S latent space.
        first_agg (`str`, *optional*, defaults to `"attention"`):
            The first aggregation method to use in the S latent space. Options are `"attention"` and `"lstm"`.
        second_agg (`str`, *optional*, defaults to `"avg"`):
            The second aggregation method to use in the S latent space. Options are `"max"` and `"avg"`.
        attention_heads_z (`int`, *optional*, defaults to 2):
            Number of attention heads to use in the self-attention module of the Z branch.
        attention_heads_s (`int`, *optional*, defaults to 6):
            Number of attention heads to use in the self-attention module of the S branch.
        use_learnable_embed_OC1 (`bool`, *optional*, defaults to `False`):
            Whether to use learnable embeddings to fill with a centroid embedding the first (lowest frequency) observed component when it's been identified as 0.
        use_learnable_embed_OC2 (`bool`, *optional*, defaults to `False`):
            Whether to use learnable embeddings with a centroid embedding the second observed component when it's been identified as 0.
        use_learnable_embed_OC3_or_more (`bool`, *optional*, defaults to `False`):
            Whether to use learnable embeddings with a centroid embedding the third (or more) observed components when it's been identified as 0.
        hidden_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all hidden fully connected layers right after the Dec2Vec encoder.
        feat_proj_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for output of the feature encoder.
        final_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for the final projection layer of DecVAE.
        encoder_dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability for all layers in the Dec2Vec encoder.
        feat_extract_norm (`str`, *optional*, defaults to `"layer"`):
            The norm to be applied to 1D convolutional layers in feature encoder. One of `"group"` for group
                normalization of only the first 1D convolutional layer, or `"instance"` /`"batch"` / `"layer"` for instance/batch/layer normalization of all 1D
                convolutional layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        add_skip_connection (`bool`, *optional*, defaults to `False`):
            Whether to add skip connections in the input of the DecVAE encoder convolutional layers to their output.
        hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the Dec2Vec encoder layers output and the encoder layers and the pooler layer of Wav2vec2. Use 768 for the original Wav2Vec2 implementation.
        z_proj_input_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the input features to the projection layer of the Z branch.
        s_proj_input_dim (`int`, *optional*, defaults to 192):
            Dimensionality of the input features to the projection layer of the S branch.
        proj_codevector_dim_z (`int`, *optional*, defaults to 192):
            Dimensionality of the final projection of the Z branch.
        proj_codevector_dim_s (`int`, *optional*, defaults to 192):
            Dimensionality of the final projection of the S branch.
        proj_intermediate_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the intermediate layer in the two-layered projectors.
        z_latent_dim (`int`, *optional*, defaults to 48):
            Dimensionality of the latent space Z.    
        s_latent_dim (`int`, *optional*, defaults to 24):
            Dimensionality of the latent space S.
        conv_dim_z (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
            Dimensionality of the convolutional layers in the Z branch.
        conv_dim_s (`Tuple[int]` or `List[int]`, *optional*, defaults to `(192, 192, 192, 192, 192, 192, 192)`):
            Dimensionality of the convolutional layers in the S branch.
        fc_input_size (`int`, *optional*, defaults to 3000):
            The size of the input features to the fully connected layer in case the latent sequence branch S is modelled as a FC network.
        fc_kernels (`Tuple[int]` or `List[int]`, *optional*, defaults to `(2048, 1024, 512, 512, 512, 256, 256)`):
            A tuple of integers defining the number of output units of each fully connected layer in the latent sequence branch S when it's modelled as a FC network. 
            The length of *fc_kernels* defines the number of fully connected layers.
        fc_or_conv_s (`str`, *optional*, defaults to `"conv"`):
            Whether the latent sequence branch S is modelled as a fully connected (FC) network or as a convolutional (conv) network. Options are `"fc"` and `"conv"`.



        ''Wav2Vec2'' args:
        vocab_size (`int`, *optional*, defaults to 32):
            Vocabulary size of the Wav2Vec2 model. Defines the number of different tokens that can be represented by
            the `inputs_ids` passed when calling [`Wav2Vec2Model`] or [`TFWav2Vec2Model`]. Vocabulary size of the
            model. Defines the different tokens that can be represented by the *inputs_ids* passed to the forward
            method of [`Wav2Vec2Model`].
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 2560):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities if a transformer is used as an extra feature extraction stage after the SSL calculation after the Dec2Vec encoder.
        layerdrop (`float`, *optional*, defaults to 0.1):
            The LayerDrop probability. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more
            details.
        agg_norm (`str`, *optional*, defaults to `"layer"`):
            The norm to be applied to the Wave2vec2 Transformer encoder. One of `"group"` for group normalization or
            `"layer"` for layer normalization.
        feat_extract_activation (`str, `optional`, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the 1D convolutional layers of the feature
            extractor. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"` are supported.
        conv_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`):
            A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the
            feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.
        conv_stride (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`):
            A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length
            of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.
        conv_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 3, 3)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The
            length of *conv_kernel* defines the number of convolutional layers and has to match the length of
            *conv_dim*.
        conv_bias (`bool`, *optional*, defaults to `False`):
            Whether the 1D convolutional layers have a bias.
        num_conv_pos_embeddings (`int`, *optional*, defaults to 128):
            Number of convolutional positional embeddings. Defines the kernel size of 1D convolutional positional
            embeddings layer.
        num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16):
            Number of groups of 1D convolutional positional embeddings layer.
        do_stable_layer_norm (`bool`, *optional*, defaults to `False`):
            Whether to apply *stable* layer norm architecture of the Transformer encoder. `do_stable_layer_norm is
            True` corresponds to applying layer norm before the attention layer, whereas `do_stable_layer_norm is
            False` corresponds to applying layer norm after the attention layer.
        apply_spec_augment (`bool`, *optional*, defaults to `True`):
            Whether to apply *SpecAugment* data augmentation to the outputs of the feature encoder. For reference see
            [SpecAugment: A Simple Data Augmentation Method for Automatic Speech
            Recognition](https://arxiv.org/abs/1904.08779).
        mask_time_prob (`float`, *optional*, defaults to 0.05):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked. The masking
            procecure generates ''mask_time_prob*len(time_axis)/mask_time_length'' independent masks over the axis. If
            reasoning from the propability of each feature vector to be chosen as the start of the vector span to be
            masked, *mask_time_prob* should be `prob_vector_start*mask_time_length`. Note that overlap may decrease the
            actual percentage of masked vectors. This is only relevant if `apply_spec_augment is True`.
        mask_time_length (`int`, *optional*, defaults to 10):
            Length of vector span along the time axis.
        mask_time_min_masks (`int`, *optional*, defaults to 2),:
            The minimum number of masks of length `mask_feature_length` generated along the time axis, each time step,
            irrespectively of `mask_feature_prob`. Only relevant if ''mask_time_prob*len(time_axis)/mask_time_length <
            mask_time_min_masks''
        mask_feature_prob (`float`, *optional*, defaults to 0.0):
            Percentage (between 0 and 1) of all feature vectors along the feature axis which will be masked. The
            masking procecure generates ''mask_feature_prob*len(feature_axis)/mask_time_length'' independent masks over
            the axis. If reasoning from the propability of each feature vector to be chosen as the start of the vector
            span to be masked, *mask_feature_prob* should be `prob_vector_start*mask_feature_length`. Note that overlap
            may decrease the actual percentage of masked vectors. This is only relevant if `apply_spec_augment is
            True`.
        mask_feature_length (`int`, *optional*, defaults to 10):
            Length of vector span along the feature axis.
        mask_feature_min_masks (`int`, *optional*, defaults to 0),:
            The minimum number of masks of length `mask_feature_length` generated along the feature axis, each time
            step, irrespectively of `mask_feature_prob`. Only relevant if
            ''mask_feature_prob*len(feature_axis)/mask_feature_length < mask_feature_min_masks''
        num_codevectors_per_group (`int`, *optional*, defaults to 320):
            Number of entries in each quantization codebook (group).
        num_codevector_groups (`int`, *optional*, defaults to 2):
            Number of codevector groups for product codevector quantization.
        contrastive_logits_temperature (`float`, *optional*, defaults to 0.1):
            The temperature *kappa* in the contrastive loss.
        num_negatives (`int`, *optional*, defaults to 100):
            Number of negative samples for the contrastive loss.
        codevector_dim (`int`, *optional*, defaults to 256):
            Dimensionality of the quantized feature vectors.
        diversity_loss_weight (`int`, *optional*, defaults to 0.1):
            The weight of the codebook diversity loss component.
        ctc_loss_reduction (`str`, *optional*, defaults to `"sum"`):
            Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an
            instance of [`Wav2Vec2ForCTC`].
        ctc_zero_infinity (`bool`, *optional*, defaults to `False`):
            Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly
            occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance
            of [`Wav2Vec2ForCTC`].
        use_weighted_layer_sum (`bool`, *optional*, defaults to `False`):
            Whether to use a weighted average of layer outputs with learned weights. Only relevant when using an
            instance of [`Wav2Vec2ForSequenceClassification`].
        classifier_proj_size (`int`, *optional*, defaults to 256):
            Dimensionality of the projection before token mean-pooling for classification.
        tdnn_dim (`Tuple[int]` or `List[int]`, *optional*, defaults to `(512, 512, 512, 512, 1500)`):
            A tuple of integers defining the number of output channels of each 1D convolutional layer in the *TDNN*
            module of the *XVector* model. The length of *tdnn_dim* defines the number of *TDNN* layers.
        tdnn_kernel (`Tuple[int]` or `List[int]`, *optional*, defaults to `(5, 3, 3, 1, 1)`):
            A tuple of integers defining the kernel size of each 1D convolutional layer in the *TDNN* module of the
            *XVector* model. The length of *tdnn_kernel* has to match the length of *tdnn_dim*.
        tdnn_dilation (`Tuple[int]` or `List[int]`, *optional*, defaults to `(1, 2, 3, 1, 1)`):
            A tuple of integers defining the dilation factor of each 1D convolutional layer in *TDNN* module of the
            *XVector* model. The length of *tdnn_dilation* has to match the length of *tdnn_dim*.
        xvector_output_dim (`int`, *optional*, defaults to 512):
            Dimensionality of the *XVector* embedding vectors.
        add_adapter (`bool`, *optional*, defaults to `False`):
            Whether a convolutional network should be stacked on top of the Wav2Vec2 Encoder. Can be very useful for
            warm-starting Wav2Vec2 for SpeechEncoderDecoder models.
        adapter_kernel_size (`int`, *optional*, defaults to 3):
            Kernel size of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        adapter_stride (`int`, *optional*, defaults to 2):
            Stride of the convolutional layers in the adapter network. Only relevant if `add_adapter is True`.
        num_adapter_layers (`int`, *optional*, defaults to 3):
            Number of convolutional layers that should be used in the adapter network. Only relevant if `add_adapter is
            True`.
        adapter_attn_dim (`int`, *optional*):
            Dimension of the attention adapter weights to be used in each attention block. An example of a model using
            attention adapters is [facebook/mms-1b-all](https://huggingface.co/facebook/mms-1b-all).
        output_hidden_size (`int`, *optional*):
            Dimensionality of the encoder output layer. If not defined, this defaults to *hidden-size*. Only relevant
            if `add_adapter is True`.
            
    ```"""

    def __init__(
        self,
        decomp_to_perform = "filter",
        fs = 16000,
        freq_groups = [[0,  750],
           [750, 1500],
           [1500, 2250],
           [2250, 3000],
           [3000, 3750],
           [3750, 4500],
           [4500, 5200],
           [5250, 6000],
           [6000, 6750],
           [6750, 7500]],
        receptive_field = 0.025,
        stride = 0.02, 
        dual_branched_latent = True,
        only_z_branch = False,
        only_s_branch = False,
        project_OCs = False,
        use_self_attention_z = False,
        use_self_attention_s = True,
        use_first_agg = True,
        use_second_agg = True,
        first_agg = "attention",
        second_agg = "avg",
        attention_heads_z = 2,
        attention_heads_s = 6,
        add_skip_connection = False,
        use_learnable_embed_OC1 = True,
        use_learnable_embed_OC2 = True,
        use_learnable_embed_OC3_or_more = True,
        z_proj_input_dim=512,
        s_proj_input_dim=192,
        proj_codevector_dim_z=192,
        proj_codevector_dim_s=192,
        proj_intermediate_dim = 512,
        z_latent_dim = 48,
        s_latent_dim = 24,
        lower_speech_freq = 50,
        higher_speech_freq = 7500,
        max_silence_freq = 250,
        notch_band_low = 55,
        notch_band_high = 70,
        detection_intervals = 6,
        spec_amp_tolerance = 5e-6,
        spec_amp_tolerance_seq = 1e-6,
        global_thres = 0.01,
        power_law = 1.7,
        nfft = 512,
        min_distance = 300,
        peak_bandwidth = 500,
        prom_thres = 0.01,
        N_peaks_to_select = 2,
        buttord = 2,
        remove_silence = False,
        NoC = 3,
        NoC_seq = 3,
        group_OCs_by_frame = 'equally_distribute',
        group_OCs_by_seq = 'equally_distribute',
        max_frames_per_batch = 100,
        decomp_loss_reduction = "sum",
        div_pos_weight = 0.3333,
        div_neg_weight = 0.3333,
        weight_0_1 = 0.3333,
        weight_0_2 = 0.3333,
        weight_0_3_and_above = 0.3333,
        decomp_loss_s_weight = 10,
        use_prior_regularization = False,
        beta_kl_prior_z = 1.0,
        beta_kl_prior_s = 1.0,
        prior_reg_weighting_z = 10,
        prior_reg_weighting_s = 400,
        divergence_type = "js",
        vocab_size=32,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=2560,
        hidden_act="gelu",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        feat_proj_dropout=0.0,
        encoder_dropout=0.0,
        final_dropout=0.0,
        activation_dropout=0.0,
        layerdrop=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        feat_extract_norm="layer",
        agg_norm = "layer",
        feat_extract_activation="gelu",
        conv_dim_z=(512, 512, 512, 512, 512, 512, 512),
        conv_dim_s=(192, 192, 192, 192, 192, 192, 192),
        conv_dim=(512, 512, 512, 512, 512, 512, 512),
        conv_stride=(5, 2, 2, 2, 2, 2, 2),
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        conv_bias=True,
        fc_input_size = 3000,
        fc_kernels=(2048, 1024, 512, 512, 512, 256, 256),
        fc_or_conv_s="conv",
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=True,
        apply_spec_augment=True,
        mask_time_prob=0.1,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        num_codevectors_per_group=320,
        num_codevector_groups=2,
        contrastive_logits_temperature=0.1,
        num_negatives=100,
        codevector_dim=256,

        contrastive_loss_weight = 1.0,
        diversity_loss_weight=0.1,
        ctc_loss_reduction="sum",
        ctc_zero_infinity=False,
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        tdnn_dim=(512, 512, 512, 512, 1500),
        tdnn_kernel=(5, 3, 3, 1, 1),
        tdnn_dilation=(1, 2, 3, 1, 1),
        xvector_output_dim=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        add_adapter=False,
        adapter_kernel_size=3,
        adapter_stride=2,
        num_adapter_layers=3,
        output_hidden_size=None,
        adapter_attn_dim=None,
        **kwargs,
    ):
        super().__init__(**kwargs, pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id)

        #Decomposition module parameters
        self.decomp_to_perform = decomp_to_perform
        self.fs = fs
        self.freq_groups = freq_groups
        self.receptive_field = receptive_field
        self.stride = stride   
        self.lower_speech_freq = lower_speech_freq
        self.higher_speech_freq = higher_speech_freq
        self.max_silence_freq = max_silence_freq
        self.notch_band_low = notch_band_low
        self.notch_band_high = notch_band_high
        self.detection_intervals = detection_intervals
        self.global_thres = global_thres
        self.spec_amp_tolerance = spec_amp_tolerance
        self.spec_amp_tolerance_seq = spec_amp_tolerance_seq
        self.power_law = power_law
        self.nfft = nfft
        self.min_distance = min_distance
        self.peak_bandwidth = peak_bandwidth
        self.prom_thres = prom_thres
        self.N_peaks_to_select = N_peaks_to_select
        self.buttord = buttord
        self.remove_silence = remove_silence
        self.NoC = NoC
        self.NoC_seq = NoC_seq
        self.group_OCs_by_frame = group_OCs_by_frame
        self.group_OCs_by_seq = group_OCs_by_seq

        #Decomposition loss parameters
        self.max_frames_per_batch = max_frames_per_batch
        self.decomp_loss_reduction = decomp_loss_reduction
        self.div_pos_weight = div_pos_weight
        self.div_neg_weight = div_neg_weight
        self.weight_0_1 = weight_0_1
        self.weight_0_2 = weight_0_2
        self.weight_0_3_and_above = weight_0_3_and_above
        self.decomp_loss_s_weight = decomp_loss_s_weight,
        self.use_prior_regularization = use_prior_regularization
        self.beta_kl_prior_z = beta_kl_prior_z
        self.beta_kl_prior_s = beta_kl_prior_s
        self.prior_reg_weighting_z = prior_reg_weighting_z
        self.prior_reg_weighting_s = prior_reg_weighting_s
        self.divergence_type = divergence_type
        
        #Dec2Vec and DecVAE parameters
        self.dual_branched_latent = dual_branched_latent
        self.only_z_branch = only_z_branch
        self.only_s_branch = only_s_branch
        self.project_OCs = project_OCs
        self.use_self_attention_z = use_self_attention_z
        self.use_self_attention_s = use_self_attention_s
        self.use_first_agg = use_first_agg
        self.use_second_agg = use_second_agg
        self.first_agg = first_agg
        self.second_agg = second_agg
        self.attention_heads_z = attention_heads_z
        self.attention_heads_s = attention_heads_s
        self.hidden_size = hidden_size
        self.feat_extract_norm = feat_extract_norm
        self.agg_norm = agg_norm
        self.feat_extract_activation = feat_extract_activation
        self.conv_dim_z = list(conv_dim_z)
        self.conv_dim_s = list(conv_dim_s)
        self.conv_dim = list(conv_dim)
        self.conv_stride = list(conv_stride)
        self.conv_kernel = list(conv_kernel)
        self.conv_bias = conv_bias
        self.fc_input_size = fc_input_size
        self.fc_kernels = list(fc_kernels)
        self.fc_or_conv_s = fc_or_conv_s
        self.hidden_dropout = hidden_dropout
        self.encoder_dropout = encoder_dropout
        self.feat_proj_dropout = feat_proj_dropout
        self.final_dropout = final_dropout
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.add_skip_connection = add_skip_connection
        self.use_learnable_embed_OC1 = use_learnable_embed_OC1
        self.use_learnable_embed_OC2 = use_learnable_embed_OC2
        self.use_learnable_embed_OC3_or_more = use_learnable_embed_OC3_or_more
        self.z_proj_input_dim = z_proj_input_dim
        self.s_proj_input_dim = s_proj_input_dim
        self.proj_codevector_dim_z = proj_codevector_dim_z
        self.proj_codevector_dim_s = proj_codevector_dim_s
        self.proj_intermediate_dim = proj_intermediate_dim
        self.z_latent_dim = z_latent_dim
        self.s_latent_dim = s_latent_dim


        #Wav2vec2 parameters
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.num_conv_pos_embedding_groups = num_conv_pos_embedding_groups
        self.num_feat_extract_layers = len(self.conv_dim_z)
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.activation_dropout = activation_dropout
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.layerdrop = layerdrop
        self.vocab_size = vocab_size
        self.do_stable_layer_norm = do_stable_layer_norm
        self.use_weighted_layer_sum = use_weighted_layer_sum

        if (
            (len(self.conv_stride) != self.num_feat_extract_layers)
            or (len(self.conv_kernel) != self.num_feat_extract_layers)
            or (len(self.conv_dim_z) != self.num_feat_extract_layers)
        ):
            raise ValueError(
                "Configuration for convolutional layers is incorrect. It is required that `len(config.conv_dim)` =="
                " `len(config.conv_stride)` == `len(config.conv_kernel)`, but is `len(config.conv_dim) ="
                f" {len(self.conv_dim_z)}`, `len(config.conv_stride) = {len(self.conv_stride)}`,"
                f" `len(config.conv_kernel) = {len(self.conv_kernel)}`."
            )

        # fine-tuning config parameters for SpecAugment: https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        self.mask_time_prob = mask_time_prob
        self.mask_time_length = mask_time_length
        self.mask_time_min_masks = mask_time_min_masks
        self.mask_feature_prob = mask_feature_prob
        self.mask_feature_length = mask_feature_length
        self.mask_feature_min_masks = mask_feature_min_masks

        # parameters for pretraining with codevector quantized representations
        self.num_codevectors_per_group = num_codevectors_per_group
        self.num_codevector_groups = num_codevector_groups
        self.contrastive_logits_temperature = contrastive_logits_temperature
        self.num_negatives = num_negatives
        self.codevector_dim = codevector_dim
        self.diversity_loss_weight = diversity_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        #proj_codevector_dim in DecVAE parameters as z,s

        # ctc loss
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity

        # adapter
        self.add_adapter = add_adapter
        self.adapter_kernel_size = adapter_kernel_size
        self.adapter_stride = adapter_stride
        self.num_adapter_layers = num_adapter_layers
        self.output_hidden_size = output_hidden_size or hidden_size
        self.adapter_attn_dim = adapter_attn_dim

        # SequenceClassification-specific parameter. Feel free to ignore for other classes.
        self.classifier_proj_size = classifier_proj_size

        # XVector-specific parameters. Feel free to ignore for other classes.
        self.tdnn_dim = list(tdnn_dim)
        self.tdnn_kernel = list(tdnn_kernel)
        self.tdnn_dilation = list(tdnn_dilation)
        self.xvector_output_dim = xvector_output_dim

    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)


__all__ = ["DecVAEConfig"]
