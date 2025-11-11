from dataclasses import dataclass, field
from typing import List, Optional
from utils import list_field

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config we are going to pre-train.
    """
    comment_model_args: str = field(
        metadata={"help": "A comment to add to the model arguments."},
    )
    dual_branched_latent: bool = field(
        default=True,
        metadata={"help": "Whether to use a dual-branched model with two separated latent variables."},
    )
    only_z_branch: bool = field(
        default=False,
        metadata={"help": "Whether to use only the z branch in the dual-branched model."},
    )
    only_s_branch: bool = field(
        default=False,
        metadata={"help": "Whether to use only the s branch in the dual-branched model."},
    )
    project_OCs: bool = field(
        default=False,
        metadata={"help": "Whether to project the components to a new latent dimension."},
    )
    use_self_attention_z: bool = field(
        default=False,
        metadata={"help": "Whether to use self-attention in the z branch."},
    )
    use_self_attention_s: bool = field(
        default=True,
        metadata={"help": "Whether to use self-attention in the s branch."},
    )
    use_first_agg: bool = field(
        default=True,
        metadata={"help": "Whether to use a first aggregation layer for hidden states."},
    )
    use_second_agg: bool = field(
        default=True,
        metadata={"help": "Whether to use a second aggregation layer for hidden states."},
    )
    first_agg: str = field(
        default="attention",
        metadata={"help": "The aggregation method to use in the first aggregation layer - attention or lstm. "},
    )
    second_agg: str = field(
        default="avg",
        metadata={"help": "The aggregation method to use in the second aggregation layer - avg or max."},
    )
    attention_heads_z: int = field(
        default=2,
        metadata={"help": "The number of self-attention heads in the z branch."},
    )
    attention_heads_s: int = field(
        default=6,
        metadata={"help": "The number of self-attention heads in the s branch."},
    )
    fine_tuning_classifier_z: str = field(
        default=None,
        metadata={"help": "The type of classifier to use as a head on z-branch for fine-tuning - mlp, lstm, transformer, cnn."},
    )
    fine_tuning_classifier_s: str = field(
        default=None,
        metadata={"help": "The type of classifier to use as a head on s-branch for fine-tuning - mlp, lstm, transformer, cnn."},
    )
    vae_fine_tuning_classifier: str = field(
        default=None,
        metadata={"help": "The type of classifier to use as a head on VAE latent space for fine-tuning - mlp, lstm, transformer, cnn."},
    )
    fine_tuning_output_classes: int = field(
        default=4,
        metadata={"help": "The number of output classes for the fine-tuning classifier."},
    )
    aggregate_branch_features: bool = field(
        default=False,
        metadata={"help": "Whether to aggregate features from both branches during supervised classification."},
    )
    freeze_backbone: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the backbone model during fine-tuning."},
    )
    use_learnable_embed_OC1: bool = field(
        default=True,
        metadata={"help": "Whether to use a learnable embedding when first decomposition component is missing."},
    )
    use_learnable_embed_OC2: bool = field(
        default=True,
        metadata={"help": "Whether to use a learnable embedding when second decomposition component is missing."},
    )
    use_learnable_embed_OC3_or_more: bool = field(
        default=True,
        metadata={"help": "Whether to use a learnable embedding when the third and above decomposition component(s) are missing."},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models. Used to obtain some default parameters values from a pretrained Wav2Vec2 encoder."}
    )
    attention_dropout: float = field(
        default=0.0, 
        metadata={"help": "The dropout ratio for the attention probabilities if a transformer is used as an extra feature extraction stage after the SSL calculation after the Dec2Vec encoder."}
    )
    feat_proj_dropout: float = field(
        default=0.0, 
        metadata={"help": "The dropout ratio for the feature projection between the Dec2Vec encoder and the transformer."}
    )
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all hidden fully connected layers right after the Dec2Vec encoder."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer of DecVAE."},
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all layers in the Dec2Vec encoder."
        },
    )
    proj_codevector_dim_z: int = field(
        default=192,
        metadata={"help": "The dimension of the projection layer that takes inputs H_z and projects to the Z_tilde latent subspaces, just before the aggregation operation."},
    )
    proj_codevector_dim_s: int = field(
        default=192,
        metadata={"help": "The dimension of the projection layer that takes inputs H_s and projects to the S_tilde latent subspaces, just before the aggregation operation."},
    )
    proj_intermediate_dim: int = field(
        default=512,
        metadata={"help": "The dimension of the intermediate layer for the projector that takes Dec2vec features to SSL calculation spaces H_z and H_s."},
    )
    z_latent_dim: int = field(
        default=48,
        metadata={"help": "The dimension of the latent z variables in individual subspaces Z_tilde - the code (the output of the mu and logvar layers)."},
    )
    s_latent_dim: int = field(
        default=24,
        metadata={"help": "The dimension of the latent s variables in individual subspaces S_tilde - the code (the output of the mu and logvar layers)."},
    )
    mask_time_prob: float = field(
        default=0.1,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )
    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    
    feat_extract_norm: str = field(
        default="layer",
        metadata={"help": "The normalization to be applied after the feature extractor. Should be 'layer' or 'instance'."},
    )
    agg_norm: str = field(
        default="layer",
        metadata={"help": "The normalization to be applied after the aggregation layer. Should be 'batch' or 'layer'."},
    )
    layer_norm_eps: float = field(
        default=1e-5,
        metadata={"help": "The epsilon used by LayerNorm layers."},
    )

    apply_spec_augment: bool = field(
        default=True,
        metadata={"help": "Whether to apply SpecAugment to the input features or not."},
    )
    architectures: List[str] = list_field(
        default=["DecVAEForPreTraining"],
        metadata={"help": "The architectures to use for the model."},
    )
    model_type: str = field(
        default="dec2vec",
        metadata={"help": "The model type to use."},
    )
    conv_bias: bool = field(
        default=True,
        metadata={"help": "Whether to use bias in the convolutional layers in the feature extractor."},
    )
    conv_dim_z: List[int] = list_field(
        default=[512, 512, 512, 512, 512, 512, 512],
        metadata={"help": "The dimensions of the convolutional layers in the feature extractor."},
    )
    conv_kernel: List[int] = list_field(
        default=[10, 3, 3, 3, 3, 2, 2],
        metadata={"help": "The kernel sizes of the convolutional layers in the feature extractor."},
    )
    conv_stride: List[int] = list_field(
        default=[5, 2, 2, 2, 2, 2, 2],
        metadata={"help": "The stride sizes of the convolutional layers in the feature extractor."},
    )
    conv_dim_s: List[int] = list_field(
        default=[192, 192, 192, 192, 192],
        metadata={"help": "The dimensions of the convolutional layers in the feature extractor for the s branch."},
    )
    fc_input_size: int = field(
        default=3000,
        metadata={"help": "The input size for the fully connected layers in the feature extractor."},
    )
    fc_kernels: List[int] = list_field(
        default=[2048, 1024, 512, 512, 512, 256, 256],
        metadata={"help": "The kernel sizes of the fully connected layers in the feature extractor."},
    )
    fc_or_conv_s: str = field(
        default="conv",
        metadata={"help": "Whether to use fully connected layers or convolutional layers for the s branch feature extractor."},
    )
    add_skip_connection: bool = field(
        default=False,
        metadata={"help": "Whether to add skip connections to the convolutional layers in the feature extractor."},
    )
    num_feat_extract_layers: int = field(
        default=7,
        metadata={"help": "The number of layers in the feature extractor."},
    )
    do_stable_layer_norm: bool = field(
        default=True,
        metadata={"help": "Whether to use a stable layer norm implementation in the transformer aggregator."},
    )
    feat_extract_activation: str = field(
        default="gelu",
        metadata={"help": "The activation function to be used in the feature extractor."},
    )
    hidden_act: str = field(
        default="gelu",
        metadata={"help": "The activation function to be used in the transformer encoder."},
    )
    hidden_size: int = field(
        default=512,
        metadata={"help": "The dimension of the projection network after Dec2Vec. If a transformer encoder is used, it will also be the size of the feedforward network."},
    )
    z_proj_input_dim: int = field(
        default=512,
        metadata={"help": "The input dimension of the projection layer that takes Dec2vec frame-level (Z) features to the SSL calculation space H_z."},
    )
    s_proj_input_dim: int = field(
        default=192,
        metadata={"help": "The input dimension of the projection layer that takes Dec2vec frame-level (S) features to the SSL calculation space H_s."},
    )
    initializer_range: float = field(
        default=0.02,
        metadata={"help": "The standard deviation of the truncated_normal_initializer for initializing all weights."},
    )
    vae_z_dim: int = field(
        default=32,
        metadata={"help": "The dimension of the z latent variable in the VAE."},
    )
    vae_proj_intermediate_dim: int = field(
        default=2048,
        metadata={"help": "The intermediate dimension of the projection layer in the VAE."},
    )
    vae_conv_dim: int = list_field(
        default=256,
        metadata={"help": "The channel dimension of the convolutional layers in the VAE."},
    )
    vae_fc_dims: List[int] = list_field(
        default=[2048,1024,512,512,512],
        metadata={"help": "The dimensions of the fully connected layers in the VAE."},
    )
    vae_kernel_sizes: List[int] = list_field(
        default = [10, 3, 3, 3, 3, 2, 2],
        metadata={"help": "The kernel sizes of the convolutional layers in the VAE."},
    )
    vae_strides: List[int] = list_field(
        default = [5, 2, 2, 2, 2, 2, 2],
        metadata={"help": "The stride sizes of the convolutional layers in the VAE."},
    )
    vae_hidden_dim: int = field(
        default=192,
        metadata={"help": "The hidden dimension of the VAE."},
    )
    vae_beta: float = field(
        default=1.0,
        metadata={"help": "The beta parameter of the VAE."},
    )
    vae_norm_type: str = field(
        default="batch",
        metadata={"help": "The normalization type to use in the VAE."},
    )
    vae_type: str = field(
        default="VAE_1D",
        metadata={"help": "The type of VAE to use."},
    )
    vae_input_type: str = field(
        default="waveform",
        metadata={"help": "The input type to the VAE."},
    )
    n_mels_vae: int = field(
        default=80,
        metadata={"help": "The number of mel bands to use."},
    )
    mel_norm_vae: str = field(
        default=None,
        metadata={"help": "The normalization to apply to the mel spectrogram."},
    )
    kl_annealing: bool = field(
        default=False,
        metadata={"help": "Whether to use KL annealing in the VAE."},
    )



@dataclass
class ModelArgumentsPost:
    """
    Arguments pertaining to which model/config we are going to pre-train.
    """
    comment_model_args: str = field(
        metadata={"help": "A comment to add to the model arguments."},
    )
    dual_branched_latent: bool = field(
        default=True,
        metadata={"help": "Whether to use a dual-branched model with two separated latent variables."},
    )
    only_z_branch: bool = field(
        default=False,
        metadata={"help": "Whether to use only the z branch in the dual-branched model."},
    )
    only_s_branch: bool = field(
        default=False,
        metadata={"help": "Whether to use only the s branch in the dual-branched model."},
    )
    project_OCs: bool = field(
        default=False,
        metadata={"help": "Whether to project the components to a new latent dimension."},
    )
    use_self_attention_z: bool = field(
        default=False,
        metadata={"help": "Whether to use self-attention in the z branch."},
    )
    use_self_attention_s: bool = field(
        default=True,
        metadata={"help": "Whether to use self-attention in the s branch."},
    )
    use_first_agg: bool = field(
        default=True,
        metadata={"help": "Whether to use a first aggregation layer for hidden states."},
    )
    use_second_agg: bool = field(
        default=True,
        metadata={"help": "Whether to use a second aggregation layer for hidden states."},
    )
    first_agg: str = field(
        default="attention",
        metadata={"help": "The aggregation method to use in the first aggregation layer - attention or lstm. "},
    )
    second_agg: str = field(
        default="avg",
        metadata={"help": "The aggregation method to use in the second aggregation layer - avg or max."},
    )
    attention_heads_z: int = field(
        default=2,
        metadata={"help": "The number of self-attention heads in the z branch."},
    )
    attention_heads_s: int = field(
        default=6,
        metadata={"help": "The number of self-attention heads in the s branch."},
    )
    use_learnable_embed_OC1: bool = field(
        default=True,
        metadata={"help": "Whether to use a learnable embedding when first decomposition component is missing."},
    )
    use_learnable_embed_OC2: bool = field(
        default=True,
        metadata={"help": "Whether to use a learnable embedding when second decomposition component is missing."},
    )
    use_learnable_embed_OC3_or_more: bool = field(
        default=True,
        metadata={"help": "Whether to use a learnable embedding when the third and above decomposition component(s) are missing."},
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    attention_dropout: float = field(
        default=0.0, 
        metadata={"help": "The dropout ratio for the attention probabilities if a transformer is used as an extra feature extraction stage after the SSL calculation after the Dec2Vec encoder."}
    )
    feat_proj_dropout: float = field(
        default=0.0, 
        metadata={"help": "The dropout ratio for the feature projection between the Dec2Vec encoder and the transformer."}
    )
    hidden_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all hidden fully connected layers right after the Dec2Vec encoder."
        },
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "The dropout probability for the final projection layer of DecVAE."},
    )
    encoder_dropout: float = field(
        default=0.0,
        metadata={
            "help": "The dropout probability for all layers in the Dec2Vec encoder."
        },
    )
    proj_codevector_dim_z: int = field(
        default=192,
        metadata={"help": "The dimension of the projection layer that takes inputs H_z and projects to the Z_tilde latent subspaces, just before the aggregation operation."},
    )
    proj_codevector_dim_s: int = field(
        default=192,
        metadata={"help": "The dimension of the projection layer that takes inputs H_s and projects to the S_tilde latent subspaces, just before the aggregation operation."},
    )
    proj_intermediate_dim: int = field(
        default=512,
        metadata={"help": "The dimension of the intermediate layer for the projector that takes Dec2vec features to SSL calculation spaces H_z and H_s."},
    )
    z_latent_dim: int = field(
        default=48,
        metadata={"help": "The dimension of the latent z variables in individual subspaces Z_tilde - the code (the output of the mu and logvar layers)."},
    )
    s_latent_dim: int = field(
        default=24,
        metadata={"help": "The dimension of the latent s variables in individual subspaces S_tilde - the code (the output of the mu and logvar layers)."},
    )
    mask_time_prob: float = field(
        default=0.1,
        metadata={
            "help": (
                "Probability of each feature vector along the time axis to be chosen as the start of the vector "
                "span to be masked. Approximately ``mask_time_prob * sequence_length // mask_time_length`` feature "
                "vectors will be masked along the time axis."
            )
        },
    )
    mask_time_length: int = field(
        default=10,
        metadata={"help": "Length of vector span to mask along the time axis."},
    )

    layerdrop: float = field(default=0.0, metadata={"help": "The LayerDrop probability."})
    
    feat_extract_norm: str = field(
        default="layer",
        metadata={"help": "The normalization to be applied after the feature extractor. Should be 'layer' or 'instance'."},
    )
    agg_norm: str = field(
        default="layer",
        metadata={"help": "The normalization to be applied after the aggregation layer. Should be 'batch' or 'layer'."},
    )
    layer_norm_eps: float = field(
        default=1e-5,
        metadata={"help": "The epsilon used by LayerNorm layers."},
    )
    apply_spec_augment: bool = field(
        default=True,
        metadata={"help": "Whether to apply SpecAugment to the input features or not."},
    )
    architectures: List[str] = list_field(
        default=["DecVAEForPreTraining"],
        metadata={"help": "The architectures to use for the model."},
    )
    model_type: str = field(
        default="dec2vec",
        metadata={"help": "The model type to use."},
    )
    conv_bias: bool = field(
        default=True,
        metadata={"help": "Whether to use bias in the convolutional layers in the feature extractor."},
    )
    conv_dim_z: List[int] = list_field(
        default=[512, 512, 512, 512, 512, 512, 512],
        metadata={"help": "The dimensions of the convolutional layers in the feature extractor."},
    )
    conv_kernel: List[int] = list_field(
        default=[10, 3, 3, 3, 3, 2, 2],
        metadata={"help": "The kernel sizes of the convolutional layers in the feature extractor."},
    )
    conv_stride: List[int] = list_field(
        default=[5, 2, 2, 2, 2, 2, 2],
        metadata={"help": "The stride sizes of the convolutional layers in the feature extractor."},
    )
    conv_dim_s: List[int] = list_field(
        default=[192, 192, 192, 192, 192],
        metadata={"help": "The dimensions of the convolutional layers in the feature extractor for the s branch."},
    )
    fc_input_size: int = field(
        default=3000,
        metadata={"help": "The input size for the fully connected layers in the feature extractor."},
    )
    fc_kernels: List[int] = list_field(
        default=[2048, 1024, 512, 512, 512, 256, 256],
        metadata={"help": "The kernel sizes of the fully connected layers in the feature extractor."},
    )
    fc_or_conv_s: str = field(
        default="conv",
        metadata={"help": "Whether to use fully connected layers or convolutional layers for the s branch feature extractor."},
    )
    add_skip_connection: bool = field(
        default=False,
        metadata={"help": "Whether to add skip connections to the convolutional layers in the feature extractor."},
    )
    num_feat_extract_layers: int = field(
        default=7,
        metadata={"help": "The number of layers in the feature extractor."},
    )
    do_stable_layer_norm: bool = field(
        default=True,
        metadata={"help": "Whether to use a stable layer norm implementation in the transformer aggregator."},
    )
    feat_extract_activation: str = field(
        default="gelu",
        metadata={"help": "The activation function to be used in the feature extractor."},
    )
    hidden_act: str = field(
        default="gelu",
        metadata={"help": "The activation function to be used in the transformer encoder."},
    )
    hidden_size: int = field(
        default=512,
        metadata={"help": "The dimension of the projection network after Dec2Vec. If a transformer encoder is used, it will also be the size of the feedforward network."},
    )
    z_proj_input_dim: int = field(
        default=512,
        metadata={"help": "The input dimension of the projection layer that takes Dec2vec frame-level (Z) features to the SSL calculation space H_z."},
    )
    s_proj_input_dim: int = field(
        default=192,
        metadata={"help": "The input dimension of the projection layer that takes Dec2vec frame-level (S) features to the SSL calculation space H_s."},
    )
    initializer_range: float = field(
        default=0.02,
        metadata={"help": "The standard deviation of the truncated_normal_initializer for initializing all weights."},
    )
    vae_z_dim: int = field(
        default=32,
        metadata={"help": "The dimension of the z latent variable in the VAE."},
    )
    vae_proj_intermediate_dim: int = field(
        default=2048,
        metadata={"help": "The intermediate dimension of the projection layer in the VAE."},
    )
    vae_conv_dim: int = list_field(
        default=256,
        metadata={"help": "The channel dimension of the convolutional layers in the VAE."},
    )
    vae_fc_dims: List[int] = list_field(
        default=[2048,1024,512,512,512],
        metadata={"help": "The dimensions of the fully connected layers in the VAE."},
    )
    vae_kernel_sizes: List[int] = list_field(
        default = [10, 3, 3, 3, 3, 2, 2],
        metadata={"help": "The kernel sizes of the convolutional layers in the VAE."},
    )
    vae_strides: List[int] = list_field(
        default = [5, 2, 2, 2, 2, 2, 2],
        metadata={"help": "The stride sizes of the convolutional layers in the VAE."},
    )
    vae_hidden_dim: int = field(
        default=192,
        metadata={"help": "The hidden dimension of the VAE."},
    )
    vae_beta: float = field(
        default=1.0,
        metadata={"help": "The beta parameter of the VAE."},
    )
    vae_norm_type: str = field(
        default="batch",
        metadata={"help": "The normalization type to use in the VAE."},
    )
    vae_type: str = field(
        default="VAE_1D",
        metadata={"help": "The type of VAE to use."},
    )
    raw_mels: bool = field(
        default=False,
        metadata={"help": "Whether to use raw mels instead of VAE features as benchmark."},
    )
    eigenprojection: str = field(
        default=None,
        metadata={"help": "The eigenprojection to use on raw data."},
    )
    vae_input_type: str = field(
        default="audio",
        metadata={"help": "The input type to the VAE."},
    )
    n_mels_vae: int = field(
        default=80,
        metadata={"help": "The number of mel bands to use."},
    )
    mel_norm_vae: str = field(
        default=None,
        metadata={"help": "The normalization to apply to the mel spectrogram."},
    )
    kl_annealing: bool = field(
        default=False,
        metadata={"help": "Whether to use KL annealing in the VAE."},
    )
    
