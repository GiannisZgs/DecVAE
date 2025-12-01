# Arguments Configuration Guide

This directory contains dataclass definitions for all configurable parameters used in DecVAE experiments. Each argument class groups related parameters for specific aspects of the model, training, and evaluation pipelines.

## Accessing Parameter Information

To access parameter descriptions without running scripts:
(example for DataTrainingArguments, same logic can be followed for other args.py files)

```python
from args_configs import DataTrainingArguments
from dataclasses import fields

# Print all parameters with their help text
for field in fields(DataTrainingArguments):
    help_text = field.metadata.get('help', 'No help available')
    print(f"{field.name}: {help_text}")
```

---

## Argument Classes

### 1. DataTrainingArguments (`data_training_args.py`)

Parameters for data loading, preprocessing, caching, and training configuration.

#### Dataset Configuration
- **`dataset_name`**: Dataset identifier (`"sim_vowels"`, `"timit"`, `"iemocap"`, `"VOC_ALS"`)
- **`data_dir`**: Path to the raw data directory
- **`cache_dir`**: Directory for storing downloaded datasets
- **`dataloader_file`**: File to use for loading a dataset (e.g., for LibriSpeech)
- **`train_val_test_split`**: Whether the dataset is split into train, validation, and test files

#### Dataset-Specific Parameters
- **`sim_snr_db`**: SNR for simulated vowels datasets (e.g., 15 dB)
- **`sim_vowels_number`**: Number of vowel types in sim_vowels (e.g., 5)
- **`sim_vowels_duration`**: Duration of simulated vowels in seconds
- **`parts_to_use`**: Number of parts to use from the vowels or atoms dataset
- **`validation_split_percentage`**: Percentage of training data to use for validation if not specified
- **`path_to_timit_phoneme48_to_id_file`**: TIMIT 48-phoneme set dictionary path
- **`path_to_timit_phoneme39_to_id_file`**: TIMIT 39-phoneme set dictionary path
- **`path_to_timit_speaker_dict_file`**: TIMIT speaker dictionary path
- **`path_to_iemocap_phoneme_to_id_file`**: IEMOCAP phoneme-to-ID dictionary path
- **`path_to_iemocap_emotion_to_id_file`**: IEMOCAP emotion label dictionary
- **`path_to_iemocap_speaker_dict_file`**: IEMOCAP speaker dictionary path
- **`path_to_voc_als_encodings`**: VOC ALS dataset variable encodings file path
- **`dev_spkrs_list`**: Path to development set speaker list (TIMIT)
- **`core_test_spkrs_list`**: Path to TIMIT core test set speakers

#### Caching
- **`train_cache_file_name`**: Path to preprocessed training data cache
- **`validation_cache_file_name`**: Path to validation cache
- **`test_cache_file_name`**: Path to test cache
- **`dev_cache_file_name`**: Path to development cache
- **`preprocessing_only`**: Set to `true` to only preprocess data without training

#### Audio Processing
- **`input_type`**: Input representation (`"waveform"` or `"mel"`)
- **`n_mels`**: Number of mel-frequency bins (typically 80)
- **`mel_hops`**: Hop length for mel-spectrogram computation (typically 4)
- **`mel_norm`**: Mel-spectrogram normalization method (`"global"`, `"instance"`, or `null`)
- **`max_duration_in_seconds`**: Maximum audio clip duration
- **`min_duration_in_seconds`**: Minimum audio clip duration
- **`skip_first_n_seconds`**: Skip initial seconds (useful for removing silence)

#### Training Configuration
- **`output_dir`**: Directory for saving model checkpoints and logs
- **`parent_dir`**: Path to pre-trained model for transfer learning
- **`save_model`**: Whether to save model checkpoints during training
- **`num_train_epochs`**: Total training epochs
- **`max_train_steps`**: Maximum training steps (overrides `num_train_epochs`)
- **`per_device_train_batch_size`**: Batch size per GPU/device for training
- **`per_device_eval_batch_size`**: Batch size per device for evaluation
- **`gradient_accumulation_steps`**: Number of steps to accumulate gradients
- **`gradient_checkpointing`**: Enable to save memory at cost of speed

#### Optimization
- **`learning_rate`**: Initial learning rate (can be list `[lr_z, lr_s]` for dual branches)
- **`lr_scheduler_type`**: Scheduler type (`"linear"`, `"constant_with_warmup"`, `"cosine"`)
- **`lr_scheduler_num_cycles`**: Number of cycles for the learning rate scheduler
- **`num_warmup_steps`**: Warmup steps for learning rate scheduler
- **`weight_decay`**: Weight decay coefficient (L2 regularization)
- **`adam_beta1`**: β₁ for Adam optimizer (typically 0.5 or 0.9)
- **`adam_beta2`**: β₂ for Adam optimizer (typically 0.999)
- **`adam_epsilon`**: ε for Adam optimizer (typically 1e-6 or 1e-8)
- **`pad_to_multiple_of`**: Pad sequences to multiple of this value (used in data collator)

#### Early Stopping
- **`early_stop_patience_epochs`**: Epochs to wait before stopping
- **`early_stop_min_delta_percent`**: Minimum improvement percentage to not trigger early stopping
- **`early_stop_warmup_steps`**: Steps before early stopping becomes active

#### Logging & Checkpointing
- **`logging_steps`**: Log metrics every N steps
- **`saving_steps`**: Save checkpoint every N steps
- **`with_wandb`**: Enable Weights & Biases logging
- **`wandb_project`**: W&B project name
- **`wandb_group`**: W&B run group name

#### Transfer Learning & HuggingFace Hub
- **`transfer_learning`**: Enable transfer learning mode
- **`transfer_from`**: Source dataset name for pre-trained weights
- **`which_checkpoint`**: Checkpoint index to load (`-1` = last, `-2` = second-to-last)
- **`pretrain`**: Whether to pretrain the model (if `false`, only decomposition is performed)
- **`push_to_hub`**: Whether to push the model to HuggingFace Hub after training
- **`hub_model_id`**: Name of the HuggingFace repository to push to
- **`hub_token`**: Token to use as HTTP bearer authorization for HuggingFace Hub
- **`trust_remote_code`**: Whether to trust execution of code from datasets/models on the Hub

#### Experiment Configuration
- **`experiment`**: Experiment type (`"snr"`, `"ssl_loss"`) - set to `"snr"` for most cases
- **`ssl_loss_frame_perc`**: Percentage of frames used for SSL loss during pre-training

#### Evaluation (DataTrainingArgumentsPost)
- **`classify`**: Enable classification evaluation
- **`classification_tasks`**: List of classification tasks (e.g., `["vowel", "speaker_frame"]`)
- **`measure_disentanglement`**: Compute disentanglement metrics
- **`sup_eval`**: Enable supervised evaluation mode
- **`unsup_eval`**: Enable unsupervised evaluation mode
- **`aggregations_to_use`**: Aggregation strategies to evaluate (e.g., `["all", "X_OCs_freq"]`)
- **`epoch_range_to_evaluate`**: List of epoch indices to evaluate
- **`dev_data_percent`**: Percentage of training data to use for validation in CV tasks
- **`train_data_percent`**: Percentage of training data to use for training in CV tasks
- **`random_states`**: Number of random seeds for cross-validation
- **`random_states_unsup`**: Number of random seeds for unsupervised k-Means initialization
- **`mod_expl_random_states`**: Number of random states for modularity explicitness evaluation
- **`disentanglement_eval_cv_splits`**: CV splits for disentanglement metrics
- **`classif_eval_cv_splits`**: CV splits for supervised classification evaluation
- **`discard_label_overlaps`**: Remove frames with multiple simultaneous labels
- **`classif_num_workers`**: Number of CPU workers for sklearn processes

#### Miscellaneous
- **`seed`**: Random seed for reproducibility
- **`preprocessing_num_workers`**: CPU workers for data preprocessing
- **`audio_column_name`**: Column name containing audio data in dataset

---

### 2. ModelArguments (`model_args.py`)

Parameters defining the DecVAE architecture and model structure.

#### Branch Architecture
- **`dual_branched_latent`**: Enable dual-branch architecture (content Z + style S)
- **`only_z_branch`**: Train only content (Z) branch
- **`only_s_branch`**: Train only style (S) branch

#### Component Decomposition
- **`project_OCs`**: Project components to new latent dimensions
- **`use_learnable_embed_OC1`**: Use learnable embeddings when 1st component is missing
- **`use_learnable_embed_OC2`**: Use learnable embeddings when 2nd component is missing
- **`use_learnable_embed_OC3_or_more`**: Use learnable embeddings for 3rd+ components

#### Convolutional Encoder (Z Branch)
- **`conv_dim_z`**: Channel dimensions for each conv layer (e.g., `[512, 512, 512, 512, 512, 512, 512]`)
- **`conv_kernel`**: Kernel sizes for conv layers (e.g., `[10, 3, 3, 3, 3, 2, 2]`)
- **`conv_stride`**: Stride sizes for conv layers (e.g., `[5, 2, 2, 2, 2, 2, 2]`)
- **`conv_bias`**: Use bias in convolutional layers
- **`feat_extract_norm`**: Normalization after feature extractor (`"layer"` or `"instance"`)
- **`feat_extract_activation`**: Activation function (`"gelu"`, `"relu"`)

#### Convolutional Encoder (S Branch)
- **`conv_dim_s`**: Channel dimensions for S branch (e.g., `[192, 192, 192, 192, 192]`)
- **`fc_or_conv_s`**: Use fully-connected or convolutional layers for S branch
- **`fc_input_size`**: Input size for fully connected layers in feature extractor
- **`fc_kernels`**: Kernel sizes for fully connected layers (e.g., `[2048, 1024, 512, 512, 512, 256, 256]`)
- **`add_skip_connection`**: Add skip connections in convolutional layers
- **`num_feat_extract_layers`**: Number of feature extraction layers (typically 7)

#### Encoder Architecture
- **`do_stable_layer_norm`**: Use stable layer norm implementation in transformer aggregator
- **`num_hidden_layers`**: Number of hidden layers in transformer (typically 8)
- **`hidden_act`**: Activation function for hidden layers (`"gelu"`)

#### Latent Space Dimensions
- **`z_latent_dim`**: Dimension of Z latent variables per component (e.g., 48)
- **`s_latent_dim`**: Dimension of S latent variables per component (e.g., 24)
- **`proj_codevector_dim_z`**: Projection dimension before Z aggregation (typically 192)
- **`proj_codevector_dim_s`**: Projection dimension before S aggregation (typically 192)
- **`proj_intermediate_dim`**: Intermediate projection dimension (typically 512)
- **`z_proj_input_dim`**: Input dimension for Z branch projection layer (typically 512)
- **`s_proj_input_dim`**: Input dimension for S branch projection layer (typically 192)
- **`hidden_size`**: Size of hidden layers after encoder (typically 512)

#### Attention Mechanisms
- **`use_self_attention_z`**: Enable self-attention in Z branch
- **`use_self_attention_s`**: Enable self-attention in S branch
- **`attention_heads_z`**: Number of attention heads for Z branch (e.g., 2)
- **`attention_heads_s`**: Number of attention heads for S branch (e.g., 6)

#### Aggregation Strategies for the Sequence Latent Variable S
- **`use_first_agg`**: Enable first aggregation method
- **`use_second_agg`**: Enable second aggregation method
- **`first_agg`**: First aggregation method (`"attention"` or `"lstm"`)
- **`second_agg`**: Second aggregation method (`"avg"` or `"max"`)
- **`agg_norm`**: Normalization after aggregation (`"batch"` or `"layer"`)

#### Masking (Contrastive Learning)
- **`mask_time_prob`**: Probability of masking time steps (typically 1.0 for full masking)
- **`mask_time_length`**: Length of masked spans (typically 10)
- **`apply_spec_augment`**: Enable SpecAugment masking

#### Dropout
- **`feat_proj_dropout`**: Dropout after feature projection
- **`hidden_dropout`**: Dropout for hidden layers
- **`final_dropout`**: Dropout for final projection layer
- **`encoder_dropout`**: Dropout in encoder layers
- **`attention_dropout`**: Dropout for attention probabilities
- **`layerdrop`**: LayerDrop probability for transformer layers(not used)

#### Fine-Tuning (Supervised)
- **`fine_tuning_classifier_z`**: Classifier type for Z branch (`"mlp"`, `"lstm"`, `"transformer"`, `"cnn"`)
- **`fine_tuning_classifier_s`**: Classifier type for S branch
- **`vae_fine_tuning_classifier`**: Classifier type for VAE latent space (for baselines)
- **`fine_tuning_output_classes`**: Number of output classes (e.g., 4 for emotions)
- **`freeze_backbone`**: Freeze encoder during fine-tuning
- **`aggregate_branch_features`**: Combine Z and S features for classification

#### Model Metadata
- **`architectures`**: List of model architectures (e.g., `["DecVAEForPreTraining"]`)
- **`model_type`**: Model type identifier (`"dec2vec"`)

#### Initialization
- **`model_name_or_path`**: HuggingFace model for initialization (e.g., `"patrickvonplaten/wav2vec2-base-v2"`)
- **`initializer_range`**: Standard deviation for weight initialization (typically 0.02)
- **`layer_norm_eps`**: Epsilon for LayerNorm (typically 1e-5)

#### VAE-Specific (for baseline comparisons)
- **`vae_z_dim`**: VAE latent dimension
- **`vae_beta`**: β coefficient for β-VAE
- **`vae_conv_dim`**: VAE convolutional channel dimensions
- **`vae_fc_dims`**: VAE fully connected layer dimensions
- **`vae_kernel_sizes`**: VAE convolutional kernel sizes
- **`vae_strides`**: VAE convolutional stride sizes
- **`vae_hidden_dim`**: VAE hidden dimension
- **`vae_proj_intermediate_dim`**: VAE projection intermediate dimension
- **`vae_norm_type`**: VAE normalization type (`"batch"`)
- **`vae_type`**: VAE architecture type (`"VAE_1D"`)
- **`vae_input_type`**: Input type for VAE (`"waveform"`, `"audio"`)
- **`n_mels_vae`**: Number of mel bands for VAE
- **`mel_norm_vae`**: Mel-spectrogram normalization for VAE
- **`kl_annealing`**: Whether to use KL annealing in VAE
- **`raw_mels`**: Use raw mel-spectrograms as benchmark (ModelArgumentsPost)
- **`eigenprojection`**: Eigenprojection method for raw data (ModelArgumentsPost)

---

### 3. TrainingObjectiveArguments (`training_obj_args.py`)

Parameters controlling loss functions and optimization objectives.

#### Decomposition Loss
- **`decomp_loss_reduction`**: Reduction method for decomposition loss (`"sum"` or `"mean"`)
- **`max_frames_per_batch`**: Maximum frames per batch for loss calculation (`"all"` or integer)

#### Divergence Weights
- **`div_pos_weight`**: Weight for positive divergence forces (encourages component separation)
- **`div_neg_weight`**: Weight for negative divergence forces (discourages within-component variance)
- **`weight_0_1`**: Weight for original-vs-1st-component divergence (typically 0.33)
- **`weight_0_2`**: Weight for original-vs-2nd-component divergence
- **`weight_0_3_and_above`**: Weight for original-vs-3rd+-component divergence
- **`divergence_type`**: Divergence metric (`"js"` for Jensen-Shannon, `"kl"` for KL divergence)

#### Prior Regularization (VAE-like)
- **`use_prior_regularization`**: Enable KL divergence to prior (Gaussian)
- **`beta_kl_prior_z`**: β coefficient for Z prior KL term (β=1 is vanilla VAE)
- **`beta_kl_prior_s`**: β coefficient for S prior KL term
- **`prior_reg_weighting_z`**: Overall weight for Z prior regularization
- **`prior_reg_weighting_s`**: Overall weight for S prior regularization

#### Multi-Objective Balancing
- **`decomp_loss_s_weight`**: Weight for S branch decomposition loss
- **`prior_reg_loss_total_weight`**: Total prior regularization weight (for fine-tuning)
- **`div_loss_total_weight`**: Total divergence loss weight (for fine-tuning)
- **`supervised_loss_weight`**: Supervised classification loss weight (fine-tuning)
- **`supervised_loss_rel_weight`**: Relative scaling factor for supervised loss

#### Supervised Loss (Fine-Tuning)
- **`supervised_loss_type`**: Loss function (`"cross_entropy"`, `"focal"`, `"label_smoothed_ce"`)
- **`supervised_loss_reduction`**: Reduction method (`"sum"`, `"mean"`, `"none"`)
- **`vae_loss_weight`**: VAE reconstruction + KL loss weight (if applicable)

#### Gradient Clipping
- **`clip_grad_value`**: Maximum gradient norm for clipping (e.g., 1.0 or 5.0)

---

### 4. DecompositionArguments (`decomposition_args.py`)

Parameters for signal decomposition preprocessing.

#### Decomposition Method
- **`decomp_to_perform`**: Decomposition algorithm (`"ewt"` for Empirical Wavelet Transform, `"filter"` for filterbank, `"emd"`, `"vmd"`)
- **`frame_decomp`**: Perform frame-level decomposition
- **`seq_decomp`**: Perform sequence-level decomposition
- **`NoC`**: Number of components for frame decomposition (typically 2-4)
- **`NoC_seq`**: Number of components for sequence decomposition

#### Component Grouping
- **`group_OCs_by_frame`**: How to group frame components (`"equally_distribute"`, `"high_freqs_first"`, `"low_freqs_first"`)
- **`group_OCs_by_seq`**: How to group sequence components

#### Audio Properties
- **`fs`**: Sampling frequency (typically 16000 Hz)
- **`receptive_field`**: Receptive field in seconds (must match encoder, typically 0.025s)
- **`stride`**: Stride in seconds (must match encoder, typically 0.02s)

#### Frequency Bands
- **`lower_speech_freq`**: Lower frequency bound for speech (typically 50 Hz)
- **`higher_speech_freq`**: Upper frequency bound (typically 7500 Hz)
- **`max_silence_freq`**: Maximum frequency for silence detection (typically 250 Hz)
- **`freq_groups`**: Frequency band definitions for oscillatory component detection

#### Noise Filtering
- **`use_notch_filter`**: Apply notch filter to remove power line noise
- **`notch_band_low`**: Notch filter lower bound (typically 55 Hz)
- **`notch_band_high`**: Notch filter upper bound (typically 70 Hz)

#### Peak Detection
- **`detection_intervals`**: Number of intervals to split spectrum for peak detection
- **`N_peaks_to_select`**: Number of largest peaks per interval
- **`min_distance`**: Minimum distance between peaks (in Hz)
- **`peak_bandwidth`**: Minimum required peak bandwidth
- **`prom_thres`**: Prominence threshold for peak detection
- **`global_thres`**: Global amplitude threshold for peaks

#### Spectrum Analysis
- **`spec_amp_tolerance`**: Amplitude tolerance for frame-level spectrum
- **`spec_amp_tolerance_seq`**: Amplitude tolerance for sequence-level spectrum
- **`nfft`**: FFT size (typically 512)
- **`power_law`**: Power law for non-uniform spectrum splitting

#### Filterbank
- **`buttord`**: Butterworth filter order (typically 2)
- **`remove_silence`**: Remove silent frames based on frequency criterion

#### EWT-Specific
- **`ewt_completion`**: Complete EWT if fewer modes found than specified
- **`ewt_filter`**: Filter type for EWT (`"gaussian"`)
- **`ewt_filter_length`**: Filter length
- **`ewt_filter_sigma`**: Gaussian filter sigma
- **`ewt_log_spectrum`**: Work with log spectrum
- **`ewt_detect`**: Peak detection method (`"locmax"`)

#### EMD-Specific
- **`emd_spline_kind`**: Spline type for EMD (`"pchip"`)
- **`emd_max_iter`**: Maximum iterations per sifting (typically 1000)
- **`emd_energy_ratio_thr`**: Energy ratio threshold per IMF check (typically 5e-3)
- **`emd_std_thr`**: Standard deviation threshold per IMF check (typically 0.1)
- **`emd_svar_thr`**: Scaled variance threshold per IMF check (typically 0.01)
- **`emd_total_power_thr`**: Total power threshold per EMD decomposition (typically 0.0001)
- **`emd_range_thr`**: Amplitude range threshold after scaling (typically 0.001)
- **`emd_extrema_detection`**: Method for finding extrema (`"simple"`)

#### VMD-Specific
- **`vmd_alpha`**: Balancing parameter (controls bandwidth/smoothness)
- **`vmd_tau`**: Time-step for dual ascent (0 for noise-slack)
- **`vmd_DC`**: Keep first mode at DC (0-frequency)
- **`vmd_init`**: Center frequency initialization (0=zeros, 1=uniform, 2=random)
- **`vmd_tol`**: Convergence tolerance
- **`use_vmd_correction`**: Apply VMD frequency correction

---

### 5. VisualizationsArguments (`visualization_args.py`)

Parameters for visualizing decomposed inputs and learned latent spaces.

#### General Visualization
- **`save_vis_dir`**: Directory to save visualization outputs
- **`random_seed_vis`**: Random seed for reproducibility
- **`use_umap`**: Use UMAP in addition to t-SNE for dimensionality reduction
- **`plot_3d`**: Generate 3D plots

#### Input Decomposition Visualization
- **`frames_to_vis`**: Number of utterances to visualize for frame variable
- **`seq_to_vis`**: Number of utterances to visualize for sequence variable
- **`vis_td_frames`**: Visualize time-domain frame features
- **`vis_mel_frames`**: Visualize mel-spectrogram frame features
- **`vis_td_seq`**: Visualize time-domain sequence features
- **`vis_mel_seq`**: Visualize mel-spectrogram sequence features
- **`set_to_use_for_vis`**: Dataset split to visualize (`"train"`, `"dev"`, `"test"`, `"all"`)

#### Latent Space Visualization
- **`visualize_latent_frame`**: Visualize frame-level latent space (Z)
- **`visualize_latent_sequence`**: Visualize sequence-level latent space (S)
- **`visualize_train_set`**: Visualize training set latents
- **`visualize_dev_set`**: Visualize development set latents
- **`visualize_test_set`**: Visualize test set latents
- **`latent_train_set_frames_to_vis`**: Number of train utterances for frame visualization
- **`latent_dev_set_frames_to_vis`**: Number of dev utterances for frame visualization
- **`latent_test_set_frames_to_vis`**: Number of test utterances for frame visualization
- **`latent_train_set_seq_to_vis`**: Number of train utterances for sequence visualization
- **`latent_dev_set_seq_to_vis`**: Number of dev utterances for sequence visualization
- **`latent_test_set_seq_to_vis`**: Number of test utterances for sequence visualization

#### Visualization Variables
- **`variables_to_plot_latent`**: Variables to color-code in frame latent plots (e.g., `["vowel", "speaker_id"]`)
- **`variables_to_plot_latent_seq`**: Variables for sequence latent plots
- **`aggregation_strategies_to_plot_frame`**: Aggregations to visualize (e.g., `["all", "X_OCs_freq"]`)
- **`aggregation_strategies_to_plot_seq`**: Aggregations for sequence plots
- **`vis_isotropic_gaussian_sphere`**: Overlay isotropic Gaussian sphere for reference

#### Dataset-Specific Variables
- **`variables_to_plot`**: VOC-ALS frame variables (e.g., `["speaker_id", "phoneme", "king_stage"]`)
- **`variables_to_plot_seq`**: VOC-ALS sequence variables for visualization
- **`sel_vowels_list_timit`**: TIMIT vowels to select (e.g., `['iy', 'ey', 'ay', 'aw', 'ow', 'uh', 'uw']`)
- **`sel_consonants_list_timit`**: TIMIT consonants to select (e.g., `['b', 'd', 'f', 'k', 'l', 's']`)
- **`sel_phonemes_list_timit`**: TIMIT phonemes to select for visualization
- **`sel_phonemes_list_iemocap`**: IEMOCAP phonemes to select
- **`sel_non_verbal_phonemes_iemocap`**: Non-verbal sounds (e.g., `['SIL', '+BREATHING+', '+LIPSMACK+', '+LAUGHTER+']`)

---




