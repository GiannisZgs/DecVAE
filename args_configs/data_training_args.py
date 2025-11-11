from dataclasses import dataclass, field
from typing import List, Optional
from utils import list_field

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    comment_data_args: str = field(
        metadata={"help": "A comment to add to the data arguments."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the downloaded datasets."},
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the Datasets library)."}
    )
    dataloader_file: str = field(
        default=None,
        metadata={"help": "The file to use for loading a dataset e.g. for LibriSpeech."},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the data directory."},
    )
    train_val_test_split: bool = field(
        default=False,
        metadata={"help": "Whether the dataset is split into train, validation and test files."},
    )
    dev_spkrs_list: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the file with list of speakers for the development set."},
    )
    core_test_spkrs_list: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the file with list of speakers for the TIMIT core test set."},
    )
    sim_snr_db: float = field(
        default=0.0,
        metadata={"help": "The SNR for the simulated datasets."},
    )
    sim_vowels_number: int = field(
        default=0,
        metadata={"help": "The number of vowels contained in the version of the sim_vowels dataset."},
    )
    sim_vowels_duration: float = field(
        default=4.0,
        metadata={"help": "The duration of the sim_vowels dataset."},
    )
    parts_to_use: int = field(
        default=1,
        metadata={"help": "The number of parts to use from the vowels or atoms dataset."},
    )
    validation_split_percentage: float = field(
        default=None,
        metadata={"help": "The percentage of the training data to use for validation if a validation split is not specified."},
    )
    train_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed train data cached file name"},
    )
    validation_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed validation data cached file name"},
    )
    test_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed test data cached file name"},
    )
    dev_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed development data cached file name"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store model logs and other data."},
    )
    parent_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to load a pretrained model from, in case of transfer learning."},
    )
    save_model: bool = field(
        default=False,
        metadata={"help": "Whether to save model checkpoints during training to disk."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    with_wandb: bool = field(
        default=False,
        metadata={"help": ("Whether or not to log training and evaluation metrics to Weights & Biases.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The Weights & Biases project name to log into."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "The Weights & Biases project group name to log into."},
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "Number of completed steps between each logging"},
    )
    saving_steps: int = field(
        default=1000,
        metadata={"help": "Number of completed steps between each model checkpoint save"},
    )
    tsne_plot_2d_3d: str = field(
        default='both',
        metadata={"help": "Whether to plot the tsne in 2d, 3d or both"},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size (per device) for the evaluation dataloader."},
    )
    learning_rate: float = field(
        default=None,
        metadata={"help": "Initial learning rate (after the potential warmup period) to use."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={"help": "The learning rate scheduler type to use."},
    )
    lr_scheduler_num_cycles: int = field(
        default=1,
        metadata={"help": "The number of cycles for the learning rate scheduler."},
    )
    num_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of steps for the warmup in the lr scheduler."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to use."},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Total number of training steps to perform. If provided, overrides num_train_epochs."},
    )
    num_train_epochs: int = field(
        default=None,
        metadata={"help": "Total number of training epochs to perform.Overridden by max_train_steps"},
    )
    early_stop_patience_epochs: int = field(
        default = 5,
        metadata= {"help": "Maximum epochs tolerated before triggering early stopping when no change is observed in validation loss"}
    )
    early_stop_min_delta_percent: float = field(
        default = 1.0,
        metadata = {"help": "Change required (in %) in the validation loss to not trigger early stopping."}
    )
    early_stop_warmup_steps: int = field(
        default = 100000,
        metadata={"help": "Number of initial steps during which early stopping is not active."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, 
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    skip_first_n_seconds: float = field(
        default=0.0,
        metadata={"help": "Skip the first n seconds of the audio file."},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Beta1 for AdamW optimizer"},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Beta2 for AdamW optimizer"},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for AdamW optimizer"},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "Pad to multiple of - used in the data collator."},
    )
    seed: int = field(
        default=0,
        metadata={"help": "A seed for reproducible training."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    push_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether to push the model to the Hugging Face model hub at the end of training."},
    )
    hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the Hugging Face repository to which the model should be pushed."},
    )
    hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    pretrain: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pretrain the model. If set to False, no model will be trained."
            )
        },
    )
    transfer_learning: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use transfer learning. If set to True, the model will be initialized with weights from a pretrained model."
            )
        },
    )
    transfer_from: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to transfer the pre-trained model from."},
    )
    experiment: str = field(
        default = None,
        metadata={"help": "The experiment to perform: 'snr','ssl_loss'. If set to SSL loss in evaluation, a model that contains a SSL loss percentage in its name will be expected. Set to 'snr' for most cases."},
    )
    ssl_loss_frame_perc: int = field(
        default=50,
        metadata={"help": "The percentage of frames that was used for pre-training with the SSL loss."},
    )
    which_checkpoint: int = field(
        default=-1,
        metadata={"help": "The index of the checkpoint to use for transfer learning. If -1, the last checkpoint will be used."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    path_to_timit_phoneme48_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the phoneme48-to-int dictionary for TIMIT."},
    )
    path_to_timit_phoneme39_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the phoneme39-to-int dictionary for TIMIT."},
    )
    path_to_timit_speaker_dict_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the speaker dictionary for TIMIT."},
    )
    input_type: str = field(
        default="waveform",
        metadata={"help": "The input type to the model - waveform or mel."},
    )
    n_mels: int = field(
        default=80,
        metadata={"help": "The number of mel bands to use."},
    )
    mel_norm: str = field(
        default=None,
        metadata={"help": "The normalization to apply to the mel spectrogram."},
    )
    mel_hops: int = field(
        default=4,
        metadata={"help": "The number of hops to use for computing the mel spectrogram."},
    )
    path_to_voc_als_encodings: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the encodings for the VOC ALS dataset variables."},
    )
    path_to_iemocap_phoneme_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the phoneme-to-int dictionary for IEMOCAP."},
    )
    path_to_iemocap_emotion_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the emotion-to-int dictionary for IEMOCAP."},
    )
    path_to_iemocap_speaker_dict_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the speaker dictionary for IEMOCAP."},
    )

@dataclass
class DataTrainingArgumentsPost:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    comment_data_args: str = field(
        metadata={"help": "A comment to add to the data arguments."},
    )
    parent_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The project parent directory."},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the data directory."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the downloaded datasets."},
    )
    dev_spkrs_list: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the file with list of speakers for the development set."},
    )
    preprocessing_only: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to only do data preprocessing and skip training. This is especially useful when data"
                " preprocessing errors out in distributed training due to timeout. In this case, one should run the"
                " preprocessing in a non-distributed setup with `preprocessing_only=True` so that the cached datasets"
                " can consequently be loaded in distributed training"
            )
        },
    )
    core_test_spkrs_list: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the file with list of speakers for the TIMIT core test set."},
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    transfer_from: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to transfer the pre-trained model from."},
    )
    experiment: str = field(
        default = None,
        metadata={"help": "The experiment to perform: 'snr','ssl_loss'. If set to SSL loss in evaluation, a model that contains a SSL loss percentage in its name will be expected. Set to 'snr' for most cases."},
    )
    ssl_loss_frame_perc: int = field(
        default=50,
        metadata={"help": "The percentage of frames that was used for pre-training with the SSL loss."},
    )
    dev_data_percent: float = field(
        default=0.1,
        metadata={"help": "The percentage of the training data to use as a validation set in the downstream cross-validation classification tasks."},
    )
    train_data_percent: float = field(
        default=0.9,
        metadata={"help": "The percentage of the training data to use as a training set in the downstream cross-validation classification tasks."},
    )
    random_states: int = list_field(
        default=None,
        metadata={"help": "The number of different random states to use for different dataset splits and classifier parameters in cross-validation tasks."},
    )
    random_states_unsup: int = list_field(
        default=None,
        metadata={"help": "The number of different random states to use for the unsupervised classification task for k-Means initialization."},
    )
    mod_expl_random_states: int = field(
        default=5,
        metadata={"help": "The number of random states to use for modularity explicitness evaluation."},
    )
    disentanglement_eval_cv_splits: int = field(
        default=5,
        metadata={"help": "The number of cross-validation splits to use for disentanglement metrics evaluation."},
    )
    classif_eval_cv_splits: int = field(
        default=5,
        metadata={"help": "The number of cross-validation splits to use for supervised classification evaluation."},
    )
    sim_snr_db: float = field(
        default=None,
        metadata={"help": "The SNR for the simulated datasets."},
    )
    sim_vowels_number: int = field(
        default=None,
        metadata={"help": "The number of vowels contained in the version of the sim_vowels dataset."},
    )
    sim_vowels_duration: float = field(
        default=4.0,
        metadata={"help": "The duration of the sim_vowels dataset."},
    )
    discard_label_overlaps: bool = field(
        default=False,
        metadata={"help": "Whether to discard frames with label overlaps (frames where the label changes) for evaluation."},
    )
    parts_to_use: int = field(
        default=1,
        metadata={"help": "The number of parts to use from the vowels or atoms dataset."},
    )
    validation_split_percentage: float = field(
        default=None,
        metadata={"help": "The percentage of the training data to use for validation if a validation split is not specified."},
    )
    train_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed train data cached file name"},
    )
    validation_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed validation data cached file name"},
    )
    test_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed test data cached file name"},
    )
    dev_cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pre-processed development data cached file name"},
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store model logs and other data."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    with_wandb: bool = field(
        default=False,
        metadata={"help": ("Whether or not to log training and evaluation metrics to Weights & Biases.")},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The Weights & Biases project name to log into."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "The Weights & Biases project group name to log into."},
    )
    epoch_range_to_evaluate: List[int] = list_field(
        default=None,
        metadata={"help": "The range of epochs to evaluate the model on."},
    )
    which_checkpoint: int = field(
        default=-1,
        metadata={"help": "The index of the checkpoint to use for transfer learning. If -1, the last checkpoint will be used."},
    )
    classify: bool = field(
        default=False,
        metadata={"help": "Whether to perform the classification task."},
    )
    classification_tasks: List[str] = list_field(
        default=None,
        metadata={"help": "The classification tasks to perform in latent post-training analysis, as names of label variables. 'all' performs all available tasks."
        "Available variables: SimVowels - 'vowel', 'speaker_frame', 'speaker_seq'. TIMIT - 'phoneme', 'speaker_frame', 'speaker_seq'."
        "VOC-ALS - 'phoneme_frame', 'speaker_frame', 'kings_stage_frame', 'disease_duration_frame', 'alsfrs_total_frame', 'alsfrs_speech_frame', 'group_frame', 'cantagallo_frame',"
        " 'phoneme_seq', 'speaker_seq', 'kings_stage_seq', 'disease_duration_seq', 'alsfrs_total_seq', 'alsfrs_speech_seq', 'group_seq', 'cantagallo_seq'. "
        "IEMOCAP - 'phoneme', 'emotion_frame', 'speaker_frame', 'emotion_seq', 'speaker_seq'."},
    )
    aggregations_to_use: List[str] = list_field(
        default=None,
        metadata={"help": "The aggregation methods to use for the classification tasks: 'all', 'OCs_joint_emb', 'OCs_proj', 'X', 'OCs'. 'all' refers to the case where all available subspaces (X,OC1,...,OCn) are aggregated together."},
    )
    measure_disentanglement: bool = field(
        default=False,
        metadata={"help": "Whether to measure the disentanglement properties of the latent variables."},
    )
    sup_eval: bool = field(
        default=True,
        metadata={"help": "Whether to evaluate representations with a supervised classification task."},
    )
    unsup_eval: bool = field(
        default=False,
        metadata={"help": "Whether to evaluate representations with an unsupervised classification task."},
    )
    vis_method: str = field(
        default='tsne',
        metadata={"help": "The dimensionality reduction method to use for visualization."},
    )
    generative_factors_vis: bool = field(
        default=False,
        metadata={"help": "Whether to visualize manifold with colored generative factors."},
    )
    frequency_vis: bool = field(
        default=False,
        metadata={"help": "Whether to visualize with colored frequency components factors."},
    )
    vis_sphere: bool = field(
        default=False,
        metadata={"help": "Whether to visualize the latent space as a sphere."},
    )
    tsne_plot_2d_3d: str = field(
        default='both',
        metadata={"help": "Whether to plot the tsne in 2d, 3d or both"},
    )
    per_device_train_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size (per device) for the training dataloader."},
    )
    per_device_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size (per device) for the evaluation dataloader."},
    )
    max_duration_in_seconds: float = field(
        default=20.0,
        metadata={
            "help": (
                "Filter audio files that are longer than `max_duration_in_seconds` seconds to"
                " 'max_duration_in_seconds`"
            )
        },
    )
    min_duration_in_seconds: float = field(
        default=0.0, 
        metadata={"help": "Filter audio files that are shorter than `min_duration_in_seconds` seconds"}
    )
    skip_first_n_seconds: float = field(
        default=0.0,
        metadata={"help": "Skip the first n seconds of the audio file."},
    )
    pad_to_multiple_of: Optional[int] = field(
        default=None,
        metadata={"help": "Pad to multiple of - used in the data collator.s"},
    )
    seed: int = field(
        default=0,
        metadata={"help": "A seed for reproducible training."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes used in preprocessing; necessary to correctly load the data."},
    )
    classif_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the sklearn processes."},
    )
    hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to use as HTTP bearer authorization for remote files."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    path_to_timit_phoneme48_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the phoneme48-to-int dictionary for TIMIT."},
    )
    path_to_timit_phoneme39_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the phoneme39-to-int dictionary for TIMIT."},
    )
    path_to_timit_speaker_dict_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the speaker dictionary for TIMIT."},
    )
    input_type: str = field(
        default="waveform",
        metadata={"help": "The input type to the model - waveform or mel."},
    )
    n_mels: int = field(
        default=80,
        metadata={"help": "The number of mel bands to use."},
    )
    mel_norm: str = field(
        default=None,
        metadata={"help": "The normalization to apply to the mel spectrogram."},
    )
    mel_hops: int = field(
        default=4,
        metadata={"help": "The number of hops to use for computing the mel spectrogram."},
    )
    path_to_voc_als_encodings: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the encodings for the VOC ALS dataset variables."},
    )
    path_to_iemocap_phoneme_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the phoneme-to-int dictionary for IEMOCAP."},
    )
    path_to_iemocap_emotion_to_id_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the emotion-to-int dictionary for IEMOCAP."},
    )
    path_to_iemocap_speaker_dict_file: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the file containing the speaker dictionary for IEMOCAP."},
    )
