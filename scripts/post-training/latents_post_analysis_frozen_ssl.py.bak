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

"""This script handles all latent evaluations (classification, disentanglement) for the frozen speech
SSL baselines - wav2vec2, HuBERT and WavLM. The encoders are used as published, with no pre-training
and no fine-tuning here, so there are no checkpoints to iterate over.

They consume the un-framed utterance waveform held in input_seq_values and emit one embedding per
frame. Their conv stack has receptive field 400 and total stride 320, which is the same grid the
labels are interpolated onto at preprocessing, so the embeddings align with the labels frame for
frame. Supported for SimVowels, TIMIT and IEMOCAP.

Decomposition of inputs is not supported here so if it's not already calculated then another script
like base_models_ssl_pretraining.py should be ran first."""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import build_frozen_ssl
from data_collation import DataCollatorForDecVAELatentPostAnalysis_NoFeatureExtraction
from config_files import DecVAEConfig
from args_configs import (
    ModelArgumentsPost,
    DataTrainingArgumentsPost,
    DecompositionArguments,
    TrainingObjectiveArguments,
    FrozenSSLArguments,
)
from utils import parse_args, debugger_is_active
from utils.cache_utils import build_cache_file_names
from latent_analysis_utils import prediction_eval
from disentanglement_utils import compute_disentanglement_metrics
from sklearn.decomposition import PCA
import joblib
import numpy as np

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

import pandas as pd
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from torch.utils.data.dataloader import DataLoader
import time

JSON_FILE_NAME_MANUAL = "config_files/baselines/wav2vec2/sim_vowels/latent_evaluations/config_wav2vec2_latent_anal_sim_vowels.json"

logger = get_logger(__name__)

SUPPORTED_DATASETS = ["sim_vowels", "timit", "iemocap"]


def _common_device(*values):
    """
    Device of the tensors found in values, which may be nested lists of them.

    Returns None when none of them is a tensor, so torch keeps its default, and raises when they
    disagree - labels built on different devices cannot be concatenated later on.
    """
    devices = set()
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, torch.Tensor):
            devices.add(value.device)
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    if len(devices) > 1:
        raise ValueError(f"The labels and the overlap mask are on different devices: {sorted(str(d) for d in devices)}")
    return next(iter(devices)) if devices else None


def _expand_to_frames(seq_values, overlap_mask_batch):
    "Repeat an utterance-level factor once per kept frame of that utterance"
    # torch.tensor builds on the CPU by default, so place the result on the inputs' device
    device = _common_device(seq_values, overlap_mask_batch)
    return torch.cat([
        torch.tensor([factor for _ in range(int((~overlap_mask_batch[i]).sum()))], device=device)
        for i, factor in enumerate(seq_values)
    ])


def fit_pca(z_fit, frozen_ssl_args, projection_path):
    """
    Fit a PCA on a seeded sample of the collected embeddings, or load one fitted earlier.

    Args:
        z_fit (torch.Tensor): (frames, hidden) embeddings to fit on.
        frozen_ssl_args (:class:`~args_configs.frozen_ssl_args.FrozenSSLArguments`)
        projection_path (str): Where the fitted PCA is cached.
    Returns:
        The fitted :class:`~sklearn.decomposition.PCA`.
    """
    if os.path.exists(projection_path):
        print(f"Loading the fitted PCA from {projection_path}")
        return joblib.load(projection_path)

    n_components = frozen_ssl_args.ssl_pca_components
    "PCA needs at least as many samples as components, so the fraction cannot cut below that"
    n_fit = int(z_fit.shape[0] * frozen_ssl_args.ssl_pca_fit_fraction)
    n_fit = min(max(n_fit, n_components), z_fit.shape[0])

    rng = np.random.default_rng(seed=frozen_ssl_args.ssl_pca_seed)
    indices = rng.choice(z_fit.shape[0], size=n_fit, replace=False)
    print(f"Fitting PCA to {n_components} components on {n_fit} of {z_fit.shape[0]} frames")

    pca = PCA(n_components=n_components, random_state=0)
    pca.fit(z_fit[indices])
    print(f"Explained variance ratio, summed: {pca.explained_variance_ratio_.sum(): .4f}")

    os.makedirs(os.path.dirname(projection_path), exist_ok=True)
    joblib.dump(pca, projection_path)
    return pca


def gather_split(dataloader, representation_function, data_training_args):
    """
    Run the frozen encoder over a split and collect frame-level embeddings with their labels.

    Args:
        dataloader: The dataloader of the split.
        representation_function: The frozen SSL encoder.
        data_training_args: Data and training related arguments.
    Returns:
        z (torch.Tensor): (frames, hidden) embeddings of every kept frame.
        labels (dict): Label name -> tensor, aligned with z along the frame axis.
    """
    dataset_name = data_training_args.dataset_name
    z = None
    labels = {}

    def append(name, value):
        labels[name] = value if name not in labels else torch.cat((labels[name], value), dim=0)

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            batch_size = batch["input_values"].shape[0]
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            overlap_mask_batch = batch.pop("overlap_mask", None)

            assert overlap_mask_batch is not None if dataset_name in ["timit", "iemocap"] else True
            if overlap_mask_batch is None or not data_training_args.discard_label_overlaps:
                overlap_mask_batch = torch.zeros_like(sub_attention_mask, dtype=torch.bool)
            else:
                "Frames corresponding to padding are set as True in the overlap and discarded"
                padded = sub_attention_mask.sum(dim=-1)
                for b in range(batch_size):
                    overlap_mask_batch[b, padded[b]:] = 1
                overlap_mask_batch = overlap_mask_batch.bool()

            if dataset_name == "sim_vowels":
                vowel_labels_batch = batch.pop("vowel_labels", None)
                speaker_vt_factor_batch = batch.pop("speaker_vt_factor", None)
                vowel_labels_batch = [
                    [ph for i, ph in enumerate(utt) if not overlap_mask_batch[j, i]]
                    for j, utt in enumerate(vowel_labels_batch)
                ]
            elif dataset_name == "timit":
                phonemes39_batch = batch.pop("phonemes39", None)[~overlap_mask_batch]
                phonemes48_batch = batch.pop("phonemes48", None)[~overlap_mask_batch]
                batch.pop("start_phonemes", None)
                batch.pop("stop_phonemes", None)
                speaker_id_batch = list(batch.pop("speaker_id", None))
            elif dataset_name == "iemocap":
                phonemes_batch = batch.pop("phonemes", None)[~overlap_mask_batch]
                emotion_batch = list(batch.pop("emotion", None))
                batch.pop("start_phonemes", None)
                batch.pop("stop_phonemes", None)
                speaker_id_batch = list(batch.pop("speaker_id", None))

            "The frozen encoders read the un-framed sequence - component 0 is the original signal"
            batch["input_seq_values"] = batch["input_seq_values"].squeeze(1)
            waveform = batch["input_seq_values"][:, 0, :]
            outputs = representation_function(input_values=waveform, attention_mask=batch["attention_mask"])
            del batch

            "Gather labels for evaluations"
            if dataset_name == "sim_vowels":
                vowel_device = _common_device(vowel_labels_batch, overlap_mask_batch)
                append("vowel", torch.cat([torch.tensor(v, device=vowel_device) for v in vowel_labels_batch]))
                append("speaker_frame", _expand_to_frames(speaker_vt_factor_batch, overlap_mask_batch))
                append("speaker_seq", speaker_vt_factor_batch.clone())
            elif dataset_name == "timit":
                append("phoneme39", phonemes39_batch.clone())
                append("phoneme48", phonemes48_batch.clone())
                append("speaker_frame", _expand_to_frames(speaker_id_batch, overlap_mask_batch))
                append("speaker_seq", torch.stack(speaker_id_batch))
            elif dataset_name == "iemocap":
                append("phoneme", phonemes_batch.clone())
                append("speaker_frame", _expand_to_frames(speaker_id_batch, overlap_mask_batch))
                append("emotion_frame", _expand_to_frames(emotion_batch, overlap_mask_batch))
                append("emotion_seq", torch.stack(emotion_batch))

            "Gather latents for evaluations - the encoder emits every frame, including the padded ones"
            z_batch = torch.masked_select(
                outputs[0], ~overlap_mask_batch[..., None]
            ).reshape(-1, outputs[0].shape[-1])
            z = z_batch.detach().cpu() if step == 0 else torch.cat((z, z_batch.detach().cpu()), dim=0)

    return z, labels


def main():
    "Parse the arguments"
    parser = HfArgumentParser((ModelArgumentsPost, DataTrainingArgumentsPost, TrainingObjectiveArguments,
                               DecompositionArguments, FrozenSSLArguments))
    if debugger_is_active() or ('TERM_PROGRAM' in os.environ.keys() and os.environ['TERM_PROGRAM'] == 'vscode'):
        model_args, data_training_args, training_obj_args, decomp_args, frozen_ssl_args = parser.parse_json_file(json_file=JSON_FILE_NAME_MANUAL)
    else:
        args = parse_args()
        model_args, data_training_args, training_obj_args, decomp_args, frozen_ssl_args = parser.parse_json_file(json_file=args.config_file)
    delattr(model_args, "comment_model_args")
    delattr(data_training_args, "comment_data_args")
    delattr(training_obj_args, "comment_tr_obj_args")
    delattr(decomp_args, "comment_decomp_args")
    delattr(frozen_ssl_args, "comment_frozen_ssl_args")

    if data_training_args.dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"The frozen SSL baselines are set up for {SUPPORTED_DATASETS}, got "
            f"'{data_training_args.dataset_name}'."
        )

    "Initialize the accelerator. Accelerator handles device placement for us"
    kwargs = DDPK(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[kwargs])
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()

        # set up weights and biases if available
        if is_wandb_available() and data_training_args.with_wandb:
            import wandb

            wandb.init(project=data_training_args.wandb_project, group=data_training_args.wandb_group)
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    "If passed along, set the training seed now."
    if data_training_args.seed is not None:
        set_seed(data_training_args.seed)

    accelerator.wait_for_everyone()

    "preprocess the datasets including loading the audio, resampling and normalization"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)

    "set max & min audio length in number of samples"
    max_length = int(data_training_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_length = int(data_training_args.min_duration_in_seconds * feature_extractor.sampling_rate)

    "load cached preprocessed files"
    if data_training_args.train_cache_file_name is None or data_training_args.validation_cache_file_name is None:
        raise ValueError("cache_file_names is not defined. Please define it in the config file.")
    else:
        cache_file_names = build_cache_file_names(data_training_args, data_training_args.input_type)

    "Load model with hyperparameters"
    model_args.max_duration_in_seconds = data_training_args.max_duration_in_seconds
    config = DecVAEConfig(**{**model_args.__dict__, **training_obj_args.__dict__, **decomp_args.__dict__})

    "load audio files into numpy arrays"
    with accelerator.main_process_first():

        vectorized_datasets = DatasetDict()
        vectorized_datasets["train"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["train"]])
        if data_training_args.dataset_name == "timit":
            vectorized_datasets["validation"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["dev"]])
        else:
            vectorized_datasets["validation"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["validation"]])
        vectorized_datasets["test"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["test"]])

        if min_length > 0.0:
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: x > min_length,
                num_proc=data_training_args.preprocessing_num_workers,
                input_columns=["input_length"],
            )
        vectorized_datasets = vectorized_datasets.remove_columns("input_length")

    "Make sure to obtain all the samples in the dataset"
    assert config.max_frames_per_batch == "all"

    "There is nothing to load - the published weights are the representation. The run is named after"
    "the encoder and layer so that results from different encoders do not overwrite each other"
    encoder_name = frozen_ssl_args.ssl_model_name_or_path.split("/")[-1]
    ckp = encoder_name + "_layer" + str(frozen_ssl_args.ssl_hidden_layer)
    print(f"Evaluating frozen encoder {frozen_ssl_args.ssl_model_name_or_path}, "
          f"hidden layer {frozen_ssl_args.ssl_hidden_layer}")

    representation_function = build_frozen_ssl(frozen_ssl_args)
    representation_function.eval()
    for param in representation_function.parameters():
        param.requires_grad = False

    "data collator"
    mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
    mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

    data_collator = DataCollatorForDecVAELatentPostAnalysis_NoFeatureExtraction(
        model=representation_function,
        feature_extractor=feature_extractor,
        model_args=model_args,
        data_training_args=data_training_args,
        config=config,
        input_type=data_training_args.input_type,
        dataset_name=data_training_args.dataset_name,
        pad_to_multiple_of=data_training_args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length
    )

    "Optional PCA on the collected embeddings, fitted on the train split. The encoders emit 768"
    "dimensions, which the metrics that follow are sensitive to"
    use_pca = frozen_ssl_args.ssl_pca_components is not None and frozen_ssl_args.ssl_pca_components > 0
    if use_pca:
        ckp += "_pca" + str(frozen_ssl_args.ssl_pca_components)
        projection_path = os.path.join(data_training_args.parent_dir, "pca_projections", ckp + "_model.joblib")

    "The frozen encoders are evaluated on ordered frames"
    if data_training_args.dataset_name == "iemocap":
        eval_dataset = concatenate_datasets([vectorized_datasets["train"], vectorized_datasets["validation"], vectorized_datasets["test"]])
        eval_dataloader = DataLoader(
            eval_dataset,
            shuffle=False,
            collate_fn=data_collator,
            batch_size=data_training_args.per_device_train_batch_size
        )
    else:
        eval_dataloader = DataLoader(
            vectorized_datasets["validation"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=data_training_args.per_device_eval_batch_size
        )
        test_dataloader = DataLoader(
            vectorized_datasets["test"],
            shuffle=False,
            collate_fn=data_collator,
            batch_size=data_training_args.per_device_eval_batch_size
        )
        if use_pca:
            "The train split is read for the PCA fit alone - it is not evaluated"
            train_dataloader = DataLoader(
                vectorized_datasets["train"],
                shuffle=False,
                collate_fn=data_collator,
                batch_size=data_training_args.per_device_eval_batch_size
            )

    "Prepare everything with HF accelerator"
    if data_training_args.dataset_name == "iemocap":
        "Evaluates on a single set"
        representation_function, eval_dataloader = accelerator.prepare(
            representation_function, eval_dataloader
        )
    else:
        representation_function, eval_dataloader, test_dataloader = accelerator.prepare(
            representation_function, eval_dataloader, test_dataloader
        )
        if use_pca:
            train_dataloader = accelerator.prepare(train_dataloader)

    "Measure total loading time"
    start_time = time.time()
    "Get the representations"
    z, labels = gather_split(eval_dataloader, representation_function, data_training_args)
    if data_training_args.dataset_name == "iemocap":
        z_test, labels_test = None, {}
    else:
        z_test, labels_test = gather_split(test_dataloader, representation_function, data_training_args)
    print(f"Total loading time: {time.time() - start_time: .4f} seconds")

    if use_pca:
        if data_training_args.dataset_name == "iemocap":
            "Every split is already in the single evaluation set - fit on a sample of it"
            z_fit = z
        else:
            z_fit, _ = gather_split(train_dataloader, representation_function, data_training_args)
        pca = fit_pca(z_fit, frozen_ssl_args, projection_path)
        del z_fit

        z = torch.tensor(pca.transform(z), dtype=torch.float32)
        if z_test is not None:
            z_test = torch.tensor(pca.transform(z_test), dtype=torch.float32)

    "Now use train/val representations to get the evaluation metrics"
    "Linear/non-linear classification"
    tasks = data_training_args.classification_tasks
    if data_training_args.classify:
        "Label to gather, target name to record it under, and the task that switches it on -"
        "the three differ, following the naming latents_post_analysis.py already uses"
        if data_training_args.dataset_name == "sim_vowels":
            frame_targets = [("vowel", "vowel", "vowel"),
                             ("speaker_frame", "speaker_frame", "speaker_frame")]
        elif data_training_args.dataset_name == "timit":
            frame_targets = [("phoneme48", "phoneme48", "phoneme"),
                             ("speaker_frame", "speaker_frame", "speaker_frame")]
        else:
            frame_targets = [("phoneme", "phoneme_frame", "phoneme"),
                             ("speaker_frame", "speaker_frame", "speaker_frame"),
                             ("emotion_frame", ["cat_emotion_frame", "speaker_frame"], "emotion_frame")]

        for label_name, target, task in frame_targets:
            if task not in tasks and "all" not in tasks:
                continue
            prediction_eval(data_training_args, config,
                X=z, X_test=z_test,
                y=labels[label_name], y_test=labels_test.get(label_name),
                checkpoint=ckp, latent_type="ssl", target=target
            )

    "Disentanglement metrics"
    if data_training_args.measure_disentanglement:
        if data_training_args.dataset_name == "sim_vowels":
            columns = ["vowel", "speaker_frame"]
            names = ["vowel", "speaker_frame"]
        elif data_training_args.dataset_name == "timit":
            columns = ["phoneme", "speaker_frame"]
            names = ["phoneme39", "speaker_frame"]
        else:
            columns = ["phoneme", "speaker_frame", "cat_emotion_frame"]
            names = ["phoneme", "speaker_frame", "emotion_frame"]

        y_frame_train = torch.cat([labels[n].reshape(-1, 1) for n in names], dim=1)
        y_frame_train = pd.DataFrame(y_frame_train.cpu().numpy(), columns=columns)
        if z_test is not None:
            y_frame_test = torch.cat([labels_test[n].reshape(-1, 1) for n in names], dim=1)
            y_frame_test = pd.DataFrame(y_frame_test.cpu().numpy(), columns=columns)
        else:
            y_frame_test = None

        compute_disentanglement_metrics(data_training_args, config, checkpoint=ckp,
            latent_type="ssl", mu_train=z, y_train=y_frame_train,
            mu_test=z_test, y_test=y_frame_test, target=columns
        )


if __name__ == "__main__":
    main()
