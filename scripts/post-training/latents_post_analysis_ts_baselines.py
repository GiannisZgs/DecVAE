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

"""This script handles all latent evaluations (classification, disentanglement) for the pre-trained
time-series baselines (CoST, TF-C, TCL, CPC, FHVAE), checkpoint by checkpoint as pre-trained by
ts_baselines_pretraining.py. The methods are frame-level: they read the framed decomposition and
emit one embedding per frame of the DecVAE grid. Sequence-level targets are read off the mean of the
frame embeddings of each utterance, as latents_post_analysis_frozen_ssl.py does.
Supported for SimVowels, TIMIT and IEMOCAP. The caches must already exist."""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import build_cost, build_tfc, build_tcl, build_cpc, build_fhvae
from data_collation import DataCollatorForDecVAELatentPostAnalysis_NoFeatureExtraction
from config_files import DecVAEConfig
from args_configs import (
    ModelArgumentsPost,
    DataTrainingArgumentsPost,
    DecompositionArguments,
    TrainingObjectiveArguments,
    BaselinePretrainingArguments,
    CoSTArguments,
    TFCArguments,
    TCLArguments,
    CPCArguments,
    FHVAEArguments,
)
from utils import parse_args, debugger_is_active
from utils.misc import extract_epoch
from utils.cache_utils import build_cache_file_names
from latent_analysis_utils import prediction_eval
from disentanglement_utils import compute_disentanglement_metrics
from safetensors import safe_open
from safetensors.torch import load_file
import copy
import json
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

JSON_FILE_NAME_MANUAL = "config_files/baselines/fhvae/sim_vowels/latent_evaluations/config_fhvae_latent_anal_sim_vowels.json"

logger = get_logger(__name__)

SUPPORTED_DATASETS = ["sim_vowels", "timit", "iemocap"]

"Random initialization, evaluated alongside the checkpoints as the VAE script does"
RANDOM_INIT = "epoch_-01"

"Settings that must agree with pre-training, or the checkpoint is read with the wrong front-end"
"or architecture. Mel and waveform frames are both 400 wide, so a strict load cannot catch it"
MUST_MATCH_PRETRAINING = {
    "cost": ("baseline_method", "baseline_component", "cost_input_type", "cost_n_mels", "cost_mel_norm",
             "cost_pool_mel_bins", "cost_output_dims", "cost_hidden_dims", "cost_depth", "cost_kernels",
             "mel_hops", "decomp_to_perform", "NoC", "receptive_field", "stride", "max_duration_in_seconds"),
    "tfc": ("baseline_method", "baseline_component", "tfc_input_type", "tfc_n_mels", "tfc_mel_norm",
            "tfc_pool_mel_bins", "tfc_transformer_layers", "tfc_attention_heads", "tfc_feedforward_multiplier",
            "tfc_projector_hidden", "tfc_projector_dim", "tfc_representation",
            "mel_hops", "decomp_to_perform", "NoC", "receptive_field", "stride", "max_duration_in_seconds"),
    "tcl": ("baseline_method", "baseline_component", "tcl_input_type", "tcl_n_mels", "tcl_mel_norm",
            "tcl_pool_mel_bins", "tcl_hidden_nodes", "tcl_maxout_k", "tcl_feature_nonlinearity",
            "tcl_pca_components", "tcl_segment_duration_in_seconds",
            "mel_hops", "decomp_to_perform", "NoC", "receptive_field", "stride", "max_duration_in_seconds"),
    "cpc": ("baseline_method", "baseline_component", "cpc_input_type", "cpc_n_mels", "cpc_mel_norm",
            "cpc_pool_mel_bins", "cpc_conv_kernel", "cpc_conv_stride", "cpc_encoder_hidden",
            "cpc_encoder_dim", "cpc_ar_dim", "cpc_ar_layers", "cpc_ar_mode", "cpc_encoder_norm",
            "cpc_representation",
            "mel_hops", "decomp_to_perform", "NoC", "receptive_field", "stride", "max_duration_in_seconds"),
    "fhvae": ("baseline_method", "baseline_component", "fhvae_input_type", "fhvae_n_mels",
              "fhvae_mel_norm", "fhvae_pool_mel_bins", "fhvae_seg_len", "fhvae_hidden",
              "fhvae_d_seg", "fhvae_d_seq", "fhvae_seq_std", "fhvae_representation",
              "mel_hops", "decomp_to_perform", "NoC", "receptive_field", "stride", "max_duration_in_seconds"),
}


def resolve_method_args(baseline_args, cost_args, tfc_args, tcl_args, cpc_args, fhvae_args):
    """
    Select the argument group of the configured method and the front-end it reads.

    Returns:
        method_args, input_type (str), n_mels (int), mel_norm (str or None), pool_mel_bins (bool)
    """
    if baseline_args.baseline_method == "cost":
        if cost_args.cost_input_type not in ("mel", "waveform"):
            raise ValueError(f"Unknown cost_input_type {cost_args.cost_input_type}, expected 'mel' or 'waveform'.")
        return (cost_args, cost_args.cost_input_type, cost_args.cost_n_mels,
                cost_args.cost_mel_norm, cost_args.cost_pool_mel_bins)

    if baseline_args.baseline_method == "tfc":
        if tfc_args.tfc_input_type not in ("mel", "waveform"):
            raise ValueError(f"Unknown tfc_input_type {tfc_args.tfc_input_type}, expected 'mel' or 'waveform'.")
        return (tfc_args, tfc_args.tfc_input_type, tfc_args.tfc_n_mels,
                tfc_args.tfc_mel_norm, tfc_args.tfc_pool_mel_bins)

    if baseline_args.baseline_method == "tcl":
        if tcl_args.tcl_input_type not in ("mel", "waveform"):
            raise ValueError(f"Unknown tcl_input_type {tcl_args.tcl_input_type}, expected 'mel' or 'waveform'.")
        return (tcl_args, tcl_args.tcl_input_type, tcl_args.tcl_n_mels,
                tcl_args.tcl_mel_norm, tcl_args.tcl_pool_mel_bins)

    if baseline_args.baseline_method == "cpc":
        if cpc_args.cpc_input_type not in ("mel", "waveform"):
            raise ValueError(f"Unknown cpc_input_type {cpc_args.cpc_input_type}, expected 'mel' or 'waveform'.")
        return (cpc_args, cpc_args.cpc_input_type, cpc_args.cpc_n_mels,
                cpc_args.cpc_mel_norm, cpc_args.cpc_pool_mel_bins)

    if baseline_args.baseline_method == "fhvae":
        if fhvae_args.fhvae_input_type not in ("mel", "waveform"):
            raise ValueError(f"Unknown fhvae_input_type {fhvae_args.fhvae_input_type}, expected 'mel' or 'waveform'.")
        return (fhvae_args, fhvae_args.fhvae_input_type, fhvae_args.fhvae_n_mels,
                fhvae_args.fhvae_mel_norm, fhvae_args.fhvae_pool_mel_bins)

    raise ValueError(
        f"Unknown baseline_method {baseline_args.baseline_method}, expected one of 'cost', 'tfc', "
        "'tcl', 'cpc', 'fhvae'."
    )


def build_baseline_model(baseline_args, method_args, input_dims, seq_length, config, n_train_utts=None):
    if baseline_args.baseline_method == "cost":
        return build_cost(method_args, input_dims, seq_length, config)

    if baseline_args.baseline_method == "tfc":
        return build_tfc(method_args, input_dims, seq_length, config)

    if baseline_args.baseline_method == "tcl":
        return build_tcl(method_args, input_dims, seq_length, config)

    if baseline_args.baseline_method == "cpc":
        return build_cpc(method_args, input_dims, seq_length, config)

    if baseline_args.baseline_method == "fhvae":
        if n_train_utts is None:
            raise ValueError(
                "FHVAE carries a prior-mean table with one row per training utterance of the run "
                "that produced the checkpoint, so it has to be built with n_train_utts."
            )
        return build_fhvae(method_args, input_dims, seq_length, config, n_train_utts)

    raise ValueError(
        f"Unknown baseline_method {baseline_args.baseline_method}, expected one of 'cost', 'tfc', "
        "'tcl', 'cpc', 'fhvae'."
    )


def representation_name(baseline_args, method_args):
    "Variant of the representation, recorded in the checkpoint name so variants do not overwrite each other"
    if baseline_args.baseline_method == "cost":
        return method_args.cost_representation
    if baseline_args.baseline_method == "tfc":
        return method_args.tfc_representation
    if baseline_args.baseline_method == "cpc":
        return method_args.cpc_representation
    if baseline_args.baseline_method == "fhvae":
        return method_args.fhvae_representation
    return baseline_args.baseline_method


def resolve_seq_pooling(baseline_args, method_args):
    "How the frames of an utterance are pooled for the '*_seq' targets, per method"
    if baseline_args.baseline_method == "cost":
        return method_args.cost_seq_pooling
    if baseline_args.baseline_method == "tfc":
        return method_args.tfc_seq_pooling
    if baseline_args.baseline_method == "tcl":
        return method_args.tcl_seq_pooling
    if baseline_args.baseline_method == "cpc":
        return method_args.cpc_seq_pooling
    if baseline_args.baseline_method == "fhvae":
        return method_args.fhvae_seq_pooling

    raise ValueError(
        f"Unknown baseline_method {baseline_args.baseline_method}, expected one of 'cost', 'tfc', "
        "'tcl', 'cpc', 'fhvae'."
    )


def series_shape(dataset, component, n_mels, pool_mel_bins):
    """
    Frame count and frame width the model reads, measured off the cache as pre-training does.

    Returns:
        seq_length (int), input_dims (int)
    """
    example = np.asarray(dataset[0]["input_values"])
    if example.ndim != 4 or example.shape[0] != 1:
        raise ValueError(
            f"Expected (1, components, frames, features) cached inputs, as DecVAE's preprocessing "
            f"writes them, got shape {example.shape}."
        )
    if component >= example.shape[1]:
        raise ValueError(f"baseline_component {component} was asked for, but the cache holds only "
                         f"{example.shape[1]} components.")
    return int(example.shape[2]), int(n_mels if pool_mel_bins else example.shape[3])


def check_against_pretraining(checkpoint_dir, config_path, baseline_args, method_args, data_training_args, decomp_args):
    """
    Compare the settings that fix the front-end and the architecture with the pre-training config
    that ts_baselines_pretraining.py copied next to the checkpoints.
    """
    pretraining_configs = [f for f in os.listdir(checkpoint_dir) if f.endswith(".json")]
    if len(pretraining_configs) != 1:
        print(f"Warning: expected one pre-training config in {checkpoint_dir}, found {pretraining_configs}. "
              "The front-end and architecture cannot be checked against pre-training.")
        return

    with open(os.path.join(checkpoint_dir, pretraining_configs[0]), "r") as f:
        pretraining = json.load(f)
    current = {**vars(decomp_args), **vars(data_training_args), **vars(baseline_args), **vars(method_args)}

    mismatches = []
    for key in MUST_MATCH_PRETRAINING[baseline_args.baseline_method]:
        if key not in pretraining:
            continue
        if pretraining[key] != current.get(key):
            mismatches.append(f"{key}: pre-training {pretraining[key]!r}, evaluation {current.get(key)!r}")
    if mismatches:
        raise ValueError(f"{config_path} disagrees with the pre-training config {pretraining_configs[0]}:\n  "
                         + "\n  ".join(mismatches))


def list_checkpoints(checkpoint_dir, epoch_range_to_evaluate):
    "Checkpoints sorted by epoch, the random initialization first, selected as in the VAE script"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                        if os.path.isdir(os.path.join(checkpoint_dir, f)) and "epoch_" in f]
    checkpoint_files.append(RANDOM_INIT)
    checkpoint_files.sort(key=extract_epoch)

    if epoch_range_to_evaluate is None:
        return checkpoint_files
    if len(epoch_range_to_evaluate) == 2:
        if epoch_range_to_evaluate[1] == -1:
            return checkpoint_files[epoch_range_to_evaluate[0]:]
        return checkpoint_files[epoch_range_to_evaluate[0]:epoch_range_to_evaluate[1]]
    if len(epoch_range_to_evaluate) == 1:
        return [checkpoint_files[epoch_range_to_evaluate[0]]]
    raise ValueError("epoch_range_to_evaluate should be a list of 1 or 2 integers, or None. Please check your config file.")


def table_rows_from_checkpoint(checkpoint_dir, checkpoints):
    """
    Rows of FHVAE's prior-mean table, read off a checkpoint.

    The table has one row per training utterance of the run that produced the checkpoint, a count
    the evaluation config does not carry, so it is read from the checkpoint itself. The randomly
    initialized model is then built the same size, so the two stay comparable and the strict load
    of every other checkpoint succeeds.

    Args:
        checkpoint_dir (str): Directory the checkpoints sit in
        checkpoints (list): Checkpoint names, the random initialization included
    Returns:
        int
    """
    for name in checkpoints:
        if name == RANDOM_INIT:
            continue
        path = os.path.join(checkpoint_dir, name, "model.safetensors")
        if not os.path.exists(path):
            continue
        with safe_open(path, framework="pt") as handle:
            if "mu_seq_table" in handle.keys():
                return int(handle.get_slice("mu_seq_table").get_shape()[0])

    raise ValueError(
        f"No checkpoint under {checkpoint_dir} holds an FHVAE prior-mean table, so the number of "
        "training utterances it was pre-trained with cannot be recovered. Point parent_dir at the "
        "directory the FHVAE pre-training run wrote."
    )


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
    device = _common_device(seq_values, overlap_mask_batch)
    return torch.cat([
        torch.tensor([factor for _ in range(int((~overlap_mask_batch[i]).sum()))], device=device)
        for i, factor in enumerate(seq_values)
    ])


def gather_split(dataloader, representation_function, data_training_args, component, n_mels, pool_mel_bins, seq_pooling):
    """
    Run the baseline over a split and collect frame-level embeddings with their labels.

    Returns:
        z (torch.Tensor): (frames, dim) embeddings of every kept frame.
        z_seq (torch.Tensor): (utterances, dim) pooled embeddings, or None when pooling is off.
        labels (dict): Label name -> tensor. The frame-level entries are aligned with z, the '*_seq'
            ones with z_seq.
    """
    dataset_name = data_training_args.dataset_name
    if seq_pooling not in (None, "mean", "last", "mu"):
        raise ValueError(
            f"The sequence pooling must be None, 'mean', 'last' or 'mu', got '{seq_pooling}'."
        )
    if seq_pooling == "mu" and not hasattr(representation_function, "pooled_sequence_embedding"):
        raise ValueError(
            "The 'mu' pooling is FHVAE's closed-form prior-mean estimate, and this model does not "
            "define pooled_sequence_embedding."
        )
    z = None
    z_seq = None
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

            "Same front-end as pre-training: one component, optionally pooled mel bins, padding zeroed"
            frames = batch["input_values"][:, component, ...]
            if pool_mel_bins:
                frames = frames.reshape(*frames.shape[:-1], n_mels, -1).mean(dim=-1)
            frames = frames * sub_attention_mask.to(frames.dtype).unsqueeze(-1)
            "FHVAE slides a window over the frames, so it has to be told where the valid ones end."
            "The frame-level methods read each frame on its own and are handed no mask, as before"
            if getattr(representation_function, "encode_requires_mask", False):
                outputs = representation_function.encode(frames, sub_attention_mask)
            else:
                outputs = representation_function.encode(frames)
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
                append("speaker_seq", torch.stack(speaker_id_batch))
                append("emotion_seq", torch.stack(emotion_batch))

            "Gather latents for evaluations"
            z_batch = torch.masked_select(
                outputs[0], ~overlap_mask_batch[..., None]
            ).reshape(-1, outputs[0].shape[-1])
            z = z_batch.detach().cpu() if step == 0 else torch.cat((z, z_batch.detach().cpu()), dim=0)

            "Mean over every frame of the padded sequence, as the frozen SSL baselines and DecVAE's SequenceAggregator do"
            if seq_pooling == "mean":
                z_seq_batch = outputs[0].mean(dim=1).detach().cpu()
                z_seq = z_seq_batch if step == 0 else torch.cat((z_seq, z_seq_batch), dim=0)
            elif seq_pooling == "last":
                "The embedding at the last non-padded frame. Only a method with a recurrent context"
                "has a frame that has already seen the whole utterance, so this is CPC's alone"
                last = (sub_attention_mask.sum(dim=-1).clamp(min=1) - 1).to(outputs[0].device)
                rows = torch.arange(batch_size, device=outputs[0].device)
                z_seq_batch = outputs[0][rows, last].detach().cpu()
                z_seq = z_seq_batch if step == 0 else torch.cat((z_seq, z_seq_batch), dim=0)
            elif seq_pooling == "mu":
                "FHVAE's s-vector, the closed-form prior-mean estimate over the non-overlapping"
                "segments. Up to the factor N/(N+sigma^2) it is mean pooling of the sequence latent"
                z_seq_batch = representation_function.pooled_sequence_embedding(
                    frames, sub_attention_mask
                ).detach().cpu()
                z_seq = z_seq_batch if step == 0 else torch.cat((z_seq, z_seq_batch), dim=0)

    return z, z_seq, labels


def main():
    "Parse the arguments"
    parser = HfArgumentParser((ModelArgumentsPost, DataTrainingArgumentsPost, TrainingObjectiveArguments,
                               DecompositionArguments, BaselinePretrainingArguments, CoSTArguments,
                               TFCArguments, TCLArguments, CPCArguments, FHVAEArguments))
    if debugger_is_active() or ('TERM_PROGRAM' in os.environ.keys() and os.environ['TERM_PROGRAM'] == 'vscode'):
        config_path = JSON_FILE_NAME_MANUAL
    else:
        args = parse_args()
        config_path = args.config_file
    model_args, data_training_args, training_obj_args, decomp_args, baseline_args, cost_args, tfc_args, tcl_args, cpc_args, fhvae_args = parser.parse_json_file(json_file=config_path)
    delattr(model_args, "comment_model_args")
    delattr(data_training_args, "comment_data_args")
    delattr(training_obj_args, "comment_tr_obj_args")
    delattr(decomp_args, "comment_decomp_args")
    delattr(baseline_args, "comment_baseline_args")
    delattr(cost_args, "comment_cost_args")
    delattr(tfc_args, "comment_tfc_args")
    delattr(tcl_args, "comment_tcl_args")
    delattr(cpc_args, "comment_cpc_args")
    delattr(fhvae_args, "comment_fhvae_args")

    if data_training_args.dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"The time-series baselines are set up for {SUPPORTED_DATASETS}, got "
            f"'{data_training_args.dataset_name}'."
        )

    method_args, input_type, n_mels, mel_norm, pool_mel_bins = resolve_method_args(baseline_args, cost_args, tfc_args, tcl_args, cpc_args, fhvae_args)
    component = baseline_args.baseline_component
    seq_pooling = resolve_seq_pooling(baseline_args, method_args)

    "The collator normalizes with the data arguments - hand it the front-end the method was pre-trained with"
    collator_data_args = copy.copy(data_training_args)
    collator_data_args.input_type = input_type
    collator_data_args.n_mels = n_mels
    collator_data_args.mel_norm = mel_norm

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
        cache_file_names = build_cache_file_names(data_training_args, input_type)

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

    "The checkpoints are the pre-training output_dir"
    checkpoint_dir = data_training_args.parent_dir
    check_against_pretraining(checkpoint_dir, config_path, baseline_args, method_args, data_training_args, decomp_args)
    checkpoints = list_checkpoints(checkpoint_dir, data_training_args.epoch_range_to_evaluate)
    seq_length, input_dims = series_shape(vectorized_datasets["train"], component, n_mels, pool_mel_bins)
    print(f"{baseline_args.baseline_method} reads {seq_length} frames of {input_dims} features")

    "FHVAE's table is sized by the pre-training split, so its size comes off the checkpoint"
    n_train_utts = None
    if baseline_args.baseline_method == "fhvae":
        n_train_utts = table_rows_from_checkpoint(checkpoint_dir, checkpoints)
        print(f"fhvae was pre-trained with {n_train_utts} utterances in its prior-mean table")

    "Below this point we need to iterate across checkpoints"
    for ckp_dir in checkpoints:
        print(f"Loading model from checkpoint directory: {checkpoint_dir}")
        print(f"Processing checkpoint {ckp_dir}...")

        "initialize random model and load pretrained weights"
        representation_function = build_baseline_model(baseline_args, method_args, input_dims,
                                                       seq_length, config, n_train_utts=n_train_utts)
        if ckp_dir != RANDOM_INIT:
            weights = load_file(os.path.join(checkpoint_dir, ckp_dir, "model.safetensors"))
            representation_function.load_state_dict(weights, strict=True)

        representation_function.eval()
        for param in representation_function.parameters():
            param.requires_grad = False

        ckp = ckp_dir + "_" + representation_name(baseline_args, method_args)

        "data collator"
        mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
        mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

        data_collator = DataCollatorForDecVAELatentPostAnalysis_NoFeatureExtraction(
            model=representation_function,
            feature_extractor=feature_extractor,
            model_args=model_args,
            data_training_args=collator_data_args,
            config=config,
            input_type=input_type,
            dataset_name=data_training_args.dataset_name,
            pad_to_multiple_of=data_training_args.pad_to_multiple_of,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length
        )

        "The baselines are evaluated on ordered frames"
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
        representation_function = accelerator.unwrap_model(representation_function)

        "Measure total loading time"
        start_time = time.time()
        "Get the representations"
        z, z_seq, labels = gather_split(eval_dataloader, representation_function, data_training_args,
                                        component, n_mels, pool_mel_bins, seq_pooling)
        if data_training_args.dataset_name == "iemocap":
            z_test, z_seq_test, labels_test = None, None, {}
        else:
            z_test, z_seq_test, labels_test = gather_split(test_dataloader, representation_function, data_training_args,
                                                           component, n_mels, pool_mel_bins, seq_pooling)
        print(f"Total loading time: {time.time() - start_time: .4f} seconds")

        "Now use train/val representations to get the evaluation metrics"
        "Linear/non-linear classification"
        tasks = data_training_args.classification_tasks
        if data_training_args.classify:
            "Label to gather, target name to record it under, and the task that switches it on"
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

            def stack_support(y, y_test, target, labels, labels_test):
                "A two-name target asks for a grouped CV split, which reads the group off the second column"
                if not isinstance(target, (list, tuple)):
                    return y, y_test
                y = torch.stack((y, labels[target[1]]), dim=1)
                if y_test is not None:
                    y_test = torch.stack((y_test, labels_test[target[1]]), dim=1)
                return y, y_test

            for label_name, target, task in frame_targets:
                if task not in tasks and "all" not in tasks:
                    continue
                y, y_test = stack_support(labels[label_name], labels_test.get(label_name),
                                          target, labels, labels_test)
                prediction_eval(data_training_args, config,
                    X=z, X_test=z_test,
                    y=y, y_test=y_test,
                    checkpoint=ckp, latent_type="z", target=target
                )

            "Sequence-level targets, read off the pooled embeddings"
            if z_seq is not None:
                if data_training_args.dataset_name == "iemocap":
                    seq_targets = [("speaker_seq", "speaker_seq", "speaker_seq"),
                                   ("emotion_seq", ["cat_emotion_seq", "speaker_seq"], "emotion_seq")]
                else:
                    seq_targets = [("speaker_seq", "speaker_seq", "speaker_seq")]

                for label_name, target, task in seq_targets:
                    if task not in tasks and "all" not in tasks:
                        continue
                    y, y_test = stack_support(labels[label_name], labels_test.get(label_name),
                                              target, labels, labels_test)
                    prediction_eval(data_training_args, config,
                        X=z_seq, X_test=z_seq_test,
                        y=y, y_test=y_test,
                        checkpoint=ckp, latent_type="z", target=target
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
                latent_type="z", mu_train=z, y_train=y_frame_train,
                mu_test=z_test, y_test=y_frame_test, target=columns
            )

            "Sequence-level disentanglement on IEMOCAP, emotion against speaker, off the pooled embeddings"
            if data_training_args.dataset_name == "iemocap":
                if z_seq is None:
                    print("Skipping the sequence-level disentanglement - the sequence pooling is not set, so "
                          "there are no pooled embeddings to evaluate")
                else:
                    seq_columns = ["speaker_seq", "cat_emotion_seq"]
                    seq_names = ["speaker_seq", "emotion_seq"]

                    y_seq_train = torch.cat([labels[n].reshape(-1, 1) for n in seq_names], dim=1)
                    y_seq_train = pd.DataFrame(y_seq_train.cpu().numpy(), columns=seq_columns)
                    if z_seq_test is not None:
                        y_seq_test = torch.cat([labels_test[n].reshape(-1, 1) for n in seq_names], dim=1)
                        y_seq_test = pd.DataFrame(y_seq_test.cpu().numpy(), columns=seq_columns)
                    else:
                        y_seq_test = None

                    compute_disentanglement_metrics(data_training_args, config, checkpoint=ckp,
                        latent_type="z", mu_train=z_seq, y_train=y_seq_train,
                        mu_test=z_seq_test, y_test=y_seq_test, target=seq_columns
                    )


if __name__ == "__main__":
    main()
