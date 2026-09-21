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


"""Pre-Training a time-series representation baseline (CoST, TF-C, TCL, CPC, FHVAE) on unlabeled audio data.

The core of the script - data loading, training parameters, the training/validation loop, early
stopping, checkpointing and logging - is the one vaes_pretraining.py uses. What differs per method
is the architecture, the front-end it reads and the loss, which live in models/baselines and in the
method's own argument group. Every method here is frame-level: it emits one embedding per frame of
the DecVAE grid, and sequence-level performance is measured by pooling in the evaluation script.
"""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import build_cost, build_tfc, build_tcl, build_cpc, build_fhvae
from models.baselines.tcl import fit_pca_whitening
import joblib
from data_collation import DataCollatorForBaselinePretraining_NoFeatureExtraction
from data_preprocessing import prepare_extract_features_pretraining_dataset
from config_files import DecVAEConfig

import transformers
from transformers import (
    AdamW,
    Wav2Vec2FeatureExtractor,
    get_scheduler,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from functools import partial
import copy
import math
import shutil
from pathlib import Path

from args_configs import (
    ModelArguments,
    DataTrainingArguments,
    DecompositionArguments,
    TrainingObjectiveArguments,
    BaselinePretrainingArguments,
    CoSTArguments,
    TFCArguments,
    TCLArguments,
    CPCArguments,
    FHVAEArguments,
)
from dataset_loading import load_timit, load_sim_vowels, load_iemocap, load_voc_als
from utils.misc import parse_args, debugger_is_active
from utils.cache_utils import build_cache_file_names, build_map_cache_file_names
from utils.training_utils import count_parameters, EarlyStopping, get_grad_norm
from safetensors.torch import save_model
import numpy as np
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from huggingface_hub import HfApi
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
import time

JSON_FILE_NAME_MANUAL = "config_files/baselines/fhvae/sim_vowels/pre-training/config_pretraining_fhvae_sim_vowels.json" #for debugging purposes only

logger = get_logger(__name__)

"Hardcoded intervention: keep only this fraction of every TIMIT split. Set to None for the full dataset"
TIMIT_SUBSET_FRACTION = 0.05


def redirect_subset_outputs(data_training_args, fraction):
    "Point the cache files and output_dir of a subset run away from the full-dataset ones"
    suffix = f"_subset{fraction * 100:g}pct"
    for attr in ("train_cache_file_name", "validation_cache_file_name", "test_cache_file_name", "dev_cache_file_name"):
        path = getattr(data_training_args, attr, None)
        if path is not None:
            stem = path[:-len(".arrow")] if path.endswith(".arrow") else path
            setattr(data_training_args, attr, stem + suffix + ".arrow")
    if data_training_args.output_dir is not None:
        data_training_args.output_dir = data_training_args.output_dir.rstrip("/\\") + suffix


def subset_raw_datasets(raw_datasets, fraction, seed):
    "Keep a seeded random fraction of every split, at least one example each"
    for split in raw_datasets:
        n_keep = max(1, int(raw_datasets[split].num_rows * fraction))
        raw_datasets[split] = raw_datasets[split].shuffle(seed=seed).select(range(n_keep))
    return raw_datasets


"Per-method losses that are logged alongside the total loss"
METHOD_LOSS_KEYS = {
    "cost": ("trend_loss", "seasonal_loss"),
    "tfc": ("time_loss", "freq_loss", "time_frequency_loss"),
    "tcl": ("segment_accuracy", "classifier_only"),
    "cpc": ("prediction_accuracy", "step_1_accuracy"),
    "fhvae": ("log_px_z", "kld_seg", "kld_seq", "log_qy"),
}


def resolve_method_args(baseline_args, cost_args, tfc_args, tcl_args, cpc_args, fhvae_args):
    """
    Select the argument group of the configured method and the front-end it reads.

    Args:
        baseline_args (:class:`~args_configs.baseline_pretraining_args.BaselinePretrainingArguments`)
        cost_args (:class:`~args_configs.cost_args.CoSTArguments`)
        tfc_args (:class:`~args_configs.tfc_args.TFCArguments`)
        tcl_args (:class:`~args_configs.tcl_args.TCLArguments`)
        cpc_args (:class:`~args_configs.cpc_args.CPCArguments`)
        fhvae_args (:class:`~args_configs.fhvae_args.FHVAEArguments`)
    Returns:
        method_args: The argument group of the configured method
        input_type (str): The front-end the method reads - 'mel' or 'waveform'
        n_mels (int): Number of mel bands, for the 'mel' front-end
        mel_norm (str or None): Mel normalization applied in the collator
        pool_mel_bins (bool): Whether the collator averages the time bins inside each frame
    """
    if baseline_args.baseline_method == "cost":
        resolved = (cost_args, cost_args.cost_input_type, cost_args.cost_n_mels,
                    cost_args.cost_mel_norm, cost_args.cost_pool_mel_bins)
        if resolved[1] not in ("mel", "waveform"):
            raise ValueError(f"Unknown cost_input_type {resolved[1]}, expected 'mel' or 'waveform'.")
        return resolved

    if baseline_args.baseline_method == "tfc":
        resolved = (tfc_args, tfc_args.tfc_input_type, tfc_args.tfc_n_mels,
                    tfc_args.tfc_mel_norm, tfc_args.tfc_pool_mel_bins)
        if resolved[1] not in ("mel", "waveform"):
            raise ValueError(f"Unknown tfc_input_type {resolved[1]}, expected 'mel' or 'waveform'.")
        return resolved

    if baseline_args.baseline_method == "tcl":
        resolved = (tcl_args, tcl_args.tcl_input_type, tcl_args.tcl_n_mels,
                    tcl_args.tcl_mel_norm, tcl_args.tcl_pool_mel_bins)
        if resolved[1] not in ("mel", "waveform"):
            raise ValueError(f"Unknown tcl_input_type {resolved[1]}, expected 'mel' or 'waveform'.")
        return resolved

    if baseline_args.baseline_method == "cpc":
        resolved = (cpc_args, cpc_args.cpc_input_type, cpc_args.cpc_n_mels,
                    cpc_args.cpc_mel_norm, cpc_args.cpc_pool_mel_bins)
        if resolved[1] not in ("mel", "waveform"):
            raise ValueError(f"Unknown cpc_input_type {resolved[1]}, expected 'mel' or 'waveform'.")
        return resolved

    if baseline_args.baseline_method == "fhvae":
        resolved = (fhvae_args, fhvae_args.fhvae_input_type, fhvae_args.fhvae_n_mels,
                    fhvae_args.fhvae_mel_norm, fhvae_args.fhvae_pool_mel_bins)
        if resolved[1] not in ("mel", "waveform"):
            raise ValueError(f"Unknown fhvae_input_type {resolved[1]}, expected 'mel' or 'waveform'.")
        return resolved

    raise ValueError(
        f"Unknown baseline_method {baseline_args.baseline_method}, expected one of 'cost', 'tfc', "
        "'tcl', 'cpc', 'fhvae'."
    )


def build_baseline_model(baseline_args, method_args, input_dims, seq_length, config, n_train_utts=None):
    """
    Build the model of the configured method.

    Args:
        baseline_args (:class:`~args_configs.baseline_pretraining_args.BaselinePretrainingArguments`)
        method_args: The argument group of the configured method
        input_dims (int): Width of one frame of the input series
        seq_length (int): Number of frames per utterance
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): Carries the frame grid
        n_train_utts (int, optional): Number of training utterances, which FHVAE needs for its
            table of per-utterance prior means. Unused by the other methods
    Returns:
        torch.nn.Module: The model, whose forward returns a dict holding at least 'loss'
    """
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
                "FHVAE keeps one trainable prior mean per training utterance, so it has to be built "
                "with n_train_utts."
            )
        return build_fhvae(method_args, input_dims, seq_length, config, n_train_utts)

    raise ValueError(
        f"Unknown baseline_method {baseline_args.baseline_method}, expected one of 'cost', 'tfc', "
        "'tcl', 'cpc', 'fhvae'."
    )


def build_optimizer(baseline_args, method_args, model, data_training_args):
    """
    Build the optimizer of the configured method's reference implementation. The learning rate and
    the weight decay come from the data arguments, where the config carries the reference values.

    Args:
        baseline_args (:class:`~args_configs.baseline_pretraining_args.BaselinePretrainingArguments`)
        method_args: The argument group of the configured method
        model (:obj:`torch.nn.Module`): The model being pre-trained
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`)
    Returns:
        torch.optim.Optimizer
    """
    "CoST freezes its momentum encoder, and the reference hands the optimizer only what it trains"
    trainable = [param for param in model.parameters() if param.requires_grad]

    if baseline_args.baseline_method == "cost":
        if method_args.cost_optimizer == "sgd":
            return torch.optim.SGD(
                trainable,
                lr=data_training_args.learning_rate,
                momentum=method_args.cost_sgd_momentum,
                weight_decay=data_training_args.weight_decay,
            )
        if method_args.cost_optimizer == "adamw":
            return AdamW(
                trainable,
                lr=data_training_args.learning_rate,
                betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                eps=data_training_args.adam_epsilon,
                weight_decay=data_training_args.weight_decay,
            )
        raise ValueError(f"Unknown cost_optimizer {method_args.cost_optimizer}, expected 'sgd' or 'adamw'.")

    if baseline_args.baseline_method == "tfc":
        if method_args.tfc_optimizer == "adam":
            return torch.optim.Adam(
                trainable,
                lr=data_training_args.learning_rate,
                betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                eps=data_training_args.adam_epsilon,
                weight_decay=data_training_args.weight_decay,
            )
        if method_args.tfc_optimizer == "adamw":
            return AdamW(
                trainable,
                lr=data_training_args.learning_rate,
                betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                eps=data_training_args.adam_epsilon,
                weight_decay=data_training_args.weight_decay,
            )
        raise ValueError(f"Unknown tfc_optimizer {method_args.tfc_optimizer}, expected 'adam' or 'adamw'.")

    if baseline_args.baseline_method == "tcl":
        "The reference decays the weights but not the biases"
        weights = [param for name, param in model.named_parameters() if param.requires_grad and name.endswith("weight")]
        biases = [param for name, param in model.named_parameters() if param.requires_grad and not name.endswith("weight")]
        groups = [{"params": weights, "weight_decay": data_training_args.weight_decay},
                  {"params": biases, "weight_decay": 0.0}]
        if method_args.tcl_optimizer == "sgd":
            return torch.optim.SGD(groups, lr=data_training_args.learning_rate,
                                   momentum=method_args.tcl_sgd_momentum)
        if method_args.tcl_optimizer == "adamw":
            return AdamW(groups, lr=data_training_args.learning_rate,
                         betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                         eps=data_training_args.adam_epsilon)
        raise ValueError(f"Unknown tcl_optimizer {method_args.tcl_optimizer}, expected 'sgd' or 'adamw'.")

    if baseline_args.baseline_method == "cpc":
        "Both references build a plain Adam over the encoder, the context network and the predictors"
        if method_args.cpc_optimizer == "adam":
            return torch.optim.Adam(
                trainable,
                lr=data_training_args.learning_rate,
                betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                eps=data_training_args.adam_epsilon,
                weight_decay=data_training_args.weight_decay,
            )
        if method_args.cpc_optimizer == "adamw":
            return AdamW(
                trainable,
                lr=data_training_args.learning_rate,
                betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                eps=data_training_args.adam_epsilon,
                weight_decay=data_training_args.weight_decay,
            )
        raise ValueError(f"Unknown cpc_optimizer {method_args.cpc_optimizer}, expected 'adam' or 'adamw'.")

    if baseline_args.baseline_method == "fhvae":
        "The reference builds a plain Adam over every parameter, the prior-mean table included."
        "It must be dense: TF1 decays the moments of every row each step, which a sparse or lazy"
        "Adam would not, and the table is what the discriminative term shapes"
        if method_args.fhvae_optimizer == "adam":
            return torch.optim.Adam(
                trainable,
                lr=data_training_args.learning_rate,
                betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                eps=data_training_args.adam_epsilon,
                weight_decay=data_training_args.weight_decay,
            )
        if method_args.fhvae_optimizer == "adamw":
            return AdamW(
                trainable,
                lr=data_training_args.learning_rate,
                betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
                eps=data_training_args.adam_epsilon,
                weight_decay=data_training_args.weight_decay,
            )
        raise ValueError(f"Unknown fhvae_optimizer {method_args.fhvae_optimizer}, expected 'adam' or 'adamw'.")

    raise ValueError(
        f"Unknown baseline_method {baseline_args.baseline_method}, expected one of 'cost', 'tfc', "
        "'tcl', 'cpc', 'fhvae'."
    )


def fit_tcl_whitening(model, dataset, data_collator, data_training_args, method_args, component, projection_path):
    """
    Fit the PCA whitening TCL applies before its network, on the training split alone, or load the
    one an earlier run fitted. The mean and the covariance are accumulated over the frames rather
    than held, so the pass costs nothing in memory.

    Args:
        model (:class:`~models.baselines.tcl.TCLForPreTraining`): Receives the fitted transform
        dataset: The training split
        data_collator: The collator, so the frames are the ones training will read
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`)
        method_args (:class:`~args_configs.tcl_args.TCLArguments`)
        component (int): Which decomposition component feeds the encoder
        projection_path (str): Where the fitted transform is cached
    """
    if os.path.exists(projection_path):
        print(f"Loading the fitted TCL whitening from {projection_path}")
        model.set_whitening(joblib.load(projection_path))
        return

    loader = DataLoader(dataset, shuffle=False, collate_fn=data_collator,
                        batch_size=data_training_args.per_device_eval_batch_size)
    generator = np.random.default_rng(seed=method_args.tcl_pca_seed)
    fraction = method_args.tcl_pca_fit_fraction
    width = model.input_dims
    count, total, scatter = 0, np.zeros(width, dtype=np.float64), np.zeros((width, width), dtype=np.float64)

    with torch.no_grad():
        for batch in loader:
            frames = batch["input_values"]
            keep = batch["sub_attention_mask"].reshape(-1).to(torch.bool)
            frames = frames.reshape(-1, frames.shape[-1])[keep].to(torch.float64).numpy()
            if fraction is not None and fraction < 1.0:
                take = max(1, int(frames.shape[0] * fraction))
                frames = frames[generator.choice(frames.shape[0], size=take, replace=False)]
            count += frames.shape[0]
            total += frames.sum(axis=0)
            scatter += frames.T @ frames

    if count <= width:
        raise ValueError(
            f"The whitening needs more training frames than features, got {count} frames of {width}."
        )
    mean = total / count
    covariance = (scatter - count * np.outer(mean, mean)) / (count - 1)
    "The model resolved the count, since an unset one falls back to the feature width"
    params = fit_pca_whitening(mean, covariance, num_comp=model.pca_components)
    print(f"Fitted the TCL whitening on {count} training frames, keeping {params['W'].shape[0]} of "
          f"{width} components, contribution ratio {params['contribution_ratio']:.4f}")

    os.makedirs(os.path.dirname(projection_path), exist_ok=True)
    joblib.dump(params, projection_path)
    model.set_whitening(params)


def build_scheduler(baseline_args, method_args, optimizer, data_training_args):
    """
    Build the learning rate schedule. The project's schedules cover the methods whose reference
    recipe uses one of them; TCL's staircase exponential decay is not among them, so it is asked
    for by name and built here.

    Args:
        baseline_args (:class:`~args_configs.baseline_pretraining_args.BaselinePretrainingArguments`)
        method_args: The argument group of the configured method
        optimizer (:obj:`torch.optim.Optimizer`)
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`)
    Returns:
        A scheduler whose step() is called once per optimization step
    """
    if data_training_args.lr_scheduler_type == "tcl_staircase":
        if baseline_args.baseline_method != "tcl":
            raise ValueError("The 'tcl_staircase' schedule is TCL's own, and this run is "
                             f"{baseline_args.baseline_method}.")
        "tf.train.exponential_decay with staircase set is a step decay"
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=method_args.tcl_decay_steps,
            gamma=method_args.tcl_decay_factor,
        )

    return get_scheduler(
        name=data_training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=data_training_args.num_warmup_steps,
        num_training_steps=data_training_args.max_train_steps,
        scheduler_specific_kwargs={'num_cycles': data_training_args.lr_scheduler_num_cycles} if data_training_args.lr_scheduler_type == "cosine_with_restarts" else None,
    )


def series_shape(vectorized_datasets, component):
    """
    Read the frame count and the frame width off the cached features, so the model is built for the
    series it will actually receive rather than for a width derived by hand.

    Args:
        vectorized_datasets (:class:`~datasets.DatasetDict`): The preprocessed dataset
        component (int): Which decomposition component feeds the encoder
    Returns:
        seq_length (int), input_dims (int)
    """
    example = np.asarray(vectorized_datasets["train"][0]["input_values"])
    if example.ndim != 4 or example.shape[0] != 1:
        raise ValueError(
            f"Expected (1, components, frames, features) cached inputs, as DecVAE's preprocessing "
            f"writes them, got shape {example.shape}. A cache written by an older baseline "
            "preprocessing has no leading axis - delete it so it is rebuilt. The cache must also "
            "hold the framed decomposition, which it only does when it ran with frame_decomp=true."
        )
    example = example[0]
    if component >= example.shape[0]:
        raise ValueError(
            f"baseline_component {component} was asked for, but the cache holds only "
            f"{example.shape[0]} components."
        )
    return int(example.shape[1]), int(example.shape[2])


def main():
    "Parse the arguments"
    parser = HfArgumentParser((ModelArguments, TrainingObjectiveArguments, DecompositionArguments,
                               DataTrainingArguments, BaselinePretrainingArguments, CoSTArguments,
                               TFCArguments, TCLArguments, CPCArguments, FHVAEArguments))

    if debugger_is_active():
        model_args, training_obj_args, decomp_args, data_training_args, baseline_args, cost_args, tfc_args, tcl_args, cpc_args, fhvae_args = \
            parser.parse_json_file(json_file=JSON_FILE_NAME_MANUAL)
    else:
        args = parse_args()
        model_args, training_obj_args, decomp_args, data_training_args, baseline_args, cost_args, tfc_args, tcl_args, cpc_args, fhvae_args = \
            parser.parse_json_file(json_file=args.config_file)
    delattr(model_args, "comment_model_args")
    delattr(training_obj_args, "comment_tr_obj_args")
    delattr(decomp_args, "comment_decomp_args")
    delattr(baseline_args, "comment_baseline_args")
    delattr(cost_args, "comment_cost_args")
    delattr(tfc_args, "comment_tfc_args")
    delattr(tcl_args, "comment_tcl_args")
    delattr(cpc_args, "comment_cpc_args")
    delattr(fhvae_args, "comment_fhvae_args")

    method_args, input_type, n_mels, mel_norm, pool_mel_bins = resolve_method_args(baseline_args, cost_args, tfc_args, tcl_args, cpc_args, fhvae_args)

    use_timit_subset = "timit" in data_training_args.dataset_name and TIMIT_SUBSET_FRACTION is not None
    if use_timit_subset:
        "A subset run must neither load nor overwrite the full-dataset cache and checkpoints"
        redirect_subset_outputs(data_training_args, TIMIT_SUBSET_FRACTION)
        print(f"TIMIT subset of {TIMIT_SUBSET_FRACTION:.2%}: cache and output_dir redirected to {data_training_args.output_dir}")

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

    "Handle the repository creation"
    if accelerator.is_main_process:
        if data_training_args.push_to_hub and not data_training_args.preprocessing_only:
            # Retrieve of infer repo_name
            repo_name = data_training_args.hub_model_id
            if repo_name is None:
                repo_name = Path(data_training_args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=data_training_args.hub_token).repo_id

            with open(os.path.join(data_training_args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif data_training_args.output_dir is not None:
            os.makedirs(data_training_args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    "load cached preprocessed files. The front-end of the method decides which cache is read"
    cache_file_names = build_cache_file_names(data_training_args, input_type)

    "preprocess the datasets including loading the audio, resampling and normalization"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)

    "set max & min audio length in number of samples"
    max_length = int(data_training_args.max_duration_in_seconds * feature_extractor.sampling_rate)
    min_length = int(data_training_args.min_duration_in_seconds * feature_extractor.sampling_rate)

    "Load model with hyperparameters"
    model_args.max_duration_in_seconds = data_training_args.max_duration_in_seconds
    config = DecVAEConfig(**{**model_args.__dict__, **training_obj_args.__dict__, **decomp_args.__dict__})

    try:
        if data_training_args.train_cache_file_name is None or data_training_args.validation_cache_file_name is None or cache_file_names["train"] is None:
            raise FileNotFoundError("one or more cache_file_names were not defined. Proceeding with computing preprocessing.")

        with accelerator.main_process_first():
            vectorized_datasets = DatasetDict()
            vectorized_datasets["train"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["train"]])
            vectorized_datasets["validation"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["validation"]])
            try:
                vectorized_datasets["test"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["test"]])
            except KeyError:
                pass
            try:
                vectorized_datasets["dev"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["dev"]])
            except KeyError:
                pass
            if min_length > 0.0:
                vectorized_datasets = vectorized_datasets.filter(
                    lambda x: x > min_length,
                    num_proc=data_training_args.preprocessing_num_workers,
                    input_columns=["input_length"],
                )
            vectorized_datasets = vectorized_datasets.remove_columns("input_length")

    except FileNotFoundError:
        "Download and create train, validation dataset"

        if "timit" in data_training_args.dataset_name:
            raw_datasets = load_timit(data_training_args)
            if use_timit_subset:
                raw_datasets = subset_raw_datasets(raw_datasets, TIMIT_SUBSET_FRACTION,
                                                   seed=data_training_args.seed if data_training_args.seed is not None else 0)
                print("TIMIT subset sizes:", {split: raw_datasets[split].num_rows for split in raw_datasets})

        elif "sim_vowels" in data_training_args.dataset_name:
            raw_datasets = load_sim_vowels(data_training_args)

        elif "VOC_ALS" in data_training_args.dataset_name:
            raw_datasets = load_voc_als(data_training_args)

        elif "iemocap" in data_training_args.dataset_name:
            raw_datasets = load_iemocap(data_training_args)

        if "timit" in data_training_args.dataset_name:
            # load_timit decodes the audio at TIMIT's native 16 kHz, so there is nothing to resample
            if feature_extractor.sampling_rate != 16000:
                raise ValueError(f"TIMIT audio is 16 kHz, but the feature extractor expects {feature_extractor.sampling_rate} Hz")

        "only normalized-inputs-training is supported"
        if not feature_extractor.do_normalize:
            raise ValueError(
                "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
            )

        "load via mapped files via path"
        cache_file_names = build_map_cache_file_names(data_training_args, input_type)

        "make the directory that will store the decomposition"
        os.makedirs(os.path.dirname(cache_file_names['train']), exist_ok=True)

        "DecVAE's own preparation, so the caches are identical to DecVAE's and can be shared."
        "It reads the front-end off the data arguments, so hand it the method's"
        prep_data_args = copy.copy(data_training_args)
        prep_data_args.input_type = input_type
        prep_data_args.n_mels = n_mels

        "load audio files into numpy arrays"
        with accelerator.main_process_first():

            vectorized_datasets = raw_datasets.map(
                partial(
                    prepare_extract_features_pretraining_dataset,
                    feature_extractor=feature_extractor,
                    data_training_args=prep_data_args,
                    decomp_args=decomp_args,
                    config=config,
                    max_length=max_length
                ),
                num_proc=data_training_args.preprocessing_num_workers,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=True,
                cache_file_names=cache_file_names,
            )

            if min_length > 0.0:
                vectorized_datasets = vectorized_datasets.filter(
                    lambda x: x > min_length,
                    num_proc=data_training_args.preprocessing_num_workers,
                    input_columns=["input_length"],
                )

            vectorized_datasets = vectorized_datasets.remove_columns("input_length")

    if data_training_args.preprocessing_only:
        return

    if not data_training_args.pretrain:
        "For cases where we only want to decompose the data and not train the model"
        print("Pretrain = false, so we will not train the model, exiting after decomposition")
        return

    "initialize random model. The series it reads is measured off the cache, so the model is built"
    "for the frame count and the frame width the collator will actually hand it"
    seq_length, input_dims = series_shape(vectorized_datasets, baseline_args.baseline_component)
    if input_type.startswith("mel") and pool_mel_bins:
        input_dims = n_mels

    "FHVAE keeps one trainable prior mean per training utterance, so a batch has to carry which"
    "utterance it came from. The indices are written after any subsetting, so they run 0..n-1 over"
    "the split the table is sized for. A held-out split carries none, and the mean is estimated"
    n_train_utts = vectorized_datasets["train"].num_rows
    if baseline_args.baseline_method == "fhvae":
        vectorized_datasets["train"] = vectorized_datasets["train"].add_column(
            "utt_index", list(range(n_train_utts))
        )
        print(f"fhvae indexes {n_train_utts} training utterances for its prior-mean table")

    model = build_baseline_model(baseline_args, method_args, input_dims, seq_length, config,
                                 n_train_utts=n_train_utts)
    print(f"{baseline_args.baseline_method} reads {seq_length} frames of {input_dims} features, "
          f"{count_parameters(model)} trainable parameters")

    "data collator, optimizer and scheduler"
    data_collator = DataCollatorForBaselinePretraining_NoFeatureExtraction(
        model=model,
        feature_extractor=feature_extractor,
        input_type=input_type,
        n_mels=n_mels,
        mel_norm=mel_norm,
        pool_mel_bins=pool_mel_bins,
        component=baseline_args.baseline_component,
        pad_to_multiple_of=data_training_args.pad_to_multiple_of,
    )

    "CoST sizes its queue of negatives against the batch, and TF-C its contrastive mask, so an"
    "incomplete last batch cannot be used by those methods"
    drop_last = getattr(model, "requires_fixed_batch_size", False)

    train_dataloader = DataLoader(
        vectorized_datasets['train'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=data_training_args.per_device_train_batch_size,
        drop_last=drop_last,
    )
    eval_dataloader = DataLoader(
        vectorized_datasets["validation"],
        collate_fn=data_collator,
        batch_size=data_training_args.per_device_eval_batch_size,
        drop_last=drop_last,
    )

    if len(train_dataloader) == 0 or len(eval_dataloader) == 0:
        raise ValueError(
            f"A split is smaller than the batch size and {baseline_args.baseline_method} needs whole "
            "batches, so the dataloader came out empty. Lower per_device_train_batch_size or "
            "per_device_eval_batch_size."
        )

    "TCL whitens its inputs, fitted on the training split before any weight is updated"
    if baseline_args.baseline_method == "tcl":
        fit_tcl_whitening(model, vectorized_datasets["train"], data_collator, data_training_args,
                          method_args, baseline_args.baseline_component,
                          os.path.join(data_training_args.output_dir, "tcl_pca_whitening.joblib"))

    "Optimizer of the method's reference implementation. The learning rate, weight decay, schedule"
    "shape and batch size are the reference ones too, carried in the data arguments of the config"
    optimizer = build_optimizer(baseline_args, method_args, model, data_training_args)

    "Prepare everything with HF accelerator"
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    "Scheduler and math around the number of training steps."
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / data_training_args.gradient_accumulation_steps)

    if data_training_args.max_train_steps is None:
        data_training_args.max_train_steps = data_training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = build_scheduler(baseline_args, method_args, optimizer, data_training_args)

    "calculate number of training epochs"
    data_training_args.num_train_epochs = math.ceil(data_training_args.max_train_steps / num_update_steps_per_epoch)

    "Train"
    total_batch_size = data_training_args.per_device_train_batch_size * accelerator.num_processes * data_training_args.gradient_accumulation_steps
    print(f"Accelerate uses {accelerator.num_processes} processes")
    print(f"Dataloader has {len(train_dataloader)} steps in an epoch")

    logger.info("***** Running training *****")
    logger.info(f"  Method = {baseline_args.baseline_method}")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {data_training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {data_training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {data_training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {data_training_args.max_train_steps}")

    "Only show the progress bar once on each machine."
    progress_bar = tqdm(range(data_training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    min_val_loss = 1000000
    loss_keys = METHOD_LOSS_KEYS[baseline_args.baseline_method]

    early_stopping = EarlyStopping()
    early_stopping.min_delta_percent = data_training_args.early_stop_min_delta_percent
    early_stopping.patience = data_training_args.early_stop_patience_epochs
    early_stopping.min_steps = data_training_args.early_stop_warmup_steps

    "save config with all parameters for the model + the run"
    if debugger_is_active():
        destination_config = os.path.join(data_training_args.output_dir, os.path.basename(JSON_FILE_NAME_MANUAL))
        shutil.copy(JSON_FILE_NAME_MANUAL, destination_config)
    else:
        destination_config = os.path.join(data_training_args.output_dir, os.path.basename(args.config_file))
        shutil.copy(args.config_file, destination_config)

    for epoch in range(starting_epoch, data_training_args.num_train_epochs):
        model.train()
        saved_this_epoch = False
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()

            "The methods read the frame series and its frame-level mask, and nothing else"
            batch.pop("attention_mask", None)

            if (batch["input_values"] != batch["input_values"]).any():
                print("NaNs in input_values")

            "Forward pass"
            outputs = model(**batch)

            "Calculate the loss after gradient accumulation. The losses are already averaged over"
            "the batch, so unlike the VAE and DecVAE paths there is nothing to renormalize by"
            loss = outputs["loss"] / data_training_args.gradient_accumulation_steps

            accelerator.backward(loss)

            "clip gradients"
            if training_obj_args.clip_grad_value is not None:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=training_obj_args.clip_grad_value)

            "update step"
            if (step + 1) % data_training_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                "compute grad norm for monitoring"
                scale = (
                    accelerator.scaler._scale.item()
                    if hasattr(accelerator, "scaler") and accelerator.scaler is not None
                    else 1
                )
                if accelerator.state.num_processes > 1:
                    grad_norm = get_grad_norm(model.module.named_parameters(), scale)
                else:
                    grad_norm = get_grad_norm(model.named_parameters(), scale)

                if grad_norm != grad_norm:
                    print("Gradient norm is NaN")

                "update parameters"
                optimizer.step()
                optimizer.zero_grad()

                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()
                elif accelerator.is_local_main_process:
                    progress_bar.write(
                        f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                    )

                "Methods that stage their training count optimization steps, not forward passes"
                unwrapped_model = accelerator.unwrap_model(model)
                if hasattr(unwrapped_model, "on_optimizer_step"):
                    unwrapped_model.on_optimizer_step()

                progress_bar.update(1)
                completed_steps += 1

            "Log all results"
            if (completed_steps+1) % (data_training_args.gradient_accumulation_steps * data_training_args.logging_steps) == 0:
                train_logs = {k: outputs[k].detach() for k in loss_keys}
                train_logs["loss"] = (loss * data_training_args.gradient_accumulation_steps).detach()

                "Aggregate all metrics over devices"
                if accelerator.state.num_processes > 1:
                    train_logs = {k: accelerator.gather_for_metrics(v).sum() for k, v in train_logs.items()}

                train_logs["lr"] = torch.tensor(optimizer.param_groups[0]["lr"])
                train_logs["grad_norm"] = torch.tensor(grad_norm)

                log_str = ""
                for k, v in train_logs.items():
                    try:
                        log_str += "| {}: {:.3e}".format(k, v.item())
                    except AttributeError:
                        log_str += "| {}: {:.3e}".format(k, v)
                if accelerator.is_local_main_process:
                    progress_bar.write(log_str)
                    if is_wandb_available() and data_training_args.with_wandb:
                        wandb.log(train_logs)

            "save model every `args.saving_steps` steps"
            if (completed_steps+1) % (data_training_args.gradient_accumulation_steps * data_training_args.saving_steps) == 0:
                saved_this_epoch = True
                "Save model"
                if (data_training_args.push_to_hub and epoch < data_training_args.num_train_epochs - 1) or data_training_args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    if epoch < 10:
                        model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_0" + str(epoch))
                    else:
                        model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_" + str(epoch))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    save_model(unwrapped_model, os.path.join(model_dir, "model.safetensors"))

                "Push to HF hub"
                if (data_training_args.push_to_hub and epoch < data_training_args.num_train_epochs - 1) and accelerator.is_main_process:
                    try:
                        api.upload_folder(
                            commit_message=f"Training in progress epoch {epoch}",
                            folder_path=data_training_args.output_dir,
                            repo_id=repo_id,
                            repo_type="model",
                            token=data_training_args.hub_token,
                        )
                    except: #ConnectionError:
                        logger.warning("Could not push to the hub. Connection error.")

            "if completed steps > `args.max_train_steps` stop"
            if completed_steps >= data_training_args.max_train_steps:
                break

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Step time: {elapsed_time: .4f} seconds")

        if epoch == 0:
            saved_this_epoch = True
            "Save model"
            if (data_training_args.push_to_hub and epoch < data_training_args.num_train_epochs - 1) or data_training_args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                if epoch < 10:
                    model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_0" + str(epoch))
                else:
                    model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_" + str(epoch))
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                save_model(unwrapped_model, os.path.join(model_dir, "model.safetensors"))

        "Validate"
        model.eval()

        val_logs = {"val_loss": 0}
        val_logs.update({"val_" + k: 0 for k in loss_keys})

        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                batch.pop("attention_mask", None)

                outputs = model(**batch)

                val_logs["val_loss"] += outputs["loss"]
                for k in loss_keys:
                    val_logs["val_" + k] += outputs[k]

        for k in val_logs.keys():
            val_logs[k] = val_logs[k] / len(eval_dataloader)

        "sum over devices in multi-processing"
        if accelerator.num_processes > 1:
            val_logs = {k: accelerator.gather_for_metrics(v).sum() for k, v in val_logs.items()}

        log_str = ""
        for k, v in val_logs.items():
            try:
                log_str += "| {}: {:.3e}".format(k, v.item())
            except AttributeError:
                log_str += "| {}: {:.3e}".format(k, v)

        if accelerator.is_local_main_process:
            progress_bar.write(log_str)
            if is_wandb_available() and data_training_args.with_wandb:
                wandb.log(val_logs)

        "Save model if validation loss is lower than min_val_loss"
        if data_training_args.output_dir is not None and min_val_loss >= val_logs["val_loss"]:
            accelerator.wait_for_everyone()
            if data_training_args.save_model and not saved_this_epoch:
                saved_this_epoch = True
                unwrapped_model = accelerator.unwrap_model(model)
                if epoch < 10:
                    model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_0" + str(epoch))
                else:
                    model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_" + str(epoch))
                model_dir += "_min_val_loss"
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                save_model(unwrapped_model, os.path.join(model_dir, "model.safetensors"))

            "Push model dir to HF hub"
            if accelerator.is_main_process and data_training_args.push_to_hub:
                try:
                    api.upload_folder(
                        commit_message=f"Training in progress epoch {epoch}",
                        folder_path=data_training_args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=data_training_args.hub_token,
                    )
                except: # ConnectionError:
                    logger.warning("Could not push to the hub. Connection error.")

        "Check val_loss and replace min_val_loss with current loss"
        if val_logs["val_loss"] < min_val_loss:
            min_val_loss = val_logs["val_loss"]

        "Check early stopping"
        early_stopping(val_logs["val_loss"], completed_steps)

        if early_stopping.early_stop:
            print("Early stopping triggered. End of training after {} epochs".format(epoch+1))
            break

    "Save last model"
    if not saved_this_epoch and data_training_args.output_dir is not None and data_training_args.save_model:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if epoch < 10:
            model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_0" + str(epoch))
        else:
            model_dir = os.path.join(data_training_args.output_dir, "training_ckp_epoch_" + str(epoch))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(unwrapped_model, os.path.join(model_dir, "model.safetensors"))

        "Push model dir to HF hub"
        if accelerator.is_main_process and data_training_args.push_to_hub:
            try:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=data_training_args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=data_training_args.hub_token,
                )
            except: # ConnectionError:
                logger.warning("Could not push to the hub. Connection error.")


if __name__ == "__main__":
    main()
