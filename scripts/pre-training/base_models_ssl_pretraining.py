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

"""Pre-Training a DecVAE model on unlabeled audio data with 🤗 Transformers utilities"""

import os
import sys
# Adds project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import DecVAEForPreTraining
from data_collation import DataCollatorForDecVAEPretraining#, DataCollatorForDecVAEPretraining_NoFeatureExtraction
from config_files import DecVAEConfig
from data_preprocessing import prepare_pretraining_dataset, prepare_extract_features_pretraining_dataset
from args_configs import ModelArguments, DataTrainingArguments, DecompositionArguments, TrainingObjectiveArguments
from dataset_loading import load_timit, load_sim_vowels, load_iemocap, load_voc_als
from utils import (
    parse_args, 
    debugger_is_active,
    multiply_grads, 
    count_parameters, 
    EarlyStopping, 
    get_grad_norm
)
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
import math
import shutil
from pathlib import Path
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


#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

JSON_FILE_NAME_MANUAL = "config_files/DecVAEs/sim_vowels/pre-training/config_pretraining_sim_vowels_NoC3.json" #for debugging purposes only

logger = get_logger(__name__)


def main():
    "Parse the arguments"       
    parser = HfArgumentParser((ModelArguments, TrainingObjectiveArguments, DecompositionArguments,DataTrainingArguments))
    if debugger_is_active():
        model_args, training_obj_args, decomp_args, data_training_args = parser.parse_json_file(json_file=JSON_FILE_NAME_MANUAL)
    else:
        args = parse_args()
        model_args, training_obj_args, decomp_args, data_training_args = parser.parse_json_file(json_file=args.config_file)
    delattr(model_args,"comment_model_args")
    delattr(training_obj_args,"comment_tr_obj_args")
    delattr(decomp_args,"comment_decomp_args")
    
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

    "load cached preprocessed files"
    if data_training_args.preprocessing_num_workers is not None and data_training_args.preprocessing_num_workers > 1:
        cache_file_names = {"train": [data_training_args.train_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)],
                "validation": [data_training_args.validation_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
        }
        if data_training_args.test_cache_file_name is not None:
            cache_file_names["test"] = [data_training_args.test_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
        if data_training_args.dev_cache_file_name is not None:
            cache_file_names["dev"] = [data_training_args.dev_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
    elif data_training_args.preprocessing_num_workers == 1 or data_training_args.preprocessing_num_workers is None:
        cache_file_names = {"train": [data_training_args.train_cache_file_name],
                "validation": [data_training_args.validation_cache_file_name]
        }
        if data_training_args.test_cache_file_name is not None:
            cache_file_names["test"] = [data_training_args.test_cache_file_name]
        if data_training_args.dev_cache_file_name is not None:
            cache_file_names["dev"] = [data_training_args.dev_cache_file_name]
    else:
        cache_file_names = {"train": None,
                            "validation":None}
    
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

        elif "sim_vowels" in data_training_args.dataset_name:
            raw_datasets = load_sim_vowels(data_training_args)
        
        elif "VOC_ALS" in data_training_args.dataset_name:
            raw_datasets = load_voc_als(data_training_args)
        
        elif "iemocap" in data_training_args.dataset_name:
            raw_datasets = load_iemocap(data_training_args)

        "Call .map to pre-process columns first"
        if "timit" in data_training_args.dataset_name:
            # make sure that dataset decodes audio with correct sampling rate
            raw_datasets = raw_datasets.cast_column(
                data_training_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
            )

        "only normalized-inputs-training is supported"
        if not feature_extractor.do_normalize:
            raise ValueError(
                "Training is only supported for normalized inputs. Make sure ``feature_extractor.do_normalize == True``"
            )

        "set max & min audio length in number of samples"
        max_length = int(data_training_args.max_duration_in_seconds * feature_extractor.sampling_rate)
        min_length = int(data_training_args.min_duration_in_seconds * feature_extractor.sampling_rate)

        "load via mapped files via path"
        cache_file_names = None 
        if data_training_args.dataset_name == "VOC_ALS" and not data_training_args.train_val_test_split:
            cache_file_names = {"train": data_training_args.train_cache_file_name,
                                "validation": data_training_args.validation_cache_file_name,
                                "dev": data_training_args.dev_cache_file_name,
                                "test": data_training_args.test_cache_file_name
                            }
        else:
            if data_training_args.train_cache_file_name is not None:
                cache_file_names = {"train": data_training_args.train_cache_file_name, 
                                    "validation": data_training_args.validation_cache_file_name}
            if data_training_args.test_cache_file_name is not None:
                cache_file_names = {**cache_file_names,
                                **{"test": data_training_args.test_cache_file_name}
                                }
            if data_training_args.dev_cache_file_name is not None:
                cache_file_names = {**cache_file_names,
                                **{"dev": data_training_args.dev_cache_file_name}
                                }

        "make the directory that will store the decomposition"
        os.makedirs(os.path.dirname(cache_file_names['train']), exist_ok=True)

        "load audio files into numpy arrays"
        with accelerator.main_process_first():
          
            vectorized_datasets = raw_datasets.map(
                partial(
                    prepare_extract_features_pretraining_dataset,
                    feature_extractor=feature_extractor,
                    data_training_args=data_training_args,
                    decomp_args=decomp_args,
                    config=config,
                    max_length=max_length
                ),#prepare_pretraining_dataset
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
        "For cases where we only want to decompose the data and not train the model - e.g. for transfer learning"
        print("Pretrain = false, so we will not train the model, exiting after decomposition")
        return
    
    "initialize random model"
    config.seq_length = max_length
    if hasattr(config,'dataset_name'):
        delattr(config,'dataset_name')
    model = DecVAEForPreTraining(config)

    "Activate gradient checkpointing if needed"
    if data_training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    "data collator, optimizer and scheduler"

    mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
    mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

    data_collator = DataCollatorForDecVAEPretraining_NoFeatureExtraction(
        model=model,
        feature_extractor=feature_extractor,
        model_args=model_args,
        data_training_args=data_training_args,
        config=config,
        input_type = data_training_args.input_type,
        pad_to_multiple_of=data_training_args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
        dataset_name = data_training_args.dataset_name,
    ) #DataCollatorForDecVAEPretraining

    train_dataloader = DataLoader(
        vectorized_datasets['train'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=data_training_args.per_device_train_batch_size,
    )
    
    eval_dataloader = DataLoader(
        vectorized_datasets["validation"], collate_fn=data_collator, batch_size=data_training_args.per_device_eval_batch_size
    )

    "Scheduler and math around the number of training steps."
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / data_training_args.gradient_accumulation_steps)

    if data_training_args.max_train_steps is None:
        data_training_args.max_train_steps = data_training_args.num_train_epochs * num_update_steps_per_epoch

    "Set different learning rates in case of dual branch"
    if config.dual_branched_latent:
        params_z_var = [param for name, param in model.named_parameters() if '_z' in name]
        params_s_var = [param for name, param in model.named_parameters() if '_s' in name or 'pool' in name]
        param_groups = [
            {'params': params_z_var, 'lr': data_training_args.learning_rate[0]},
            {'params': params_s_var, 'lr': data_training_args.learning_rate[1]},
        ]
        optimizer_z = AdamW([param_groups[0]],betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
            eps=data_training_args.adam_epsilon,
            weight_decay=data_training_args.weight_decay,)
        optimizer_s = AdamW([param_groups[1]],betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
            eps=data_training_args.adam_epsilon,
            weight_decay=data_training_args.weight_decay,)
        
        "Prepare everything with HF accelerator"
        model, optimizer_z, optimizer_s, train_dataloader, eval_dataloader  = accelerator.prepare(
            model, optimizer_z, optimizer_s, train_dataloader, eval_dataloader
        )

        lr_scheduler_z = get_scheduler(
            name=data_training_args.lr_scheduler_type[0],
            optimizer=optimizer_z,
            num_warmup_steps=data_training_args.num_warmup_steps[0],
            num_training_steps=data_training_args.max_train_steps,
        ) 

        lr_scheduler_s = get_scheduler(
            name=data_training_args.lr_scheduler_type[1],
            optimizer=optimizer_s,
            num_warmup_steps=data_training_args.num_warmup_steps[1],
            num_training_steps=data_training_args.max_train_steps,
        ) 
        
    else:
        "Optimizer"
        optimizer = AdamW(
            list(model.parameters()),
            lr=data_training_args.learning_rate[0] if config.only_z_branch else data_training_args.learning_rate[1],
            betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
            eps=data_training_args.adam_epsilon,
            weight_decay=data_training_args.weight_decay,
        )

        "Prepare everything with HF accelerator"
        model, optimizer, train_dataloader, eval_dataloader  = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        lr_scheduler = get_scheduler(
                name=data_training_args.lr_scheduler_type[0] if config.only_z_branch else data_training_args.lr_scheduler_type[1],
                optimizer=optimizer,
                num_warmup_steps=data_training_args.num_warmup_steps[0] if config.only_z_branch else data_training_args.num_warmup_steps[1],
                num_training_steps=data_training_args.max_train_steps,
        ) 
            

    #n_params = count_parameters(model)

    "calculate number of training epochs"
    data_training_args.num_train_epochs = math.ceil(data_training_args.max_train_steps / num_update_steps_per_epoch)

    "Train"
    total_batch_size = data_training_args.per_device_train_batch_size * accelerator.num_processes * data_training_args.gradient_accumulation_steps
    print(f"Accelerate uses {accelerator.num_processes} processes")
    print(f"Dataloader has {len(train_dataloader)} steps in an epoch")

    logger.info("***** Running training *****")
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
    
    early_stopping = EarlyStopping()
    early_stopping.min_delta_percent = data_training_args.early_stop_min_delta_percent
    early_stopping.patience = data_training_args.early_stop_patience_epochs
    early_stopping.min_steps = data_training_args.early_stop_warmup_steps

    "save config with all parameters for the model + the run"
    if debugger_is_active():
        destination_config = os.path.join(data_training_args.output_dir, JSON_FILE_NAME_MANUAL)
        shutil.copy(JSON_FILE_NAME_MANUAL, destination_config)
    else:
        destination_config = os.path.join(data_training_args.output_dir,os.path.basename(args.config_file)) 
        shutil.copy(args.config_file,destination_config)

    for epoch in range(starting_epoch, data_training_args.num_train_epochs):
        model.train()
        saved_this_epoch = False
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch_size = batch["input_values"].shape[0] 
            mask_indices_seq_length = batch["input_values"].shape[2]

            "compute num of losses"                
            if data_training_args.dataset_name == "sim_vowels":
                batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)
                if hasattr(batch,"vowel_labels"):
                    batch.pop("vowel_labels")
                if hasattr(batch,"speaker_vt_factor"):
                    batch.pop("speaker_vt_factor")
            
            num_losses_orig = batch["mask_time_indices"].sum()
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )
            percent_masked = num_losses_orig / sub_attention_mask.sum()

            if data_training_args.dataset_name == "timit":
                batch["mask_time_indices"] = sub_attention_mask.clone()
                if data_training_args.dataset_name == "timit":
                    batch.pop("phonemes39", None)
                    batch.pop("phonemes48", None)
                batch.pop("start_phonemes", None)
                batch.pop("stop_phonemes", None)
                batch.pop("speaker_id", None)

            batch.pop("overlap_mask", None)

            if (batch["input_values"] != batch["input_values"]).any():
                print("NaNs in input_values")

            "Forward pass"
            outputs = model(**batch)

            "Find the percentage of masks used for the loss calculation"
            decomp_mask = outputs.mask_time_indices
            num_losses_from_mask = decomp_mask.sum().clone() #number of masked frames after silence removal if applicable
            "Different from num_losses as silent frames are excluded- e.g. with N = 100 not all masks are used"
            percent_masked_decomp = num_losses_from_mask / sub_attention_mask.sum()
            "Unpack positive/negative divergences"
            if config.dual_branched_latent or config.only_z_branch:
                div_dict_z = outputs.divergence_dict_z
                N_div_pos_z = outputs.N_div_pos_z
                N_div_neg_z = outputs.N_div_neg_z
            if config.dual_branched_latent or config.only_s_branch:
                div_dict_s = outputs.divergence_dict_s
                N_div_pos_s = outputs.N_div_pos_s
                N_div_neg_s = outputs.N_div_neg_s

            if config.only_z_branch:
                weighted_num_losses = N_div_neg_z*config.div_neg_weight + N_div_pos_z*config.div_pos_weight
            if config.dual_branched_latent:
                "The two branches will have equal number of decomposition losses"
                weighted_num_losses = 2*(N_div_neg_z*config.div_neg_weight + N_div_pos_z*config.div_pos_weight)
            if config.only_s_branch:
                weighted_num_losses = N_div_neg_s*config.div_neg_weight + N_div_pos_s*config.div_pos_weight

                
            if not weighted_num_losses.device == 'cuda':
                weighted_num_losses = weighted_num_losses.to('cuda')
                percent_masked_decomp = percent_masked_decomp.to('cuda')

            "Calculate the loss after gradient accumulation"
            loss = outputs.loss / data_training_args.gradient_accumulation_steps
            
            accelerator.backward(loss)

            "make sure that `num_losses` is summed for distributed training and average gradients over losses of all devices"
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather_for_metrics(weighted_num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / weighted_num_losses)
                
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
                if config.dual_branched_latent:
                    optimizer_z.step()
                    optimizer_s.step()
                    optimizer_z.zero_grad()
                    optimizer_s.zero_grad()
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler_z.step()
                        lr_scheduler_s.step()
                    elif accelerator.is_local_main_process:
                        progress_bar.write(
                            f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                        )
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                    if not accelerator.optimizer_step_was_skipped:
                        lr_scheduler.step()
                    elif accelerator.is_local_main_process:
                        progress_bar.write(
                            f"Gradients have overflown - skipping update step... Updating gradient scale to {scale}..."
                        )

                progress_bar.update(1)
                completed_steps += 1

            "Log all results"
            if (completed_steps+1) % (data_training_args.gradient_accumulation_steps * data_training_args.logging_steps) == 0:
                loss.detach()
                if config.dual_branched_latent or config.only_z_branch:
                    outputs.decomposition_loss_z.detach()
                    outputs.prior_loss_z.detach()
                    outputs.div_pos_z.detach()
                    outputs.div_neg_z.detach()
                    outputs.ce_pos_z.detach()
                    outputs.ce_neg_z.detach()
                if config.dual_branched_latent or config.only_s_branch:
                    outputs.decomposition_loss_s.detach()
                    outputs.prior_loss_s.detach()
                    outputs.div_pos_s.detach()
                    outputs.div_neg_s.detach()
                    outputs.ce_pos_s.detach()
                    outputs.ce_neg_s.detach()
                
                "Aggregate all metrics over devices"
                if accelerator.state.num_processes > 1:
                    loss = accelerator.gather_for_metrics(loss).sum()
                    if config.dual_branched_latent or config.only_z_branch:
                        outputs.decomposition_loss_z = accelerator.gather_for_metrics(outputs.decomposition_loss_z).sum()
                        outputs.prior_loss_z = accelerator.gather_for_metrics(outputs.prior_loss_z).sum()
                        outputs.div_pos_z = accelerator.gather_for_metrics(outputs.div_pos_z).sum()
                        outputs.div_neg_z = accelerator.gather_for_metrics(outputs.div_neg_z).sum()
                        outputs.ce_pos_z = accelerator.gather_for_metrics(outputs.ce_pos_z).sum()
                        outputs.ce_neg_z = accelerator.gather_for_metrics(outputs.ce_neg_z).sum()
                        N_div_pos_z = accelerator.gather_for_metrics(N_div_pos_z).sum()
                        N_div_neg_z = accelerator.gather_for_metrics(N_div_neg_z).sum()
                        div_dict_z = {key: accelerator.gather_for_metrics(values.detach()).sum() for key, values in div_dict_z.items()} 

                    if config.dual_branched_latent or config.only_s_branch:
                        outputs.decomposition_loss_s = accelerator.gather_for_metrics(outputs.decomposition_loss_s).sum()
                        outputs.prior_loss_s = accelerator.gather_for_metrics(outputs.prior_loss_s).sum()
                        outputs.div_pos_s = accelerator.gather_for_metrics(outputs.div_pos_s).sum()
                        outputs.div_neg_s = accelerator.gather_for_metrics(outputs.div_neg_s).sum()
                        outputs.ce_pos_s = accelerator.gather_for_metrics(outputs.ce_pos_s).sum()
                        outputs.ce_neg_s = accelerator.gather_for_metrics(outputs.ce_neg_s).sum()
                        N_div_pos_s = accelerator.gather_for_metrics(N_div_pos_s).sum()
                        N_div_neg_s = accelerator.gather_for_metrics(N_div_neg_s).sum()
                        div_dict_s = {key: accelerator.gather_for_metrics(values.detach()).sum() for key, values in div_dict_s.items()} 

                    percent_masked = accelerator.gather_for_metrics(percent_masked).sum()
                    percent_masked_decomp = accelerator.gather_for_metrics(percent_masked_decomp).sum()
                    
                if config.dual_branched_latent or config.only_z_branch:
                    div_dict_z = {key: values.detach() for key, values in div_dict_z.items()} 
                if config.dual_branched_latent or config.only_s_branch:
                    div_dict_s = {key: values.detach() for key, values in div_dict_s.items()}
                                
                "Training logs dictionary"
                if config.dual_branched_latent:
                    train_logs = {
                            "loss": (loss * data_training_args.gradient_accumulation_steps),
                            "%_mask_idx_decomp": percent_masked_decomp / accelerator.num_processes,                                
                            "lr_z": torch.tensor(optimizer_z.param_groups[0]["lr"]),                  
                            "lr_s": torch.tensor(optimizer_s.param_groups[0]["lr"]),
                            "grad_norm": torch.tensor(grad_norm),
                    }
                elif config.only_z_branch:
                    train_logs = {
                            "loss": (loss * data_training_args.gradient_accumulation_steps),
                            "%_mask_idx_decomp": percent_masked_decomp / accelerator.num_processes,                                
                            "lr_z": torch.tensor(optimizer.param_groups[0]["lr"]),                  
                            "lr_s": 0,
                            "grad_norm": torch.tensor(grad_norm),
                    }
                elif config.only_s_branch:
                    train_logs = {
                            "loss": (loss * data_training_args.gradient_accumulation_steps),
                            "%_mask_idx_decomp": percent_masked_decomp / accelerator.num_processes,                                
                            "lr_z": 0,                  
                            "lr_s": torch.tensor(optimizer.param_groups[0]["lr"]),
                            "grad_norm": torch.tensor(grad_norm),
                    }
                if config.dual_branched_latent or config.only_z_branch:
                    train_logs_z = {
                        "decomposition_loss_z": outputs.decomposition_loss_z,
                        "prior_loss_z": outputs.prior_loss_z,
                        "divergence_positive_z": outputs.div_pos_z, 
                        "divergence_negative_z": outputs.div_neg_z,
                        "cross_entropy_positive_z": outputs.ce_pos_z,
                        "cross_entropy_negative_z": outputs.ce_neg_z,
                    }
                    train_logs_z = {**train_logs_z,**{"div_0_{}_z".format(i+1): 0 for i in range(config.NoC)}}
                    train_logs_z = {**train_logs_z, **{"div_{}_{}_z".format(i+1,j+1): 0 for i in range(config.NoC) for j in range(i+1) if i != j}}
                
                    for k in div_dict_z.keys():                
                        train_logs_z[k+"_z"] += div_dict_z[k] 

                    train_logs = {**train_logs, **train_logs_z}
                
                if config.dual_branched_latent or config.only_s_branch:
                    train_logs_s = {
                        "decomposition_loss_s": outputs.decomposition_loss_s,
                        "prior_loss_s": outputs.prior_loss_s,
                        "divergence_positive_s": outputs.div_pos_s, 
                        "divergence_negative_s": outputs.div_neg_s,
                        "cross_entropy_positive_s": outputs.ce_pos_s,
                        "cross_entropy_negative_s": outputs.ce_neg_s,
                    }
                    train_logs_s = {**train_logs_s,**{"div_0_{}_s".format(i+1): 0 for i in range(config.NoC_seq)}}
                    train_logs_s = {**train_logs_s, **{"div_{}_{}_s".format(i+1,j+1): 0 for i in range(config.NoC_seq) for j in range(i+1) if i != j}}

                    for k in div_dict_s.keys():                
                        train_logs_s[k+"_s"] += div_dict_s[k] 

                    train_logs = {**train_logs, **train_logs_s}
                    
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
                        

            "save model and latents every `args.saving_steps` steps"
            if (completed_steps+1) % (data_training_args.gradient_accumulation_steps * data_training_args.saving_steps) == 0:
                saved_this_epoch = True
                "Save model"
                if (data_training_args.push_to_hub and epoch < data_training_args.num_train_epochs - 1) or data_training_args.output_dir is not None:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)   
                    if epoch < 10:
                        model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                    else:
                        model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                    unwrapped_model.save_pretrained(
                        model_name, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )

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
            
        del outputs

        
        if epoch == 0:
            saved_this_epoch = True
            "Save model"
            if (data_training_args.push_to_hub and epoch < data_training_args.num_train_epochs - 1) or data_training_args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)   
                if epoch < 10:
                    model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                else:
                    model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                unwrapped_model.save_pretrained(
                    model_name, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )

        "Validate"
        if eval_dataloader is not None:
            model.eval()

            "initialize validation logs"
            if config.dual_branched_latent or config.only_z_branch:
                div_dict_z = {"val_div_0_{}_z".format(i+1): 0 for i in range(config.NoC)}
                div_dict_z = {**div_dict_z, **{"val_div_{}_{}_z".format(i+1,j+1): 0 for i in range(config.NoC) for j in range(i+1) if i != j}}
            if config.dual_branched_latent or config.only_s_branch:
                div_dict_s = {"val_div_0_{}_s".format(i+1): 0 for i in range(config.NoC_seq)}
                div_dict_s = {**div_dict_s, **{"val_div_{}_{}_s".format(i+1,j+1): 0 for i in range(config.NoC_seq) for j in range(i+1) if i != j}}

            val_logs = {
                "val_loss": 0,
            }
            if config.dual_branched_latent or config.only_z_branch:
                val_logs_z = {
                    "val_decomposition_loss_z": 0,
                    "val_prior_loss_z": 0,
                    "val_divergence_positive_z": 0,
                    "val_divergence_negative_z": 0,
                    "val_cross_entropy_positive_z": 0,
                    "val_cross_entropy_negative_z": 0,

                }
                val_logs_z = {**val_logs_z, **div_dict_z}
                val_logs = {**val_logs,**val_logs_z}
            if config.dual_branched_latent or config.only_s_branch:
                val_logs_s = {
                    "val_decomposition_loss_s": 0,
                    "val_prior_loss_s": 0,
                    "val_divergence_positive_s": 0,
                    "val_divergence_negative_s": 0,
                    "val_cross_entropy_positive_s": 0,
                    "val_cross_entropy_negative_s": 0,
                }
                val_logs_s = {**val_logs_s, **div_dict_s}
                val_logs = {**val_logs,**val_logs_s}

            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    if data_training_args.dataset_name == "sim_vowels":
                        batch["mask_time_indices"] = torch.ones_like(batch["mask_time_indices"])
                        if hasattr(batch,"vowel_labels"):
                            batch.pop("vowel_labels")
                        if hasattr(batch,"speaker_vt_factor"):
                            batch.pop("speaker_vt_factor")
                    elif data_training_args.dataset_name == "timit":
                        batch["mask_time_indices"] = batch["sub_attention_mask"].clone()
                        if data_training_args.dataset_name == "timit":
                            batch.pop("phonemes39", None)
                            batch.pop("phonemes48", None)
                        batch.pop("start_phonemes", None)
                        batch.pop("stop_phonemes", None)
                        batch.pop("speaker_id", None)

                    batch.pop("sub_attention_mask", None)
                    batch.pop("overlap_mask", None)

                    outputs = model(**batch)

                    del batch

                
                val_logs["val_loss"] += outputs.loss
                if config.dual_branched_latent or config.only_z_branch:
                    for k in outputs.divergence_dict_z.keys():                
                        val_logs["val_"+k+"_z"] += outputs.divergence_dict_z[k] 
                    val_logs["val_decomposition_loss_z"] += outputs.decomposition_loss_z
                    val_logs["val_prior_loss_z"] += outputs.prior_loss_z
                    val_logs["val_divergence_positive_z"] += outputs.div_pos_z
                    val_logs["val_divergence_negative_z"] += outputs.div_neg_z
                    val_logs["val_cross_entropy_positive_z"] += outputs.ce_pos_z
                    val_logs["val_cross_entropy_negative_z"] += outputs.ce_neg_z
                if config.dual_branched_latent or config.only_s_branch:
                    for k in outputs.divergence_dict_s.keys():                
                        val_logs["val_"+k+"_s"] += outputs.divergence_dict_s[k] 
                    val_logs["val_decomposition_loss_s"] += outputs.decomposition_loss_s
                    val_logs["val_prior_loss_s"] += outputs.prior_loss_s
                    val_logs["val_divergence_positive_s"] += outputs.div_pos_s
                    val_logs["val_divergence_negative_s"] += outputs.div_neg_s
                    val_logs["val_cross_entropy_positive_s"] += outputs.ce_pos_s
                    val_logs["val_cross_entropy_negative_s"] += outputs.ce_neg_s

            for k in val_logs.keys():
                val_logs[k] = val_logs[k] /  len(eval_dataloader)

            "sum over devices in multi-processing"
            if accelerator.num_processes > 1:
                val_logs = {k: accelerator.gather_for_metrics(v).sum() for k, v in val_logs.items()}

            log_str = ""
            for k, v in val_logs.items():
                try:
                    log_str += "| {}: {:.3e}".format(k, v.item())
                except:
                    log_str += "| {}: {:.3e}".format(k, v)         

            if accelerator.is_local_main_process:
                progress_bar.write(log_str)
                if is_wandb_available() and data_training_args.with_wandb:
                    wandb.log(val_logs)

            "Check val_loss and replace min_val_loss with current loss"   
            if val_logs["val_loss"] < min_val_loss:
                min_val_loss = val_logs["val_loss"]

            "Save model if validation loss is lower than min_val_loss"
            if data_training_args.output_dir is not None and \
                min_val_loss >= val_logs["val_loss"]: 
        
                accelerator.wait_for_everyone()
                "Save model"
                if data_training_args.save_model and not saved_this_epoch:
                    saved_this_epoch = True
                    unwrapped_model = accelerator.unwrap_model(model)
                    if epoch < 10:
                        model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                    else:
                        model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                    model_name += "_min_val_loss" 
                    
                    unwrapped_model.save_pretrained(
                        model_name, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    
                "Push model dir to HF hub"
                if accelerator.is_main_process:
                    if data_training_args.push_to_hub:
                        try:
                            api.upload_folder(
                                commit_message=f"Training in progress epoch {epoch}",
                                folder_path=data_training_args.output_dir,
                                repo_id=repo_id,
                                repo_type="model",
                                token=data_training_args.hub_token,
                            )
                        except:# ConnectionError:
                            logger.warning("Could not push to the hub. Connection error.")

            "Check early stopping"
            early_stopping(val_logs["val_loss"],completed_steps)

            if early_stopping.early_stop:
                print("Early stopping triggered. End of training after {} epochs".format(epoch+1))
                if saved_this_epoch:
                    break
                else:
                    "Save last model"
                    saved_this_epoch = True
                    if data_training_args.output_dir is not None and data_training_args.save_model:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        if epoch < 10:
                            model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                        else:
                            model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                        if min_val_loss >= val_logs["val_loss"]:
                            model_name += "_min_val_loss" 
                        
                        unwrapped_model.save_pretrained(
                            model_name, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                        )
                    
                    "Push model dir to HF hub"
                    if accelerator.is_main_process:
                        if data_training_args.push_to_hub:
                            try:
                                api.upload_folder(
                                    commit_message="End of training",
                                    folder_path=data_training_args.output_dir,
                                    repo_id=repo_id,
                                    repo_type="model",
                                    token=data_training_args.hub_token,
                                )
                            except:# ConnectionError:
                                logger.warning("Could not push to the hub. Connection error.")
                    break
    
    "Save last model"
    if not saved_this_epoch and data_training_args.output_dir is not None and data_training_args.save_model:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if epoch < 10:
            model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
        else:
            model_name = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
        
        unwrapped_model.save_pretrained(
            model_name, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
    
        "Push model dir to HF hub"
        if accelerator.is_main_process:
            if data_training_args.push_to_hub:
                try:
                    api.upload_folder(
                        commit_message="End of training",
                        folder_path=data_training_args.output_dir,
                        repo_id=repo_id,
                        repo_type="model",
                        token=data_training_args.hub_token,
                    )
                except:# ConnectionError:
                    logger.warning("Could not push to the hub. Connection error.")



if __name__ == "__main__":
    main()
