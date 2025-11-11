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
"""Fine-tune (domain adaptation with VAE loss) pretrained VAE models on speech datasets. Supports finetuning
 on IEMOCAP, SimVowels, and TIMIT for now. VOC-ALS is directly processed in latents_post_analysis_vae1D.py."""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

import transformers
from transformers import (
    AdamW,
    Wav2Vec2FeatureExtractor,
    get_scheduler,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from models import VAE_1D, VAE_1D_FC, DecompositionModule
from data_collation import DataCollatorForVAE1D_SSL_FineTuning
from data_preprocessing import prepare_pretraining_dataset
from config_files import DecVAEConfig
from utils import (
    parse_args, 
    debugger_is_active, 
    extract_epoch,
    multiply_grads, 
    count_parameters, 
    EarlyStopping, 
    get_grad_norm
)

from args_configs import ModelArguments, DataTrainingArguments, DecompositionArguments, TrainingObjectiveArguments
from dataset_loading import load_librispeech, load_timit, load_sim_vowels, load_iemocap, load_voc_als, load_scRNA_seq

from functools import partial
import math
import os
import shutil
from pathlib import Path

from safetensors.torch import save_model, load_file
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
import json
from feature_extraction.mel_features import extract_mel_spectrogram
from feature_extraction import extract_fft_psd

#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"

JSON_FILE_NAME_MANUAL = "config_files/VAEs/iemocap/fine-tuning/config_finetune_vae1d_iemocap.json" #for debugging purposes only

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

            wandb.init(data_training_args.wandb_project, group=data_training_args.wandb_group) #
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
        
        "load audio files into numpy arrays"
        with accelerator.main_process_first():

            vectorized_datasets = raw_datasets.map(
                partial(
                    prepare_pretraining_dataset,
                    feature_extractor=feature_extractor,
                    data_training_args=data_training_args,
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

    if data_training_args.transfer_learning and data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
        print(f"Using transfer learning for {data_training_args.dataset_name}")

        "Checkpoint directory"
        if model_args.vae_type == "VAE_1D":
            model_type = "vae1d"
        elif model_args.vae_type == "VAE_1D_FC":
            model_type = "vae1d_fc"
        elif model_args.vae_type == "VAE_1D_seq":
            model_type = "vae1d_seq"
        elif model_args.vae_type == "VAE_1D_FC_seq":
            model_type = "vae1d_fc_seq"
        
        if model_args.vae_beta == 0.1:
            beta = "_b01"
        else:
            beta = "_b" + str(int(model_args.vae_beta))
        
        if "mel" in model_args.vae_input_type:
            str_input_type = "mel"
        elif "waveform" in model_args.vae_input_type:
            str_input_type = "waveform"
    
        # Construct checkpoint directory path
        if data_training_args.transfer_from == "timit":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                "timit" + beta + "_" + model_type + "_" + str_input_type + "_bs" + str(data_training_args.per_device_train_batch_size))
        elif data_training_args.transfer_from == "sim_vowels":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                "snr" + str(data_training_args.sim_snr_db) \
                + beta + "_" + model_type + "_" + str_input_type + "_bs" + str(data_training_args.per_device_train_batch_size))
        
        if data_training_args.experiment == "z_size":
            checkpoint_dir += "_z" + str(model_args.vae_z_dim)

        print(f"Loading pretrained model from checkpoint directory: {checkpoint_dir}")
        
        # Find the latest checkpoint
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'config' not in f]
        checkpoint_files.sort(key=extract_epoch)
        
        # Initialize the model based on model type
        if model_args.vae_type == "VAE_1D":
            in_size = max_length
            model = VAE_1D(z_dim=model_args.vae_z_dim, 
                    proj_intermediate_dim=model_args.vae_proj_intermediate_dim, 
                    conv_dim=model_args.vae_conv_dim, 
                    treat_as_sequence=False,
                    kernel_sizes=model_args.vae_kernel_sizes,
                    strides=model_args.vae_strides, 
                    in_size=in_size, 
                    norm_type=model_args.vae_norm_type,
                    hidden_dim=model_args.vae_hidden_dim,  
                    beta=model_args.vae_beta,
                    warmup_steps=data_training_args.num_warmup_steps,
                    kl_annealing=model_args.kl_annealing,
                )
        elif model_args.vae_type == "VAE_1D_seq":
            in_size = max_length
            model = VAE_1D(z_dim=model_args.vae_z_dim, 
                    proj_intermediate_dim=model_args.vae_proj_intermediate_dim, 
                    conv_dim=model_args.vae_conv_dim, 
                    treat_as_sequence=True,
                    kernel_sizes=model_args.vae_kernel_sizes,
                    strides=model_args.vae_strides, 
                    in_size=in_size, 
                    norm_type=model_args.vae_norm_type,
                    hidden_dim=model_args.vae_hidden_dim,  
                    beta=model_args.vae_beta,
                    warmup_steps=data_training_args.num_warmup_steps,
                    kl_annealing=model_args.kl_annealing,
                )
        elif model_args.vae_type == "VAE_1D_FC":
            if model_args.vae_input_type == "mel":
                in_size = int(model_args.n_mels_vae)
            elif model_args.vae_input_type == "waveform":
                in_size = int(config.receptive_field * config.fs)
            
            model = VAE_1D_FC(z_dim=model_args.vae_z_dim,
                    hidden_dims=model_args.vae_fc_dims,
                    kernel_sizes=model_args.vae_kernel_sizes,
                    treat_as_sequence=False,
                    strides=model_args.vae_strides, 
                    in_size=in_size,
                    norm_type=model_args.vae_norm_type,
                    beta=model_args.vae_beta,
                    warmup_steps=data_training_args.num_warmup_steps,
                    kl_annealing=model_args.kl_annealing
                )
        elif model_args.vae_type == "VAE_1D_FC_seq":
            in_size = max_length
            model = VAE_1D_FC(z_dim=model_args.vae_z_dim,
                    hidden_dims=model_args.vae_fc_dims,
                    kernel_sizes=model_args.vae_kernel_sizes,
                    treat_as_sequence=True,
                    strides=model_args.vae_strides, 
                    in_size=in_size,
                    norm_type=model_args.vae_norm_type,
                    beta=model_args.vae_beta,
                    warmup_steps=data_training_args.num_warmup_steps,
                    kl_annealing=model_args.kl_annealing
                )

        # Load pretrained weights if available
        if checkpoint_files and checkpoint_files[0] != 'epoch_-01':
            checkpoint = checkpoint_files[data_training_args.which_checkpoint]
            print(f"Loading weights from checkpoint {checkpoint}")
            pretrained_model_file = os.path.join(checkpoint_dir, checkpoint, "model.safetensors")
            
            if os.path.exists(pretrained_model_file):
                weights = load_file(pretrained_model_file)
                model.load_state_dict(weights, strict=False)
                print(f"Successfully loaded pretrained weights from {pretrained_model_file}")
            else:
                print(f"Warning: Could not find model file at {pretrained_model_file}, initializing model from scratch")
        else:
            print("Training from scratch (no pretrained weights found)")
    else:
        print("Only supporting transfer learning, exiting...")
        return

    # Activate gradient checkpointing if needed
    if data_training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Data collator
    mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
    mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

    data_collator = DataCollatorForVAE1D_SSL_FineTuning(
        model=model,
        model_name=model_args.vae_type,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=data_training_args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
        dataset_name=data_training_args.dataset_name,
    )

    if data_training_args.dataset_name == "iemocap":
        vectorized_datasets['train'] = concatenate_datasets([vectorized_datasets['train'],vectorized_datasets['test']])
    train_dataloader = DataLoader(
        vectorized_datasets['train'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=data_training_args.per_device_train_batch_size,
    )
    
    eval_dataloader = DataLoader(
        vectorized_datasets["validation"], 
        collate_fn=data_collator, 
        batch_size=data_training_args.per_device_eval_batch_size
    )

    # Optimizer
    optimizer = AdamW(
        list(model.parameters()),
        lr=data_training_args.learning_rate,
        betas=[data_training_args.adam_beta1, data_training_args.adam_beta2],
        eps=data_training_args.adam_epsilon,
        weight_decay=data_training_args.weight_decay,
    )

    # Prepare everything with HF accelerator
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"Model has {n_params:,} trainable parameters")

    # Scheduler and math around the number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / data_training_args.gradient_accumulation_steps)

    if data_training_args.max_train_steps is None:
        data_training_args.max_train_steps = data_training_args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=data_training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=data_training_args.num_warmup_steps,
        num_training_steps=data_training_args.max_train_steps,
        scheduler_specific_kwargs={'num_cycles': data_training_args.lr_scheduler_num_cycles} if data_training_args.lr_scheduler_type == "cosine_with_restarts" else None,
    )

    # Calculate number of training epochs
    data_training_args.num_train_epochs = math.ceil(data_training_args.max_train_steps / num_update_steps_per_epoch)

    # Training
    total_batch_size = data_training_args.per_device_train_batch_size * accelerator.num_processes * data_training_args.gradient_accumulation_steps
    
    print(f"Accelerate uses {accelerator.num_processes} processes")
    print(f"Dataloader has {len(train_dataloader)} steps in an epoch")

    logger.info("***** Running fine-tuning *****")
    logger.info(f"  Num examples = {len(vectorized_datasets['train'])}")
    logger.info(f"  Num Epochs = {data_training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {data_training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {data_training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {data_training_args.max_train_steps}")

    # Only show the progress bar once on each machine
    progress_bar = tqdm(range(data_training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    min_val_loss = 1000000
    
    early_stopping = EarlyStopping()
    early_stopping.min_delta_percent = data_training_args.early_stop_min_delta_percent
    early_stopping.patience = data_training_args.early_stop_patience_epochs
    early_stopping.min_steps = data_training_args.early_stop_warmup_steps

    # Save config with all parameters for the model + the run
    if debugger_is_active():
        destination_config = os.path.join(data_training_args.output_dir, JSON_FILE_NAME_MANUAL)
        shutil.copy(JSON_FILE_NAME_MANUAL, destination_config)
    else:
        destination_config = os.path.join(data_training_args.output_dir, args.config_file) 
        shutil.copy(args.config_file, destination_config)

    for epoch in range(starting_epoch, data_training_args.num_train_epochs):
        model.train()
        saved_this_epoch = False
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            batch_size = batch["input_values"].shape[0]
            mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
            
            "compute num of losses"                
            if data_training_args.dataset_name == "sim_vowels":
                batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)
            

                if hasattr(batch,"vowel_labels"):
                    batch.pop("vowel_labels")
                if hasattr(batch,"speaker_vt_factor"):
                    batch.pop("speaker_vt_factor")
            
            sub_attention_mask = batch.pop("sub_attention_mask", None)
            sub_attention_mask = (
                sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["mask_time_indices"])
            )

            if data_training_args.dataset_name in ["timit", "iemocap"]:
                batch["mask_time_indices"] = sub_attention_mask.clone()
                if data_training_args.dataset_name == "timit":
                    batch.pop("phonemes39", None)
                    batch.pop("phonemes48", None)
                elif data_training_args.dataset_name == "iemocap":
                    batch.pop("phonemes", None)
                    batch.pop("emotion", None)
                batch.pop("start_phonemes", None)
                batch.pop("stop_phonemes", None)
                batch.pop("speaker_id", None)
            batch.pop("overlap_mask", None)

            if (batch["input_values"] != batch["input_values"]).any():
                print("NaNs in input_values")
            
            batch["global_step"] = completed_steps
            
            "Forward pass"

            if model_args.vae_type == "VAE_1D_FC":
                if model_args.vae_input_type == "waveform":
                    batch["input_values"] = batch["input_values"][:,0,:,:]
                elif model_args.vae_input_type == "mel":
                    batch["input_values"], _ = extract_mel_spectrogram(batch["input_values"][:,0,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                elif model_args.vae_input_type == "fft":              
                    "Apply fft using welch's power spectral density estimation"
                    batch["input_values"], _ = extract_fft_psd(
                        batch, 
                        normalize=True, #self.data_training_args.mel_norm,
                        device=batch["input_values"].device
                    )

                batch["attention_mask"] = sub_attention_mask
            elif model_args.vae_type == "VAE_1D_FC_seq":
                "If input will be mel filterbank features, split sequence in frames"
                if model_args.vae_input_type == 'mel':
                    frame_len = int(batch["input_values"].shape[-1]/10) #int(config.receptive_field*config.fs)
                    frames = batch["input_values"].shape[-1]/frame_len
                    new_input_seq_values = torch.zeros((batch_size,int(frames),frame_len),device = batch["input_values"].device)
                    for o in range(batch_size):
                        sequence = batch["input_values"][o,:].clone()
                        for f in range(int(frames)):
                            framed_sequence = sequence[f*frame_len:(f+1)*frame_len]
                            new_input_seq_values[o,f,:] = framed_sequence.clone()
                    batch["input_values"] = new_input_seq_values.clone()

                    for o in range(batch_size):
                        batch["input_values"][o,...], _ = extract_mel_spectrogram(batch["input_values"][o,...].unsqueeze(0),config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len)
                       
                    "Flatten sequence - Reverse framing"
                    batch["input_values"] = batch["input_values"].reshape(batch["input_values"].shape[0],-1)
                
                elif model_args.vae_input_type == "fft":              
                    "Apply fft using welch's power spectral density estimation"
                    batch["input_values"], _ = extract_fft_psd(
                        batch, 
                        normalize=True, #self.data_training_args.mel_norm,
                        device=batch["input_values"].device
                    )

            outputs = model(**batch)
            z_mean, z_logvar, z, recon_x, vae_loss, recon_loss, kld_loss = outputs
            attention_mask = batch["attention_mask"]
            
            "Find the percentage of masks used for the loss calculation"
            num_losses = sub_attention_mask.sum()

            "Calculate the loss after gradient accumulation"
            loss = vae_loss / data_training_args.gradient_accumulation_steps
            
            accelerator.backward(loss)

            "make sure that `num_losses` is summed for distributed training and average gradients over losses of all devices"
            if accelerator.state.num_processes > 1:
                num_losses = accelerator.gather_for_metrics(num_losses).sum()
                gradient_multiplier = accelerator.state.num_processes / num_losses
                multiply_grads(model.module.parameters(), gradient_multiplier)
            else:
                multiply_grads(model.parameters(), 1 / num_losses)
                
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

                progress_bar.update(1)
                completed_steps += 1

            "Log all results"
            if (completed_steps+1) % (data_training_args.gradient_accumulation_steps * data_training_args.logging_steps) == 0:
                vae_loss.detach()
                recon_loss.detach()
                kld_loss.detach()
                
                "Aggregate all metrics over devices"
                if accelerator.state.num_processes > 1:
                    vae_loss = accelerator.gather_for_metrics(vae_loss).sum()
                    recon_loss = accelerator.gather_for_metrics(recon_loss).sum()
                    kld_loss = accelerator.gather_for_metrics(kld_loss).sum()
                                
                "Training logs dictionary"
                train_logs = {
                        "vae_loss": vae_loss,
                        "recon_loss": recon_loss,
                        "kld_loss": kld_loss,
                        "num_losses": num_losses / accelerator.num_processes,                                
                        "lr": torch.tensor(optimizer.param_groups[0]["lr"]),                  
                        "grad_norm": torch.tensor(grad_norm),
                }
                    
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
                        model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                    else:
                        model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    save_model(unwrapped_model, os.path.join(model_dir,"model.safetensors"))

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
                    model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                else:
                    model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                save_model(unwrapped_model, os.path.join(model_dir,"model.safetensors"))

        "Validate"
        model.eval()

        val_logs = {
            "val_vae_loss": 0,
            "val_recon_x_loss": 0,
            "val_kld_loss": 0,
            "val_num_losses": 0,
        }
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):        
                batch_size = batch["input_values"].shape[0]
                if data_training_args.dataset_name == "sim_vowels":
                    batch["mask_time_indices"] = torch.ones_like(batch["mask_time_indices"])
                    if model_args.vae_type == "VAE_1D_FC":
                        batch["attention_mask"] = batch["sub_attention_mask"]
                    if hasattr(batch,"ground_truth_sources"):
                        ground_truth_sources = batch.pop("ground_truth_sources")
                    if hasattr(batch,"detected_sources"):
                        detected_sources = batch.pop("detected_sources")
                    if hasattr(batch,"vowel_labels"):
                        vowel_labels = batch.pop("vowel_labels")
                    if hasattr(batch,"speaker_vt_factor"):
                        speaker_vt_factor = batch.pop("speaker_vt_factor")
                elif data_training_args.dataset_name in ["timit", "iemocap"]:
                    if model_args.vae_type == "VAE_1D_FC":
                        batch["attention_mask"] = batch["sub_attention_mask"]
                    batch["mask_time_indices"] = batch["sub_attention_mask"].clone()
                    if data_training_args.dataset_name == "timit":
                        batch.pop("phonemes39", None)
                        batch.pop("phonemes48", None)
                    elif data_training_args.dataset_name == "iemocap":
                        batch.pop("phonemes", None)
                        batch.pop("emotion", None)
                    batch.pop("start_phonemes", None)
                    batch.pop("stop_phonemes", None)
                    batch.pop("speaker_id", None)
                
                batch.pop("sub_attention_mask", None)
                batch.pop("overlap_mask", None)
                batch["global_step"] = completed_steps
                num_losses = batch["mask_time_indices"].sum()

                if model_args.vae_type == "VAE_1D_FC":
                    if model_args.vae_input_type == "waveform":
                        batch["input_values"] = batch["input_values"][:,0,:,:]
                    elif model_args.vae_input_type == "mel":
                        batch["input_values"], _ = extract_mel_spectrogram(batch["input_values"][:,0,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                    elif model_args.vae_input_type == "fft": 
                        "Apply fft using welch's power spectral density estimation"      
                        batch["input_values"], _ = extract_fft_psd(
                            batch, 
                            normalize=True, #self.data_training_args.mel_norm,
                            device=batch["input_values"].device
                        )
                elif model_args.vae_type == "VAE_1D_FC_seq":
                    "If input will be mel filterbank features, split sequence in frames"
                    if model_args.vae_input_type == 'mel':
                        frame_len = int(batch["input_values"].shape[-1]/10) #int(config.receptive_field*config.fs)
                        frames = batch["input_values"].shape[-1]/frame_len
                        new_input_seq_values = torch.zeros((batch_size,int(frames),frame_len),device = batch["input_values"].device)
                        for o in range(batch_size):
                            sequence = batch["input_values"][o,:].clone()
                            for f in range(int(frames)):
                                framed_sequence = sequence[f*frame_len:(f+1)*frame_len]
                                new_input_seq_values[o,f,:] = framed_sequence.clone()
                        batch["input_values"] = new_input_seq_values.clone()

                        for o in range(batch_size):
                            batch["input_values"][o,...], _ = extract_mel_spectrogram(batch["input_values"][o,...].unsqueeze(0),config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len)
                        
                        "Flatten sequence - Reverse framing"
                        batch["input_values"] = batch["input_values"].reshape(batch["input_values"].shape[0],-1)
                    elif model_args.vae_input_type == "fft": 
                        "Apply fft using welch's power spectral density estimation"      
                        batch["input_values"], _ = extract_fft_psd(
                            batch, 
                            normalize=True, #self.data_training_args.mel_norm,
                            device=batch["input_values"].device
                        )
                        
                outputs = model(**batch)
                z_mean, z_logvar, z, recon_x, vae_loss, recon_loss, kld_loss = outputs
                attention_mask = batch["attention_mask"]
 
            val_logs["val_vae_loss"] += vae_loss
            val_logs["val_recon_x_loss"] += recon_loss
            val_logs["val_kld_loss"] += kld_loss
            val_logs["val_num_losses"] += num_losses

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
        if val_logs["val_vae_loss"] < min_val_loss:
            min_val_loss = val_logs["val_vae_loss"]

        "Save model if validation loss is lower than min_val_loss, every #check_val_loss_every_epochs epochs"
        if data_training_args.output_dir is not None and \
            min_val_loss >= val_logs["val_vae_loss"]: 
    
            accelerator.wait_for_everyone()
            "Save model"
            if data_training_args.save_model and not saved_this_epoch:
                saved_this_epoch = True
                unwrapped_model = accelerator.unwrap_model(model)                
                if epoch < 10:
                    model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                else:
                    model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                model_dir += "_min_val_loss" 
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                save_model(unwrapped_model, os.path.join(model_dir,"model.safetensors"))

                
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
        early_stopping(val_logs["val_vae_loss"],completed_steps)

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
                    model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
                else:
                    model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
                if min_val_loss >= val_logs["val_vae_loss"]:
                    model_dir += "_min_val_loss" 
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                save_model(unwrapped_model, os.path.join(model_dir,"model.safetensors"))
                
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
            model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_0" + str(epoch))
        else:
            model_dir = os.path.join(data_training_args.output_dir,"training_ckp_epoch_" + str(epoch))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        save_model(unwrapped_model, os.path.join(model_dir,"model.safetensors"))

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
