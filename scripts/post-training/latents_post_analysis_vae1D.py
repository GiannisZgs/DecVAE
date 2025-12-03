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

"""This script handles all latent evaluations (classification, disentanglement) for VAE models. 
This script loads pretrained VAE models from specified checkpoints, or initializes random VAE models or ICA/PCA/kPCA models, 
gathers representations per data point and computes classification (task-related) and disentanglement metrics. 
See arguments data_training_args.classification_tasks in args_configs.data_training_args.DataTrainingArgumentsPost for more details
on selecting which variables to classify for each dataset.
Decomposition of inputs is not supported here so if it's not already calculated then another script like vaes_pretraining.py should be ran first."""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import VAE_1D, VAE_1D_FC
from data_collation import DataCollatorForVAE1DLatentPostAnalysis
from config_files import DecVAEConfig
from sklearn.decomposition import PCA, FastICA, KernelPCA 
import joblib
from args_configs import ModelArgumentsPost, DataTrainingArgumentsPost, DecompositionArguments, TrainingObjectiveArguments
from utils import parse_args, debugger_is_active, extract_epoch
from latent_analysis_utils import prediction_eval
from disentanglement_utils import compute_disentanglement_metrics
from feature_extraction import extract_mel_spectrogram

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from safetensors.torch import load_file
import pandas as pd
import numpy as np
from dataclasses import field
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from torch.utils.data.dataloader import DataLoader
import time
import json
import gzip
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

JSON_FILE_NAME_MANUAL = "config_files/VAEs/voc_als/latent_evaluations/config_vae1d_latent_anal_voc_als.json" #for debugging purposes only
SAVE_DIR_PLOTS = '/home/giannis/Documents/DecSSL/R_vis/latent_quality/low_dim_vis_latents/' #'/home/giannis/Documents/DecSSL/R_vis/latent_quality/low_dim_vis_latents/'

logger = get_logger(__name__)

def main():
    "Parse the arguments"
    parser = HfArgumentParser((ModelArgumentsPost, DataTrainingArgumentsPost, TrainingObjectiveArguments, DecompositionArguments))
    if debugger_is_active() or ('TERM_PROGRAM' in os.environ.keys() and os.environ['TERM_PROGRAM'] == 'vscode'):
        model_args, data_training_args, training_obj_args, decomp_args = parser.parse_json_file(json_file=JSON_FILE_NAME_MANUAL)
    else:
        args = parse_args()
        model_args, data_training_args, training_obj_args, decomp_args = parser.parse_json_file(json_file=args.config_file)
    delattr(model_args,"comment_model_args")
    delattr(data_training_args,"comment_data_args")
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
        if data_training_args.preprocessing_num_workers is None or data_training_args.preprocessing_num_workers == 1:
            cache_file_names = {"train": [data_training_args.train_cache_file_name],
                "validation": [data_training_args.validation_cache_file_name]
            }
            if data_training_args.test_cache_file_name is not None:
                cache_file_names["test"] = [data_training_args.test_cache_file_name]
            if data_training_args.dev_cache_file_name is not None:
                cache_file_names["dev"] = [data_training_args.dev_cache_file_name]
        else:   
            cache_file_names = {"train": [data_training_args.train_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)],
                    "validation": [data_training_args.validation_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
            }
            if data_training_args.test_cache_file_name is not None:
                cache_file_names["test"] = [data_training_args.test_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
            if data_training_args.dev_cache_file_name is not None:
                cache_file_names["dev"] = [data_training_args.dev_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
    
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
        if data_training_args.test_cache_file_name is not None:
            vectorized_datasets["test"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["test"]])

        if data_training_args.dataset_name == "VOC_ALS":
            vectorized_datasets["dev"] = concatenate_datasets([Dataset.from_file(file) for file in cache_file_names["dev"]])

        if min_length > 0.0:
            vectorized_datasets = vectorized_datasets.filter(
                lambda x: x > min_length,
                num_proc=data_training_args.preprocessing_num_workers,
                input_columns=["input_length"],
            )
        vectorized_datasets = vectorized_datasets.remove_columns("input_length")

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
    if "vowels" in data_training_args.dataset_name:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            "snr" + str(data_training_args.sim_snr_db) \
            + beta + "_" + model_type + "_" + str_input_type + "_bs" + str(data_training_args.per_device_train_batch_size))
    elif "timit" in data_training_args.dataset_name:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            "timit" + beta +  "_" + model_type + "_" + str_input_type + "_bs" + str(data_training_args.per_device_train_batch_size))
    elif "iemocap" in data_training_args.dataset_name:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            beta[1:] + "_" + model_type + "_" + str_input_type + "_bs" + str(data_training_args.per_device_train_batch_size))
    elif "VOC_ALS" in data_training_args.dataset_name:
        if data_training_args.transfer_from == "timit":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                "timit" + beta + "_" + model_type + "_" + str_input_type + "_bs" + str(data_training_args.per_device_train_batch_size))
        elif data_training_args.transfer_from == "sim_vowels":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                "snr" + str(data_training_args.sim_snr_db) \
                + beta + "_" + model_type + "_" + str_input_type + "_bs" + str(data_training_args.per_device_train_batch_size))

    if data_training_args.experiment == "z_size":
        checkpoint_dir += "_z" + str(model_args.vae_z_dim)

    if model_args.eigenprojection is not None:
        if "VOC_ALS" in data_training_args.dataset_name:
            projection_dir = os.path.join(os.path.dirname(data_training_args.parent_dir),model_args.eigenprojection + "_" + model_args.vae_input_type + "_vocals")
        else:
            projection_dir = os.path.join(os.path.dirname(data_training_args.parent_dir),model_args.eigenprojection + "_" + model_args.vae_input_type)
        os.makedirs(projection_dir, exist_ok=True)

    "Make sure to obtain all the samples in the dataset"
    assert config.max_frames_per_batch == "all"

    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'config' not in f]
    checkpoint_files.append('epoch_-01')
    # Sort the training dirs by epoch number
    checkpoint_files.sort(key=extract_epoch)

    if data_training_args.epoch_range_to_evaluate is None:
        data_training_args.epoch_range_to_evaluate = checkpoint_files
    else:
        if len(data_training_args.epoch_range_to_evaluate) == 2:
            if data_training_args.epoch_range_to_evaluate[1] == -1:
                data_training_args.epoch_range_to_evaluate = checkpoint_files[data_training_args.epoch_range_to_evaluate[0]:]   
            else:
                data_training_args.epoch_range_to_evaluate = checkpoint_files[data_training_args.epoch_range_to_evaluate[0]:data_training_args.epoch_range_to_evaluate[1]]  
        elif len(data_training_args.epoch_range_to_evaluate) == 1:
            data_training_args.epoch_range_to_evaluate = [checkpoint_files[data_training_args.epoch_range_to_evaluate[0]]]
        else:
            raise ValueError("epoch_range_to_evaluate should be a list of 1 or 2 integers, or None. Please check your config file.")
    "Below this point we need to iterate across checkpoints"
    for ckp in data_training_args.epoch_range_to_evaluate:
        
        print(f"Loading model from checkpoint directory: {checkpoint_dir}")
        print(f"Processing checkpoint {ckp}...")

        "initialize random model and load pretrained weights"
        if model_args.vae_type == "VAE_1D": 
            representation_function = VAE_1D(z_dim = model_args.vae_z_dim, 
                    proj_intermediate_dim=model_args.vae_proj_intermediate_dim, 
                    conv_dim=model_args.vae_conv_dim, 
                    treat_as_sequence=False,
                    kernel_sizes=model_args.vae_kernel_sizes,
                    strides=model_args.vae_strides, 
                    in_size = max_length, 
                    norm_type = model_args.vae_norm_type,
                    hidden_dim=model_args.vae_hidden_dim,  
                    beta = model_args.vae_beta,
                    warmup_steps=5000,
                )
        elif model_args.vae_type == "VAE_1D_FC":
            if model_args.vae_input_type == "mel":
                reduction_components = 20 #for ica,pca only
                in_size = int(model_args.n_mels_vae)#np.ceil((config.receptive_field/config.stride)))
            elif model_args.vae_input_type == "mel_ocs":
                reduction_components = 30 #for ica,pca only
                in_size = int(model_args.n_mels_vae)#np.ceil((config.receptive_field/config.stride)))
            elif model_args.vae_input_type == "mel_all":
                reduction_components = 35 #for ica,pca only
                in_size = int(model_args.n_mels_vae)#np.ceil((config.receptive_field/config.stride)))
            elif model_args.vae_input_type == "waveform":
                reduction_components = 50 #for ica,pca only
                in_size = int(config.receptive_field*config.fs)
            elif model_args.vae_input_type == "waveform_ocs":
                reduction_components = 65 #for ica,pca only
                in_size = int(config.receptive_field*config.fs)
            elif model_args.vae_input_type == "waveform_all":
                reduction_components = 70 #for ica,pca only
                in_size = int(config.receptive_field*config.fs)
            representation_function = VAE_1D_FC(z_dim=model_args.vae_z_dim,
                    hidden_dims=model_args.vae_fc_dims,
                    kernel_sizes=model_args.vae_kernel_sizes,
                    treat_as_sequence=False,
                    strides=model_args.vae_strides, 
                    in_size=in_size,
                    norm_type = model_args.vae_norm_type,
                    beta=model_args.vae_beta,
                    warmup_steps=5000,
                    kl_annealing = model_args.kl_annealing
                )
            
        if model_args.eigenprojection == 'ica':
            eigenprojection_function = FastICA(n_components=reduction_components,random_state=0,whiten='unit-variance')
        elif model_args.eigenprojection == 'pca':
            eigenprojection_function = PCA(n_components=reduction_components,random_state=0)
        elif model_args.eigenprojection == 'kpca-rbf':
            eigenprojection_function = KernelPCA(n_components=reduction_components, kernel='rbf', gamma=0.1, random_state=0)
        elif model_args.eigenprojection == 'kpca-poly':
            eigenprojection_function = KernelPCA(n_components=reduction_components, kernel='poly', gamma=0.1, random_state=0)
        elif model_args.eigenprojection == 'kpca-sigmoid':
            eigenprojection_function = KernelPCA(n_components=reduction_components, kernel='sigmoid', gamma=0.1, random_state=0)
        elif model_args.eigenprojection is None:
            pass

        if ckp != 'epoch_-01' and not model_args.raw_mels:
            "No weights, random initialization"

            pretrained_model_file = os.path.join(checkpoint_dir,ckp,"model.safetensors")
        
            weights = load_file(pretrained_model_file)

            representation_function.load_state_dict(weights, strict=False)
        
        
        representation_function.eval()
        for param in representation_function.parameters():
            param.requires_grad = False

        "data collator, optimizer and scheduler"
        mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
        mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

        data_collator = DataCollatorForVAE1DLatentPostAnalysis(
            model=representation_function,
            model_name=model_args.vae_type,
            feature_extractor=feature_extractor,
            dataset_name = data_training_args.dataset_name,
            pad_to_multiple_of=data_training_args.pad_to_multiple_of,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length,
        )
        if data_training_args.dataset_name == "VOC_ALS":
            eval_dataset = concatenate_datasets([vectorized_datasets["train"], vectorized_datasets["validation"],vectorized_datasets["test"],vectorized_datasets["dev"]])                                                    
            eval_dataloader = DataLoader(
                eval_dataset, 
                shuffle=True,
                collate_fn=data_collator, 
                batch_size=data_training_args.per_device_train_batch_size
            )
        elif data_training_args.dataset_name == "iemocap":
            eval_dataset = concatenate_datasets([vectorized_datasets["train"], vectorized_datasets["validation"],vectorized_datasets["test"]])                                                    
            eval_dataloader = DataLoader(
                eval_dataset, 
                shuffle=True,
                collate_fn=data_collator, 
                batch_size=data_training_args.per_device_train_batch_size
            )

        else:
            train_dataloader = DataLoader(
                vectorized_datasets["train"],
                shuffle=False,
                collate_fn=data_collator,
                batch_size=data_training_args.per_device_train_batch_size,
            )
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

        if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            "Evaluates on a single set"
            representation_function, eval_dataloader = accelerator.prepare(
                representation_function, eval_dataloader
            )
        else:
            "Prepare everything with HF accelerator"
            representation_function, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
                representation_function, train_dataloader, eval_dataloader, test_dataloader
            )

        "Only for the cases of ICA, PCA, kPCA, get the training data"
        if model_args.raw_mels and model_args.eigenprojection is not None and data_training_args.dataset_name not in ["VOC_ALS", "iemocap"]:
            if os.path.exists(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib')):
                print(f"Loading the fitted {model_args.eigenprojection}_{model_args.vae_input_type} model from memory")
                start_time = time.time()
                eigenprojection_function = joblib.load(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib'))
            else:
                "Get the training data"
                with torch.no_grad():
                    for step, batch in enumerate(train_dataloader):
                        batch_size = batch["input_values"].shape[0]
                        mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
                        sub_attention_mask = batch.pop("sub_attention_mask", None)
                        attention_mask = batch["attention_mask"].bool()
                        overlap_mask_batch = batch.pop("overlap_mask", None)
                        if hasattr(batch,"reconstruction_NRMSEs"):
                            batch.pop("reconstruction_NRMSEs", None)
                        if hasattr(batch,"reconstruction_NRMSE_seq"):
                            batch.pop("reconstruction_NRMSE_seq", None)
                        if hasattr(batch,"correlograms"):
                            batch.pop("correlograms", None)
                        if hasattr(batch,"correlogram_seq"):
                            batch.pop("correlogram_seq", None)
                        batch["global_step"] = 0
                        assert overlap_mask_batch != None if data_training_args.dataset_name in ["timit"] else True
                        if overlap_mask_batch is None or not data_training_args.discard_label_overlaps:
                            overlap_mask_batch = torch.zeros_like(sub_attention_mask).astype(torch.bool)
                        else:
                            "Frames corresponding to padding are set as True in the overlap and discarded"
                            padded = sub_attention_mask.sum(dim = -1)
                            for b in range(batch_size):
                                overlap_mask_batch[b,padded[b]:] = 1
                            overlap_mask_batch = overlap_mask_batch.bool()
                        
                        if data_training_args.dataset_name == "sim_vowels":
                            batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)                
                            if hasattr(batch,"vowel_labels"):
                                vowel_labels_batch = batch.pop("vowel_labels")
                            if hasattr(batch,"speaker_vt_factor"):
                                speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                            
                            vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                        elif data_training_args.dataset_name in ["timit"]:
                            batch["mask_time_indices"] = sub_attention_mask.clone()
                            phonemes39_batch = batch.pop("phonemes39", None)
                            phonemes48_batch = batch.pop("phonemes48", None)                                
                            phonemes39_batch = phonemes39_batch[~overlap_mask_batch]
                            phonemes48_batch = phonemes48_batch[~overlap_mask_batch]

                            batch.pop("start_phonemes", None)
                            batch.pop("stop_phonemes", None)
                            speaker_id_batch = list(batch.pop("speaker_id", None))
                        

                        if model_args.vae_type == "VAE_1D_FC":
                            if model_args.vae_input_type == "waveform_ocs":
                                if model_args.raw_mels:
                                    batch["input_values"] = batch["input_values"][:,1:,:,:].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                                    "Reshape"
                                else:
                                    raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_ocs input type")
                            elif model_args.vae_input_type == "waveform_all":
                                if model_args.raw_mels:
                                    batch["input_values"] = batch["input_values"].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                                else:
                                    raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_all input type")
                            elif model_args.vae_input_type == "mel_ocs":
                                if model_args.raw_mels:
                                    new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1]-1,batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                    for i in range(1,config.NoC+1):
                                        new_input_values[:,i-1,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                    batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                                else:
                                    raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_ocs input type")
                            elif model_args.vae_input_type == "mel_all":
                                if model_args.raw_mels:
                                    new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1],batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                    for i in range(0,config.NoC+1):
                                        new_input_values[:,i,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                    batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                                else:
                                    raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_all input type")
                            elif model_args.vae_input_type == "waveform":
                                batch["input_values"] = batch["input_values"][:,0,:,:]
                            elif model_args.vae_input_type == "mel":
                                batch["input_values"], _ = extract_mel_spectrogram(batch["input_values"][:,0,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)

                            batch["attention_mask"] = sub_attention_mask
                        
                        if model_args.raw_mels and not model_args.vae_type == "VAE_1D_FC":
                            if model_args.vae_input_type == "mel":
                                batch["input_values"] = extract_mel_spectrogram(batch["input_values"],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs)+1, normalize=model_args.mel_norm_vae)
                            elif model_args.vae_input_type == "waveform":
                                batch["input_values"] = batch["input_values"][:,0,:,:]
                            batch["attention_mask"] = sub_attention_mask

                        if not model_args.raw_mels:
                            outputs = representation_function(**batch)
                        else:                    
                            outputs = [batch["input_values"]]
                        del batch

                        if "vowels" in data_training_args.dataset_name:
                            if step == 0:
                                vowel_labels = torch.cat([torch.tensor(v) for v in vowel_labels_batch]) 
                            else:
                                vowel_labels = torch.cat((vowel_labels,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                            if step == 0:
                                speaker_vt_factor_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)]) 
                                speaker_vt_factor_seq = speaker_vt_factor_batch.clone() 
                            else:
                                speaker_vt_factor_frame = torch.cat((speaker_vt_factor_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                                speaker_vt_factor_seq = torch.cat((speaker_vt_factor_seq,speaker_vt_factor_batch),dim = 0) 

                        elif data_training_args.dataset_name in ["timit"]:                            
                            if step == 0:
                                phonemes39 = phonemes39_batch.clone()
                                phonemes48 = phonemes48_batch.clone()
                            else:
                                phonemes39 = torch.cat((phonemes39,phonemes39_batch))
                                phonemes48 = torch.cat((phonemes48,phonemes48_batch))
                            if step == 0:
                                speaker_id_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) 
                                speaker_id_seq = torch.stack(speaker_id_batch) 
                            else:
                                speaker_id_frame = torch.cat((speaker_id_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                                speaker_id_seq = torch.cat((speaker_id_seq,torch.stack(speaker_id_batch)),dim = 0) 

                        "Gather latents for evaluations"
                        if data_training_args.dataset_name == "sim_vowels":
                            overlap_mask_batch = overlap_mask_batch[sub_attention_mask].view(batch_size,-1)
                        z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                        if step == 0:                    
                            z_mean = z_mean_batch.detach().cpu()
                        else:
                            z_mean = torch.cat((z_mean,z_mean_batch.detach().cpu()),dim = 0)
                    
                "Fit projection on train set"
                if 'kpca-' in model_args.eigenprojection: 
                    "Fit the projection on a subset of the training set"
                    rng = np.random.default_rng(seed=42) 
                    indices = rng.choice(z_mean.shape[0], size=int(z_mean.shape[0]*0.01), replace=False)
                    eigenprojection_function.fit(z_mean[indices])
                elif (model_args.vae_input_type == "waveform_all" or model_args.vae_input_type == "waveform_ocs"):
                    rng = np.random.default_rng(seed=42) 
                    indices = rng.choice(z_mean.shape[0], size=int(z_mean.shape[0]*0.2), replace=False)
                    eigenprojection_function.fit(z_mean[indices])
                else:
                    eigenprojection_function.fit(z_mean)
                "Save the fitted model"
                joblib.dump(eigenprojection_function, os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib'))

        elif data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            if model_args.raw_mels and model_args.eigenprojection is not None and os.path.exists(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib')):
                print(f"Loading the fitted {model_args.eigenprojection}_{model_args.vae_input_type} model from memory")
                start_time = time.time()
                eigenprojection_function = joblib.load(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib'))     

            "Get the data for VOC_ALS or iemocap"
            with torch.no_grad():
                start_time = time.time()
                for step, batch in enumerate(eval_dataloader):
                    batch_size = batch["input_values"].shape[0]
                    mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
                    sub_attention_mask = batch.pop("sub_attention_mask", None)
                    attention_mask = batch["attention_mask"].bool()
                    overlap_mask_batch = batch.pop("overlap_mask", None)
                    if hasattr(batch,"reconstruction_NRMSEs"):
                        batch.pop("reconstruction_NRMSEs", None)
                    if hasattr(batch,"reconstruction_NRMSE_seq"):
                        batch.pop("reconstruction_NRMSE_seq", None)
                    if hasattr(batch,"correlograms"):
                        batch.pop("correlograms", None)
                    if hasattr(batch,"correlogram_seq"):
                        batch.pop("correlogram_seq", None)
                    batch["global_step"] = 0
                    assert overlap_mask_batch != None if data_training_args.dataset_name in ["iemocap"] else True
                    if (overlap_mask_batch is None or not data_training_args.discard_label_overlaps) and not "VOC_ALS" in data_training_args.dataset_name:
                        overlap_mask_batch = torch.zeros_like(sub_attention_mask).astype(torch.bool)
                    else:
                        if "VOC_ALS" in data_training_args.dataset_name:
                            overlap_mask_batch = torch.zeros_like(sub_attention_mask,dtype=torch.bool)
                        "Frames corresponding to padding are set as True in the overlap and discarded"
                        padded = sub_attention_mask.sum(dim = -1)
                        for b in range(batch_size):
                            overlap_mask_batch[b,padded[b]:] = 1
                        overlap_mask_batch = overlap_mask_batch.bool()
                    
                    if data_training_args.dataset_name in ["iemocap"]:
                        batch["mask_time_indices"] = sub_attention_mask.clone()
                        phonemes_batch = batch.pop("phonemes", None)
                        emotion_batch = list(batch.pop("emotion", None))
                        phonemes_batch = phonemes_batch[~overlap_mask_batch]
                        batch.pop("start_phonemes", None)
                        batch.pop("stop_phonemes", None)
                        speaker_id_batch = list(batch.pop("speaker_id", None))
                    
                    elif data_training_args.dataset_name == "VOC_ALS":
                        batch["mask_time_indices"] = sub_attention_mask.clone()
                        alsfrs_total_batch = list(batch.pop("alsfrs_total", None))
                        disease_duration_batch = list(batch.pop("disease_duration", None))
                        king_stage_batch = list(batch.pop("king_stage", None))
                        alsfrs_speech_batch = list(batch.pop("alsfrs_speech", None))
                        cantagallo_batch = list(batch.pop("cantagallo", None))
                        phonemes_batch = list(batch.pop("phonemes", None))
                        speaker_id_batch = list(batch.pop("speaker_id", None))
                        group_batch = list(batch.pop("group", None))

                    if model_args.vae_type == "VAE_1D_FC":
                        if model_args.vae_input_type == "waveform_ocs":
                            if model_args.raw_mels:
                                batch["input_values"] = batch["input_values"][:,1:,:,:].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                                "Reshape"
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_ocs input type")
                        elif model_args.vae_input_type == "waveform_all":
                            if model_args.raw_mels:
                                batch["input_values"] = batch["input_values"].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_all input type")
                        elif model_args.vae_input_type == "mel_ocs":
                            if model_args.raw_mels:
                                new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1]-1,batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                for i in range(1,config.NoC+1):
                                    new_input_values[:,i-1,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_ocs input type")
                        elif model_args.vae_input_type == "mel_all":
                            if model_args.raw_mels:
                                new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1],batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                for i in range(0,config.NoC+1):
                                    new_input_values[:,i,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_all input type")
                        elif model_args.vae_input_type == "waveform":
                            batch["input_values"] = batch["input_values"][:,0,:,:]
                        elif model_args.vae_input_type == "mel":
                            batch["input_values"], _ = extract_mel_spectrogram(batch["input_values"][:,0,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)

                        batch["attention_mask"] = sub_attention_mask
                    
                    if model_args.raw_mels and not model_args.vae_type == "VAE_1D_FC":
                        if model_args.vae_input_type == "mel":
                            batch["input_values"], _ = extract_mel_spectrogram(batch["input_values"],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs)+1, normalize=model_args.mel_norm_vae)
                        elif model_args.vae_input_type == "waveform":
                            batch["input_values"], _ = batch["input_values"][:,0,:,:]
                        batch["attention_mask"] = sub_attention_mask

                    if not model_args.raw_mels:
                        outputs = representation_function(**batch)
                    else:                    
                        outputs = [batch["input_values"]]
                    del batch

                    if "vowels" in data_training_args.dataset_name:
                        if step == 0:
                            vowel_labels = torch.cat([torch.tensor(v) for v in vowel_labels_batch]) 
                        else:
                            vowel_labels = torch.cat((vowel_labels,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                        if step == 0:
                            speaker_vt_factor_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])
                            speaker_vt_factor_seq = speaker_vt_factor_batch.clone()
                        else:
                            speaker_vt_factor_frame = torch.cat((speaker_vt_factor_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                            speaker_vt_factor_seq = torch.cat((speaker_vt_factor_seq,speaker_vt_factor_batch),dim = 0)

                    elif data_training_args.dataset_name in ["iemocap"]:
                        if step == 0:
                            phonemes = phonemes_batch.clone()
                            emotion_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)]) 
                            emotion_seq = torch.stack(emotion_batch) 
                        else:
                            phonemes = torch.cat((phonemes,phonemes_batch))
                            emotion_frame = torch.cat((emotion_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)])),dim = 0)
                            emotion_seq = torch.cat((emotion_seq,torch.stack(emotion_batch)),dim = 0) 

                        if step == 0:
                            speaker_id_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) 
                            speaker_id_seq = torch.stack(speaker_id_batch)
                        else:
                            speaker_id_frame = torch.cat((speaker_id_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            speaker_id_seq = torch.cat((speaker_id_seq,torch.stack(speaker_id_batch)),dim = 0) 

                    elif "VOC_ALS" in data_training_args.dataset_name:
                        if step == 0:
                            alsfrs_total_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_total_batch)]) 
                            alsfrs_total_seq = torch.stack(alsfrs_total_batch)
                            alsfrs_speech_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_speech_batch)]) 
                            alsfrs_speech_seq = torch.stack(alsfrs_speech_batch)
                            disease_duration_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(disease_duration_batch)]) 
                            disease_duration_seq = torch.stack(disease_duration_batch)
                            king_stage_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(king_stage_batch)]) 
                            king_stage_seq = torch.stack(king_stage_batch)
                            cantagallo_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(cantagallo_batch)]) 
                            cantagallo_seq = torch.stack(cantagallo_batch)
                            phonemes_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(phonemes_batch)])
                            phonemes_seq = torch.stack(phonemes_batch)
                            speaker_id_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])
                            speaker_id_seq = torch.stack(speaker_id_batch)
                            group_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(group_batch)])
                            group_seq = torch.stack(group_batch)
                        else:
                            alsfrs_total_frame = torch.cat((alsfrs_total_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_total_batch)])),dim = 0)
                            alsfrs_total_seq = torch.cat((alsfrs_total_seq,torch.stack(alsfrs_total_batch)),dim = 0) 
                            alsfrs_speech_frame = torch.cat((alsfrs_speech_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_speech_batch)])),dim = 0)
                            alsfrs_speech_seq = torch.cat((alsfrs_speech_seq,torch.stack(alsfrs_speech_batch)),dim = 0) 
                            disease_duration_frame = torch.cat((disease_duration_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(disease_duration_batch)])),dim = 0)
                            disease_duration_seq = torch.cat((disease_duration_seq,torch.stack(disease_duration_batch)),dim = 0)
                            king_stage_frame = torch.cat((king_stage_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(king_stage_batch)])),dim = 0)
                            king_stage_seq = torch.cat((king_stage_seq,torch.stack(king_stage_batch)),dim = 0)
                            cantagallo_frame = torch.cat((cantagallo_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(cantagallo_batch)])),dim = 0)
                            cantagallo_seq = torch.cat((cantagallo_seq,torch.stack(cantagallo_batch)),dim = 0)
                            phonemes_frame = torch.cat((phonemes_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(phonemes_batch)])),dim = 0)
                            phonemes_seq = torch.cat((phonemes_seq,torch.stack(phonemes_batch)),dim = 0)
                            speaker_id_frame = torch.cat((speaker_id_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            speaker_id_seq = torch.cat((speaker_id_seq,torch.stack(speaker_id_batch)),dim = 0)
                            group_frame = torch.cat((group_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(group_batch)])),dim = 0)
                            group_seq = torch.cat((group_seq,torch.stack(group_batch)),dim = 0)

                    "Gather latents for evaluations"
                    z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                    if step == 0:                    
                        z_mean = z_mean_batch.detach().cpu()
                    else:
                        z_mean = torch.cat((z_mean,z_mean_batch.detach().cpu()),dim = 0)
                    
            if model_args.raw_mels and model_args.eigenprojection is not None:
                
                if not os.path.exists(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib')):
                    "Fit projection on train set"
                    if 'kpca-' in model_args.eigenprojection: 
                        "Fit the projection on a subset of the training set"
                        rng = np.random.default_rng(seed=42) 
                        indices = rng.choice(z_mean.shape[0], size=int(z_mean.shape[0]*0.01), replace=False)
                        eigenprojection_function.fit(z_mean[indices])
                    elif (model_args.vae_input_type == "waveform_all" or model_args.vae_input_type == "waveform_ocs"):
                        rng = np.random.default_rng(seed=42) 
                        indices = rng.choice(z_mean.shape[0], size=int(z_mean.shape[0]*0.2), replace=False)
                        eigenprojection_function.fit(z_mean[indices])
                    else:
                        eigenprojection_function.fit(z_mean)
                    "Save the fitted model"
                    joblib.dump(eigenprojection_function, os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib'))
            
                "Transform z_mean"
                z_mean = torch.tensor(eigenprojection_function.transform(z_mean))

        else:
            pass
        
        if data_training_args.dataset_name not in ["VOC_ALS", "iemocap"]:

            "Measure total loading time"
            start_time = time.time()
            "Get the representations"
            with torch.no_grad():
                "Eval set for loop"
                for step, batch in enumerate(eval_dataloader):
                    batch_size = batch["input_values"].shape[0]
                    mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
                    sub_attention_mask = batch.pop("sub_attention_mask", None)
                    attention_mask = batch["attention_mask"].bool()
                    overlap_mask_batch = batch.pop("overlap_mask", None)
                    if hasattr(batch,"reconstruction_NRMSEs"):
                        batch.pop("reconstruction_NRMSEs", None)
                    if hasattr(batch,"reconstruction_NRMSE_seq"):
                        batch.pop("reconstruction_NRMSE_seq", None)
                    if hasattr(batch,"correlograms"):
                        batch.pop("correlograms", None)
                    if hasattr(batch,"correlogram_seq"):
                        batch.pop("correlogram_seq", None)
                    batch["global_step"] = 0
                    assert overlap_mask_batch != None if data_training_args.dataset_name in ["timit"] else True
                    if overlap_mask_batch is None or not data_training_args.discard_label_overlaps:
                        overlap_mask_batch = torch.zeros_like(sub_attention_mask).astype(torch.bool)
                    else:
                        "Frames corresponding to padding are set as True in the overlap and discarded"
                        padded = sub_attention_mask.sum(dim = -1)
                        for b in range(batch_size):
                            overlap_mask_batch[b,padded[b]:] = 1
                        overlap_mask_batch = overlap_mask_batch.bool()
                    
                    if data_training_args.dataset_name == "sim_vowels":
                        batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)                
                        if hasattr(batch,"vowel_labels"):
                            vowel_labels_batch = batch.pop("vowel_labels")
                        if hasattr(batch,"speaker_vt_factor"):
                            speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                        
                        vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                    elif data_training_args.dataset_name in ["timit"]:
                        batch["mask_time_indices"] = sub_attention_mask.clone()
                        phonemes39_batch = batch.pop("phonemes39", None)
                        phonemes48_batch = batch.pop("phonemes48", None)
                        phonemes39_batch = phonemes39_batch[~overlap_mask_batch]
                        phonemes48_batch = phonemes48_batch[~overlap_mask_batch]
                            
                        batch.pop("start_phonemes", None)
                        batch.pop("stop_phonemes", None)
                        speaker_id_batch = list(batch.pop("speaker_id", None))
                    
                    if model_args.vae_type == "VAE_1D_FC":
                        if model_args.vae_input_type == "waveform_ocs":
                            if model_args.raw_mels:
                                batch["input_values"] = batch["input_values"][:,1:,:,:].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                                "Reshape"
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_ocs input type")
                        elif model_args.vae_input_type == "waveform_all":
                            if model_args.raw_mels:
                                batch["input_values"] = batch["input_values"].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_all input type")
                        elif model_args.vae_input_type == "mel_ocs":
                            if model_args.raw_mels:
                                new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1]-1,batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                for i in range(1,config.NoC+1):
                                    new_input_values[:,i-1,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_ocs input type")
                        elif model_args.vae_input_type == "mel_all":
                            if model_args.raw_mels:
                                new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1],batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                for i in range(0,config.NoC+1):
                                    new_input_values[:,i,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_all input type")
                        elif model_args.vae_input_type == "mel":
                            batch["input_values"], _ = extract_mel_spectrogram(batch["input_values"][:,0,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                        elif model_args.vae_input_type == "waveform":
                            batch["input_values"] = batch["input_values"][:,0,:,:]
                        batch["attention_mask"] = sub_attention_mask
                    
                    if model_args.raw_mels and not model_args.vae_type == "VAE_1D_FC":
                        if model_args.vae_input_type == "mel":
                            batch["input_values"] = extract_mel_spectrogram(batch["input_values"],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs), normalize=model_args.mel_norm_vae)
                        elif model_args.vae_input_type == "waveform":
                            batch["input_values"] = batch["input_values"][:,0,:,:]
                        batch["attention_mask"] = sub_attention_mask

                    if not model_args.raw_mels:
                        outputs = representation_function(**batch)
                    else:
                        outputs = [batch["input_values"]]
                        
                    del batch

                    if "vowels" in data_training_args.dataset_name:
                        if step == 0:
                            vowel_labels = torch.cat([torch.tensor(v) for v in vowel_labels_batch]) 
                        else:
                            vowel_labels = torch.cat((vowel_labels,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                        if step == 0:
                            speaker_vt_factor_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])
                            speaker_vt_factor_seq = speaker_vt_factor_batch.clone()
                        else:
                            speaker_vt_factor_frame = torch.cat((speaker_vt_factor_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                            speaker_vt_factor_seq = torch.cat((speaker_vt_factor_seq,speaker_vt_factor_batch),dim = 0)

                    elif "timit" in data_training_args.dataset_name:
                        if step == 0:
                            phonemes39 = phonemes39_batch.clone()
                            phonemes48 = phonemes48_batch.clone()
                        else:
                            phonemes39 = torch.cat((phonemes39,phonemes39_batch))
                            phonemes48 = torch.cat((phonemes48,phonemes48_batch))

                        if step == 0:
                            speaker_id_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch) for _ in used_indices[i]])
                            speaker_id_seq = torch.stack(speaker_id_batch) 
                        else:
                            speaker_id_frame = torch.cat((speaker_id_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            speaker_id_seq = torch.cat((speaker_id_seq,torch.stack(speaker_id_batch)),dim = 0) 

                    "Gather latents for evaluations"
                    if data_training_args.dataset_name == "sim_vowels":
                        overlap_mask_batch = overlap_mask_batch[sub_attention_mask].view(batch_size,-1)
                    z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                    if step == 0:                    
                        z_mean = z_mean_batch.detach().cpu()
                    else:
                        z_mean = torch.cat((z_mean,z_mean_batch.detach().cpu()),dim = 0)

                if model_args.eigenprojection is not None:
                    "Transform dev set"
                    z_mean = torch.tensor(eigenprojection_function.transform(z_mean))

                "Test set for loop"
                for step, batch in enumerate(test_dataloader):
                    x = batch["input_values"]
                    batch_size = batch["input_values"].shape[0]
                    mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
                    sub_attention_mask = batch.pop("sub_attention_mask", None)
                    attention_mask = batch["attention_mask"].bool()
                    overlap_mask_batch = batch.pop("overlap_mask", None)
                    if hasattr(batch,"reconstruction_NRMSEs"):
                        batch.pop("reconstruction_NRMSEs", None)
                    if hasattr(batch,"reconstruction_NRMSE_seq"):
                        batch.pop("reconstruction_NRMSE_seq", None)
                    if hasattr(batch,"correlograms"):
                        batch.pop("correlograms", None)
                    if hasattr(batch,"correlogram_seq"):
                        batch.pop("correlogram_seq", None)
                    batch["global_step"] = 0
                    assert overlap_mask_batch != None if data_training_args.dataset_name in ["timit"] else True
                    if overlap_mask_batch is None or not data_training_args.discard_label_overlaps:
                        overlap_mask_batch = torch.zeros_like(sub_attention_mask).astype(torch.bool)
                    else:
                        "Frames corresponding to padding are set as True in the overlap and discarded"
                        padded = sub_attention_mask.sum(dim = -1)
                        for b in range(batch_size):
                            overlap_mask_batch[b,padded[b]:] = 1
                        overlap_mask_batch = overlap_mask_batch.bool()
                    if data_training_args.dataset_name == "sim_vowels":
                        batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)                
                        if hasattr(batch,"vowel_labels"):
                            vowel_labels_batch = batch.pop("vowel_labels")
                        if hasattr(batch,"speaker_vt_factor"):
                            speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                        vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                    elif data_training_args.dataset_name in ["timit"]:
                        batch["mask_time_indices"] = sub_attention_mask.clone()
                        if data_training_args.dataset_name == "timit":
                            phonemes39_batch = batch.pop("phonemes39", None)
                            phonemes48_batch = batch.pop("phonemes48", None)

                            phonemes39_batch = phonemes39_batch[~overlap_mask_batch]
                            phonemes48_batch = phonemes48_batch[~overlap_mask_batch]

                        batch.pop("start_phonemes", None)
                        batch.pop("stop_phonemes", None)
                        speaker_id_batch = list(batch.pop("speaker_id", None))

                    if model_args.vae_type == "VAE_1D_FC":
                        if model_args.vae_input_type == "waveform_ocs":
                            if model_args.raw_mels:
                                batch["input_values"] = batch["input_values"][:,1:,:,:].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                                "Reshape"
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_ocs input type")
                        elif model_args.vae_input_type == "waveform_all":
                            if model_args.raw_mels:
                                batch["input_values"] = batch["input_values"].transpose(1,2).reshape(batch_size,batch["input_values"].shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with waveform_all input type")
                        elif model_args.vae_input_type == "mel_ocs":
                            if model_args.raw_mels:
                                new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1]-1,batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                for i in range(1,config.NoC+1):
                                    new_input_values[:,i-1,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_ocs input type")
                        elif model_args.vae_input_type == "mel_all":
                            if model_args.raw_mels:
                                new_input_values = torch.zeros((batch["input_values"].shape[0],batch["input_values"].shape[1],batch["input_values"].shape[2],model_args.n_mels_vae),dtype=batch["input_values"].dtype,device=batch["input_values"].device)
                                for i in range(0,config.NoC+1):
                                    new_input_values[:,i,...], _ = extract_mel_spectrogram(batch["input_values"][:,i,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                                batch["input_values"] = new_input_values.transpose(1,2).reshape(batch_size,new_input_values.shape[2],-1)
                            else:
                                raise ValueError("model_args.raw_mels should be True for VAE_1D_FC with mel_all input type")
                        elif model_args.vae_input_type == "waveform":
                            batch["input_values"] = batch["input_values"][:,0,:,:]
                        elif model_args.vae_input_type == "mel":
                            batch["input_values"], _ = extract_mel_spectrogram(batch["input_values"][:,0,:,:],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs) + 1, normalize=model_args.mel_norm_vae)
                        batch["attention_mask"] = sub_attention_mask
        
                    if not model_args.raw_mels:
                        outputs = representation_function(**batch)
                    else:
                        outputs = [batch["input_values"]]
                    del batch

                    if "vowels" in data_training_args.dataset_name:
                        if step == 0:
                            vowel_labels_test = torch.cat([torch.tensor(v) for v in vowel_labels_batch]) 
                        else:
                            vowel_labels_test = torch.cat((vowel_labels_test,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                        if step == 0:
                            speaker_vt_factor_frame_test = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])
                            speaker_vt_factor_seq_test = speaker_vt_factor_batch.clone()
                        else:
                            speaker_vt_factor_frame_test = torch.cat((speaker_vt_factor_frame_test,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                            speaker_vt_factor_seq_test = torch.cat((speaker_vt_factor_seq_test,speaker_vt_factor_batch),dim = 0) 

                    elif "timit" in data_training_args.dataset_name:
                        if "timit" in data_training_args.dataset_name:
                            if step == 0:
                                phonemes39_test = phonemes39_batch.clone()
                                phonemes48_test = phonemes48_batch.clone()
                            else:
                                phonemes39_test = torch.cat((phonemes39_test,phonemes39_batch))
                                phonemes48_test = torch.cat((phonemes48_test,phonemes48_batch))
                                
                        if step == 0:
                            speaker_id_frame_test = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch) for _ in used_indices[i]])
                            speaker_id_seq_test = torch.stack(speaker_id_batch) 
                        else:
                            speaker_id_frame_test = torch.cat((speaker_id_frame_test,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            speaker_id_seq_test = torch.cat((speaker_id_seq_test,torch.stack(speaker_id_batch)),dim = 0)

                    "Gather latents for evaluations"
                    if data_training_args.dataset_name == "sim_vowels":
                        overlap_mask_batch = overlap_mask_batch[sub_attention_mask].view(batch_size,-1)
                    z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                    if step == 0:                    
                        z_mean_test = z_mean_batch.detach().cpu()
                    else:
                        z_mean_test = torch.cat((z_mean_test,z_mean_batch.detach().cpu()),dim = 0)

                if model_args.eigenprojection is not None:
                    "Transform test set"
                    z_mean_test = torch.tensor(eigenprojection_function.transform(z_mean_test))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total loading time: {elapsed_time: .4f} seconds")

        "Now use train/val representations to get the evaluation metrics"
        "1. Projection evaluation"
        "2. Components evaluation separately (1st,2nd,3rd)"
        
        "Linear/non-linear classification"
        #Create combined labels - Speaker-vowel
        #Comparisons inside speaker / inside vowel
        if data_training_args.classify:
            if "vowels" in data_training_args.dataset_name:        
                if "vowel" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:        
                    "Check vowels in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = z_mean_test,
                        y = vowel_labels, y_test = vowel_labels_test,
                        checkpoint = ckp, latent_type="z",target = "vowel"
                    )
                if "speaker" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                    "Check speakers in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = z_mean_test,
                        y = speaker_vt_factor_frame, y_test = speaker_vt_factor_frame_test,
                        checkpoint = ckp, latent_type="z",target = "speaker_frame"
                    )

            elif "timit" in data_training_args.dataset_name:
                if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                    "Check phoneme frame accuracy in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = z_mean_test,
                        y = phonemes48, y_test = phonemes48_test,
                        checkpoint = ckp, latent_type="z",target = "phoneme48" 
                    )
                if "speaker" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                    "Check speakers in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = z_mean_test,
                        y = speaker_id_frame, y_test = speaker_id_frame_test,
                        checkpoint = ckp, latent_type="z",target = "speaker_frame"
                    )

            elif "iemocap" in data_training_args.dataset_name:
                if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                    "Check phoneme accuracy in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = phonemes, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "phoneme" 
                    )
                if "speaker" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                    "Check speaker identification in z" 
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = speaker_id_frame, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "speaker_frame" 
                    )
                if "emotion" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                    "Check emotion recognition in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = torch.stack((emotion_frame,speaker_id_frame), dim = 1), y_test = None,
                        checkpoint = ckp, latent_type="z",target = ["cat_emotion_frame", "speaker_frame"]
                    )

            elif "VOC_ALS" in data_training_args.dataset_name:
                if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                    "Check phoneme accuracy in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = phonemes_frame, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "phoneme_frame" 
                    )
                if "speaker" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                    "Check speaker identification in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = speaker_id_frame, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "speaker_frame" 
                )
                if "group" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                    "Check group in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = group_frame, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "group_frame"
                    )
                if "kings_stage" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                    "Check King's staging in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = king_stage_frame, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "kings_stage_frame"
                    )
                if "disease_duration" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                    "Check Disease Duration staging in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = disease_duration_frame, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "disease_duration_frame"
                    )
                if "alsfrs_total" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                    "Check ALSFRS-total staging in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = alsfrs_total_frame, y_test = None,
                        checkpoint = ckp, latent_type="z",target = "alsfrs_total_frame"
                    )
                if "alsfrs_speech" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                    "Check ALSFRS-speech subitem staging in z"
                    prediction_eval(data_training_args,config,
                        X = z_mean, X_test = None,
                        y = alsfrs_speech_frame, y_test = None,
                        checkpoint = ckp, latent_type="OCs_joint_emb",target = "alsfrs_speech_frame"
                    )

        "Disentanglement evaluation"
        if data_training_args.measure_disentanglement:
            if "vowels" in data_training_args.dataset_name:
                y_frame_test = torch.cat([vowel_labels_test.reshape(-1,1),speaker_vt_factor_frame_test.reshape(-1,1)],dim = 1)
                y_frame_train = torch.cat([vowel_labels.reshape(-1,1),speaker_vt_factor_frame.reshape(-1,1)],dim = 1)
                y_frame_test = pd.DataFrame(y_frame_test.cpu().numpy(),columns=["vowel","speaker_frame"])
                y_frame_train = pd.DataFrame(y_frame_train.cpu().numpy(),columns=["vowel","speaker_frame"])
                y_seq_test = pd.DataFrame(speaker_vt_factor_seq_test.cpu().numpy(),columns=["speaker_seq"])
                y_seq_train = pd.DataFrame(speaker_vt_factor_seq.cpu().numpy(),columns=["speaker_seq"])

            elif "timit" in data_training_args.dataset_name:
                speaker_id_frame_test = speaker_id_frame_test.to(phonemes39_test.device)
                speaker_id_frame = speaker_id_frame.to(phonemes39_test.device)
                y_frame_test = torch.cat([phonemes39_test.reshape(-1,1),speaker_id_frame_test.reshape(-1,1)],dim = 1)
                y_frame_train = torch.cat([phonemes39.reshape(-1,1),speaker_id_frame.reshape(-1,1)],dim = 1)
                y_frame_test = pd.DataFrame(y_frame_test.cpu().numpy(),columns=["phoneme","speaker_frame"])
                y_frame_train = pd.DataFrame(y_frame_train.cpu().numpy(),columns=["phoneme","speaker_frame"])
                y_seq_test = pd.DataFrame(speaker_id_seq_test.cpu().numpy(),columns=["speaker_seq"])
                y_seq_train = pd.DataFrame(speaker_id_seq.cpu().numpy(),columns=["speaker_seq"])
            
            elif "iemocap" in data_training_args.dataset_name:
                if phonemes.device != emotion_frame.device:
                    phonemes = phonemes.to(emotion_frame.device)
                y_frame_train = torch.cat([phonemes.reshape(-1,1),speaker_id_frame.reshape(-1,1),emotion_frame.reshape(-1,1)],dim = 1)
                y_frame_train = pd.DataFrame(y_frame_train.cpu().numpy(),columns=["phoneme","speaker_frame","cat_emotion_frame"])
                y_seq_train = torch.cat([speaker_id_seq.reshape(-1,1),emotion_seq.reshape(-1,1)],dim = 1)
                y_seq_train = pd.DataFrame(y_seq_train.cpu().numpy(),columns=["speaker_seq","cat_emotion_seq"])

            elif "VOC_ALS" in data_training_args.dataset_name:
                "Create multiple mu latents for VOC_ALS to make different comparisons - Multiple variables"
                "King's stage - Phoneme - Speaker"
                y_frame_train_king = torch.cat([king_stage_frame.reshape(-1,1),phonemes_frame.reshape(-1,1),speaker_id_frame.reshape(-1,1)],dim = 1)
                y_seq_train_king = torch.cat([king_stage_seq.reshape(-1,1),phonemes_seq.reshape(-1,1),speaker_id_seq.reshape(-1,1)],dim = 1)
                y_frame_train_king = pd.DataFrame(y_frame_train_king.cpu().numpy(),columns=["king_stage_frame","phoneme_frame","speaker_frame"])
                y_seq_train_king = pd.DataFrame(y_seq_train_king.cpu().numpy(),columns=["king_stage_seq","phoneme_seq","speaker_seq"])

                #"Disease Duration - Phoneme - Speaker"
                #y_frame_train_dis = torch.cat([disease_duration_frame.reshape(-1,1),phonemes_frame.reshape(-1,1),speaker_id_frame.reshape(-1,1)],dim = 1)
                #y_seq_train_dis = torch.cat([disease_duration_seq.reshape(-1,1),phonemes_seq.reshape(-1,1),speaker_id_seq.reshape(-1,1)],dim = 1)
                #y_frame_train_dis = pd.DataFrame(y_frame_train_dis.cpu().numpy(),columns=["disease_duration_frame","phoneme_frame","speaker_frame"])
                #y_seq_train_dis = pd.DataFrame(y_seq_train_dis.cpu().numpy(),columns=["disease_duration_seq","phoneme_seq","speaker_seq"])


            if "VOC_ALS" in data_training_args.dataset_name:
                "Check vowels/speakers disentanglement in z - With King's Clinical Staging"
                compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                    latent_type="z", mu_train = z_mean, y_train = y_frame_train_king, 
                    mu_test = None, y_test = None, target = ["king_stage_frame","phoneme_frame","speaker_frame"]
                )

                "Check vowels/speakers disentanglement in z - With Disease Duration"
                compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                    latent_type="z", mu_train = z_mean, y_train = y_frame_train_dis, 
                    mu_test = None, y_test = None, target = ["disease_duration_frame","phoneme_frame","speaker_frame"]
                )
            elif "iemocap" in data_training_args.dataset_name:
                "Check phonemes/speakers/emotions disentanglement in z"
                compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                    latent_type="z", mu_train = z_mean, y_train = y_frame_train, 
                    mu_test = None, y_test = None, target = ["phoneme","speaker_frame","cat_emotion_frame"]
                )
            else:    
                "Check vowels/speakers disentanglement in z"
                compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                    latent_type="z", mu_train = z_mean, y_train = y_frame_train, 
                    mu_test = z_mean_test, y_test = y_frame_test, target = ["vowel","speaker_frame"] if "vowels" in data_training_args.dataset_name else ["phoneme","speaker_frame"]
                )
            

        if model_args.raw_mels:
            print("Raw mel filterbank features evaluation finished")
            break
            

if __name__ == "__main__":
    main()