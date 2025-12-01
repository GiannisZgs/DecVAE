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

"""This script visualizes latent representations of VAE-based or PCA/ICA/kPCA models through the latent_analysis.visualize utility.
Visualization supports generative factors embedding (colored by labels) in 2D and 3D space.
This script selects a number of instances of a class to avoid cluttered visualizations i.e. in cases where speakers are > 20, we select a few speakers only to visualize.
Decomposition is not supported here so inputs have to be preprocessed from another script for this script to work. 
Subgroups of generative factors are supported for visualization; e.g. in TIMIT we gather consonants and vowels that are subgroups of phonemes.
"""

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
from args_configs import ModelArgumentsPost, DataTrainingArgumentsPost, DecompositionArguments, TrainingObjectiveArguments, VisualizationsArguments
from utils.misc import parse_args, debugger_is_active, extract_epoch
from feature_extraction import extract_mel_spectrogram

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from sklearn.decomposition import PCA, FastICA, KernelPCA 
import joblib
from safetensors.torch import load_file
import numpy as np

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from torch.utils.data.dataloader import DataLoader
import json
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from latent_analysis_utils import visualize
import warnings
import time

warnings.simplefilter("ignore")
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_USE_CUDA_DSA"] = "1"
#os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

JSON_FILE_NAME_MANUAL = "config_files/VAEs/sim_vowels/latent_visualizations/config_vae1d_latent_frames_visualization_vowels.json" #for debugging purposes only

logger = get_logger(__name__)


def main():
    "Parse the arguments"       
    parser = HfArgumentParser((ModelArgumentsPost, TrainingObjectiveArguments, DecompositionArguments, DataTrainingArgumentsPost))
    if debugger_is_active():
        model_args, training_obj_args, decomp_args, data_training_args, vis_args = parser.parse_json_file(json_file=JSON_FILE_NAME_MANUAL)
    else:
        args = parse_args()
        model_args, training_obj_args, decomp_args, data_training_args, vis_args = parser.parse_json_file(json_file=args.config_file)
    delattr(model_args,"comment_model_args")
    delattr(data_training_args,"comment_data_args")
    delattr(training_obj_args,"comment_tr_obj_args")
    delattr(decomp_args,"comment_decomp_args")
    delattr(vis_args,"comment_vis_args")

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
        projection_dir = os.path.join(os.path.dirname(data_training_args.parent_dir),model_args.eigenprojection + "_" + model_args.vae_input_type)
        os.makedirs(projection_dir, exist_ok=True)

    BETAS = beta[1:]

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
                in_size = int(model_args.n_mels_vae)
            elif model_args.vae_input_type == "mel_ocs":
                reduction_components = 30 #for ica,pca only
                in_size = int(model_args.n_mels_vae)
            elif model_args.vae_input_type == "mel_all":
                reduction_components = 35 #for ica,pca only
                in_size = int(model_args.n_mels_vae)
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
                    treat_as_sequence=False,
                    kernel_sizes=model_args.vae_kernel_sizes,
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

            representation_function.load_state_dict(weights)
        
        
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
            train_dataset = concatenate_datasets([vectorized_datasets["train"], vectorized_datasets["validation"],vectorized_datasets["test"],vectorized_datasets["dev"]])                                                    
            train_dataloader = DataLoader(
                train_dataset, 
                shuffle=True,
                collate_fn=data_collator, 
                batch_size=data_training_args.per_device_train_batch_size
            )
        elif data_training_args.dataset_name == "iemocap":
            train_dataset = concatenate_datasets([vectorized_datasets["train"], vectorized_datasets["validation"],vectorized_datasets["test"]])                                                    
            train_dataloader = DataLoader(
                train_dataset, 
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
            representation_function, train_dataloader = accelerator.prepare(
                representation_function, train_dataloader
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
                eigenprojection_function = joblib.load(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib'))
            else:
                "Get the training data"
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

                    elif "timit" in data_training_args.dataset_name:
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

                    elif "timit" in data_training_args.dataset_name:
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
                    rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
                    indices = rng.choice(z_mean.shape[0], size=int(z_mean.shape[0]*0.01), replace=False)
                    eigenprojection_function.fit(z_mean[indices])
                elif (model_args.vae_input_type == "waveform_all" or model_args.vae_input_type == "waveform_ocs"):
                    rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
                    indices = rng.choice(z_mean.shape[0], size=int(z_mean.shape[0]*0.2), replace=False)
                    eigenprojection_function.fit(z_mean[indices])
                else:
                    eigenprojection_function.fit(z_mean)
                "Save the fitted model"
                joblib.dump(eigenprojection_function, os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib'))


        elif data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            if model_args.raw_mels and model_args.eigenprojection is not None and os.path.exists(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib')):
                print(f"Loading the fitted {model_args.eigenprojection}_{model_args.vae_input_type} model from memory")
                eigenprojection_function = joblib.load(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib'))

            "Get the data for VOC_ALS or iemocap"
            with torch.no_grad():
                start_time = time.time()
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
                    
                    if "iemocap" in data_training_args.dataset_name:
                        batch["mask_time_indices"] = sub_attention_mask.clone()
                        with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
                            phoneme_to_id = json.load(json_file)
                        id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
                        phonemes_batch = batch.pop("phonemes", None)
                        emotion_batch = list(batch.pop("emotion", None))
                        phonemes_batch = phonemes_batch[~overlap_mask_batch]

                        batch.pop("words", None)
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
                            batch["input_values"] = extract_mel_spectrogram(batch["input_values"],config.fs,n_mels=model_args.n_mels_vae, n_fft=int(config.receptive_field*config.fs), hop_length=int(config.receptive_field*config.fs)+1, normalize=model_args.mel_norm_vae)
                        elif model_args.vae_input_type == "waveform":
                            batch["input_values"] = batch["input_values"][:,0,:,:]
                        batch["attention_mask"] = sub_attention_mask

                    if not model_args.raw_mels:
                        outputs = representation_function(**batch)
                    else:                    
                        outputs = [batch["input_values"]]
                    del batch

                    if "iemocap" in data_training_args.dataset_name:
                        if step == 0:
                            phonemes_frame = phonemes_batch.clone()
                            emotion_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)]) 
                            #emotion_seq = torch.stack(emotion_batch) 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): # or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                phonemes_frame = torch.cat((phonemes_frame,phonemes_batch))
                                emotion_frame = torch.cat((emotion_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)])),dim = 0)
                            #emotion_seq = torch.cat((emotion_seq,torch.stack(emotion_batch)),dim = 0) 

                        if step == 0:
                            speaker_id_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) 
                            #speaker_id_seq = torch.stack(speaker_id_batch) 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): # or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                speaker_id_frame = torch.cat((speaker_id_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            #speaker_id_seq = torch.cat((speaker_id_seq,torch.stack(speaker_id_batch)),dim = 0) 

                    elif "VOC_ALS" in data_training_args.dataset_name:
                        for j in range(overlap_mask_batch.shape[0]):
                            for i in range(overlap_mask_batch.shape[1]):
                                if not sub_attention_mask[j,i]:
                                    overlap_mask_batch[j,i] = True

                        if step == 0:
                            alsfrs_total_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_total_batch)]) 
                            #alsfrs_total_seq = torch.stack(alsfrs_total_batch)
                            alsfrs_speech_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_speech_batch)]) 
                            #alsfrs_speech_seq = torch.stack(alsfrs_speech_batch)
                            disease_duration_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(disease_duration_batch)]) 
                            #disease_duration_seq = torch.stack(disease_duration_batch)
                            king_stage_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(king_stage_batch)]) 
                            #king_stage_seq = torch.stack(king_stage_batch)
                            cantagallo_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(cantagallo_batch)]) 
                            #cantagallo_seq = torch.stack(cantagallo_batch)
                            phonemes_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(phonemes_batch)])
                            #phonemes_seq = torch.stack(phonemes_batch)
                            speaker_id_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])
                            #speaker_id_seq = torch.stack(speaker_id_batch)
                            group_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(group_batch)])
                            #group_seq = torch.stack(group_batch)
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): # or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                alsfrs_total_frame = torch.cat((alsfrs_total_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_total_batch)])),dim = 0)
                                #alsfrs_total_seq = torch.cat((alsfrs_total_seq,torch.stack(alsfrs_total_batch)),dim = 0) 
                                alsfrs_speech_frame = torch.cat((alsfrs_speech_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_speech_batch)])),dim = 0)
                                #alsfrs_speech_seq = torch.cat((alsfrs_speech_seq,torch.stack(alsfrs_speech_batch)),dim = 0) 
                                disease_duration_frame = torch.cat((disease_duration_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(disease_duration_batch)])),dim = 0)
                                #disease_duration_seq = torch.cat((disease_duration_seq,torch.stack(disease_duration_batch)),dim = 0)
                                king_stage_frame = torch.cat((king_stage_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(king_stage_batch)])),dim = 0)
                                #king_stage_seq = torch.cat((king_stage_seq,torch.stack(king_stage_batch)),dim = 0)
                                cantagallo_frame = torch.cat((cantagallo_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(cantagallo_batch)])),dim = 0)
                                #cantagallo_seq = torch.cat((cantagallo_seq,torch.stack(cantagallo_batch)),dim = 0)
                                phonemes_frame = torch.cat((phonemes_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(phonemes_batch)])),dim = 0)
                                #phonemes_seq = torch.cat((phonemes_seq,torch.stack(phonemes_batch)),dim = 0)
                                speaker_id_frame = torch.cat((speaker_id_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                                #speaker_id_seq = torch.cat((speaker_id_seq,torch.stack(speaker_id_batch)),dim = 0)
                                group_frame = torch.cat((group_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(group_batch)])),dim = 0)
                                #group_seq = torch.cat((group_seq,torch.stack(group_batch)),dim = 0)

                    "Gather latents for evaluations"
                    if data_training_args.dataset_name == "sim_vowels":
                        overlap_mask_batch = overlap_mask_batch[sub_attention_mask].view(batch_size,-1)
                    z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                    if vis_args.visualize_latent_frame:
                        if step == 0:                    
                            z_mean = z_mean_batch.detach().cpu()
                        elif step*batch_size < vis_args.latent_train_set_frames_to_vis and step > 0:
                            z_mean = torch.cat((z_mean,z_mean_batch.detach().cpu()),dim = 0)
                        elif step*batch_size >= vis_args.latent_train_set_frames_to_vis:
                            break

                    if step*batch_size >= vis_args.latent_train_set_frames_to_vis: 
                        break
                
            if model_args.raw_mels and model_args.eigenprojection is not None:

                if not os.path.exists(os.path.join(projection_dir, model_args.eigenprojection + "_" + model_args.vae_input_type + '_model.joblib')):
                    "Fit projection on train set"
                    if 'kpca-' in model_args.eigenprojection: 
                        "Fit the projection on a subset of the training set"
                        rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
                        indices = rng.choice(z_mean.shape[0], size=int(z_mean.shape[0]*0.01), replace=False)
                        eigenprojection_function.fit(z_mean[indices])
                    elif (model_args.vae_input_type == "waveform_all" or model_args.vae_input_type == "waveform_ocs"):
                        rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
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
            "Get all sets representations"
            start_time = time.time()
            with torch.no_grad():
                if vis_args.visualize_train_set:
                    "Train set for loop"
                    for step, batch in enumerate(train_dataloader):
                        
                        batch_size = batch["input_values"].shape[0]
                        mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
                        sub_attention_mask = batch.pop("sub_attention_mask", None)
                        overlap_mask_batch = batch.pop("overlap_mask", None)
                        if hasattr(batch,"reconstruction_NRMSE_seq"):
                            batch.pop("reconstruction_NRMSE_seq", None)
                        if hasattr(batch,"reconstruction_NRMSEs"):
                            batch.pop("reconstruction_NRMSEs", None)
                        if hasattr(batch,"correlograms"):
                            batch.pop("correlograms", None)
                        if hasattr(batch,"correlogram_seq"):
                            batch.pop("correlogram_seq", None)
                        batch["global_step"] = 0
                        assert overlap_mask_batch != None if data_training_args.dataset_name == "timit" else True
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
                            
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): # or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                        elif "timit" in data_training_args.dataset_name:
                            batch["mask_time_indices"] = sub_attention_mask.clone()
                            "Phonemes39 (vowels/consonants)"
                            phonemes39_batch = batch.pop("phonemes39", None)
                            if hasattr(batch,"phonemes48"):
                                phonemes48_batch = batch.pop("phonemes48", None)
                            with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                                phoneme39_to_id = json.load(json_file)
                            id_to_phoneme39 = {v: k for k, v in phoneme39_to_id.items()}
                            batch_phonemes = []
                            batch_vowels = []
                            batch_consonants = []
                            pho39 = phonemes39_batch[~overlap_mask_batch]
                            for ph in pho39:
                                if ph != -100:
                                    batch_phonemes.append(id_to_phoneme39[ph.item()])
                                    if id_to_phoneme39[ph.item()] in ['ae','ao','aw','ax','ay','eh','er','ey','ih','ix','iy','ow','oy','uh','uw','y']:
                                        batch_vowels.append(id_to_phoneme39[ph.item()])
                                    else:
                                        batch_vowels.append('NO')
                                    if id_to_phoneme39[ph.item()] in ['b','ch','d','dh','dx','f','g','hh','jh','k','l','m','n','p','r','s','sh','t','th','v','w','z']:
                                        batch_consonants.append(id_to_phoneme39[ph.item()])
                                    else:
                                        batch_consonants.append('NO')
                                else:
                                    batch_phonemes.append('sil')
                                    batch_vowels.append('NO')
                                    batch_consonants.append('NO')
                            
                            phonemes39_batch = np.array(batch_phonemes)
                            vowels_batch = np.array(batch_vowels)
                            consonants_batch = np.array(batch_consonants)
                            
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
                                vowel_labels_train = torch.cat([torch.tensor(v) for v in vowel_labels_batch]) 
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): # or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                    vowel_labels_train = torch.cat((vowel_labels_train,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                                speaker_vt_factor_frame_train = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)]) 
                                #speaker_vt_factor_seq_train = speaker_vt_factor_batch.clone() #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch)])
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): # or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                    speaker_vt_factor_frame_train = torch.cat((speaker_vt_factor_frame_train,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                                #if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_train_set_seq_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                #    speaker_vt_factor_seq_train = torch.cat((speaker_vt_factor_seq_train,speaker_vt_factor_batch),dim = 0) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch)])

                        elif "timit" in data_training_args.dataset_name:
                            if step == 0:
                                phonemes39_train = phonemes39_batch.copy()
                                vowels_train = vowels_batch.copy()
                                consonants_train = consonants_batch.copy()
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): 
                                    phonemes39_train = np.concatenate((phonemes39_train,phonemes39_batch))
                                    vowels_train = np.concatenate((vowels_train,vowels_batch))
                                    consonants_train = np.concatenate((consonants_train,consonants_batch))

                            if step == 0:
                                speaker_id_frame_train = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) 
                                speaker_id_seq_train = torch.stack(speaker_id_batch) 
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis): # or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                    speaker_id_frame_train = torch.cat((speaker_id_frame_train,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                                #if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_train_set_seq_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                #    speaker_id_seq_train = torch.cat((speaker_id_seq_train,torch.stack(speaker_id_batch)),dim = 0) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch)])

                        "Gather latents for evaluations"
                        if data_training_args.dataset_name == "sim_vowels":
                            overlap_mask_batch = overlap_mask_batch[sub_attention_mask].view(batch_size,-1)
                        z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                        if vis_args.visualize_latent_frame:
                            if step == 0:
                                z_mean_train = z_mean_batch.detach().cpu()
                            elif step*batch_size < vis_args.latent_train_set_frames_to_vis and step > 0:
                                z_mean_train = torch.cat((z_mean_train,z_mean_batch.detach().cpu()),dim = 0)
                            elif step*batch_size >= vis_args.latent_train_set_frames_to_vis:
                                break                  
                    
                        if step*batch_size >= vis_args.latent_train_set_frames_to_vis: 
                            break
                    
                    if model_args.eigenprojection is not None:
                        "Transform train set"
                        z_mean_train = torch.tensor(eigenprojection_function.transform(z_mean_train))

                
                if vis_args.visualize_dev_set:
                    "Dev set for loop"
                    for step, batch in enumerate(eval_dataloader):
                        
                        batch_size = batch["input_values"].shape[0]
                        mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
                        sub_attention_mask = batch.pop("sub_attention_mask", None)
                        overlap_mask_batch = batch.pop("overlap_mask", None)
                        if hasattr(batch,"reconstruction_NRMSE_seq"):
                            batch.pop("reconstruction_NRMSE_seq", None)
                        if hasattr(batch,"reconstruction_NRMSEs"):
                            batch.pop("reconstruction_NRMSEs", None)
                        if hasattr(batch,"correlograms"):
                            batch.pop("correlograms", None)
                        if hasattr(batch,"correlogram_seq"):
                            batch.pop("correlogram_seq", None)
                        batch["global_step"] = 0
                        assert overlap_mask_batch != None if data_training_args.dataset_name == "timit" else True
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
                            
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis): # or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                        elif data_training_args.dataset_name == "timit:
                            batch["mask_time_indices"] = sub_attention_mask.clone()
                            batch.pop("phonemes48", None)
                            "Phonemes39 (vowels/consonants)"
                            "We do not need phonemes48 as we only visualize here"
                            phonemes39_batch = batch.pop("phonemes39", None)
                            with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                                phoneme39_to_id = json.load(json_file)
                            id_to_phoneme39 = {v: k for k, v in phoneme39_to_id.items()}
                            batch_phonemes = []
                            batch_vowels = []
                            batch_consonants = []
                            pho39 = phonemes39_batch[~overlap_mask_batch]
                            for ph in pho39:
                                if ph != -100:
                                    batch_phonemes.append(id_to_phoneme39[ph.item()])
                                    if id_to_phoneme39[ph.item()] in ['ae','ao','aw','ax','ay','eh','er','ey','ih','ix','iy','ow','oy','uh','uw','y']:
                                        batch_vowels.append(id_to_phoneme39[ph.item()])
                                    else:
                                        batch_vowels.append('NO')
                                    if id_to_phoneme39[ph.item()] in ['b','ch','d','dh','dx','f','g','hh','jh','k','l','m','n','p','r','s','sh','t','th','v','w','z']:
                                        batch_consonants.append(id_to_phoneme39[ph.item()])
                                    else:
                                        batch_consonants.append('NO')
                                else:
                                    batch_phonemes.append('sil')
                                    batch_vowels.append('NO')
                                    batch_consonants.append('NO')
                            
                            phonemes39_batch = np.array(batch_phonemes)
                            vowels_batch = np.array(batch_vowels)
                            consonants_batch = np.array(batch_consonants)
                                
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
                                vowel_labels_dev = torch.cat([torch.tensor(v) for v in vowel_labels_batch])
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis): # or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                    vowel_labels_dev = torch.cat((vowel_labels_dev,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                            if step == 0:
                                speaker_vt_factor_frame_dev = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)]) 
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis): # or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                    speaker_vt_factor_frame_dev = torch.cat((speaker_vt_factor_frame_dev,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                                #if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_dev_set_seq_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                #    speaker_vt_factor_seq_dev = torch.cat((speaker_vt_factor_seq_dev,speaker_vt_factor_batch),dim = 0) 

                        elif "timit" in data_training_args.dataset_name:
                            if step == 0:
                                phonemes39_dev = phonemes39_batch.copy()
                                vowels_dev = vowels_batch.copy()
                                consonants_dev = consonants_batch.copy()
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis): # or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                    phonemes39_dev = np.concatenate((phonemes39_dev,phonemes39_batch))
                                    vowels_dev = np.concatenate((vowels_dev,vowels_batch))
                                    consonants_dev = np.concatenate((consonants_dev,consonants_batch))

                            if step == 0:
                                speaker_id_frame_dev = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])
                                speaker_id_seq_dev = torch.stack(speaker_id_batch) 
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis): # or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                    speaker_id_frame_dev = torch.cat((speaker_id_frame_dev,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                                #if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_dev_set_seq_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                #    speaker_id_seq_dev = torch.cat((speaker_id_seq_dev,torch.stack(speaker_id_batch)),dim = 0) 

                        "Gather latents for evaluations"
                        if data_training_args.dataset_name == "sim_vowels":
                            overlap_mask_batch = overlap_mask_batch[sub_attention_mask].view(batch_size,-1)
                        z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                        if vis_args.visualize_latent_frame:
                            if step == 0:
                                z_mean_dev = z_mean_batch.detach().cpu()
                            elif step*batch_size < vis_args.latent_dev_set_frames_to_vis and step > 0:
                                z_mean_dev = torch.cat((z_mean_dev,z_mean_batch.detach().cpu()),dim = 0)
                            elif step*batch_size >= vis_args.latent_dev_set_frames_to_vis:
                                break                  
                    
                        if step*batch_size >= vis_args.latent_dev_set_frames_to_vis: 
                            break
                    
                    if model_args.eigenprojection is not None:
                        "Transform dev set"
                        z_mean_dev = torch.tensor(eigenprojection_function.transform(z_mean_dev))


                if vis_args.visualize_test_set:
                    "Test set for loop"
                    for step, batch in enumerate(test_dataloader):
                        
                        batch_size = batch["input_values"].shape[0]
                        mask_indices_seq_length = batch["sub_attention_mask"].shape[1]
                        sub_attention_mask = batch.pop("sub_attention_mask", None)
                        overlap_mask_batch = batch.pop("overlap_mask", None)
                        if hasattr(batch,"reconstruction_NRMSE_seq"):
                            batch.pop("reconstruction_NRMSE_seq", None)
                        if hasattr(batch,"reconstruction_NRMSEs"):
                            batch.pop("reconstruction_NRMSEs", None)
                        if hasattr(batch,"correlograms"):
                            batch.pop("correlograms", None)
                        if hasattr(batch,"correlogram_seq"):
                            batch.pop("correlogram_seq", None)
                        batch["global_step"] = 0
                        assert overlap_mask_batch != None if data_training_args.dataset_name == "timit" else True
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
                            
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis): # or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                        elif data_training_args.dataset_name == "timit":
                            batch["mask_time_indices"] = sub_attention_mask.clone()
                            batch.pop("phonemes48", None)
                            "Phonemes39 (vowels/consonants)"
                            phonemes39_batch = batch.pop("phonemes39", None)
                            with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                                phoneme39_to_id = json.load(json_file)
                            id_to_phoneme39 = {v: k for k, v in phoneme39_to_id.items()}
                            batch_phonemes = []
                            batch_vowels = []
                            batch_consonants = []
                            pho39 = phonemes39_batch[~overlap_mask_batch]
                            for ph in pho39:
                                if ph != -100:
                                    batch_phonemes.append(id_to_phoneme39[ph.item()])
                                    if id_to_phoneme39[ph.item()] in ['ae','ao','aw','ax','ay','eh','er','ey','ih','ix','iy','ow','oy','uh','uw','y']:
                                        batch_vowels.append(id_to_phoneme39[ph.item()])
                                    else:
                                        batch_vowels.append('NO')
                                    if id_to_phoneme39[ph.item()] in ['b','ch','d','dh','dx','f','g','hh','jh','k','l','m','n','p','r','s','sh','t','th','v','w','z']:
                                        batch_consonants.append(id_to_phoneme39[ph.item()])
                                    else:
                                        batch_consonants.append('NO')
                                else:
                                    batch_phonemes.append('sil')
                                    batch_vowels.append('NO')
                                    batch_consonants.append('NO')
                            
                            phonemes39_batch = np.array(batch_phonemes)
                            vowels_batch = np.array(batch_vowels)
                            consonants_batch = np.array(batch_consonants)
                                
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
                                vowel_labels_test = torch.cat([torch.tensor(v) for v in vowel_labels_batch])
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis): # or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                    vowel_labels_test = torch.cat((vowel_labels_test,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                            if step == 0:
                                speaker_vt_factor_frame_test = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])
                                #speaker_vt_factor_seq_test = speaker_vt_factor_batch.clone() 
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis): # or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                    speaker_vt_factor_frame_test = torch.cat((speaker_vt_factor_frame_test,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                                #if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_test_set_seq_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                #    speaker_vt_factor_seq_test = torch.cat((speaker_vt_factor_seq_test,speaker_vt_factor_batch),dim = 0)

                        elif "timit" in data_training_args.dataset_name:
                            if step == 0:
                                phonemes39_test = phonemes39_batch.copy()
                                vowels_test = vowels_batch.copy()
                                consonants_test = consonants_batch.copy()
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis): # or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                    phonemes39_test = np.concatenate((phonemes39_test,phonemes39_batch))
                                    vowels_test = np.concatenate((vowels_test,vowels_batch))
                                    consonants_test = np.concatenate((consonants_test,consonants_batch))

                            if step == 0:
                                speaker_id_frame_test = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) 
                                speaker_id_seq_test = torch.stack(speaker_id_batch) 
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis): # or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                    speaker_id_frame_test = torch.cat((speaker_id_frame_test,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                                #if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_test_set_seq_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                #    speaker_id_seq_test = torch.cat((speaker_id_seq_test,torch.stack(speaker_id_batch)),dim = 0) 

                        "Gather latents for evaluations"
                        if data_training_args.dataset_name == "sim_vowels":
                            overlap_mask_batch = overlap_mask_batch[sub_attention_mask].view(batch_size,-1)
                        z_mean_batch = torch.masked_select(outputs[0],~overlap_mask_batch[...,None]).reshape(-1,outputs[0].shape[-1])
                        if vis_args.visualize_latent_frame:
                            if step == 0:
                                z_mean_test = z_mean_batch.detach().cpu()
                            elif step*batch_size < vis_args.latent_test_set_frames_to_vis and step > 0:
                                z_mean_test = torch.cat((z_mean_test,z_mean_batch.detach().cpu()),dim = 0)
                            elif step*batch_size >= vis_args.latent_test_set_frames_to_vis:
                                break                  
                    
                        if step*batch_size >= vis_args.latent_test_set_frames_to_vis: 
                            break
                    
                    if model_args.eigenprojection is not None:
                        "Transform test set"
                        z_mean_test = torch.tensor(eigenprojection_function.transform(z_mean_test))


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total loading time: {elapsed_time: .4f} seconds")


        "-------------------------------------------------------------------------------------------"
        "Visualizations"
        "-------------------------------------------------------------------------------------------"
        def sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_that_subset,
            vowel_labels = None,speaker_labels_frame = None,speaker_labels_seq = None, 
            mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
            mu_joint_components_z = None,mu_all_z = None,
            mu_originals_s = None,mu_components_s = None,mu_projections_s = None,
            mu_joint_components_s = None,mu_all_s = None
            ):

        
            if vis_that_subset and vis_args.visualize_latent_frame:
                "-------------------------------------------------------------------------------------------"
                "Vowel frame"
                "--------------------------------------------------------------------------------------------"
                if "vowel" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Vowel Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Vowels"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = vowel_labels,
                        target = "vowel",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowels',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Vowel Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    
                    if vis_args.plot_3d:
                        "TSNE - X - Vowels - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = vowel_labels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowels',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Vowels"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = vowel_labels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowels',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        
                        if vis_args.plot_3d:
                            "UMAP - X - Vowels"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowels',data_training_args.vis_method)
                            )

                "-------------------------------------------------------------------------------------------"
                "Speaker frame"
                "--------------------------------------------------------------------------------------------"
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    "2D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Vowels"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = speaker_labels_frame,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                    )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    if vis_args.plot_3d:
                        "TSNE - X - Vowels"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'

                        "UMAP - X / OCs - Vowels & Frequency"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )
                        
                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"

                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        if vis_args.plot_3d:
                            "UMAP - X / OCs - Vowels & Frequency"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                            )


        def timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_that_subset,
                phoneme_labels = None, consonant_labels = None,vowel_labels = None,
                speaker_labels_frame = None,speaker_labels_seq = None, 
                mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                mu_joint_components_z = None,mu_all_z = None,
                mu_originals_s = None,mu_components_s = None,mu_projections_s = None,
                mu_joint_components_s = None,mu_all_s = None
            ):

            "Select 10-20 speakers to visualize"
            speaker_labels_frame = speaker_labels_frame.detach().cpu().numpy()
            rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
            if speaker_labels_frame is not None:
                all_speakers = np.unique(speaker_labels_frame)
            else:
                all_speakers = np.array([])
            if len(all_speakers) >= 10:
                sel_10_speakers_list = rng.choice(all_speakers, size=10, replace=False)
                sel_10_sp_mask = np.isin(speaker_labels_frame, sel_10_speakers_list)
                sel_10_speakers = speaker_labels_frame[sel_10_sp_mask]
            elif len(all_speakers) > 0 and len(all_speakers) < 10:
                sel_10_speakers = speaker_labels_frame.copy()
                sel_10_sp_mask = np.ones_like(speaker_labels_frame, dtype=bool)
            
            "Select specific vowels and consonants to be visualized and remove the NO flags"
            if vowel_labels is not None:
                vowel_mask = np.isin(vowel_labels, vis_args.sel_vowels_list_timit)
                sel_vowels = vowel_labels[vowel_mask]
            if consonant_labels is not None:               
                consonant_mask = np.isin(consonant_labels, vis_args.sel_consonants_list_timit)
                sel_consonants = consonant_labels[consonant_mask]
            if phoneme_labels is not None:
                phoneme_mask = np.isin(phoneme_labels, vis_args.sel_phonemes_list_timit)
                sel_phonemes = phoneme_labels[phoneme_mask]

                    
            if vis_that_subset and vis_args.visualize_latent_frame:
                    
                "For speakers we need to index using the speaker mask"
                mu_originals_z_sel_speakers = mu_originals_z[sel_10_sp_mask]
                "Use other masks similarly"
                mu_originals_z_sel_phonemes = mu_originals_z[phoneme_mask]
                mu_originals_z_sel_consonants = mu_originals_z[consonant_mask]
                mu_originals_z_sel_vowels = mu_originals_z[vowel_mask]

                "-------------------------------------------------------------------------------------------"
                "Phonemes"
                "-------------------------------------------------------------------------------------------"                
                if "phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Phonemes"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z_sel_phonemes,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_phonemes,
                        target = "phoneme39",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    
                    if vis_args.plot_3d:
                        "TSNE - X - Phonemes - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme39",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Phonemes"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme39",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        "UMAP - X - Phonemes"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme39",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                        )


                "-------------------------------------------------------------------------------------------"
                "Vowels"
                "-------------------------------------------------------------------------------------------"                
                if "vowel" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Vowels Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Vowels"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z_sel_vowels,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_vowels,
                        target = "vowel",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowel',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Vowel Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    
                    if vis_args.plot_3d:
                        "TSNE - X - Vowels - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_vowels,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_vowels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowel',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Vowels"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_vowels,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_vowels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowel',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - Vowels"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_vowels,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','vowel',data_training_args.vis_method)
                            )


                "-------------------------------------------------------------------------------------------"
                "Consonants"
                "-------------------------------------------------------------------------------------------"                
                if "consonant" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Consonants Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Consonants"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z_sel_consonants,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_consonants,
                        target = "consonant",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','consonant',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Consonant Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - Consonants - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_consonants,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_consonants,
                            target = "consonant",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','consonant',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Consonant Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Consonants"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_consonants,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_consonants,
                            target = "consonant",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','consonant',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Consonant Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - Consonants"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_consonants,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','consonant',data_training_args.vis_method)
                            )


                "-------------------------------------------------------------------------------------------"
                "Speaker frame"
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Speakers"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z_sel_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_10_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                    )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    if vis_args.plot_3d:
                        "TSNE - X - Speakers"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )
                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'

                        "UMAP - X / OCs - Speakers"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"

                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        if vis_args.plot_3d:
                            "UMAP - X / OCs - Speakers"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                            )


        def iemocap_latent_vis(config,data_training_args,decomp_args,data_subset,vis_that_subset,
                phoneme_labels = None, emotion_labels_frame = None, emotion_labels_seq = None,
                speaker_labels_frame = None,speaker_labels_seq = None, 
                mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                mu_joint_components_z = None,mu_all_z = None,
                mu_originals_s = None,mu_components_s = None,mu_projections_s = None,
                mu_joint_components_s = None,mu_all_s = None
            ):
            
            phoneme_labels = phoneme_labels.detach().cpu().numpy()
            "convert phoneme ids to characters"
            phoneme_labels = np.array([id_to_phoneme[ph.item()] for ph in phoneme_labels])
            "Select phonemes to be visualized and remove the NO flags"
            if phoneme_labels is not None:
                phoneme_mask = np.isin(phoneme_labels, vis_args.sel_phonemes_list_iemocap)
                sel_phonemes = phoneme_labels[phoneme_mask]
                non_verbal_mask = np.isin(phoneme_labels, vis_args.sel_non_verbal_phonemes_iemocap)
                sel_non_verbal_phonemes = phoneme_labels[non_verbal_mask]

                    
            if vis_that_subset and vis_args.visualize_latent_frame:
                    
                "Use masks to select the relevant data"
                mu_originals_z_sel_phonemes = mu_originals_z[phoneme_mask]
                mu_originals_z_sel_non_verbal_phonemes = mu_originals_z[non_verbal_mask]

                "-------------------------------------------------------------------------------------------"
                "Phonemes"
                "-------------------------------------------------------------------------------------------"                
                if "phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Phonemes"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z_sel_phonemes,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_phonemes,
                        target = "phoneme",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - Phonemes - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Phonemes"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - Phonemes"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phonemes',data_training_args.vis_method)
                            )

                "-------------------------------------------------------------------------------------------"
                "Non-verbal Phonemes"
                "-------------------------------------------------------------------------------------------"                
                if "non_verbal_phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Non-verbal Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Non-verbal Phonemes"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z_sel_non_verbal_phonemes,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_non_verbal_phonemes,
                        target = "non_verbal_phoneme",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','non_verbal_phonemes',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Non-verbal Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - Non-verbal Phonemes - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_non_verbal_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_non_verbal_phonemes,
                            target = "non_verbal_phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','non_verbal_phonemes',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Non-verbal Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False

                        "UMAP - X - Non-verbal Phonemes"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_non_verbal_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_non_verbal_phonemes,
                            target = "non_verbal_phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','non_verbal_phonemes',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Non-verbal Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - Non-verbal Phonemes"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_non_verbal_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','non_verbal_phonemes',data_training_args.vis_method)
                            )

                "-------------------------------------------------------------------------------------------"
                "Speaker frame"
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Speakers"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = speaker_labels_frame,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                    )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    if vis_args.plot_3d:
                        "TSNE - X - Speakers"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        
                        "UMAP - X / OCs - Speakers"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"

                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        if vis_args.plot_3d:
                            "UMAP - X / OCs - Speakers"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                            )

                "-------------------------------------------------------------------------------------------"
                "Emotion frame"
                if "emotion" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Emotion Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Speakers"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = emotion_labels_frame,
                        target = "emotion",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','categorical_emotions',data_training_args.vis_method)
                    )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Emotion Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    if vis_args.plot_3d:
                        "TSNE - X - Speakers"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = emotion_labels_frame,
                            target = "emotion",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','categorical_emotions',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Emotion Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        
                        "UMAP - X / OCs - Speakers"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = emotion_labels_frame,
                            target = "emotion",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','categorical_emotions',data_training_args.vis_method)
                        )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Emotion Visualizations"
                        "--------------------------------------------------------------------------------------------"

                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        if vis_args.plot_3d:
                            "UMAP - X / OCs - Speakers"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','categorical_emotions',data_training_args.vis_method)
                            )


        def voc_als_latent_vis(config,data_training_args,decomp_args,data_subset,vis_that_subset,
            phoneme_labels = None, alsfrs_total = None,alsfrs_speech = None,speaker_labels_frame = None, 
            disease_duration = None, king_stage = None, group = None, cantagallo = None,
            mu_originals_z = None):

            rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
            "VOC-ALS has 153 different speakers - Select 10 to visualize"       
            speaker_labels_frame = speaker_labels_frame.detach().cpu().numpy()
            all_speakers = np.unique(speaker_labels_frame)
            if len(all_speakers) >= 10:
                sel_10_speakers_list = rng.choice(all_speakers, size=10, replace=False)
                sel_10_sp_mask = np.isin(speaker_labels_frame, sel_10_speakers_list)
                sel_10_speakers = speaker_labels_frame[sel_10_sp_mask]
            elif len(all_speakers) > 0 and len(all_speakers) < 10:
                sel_10_speakers = speaker_labels_frame.copy()
                sel_10_sp_mask = np.ones_like(speaker_labels_frame, dtype=bool)
            
            "For speakers we need to index using the speaker mask"
            mu_originals_z_sel_speakers = mu_originals_z[sel_10_sp_mask]

            if vis_that_subset and vis_args.visualize_latent_frame:
                    
                "--------------------------------------------------------------------------------------------"
                "Phoneme"
                "--------------------------------------------------------------------------------------------"
                if "phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Phonemes"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = phoneme_labels,
                        target = "phoneme",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phoneme',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - Phonemes - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = phoneme_labels,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phoneme',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Phonemes"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = phoneme_labels,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phoneme',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - Phonemes"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','phoneme',data_training_args.vis_method)
                            )


                "--------------------------------------------------------------------------------------------"
                "ALSFRS-Total"
                "--------------------------------------------------------------------------------------------"
                if "alsfrs_total" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE ALSFRS-Total Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - ALSFRS-Total"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = alsfrs_total,
                        target = "alsfrs_total",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_total',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE ALSFRS-Total Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - ALSFRS-Total - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_total,
                            target = "alsfrs_total",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_total',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP ALSFRS-Total Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - ALSFRS-Total"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_total,
                            target = "alsfrs_total",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_total',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP ALSFRS-Total Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - ALSFRS-Total"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_total',data_training_args.vis_method)
                            )


                "--------------------------------------------------------------------------------------------"
                "ALSFRS-Speech"
                "--------------------------------------------------------------------------------------------"
                if "alsfrs_speech" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE ALSFRS-Speech Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - ALSFRS-Speech"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = alsfrs_speech,
                        target = "alsfrs_speech",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_speech',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE ALSFRS-Speech Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - ALSFRS-Speech - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech,
                            target = "alsfrs_speech",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_speech',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP ALSFRS-Speech Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - ALSFRS-Speech"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech,
                            target = "alsfrs_speech",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_speech',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP ALSFRS-Speech Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - ALSFRS-Speech"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','alsfrs_speech',data_training_args.vis_method)
                            )

                "--------------------------------------------------------------------------------------------"
                "Disease Duration"
                "--------------------------------------------------------------------------------------------"
                if "disease_duration" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Disease Duration Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Disease Duration"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = disease_duration,
                        target = "disease_duration",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','disease_duration',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Disease Duration Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - Disease Duration - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = disease_duration,
                            target = "disease_duration",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','disease_duration',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Disease Duration Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Disease Duration"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = disease_duration,
                            target = "disease_duration",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','disease_duration',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Disease Duration Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - ALSFRS-Speech"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','disease_duration',data_training_args.vis_method)
                            )


                "--------------------------------------------------------------------------------------------"
                "King's Stage"
                "--------------------------------------------------------------------------------------------"
                if "king_stage" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE King's Stage Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - King's Stage"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = king_stage,
                        target = "king_stage",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','king_stage',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE King's Stage Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - King's Stage - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = king_stage,
                            target = "king_stage",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','king_stage',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP King's Stage Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - King's Stage"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = king_stage,
                            target = "king_stage",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','king_stage',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP King's Stage Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - ALSFRS-Speech"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','king_stage',data_training_args.vis_method)
                            )

                "--------------------------------------------------------------------------------------------"
                "Disease category / group"
                "--------------------------------------------------------------------------------------------"
                if "group" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Disease category / group Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Disease category / group"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = group,
                        target = "group",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','group',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Disease category / group Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - Disease category / group - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = group,
                            target = "group",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','group',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Disease category / group Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Disease category / group"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = group,
                            target = "group",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','group',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Disease category / group Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - ALSFRS-Speech"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','group',data_training_args.vis_method)
                            )


                "--------------------------------------------------------------------------------------------"
                "Cantagallo"
                "--------------------------------------------------------------------------------------------"
                if "group" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Cantagallo Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Cantagallo"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = cantagallo,
                        target = "cantagallo",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','cantagallo',data_training_args.vis_method)
                    )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Cantagallo Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        "TSNE - X - Cantagallo - 3D sphere"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = cantagallo,
                            target = "cantagallo",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','cantagallo',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Cantagallo Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        
                        "UMAP - X - Cantagallo"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = cantagallo,
                            target = "cantagallo",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','cantagallo',data_training_args.vis_method)
                        )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Cantagallo Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        if vis_args.plot_3d:
                            "UMAP - X - ALSFRS-Speech"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','cantagallo',data_training_args.vis_method)
                            )

                "-------------------------------------------------------------------------------------------"
                "Speaker frame"
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    "TSNE - X - Speakers"
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_z_sel_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_10_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = False,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                    )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    if vis_args.plot_3d:
                        "TSNE - X - Speakers"
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'

                        "UMAP - X / OCs - Speakers"
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = False,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                        )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"

                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        if vis_args.plot_3d:
                            "UMAP - X / OCs - Speakers"
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = False,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,model_args.vae_type + '_' + model_args.vae_input_type,data_training_args.dataset_name,BETAS,data_subset,'X','speakers',data_training_args.vis_method)
                            )


        if data_training_args.dataset_name == 'sim_vowels':
            "-------------------------------------------------------------------------------------------"
            "Sim_vowels Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                    vowel_labels_train,speaker_vt_factor_frame_train,speaker_labels_seq = None, 
                    mu_originals_z = z_mean_train
                    )
            
            "-------------------------------------------------------------------------------------------"
            "Sim_vowels Dev set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_dev_set:
                data_subset = 'dev'
                sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                    vowel_labels_dev,speaker_vt_factor_frame_dev,speaker_labels_seq = None, 
                    mu_originals_z = z_mean_dev                    
                )

            "-------------------------------------------------------------------------------------------"
            "Sim_vowels Test set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_test_set:
                data_subset = 'test'
                sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                    vowel_labels_test,speaker_vt_factor_frame_test,speaker_labels_seq = None, 
                    mu_originals_z = z_mean_test
                )

        if data_training_args.dataset_name == 'timit':
            "-------------------------------------------------------------------------------------------"
            "TIMIT Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                    phonemes39_train,consonants_train,vowels_train, speaker_id_frame_train,
                    mu_originals_z = z_mean_train
                )
            
            "-------------------------------------------------------------------------------------------"
            "TIMIT Dev set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_dev_set:
                data_subset = 'dev'
                timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                    phonemes39_dev,consonants_dev,vowels_dev, speaker_id_frame_dev,
                    mu_originals_z = z_mean_dev
                )

            "-------------------------------------------------------------------------------------------"
            "TIMIT Test set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_test_set:
                data_subset = 'test'
                timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                    phonemes39_test,consonants_test,vowels_test, speaker_id_frame_test,
                    mu_originals_z = z_mean_test
                )

        if data_training_args.dataset_name == 'iemocap':
            "-------------------------------------------------------------------------------------------"
            "IEMOCAP Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                iemocap_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                    phonemes_frame, emotion_labels_frame = emotion_frame, emotion_labels_seq = None, 
                    speaker_labels_frame = speaker_id_frame, speaker_labels_seq = None,
                    mu_originals_z = z_mean
                )

        if data_training_args.dataset_name == 'VOC_ALS':
            "-------------------------------------------------------------------------------------------"
            "VOC_ALS Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                voc_als_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                    phonemes_frame, alsfrs_total_frame,alsfrs_speech_frame,speaker_id_frame, 
                    disease_duration_frame, king_stage_frame, group_frame, cantagallo_frame,
                    z_mean)


if __name__ == "__main__":
    main()