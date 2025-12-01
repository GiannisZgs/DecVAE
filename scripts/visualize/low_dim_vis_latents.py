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

"""This script visualizes latent representations of DecVAE models through the latent_analysis.visualize utility.
Visualization supports frequency embedding and generative factors embedding (colored by labels) in 2D and 3D space.
It also selects one or more of different latent subspace aggregation strategies to visualize.
This script selects a number of instances of a class to avoid cluttered visualizations i.e. in cases where speakers are > 20, we select a few speakers only to visualize.
Decomposition is not supported here so inputs have to be preprocessed from another script for this script to work. 
Subgroups of generative factors are supported for visualization; e.g. in TIMIT we gather consonants and vowels that are subgroups of phonemes.
"""
"""
Dataset sizings:
IEMOCAP: seq - 5530, frame - 300 or 50 for small scale
VOC_ALS: seq - 1224, frame - 300 or 30 for small scale
TIMIT: train seq - 3458, train: 600, test, 192, dev 400 frame / or 50 for small scale
Vowels: seq train: 4000, test: 300, dev: 500 / frame train: 600, test:300, dev:500 / or 30 for small scale
"""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import DecVAEForPreTraining
from config_files import DecVAEConfig
from data_collation import DataCollatorForDecVAELatentVisualization
from args_configs import ModelArgumentsPost, DataTrainingArgumentsPost, DecompositionArguments, TrainingObjectiveArguments, VisualizationsArguments
from utils.misc import parse_args, debugger_is_active, extract_epoch

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from safetensors.torch import load_file
import numpy as np
import re
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

JSON_FILE_NAME_MANUAL = "config_files/DecVAEs/sim_vowels/latent_visualizations/config_latent_frames_visualization_vowels.json" #for debugging purposes only

logger = get_logger(__name__)

def main():
    "Parse the arguments"       
    parser = HfArgumentParser((ModelArgumentsPost, TrainingObjectiveArguments, DecompositionArguments, DataTrainingArgumentsPost, VisualizationArguments))
    if debugger_is_active():
        model_args, training_obj_args, decomp_args, data_training_args, vis_args = parser.parse_json_file(json_file=JSON_FILE_NAME_MANUAL)
    else:
        args = parse_args()
        model_args, training_obj_args, decomp_args, data_training_args, vis_args = parser.parse_json_file(json_file=args.config_file)
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
    if model_args.dual_branched_latent:
        model_type = "dual"
        if training_obj_args.beta_kl_prior_z == 0.1 and training_obj_args.beta_kl_prior_s == 0.1:
            betas = "_bz01" + "_bs01" 
        elif training_obj_args.beta_kl_prior_z == 0.1 and training_obj_args.beta_kl_prior_s != 0.1:
            betas = "_bz01" + "_bs" + str(int(training_obj_args.beta_kl_prior_s))
        elif training_obj_args.beta_kl_prior_z != 0.1 and training_obj_args.beta_kl_prior_s == 0.1:
            betas = "_bz" + str(int(training_obj_args.beta_kl_prior_z)) + "_bs01"
        else:
            betas = "_bz" + str(int(training_obj_args.beta_kl_prior_z)) + "_bs" + str(int(training_obj_args.beta_kl_prior_s))
    elif model_args.only_z_branch:
        model_type = "single_z"
        if training_obj_args.beta_kl_prior_z == 0.1:
            betas = "_bz01"
        else:
            betas = "_bz" + str(int(training_obj_args.beta_kl_prior_z))
    elif model_args.only_s_branch:
        model_type = "single_s"
        if training_obj_args.beta_kl_prior_s == 0.1:
            betas = "_bs01"
        else:
            betas = "_bs" + str(int(training_obj_args.beta_kl_prior_s))
    if "vowels" in data_training_args.dataset_name:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            "snr" + str(data_training_args.sim_snr_db) \
            + betas + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))
    elif data_training_args.dataset_name in ["timit", "iemocap"]:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            betas[1:] + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))
    elif "VOC_ALS" in data_training_args.dataset_name:
        if data_training_args.transfer_from == "timit":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                betas[1:] + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))
        elif data_training_args.transfer_from == "sim_vowels":
            checkpoint_dir = os.path.join(data_training_args.parent_dir,
                "snr" + str(data_training_args.sim_snr_db) \
                + betas + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))

    BETAS = betas[1:]

    if data_training_args.experiment == "ssl_loss":
        checkpoint_dir += "_" + str(data_training_args.ssl_loss_frame_perc) +"percent_frames"

    "Make sure to use all frames in a batch for visualization"
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
        representation_function = DecVAEForPreTraining(config)
        if ckp != 'epoch_-01':
            "No weights, random initialization"

            pretrained_model_file = os.path.join(checkpoint_dir,ckp,"model.safetensors")
        
            weights = load_file(pretrained_model_file)

            keys_to_remove = [key for key in weights.keys() if 'project_hid' in key or 'project_q' in key]
            if keys_to_remove:
                for key in keys_to_remove:
                    del weights[key]
                    print(f"Removed deprecated module {key} from weights.")

            representation_function.load_state_dict(weights)
        
        representation_function.eval()
        for param in representation_function.parameters():
            param.requires_grad = False

        "data collator, optimizer and scheduler"
        mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
        mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

        data_collator = DataCollatorForDecVAELatentVisualization(
            model=representation_function,
            feature_extractor=feature_extractor,
            model_args=model_args,
            data_training_args=data_training_args,
            config=config,
            input_type = data_training_args.input_type,
            dataset_name = data_training_args.dataset_name,
            pad_to_multiple_of=data_training_args.pad_to_multiple_of,
            mask_time_prob=mask_time_prob,
            mask_time_length=mask_time_length
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

        "Measure total loading time"
        start_time = time.time()

        "Get all sets representations"
        with torch.no_grad():
            if vis_args.visualize_train_set:
                "Train set for loop"
                for step, batch in enumerate(train_dataloader):
                    batch_size = batch["input_values"].shape[0]
                    mask_indices_seq_length = batch["input_values"].shape[2]
                    sub_attention_mask = batch.pop("sub_attention_mask", None)
                    overlap_mask_batch = batch.pop("overlap_mask", None)
                    assert overlap_mask_batch != None if data_training_args.dataset_name in ["timit", "iemocap"] else True
                    if hasattr(batch,"reconstruction_NRMSE_seq"):
                        batch.pop("reconstruction_NRMSE_seq", None)
                    if hasattr(batch,"correlograms"):
                        batch.pop("correlograms", None)
                    if hasattr(batch,"correlogram_seq"):
                        batch.pop("correlogram_seq", None)

                    if overlap_mask_batch is None or not data_training_args.discard_label_overlaps:
                        overlap_mask_batch = torch.zeros_like(sub_attention_mask, dtype = torch.bool)
                    else:
                        if "VOC_ALS" in data_training_args.dataset_name:
                            overlap_mask_batch = torch.zeros_like(sub_attention_mask,dtype=torch.bool)
                        "Frames corresponding to padding are set as True in the overlap and discarded"
                        padded = sub_attention_mask.sum(dim = -1)
                        for b in range(batch_size):
                            overlap_mask_batch[b,padded[b]:] = 1
                        overlap_mask_batch = overlap_mask_batch.bool()
                    
                    if "sim_vowels" in data_training_args.dataset_name:
                        batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)                
                        if hasattr(batch,"vowel_labels"):
                            vowel_labels_batch = batch.pop("vowel_labels")
                        if hasattr(batch,"speaker_vt_factor"):
                            speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                        
                        if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                            vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                    elif data_training_args.dataset_name in ["timit", "iemocap"]:
                        batch["mask_time_indices"] = sub_attention_mask.clone()
                        if data_training_args.dataset_name == "timit":
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
                        elif data_training_args.dataset_name == "iemocap":
                            "Phonemes(vowels/consonants)"
                            phonemes_batch = batch.pop("phonemes", None)
                            with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
                                phoneme_to_id = json.load(json_file)
                            id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
                            batch_phonemes = []
                            phonemes = phonemes_batch[~overlap_mask_batch]
                            for ph in phonemes:
                                if ph != -100:
                                    batch_phonemes.append(id_to_phoneme[ph.item()])
                                else:
                                    batch_phonemes.append('SIL')
                          
                            phonemes_batch = np.array(batch_phonemes)

                            "Emotion"
                            emotion_batch = list(batch.pop("emotion", None))

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


                    outputs = representation_function(**batch)
                    del batch

                    if "vowels" in data_training_args.dataset_name:
                        if step == 0:
                            vowel_labels_train = torch.cat([torch.tensor(v) for v in vowel_labels_batch]) 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                vowel_labels_train = torch.cat((vowel_labels_train,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                        if step == 0:
                            speaker_vt_factor_frame_train = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])
                            speaker_vt_factor_seq_train = speaker_vt_factor_batch.clone() 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                speaker_vt_factor_frame_train = torch.cat((speaker_vt_factor_frame_train,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                            if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_train_set_seq_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                speaker_vt_factor_seq_train = torch.cat((speaker_vt_factor_seq_train,speaker_vt_factor_batch),dim = 0) 

                    elif data_training_args.dataset_name in ["timit","iemocap"]:
                        if "timit" in data_training_args.dataset_name:
                            if step == 0:
                                phonemes39_train = phonemes39_batch.copy()
                                vowels_train = vowels_batch.copy()
                                consonants_train = consonants_batch.copy()
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                    phonemes39_train = np.concatenate((phonemes39_train,phonemes39_batch))
                                    vowels_train = np.concatenate((vowels_train,vowels_batch))
                                    consonants_train = np.concatenate((consonants_train,consonants_batch))

                        elif "iemocap" in data_training_args.dataset_name:
                            if step == 0:
                                phonemes_train = phonemes_batch.copy()
                                emotion_frame_train = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)]) 
                                emotion_seq_train = torch.stack(emotion_batch) 
                            else:
                                if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                    phonemes_train = np.concatenate((phonemes_train,phonemes_batch))
                                    emotion_frame_train = torch.cat((emotion_frame_train,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)])),dim = 0)
                                if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_train_set_seq_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                    emotion_seq_train = torch.cat((emotion_seq_train,torch.stack(emotion_batch)),dim = 0)

                        if step == 0:
                            speaker_id_frame_train = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) 
                            speaker_id_seq_train = torch.stack(speaker_id_batch) 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                speaker_id_frame_train = torch.cat((speaker_id_frame_train,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_train_set_seq_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                speaker_id_seq_train = torch.cat((speaker_id_seq_train,torch.stack(speaker_id_batch)),dim = 0)

                    elif "VOC_ALS" in data_training_args.dataset_name:                       
                        for j in range(overlap_mask_batch.shape[0]):
                            for i in range(overlap_mask_batch.shape[1]):
                                if not sub_attention_mask[j,i]:
                                    overlap_mask_batch[j,i] = True
                        #For-loop used here to avoid the problem of concatenating tensors with different lengths
                        if step == 0:
                            "Interpolate sequence-level labels to match frames"
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
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                alsfrs_total_frame = torch.cat((alsfrs_total_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_total_batch)])),dim = 0)
                                alsfrs_speech_frame = torch.cat((alsfrs_speech_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(alsfrs_speech_batch)])),dim = 0)
                                disease_duration_frame = torch.cat((disease_duration_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(disease_duration_batch)])),dim = 0)
                                king_stage_frame = torch.cat((king_stage_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(king_stage_batch)])),dim = 0)
                                cantagallo_frame = torch.cat((cantagallo_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(cantagallo_batch)])),dim = 0)
                                phonemes_frame = torch.cat((phonemes_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(phonemes_batch)])),dim = 0)
                                speaker_id_frame = torch.cat((speaker_id_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                                group_frame = torch.cat((group_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(group_batch)])),dim = 0)
                            if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_train_set_seq_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis):
                                alsfrs_total_seq = torch.cat((alsfrs_total_seq,torch.stack(alsfrs_total_batch)),dim = 0) 
                                alsfrs_speech_seq = torch.cat((alsfrs_speech_seq,torch.stack(alsfrs_speech_batch)),dim = 0) 
                                disease_duration_seq = torch.cat((disease_duration_seq,torch.stack(disease_duration_batch)),dim = 0)
                                king_stage_seq = torch.cat((king_stage_seq,torch.stack(king_stage_batch)),dim = 0)
                                cantagallo_seq = torch.cat((cantagallo_seq,torch.stack(cantagallo_batch)),dim = 0)
                                phonemes_seq = torch.cat((phonemes_seq,torch.stack(phonemes_batch)),dim = 0)
                                speaker_id_seq = torch.cat((speaker_id_seq,torch.stack(speaker_id_batch)),dim = 0)
                                group_seq = torch.cat((group_seq,torch.stack(group_batch)),dim = 0)




                    "Gather latents for evaluations"
                    overlap_mask_batch = overlap_mask_batch[sub_attention_mask]
                    if step == 0:
                        if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                            "Z latents"
                            outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                            mu_components_z_train = outputs.mu_components_z.detach().cpu()
                            outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                            mu_originals_z_train = outputs.mu_originals_z.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                mu_projections_z_train = outputs.mu_projections_z.detach().cpu()

                                
                        if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                            "S latents"
                            mu_components_s_train = outputs.mu_components_s.detach().cpu()
                            mu_originals_s_train = outputs.mu_originals_s.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_s') and config.project_OCs:
                                mu_projections_s_train = outputs.mu_projections_s.detach().cpu()


                    else:
                        if step*batch_size < vis_args.latent_train_set_frames_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis:
                            if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                                "Z latents"
                                outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                                mu_components_z_train = torch.cat((mu_components_z_train,outputs.mu_components_z.detach().cpu()),dim = 1)
                                outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                                mu_originals_z_train = torch.cat((mu_originals_z_train,outputs.mu_originals_z.detach().cpu()),dim = 0)
                                "OCs projection if available"
                                if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                    outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                    mu_projections_z_train = torch.cat((mu_projections_z_train,outputs.mu_projections_z.detach().cpu()),dim = 0)


                        if step*batch_size < vis_args.latent_train_set_seq_to_vis or vis_args.latent_train_set_frames_to_vis == vis_args.latent_train_set_seq_to_vis:
                            if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                                "S latents"
                                mu_components_s_train = torch.cat((mu_components_s_train,outputs.mu_components_s.detach().cpu()),dim = 1)
                                #logvar_components_s = torch.cat((logvar_components_s,outputs.logvar_components_s.detach().cpu()),dim = 1)
                                mu_originals_s_train = torch.cat((mu_originals_s_train,outputs.mu_originals_s.detach().cpu()),dim = 0)
                                "OCs projection if available"
                                if hasattr(outputs,'used_projected_components_s') and config.project_OCs:                        
                                    mu_projections_s_train = torch.cat((mu_projections_s_train,outputs.mu_projections_s.detach().cpu()),dim = 0)
                  
                
                    if not vis_args.visualize_latent_sequence and vis_args.visualize_latent_frame:
                        if step >= vis_args.latent_train_set_frames_to_vis: 
                            break
                    else:
                        if step >= vis_args.latent_train_set_seq_to_vis: 
                            break
            
            if vis_args.visualize_dev_set:
                "Eval set for loop"
                for step, batch in enumerate(eval_dataloader):                   
                    batch_size = batch["input_values"].shape[0]
                    mask_indices_seq_length = batch["input_values"].shape[2]
                    sub_attention_mask = batch.pop("sub_attention_mask", None)
                    overlap_mask_batch = batch.pop("overlap_mask", None)
                    if hasattr(batch,"reconstruction_NRMSE_seq"):
                        batch.pop("reconstruction_NRMSE_seq", None)
                    if hasattr(batch,"correlograms"):
                        batch.pop("correlograms", None)
                    if hasattr(batch,"correlogram_seq"):
                        batch.pop("correlogram_seq", None)
                    assert overlap_mask_batch != None if "timit" in data_training_args.dataset_name else True
                    if overlap_mask_batch is None or not data_training_args.discard_label_overlaps:
                        overlap_mask_batch = torch.zeros_like(sub_attention_mask, dtype = torch.bool)
                    else:
                        "Frames corresponding to padding are set as True in the overlap and discarded"
                        padded = sub_attention_mask.sum(dim = -1)
                        for b in range(batch_size):
                            overlap_mask_batch[b,padded[b]:] = 1
                        overlap_mask_batch = overlap_mask_batch.bool()

                    if "vowels" in data_training_args.dataset_name:
                        batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)
                        if hasattr(batch,"vowel_labels"):
                            vowel_labels_batch = batch.pop("vowel_labels")
                        if hasattr(batch,"speaker_vt_factor"):
                            speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                        
                        if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                            vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                    elif "timit" in data_training_args.dataset_name:
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
                    
                    outputs = representation_function(**batch)
                    del batch

                    if "vowels" in data_training_args.dataset_name:
                        if step == 0:
                            vowel_labels_dev = torch.cat([torch.tensor(v) for v in vowel_labels_batch])
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                vowel_labels_dev = torch.cat((vowel_labels_dev,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                        if step == 0:
                            speaker_vt_factor_frame_dev = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)]) 
                            speaker_vt_factor_seq_dev = speaker_vt_factor_batch.clone()
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                speaker_vt_factor_frame_dev = torch.cat((speaker_vt_factor_frame_dev,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                            if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_dev_set_seq_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                speaker_vt_factor_seq_dev = torch.cat((speaker_vt_factor_seq_dev,speaker_vt_factor_batch),dim = 0) 

                    elif "timit" in data_training_args.dataset_name:
                        if step == 0:
                            phonemes39_dev = phonemes39_batch.copy()
                            vowels_dev = vowels_batch.copy()
                            consonants_dev = consonants_batch.copy()
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                phonemes39_dev = np.concatenate((phonemes39_dev,phonemes39_batch))
                                vowels_dev = np.concatenate((vowels_dev,vowels_batch))
                                consonants_dev = np.concatenate((consonants_dev,consonants_batch))

                        if step == 0:
                            speaker_id_frame_dev = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) 
                            speaker_id_seq_dev = torch.stack(speaker_id_batch) 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_dev_set_frames_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                speaker_id_frame_dev = torch.cat((speaker_id_frame_dev,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_dev_set_seq_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis):
                                speaker_id_seq_dev = torch.cat((speaker_id_seq_dev,torch.stack(speaker_id_batch)),dim = 0) 

                    "Gather latents for evaluations"
                    overlap_mask_batch = overlap_mask_batch[sub_attention_mask]
                    if step == 0:
                        if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                            "Z latents"
                            outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                            mu_components_z_dev = outputs.mu_components_z.detach().cpu()
                            outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                            mu_originals_z_dev = outputs.mu_originals_z.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                mu_projections_z_dev = outputs.mu_projections_z.detach().cpu()

                                
                        if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                            "S latents"
                            mu_components_s_dev = outputs.mu_components_s.detach().cpu()
                            mu_originals_s_dev = outputs.mu_originals_s.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_s') and config.project_OCs:
                                mu_projections_s_dev = outputs.mu_projections_s.detach().cpu()
                            

                    else:
                        if step*batch_size < vis_args.latent_dev_set_frames_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis:
                            if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                                "Z latents"
                                outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                                mu_components_z_dev = torch.cat((mu_components_z_dev,outputs.mu_components_z.detach().cpu()),dim = 1)
                                #logvar_components_z = torch.cat((logvar_components_z,outputs.logvar_components_z.detach().cpu()),dim = 1)
                                outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                                mu_originals_z_dev = torch.cat((mu_originals_z_dev,outputs.mu_originals_z.detach().cpu()),dim = 0)
                                "OCs projection if available"
                                if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                    outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                    mu_projections_z_dev = torch.cat((mu_projections_z_dev,outputs.mu_projections_z.detach().cpu()),dim = 0)


                        if step*batch_size < vis_args.latent_dev_set_seq_to_vis or vis_args.latent_dev_set_frames_to_vis == vis_args.latent_dev_set_seq_to_vis:
                            if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                                "S latents"
                                mu_components_s_dev = torch.cat((mu_components_s_dev,outputs.mu_components_s.detach().cpu()),dim = 1)
                                mu_originals_s_dev = torch.cat((mu_originals_s_dev,outputs.mu_originals_s.detach().cpu()),dim = 0)
                                "OCs projection if available"
                                if hasattr(outputs,'used_projected_components_s') and config.project_OCs:                        
                                    mu_projections_s_dev = torch.cat((mu_projections_s_dev,outputs.mu_projections_s.detach().cpu()),dim = 0)


                    if not vis_args.visualize_latent_sequence and vis_args.visualize_latent_frame:
                        if step >= vis_args.latent_dev_set_frames_to_vis: 
                            break
                    else:
                        if step >= vis_args.latent_dev_set_seq_to_vis: 
                            break

            if vis_args.visualize_test_set:
                "Test set for loop"
                for step, batch in enumerate(test_dataloader):                   
                    batch_size = batch["input_values"].shape[0]
                    mask_indices_seq_length = batch["input_values"].shape[2]
                    sub_attention_mask = batch.pop("sub_attention_mask", None)
                    overlap_mask_batch = batch.pop("overlap_mask", None)
                    if hasattr(batch,"reconstruction_NRMSE_seq"):
                        batch.pop("reconstruction_NRMSE_seq", None)
                    if hasattr(batch,"correlograms"):
                        batch.pop("correlograms", None)
                    if hasattr(batch,"correlogram_seq"):
                        batch.pop("correlogram_seq", None)
                    assert overlap_mask_batch != None if "timit" in data_training_args.dataset_name else True
                    if overlap_mask_batch is None or not data_training_args.discard_label_overlaps:
                        overlap_mask_batch = torch.zeros_like(sub_attention_mask, dtype = torch.bool)
                    else:
                        "Frames corresponding to padding are set as True in the overlap and discarded"
                        padded = sub_attention_mask.sum(dim = -1)
                        for b in range(batch_size):
                            overlap_mask_batch[b,padded[b]:] = 1
                        overlap_mask_batch = overlap_mask_batch.bool()
                    
                    if "vowels" in data_training_args.dataset_name:
                        batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)                
                        if hasattr(batch,"vowel_labels"):
                            vowel_labels_batch = batch.pop("vowel_labels")
                        if hasattr(batch,"speaker_vt_factor"):
                            speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                        
                        if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                            vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                    elif "timit" in data_training_args.dataset_name:
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
                    
                    outputs = representation_function(**batch)
                    del batch

                    if "vowels" in data_training_args.dataset_name:
                        if step == 0:
                            vowel_labels_test = torch.cat([torch.tensor(v) for v in vowel_labels_batch]) 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                vowel_labels_test = torch.cat((vowel_labels_test,torch.cat([torch.tensor(v) for v in vowel_labels_batch])))
                        if step == 0:
                            speaker_vt_factor_frame_test = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)]) 
                            speaker_vt_factor_seq_test = speaker_vt_factor_batch.clone() 
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                speaker_vt_factor_frame_test = torch.cat((speaker_vt_factor_frame_test,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_vt_factor_batch)])),dim = 0)
                            if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_test_set_seq_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                speaker_vt_factor_seq_test = torch.cat((speaker_vt_factor_seq_test,speaker_vt_factor_batch),dim = 0) 

                    elif "timit" in data_training_args.dataset_name:
                        if step == 0:
                            phonemes39_test = phonemes39_batch.copy()
                            vowels_test = vowels_batch.copy()
                            consonants_test = consonants_batch.copy()
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                phonemes39_test = np.concatenate((phonemes39_test,phonemes39_batch))
                                vowels_test = np.concatenate((vowels_test,vowels_batch))
                                consonants_test = np.concatenate((consonants_test,consonants_batch))

                        if step == 0:
                            speaker_id_frame_test = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch) for _ in used_indices[i]])
                            speaker_id_seq_test = torch.stack(speaker_id_batch) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch)])
                        else:
                            if vis_args.visualize_latent_frame and (step*batch_size < vis_args.latent_test_set_frames_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                speaker_id_frame_test = torch.cat((speaker_id_frame_test,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            if vis_args.visualize_latent_sequence and (step*batch_size < vis_args.latent_test_set_seq_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis):
                                speaker_id_seq_test = torch.cat((speaker_id_seq_test,torch.stack(speaker_id_batch)),dim = 0) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch)])

                    "Gather latents for evaluations"
                    overlap_mask_batch = overlap_mask_batch[sub_attention_mask]
                    if step == 0:
                        if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                            "Z latents"
                            outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                            mu_components_z_test = outputs.mu_components_z.detach().cpu()
                            outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                            mu_originals_z_test = outputs.mu_originals_z.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                mu_projections_z_test = outputs.mu_projections_z.detach().cpu()
                            

                        if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                            "S latents"
                            mu_components_s_test = outputs.mu_components_s.detach().cpu()
                            mu_originals_s_test = outputs.mu_originals_s.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_s') and config.project_OCs:
                                mu_projections_s_test = outputs.mu_projections_s.detach().cpu()

                    else:
                        if step*batch_size < vis_args.latent_test_set_frames_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis:
                            if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                                "Z latents"
                                outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                                mu_components_z_test = torch.cat((mu_components_z_test,outputs.mu_components_z.detach().cpu()),dim = 1)
                                #logvar_components_z = torch.cat((logvar_components_z,outputs.logvar_components_z.detach().cpu()),dim = 1)
                                outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                                mu_originals_z_test = torch.cat((mu_originals_z_test,outputs.mu_originals_z.detach().cpu()),dim = 0)
                                "OCs projection if available"
                                if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                    outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                    mu_projections_z_test = torch.cat((mu_projections_z_test,outputs.mu_projections_z.detach().cpu()),dim = 0)


                        if step*batch_size < vis_args.latent_test_set_seq_to_vis or vis_args.latent_test_set_frames_to_vis == vis_args.latent_test_set_seq_to_vis:
                            if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                                "S latents"
                                mu_components_s_test = torch.cat((mu_components_s_test,outputs.mu_components_s.detach().cpu()),dim = 1)
                                #logvar_components_s = torch.cat((logvar_components_s,outputs.logvar_components_s.detach().cpu()),dim = 1)
                                mu_originals_s_test = torch.cat((mu_originals_s_test,outputs.mu_originals_s.detach().cpu()),dim = 0)
                                "OCs projection if available"
                                if hasattr(outputs,'used_projected_components_s'):                        
                                    mu_projections_s_test = torch.cat((mu_projections_s_test,outputs.mu_projections_s.detach().cpu()),dim = 0)
                                
                    if not vis_args.visualize_latent_sequence and vis_args.visualize_latent_frame:
                        if step >= vis_args.latent_test_set_frames_to_vis: 
                            break
                    else:
                        if step >= vis_args.latent_test_set_seq_to_vis: 
                            break

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total loading time: {elapsed_time: .4f} seconds")

        "We have already collected: 1. X latents, 2. OCs latents, 3. OC projected latents"
        "Furthermore, 4. combine components into a joint embedding - Extend labels accordingly"
        "Also 5. combine all embeddings to a single one - X + OCs"
        if (config.dual_branched_latent or config.only_z_branch) and vis_args.visualize_latent_frame:
            if vis_args.visualize_train_set:
                mu_joint_components_z_train = torch.cat([b.reshape(1,-1) for b in mu_components_z_train.transpose(0,1)])
                all_embs = torch.cat([mu_originals_z_train.unsqueeze(0),mu_components_z_train])
                mu_all_z_train = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
                #logvar_joint_components_z = torch.cat([b.reshape(1,-1) for b in logvar_components_z.transpose(0,1)])
            if vis_args.visualize_dev_set:
                mu_joint_components_z_dev = torch.cat([b.reshape(1,-1) for b in mu_components_z_dev.transpose(0,1)])
                all_embs = torch.cat([mu_originals_z_dev.unsqueeze(0),mu_components_z_dev])
                mu_all_z_dev = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
            if vis_args.visualize_test_set:
                mu_joint_components_z_test = torch.cat([b.reshape(1,-1) for b in mu_components_z_test.transpose(0,1)])
                all_embs = torch.cat([mu_originals_z_test.unsqueeze(0),mu_components_z_test])
                mu_all_z_test = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
            
        if (config.dual_branched_latent or config.only_s_branch) and vis_args.visualize_latent_sequence:
            if vis_args.visualize_train_set:
                mu_joint_components_s_train = torch.cat([b.reshape(1,-1) for b in mu_components_s_train.transpose(0,1)])
                all_embs = torch.cat([mu_originals_s_train.unsqueeze(0),mu_components_s_train])
                mu_all_s_train = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
                #logvar_joint_components_s = torch.cat([b.reshape(1,-1) for b in logvar_components_s.transpose(0,1)])
            if vis_args.visualize_dev_set:
                mu_joint_components_s_dev = torch.cat([b.reshape(1,-1) for b in mu_components_s_dev.transpose(0,1)])
                all_embs = torch.cat([mu_originals_s_dev.unsqueeze(0),mu_components_s_dev])
                mu_all_s_dev = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
            if vis_args.visualize_test_set:            
                mu_joint_components_s_test = torch.cat([b.reshape(1,-1) for b in mu_components_s_test.transpose(0,1)])
                all_embs = torch.cat([mu_originals_s_test.unsqueeze(0),mu_components_s_test])
                mu_all_s_test = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
     
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

        
            if vis_that_subset and vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                            
                n_components = 50

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_frame = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_frame_reduced = torch.tensor(pca_OCs_joint_frame.fit_transform(mu_joint_components_z))
                    explained_var_OCs_joint = sum(pca_OCs_joint_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint frame PCA: {explained_var_OCs_joint:.2f}%")
                
                if "all" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_frame = PCA(n_components=n_components, random_state=0)
                    mu_all_frame_reduced = torch.tensor(pca_all_frame.fit_transform(mu_all_z))
                    explained_var_all = sum(pca_all_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding frame PCA: {explained_var_all:.2f}%")


                if "vowel" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Vowel Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                            'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }  
                    
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Vowels & Frequency" 
                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = vowel_labels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = vowel_labels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Vowels & Frequency"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = vowel_labels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Vowel Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    manifold_dict = {
                        'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if vis_args.plot_3d:
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Vowels & Frequency - 3D sphere"

                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = vowel_labels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }    
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Vowels & Frequency"
 
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = vowel_labels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = vowel_labels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        } 
                        if vis_args.plot_3d:
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Vowels & Frequency"    
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = vowel_labels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = vowel_labels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Vowels & Frequency"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = vowel_labels,
                                        target = "vowel",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = vowel_labels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                                )

                "-------------------------------------------------------------------------------------------"
                "Speaker frame"

                "--------------------------------------------------------------------------------------------"
                "2D TSNE Speaker Visualizations"
                "--------------------------------------------------------------------------------------------"
                
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Vowels & Frequency"
 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Vowels & Frequency"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                            )

                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False #already visualized in vowel
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if vis_args.plot_3d:
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Vowels & Frequency"
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'

                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Vowels & Frequency"   
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )

                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }   
                        if vis_args.plot_3d:
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Vowels & Frequency" 
                            
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                                )

                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Vowels & Frequency"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = speaker_labels_frame,
                                        target = "speaker_frame",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                    )
                            
                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                                )



            if vis_that_subset and vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):

                "Try using PCA to see if it gives better visualization"
                n_components = 50

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_seq = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_seq_reduced = torch.tensor(pca_OCs_joint_seq.fit_transform(mu_joint_components_s))
                    explained_var_OCs_joint = sum(pca_OCs_joint_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint seq PCA: {explained_var_OCs_joint:.2f}%")

                if "all" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_seq = PCA(n_components=n_components, random_state=0)
                    mu_all_seq_reduced = torch.tensor(pca_all_seq.fit_transform(mu_all_s))
                    explained_var_all = sum(pca_all_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding seq PCA: {explained_var_all:.2f}%")

                "--------------------------------------------------------------------------------------------------------------"
                "2D TSNE - Speakers Sequence"
                "--------------------------------------------------------------------------------------------------------------"

                data_training_args.frequency_vis = True
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'
                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                init='pca'),
                }  
                if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                    "TSNE - X - OCs - Speakers Sequence"
 
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_s,
                        OCs = mu_components_s,
                        z_or_h = 'z',
                        y_vec = speaker_labels_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                    )

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                    data_training_args.frequency_vis = False
                    "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                    visualize(data_training_args, 
                        config,
                        X = mu_OCs_joint_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = speaker_labels_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                    )

                if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                    data_training_args.frequency_vis = False
                    "TSNE - OCs projection - Speakers Sequence"
                    visualize(data_training_args, 
                        config,
                        X = mu_projections_s,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = speaker_labels_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                    )

                if "all" in vis_args.aggregation_strategies_to_plot_seq:
                    data_training_args.frequency_vis = False
                    "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                    visualize(data_training_args, 
                        config,
                        X = mu_all_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = speaker_labels_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                    )

                "--------------------------------------------------------------------------------------------------------------"
                "3D TSNE - Speakers Sequence"
                "--------------------------------------------------------------------------------------------------------------"

                data_training_args.frequency_vis = True
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                data_training_args.tsne_plot_2d_3d = '3d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                init='pca'),
                } 
                if vis_args.plot_3d:
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        
                        "TSNE - X - OCs - Speakers Sequence"  
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                        )
                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                if vis_args.use_umap:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D UMAP - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.vis_method = 'umap'
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "UMAP - X - OCs - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "UMAP - OCs projection - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D UMAP - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.vis_method = 'umap'
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }                      
                    if vis_args.plot_3d:
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:    
                            "UMAP - X - OCs - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                            )


        def timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_that_subset,
                phoneme_labels = None, consonant_labels = None,vowel_labels = None,speaker_labels_frame = None,speaker_labels_seq = None, 
                mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                mu_joint_components_z = None,mu_all_z = None,
                mu_originals_s = None,mu_components_s = None,mu_projections_s = None,
                mu_joint_components_s = None,mu_all_s = None
                ):

            
            "Select 10-20 speakers to visualize"
            speaker_labels_frame = speaker_labels_frame.detach().cpu().numpy()
            if vis_args.visualize_latent_sequence:
                speaker_labels_seq = speaker_labels_seq.detach().cpu().numpy()
            rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
            if speaker_labels_frame is not None:
                all_speakers = np.unique(speaker_labels_frame)
            else:
                all_speakers = np.array([])
            if vis_args.visualize_latent_sequence:
                if speaker_labels_seq is not None:
                    all_speakers_seq = np.unique(speaker_labels_seq)
                else:
                    all_speakers_seq = np.array([])
            if len(all_speakers) >= 10:
                sel_10_speakers_list = rng.choice(all_speakers, size=10, replace=False)
                sel_10_sp_mask = np.isin(speaker_labels_frame, sel_10_speakers_list)
                sel_10_speakers = speaker_labels_frame[sel_10_sp_mask]
            elif len(all_speakers) > 0 and len(all_speakers) < 10:
                sel_10_speakers = speaker_labels_frame.copy()

            if vis_args.visualize_latent_sequence:
                if len(all_speakers_seq) >= 10:
                    sel_10_speakers_seq_list = rng.choice(all_speakers_seq, size=10, replace=False)
                    sel_10_sp_seq_mask = np.isin(speaker_labels_seq, sel_10_speakers_seq_list)
                    sel_10_speakers_seq = speaker_labels_seq[sel_10_sp_seq_mask]
                elif len(all_speakers_seq) > 0 and len(all_speakers_seq) < 10:
                    sel_10_speakers_seq = speaker_labels_seq.copy()
            
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


            if vis_that_subset and vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                            
                n_components = 50

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_frame = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_frame_reduced = torch.tensor(pca_OCs_joint_frame.fit_transform(mu_joint_components_z))
                    explained_var_OCs_joint = sum(pca_OCs_joint_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint frame PCA: {explained_var_OCs_joint:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_OCs_joint_frame_reduced_sel_speakers = mu_OCs_joint_frame_reduced[sel_10_sp_mask]
                    "Use other masks similarly"  
                    mu_OCs_joint_frame_reduced_sel_phonemes = mu_OCs_joint_frame_reduced[phoneme_mask]
                    mu_OCs_joint_frame_reduced_sel_consonants = mu_OCs_joint_frame_reduced[consonant_mask]
                    mu_OCs_joint_frame_reduced_sel_vowels = mu_OCs_joint_frame_reduced[vowel_mask]

                if "all" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_frame = PCA(n_components=n_components, random_state=0)
                    mu_all_frame_reduced = torch.tensor(pca_all_frame.fit_transform(mu_all_z))
                    explained_var_all = sum(pca_all_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding frame PCA: {explained_var_all:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_all_frame_reduced_sel_speakers = mu_all_frame_reduced[sel_10_sp_mask]
                    "Use other masks similarly" 
                    mu_all_frame_reduced_sel_phonemes = mu_all_frame_reduced[phoneme_mask]
                    mu_all_frame_reduced_sel_consonants = mu_all_frame_reduced[consonant_mask]
                    mu_all_frame_reduced_sel_vowels = mu_all_frame_reduced[vowel_mask]
                    
                "For speakers we need to index using the speaker mask"
                mu_originals_z_sel_speakers = mu_originals_z[sel_10_sp_mask]
                mu_components_z_sel_speakers = mu_components_z[:,sel_10_sp_mask,:]
                if config.project_OCs:
                    mu_projections_z_sel_speakers = mu_projections_z[sel_10_sp_mask]
                "Use other masks similarly" 
                mu_originals_z_sel_phonemes = mu_originals_z[phoneme_mask]
                mu_components_z_sel_phonemes = mu_components_z[:,phoneme_mask,:]
                if config.project_OCs:
                    mu_projections_z_sel_phonemes = mu_projections_z[phoneme_mask]
                mu_originals_z_sel_consonants = mu_originals_z[consonant_mask]
                mu_components_z_sel_consonants = mu_components_z[:,consonant_mask,:]
                if config.project_OCs:
                    mu_projections_z_sel_consonants = mu_projections_z[consonant_mask]
                mu_originals_z_sel_vowels = mu_originals_z[vowel_mask]
                mu_components_z_sel_vowels = mu_components_z[:,vowel_mask,:]
                if config.project_OCs:
                    mu_projections_z_sel_vowels = mu_projections_z[vowel_mask]

                "-------------------------------------------------------------------------------------------"
                "Phoneme frame"
                "-------------------------------------------------------------------------------------------"                
                if "phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Phonemes & Frequency"                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_phonemes,
                            OCs = mu_components_z_sel_phonemes,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme39",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Phonemes "
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme39",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Phonemes"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme39",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Phonemes"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme39",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Phonemes & Frequency - 3D sphere"                      
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_phonemes,
                                OCs = mu_components_z_sel_phonemes,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme39",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Phonemes"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme39",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Phoneme"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme39",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme39",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Phonemes & Frequency"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_phonemes,
                                OCs = mu_components_z_sel_phonemes,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme39",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Phonemes"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme39",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Phonemes"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme39",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme39",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }                               
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Phonemes & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z_sel_phonemes,
                                    OCs = mu_components_z_sel_phonemes,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme39",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Phonemes"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme39",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Phonemes"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z_sel_phonemes,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = sel_phonemes,
                                        target = "phoneme39",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Phonemes"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme39",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "Consonant frame"
                "-------------------------------------------------------------------------------------------"                
                if "consonant" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Consonant Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Consonants & Frequency"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_consonants,
                            OCs = mu_components_z_sel_consonants,
                            z_or_h = 'z',
                            y_vec = sel_consonants,
                            target = "consonant",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','consonants',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Consonants & Frequency"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced_sel_consonants,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_consonants,
                            target = "consonant",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','consonants',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Consonants & Frequency"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z_sel_consonants,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','consonants',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Consonants & Frequency"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced_sel_consonants,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_consonants,
                            target = "consonant",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','consonants',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Consonant Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        } 
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Consonants & Frequency - 3D sphere"
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_consonants,
                                OCs = mu_components_z_sel_consonants,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','consonants',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Consonants & Frequency"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_consonants,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','consonants',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Consonants & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_consonants,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_consonants,
                                    target = "consonant",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','consonants',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Consonants & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_consonants,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','consonants',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Consonant Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        } 
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Consonants & Frequency"    
                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_consonants,
                                OCs = mu_components_z_sel_consonants,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','consonants',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Consonants & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_consonants,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','consonants',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Consonants & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_consonants,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_consonants,
                                    target = "consonant",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','consonants',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Consonants & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_consonants,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_consonants,
                                target = "consonant",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','consonants',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Consonant Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
    
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            } 
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Consonants & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z_sel_consonants,
                                    OCs = mu_components_z_sel_consonants,
                                    z_or_h = 'z',
                                    y_vec = sel_consonants,
                                    target = "consonant",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','consonants',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Consonants & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced_sel_consonants,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_consonants,
                                    target = "consonant",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','consonants',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Consonants & Frequency"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z_sel_consonants,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = sel_consonants,
                                        target = "consonant",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','consonants',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Consonants & Frequency"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced_sel_consonants,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_consonants,
                                    target = "consonant",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','consonants',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "Vowel frame"
                "-------------------------------------------------------------------------------------------"                
                if "vowel" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Vowel Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    } 
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Vowels & Frequency"  
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_vowels,
                            OCs = mu_components_z_sel_vowels,
                            z_or_h = 'z',
                            y_vec = sel_vowels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced_sel_vowels,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_vowels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Vowels & Frequency"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z_sel_vowels,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced_sel_vowels,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_vowels,
                            target = "vowel",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Vowel Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }                      
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Vowels & Frequency - 3D sphere" 
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_vowels,
                                OCs = mu_components_z_sel_vowels,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_vowels,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_vowels,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_vowels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_vowels,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Vowels & Frequency"   
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_vowels,
                                OCs = mu_components_z_sel_vowels,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_vowels,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_vowels,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_vowels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_vowels,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_vowels,
                                target = "vowel",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Vowel Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }  
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Vowels & Frequency"  
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z_sel_vowels,
                                    OCs = mu_components_z_sel_vowels,
                                    z_or_h = 'z',
                                    y_vec = sel_vowels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','vowels',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced_sel_vowels,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_vowels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','vowels',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Vowels & Frequency"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z_sel_vowels,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = sel_vowels,
                                        target = "vowel",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','vowels',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced_sel_vowels,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_vowels,
                                    target = "vowel",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','vowels',data_training_args.vis_method)
                                )

                "-------------------------------------------------------------------------------------------"
                "Speaker frame"
                "-------------------------------------------------------------------------------------------"
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    "2D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Vowels & Frequency" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_speakers,
                            OCs = mu_components_z_sel_speakers,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Vowels & Frequency"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                            )

                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False #already visualized in vowel
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Vowels & Frequency"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z_sel_speakers,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Vowels & Frequency"  
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_speakers,
                                OCs = mu_components_z_sel_speakers,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )

                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }    
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Vowels & Frequency"                                
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z_sel_speakers,
                                    OCs = mu_components_z_sel_speakers,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                                )

                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Vowels & Frequency"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z_sel_speakers,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = sel_10_speakers,
                                        target = "speaker_frame",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                    )
                            
                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                                )


            if vis_that_subset and vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                "-------------------------------------------------------------------------------------------"
                "Speaker sequence"
                "--------------------------------------------------------------------------------------------"

                "Try using PCA to see if it gives better visualization"
                n_components = 50

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_seq = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_seq_reduced = torch.tensor(pca_OCs_joint_seq.fit_transform(mu_joint_components_s))
                    explained_var_OCs_joint = sum(pca_OCs_joint_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint seq PCA: {explained_var_OCs_joint:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_OCs_joint_seq_reduced_sel_speakers = mu_OCs_joint_seq_reduced[sel_10_sp_seq_mask]

                if "all" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_seq = PCA(n_components=n_components, random_state=0)
                    mu_all_seq_reduced = torch.tensor(pca_all_seq.fit_transform(mu_all_s))
                    explained_var_all = sum(pca_all_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding seq PCA: {explained_var_all:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_all_seq_reduced_sel_speakers = mu_all_seq_reduced[sel_10_sp_seq_mask]

                "For speakers we need to index using the speaker mask"
                mu_originals_s_sel_speakers = mu_originals_s[sel_10_sp_seq_mask]
                mu_components_s_sel_speakers = mu_components_s[:,sel_10_sp_seq_mask,:]
                mu_projections_s_sel_speakers = mu_projections_s[sel_10_sp_seq_mask]

                "--------------------------------------------------------------------------------------------------------------"
                "2D TSNE - Speakers Sequence"
                "--------------------------------------------------------------------------------------------------------------"

                data_training_args.frequency_vis = True
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'
                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                init='pca'),
                }  
                if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                    "TSNE - X - OCs - Speakers Sequence" 
                    visualize(data_training_args, 
                        config,
                        X = mu_originals_s_sel_speakers,
                        OCs = mu_components_s_sel_speakers,
                        z_or_h = 'z',
                        y_vec = sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                    )

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                    data_training_args.frequency_vis = False
                    "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                    visualize(data_training_args, 
                        config,
                        X = mu_OCs_joint_seq_reduced_sel_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                    )
                if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                    data_training_args.frequency_vis = False
                    "TSNE - OCs projection - Speakers Sequence"
                    visualize(data_training_args, 
                        config,
                        X = mu_projections_s_sel_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                    )

                if "all" in vis_args.aggregation_strategies_to_plot_seq:
                    data_training_args.frequency_vis = False
                    "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                    visualize(data_training_args, 
                        config,
                        X = mu_all_seq_reduced_sel_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec = sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                    )

                "--------------------------------------------------------------------------------------------------------------"
                "3D TSNE - Speakers Sequence"
                "--------------------------------------------------------------------------------------------------------------"

                data_training_args.frequency_vis = True
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                data_training_args.tsne_plot_2d_3d = '3d'
                data_training_args.vis_method = 'tsne'
                
                if vis_args.plot_3d:
                    manifold_dict = {
                        'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Speakers Sequence" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s_sel_speakers,
                            OCs = mu_components_s_sel_speakers,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                        )
                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                if vis_args.use_umap:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D UMAP - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.vis_method = 'umap'
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "UMAP - X - OCs - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s_sel_speakers,
                            OCs = mu_components_s_sel_speakers,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                        )
                    
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "UMAP - OCs projection - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D UMAP - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.vis_method = 'umap'
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:                            
                            "UMAP - X - OCs - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s_sel_speakers,
                                OCs = mu_components_s_sel_speakers,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False    
                            "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                            )


        def iemocap_latent_vis(config,data_training_args,decomp_args,data_subset,vis_that_subset,
                phoneme_labels = None, emotion_labels_frame = None, emotion_labels_seq = None, speaker_labels_frame = None, speaker_labels_seq = None, 
                mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                mu_joint_components_z = None,mu_all_z = None,
                mu_originals_s = None,mu_components_s = None,mu_projections_s = None,
                mu_joint_components_s = None,mu_all_s = None
                ):

            "Select phonemes to be visualized and remove the NO flags"
            if phoneme_labels is not None:
                phoneme_mask = np.isin(phoneme_labels, vis_args.sel_phonemes_list_iemocap)
                sel_phonemes = phoneme_labels[phoneme_mask]
                non_verbal_mask = np.isin(phoneme_labels, vis_args.sel_non_verbal_phonemes_iemocap)
                sel_non_verbal_phonemes = phoneme_labels[non_verbal_mask]

            if vis_that_subset and vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                            
                n_components = 50

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_frame = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_frame_reduced = torch.tensor(pca_OCs_joint_frame.fit_transform(mu_joint_components_z))
                    explained_var_OCs_joint = sum(pca_OCs_joint_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint frame PCA: {explained_var_OCs_joint:.2f}%")

                    "Use other masks similarly"  
                    mu_OCs_joint_frame_reduced_sel_phonemes = mu_OCs_joint_frame_reduced[phoneme_mask]
                    mu_OCs_joint_frame_reduced_sel_non_verbal_phonemes = mu_OCs_joint_frame_reduced[non_verbal_mask]

                if "all" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_frame = PCA(n_components=n_components, random_state=0)
                    mu_all_frame_reduced = torch.tensor(pca_all_frame.fit_transform(mu_all_z))
                    explained_var_all = sum(pca_all_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding frame PCA: {explained_var_all:.2f}%")

                    "Use other masks similarly" 
                    mu_all_frame_reduced_sel_phonemes = mu_all_frame_reduced[phoneme_mask]
                    mu_all_frame_reduced_sel_non_verbal_phonemes = mu_all_frame_reduced[non_verbal_mask]

                    
                "Use other masks similarly" 
                mu_originals_z_sel_phonemes = mu_originals_z[phoneme_mask]
                mu_components_z_sel_phonemes = mu_components_z[:,phoneme_mask,:]
                if config.project_OCs:
                    mu_projections_z_sel_phonemes = mu_projections_z[phoneme_mask]
                mu_originals_z_sel_non_verbal_phonemes = mu_originals_z[non_verbal_mask]
                mu_components_z_sel_non_verbal_phonemes = mu_components_z[:,non_verbal_mask,:]
                if config.project_OCs:
                    mu_projections_z_sel_non_verbal_phonemes = mu_projections_z[non_verbal_mask]

                "-------------------------------------------------------------------------------------------"
                "Phonemes frame"
                "-------------------------------------------------------------------------------------------"                
                if "phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Phonemes & Frequency"                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_phonemes,
                            OCs = mu_components_z_sel_phonemes,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Phonemes "
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Phonemes"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Phonemes"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced_sel_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_phonemes,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Phonemes & Frequency - 3D sphere"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_phonemes,
                                OCs = mu_components_z_sel_phonemes,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Phonemes"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Phoneme"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Phonemes & Frequency"   
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_phonemes,
                                OCs = mu_components_z_sel_phonemes,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Phonemes"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Phonemes"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_phonemes,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Phonemes & Frequency"                                
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z_sel_phonemes,
                                    OCs = mu_components_z_sel_phonemes,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Phonemes"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Phonemes"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z_sel_phonemes,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = sel_phonemes,
                                        target = "phoneme",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Phonemes"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced_sel_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_phonemes,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                                )

                "-------------------------------------------------------------------------------------------"
                "Non-verbal Phonemes frame"
                "-------------------------------------------------------------------------------------------"                
                if "non_verbal_phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Non-verbal Phonemes & Frequency"                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_non_verbal_phonemes,
                            OCs = mu_components_z_sel_non_verbal_phonemes,
                            z_or_h = 'z',
                            y_vec = sel_non_verbal_phonemes,
                            target = "non_verbal_phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','non_verbal_phonemes',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Non-verbal Phonemes"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced_sel_non_verbal_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_non_verbal_phonemes,
                            target = "non_verbal_phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','non_verbal_phonemes',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Non-verbal Phonemes"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z_sel_non_verbal_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','non_verbal_phonemes',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Non-verbal Phonemes"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced_sel_non_verbal_phonemes,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_non_verbal_phonemes,
                            target = "non_verbal_phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','non_verbal_phonemes',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Non-verbal Phonemes Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Phonemes & Frequency - 3D sphere"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_non_verbal_phonemes,
                                OCs = mu_components_z_sel_non_verbal_phonemes,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','non_verbal_phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Non-verbal Phonemes"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_non_verbal_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','non_verbal_phonemes',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Non-verbal Phoneme"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_non_verbal_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_non_verbal_phonemes,
                                    target = "non_verbal_phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','non_verbal_phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Non-verbal Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_non_verbal_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','non_verbal_phonemes',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Non-Verbal Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs -  Non-Verbal Phoneme & Frequency"  
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_non_verbal_phonemes,
                                OCs = mu_components_z_sel_non_verbal_phonemes,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','non_verbal_phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Non-Verbal Phonemes"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_non_verbal_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','non_verbal_phonemes',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Non-Verbal Phonemes"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_non_verbal_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_non_verbal_phonemes,
                                    target = "non_verbal_phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','non_verbal_phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Non-Verbal Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_non_verbal_phonemes,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_non_verbal_phonemes,
                                target = "non_verbal_phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','non_verbal_phonemes',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Non-Verbal Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Non-Verbal Phoneme & Frequency"  
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z_sel_non_verbal_phonemes,
                                    OCs = mu_components_z_sel_non_verbal_phonemes,
                                    z_or_h = 'z',
                                    y_vec = sel_non_verbal_phonemes,
                                    target = "non_verbal_phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','non_verbal_phonemes',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Non-Verbal Phonemes"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced_sel_non_verbal_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_non_verbal_phonemes,
                                    target = "non_verbal_phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','non_verbal_phonemes',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Non-Verbal Phoneme"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z_sel_non_verbal_phonemes,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = sel_non_verbal_phonemes,
                                        target = "non_verbal_phoneme",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','non_verbal_phonemes',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Non-Verbal Phonemes"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced_sel_non_verbal_phonemes,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_non_verbal_phonemes,
                                    target = "non_verbal_phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','non_verbal_phonemes',data_training_args.vis_method)
                                )
                                
                "-------------------------------------------------------------------------------------------"
                "Speaker frame"
                "-------------------------------------------------------------------------------------------"
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    "2D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'

                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Speakers"                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - OCs joint embedding (concatenation) - Speakers"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Speakers"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                            )

                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Speakers"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_frame,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False #already visualized in vowel
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Speakers"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Speakers"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Speakers"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Speakers"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        } 
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Speakers"   
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Speakers"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Speakers"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )

                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Speakers"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_frame,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }  
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Speakers"   
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Speakers"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                                )

                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Speakers"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = speaker_labels_frame,
                                        target = "speaker_frame",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                    )
                            
                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Speakers"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_frame,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                                )

                "-------------------------------------------------------------------------------------------"
                "Emotion frame"
                "-------------------------------------------------------------------------------------------"
                if "emotion" in vis_args.variables_to_plot_latent:
                    "2D TSNE Emotion Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    } 
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Emotions"  
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = emotion_labels_frame,
                            target = "emotion_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','categorical_emotions',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - OCs joint embedding (concatenation) - Emotions"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = emotion_labels_frame,
                            target = "emotion_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','categorical_emotions',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Emotions"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','categorical_emotions',data_training_args.vis_method)
                            )

                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Emotions"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = emotion_labels_frame,
                            target = "emotion_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','categorical_emotions',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Emotion Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False #already visualized in vowel
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Emotions" 
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','categorical_emotions',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Emotions"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','categorical_emotions',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Emotions"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_frame,
                                    target = "emotion_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','categorical_emotions',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Emotions"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','categorical_emotions',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Emotion Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        } 
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Emotions"    
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','categorical_emotions',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Emotions"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','categorical_emotions',data_training_args.vis_method)

                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Emotions"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_frame,
                                    target = "emotion_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','categorical_emotions',data_training_args.vis_method)
                                )

                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Emotions"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_frame,
                                target = "emotion_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','categorical_emotions',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Emotion Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }  
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Emotions"   
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_frame,
                                    target = "emotion_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','categorical_emotions',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Emotions"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_frame,
                                    target = "emotion_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','categorical_emotions',data_training_args.vis_method)

                                )

                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Emotions"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = emotion_labels_frame,
                                        target = "emotion_frame",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','categorical_emotions',data_training_args.vis_method)
                                    )
                            
                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Emotions"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_frame,
                                    target = "emotion_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','categorical_emotions',data_training_args.vis_method)
                                )


            if vis_that_subset and vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                "-------------------------------------------------------------------------------------------"
                "Speaker sequence"
                "--------------------------------------------------------------------------------------------"

                "Try using PCA to see if it gives better visualization"
                n_components = 50

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_seq = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_seq_reduced = torch.tensor(pca_OCs_joint_seq.fit_transform(mu_joint_components_s))
                    explained_var_OCs_joint = sum(pca_OCs_joint_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint seq PCA: {explained_var_OCs_joint:.2f}%")

                if "all" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_seq = PCA(n_components=n_components, random_state=0)
                    mu_all_seq_reduced = torch.tensor(pca_all_seq.fit_transform(mu_all_s))
                    explained_var_all = sum(pca_all_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding seq PCA: {explained_var_all:.2f}%")


                if "speaker_id" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Speakers Sequence" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = speaker_labels_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - Speakers Sequence"   
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - Speakers Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = speaker_labels_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - Speakers Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                                "UMAP - X - OCs - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = speaker_labels_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                                )

                if "emotion" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - Emotion Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Emotion Sequence"   
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = emotion_labels_seq,
                            target = "emotion_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','emotion_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Emotion Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = emotion_labels_seq,
                            target = "emotion_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','emotion_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Emotion Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = emotion_labels_seq,
                            target = "emotion_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','emotion_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Emotion Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = emotion_labels_seq,
                            target = "emotion_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','emotion_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - Emotion Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - Emotion Sequence" 
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','emotion_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Emotion Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','emotion_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Emotion Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','emotion_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Emotion Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','emotion_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - Emotion Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:   
                            "UMAP - X - OCs - Emotion Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','emotion_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Emotion Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','emotion_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Emotion Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','emotion_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Emotion Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = emotion_labels_seq,
                                target = "emotion_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','emotion_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - Emotion Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:                                
                                "UMAP - X - OCs - Emotion Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_seq,
                                    target = "emotion_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','emotion_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Emotion Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_seq,
                                    target = "emotion_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','emotion_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Emotion Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_seq,
                                    target = "emotion_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','emotion_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - Emotion Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = emotion_labels_seq,
                                    target = "emotion_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','emotion_seq',data_training_args.vis_method)
                                )


        def voc_als_latent_vis(config,data_training_args,decomp_args,data_subset,vis_that_subset,
                phoneme_labels = None, alsfrs_total = None,alsfrs_speech = None, speaker_labels_frame = None,
                disease_duration = None, king_stage = None,group = None, cantagallo = None,
                phoneme_labels_seq = None, alsfrs_total_seq = None, alsfrs_speech_seq = None, speaker_labels_seq = None, 
                disease_duration_seq = None, king_stage_seq = None, group_seq = None, cantagallo_seq = None,                
                mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                mu_joint_components_z = None,mu_all_z = None,
                mu_originals_s = None,mu_components_s = None,mu_projections_s = None,
                mu_joint_components_s = None,mu_all_s = None
                ):

            
            rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
            "VOC-ALS has 153 different speakers - Select 10 to visualize"   
            if speaker_labels_frame is not None:    
                speaker_labels_frame = speaker_labels_frame.detach().cpu().numpy()
                all_speakers = np.unique(speaker_labels_frame)
                if len(all_speakers) >= 10:
                    sel_10_speakers_list = rng.choice(all_speakers, size=10, replace=False)
                    sel_10_sp_mask = np.isin(speaker_labels_frame, sel_10_speakers_list)
                    sel_10_speakers = speaker_labels_frame[sel_10_sp_mask]
                else:
                    sel_10_speakers = speaker_labels_frame.copy()
            if speaker_labels_seq is not None:
                speaker_labels_seq = speaker_labels_seq.detach().cpu().numpy()        
                all_speakers_seq = np.unique(speaker_labels_seq)
                if len(all_speakers_seq) >= 10:
                    sel_10_speakers_seq_list = rng.choice(all_speakers_seq, size=10, replace=False)
                    sel_10_sp_seq_mask = np.isin(speaker_labels_seq, sel_10_speakers_seq_list)
                    sel_10_speakers_seq = speaker_labels_seq[sel_10_sp_seq_mask]
                else:
                    sel_10_speakers_seq = speaker_labels_seq.copy()


            if vis_that_subset and vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch):
                            
                n_components = 100

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_frame = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_frame_reduced = torch.tensor(pca_OCs_joint_frame.fit_transform(mu_joint_components_z))
                    explained_var_OCs_joint = sum(pca_OCs_joint_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint frame PCA: {explained_var_OCs_joint:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_OCs_joint_frame_reduced_sel_speakers = mu_OCs_joint_frame_reduced[sel_10_sp_mask]

                if "all" in vis_args.aggregation_strategies_to_plot_frame:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_frame = PCA(n_components=n_components, random_state=0)
                    mu_all_frame_reduced = torch.tensor(pca_all_frame.fit_transform(mu_all_z))
                    explained_var_all = sum(pca_all_frame.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding frame PCA: {explained_var_all:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_all_frame_reduced_sel_speakers = mu_all_frame_reduced[sel_10_sp_mask]

                "For speakers we need to index using the speaker mask"
                mu_originals_z_sel_speakers = mu_originals_z[sel_10_sp_mask]
                mu_components_z_sel_speakers = mu_components_z[:,sel_10_sp_mask,:]
                if config.project_OCs:
                    mu_projections_z_sel_speakers = mu_projections_z[sel_10_sp_mask]



                "-------------------------------------------------------------------------------------------"
                "Phoneme frame"
                "-------------------------------------------------------------------------------------------"                
                if "phoneme" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Phonemes & Frequency" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = phoneme_labels,
                            target = "phoneme39",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Phonemes "
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = phoneme_labels,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Phonemes"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Phonemes"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = phoneme_labels,
                            target = "phoneme",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Phoneme Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Phonemes & Frequency - 3D sphere"    
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Phonemes"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Phoneme"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Phonemes & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Phonemes"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Phonemes"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Phonemes"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels,
                                target = "phoneme",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Phoneme Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Phonemes & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phonemes',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Phonemes"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phonemes',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Phonemes"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = phoneme_labels,
                                        target = "phoneme",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phonemes',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Phonemes"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels,
                                    target = "phoneme",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phonemes',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "ALSFRS-Total frame"
                "-------------------------------------------------------------------------------------------"                
                if "alsfrs_total" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE ALSFRS-Total Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - ALSFRS-Total & Frequency"                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = alsfrs_total,
                            target = "alsfrs_total",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - ALSFRS-Total"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_total,
                            target = "alsfrs_total",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - ALSFRS-Total"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - ALSFRS-Total"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_total,
                            target = "alsfrs_total",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE ALSFRS-Total Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - ALSFRS-Total & Frequency - 3D sphere"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - ALSFRS-Total"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - ALSFRS-Total"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total,
                                    target = "alsfrs_total",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - ALSFRS-Total"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP ALSFRS-Total Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - ALSFRS-Total & Frequency"  
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - ALSFRS-Total"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - ALSFRS-Total"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total,
                                    target = "alsfrs_total",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - ALSFRS-Total"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total,
                                target = "alsfrs_total",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP ALSFRS-Total Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - ALSFRS-Total & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total,
                                    target = "alsfrs_total",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - ALSFRS-Total"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total,
                                    target = "alsfrs_total",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - ALSFRS-Total"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = alsfrs_total,
                                        target = "alsfrs_total",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - ALSFRS-Total"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total,
                                    target = "alsfrs_total",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "ALSFRS-Speech frame"
                "-------------------------------------------------------------------------------------------"                
                if "alsfrs_speech" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE ALSFRS-Total Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - ALSFRS-Total & Frequency"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech,
                            target = "alsfrs_speech",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - ALSFRS-Total"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech,
                            target = "alsfrs_speech",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - ALSFRS-Total"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - ALSFRS-Total"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech,
                            target = "alsfrs_speech",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE ALSFRS-Total Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - ALSFRS-Total & Frequency - 3D sphere"  
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - ALSFRS-Total"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - ALSFRS-Total"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech,
                                    target = "alsfrs_speech",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - ALSFRS-Total"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP ALSFRS-Total Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }                            
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - ALSFRS-Total & Frequency" 
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - ALSFRS-Total"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - ALSFRS-Total"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech,
                                    target = "alsfrs_speech",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - ALSFRS-Total"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech,
                                target = "alsfrs_speech",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP ALSFRS-Total Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - ALSFRS-Total & Frequency"  
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech,
                                    target = "alsfrs_speech",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - ALSFRS-Total"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech,
                                    target = "alsfrs_speech",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - ALSFRS-Total"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = alsfrs_speech,
                                        target = "alsfrs_speech",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - ALSFRS-Total"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech,
                                    target = "alsfrs_speech",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "Speaker frame"
                "-------------------------------------------------------------------------------------------"
                if "speaker_id" in vis_args.variables_to_plot_latent:
                    "2D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    } 
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Vowels & Frequency"  
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z_sel_speakers,
                            OCs = mu_components_z_sel_speakers,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                        )

                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Vowels & Frequency"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                            )

                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers,
                            target = "speaker_frame",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Speaker Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = False #already visualized in vowel
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Vowels & Frequency"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_speakers,
                                OCs = mu_components_z_sel_speakers,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Vowels & Frequency"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z_sel_speakers,
                                OCs = mu_components_z_sel_speakers,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Vowels & Frequency"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                )

                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers,
                                target = "speaker_frame",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Speaker Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = False #already visualized in vowel
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            } 
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Vowels & Frequency"    
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z_sel_speakers,
                                    OCs = mu_components_z_sel_speakers,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Vowels & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers',data_training_args.vis_method)

                                )

                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Vowels & Frequency"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z_sel_speakers,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = sel_10_speakers,
                                        target = "speaker_frame",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers',data_training_args.vis_method)
                                    )
                            
                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Vowels & Frequency"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers,
                                    target = "speaker_frame",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers',data_training_args.vis_method)
                                )

                "-------------------------------------------------------------------------------------------"
                "Disease Duration frame"
                "-------------------------------------------------------------------------------------------"                
                if "disease_duration" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Disease Duration Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Disease Duration & Frequency"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = disease_duration,
                            target = "disease_duration",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Disease Duration"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = disease_duration,
                            target = "disease_duration",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Disease Duration"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Disease Duration"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = disease_duration,
                            target = "disease_duration",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Disease Duration Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Disease Duration & Frequency - 3D sphere" 
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Disease Duration"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Disease Duration"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = disease_duration,
                                    target = "disease_duration",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Disease Duration"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Disease Duration Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Disease Duration & Frequency"  
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Disease Duration"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Disease Duration"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = disease_duration,
                                    target = "disease_duration",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Disease Duration"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration,
                                target = "disease_duration",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Disease Duration Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Disease Duration & Frequency"                                
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = disease_duration,
                                    target = "disease_duration",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Disease Duration"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = disease_duration,
                                    target = "disease_duration",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Disease Duration"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = disease_duration,
                                        target = "disease_duration",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Disease Duration"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = disease_duration,
                                    target = "disease_duration",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "King's stage frame"
                "-------------------------------------------------------------------------------------------"                
                if "king_stage" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE King's stage Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - King's stage & Frequency"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = king_stage,
                            target = "king_stage",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - King's stage"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = king_stage,
                            target = "king_stage",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - King's stage"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - King's stage"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = king_stage,
                            target = "king_stage",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Disease Duration Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - King's stage & Frequency - 3D sphere"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - King's stage"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - King's stage"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = king_stage,
                                    target = "king_stage",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - King's stage"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP King's stage Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - King's stage & Frequency"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - King's stage"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - King's stage"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = king_stage,
                                    target = "king_stage",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - King's stage"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage,
                                target = "king_stage",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP King's stage Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - King's stage & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = king_stage,
                                    target = "king_stage",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - King's stage"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = king_stage,
                                    target = "king_stage",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - King's stage"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = king_stage,
                                        target = "king_stage",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - King's stage"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = king_stage,
                                    target = "king_stage",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "Disease category / group frame"
                "-------------------------------------------------------------------------------------------"                
                if "group" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Disease category / group Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Disease category / group & Frequency" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = group,
                            target = "group",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Disease category / group"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = group,
                            target = "group",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Disease category / group"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Disease category / group"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = group,
                            target = "group",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Disease Duration Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Disease category / group & Frequency - 3D sphere"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Disease category / group"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Disease category / group"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = group,
                                    target = "group",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Disease category / group"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Disease category / group Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }     
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Disease category / group & Frequency"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Disease category / group"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Disease category / group"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = group,
                                    target = "group",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Disease category / group"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group,
                                target = "group",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Disease category / group Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Disease category / group & Frequency"                                
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = group,
                                    target = "group",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Disease category / group"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = group,
                                    target = "group",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Disease category / group"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = group,
                                        target = "group",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Disease category / group"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = group,
                                    target = "group",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group',data_training_args.vis_method)
                                )

                "-------------------------------------------------------------------------------------------"
                "Cantagallo frame"
                "-------------------------------------------------------------------------------------------"                
                if "cantagallo" in vis_args.variables_to_plot_latent:
                    "--------------------------------------------------------------------------------------------"
                    "2D TSNE Cantagallo Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                        "TSNE - X / OCs - Cantagallo & Frequency"                        
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_z,
                            OCs = mu_components_z,
                            z_or_h = 'z',
                            y_vec = cantagallo,
                            target = "cantagallo",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo',data_training_args.vis_method)
                        )
                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Cantagallo"
                        data_training_args.frequency_vis = False
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = cantagallo,
                            target = "cantagallo",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Cantagallo"
                        if config.project_OCs:
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_z,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo',data_training_args.vis_method)
                            )
                    if "all" in vis_args.aggregation_strategies_to_plot_frame:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Cantagallo"

                        visualize(data_training_args, 
                            config,
                            X = mu_all_frame_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = cantagallo,
                            target = "cantagallo",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo',data_training_args.vis_method)
                        )
                    
                    "--------------------------------------------------------------------------------------------"
                    "3D TSNE Cantagallo Visualizations"
                    "--------------------------------------------------------------------------------------------"
                    data_training_args.frequency_vis = True
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - X / OCs - Cantagallo - 3D sphere"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            "TSNE - OCs joint embedding (concatenation) - Cantagallo"
                            data_training_args.frequency_vis = False
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo',data_training_args.vis_method)
                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Cantagallo"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = cantagallo,
                                    target = "cantagallo",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Cantagallo"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                    
                        "--------------------------------------------------------------------------------------------"
                        "2D UMAP Cantagallo Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '2d'
                        data_training_args.vis_sphere= False
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                            n_neighbors=30,min_dist=0.2,densmap=False)        
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                            "UMAP - X / OCs - Cantagallo & Frequency"                            
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_z,
                                OCs = mu_components_z,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo',data_training_args.vis_method)
                            )
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Cantagallo"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo',data_training_args.vis_method)

                            )
                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Cantagallo"
                            if config.project_OCs:
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_z,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = cantagallo,
                                    target = "cantagallo",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo',data_training_args.vis_method)
                                )
                        if "all" in vis_args.aggregation_strategies_to_plot_frame:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Cantagallo"

                            visualize(data_training_args, 
                                config,
                                X = mu_all_frame_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo,
                                target = "cantagallo",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo',data_training_args.vis_method)
                            )


                        "--------------------------------------------------------------------------------------------"
                        "3D UMAP Cantagallo Visualizations"
                        "--------------------------------------------------------------------------------------------"
                        data_training_args.frequency_vis = True
                        data_training_args.tsne_plot_2d_3d = '3d'
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere

                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=3, random_state=vis_args.random_seed_vis, metric = 'cosine',
                                                n_neighbors=30,min_dist=0.2,densmap=False)        
                            }     
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_frame:
                                "UMAP - X / OCs - Cantagallo & Frequency"                                
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_z,
                                    OCs = mu_components_z,
                                    z_or_h = 'z',
                                    y_vec = cantagallo,
                                    target = "cantagallo",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo',data_training_args.vis_method)
                                )
                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Cantagallo"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = cantagallo,
                                    target = "cantagallo",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo',data_training_args.vis_method)

                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False

                                "UMAP - OCs projection - Cantagallo"
                                if config.project_OCs:
                                    visualize(data_training_args, 
                                        config,
                                        X = mu_projections_z,
                                        OCs = None,
                                        z_or_h = 'z',
                                        y_vec = cantagallo,
                                        target = "cantagallo",
                                        data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                        manifold_dict = manifold_dict,
                                        return_data = True,
                                        display_figures = True,
                                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo',data_training_args.vis_method)
                                    )

                            if "all" in vis_args.aggregation_strategies_to_plot_frame:
                                data_training_args.frequency_vis = False
                                "UMAP - All / total embedding (X + OCs) - Cantagallo"

                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_frame_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = cantagallo,
                                    target = "cantagallo",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' + str(vis_args.latent_train_set_frames_to_vis) + '_frames',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo',data_training_args.vis_method)
                                )


            if vis_that_subset and vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch):
                
                "Try using PCA to see if it gives better visualization"
                n_components = 50

                if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on joint/concatenated OCs - Use as X"
                    pca_OCs_joint_seq = PCA(n_components=n_components, random_state=0)
                    mu_OCs_joint_seq_reduced = torch.tensor(pca_OCs_joint_seq.fit_transform(mu_joint_components_s))
                    explained_var_OCs_joint = sum(pca_OCs_joint_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for OCs joint seq PCA: {explained_var_OCs_joint:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_OCs_joint_seq_reduced_sel_speakers = mu_OCs_joint_seq_reduced[sel_10_sp_seq_mask]

                if "all" in vis_args.aggregation_strategies_to_plot_seq:
                    "PCA on All / total embedding (X + OCs) - Use as X"
                    pca_all_seq = PCA(n_components=n_components, random_state=0)
                    mu_all_seq_reduced = torch.tensor(pca_all_seq.fit_transform(mu_all_s))
                    explained_var_all = sum(pca_all_seq.explained_variance_ratio_) * 100
                    print(f"Explained variance for total embedding seq PCA: {explained_var_all:.2f}%")

                    "For speakers we need to index using the speaker mask"
                    mu_all_seq_reduced_sel_speakers = mu_all_seq_reduced[sel_10_sp_seq_mask]

                "For speakers we need to index using the speaker mask"
                mu_originals_s_sel_speakers = mu_originals_s[sel_10_sp_seq_mask]
                mu_components_s_sel_speakers = mu_components_s[:,sel_10_sp_seq_mask,:]
                if config.project_OCs:
                    mu_projections_s_sel_speakers = mu_projections_s[sel_10_sp_seq_mask]


                "-------------------------------------------------------------------------------------------"
                "Speaker sequence"
                "--------------------------------------------------------------------------------------------"
                if "speaker_id" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Speakers Sequence" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s_sel_speakers,
                            OCs = mu_components_s_sel_speakers,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced_sel_speakers,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = sel_10_speakers_seq,
                            target = "speaker_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - Speakers Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s_sel_speakers,
                                OCs = mu_components_s_sel_speakers,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - Speakers Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s_sel_speakers,
                                OCs = mu_components_s_sel_speakers,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced_sel_speakers,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = sel_10_speakers_seq,
                                target = "speaker_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - Speakers Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                                "UMAP - X - OCs - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s_sel_speakers,
                                    OCs = mu_components_s_sel_speakers,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','speakers_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','speakers_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','speakers_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - Speakers Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced_sel_speakers,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = sel_10_speakers_seq,
                                    target = "speaker_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','speakers_seq',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "Phoneme sequence"                
                "--------------------------------------------------------------------------------------------"
                if "phoneme" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - Phoneme Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Phoneme Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = phoneme_labels_seq,
                            target = "phoneme_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phoneme_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Phoneme Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = phoneme_labels_seq,
                            target = "phoneme_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phoneme_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Phoneme Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = phoneme_labels_seq,
                            target = "phoneme_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phoneme_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Phoneme Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = phoneme_labels_seq,
                            target = "phoneme_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phoneme_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - Phoneme Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - Phoneme Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phoneme_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Phoneme Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phoneme_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Phoneme Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phoneme_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Phoneme Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phoneme_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - Phoneme Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - Phoneme Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phoneme_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Phoneme Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phoneme_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Phoneme Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phoneme_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Phoneme Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = phoneme_labels_seq,
                                target = "phoneme_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phoneme_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - Phoneme Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:                                
                                "UMAP - X - OCs - Phoneme Sequence & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels_seq,
                                    target = "phoneme_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','phoneme_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Phoneme Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels_seq,
                                    target = "phoneme_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','phoneme_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Phoneme Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels_seq,
                                    target = "phoneme_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','phoneme_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - Phoneme Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = phoneme_labels_seq,
                                    target = "phoneme_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','phoneme_seq',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "ALSFRS-Total sequence"                
                "--------------------------------------------------------------------------------------------"
                if "alsfrs_total" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - ALSFRS-Total Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - ALSFRS-Total Sequence" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = alsfrs_total_seq,
                            target = "alsfrs_total_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - ALSFRS-Total Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_total_seq,
                            target = "alsfrs_total_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Phoneme Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_total_seq,
                            target = "alsfrs_total_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - ALSFRS-Total Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_total_seq,
                            target = "alsfrs_total_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - ALSFRS-Total Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - ALSFRS-Total Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - ALSFRS-Total Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - ALSFRS-Total Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - ALSFRS-Total Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - ALSFRS-Total Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - ALSFRS-Total Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - ALSFRS-Total Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - ALSFRS-Total Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - ALSFRS-Total Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_total_seq,
                                target = "alsfrs_total_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - ALSFRS-Total Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }  
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:                                
                                "UMAP - X - OCs - ALSFRS-Total Sequence & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total_seq,
                                    target = "alsfrs_total_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_total_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - ALSFRS-Total Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total_seq,
                                    target = "alsfrs_total_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - ALSFRS-Total Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total_seq,
                                    target = "alsfrs_total_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_total_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - ALSFRS-Total Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_total_seq,
                                    target = "alsfrs_total_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_total_seq',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "ALSFRS-Speech sequence"                
                "--------------------------------------------------------------------------------------------"
                if "alsfrs_speech" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - ALSFRS-Speech Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - ALSFRS-Speech Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech_seq,
                            target = "alsfrs_speech_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - ALSFRS-Speech Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech_seq,
                            target = "alsfrs_speech_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - ALSFRS-Speech Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech_seq,
                            target = "alsfrs_speech_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - ALSFRS-Speech Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = alsfrs_speech_seq,
                            target = "alsfrs_speech_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - ALSFRS-Speech Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - ALSFRS-Speech Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - ALSFRS-Speech Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - ALSFRS-Speech Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - ALSFRS-Speech Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - ALSFRS-Speech Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq: 
                            "UMAP - X - OCs - ALSFRS-Speech Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - ALSFRS-Speech Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - ALSFRS-Speech Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - ALSFRS-Speech Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = alsfrs_speech_seq,
                                target = "alsfrs_speech_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - ALSFRS-Speech Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                                "UMAP - X - OCs - ALSFRS-Speech Sequence & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech_seq,
                                    target = "alsfrs_speech_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','alsfrs_speech_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - ALSFRS-Speech Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech_seq,
                                    target = "alsfrs_speech_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - ALSFRS-Speech Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech_seq,
                                    target = "alsfrs_speech_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','alsfrs_speech_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - ALSFRS-Speech Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = alsfrs_speech_seq,
                                    target = "alsfrs_speech_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','alsfrs_speech_seq',data_training_args.vis_method)
                                )

                
                "-------------------------------------------------------------------------------------------"
                "Disease Duration sequence"                
                "--------------------------------------------------------------------------------------------"
                if "disease_duration" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - Disease Duration Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Disease Duration Sequence" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = disease_duration_seq,
                            target = "disease_duration_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Disease Duration Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = disease_duration_seq,
                            target = "disease_duration_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Disease Duration Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = disease_duration_seq,
                            target = "disease_duration_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Disease Duration Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = disease_duration_seq,
                            target = "disease_duration_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - Disease Duration Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - Disease Duration Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Disease Duration Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Disease Duration Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Disease Duration Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - Disease Duration Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - Disease Duration Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Disease Duration Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Disease Duration Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Disease Duration Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = disease_duration_seq,
                                target = "disease_duration_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - Disease Duration Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:                                
                                "UMAP - X - OCs - Disease Duration Sequence & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = disease_duration_seq,
                                    target = "disease_duration_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','disease_duration_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Disease Duration Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = disease_duration_seq,
                                    target = "disease_duration_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','disease_duration_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Disease Duration Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = disease_duration_seq,
                                    target = "disease_duration_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','disease_duration_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - Disease Duration Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = disease_duration_seq,
                                    target = "disease_duration_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','disease_duration_seq',data_training_args.vis_method)
                                )

                
                "-------------------------------------------------------------------------------------------"
                "King's stage sequence"                
                "--------------------------------------------------------------------------------------------"
                if "king_stage" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - King's stage Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                            'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    } 
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - King's stage Sequence"  
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = king_stage_seq,
                            target = "king_stage_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - King's stage Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = king_stage_seq,
                            target = "king_stage_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - King's stage Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = king_stage_seq,
                            target = "king_stage_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - King's stage Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = king_stage_seq,
                            target = "king_stage_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - King's stage Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - King's stage Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - King's stage Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - King's stage Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - King's stage Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - King's stage Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - King's stage Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - King's stage Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - King's stage Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - King's stage Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = king_stage_seq,
                                target = "king_stage_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - King's stage Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                                "UMAP - X - OCs - King's stage Sequence & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = king_stage_seq,
                                    target = "king_stage_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','king_stage_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - King's stage Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = king_stage_seq,
                                    target = "king_stage_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','king_stage_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - King's stage Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = king_stage_seq,
                                    target = "king_stage_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','king_stage_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - King's stage Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = king_stage_seq,
                                    target = "king_stage_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','king_stage_seq',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "Disease category / group sequence"                
                "--------------------------------------------------------------------------------------------"
                if "group" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - Disease category / group Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }   
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Disease category / group Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = group_seq,
                            target = "group_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Disease category / group Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = group_seq,
                            target = "group_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Disease category / group Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = group_seq,
                            target = "group_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Disease category / group Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = group_seq,
                            target = "group_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - Disease category / group Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }  
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - Disease category / group Sequence & Frequency" 
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Disease category / group Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Disease category / group Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Disease category / group Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - Disease category / group Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - Disease category / group Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Disease category / group Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Disease category / group Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Disease category / group Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = group_seq,
                                target = "group_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - Disease category / group Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:                                
                                "UMAP - X - OCs - Disease category / group Sequence & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = group_seq,
                                    target = "group_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','group_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Disease category / group Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = group_seq,
                                    target = "group_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','group_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Disease category / group Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = group_seq,
                                    target = "group_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','group_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - Disease category / group Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = group_seq,
                                    target = "group_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','group_seq',data_training_args.vis_method)
                                )


                "-------------------------------------------------------------------------------------------"
                "Cantagallo sequence"                
                "--------------------------------------------------------------------------------------------"
                if "cantagallo" in vis_args.variables_to_plot_latent_seq:
                    "--------------------------------------------------------------------------------------------------------------"
                    "2D TSNE - Cantagallo Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= False
                    data_training_args.tsne_plot_2d_3d = '2d'
                    data_training_args.vis_method = 'tsne'
                    manifold_dict = {
                        'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                    max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                    init='pca'),
                    }  
                    if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                        "TSNE - X - OCs - Cantagallo Sequence" 
                        visualize(data_training_args, 
                            config,
                            X = mu_originals_s,
                            OCs = mu_components_s,
                            z_or_h = 'z',
                            y_vec = cantagallo_seq,
                            target = "cantagallo_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo_seq',data_training_args.vis_method)
                        )

                    if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs joint embedding (concatenation) - Cantagallo Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_OCs_joint_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = cantagallo_seq,
                            target = "cantagallo_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo_seq',data_training_args.vis_method)
                        )
                    if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - OCs projection - Cantagallo Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_projections_s,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = cantagallo_seq,
                            target = "cantagallo_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo_seq',data_training_args.vis_method)
                        )

                    if "all" in vis_args.aggregation_strategies_to_plot_seq:
                        data_training_args.frequency_vis = False
                        "TSNE - All / total embedding (X + OCs) - Cantagallo Sequence"
                        visualize(data_training_args, 
                            config,
                            X = mu_all_seq_reduced,
                            OCs = None,
                            z_or_h = 'z',
                            y_vec = cantagallo_seq,
                            target = "cantagallo_seq",
                            data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                            manifold_dict = manifold_dict,
                            return_data = True,
                            display_figures = True,
                            save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo_seq',data_training_args.vis_method)
                        )

                    "--------------------------------------------------------------------------------------------------------------"
                    "3D TSNE - Cantagallo Sequence"
                    "--------------------------------------------------------------------------------------------------------------"

                    data_training_args.frequency_vis = True
                    data_training_args.generative_factors_vis= True
                    data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                    data_training_args.tsne_plot_2d_3d = '3d'
                    data_training_args.vis_method = 'tsne'
                    
                    if vis_args.plot_3d:
                        manifold_dict = {
                            'tsne': TSNE(n_components=3, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                        max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                                        init='pca'),
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "TSNE - X - OCs - Cantagallo Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo_seq',data_training_args.vis_method)
                            )

                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs joint embedding (concatenation) - Cantagallo Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - OCs projection - Cantagallo Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo_seq',data_training_args.vis_method)
                            )
                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "TSNE - All / total embedding (X + OCs) - Cantagallo Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo_seq',data_training_args.vis_method)
                            )

                    if vis_args.use_umap:
                        "--------------------------------------------------------------------------------------------------------------"
                        "2D UMAP - Cantagallo Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= False
                        data_training_args.tsne_plot_2d_3d = '2d'
                        manifold_dict = {
                            'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                        }   
                        if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:
                            "UMAP - X - OCs - Cantagallo Sequence & Frequency"
                            visualize(data_training_args, 
                                config,
                                X = mu_originals_s,
                                OCs = mu_components_s,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo_seq',data_training_args.vis_method)
                            )
                        
                        if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs joint embedding (concatenation) - Cantagallo Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_OCs_joint_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo_seq',data_training_args.vis_method)
                            )

                        if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - OCs projection - Cantagallo Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_projections_s,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo_seq',data_training_args.vis_method)
                            )

                        if "all" in vis_args.aggregation_strategies_to_plot_seq:
                            data_training_args.frequency_vis = False
                            "UMAP - All / total embedding (X + OCs) - Cantagallo Sequence"
                            visualize(data_training_args, 
                                config,
                                X = mu_all_seq_reduced,
                                OCs = None,
                                z_or_h = 'z',
                                y_vec = cantagallo_seq,
                                target = "cantagallo_seq",
                                data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                manifold_dict = manifold_dict,
                                return_data = True,
                                display_figures = True,
                                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo_seq',data_training_args.vis_method)
                            )

                        "--------------------------------------------------------------------------------------------------------------"
                        "3D UMAP - Cantagallo Sequence"
                        "--------------------------------------------------------------------------------------------------------------"

                        data_training_args.vis_method = 'umap'
                        data_training_args.frequency_vis = True
                        data_training_args.generative_factors_vis= True
                        data_training_args.vis_sphere= vis_args.vis_isotropic_gaussian_sphere
                        data_training_args.tsne_plot_2d_3d = '3d'
                        
                        if vis_args.plot_3d:
                            manifold_dict = {
                                'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                            }   
                            if "X_OCs_freq" in vis_args.aggregation_strategies_to_plot_seq:                                
                                "UMAP - X - OCs - Cantagallo Sequence & Frequency"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_originals_s,
                                    OCs = mu_components_s,
                                    z_or_h = 'z',
                                    y_vec = cantagallo_seq,
                                    target = "cantagallo_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'X_OCs','cantagallo_seq',data_training_args.vis_method)
                                )

                            if "OCs_joint" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs joint embedding (concatenation) - Cantagallo Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_OCs_joint_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = cantagallo_seq,
                                    target = "cantagallo_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_joint_emb','cantagallo_seq',data_training_args.vis_method)
                                )
                            if "OCs_proj" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False
                                "UMAP - OCs projection - Cantagallo Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_projections_s,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = cantagallo_seq,
                                    target = "cantagallo_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'OCs_projection','cantagallo_seq',data_training_args.vis_method)
                                )
                            if "all" in vis_args.aggregation_strategies_to_plot_seq:
                                data_training_args.frequency_vis = False    
                                "UMAP - All / total embedding (X + OCs) - Cantagallo Sequence"
                                visualize(data_training_args, 
                                    config,
                                    X = mu_all_seq_reduced,
                                    OCs = None,
                                    z_or_h = 'z',
                                    y_vec = cantagallo_seq,
                                    target = "cantagallo_seq",
                                    data_set = data_training_args.dataset_name + '_' + data_subset + '_' +  str(vis_args.latent_train_set_seq_to_vis) + '_seqs',
                                    manifold_dict = manifold_dict,
                                    return_data = True,
                                    display_figures = True,
                                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,data_training_args.dataset_name,BETAS,data_subset,'all_joint_emb','cantagallo_seq',data_training_args.vis_method)
                                )



        if data_training_args.dataset_name == 'sim_vowels':
            "-------------------------------------------------------------------------------------------"
            "Sim_vowels Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                        vowel_labels_train,speaker_vt_factor_frame_train,speaker_vt_factor_seq_train, 
                        mu_originals_z_train,mu_components_z_train,mu_projections_z_train,
                        mu_joint_components_z_train,mu_all_z_train,
                        mu_originals_s_train, mu_components_s_train, mu_projections_s_train ,
                        mu_joint_components_s_train ,mu_all_s_train
                        )
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                                    vowel_labels_train,speaker_vt_factor_frame_train,speaker_labels_seq = None, 
                                    mu_originals_z = mu_originals_z_train,mu_components_z =  mu_components_z_train,mu_projections_z = mu_projections_z_train,
                                    mu_joint_components_z = mu_joint_components_z_train,mu_all_z = mu_all_z_train
                                    )
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                                    vowel_labels = None,speaker_labels_frame = None,speaker_labels_seq = speaker_vt_factor_seq_train, 
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_train,mu_components_s = mu_components_s_train,mu_projections_s = mu_projections_s_train,
                                    mu_joint_components_s = mu_joint_components_s_train,mu_all_s = mu_all_s_train
                                    )

            "-------------------------------------------------------------------------------------------"
            "Sim_vowels Dev set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_dev_set:
                data_subset = 'dev'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                        vowel_labels_dev,speaker_vt_factor_frame_dev,speaker_vt_factor_seq_dev, 
                        mu_originals_z_dev,mu_components_z_dev,mu_projections_z_dev,
                        mu_joint_components_z_dev,mu_all_z_dev,
                        mu_originals_s_dev, mu_components_s_dev, mu_projections_s_dev,
                        mu_joint_components_s_dev,mu_all_s_dev
                        )
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                                    vowel_labels_dev,speaker_vt_factor_frame_dev,speaker_labels_seq = None, 
                                    mu_originals_z = mu_originals_z_dev,mu_components_z =  mu_components_z_dev,mu_projections_z = mu_projections_z_dev,
                                    mu_joint_components_z = mu_joint_components_z_dev,mu_all_z = mu_all_z_dev
                                    )
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                                    vowel_labels = None,speaker_labels_frame = None,speaker_labels_seq = speaker_vt_factor_seq_dev, 
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_dev,mu_components_s = mu_components_s_dev,mu_projections_s = mu_projections_s_dev,
                                    mu_joint_components_s = mu_joint_components_s_dev,mu_all_s = mu_all_s_dev
                                    )


            "-------------------------------------------------------------------------------------------"
            "Sim_vowels Test set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_test_set:
                data_subset = 'test'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                        vowel_labels_test,speaker_vt_factor_frame_test,speaker_vt_factor_seq_test, 
                        mu_originals_z_test,mu_components_z_test,mu_projections_z_test,
                        mu_joint_components_z_test,mu_all_z_test,
                        mu_originals_s_test, mu_components_s_test, mu_projections_s_test,
                        mu_joint_components_s_test,mu_all_s_test
                        )
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                                    vowel_labels_test,speaker_vt_factor_frame_test,speaker_labels_seq = None, 
                                    mu_originals_z = mu_originals_z_test,mu_components_z =  mu_components_z_test,mu_projections_z = mu_projections_z_test,
                                    mu_joint_components_z = mu_joint_components_z_test,mu_all_z = mu_all_z_test
                                    )
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    sim_vowels_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                                    vowel_labels = None,speaker_labels_frame = None,speaker_labels_seq = speaker_vt_factor_seq_test, 
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_test,mu_components_s = mu_components_s_test,mu_projections_s = mu_projections_s_test,
                                    mu_joint_components_s = mu_joint_components_s_test,mu_all_s = mu_all_s_test
                                    )

        elif data_training_args.dataset_name == 'timit':
            "-------------------------------------------------------------------------------------------"
            "TIMIT Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                        phonemes39_train, consonants_train, vowels_train, speaker_id_frame_train, speaker_id_seq_train, 
                        mu_originals_z_train,mu_components_z_train,mu_projections_z_train if config.project_OCs else None,
                        mu_joint_components_z_train,mu_all_z_train,
                        mu_originals_s_train, mu_components_s_train, mu_projections_s_train if config.project_OCs else None,
                        mu_joint_components_s_train ,mu_all_s_train
                        )
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                                    phonemes39_train, consonants_train, vowels_train, speaker_id_frame_train, speaker_labels_seq = None, 
                                    mu_originals_z = mu_originals_z_train,mu_components_z =  mu_components_z_train,mu_projections_z = mu_projections_z_train if config.project_OCs else None,
                                    mu_joint_components_z = mu_joint_components_z_train,mu_all_z = mu_all_z_train
                                    )
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                                    phoneme_labels = None, consonant_labels = None, vowel_labels = None, speaker_labels_frame = None, speaker_labels_seq = speaker_id_seq_train, 
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_train,mu_components_s = mu_components_s_train,mu_projections_s = mu_projections_s_train if config.project_OCs else None,
                                    mu_joint_components_s = mu_joint_components_s_train,mu_all_s = mu_all_s_train
                                    )

            "-------------------------------------------------------------------------------------------"
            "TIMIT Dev set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_dev_set:
                data_subset = 'dev'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                        phonemes39_dev, consonants_dev, vowels_dev,speaker_id_frame_dev,speaker_id_seq_dev, 
                        mu_originals_z_dev,mu_components_z_dev,mu_projections_z_dev if config.project_OCs else None,
                        mu_joint_components_z_dev,mu_all_z_dev,
                        mu_originals_s_dev, mu_components_s_dev, mu_projections_s_dev if config.project_OCs else None,
                        mu_joint_components_s_dev,mu_all_s_dev
                        )
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                                    phonemes39_dev, consonants_dev, vowels_dev,speaker_id_frame_dev,speaker_labels_seq = None, 
                                    mu_originals_z = mu_originals_z_dev,mu_components_z =  mu_components_z_dev,mu_projections_z = mu_projections_z_dev if config.project_OCs else None,
                                    mu_joint_components_z = mu_joint_components_z_dev,mu_all_z = mu_all_z_dev
                                    )
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_dev_set,
                                    vowel_labels = None,speaker_labels_frame = None,speaker_labels_seq = speaker_id_seq_dev, 
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_dev,mu_components_s = mu_components_s_dev,mu_projections_s = mu_projections_s_dev if config.project_OCs else None,
                                    mu_joint_components_s = mu_joint_components_s_dev,mu_all_s = mu_all_s_dev
                                    )

            "-------------------------------------------------------------------------------------------"
            "TIMIT Test set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_test_set:
                data_subset = 'test'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                        phonemes39_test, consonants_test, vowels_test,speaker_id_frame_test,speaker_id_seq_test, 
                        mu_originals_z_test,mu_components_z_test,mu_projections_z_test if config.project_OCs else None,
                        mu_joint_components_z_test,mu_all_z_test,
                        mu_originals_s_test, mu_components_s_test, mu_projections_s_test if config.project_OCs else None,
                        mu_joint_components_s_test,mu_all_s_test
                        )
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                                    phonemes39_test, consonants_test, vowels_test,speaker_id_frame_test,speaker_labels_seq = None, 
                                    mu_originals_z = mu_originals_z_test,mu_components_z =  mu_components_z_test,mu_projections_z = mu_projections_z_test if config.project_OCs else None,
                                    mu_joint_components_z = mu_joint_components_z_test,mu_all_z = mu_all_z_test
                                    )
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    timit_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_test_set,
                                    vowel_labels = None,speaker_labels_frame = None,speaker_labels_seq = speaker_id_seq_test, 
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_test,mu_components_s = mu_components_s_test,mu_projections_s = mu_projections_s_test if config.project_OCs else None,
                                    mu_joint_components_s = mu_joint_components_s_test,mu_all_s = mu_all_s_test
                                    )

        elif data_training_args.dataset_name == 'iemocap':
            "-------------------------------------------------------------------------------------------"
            "IEMOCAP Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    iemocap_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                        phonemes_train, emotion_frame_train, emotion_seq_train, speaker_id_frame_train, speaker_id_seq_train, 
                        mu_originals_z_train,mu_components_z_train,mu_projections_z_train if config.project_OCs else None,
                        mu_joint_components_z_train,mu_all_z_train,
                        mu_originals_s_train, mu_components_s_train, mu_projections_s_train if config.project_OCs else None,
                        mu_joint_components_s_train ,mu_all_s_train
                        )
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    iemocap_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                                    phonemes_train, emotion_frame_train, emotion_labels_seq = None, speaker_labels_frame = speaker_id_frame_train, speaker_labels_seq = None, 
                                    mu_originals_z = mu_originals_z_train,mu_components_z =  mu_components_z_train,mu_projections_z = mu_projections_z_train if config.project_OCs else None,
                                    mu_joint_components_z = mu_joint_components_z_train,mu_all_z = mu_all_z_train
                                    )
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    iemocap_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                                    phoneme_labels = None, emotion_labels_frame = None, emotion_labels_seq = emotion_seq_train, speaker_labels_frame = None, speaker_labels_seq = speaker_id_seq_train, 
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_train,mu_components_s = mu_components_s_train,mu_projections_s = mu_projections_s_train if config.project_OCs else None,
                                    mu_joint_components_s = mu_joint_components_s_train,mu_all_s = mu_all_s_train
                                    )
                    

        elif data_training_args.dataset_name == 'VOC_ALS':
            "-------------------------------------------------------------------------------------------"
            "VOC_ALS Train set visualization"
            "-------------------------------------------------------------------------------------------"
            if vis_args.visualize_train_set:
                data_subset = 'train'
                if vis_args.visualize_latent_frame and config.dual_branched_latent and vis_args.visualize_latent_sequence:
                    voc_als_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                        phonemes_frame, alsfrs_total_frame, alsfrs_speech_frame, speaker_id_frame, 
                        disease_duration_frame, king_stage_frame, group_frame, cantagallo_frame,
                        phonemes_seq, alsfrs_total_seq, alsfrs_speech_seq, speaker_id_seq, 
                        disease_duration_seq, king_stage_seq, group_seq, cantagallo_seq,                
                        mu_originals_z_train,mu_components_z_train,mu_projections_z_train if config.project_OCs else None,
                        mu_joint_components_z_train,mu_all_z_train,
                        mu_originals_s_train, mu_components_s_train, mu_projections_s_train if config.project_OCs else None,
                        mu_joint_components_s_train ,mu_all_s_train
                        )
                    
                if vis_args.visualize_latent_frame and (config.dual_branched_latent or config.only_z_branch) and not vis_args.visualize_latent_sequence:
                    voc_als_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                        phonemes_frame, alsfrs_total_frame, alsfrs_speech_frame, speaker_id_frame, 
                        disease_duration_frame, king_stage_frame, group_frame, cantagallo_frame,
                        phoneme_labels_seq = None, alsfrs_total_seq = None, alsfrs_speech_seq = None, speaker_labels_seq = None, 
                        disease_duration_seq = None, king_stage_seq = None, group_seq = None, cantagallo_seq = None,                                             
                        mu_originals_z = mu_originals_z_train,mu_components_z =  mu_components_z_train,mu_projections_z = mu_projections_z_train if config.project_OCs else None,
                        mu_joint_components_z = mu_joint_components_z_train,mu_all_z = mu_all_z_train
                    )
                    
                if vis_args.visualize_latent_sequence and (config.dual_branched_latent or config.only_s_branch) and not vis_args.visualize_latent_frame:
                    voc_als_latent_vis(config,data_training_args,decomp_args,data_subset,vis_args.visualize_train_set,
                                    phoneme_labels = None, alsfrs_total = None, alsfrs_speech = None, speaker_labels_frame = None, 
                                    disease_duration = None, king_stage = None, group = None, cantagallo = None,
                                    phoneme_labels_seq = phonemes_seq, alsfrs_total_seq = alsfrs_total_seq, alsfrs_speech_seq = alsfrs_speech_seq, speaker_labels_seq = speaker_id_seq, 
                                    disease_duration_seq = disease_duration_seq, king_stage_seq = king_stage_seq, group_seq = group_seq, cantagallo_seq = cantagallo_seq,                                             
                                    mu_originals_z = None,mu_components_z = None,mu_projections_z = None,
                                    mu_joint_components_z = None,mu_all_z = None,
                                    mu_originals_s = mu_originals_s_train,mu_components_s = mu_components_s_train,mu_projections_s = mu_projections_s_train if config.project_OCs else None,
                                    mu_joint_components_s = mu_joint_components_s_train,mu_all_s = mu_all_s_train
                                    )




if __name__ == "__main__":
    main()