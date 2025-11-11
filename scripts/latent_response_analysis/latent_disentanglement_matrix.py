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

"""Calculate latent disentanglement matrices for DecVAE models. This script loads pretrained DecVAE models from specified checkpoints,
or initializes random DecVAE models, and computes disentanglement matrices based on the whole dataset or a subset of it. 
In SimVowels and TIMIT it uses the development and test sets, whereas in VOC_ALS and IEMOCAP it uses the entire dataset.
Decomposition of inputs is not supported here so if it's not already calculated then another script like vaes_pretraining.py should be ran first."""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import DecVAEForPreTraining
from config_files import DecVAEConfig
from data_collation import DataCollatorForDecVAELatentDisentanglement
from args_configs import ModelArgumentsPost, DataTrainingArgumentsPost, DecompositionArguments, TrainingObjectiveArguments
from utils import (
    parse_args, 
    debugger_is_active, 
    extract_epoch
)

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from safetensors.torch import load_file
import argparse
import pandas as pd
import re
import json
import gzip
import os
from dataclasses import field
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from torch.utils.data.dataloader import DataLoader
import time
import sys 
from disentanglement_utils import compute_disentanglement_matrices
import warnings

warnings.simplefilter("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

JSON_FILE_NAME_MANUAL = "config_files/DecVAEs/voc_als/latent_evaluations/config_latent_anal_voc_als.json" #for debugging purposes only
SAVE_DIR = '../latent_disentanglement_matrices_decvaes/voc_als'

logger = get_logger(__name__)

def main():
    "Parse the arguments"
    parser = HfArgumentParser((ModelArgumentsPost, TrainingObjectiveArguments, DecompositionArguments, DataTrainingArgumentsPost))
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
   
    if data_training_args.experiment == "ssl_loss":
        checkpoint_dir += "_" + str(data_training_args.ssl_loss_frame_perc) +"percent_frames"

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
    "Below this point we iterate across checkpoints"
    for ckp in data_training_args.epoch_range_to_evaluate:

        print(f"Loading model from checkpoint directory: {checkpoint_dir}")
        print(f"Processing checkpoint {ckp}...")

        "initialize random model and load pretrained weights"
        representation_function = DecVAEForPreTraining(config)
        if ckp != 'epoch_-01':
            "In case of epoch = -1, no weights, random initialization"

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

        data_collator = DataCollatorForDecVAELatentDisentanglement(
            model=representation_function,
            feature_extractor=feature_extractor,
            model_args=model_args,
            data_training_args=data_training_args,
            config=config,
            input_type = data_training_args.input_type,
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
        if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            "Evaluates on a single set"
            representation_function, eval_dataloader = accelerator.prepare(
                representation_function, eval_dataloader
            )
        else:
            representation_function, eval_dataloader, test_dataloader = accelerator.prepare(
                representation_function, eval_dataloader, test_dataloader
            )

        "Measure total loading time"
        start_time = time.time()
        "Get the representations"
        with torch.no_grad():
            "Eval set for loop"
            for step, batch in enumerate(eval_dataloader):
                batch_size = batch["input_values"].shape[0]
                mask_indices_seq_length = batch["input_values"].shape[2]
                sub_attention_mask = batch.pop("sub_attention_mask", None)
                overlap_mask_batch = batch.pop("overlap_mask", None)
                assert overlap_mask_batch != None if data_training_args.dataset_name in ["timit", "iemocap"] else True
                if (overlap_mask_batch is None or not data_training_args.discard_label_overlaps) and not "VOC_ALS" in data_training_args.dataset_name:
                    overlap_mask_batch = torch.zeros_like(sub_attention_mask,dtype=torch.bool)
                else:
                    if "VOC_ALS" in data_training_args.dataset_name:
                        overlap_mask_batch = torch.zeros_like(sub_attention_mask,dtype=torch.bool)
                    "Frames corresponding to padding are set as True in the overlap and discarded"
                    padded = sub_attention_mask.sum(dim = -1)
                    for b in range(batch_size):
                        overlap_mask_batch[b,padded[b]:] = 1
                    overlap_mask_batch = overlap_mask_batch.bool()

                if data_training_args.dataset_name in ["sim_vowels"]:
                    batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)
                    if hasattr(batch,"vowel_labels"):
                        vowel_labels_batch = batch.pop("vowel_labels")
                    if hasattr(batch,"speaker_vt_factor"):
                        speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                    
                    vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                elif data_training_args.dataset_name in ["timit", "iemocap"]:
                    batch["mask_time_indices"] = sub_attention_mask.clone()
                    if data_training_args.dataset_name == "timit":
                        phonemes39_batch = batch.pop("phonemes39", None)
                        phonemes48_batch = batch.pop("phonemes48", None)
                        
                        phonemes39_batch = phonemes39_batch[~overlap_mask_batch]
                        phonemes48_batch = phonemes48_batch[~overlap_mask_batch]

                    elif data_training_args.dataset_name == "iemocap":
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


                outputs = representation_function(**batch)
                del batch
                "Gather labels & latents for evaluations"
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

                elif data_training_args.dataset_name in ["timit", "iemocap"]:
                    if "timit" in data_training_args.dataset_name:
                        if step == 0:
                            phonemes39 = phonemes39_batch.clone()
                            phonemes48 = phonemes48_batch.clone()
                        else:
                            phonemes39 = torch.cat((phonemes39,phonemes39_batch))
                            phonemes48 = torch.cat((phonemes48,phonemes48_batch))
                    elif "iemocap" in data_training_args.dataset_name:
                        if step == 0:
                            phonemes = phonemes_batch.clone()
                            emotion_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)]) 
                            emotion_seq = torch.stack(emotion_batch) 
                        else:
                            phonemes = torch.cat((phonemes,phonemes_batch))
                            emotion_frame = torch.cat((emotion_frame,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(emotion_batch)])),dim = 0)
                            emotion_seq = torch.cat((emotion_seq,torch.stack(emotion_batch)),dim = 0) 
                    
                    if step == 0:
                        speaker_id_frame = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)]) #torch.stack([factor for i,factor in enumerate(speaker_vt_factor_batch) for _ in used_indices[i]])
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
                overlap_mask_batch = overlap_mask_batch[sub_attention_mask]
                if step == 0:
                    if config.dual_branched_latent or config.only_z_branch:
                        "Z latents"
                        outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                        mu_components_z = outputs.mu_components_z.detach().cpu()
                        outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                        mu_originals_z = outputs.mu_originals_z.detach().cpu()
                        "OCs projection if available"
                        if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                            outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                            mu_projections_z = outputs.mu_projections_z.detach().cpu()
                            
                    if config.dual_branched_latent or config.only_s_branch:
                        "S latents"
                        mu_components_s = outputs.mu_components_s.detach().cpu()
                        mu_originals_s = outputs.mu_originals_s.detach().cpu()
                        "OCs projection if available"
                        if hasattr(outputs,'used_projected_components_s') and config.project_OCs:
                            mu_projections_s = outputs.mu_projections_s.detach().cpu()
                else:
                    if config.dual_branched_latent or config.only_z_branch:
                        "Z latents"
                        outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                        mu_components_z = torch.cat((mu_components_z,outputs.mu_components_z.detach().cpu()),dim = 1)
                        outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                        mu_originals_z = torch.cat((mu_originals_z,outputs.mu_originals_z.detach().cpu()),dim = 0)
                        "OCs projection if available"
                        if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                            outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                            mu_projections_z = torch.cat((mu_projections_z,outputs.mu_projections_z.detach().cpu()),dim = 0)
                    
                    if config.dual_branched_latent or config.only_s_branch:
                        "S latents"
                        mu_components_s = torch.cat((mu_components_s,outputs.mu_components_s.detach().cpu()),dim = 1)
                        mu_originals_s = torch.cat((mu_originals_s,outputs.mu_originals_s.detach().cpu()),dim = 0)
                        "OCs projection if available"
                        if hasattr(outputs,'used_projected_components_s') and config.project_OCs:                        
                            mu_projections_s = torch.cat((mu_projections_s,outputs.mu_projections_s.detach().cpu()),dim = 0)


            if not (data_training_args.dataset_name in ["VOC_ALS", "iemocap"]):
                "Test set for loop"
                for step, batch in enumerate(test_dataloader):
                    batch_size = batch["input_values"].shape[0]
                    mask_indices_seq_length = batch["input_values"].shape[2]
                    sub_attention_mask = batch.pop("sub_attention_mask", None)
                    overlap_mask_batch = batch.pop("overlap_mask", None)
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

                    outputs = representation_function(**batch)

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
                        if step == 0:
                            phonemes39_test = phonemes39_batch.clone()
                            phonemes48_test = phonemes48_batch.clone()
                        else:
                            phonemes39_test = torch.cat((phonemes39_test,phonemes39_batch))
                            phonemes48_test = torch.cat((phonemes48_test,phonemes48_batch))
                            
                        if step == 0:
                            speaker_id_frame_test = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])
                            speaker_id_seq_test = torch.stack(speaker_id_batch) 
                        else:
                            speaker_id_frame_test = torch.cat((speaker_id_frame_test,torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask_batch[i]))]) for i,factor in enumerate(speaker_id_batch)])),dim = 0)
                            speaker_id_seq_test = torch.cat((speaker_id_seq_test,torch.stack(speaker_id_batch)),dim = 0) 

                    "Gather latents for evaluations"
                    overlap_mask_batch = overlap_mask_batch[sub_attention_mask]
                    if step == 0:
                        if config.dual_branched_latent or config.only_z_branch:
                            "Z latents"
                            outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                            mu_components_z_test = outputs.mu_components_z.detach().cpu()
                            outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                            mu_originals_z_test = outputs.mu_originals_z.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                mu_projections_z_test = outputs.mu_projections_z.detach().cpu()

                        if config.dual_branched_latent or config.only_s_branch:
                            "S latents"
                            mu_components_s_test = outputs.mu_components_s.detach().cpu()
                            mu_originals_s_test = outputs.mu_originals_s.detach().cpu()
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_s') and config.project_OCs:
                                mu_projections_s_test = outputs.mu_projections_s.detach().cpu()

                    else:
                        if config.dual_branched_latent or config.only_z_branch:
                            "Z latents"
                            outputs.mu_components_z = torch.masked_select(outputs.mu_components_z,~overlap_mask_batch[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
                            mu_components_z_test = torch.cat((mu_components_z_test,outputs.mu_components_z.detach().cpu()),dim = 1)
                            outputs.mu_originals_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1])
                            mu_originals_z_test = torch.cat((mu_originals_z_test,outputs.mu_originals_z.detach().cpu()),dim = 0)
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_z') and config.project_OCs:
                                outputs.mu_projections_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask_batch[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1])
                                mu_projections_z_test = torch.cat((mu_projections_z_test,outputs.mu_projections_z.detach().cpu()),dim = 0)

                        if config.dual_branched_latent or config.only_s_branch:
                            "S latents"
                            mu_components_s_test = torch.cat((mu_components_s_test,outputs.mu_components_s.detach().cpu()),dim = 1)
                            mu_originals_s_test = torch.cat((mu_originals_s_test,outputs.mu_originals_s.detach().cpu()),dim = 0)
                            "OCs projection if available"
                            if hasattr(outputs,'used_projected_components_s') and config.project_OCs:                        
                                mu_projections_s_test = torch.cat((mu_projections_s_test,outputs.mu_projections_s.detach().cpu()),dim = 0)


        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total loading time: {elapsed_time: .4f} seconds")

        "Z and S latents"
        "Combine components into a joint embedding - Extend labels accordingly"
        "Also combine all embeddings to a single one - X + OCs"
        if config.dual_branched_latent or config.only_z_branch:
            mu_joint_components_z = torch.cat([b.reshape(1,-1) for b in mu_components_z.transpose(0,1)])
            all_embs = torch.cat([mu_originals_z.unsqueeze(0),mu_components_z])
            mu_all_z = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
            if data_training_args.dataset_name not in ["VOC_ALS", "iemocap"]:
                mu_joint_components_z_test = torch.cat([b.reshape(1,-1) for b in mu_components_z_test.transpose(0,1)])
                all_embs_test = torch.cat([mu_originals_z_test.unsqueeze(0),mu_components_z_test])
                mu_all_z_test = torch.cat([b.reshape(1,-1) for b in all_embs_test.transpose(0,1)])
        if config.dual_branched_latent or config.only_s_branch:
            if not config.use_first_agg and not config.use_second_agg:
                "If no aggregation strategy is used, then sequence case is same as frame - same shape"
                mu_components_s = mu_components_s.reshape(mu_components_s.shape[0],-1,mu_components_s.shape[-1])
                mu_joint_components_s = torch.cat([b.reshape(1,-1) for b in mu_components_s.transpose(0,1)])
            else:
                mu_joint_components_s = torch.cat([b.reshape(1,-1) for b in mu_components_s.transpose(0,1)])
            all_embs = torch.cat([mu_originals_s.unsqueeze(0),mu_components_s])
            mu_all_s = torch.cat([b.reshape(1,-1) for b in all_embs.transpose(0,1)])
            if data_training_args.dataset_name not in ["VOC_ALS", "iemocap"]:
                if not config.use_first_agg and not config.use_second_agg:
                    "If no aggregation strategy is used, then sequence case is same as frame - same shape"
                    mu_components_s_test = mu_components_s_test.reshape(mu_components_s_test.shape[0],-1,mu_components_s_test.shape[-1])
                    mu_joint_components_s_test = torch.cat([b.reshape(1,-1) for b in mu_components_s_test.transpose(0,1)])
                else:
                    mu_joint_components_s_test = torch.cat([b.reshape(1,-1) for b in mu_components_s_test.transpose(0,1)])                
                all_embs_test = torch.cat([mu_originals_s_test.unsqueeze(0),mu_components_s_test])
                mu_all_s_test = torch.cat([b.reshape(1,-1) for b in all_embs_test.transpose(0,1)])


        "Disentanglement matrix calculation"
        if "vowels" in data_training_args.dataset_name:
            y_frame_test = torch.cat([vowel_labels_test.reshape(-1,1),speaker_vt_factor_frame_test.reshape(-1,1)],dim = 1)
            y_frame_train = torch.cat([vowel_labels.reshape(-1,1),speaker_vt_factor_frame.reshape(-1,1)],dim = 1)
            y_frame_test = pd.DataFrame(y_frame_test.cpu().numpy(),columns=["vowel","speaker_frame"])
            y_frame_train = pd.DataFrame(y_frame_train.cpu().numpy(),columns=["vowel","speaker_frame"])
            y_seq_test = pd.DataFrame(speaker_vt_factor_seq_test.cpu().numpy(),columns=["speaker_seq"])
            y_seq_train = pd.DataFrame(speaker_vt_factor_seq.cpu().numpy(),columns=["speaker_seq"])

        elif data_training_args.dataset_name in ["timit"]:
            speaker_id_frame_test = speaker_id_frame_test.to(phonemes39_test.device)
            speaker_id_frame = speaker_id_frame.to(phonemes39_test.device)
            y_frame_test = torch.cat([phonemes39_test.reshape(-1,1),speaker_id_frame_test.reshape(-1,1)],dim = 1)
            y_frame_train = torch.cat([phonemes39.reshape(-1,1),speaker_id_frame.reshape(-1,1)],dim = 1)
            y_frame_test = pd.DataFrame(y_frame_test.cpu().numpy(),columns=["phoneme","speaker_frame"])
            y_frame_train = pd.DataFrame(y_frame_train.cpu().numpy(),columns=["phoneme","speaker_frame"])
            y_seq_test = pd.DataFrame(speaker_id_seq_test.cpu().numpy(),columns=["speaker_seq"])
            y_seq_train = pd.DataFrame(speaker_id_seq.cpu().numpy(),columns=["speaker_seq"])
        
        elif "iemocap" in data_training_args.dataset_name:
            speaker_id_frame = speaker_id_frame.to(phonemes.device)
            emotion_frame = emotion_frame.to(phonemes.device)
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

            "Disease Duration - Phoneme - Speaker"
            y_frame_train_dis = torch.cat([disease_duration_frame.reshape(-1,1),phonemes_frame.reshape(-1,1),speaker_id_frame.reshape(-1,1)],dim = 1)
            y_seq_train_dis = torch.cat([disease_duration_seq.reshape(-1,1),phonemes_seq.reshape(-1,1),speaker_id_seq.reshape(-1,1)],dim = 1)
            y_frame_train_dis = pd.DataFrame(y_frame_train_dis.cpu().numpy(),columns=["disease_duration_frame","phoneme_frame","speaker_frame"])
            y_seq_train_dis = pd.DataFrame(y_seq_train_dis.cpu().numpy(),columns=["disease_duration_seq","phoneme_seq","speaker_seq"])


        "Frame-level latent Z"
        if config.dual_branched_latent or config.only_z_branch:
            # Prepare latent representations
            if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
                mu_X_z = mu_originals_z.clone()
                mu_OC1_z = mu_components_z[0].clone()
                mu_OC2_z = mu_components_z[1].clone()
                if config.NoC >= 3:
                    mu_OC3_z = mu_components_z[2].clone()
                if config.NoC >= 4:
                    mu_OC4_z = mu_components_z[3].clone()
                if config.NoC >= 5:
                    mu_OC5_z = mu_components_z[4].clone()
                if config.project_OCs:
                    mu_OCs_proj_z = mu_projections_z.clone()
                    
                mu_OCs_joint_z = mu_joint_components_z.clone() 
                mu_X_OCs_joint_z = mu_all_z.clone()
                
            elif data_training_args.dataset_name in ["sim_vowels", "timit"]:
                mu_X_z = torch.cat((mu_originals_z, mu_originals_z_test), dim=0)
                mu_OC1_z = torch.cat((mu_components_z[0], mu_components_z_test[0]), dim=0)
                mu_OC2_z = torch.cat((mu_components_z[1], mu_components_z_test[1]), dim=0)
                if config.NoC >= 3:
                    mu_OC3_z = torch.cat((mu_components_z[2], mu_components_z_test[2]), dim=0)
                if config.NoC >= 4:
                    mu_OC4_z = torch.cat((mu_components_z[3], mu_components_z_test[3]), dim=0)
                if config.NoC >= 5:
                    mu_OC5_z = torch.cat((mu_components_z[4], mu_components_z_test[4]), dim=0)
                if config.project_OCs:
                    mu_OCs_proj_z = torch.cat((mu_projections_z, mu_projections_z_test), dim=0)
                    
                mu_OCs_joint_z = torch.cat((mu_joint_components_z, mu_joint_components_z_test), dim=0)
                mu_X_OCs_joint_z = torch.cat((mu_all_z, mu_all_z_test), dim=0)
            
            # Create factors dictionary
            if data_training_args.dataset_name in ["sim_vowels","timit"]:
                if "sim_vowels" in data_training_args.dataset_name:
                    phonemes = pd.concat((y_frame_train["vowel"], y_frame_test["vowel"]), axis=0, ignore_index=True)
                elif "timit" in data_training_args.dataset_name:
                    phonemes = pd.concat((y_frame_train["phoneme"], y_frame_test["phoneme"]), axis=0, ignore_index=True)
                speakers = pd.concat((y_frame_train["speaker_frame"], y_frame_test["speaker_frame"]), axis=0, ignore_index=True)
                factor_dict = {"phonemes": phonemes, "speakers": speakers}
            elif data_training_args.dataset_name in ["VOC_ALS", "iemocap", "scRNA_seq"]:
                if "iemocap" in data_training_args.dataset_name:
                    phonemes = y_frame_train["phoneme"]
                    speakers = y_frame_train["speaker_frame"]
                    emotions = y_frame_train["cat_emotion_frame"]
                    factor_dict = {"phonemes": phonemes, "speakers": speakers, "emotions": emotions}
                elif "VOC_ALS" in data_training_args.dataset_name:
                    phonemes = y_frame_train_king["phoneme_frame"]
                    speakers = y_frame_train_king["speaker_frame"]
                    king_stage = y_frame_train_king["king_stage_frame"]
                    factor_dict = {"phonemes": phonemes, "speakers": speakers, "king_stage": king_stage}

            # Create directory for storing results
            store_dir = os.path.join(SAVE_DIR, os.path.basename(checkpoint_dir), 'frame',
                                    f"checkpoint-{ckp}")
            os.makedirs(store_dir, exist_ok=True)
            
            latent_vis_dict_z = {}

            latent_vis_dict_z["X"] = compute_disentanglement_matrices(mu_X_z, factor_dict, store_dir, "X", compute_3d_slices = False, save_visualizations=True)

            latent_vis_dict_z["OC1"] = compute_disentanglement_matrices(mu_OC1_z, factor_dict, store_dir, "OC1", compute_3d_slices = False, save_visualizations=True)

            latent_vis_dict_z["OC2"] = compute_disentanglement_matrices(mu_OC2_z, factor_dict, store_dir, "OC2", compute_3d_slices = False, save_visualizations=True)

            if config.NoC >= 3:
                latent_vis_dict_z["OC3"] = compute_disentanglement_matrices(mu_OC3_z, factor_dict, store_dir, "OC3", compute_3d_slices = False, save_visualizations=True)
            if config.NoC >= 4:
                latent_vis_dict_z["OC4"] = compute_disentanglement_matrices(mu_OC4_z, factor_dict, store_dir, "OC4", compute_3d_slices = False, save_visualizations=True)
            if config.NoC >= 5:
                latent_vis_dict_z["OC5"] = compute_disentanglement_matrices(mu_OC5_z, factor_dict, store_dir, "OC5", compute_3d_slices = False, save_visualizations=True)

            if config.project_OCs:
                latent_vis_dict_z["OCs_proj"] = compute_disentanglement_matrices(mu_OCs_proj_z, factor_dict, store_dir, "OCs_proj", compute_3d_slices = False, save_visualizations=True)
            
            # Process OCs_joint latent
            latent_vis_dict_z["OCs_joint"] = compute_disentanglement_matrices(mu_OCs_joint_z, factor_dict, store_dir, "OCs_joint", compute_3d_slices = False, save_visualizations=True)

            # Process all latent
            "Do we need to process the all latent?"
            latent_vis_dict_z["all"] = compute_disentanglement_matrices(mu_X_OCs_joint_z, factor_dict, store_dir, "all", compute_3d_slices = False, save_visualizations=True)

            # Store the factor values for reference
            if data_training_args.dataset_name in ["sim_vowels", "timit"]:
                latent_vis_dict_z["phonemes"] = phonemes.tolist()
                latent_vis_dict_z["speakers"] = speakers.tolist()
            elif data_training_args.dataset_name in "VOC_ALS":
                latent_vis_dict_z["phonemes"] = phonemes.tolist()
                latent_vis_dict_z["speakers"] = speakers.tolist()
                latent_vis_dict_z["king_stage"] = king_stage.tolist()
            elif data_training_args.dataset_name in "iemocap":
                latent_vis_dict_z["phonemes"] = phonemes.tolist()
                latent_vis_dict_z["speakers"] = speakers.tolist()
                latent_vis_dict_z["emotions"] = emotions.tolist()

            # Save all results to a single compressed JSON file
            with gzip.open(os.path.join(store_dir, "latent_vis_dict_z.json"), 'wt') as f:
                json.dump(latent_vis_dict_z, f)
       
        "Sequence-level latent S"
        if config.dual_branched_latent or config.only_s_branch:
            "Prepare latent representations"
            if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
                mu_X_s = mu_originals_s.clone()
                mu_OC1_s = mu_components_s[0].clone()
                mu_OC2_s = mu_components_s[1].clone()
                if config.NoC_seq >= 3:
                    mu_OC3_s = mu_components_s[2].clone()
                if config.NoC_seq >= 4:
                    mu_OC4_s = mu_components_s[3].clone()
                if config.NoC_seq >= 5:
                    mu_OC5_s = mu_components_s[4].clone()
                if config.project_OCs:
                    mu_OCs_proj_s = mu_projections_s.clone()

                mu_OCs_joint_s = mu_joint_components_s.clone() 
                mu_X_OCs_joint_s = mu_all_s.clone()

            elif data_training_args.dataset_name in ["sim_vowels", "timit"]:
                print("Sim vowels and TIMIT have a single generative factor in the sequence level. Exiting without calculating disentanglement matrix")
                break

            # Create factors dictionary
            if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
                if "iemocap" in data_training_args.dataset_name:
                    speakers = y_seq_train["speaker_seq"]
                    emotions = y_seq_train["cat_emotion_seq"]
                    factor_dict = {"speakers": speakers, "emotions": emotions}
                elif "VOC_ALS" in data_training_args.dataset_name:
                    phonemes = y_seq_train_king["phoneme_seq"]
                    speakers = y_seq_train_king["speaker_seq"]
                    king_stage = y_seq_train_king["king_stage_seq"]
                    factor_dict = {"phonemes": phonemes, "speakers": speakers, "king_stage": king_stage}

            # Create directory for storing results
            store_dir = os.path.join(SAVE_DIR, os.path.basename(checkpoint_dir), 'sequence',
                                    f"checkpoint-{ckp}")
            os.makedirs(store_dir, exist_ok=True)
            
            latent_vis_dict_s = {}
            
            latent_vis_dict_s["X"] = compute_disentanglement_matrices(mu_X_s, factor_dict, store_dir, "X", compute_3d_slices = False, save_visualizations=False)
            
            latent_vis_dict_s["OC1"] = compute_disentanglement_matrices(mu_OC1_s, factor_dict, store_dir, "OC1", compute_3d_slices = False, save_visualizations=False)
            
            latent_vis_dict_s["OC2"] = compute_disentanglement_matrices(mu_OC2_s, factor_dict, store_dir, "OC2", compute_3d_slices = False, save_visualizations=False)
            
            if config.NoC_seq >= 3:
                latent_vis_dict_s["OC3"] = compute_disentanglement_matrices(mu_OC3_s, factor_dict, store_dir, "OC3", compute_3d_slices = False, save_visualizations=False)
            if config.NoC_seq >= 4:
                latent_vis_dict_s["OC4"] = compute_disentanglement_matrices(mu_OC4_s, factor_dict, store_dir, "OC4", compute_3d_slices = False, save_visualizations=False)
            if config.NoC_seq >= 5:
                latent_vis_dict_s["OC5"] = compute_disentanglement_matrices(mu_OC5_s, factor_dict, store_dir, "OC5", compute_3d_slices = False, save_visualizations=False)

            if config.project_OCs:
                latent_vis_dict_s["OCs_proj"] = compute_disentanglement_matrices(mu_OCs_proj_s, factor_dict, store_dir, "OCs_proj", compute_3d_slices = False, save_visualizations=False)
            
            # Process OCs_joint latent
            latent_vis_dict_s["OCs_joint"] = compute_disentanglement_matrices(mu_OCs_joint_s, factor_dict, store_dir, "OCs_joint", compute_3d_slices = False, save_visualizations=False)

            # Process all latent
            latent_vis_dict_s["all"] = compute_disentanglement_matrices(mu_X_OCs_joint_s, factor_dict, store_dir, "all", compute_3d_slices = False, save_visualizations=False)

            # Store the factor values for reference
            if data_training_args.dataset_name in "VOC_ALS":
                latent_vis_dict_s["phonemes"] = phonemes.tolist()
                latent_vis_dict_s["speakers"] = speakers.tolist()
                latent_vis_dict_s["king_stage"] = king_stage.tolist()
            elif data_training_args.dataset_name in "iemocap":
                latent_vis_dict_s["speakers"] = speakers.tolist()
                latent_vis_dict_s["emotions"] = emotions.tolist()
            
            # Save all results to a single compressed JSON file
            with gzip.open(os.path.join(store_dir, "latent_vis_dict_s.json"), 'wt') as f:
                json.dump(latent_vis_dict_s, f)

if __name__ == "__main__":
    main()