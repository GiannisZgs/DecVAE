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

"""This script handles all latent evaluations (classification, disentanglement) for DecVAE models. 
This script loads pretrained DecVAE models from specified checkpoints, or initializes random DecVAE models, 
gathers representations per data point and computes classification (task-related) and disentanglement metrics. 
See arguments data_training_args.aggregations_to_use and .classification_tasks in args_configs.data_training_args.DataTrainingArgumentsPost for more details
on selecting which variables to classify and what embeddings to use for each dataset.
Decomposition of inputs is not supported here so if it's not already calculated then another script like base_models_ssl_pretraining.py should be ran first."""

from logging import config
import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import DecVAEForPreTraining
from data_collation import DataCollatorForDecVAELatentPostAnalysis
from config_files import DecVAEConfig
from args_configs import ModelArgumentsPost, DataTrainingArgumentsPost, DecompositionArguments, TrainingObjectiveArguments
from utils import parse_args, debugger_is_active, extract_epoch
from latent_analysis_utils import prediction_eval, visualize
from disentanglement_utils import compute_disentanglement_metrics

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from safetensors.torch import load_file
import pandas as pd
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from torch.utils.data.dataloader import DataLoader
import time
import warnings

warnings.simplefilter("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

JSON_FILE_NAME_MANUAL = "config_files/DecVAEs/sim_vowels/latent_evaluations/config_latent_anal_sim_vowels.json" #for debugging purposes only

logger = get_logger(__name__)

def main():
    "Parse the arguments"       
    parser = HfArgumentParser((ModelArgumentsPost, TrainingObjectiveArguments, DecompositionArguments,DataTrainingArgumentsPost))
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
    elif "scRNA_seq" in data_training_args.dataset_name:
        checkpoint_dir = os.path.join(data_training_args.parent_dir,
            betas[1:] + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type + "-bs" + str(data_training_args.per_device_train_batch_size))

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
            "In case of epoch = -1, no weights, random initialization"

            pretrained_model_file = os.path.join(checkpoint_dir,ckp,"model.safetensors")
        
            weights = load_file(pretrained_model_file)
    
            keys_to_remove = [key for key in weights.keys() if 'project_hid' in key or 'project_q' in key]
            if keys_to_remove:
                for key in keys_to_remove:
                    del weights[key]
                    print(f"Removed deprecated module {key} from weights.")

            representation_function.load_state_dict(weights, strict=False)
        
        representation_function.eval()
        for param in representation_function.parameters():
            param.requires_grad = False

        "data collator, optimizer and scheduler"
        mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
        mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

        data_collator = DataCollatorForDecVAELatentPostAnalysis(
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
            ) #train_dataloader

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

            if not(data_training_args.dataset_name in ["VOC_ALS", "iemocap"]):
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
                    if "vowels" in data_training_args.dataset_name:
                        batch["mask_time_indices"] = torch.ones((batch_size, mask_indices_seq_length), dtype=torch.bool, device=batch["mask_time_indices"].device)                
                        if hasattr(batch,"vowel_labels"):
                            vowel_labels_batch = batch.pop("vowel_labels")
                        if hasattr(batch,"speaker_vt_factor"):
                            speaker_vt_factor_batch = batch.pop("speaker_vt_factor")
                        vowel_labels_batch = [[ph for i,ph in enumerate(batch) if not overlap_mask_batch[j,i]] for j,batch in enumerate(vowel_labels_batch)] 

                    elif data_training_args.dataset_name == "timit":
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
        "Here we create different aggregation strategies to aggregate subspaces into a single latent space"
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


        "Now use train/val representations to get the evaluation metrics"
        "1. Projection evaluation"
        "2. Components evaluation separately (1st,2nd,3rd)"
        
        "Linear/non-linear classification"
        #Create combined labels - Speaker-vowel
        #Comparisons inside speaker / inside vowel
        if data_training_args.classify:
            if "vowels" in data_training_args.dataset_name:
                if config.dual_branched_latent or config.only_z_branch:
                    
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        if "vowel" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                            "OCs combination ([OC1,OC2,...,OCn])"
                            "Check vowels in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = mu_joint_components_z_test,
                                y = vowel_labels, y_test = vowel_labels_test,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "vowel"
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = mu_joint_components_z_test,
                                y = speaker_vt_factor_frame, y_test = speaker_vt_factor_frame_test,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "speaker_frame"
                            )

                    "OCs projection <- f([OC1,OC2,...,OCn])"
                    if config.project_OCs and "OCs_proj" in data_training_args.aggregations_to_use:
                        if "vowel" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                            "Check vowels in z"
                            prediction_eval(data_training_args,config,
                                X = mu_projections_z, X_test = mu_projections_z_test,
                                y = vowel_labels, y_test = vowel_labels_test,
                                checkpoint = ckp, latent_type="OCs_proj",target = "vowel"
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks: 
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_projections_z, X_test = mu_projections_z_test,
                                y = speaker_vt_factor_frame, y_test = speaker_vt_factor_frame_test,
                                checkpoint = ckp, latent_type="OCs_proj",target = "speaker_frame"
                            )

                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        if "vowel" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                            "Check vowels in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = mu_all_z_test,
                                y = vowel_labels, y_test = vowel_labels_test,
                                checkpoint = ckp, latent_type="all",target = "vowel"
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = mu_all_z_test,
                                y = speaker_vt_factor_frame, y_test = speaker_vt_factor_frame_test,
                                checkpoint = ckp, latent_type="all",target = "speaker_frame"
                            )

                    if "X" in data_training_args.aggregations_to_use:
                        "Original X"
                        if "vowel" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                            "Check vowels in z"
                            prediction_eval(data_training_args,config,
                                X = mu_originals_z, X_test = mu_originals_z_test,
                                y = vowel_labels, y_test = vowel_labels_test,
                                checkpoint = ckp, latent_type="X",target = "vowel"
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_originals_z, X_test = mu_originals_z_test,
                                y = speaker_vt_factor_frame, y_test = speaker_vt_factor_frame_test,
                                checkpoint = ckp, latent_type="X",target = "speaker_frame"
                            )

                    
                    if "OCs" in data_training_args.aggregations_to_use:
                        "Individual OCs"
                        if "vowel" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check vowels in z"
                            for i in range(decomp_args.NoC):                    
                                prediction_eval(data_training_args,config,
                                    X = mu_components_z[i], X_test = mu_components_z_test[i],
                                    y = vowel_labels, y_test = vowel_labels_test,
                                    checkpoint = ckp, latent_type=f'OC{i+1}',target = "vowel"
                                )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            for i in range(decomp_args.NoC):                    
                                prediction_eval(data_training_args,config,
                                    X = mu_components_z[i], X_test = mu_components_z_test[i],
                                    y = speaker_vt_factor_frame, y_test = speaker_vt_factor_frame_test,
                                    checkpoint = ckp, latent_type=f'OC{i+1}',target = "speaker_frame"
                                )

                if config.dual_branched_latent or config.only_s_branch:

                    if "OCs_joint_emb" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "OCs combination ([OC1,OC2,...,OCn])"
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_joint_components_s, X_test = mu_joint_components_s_test,
                            y = speaker_vt_factor_seq, y_test = speaker_vt_factor_seq_test,
                            checkpoint = ckp, latent_type="OCs_joint_emb",target = "speaker_seq"
                        )

                    "OCs projection <- f([OC1,OC2,...,OCn])"
                    if "OCs_proj" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks) and config.project_OCs:
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_projections_s, X_test = mu_projections_s_test,
                            y = speaker_vt_factor_seq, y_test = speaker_vt_factor_seq_test,
                            checkpoint = ckp, latent_type="OCs_proj",target = "speaker_seq"
                        )
                        
                    if "all" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_all_s, X_test = mu_all_s_test,
                            y = speaker_vt_factor_seq, y_test = speaker_vt_factor_seq_test,
                            checkpoint = ckp, latent_type="all",target = "speaker_seq"
                        )

                    if "X" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "Original X"
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_originals_s, X_test = mu_originals_s_test,
                            y = speaker_vt_factor_seq, y_test = speaker_vt_factor_seq_test,
                            checkpoint = ckp, latent_type="X",target = "speaker_seq"
                        )
                    
                    if "OCs" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "Individual OCs"
                        "Check speakers in s"
                        for i in range(decomp_args.NoC_seq):                    
                            prediction_eval(data_training_args,config,
                                X = mu_components_s[i], X_test = mu_components_s_test[i],
                                y = speaker_vt_factor_seq, y_test = speaker_vt_factor_seq_test,
                                checkpoint = ckp, latent_type=f'OC{i+1}',target = "speaker_seq"
                            )


            elif "timit" in data_training_args.dataset_name:
                if config.dual_branched_latent or config.only_z_branch:
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs combination ([OC1,OC2,...,OCn])"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = mu_joint_components_z_test,
                                y = phonemes48, y_test = phonemes48_test,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "phoneme48" 
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = mu_joint_components_z_test,
                                y = speaker_id_frame, y_test = speaker_id_frame_test,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "speaker_frame"
                            )

                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            prediction_eval(data_training_args,config,
                                X = mu_projections_z, X_test = mu_projections_z_test,
                                y = phonemes48, y_test = phonemes48_test,
                                checkpoint = ckp, latent_type="OCs_proj",target = "phoneme48" 
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:  
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_projections_z, X_test = mu_projections_z_test,
                                y = speaker_id_frame, y_test = speaker_id_frame_test,
                                checkpoint = ckp, latent_type="OCs_proj",target = "speaker_frame"
                            )

                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = mu_all_z_test,
                                y = phonemes48, y_test = phonemes48_test,
                                checkpoint = ckp, latent_type="all",target = "phoneme48"
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = mu_all_z_test,
                                y = speaker_id_frame, y_test = speaker_id_frame_test,
                                checkpoint = ckp, latent_type="all",target = "speaker_frame"
                            )

                    if "X" in data_training_args.aggregations_to_use:
                        "Original X"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            prediction_eval(data_training_args,config,
                                X = mu_originals_z, X_test = mu_originals_z_test,
                                y = phonemes48, y_test = phonemes48_test,
                                checkpoint = ckp, latent_type="X",target = "phoneme48" 
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_originals_z, X_test = mu_originals_z_test,
                                y = speaker_id_frame, y_test = speaker_id_frame_test,
                                checkpoint = ckp, latent_type="X",target = "speaker_frame"
                            )

                    if "OCs" in data_training_args.aggregations_to_use:
                        "Individual OCs"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            for i in range(decomp_args.NoC):                    
                                prediction_eval(data_training_args,config,
                                    X = mu_components_z[i], X_test = mu_components_z_test[i],
                                    y = phonemes48, y_test = phonemes48_test,
                                    checkpoint = ckp, latent_type=f'OC{i+1}',target = "phoneme48"
                                )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            for i in range(decomp_args.NoC):                    
                                prediction_eval(data_training_args,config,
                                    X = mu_components_z[i], X_test = mu_components_z_test[i],
                                    y = speaker_id_frame, y_test = speaker_id_frame_test,
                                    checkpoint = ckp, latent_type=f'OC{i+1}',target = "speaker_frame"
                                )
                    

                if config.dual_branched_latent or config.only_s_branch:
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "OCs combination ([OC1,OC2,...,OCn])"
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_joint_components_s, X_test = mu_joint_components_s_test,
                            y = speaker_id_seq, y_test = speaker_id_seq_test,
                            checkpoint = ckp, latent_type="OCs_joint_emb",target = "speaker_seq"
                        )
                    
                    if "OCs_proj" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks) and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_projections_s, X_test = mu_projections_s_test,
                            y = speaker_id_seq, y_test = speaker_id_seq_test,
                            checkpoint = ckp, latent_type="OCs_proj",target = "speaker_seq"
                        )

                    if "all" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_all_s, X_test = mu_all_s_test,
                            y = speaker_id_seq, y_test = speaker_id_seq_test,
                            checkpoint = ckp, latent_type="all",target = "speaker_seq"
                        )

                    if "X" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "Original X"
                        "Check speakers in s"                    
                        prediction_eval(data_training_args,config,
                            X = mu_originals_s, X_test = mu_originals_s_test,
                            y = speaker_id_seq, y_test = speaker_id_seq_test,
                            checkpoint = ckp, latent_type="X",target = "speaker_seq"
                        )
                    
                    if "OCs" in data_training_args.aggregations_to_use and ("speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks):
                        "Individual OCs"
                        "Check speakers in s"
                        for i in range(decomp_args.NoC_seq):                    
                            prediction_eval(data_training_args,config,
                                X = mu_components_s[i], X_test = mu_components_s_test[i],
                                y = speaker_id_seq, y_test = speaker_id_seq_test,
                                checkpoint = ckp, latent_type=f'OC{i+1}',target = "speaker_seq"
                            )


            elif "VOC_ALS" in data_training_args.dataset_name:
                if config.dual_branched_latent or config.only_z_branch:
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs combination ([OC1,OC2,...,OCn])"
                        if "phoneme_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phoneme in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = phonemes_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "phoneme_frame"
                            )
                        if "group_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check group in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = group_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "group_frame"
                            )
                        if "kings_stage_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check King's staging in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = king_stage_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "kings_stage_frame"
                            )
                        if "disease_duration_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                            "Check disease duration in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = disease_duration_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "disease_duration_frame"
                            )
                        if "alsfrs_total_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                            "Check ALSFRS-total staging in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = alsfrs_total_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "alsfrs_total_frame"
                            )
                        if "alsfrs_speech_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:   
                            "Check ALSFRS-speech subitem staging in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = alsfrs_speech_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "alsfrs_speech_frame"
                            )
                        if "cantagallo_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check Cantagallo Questionnaire Scale in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = cantagallo_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "cantagallo_frame"
                            )
                    
                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        if "phoneme_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phoneme in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = phonemes_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "phoneme_frame"
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speaker in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = speaker_id_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "speaker_frame"
                            )
                        if "disease_duration_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check disease duration in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = disease_duration_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "disease_duration_frame"
                            )
                        if "group_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check group in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = group_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "group_frame"
                            )
                        if "kings_stage_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check King's staging in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = king_stage_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "kings_stage_frame"
                            )
                        if "alsfrs_total_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check ALSFRS-total in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = alsfrs_total_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "alsfrs_total_frame"
                            )
                        if "alsfrs_speech_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:             
                            "Check ALSFRS-speech in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = alsfrs_speech_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "alsfrs_speech_frame"
                            )
                        if "cantagallo_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check Cantagallo in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = cantagallo_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "cantagallo_frame"
                            )

                    
                if config.dual_branched_latent or config.only_s_branch:
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs combination ([OC1,OC2,...,OCn])"
                        if "phoneme_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phoneme in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = phonemes_seq, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "phoneme_seq"
                            )
                        if "group_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check group in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = group_seq, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "group_seq"
                            )
                        if "kings_stage_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check King's staging in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = king_stage_seq, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "kings_stage_seq"
                            )
                        if "disease_duration_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check disease duration staging in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = disease_duration_seq, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "disease_duration_seq"
                            )
                        if "alsfrs_total_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check ALSFRS-total staging in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = alsfrs_total_seq, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "alsfrs_total_seq"
                            )
                        if "alsfrs_speech_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check ALSFRS-speech staging in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = alsfrs_speech_seq, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "alsfrs_speech_seq"
                            )

                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        if "phoneme_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phoneme in s"
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = phonemes_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "phoneme_seq"
                            )
                        if "speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speaker in s"
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = speaker_id_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "speaker_seq"
                            )
                        if "disease_duration_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = disease_duration_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "disease_duration_seq"
                            )

                        if "group_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check group in s"
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = group_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "group_seq"
                            )
                        if "kings_stage_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check King's staging in s"
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = king_stage_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "kings_stage_seq"
                            )
                        if "alsfrs_total_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check ALSFRS-total in s"
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = alsfrs_total_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "alsfrs_total_seq"
                            )
                        if "alsfrs_speech_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check ALSFRS-speech subitem in s"
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = alsfrs_speech_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "alsfrs_speech_seq"
                            )
                        if "cantagallo_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check Cantagallo in s"
                            prediction_eval(data_training_args,config,
                                X = mu_all_s, X_test = None,
                                y = cantagallo_seq, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "cantagallo_seq"
                            )


            elif "iemocap" in data_training_args.dataset_name:
                if config.dual_branched_latent or config.only_z_branch:
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs combination ([OC1,OC2,...,OCn])"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phoneme frame accuracy in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = phonemes, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "phoneme_frame" 
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = speaker_id_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "speaker_frame"
                            )
                        if "emotion_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check emotion in z"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_z, X_test = None,
                                y = torch.stack((emotion_frame,speaker_id_frame), dim = 1), y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = ["cat_emotion_frame", "speaker_frame"]
                            )
                            
                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phoneme in z"
                            prediction_eval(data_training_args,config,
                                X = mu_projections_z, X_test = None,
                                y = phonemes, y_test = None,
                                checkpoint = ckp, latent_type="OCs_proj",target = "phoneme_frame" 
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_projections_z, X_test = None,
                                y = speaker_id_frame, y_test = None,
                                checkpoint = ckp, latent_type="OCs_proj",target = "speaker_frame"
                            )
                        if "emotion_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check emotion in z"
                            prediction_eval(data_training_args,config,
                                X = mu_projections_z, X_test = None,
                                y = torch.stack((emotion_frame,speaker_id_frame), dim = 1), y_test = None,
                                checkpoint = ckp, latent_type="OCs_proj",target = ["cat_emotion_frame", "speaker_frame"]
                            )

                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = phonemes, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "phoneme_frame" 
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = speaker_id_frame, y_test = None,
                                checkpoint = ckp, latent_type="all",target = "speaker_frame"
                            )
                        if "emotion_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check emotion in z"
                            prediction_eval(data_training_args,config,
                                X = mu_all_z, X_test = None,
                                y = torch.stack((emotion_frame,speaker_id_frame), dim = 1), y_test = None,
                                checkpoint = ckp, latent_type="all",target = ["cat_emotion_frame", "speaker_frame"]
                            )

                    if "X" in data_training_args.aggregations_to_use:
                        "Original X"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            prediction_eval(data_training_args,config,
                                X = mu_originals_z, X_test = None,
                                y = phonemes, y_test = None,
                                checkpoint = ckp, latent_type="X",target = "phoneme_frame" 
                            )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            prediction_eval(data_training_args,config,
                                X = mu_originals_z, X_test = None,
                                y = speaker_id_frame, y_test = None,
                                checkpoint = ckp, latent_type="X",target = "speaker_frame"
                            )
                        if "emotion_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check emotion in z"
                            prediction_eval(data_training_args,config,
                                X = mu_originals_z, X_test = None,
                                y = torch.stack((emotion_frame,speaker_id_frame), dim = 1), y_test = None,
                                checkpoint = ckp, latent_type="X",target = ["cat_emotion_frame", "speaker_frame"]

                            )

                    if "OCs" in data_training_args.aggregations_to_use:
                        "Individual OCs"
                        if "phoneme" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check phonemes in z"
                            for i in range(decomp_args.NoC):                    
                                prediction_eval(data_training_args,config,
                                    X = mu_components_z[i], X_test = None,
                                    y = phonemes, y_test = None,
                                    checkpoint = ckp, latent_type=f'OC{i+1}',target = "phoneme_frame" 
                                )
                        if "speaker_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in z"
                            for i in range(decomp_args.NoC):                    
                                prediction_eval(data_training_args,config,
                                    X = mu_components_z[i], X_test = None,
                                    y = speaker_id_frame, y_test = None,
                                    checkpoint = ckp, latent_type=f'OC{i+1}',target = "speaker_frame"
                                )
                        if "emotion_frame" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check emotion in z"
                            for i in range(decomp_args.NoC):
                                prediction_eval(data_training_args,config,
                                    X = mu_components_z[i], X_test = None,
                                    y = torch.stack((emotion_frame,speaker_id_frame), dim = 1), y_test = None,
                                    checkpoint = ckp, latent_type=f'OC{i+1}',target = ["cat_emotion_frame", "speaker_frame"]
                                )

                if config.dual_branched_latent or config.only_s_branch:
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs combination ([OC1,OC2,...,OCn])"
                        if "speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check speakers in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = speaker_id_seq, y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = "speaker_seq"
                            )
                        if "emotion_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                            "Check emotion in s"
                            prediction_eval(data_training_args,config,
                                X = mu_joint_components_s, X_test = None,
                                y = torch.stack((emotion_seq,speaker_id_seq), dim = 1), y_test = None,
                                checkpoint = ckp, latent_type="OCs_joint_emb",target = ["cat_emotion_seq", "speaker_seq"]
                            )

                if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                    "OCs projection <- f([OC1,OC2,...,OCn])"
                    if "speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_projections_s, X_test = None,
                            y = speaker_id_seq, y_test = None,
                            checkpoint = ckp, latent_type="OCs_proj",target = "speaker_seq"
                        )
                    if "emotion_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check emotion in s"
                        prediction_eval(data_training_args,config,
                            X = mu_projections_s, X_test = None,
                            y = torch.stack((emotion_seq,speaker_id_seq), dim = 1), y_test = None,
                            checkpoint = ckp, latent_type="OCs_proj",target = ["cat_emotion_seq", "speaker_seq"]
                        )

                if "all" in data_training_args.aggregations_to_use:
                    "All - ([X,OC1,OC2,...,OCn])"
                    if "speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_all_s, X_test = None,
                            y = speaker_id_seq, y_test = None,
                            checkpoint = ckp, latent_type="all",target = "speaker_seq"
                        )
                    if "emotion_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check emotion in s"
                        prediction_eval(data_training_args,config,
                            X = mu_all_s, X_test = None,
                            y = torch.stack((emotion_seq,speaker_id_seq), dim = 1), y_test = None,
                            checkpoint = ckp, latent_type="all",target = ["cat_emotion_seq", "speaker_seq"]
                        )

                if "X" in data_training_args.aggregations_to_use:
                    "Original X"
                    if "speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check speakers in s"
                        prediction_eval(data_training_args,config,
                            X = mu_originals_s, X_test = None,
                            y = speaker_id_seq, y_test = None,
                            checkpoint = ckp, latent_type="X",target = "speaker_seq"
                        )
                    if "emotion_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check emotion in s"
                        prediction_eval(data_training_args,config,
                            X = mu_originals_s, X_test = None,
                            y = torch.stack((emotion_seq,speaker_id_seq), dim = 1), y_test = None,
                            checkpoint = ckp, latent_type="X",target = ["cat_emotion_seq", "speaker_seq"]
                        )

                if "OCs" in data_training_args.aggregations_to_use:
                    "Individual OCs"
                    if "speaker_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check speakers in s"
                        for i in range(decomp_args.NoC_seq):                    
                            prediction_eval(data_training_args,config,
                                X = mu_components_s[i], X_test = None,
                                y = speaker_id_seq, y_test = None,
                                checkpoint = ckp, latent_type=f'OC{i+1}',target = "speaker_seq"
                            )
                    if "emotion_seq" in data_training_args.classification_tasks or "all" in data_training_args.classification_tasks:
                        "Check emotion in s"
                        for i in range(decomp_args.NoC_seq):
                            prediction_eval(data_training_args,config,
                                X = mu_components_s[i], X_test = None,
                                y = torch.stack((emotion_seq,speaker_id_seq), dim = 1), y_test = None,
                                checkpoint = ckp, latent_type=f'OC{i+1}',target = ["cat_emotion_seq", "speaker_seq"]
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


            if config.dual_branched_latent or config.only_z_branch:
                if "VOC_ALS" in data_training_args.dataset_name:
                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z - With King's Clinical Staging"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="all", mu_train = mu_all_z, y_train = y_frame_train_king, 
                            mu_test = None, y_test = None, target = ["king_stage_frame","phoneme_frame","speaker_frame"]
                        )

                        #"Check vowels/speakers disentanglement in z - With Disease Duration"
                        #compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                        #    latent_type="all", mu_train = mu_all_z, y_train = y_frame_train_dis, 
                        #    mu_test = None, y_test = None, target = ["disease_duration_frame","phoneme_frame","speaker_frame"]
                        #)
                    
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs joint embedding - ([OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z - With King's Clinical Staging"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_joint_emb", mu_train = mu_joint_components_z, y_train = y_frame_train_king, 
                            mu_test = None, y_test = None, target = ["king_stage_frame","phoneme_frame","speaker_frame"]
                        )
                    
                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z - With King's Clinical Staging"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_proj", mu_train = mu_projections_z, y_train = y_frame_train_king, 
                            mu_test = None, y_test = None, target = ["king_stage_frame","phoneme_frame","speaker_frame"]
                        )


                elif "iemocap" in data_training_args.dataset_name:
                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check vowels/speakers/emotions disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="all", mu_train = mu_all_z, y_train = y_frame_train, 
                            mu_test = None, y_test = None, target = ["phoneme","speaker_frame","cat_emotion_frame"]
                        )
                    
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs joint embedding - ([OC1,OC2,...,OCn])"
                        "Check vowels/speakers/emotions disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_joint_emb", mu_train = mu_joint_components_z, y_train = y_frame_train, 
                            mu_test = None, y_test = None, target = ["phoneme","speaker_frame","cat_emotion_frame"]
                        )
                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        "Check vowels/speakers/emotions disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_proj", mu_train = mu_projections_z, y_train = y_frame_train, 
                            mu_test = None, y_test = None, target = ["phoneme","speaker_frame","cat_emotion_frame"]
                        )


                else: #SimVowels and TIMIT
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs combination ([OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_joint_emb", mu_train = mu_joint_components_z, y_train = y_frame_train, 
                            mu_test = mu_joint_components_z_test, y_test = y_frame_test, target = ["vowel","speaker_frame"] if "vowels" in data_training_args.dataset_name else ["phoneme","speaker_frame"]
                        )
                    
                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_proj", mu_train = mu_projections_z, y_train = y_frame_train, 
                            mu_test = mu_projections_z_test, y_test = y_frame_test, target = ["vowel","speaker_frame"] if "vowels" in data_training_args.dataset_name else ["phoneme","speaker_frame"]
                        )

                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="all", mu_train = mu_all_z, y_train = y_frame_train, 
                            mu_test = mu_all_z_test, y_test = y_frame_test, target = ["vowel","speaker_frame"] if "vowels" in data_training_args.dataset_name else ["phoneme","speaker_frame"]
                        )
                    
                    if "X" in data_training_args.aggregations_to_use:
                        "Original X"
                        "Check vowels/speakers disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="X", mu_train = mu_originals_z, y_train = y_frame_train, 
                            mu_test = mu_originals_z_test, y_test = y_frame_test, target = ["vowel","speaker_frame"] if "vowels" in data_training_args.dataset_name else ["phoneme","speaker_frame"]
                        )
                    
                    if "OCs" in data_training_args.aggregations_to_use:
                        "Individual OCs"
                        "Check vowels/speakers disentanglement in z"
                        for i in range(decomp_args.NoC):                    
                            compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                                latent_type=f'OC{i+1}', mu_train = mu_components_z[i], y_train = y_frame_train, 
                                mu_test = mu_components_z_test[i], y_test = y_frame_test, target = ["vowel","speaker_frame"] if "vowels" in data_training_args.dataset_name else ["phoneme","speaker_frame"]
                            )
                    

            if config.dual_branched_latent or config.only_s_branch:
                if "VOC_ALS" in data_training_args.dataset_name:
                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z - With King's Clinical Staging"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="all", mu_train = mu_all_s, y_train = y_seq_train_king, 
                            mu_test = None, y_test = None, target = ["king_stage_seq","phoneme_seq","speaker_seq"]
                        )

                        #"Check vowels/speakers disentanglement in z - With Disease Duration"
                        #compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                        #    latent_type="all", mu_train = mu_all_s, y_train = y_seq_train_dis, 
                        #    mu_test = None, y_test = None, target = ["disease_duration_seq","phoneme_seq","speaker_seq"]
                        #)

                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs joint embedding - ([OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z - With King's Clinical Staging"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_joint_emb", mu_train = mu_joint_components_s, y_train = y_seq_train_king, 
                            mu_test = None, y_test = None, target = ["king_stage_seq","phoneme_seq","speaker_seq"]
                        )

                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        "Check vowels/speakers disentanglement in z - With King's Clinical Staging"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_proj", mu_train = mu_projections_s, y_train = y_seq_train_king, 
                            mu_test = None, y_test = None, target = ["king_stage_seq","phoneme_seq","speaker_seq"]
                        )


                elif "iemocap" in data_training_args.dataset_name:
                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check speakers/emotions disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="all", mu_train = mu_all_s, y_train = y_seq_train, 
                            mu_test = None, y_test = None, target = ["speaker_seq","cat_emotion_seq"]
                        )
                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs joint embedding - ([OC1,OC2,...,OCn])"
                        "Check speakers/emotions disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_joint_emb", mu_train = mu_joint_components_s, y_train = y_seq_train, 
                            mu_test = None, y_test = None, target = ["speaker_seq","cat_emotion_seq"]
                        )
                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        "Check speakers/emotions disentanglement in z"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_proj", mu_train = mu_projections_s, y_train = y_seq_train, 
                            mu_test = None, y_test = None, target = ["speaker_seq","cat_emotion_seq"]
                        )


                else: #SimVowels and TIMIT

                    if "OCs_joint_emb" in data_training_args.aggregations_to_use:
                        "OCs combination ([OC1,OC2,...,OCn])"
                        "Check only speakers in s"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_joint_emb", mu_train = mu_joint_components_s, y_train = y_seq_train, 
                            mu_test = mu_joint_components_s_test, y_test = y_seq_test, target = ["speaker_seq"]
                        )
                    if "OCs_proj" in data_training_args.aggregations_to_use and config.project_OCs:
                        "OCs projection <- f([OC1,OC2,...,OCn])"
                        "Check only speakers in s"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="OCs_proj", mu_train = mu_projections_s, y_train = y_seq_train, 
                            mu_test = mu_projections_s_test, y_test = y_seq_test, target = ["speaker_seq"]
                        )
                    if "all" in data_training_args.aggregations_to_use:
                        "All - ([X,OC1,OC2,...,OCn])"
                        "Check only speakers in s"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="all", mu_train = mu_all_s, y_train = y_seq_train, 
                            mu_test = mu_all_s_test, y_test = y_seq_test, target = ["speaker_seq"]
                        )
                    if "X" in data_training_args.aggregations_to_use:
                        "Original X"
                        "Check only speakers in s"
                        compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                            latent_type="X", mu_train = mu_originals_s, y_train = y_seq_train, 
                            mu_test = mu_originals_s_test, y_test = y_seq_test, target = ["speaker_seq"]
                        )
                    if "OCs" in data_training_args.aggregations_to_use:
                        "Individual OCs"
                        "Check only speakers in s"
                        for i in range(decomp_args.NoC_seq):                    
                            compute_disentanglement_metrics(data_training_args,config,checkpoint = ckp,
                                latent_type=f'OC{i+1}', mu_train = mu_components_s[i], y_train = y_seq_train, 
                                mu_test = mu_components_s_test[i], y_test = y_seq_test, target = ["speaker_seq"]
                            )


if __name__ == "__main__":
    main()