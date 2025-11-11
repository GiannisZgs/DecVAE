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

"""Calculate latent traversals for DecVAE models. This script loads pretrained DecVAE models from specified checkpoints,
or initializes random DecVAE models, and obtains the response of the model to a small set of traversal inputs.
We design the traversal datasets a priori to have a small controlled set where all instances of each factor occur in all possible combinations 
between factors and their instances. This is done through the load_traversal_subset_* functions in the dataset_loading module. Exception is the 
SimVowels dataset where we generate this subset with the scripts/simulations/simulated_vowels_for_latent_traversal.py script. 
The decomposition of these traversal subsets is also performed here.
Currently supports up to 5 components, for more it needs to be modified accordingly.
"""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

from models import DecVAEForPreTraining, DecompositionModule
from config_files import DecVAEConfig
from data_collation import DataCollatorForDecVAELatentTraversals
from data_preprocessing import prepare_traversal_dataset

from args_configs import ModelArgumentsPost, DataTrainingArgumentsPost, DecompositionArguments, TrainingObjectiveArguments
from dataset_loading import load_traversal_subset_timit, load_sim_vowels, load_traversal_subset_iemocap, load_traversal_subset_voc_als
from utils import parse_args, debugger_is_active, extract_epoch
from latent_analysis_utils import (
    calculate_variance_dimensions,
    save_latent_representation,
    average_latent_representations
)

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from safetensors.torch import load_file
from functools import partial
import json
import gzip
import numpy as np
import datasets
import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from torch.utils.data.dataloader import DataLoader
import warnings

warnings.simplefilter("ignore")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

JSON_FILE_NAME_MANUAL = "config_latent_traversals_voc_als_renderex.json" #for debugging purposes only

logger = get_logger(__name__)

CANONICAL_FORMANT_FREQUENCIES = {
    'a':  [710, 1100, 2540],
    'e': [550, 1770, 2490],
    'I': [400, 1920, 2560],
    'aw': [590, 880, 2540],
    'u': [310, 870, 2250],
}

CANONICAL_FORMANT_FREQUENCIES_EXTENDED = {
    'i': [280, 2250, 2890],
    'I': [400, 1920, 2560],
    'e': [550, 1770, 2490],
    'ae': [690, 1660, 2490],
    'a': [710, 1100, 2540],
    'aw': [590, 880, 2540],
    'y': [450, 1030, 2380],
    'u': [310, 870, 2250],
}


VOWEL_TO_INT = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES.keys())}
INT_TO_VOWEL = {idx: vowel for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES.keys())}

VOWEL_TO_INT_EXTENDED = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES_EXTENDED.keys())}
INT_TO_VOWEL_EXTENDED = {idx: vowel for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES_EXTENDED.keys())}

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

            wandb.init(project=data_training_args.wandb_project, group=data_training_args.wandb_group) #
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    "If passed along, set the training seed now."
    if data_training_args.seed is not None:
        set_seed(data_training_args.seed)

    accelerator.wait_for_everyone()
    
    "load cached preprocessed files"
    if data_training_args.preprocessing_num_workers is not None and data_training_args.preprocessing_num_workers > 1:
        if "vowels" in data_training_args.dataset_name:
            cache_file_names = {"train": [data_training_args.train_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]}
        else:
            cache_file_names = {"train": [data_training_args.train_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)],
                    "validation": [data_training_args.validation_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
            }
        if data_training_args.test_cache_file_name is not None:
            cache_file_names["test"] = [data_training_args.test_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
        if data_training_args.dev_cache_file_name is not None:
            cache_file_names["dev"] = [data_training_args.dev_cache_file_name[:-6] + "_0000"+str(i)+"_of_0000"+str(data_training_args.preprocessing_num_workers)+".arrow" for i in range(data_training_args.preprocessing_num_workers)]
    elif data_training_args.preprocessing_num_workers == 1 or data_training_args.preprocessing_num_workers is None:
        if "vowels" in data_training_args.dataset_name:
            cache_file_names = {"train": [data_training_args.train_cache_file_name]}
        else: 
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

            if data_training_args.dataset_name == "timit":
                with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                    phoneme39_to_id = json.load(json_file)
                with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'r') as json_file:
                    phoneme48_to_id = json.load(json_file)

    except FileNotFoundError:
        if "timit" in data_training_args.dataset_name:
            raw_datasets = load_traversal_subset_timit(data_training_args)
        
        elif "sim_vowels" in data_training_args.dataset_name:
            sdatasets = load_sim_vowels(data_training_args)

            raw_datasets = DatasetDict()
            raw_datasets["train"] = sdatasets["train"]
        
        elif "VOC_ALS" in data_training_args.dataset_name:
            raw_datasets = load_traversal_subset_voc_als(data_training_args)
        
        elif "iemocap" in data_training_args.dataset_name:
            raw_datasets = load_traversal_subset_iemocap(data_training_args)

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
        if "timit" in data_training_args.dataset_name:
            cache_file_names = {"fixed_phoneme": data_training_args.train_cache_file_name,
                                "fixed_speaker": data_training_args.validation_cache_file_name
                            }
        elif "iemocap" in data_training_args.dataset_name:
            cache_file_names = {"fixed_emotion_phoneme_speaker": data_training_args.train_cache_file_name,
                                "fixed_phoneme_emotion": data_training_args.validation_cache_file_name,
                                "fixed_speaker_emotion": data_training_args.dev_cache_file_name,
                                "fixed_nonverbal_emotion": data_training_args.test_cache_file_name
                            }
        elif "VOC_ALS" in data_training_args.dataset_name:
            cache_file_names = {"fixed_phoneme": data_training_args.train_cache_file_name,
                                "fixed_kings_stage": data_training_args.validation_cache_file_name
                            }
            
        elif "vowels" in data_training_args.dataset_name:
            if data_training_args.train_cache_file_name is not None:
                cache_file_names = {"train": data_training_args.train_cache_file_name}
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

            if "vowels" in data_training_args.dataset_name:
                set_id = "train"
            elif data_training_args.dataset_name in ["timit","VOC_ALS"]:
                set_id = "fixed_phoneme"
            elif "iemocap" in data_training_args.dataset_name:
                set_id = "fixed_phoneme_emotion"

            vectorized_datasets = raw_datasets.map(
                partial(
                    prepare_traversal_dataset,
                    feature_extractor=feature_extractor,
                    data_training_args=data_training_args,
                    decomp_args=decomp_args,
                    config=config,
                    max_length=max_length
                ),
                num_proc=data_training_args.preprocessing_num_workers,
                remove_columns=raw_datasets[set_id].column_names,
                load_from_cache_file=True,
                cache_file_names=cache_file_names
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

    "Set up the model - either initialize from scratch or load pretrained model"
    config.seq_length = max_length
    if hasattr(config,'dataset_name'):
        delattr(config,'dataset_name')
    
        
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

    if 'SSL_loss' in data_training_args.parent_dir:
        checkpoint_dir += "_" + str(data_training_args.ssl_loss_frame_perc) +"percent_frames"

    # Find the latest checkpoint
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if 'config' not in f]
    checkpoint_files.sort(key=extract_epoch)
    
    # Initialize random model and load pretrained weights
    representation_function = DecVAEForPreTraining(config)
    
    if checkpoint_files:
        checkpoint = checkpoint_files[data_training_args.which_checkpoint]
        print(f"Loading weights from checkpoint {checkpoint}")
        pretrained_model_file = os.path.join(checkpoint_dir, checkpoint, "model.safetensors")
        
        if os.path.exists(pretrained_model_file):
            weights = load_file(pretrained_model_file)
            keys_to_remove = [key for key in weights.keys() if 'project_hid' in key or 'project_q' in key]
            if keys_to_remove:
                for key in keys_to_remove:
                    del weights[key]
                    print(f"Removed deprecated module {key} from weights.")

            representation_function.load_state_dict(weights, strict=False)
            print(f"Successfully loaded pretrained weights from {pretrained_model_file}")
        else:
            print(f"Warning: Could not find model file at {pretrained_model_file}, initializing model from scratch")
    else:
        print(f"Warning: No checkpoints found in {checkpoint_dir}, initializing model from scratch")

    representation_function.eval()
    for param in representation_function.parameters():
        param.requires_grad = False

    "data collator"
    mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
    mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

    data_collator = DataCollatorForDecVAELatentTraversals(
        model=representation_function,
        feature_extractor=feature_extractor,
        model_args=model_args,
        data_training_args=data_training_args,
        config=config,
        input_type = data_training_args.input_type,
        pad_to_multiple_of=data_training_args.pad_to_multiple_of,
        mask_time_prob=mask_time_prob,
        mask_time_length=mask_time_length,
        dataset_name = data_training_args.dataset_name,
    )

    if "timit" in data_training_args.dataset_name:
        to_use = {
            'fixed_phoneme': 'train', 
            'fixed_speaker': 'validation', 
        }
    elif "iemocap" in data_training_args.dataset_name:
        to_use = {
            'fixed_emotion_phoneme_speaker': 'train', 
            'fixed_phoneme_emotion': 'validation', 
            'fixed_speaker_emotion':'dev',
            'fixed_nonverbal_emotion':'test'
        }
    elif "VOC_ALS" in data_training_args.dataset_name:
        to_use = {
            'fixed_phoneme': 'train', 
            'fixed_kings_stage':'validation'
        }
    elif "vowels" in data_training_args.dataset_name:
        to_use = {
            'fixed_vowels': 'train', 
            'fixed_speakers': 'validation', 
        }

    try:
        train_dataloader = DataLoader(
            vectorized_datasets[data_training_args.experiment],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=[dat.num_rows for name, dat in vectorized_datasets.items() if name == data_training_args.experiment][0],
        )
    except KeyError: 
        train_dataloader = DataLoader(
            vectorized_datasets[to_use[data_training_args.experiment]],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=[dat.num_rows for name, dat in vectorized_datasets.items() if name == to_use[data_training_args.experiment]][0],
        )

    "For vowels, we have already created the dataset in another script"
    "For real data, the was created here before doing the traversals"

    "Load the single batch of the dataloader for the traversal analysis"
    x_batch = next(iter(train_dataloader))
    batch_size = x_batch["input_values"].shape[0]
        
    sub_attention_mask = x_batch.pop("sub_attention_mask")
    if "vowels" in data_training_args.dataset_name:
        x_batch["mask_time_indices"] = torch.ones_like(x_batch["mask_time_indices"])
        vowels = x_batch.pop("vowel_labels")
        speaker_vt_factor = np.round(x_batch.pop("speaker_vt_factor"),decimals=3)
        if hasattr(x_batch, "overlap_mask"):
            x_batch.pop("overlap_mask")
    elif data_training_args.dataset_name in ["timit", "iemocap"]:
        x_batch.pop("start_phonemes")
        x_batch.pop("stop_phonemes")
        overlap_mask = x_batch.pop("overlap_mask")
        speaker_id = x_batch.pop("speaker_id")
        if "timit" in data_training_args.dataset_name:
            x_batch.pop("phonemes48")
            phonemes = x_batch.pop("phonemes39")
        elif "iemocap" in data_training_args.dataset_name:
            phonemes = x_batch.pop("phonemes")
            emotion_labels = x_batch.pop("emotion")
        
    elif "VOC_ALS" in data_training_args.dataset_name:
        king_stage = list(x_batch.pop("king_stage", None))
        phonemes = list(x_batch.pop("phonemes", None))
        x_batch.pop("alsfrs_total", None)
        x_batch.pop("alsfrs_speech", None)
        x_batch.pop("cantagallo", None)
        x_batch.pop("speaker_id", None)
        x_batch.pop("group", None)
        x_batch.pop("disease_duration", None)

    
    if not "vowels" in data_training_args.dataset_name:
        if "VOC_ALS" in data_training_args.dataset_name:
            overlap_mask = torch.zeros_like(sub_attention_mask,dtype=torch.bool)
        "Frames corresponding to padding are set as True in the overlap and discarded"
        padded = sub_attention_mask.sum(dim = -1)
        for b in range(batch_size):
            overlap_mask[b,padded[b]:] = 1
        overlap_mask = overlap_mask.bool()
        x_batch["mask_time_indices"] = sub_attention_mask.clone()
        #mask_time_indices set above for vowels

        "Labels"
        if data_training_args.dataset_name in ["timit", "iemocap"]:
            phonemes = phonemes[~overlap_mask]
            speaker_id = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask[i]))]) for i,factor in enumerate(speaker_id)])
            if "iemocap" in data_training_args.dataset_name:
                emotion = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask[i]))]) for i,factor in enumerate(emotion_labels)])
        elif "VOC_ALS" in data_training_args.dataset_name:
            king_stage = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask[i]))]) for i,factor in enumerate(king_stage)]) 
            phonemes = torch.cat([torch.tensor([factor for j in range(sum(~overlap_mask[i]))]) for i,factor in enumerate(phonemes)])
            
        overlap_mask = overlap_mask[sub_attention_mask]
    
    "In case of dual model we also need to pass a sequence"
    if not hasattr(x_batch,"input_seq_values") and model_args.dual_branched_latent:
        repeatfor = int(4/decomp_args.receptive_field)
        x_batch["input_seq_values"] = x_batch["input_values"].repeat(1,1,1,repeatfor).squeeze(2)

    
    outputs = representation_function(**x_batch)
    if not "vowels" in data_training_args.dataset_name:
        mu_OCs_z = torch.masked_select(outputs.mu_components_z,~overlap_mask[None,:,None]).reshape(outputs.mu_components_z.shape[0],-1,outputs.mu_components_z.shape[-1])
        logvar_OCs_z = torch.masked_select(outputs.logvar_components_z,~overlap_mask[None,:,None]).reshape(outputs.logvar_components_z.shape[0],-1,outputs.logvar_components_z.shape[-1]) 
    else:
        mu_OCs_z = outputs.mu_components_z
        logvar_OCs_z = outputs.logvar_components_z
    mu_OC1_z = mu_OCs_z[0].detach()
    logvar_OC1_z = logvar_OCs_z[0].detach()
    mu_OC2_z = mu_OCs_z[1].detach()
    logvar_OC2_z = logvar_OCs_z[1].detach()
    if decomp_args.NoC >= 3:
        mu_OC3_z = mu_OCs_z[2].detach()
        logvar_OC3_z = logvar_OCs_z[2].detach()
    if decomp_args.NoC >= 4:
        mu_OC4_z = mu_OCs_z[3].detach()
        logvar_OC4_z = logvar_OCs_z[3].detach()
    if decomp_args.NoC >= 5:
        mu_OC5_z = mu_OCs_z[4].detach()
        logvar_OC5_z = logvar_OCs_z[4].detach()
    mu_OCs_z = torch.cat([b.reshape(1,-1) for b in mu_OCs_z.transpose(0,1)]).detach().numpy()
    logvar_OCs_z = torch.cat([b.reshape(1,-1) for b in logvar_OCs_z.transpose(0,1)])
    if not "vowels" in data_training_args.dataset_name:
        mu_X_z = torch.masked_select(outputs.mu_originals_z,~overlap_mask[:,None]).reshape(-1,outputs.mu_originals_z.shape[-1]).detach()
        logvar_X_z = torch.masked_select(outputs.logvar_originals_z,~overlap_mask[:,None]).reshape(-1,outputs.logvar_originals_z.shape[-1]).detach()

        if config.project_OCs:
            mu_OCs_proj_z = torch.masked_select(outputs.mu_projections_z,~overlap_mask[:,None]).reshape(-1,outputs.mu_projections_z.shape[-1]).detach()
            logvar_OCs_proj_z = torch.masked_select(outputs.logvar_projections_z,~overlap_mask[:,None]).reshape(-1,outputs.logvar_projections_z.shape[-1]).detach()
    else:
        mu_X_z = outputs.mu_originals_z.detach()
        logvar_X_z = outputs.logvar_originals_z.detach()
        if config.project_OCs:
            mu_OCs_proj_z = outputs.mu_projections_z.detach()
            logvar_OCs_proj_z = outputs.logvar_projections_z.detach()

    "Data will be saved here"
    if data_training_args.dataset_name in ["VOC_ALS","iemocap"]:
        store_dir = os.path.join(data_training_args.output_dir,
                f'{data_training_args.dataset_name}_{decomp_args.decomp_to_perform}_transfer_from_{data_training_args.transfer_from}_{data_training_args.experiment}',
                betas[1:] + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type,
                checkpoint)
    else:
        store_dir = os.path.join(data_training_args.output_dir,
                f'{data_training_args.dataset_name}_{decomp_args.decomp_to_perform}_{data_training_args.experiment}',
                betas[1:] + "_NoC" + str(decomp_args.NoC) + "_" + data_training_args.input_type + "_" + model_type,
                checkpoint)
    os.makedirs(store_dir, exist_ok=True)

    if "timit" in data_training_args.dataset_name:
        # Prepare latent spaces dictionary
        latent_spaces = {
            'X': (mu_X_z, logvar_X_z),
            'OC1': (mu_OC1_z, logvar_OC1_z),
            'OC2': (mu_OC2_z, logvar_OC2_z)
        }
        
        if decomp_args.NoC >= 3:
            latent_spaces['OC3'] = (mu_OC3_z, logvar_OC3_z)
        if decomp_args.NoC >= 4:
            latent_spaces['OC4'] = (mu_OC4_z, logvar_OC4_z)
        if decomp_args.NoC >= 5:
            latent_spaces['OC5'] = (mu_OC5_z, logvar_OC5_z)
        if config.project_OCs:
            latent_spaces['OCs_proj'] = (mu_OCs_proj_z, logvar_OCs_proj_z)
        
        # Prepare factor labels
        factor_labels = {
            'phoneme': phonemes,
            'speaker': speaker_id
        }
        
        # Get averaged representations
        result = average_latent_representations(
            latent_spaces, 
            factor_labels, 
            data_training_args.experiment
        )
        
        # Extract results
        avg_mus = result['avg_mus']
        avg_logvars = result['avg_logvars']
        phonemes = result['factor_tensors']['phoneme']
        speaker_id = result['factor_tensors']['speaker']
        
        # Update the latent representations with averaged values
        mu_X_z = avg_mus['X']
        logvar_X_z = avg_logvars['X']
        mu_OC1_z = avg_mus['OC1']
        logvar_OC1_z = avg_logvars['OC1']
        mu_OC2_z = avg_mus['OC2']
        logvar_OC2_z = avg_logvars['OC2']
        if decomp_args.NoC >= 3:
            mu_OC3_z = avg_mus['OC3']
            logvar_OC3_z = avg_logvars['OC3']
        if decomp_args.NoC >= 4:
            mu_OC4_z = avg_mus['OC4']
            logvar_OC4_z = avg_logvars['OC4']
        if decomp_args.NoC >= 5:
            mu_OC5_z = avg_mus['OC5']
            logvar_OC5_z = avg_logvars['OC5']
        if config.project_OCs:
            mu_OCs_proj_z = avg_mus['OCs_proj']
            logvar_OCs_proj_z = avg_logvars['OCs_proj']

    elif "iemocap" in data_training_args.dataset_name:
        with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
            phoneme_to_id = json.load(json_file)
        with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'r') as json_file:
            emotion_to_id = json.load(json_file)
        with open(data_training_args.path_to_iemocap_speaker_dict_file, 'r') as json_file:
            speaker_to_id = json.load(json_file)
        
        id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
        id_to_emotion = {v: k for k, v in emotion_to_id.items()}
        id_to_speaker = {v: k for k, v in speaker_to_id.items()}
  
        # Prepare latent spaces dictionary
        latent_spaces = {
            'X': (mu_X_z, logvar_X_z),
            'OC1': (mu_OC1_z, logvar_OC1_z),
            'OC2': (mu_OC2_z, logvar_OC2_z)
        }
        
        if decomp_args.NoC >= 3:
            latent_spaces['OC3'] = (mu_OC3_z, logvar_OC3_z)
        if decomp_args.NoC >= 4:
            latent_spaces['OC4'] = (mu_OC4_z, logvar_OC4_z)
        if decomp_args.NoC >= 5:
            latent_spaces['OC5'] = (mu_OC5_z, logvar_OC5_z)
        if config.project_OCs:
            latent_spaces['OCs_proj'] = (mu_OCs_proj_z, logvar_OCs_proj_z)
        
        
        if data_training_args.experiment == "fixed_emotion_phoneme_speaker":
            factor_labels = {
                'phoneme': phonemes,
                'speaker': speaker_id,
                'emotion': emotion
            }
            
            result = average_latent_representations(
                latent_spaces, 
                factor_labels, 
                data_training_args.experiment
            )
            # Process 3-factor case results
            emotion_avg_mus = result['emotion_avg_mus']
            emotion_avg_logvars = result['emotion_avg_logvars']
            emotion_phoneme_speaker_pairs = result['emotion_phoneme_speaker_pairs']
            all_emotions = result['all_emotions']
            
            # For each emotion, process and save results separately
            for emotion_val in all_emotions:
                emotion_name = id_to_emotion[emotion_val.item()]
                pairs = emotion_phoneme_speaker_pairs[emotion_val.item()]
                
                if not pairs:
                    continue  # Skip if no pairs for this emotion
                    
                # Extract phonemes and speakers from pairs
                phoneme_vals = torch.tensor([p for p, s in pairs])
                speaker_vals = torch.tensor([s for p, s in pairs])
                
                # Process each latent space for this emotion
                for latent_name in ['X', 'OC1', 'OC2', 'OC3', 'OC4', 'OC5', 'OCs_proj']:
                    if latent_name not in emotion_avg_mus[emotion_val.item()]:
                        continue
                        
                    # Get the averaged latent values for this emotion
                    mu_data = emotion_avg_mus[emotion_val.item()][latent_name]
                    logvar_data = emotion_avg_logvars[emotion_val.item()][latent_name]
                    
                    # Calculate variance dimensions for this latent space and emotion
                    latent_results = calculate_variance_dimensions(mu_data, logvar_data, latent_name)
                    
                    # Save the data
                    fname = os.path.join(store_dir, f"{latent_name}_varying_phonemes_speakers_fixed_emotion_{emotion_name}.json")
                    data_dict = {
                        'mu': latent_results['reduced_min_max_var'].detach().cpu().numpy().tolist(),
                        'logvar': latent_results['logvar_reduced_min_max_var'].detach().cpu().numpy().tolist(),
                        'phoneme': phoneme_vals.detach().cpu().numpy().tolist(),
                        'speaker': speaker_vals.detach().cpu().numpy().tolist(),
                        'emotion': [emotion_val.item()] * len(pairs),
                        'var_dims': latent_results['var_dims'].detach().cpu().numpy().tolist(),
                        'min_var_latents': latent_results['min_var_dims'].detach().cpu().numpy().tolist(),
                        'varying': ['phoneme_speaker'],
                        'fixed_factor': 'emotion',
                        'fixed_value': emotion_name,
                        'latent': [latent_name]
                    }
                    
                    with gzip.open(fname, "wt") as f:
                        json.dump(data_dict, f)
        
        elif data_training_args.experiment == "fixed_phoneme_emotion":
            # Prepare factor labels specific to this experiment
            factor_labels = {
                'phoneme': phonemes,
                'emotion': emotion
            }
            
            # Get averaged representations specific to this experiment
            result = average_latent_representations(
                latent_spaces, 
                factor_labels, 
                "fixed_phoneme_emotion"
            )
            
            # Process 2-factor case results
            avg_mus = result['avg_mus']
            avg_logvars = result['avg_logvars']
            factor_tensors = result['factor_tensors']
            
            # Update the latent representations with averaged values
            mu_X_z = avg_mus['X']
            logvar_X_z = avg_logvars['X']
            mu_OC1_z = avg_mus['OC1']
            logvar_OC1_z = avg_logvars['OC1']
            mu_OC2_z = avg_mus['OC2']
            logvar_OC2_z = avg_logvars['OC2']
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                mu_OC3_z = avg_mus['OC3']
                logvar_OC3_z = avg_logvars['OC3']
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                mu_OC4_z = avg_mus['OC4']
                logvar_OC4_z = avg_logvars['OC4']
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                mu_OC5_z = avg_mus['OC5']
                logvar_OC5_z = avg_logvars['OC5']
            if config.project_OCs and 'OCs_proj' in avg_mus:
                mu_OCs_proj_z = avg_mus['OCs_proj']
                logvar_OCs_proj_z = avg_logvars['OCs_proj']
                
            # Calculate variance dimensions
            X_results = calculate_variance_dimensions(mu_X_z, logvar_X_z, "X")
            OC1_results = calculate_variance_dimensions(mu_OC1_z, logvar_OC1_z, "OC1")
            OC2_results = calculate_variance_dimensions(mu_OC2_z, logvar_OC2_z, "OC2")
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                OC3_results = calculate_variance_dimensions(mu_OC3_z, logvar_OC3_z, "OC3")
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                OC4_results = calculate_variance_dimensions(mu_OC4_z, logvar_OC4_z, "OC4")
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                OC5_results = calculate_variance_dimensions(mu_OC5_z, logvar_OC5_z, "OC5")
            if config.project_OCs and 'OCs_proj' in avg_mus:
                OCs_proj_results = calculate_variance_dimensions(mu_OCs_proj_z, logvar_OCs_proj_z, "OCs_proj")
                
            # Save latent representations with emotion varying, phoneme fixed
            save_latent_representation(X_results, factor_tensors['emotion'], factor_tensors['phoneme'], store_dir, 'X', 'emotion', 'phoneme', 'iemocap')
            save_latent_representation(OC1_results, factor_tensors['emotion'], factor_tensors['phoneme'], store_dir, 'OC1', 'emotion', 'phoneme', 'iemocap')
            save_latent_representation(OC2_results, factor_tensors['emotion'], factor_tensors['phoneme'], store_dir, 'OC2', 'emotion', 'phoneme', 'iemocap')
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                save_latent_representation(OC3_results, factor_tensors['emotion'], factor_tensors['phoneme'], store_dir, 'OC3', 'emotion', 'phoneme', 'iemocap')
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                save_latent_representation(OC4_results, factor_tensors['emotion'], factor_tensors['phoneme'], store_dir, 'OC4', 'emotion', 'phoneme', 'iemocap')
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                save_latent_representation(OC5_results, factor_tensors['emotion'], factor_tensors['phoneme'], store_dir, 'OC5', 'emotion', 'phoneme', 'iemocap')
            if config.project_OCs and 'OCs_proj' in avg_mus:
                save_latent_representation(OCs_proj_results, factor_tensors['emotion'], factor_tensors['phoneme'], store_dir, 'OCs_proj', 'emotion', 'phoneme', 'iemocap')
                
        elif data_training_args.experiment == "fixed_speaker_emotion":
            # Prepare factor labels specific to this experiment
            factor_labels = {
                'speaker': speaker_id,
                'emotion': emotion
            }
            
            # Get averaged representations specific to this experiment
            result = average_latent_representations(
                latent_spaces, 
                factor_labels, 
                "fixed_speaker_emotion"
            )
            
            # Process 2-factor case results
            avg_mus = result['avg_mus']
            avg_logvars = result['avg_logvars']
            factor_tensors = result['factor_tensors']
            
            # Update the latent representations with averaged values
            mu_X_z = avg_mus['X']
            logvar_X_z = avg_logvars['X']
            mu_OC1_z = avg_mus['OC1']
            logvar_OC1_z = avg_logvars['OC1']
            mu_OC2_z = avg_mus['OC2']
            logvar_OC2_z = avg_logvars['OC2']
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                mu_OC3_z = avg_mus['OC3']
                logvar_OC3_z = avg_logvars['OC3']
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                mu_OC4_z = avg_mus['OC4']
                logvar_OC4_z = avg_logvars['OC4']
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                mu_OC5_z = avg_mus['OC5']
                logvar_OC5_z = avg_logvars['OC5']
            if config.project_OCs and 'OCs_proj' in avg_mus:
                mu_OCs_proj_z = avg_mus['OCs_proj']
                logvar_OCs_proj_z = avg_logvars['OCs_proj']
                
            # Calculate variance dimensions
            X_results = calculate_variance_dimensions(mu_X_z, logvar_X_z, "X")
            OC1_results = calculate_variance_dimensions(mu_OC1_z, logvar_OC1_z, "OC1")
            OC2_results = calculate_variance_dimensions(mu_OC2_z, logvar_OC2_z, "OC2")
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                OC3_results = calculate_variance_dimensions(mu_OC3_z, logvar_OC3_z, "OC3")
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                OC4_results = calculate_variance_dimensions(mu_OC4_z, logvar_OC4_z, "OC4")
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                OC5_results = calculate_variance_dimensions(mu_OC5_z, logvar_OC5_z, "OC5")
            if config.project_OCs and 'OCs_proj' in avg_mus:
                OCs_proj_results = calculate_variance_dimensions(mu_OCs_proj_z, logvar_OCs_proj_z, "OCs_proj")
                
            # Save latent representations with emotion varying, speaker fixed
            save_latent_representation(X_results, factor_tensors['emotion'], factor_tensors['speaker'], store_dir, 'X', 'emotion', 'speaker', 'iemocap')
            save_latent_representation(OC1_results, factor_tensors['emotion'], factor_tensors['speaker'], store_dir, 'OC1', 'emotion',  'speaker', 'iemocap')
            save_latent_representation(OC2_results, factor_tensors['emotion'], factor_tensors['speaker'], store_dir, 'OC2', 'emotion',  'speaker', 'iemocap')
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                save_latent_representation(OC3_results, factor_tensors['emotion'], factor_tensors['speaker'], store_dir, 'OC3', 'emotion',  'speaker', 'iemocap')
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                save_latent_representation(OC4_results, factor_tensors['emotion'], factor_tensors['speaker'], store_dir, 'OC4', 'emotion',  'speaker', 'iemocap')
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                save_latent_representation(OC5_results, factor_tensors['emotion'], factor_tensors['speaker'], store_dir, 'OC5', 'emotion', 'speaker', 'iemocap')
            if config.project_OCs and 'OCs_proj' in avg_mus:
                save_latent_representation(OCs_proj_results, factor_tensors['emotion'], factor_tensors['speaker'], store_dir, 'OCs_proj', 'emotion', 'speaker', 'iemocap')
                
        elif data_training_args.experiment == "fixed_nonverbal_emotion":
            # Prepare factor labels specific to this experiment
            factor_labels = {
                'phoneme': phonemes,  # Nonverbal phonemes
                'emotion': emotion
            }
            
            # Get averaged representations specific to this experiment
            result = average_latent_representations(
                latent_spaces, 
                factor_labels, 
                "fixed_nonverbal_emotion"
            )
            
            # Process 2-factor case results
            avg_mus = result['avg_mus']
            avg_logvars = result['avg_logvars']
            factor_tensors = result['factor_tensors']
            
            # Update the latent representations with averaged values
            mu_X_z = avg_mus['X']
            logvar_X_z = avg_logvars['X']
            mu_OC1_z = avg_mus['OC1']
            logvar_OC1_z = avg_logvars['OC1']
            mu_OC2_z = avg_mus['OC2']
            logvar_OC2_z = avg_logvars['OC2']
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                mu_OC3_z = avg_mus['OC3']
                logvar_OC3_z = avg_logvars['OC3']
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                mu_OC4_z = avg_mus['OC4']
                logvar_OC4_z = avg_logvars['OC4']
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                mu_OC5_z = avg_mus['OC5']
                logvar_OC5_z = avg_logvars['OC5']
            if config.project_OCs and 'OCs_proj' in avg_mus:
                mu_OCs_proj_z = avg_mus['OCs_proj']
                logvar_OCs_proj_z = avg_logvars['OCs_proj']
                
            # Calculate variance dimensions
            X_results = calculate_variance_dimensions(mu_X_z, logvar_X_z, "X")
            OC1_results = calculate_variance_dimensions(mu_OC1_z, logvar_OC1_z, "OC1")
            OC2_results = calculate_variance_dimensions(mu_OC2_z, logvar_OC2_z, "OC2")
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                OC3_results = calculate_variance_dimensions(mu_OC3_z, logvar_OC3_z, "OC3")
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                OC4_results = calculate_variance_dimensions(mu_OC4_z, logvar_OC4_z, "OC4")
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                OC5_results = calculate_variance_dimensions(mu_OC5_z, logvar_OC5_z, "OC5")
            if config.project_OCs and 'OCs_proj' in avg_mus:
                OCs_proj_results = calculate_variance_dimensions(mu_OCs_proj_z, logvar_OCs_proj_z, "OCs_proj")
                
            # Save latent representations with emotion varying, nonverbal phoneme fixed
            save_latent_representation(X_results, factor_tensors['emotion'], factor_tensors['nonverbal'], store_dir, 'X', 'emotion', 'phoneme', 'iemocap')
            save_latent_representation(OC1_results, factor_tensors['emotion'], factor_tensors['nonverbal'], store_dir, 'OC1', 'emotion', 'phoneme', 'iemocap')
            save_latent_representation(OC2_results, factor_tensors['emotion'], factor_tensors['nonverbal'], store_dir, 'OC2', 'emotion', 'phoneme', 'iemocap')
            if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
                save_latent_representation(OC3_results, factor_tensors['emotion'], factor_tensors['nonverbal'], store_dir, 'OC3', 'emotion', 'phoneme', 'iemocap')
            if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
                save_latent_representation(OC4_results, factor_tensors['emotion'], factor_tensors['nonverbal'], store_dir, 'OC4', 'emotion', 'phoneme', 'iemocap')
            if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
                save_latent_representation(OC5_results, factor_tensors['emotion'], factor_tensors['nonverbal'], store_dir, 'OC5', 'emotion', 'phoneme', 'iemocap')
            if config.project_OCs and 'OCs_proj' in avg_mus:
                save_latent_representation(OCs_proj_results, factor_tensors['emotion'], factor_tensors['nonverbal'], store_dir, 'OCs_proj', 'emotion', 'phoneme', 'iemocap')

    elif "VOC_ALS" in data_training_args.dataset_name:
        # Prepare latent spaces dictionary
        latent_spaces = {
            'X': (mu_X_z, logvar_X_z),
            'OC1': (mu_OC1_z, logvar_OC1_z),
            'OC2': (mu_OC2_z, logvar_OC2_z)
        }
        
        if decomp_args.NoC >= 3:
            latent_spaces['OC3'] = (mu_OC3_z, logvar_OC3_z)
        if decomp_args.NoC >= 4:
            latent_spaces['OC4'] = (mu_OC4_z, logvar_OC4_z)
        if decomp_args.NoC >= 5:
            latent_spaces['OC5'] = (mu_OC5_z, logvar_OC5_z)
        if config.project_OCs:
            latent_spaces['OCs_proj'] = (mu_OCs_proj_z, logvar_OCs_proj_z)
        
        factor_labels = {
                'phoneme': phonemes,
                'king_stage': king_stage
        }
        
        result = average_latent_representations(
            latent_spaces, 
            factor_labels, 
            data_training_args.experiment
        )
        
        # Extract results
        avg_mus = result['avg_mus']
        avg_logvars = result['avg_logvars']
        factor_tensors = result['factor_tensors']

        
        # Update the latent representations with averaged values
        mu_X_z = avg_mus['X']
        logvar_X_z = avg_logvars['X']
        mu_OC1_z = avg_mus['OC1']
        logvar_OC1_z = avg_logvars['OC1']
        mu_OC2_z = avg_mus['OC2']
        logvar_OC2_z = avg_logvars['OC2']
        if decomp_args.NoC >= 3:
            mu_OC3_z = avg_mus['OC3']
            logvar_OC3_z = avg_logvars['OC3']
        if decomp_args.NoC >= 4:
            mu_OC4_z = avg_mus['OC4']
            logvar_OC4_z = avg_logvars['OC4']
        if decomp_args.NoC >= 5:
            mu_OC5_z = avg_mus['OC5']
            logvar_OC5_z = avg_logvars['OC5']
        if config.project_OCs:
            mu_OCs_proj_z = avg_mus['OCs_proj']
            logvar_OCs_proj_z = avg_logvars['OCs_proj']

        # Calculate variance dimensions
        X_results = calculate_variance_dimensions(mu_X_z, logvar_X_z, "X")
        OC1_results = calculate_variance_dimensions(mu_OC1_z, logvar_OC1_z, "OC1")
        OC2_results = calculate_variance_dimensions(mu_OC2_z, logvar_OC2_z, "OC2")
        if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
            OC3_results = calculate_variance_dimensions(mu_OC3_z, logvar_OC3_z, "OC3")
        if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
            OC4_results = calculate_variance_dimensions(mu_OC4_z, logvar_OC4_z, "OC4")
        if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
            OC5_results = calculate_variance_dimensions(mu_OC5_z, logvar_OC5_z, "OC5")
        if config.project_OCs and 'OCs_proj' in avg_mus:
            OCs_proj_results = calculate_variance_dimensions(mu_OCs_proj_z, logvar_OCs_proj_z, "OCs_proj")

        if data_training_args.experiment == "fixed_phoneme":  
            factor_fixed = 'phoneme'
            factor_varying = 'king_stage'
        elif data_training_args.experiment == "fixed_kings_stage":
            factor_fixed = 'king_stage'
            factor_varying = 'phoneme'
        # Save latent representations 
        save_latent_representation(X_results, factor_tensors[factor_varying], factor_tensors[factor_fixed], store_dir, 'X', factor_varying, factor_fixed, 'VOC_ALS')
        save_latent_representation(OC1_results, factor_tensors[factor_varying], factor_tensors[factor_fixed], store_dir, 'OC1', factor_varying, factor_fixed, 'VOC_ALS')
        save_latent_representation(OC2_results, factor_tensors[factor_varying], factor_tensors[factor_fixed], store_dir, 'OC2', factor_varying, factor_fixed, 'VOC_ALS')
        if decomp_args.NoC >= 3 and 'OC3' in avg_mus:
            save_latent_representation(OC3_results, factor_tensors[factor_varying], factor_tensors[factor_fixed], store_dir, 'OC3', factor_varying, factor_fixed, 'VOC_ALS')
        if decomp_args.NoC >= 4 and 'OC4' in avg_mus:
            save_latent_representation(OC4_results, factor_tensors[factor_varying], factor_tensors[factor_fixed], store_dir, 'OC4', factor_varying, factor_fixed, 'VOC_ALS')
        if decomp_args.NoC >= 5 and 'OC5' in avg_mus:
            save_latent_representation(OC5_results, factor_tensors[factor_varying], factor_tensors[factor_fixed], store_dir, 'OC5', factor_varying, factor_fixed, 'VOC_ALS')
        if config.project_OCs and 'OCs_proj' in avg_mus:
            save_latent_representation(OCs_proj_results, factor_tensors[factor_varying], factor_tensors[factor_fixed], store_dir, 'OCs_proj', factor_varying, factor_fixed, 'VOC_ALS')
                

    if data_training_args.dataset_name in ["timit", "sim_vowels"]:
        "Max variance on individual spaces"
        X_results = calculate_variance_dimensions(mu_X_z, logvar_X_z, "X")
        OC1_results = calculate_variance_dimensions(mu_OC1_z, logvar_OC1_z, "OC1")
        OC2_results = calculate_variance_dimensions(mu_OC2_z, logvar_OC2_z, "OC2")
        if decomp_args.NoC >= 3:
            OC3_results = calculate_variance_dimensions(mu_OC3_z, logvar_OC3_z, "OC3")
        if decomp_args.NoC >= 4:
            OC4_results = calculate_variance_dimensions(mu_OC4_z, logvar_OC4_z, "OC4")
        if decomp_args.NoC >= 5:
            OC5_results = calculate_variance_dimensions(mu_OC5_z, logvar_OC5_z, "OC5")
        if config.project_OCs:
            OCs_proj_results = calculate_variance_dimensions(mu_OCs_proj_z, logvar_OCs_proj_z, "OCs_proj")

        if "vowels" in data_training_args.dataset_name:
            all_speakers = torch.unique(speaker_vt_factor)
            all_vowels = torch.unique(vowels)
            if data_training_args.sim_vowels_number == 5:
                vowels_str = [INT_TO_VOWEL[v.item()] for v in all_vowels]
                all_vowels_str = [INT_TO_VOWEL[v.item()] for v in vowels]
            elif data_training_args.sim_vowels_number == 8:
                vowels_str = [INT_TO_VOWEL_EXTENDED[v.item()] for v in all_vowels]
                all_vowels_str = [INT_TO_VOWEL_EXTENDED[v.item()] for v in vowels]
        elif "timit" in data_training_args.dataset_name:
            all_speakers = torch.unique(speaker_id)
            all_phonemes = torch.unique(phonemes)
            all_phonemes = all_phonemes[all_phonemes != -100]


        if "timit" in data_training_args.dataset_name:
            if data_training_args.experiment == "fixed_phoneme":
                "Data are already selected and filtered"
                save_latent_representation(X_results, speaker_id, phonemes, store_dir, 'X', 'speaker', 'phoneme', 'timit')
                if config.project_OCs:
                    save_latent_representation(OCs_proj_results, speaker_id, phonemes, store_dir, 'OCs_proj', 'speaker', 'phoneme', 'timit')
                save_latent_representation(OC1_results, speaker_id, phonemes, store_dir, 'OC1', 'speaker', 'phoneme', 'timit')
                save_latent_representation(OC2_results, speaker_id, phonemes, store_dir, 'OC2', 'speaker', 'phoneme', 'timit')
                if decomp_args.NoC >= 3:
                    save_latent_representation(OC3_results, speaker_id, phonemes, store_dir, 'OC3', 'speaker', 'phoneme', 'timit')
                if decomp_args.NoC >= 4:
                    save_latent_representation(OC4_results, speaker_id, phonemes, store_dir, 'OC4', 'speaker', 'phoneme', 'timit')
                if decomp_args.NoC >= 5:
                    save_latent_representation(OC5_results, speaker_id, phonemes, store_dir, 'OC5', 'speaker', 'phoneme', 'timit')
            elif data_training_args.experiment == "fixed_speaker":
                "Data are already selected and filtered"
                save_latent_representation(X_results, phonemes, speaker_id,  store_dir, 'X', 'phoneme', 'speaker', 'timit')
                if config.project_OCs:
                    save_latent_representation(OCs_proj_results, phonemes, speaker_id, store_dir, 'OCs_proj', 'phoneme', 'speaker', 'timit')
                save_latent_representation(OC1_results, phonemes, speaker_id, store_dir, 'OC1', 'phoneme', 'speaker', 'timit')
                save_latent_representation(OC2_results, phonemes, speaker_id, store_dir, 'OC2', 'phoneme', 'speaker', 'timit')
                if decomp_args.NoC >= 3:
                    save_latent_representation(OC3_results, phonemes, speaker_id, store_dir, 'OC3', 'phoneme', 'speaker', 'timit')
                if decomp_args.NoC >= 4:
                    save_latent_representation(OC4_results, phonemes, speaker_id, store_dir, 'OC4', 'phoneme', 'speaker', 'timit')
                if decomp_args.NoC >= 5:
                    save_latent_representation(OC5_results, phonemes, speaker_id, store_dir, 'OC5', 'phoneme', 'speaker', 'timit')

        elif "vowels" in data_training_args.dataset_name:
            if data_training_args.experiment == "fixed_vowels":
                "Select some speakers to avoid cluttered plots"
                sel_speakers = all_speakers[0:len(all_speakers):8]
                sel_vt_factors = []
                inds = []
                for i,f in enumerate(speaker_vt_factor):
                    if f in sel_speakers:
                        inds.append(i)
                        sel_vt_factors.append(f)
                sel_vt_factors = torch.cat(sel_vt_factors)
                sel_vowels = vowels[inds]

                sel_results = {
                    'X': {k: v[inds] if 'reduced' in k else v for k, v in X_results.items()},
                    'OC1': {k: v[inds] if 'reduced' in k else v for k, v in OC1_results.items()},
                    'OC2': {k: v[inds] if 'reduced' in k else v for k, v in OC2_results.items()}
                }
                if config.project_OCs:
                    sel_results['OCs_proj'] = {k: v[inds] if 'reduced' in k else v for k, v in OCs_proj_results.items()}
                if decomp_args.NoC >= 3:
                    sel_results['OC3'] = {k: v[inds] if 'reduced' in k else v for k, v in OC3_results.items()}
                if decomp_args.NoC >= 4:
                    sel_results['OC4'] = {k: v[inds] if 'reduced' in k else v for k, v in OC4_results.items()}
                if decomp_args.NoC >= 5:
                    sel_results['OC5'] = {k: v[inds] if 'reduced' in k else v for k, v in OC5_results.items()}

                # Save all latent representations using the helper function
                save_latent_representation(sel_results['X'], sel_vt_factors, sel_vowels, store_dir, 'X', 'speaker', 'vowel', 'sim_vowels', data_training_args.sim_vowels_number)
                if config.project_OCs:
                    save_latent_representation(sel_results['OCs_proj'], sel_vt_factors, sel_vowels, store_dir, 'OCs_proj', 'speaker', 'vowel', 'sim_vowels',data_training_args.sim_vowels_number)
                save_latent_representation(sel_results['OC1'], sel_vt_factors, sel_vowels, store_dir, 'OC1', 'speaker', 'vowel', 'sim_vowels',data_training_args.sim_vowels_number)
                save_latent_representation(sel_results['OC2'], sel_vt_factors, sel_vowels, store_dir, 'OC2', 'speaker', 'vowel', 'sim_vowels',data_training_args.sim_vowels_number)
                if decomp_args.NoC >= 3:
                    save_latent_representation(sel_results['OC3'], sel_vt_factors, sel_vowels, store_dir, 'OC3', 'speaker', 'vowel', 'sim_vowels',data_training_args.sim_vowels_number)
                if decomp_args.NoC >= 4:
                    save_latent_representation(sel_results['OC4'], sel_vt_factors, sel_vowels, store_dir, 'OC4', 'speaker', 'vowel', 'sim_vowels',data_training_args.sim_vowels_number)
                if decomp_args.NoC >= 5:
                    save_latent_representation(sel_results['OC5'], sel_vt_factors, sel_vowels, store_dir, 'OC5', 'speaker', 'vowel', 'sim_vowels',data_training_args.sim_vowels_number)

                
            elif data_training_args.experiment == "fixed_speakers":
                "Select some speakers to avoid cluttered plots"
                sel_speakers = all_speakers[0:len(all_speakers):8]#[10:51]
                sel_vt_factors = []
                inds = []
                for i,f in enumerate(speaker_vt_factor):
                    if f in sel_speakers:
                        inds.append(i)
                        sel_vt_factors.append(f)
                sel_vt_factors = torch.cat(sel_vt_factors)
                sel_vowels = vowels[inds]

                sel_results = {
                    'X': {k: v[inds] if 'reduced' in k else v for k, v in X_results.items()},
                    'OC1': {k: v[inds] if 'reduced' in k else v for k, v in OC1_results.items()},
                    'OC2': {k: v[inds] if 'reduced' in k else v for k, v in OC2_results.items()}                
                }
                if config.project_OCs:
                    sel_results['OCs_proj'] = {k: v[inds] if 'reduced' in k else v for k, v in OCs_proj_results.items()}
                if decomp_args.NoC >= 3:
                    sel_results['OC3'] = {k: v[inds] if 'reduced' in k else v for k, v in OC3_results.items()}
                if decomp_args.NoC >= 4:
                    sel_results['OC4'] = {k: v[inds] if 'reduced' in k else v for k, v in OC4_results.items()}
                if decomp_args.NoC >= 5:
                    sel_results['OC5'] = {k: v[inds] if 'reduced' in k else v for k, v in OC5_results.items()}

                # Save all latent representations using the helper function
                save_latent_representation(sel_results['X'], sel_vowels, sel_vt_factors, store_dir, 'X', 'vowel','speaker', 'sim_vowels', data_training_args.sim_vowels_number)
                if config.project_OCs:
                    save_latent_representation(sel_results['OCs_proj'], sel_vowels, sel_vt_factors, store_dir, 'OCs_proj', 'vowel','speaker', 'sim_vowels', data_training_args.sim_vowels_number)
                save_latent_representation(sel_results['OC1'], sel_vowels, sel_vt_factors, store_dir, 'OC1', 'vowel', 'speaker', 'sim_vowels',data_training_args.sim_vowels_number)
                save_latent_representation(sel_results['OC2'], sel_vowels, sel_vt_factors, store_dir, 'OC2', 'vowel', 'speaker', 'sim_vowels',data_training_args.sim_vowels_number)
                if decomp_args.NoC >= 3:
                    save_latent_representation(sel_results['OC3'], sel_vowels, sel_vt_factors, store_dir, 'OC3', 'vowel', 'speaker', 'sim_vowels',data_training_args.sim_vowels_number)
                if decomp_args.NoC >= 4:
                    save_latent_representation(sel_results['OC4'], sel_vowels, sel_vt_factors, store_dir, 'OC4', 'vowel', 'speaker', 'sim_vowels',data_training_args.sim_vowels_number)
                if decomp_args.NoC >= 5:
                    save_latent_representation(sel_results['OC5'], sel_vowels, sel_vt_factors, store_dir, 'OC5', 'vowel', 'speaker', 'sim_vowels',data_training_args.sim_vowels_number)
    
        

if __name__ == '__main__':
    main()
