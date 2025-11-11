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

"""This script visualizes input data to the DecVAE and VAE-based models through the latent_analysis.visualize utility. Different options for input data are provided - raw, decomposed, aggregated.
Visualization supports frequency embedding (samples colored by component correspondence), generative factors embedding (colored by labels), 2D and 3D space.
This script selects a number of instances of a class to avoid cluttered visualizations i.e. in cases where speakers are > 20, we select a few speakers only to visualize.
A PCA reduction is applied on the input data before they are fed to a TSNE or UMAP manifold algorithm. 
Check the global variables after the imports to set the parameters of the visualization.
Decomposition is supported here. 
Subgroups of generative factors are supported for visualization; e.g. in TIMIT we gather consonants and vowels that are subgroups of phonemes.
"""

"""
Dataset sizings:
IEMOCAP: seq - 5530, frame - 600 - small scale: 50 frames from 'all'
VOC_ALS: seq - 1224, frame - 300 - small scale: 30 frames from 'all'
TIMIT: train seq - 3458, train: 600, test, 192, dev 400 frame - small scale: 50 frames from dev
Vowels: seq train: 4000, test: 300, dev: 500 / frame train: 600, test:300, dev:500 - small scale: 30 frames from dev
"""

"""
Which variables are plotted 
TIMIT: ["phonemes39", "vowels", "consonants", "speaker_id"]
VOWELS: ["vowels", "speaker_id"]
IEMOCAP: ["phonemes", "emotion", "speaker_id"]
TIMIT_seq/Vowels_seq: ["speaker_id"]
VOC-ALS: ["speaker_id", "phoneme", "king_stage"]
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
from data_collation import DataCollatorForInputVisualization
from data_preprocessing import prepare_data_for_quality_assessment
from args_configs import ModelArguments, DataTrainingArguments, DecompositionArguments, TrainingObjectiveArguments, VisualizationsArguments
from dataset_loading import load_timit, load_sim_vowels, load_iemocap, load_voc_als
from utils.misc import parse_args, debugger_is_active, find_speaker_gender

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from functools import partial
import numpy as np
import os
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from huggingface_hub import HfApi
from torch.utils.data.dataloader import DataLoader
import json
import os
import sys 
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from feature_extraction import extract_mel_spectrogram
from latent_analysis_utils import visualize


JSON_FILE_NAME_MANUAL = "config_files/input_visualizations/config_visualizing_input_frames_voc_als.json" #for debugging purposes only

logger = get_logger(__name__)

def main():
    "Parse the arguments"       
    parser = HfArgumentParser((ModelArguments, TrainingObjectiveArguments, DecompositionArguments, DataTrainingArguments, VisualizationsArguments))
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

    assert data_training_args.input_type == "waveform"
    assert vis_args.seq_to_vis == vis_args.frames_to_vis, "Set vis_args.seq_to_vis equal to vis_args.frames_to_vis to make sure the correct number of samples are gathered. Run two times if different number of sequences and frames is needed."
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
    #if data_training_args.train_cache_file_name is not None and data_training_args.validation_cache_file_name is not None \
    #    and data_training_args.test_cache_file_name is not None:
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

    except FileNotFoundError: #else:#
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
                    prepare_data_for_quality_assessment,
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

    "initialize random model"
    config.seq_length = max_length
    model = DecVAEForPreTraining(config)

    "data collator, optimizer and scheduler"

    mask_time_prob = config.mask_time_prob if model_args.mask_time_prob is None else model_args.mask_time_prob
    mask_time_length = config.mask_time_length if model_args.mask_time_length is None else model_args.mask_time_length

    data_collator = DataCollatorForInputVisualization(
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
    )
    if vis_args.set_to_use_for_vis == 'all':
        if data_training_args.dataset_name == "VOC_ALS":
            vectorized_datasets['validation'] = concatenate_datasets([vectorized_datasets['validation'], vectorized_datasets['test'],vectorized_datasets['train'],vectorized_datasets['dev']])
        elif data_training_args.dataset_name == "iemocap":
            vectorized_datasets['validation'] = concatenate_datasets([vectorized_datasets['validation'], vectorized_datasets['test'],vectorized_datasets['train']])
        dataloader = DataLoader(
            vectorized_datasets['validation'],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=data_training_args.per_device_train_batch_size,
        )
    else:
        if data_training_args.dataset_name == "sim_vowels":
            dataloader = DataLoader(
                vectorized_datasets['validation' if vis_args.set_to_use_for_vis == 'dev' else vis_args.set_to_use_for_vis],
                shuffle=True,
                collate_fn=data_collator,
                batch_size=data_training_args.per_device_train_batch_size,
            )
        else:
            dataloader = DataLoader(
                vectorized_datasets[vis_args.set_to_use_for_vis],
                shuffle=True,
                collate_fn=data_collator,
                batch_size=data_training_args.per_device_train_batch_size,
            )

    "Initialize variables we will be gathering for plotting as we iterate the datasets"
    if data_training_args.dataset_name == 'timit':
        quality_frame = {'phonemes39': [], "vowels": [], "consonants": [], 'speaker_id': [], 'gender': [], 'mel': [], 'time_domain': [], 'overlap_mask': []}
    elif data_training_args.dataset_name == 'sim_vowels':
        quality_frame = {"vowels": [], 'speaker_id': [], 'gender': [], 'mel': [], 'time_domain': [], 'overlap_mask': []}
    elif data_training_args.dataset_name == 'VOC_ALS':
        quality_frame = {'group': [], 'alsfrs_total': [], 'alsfrs_speech': [], 'king_stage': [], 'cantagallo': [], 'speaker_id': [], 'phonemes':[], 'disease_duration': [],'mel': [], 'time_domain': [], 'overlap_mask': []}
    elif data_training_args.dataset_name == 'iemocap':
        quality_frame = {'emotion': [], 'speaker_id': [], 'gender': [], 'phonemes':[],'mel': [], 'time_domain': [], 'overlap_mask': []}

    if data_training_args.dataset_name == 'timit' or data_training_args.dataset_name == 'sim_vowels':
        quality_seq = {'speaker_id': [], 'gender': [], 'mel': [],'time_domain': []}
    elif data_training_args.dataset_name == 'VOC_ALS':
        quality_seq = {'group': [], 'alsfrs_total': [], 'alsfrs_speech': [], 'king_stage': [], 'cantagallo': [], 'speaker_id': [], 'phonemes':[], 'disease_duration': [], 'mel': [], 'time_domain': []}
    elif data_training_args.dataset_name == 'iemocap':
        quality_seq = {'emotion': [], 'speaker_id': [], 'gender': [], 'mel': [], 'time_domain': []}
    
    "Save one or several batches of data for frame, all batches for sequence"
    for step, batch in enumerate(dataloader):
        #print(step)
        batch_size = batch["input_values"].shape[0]
        assert batch_size == 1, "Set batch size to 1 so vis_args.frames_to_vis correspond to the actual number of samples"
        frames = batch["sub_attention_mask"].sum()
        frame_len = batch["input_values"].shape[-1]
        seq_len = batch["attention_mask"].sum()
        if hasattr(batch,"overlap_mask"):
            overlap_mask = batch["overlap_mask"].squeeze(0)
            for o,ov in enumerate(overlap_mask[:frames]):
                if not ov == 0 and not ov == 1:
                    overlap_mask[o] = 1 
            overlap_mask = overlap_mask[:frames]
        else:
            overlap_mask = torch.zeros_like(batch["sub_attention_mask"], dtype=torch.bool)
            "Frames corresponding to padding are set as True in the overlap and discarded"
            padded = batch["sub_attention_mask"].sum(dim = -1)
            for b in range(batch_size):
                overlap_mask[b,padded[b]:] = 1
            overlap_mask = overlap_mask.bool()
        if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
            quality_frame['overlap_mask'].append(overlap_mask.detach().cpu().numpy()) 
        else:
            #if frames to visualize is a number less than sequences to visualize
            pass

        if data_training_args.dataset_name == "timit":
            if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                "Discard padded frames"
                batch["input_values"] = batch["input_values"][:,:,:frames,...]

                "Phonemes39 (vowels/consonants)"
                with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                    phoneme39_to_id = json.load(json_file)
                id_to_phoneme39 = {v: k for k, v in phoneme39_to_id.items()}
                batch_phonemes = []
                batch_vowels = []
                batch_consonants = []
                pho39 = batch["phonemes39"].squeeze(0)[:frames]
                for ph in pho39:
                    if ph != -100:
                        batch_phonemes.append(id_to_phoneme39[ph.item()])
                        "Gather vowels in TIMIT"
                        if id_to_phoneme39[ph.item()] in ['ae','ao','aw','ax','ay','eh','er','ey','ih','ix','iy','ow','oy','uh','uw','y']:
                            batch_vowels.append(id_to_phoneme39[ph.item()])
                        else:
                            batch_vowels.append('NO')
                        "Gather consonants in TIMIT"
                        if id_to_phoneme39[ph.item()] in ['b','ch','d','dh','dx','f','g','hh','jh','k','l','m','n','p','r','s','sh','t','th','v','w','z']:
                            batch_consonants.append(id_to_phoneme39[ph.item()])
                        else:
                            batch_consonants.append('NO')
                    else:
                        batch_phonemes.append('sil')
                        batch_vowels.append('NO')
                        batch_consonants.append('NO')

                quality_frame["phonemes39"].append(batch_phonemes)
                quality_frame["vowels"].append(batch_vowels)
                quality_frame["consonants"].append(batch_consonants)
                        
            else:
                pass


            "Speaker_id (M/F)"
            with open(data_training_args.path_to_timit_speaker_dict_file, 'r') as json_file:
                speaker_id_to_id = json.load(json_file)
            speaker_id_to_id = {v: k for k, v in speaker_id_to_id.items()}
            if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                quality_frame["speaker_id"].append(batch["speaker_id"].detach().cpu().numpy().tolist())
            else:
                pass
            quality_seq["speaker_id"].append(batch["speaker_id"].detach().cpu().numpy().tolist())

            speaker_dir = os.path.join(data_training_args.data_dir,"/data/lisa/data/timit/raw/TIMIT/TRAIN")

            gender = find_speaker_gender(speaker_dir,speaker_id_to_id[batch["speaker_id"].item()])
            if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                quality_frame["gender"].append(gender)
            else:
                pass
            quality_seq["gender"].append(gender)

        elif data_training_args.dataset_name == "iemocap":
            if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                "Discard padded frames"
                batch["input_values"] = batch["input_values"][:,:,:frames,...]

                "Phonemes"
                with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
                    phoneme_to_id = json.load(json_file)
                id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
                batch_phonemes = []
                batch_vowels = []
                batch_consonants = []
                phonemes = batch["phonemes"].squeeze(0)[:frames]
                for ph in phonemes:
                    if ph != -100:
                        batch_phonemes.append(id_to_phoneme[ph.item()])
                    else:
                        batch_phonemes.append('SIL')

                quality_frame["phonemes"].append(batch_phonemes)                        
            else:
                pass


            "Speaker_id"
            with open(data_training_args.path_to_iemocap_speaker_dict_file, 'r') as json_file:
                speaker_id_to_id = json.load(json_file)
            id_to_speaker_id = {v: k for k, v in speaker_id_to_id.items()}
            if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                quality_frame["speaker_id"].append(batch["speaker_id"].item())
            else:
                pass
            quality_seq["speaker_id"].append(batch["speaker_id"].item()) 

            "Emotion"
            if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                quality_frame["emotion"].append(batch["emotion"].item()) 
            else:
                pass
            quality_seq["emotion"].append(batch["emotion"].item())

            if 'M' in id_to_speaker_id[batch["speaker_id"].item()]:
                quality_frame["gender"].append("M")
                quality_seq["gender"].append("M")
            elif 'F' in id_to_speaker_id[batch["speaker_id"].item()]:
                quality_frame["gender"].append("F")
                quality_seq["gender"].append("F")

        elif data_training_args.dataset_name == "sim_vowels":
            "Vowels"
            if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                quality_frame["vowels"].append(batch["vowel_labels"].squeeze(0)[:frames].detach().cpu().numpy().tolist())
                quality_frame["speaker_id"].append(batch["speaker_vt_factor"].detach().cpu().numpy().tolist())
            else:
                pass
            quality_seq["speaker_id"].append(batch["speaker_vt_factor"].detach().cpu().numpy().tolist())
            if batch["speaker_vt_factor"] <= 0.95:
                if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                    quality_frame["gender"].append("F")
                else:
                    pass
                quality_seq["gender"].append("F")
            elif batch["speaker_vt_factor"] >= 1.05:
                if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                    quality_frame["gender"].append("M")
                else:
                    pass
                quality_seq["gender"].append("M")
            else:
                if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
                    quality_frame["gender"].append("U")
                else:
                    pass
                quality_seq["gender"].append("U")

        elif data_training_args.dataset_name == "VOC_ALS":
            "Discard padded frames"
            batch["input_values"] = batch["input_values"][:,:,:frames,...]
            
            "Sequence variables"
            quality_seq["alsfrs_total"].append(list(batch["alsfrs_total"]))
            quality_seq["alsfrs_speech"].append(list(batch["alsfrs_speech"]))
            quality_seq["king_stage"].append(list(batch["king_stage"]))
            quality_seq["disease_duration"].append(list(batch["disease_duration"]))
            quality_seq["speaker_id"].append(list(batch["speaker_id"]))
            quality_seq["group"].append(list(batch["group"]))
            quality_seq["phonemes"].append(list(batch["phonemes"]))
            quality_seq["cantagallo"].append(list(batch["cantagallo"]))
            "Frame variables"
            quality_frame["alsfrs_total"].append(list(batch["alsfrs_total"]))
            quality_frame["alsfrs_speech"].append(list(batch["alsfrs_speech"]))
            quality_frame["king_stage"].append(list(batch["king_stage"]))
            quality_frame["disease_duration"].append(list(batch["disease_duration"]))
            quality_frame["speaker_id"].append(list(batch["speaker_id"]))
            quality_frame["group"].append(list(batch["group"]))
            quality_frame["phonemes"].append(list(batch["phonemes"]))
            quality_frame["cantagallo"].append(list(batch["cantagallo"]))
            

        "Time domain - Frame"
        if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis:
            quality_frame["time_domain"].append(batch["input_values"].squeeze(0).detach().cpu().numpy()) #.tolist())
        else:
            pass
        
        quality_seq["time_domain"].append(batch["input_seq_values"].squeeze(0).detach().cpu().numpy()) #.tolist())
        
        if step <vis_args.frames_to_vis or vis_args.frames_to_vis == vis_args.seq_to_vis and vis_args.vis_mel_frames:
            "Mel_features - Frame"
            mel_features = torch.zeros((batch_size,config.NoC+1,frames,frame_len))
            mel_features[:,0,...], spec_max = extract_mel_spectrogram(batch["input_values"][:,0,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = None)
            for o in range(1,batch["input_values"].shape[1]):
                mel_features[:,o,...],_ = extract_mel_spectrogram(batch["input_values"][:,o,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = spec_max)

            mel_features = mel_features.squeeze(0)
            quality_frame["mel"].append(mel_features.detach().cpu().numpy()) #.tolist())
        else:
            pass

        "Mel_features - Sequence" 
        if vis_args.vis_mel_seq:
            "First bring sequence to a framed format"
            if data_training_args.dataset_name in ["timit", "VOC_ALS", "iemocap"]:
                batch["input_seq_values"] = batch["input_seq_values"][...,:seq_len]
            frames_seq = batch["input_seq_values"].shape[-1]//frame_len
            seq_len = frames_seq*frame_len
            new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames_seq),frame_len),device = batch["input_seq_values"].device)
            for o in range(batch["input_values"].shape[1]):
                sequence = batch["input_seq_values"][:,o,:].clone()
                for f in range(int(frames_seq)):
                    framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                    new_input_seq_values[:,o,f,:] = framed_sequence.clone()
            batch["input_seq_values"] = new_input_seq_values.clone()
            
            mel_seq_features = torch.zeros((batch_size,config.NoC_seq+1,seq_len))
            seq_mel,spec_max_seq = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = None)

            mel_seq_features[:,0,...] = seq_mel.reshape(batch["input_seq_values"].shape[0],-1)

            for o in range(1,batch["input_seq_values"].shape[1]):
                seq_mel,_ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = spec_max_seq)
                mel_seq_features[:,o,...] = seq_mel.reshape(batch["input_seq_values"].shape[0],-1)
            
            mel_seq_features = mel_seq_features.squeeze(0)
            quality_seq["mel"].append(mel_seq_features.detach().cpu().numpy()) #.tolist())

        if not (vis_args.vis_mel_seq or vis_args.vis_td_seq) and (vis_args.vis_mel_frames or vis_args.vis_td_frames):
            if step + 1 >= vis_args.frames_to_vis: 
                break
        else:
            if step + 1 >= vis_args.seq_to_vis: 
                break


    "Perform dimensionality reduction on Time and Mel domain features"
    "Transform data into tensors for easier handling"
    "Real mel data have to be padded to a common length - in training this is handled by the dataloader"
    
    "First labels / variables at sequence and frame level"
    if data_training_args.dataset_name in ["timit", "sim_vowels", "iemocap"]:
        gender_seq = np.array(quality_seq["gender"])
        speaker_id_seq = torch.tensor(np.array(quality_seq["speaker_id"])).squeeze(-1)
        if data_training_args.dataset_name == "iemocap":
            emotion_seq = torch.tensor(np.array(quality_seq["emotion"])).squeeze(-1)
        if data_training_args.dataset_name == "sim_vowels":
            overlap_mask = torch.tensor(np.array(quality_frame["overlap_mask"])).flatten()
            quality_frame["speaker_id"] = torch.tensor(np.array(quality_frame["speaker_id"])).repeat(1,len(quality_frame["vowels"][0]))
            gender = np.repeat(quality_frame["gender"], len(quality_frame["vowels"][0])) #.reshape(len(quality_frame["gender"]),-1)
            speaker_id = torch.cat([x for x in quality_frame["speaker_id"]],dim=0)
            quality_frame["vowels"] = torch.tensor(np.array(quality_frame["vowels"])) 
            vowels = torch.cat([x for x in quality_frame["vowels"]],dim=0)
        elif data_training_args.dataset_name in ["timit", "iemocap"]:
            "For overlap mask"
            max_length = max([len(x) for x in quality_frame["overlap_mask"]])
            new_overlap_mask = torch.ones((len(quality_frame["overlap_mask"]),max_length),dtype=torch.bool)
            for i in range(len(quality_frame["overlap_mask"])):
                new_overlap_mask[i,:len(quality_frame["overlap_mask"][i])] = torch.tensor(quality_frame["overlap_mask"][i])
            quality_frame["overlap_mask"] = new_overlap_mask
            overlap_mask = quality_frame["overlap_mask"].reshape(-1)
            "For speaker id and gender"
            speaker_id = torch.tensor(np.array(quality_frame["speaker_id"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
            "Use the speaker_id for gender and convert to speaker_id str to find the gender later"
            gender = torch.tensor(np.array(quality_frame["speaker_id"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
            
            "For emotion"
            if data_training_args.dataset_name == "iemocap":
                emotion = torch.tensor(np.array(quality_frame["emotion"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
            "For phonemes"
            if data_training_args.dataset_name == "timit":
                max_length = max([len(x) for x in quality_frame["phonemes39"]])
                new_phonemes39 = np.full((len(quality_frame["phonemes39"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["phonemes39"])):
                    new_phonemes39[i,:len(quality_frame["phonemes39"][i])] = quality_frame["phonemes39"][i]
                phonemes39 = new_phonemes39.reshape(-1)
            elif data_training_args.dataset_name == "iemocap":
                max_length = max([len(x) for x in quality_frame["phonemes"]])
                new_phonemes = np.full((len(quality_frame["phonemes"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["phonemes"])):
                    new_phonemes[i,:len(quality_frame["phonemes"][i])] = quality_frame["phonemes"][i]
                phonemes = new_phonemes.reshape(-1)

            if data_training_args.dataset_name == "timit":
                "For consonants"
                max_length = max([len(x) for x in quality_frame["consonants"]])
                new_consonants = np.full((len(quality_frame["consonants"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["consonants"])):
                    new_consonants[i,:len(quality_frame["consonants"][i])] = quality_frame["consonants"][i]
                consonants = new_consonants.reshape(-1)

                "For vowels"
                max_length = max([len(x) for x in quality_frame["vowels"]])
                new_vowels = np.full((len(quality_frame["vowels"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["vowels"])):
                    new_vowels[i,:len(quality_frame["vowels"][i])] = quality_frame["vowels"][i]
                vowels = new_vowels.reshape(-1)

    elif data_training_args.dataset_name == "VOC_ALS":
        phoneme_seq = np.array(quality_seq["phonemes"])
        group_seq = np.array(quality_seq["group"])
        alsfrs_total_seq = np.array(quality_seq["alsfrs_total"])
        alsfrs_speech_seq = np.array(quality_seq["alsfrs_speech"])
        king_stage_seq = np.array(quality_seq["king_stage"])
        cantagallo_seq = np.array(quality_seq["cantagallo"])
        disease_duration_seq = np.array(quality_seq["disease_duration"])
        speaker_id_seq = torch.tensor(np.array(quality_seq["speaker_id"])).squeeze(-1)

        "For overlap mask"
        max_length = max([len(x[0]) for x in quality_frame["overlap_mask"]])
        new_overlap_mask = torch.ones((len(quality_frame["overlap_mask"]),max_length),dtype=torch.bool)
        for i in range(len(quality_frame["overlap_mask"])):
            new_overlap_mask[i,:quality_frame["overlap_mask"][i].shape[-1]] = torch.tensor(quality_frame["overlap_mask"][i])
        quality_frame["overlap_mask"] = new_overlap_mask
        overlap_mask = quality_frame["overlap_mask"].reshape(-1)
        "For all other variables"
        phoneme = torch.tensor(np.array(quality_frame["phonemes"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        group = torch.tensor(np.array(quality_frame["group"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        alsfrs_total = torch.tensor(np.array(quality_frame["alsfrs_total"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        alsfrs_speech = torch.tensor(np.array(quality_frame["alsfrs_speech"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        king_stage = torch.tensor(np.array(quality_frame["king_stage"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)  
        cantagallo = torch.tensor(np.array(quality_frame["cantagallo"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        disease_duration = torch.tensor(np.array(quality_frame["disease_duration"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        speaker_id = torch.tensor(np.array(quality_frame["speaker_id"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
   
    elif data_training_args.dataset_name == "scRNA_seq":
        cell_type_seq = np.array(quality_seq["celltype"])
        condition_seq = np.array(quality_seq["condition"])
        batch_information_seq = np.array(quality_seq["batch"])

        "For overlap mask"
        max_length = max([len(x[0]) for x in quality_frame["overlap_mask"]])
        new_overlap_mask = torch.ones((len(quality_frame["overlap_mask"]),max_length),dtype=torch.bool)
        for i in range(len(quality_frame["overlap_mask"])):
            new_overlap_mask[i,:quality_frame["overlap_mask"][i].shape[-1]] = torch.tensor(quality_frame["overlap_mask"][i])
        quality_frame["overlap_mask"] = new_overlap_mask
        overlap_mask = quality_frame["overlap_mask"].reshape(-1)
        "For all other variables"
        cell_type = torch.tensor(np.array(quality_frame["celltype"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        condition = torch.tensor(np.array(quality_frame["condition"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
        batch_information = torch.tensor(np.array(quality_frame["batch"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)

    "Then sequence data"
    quality_seq["time_domain"] = torch.tensor(np.array(quality_seq["time_domain"]))
    td_seq = quality_seq["time_domain"][:,0,:]
    
    if data_training_args.dataset_name == "sim_vowels":
        td_OCs_seq = quality_seq["time_domain"][:,1:,:]
        td_OCs_concat_seq = td_OCs_seq.reshape(td_OCs_seq.shape[0],-1)
    else:
        td_OCs_seq = quality_seq["time_domain"][:,1:,:].transpose(0,1)
        td_OCs_concat_seq = td_OCs_seq.transpose(0,1).reshape(td_OCs_seq.shape[1],-1)

    "Real mel data have to be padded to a common length - in training this is handled by the dataloader"
    if vis_args.vis_mel_seq:
        if data_training_args.dataset_name == "sim_vowels" or data_training_args.dataset_name == "scRNA_seq":
            quality_seq["mel"] = torch.tensor(np.array(quality_seq["mel"]))
        else:
            max_length = max([x.shape[-1] for x in quality_seq["mel"]])
            new_mel_seq = torch.zeros((len(quality_seq["mel"]),config.NoC_seq+1,max_length))
            for i in range(len(quality_seq["mel"])):
                new_mel_seq[i,:,:quality_seq["mel"][i].shape[-1]] = torch.tensor(quality_seq["mel"][i])
            quality_seq["mel"] = new_mel_seq

        mel_seq = quality_seq["mel"][:,0,:]
        if data_training_args.dataset_name == "sim_vowels" or data_training_args.dataset_name == "scRNA_seq":
            mel_OCs_seq = quality_seq["mel"][:,1:,:]
            mel_OCs_concat_seq = mel_OCs_seq.reshape(mel_OCs_seq.shape[0],-1)
        else:
            mel_OCs_seq = quality_seq["mel"][:,1:,:].transpose(0,1)
            mel_OCs_concat_seq = mel_OCs_seq.transpose(0,1).reshape(mel_OCs_seq.shape[1],-1)
        #mel_OCs_seq = mel_OCs_seq.reshape(mel_OCs_seq.shape[0],-1)
    
    
    "Then frame data"
    if data_training_args.dataset_name == "sim_vowels":
        quality_frame["mel"] = torch.tensor(np.array(quality_frame["mel"]))
        quality_frame["time_domain"] = torch.tensor(np.array(quality_frame["time_domain"]))
    else:
        "For mel domain"
        max_length = max([x.shape[-2] for x in quality_frame["mel"]])
        new_mel_frame = torch.zeros((len(quality_frame["mel"]),config.NoC+1,max_length,frame_len))
        for i in range(len(quality_frame["mel"])):
            new_mel_frame[i,:,:quality_frame["mel"][i].shape[-2],:] = torch.tensor(quality_frame["mel"][i])
        quality_frame["mel"] = new_mel_frame
        "For time domain"
        max_length = max([x.shape[-2] for x in quality_frame["time_domain"]])
        new_td_frame = torch.zeros((len(quality_frame["time_domain"]),config.NoC+1,max_length,frame_len))
        for i in range(len(quality_frame["time_domain"])):
            new_td_frame[i,:,:quality_frame["time_domain"][i].shape[-2],:] = torch.tensor(quality_frame["time_domain"][i])
        quality_frame["time_domain"] = new_td_frame
        "For overlap mask"
        max_length = max([len(x) for x in quality_frame["overlap_mask"]])
        new_overlap_mask = torch.ones((len(quality_frame["overlap_mask"]),max_length),dtype=torch.bool)
        for i in range(len(quality_frame["overlap_mask"])):
            new_overlap_mask[i,:len(quality_frame["overlap_mask"][i])] = torch.tensor(quality_frame["overlap_mask"][i])
        quality_frame["overlap_mask"] = new_overlap_mask
        overlap_mask = quality_frame["overlap_mask"].reshape(-1)
        if data_training_args.dataset_name in ["timit", "iemocap"]:
            "For speaker id and gender"
            speaker_id = torch.tensor(np.array(quality_frame["speaker_id"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
            gender = torch.tensor(np.array(quality_frame["speaker_id"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)
            "For emotion"
            if data_training_args.dataset_name == "iemocap":
                emotion = torch.tensor(np.array(quality_frame["emotion"])).repeat(1,quality_frame["overlap_mask"].shape[-1]).reshape(-1)

            "For phonemes"
            if data_training_args.dataset_name == "timit":
                max_length = max([len(x) for x in quality_frame["phonemes39"]])
                new_phonemes39 = np.full((len(quality_frame["phonemes39"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["phonemes39"])):
                    new_phonemes39[i,:len(quality_frame["phonemes39"][i])] = quality_frame["phonemes39"][i]
                phonemes39 = new_phonemes39.reshape(-1)
            elif data_training_args.dataset_name == "iemocap":
                max_length = max([len(x) for x in quality_frame["phonemes"]])
                new_phonemes = np.full((len(quality_frame["phonemes"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["phonemes"])):
                    new_phonemes[i,:len(quality_frame["phonemes"][i])] = quality_frame["phonemes"][i]
                phonemes = new_phonemes.reshape(-1)
            
            if data_training_args.dataset_name == "timit":
                "For consonants"
                max_length = max([len(x) for x in quality_frame["consonants"]])
                new_consonants = np.full((len(quality_frame["consonants"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["consonants"])):
                    new_consonants[i,:len(quality_frame["consonants"][i])] = quality_frame["consonants"][i]
                consonants = new_consonants.reshape(-1)

                "For vowels"
                max_length = max([len(x) for x in quality_frame["vowels"]])
                new_vowels = np.full((len(quality_frame["vowels"]), max_length), 'NO', dtype=object)
                for i in range(len(quality_frame["vowels"])):
                    new_vowels[i,:len(quality_frame["vowels"][i])] = quality_frame["vowels"][i]
                vowels = new_vowels.reshape(-1)
            
    
    td_frame = quality_frame["time_domain"][:,0,...]
    td_frame = torch.cat([x for x in td_frame],dim=0)
    td_OCs_frame = quality_frame["time_domain"][:,1:,...]
    td_OCs_frame = torch.stack([torch.cat([o for o in td_OCs_frame[:,c,...]],dim=0) for c in range(config.NoC)],dim=0)
    mel_frame = quality_frame["mel"][:,0,...]
    mel_frame = torch.cat([x for x in mel_frame],dim=0)
    mel_OCs_frame = quality_frame["mel"][:,1:,...]
    mel_OCs_frame = torch.stack([torch.cat([o for o in mel_OCs_frame[:,c,...]],dim=0) for c in range(config.NoC)],dim=0)

    "Apply overlap mask to all frame-level data"
    if data_training_args.dataset_name in ["timit", "sim_vowels", "iemocap"]:
        if data_training_args.dataset_name == "timit":
            vowels = vowels[~overlap_mask]
            phonemes39 = phonemes39[~overlap_mask]
            consonants = consonants[~overlap_mask]
        elif data_training_args.dataset_name == "iemocap":
            phonemes = phonemes[~overlap_mask]
            emotion = emotion[~overlap_mask]
        else:
            vowels = torch.masked_select(vowels,~overlap_mask)
        speaker_id = torch.masked_select(speaker_id,~overlap_mask)
        gender = gender[~overlap_mask]
    elif data_training_args.dataset_name == "VOC_ALS":
        group = torch.masked_select(group,~overlap_mask)
        alsfrs_total = torch.masked_select(alsfrs_total,~overlap_mask)
        alsfrs_speech = torch.masked_select(alsfrs_speech,~overlap_mask)
        king_stage = torch.masked_select(king_stage,~overlap_mask)
        cantagallo = torch.masked_select(cantagallo,~overlap_mask)
        disease_duration = torch.masked_select(disease_duration,~overlap_mask)
        speaker_id = torch.masked_select(speaker_id,~overlap_mask)
        phoneme = torch.masked_select(phoneme,~overlap_mask)

    td_frame = torch.masked_select(td_frame,~overlap_mask[:,None]).reshape(-1,td_frame.shape[-1])
    td_OCs_frame = torch.masked_select(td_OCs_frame,~overlap_mask[None,:,None]).reshape(td_OCs_frame.shape[0],-1,td_OCs_frame.shape[-1])
    mel_frame = torch.masked_select(mel_frame,~overlap_mask[:,None]).reshape(-1,mel_frame.shape[-1])
    mel_OCs_frame = torch.masked_select(mel_OCs_frame,~overlap_mask[None,:,None]).reshape(mel_OCs_frame.shape[0],-1,mel_OCs_frame.shape[-1])


    if data_training_args.dataset_name == "sim_vowels":
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "Sim Vowels"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"

        "Frame-level Variable"
        "Time domain"

        if vis_args.vis_td_frames:
            "Try using PCA to see if it gives better visualization"
            n_components = 80  # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            td_frame_reduced = torch.tensor(pca_frame.fit_transform(td_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            td_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(td_OCs_frame[oc]))
                td_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for OC {oc+1} frame PCA: {explained_var:.2f}%")
            td_OCs_frame_reduced = torch.stack(td_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            td_OCs_concat_frame = td_OCs_frame.transpose(0,1).reshape(td_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            td_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(td_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")
            

            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            "TSNE - Time domain - Vowels & Frequency"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = td_frame_reduced,
                OCs = td_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   

                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  vowels,
                    target = "vowel",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  vowels,
                    target = "vowel",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Time domain - Frame-level Speaker"
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
            visualize(data_training_args, 
                config,
                X = td_frame_reduced,
                OCs = td_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
            )
            
            if vis_args.use_umap:
                "UMAP - Time domain - Frame-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }    
                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )
            
        "--------------------------------------------------------------------------------------------------------------"
        data_training_args.frequency_vis = True #Reset
        "--------------------------------------------------------------------------------------------------------------"

        "Mel Filterbank domain - Vowels & Frequency"

        if vis_args.vis_mel_frames:
            n_components = 20  # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            mel_frame_reduced = torch.tensor(pca_frame.fit_transform(mel_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            mel_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(mel_OCs_frame[oc]))
                mel_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel OC {oc+1} frame PCA: {explained_var:.2f}%")
            mel_OCs_frame_reduced = torch.stack(mel_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            mel_OCs_concat_frame = mel_OCs_frame.transpose(0,1).reshape(mel_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            mel_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(mel_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")

            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            "TSNE - Mel domain - Vowels & Frequency"
            "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = mel_frame_reduced,
                OCs = mel_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
            )
            
            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Vowels & Frequency"
                #braycurtis, canberra, euclidean, cosine, correlation
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  vowels,
                    target = "vowel",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  vowels,
                    target = "vowel",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Mel domain - Frame-level Speaker"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = mel_frame_reduced,
                OCs = mel_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Frame-level Speaker"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "Sequence-level Variable"
        "Time domain"
        
        if vis_args.vis_td_seq:
            if vis_args.set_to_use_for_vis == 'test':
                n_components = 200
            else:
                n_components = 400  # Choose number of components to keep

            # PCA for original sequence
            pca_seq = PCA(n_components=n_components, random_state=0)
            td_seq_reduced = torch.tensor(pca_seq.fit_transform(td_seq))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for original sequence PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            td_OCs_seq_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(td_OCs_seq[:,oc,:]))
                td_OCs_seq_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for time domain OC {oc+1} sequence PCA: {explained_var:.2f}%")
            td_OCs_seq_reduced = torch.stack(td_OCs_seq_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            td_OCs_concat_seq = td_OCs_seq.transpose(0,1).reshape(td_OCs_seq.shape[0],-1)
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            td_OCs_concat_seq_reduced = torch.tensor(pca_OCs_concat_seq.fit_transform(td_OCs_concat_seq))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat sequence PCA: {explained_var_OCs:.2f}%")

            "--------------------------------------------------------------------------------------------------------------"
            "TSNE - Time domain - Sequence-level Speaker"
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
            visualize(data_training_args, 
                config,
                X = td_seq_reduced,
                OCs = td_OCs_seq_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_seq_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
            )

            if vis_args.use_umap:
                "UMAP - Time domain - Sequence-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                }   
                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                )
        
        "--------------------------------------------------------------------------------------------------------------"
        "Mel domain - Sequence-level Speaker"

        if vis_args.vis_mel_seq:
            n_components = 10  # Choose number of components to keep

            # PCA for original sequence
            pca_seq = PCA(n_components=n_components, random_state=0)
            mel_seq_reduced = torch.tensor(pca_seq.fit_transform(mel_seq))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel original sequence PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            mel_OCs_seq_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(mel_OCs_seq[:,oc,:]))
                mel_OCs_seq_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel domain OC {oc+1} sequence PCA: {explained_var:.2f}%")
            mel_OCs_seq_reduced = torch.stack(mel_OCs_seq_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            mel_OCs_concat_seq = mel_OCs_seq.transpose(0,1).reshape(mel_OCs_seq.shape[0],-1)
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            mel_OCs_concat_seq_reduced = torch.tensor(pca_OCs_concat_seq.fit_transform(mel_OCs_concat_seq))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat sequence PCA: {explained_var_OCs:.2f}%")

            "--------------------------------------------------------------------------------------------------------------"
            "TSNE - Mel domain - Sequence-level Speaker"
            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=50, metric='braycurtis',early_exaggeration=10,
                            init='pca'),
            }   
            visualize(data_training_args, 
                config,
                X = mel_seq_reduced,
                OCs = mel_OCs_seq_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_seq_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
            )

            if vis_args.use_umap:
                "UMAP - Mel domain - Sequence-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=50,min_dist=0.9,densmap=False)        
                }   
                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                )

    elif data_training_args.dataset_name == "timit":
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "TIMIT"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
 
        "Select 10-20 speakers to visualize"
        rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
        all_speakers = np.unique(speaker_id)
        all_speakers_seq = np.unique(speaker_id_seq)
        sel_10_speakers_list = rng.choice(all_speakers, size=10, replace=False)
        sel_10_sp_mask = np.isin(speaker_id, sel_10_speakers_list)
        sel_10_speakers_seq_list = rng.choice(all_speakers_seq, size=10, replace=False)
        sel_10_sp_seq_mask = np.isin(speaker_id_seq, sel_10_speakers_seq_list)
        sel_10_speakers = speaker_id[sel_10_sp_mask]
        sel_10_speakers_seq = speaker_id_seq[sel_10_sp_seq_mask]
        if len(all_speakers) >= 20:
            sel_20_speakers_list = rng.choice(all_speakers, size=20, replace=False)
            sel_20_sp_mask = np.isin(speaker_id, sel_20_speakers_list)
            sel_20_speakers = speaker_id[sel_20_sp_mask]
        if len(all_speakers_seq) >= 20:
            sel_20_speakers_seq_list = rng.choice(all_speakers_seq, size=20, replace=False)
            sel_20_sp_seq_mask = np.isin(speaker_id_seq, sel_20_speakers_seq_list)
            sel_20_speakers_seq = speaker_id_seq[sel_20_sp_seq_mask]

        "Select specific vowels and consonants to be visualized and remove the NO flags"
        all_vowels_list = np.unique(vowels)[1:]
        all_consonants_list = np.unique(consonants)[1:]
        all_vowel_mask = np.isin(vowels, all_vowels_list)
        all_consonant_mask = np.isin(consonants, all_consonants_list)
        vowel_mask = np.isin(vowels, vis_args.sel_vowels_list_timit)
        sel_vowels = vowels[vowel_mask]
        consonant_mask = np.isin(consonants, vis_args.sel_consonants_list_timit)
        sel_consonants = consonants[consonant_mask]
        phoneme_mask = np.isin(phonemes39, vis_args.sel_phonemes_list_timit)
        sel_phonemes = phonemes39[phoneme_mask]

        if vis_args.vis_td_frames:
            "Frame-level Variable"
            "Time domain"
            "Try using PCA to see if it gives better visualization"
            n_components = 100  # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            td_frame_reduced = torch.tensor(pca_frame.fit_transform(td_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            td_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(td_OCs_frame[oc]))
                td_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for OC {oc+1} frame PCA: {explained_var:.2f}%")
            td_OCs_frame_reduced = torch.stack(td_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            td_OCs_concat_frame = td_OCs_frame.transpose(0,1).reshape(td_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            td_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(td_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")

            "Select subsets after PCA"
            "Time domain - X"
            sel_td_frame_vowels = td_frame_reduced[vowel_mask]
            sel_td_frame_consonants = td_frame_reduced[consonant_mask]
            sel_td_frame_phonemes = td_frame_reduced[phoneme_mask]
            all_td_frame_vowels = td_frame_reduced[all_vowel_mask]
            all_td_frame_consonants = td_frame_reduced[all_consonant_mask]
            "Time domain - OCs"
            sel_td_OCs_frame_vowels = td_OCs_frame_reduced[:,vowel_mask,:]
            sel_td_OCs_frame_consonants = td_OCs_frame_reduced[:,consonant_mask,:]
            sel_td_OCs_frame_phonemes = td_OCs_frame_reduced[:,phoneme_mask,:]
            all_td_OCs_frame_vowels = td_OCs_frame_reduced[:,all_vowel_mask,:]
            all_td_OCs_frame_consonants = td_OCs_frame_reduced[:,all_consonant_mask,:]
            "Time domain - OCs concatenated"
            sel_td_OCs_concat_frame_vowels = td_OCs_concat_frame_reduced[vowel_mask]
            sel_td_OCs_concat_frame_consonants = td_OCs_concat_frame_reduced[consonant_mask]
            sel_td_OCs_concat_frame_phonemes = td_OCs_concat_frame_reduced[phoneme_mask]
            all_td_OCs_concat_frame_vowels = td_OCs_concat_frame_reduced[all_vowel_mask]
            all_td_OCs_concat_frame_consonants = td_OCs_concat_frame_reduced[all_consonant_mask]       

            "Frame Speakers - Time Domain"
            sel_td_frame_10_speakers = td_frame_reduced[sel_10_sp_mask]
            sel_td_OCs_frame_10_speakers = td_OCs_frame_reduced[:,sel_10_sp_mask,:]
            sel_td_OCs_concat_frame_10_speakers = td_OCs_concat_frame_reduced[sel_10_sp_mask] 
            if len(all_speakers) >= 20:
                sel_td_frame_20_speakers = td_frame_reduced[sel_20_sp_mask]
                sel_td_OCs_frame_20_speakers = td_OCs_frame_reduced[:,sel_20_sp_mask,:]
                sel_td_OCs_concat_frame_20_speakers = td_OCs_concat_frame_reduced[sel_20_sp_mask]

            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            "Frequency and all Phonemes"
            
            "TSNE - Time domain - Vowels & Frequency"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = td_frame_reduced,
                OCs = td_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  phonemes39,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
            )
           
            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  phonemes39,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   
                
                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  phonemes39,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  phonemes39,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "Selected Vowels"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            "TSNE - Time domain - Vowels & Frequency"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = sel_td_frame_vowels,
                OCs = sel_td_OCs_frame_vowels,
                z_or_h = 'z',
                y_vec =  sel_vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_td_OCs_concat_frame_vowels,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   
                
                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_vowels,
                    OCs = sel_td_OCs_frame_vowels,
                    z_or_h = 'z',
                    y_vec =  sel_vowels,
                    target = "vowel",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_vowels,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_vowels,
                    target = "vowel",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'vowels')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "Selected Consonants"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            "TSNE - Time domain - Vowels & Frequency"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = sel_td_frame_consonants,
                OCs = sel_td_OCs_frame_consonants,
                z_or_h = 'z',
                y_vec =  sel_consonants,
                target = "consonant",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'consonants')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_td_OCs_concat_frame_consonants,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_consonants,
                target = "consonant",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'consonants')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   

                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_consonants,
                    OCs = sel_td_OCs_frame_consonants,
                    z_or_h = 'z',
                    y_vec =  sel_consonants,
                    target = "consonant",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'consonants')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_consonants,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_consonants,
                    target = "consonant",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'consonants')
                )

            "--------------------------------------------------------------------------------------------------------------"
            
            "Selected Vowels & Consonants"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            "TSNE - Time domain - Vowels & Frequency"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = sel_td_frame_phonemes,
                OCs = sel_td_OCs_frame_phonemes,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'selected_phonemes')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_td_OCs_concat_frame_phonemes,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'selected_phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   

                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_phonemes,
                    OCs = sel_td_OCs_frame_phonemes,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'selected_phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_phonemes,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'selected_phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Time domain - Frame-level Speaker"
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
            "10 random speakers"
            visualize(data_training_args, 
                config,
                X = sel_td_frame_10_speakers,
                OCs = sel_td_OCs_frame_10_speakers,
                z_or_h = 'z',
                y_vec =  sel_10_speakers,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_td_OCs_concat_frame_10_speakers,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_10_speakers,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
            )
            
            if vis_args.use_umap:
                "UMAP - Time domain - Frame-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }    
                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_10_speakers,
                    OCs = sel_td_OCs_frame_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )

            "20 random speakers"
            if len(all_speakers) >= 20:
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                init='pca'),
                }  
                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_20_speakers,
                    OCs = sel_td_OCs_frame_20_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_20_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level Speaker"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = sel_td_frame_20_speakers,
                        OCs = sel_td_OCs_frame_20_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_td_OCs_concat_frame_20_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                    )
                
        "--------------------------------------------------------------------------------------------------------------"
        data_training_args.frequency_vis = True #Reset
        "--------------------------------------------------------------------------------------------------------------"

        "Mel Filterbank domain - Vowels & Frequency"
        if vis_args.vis_mel_frames:
            n_components = 25 # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            mel_frame_reduced = torch.tensor(pca_frame.fit_transform(mel_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            mel_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(mel_OCs_frame[oc]))
                mel_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel OC {oc+1} frame PCA: {explained_var:.2f}%")
            mel_OCs_frame_reduced = torch.stack(mel_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            mel_OCs_concat_frame = mel_OCs_frame.transpose(0,1).reshape(mel_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            mel_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(mel_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")

            "Select subsets after PCA"
            "Mel Domain - X"
            sel_mel_frame_vowels = mel_frame_reduced[vowel_mask]
            sel_mel_frame_consonants = mel_frame_reduced[consonant_mask]
            sel_mel_frame_phonemes = mel_frame_reduced[phoneme_mask]
            all_mel_frame_vowels = mel_frame_reduced[all_vowel_mask]
            all_mel_frame_consonants = mel_frame_reduced[all_consonant_mask]
            "Mel Domain - OCs"
            sel_mel_OCs_frame_vowels = mel_OCs_frame_reduced[:,vowel_mask,:]
            sel_mel_OCs_frame_consonants = mel_OCs_frame_reduced[:,consonant_mask,:]
            sel_mel_OCs_frame_phonemes = mel_OCs_frame_reduced[:,phoneme_mask,:]
            all_mel_OCs_frame_vowels = mel_OCs_frame_reduced[:,all_vowel_mask,:]
            all_mel_OCs_frame_consonants = mel_OCs_frame_reduced[:,all_consonant_mask,:]
            "Mel Domain - OCs concatenated"
            sel_mel_OCs_concat_frame_vowels = mel_OCs_concat_frame_reduced[vowel_mask]
            sel_mel_OCs_concat_frame_consonants = mel_OCs_concat_frame_reduced[consonant_mask]
            sel_mel_OCs_concat_frame_phonemes = mel_OCs_concat_frame_reduced[phoneme_mask]
            all_mel_OCs_concat_frame_vowels = mel_OCs_concat_frame_reduced[all_vowel_mask]
            all_mel_OCs_concat_frame_consonants = mel_OCs_concat_frame_reduced[all_consonant_mask]

            "Speakers - Mel Domain"
            sel_mel_frame_10_speakers = mel_frame_reduced[sel_10_sp_mask]
            sel_mel_OCs_frame_10_speakers = mel_OCs_frame_reduced[:,sel_10_sp_mask,:]
            sel_mel_OCs_concat_frame_10_speakers = mel_OCs_concat_frame_reduced[sel_10_sp_mask] 
            if len(all_speakers) >= 20:
                sel_mel_frame_20_speakers = mel_frame_reduced[sel_20_sp_mask]
                sel_mel_OCs_frame_20_speakers = mel_OCs_frame_reduced[:,sel_20_sp_mask,:]
                sel_mel_OCs_concat_frame_20_speakers = mel_OCs_concat_frame_reduced[sel_20_sp_mask]


            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            "TSNE - Mel domain - Frequency and all Phonemes"
            "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = mel_frame_reduced,
                OCs = mel_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  phonemes39,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  phonemes39,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Frequency & Phonemes"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  phonemes39,
                    target = "phoneme",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  phonemes39,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Mel domain - Selected Vowels"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = sel_mel_frame_vowels,
                OCs = sel_mel_OCs_frame_vowels,
                z_or_h = 'z',
                y_vec =  sel_vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_mel_OCs_concat_frame_vowels,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_vowels,
                target = "vowel",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_vowels,
                    OCs = sel_mel_OCs_frame_vowels,
                    z_or_h = 'z',
                    y_vec =  sel_vowels,
                    target = "vowel",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_vowels,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_vowels,
                    target = "vowel",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'vowels')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Mel domain - Selected Consonants"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = sel_mel_frame_consonants,
                OCs = sel_mel_OCs_frame_consonants,
                z_or_h = 'z',
                y_vec =  sel_consonants,
                target = "consonant",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'consonants')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_mel_OCs_concat_frame_consonants,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_consonants,
                target = "consonant",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'consonants')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_consonants,
                    OCs = sel_mel_OCs_frame_consonants,
                    z_or_h = 'z',
                    y_vec =  sel_consonants,
                    target = "consonant",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'consonants')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_consonants,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_consonants,
                    target = "consonant",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'consonants')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Mel domain - Selected Vowels & Consonants"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = sel_mel_frame_phonemes,
                OCs = sel_mel_OCs_frame_phonemes,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'selected_phonemes')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_mel_OCs_concat_frame_phonemes,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'selected_phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_phonemes,
                    OCs = sel_mel_OCs_frame_phonemes,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'selected_phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_phonemes,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'selected_phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"


            "TSNE - Mel domain - Frame-level Speaker"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 

            "10 random speakers"
            visualize(data_training_args, 
                config,
                X = sel_mel_frame_10_speakers,
                OCs = sel_mel_OCs_frame_10_speakers,
                z_or_h = 'z',
                y_vec =  sel_10_speakers,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_mel_OCs_concat_frame_10_speakers,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_10_speakers,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Frame-level Speaker"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_10_speakers,
                    OCs = sel_mel_OCs_frame_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

            "20 random speakers"
            if len(all_speakers) >= 20:
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_20_speakers,
                    OCs = sel_mel_OCs_frame_20_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames_20_speakers',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_20_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames_20_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level Speaker"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_frame_20_speakers,
                        OCs = sel_mel_OCs_frame_20_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers,
                        target = "speaker_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames_20_speakers',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_OCs_concat_frame_20_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames_20_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                    )


        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "Sequence-level Variable"
        "Time domain"

        if vis_args.vis_td_seq:

            "Sequence Selected Speakers - Time Domain - First select then reduce"
            sel_td_seq_10_speakers = td_seq[sel_10_sp_seq_mask]
            sel_td_OCs_seq_10_speakers = td_OCs_seq[:,sel_10_sp_seq_mask,:].transpose(0,1)
            sel_td_OCs_concat_seq_10_speakers = td_OCs_concat_seq[sel_10_sp_seq_mask]
            if len(all_speakers_seq) >= 20:
                sel_td_seq_20_speakers = td_seq[sel_20_sp_seq_mask]
                sel_td_OCs_seq_20_speakers = td_OCs_seq[:,sel_20_sp_seq_mask,:].transpose(0,1)
                sel_td_OCs_concat_seq_20_speakers = td_OCs_concat_seq[sel_20_sp_seq_mask]

            n_components = 75  # Choose number of components to keep

            "Reduce every sequence separately - for 10 and 20 speakers"
            "PCA 10 speakers"
            pca_seq = PCA(n_components=n_components, random_state=0)
            sel_td_seq_10_speakers = torch.tensor(pca_seq.fit_transform(sel_td_seq_10_speakers))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for original sequence 10 speakers PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            sel_td_OCs_seq_10_speakers_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(sel_td_OCs_seq_10_speakers[:,oc,:]))
                sel_td_OCs_seq_10_speakers_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for time domain OC {oc+1} sequence 10 speakers PCA: {explained_var:.2f}%")
            sel_td_OCs_seq_10_speakers = torch.stack(sel_td_OCs_seq_10_speakers_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            sel_td_OCs_concat_seq_10_speakers = torch.tensor(pca_OCs_concat_seq.fit_transform(sel_td_OCs_concat_seq_10_speakers))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat sequence 10 speakers PCA: {explained_var_OCs:.2f}%")

            "PCA 20 speakers"
            n_components = 150

            pca_seq = PCA(n_components=n_components, random_state=0)
            sel_td_seq_20_speakers = torch.tensor(pca_seq.fit_transform(sel_td_seq_20_speakers))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for original sequence 20 speakers PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            sel_td_OCs_seq_20_speakers_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(sel_td_OCs_seq_20_speakers[:,oc,:]))
                sel_td_OCs_seq_20_speakers_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for time domain OC {oc+1} sequence 20 speakers PCA: {explained_var:.2f}%")
            sel_td_OCs_seq_20_speakers = torch.stack(sel_td_OCs_seq_20_speakers_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            sel_td_OCs_concat_seq_20_speakers = torch.tensor(pca_OCs_concat_seq.fit_transform(sel_td_OCs_concat_seq_20_speakers))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat sequence 20 speakers PCA: {explained_var_OCs:.2f}%")

            "--------------------------------------------------------------------------------------------------------------"
            "TSNE - Time domain - Sequence-level Speaker"
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

            "10 random speakers"
            visualize(data_training_args, 
                config,
                X = sel_td_seq_10_speakers,
                OCs = sel_td_OCs_seq_10_speakers,
                z_or_h = 'z',
                y_vec =  sel_10_speakers_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_td_OCs_concat_seq_10_speakers,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_10_speakers_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
            )

            if vis_args.use_umap:
                "UMAP - Time domain - Sequence-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                }   
                visualize(data_training_args, 
                    config,
                    X = sel_td_seq_10_speakers,
                    OCs = sel_td_OCs_seq_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_seq_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                )
            
            "20 random speakers"
            if len(all_speakers_seq) >= 20:
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

                visualize(data_training_args, 
                    config,
                    X = sel_td_seq_20_speakers,
                    OCs = sel_td_OCs_seq_20_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs_20_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_seq_20_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_20_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level Speaker"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = sel_td_seq_20_speakers,
                        OCs = sel_td_OCs_seq_20_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs_20_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_td_OCs_concat_seq_20_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_20_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                    )

        "--------------------------------------------------------------------------------------------------------------"
        "Mel domain - Sequence-level Speaker"

        if vis_args.vis_mel_seq:
            "Sequence Selected Speakers - Mel Domain"
            sel_mel_seq_10_speakers = mel_seq[sel_10_sp_seq_mask]
            sel_mel_OCs_seq_10_speakers = mel_OCs_seq[:,sel_10_sp_seq_mask,:].transpose(0,1)
            sel_mel_OCs_concat_seq_10_speakers = mel_OCs_concat_seq[sel_10_sp_seq_mask]
            if len(all_speakers_seq) >= 20:
                sel_mel_seq_20_speakers = mel_seq[sel_20_sp_seq_mask]
                sel_mel_OCs_seq_20_speakers = mel_OCs_seq[:,sel_20_sp_seq_mask,:].transpose(0,1)
                sel_mel_OCs_concat_seq_20_speakers = mel_OCs_concat_seq[sel_20_sp_seq_mask]

            "Reduce every sequence separately - for 10 and 20 speakers"
            "PCA 10 speakers"
            n_components = 20  # Choose number of components to keep

            # PCA for original sequence
            pca_seq = PCA(n_components=n_components, random_state=0)
            sel_mel_seq_10_speakers = torch.tensor(pca_seq.fit_transform(sel_mel_seq_10_speakers))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel original sequence 10 speakers PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            sel_mel_OCs_seq_10_speakers_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(sel_mel_OCs_seq_10_speakers[:,oc,:]))
                sel_mel_OCs_seq_10_speakers_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel domain OC {oc+1} sequence 10 speakers PCA: {explained_var:.2f}%")
            sel_mel_OCs_seq_10_speakers = torch.stack(sel_mel_OCs_seq_10_speakers_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            sel_mel_OCs_concat_seq_10_speakers = torch.tensor(pca_OCs_concat_seq.fit_transform(sel_mel_OCs_concat_seq_10_speakers))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat sequence 10 speakers PCA: {explained_var_OCs:.2f}%")

            "PCA 20 speakers"
            n_components = 25  # Choose number of components to keep

            # PCA for original sequence
            pca_seq = PCA(n_components=n_components, random_state=0)
            sel_mel_seq_20_speakers = torch.tensor(pca_seq.fit_transform(sel_mel_seq_20_speakers))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel original sequence 20 speakers PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            sel_mel_OCs_seq_20_speakers_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(sel_mel_OCs_seq_20_speakers[:,oc,:]))
                sel_mel_OCs_seq_20_speakers_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel domain OC {oc+1} sequence 20 speakers PCA: {explained_var:.2f}%")
            sel_mel_OCs_seq_20_speakers = torch.stack(sel_mel_OCs_seq_20_speakers_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            sel_mel_OCs_concat_seq_20_speakers = torch.tensor(pca_OCs_concat_seq.fit_transform(sel_mel_OCs_concat_seq_20_speakers))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat sequence 20 speakers PCA: {explained_var_OCs:.2f}%")

            "--------------------------------------------------------------------------------------------------------------"
            "TSNE - Mel domain - Sequence-level Speaker"
            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=5, metric='cityblock',early_exaggeration=10,
                            init='pca'),
            }

            "10 random speakers"
            visualize(data_training_args, 
                config,
                X = sel_mel_seq_10_speakers,
                OCs = sel_mel_OCs_seq_10_speakers,
                z_or_h = 'z',
                y_vec =  sel_10_speakers_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_mel_OCs_concat_seq_10_speakers,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_10_speakers_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
            )

            if vis_args.use_umap:
                "UMAP - Mel domain - Sequence-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                } 
                visualize(data_training_args, 
                    config,
                    X = sel_mel_seq_10_speakers,
                    OCs = sel_mel_OCs_seq_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) +'_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_seq_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                )

            "20 random speakers"
            if len(all_speakers_seq) >= 20:
                data_training_args.frequency_vis = True
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=5, metric='cityblock',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = sel_mel_seq_20_speakers,
                    OCs = sel_mel_OCs_seq_20_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) + '_seqs_20_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_seq_20_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_20_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_20_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level Speaker"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'wminkowski',n_neighbors=5,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_seq_20_speakers,
                        OCs = sel_mel_OCs_seq_20_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) +'_seqs_20_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_OCs_concat_seq_20_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_20_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_20_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name)
                    )

    elif data_training_args.dataset_name == "iemocap":
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "IEMOCAP"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"

        "Select phonemes to be visualized and remove the NO flags"
        phoneme_mask = np.isin(phonemes, vis_args.sel_phonemes_list_iemocap)
        sel_phonemes = phonemes[phoneme_mask]
        non_verbal_mask = np.isin(phonemes, vis_args.sel_non_verbal_phonemes_iemocap)
        sel_non_verbal_phonemes = phonemes[non_verbal_mask]

        if vis_args.vis_td_frames:
            "Frame-level Variable"
            "Time domain"
            "Try using PCA to see if it gives better visualization"
            n_components = 100  # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            td_frame_reduced = torch.tensor(pca_frame.fit_transform(td_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            td_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(td_OCs_frame[oc]))
                td_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for OC {oc+1} frame PCA: {explained_var:.2f}%")
            td_OCs_frame_reduced = torch.stack(td_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            td_OCs_concat_frame = td_OCs_frame.transpose(0,1).reshape(td_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            td_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(td_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")

            "Select subsets after PCA"
            "Time domain - X"
            sel_td_frame_phonemes = td_frame_reduced[phoneme_mask]
            sel_td_frame_non_verbal_phonemes = td_frame_reduced[non_verbal_mask]
            "Time domain - OCs"
            sel_td_OCs_frame_phonemes = td_OCs_frame_reduced[:,phoneme_mask,:]
            sel_td_OCs_frame_non_verbal_phonemes = td_OCs_frame_reduced[:,non_verbal_mask,:]

            "Time domain - OCs concatenated"
            sel_td_OCs_concat_frame_phonemes = td_OCs_concat_frame_reduced[phoneme_mask]
            sel_td_OCs_concat_frame_non_verbal_phonemes = td_OCs_concat_frame_reduced[non_verbal_mask]

            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            "Frequency and selected Phonemes"
            
            "TSNE - Time domain - Phonemes & Frequency"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = sel_td_frame_phonemes,
                OCs = sel_td_OCs_frame_phonemes,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_td_OCs_concat_frame_phonemes,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Phonemes & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   

                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_phonemes,
                    OCs = sel_td_OCs_frame_phonemes,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_phonemes,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "Selected Non-verbal Phonemes"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            "TSNE - Time domain - Non-verbal Phonemes"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = sel_td_frame_non_verbal_phonemes,
                OCs = sel_td_OCs_frame_non_verbal_phonemes,
                z_or_h = 'z',
                y_vec =  sel_non_verbal_phonemes,
                target = "non_verbal_phoneme",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'non_verbal_phonemes')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_td_OCs_concat_frame_non_verbal_phonemes,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_non_verbal_phonemes,
                target = "non_verbal_phoneme",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'non_verbal_phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Non-verbal Phonemes"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   
                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_non_verbal_phonemes,
                    OCs = sel_td_OCs_frame_non_verbal_phonemes,
                    z_or_h = 'z',
                    y_vec =  sel_non_verbal_phonemes,
                    target = "non_verbal_phoneme",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'non_verbal_phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_non_verbal_phonemes,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_non_verbal_phonemes,
                    target = "non_verbal_phoneme",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'non_verbal_phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "Emotions"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            "TSNE - Time domain - Emotions"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                            init='pca'),
            }   
            
            visualize(data_training_args, 
                config,
                X = td_frame_reduced,
                OCs = td_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  emotion,
                target = "emotion",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'categorical_emotions')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  emotion,
                target = "emotion",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'categorical_emotions')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Time domain - Emotions"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }   
    
                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  emotion,
                    target = "emotion",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'categorical_emotions')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  emotion,
                    target = "emotion",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'categorical_emotions')
                )

            "--------------------------------------------------------------------------------------------------------------"
            "Speakers"

            "TSNE - Time domain - Frame-level Speaker"
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
            "10 random speakers"
            visualize(data_training_args, 
                config,
                X = td_frame_reduced,
                OCs = td_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
            )
            
            if vis_args.use_umap:
                "UMAP - Time domain - Frame-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                }    
                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )
                
        "--------------------------------------------------------------------------------------------------------------"
        data_training_args.frequency_vis = True #Reset

        "--------------------------------------------------------------------------------------------------------------"

        "Mel Filterbank domain - Vowels & Frequency"
        if vis_args.vis_mel_frames:
            n_components = 25 # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            mel_frame_reduced = torch.tensor(pca_frame.fit_transform(mel_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            mel_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(mel_OCs_frame[oc]))
                mel_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel OC {oc+1} frame PCA: {explained_var:.2f}%")
            mel_OCs_frame_reduced = torch.stack(mel_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            mel_OCs_concat_frame = mel_OCs_frame.transpose(0,1).reshape(mel_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            mel_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(mel_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")

            "Select subsets after PCA"
            "Mel Domain - X"
            sel_mel_frame_phonemes = mel_frame_reduced[phoneme_mask]
            sel_mel_frame_non_verbal_phonemes = mel_frame_reduced[non_verbal_mask]
            
            "Mel Domain - OCs"
            sel_mel_OCs_frame_phonemes = mel_OCs_frame_reduced[:,phoneme_mask,:]
            sel_mel_OCs_frame_non_verbal_phonemes = mel_OCs_frame_reduced[:,non_verbal_mask,:]
            
            "Mel Domain - OCs concatenated"
            sel_mel_OCs_concat_frame_phonemes = mel_OCs_concat_frame_reduced[phoneme_mask]
            sel_mel_OCs_concat_frame_non_verbal_phonemes = mel_OCs_concat_frame_reduced[non_verbal_mask]

            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            "TSNE - Mel domain - Frequency and Selected Phonemes"
            "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = sel_mel_frame_phonemes,
                OCs = sel_mel_OCs_frame_phonemes,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_mel_OCs_concat_frame_phonemes,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_phonemes,
                target = "phoneme",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Frequency & Selected Phonemes"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_phonemes,
                    OCs = sel_mel_OCs_frame_phonemes,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_phonemes,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_phonemes,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Mel domain - Non-verbal Phonemes"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = sel_mel_frame_non_verbal_phonemes,
                OCs = sel_mel_OCs_frame_non_verbal_phonemes,
                z_or_h = 'z',
                y_vec =  sel_non_verbal_phonemes,
                target = "non_verbal_phoneme",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'non_verbal_phonemes')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = sel_mel_OCs_concat_frame_non_verbal_phonemes,
                OCs = None,
                z_or_h = 'z',
                y_vec =  sel_non_verbal_phonemes,
                target = "non_verbal_phoneme",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'non_verbal_phonemes')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_non_verbal_phonemes,
                    OCs = sel_mel_OCs_frame_non_verbal_phonemes,
                    z_or_h = 'z',
                    y_vec =  sel_non_verbal_phonemes,
                    target = "non_verbal_phoneme",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'non_verbal_phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_non_verbal_phonemes,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_non_verbal_phonemes,
                    target = "non_verbal_phoneme",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'non_verbal_phonemes')
                )

            "--------------------------------------------------------------------------------------------------------------"

            "TSNE - Mel domain - Emotions"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = mel_frame_reduced,
                OCs = mel_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  emotion,
                target = "emotion",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'categorical_emotions')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  emotion,
                target = "emotion",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'categorical_emotions')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Vowels & Frequency"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  emotion,
                    target = "emotion",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'categorical_emotions')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  emotion,
                    target = "emotion",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'categorical_emotions')
                )

            "--------------------------------------------------------------------------------------------------------------"
            "Speakers"

            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                            init='pca'),
            } 

            "TSNE - Mel domain - Frame-level Speaker"
            visualize(data_training_args, 
                config,
                X = mel_frame_reduced,
                OCs = mel_OCs_frame_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict= manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_frame_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id,
                target = "speaker_frame",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
            )

            if vis_args.use_umap:
                data_training_args.vis_method = 'umap'
                "UMAP - Mel domain - Frame-level Speaker"
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                }
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )


        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "Sequence-level Variable"
        "Time domain"

        if vis_args.vis_td_seq:
            n_components = 75 

            "Reduce sequence separately"
            "PCA 10 speakers"
            pca_seq = PCA(n_components=n_components, random_state=0)
            td_seq_reduced = torch.tensor(pca_seq.fit_transform(td_seq))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for original sequence PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            td_OCs_seq_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(td_OCs_seq[oc,:,:]))
                td_OCs_seq_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for time domain OC {oc+1} sequence PCA: {explained_var:.2f}%")
            td_OCs_seq_reduced = torch.stack(td_OCs_seq_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            td_OCs_concat_seq_reduced = torch.tensor(pca_OCs_concat_seq.fit_transform(td_OCs_concat_seq))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat sequence PCA: {explained_var_OCs:.2f}%")

            
            "--------------------------------------------------------------------------------------------------------------"
            "Sequence-level Speaker"

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

            "TSNE - Time domain - Sequence-level Speaker"
            visualize(data_training_args, 
                config,
                X = td_seq_reduced,
                OCs = td_OCs_seq_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'speakers_seq')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_seq_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'speakers_seq')
            )

            if vis_args.use_umap:
                "UMAP - Time domain - Sequence-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                }   
                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'speakers_seq')
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'speakers_seq')
                )
            
            "--------------------------------------------------------------------------------------------------------------"
            "Sequence-level Emotions"

            "TSNE - Time domain - Sequence-level Emotions"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=15, metric='canberra',early_exaggeration=10,
                            init='pca'),
            } 

            visualize(data_training_args, 
                config,
                X = td_seq_reduced,
                OCs = td_OCs_seq_reduced,
                z_or_h = 'z',
                y_vec =  emotion_seq,
                target = "emotion_seq",
                data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'emotion_seq')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = td_OCs_concat_seq_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  emotion_seq,
                target = "emotion_seq",
                data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'emotion_seq')
            )

            if vis_args.use_umap:
                "UMAP - Time domain - Sequence-level Emotions"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                }   
                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  emotion_seq,
                    target = "emotion_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'emotion_seq')
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  emotion_seq,
                    target = "emotion_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name, 'emotion_seq')
                )

        "--------------------------------------------------------------------------------------------------------------"
        "Mel domain - Sequence-level Speaker"

        if vis_args.vis_mel_seq:
            "Reduce sequence"
            
            n_components = 20  # Choose number of components to keep

            # PCA for original sequence
            pca_seq = PCA(n_components=n_components, random_state=0)
            mel_seq_reduced = torch.tensor(pca_seq.fit_transform(mel_seq))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel original sequence 10 speakers PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            mel_OCs_seq_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(mel_OCs_seq[oc,:,:]))
                mel_OCs_seq_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel domain OC {oc+1} sequence 10 speakers PCA: {explained_var:.2f}%")
            mel_OCs_seq_reduced = torch.stack(mel_OCs_seq_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            mel_OCs_concat_seq_reduced = torch.tensor(pca_OCs_concat_seq.fit_transform(mel_OCs_concat_seq))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat sequence 10 speakers PCA: {explained_var_OCs:.2f}%")

            "--------------------------------------------------------------------------------------------------------------"
            "Speakers"
            
            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=5, metric='cityblock',early_exaggeration=10,
                            init='pca'),
            } 

            "TSNE - Mel domain - Sequence-level Speaker"
            visualize(data_training_args, 
                config,
                X = mel_seq_reduced,
                OCs = mel_OCs_seq_reduced,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'speakers_seq')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_seq_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  speaker_id_seq,
                target = "speaker_seq",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'speakers_seq')
            )

            if vis_args.use_umap:
                "UMAP - Mel domain - Sequence-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) +'_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'speakers_seq')
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  speaker_id_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'speakers_seq')
                )

            "--------------------------------------------------------------------------------------------------------------"
            "Emotions"
            
            "TSNE - Mel domain - Sequence-level Emotions"
            
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            manifold_dict = {
                'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                            max_iter = 1000, perplexity=5, metric='cityblock',early_exaggeration=10,
                            init='pca'),
            } 
            visualize(data_training_args, 
                config,
                X = mel_seq_reduced,
                OCs = mel_OCs_seq_reduced,
                z_or_h = 'z',
                y_vec =  emotion_seq,
                target = "emotion_seq",
                data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'emotion_seq')
            )

            "Also plot for the concatenated OCs"
            visualize(data_training_args, 
                config,
                X = mel_OCs_concat_seq_reduced,
                OCs = None,
                z_or_h = 'z',
                y_vec =  emotion_seq,
                target = "emotion_seq",
                data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                manifold_dict = manifold_dict,
                return_data = True,
                display_figures = True,
                save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'emotion_seq')
            )

            if vis_args.use_umap:
                "UMAP - Mel domain - Sequence-level Speaker"
                data_training_args.vis_method = 'umap'
                manifold_dict = {
                    'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'wminkowski',n_neighbors=5,min_dist=0.9,densmap=False)        
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  emotion_seq,
                    target = "emotion_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) +'_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'emotion_seq')
                )
                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  emotion_seq,
                    target = "emotion_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name, 'emotion_seq')
                )

    elif data_training_args.dataset_name == "VOC_ALS":
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "VOC-ALS"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        
        rng = np.random.default_rng(seed=vis_args.random_seed_vis) 
        "VOC-ALS has 153 different speakers - Select 10 to visualize"       
        all_speakers = np.unique(speaker_id)
        all_speakers_seq = np.unique(speaker_id_seq)
        if len(all_speakers) >= 10:
            sel_10_speakers_list = rng.choice(all_speakers, size=10, replace=False)
            sel_10_sp_mask = np.isin(speaker_id, sel_10_speakers_list)
            sel_10_speakers = speaker_id[sel_10_sp_mask]
        else:
            sel_10_speakers = speaker_id.clone()
        if len(all_speakers_seq) >= 10:
            sel_10_speakers_seq_list = rng.choice(all_speakers_seq, size=10, replace=False)
            sel_10_sp_seq_mask = np.isin(speaker_id_seq, sel_10_speakers_seq_list)
            sel_10_speakers_seq = speaker_id_seq[sel_10_sp_seq_mask]
        else:
            sel_10_speakers_seq = speaker_id_seq.clone()

        if vis_args.vis_td_frames:
            "Frame-level Variable"
            "Time domain"
            "Try using PCA to see if it gives better visualization"
            n_components = 100  # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            td_frame_reduced = torch.tensor(pca_frame.fit_transform(td_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            td_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(td_OCs_frame[oc]))
                td_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for OC {oc+1} frame PCA: {explained_var:.2f}%")
            td_OCs_frame_reduced = torch.stack(td_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            td_OCs_concat_frame = td_OCs_frame.transpose(0,1).reshape(td_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            td_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(td_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")      

            "Frame Speakers - Time Domain"
            if "speaker_id" in vis_args.variables_to_plot:        
                sel_td_frame_10_speakers = td_frame_reduced[sel_10_sp_mask]
                sel_td_OCs_frame_10_speakers = td_OCs_frame_reduced[:,sel_10_sp_mask,:]
                sel_td_OCs_concat_frame_10_speakers = td_OCs_concat_frame_reduced[sel_10_sp_mask] 

            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            
            if "phoneme" in vis_args.variables_to_plot:        
                "Frequency and all Phonemes"
                
                "TSNE - Time domain - Vowels & Frequency"
                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='cosine',early_exaggeration=10,
                                init='pca'),
                }   
                
                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec = phoneme,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  phoneme,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                )

                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Time domain - Vowels & Frequency"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }   
                    
                    visualize(data_training_args, 
                        config,
                        X = td_frame_reduced,
                        OCs = td_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  phoneme,
                        target = "phoneme",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  phonemes39,
                        target = "phoneme",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'phonemes')
                    )
            
            "--------------------------------------------------------------------------------------------------------------"

            if "speaker_id" in vis_args.variables_to_plot:        
                "TSNE - Time domain - Frame-level Speaker"
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
                "10 random speakers"
                visualize(data_training_args, 
                    config,
                    X = sel_td_frame_10_speakers,
                    OCs = sel_td_OCs_frame_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_frame_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level Speaker"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = sel_td_frame_10_speakers,
                        OCs = sel_td_OCs_frame_10_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_td_OCs_concat_frame_10_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'speakers')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "group" in vis_args.variables_to_plot:        
                "TSNE - Time domain - Frame-level Group / Disease Category"
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

                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  group,
                    target = "group_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'group')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  group,
                    target = "group_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'group')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level Group / Disease Category"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = td_frame_reduced,
                        OCs = td_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  group,
                        target = "group_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'group')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  group,
                        target = "group_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'group')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "disease_duration" in vis_args.variables_to_plot:        
                "TSNE - Time domain - Frame-level Disease Progression"
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

                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  disease_duration,
                    target = "disease_duration_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'disease_duration')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  disease_duration,
                    target = "disease_duration_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'disease_duration')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level Disease Progression"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = td_frame_reduced,
                        OCs = td_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  disease_duration,
                        target = "disease_duration_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'disease_duration')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  disease_duration,
                        target = "disease_duration_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'disease_duration')
                    )

            
            "--------------------------------------------------------------------------------------------------------------"
            if "king_stage" in vis_args.variables_to_plot:        
                "TSNE - Time domain - Frame-level King's Progression Stage Scale"
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

                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  king_stage,
                    target = "king_stage_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'king_stage')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  king_stage,
                    target = "king_stage_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'king_stage')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level King's Progression Stage Scale"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = td_frame_reduced,
                        OCs = td_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  king_stage,
                        target = "king_stage_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'king_stage')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  king_stage,
                        target = "king_stage_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'king_stage')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "cantagallo" in vis_args.variables_to_plot:        
                "TSNE - Time domain - Frame-level Cantagallo Questionnaire Scale"
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

                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  cantagallo,
                    target = "cantagallo_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'cantagallo')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  cantagallo,
                    target = "cantagallo_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'cantagallo')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level Cantagallo Questionnaire Scale"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = td_frame_reduced,
                        OCs = td_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  cantagallo,
                        target = "cantagallo_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'cantagallo')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  cantagallo,
                        target = "cantagallo_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'cantagallo')
                    )

            
            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_total" in vis_args.variables_to_plot:        
                "TSNE - Time domain - Frame-level ALSFRS Scale Total Score"
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

                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total,
                    target = "alsfrs_total_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_total')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total,
                    target = "alsfrs_total_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_total')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level ALSFRS Scale Total Score"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = td_frame_reduced,
                        OCs = td_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total,
                        target = "alsfrs_total_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_total')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total,
                        target = "alsfrs_total_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_total')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_speech" in vis_args.variables_to_plot:        
                "TSNE - Time domain - Frame-level ALSFRS Scale Speech Subitem"
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

                visualize(data_training_args, 
                    config,
                    X = td_frame_reduced,
                    OCs = td_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech,
                    target = "alsfrs_speech_frame",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_speech')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech,
                    target = "alsfrs_speech_frame",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_speech')
                )
                
                if vis_args.use_umap:
                    "UMAP - Time domain - Frame-level ALSFRS Scale Speech Subitem"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'cosine',n_neighbors=30,min_dist=0.2,densmap=False)        
                    }    
                    visualize(data_training_args, 
                        config,
                        X = td_frame_reduced,
                        OCs = td_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech,
                        target = "alsfrs_speech_frame",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_speech')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech,
                        target = "alsfrs_speech_frame",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_frame',data_training_args.dataset_name,'alsfrs_speech')
                    )


        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"

        "Mel Filterbank domain - Vowels & Frequency"
        if vis_args.vis_mel_frames:
            n_components = 25 # Choose number of components to keep

            "PCA for original frame"
            pca_frame = PCA(n_components=n_components, random_state=0)
            mel_frame_reduced = torch.tensor(pca_frame.fit_transform(mel_frame))
            explained_var_orig = sum(pca_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain original frame PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components frame" 
            mel_OCs_frame_reduced = []
            for oc in range(config.NoC):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(mel_OCs_frame[oc]))
                mel_OCs_frame_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel OC {oc+1} frame PCA: {explained_var:.2f}%")
            mel_OCs_frame_reduced = torch.stack(mel_OCs_frame_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            mel_OCs_concat_frame = mel_OCs_frame.transpose(0,1).reshape(mel_OCs_frame.shape[1],-1)
            pca_OCs_concat_frame = PCA(n_components=n_components, random_state=0)
            mel_OCs_concat_frame_reduced = torch.tensor(pca_OCs_concat_frame.fit_transform(mel_OCs_concat_frame))
            explained_var_OCs = sum(pca_OCs_concat_frame.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat frame PCA: {explained_var_OCs:.2f}%")

            "Speakers - Mel Domain"
            if "speaker_id" in vis_args.variables_to_plot:        
                sel_mel_frame_10_speakers = mel_frame_reduced[sel_10_sp_mask]
                sel_mel_OCs_frame_10_speakers = mel_OCs_frame_reduced[:,sel_10_sp_mask,:]
                sel_mel_OCs_concat_frame_10_speakers = mel_OCs_concat_frame_reduced[sel_10_sp_mask] 

            data_training_args.frequency_vis = True
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'

            if "phoneme" in vis_args.variables_to_plot:        
                "TSNE - Mel domain - Frequency and all Phonemes"
                "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  phoneme,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  phoneme,
                    target = "phoneme",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                )
                
                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frequency & Phonemes"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_frame_reduced,
                        OCs = mel_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  phoneme,
                        target = "phoneme",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  phoneme,
                        target = "phoneme",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'phonemes')
                    )

            if "speaker_id" in vis_args.variables_to_plot:  
                "TSNE - Mel domain - Frame-level Speaker"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 

                "10 random speakers"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_frame_10_speakers,
                    OCs = sel_mel_OCs_frame_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_frame_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers,
                    target = "speaker_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                )

                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level Speaker"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_frame_10_speakers,
                        OCs = sel_mel_OCs_frame_10_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers,
                        target = "speaker_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_OCs_concat_frame_10_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers,
                        target = "speaker_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames_10_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'speakers')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "group" in vis_args.variables_to_plot:  
                "TSNE - Mel domain - Frame-level Group / Disease Category"
                "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
                data_training_args.frequency_vis = False
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  group,
                    target = "group_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'group')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  group,
                    target = "group_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'group')
                )
                
                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level Group / Disease Category"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_frame_reduced,
                        OCs = mel_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  group,
                        target = "group_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'group')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  group,
                        target = "group_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'group')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "disease_duration" in vis_args.variables_to_plot:  
                "TSNE - Mel domain - Frame-level Disease Duration"
                "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
                data_training_args.frequency_vis = False
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  disease_duration,
                    target = "disease_duration_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'disease_duration')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  disease_duration,
                    target = "disease_duration_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'disease_duration')
                )
                
                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level Group / Disease Category"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_frame_reduced,
                        OCs = mel_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  disease_duration,
                        target = "disease_duration_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'disease_duration')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  disease_duration,
                        target = "disease_duration_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'disease_duration')
                    ) 

            "--------------------------------------------------------------------------------------------------------------"
            if "king_stage" in vis_args.variables_to_plot:  
                "TSNE - Mel domain - Frame-level King's Progression Stage Scale"
                "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
                data_training_args.frequency_vis = False
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  king_stage,
                    target = "king_stage_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'king_stage')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  king_stage,
                    target = "king_stage_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'king_stage')
                )
                
                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level King's Progression Stage Scale"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_frame_reduced,
                        OCs = mel_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  king_stage,
                        target = "king_stage_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'king_stage')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  king_stage,
                        target = "king_stage_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'king_stage')
                    ) 

            "--------------------------------------------------------------------------------------------------------------"
            if "cantagallo" in vis_args.variables_to_plot:  
                "TSNE - Mel domain - Frame-level Cantagallo Questionnaire Scale"
                "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
                data_training_args.frequency_vis = False
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  cantagallo,
                    target = "cantagallo_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'cantagallo')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  cantagallo,
                    target = "cantagallo_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'cantagallo')
                )
                
                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level Cantagallo Questionnaire Scale"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_frame_reduced,
                        OCs = mel_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  cantagallo,
                        target = "cantagallo_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'cantagallo')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  cantagallo,
                        target = "cantagallo_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'cantagallo')
                    ) 

            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_total" in vis_args.variables_to_plot:  
                "TSNE - Mel domain - Frame-level ALSFRS Scale Total Score"
                "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
                data_training_args.frequency_vis = False
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total,
                    target = "alsfrs_total_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_total')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total,
                    target = "alsfrs_total_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_total')
                )
                
                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level ALSFRS Scale Total Score"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_frame_reduced,
                        OCs = mel_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total,
                        target = "alsfrs_total_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_total')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total,
                        target = "alsfrs_total_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_total')
                    ) 

            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_speech" in vis_args.variables_to_plot:  
                "TSNE - Mel domain - Frame-level ALSFRS Scale Speech Subitem Score"
                "Result is robust to changes in perplexity, metric, learning rate, and early exaggeration"
                data_training_args.frequency_vis = False
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=12,
                                init='pca'),
                } 
                visualize(data_training_args, 
                    config,
                    X = mel_frame_reduced,
                    OCs = mel_OCs_frame_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech,
                    target = "alsfrs_speech_frame",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict= manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_speech')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_frame_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech,
                    target = "alsfrs_speech_frame",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_speech')
                )
                
                if vis_args.use_umap:
                    data_training_args.vis_method = 'umap'
                    "UMAP - Mel domain - Frame-level ALSFRS Scale Speech Subitem Score"
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=100,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_frame_reduced,
                        OCs = mel_OCs_frame_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech,
                        target = "alsfrs_speech_frame",
                        data_set =  data_training_args.dataset_name + '_mel_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict= manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_speech')
                    )

                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_frame_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech,
                        target = "alsfrs_speech_frame",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.frames_to_vis) + '_frames',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_frame',data_training_args.dataset_name,'alsfrs_speech')
                    ) 

        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "--------------------------------------------------------------------------------------------------------------"
        "Sequence-level Variable"
        "Time domain"
        

        if vis_args.vis_td_seq:
            n_components = min(75,td_seq.shape[0])  # Choose number of components to keep

            "Reduce every sequence separately"
            "PCA 10 speakers"
            pca_seq = PCA(n_components=n_components, random_state=0)
            td_seq_reduced = torch.tensor(pca_seq.fit_transform(td_seq))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for original sequence 10 speakers PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            td_OCs_seq_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(td_OCs_seq[oc,:,:]))
                td_OCs_seq_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for time domain OC {oc+1} sequence 10 speakers PCA: {explained_var:.2f}%")
            td_OCs_seq_reduced = torch.stack(td_OCs_seq_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            td_OCs_concat_seq_reduced = torch.tensor(pca_OCs_concat_seq.fit_transform(td_OCs_concat_seq))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for time domain OCs_concat sequence 10 speakers PCA: {explained_var_OCs:.2f}%")

            "Sequence Selected Speakers - Time Domain - First reduce then select"
            "Contrary to vowels and TIMIT, here we have more than one sequence variables so we do not want to select for them as well"
            if "speaker_id" in vis_args.variables_to_plot_seq:  
                sel_td_seq_10_speakers = td_seq_reduced[sel_10_sp_seq_mask]
                sel_td_OCs_seq_10_speakers = td_OCs_seq_reduced[:,sel_10_sp_seq_mask,:].transpose(0,1)
                sel_td_OCs_concat_seq_10_speakers = td_OCs_concat_seq_reduced[sel_10_sp_seq_mask]

            "--------------------------------------------------------------------------------------------------------------"
            "TSNE - Time domain - Sequence-level Speaker"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            if "speaker_id" in vis_args.variables_to_plot_seq:  
                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,sel_td_seq_10_speakers.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                } 

                "10 random speakers"
                visualize(data_training_args, 
                    config,
                    X = sel_td_seq_10_speakers,
                    OCs = sel_td_OCs_seq_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_td_OCs_concat_seq_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level Speaker"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = sel_td_seq_10_speakers,
                        OCs = sel_td_OCs_seq_10_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_td_OCs_concat_seq_10_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name)
                    )
            
            "--------------------------------------------------------------------------------------------------------------"
            if "phoneme" in vis_args.variables_to_plot_seq:  
                "TSNE - Time domain - Sequence-level Phoneme and Frequency"
                data_training_args.frequency_vis = True
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,td_seq_reduced.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  phoneme_seq,
                    target = "phoneme_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  phoneme_seq,
                    target = "phoneme_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'phonemes')
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level Phoneme and Frequency"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = td_seq_reduced,
                        OCs = td_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  phoneme_seq,
                        target = "phoneme_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'phonemes')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  phoneme_seq,
                        target = "phoneme_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'phonemes')
                    )
           
            "--------------------------------------------------------------------------------------------------------------"
            if "group" in vis_args.variables_to_plot_seq:  
                "TSNE - Time domain - Sequence-level Group / Disease Category"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,td_seq_reduced.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                }

                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  group_seq,
                    target = "group_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'group')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  group_seq,
                    target = "group_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'group')
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level Group / Disease Category"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = td_seq_reduced,
                        OCs = td_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  group_seq,
                        target = "group_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'group')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  group_seq,
                        target = "group_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'group')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "disease_duration" in vis_args.variables_to_plot_seq:  
                "TSNE - Time domain - Sequence-level Disease Duration"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,td_seq_reduced.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  disease_duration_seq,
                    target = "disease_duration_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'disease_duration')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  disease_duration_seq,
                    target = "disease_duration_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'disease_duration')
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level Disease Duration"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = td_seq_reduced,
                        OCs = td_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  disease_duration_seq,
                        target = "disease_duration_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'disease_duration')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  disease_duration_seq,
                        target = "disease_duration_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'disease_duration')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "king_stage" in vis_args.variables_to_plot_seq:  
                "TSNE - Time domain - Sequence-level King's Disease Staging Scale"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,td_seq_reduced.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                }

                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  king_stage_seq,
                    target = "king_stage_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'king_stage')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  king_stage_seq,
                    target = "king_stage_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'king_stage')
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level King's Disease Staging Scale"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = td_seq_reduced,
                        OCs = td_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  king_stage_seq,
                        target = "king_stage_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'king_stage')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  king_stage_seq,
                        target = "king_stage_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'king_stage')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "cantagallo" in vis_args.variables_to_plot_seq:  
                "TSNE - Time domain - Sequence-level Cantagallo Questionnaire Scale"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,td_seq_reduced.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  cantagallo_seq,
                    target = "cantagallo_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'cantagallo')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  cantagallo_seq,
                    target = "cantagallo_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'cantagallo')
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level Cantagallo Questionnaire Scale"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = td_seq_reduced,
                        OCs = td_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  cantagallo_seq,
                        target = "cantagallo_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'cantagallo')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  cantagallo_seq,
                        target = "cantagallo_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'cantagallo')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_total" in vis_args.variables_to_plot_seq:  
                "TSNE - Time domain - Sequence-level ALSFRS Scale Total Score"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,td_seq_reduced.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total_seq,
                    target = "alsfrs_total_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_total')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total_seq,
                    target = "alsfrs_total_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_total')
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level ALSFRS Scale Total Score"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = td_seq_reduced,
                        OCs = td_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total_seq,
                        target = "alsfrs_total_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_total')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total_seq,
                        target = "alsfrs_total_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_total')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_speech" in vis_args.variables_to_plot_seq:  
                "TSNE - Time domain - Sequence-level ALSFRS Speech Subitem Score"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(15,td_seq_reduced.shape[0]-1), metric='canberra',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = td_seq_reduced,
                    OCs = td_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech_seq,
                    target = "alsfrs_speech_seq",
                    data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_speech')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = td_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech_seq,
                    target = "alsfrs_speech_seq",
                    data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_speech')
                )

                if vis_args.use_umap:
                    "UMAP - Time domain - Sequence-level ALSFRS Speech Subitem Score"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'canberra',n_neighbors=15,min_dist=0.9,densmap=False)        
                    }   
                    visualize(data_training_args, 
                        config,
                        X = td_seq_reduced,
                        OCs = td_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech_seq,
                        target = "alsfrs_speech_seq",
                        data_set = data_training_args.dataset_name + '_td_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_speech')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = td_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech_seq,
                        target = "alsfrs_speech_seq",
                        data_set = data_training_args.dataset_name + '_td_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'time_domain_sequence',data_training_args.dataset_name,'alsfrs_speech')
                    )


        "--------------------------------------------------------------------------------------------------------------"
        "Mel domain - Sequence-level Speaker"

        if vis_args.vis_mel_seq:
            "Reduce every sequence separately - for 10 and 20 speakers"
            "PCA 10 speakers"
            n_components = min(20,mel_seq.shape[0])  # Choose number of components to keep

            # PCA for original sequence
            pca_seq = PCA(n_components=n_components, random_state=0)
            mel_seq_reduced = torch.tensor(pca_seq.fit_transform(mel_seq))
            explained_var_orig = sum(pca_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel original sequence 10 speakers PCA: {explained_var_orig:.2f}%")

            "PCA for orthogonal components sequence" 
            mel_OCs_seq_reduced = []
            for oc in range(config.NoC_seq):
                pca_OC = PCA(n_components=n_components, random_state=0)
                oc_reduced = torch.tensor(pca_OC.fit_transform(mel_OCs_seq[oc,:,:]))
                mel_OCs_seq_reduced.append(oc_reduced)
                explained_var = sum(pca_OC.explained_variance_ratio_) * 100
                print(f"Explained variance for mel domain OC {oc+1} sequence 10 speakers PCA: {explained_var:.2f}%")
            mel_OCs_seq_reduced = torch.stack(mel_OCs_seq_reduced, dim=0)
            
            "PCA on concatenated OCs - Use as X"
            pca_OCs_concat_seq = PCA(n_components=n_components, random_state=0)
            mel_OCs_concat_seq_reduced = torch.tensor(pca_OCs_concat_seq.fit_transform(mel_OCs_concat_seq))
            explained_var_OCs = sum(pca_OCs_concat_seq.explained_variance_ratio_) * 100
            print(f"Explained variance for mel domain OCs_concat sequence 10 speakers PCA: {explained_var_OCs:.2f}%")

            if "speaker_id" in vis_args.variables_to_plot_seq:  
                "Sequence Selected Speakers - Mel Domain"
                sel_mel_seq_10_speakers = mel_seq_reduced[sel_10_sp_seq_mask]
                sel_mel_OCs_seq_10_speakers = mel_OCs_seq_reduced[:,sel_10_sp_seq_mask,:].transpose(0,1)
                sel_mel_OCs_concat_seq_10_speakers = mel_OCs_concat_seq_reduced[sel_10_sp_seq_mask]
            
            "--------------------------------------------------------------------------------------------------------------"
            "TSNE - Mel domain - Sequence-level Speaker"
            data_training_args.frequency_vis = False
            data_training_args.generative_factors_vis= True
            data_training_args.vis_sphere= False
            data_training_args.tsne_plot_2d_3d = '2d'
            data_training_args.vis_method = 'tsne'
            if "speaker_id" in vis_args.variables_to_plot_seq:  
                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,sel_mel_seq_10_speakers.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = sel_mel_seq_10_speakers,
                    OCs = sel_mel_OCs_seq_10_speakers,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'speakers')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = sel_mel_OCs_concat_seq_10_speakers,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  sel_10_speakers_seq,
                    target = "speaker_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'speakers')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level Speaker"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    }  
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_seq_10_speakers,
                        OCs = sel_mel_OCs_seq_10_speakers,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis) +'_seqs_10_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'speakers')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = sel_mel_OCs_concat_seq_10_speakers,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  sel_10_speakers_seq,
                        target = "speaker_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis) + '_seqs_10_speakers',
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'speakers')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "phoneme" in vis_args.variables_to_plot_seq:  
                "TSNE - Mel domain - Sequence-level Phonemes & Frequency"
                data_training_args.frequency_vis = True
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,mel_seq_reduced.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                }  

                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  phoneme_seq,
                    target = "phoneme_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'phonemes')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  phoneme_seq,
                    target = "phoneme_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'phonemes')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level Phonemes & Frequency"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    } 
                    visualize(data_training_args, 
                        config,
                        X = mel_seq_reduced,
                        OCs = mel_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  phoneme_seq,
                        target = "phoneme_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'phonemes')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  phoneme_seq,
                        target = "phoneme_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'phonemes')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "group" in vis_args.variables_to_plot_seq:  
                "TSNE - Mel domain - Sequence-level Group / Disease Category"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,mel_seq_reduced.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  group_seq,
                    target = "group_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'group')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  group_seq,
                    target = "group_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'group')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level Group / Disease Category"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    }
                    visualize(data_training_args, 
                        config,
                        X = mel_seq_reduced,
                        OCs = mel_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  group_seq,
                        target = "group_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'group')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  group_seq,
                        target = "group_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'group')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "disease_duration" in vis_args.variables_to_plot_seq:  
                "TSNE - Mel domain - Sequence-level Disease Duration"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,mel_seq_reduced.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  disease_duration_seq,
                    target = "disease_duration_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'disease_duration')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  disease_duration_seq,
                    target = "disease_duration_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'disease_duration')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level Disease Duration"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    } 
                    visualize(data_training_args, 
                        config,
                        X = mel_seq_reduced,
                        OCs = mel_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  disease_duration_seq,
                        target = "disease_duration_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'disease_duration')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  disease_duration_seq,
                        target = "disease_duration_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'disease_duration')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "king_stage" in vis_args.variables_to_plot_seq:  
                "TSNE - Mel domain - Sequence-level King's Disease Staging Scale"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,mel_seq_reduced.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  king_stage_seq,
                    target = "king_stage_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'king_stage')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  king_stage_seq,
                    target = "king_stage_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'king_stage')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level King's Disease Staging Scale"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    }  
                    visualize(data_training_args, 
                        config,
                        X = mel_seq_reduced,
                        OCs = mel_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  king_stage_seq,
                        target = "king_stage_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'king_stage')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  king_stage_seq,
                        target = "king_stage_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'king_stage')
                    )


            "--------------------------------------------------------------------------------------------------------------"
            if "cantagallo" in vis_args.variables_to_plot_seq:  
                "TSNE - Mel domain - Sequence-level Cantagallo Questionnaire Scale"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,mel_seq_reduced.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                } 

                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  cantagallo_seq,
                    target = "cantagallo_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'cantagallo')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  cantagallo_seq,
                    target = "cantagallo_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'cantagallo')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level Cantagallo Questionnaire Scale"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    }  
                    visualize(data_training_args, 
                        config,
                        X = mel_seq_reduced,
                        OCs = mel_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  cantagallo_seq,
                        target = "cantagallo_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'cantagallo')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  cantagallo_seq,
                        target = "cantagallo_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'cantagallo')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_total" in vis_args.variables_to_plot_seq:  
                "TSNE - Mel domain - Sequence-level ALSFRS Scale Total Score"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,mel_seq_reduced.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                }  

                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total_seq,
                    target = "alsfrs_total_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_total')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_total_seq,
                    target = "alsfrs_total_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_total')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level ALSFRS Scale Total Score"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    } 
                    visualize(data_training_args, 
                        config,
                        X = mel_seq_reduced,
                        OCs = mel_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total_seq,
                        target = "alsfrs_total_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_total')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_total_seq,
                        target = "alsfrs_total_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_total')
                    )

            "--------------------------------------------------------------------------------------------------------------"
            if "alsfrs_speech" in vis_args.variables_to_plot_seq:  
                "TSNE - Mel domain - Sequence-level ALSFRS Speech Subitem Score"
                data_training_args.frequency_vis = False
                data_training_args.generative_factors_vis= True
                data_training_args.vis_sphere= False
                data_training_args.tsne_plot_2d_3d = '2d'
                data_training_args.vis_method = 'tsne'

                manifold_dict = {
                    'tsne': TSNE(n_components=2, random_state=vis_args.random_seed_vis, learning_rate= 'auto', 
                                max_iter = 1000, perplexity=min(5,mel_seq_reduced.shape[0]-1), metric='cityblock',early_exaggeration=10,
                                init='pca'),
                }  

                visualize(data_training_args, 
                    config,
                    X = mel_seq_reduced,
                    OCs = mel_OCs_seq_reduced,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech_seq,
                    target = "alsfrs_speech_seq",
                    data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_speech')
                )

                "Also plot for the concatenated OCs"
                visualize(data_training_args, 
                    config,
                    X = mel_OCs_concat_seq_reduced,
                    OCs = None,
                    z_or_h = 'z',
                    y_vec =  alsfrs_speech_seq,
                    target = "alsfrs_speech_seq",
                    data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                    manifold_dict = manifold_dict,
                    return_data = True,
                    display_figures = True,
                    save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_speech')
                )

                if vis_args.use_umap:
                    "UMAP - Mel domain - Sequence-level ALSFRS Speech Subitem Score"
                    data_training_args.vis_method = 'umap'
                    manifold_dict = {
                        'umap': umap.UMAP(n_components=2, random_state=vis_args.random_seed_vis, metric = 'braycurtis',n_neighbors=10,min_dist=0.9,densmap=False)        
                    }  
                    visualize(data_training_args, 
                        config,
                        X = mel_seq_reduced,
                        OCs = mel_OCs_seq_reduced,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech_seq,
                        target = "alsfrs_speech_seq",
                        data_set = data_training_args.dataset_name + '_mel_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_speech')
                    )
                    "Also plot for the concatenated OCs"
                    visualize(data_training_args, 
                        config,
                        X = mel_OCs_concat_seq_reduced,
                        OCs = None,
                        z_or_h = 'z',
                        y_vec =  alsfrs_speech_seq,
                        target = "alsfrs_speech_seq",
                        data_set = data_training_args.dataset_name + '_mel_OCs_concat_' + str(vis_args.seq_to_vis),
                        manifold_dict = manifold_dict,
                        return_data = True,
                        display_figures = True,
                        save_dir = os.path.join(vis_args.save_vis_dir,decomp_args.decomp_to_perform,'low_input_dim',vis_args.set_to_use_for_vis,data_training_args.vis_method,'mel_sequence',data_training_args.dataset_name,'alsfrs_speech')
                    )


if __name__ == "__main__":
    main()