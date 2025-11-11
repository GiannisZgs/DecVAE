"""
This script performs decomposition and extracts features and metrics for the SimVowels and TIMIT datasets, to assess
how well the decomposition can recover the true formants in the SimVowels dataset and compare the results with the real speech
TIMIT dataset. These data are used to obtain Supplementary Information Figure 4 and 6.
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
from args_configs import ModelArguments, DataTrainingArguments, DecompositionArguments, TrainingObjectiveArguments
from dataset_loading import load_timit, load_sim_vowels
from utils.misc import parse_args, debugger_is_active
from utils.audio_handling import find_speaker_gender

import transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    is_wandb_available,
    set_seed,
    HfArgumentParser,
)

from functools import partial
import scipy
import numpy as np
from pathlib import Path
import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs as DDPK
from datasets import DatasetDict, concatenate_datasets, Dataset
from huggingface_hub import HfApi
from torch.utils.data.dataloader import DataLoader
from feature_extraction import extract_mel_spectrogram
import json


JSON_FILE_NAME_MANUAL = "config_files/DecVAEs/sim_vowels/pre-training/config_pretraining_sim_vowels_NoC3.json"
SAVE_DIR = '../data_for_figures/decomposition_quality' 

logger = get_logger(__name__)

def main():
    "Parse the arguments"
    parser = HfArgumentParser((ModelArguments, TrainingObjectiveArguments, DecompositionArguments, DataTrainingArguments))
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

            if data_training_args.dataset_name == "timit":
                with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                    phoneme39_to_id = json.load(json_file)
                with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'r') as json_file:
                    phoneme48_to_id = json.load(json_file)

    except FileNotFoundError: #else:#
        "Download and create train, validation dataset"
        
        if "timit" in data_training_args.dataset_name:
            raw_datasets = load_timit(data_training_args)

        elif "sim_vowels" in data_training_args.dataset_name:
            raw_datasets = load_sim_vowels(data_training_args)
        
        
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

    "Activate gradient checkpointing if needed"
    if data_training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        loss_mode = training_obj_args.loss_mode,
        dataset_name = data_training_args.dataset_name,
    )
    train_dataloader = DataLoader(
        vectorized_datasets['train'],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=data_training_args.per_device_train_batch_size,
    )


    print(f"Dataloader has {len(train_dataloader)} steps in an epoch")

    "Only show the progress bar once on each machine."
    starting_epoch = 0
    if data_training_args.dataset_name == 'timit':
        quality_frame = {'NRMSEs': [], 'correlograms': [], 'phonemes39': [], "vowels": [], "consonants": [], 'speaker_id': [], 'gender': [], 'mel': [], 'detected_frequencies': [], 'overlap_mask': []}
    elif data_training_args.dataset_name == 'sim_vowels':
        quality_frame = {'NRMSEs': [], 'correlograms': [], "vowels": [], 'speaker_id': [], 'gender': [], 'mel': [], 'detected_frequencies': [], 'overlap_mask': []}

    quality_seq = {'NRMSEs': [], 'correlograms': [], 'speaker_id': [], 'gender': [], 'mel': []}
    for step, batch in enumerate(train_dataloader):

        batch_size = batch["input_values"].shape[0]
        assert batch_size == 1
        frames = batch["sub_attention_mask"].sum()
        frame_len = batch["input_values"].shape[-1]
        seq_len = batch["attention_mask"].sum()
        overlap_mask = batch["overlap_mask"].squeeze(0)
        for o,ov in enumerate(overlap_mask[:frames]):
            if not ov == 0 and not ov == 1:
                overlap_mask[o] = 0
        overlap_mask = overlap_mask[:frames]
        quality_frame['overlap_mask'].append(overlap_mask.detach().cpu().numpy().tolist())
        "Keep: NRMSEs (frame/seq), correlograms (frame/seq), phonemes39 (vowels/consonants), speaker_id (M/F)"
        "Keep: mel_features, detected frequencies of OCs"
        "For each frame"

        "NRMSEs (frame/seq)"
        if batch["reconstruction_NRMSEs"].shape[-1] != frames:
            if batch["reconstruction_NRMSEs"].shape[-1] > frames:
                batch["reconstruction_NRMSEs"] = batch["reconstruction_NRMSEs"][:,:frames]
            else:
                batch["reconstruction_NRMSEs"] = torch.cat((batch["reconstruction_NRMSEs"],-1*torch.ones((1,frames-batch["reconstruction_NRMSEs"].shape[-1]),device = batch["reconstruction_NRMSEs"].device)),dim=-1)
        #assert batch["reconstruction_NRMSEs"].shape[-1] == frames
        #assert batch["correlograms"].shape[1] == frames
        quality_frame["NRMSEs"].append(batch["reconstruction_NRMSEs"].squeeze(0).detach().cpu().numpy().tolist())
        quality_seq["NRMSEs"].append(batch["reconstruction_NRMSE_seq"].detach().cpu().numpy().tolist())
        "Correlograms (frame/seq)"
        if batch["correlograms"].shape[1] != frames:
            if batch["correlograms"].shape[1] > frames:
                batch["correlograms"] = batch["correlograms"][:,:frames,...]
            else:
                batch["correlograms"] = torch.cat((batch["correlograms"],-1*torch.ones((1,frames-batch["correlograms"].shape[1],batch["correlograms"].shape[-2], batch["correlograms"].shape[-1]),device = batch["correlograms"].device)),dim=1)
        quality_frame["correlograms"].append(batch["correlograms"].squeeze(0).detach().cpu().numpy().tolist())
        quality_seq["correlograms"].append(batch["correlogram_seq"].squeeze(0).detach().cpu().numpy().tolist())
        if data_training_args.dataset_name == "timit":
            "Discard padded frames"
            batch["input_values"] = batch["input_values"][:,:,:frames,...]

            "Phonemes39 (vowels/consonants)"
            #quality_frame["phonemes39"].append(batch["phonemes39"].squeeze(0)[:frames])
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

            quality_frame["phonemes39"].append(batch_phonemes)
            quality_frame["vowels"].append(batch_vowels)
            quality_frame["consonants"].append(batch_consonants)

            "Speaker_id (M/F)"
            with open(data_training_args.path_to_timit_speaker_dict_file, 'r') as json_file:
                speaker_id_to_id = json.load(json_file)
            speaker_id_to_id = {v: k for k, v in speaker_id_to_id.items()}
            quality_frame["speaker_id"].append(batch["speaker_id"].detach().cpu().numpy().tolist())
            quality_seq["speaker_id"].append(batch["speaker_id"].detach().cpu().numpy().tolist())

            speaker_dir = "/home/giannis/Documents/TIMIT/data/lisa/data/timit/raw/TIMIT/TRAIN"
            
            gender = find_speaker_gender(speaker_dir,speaker_id_to_id[batch["speaker_id"].item()])
            quality_frame["gender"].append(gender)
            quality_seq["gender"].append(gender)

        elif data_training_args.dataset_name == "sim_vowels":
            "Vowels"
            quality_frame["vowels"].append(batch["vowel_labels"].squeeze(0)[:frames].detach().cpu().numpy().tolist())
            quality_frame["speaker_id"].append(batch["speaker_vt_factor"].detach().cpu().numpy().tolist())
            quality_seq["speaker_id"].append(batch["speaker_vt_factor"].detach().cpu().numpy().tolist())
            if batch["speaker_vt_factor"] <= 0.95:
                quality_frame["gender"].append("F")
                quality_seq["gender"].append("F")
            elif batch["speaker_vt_factor"] >= 1.05:
                quality_frame["gender"].append("M")
                quality_seq["gender"].append("M")
            else:
                quality_frame["gender"].append("U")
                quality_seq["gender"].append("U")

        "Mel_features - Frame"
        mel_features = torch.zeros((batch_size,config.NoC+1,frames,frame_len))
        mel_features[:,0,...], spec_max = extract_mel_spectrogram(batch["input_values"][:,0,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = None)
        for o in range(1,batch["input_values"].shape[1]):
            mel_features[:,o,...],_ = extract_mel_spectrogram(batch["input_values"][:,o,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = spec_max)

        mel_features = mel_features.squeeze(0)
        quality_frame["mel"].append(mel_features.detach().cpu().numpy().tolist())

        "Mel_features - Sequence" 

        "First bring sequence to a framed format"
        if data_training_args.dataset_name == "timit":
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
        
        mel_seq_features = torch.zeros((batch_size,config.NoC+1,seq_len))
        seq_mel,spec_max_seq = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = None)

        mel_seq_features[:,0,...] = seq_mel.reshape(batch["input_seq_values"].shape[0],-1)

        for o in range(1,batch["input_values"].shape[1]):
            seq_mel,_ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len, ref = spec_max_seq)
            mel_seq_features[:,o,...] = seq_mel.reshape(batch["input_seq_values"].shape[0],-1)
        
        mel_seq_features = mel_seq_features.squeeze(0)
        quality_seq["mel"].append(mel_seq_features.detach().cpu().numpy().tolist())

        "Detected frequencies of OCs - Frame"
        det_freqs = np.zeros((config.NoC,frames))
        for o in range(1,config.NoC+1):
            f,Pxx = scipy.signal.welch(batch["input_values"][0,o,:,:].detach().cpu().numpy(), fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
            for r in range(frames):
                det_freqs[o-1,r] = f[np.argmax(Pxx[r,:])]
            
        quality_frame["detected_frequencies"].append(det_freqs.tolist())

    with open(os.path.join(SAVE_DIR, decomp_args.decomp_to_perform, f'{data_training_args.dataset_name}_dec_quality_frame_NoC{decomp_args.NoC}_{decomp_args.decomp_to_perform}.json'), 'w') as f:
        json.dump(quality_frame, f)
    print("Saved frame quality successfully")
    with open(os.path.join(SAVE_DIR, decomp_args.decomp_to_perform, f'{data_training_args.dataset_name}_dec_quality_sequence_NoC{decomp_args.NoC}_{decomp_args.decomp_to_perform}.json'), 'w') as f:
        json.dump(quality_seq, f)
    print("Saved sequence quality successfully")

if __name__ == "__main__":
    main()