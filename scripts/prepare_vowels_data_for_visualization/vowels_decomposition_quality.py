"""
This script extracts features, calculates some metrics and organizes data to prepare for visualizations for the SimVowels dataset.
These data are used to obtain Supplementary Information Figures 4 and 6.
"""


import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")
    
from transformers import HfArgumentParser, Wav2Vec2FeatureExtractor
from config_files import DecVAEConfig
from args_configs import ModelArguments, DataTrainingArguments, DecompositionArguments, TrainingObjectiveArguments
from utils.misc import parse_args, debugger_is_active
import torch
import json
import gzip
from models.decomposition_masking import DecompositionModule
import numpy as np
import re

JSON_FILE_NAME_MANUAL = "config_files/DecVAEs/sim_vowels/pre-training/config_pretraining_sim_vowels_NoC3.json" #for debugging purposes only
FNAME = '../data_for_figures/sim_vowels_figures.json.gz'
SAVE_DIR = '../data_for_figures/'
SAMPLE = 'all' #'_0.705' or 'all'
MAX_LENGTH = 1600 # 0.1 seconds
START_SAMPLE = 0 #320
END_SAMPLE = 1600 #720    


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
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_args.model_name_or_path)
    config = DecVAEConfig(**{**model_args.__dict__, **training_obj_args.__dict__, **decomp_args.__dict__})

    decomposition_results = {}

    "Load figure data"
    with gzip.open(FNAME, "rt") as f:
        data_dict = json.load(f)
    patterns = set()  # Use set for unique patterns
    for key in data_dict.keys():
        matches = re.findall(r'_([^_]+)', key)  # Find all patterns after '_'
        patterns.update(matches)

    for pat in patterns:
        if SAMPLE != 'all':
            if pat != SAMPLE:
                continue
        sample_data = {k.replace(pat, ''): v 
                    for k, v in data_dict.items() 
                    if pat in k}
        
        decomposition_results[pat] = {}
        "Frame decomposition"

        for vowel in sample_data.keys():
            decomposition_results[pat][vowel] = {}
            
            formants = np.zeros((3,len(sample_data[vowel]['formant_1_waves'])))
            signal = sample_data[vowel]['formant_1_waves']
            inputs = feature_extractor(
                        signal, sampling_rate=config.fs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
            )
            f1 = inputs.input_values[0][START_SAMPLE:END_SAMPLE].reshape(-1,1).transpose() 
            formants[0] = f1.copy()

            signal = sample_data[vowel]['formant_2_waves']
            inputs = feature_extractor(
                        signal, sampling_rate=config.fs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
            )
            f2 = inputs.input_values[0][START_SAMPLE:END_SAMPLE].reshape(-1,1).transpose() 
            formants[1] = f2.copy()

            signal = sample_data[vowel]['formant_3_waves']
            inputs = feature_extractor(
                        signal, sampling_rate=config.fs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
            )
            f3 = inputs.input_values[0][START_SAMPLE:END_SAMPLE].reshape(-1,1).transpose() 
            formants[2] = f3.copy()

            "Calculate orthogonality index of the formants"
            gt_correlograms = np.zeros((4,3,3))
            for c in range(4):
                frame_start = int(c*config.fs*config.stride)
                frame_end = int(frame_start + config.fs*config.receptive_field)
                for o in range(3):
                    for r in range(3):
                        gt_correlograms[c,o,r] = np.abs(np.cov(formants[o,frame_start:frame_end],formants[r,frame_start:frame_end])[0,1] / (np.std(formants[o,frame_start:frame_end])*np.std(formants[r,frame_start:frame_end])))
                
            
            decomposition_results[pat][vowel]['gt_correlograms'] = gt_correlograms.tolist()

            "Now frame decomposition results"

            signal = sample_data[vowel]['time_domain_signal']
            inputs = feature_extractor(
                        signal, sampling_rate=config.fs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
            )

            decomp_module = DecompositionModule(config)
            mask_indices_seq_length = int(decomp_module._get_feat_extract_output_lengths(MAX_LENGTH))# inputs.input_values[0].shape[0]))

            all_ones_mask = torch.ones((1,mask_indices_seq_length),dtype = torch.bool)
            attention_mask = torch.tensor(np.expand_dims(inputs.attention_mask[0],axis = 0))
            decomposition_outcome, mask_time_indices, _, _, reconstruction_NRMSEs, _, avg_correlogram, correlograms,_, _ ,_, _,_ = decomp_module(
                                    np.expand_dims(inputs.input_values[0],axis=0),
                                    mask_time_indices=all_ones_mask,
                                    attention_mask=attention_mask,
                                    remove_silence = decomp_args.remove_silence
                                )
            
            decomposition_results[pat][vowel]['reconstruction_NRMSEs'] = reconstruction_NRMSEs
            decomposition_results[pat][vowel]['avg_correlogram'] = avg_correlogram.detach().numpy().tolist()
            decomposition_results[pat][vowel]['correlograms'] = [c.detach().numpy().tolist() for c in correlograms]
        

        "Sequence decomposition"
        decomposition_results[pat]['seq'] = {}

        "First save original signals - SEQ"
        
        signal = []
        for vowel in sample_data.keys():
            s = sample_data[vowel]['formant_1_waves']
            signal.extend(s)   

        signal = np.array(signal)
        inputs = feature_extractor(
                    signal, sampling_rate=config.fs, max_length=5*MAX_LENGTH, truncation=True, padding="max_length"
        )
        f1 = inputs.input_values[0].reshape(-1,1).transpose()

        signal = []
        for vowel in sample_data.keys():
            s = sample_data[vowel]['formant_2_waves']
            signal.extend(s)
        
        signal = np.array(signal)
        inputs = feature_extractor(
                    signal, sampling_rate=config.fs, max_length=5*MAX_LENGTH, truncation=True, padding="max_length"
        )
        f2 = inputs.input_values[0].reshape(-1,1).transpose()

        signal = []
        for vowel in sample_data.keys():
            s = sample_data[vowel]['formant_3_waves']
            signal.extend(s)
        
        signal = np.array(signal)
        inputs = feature_extractor(
                    signal, sampling_rate=config.fs, max_length=5*MAX_LENGTH, truncation=True, padding="max_length"
        )
        f3 = inputs.input_values[0].reshape(-1,1).transpose()

        "Calculate orthogonality index of the formants"
        gt_correlograms = np.zeros((3,3))
        for o in range(3):
            for r in range(3):
                gt_correlograms[o,r] = np.abs(np.cov(formants[o],formants[r])[0,1] / (np.std(formants[o])*np.std(formants[r])))
            
            
        decomposition_results[pat]['seq']['gt_correlograms'] = gt_correlograms.tolist()        

        "Now sequence decomposition results"
        signal = []
        for vowel in sample_data.keys():
            s = sample_data[vowel]['time_domain_signal']
            signal.extend(s)   

        signal = np.array(signal)
        inputs = feature_extractor(
                    signal, sampling_rate=config.fs, max_length=5*MAX_LENGTH, truncation=True, padding="max_length"
        )

        decomp_module = DecompositionModule(config)
        mask_indices_seq_length = int(decomp_module._get_feat_extract_output_lengths(5*MAX_LENGTH))# inputs.input_values[0].shape[0]))

        all_ones_mask = torch.ones((1,mask_indices_seq_length),dtype = torch.bool)
        attention_mask = torch.tensor(np.expand_dims(inputs.attention_mask[0],axis = 0))
        decomposition_outcome, mask_time_indices, _, _, _, reconstruction_NRMSEs_seq, _, _, avg_correlogram_seq, correlograms_seq,_, _ ,_ = decomp_module(
                                np.expand_dims(inputs.input_values[0],axis=0),
                                mask_time_indices=all_ones_mask,
                                attention_mask=attention_mask,
                                remove_silence = decomp_args.remove_silence
                            )
        
        seq_decomp = decomposition_outcome["sequence"] 
        decomposition_results[pat]['seq']['reconstruction_NRMSEs'] = reconstruction_NRMSEs_seq
        decomposition_results[pat]['seq']['avg_correlogram'] = avg_correlogram_seq.detach().numpy().tolist()
        decomposition_results[pat]['seq']['correlograms'] = [c.detach().numpy().tolist() for c in correlograms_seq]


        if pat == SAMPLE:
            break

    with open(os.path.join(SAVE_DIR,f'vowels_decomposition_quality_NoC{decomp_args.NoC}_{decomp_args.decomp_to_perform}_all_speakers.json'), 'w') as f:
        json.dump(decomposition_results, f)
        
if __name__ == "__main__":
    main()