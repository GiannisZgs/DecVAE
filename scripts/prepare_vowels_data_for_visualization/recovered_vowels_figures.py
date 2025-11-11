"""
This script extracts features, calculates some metrics and organizes data to prepare for visualizations for the SimVowels dataset.
These data are used to obtain Supplementary Information Figures 4 and 6.
It is the same as vowels_decomposition_quality.py but focuses on a single speaker's data.
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
from models import DecompositionModule
import numpy as np
import scipy
from feature_extraction import extract_mel_spectrogram

JSON_FILE_NAME_MANUAL = "config_files/DecVAEs/sim_vowels/pre-training/config_pretraining_sim_vowels_NoC3.json" #for debugging purposes only
FNAME = '../data_for_figures/sim_vowels_figures.json.gz'
SAVE_DIR = '../data_for_figures'
SAMPLE = '_0.705'
MAX_LENGTH = 1600 # 0.1 seconds
START_SAMPLE = 320
END_SAMPLE = 720    

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

    "Load figure data"
    with gzip.open(FNAME, "rt") as f:
        data_dict = json.load(f)
    sample_data = {k.replace(SAMPLE, ''): v 
                   for k, v in data_dict.items() 
                   if SAMPLE in k}
    decomposition_results = {}


    "Frame decomposition"

    for vowel in sample_data.keys():
        decomposition_results[vowel] = {}

        signal = sample_data[vowel]['time_domain_signal']

        "First save original signals"
        x = np.array(signal)[START_SAMPLE:END_SAMPLE].reshape(-1,1).transpose()
        x_mean = np.mean(x)
        x_std = np.std(x)
        x = (x - x_mean) / x_std
        
        signal = sample_data[vowel]['formant_1_waves']
    
        f1 = np.array(signal)[START_SAMPLE:END_SAMPLE].reshape(-1,1).transpose()
        f1_x = (f1 - x_mean) / x_std


        signal = sample_data[vowel]['formant_2_waves']
        
        f2 = np.array(signal)[START_SAMPLE:END_SAMPLE].reshape(-1,1).transpose() 
        f2_x = (f2 - x_mean) / x_std
        
        signal = sample_data[vowel]['formant_3_waves']
        
        f3 = np.array(signal)[START_SAMPLE:END_SAMPLE].reshape(-1,1).transpose() 
        f3_x = (f3 - x_mean) / x_std

        decomposition_results[vowel]['input_frame'] = np.concatenate((x,f1_x,f2_x,f3_x),axis=0).tolist()

        "Now also calculate and save the initial spectral plots for each formant - FRAME"
        f,Pxx_0 = scipy.signal.welch(decomposition_results[vowel]['input_frame'][0], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        _,Pxx_1 = scipy.signal.welch(decomposition_results[vowel]['input_frame'][1], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        _,Pxx_2 = scipy.signal.welch(decomposition_results[vowel]['input_frame'][2], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        _,Pxx_3 = scipy.signal.welch(decomposition_results[vowel]['input_frame'][3], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        decomposition_results[vowel]['spectral_density_input_frame_X'] = Pxx_0.tolist()
        decomposition_results[vowel]['spectral_density_input_frame_OC1'] = Pxx_1.tolist()
        decomposition_results[vowel]['spectral_density_input_frame_OC2'] = Pxx_2.tolist()
        decomposition_results[vowel]['spectral_density_input_frame_OC3'] = Pxx_3.tolist()

        "Now frame decomposition results"

        signal = sample_data[vowel]['time_domain_signal']
        inputs = feature_extractor(
                    signal, sampling_rate=config.fs, max_length=MAX_LENGTH, truncation=True, padding="max_length"
        )

        decomp_module = DecompositionModule(config)
        mask_indices_seq_length = int(decomp_module._get_feat_extract_output_lengths(MAX_LENGTH))# inputs.input_values[0].shape[0]))

        all_ones_mask = torch.ones((1,mask_indices_seq_length),dtype = torch.bool)
        attention_mask = torch.tensor(np.expand_dims(inputs.attention_mask[0],axis = 0))
        decomposition_outcome, mask_time_indices, _, _, reconstruction_NRMSEs, _, avg_correlogram, correlograms,_, _ , _, _,_ = decomp_module(
                                np.expand_dims(inputs.input_values[0],axis=0),
                                mask_time_indices=all_ones_mask,
                                attention_mask=attention_mask,
                                remove_silence = decomp_args.remove_silence
                            )
        
        frame_decomp = decomposition_outcome["frame"]
        frame_decomp[0] = torch.sum(frame_decomp[1:],dim=0,keepdim=True)
        decomposition_results[vowel]['frame'] = frame_decomp.squeeze(1).detach().numpy().tolist()
        decomposition_results[vowel]['reconstruction_NRMSEs'] = reconstruction_NRMSEs
        decomposition_results[vowel]['avg_correlogram'] = avg_correlogram.detach().numpy().tolist()
        decomposition_results[vowel]['correlograms'] = [c.detach().numpy().tolist() for c in correlograms]
                
        "Now also calculate and save the recovered spectral plots for each component - FRAME"
        f,Pxx_0 = scipy.signal.welch(frame_decomp[0,0,:,:], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        _,Pxx_1 = scipy.signal.welch(frame_decomp[1,0,:,:], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        _,Pxx_2 = scipy.signal.welch(frame_decomp[2,0,:,:], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        _,Pxx_3 = scipy.signal.welch(frame_decomp[3,0,:,:], fs=config.fs, window='hann', nperseg=132, noverlap=10, nfft=config.nfft, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
        decomposition_results[vowel]['spectral_density_X'] = Pxx_0.tolist()
        decomposition_results[vowel]['spectral_density_OC1'] = Pxx_1.tolist()
        decomposition_results[vowel]['spectral_density_OC2'] = Pxx_2.tolist()
        decomposition_results[vowel]['spectral_density_OC3'] = Pxx_3.tolist()
        decomposition_results[vowel]['frequencies'] = f.tolist()

        "Calculate mel spectrograms for the original signal and the decomposed components - FRAME"
        frame_len = frame_decomp.shape[-1]
        mel_features = extract_mel_spectrogram(frame_decomp[0,0,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len)
        decomposition_results[vowel][f'mel_features_X'] = mel_features.tolist()
        for o in range(1,frame_decomp.shape[0]):
            mel_features = extract_mel_spectrogram(frame_decomp[o,0,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len)
            decomposition_results[vowel][f'mel_features_OC{o}'] = mel_features.tolist()
    
    

    "Sequence decomposition"
    decomposition_results['all'] = {}
    signal = []
    for vowel in sample_data.keys():
        s = sample_data[vowel]['time_domain_signal']
        signal.extend(s)   

    signal = np.array(signal)

    "First save original signals - SEQ"
    x = signal.reshape(-1,1).transpose()
    "Standardize x sequence and apply the same transformation to the formants"
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = (x - x_mean) / x_std
    
    signal = []
    for vowel in sample_data.keys():
        s = sample_data[vowel]['formant_1_waves']
        signal.extend(s)   

    signal = np.array(signal)
    f1 = signal.reshape(-1,1).transpose()

    signal = []
    for vowel in sample_data.keys():
        s = sample_data[vowel]['formant_2_waves']
        signal.extend(s)
    
    signal = np.array(signal)
    f2 = signal.reshape(-1,1).transpose()

    signal = []
    for vowel in sample_data.keys():
        s = sample_data[vowel]['formant_3_waves']
        signal.extend(s)
    
    signal = np.array(signal)
    f3 = signal.reshape(-1,1).transpose()

    decomposition_results['all']['input_seq'] = np.concatenate((x,f1,f2,f3),axis=0).tolist()
    

    "Now also calculate and save the initial spectral plots for each formant - FRAME"
    f,Pxx_0 = scipy.signal.welch(decomposition_results['all']['input_seq'][0], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    _,Pxx_1 = scipy.signal.welch(decomposition_results['all']['input_seq'][1], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    _,Pxx_2 = scipy.signal.welch(decomposition_results['all']['input_seq'][2], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    _,Pxx_3 = scipy.signal.welch(decomposition_results['all']['input_seq'][3], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    decomposition_results['all']['spectral_density_input_seq_X'] = Pxx_0.tolist()
    decomposition_results['all']['spectral_density_input_seq_OC1'] = Pxx_1.tolist()
    decomposition_results['all']['spectral_density_input_seq_OC2'] = Pxx_2.tolist()
    decomposition_results['all']['spectral_density_input_seq_OC3'] = Pxx_3.tolist()
    decomposition_results['all']['frequencies'] = f.tolist()


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
    decomposition_outcome, mask_time_indices, _, _, _, reconstruction_NRMSEs_seq, _, _,_,_, _, _ ,_ = decomp_module(
                            np.expand_dims(inputs.input_values[0],axis=0),
                            mask_time_indices=all_ones_mask,
                            attention_mask=attention_mask,
                            remove_silence = decomp_args.remove_silence
                        )
    
    seq_decomp = decomposition_outcome["sequence"] 
    "Standardize according to the x sequence"  
    seq_decomp[0] = torch.sum(seq_decomp[1:],dim=0,keepdim=True)
    seq_decomp[0] = (seq_decomp[0] - seq_decomp[0].mean()) / seq_decomp[0].std()             
    decomposition_results['all']['sequence'] = seq_decomp.squeeze(1).detach().numpy().tolist()
    decomposition_results['all']['reconstruction_NRMSEs_seq'] = reconstruction_NRMSEs_seq
    
    "Now also calculate and save the recovered spectral plots for each component - SEQ"
    f,Pxx_0 = scipy.signal.welch(seq_decomp[0,0,:], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    _,Pxx_1 = scipy.signal.welch(seq_decomp[1,0,:], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    _,Pxx_2 = scipy.signal.welch(seq_decomp[2,0,:], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    _,Pxx_3 = scipy.signal.welch(seq_decomp[3,0,:], fs=config.fs, window='hann', nperseg=2048, noverlap=512, nfft=2048, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean')
    decomposition_results['all']['spectral_density_X_seq'] = Pxx_0.tolist()
    decomposition_results['all']['spectral_density_OC1_seq'] = Pxx_1.tolist()
    decomposition_results['all']['spectral_density_OC2_seq'] = Pxx_2.tolist()
    decomposition_results['all']['spectral_density_OC3_seq'] = Pxx_3.tolist()

    "Calculate mel spectrograms for the original signal and the decomposed components - SEQ"
    mel_features, _ = extract_mel_spectrogram(seq_decomp[0,0,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len)
    decomposition_results['all'][f'mel_features_X_seq'] = mel_features.tolist()
    for o in range(1,seq_decomp.shape[0]):
        mel_features, _ = extract_mel_spectrogram(seq_decomp[o,0,...],config.fs,n_mels=data_training_args.n_mels, n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs), hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops), normalize=data_training_args.mel_norm, feature_length=frame_len)
        decomposition_results['all'][f'mel_features_OC{o}_seq'] = mel_features.tolist()


    with open(os.path.join(SAVE_DIR,f'decomposition_results_NoC{decomp_args.NoC}_{decomp_args.decomp_to_perform}_single_speaker.json'), 'w') as f:
        json.dump(decomposition_results, f)
        
if __name__ == "__main__":
    main()