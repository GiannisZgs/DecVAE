from models import DecompositionModule
import numpy as np
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from feature_extraction import extract_mel_spectrogram, extract_fft_psd

def prepare_pretraining_dataset(batch, feature_extractor, data_training_args, decomp_args, config, max_length):
    """ 
    To be used for Apache Arrow processing in the Dataset.map() function to
    apply processing steps to single sequences (batch) for preparing data for latent traversal visualization.
    This function loads the .arrow format data, interpolates labels to match the network output length,
    preprocesses the audio (normalization, resampling, truncation/padding), performs decomposition using the DecompositionModule,
    and returns all necessary data in batched format per sample-sequence.

    Args:
        batch: A single sample from the .arrow Dataset containing audio and labels.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`): The data training arguments dictionary 
            with necessary data-related parameters.
        decomp_args (:class:`~args_configs.decomp_args.DecompositionArguments`): The decomposition arguments dictionary
            with necessary decomposition model-related parameters; supports various configurations for each different type of decomposition.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object; we need access to some parameters.
        max_length (int): The maximum length of the audio sequences after preprocessing (in number of samples).
    """
    
    sample = batch[data_training_args.audio_column_name]

    if type(sample) == list:
        sample = {'array': np.array(sample),'sampling_rate': decomp_args.fs}
    
    config.dataset_name = data_training_args.dataset_name
    "Initialize Decomposition Module (D_w^C)"
    decomp_module = DecompositionModule(config)
    "Max sequence length at the output of the feature extractor encoder Wav2vec2"
    mask_indices_seq_length = int(decomp_module._get_feat_extract_output_lengths(max_length))# inputs.input_values[0].shape[0]))
    "Start and end of frames"
    frame_len = int(decomp_args.receptive_field*decomp_args.fs)
    frame_stride = int(decomp_args.stride*decomp_args.fs)
    start_indices = np.array([i * frame_stride for i in range(mask_indices_seq_length)])
    stop_indices = start_indices + frame_len

    "Interpolate labels"
    if data_training_args.dataset_name == "timit":
        phonemes39 = batch['phonetic_detail']['utterance39']
        phonemes48 = batch['phonetic_detail']['utterance48']
        start_phonemes = batch['phonetic_detail']['start']
        stop_phonemes = batch['phonetic_detail']['stop']

        "Also load the phoneme id mappings"
        with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
            phoneme39_to_id = json.load(json_file)
        with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'r') as json_file:
            phoneme48_to_id = json.load(json_file)

        "We will need to account for overlap of the network's receptive field"
        "Phoneme labels will need to be interpolated to match the network's output"
        interp_phonemes39 = []
        interp_phonemes48 = []
        overlap_mask = []
        c = 0 #step_phonemes
        phoneme_just_ended = False
        for i in range(mask_indices_seq_length):
            if len(interp_phonemes39) > mask_indices_seq_length:
                pass
            frame_start = start_indices[i]
            frame_stop = stop_indices[i]   
            if i > 0 and c < len(phonemes39)-1:
                if (phoneme_just_ended and(frame_start + frame_len/2 < start_phonemes[c]) and (frame_start > stop_phonemes[c-1])) or (not phoneme_just_ended and frame_start + frame_len/2 < start_phonemes[c+1] and frame_start > stop_phonemes[c] - frame_len/4):
                    interp_phonemes39.append(phoneme39_to_id['sil'])
                    interp_phonemes48.append(phoneme48_to_id['sil'])
                    continue

            if frame_stop-frame_len/2 <= stop_phonemes[c]: 
                phoneme_just_ended = False
                interp_phonemes39.append(phonemes39[c])
                interp_phonemes48.append(phonemes48[c])
            else:
                "End of phoneme, phoneme changes"
                phoneme_just_ended = True
                c+=1
                if c == len(phonemes39):
                    "End of indices or end of utterance"
                    interp_phonemes39.append(phonemes39[c-1])
                    interp_phonemes48.append(phonemes48[c-1])
                    overlap_mask.append(False)
                    break
                else:
                    interp_phonemes39.append(phonemes39[c])
                    interp_phonemes48.append(phonemes48[c])
            if frame_stop >= stop_phonemes[c]:
                if c + 1 >= len(phonemes39):
                    overlap_mask.append(False)
                else:
                    if phonemes39[c] == phonemes39[c+1]:
                        overlap_mask.append(False)
                    else:
                        overlap_mask.append(True)
            else:
                overlap_mask.append(False)
            
        if len(interp_phonemes39) > mask_indices_seq_length:
            raise ValueError("Interpolated phonemes are longer than the sequence length")
        batch['phonemes39'] = np.array(interp_phonemes39)
        batch['phonemes48'] = np.array(interp_phonemes48)
        batch['speaker'] = batch['speaker_id']
        batch['words'] = batch['text']
        batch['start_phonemes'] = np.array(start_phonemes)
        batch['stop_phonemes'] = np.array(stop_phonemes)
        batch['overlap_mask'] = np.array(overlap_mask)
    
    elif data_training_args.dataset_name == "sim_vowels":
        #This is only correct for RFS/stride = 5/4
        vowels_list = batch["vowel"]#.split('_')
        num_vowels = len(vowels_list)
        vowel_sample_len = int(len(sample["array"]) / num_vowels)
        vowel_dur = vowel_sample_len / sample["sampling_rate"]
        #vowel_interp_factor = int(vowel_dur / decomp_args.receptive_field)
        #start_vowels = [i*vowel_sample_len for i in range(num_vowels)]
        stop_vowels = [i*vowel_sample_len + vowel_sample_len for i in range(num_vowels)]
        vowels_interp = []
        overlap_mask = []
        c = 0 #step_phonemes
        phoneme_just_ended = False
        for i in range(mask_indices_seq_length):
            frame_start = start_indices[i]
            frame_stop = stop_indices[i]   
            if frame_stop-frame_len/2 <= stop_vowels[c]: 
                phoneme_just_ended = False
                vowels_interp.append(vowels_list[c])
            else:
                "End of phoneme, phoneme changes"
                phoneme_just_ended = True
                c+=1
                if c == len(vowels_list):
                    "End of indices or end of utterance"
                    vowels_interp.append(vowels_list[c-1])
                    overlap_mask.append(False)
                    break
                else:
                    vowels_interp.append(vowels_list[c])
            if frame_stop >= stop_vowels[c]:
                if vowels_list[c] == vowels_list[c+1]:
                    overlap_mask.append(False)
                else:
                    overlap_mask.append(True)
            else:
                overlap_mask.append(False)

        batch["vowel_labels"] = np.array((vowels_interp))
        batch["speaker_vt_factor"] = batch["speaker_vocal_tract_factor"]
        batch['overlap_mask'] = np.array(overlap_mask)
    
    elif data_training_args.dataset_name == "VOC_ALS":
        "A single audio file has the same labels"
        "Interpolate the labels to be frame-wise"
        def truncate_from_middle(audio_array, sampling_rate, seconds_to_skip, max_length=None):
            """Truncate audio array by skipping the first 1 second"""
            # Calculate how many samples to skip (of length seconds_to_skip)
            samples_to_skip = int(seconds_to_skip*sampling_rate) 
            
            # If audio is shorter than 1 second, return empty or minimal audio
            if len(audio_array) <= samples_to_skip:
                return audio_array # Return minimal audio
            
            # Skip first second
            truncated_audio = audio_array[samples_to_skip:]
            
            # If max_length is specified, further truncate from the end
            if max_length is not None and len(truncated_audio) > max_length:
                truncated_audio = truncated_audio[:max_length]
                
            return truncated_audio
        
        sample["array"] = truncate_from_middle(
            sample["array"], 
            sampling_rate=sample["sampling_rate"],
            seconds_to_skip=data_training_args.skip_first_n_seconds,
            max_length=max_length
        )

        "Also encode healty control labels from None or - to a number e.g. -100"
        if batch['alsfrs_total'] is None:
            batch['alsfrs_total'] = -1
        if batch['disease_duration'] is None:
            batch['disease_duration'] = -1   
        if batch['king_stage'] is None:
            batch['king_stage'] = -1         
        if batch['alsfrs_speech'] == '-':
            batch['alsfrs_speech'] = -1 
        
        batch['alsfrs_total_enc'] = batch['alsfrs_total']
        batch['disease_duration_enc'] = batch['disease_duration']
        batch['king_stage_enc'] = batch['king_stage']
        batch['alsfrs_speech'] = int(batch['alsfrs_speech']) 
        batch['alsfrs_speech_enc'] = batch['alsfrs_speech'] #.astype(str)
        batch["cantagallo_enc"] = batch["cantagallo"]
        batch["phonemes"] = batch["phoneme_encoded"]
        batch["speaker"] = batch["speaker_id_encoded"]
        batch["group"] = batch["category_encoded"]
    
    elif data_training_args.dataset_name == "iemocap":

        phonemes = []
        start_phonemes = []
        stop_phonemes = []
        for ph in batch['phonemes_dict']:
            phonemes.append(ph['phoneme'])
            start_phonemes.append(int(ph['start']*sample["sampling_rate"]))
            stop_phonemes.append(int(ph['end']*sample["sampling_rate"]))

        "Also load the phoneme id, emotion and speaker mappings"
        with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
            phoneme_to_id = json.load(json_file)

        with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'r') as json_file:
            emotion_to_id = json.load(json_file)

        with open(data_training_args.path_to_iemocap_speaker_dict_file, 'r') as json_file:
            speaker_to_id = json.load(json_file)

        "We will need to account for overlap of the network's receptive field"
        "Phoneme labels will need to be interpolated to match the network's output"
        interp_phonemes = []
        overlap_mask = []
        c = 0 #step_phonemes
        phoneme_just_ended = False
        for i in range(mask_indices_seq_length):
            if len(interp_phonemes) > mask_indices_seq_length:
                pass
            frame_start = start_indices[i]
            #print(frame_start)
            frame_stop = stop_indices[i]   
            if i > 0 and c < len(phonemes)-1:
                if (phoneme_just_ended and(frame_start + frame_len/2 < start_phonemes[c]) and (frame_start > stop_phonemes[c-1])) or (not phoneme_just_ended and frame_start + frame_len/2 < start_phonemes[c+1] and frame_start > stop_phonemes[c] - frame_len/4):
                    interp_phonemes.append(phoneme_to_id['sil'])
                    continue
            
            if frame_start + 3*frame_len/4 < start_phonemes[c]: 
                "Phoneme is only partly inside this frame - This is most likely the first frame"
                phoneme_just_ended = False
                interp_phonemes.append(phoneme_to_id['SIL'])
                overlap_mask.append(False)
                continue

            if frame_stop-frame_len/2 <= stop_phonemes[c]:
                phoneme_just_ended = False
                interp_phonemes.append(phoneme_to_id[phonemes[c]])
            else:
                "End of phoneme, phoneme changes"
                phoneme_just_ended = True
                c+=1
                if c == len(phonemes):
                    "End of indices or end of utterance"
                    if not frame_start >= stop_phonemes[c-1]:
                        interp_phonemes.append(phoneme_to_id[phonemes[c-1]])
                        overlap_mask.append(False)
                    break
                else:
                    interp_phonemes.append(phoneme_to_id[phonemes[c]])

            if frame_stop >= stop_phonemes[c]:
                if c + 1 >= len(phonemes):
                    overlap_mask.append(False)
                else:
                    if phonemes[c] == phonemes[c+1]:
                        overlap_mask.append(False)
                    else:
                        overlap_mask.append(True)
            else:
                overlap_mask.append(False)
            
        if len(interp_phonemes) > mask_indices_seq_length:
            raise ValueError("Interpolated phonemes are longer than the sequence length")
        batch['phonemes'] = np.array(interp_phonemes)
        batch['speaker'] = speaker_to_id[batch['speaker_id']]
        batch['emotion_labels'] = emotion_to_id[batch['emotion']]
        batch['start_phonemes'] = np.array(start_phonemes)
        batch['stop_phonemes'] = np.array(stop_phonemes)
        batch['overlap_mask'] = np.array(overlap_mask)
  
    "Pre-process (normalize and resample if needed) and enforce max length (truncation or padding)"
    inputs = feature_extractor(
        sample["array"], sampling_rate=sample["sampling_rate"], max_length=max_length, truncation=True, padding="max_length"
    )
    
    for frame in inputs["input_values"]:
        if (frame != frame).any():
            print("Nan in input_values")
    attention_mask = torch.tensor(np.expand_dims(inputs.attention_mask[0],axis = 0))
    
    "Perform the decomposition for all frames, then mask (use) in latent space according to mask_time_indices"
    all_ones_mask = torch.ones((1,mask_indices_seq_length),dtype = torch.bool)
    if data_training_args.dataset_name in ["timit", "VOC_ALS", "iemocap"]:
        "Zero padded frames need not be decomposed - Use the masks for those"
        decomposition_outcome, _, _, _, _, _, _, _, _, _ ,_, _,_= decomp_module(
            np.expand_dims(inputs.input_values[0],axis=0),
            mask_time_indices=all_ones_mask,
            attention_mask=attention_mask,
            remove_silence = decomp_args.remove_silence
        )
    elif data_training_args.dataset_name in ["sim_vowels"]:
        decomposition_outcome = decomp_module(
            np.expand_dims(inputs.input_values[0],axis=0),
            mask_time_indices=all_ones_mask,
            attention_mask=attention_mask,
            remove_silence = decomp_args.remove_silence
        )[0]

    if decomp_args.frame_decomp and decomposition_outcome["frame"] is not None:
        frame_decomp = decomposition_outcome["frame"]
        batch["input_values"] = frame_decomp.squeeze(1)
    if decomp_args.seq_decomp and decomposition_outcome["sequence"] is not None:
        seq_decomp = decomposition_outcome["sequence"]                
        batch["input_seq_values"] = seq_decomp.squeeze(-2)
    batch["input_length"] = max_length 
    batch["attention_mask"] = attention_mask

    return batch


def prepare_extract_features_pretraining_dataset(batch, feature_extractor, data_training_args, decomp_args, config, max_length):
    """ 
    To be used for Apache Arrow processing in the Dataset.map() function to
    apply processing steps to single sequences (batch) for preparing data for latent traversal visualization.
    This function loads the .arrow format data, interpolates labels to match the network output length,
    preprocesses the audio (normalization, resampling, truncation/padding), performs decomposition using the DecompositionModule,
    extracts mel filterbank or FFT features (optional), and returns all necessary data in batched format per sample-sequence.

    Args:
        batch: A single sample from the .arrow Dataset containing audio and labels.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`): The data training arguments dictionary 
            with necessary data-related parameters.
        decomp_args (:class:`~args_configs.decomp_args.DecompositionArguments`): The decomposition arguments dictionary
            with necessary decomposition model-related parameters; supports various configurations for each different type of decomposition.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object; we need access to some parameters.
        max_length (int): The maximum length of the audio sequences after preprocessing (in number of samples).
    """
    
    sample = batch[data_training_args.audio_column_name]

    if type(sample) == list:
        sample = {'array': np.array(sample),'sampling_rate': decomp_args.fs}
    
    config.dataset_name = data_training_args.dataset_name
    "Initialize Decomposition Module (D_w^C)"
    decomp_module = DecompositionModule(config)
    "Max sequence length at the output of the feature extractor encoder Wav2vec2"
    mask_indices_seq_length = int(decomp_module._get_feat_extract_output_lengths(max_length))# inputs.input_values[0].shape[0]))
    "Start and end of frames"
    frame_len = int(decomp_args.receptive_field*decomp_args.fs)
    frame_stride = int(decomp_args.stride*decomp_args.fs)
    start_indices = np.array([i * frame_stride for i in range(mask_indices_seq_length)])
    stop_indices = start_indices + frame_len

    "Interpolate labels"
    if data_training_args.dataset_name == "timit":
        phonemes39 = batch['phonetic_detail']['utterance39']
        phonemes48 = batch['phonetic_detail']['utterance48']
        start_phonemes = batch['phonetic_detail']['start']
        stop_phonemes = batch['phonetic_detail']['stop']

        "Also load the phoneme id mappings"
        with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
            phoneme39_to_id = json.load(json_file)
        with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'r') as json_file:
            phoneme48_to_id = json.load(json_file)

        "We will need to account for overlap of the network's receptive field"
        "Phoneme labels will need to be interpolated to match the network's output"
        interp_phonemes39 = []
        interp_phonemes48 = []
        overlap_mask = []
        c = 0 #step_phonemes
        phoneme_just_ended = False
        for i in range(mask_indices_seq_length):
            if len(interp_phonemes39) > mask_indices_seq_length:
                pass
            frame_start = start_indices[i]
            frame_stop = stop_indices[i]   
            if i > 0 and c < len(phonemes39)-1:
                if (phoneme_just_ended and(frame_start + frame_len/2 < start_phonemes[c]) and (frame_start > stop_phonemes[c-1])) or (not phoneme_just_ended and frame_start + frame_len/2 < start_phonemes[c+1] and frame_start > stop_phonemes[c] - frame_len/4):
                    interp_phonemes39.append(phoneme39_to_id['sil'])
                    interp_phonemes48.append(phoneme48_to_id['sil'])
                    continue

            if frame_stop-frame_len/2 <= stop_phonemes[c]: 
                phoneme_just_ended = False
                interp_phonemes39.append(phonemes39[c])
                interp_phonemes48.append(phonemes48[c])
            else:
                "End of phoneme, phoneme changes"
                phoneme_just_ended = True
                c+=1
                if c == len(phonemes39):
                    "End of indices or end of utterance"
                    interp_phonemes39.append(phonemes39[c-1])
                    interp_phonemes48.append(phonemes48[c-1])
                    overlap_mask.append(False)
                    break
                else:
                    interp_phonemes39.append(phonemes39[c])
                    interp_phonemes48.append(phonemes48[c])
            if frame_stop >= stop_phonemes[c]:
                if c + 1 >= len(phonemes39):
                    overlap_mask.append(False)
                else:
                    if phonemes39[c] == phonemes39[c+1]:
                        overlap_mask.append(False)
                    else:
                        overlap_mask.append(True)
            else:
                overlap_mask.append(False)
            
        if len(interp_phonemes39) > mask_indices_seq_length:
            raise ValueError("Interpolated phonemes are longer than the sequence length")
        batch['phonemes39'] = np.array(interp_phonemes39)
        batch['phonemes48'] = np.array(interp_phonemes48)
        batch['speaker'] = batch['speaker_id']
        batch['words'] = batch['text']
        batch['start_phonemes'] = np.array(start_phonemes)
        batch['stop_phonemes'] = np.array(stop_phonemes)
        batch['overlap_mask'] = np.array(overlap_mask)
    
    elif data_training_args.dataset_name == "sim_vowels":
        #This is only correct for RFS/stride = 5/4
        vowels_list = batch["vowel"]#.split('_')
        num_vowels = len(vowels_list)
        vowel_sample_len = int(len(sample["array"]) / num_vowels)
        vowel_dur = vowel_sample_len / sample["sampling_rate"]
        #vowel_interp_factor = int(vowel_dur / decomp_args.receptive_field)
        #start_vowels = [i*vowel_sample_len for i in range(num_vowels)]
        stop_vowels = [i*vowel_sample_len + vowel_sample_len for i in range(num_vowels)]
        vowels_interp = []
        overlap_mask = []
        c = 0 #step_phonemes
        phoneme_just_ended = False
        for i in range(mask_indices_seq_length):
            frame_start = start_indices[i]
            frame_stop = stop_indices[i]   
            if frame_stop-frame_len/2 <= stop_vowels[c]: 
                phoneme_just_ended = False
                vowels_interp.append(vowels_list[c])
            else:
                "End of phoneme, phoneme changes"
                phoneme_just_ended = True
                c+=1
                if c == len(vowels_list):
                    "End of indices or end of utterance"
                    vowels_interp.append(vowels_list[c-1])
                    overlap_mask.append(False)
                    break
                else:
                    vowels_interp.append(vowels_list[c])
            if frame_stop >= stop_vowels[c]:
                if vowels_list[c] == vowels_list[c+1]:
                    overlap_mask.append(False)
                else:
                    overlap_mask.append(True)
            else:
                overlap_mask.append(False)

        batch["vowel_labels"] = np.array((vowels_interp))
        batch["speaker_vt_factor"] = batch["speaker_vocal_tract_factor"]
        batch['overlap_mask'] = np.array(overlap_mask)
    
    elif data_training_args.dataset_name == "VOC_ALS":
        "A single audio file has the same labels"
        "Interpolate the labels to be frame-wise"
        def truncate_from_middle(audio_array, sampling_rate, seconds_to_skip, max_length=None):
            """Truncate audio array by skipping the first 1 second"""
            # Calculate how many samples to skip (of length seconds_to_skip)
            samples_to_skip = int(seconds_to_skip*sampling_rate) 
            
            # If audio is shorter than 1 second, return empty or minimal audio
            if len(audio_array) <= samples_to_skip:
                return audio_array # Return minimal audio
            
            # Skip first second
            truncated_audio = audio_array[samples_to_skip:]
            
            # If max_length is specified, further truncate from the end
            if max_length is not None and len(truncated_audio) > max_length:
                truncated_audio = truncated_audio[:max_length]
                
            return truncated_audio
        
        sample["array"] = truncate_from_middle(
            sample["array"], 
            sampling_rate=sample["sampling_rate"],
            seconds_to_skip=data_training_args.skip_first_n_seconds,
            max_length=max_length
        )

        "Also encode healty control labels from None or - to a number e.g. -100"
        if batch['alsfrs_total'] is None:
            batch['alsfrs_total'] = -1
        if batch['disease_duration'] is None:
            batch['disease_duration'] = -1   
        if batch['king_stage'] is None:
            batch['king_stage'] = -1         
        if batch['alsfrs_speech'] == '-':
            batch['alsfrs_speech'] = -1 
        
        batch['alsfrs_total_enc'] = batch['alsfrs_total']
        batch['disease_duration_enc'] = batch['disease_duration']
        batch['king_stage_enc'] = batch['king_stage']
        batch['alsfrs_speech'] = int(batch['alsfrs_speech']) 
        batch['alsfrs_speech_enc'] = batch['alsfrs_speech'] #.astype(str)
        batch["cantagallo_enc"] = batch["cantagallo"]
        batch["phonemes"] = batch["phoneme_encoded"]
        batch["speaker"] = batch["speaker_id_encoded"]
        batch["group"] = batch["category_encoded"]
    
    elif data_training_args.dataset_name == "iemocap":

        phonemes = []
        start_phonemes = []
        stop_phonemes = []
        for ph in batch['phonemes_dict']:
            phonemes.append(ph['phoneme'])
            start_phonemes.append(int(ph['start']*sample["sampling_rate"]))
            stop_phonemes.append(int(ph['end']*sample["sampling_rate"]))

        "Also load the phoneme id, emotion and speaker mappings"
        with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
            phoneme_to_id = json.load(json_file)

        with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'r') as json_file:
            emotion_to_id = json.load(json_file)

        with open(data_training_args.path_to_iemocap_speaker_dict_file, 'r') as json_file:
            speaker_to_id = json.load(json_file)

        "We will need to account for overlap of the network's receptive field"
        "Phoneme labels will need to be interpolated to match the network's output"
        interp_phonemes = []
        overlap_mask = []
        c = 0 #step_phonemes
        phoneme_just_ended = False
        for i in range(mask_indices_seq_length):
            if len(interp_phonemes) > mask_indices_seq_length:
                pass
            frame_start = start_indices[i]
            #print(frame_start)
            frame_stop = stop_indices[i]   
            if i > 0 and c < len(phonemes)-1:
                if (phoneme_just_ended and(frame_start + frame_len/2 < start_phonemes[c]) and (frame_start > stop_phonemes[c-1])) or (not phoneme_just_ended and frame_start + frame_len/2 < start_phonemes[c+1] and frame_start > stop_phonemes[c] - frame_len/4):
                    interp_phonemes.append(phoneme_to_id['sil'])
                    continue
            
            if frame_start + 3*frame_len/4 < start_phonemes[c]: 
                "Phoneme is only partly inside this frame - This is most likely the first frame"
                phoneme_just_ended = False
                interp_phonemes.append(phoneme_to_id['SIL'])
                overlap_mask.append(False)
                continue

            if frame_stop-frame_len/2 <= stop_phonemes[c]:
                phoneme_just_ended = False
                interp_phonemes.append(phoneme_to_id[phonemes[c]])
            else:
                "End of phoneme, phoneme changes"
                phoneme_just_ended = True
                c+=1
                if c == len(phonemes):
                    "End of indices or end of utterance"
                    if not frame_start >= stop_phonemes[c-1]:
                        interp_phonemes.append(phoneme_to_id[phonemes[c-1]])
                        overlap_mask.append(False)
                    break
                else:
                    interp_phonemes.append(phoneme_to_id[phonemes[c]])

            if frame_stop >= stop_phonemes[c]:
                if c + 1 >= len(phonemes):
                    overlap_mask.append(False)
                else:
                    if phonemes[c] == phonemes[c+1]:
                        overlap_mask.append(False)
                    else:
                        overlap_mask.append(True)
            else:
                overlap_mask.append(False)
            
        if len(interp_phonemes) > mask_indices_seq_length:
            raise ValueError("Interpolated phonemes are longer than the sequence length")
        batch['phonemes'] = np.array(interp_phonemes)
        batch['speaker'] = speaker_to_id[batch['speaker_id']]
        batch['emotion_labels'] = emotion_to_id[batch['emotion']]
        batch['start_phonemes'] = np.array(start_phonemes)
        batch['stop_phonemes'] = np.array(stop_phonemes)
        batch['overlap_mask'] = np.array(overlap_mask)
  
    "Pre-process (normalize and resample if needed) and enforce max length (truncation or padding)"
    inputs = feature_extractor(
        sample["array"], sampling_rate=sample["sampling_rate"], max_length=max_length, truncation=True, padding="max_length"
    )
    
    for frame in inputs["input_values"]:
        if (frame != frame).any():
            print("Nan in input_values")
    attention_mask = torch.tensor(np.expand_dims(inputs.attention_mask[0],axis = 0))
    
    "Perform the decomposition for all frames, then mask (use) in latent space according to mask_time_indices"
    all_ones_mask = torch.ones((1,mask_indices_seq_length),dtype = torch.bool)
    if data_training_args.dataset_name in ["timit", "VOC_ALS", "iemocap"]:
        "Zero padded frames need not be decomposed - Use the masks for those"
        decomposition_outcome, _, _, _, _, _, _, _, _, _ ,_, _,_= decomp_module(
            np.expand_dims(inputs.input_values[0],axis=0),
            mask_time_indices=all_ones_mask,
            attention_mask=attention_mask,
            remove_silence = decomp_args.remove_silence
        )
    elif data_training_args.dataset_name in ["sim_vowels"]:
        decomposition_outcome = decomp_module(
            np.expand_dims(inputs.input_values[0],axis=0),
            mask_time_indices=all_ones_mask,
            attention_mask=attention_mask,
            remove_silence = decomp_args.remove_silence
        )[0]

    if decomp_args.frame_decomp and decomposition_outcome["frame"] is not None:
        frame_decomp = decomposition_outcome["frame"]
        batch["input_values"] = frame_decomp.squeeze(1)
        batch["input_values"] = batch["input_values"].unsqueeze(0)
    if decomp_args.seq_decomp and decomposition_outcome["sequence"] is not None:
        seq_decomp = decomposition_outcome["sequence"]                
        batch["input_seq_values"] = seq_decomp.squeeze(-2)
        batch["input_seq_values"] = batch["input_seq_values"].unsqueeze(0)
    batch["input_length"] = max_length 
    batch["attention_mask"] = attention_mask

    "Extract features"
    device = batch["input_values"].device
    batch_size = 1
    
    "If input will be mel filterbank features, split sequence in frames"
    if data_training_args.input_type == 'mel':
        frame_len = batch["input_values"].shape[-1]
        if batch.get("input_seq_values") is not None:
            frames = batch["input_seq_values"].shape[-1]/frame_len
            new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames),frame_len),device = batch["input_seq_values"].device)
        
            # Split sequence into frames 
            for o in range(batch["input_values"].shape[1]):
                sequence = batch["input_seq_values"][:,o,:].clone()
                for f in range(int(frames)):
                    framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                    new_input_seq_values[:,o,f,:] = framed_sequence.clone()
            batch["input_seq_values"] = new_input_seq_values.clone()
    
        if len(config.conv_kernel) == 7:
            assert data_training_args.mel_hops == 4
        elif len(config.conv_kernel) == 5:
            assert data_training_args.mel_hops == 3
        "Store the per-utterance peaks unclamped, so the collator can restore the batch reference"
        mel_spec_max = []
        mel_seq_spec_max = []
        for o in range(batch["input_values"].shape[1]):
            batch["input_values"][:,o,...], spec_max = extract_mel_spectrogram(
                    batch["input_values"][:,o,...],
                    config.fs,
                    n_mels=data_training_args.n_mels,
                    n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs),
                    hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops),
                    normalize=None,
                    feature_length=frame_len,
                    top_db=None
                )
            mel_spec_max.append(float(spec_max))

            if batch.get("input_seq_values") is not None:
                batch["input_seq_values"][:,o,...], seq_spec_max = extract_mel_spectrogram(
                    batch["input_seq_values"][:,o,...],
                    config.fs,
                    n_mels=data_training_args.n_mels,
                    n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs),
                    hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops),
                    normalize=None,
                    feature_length=frame_len,
                    top_db=None
                )
                mel_seq_spec_max.append(float(seq_spec_max))

        batch["mel_spec_max"] = np.array(mel_spec_max, dtype=np.float32)
        if len(mel_seq_spec_max) > 0:
            batch["mel_seq_spec_max"] = np.array(mel_seq_spec_max, dtype=np.float32)

        if batch.get("input_seq_values") is not None:
            "Flatten sequence - Reverse framing"
            batch["input_seq_values"] = batch["input_seq_values"].reshape(batch["input_seq_values"].shape[0],batch["input_seq_values"].shape[1],-1)

    elif data_training_args.input_type == 'fft':                               
        # Apply fft using welch's power spectral density estimation
        batch["input_values"], batch["input_seq_values"] = extract_fft_psd(
            batch, 
            normalize=True, #self.data_training_args.mel_norm,
            device=batch["input_values"].device
        )
                                

    return batch

def prepare_extract_features_vae_pretraining_dataset(batch, feature_extractor, model_args, data_training_args, decomp_args, config, max_length):
    """
    To be used for Apache Arrow processing in the Dataset.map() function, for VAE-based models.
    Performs the same steps as prepare_pretraining_dataset and additionally extracts the mel
    filterbank features that were previously extracted inside the VAE training loop. The features
    are left un-normalized here; normalization happens in the collator, which sees the whole batch.
    VAEs operate on a single signal, so the component axis is dropped.

    Args:
        batch: A single sample from the .arrow Dataset containing audio and labels.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        model_args (:class:`~args_configs.model_args.ModelArguments`): The model arguments dictionary;
            the VAE type and its mel settings (n_mels_vae, mel_norm_vae) are needed here.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`): The data training arguments dictionary
            with necessary data-related parameters.
        decomp_args (:class:`~args_configs.decomp_args.DecompositionArguments`): The decomposition arguments dictionary
            with necessary decomposition model-related parameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object.
        max_length (int): The maximum length of the audio sequences after preprocessing (in number of samples).
    """

    batch = prepare_pretraining_dataset(batch, feature_extractor, data_training_args, decomp_args, config, max_length)

    if model_args.vae_input_type.startswith("waveform"):
        "Nothing to extract - the components are kept, the collator selects the one it needs"
        return batch

    if model_args.vae_input_type == "fft":
        "Welch PSD normalizes each component of each utterance on its own, so extracting here"
        "gives the same result as extracting on the whole batch"
        if model_args.vae_type == "VAE_1D_FC_seq":
            "The collator used to take the first component of the sequence - do it here instead"
            fft_batch = {"input_values": batch["input_seq_values"][0].unsqueeze(0)}
            values, _ = extract_fft_psd(fft_batch, normalize=True, device=fft_batch["input_values"].device)
            batch["input_values"] = values.squeeze(0)
            batch.pop("input_seq_values", None)
        elif model_args.vae_type == "VAE_1D_FC":
            fft_batch = {"input_values": batch["input_values"].unsqueeze(0)}
            values, _ = extract_fft_psd(fft_batch, normalize=True, device=fft_batch["input_values"].device)
            batch["input_values"] = values.squeeze(0)
        "The remaining model types consume the raw sequence, as they did when extraction happened"
        "in the training loop - there was no fft branch for them there either"
        return batch

    if not model_args.vae_input_type.startswith('mel'):
        return batch

    if model_args.vae_type == "VAE_1D_FC":
        "Extract every component - the ICA/PCA evaluations read the components without the original"
        "signal ('mel_ocs') or with it ('mel_all'), while training selects the original signal alone"
        frame_len = batch["input_values"].shape[-1]
        components, spec_max = [], []
        for o in range(batch["input_values"].shape[0]):
            features, component_max = extract_mel_spectrogram(
                batch["input_values"][o].unsqueeze(0),
                config.fs,
                n_mels=model_args.n_mels_vae,
                n_fft=int(config.receptive_field*config.fs),
                hop_length=int(config.receptive_field*config.fs) + 1,
                normalize=None,
                feature_length=frame_len,
                top_db=None
            )
            components.append(features.squeeze(0))
            spec_max.append(float(component_max))
        batch["input_values"] = torch.stack(components)
        batch["mel_spec_max"] = np.array(spec_max, dtype=np.float32)

    elif model_args.vae_type == "VAE_1D_FC_seq":
        "The collator used to take the first component of the sequence - do it here instead"
        sequence = batch["input_seq_values"][0]
        frame_len = int(sequence.shape[-1]/10)
        frames = int(sequence.shape[-1]/frame_len)
        framed_sequence = torch.zeros((frames, frame_len), device=sequence.device)
        for f in range(frames):
            framed_sequence[f, :] = sequence[f*frame_len:(f+1)*frame_len].clone()

        "Keep the features untruncated - the collator normalizes before cutting to frame_len,"
        "otherwise the mel axis can no longer be recovered from the flattened features"
        features, spec_max = extract_mel_spectrogram(
            framed_sequence.unsqueeze(0),
            config.fs,
            n_mels=model_args.n_mels_vae,
            n_fft=int(data_training_args.mel_hops*config.receptive_field*config.fs),
            hop_length=int(((config.receptive_field*config.fs) + 1)/data_training_args.mel_hops),
            normalize=None,
            feature_length=None,
            top_db=None
        )
        batch["input_values"] = features.squeeze(0)
        batch["mel_spec_max"] = np.float32(spec_max)
        batch["mel_frame_len"] = np.int64(frame_len)
        batch.pop("input_seq_values", None)

    return batch
