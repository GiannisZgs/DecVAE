from models import DecompositionModule
import numpy as np
import json
import torch

def prepare_traversal_dataset(batch, feature_extractor, data_training_args, decomp_args, config, max_length):
    """ 
    To be used for Apache Arrow processing in the Dataset.map() function to
    apply processing steps to single sequences (batch) for preparing data for latent traversal visualization.
    This function loads the .arrow format data, interpolates labels to match the network output length,
    preprocesses the audio (normalization, resampling, truncation/padding), performs decomposition using the DecompositionModule (D_w^C),
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

    #Get audio data
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
        "Below loop interpolates labels and keeps track of segments with overlapping labels"
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
        
    elif "vowels" in data_training_args.dataset_name:
        #This is only correct for RFS/stride = 5/4
        vowels_list = batch["vowel"]
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
            # Calculate how many samples to skip (equal to seconds_to_skip)
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
            #print(frame_stop)
            #if frame_start == 40960:
            #    pass 
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
        #assert len(interp_phonemes39) == mask_indices_seq_length 
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
