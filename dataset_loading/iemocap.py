"""
Dataset loading and initial processing for the IEMOCAP dataset for different scenarios.
Returns DatasetDict objects for training, validation, and testing.
"""

from datasets import Dataset, DatasetDict, concatenate_datasets
import os
import glob
import re
import soundfile as sf
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
from sklearn.model_selection import StratifiedGroupKFold

"Some instances from the phonemes class to use for visualizations"
SEL_PHONEMES_LIST_IEMOCAP  = ['IY', 'EY', ' AY', 'OW', 'UH', 'UW', 'B', 'D', 'F', 'K', 'L', 'S']
SEL_NON_VERBAL_PHONEMES_LIST_IEMOCAP = ['+BREATHING+', "+LIPSMACK", "+LAUGHTER+"]
SEL_VOWELS_LIST_IEMOCAP = ['IY', 'EY', ' AY', 'OW', 'UH', 'UW']
MIN_ACCEPTABLE_LENGTH = 2000 
TARGET_PER_EMOTION = 10
TARGET_PER_COMBINATION = 3  # Collect at least 3 examples per combination


def load_iemocap(data_training_args):
    """
    Process and load the IEMOCAP dataset directly without saving intermediate files.
    
    Args:
        data_training_args: Arguments for data processing including:
            - dataset_path: Path to IEMOCAP_full_release directory
            - seed: Random seed for shuffling
            - split_by: How to split data ("session" or "speaker")
            - emotions: Comma-separated list of emotions to include (optional)
            - max_duration: Maximum audio duration in seconds (optional)
            
    Returns:
        DatasetDict containing train, validation, and test datasets
    """
    base_dir = data_training_args.data_dir
    print(f"Processing IEMOCAP dataset from {base_dir}")
    
    # Dictionary to store all extracted data
    all_data = []
    all_phonemes = []
    # Iterate through all sessions
    for session_idx in range(1, 6):
        session_name = f"Session{session_idx}"
        session_path = os.path.join(base_dir, session_name)
        
        print(f"Processing {session_name}...")
        
        # Get all WAV files in the sentences/wav directory
        wav_dir = os.path.join(session_path, "sentences", "wav")
        wav_files = glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)
        
        # Process each WAV file
        for wav_file in tqdm(wav_files, desc=f"Processing {session_name} files"):
            # Extract file identifiers
            file_name = os.path.basename(wav_file)
            file_id = os.path.splitext(file_name)[0]  # e.g., Ses01F_impro01_F000 or Ses01F_impro01_1_F000
            
            # Extract the conversation style and handle different formats
            parts = file_id.split('_')
            speaker_id = parts[0]  # e.g., Ses01F
            gender = speaker_id[-1]  # F or M
            if 'impro' in parts[1]:
                conv_style = 'impro'
            elif 'script' in parts[1]:
                conv_style = 'script'
            # Handle different conversation style formats
            if len(parts) >= 3 and parts[2][0].isdigit():
                # Format: Ses01F_impro01_1_F000 or Ses01F_script01_2_M000
                sentence_id = '_'.join(parts[:3])  # e.g., Ses01F_impro01_1
            else:
                # Standard format: Ses01F_impro01_F000
                sentence_id = '_'.join(parts[:2])  # e.g., Ses01F_impro01
            
            
            # Get audio data
            try:
                audio_data, sample_rate = sf.read(wav_file)
                duration = len(audio_data) / sample_rate
                # Store audio data as numpy array
                audio_array = audio_data
            except Exception as e:
                print(f"Error loading audio file {wav_file}: {e}")
                continue
            
            # Get phoneme segmentation from forced alignment
            phseg_file = os.path.join(session_path, "sentences", "ForcedAlignment", 
                                     sentence_id, file_id + ".phseg")
            phonemes_dict = []
            try:
                if os.path.exists(phseg_file):
                    with open(phseg_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip() and not line.startswith('\t SFrm'):
                                parts = line.strip().split()
                                if len(parts) >= 4:
                                    try:
                                        # Transform to seconds
                                        start_time = (float(parts[0]) + 2) / 100
                                        end_time = (float(parts[1]) + 2) / 100
                                        primary_phone = parts[3]
                                        all_phonemes.append(primary_phone)
                                        phonemes_dict.append({
                                            'start': start_time,
                                            'end': end_time,
                                            'phoneme': primary_phone
                                        })
                                    except:
                                        continue
                else:
                    print(f"Phoneme file not found: {phseg_file}")
                    continue
            except Exception as e:
                print(f"Error loading phoneme file {phseg_file}: {e}")
                phonemes_dict = []
                continue
            
            # Get emotion label from evaluation file
            emotion_file = os.path.join(session_path, "dialog", "EmoEvaluation", "Majority", 
                                        f"{sentence_id}.txt")
            emotion = "unknown"
            valence = arousal = dominance = 0.0
            
            try:
                if os.path.exists(emotion_file):
                    with open(emotion_file, 'r') as f:
                        content = f.read()
                        # Find the specific utterance in the file
                        for line in content.split('\n'):
                            if file_id in line and not line.startswith('%') and line.startswith('['):
                                # Extract categorical emotion and dimensional ratings
                                parts = line.strip().split('\t')
                                if len(parts) >= 3:
                                    emotion = parts[2]
                                    "Emotion xxx is two or more emotions"
                                    # Try to extract VAD values if available
                                    if len(parts) >= 4 and '[' in parts[3] and ']' in parts[3]:
                                        try:
                                            vad_str = parts[3].strip('[]')
                                            vad_values = [float(x.strip()) for x in vad_str.split(',')]
                                            if len(vad_values) >= 3:
                                                valence = vad_values[0]
                                                arousal = vad_values[1]
                                                dominance = vad_values[2]
                                        except:
                                            pass
                                break
                else:
                    print(f"Emotion file not found: {emotion_file}")
                    continue
            except Exception as e:
                print(f"Error loading emotion file {emotion_file}: {e}")
                continue
            
            # Create a data entry for this file
            entry = {
                'file_id': file_id,
                'session': session_name,
                'sentence_id': sentence_id,
                'speaker_id': speaker_id,
                'conv_style': conv_style,
                'gender': gender,
                'path': wav_file,
                'audio': audio_array,
                'sampling_rate': sample_rate,
                'duration': duration,
                'emotion': emotion,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'phonemes_dict': phonemes_dict
            }

            all_data.append(entry)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_data)

    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)
    # Filter out all emotions except 5
    df = df[df['emotion'].isin(['hap', 'sad', 'ang', 'exc', 'neu'])]
    #Distribution of emotions: total of 5531
    # Neu 1708, Hap + Exc: 1636 , Sad: 1084, Ang: 1103
    df['emotion'] = df['emotion'].replace({'exc': 'hap'})

    "Save encodings to .json dictionaries to be used in evaluations"
    "Create mappings to integers"
    unique_phonemes = np.unique(all_phonemes)
    unique_emotions = df['emotion'].unique()
    unique_speakers = df['speaker_id'].unique()
    if not os.path.exists(data_training_args.path_to_iemocap_phoneme_to_id_file):
        phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(sorted(unique_phonemes))}
        with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'w') as json_file:
            json.dump(phoneme_to_id, json_file, indent=4)
    
    if not os.path.exists(data_training_args.path_to_iemocap_emotion_to_id_file):
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(sorted(unique_emotions))}
        with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'w') as json_file:
            json.dump(emotion_to_id, json_file, indent=4)
    
    if not os.path.exists(data_training_args.path_to_iemocap_speaker_dict_file):
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted(unique_speakers))}
        with open(data_training_args.path_to_iemocap_speaker_dict_file, 'w') as json_file:
            json.dump(speaker_to_id, json_file, indent=4)
        
    "Dataset split by speaker-emotion stratification"
    "This is only relevant for training, during evaluation we will use LOGO"    
    print("Creating stratified split with all speakers in all sets")
    
    # Define split proportions (train: 90%, validation: 5%, test: 5%)
    train_prop = 0.90
    val_prop = 0.05
    test_prop = 0.05
    
    # Group by speaker and emotion to preserve stratification
    speaker_groups = []
    for speaker_id in df['speaker_id'].unique():
        speaker_df = df[df['speaker_id'] == speaker_id]
        # Further stratify by emotion within each speaker
        for emotion in speaker_df['emotion'].unique():
            emotion_df = speaker_df[speaker_df['emotion'] == emotion]
            speaker_groups.append(emotion_df)
    
    # Initialize empty dataframes for each split
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Split each speaker-emotion group
    for group_df in speaker_groups:
        # Shuffle the group with the specified random seed
        group_df = group_df.sample(frac=1, random_state=data_training_args.seed)
        
        # Calculate split sizes
        n_samples = len(group_df)
        n_train = int(n_samples * train_prop)
        n_val = int(n_samples * val_prop)
        # n_test will be the remainder
        
        # Split the group
        train_dfs.append(group_df.iloc[:n_train])
        test_dfs.append(group_df.iloc[n_train:n_train+n_val])
        val_dfs.append(group_df.iloc[n_train+n_val:])
    
    # Combine all splits
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    test_df = pd.concat(test_dfs)
    
    # Verify that all speakers appear in all splits
    train_speakers = set(train_df['speaker_id'].unique())
    val_speakers = set(val_df['speaker_id'].unique())
    test_speakers = set(test_df['speaker_id'].unique())
    
    all_speakers = set(df['speaker_id'].unique())
    
    # emotion distribution in each split
    #print("\nEmotion distribution:")
    #print("Train:", train_df['emotion'].value_counts().to_dict())
    #print("Val:", val_df['emotion'].value_counts().to_dict())
    #print("Test:", test_df['emotion'].value_counts().to_dict())
    
    # samples distribution
    #total_samples = len(df)
    #print(f"\nTotal samples: {total_samples}")
    #print(f"Train set: {len(train_df)} samples ({len(train_df)/total_samples*100:.1f}%)")
    #print(f"Validation set: {len(val_df)} samples ({len(val_df)/total_samples*100:.1f}%)")
    #print(f"Test set: {len(test_df)} samples ({len(test_df)/total_samples*100:.1f}%)")

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df) #.iloc[0:100])
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    train_datasets_splits = []
    val_datasets_splits = []
    test_datasets_splits = []

    train_datasets_splits.append(train_dataset)
    val_datasets_splits.append(val_dataset)
    test_datasets_splits.append(test_dataset)
    
    # Create final dataset dictionary
    raw_datasets = DatasetDict()
    
    if len(train_datasets_splits) > 0:
        if len(train_datasets_splits) > 1:
            raw_datasets["train"] = concatenate_datasets(train_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["train"] = train_datasets_splits[0].shuffle(seed=data_training_args.seed)
    
    if data_training_args.train_val_test_split:
        if data_training_args.validation_split_percentage is None:
            if len(val_datasets_splits) > 1:
                raw_datasets["validation"] = concatenate_datasets(val_datasets_splits).shuffle(seed=data_training_args.seed)
            else:
                raw_datasets["validation"] = val_datasets_splits[0].shuffle(seed=data_training_args.seed)
        else:
            num_validation_samples = raw_datasets["train"].num_rows * data_training_args.validation_split_percentage // 100
            raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
            raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))
        
        if len(test_datasets_splits) > 1:
            raw_datasets["test"] = concatenate_datasets(test_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["test"] = test_datasets_splits[0].shuffle(seed=data_training_args.seed)
    else:
        if len(val_datasets_splits) > 1:
            raw_datasets["validation"] = concatenate_datasets(val_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["validation"] = val_datasets_splits[0].shuffle(seed=data_training_args.seed)
        
        if len(test_datasets_splits) > 1:
            raw_datasets["test"] = concatenate_datasets(test_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["test"] = test_datasets_splits[0].shuffle(seed=data_training_args.seed)
    
    print("IEMOCAP dataset processed and loaded successfully.")
    return raw_datasets

def load_iemocap_speaker_dependent(data_training_args):
    """
    Process and load the IEMOCAP dataset with speaker-dependent splits.
    
    This function loads the IEMOCAP dataset in the same way as load_iemocap,
    but creates 10 different sets, one for each unique speaker, instead of the 
    standard train/val/test split. 
    
    Args:
        data_training_args: Arguments for data processing
    
    Returns:
        DatasetDict containing one split for each unique speaker
    """
    base_dir = data_training_args.data_dir
    print(f"Processing IEMOCAP dataset from {base_dir} with speaker-dependent splits")
    
    # Dictionary to store all extracted data
    all_data = []
    all_phonemes = []
    
    # Iterate through all sessions
    for session_idx in range(1, 6):
        session_name = f"Session{session_idx}"
        session_path = os.path.join(base_dir, session_name)
        
        print(f"Processing {session_name}...")
        
        # Get all WAV files in the sentences/wav directory
        wav_dir = os.path.join(session_path, "sentences", "wav")
        wav_files = glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)
        
        # Process each WAV file
        for wav_file in tqdm(wav_files, desc=f"Processing {session_name} files"):
            # Extract file identifiers
            file_name = os.path.basename(wav_file)
            file_id = os.path.splitext(file_name)[0]
            
            # Extract the conversation style and handle different formats
            parts = file_id.split('_')
            speaker_id = parts[0]  # e.g., Ses01F
            gender = speaker_id[-1]  # F or M
            
            if 'impro' in parts[1]:
                conv_style = 'impro'
            elif 'script' in parts[1]:
                conv_style = 'script'
                
            if len(parts) >= 3 and parts[2][0].isdigit():
                sentence_id = '_'.join(parts[:3])
            else:
                sentence_id = '_'.join(parts[:2])
            
            # Get audio data
            try:
                audio_data, sample_rate = sf.read(wav_file)
                duration = len(audio_data) / sample_rate
                audio_array = audio_data
            except Exception as e:
                print(f"Error loading audio file {wav_file}: {e}")
                continue
            
            # Get phoneme segmentation
            phseg_file = os.path.join(session_path, "sentences", "ForcedAlignment", 
                                     sentence_id, file_id + ".phseg")
            phonemes_dict = []
            
            try:
                if os.path.exists(phseg_file):
                    with open(phseg_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip() and not line.startswith('\t SFrm'):
                                parts = line.strip().split()
                                if len(parts) >= 4:
                                    try:
                                        # Transform to seconds
                                        start_time = (float(parts[0]) + 2) / 100
                                        end_time = (float(parts[1]) + 2) / 100
                                        primary_phone = parts[3]
                                        all_phonemes.append(primary_phone)
                                        phonemes_dict.append({
                                            'start': start_time,
                                            'end': end_time,
                                            'phoneme': primary_phone
                                        })
                                    except:
                                        continue
                else:
                    print(f"Phoneme file not found: {phseg_file}")
                    continue
            except Exception as e:
                print(f"Error loading phoneme file {phseg_file}: {e}")
                phonemes_dict = []
                continue
            
            # Get emotion label
            emotion_file = os.path.join(session_path, "dialog", "EmoEvaluation", "Majority", 
                                        f"{sentence_id}.txt")
            emotion = "unknown"
            valence = arousal = dominance = 0.0
            
            try:
                if os.path.exists(emotion_file):
                    with open(emotion_file, 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if file_id in line and not line.startswith('%') and line.startswith('['):
                                parts = line.strip().split('\t')
                                if len(parts) >= 3:
                                    emotion = parts[2]
                                    # Try to extract VAD values if available
                                    if len(parts) >= 4 and '[' in parts[3] and ']' in parts[3]:
                                        try:
                                            vad_str = parts[3].strip('[]')
                                            vad_values = [float(x.strip()) for x in vad_str.split(',')]
                                            if len(vad_values) >= 3:
                                                valence = vad_values[0]
                                                arousal = vad_values[1]
                                                dominance = vad_values[2]
                                        except:
                                            pass
                                break
                else:
                    print(f"Emotion file not found: {emotion_file}")
                    continue
            except Exception as e:
                print(f"Error loading emotion file {emotion_file}: {e}")
                continue
            
            # Create a data entry for this file
            entry = {
                'file_id': file_id,
                'session': session_name,
                'sentence_id': sentence_id,
                'speaker_id': speaker_id,
                'conv_style': conv_style,
                'gender': gender,
                'path': wav_file,
                'audio': audio_array,
                'sampling_rate': sample_rate,
                'duration': duration,
                'emotion': emotion,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'phonemes_dict': phonemes_dict
            }
            
            all_data.append(entry)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_data)

    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)
    # Filter out all emotions except 5
    df = df[df['emotion'].isin(['hap', 'sad', 'ang', 'exc', 'neu'])]
    df['emotion'] = df['emotion'].replace({'exc': 'hap'})

    "Save encodings to .json dictionaries to be used in evaluations"
    "Create mappings to integers"
    unique_phonemes = np.unique(all_phonemes)
    unique_emotions = df['emotion'].unique()
    unique_speakers = df['speaker_id'].unique()
    
    print(f"Found {len(unique_speakers)} unique speakers: {unique_speakers}")
    print(f"Found {len(unique_emotions)} emotions: {unique_emotions}")
    print(f"Found {len(unique_phonemes)} unique phonemes")
    
    # Create mapping dictionaries if they don't exist
    if not os.path.exists(data_training_args.path_to_iemocap_phoneme_to_id_file):
        phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(sorted(unique_phonemes))}
        with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'w') as json_file:
            json.dump(phoneme_to_id, json_file, indent=4)
    
    if not os.path.exists(data_training_args.path_to_iemocap_emotion_to_id_file):
        emotion_to_id = {emotion: idx for idx, emotion in enumerate(sorted(unique_emotions))}
        with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'w') as json_file:
            json.dump(emotion_to_id, json_file, indent=4)
    
    if not os.path.exists(data_training_args.path_to_iemocap_speaker_dict_file):
        speaker_to_id = {speaker: idx for idx, speaker in enumerate(sorted(unique_speakers))}
        with open(data_training_args.path_to_iemocap_speaker_dict_file, 'w') as json_file:
            json.dump(speaker_to_id, json_file, indent=4)
    
    # Create speaker-dependent splits
    print("Creating speaker-dependent splits - one dataset for each speaker")
    
    # Initialize dictionary to store datasets for each speaker
    speaker_datasets = {}
    
    # Create a dataset for each speaker
    for speaker_id in unique_speakers:
        speaker_df = df[df['speaker_id'] == speaker_id]
        print(f"Speaker {speaker_id}: {len(speaker_df)} samples")
        
        # Convert to Hugging Face dataset
        speaker_dataset = Dataset.from_pandas(speaker_df) #.iloc[0:20]
            
        speaker_datasets[speaker_id] = speaker_dataset
    
    # Create final dataset dictionary
    raw_datasets = DatasetDict()
    
    # Add each speaker dataset to the DatasetDict
    for id, (speaker_id, dataset) in enumerate(speaker_datasets.items()):
        raw_datasets[f"speaker{id+1}"] = dataset.shuffle(seed=data_training_args.seed)
    
    print("IEMOCAP dataset processed and loaded with speaker-dependent splits successfully.")
    return raw_datasets

def load_traversal_subset_iemocap(data_training_args):
    """
    Process and load the IEMOCAP dataset for latent traversal analysis.
    
    Creates traversal datasets with:
    1. Fixed emotion, varying speaker and vowel
    2. Fixed phoneme, varying emotion
    3. Fixed speaker, varying emotion
    4. Fixed non-verbal phoneme, varying emotion
    
    Tracks maximum segment length across all datasets.
    
    Args:
        data_training_args: Arguments for data processing
    
    Returns:
        DatasetDict containing various traversal datasets
    """
    base_dir = data_training_args.data_dir
    print(f"Processing IEMOCAP traversal dataset from {base_dir}")
    
    # Initialize length tracking variables
    max_segment_length = 0
    max_segment_duration = 0
    segment_lengths = []
    
    # Dictionary to store all extracted data
    all_data = []
    all_phonemes = []
    
    # Iterate through all sessions
    for session_idx in range(1, 6):
        session_name = f"Session{session_idx}"
        session_path = os.path.join(base_dir, session_name)
        
        print(f"Processing {session_name} for traversal...")
        
        # Get all WAV files in the sentences/wav directory
        wav_dir = os.path.join(session_path, "sentences", "wav")
        wav_files = glob.glob(os.path.join(wav_dir, "**", "*.wav"), recursive=True)
        
        # Process each WAV file
        for wav_file in tqdm(wav_files, desc=f"Processing {session_name} files"):
            # Extract file identifiers
            file_name = os.path.basename(wav_file)
            file_id = os.path.splitext(file_name)[0]
            
            # Extract the conversation style and handle different formats
            parts = file_id.split('_')
            speaker_id = parts[0]  # e.g., Ses01F
            gender = speaker_id[-1]  # F or M
            
            if 'impro' in parts[1]:
                conv_style = 'impro'
            elif 'script' in parts[1]:
                conv_style = 'script'
                
            if len(parts) >= 3 and parts[2][0].isdigit():
                sentence_id = '_'.join(parts[:3])
            else:
                sentence_id = '_'.join(parts[:2])
            
            # Get audio data
            try:
                audio_data, sample_rate = sf.read(wav_file)
                duration = len(audio_data) / sample_rate
                audio_array = audio_data
            except Exception as e:
                print(f"Error loading audio file {wav_file}: {e}")
                continue
            
            # Get phoneme segmentation
            phseg_file = os.path.join(session_path, "sentences", "ForcedAlignment", 
                                     sentence_id, file_id + ".phseg")
            phonemes_dict = []
            
            try:
                if os.path.exists(phseg_file):
                    with open(phseg_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if line.strip() and not line.startswith('\t SFrm'):
                                parts = line.strip().split()
                                if len(parts) >= 4:
                                    try:
                                        # Transform to seconds
                                        start_time = (float(parts[0]) + 2) / 100
                                        end_time = (float(parts[1]) + 2) / 100
                                        primary_phone = parts[3]
                                        all_phonemes.append(primary_phone)
                                        phonemes_dict.append({
                                            'start': start_time,
                                            'end': end_time,
                                            'phoneme': primary_phone
                                        })
                                    except:
                                        continue
                else:
                    print(f"Phoneme file not found: {phseg_file}")
                    continue
            except Exception as e:
                print(f"Error loading phoneme file {phseg_file}: {e}")
                continue
            
            # Get emotion label
            emotion_file = os.path.join(session_path, "dialog", "EmoEvaluation", "Majority", 
                                        f"{sentence_id}.txt")
            emotion = "unknown"
            valence = arousal = dominance = 0.0
            
            try:
                if os.path.exists(emotion_file):
                    with open(emotion_file, 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if file_id in line and not line.startswith('%') and line.startswith('['):
                                parts = line.strip().split('\t')
                                if len(parts) >= 3:
                                    emotion = parts[2]
                                    # Try to extract VAD values if available
                                    if len(parts) >= 4 and '[' in parts[3] and ']' in parts[3]:
                                        try:
                                            vad_str = parts[3].strip('[]')
                                            vad_values = [float(x.strip()) for x in vad_str.split(',')]
                                            if len(vad_values) >= 3:
                                                valence = vad_values[0]
                                                arousal = vad_values[1]
                                                dominance = vad_values[2]
                                        except:
                                            pass
                                break
                else:
                    print(f"Emotion file not found: {emotion_file}")
                    continue
            except Exception as e:
                print(f"Error loading emotion file {emotion_file}: {e}")
                continue
            
            # Create a data entry for this file
            entry = {
                'file_id': file_id,
                'session': session_name,
                'sentence_id': sentence_id,
                'speaker_id': speaker_id,
                'conv_style': conv_style,
                'gender': gender,
                'path': wav_file,
                'audio': audio_array,
                'sampling_rate': sample_rate,
                'duration': duration,
                'emotion': emotion,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'phonemes_dict': phonemes_dict
            }
            
            if len(phonemes_dict) > 0:
                all_data.append(entry)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_data)
    
    # Filter emotions to the main categories
    df = df[df['emotion'].isin(['hap', 'sad', 'ang', 'exc', 'neu'])]
    df['emotion'] = df['emotion'].replace({'exc': 'hap'})  # Combine happy and excited
    
    # Load the phoneme, emotion, and speaker mappings
    with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
        phoneme_to_id = json.load(json_file)
    
    with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'r') as json_file:
        emotion_to_id = json.load(json_file)
        
    with open(data_training_args.path_to_iemocap_speaker_dict_file, 'r') as json_file:
        speaker_to_id = json.load(json_file)
    
    # Create id to phoneme/emotion/speaker mappings
    id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
    id_to_emotion = {v: k for k, v in emotion_to_id.items()}
    id_to_speaker = {v: k for k, v in speaker_to_id.items()}
    
    # Function to extract phoneme segments
    def extract_phoneme_segment(entry, phoneme):
        segments = []
        for p_dict in entry['phonemes_dict']:
            if p_dict['phoneme'] == phoneme:
                start_time = p_dict['start']
                end_time = p_dict['end']
                
                # Calculate indices for audio segment
                start_idx = int(start_time * entry['sampling_rate'])
                end_idx = int(end_time * entry['sampling_rate'])
                
                # Skip segments that are too short
                segment_length = end_idx - start_idx
                if segment_length < MIN_ACCEPTABLE_LENGTH:
                    continue
                
                # Copy the entry
                segment = entry.copy()
                
                # Update audio with just this segment
                segment['audio'] = entry['audio'][start_idx:end_idx]
                segment['duration'] = (end_idx - start_idx) / entry['sampling_rate']
                segment['segment_length'] = segment_length
                
                # Keep track of which phoneme was extracted
                segment['extracted_phoneme'] = phoneme
                
                # Update maximum length tracking
                nonlocal max_segment_length, max_segment_duration, segment_lengths
                max_segment_length = max(max_segment_length, segment_length)
                max_segment_duration = max(max_segment_duration, segment['duration'])
                segment_lengths.append(segment_length)
                
                segments.append(segment)
        return segments
    
    # Function to check if an entry contains a specific phoneme
    def contains_phoneme(entry, phoneme):
        return any(p_dict['phoneme'] == phoneme for p_dict in entry['phonemes_dict'])
    
    # Create empty lists for each traversal type
    fixed_emotion_phoneme_speaker_datasets = []
    fixed_phoneme_emotion_datasets = []
    fixed_speaker_emotion_datasets = []
    fixed_nonverbal_emotion_datasets = []
    
    # Track per-category maximum lengths
    emotion_max_lengths = {}
    phoneme_max_lengths = {}
    speaker_max_lengths = {}
    nonverbal_max_lengths = {}
    
    # 1. Fixed emotion, varying speaker and vowel only
    for emotion in ['hap', 'sad', 'ang', 'neu']:
        emotion_phoneme_examples = []
        combination_counts = {}  # Track count of each (speaker, phoneme) combination
        emotion_max_lengths[emotion] = 0
        
        # Filter entries with this emotion
        emotion_entries = df[df['emotion'] == emotion]
        
        for speaker_id in df['speaker_id'].unique():
            for phoneme in SEL_PHONEMES_LIST_IEMOCAP:
                if phoneme not in phoneme_to_id:
                    continue
                
                combination = (speaker_id, phoneme)
                combination_counts[combination] = 0
                
                # Find entries with this emotion, this speaker, containing this phoneme
                speaker_entries = emotion_entries[emotion_entries['speaker_id'] == speaker_id]
                
                # Try to get target_per_combination examples
                count = 0
                for _, entry in speaker_entries.iterrows():
                    if count >= TARGET_PER_COMBINATION:
                        break
                        
                    if contains_phoneme(entry, phoneme):
                        segments = extract_phoneme_segment(entry, phoneme)
                        if segments:
                            segment = segments[0]
                            emotion_phoneme_examples.append(segment)
                            combination_counts[combination] = combination_counts.get(combination, 0) + 1
                            count += 1
                            # Track max length for this emotion
                            emotion_max_lengths[emotion] = max(emotion_max_lengths[emotion], segment['segment_length'])
        
        # Count how many combinations have at least one example
        combinations_with_examples = sum(1 for count in combination_counts.values() if count > 0)
        total_examples = sum(combination_counts.values())
        
        if emotion_phoneme_examples and combinations_with_examples >= 5:  # At least 5 different combinations
            # Convert to Dataset
            fixed_emotion_phoneme_dataset = Dataset.from_dict({
                k: [example[k] for example in emotion_phoneme_examples if k in example]
                for k in emotion_phoneme_examples[0].keys()
            })
            
            fixed_emotion_phoneme_speaker_datasets.append({
                'dataset': fixed_emotion_phoneme_dataset,
                'fixed_factor': 'emotion_vowel',
                'fixed_value': emotion,
                'max_length': emotion_max_lengths[emotion],
                'combination_counts': combination_counts,
                'total_examples': total_examples,
                'combinations_with_examples': combinations_with_examples
            })
    
    # 2. Fixed phoneme, varying emotion
    for phoneme in SEL_PHONEMES_LIST_IEMOCAP:
        if phoneme not in phoneme_to_id:
            continue
            
        phoneme_emotion_examples = []
        emotion_counts = {emotion: 0 for emotion in ['hap', 'sad', 'ang', 'neu']}  # Track count of each emotion
        phoneme_max_lengths[phoneme] = 0
        
        # Filter entries containing this phoneme
        for _, entry in df.iterrows():
            emotion = entry['emotion']
            
            # Skip if we already have enough examples of this emotion
            if emotion_counts.get(emotion, 0) >= TARGET_PER_EMOTION:
                continue
                
            if contains_phoneme(entry, phoneme):
                segments = extract_phoneme_segment(entry, phoneme)
                if segments:
                    segment = segments[0]
                    phoneme_emotion_examples.append(segment)
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    # Track max length for this phoneme
                    phoneme_max_lengths[phoneme] = max(phoneme_max_lengths[phoneme], segment['segment_length'])
        
        # Check if we have enough examples with different emotions
        num_emotions_with_examples = sum(1 for count in emotion_counts.values() if count > 0)
        if phoneme_emotion_examples and num_emotions_with_examples >= 3:  # At least 3 emotions
            # Convert to Dataset
            fixed_phoneme_emotion_dataset = Dataset.from_dict({
                k: [example[k] for example in phoneme_emotion_examples if k in example]
                for k in phoneme_emotion_examples[0].keys()
            })
            
            fixed_phoneme_emotion_datasets.append({
                'dataset': fixed_phoneme_emotion_dataset,
                'fixed_factor': 'phoneme_emotion',
                'fixed_value': phoneme,
                'max_length': phoneme_max_lengths[phoneme],
                'emotion_counts': emotion_counts
            })
    
    # 3. Fixed speaker, varying emotion - MODIFIED to collect 3+ examples per emotion
    for speaker_id in df['speaker_id'].unique():
        speaker_emotion_examples = []
        emotion_counts = {emotion: 0 for emotion in ['hap', 'sad', 'ang', 'neu']}  # Track count of each emotion
        speaker_max_lengths[speaker_id] = 0
        
        # Filter entries from this speaker
        speaker_entries = df[df['speaker_id'] == speaker_id]
        
        for emotion in ['hap', 'sad', 'ang', 'neu']:
            # Get all entries with this speaker and emotion
            emotion_entries = speaker_entries[speaker_entries['emotion'] == emotion]
            
            # Try to get target_per_emotion examples
            count = 0
            for _, entry in emotion_entries.iterrows():
                if count >= TARGET_PER_EMOTION:
                    break
                    
                # Find a phoneme segment (any phoneme)
                if entry['phonemes_dict']:
                    phoneme = entry['phonemes_dict'][0]['phoneme']
                    segments = extract_phoneme_segment(entry, phoneme)
                    if segments:
                        segment = segments[0]
                        speaker_emotion_examples.append(segment)
                        emotion_counts[emotion] += 1
                        count += 1
                        # Track max length for this speaker
                        speaker_max_lengths[speaker_id] = max(speaker_max_lengths[speaker_id], segment['segment_length'])
        
        # Check if we have enough examples with different emotions
        num_emotions_with_examples = sum(1 for count in emotion_counts.values() if count > 0)
        if speaker_emotion_examples and num_emotions_with_examples >= 3:  # At least 3 emotions
            # Convert to Dataset
            fixed_speaker_emotion_dataset = Dataset.from_dict({
                k: [example[k] for example in speaker_emotion_examples if k in example]
                for k in speaker_emotion_examples[0].keys()
            })
            
            fixed_speaker_emotion_datasets.append({
                'dataset': fixed_speaker_emotion_dataset,
                'fixed_factor': 'speaker_emotion',
                'fixed_value': speaker_id,
                'max_length': speaker_max_lengths[speaker_id],
                'emotion_counts': emotion_counts
            })
    
    # 4. Fixed non-verbal phoneme, varying emotion - MODIFIED to collect 3+ examples per emotion
    for nonverbal in SEL_NON_VERBAL_PHONEMES_LIST_IEMOCAP:
        if nonverbal not in phoneme_to_id:
            continue
            
        nonverbal_emotion_examples = []
        emotion_counts = {emotion: 0 for emotion in ['hap', 'sad', 'ang', 'neu']}  # Track count of each emotion
        nonverbal_max_lengths[nonverbal] = 0
        
        # Filter entries containing this non-verbal phoneme
        for _, entry in df.iterrows():
            emotion = entry['emotion']
            
            # Skip if we already have enough examples of this emotion
            if emotion_counts.get(emotion, 0) >= TARGET_PER_EMOTION:
                continue
                
            if contains_phoneme(entry, nonverbal):
                segments = extract_phoneme_segment(entry, nonverbal)
                if segments:
                    segment = segments[0]
                    nonverbal_emotion_examples.append(segment)
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                    # Track max length for this nonverbal
                    nonverbal_max_lengths[nonverbal] = max(nonverbal_max_lengths[nonverbal], segment['segment_length'])
        
        # Check if we have enough examples with different emotions
        num_emotions_with_examples = sum(1 for count in emotion_counts.values() if count > 0)
        if nonverbal_emotion_examples and num_emotions_with_examples >= 2:  # At least 2 emotions (kept at 2 since nonverbals are rarer)
            # Convert to Dataset
            fixed_nonverbal_emotion_dataset = Dataset.from_dict({
                k: [example[k] for example in nonverbal_emotion_examples if k in example]
                for k in nonverbal_emotion_examples[0].keys()
            })
            
            fixed_nonverbal_emotion_datasets.append({
                'dataset': fixed_nonverbal_emotion_dataset,
                'fixed_factor': 'nonverbal_emotion',
                'fixed_value': nonverbal,
                'max_length': nonverbal_max_lengths[nonverbal],
                'emotion_counts': emotion_counts
            })
    
    # Create the final datasets dictionary
    raw_datasets = DatasetDict()
    
    # Add all traversal datasets
    # 1. Fixed emotion, varying speaker and vowel
    if fixed_emotion_phoneme_speaker_datasets:
        emotion_phoneme_datasets = [d['dataset'] for d in fixed_emotion_phoneme_speaker_datasets]
        if emotion_phoneme_datasets:
            raw_datasets["fixed_emotion_phoneme_speaker"] = concatenate_datasets(emotion_phoneme_datasets)
    
    # 2. Fixed phoneme, varying emotion
    if fixed_phoneme_emotion_datasets:
        phoneme_emotion_datasets = [d['dataset'] for d in fixed_phoneme_emotion_datasets]
        if phoneme_emotion_datasets:
            raw_datasets["fixed_phoneme_emotion"] = concatenate_datasets(phoneme_emotion_datasets)
    
    # 3. Fixed speaker, varying emotion
    if fixed_speaker_emotion_datasets:
        speaker_emotion_datasets = [d['dataset'] for d in fixed_speaker_emotion_datasets]
        if speaker_emotion_datasets:
            raw_datasets["fixed_speaker_emotion"] = concatenate_datasets(speaker_emotion_datasets)
    
    # 4. Fixed non-verbal, varying emotion
    if fixed_nonverbal_emotion_datasets:
        nonverbal_emotion_datasets = [d['dataset'] for d in fixed_nonverbal_emotion_datasets]
        if nonverbal_emotion_datasets:
            raw_datasets["fixed_nonverbal_emotion"] = concatenate_datasets(nonverbal_emotion_datasets)
    
    # Calculate statistics about segment lengths
    if segment_lengths:
        min_segment_length = min(segment_lengths)
        avg_segment_length = sum(segment_lengths) / len(segment_lengths)
        sample_rate = 16000  # IEMOCAP sample rate
        min_duration_ms = min_segment_length * 1000 / sample_rate
        max_duration_ms = max_segment_length * 1000 / sample_rate
        avg_duration_ms = avg_segment_length * 1000 / sample_rate
    else:
        min_segment_length = 0
        avg_segment_length = 0
        min_duration_ms = 0
        max_duration_ms = 0
        avg_duration_ms = 0
    
    # Store metadata about the traversals and segment lengths
    raw_datasets.traversal_metadata = {
        'fixed_emotion_vowel_speaker': fixed_emotion_phoneme_speaker_datasets,
        'fixed_phoneme_emotion': fixed_phoneme_emotion_datasets,
        'fixed_speaker_emotion': fixed_speaker_emotion_datasets,
        'fixed_nonverbal_emotion': fixed_nonverbal_emotion_datasets
    }
    
    raw_datasets.segment_length_stats = {
        'overall_min_length': min_segment_length,
        'overall_max_length': max_segment_length,
        'overall_avg_length': avg_segment_length,
        'emotion_max_lengths': emotion_max_lengths,
        'phoneme_max_lengths': phoneme_max_lengths,
        'speaker_max_lengths': speaker_max_lengths,
        'nonverbal_max_lengths': nonverbal_max_lengths
    }
    
    # Report segment length statistics
    print("\n===== Audio Segment Length Statistics =====")
    print(f"Min segment length: {min_segment_length} samples ({min_duration_ms:.2f} ms)")
    print(f"Max segment length: {max_segment_length} samples ({max_duration_ms:.2f} ms)")
    print(f"Avg segment length: {avg_segment_length:.2f} samples ({avg_duration_ms:.2f} ms)")
    
    print("\n----- Emotion-specific Max Lengths -----")
    for emotion, max_len in sorted(emotion_max_lengths.items()):
        max_dur_ms = max_len * 1000 / sample_rate
        print(f"Emotion '{emotion}': {max_len} samples ({max_dur_ms:.2f} ms)")
    
    print("\n----- Phoneme-specific Max Lengths -----")
    for phoneme, max_len in sorted(phoneme_max_lengths.items()):
        max_dur_ms = max_len * 1000 / sample_rate
        print(f"Phoneme '{phoneme}': {max_len} samples ({max_dur_ms:.2f} ms)")
    
    print("\n----- Speaker-specific Max Lengths -----")
    for speaker, max_len in sorted(speaker_max_lengths.items()):
        max_dur_ms = max_len * 1000 / sample_rate
        print(f"Speaker '{speaker}': {max_len} samples ({max_dur_ms:.2f} ms)")
    
    print("\n----- Non-verbal-specific Max Lengths -----")
    for nonverbal, max_len in sorted(nonverbal_max_lengths.items()):
        max_dur_ms = max_len * 1000 / sample_rate
        print(f"Non-verbal '{nonverbal}': {max_len} samples ({max_dur_ms:.2f} ms)")
    
    print("\nIEMOCAP traversal datasets created successfully.")
    return raw_datasets