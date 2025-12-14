"""
Script to organize and preprocess the VOC-ALS dataset.
It loads all files, extracts metadata from the Excel file,
encodes each variable to integer-string pairs and saves them in ./vocabularies/ .json files.
If needed it can perform a train/test/validation split as well. 
"""

import os
import sys
# Add project root to Python path for module resolution
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

import pandas as pd
import re
import json
import gzip
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE_DIR = "../VOC-ALS" 
OUTPUT_DIR = "../VOC-ALS_preprocessed"
VOCAB_DIR = "./vocabularies"
SPLIT_DATA = False  # Set to True to enable train/val/test split
NUM_PARTS = 4
TARGET_FS = 16000  # Target sample rate for audio files
TEST_SIZE = 0.2
VAL_SIZE = 0.15
SEED = 42  # Random seed for reproducibility

def encode_metadata_values(metadata_df):
    """
    Encode specific metadata columns according to the specified ranges
    
    Args:
        metadata_df: DataFrame containing the metadata
        
    Returns:
        DataFrame with encoded values for specific columns
    """
    df = metadata_df.copy()
    
    encodings = {}

    # 1. Encode ALSFRS-R_TotalScore: 0-48 into intervals of 6 (0-5, 6-11, etc.)
    if 'ALSFRS-R_TotalScore' in df.columns:
        # Define the bins for ALSFRS-R
        alsfrs_bins = [0, 6, 12, 18, 24, 30, 36, 42, 48]
        alsfrs_labels = list(range(8))  # 0-7 for the 8 intervals
        
        # Convert to numeric, handling non-numeric values
        df['ALSFRS-R_TotalScore'] = pd.to_numeric(df['ALSFRS-R_TotalScore'], errors='coerce')
        
        # Create a new column with encoded values
        df['ALSFRS-R_TotalScore_encoded'] = pd.cut(
            df['ALSFRS-R_TotalScore'], 
            bins=alsfrs_bins, 
            labels=alsfrs_labels, 
            include_lowest=True,
            right=True
        )
        
        # Replace NaN with None for JSON serialization
        df['ALSFRS-R_TotalScore_encoded'] = df['ALSFRS-R_TotalScore_encoded'].astype('Int64')
        print("Encoded ALSFRS-R_TotalScore into 8 intervals")

        encoding_map = {str(i): f"{alsfrs_bins[i]}-{alsfrs_bins[i+1]-1}" for i in range(len(alsfrs_labels))}
        encodings['ALSFRS-R_TotalScore'] = encoding_map

    # 2. Encode DiseaseDuration
    if 'DiseaseDuration' in df.columns:
        # Define the bins for disease duration (in years)
        duration_bins = [0, 1, 5, 10, 20, float('inf')]
        duration_labels = [0, 1, 2, 3, 4]  # Encoded values
        
        # Convert to numeric, handling non-numeric values
        df['DiseaseDuration'] = pd.to_numeric(df['DiseaseDuration'], errors='coerce')
        
        # Create a new column with encoded values
        df['DiseaseDuration_encoded'] = pd.cut(
            df['DiseaseDuration'], 
            bins=duration_bins, 
            labels=duration_labels, 
            include_lowest=True,
            right=True
        )
        
        # Replace NaN with None for JSON serialization
        df['DiseaseDuration_encoded'] = df['DiseaseDuration_encoded'].astype('Int64')
        print("Encoded DiseaseDuration into 5 intervals")
        
        bin_labels = [f"{duration_bins[i]}-{duration_bins[i+1]}" if duration_bins[i+1] != float('inf') else f"{duration_bins[i]}+" 
                     for i in range(len(duration_labels))]
        encoding_map = {str(i): bin_labels[i] for i in range(len(duration_labels))}
        encodings['DiseaseDuration'] = encoding_map

    # 3. Encode KingClinicalStage
    if 'KingClinicalStage' in df.columns:
        # Convert stage values according to specification
        king_stage_mapping = {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            '4A': 4,
            '4B': 5,
            '4A/B': 6
        }
        
        # Function to apply mapping
        def encode_king_stage(val):
            try:
                # First try direct mapping
                if val in king_stage_mapping:
                    return king_stage_mapping[val]
                # Try converting to integer first (for values like '1', '2', etc.)
                elif str(val).isdigit() and int(val) in king_stage_mapping:
                    return king_stage_mapping[int(val)]
                # For string representations
                elif str(val) in king_stage_mapping:
                    return king_stage_mapping[str(val)]
                else:
                    return None
            except:
                return None
        
        # Create new encoded column
        df['KingClinicalStage_encoded'] = df['KingClinicalStage'].apply(encode_king_stage)
        print("Encoded KingClinicalStage into values 0-6")

        reverse_mapping = {str(v): str(k) for k, v in king_stage_mapping.items()}
        encodings['KingClinicalStage'] = reverse_mapping

    # 4. Encode Cantagallo_Questionnaire
    if 'Cantagallo_Questionnaire' in df.columns:
        # Define the bins for Cantagallo Questionnaire
        cantagallo_bins = [0, 7, 35, 70, 134, 140]
        cantagallo_labels = [0, 1, 2, 3, 4]  # Encoded values
        
        # Convert to numeric, handling non-numeric values
        df['Cantagallo_Questionnaire'] = pd.to_numeric(df['Cantagallo_Questionnaire'], errors='coerce')
        
        # Create a new column with encoded values
        df['Cantagallo_Questionnaire_encoded'] = pd.cut(
            df['Cantagallo_Questionnaire'], 
            bins=cantagallo_bins, 
            labels=cantagallo_labels, 
            include_lowest=True,
            right=True
        )
        
        # Replace NaN with None for JSON serialization
        df['Cantagallo_Questionnaire_encoded'] = df['Cantagallo_Questionnaire_encoded'].astype('Int64')
        print("Encoded Cantagallo_Questionnaire into 5 intervals")

        bin_labels = [f"{cantagallo_bins[i]}-{cantagallo_bins[i+1]-1}" for i in range(len(cantagallo_labels))]
        encoding_map = {str(i): bin_labels[i] for i in range(len(cantagallo_labels))}
        encodings['Cantagallo_Questionnaire'] = encoding_map

    # 5. Add phoneme encoding map (added for future reference)
    df.attrs['phoneme_encoding'] = {
        'A': 0,
        'E': 1,
        'I': 2,
        'O': 3,
        'U': 4,
        'KA': 5,
        'PA': 6,
        'TA': 7
    }
    encodings['phoneme'] = {str(v): k for k, v in df.attrs['phoneme_encoding'].items()}
    print("Added phoneme encoding map (A,E,I,O,U,KA,PA,TA)")
    
    # 6. Add category encoding map (added for future reference)
    df.attrs['category_encoding'] = {
        'HC': 0,
        'ALS': 1
    }
    encodings['category'] = {str(v): k for k, v in df.attrs['category_encoding'].items()}
    print("Added category encoding map (HC,ALS)")
    
    # 7. Add speaker_id encoding map
    # Create and attach a function to generate speaker_id encodings
    def get_speaker_id_encoding():
        """
        Returns a dictionary mapping speaker_ids to encoded values
        """
        speakers = [
            "CT001", "CT004", "CT010", "CT013", "CT014", "CT015", "CT018", "CT019", "CT020", 
            "CT021", "CT022", "CT023", "CT026", "CT027", "CT028", "CT029", "CT030", "CT033", 
            "CT035", "CT037", "CT038", "CT039", "CT040", "CT041", "CT043", "CT044", "CT045", 
            "CT046", "CT047", "CT048", "CT049", "CT050", "CT051", "CT052", "CT053", "CT054", 
            "CT055", "CT056", "CT057", "CT058", "CT059", "CT060", "CT061", "CT062", "CT063", 
            "CT064", "CT065", "CT066", "CT067", "CT068", "CT069", "PZ001", "PZ003", "PZ004", 
            "PZ005", "PZ006", "PZ007", "PZ008", "PZ009", "PZ010", "PZ011", "PZ012", "PZ013", 
            "PZ014", "PZ015", "PZ016", "PZ017", "PZ018", "PZ019", "PZ020", "PZ021", "PZ022", 
            "PZ023", "PZ024", "PZ025", "PZ026", "PZ028", "PZ029", "PZ030", "PZ031", "PZ032", 
            "PZ033", "PZ034", "PZ035", "PZ036", "PZ037", "PZ039", "PZ040", "PZ041", "PZ043", 
            "PZ044", "PZ045", "PZ048", "PZ049", "PZ050", "PZ051", "PZ052", "PZ053", "PZ054", 
            "PZ055", "PZ056", "PZ057", "PZ058", "PZ059", "PZ060", "PZ061", "PZ062", "PZ063", 
            "PZ064", "PZ065", "PZ066", "PZ067", "PZ068", "PZ069", "PZ070", "PZ071", "PZ072", 
            "PZ073", "PZ074", "PZ077", "PZ079", "PZ080", "PZ081", "PZ082", "PZ083", "PZ084", 
            "PZ085", "PZ087", "PZ088", "PZ089", "PZ090", "PZ091", "PZ092", "PZ094", "PZ095", 
            "PZ096", "PZ097", "PZ098", "PZ099", "PZ100", "PZ101", "PZ102", "PZ103", "PZ104", 
            "PZ105", "PZ107", "PZ108", "PZ109", "PZ110", "PZ111", "PZ112", "PZ114", "PZ115"
        ]
        return {speaker: i for i, speaker in enumerate(speakers)}
    
    # Create encoding map for speaker_id
    df.attrs['speaker_id_encoding'] = get_speaker_id_encoding()
    encodings['speaker_id'] = {str(v): k for k, v in df.attrs['speaker_id_encoding'].items()}
    print(f"Added speaker_id encoding map for {len(df.attrs['speaker_id_encoding'])} speakers")
    
    return df, encodings

def encode_audio_variables(data_dict, metadata_df):
    """
    Encode phoneme, category and speaker_id variables based on encoding maps
    
    Args:
        data_dict: Dictionary containing the data
        metadata_df: DataFrame containing the metadata and encoding maps
        
    Returns:
        Updated dictionary with encoded variables and encoding maps
    """
    # Get encoding maps
    phoneme_map = metadata_df.attrs.get('phoneme_encoding', {})
    category_map = metadata_df.attrs.get('category_encoding', {})
    speaker_id_map = metadata_df.attrs.get('speaker_id_encoding', {})
    
    # Create reverse mappings for the encodings
    phoneme_encoding = {str(v): str(k) for k, v in phoneme_map.items()}
    category_encoding = {str(v): str(k) for k, v in category_map.items()}
    speaker_id_encoding = {str(v): str(k) for k, v in speaker_id_map.items()}
    
    # Add the encoded values
    if 'phoneme' in data_dict:
        data_dict['phoneme_encoded'] = [phoneme_map.get(p, -1) for p in data_dict['phoneme']]
        
    if 'category' in data_dict:
        data_dict['category_encoded'] = [category_map.get(c, -1) for c in data_dict['category']]
        
    if 'speaker_id' in data_dict:
        data_dict['speaker_id_encoded'] = [speaker_id_map.get(s, -1) for s in data_dict['speaker_id']]
    """
    # Add encoding maps to the data dictionary
    if 'encodings' not in data_dict:
        data_dict['encodings'] = {}
        
    data_dict['encodings'].update({
        'phoneme': phoneme_encoding,
        'category': category_encoding,
        'speaker_id': speaker_id_encoding
    })
    """
    return data_dict

def process_voc_als_data(base_dir, excel_file, output_file, split_data=False, test_size=0.2, val_size=0.15, random_seed=42):
    """
    Processes VOC-ALS dataset by scanning directory structure, extracts data from .wav files
    and metadata from Excel, organizes it by subject, and optionally splits into train/val/test sets.
    
    Args:
        base_dir (str): Base directory containing experiment folders
        excel_file (str): Path to the VOC-ALS.xlsx file
        output_file (str): Path to save the processed data as .json.gz
        split_data (bool): Whether to split data into train/validation/test sets
        test_size (float): Proportion of data for testing (default: 0.2 or 20%)
        val_size (float): Proportion of data for validation (default: 0.15 or 15%)
        random_seed (int): Random seed for reproducibility
    
    Returns:
        dict: Processed data structure
    """
    # 1. Define the experiments
    experiments = ["phonationA", "phonationE", "phonationI", "phonationO", "phonationU",
                  "rhythmKA", "rhythmPA", "rhythmTA"]
    
    # 2. Read metadata from the Excel file
    print(f"Reading metadata from {excel_file}...")
    excel_df = pd.read_excel(excel_file, header = None)
    column_names = excel_df.iloc[1].values
    excel_df = pd.read_excel(excel_file, header = None,skiprows=2)
    excel_df.columns = column_names
        
    "Discard extracted features, keep only demographics and patient info"
    columns_to_keep = list(range(33)) + [len(excel_df.columns) - 1]
    excel_df = excel_df.iloc[:, columns_to_keep]

    excel_df['Subject_ID'] = excel_df['ID'].astype(str)  # Ensure Subject_ID is a string

    "Encode specific variables of interest to be in label format"
    excel_df, encoding_maps = encode_metadata_values(excel_df)
    print(f"Succesfully encoded clinical variables")
    encodings_filename = os.path.join(VOCAB_DIR, "voc_als_encodings.json")
    with open(encodings_filename, 'w', encoding='utf-8') as f:
        json.dump(encoding_maps, f, indent=2)
    print(f"Saved encoding maps to {encodings_filename}")
    
    # 3. Scan directories and collect file information
    print("Scanning directories for audio files...")
    audio_files = []
    subject_ids = set()
    
    for experiment in experiments:
        exp_dir = os.path.join(base_dir, experiment)
        if not os.path.exists(exp_dir):
            print(f"Warning: Experiment directory {exp_dir} not found. Skipping.")
            continue
        
        for file in os.listdir(exp_dir):
            if file.endswith('.wav'):
                # Extract subject ID from filename (e.g., CT001_phonationA.wav -> CT001)
                match = re.match(r'([A-Z]+\d+)_', file)
                if match:
                    subject_id = match.group(1)
                    subject_ids.add(subject_id)
                    audio_files.append({
                        'subject_id': subject_id,
                        'experiment': experiment,
                        'wav_file': file,
                        'path': os.path.join(exp_dir, file)
                    })
    
    print(f"Found {len(audio_files)} audio files across {len(subject_ids)} subjects")
    
    # 4. Organize subjects by group (patient/control)
    subject_list = list(subject_ids)
    patients = [s for s in subject_list if s.startswith('PZ')]
    controls = [s for s in subject_list if s.startswith('CT')]
    
    print(f"Subject distribution: {len(patients)} patients, {len(controls)} controls")
    
    # 5. Create train/val/test split if requested
    if split_data:
        print(f"Creating train/validation/test split (seed={random_seed})...")
        
        # Split patients
        patients_train, patients_temp = train_test_split(
            patients, test_size=test_size + val_size, random_state=random_seed)
        
        if len(patients_temp) >= 2:  # Make sure we have enough for both val and test
            val_ratio = val_size / (test_size + val_size)
            patients_val, patients_test = train_test_split(
                patients_temp, test_size=1-val_ratio, random_state=random_seed)
        else:
            # Handle edge case with very few patients
            patients_val = []
            patients_test = patients_temp
        
        # Split controls
        controls_train, controls_temp = train_test_split(
            controls, test_size=test_size + val_size, random_state=random_seed)
        
        if len(controls_temp) >= 2:  # Make sure we have enough for both val and test
            val_ratio = val_size / (test_size + val_size)
            controls_val, controls_test = train_test_split(
                controls_temp, test_size=1-val_ratio, random_state=random_seed)
        else:
            # Handle edge case with very few controls
            controls_val = []
            controls_test = controls_temp
        
        # Combine splits
        train_subjects = patients_train + controls_train
        val_subjects = patients_val + controls_val
        test_subjects = patients_test + controls_test
        
        print(f"Split summary:")
        print(f"  Training set: {len(train_subjects)} subjects ({len(patients_train)} patients, {len(controls_train)} controls)")
        print(f"  Validation set: {len(val_subjects)} subjects ({len(patients_val)} patients, {len(controls_val)} controls)")
        print(f"  Test set: {len(test_subjects)} subjects ({len(patients_test)} patients, {len(controls_test)} controls)")
        
        # Create split dictionary
        data_splits = {
            'train': train_subjects,
            'validation': val_subjects,
            'test': test_subjects
        }
    else:
        data_splits = None
    
    # 6. Process each subject's data
    print("Organizing data by subject...")
    data_by_subject = {}
    max_audio_duration = 0
    min_audio_duration = float('inf')
    durations = []  
    for subject_id in tqdm(subject_ids):
        # Determine which group the subject belongs to
        group = 'patient' if subject_id.startswith('PZ') else 'control' if subject_id.startswith('CT') else 'unknown'
        
        # Determine which split this subject belongs to (if splitting)
        if split_data:
            if subject_id in train_subjects:
                split = 'train'
            elif subject_id in val_subjects:
                split = 'validation'
            elif subject_id in test_subjects:
                split = 'test'
            else:
                split = 'unknown'  # Shouldn't happen
        else:
            split = None
        
        # Create subject entry
        data_by_subject[subject_id] = {
            'subject_id': subject_id,
            'experiments': {},
            'group': group
        }
        
        if split_data:
            data_by_subject[subject_id]['split'] = split
        
        # Extract metadata for the subject from Excel
        subject_metadata = excel_df[excel_df['Subject_ID'] == subject_id]
        if not subject_metadata.empty:
            # Convert metadata to a dictionary, handling non-JSON serializable items
            metadata_dict = {}
            for k, v in subject_metadata.iloc[0].items():
                if pd.isna(v):
                    metadata_dict[k] = None
                elif isinstance(v, (np.integer, np.floating)):
                    metadata_dict[k] = float(v) if isinstance(v, np.floating) else int(v)
                else:
                    metadata_dict[k] = str(v)
            
            data_by_subject[subject_id]['metadata'] = metadata_dict
        else:
            print(f"Warning: No metadata found for subject {subject_id}")
            data_by_subject[subject_id]['metadata'] = None
        
        # Organize experiments for this subject
        
        for experiment in experiments:
            exp_files = [f for f in audio_files if f['subject_id'] == subject_id and f['experiment'] == experiment]
            if exp_files:
                # Load audio data
                wav_data = []
                for file_info in exp_files:
                    try:
                        # Load audio file
                        audio, sr = librosa.load(file_info['path'], sr=None)
                        
                        #Resample to 16kHz if necessary
                        if sr != TARGET_FS:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                            sr = TARGET_FS

                        # Extract basic audio features
                        duration = len(audio) / sr
                        if duration > max_audio_duration:
                            max_audio_duration = duration
                        if duration < min_audio_duration:
                            min_audio_duration = duration
                        durations.append(duration)
                        wav_data.append({
                            'filename': file_info['wav_file'],
                            'duration': duration,
                            'audio': audio.tolist(),  # Convert to list for JSON serialization
                            'sample_rate': sr,
                        })
                    except Exception as e:
                        print(f"Error processing {file_info['path']}: {str(e)}")
                
                data_by_subject[subject_id]['experiments'][experiment] = {
                    'wav_files': wav_data,
                    'count': len(wav_data)
                }
            else:
                data_by_subject[subject_id]['experiments'][experiment] = None
    
    # 7. Calculate summary statistics
    patient_count = len([s for s in data_by_subject.values() if s['group'] == 'patient'])
    control_count = len([s for s in data_by_subject.values() if s['group'] == 'control'])
    
    summary = {
        'total_subjects': len(data_by_subject),
        'patient_count': patient_count,
        'control_count': control_count,
        'experiments': experiments,
        'max_audio_duration': max_audio_duration,
        'min_audio_duration': min_audio_duration,
        'average_audio_duration': np.mean(durations) if durations else 0,
        'std_audio_duration': np.std(durations) if durations else 0,
        'dataset_stats': {
            'total_files': len(audio_files)
        }
    }
    
    # Add split information to summary if applicable
    if split_data:
        summary['data_splits'] = {
            'train': {
                'total': len(train_subjects),
                'patients': len(patients_train),
                'controls': len(controls_train)
            },
            'validation': {
                'total': len(val_subjects),
                'patients': len(patients_val),
                'controls': len(controls_val)
            },
            'test': {
                'total': len(test_subjects),
                'patients': len(patients_test),
                'controls': len(controls_test)
            }
        }
    
    # 8. Create final output structure
    output_data = {
        'summary': summary,
        'subjects': data_by_subject
    }
    
    # Include original split lists if splitting was done
    if split_data:
        output_data['splits'] = data_splits
        print(f"Saving processed data to {output_file}...")
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
    else:
        subject_list = list(subject_ids)
        num_subjects = len(subject_list)
        subjects_per_part = num_subjects // NUM_PARTS
        remainder = num_subjects % NUM_PARTS
        
        start_index = 0

        #output_keys = list(data_by_subject.keys())
        for i in range(NUM_PARTS):
            data_dict = {"audio": [], "phoneme": [], "speaker_id": [], "category": [], "alsfrs_total": [],
                    "alsfrs_speech": [], "disease_duration": [], "king_stage": [], "cantagallo": []
                 }
            end_index = start_index + subjects_per_part + (1 if i < remainder else 0)
            part_subjects = subject_list[start_index:end_index]
            print(f"  Part {i + 1}: {len(part_subjects)} subjects")
        
            for k in part_subjects:
                for experiment in experiments:
                    data_dict["audio"].append(data_by_subject[k]['experiments'][experiment]['wav_files'][0]['audio'])
                    data_dict["phoneme"].append(experiment.split("phonation")[-1] if "phonation" in experiment else experiment.split("rhythm")[-1])
                    data_dict["speaker_id"].append(k)
                    data_dict["category"].append(data_by_subject[k]['metadata']['Category'])
                    data_dict["alsfrs_total"].append(data_by_subject[k]['metadata']['ALSFRS-R_TotalScore_encoded'])
                    data_dict["alsfrs_speech"].append(data_by_subject[k]['metadata']['ALSFRS-R_SpeechSubscore'])
                    data_dict["disease_duration"].append(data_by_subject[k]['metadata']['DiseaseDuration_encoded'])
                    data_dict["king_stage"].append(data_by_subject[k]['metadata']['KingClinicalStage_encoded'])
                    data_dict["cantagallo"].append(data_by_subject[k]['metadata']['Cantagallo_Questionnaire_encoded'])
            
            data_dict = encode_audio_variables(data_dict, excel_df)

            output_file_part = output_file[:-8] + f"_part{i+1}.json.gz"
            print(f"Saving part {i + 1} of processed data to {output_file_part}...")
            with gzip.open(output_file_part, 'wt', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=2)
            start_index = end_index
            print(f"Processed data saved to {output_file_part}")

    
    print(f"Data saved to {output_file}")
    return max_audio_duration


def main():
    excel_file = os.path.join(BASE_DIR, "VOC-ALS.xlsx")
    output_file = os.path.join(OUTPUT_DIR, "voc_als_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Process data
    if SPLIT_DATA:
        output_file = os.path.splitext(output_file)[0] + "_split.json.gz"
        max_audio_duration = process_voc_als_data(
            BASE_DIR, 
            excel_file, 
            output_file, 
            split_data=True, 
            test_size= TEST_SIZE, 
            val_size= VAL_SIZE,
            random_seed= SEED
        )
    else:
        output_file = os.path.splitext(output_file)[0] + ".json.gz"
        max_audio_duration = process_voc_als_data(
            BASE_DIR,
            excel_file,
            output_file,
            split_data=False
        )

if __name__ == "__main__":
    main()