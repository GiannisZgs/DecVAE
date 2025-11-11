"""
Dataset loading and initial processing for the VOC-ALS dataset for different scenarios.
Returns DatasetDict objects for training, validation, and testing.
"""
from datasets import Dataset, DatasetDict, concatenate_datasets
import os
import json
import gzip
import numpy as np
import pandas as pd

TARGET_PER_COMBINATION = 5

def load_voc_als(data_training_args):

    train_datasets_splits = []
    val_datasets_splits = []
    dev_datasets_splits = []
    test_datasets_splits = []

    n_train_files = 0
    "Load dataset files - Separate into 4 sets for better performance"
    if not data_training_args.train_val_test_split:
        files = [f for f in os.listdir(data_training_args.data_dir) if f != 'encodings.json']

        num_files = len(files)
        for file in files:
            if file != 'encodings.json': 
                file = os.path.join(data_training_args.data_dir,file)
                with gzip.open(file, "rt") as f:
                    data = json.load(f)
                if ".json" in file:
                    data["audio"] = [np.array(arr) for arr in data["audio"]]
                data = Dataset.from_dict(data)    
                train_datasets_splits.append(data)
                n_train_files+=1
                if n_train_files >= 1:
                    break
        "Load the rest of the files"
        for n,file in enumerate(files):
            if n < 1:
                continue
            file = os.path.join(data_training_args.data_dir,file)
            with gzip.open(file, "rt") as f:
                data = json.load(f)
            if ".json" in file:
                data["audio"] = [np.array(arr) for arr in data["audio"]]
            data = Dataset.from_dict(data)    
            val_datasets_splits.append(data)
            n_train_files+=1
            if n_train_files >= 2:
                break
        
        for n,file in enumerate(files):
            if n < 2:
                continue
            file = os.path.join(data_training_args.data_dir,file)
            with gzip.open(file, "rt") as f:
                data = json.load(f)
            if ".json" in file:
                data["audio"] = [np.array(arr) for arr in data["audio"]]
            data = Dataset.from_dict(data)    
            dev_datasets_splits.append(data)
            n_train_files+=1
            if n_train_files >= 3:
                break
        
        for n,file in enumerate(files):
            if n < 3:
                continue
            file = os.path.join(data_training_args.data_dir,file)
            with gzip.open(file, "rt") as f:
                data = json.load(f)
            if ".json" in file:
                data["audio"] = [np.array(arr) for arr in data["audio"]]
            data = Dataset.from_dict(data)    
            test_datasets_splits.append(data)
            n_train_files+=1
            if n_train_files >= 4:
                break

    else:
        dev_datasets_splits = []
        test_datasets_splits = []
        "Train splits"
        for file in os.listdir(data_training_args.data_dir):
            if "train" in file:
                file = os.path.join(data_training_args.data_dir,file)
                with gzip.open(file, "rt") as f:
                    data = json.load(f)
                if ".json" in file:
                    data["audio"] = [np.array(arr) for arr in data["audio"]]
                data = Dataset.from_dict(data)    
                train_datasets_splits.append(data)
                n_train_files+=1
        
        "Development splits"
        for file in os.listdir(data_training_args.data_dir):
            if "dev" in file:
                file = os.path.join(data_training_args.data_dir,file)
                with gzip.open(file, "rt") as f:
                    data = json.load(f)
                if ".json" in file:
                    data["audio"] = [np.array(arr) for arr in data["audio"]]
                data = Dataset.from_dict(data)    
                dev_datasets_splits.append(data)
                n_train_files+=1

        "Test splits"
        for file in os.listdir(data_training_args.data_dir):
            if "dev" in file:
                file = os.path.join(data_training_args.data_dir,file)
                with gzip.open(file, "rt") as f:
                    data = json.load(f)
                if ".json" in file:
                    data["audio"] = [np.array(arr) for arr in data["audio"]]
                data = Dataset.from_dict(data)    
                test_datasets_splits.append(data)
                n_train_files+=1

    
    raw_datasets = DatasetDict()
    if len(train_datasets_splits) > 1:
        raw_datasets["train"] = concatenate_datasets(train_datasets_splits).shuffle(seed=data_training_args.seed)
    else:
        raw_datasets["train"] = train_datasets_splits[0]

    if data_training_args.train_val_test_split:
        if data_training_args.validation_split_percentage is None:
            if len(dev_datasets_splits) > 1:
                raw_datasets["validation"] = concatenate_datasets(dev_datasets_splits).shuffle(seed=data_training_args.seed)
            else:
                raw_datasets["validation"] = dev_datasets_splits[0]    
        else:
            num_validation_samples = raw_datasets["train"].num_rows * data_training_args.validation_split_percentage // 100
            raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
            raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))

        if len(test_datasets_splits) > 1:
            raw_datasets["test"] = concatenate_datasets(test_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["test"] = test_datasets_splits[0]  
    else:
        if len(val_datasets_splits) > 1:
            raw_datasets["validation"] = concatenate_datasets(val_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["validation"] = val_datasets_splits[0]
        if len(dev_datasets_splits) > 1:
            raw_datasets["dev"] = concatenate_datasets(dev_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["dev"] = dev_datasets_splits[0]
        if len(test_datasets_splits) > 1:
            raw_datasets["test"] = concatenate_datasets(test_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["test"] = test_datasets_splits[0]
            
    return raw_datasets


def load_traversal_subset_voc_als(data_training_args):
    """
    Process and load the VOC_ALS dataset for latent traversal analysis.
    
    Creates traversal datasets with:
    1. Fixed phoneme, varying King's stage 
    2. Fixed King's stage, varying phoneme (not speaker)
    
    Args:
        data_training_args: Arguments for data processing
    
    Returns:
        DatasetDict containing traversal datasets
    """
    print(f"Processing VOC_ALS traversal dataset from {data_training_args.data_dir}")
    
    # Load all the data
    all_data = []
    files = [f for f in os.listdir(data_training_args.data_dir) if f != 'encodings.json']
    
    for file in files:
        file_path = os.path.join(data_training_args.data_dir, file)
        try:
            with gzip.open(file_path, "rt") as f:
                data = json.load(f)
            
            if ".json" in file_path:
                data["audio"] = [np.array(arr) for arr in data["audio"]]
            
            # Extract additional metadata if available
            if "phoneme" in data:
                phonemes = data["phoneme"]
            
            if "speaker_id" in data:
                speaker_ids = data["speaker_id"]
            
            if "king_stage" in data:
                king_stages = np.array(data["king_stage"])
                none_mask = np.equal(king_stages, None) 
                king_stages[none_mask] = -1
                king_stages = king_stages.tolist()  
            
            # Create individual entries for each sample
            for i in range(len(data["audio"])):
                entry = {
                    'audio': data["audio"][i],
                    'phoneme': phonemes[i] if i < len(phonemes) else "unknown",
                    'speaker_id': speaker_ids[i] if i < len(speaker_ids) else "unknown",
                    'king_stage': king_stages[i] if i < len(king_stages) else "unknown",
                    'file_source': file
                }
                
                # Add any other available fields
                for key in data:
                    if key not in ['audio', 'phoneme', 'speaker_id', 'king_stage'] and i < len(data[key]):
                        entry[key] = data[key][i]
                
                all_data.append(entry)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    if not all_data:
        raise ValueError("No data could be loaded for traversal analysis")
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(all_data)
    
    # Get unique values for each factor
    unique_phonemes = df['phoneme'].unique()
    unique_king_stages = df['king_stage'].unique()
    
    # Create empty lists for each traversal type
    fixed_phoneme_datasets = []
    fixed_king_stage_datasets = []
    
    # 1. Fixed phoneme, varying King's stage - MODIFIED to collect multiple examples
    for phoneme in unique_phonemes:
        if phoneme == "unknown":
            continue
            
        phoneme_examples = []
        king_stage_counts = {}  # Track count per king's stage
        
        # Filter entries with this phoneme
        phoneme_entries = df[df['phoneme'] == phoneme]
        
        for king_stage in unique_king_stages:
            if king_stage == "unknown" or king_stage == -1:
                continue
            
            # Find entries with this phoneme and king's stage
            matching_entries = phoneme_entries[phoneme_entries['king_stage'] == king_stage]
            
            # Collect up to TARGET_PER_COMBINATION examples
            count = 0
            for _, row in matching_entries.iterrows():
                if count >= TARGET_PER_COMBINATION:
                    break
                    
                example = row.to_dict()
                phoneme_examples.append(example)
                king_stage_counts[king_stage] = king_stage_counts.get(king_stage, 0) + 1
                count += 1
        
        # Only include if we have at least 2 different king's stages with examples
        king_stages_with_examples = len(king_stage_counts)
        if phoneme_examples and king_stages_with_examples >= 2:
            # Convert to Dataset
            fixed_phoneme_dataset = Dataset.from_dict({
                k: [example[k] for example in phoneme_examples if k in example]
                for k in phoneme_examples[0].keys()
            })
            
            fixed_phoneme_datasets.append({
                'dataset': fixed_phoneme_dataset,
                'fixed_factor': 'phoneme',
                'fixed_value': phoneme,
                'king_stage_counts': king_stage_counts,
                'total_examples': len(phoneme_examples),
                'king_stages_with_examples': king_stages_with_examples
            })
    
    # 2. Fixed King's stage, varying phoneme only (not speaker) - MODIFIED
    for king_stage in unique_king_stages:
        if king_stage == "unknown" or king_stage == -1:
            continue
            
        king_stage_examples = []
        phoneme_counts = {}  # Track count per phoneme
        
        # Filter entries with this King's stage
        king_stage_entries = df[df['king_stage'] == king_stage]
        
        for phoneme in unique_phonemes:
            if phoneme == "unknown":
                continue
            
            # Find entries with this King's stage and phoneme
            matching_entries = king_stage_entries[king_stage_entries['phoneme'] == phoneme]
            
            # Collect up to TARGET_PER_COMBINATION examples
            count = 0
            for _, row in matching_entries.iterrows():
                if count >= TARGET_PER_COMBINATION:
                    break
                    
                example = row.to_dict()
                king_stage_examples.append(example)
                phoneme_counts[phoneme] = phoneme_counts.get(phoneme, 0) + 1
                count += 1
        
        # Only include if we have at least 2 different phonemes with examples
        phonemes_with_examples = len(phoneme_counts)
        if king_stage_examples and phonemes_with_examples >= 2:
            # Convert to Dataset
            fixed_king_stage_dataset = Dataset.from_dict({
                k: [example[k] for example in king_stage_examples if k in example]
                for k in king_stage_examples[0].keys()
            })
            
            fixed_king_stage_datasets.append({
                'dataset': fixed_king_stage_dataset,
                'fixed_factor': 'king_stage',
                'fixed_value': king_stage,
                'phoneme_counts': phoneme_counts,
                'total_examples': len(king_stage_examples),
                'phonemes_with_examples': phonemes_with_examples
            })
    
    # Create the final datasets dictionary
    raw_datasets = DatasetDict()
    
    # Add the traversal datasets
    # 1. Fixed phoneme, varying King's stage
    if fixed_phoneme_datasets:
        phoneme_datasets = [d['dataset'] for d in fixed_phoneme_datasets]
        if phoneme_datasets:
            raw_datasets["fixed_phoneme"] = concatenate_datasets(phoneme_datasets)
            print(f"Created fixed_phoneme dataset with {raw_datasets['fixed_phoneme'].num_rows} examples")
    
    # 2. Fixed King's stage, varying phoneme only
    if fixed_king_stage_datasets:
        kings_stage_datasets = [d['dataset'] for d in fixed_king_stage_datasets]
        if kings_stage_datasets:
            raw_datasets["fixed_kings_stage"] = concatenate_datasets(kings_stage_datasets)
            print(f"Created fixed_kings_stage dataset with {raw_datasets['fixed_kings_stage'].num_rows} examples")
    
    # Store metadata about the traversals
    raw_datasets.traversal_metadata = {
        'fixed_phoneme': fixed_phoneme_datasets,
        'fixed_king_stage': fixed_king_stage_datasets,
        'target_per_combination': TARGET_PER_COMBINATION
    }
    
    # Print statistics
    total_phoneme_examples = sum(d['total_examples'] for d in fixed_phoneme_datasets)
    total_king_stage_examples = sum(d['total_examples'] for d in fixed_king_stage_datasets)
    
    print("\n===== VOC_ALS Traversal Dataset Statistics =====")
    print(f"Target examples per combination: {TARGET_PER_COMBINATION}")
    print(f"Fixed phoneme datasets: {len(fixed_phoneme_datasets)} phonemes, {total_phoneme_examples} total examples")
    print(f"Fixed king's stage datasets: {len(fixed_king_stage_datasets)} stages, {total_king_stage_examples} total examples")
    
    print("VOC_ALS traversal datasets created successfully.")
    return raw_datasets