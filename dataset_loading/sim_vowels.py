"""
Dataset loading and initial processing for the SimVowels dataset for different scenarios.
Returns DatasetDict objects for training, validation, and testing.
"""
from datasets import Dataset, DatasetDict, concatenate_datasets
import os
import json
import gzip
import numpy as np


def load_sim_vowels(data_training_args):

    train_datasets_splits = []
    val_datasets_splits = []
    test_datasets_splits = []
    n_train_files = 0

    "Train splits"
    for file in os.listdir(data_training_args.data_dir):
        if ("SNR_" + str(data_training_args.sim_snr_db)) in file \
            and ("vowels_" + str(data_training_args.sim_vowels_number)) in file \
                and (str(data_training_args.sim_vowels_duration) + "s") in file \
                    and "train" in file:   
            
            file = os.path.join(data_training_args.data_dir,file)
            with gzip.open(file, "rt") as f:
                data = json.load(f)
            #with open(file, "rb") as f:
            #    data = pickle.load(f)
            
            if ".json" in file:
                data["audio"] = [np.array(arr) for arr in data["audio"]]
            data = Dataset.from_dict(data)    
            train_datasets_splits.append(data)
            n_train_files+=1
            if n_train_files >= data_training_args.parts_to_use:
                break

    "Development splits"
    for file in os.listdir(data_training_args.data_dir):
        if ("SNR_" + str(data_training_args.sim_snr_db)) in file \
            and ("vowels_" + str(data_training_args.sim_vowels_number)) in file \
                and (str(data_training_args.sim_vowels_duration) + "s") in file \
                    and "dev" in file:
            file = os.path.join(data_training_args.data_dir,file)
            with gzip.open(file, "rt") as f:
                data = json.load(f)

            if ".json" in file:
                data["audio"] = [np.array(arr) for arr in data["audio"]]
            data = Dataset.from_dict(data) 
            #Convert to Datasets dataset
            val_datasets_splits.append(data)

    "Test splits"
    for file in os.listdir(data_training_args.data_dir):
        if ("SNR_" + str(data_training_args.sim_snr_db)) in file \
            and ("vowels_" + str(data_training_args.sim_vowels_number)) in file \
                and (str(data_training_args.sim_vowels_duration) + "s") in file \
                    and "test" in file:
            file = os.path.join(data_training_args.data_dir,file)
            with gzip.open(file, "rt") as f:
                data = json.load(f)
            #with open(file, "rb") as f:
            #    data = pickle.load(f)

            if ".json" in file:
                data["audio"] = [np.array(arr) for arr in data["audio"]]
            data = Dataset.from_dict(data) 
            #Convert to Datasets dataset
            test_datasets_splits.append(data)


    raw_datasets = DatasetDict()
    if len(train_datasets_splits) > 1:
        raw_datasets["train"] = concatenate_datasets(train_datasets_splits).shuffle(seed=data_training_args.seed)
    else:
        raw_datasets["train"] = train_datasets_splits[0]
    if data_training_args.validation_split_percentage is None:
        if len(val_datasets_splits) > 1:
            raw_datasets["validation"] = concatenate_datasets(val_datasets_splits).shuffle(seed=data_training_args.seed)
        else:
            raw_datasets["validation"] = val_datasets_splits[0]    
    else:
        num_validation_samples = raw_datasets["train"].num_rows * data_training_args.validation_split_percentage // 100
        raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
        raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))
    if len(test_datasets_splits) > 1:
        raw_datasets["test"] = concatenate_datasets(test_datasets_splits).shuffle(seed=data_training_args.seed)
    else:
        raw_datasets["test"] = test_datasets_splits[0]  

    return raw_datasets