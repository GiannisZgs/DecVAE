"""
Dataset loading and initial processing for the SimCoupled dataset (scripts/simulations/simulated_coupled.py).
Returns DatasetDict objects for training, validation, testing and the independent-factors split.
"""
from datasets import Dataset, DatasetDict
import os
import json
import gzip
import numpy as np


"Generator split name -> DatasetDict split name"
SIM_COUPLED_SPLITS = {"train": "train", "dev": "validation", "test": "test", "indep": "indep"}


def sim_coupled_file_name(data_training_args, split):
    "Exact file name, so that e.g. SNR_10 cannot match SNR_100"
    return "sim_coupled_SNR_{:g}_{}_{:g}s.json.gz".format(
        data_training_args.sim_snr_db, split, data_training_args.sim_vowels_duration
    )


def load_sim_coupled(data_training_args):

    raw_datasets = DatasetDict()
    for split, dataset_split in SIM_COUPLED_SPLITS.items():
        file = os.path.join(data_training_args.data_dir, sim_coupled_file_name(data_training_args, split))
        with gzip.open(file, "rt") as f:
            data = json.load(f)
        data["audio"] = [np.array(arr) for arr in data["audio"]]
        raw_datasets[dataset_split] = Dataset.from_dict(data)

    if data_training_args.validation_split_percentage is not None:
        num_validation_samples = int(raw_datasets["train"].num_rows * data_training_args.validation_split_percentage // 100)
        raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
        raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))

    return raw_datasets
