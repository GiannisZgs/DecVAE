#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 Ioannis Ziogas <ziogioan@ieee.org>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Disentanglement evaluation utilities.
The herein code for the computation of disentanglement metrics has been adapted 
from google-research/disentanglement_lib in accordance to Apache-2.0 License.
https://github.com/google-research/disentanglement_lib.git
All utilities have been adapted to work without the use of TensorFlow.
"""

from disentanglement_utils import dci, irs, modularity_explicitness, kl_distance_mi
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score,adjusted_mutual_info_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from transformers import is_wandb_available
import os
import json
import time
import scipy.stats as stats

def compute_disentanglement_metrics(data_training_args,config,checkpoint, latent_type,mu_train, y_train, mu_test = None, y_test = None,target = None):
    """
    Main function for computing all disentanglement evaluation metrics.
    Args:
        data_training_args: Data and training related arguments.
        config: Model configuration object.
        checkpoint: Checkpoint identifier for saving results; affects only the name of the saved results files
        latent_type: Refers to the type of latent aggregation of individual subspaces to obtain final latent prior approximation.
            Can be 'all', 'X', 'OCs_joint', 'OCs_proj'; affects only the name of the saved results files.
        mu_train: Latent mu representations of the training set data.
        y_train: Ground truth labels for the training set.
        mu_test: Latent mu representations of the test set data (optional).
        y_test: Ground truth labels for the test set (optional).
        target: Target generative factors for evaluation (optional).

    If test data are not provided (mu_test, y_test), a train/test split will be performed on the train data.
    """
    
    if is_wandb_available() and data_training_args.with_wandb:
        import wandb
    
    target1 = target[0]
    if len(target) > 1:
        target2 = target[1]
        if len(target) > 2:
            target3 = target[2]
        else:
            target3 = ['None']
    else:
        target2 = ['None']
        target3 = ['None']

    latent_dim = mu_train.shape[1]
    colnames_X = ["X" + str(i) for i in range(latent_dim)]
    mu_train = pd.DataFrame(data = mu_train, columns = colnames_X)
    if mu_test is not None:
        mu_test = pd.DataFrame(data = mu_test, columns = colnames_X)

    if "vowels" in data_training_args.dataset_name:        
        if data_training_args.sim_vowels_number == 5:
            int_to_vowel = {
                '0': 'a', '1': 'e', '2': 'I', '3': 'aw', '4': 'u'
            }           
        elif data_training_args.sim_vowels_number == 8:
             int_to_vowel = {'0':'i','1':'I','2':'e','3':'ae','4':'a','5':'aw','6':'y','7':'u'}
        
        "Convert vowels to strings / categorical"       
        if "vowel" in target1 or "vowel" in target2:
            if data_training_args.discard_label_overlaps:
                vowels_categorical = [int_to_vowel[str(int(v))] for v in y_train["vowel"] if len(int_to_vowel[str(int(v))]) <= 2]
                if mu_test is not None:
                    vowels_categorical_test = [int_to_vowel[str(int(v))] for v in y_test["vowel"] if len(int_to_vowel[str(int(v))]) <= 2]
                corresp_inds = [i for i,v in enumerate(y_train["vowel"]) if len(int_to_vowel[str(int(v))]) <= 2]
                if mu_test is not None:
                    corresp_inds_test = [i for i,v in enumerate(y_test["vowel"]) if len(int_to_vowel[str(int(v))]) <= 2]
                "Discard the corresponding rows from the data"
                mu_train = mu_train.iloc[corresp_inds].reset_index(drop=True)
                if mu_test is not None:
                    mu_test = mu_test.iloc[corresp_inds_test].reset_index(drop=True)
            else:
                vowels_categorical = [int_to_vowel[str(int(v))] for v in y_train["vowel"]]
                if mu_test is not None:
                    vowels_categorical_test = [int_to_vowel[str(int(v))] for v in y_test["vowel"]]
            
            vowels_train = pd.DataFrame(data = vowels_categorical, columns = ["vowel"]).reset_index(drop=True)
            if mu_test is not None:
                vowels_test = pd.DataFrame(data = vowels_categorical_test, columns = ["vowel"]).reset_index(drop=True)            
        
        "Speakers"
        if "speaker" in target1 or "speaker" in target2:
            if "speaker" in target1:
                speaker_target = target1 #.copy()
            elif "speaker" in target2:
                speaker_target = target2 #.copy()
            sg_train = np.stack([(0.7,0.73),(0.78,0.81),(0.82,0.85),(0.86,0.89),(0.94,0.97),(1.02,1.05),(1.1,1.13),(1.14,1.17),(1.18,1.21),(1.26,1.29)])
            sg_dev = np.stack([(0.74,0.75),(0.9,0.91),(0.98,0.99),(1.06,1.07),(1.22,1.23)])
            sg_test = np.stack([(0.76,0.77),(0.92,0.93),(1.00,1.01),(1.08,1.09),(1.24,1.25)])
            speaker_groups = np.vstack([sg_train,sg_dev,sg_test])
            speakers_str = ['SP'+str(s+1) for s in range(speaker_groups.shape[0])]
            speakers_IDs = np.zeros_like(y_train[speaker_target],dtype='object')
            if mu_test is not None:
              speakers_IDs_test = np.zeros_like(y_test[speaker_target],dtype='object')
            for h,g in enumerate(speaker_groups):
                ix_L = np.where(y_train[speaker_target] >= g[0])[0]                      
                ix_U = np.where(y_train[speaker_target] < g[1])[0]
                ix0 = np.intersect1d(ix_L,ix_U)
                speakers_IDs[ix0] = speakers_str[h]
                if mu_test is not None:
                    ix_L_test = np.where(y_test[speaker_target] >= g[0])[0]
                    ix_U_test = np.where(y_test[speaker_target] < g[1])[0]
                    ix0_test = np.intersect1d(ix_L_test,ix_U_test)
                    speakers_IDs_test[ix0_test] = speakers_str[h]
            
            speakers_train = pd.DataFrame(data = speakers_IDs.astype('str'), columns = [speaker_target]).reset_index(drop=True)
            if mu_test is not None:
                speakers_test = pd.DataFrame(data = speakers_IDs_test.astype('str'), columns = [speaker_target]).reset_index(drop=True)
            if data_training_args.discard_label_overlaps and ("vowel" in target1 or "vowel" in target2):
                speakers_train = speakers_train.iloc[corresp_inds].reset_index(drop=True)
                if mu_test is not None:
                    speakers_test = speakers_test.iloc[corresp_inds_test].reset_index(drop=True)

            "Merge train and test labels to create a common encoding - Speakers are not common in train and test"
            y_speakers = np.concatenate((speakers_train,speakers_test),axis=0)


        "Convert label categories to numericals"
        if "vowel" in target1 or "vowel" in target2:
            le_vowels = LabelEncoder().fit(np.array(vowels_train).ravel())
            vowels_train = pd.DataFrame(data = le_vowels.transform(np.array(vowels_train).ravel()), columns=["vowel"])
            if mu_test is not None:
                vowels_test = pd.DataFrame(data = le_vowels.transform(np.array(vowels_test).ravel()), columns=["vowel"])
        if "speaker" in target1 or "speaker" in target2:
            le_speakers = LabelEncoder().fit(np.array(y_speakers).ravel())
            speakers_train = pd.DataFrame(data = le_speakers.transform(np.array(speakers_train).ravel()), columns=["speaker"])
            if mu_test is not None:
                speakers_test = pd.DataFrame(data = le_speakers.transform(np.array(speakers_test).ravel()), columns=["speaker"])

        if len(target) == 1 and "vowel" in target[0]:
            y_train = np.array(vowels_train).transpose()
            if mu_test is not None:
                y_test = np.array(vowels_test).transpose()
        elif len(target) == 1 and "speaker" in target[0]:
            y_train = np.array(speakers_train).transpose()
            if mu_test is not None:
                y_test = np.array(speakers_test).transpose()
        elif len(target) == 2:
            y_train = np.array(pd.concat([vowels_train,speakers_train],axis=1).transpose())
            if mu_test is not None:
                y_test = np.array(pd.concat([vowels_test,speakers_test],axis=1).transpose())
    
    elif "timit" in data_training_args.dataset_name:
        y_train = np.array(y_train[target]).transpose()
        if mu_test is not None:
            y_test = np.array(y_test[target]).transpose()
    elif data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
        y_train = np.array(y_train[target]).transpose()
        if y_test is not None:
            y_test = np.array(y_test[target]).transpose()

    "Make sure that all arrays are in the correct dimensions"
    mu_train = np.array(mu_train.transpose())
    if mu_test is not None:
        mu_test = np.array(mu_test.transpose())

    "Train/test split if test set is not provided"
    if mu_test is None and y_test is None:
        mu_train, mu_test = split_train_test(mu_train, data_training_args.train_data_percent)
        y_train, y_test = split_train_test(y_train, data_training_args.train_data_percent)

    "Create new train/test for supervised metrics - Speakers are not present in both sets"
    if ("speaker" in target1 or "speaker" in target2): # and 'emotion' not in target3:
        mus = np.concatenate((mu_train,mu_test),axis=1).transpose()
        ys = np.concatenate((y_train,y_test),axis=1).transpose()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=data_training_args.dev_data_percent,train_size=data_training_args.train_data_percent, random_state=42)
        "Split according to speakers"
        if "speaker" in target1:
            y_to_split = ys[:,0]
        elif "speaker" in target2:
            y_to_split = ys[:,1]
        for train_index, dev_index in sss.split(mus, y_to_split):
            mu_test = mus[dev_index,:]
            y_test = ys[dev_index,:]
            mu_train = mus[train_index,:]
            y_train = ys[train_index,:]
        mu_test = mu_test.transpose()
        mu_train = mu_train.transpose()
        y_test = y_test.transpose()
        y_train = y_train.transpose()
        
    elif "group" in target: # or 'emotion' in target3:
        "For VOC_ALS data, we need to stratify based on two or more variables"
        mus = np.concatenate((mu_train,mu_test),axis=1).transpose()
        ys = np.concatenate((y_train,y_test),axis=1).transpose()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=data_training_args.dev_data_percent,train_size=data_training_args.train_data_percent, random_state=42)
        "Split according to speakers"
        stratify_targets = np.zeros(ys.shape[0], dtype=int)
        unique_values1 = len(np.unique(ys[:,0]))  # Get the number of unique values in first factor
        unique_values2 = len(np.unique(ys[:,1]))  # Get the number of unique values in second factor
        if ys.shape[1] <= 2:
            stratify_targets =  ys[:,0] * unique_values1 + ys[:,1]
        else:
            # Create a combined encoding: factor1_val * max_factor2_vals + factor2_val
            stratify_targets = ys[:,0] * unique_values1 + ys[:,1]* unique_values2 + ys[:,2]
        print(f"Stratifying on {ys.shape[1]} factors combined")

        for train_index, dev_index in sss.split(mus, stratify_targets):
            mu_test = mus[dev_index,:]
            y_test = ys[dev_index,:]
            mu_train = mus[train_index,:]
            y_train = ys[train_index,:]
        mu_test = mu_test.transpose()
        mu_train = mu_train.transpose()
        y_test = y_test.transpose()
        y_train = y_train.transpose()

    "1. Unsupervised metrics"
    start_time_unsup = time.time()
    unsupervised_scores = kl_distance_mi.unsupervised_metrics(np.concatenate((mu_train,mu_test),axis=1), total_corr=True, wass_corr = False)
    end_time_unsup = time.time()
    elapsed_time_unsup = end_time_unsup - start_time_unsup
    print(f"Total Unsupervised metrics time: {elapsed_time_unsup: .4f} seconds")
    
    "2. DCI: Disentanglement, Completeness, Informativeness - Trains GBTs"
    "Cross-validation outside the DCI function"
    start_time_dci = time.time()
    if "vowels" in data_training_args.dataset_name:
        "No cross-validation in simulations"
        dci_scores = dci.compute_dci(mu_train, y_train, mu_test, y_test)
    else:
        mus_combined = np.concatenate((mu_train, mu_test), axis=1)
        ys_combined = np.concatenate((y_train, y_test), axis=1)

        cv_dci_results = []
        # For multi-factor cases, we'll stratify based on the first factor
        if ys_combined.shape[0] > 1:  # We have multiple factors
            # Create compound stratification target combining both factors
            stratify_targets = np.zeros(ys_combined.shape[1], dtype=int)
            unique_values1 = len(np.unique(ys_combined[0]))
            # Create a combined encoding: factor1_val * max_factor2_vals + factor2_val
            if ys_combined.shape[0] > 2:
                unique_values2 = len(np.unique(ys_combined[1]))
                stratify_targets = ys_combined[0] * unique_values1 + ys_combined[1] * unique_values2 + ys_combined[2]
            else:                
                stratify_targets = ys_combined[0] * unique_values1 + ys_combined[1]
            print(f"Stratifying on {ys_combined.shape[0]} factors combined")
        else:
            # Single factor - use it directly
            stratify_targets = ys_combined[0]
            print("Stratifying on single factor")

        skf = StratifiedKFold(n_splits=data_training_args.disentanglement_eval_cv_splits, shuffle=True, random_state=42)
        print(f"Starting {data_training_args.disentanglement_eval_cv_splits}-fold cross-validation for DCI metrics")
        for fold, (train_idx, test_idx) in enumerate(skf.split(mus_combined.T, stratify_targets)):
            print(f"\nComputing DCI - Fold {fold+1}/{data_training_args.disentanglement_eval_cv_splits}")
            
            # Split data for this fold
            fold_mu_train = mus_combined[:, train_idx]
            fold_y_train = ys_combined[:, train_idx]
            fold_mu_test = mus_combined[:, test_idx]
            fold_y_test = ys_combined[:, test_idx]
            
            # Calculate DCI for this fold
            fold_dci_scores = dci.compute_dci(fold_mu_train, fold_y_train, 
                                            fold_mu_test, fold_y_test)
            
            # Store results for this fold
            cv_dci_results.append(fold_dci_scores) 

        dci_metrics = list(cv_dci_results[0].keys())
        dci_scores = {}

        print("\nDCI Cross-validation results:")
        print("----------------------------")

        for metric in dci_metrics:
            values = [fold_result[metric] for fold_result in cv_dci_results]
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            # Calculate 95% confidence interval
            z_score = stats.norm.ppf(0.975)  # 95% CI
            ci = z_score * (std_value / np.sqrt(data_training_args.disentanglement_eval_cv_splits))
            
            # Store aggregated results
            dci_scores[f"{metric}"] = mean_value
            dci_scores[f"{metric}_std"] = std_value
            dci_scores[f"{metric}_ci"] = ci
            
            print(f"{metric}: {mean_value:.4f} ± {ci:.4f} (95% CI)")

        dci_scores["fold_results"] = [
            {k: float(v) if isinstance(v, (int, float, np.number)) else v 
            for k, v in fold.items()} 
            for fold in cv_dci_results
        ]
    end_time_dci = time.time()
    elapsed_time_dci = end_time_dci - start_time_dci
    print(f"Total DCI time: {elapsed_time_dci: .4f} seconds")

    "4. IRS: Interventional Robustness Score"
    start_time_irs = time.time()
    irs_dict = irs.compute_irs(mus_combined, ys_combined, diff_quantile=0.99)
    irs_score = {"IRS": irs_dict["IRS"]}
    end_time_irs = time.time()
    elapsed_time_irs = end_time_irs - start_time_irs
    print(f"Total IRS time: {elapsed_time_irs: .4f} seconds")
    
    
    if len(target) > 1:
        "5. Modularity and Explicitness - Trains logistic regression"
        start_time_mod_expl = time.time()
        if "vowels" not in data_training_args.dataset_name:
            mod_expl_scores = modularity_explicitness.compute_modularity_explicitness(mu_train, y_train, mu_test, y_test, 
                        use_cv=True,n_splits = data_training_args.disentanglement_eval_cv_splits,n_random_states = data_training_args.mod_expl_random_states)
        else:
            mod_expl_scores = modularity_explicitness.compute_modularity_explicitness(mu_train, y_train, mu_test, y_test, use_cv=False)
        end_time_mod_expl = time.time()
        elapsed_time_mod_expl = end_time_mod_expl - start_time_mod_expl
        print(f"Total Modularity-Explicitness time: {elapsed_time_mod_expl: .4f} seconds")
    
    "Save as .csv and log to wandb"
    if len(target) > 1:
        results = {**dci_scores, **irs_score, **mod_expl_scores, **unsupervised_scores} 
    else:
        results = {**dci_scores, **irs_score,  **unsupervised_scores} 
    current_result_dir = os.path.join(data_training_args.output_dir,checkpoint)
    if not os.path.exists(current_result_dir):
        os.makedirs(current_result_dir)

    if "seq" in target1 or "seq" in target2 or "seq" in target3:
        branch_type = "seq"
    else:
        branch_type = "frame"
    if "vae" in config.model_type or "VAE" in config.model_type:
        if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            fname = f'{config.model_type}_{latent_type}_z{config.vae_z_dim}_b{config.vae_beta}_{len(config.vae_kernel_sizes)}layers_disentanglement_results.csv'
        else:
            if 'None' in target2:
                fname = f'{config.model_type}_{latent_type}_{target1}_z{config.vae_z_dim}_b{config.vae_beta}_{len(config.vae_kernel_sizes)}layers_disentanglement_results.csv'
            else:
                fname = f'{config.model_type}_{latent_type}_{target1}_{target2}_z{config.vae_z_dim}_b{config.vae_beta}_{len(config.vae_kernel_sizes)}layers_disentanglement_results.csv'
    elif config.dual_branched_latent:
        model_type = "dual"

        if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            fname = f'{config.decomp_to_perform}_NoC{config.NoC}_bz{config.beta_kl_prior_z}_bs{config.beta_kl_prior_s}_{model_type}_{latent_type}_z{config.z_latent_dim}_s{config.s_latent_dim}_h{config.proj_codevector_dim_z}_disentanglement_results_{branch_type}.csv'
        else:
            if 'None' in target2:
                fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{config.beta_kl_prior_z}_bs{config.beta_kl_prior_s}_{model_type}_{latent_type}_{target1}_z{config.z_latent_dim}_s{config.s_latent_dim}_h{config.proj_codevector_dim_z}_disentanglement_results_{branch_type}.csv'
            else:
                fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{config.beta_kl_prior_z}_bs{config.beta_kl_prior_s}_{model_type}_{latent_type}_{target1}_{target2}_z{config.z_latent_dim}_s{config.s_latent_dim}_h{config.proj_codevector_dim_z}_disentanglement_results_{branch_type}.csv'
    elif config.only_z_branch:
        model_type = "single_z"
        if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            fname = f'{config.decomp_to_perform}_NoC{config.NoC}_bz{config.beta_kl_prior_z}_{model_type}_{latent_type}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}_disentanglement_results_{branch_type}.csv'
        else:
            if 'None' in target2:
                fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{config.beta_kl_prior_z}_{model_type}_{latent_type}_{target1}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}_disentanglement_results_{branch_type}.csv'
            else:
                fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{config.beta_kl_prior_z}_{model_type}_{latent_type}_{target1}_{target2}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}_disentanglement_results_{branch_type}.csv'
    elif config.only_s_branch:
        model_type = "single_s"
        if data_training_args.dataset_name in ["VOC_ALS", "iemocap"]:
            fname = f'{config.decomp_to_perform}_NoC{config.NoC}_bs{config.beta_kl_prior_s}_{model_type}_{latent_type}_s{config.s_latent_dim}_h{config.proj_codevector_dim_s}_disentanglement_results_{branch_type}.csv'
        else:
            fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bs{config.beta_kl_prior_s}_{model_type}_{latent_type}_{target1}_s{config.s_latent_dim}_h{config.proj_codevector_dim_s}_disentanglement_results_{branch_type}.csv'

    output_path = os.path.join(current_result_dir, fname)
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    if is_wandb_available() and data_training_args.with_wandb:
        for key, value in results.items():
            log_name = key+' '+latent_type+ ' '+target1  
            if not 'None' in target2:
                log_name += ' '+target2
            if not 'None' in target3:
                log_name += ' '+target3
            wandb.log({log_name: value})

    return results

def split_train_test(observations, train_percentage):
  """Splits observations into a train and test set.

  Args:
    observations: Observations to split in train and test. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]
  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test set."
  return observations_train, observations_test


def histogram_discretize(target, num_bins=30):
  """Discretization based on histograms."""
  discretized = np.zeros_like(target)
  for i in range(target.shape[0]):
    discretized[i, :] = np.digitize(target[i, :], np.histogram(
        target[i, :], num_bins)[1][:-1])
  return discretized

def is_discrete(array):
    return np.all(np.equal(np.mod(array, 1), 0))

def normalize_data(data, mean=None, stddev=None):
  if mean is None:
    mean = np.mean(data, axis=1)
  if stddev is None:
    stddev = np.std(data, axis=1)
  return (data - mean[:, np.newaxis]) / stddev[:, np.newaxis], mean, stddev


def discrete_mutual_info(mus, ys):
  """Compute discrete mutual information."""
  num_codes = mus.shape[0]
  num_factors = ys.shape[0]
  m = np.zeros([num_codes, num_factors])
  for i in range(num_codes):
    for j in range(num_factors):
      m[i, j] = adjusted_mutual_info_score(ys[j, :], mus[i, :])
      
  return m


def discrete_entropy(ys):
  """Compute discrete mutual information."""
  num_factors = ys.shape[0]
  h = np.zeros(num_factors)
  for j in range(num_factors):
    h[j] = mutual_info_score(ys[j, :], ys[j, :])
  return h
