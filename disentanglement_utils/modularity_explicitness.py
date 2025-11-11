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

"""Modularity and explicitness metrics from the F-statistic paper.

Based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss"
(https://arxiv.org/pdf/1802.05312.pdf).

Adapted from the Google Research disentanglement_lib (https://github.com/google-research/disentanglement_lib) in accordance to Apache-2.0 License.
"""

from disentanglement_utils import disentanglement_eval
import numpy as np
#from six.moves import range
from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Ignore potential division by zero
  

def compute_modularity_explicitness(mus_train, ys_train, mus_test, ys_test, use_cv = False, n_splits = 5,n_random_states = 5):
  """Computes the modularity metric according to Sec 3.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with average modularity score and average explicitness
      (train and test).
  """

  scores = {}
  "Calculate modularity on whole set"
  mus_combined = np.concatenate((mus_train, mus_test), axis=1)
  ys_combined = np.concatenate((ys_train, ys_test), axis=1)
  discretized_mus = disentanglement_eval.histogram_discretize(mus_combined)
  mutual_information = disentanglement_eval.discrete_mutual_info(discretized_mus, ys_combined)
  # Mutual information should have shape [num_codes, num_factors].
  assert mutual_information.shape[0] == mus_train.shape[0]
  assert mutual_information.shape[1] == ys_train.shape[0]
  scores["modularity_score"] = modularity(mutual_information)
  
  if use_cv:
    "Calculate explicitness in KFold CV with random states"
    mus_combined_T = mus_combined.T
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
      # Single factor
      stratify_targets = ys_combined.copy()
      print("Stratifying on single factor")

    explicitness_train_scores = []
    explicitness_test_scores = []
    print(f"Starting {n_splits}-fold CV with {n_random_states} random states for modularity & explicitness")
    for rs in range(n_random_states):
      print(f"\nRandom state {rs+1}/{n_random_states}")
      
      # Setup cross-validation with this random state
      skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
      
      # For each fold
      for fold, (train_idx, test_idx) in enumerate(skf.split(mus_combined_T, stratify_targets)):
        print(f"  Processing fold {fold+1}/{n_splits}")
        
        # Split data for this fold
        fold_mus_train = mus_combined[:, train_idx]
        fold_ys_train = ys_combined[:, train_idx]
        fold_mus_test = mus_combined[:, test_idx]
        fold_ys_test = ys_combined[:, test_idx]
        
        # Calculate explicitness for this fold
        explicitness_score_train = np.zeros([fold_ys_train.shape[0], 1])
        explicitness_score_test = np.zeros([fold_ys_test.shape[0], 1])
        mus_train_norm, mean_mus, stddev_mus = disentanglement_eval.normalize_data(fold_mus_train)
        mus_test_norm, _, _ = disentanglement_eval.normalize_data(fold_mus_test, mean_mus, stddev_mus)

        for i in range(fold_ys_train.shape[0]):
          explicitness_score_train[i], explicitness_score_test[i] = \
            explicitness_per_factor(mus_train_norm, fold_ys_train[i, :],
                                    mus_test_norm, fold_ys_test[i, :])

        fold_explicitness_train = (np.mean(explicitness_score_train) - 0.5) / 0.5
        fold_explicitness_test = (np.mean(explicitness_score_test) - 0.5) / 0.5
      
        explicitness_train_scores.append(fold_explicitness_train)
        explicitness_test_scores.append(fold_explicitness_test)
      
        print(f"    Explicitness (train): {fold_explicitness_train:.4f}")
        print(f"    Explicitness (test): {fold_explicitness_test:.4f}")

    "Calculate statistics"
    explicitness_train_mean = np.mean(explicitness_train_scores)
    explicitness_train_std = np.std(explicitness_train_scores)
    explicitness_test_mean = np.mean(explicitness_test_scores)
    explicitness_test_std = np.std(explicitness_test_scores)
    
    # Calculate 95% confidence intervals
    n_samples = n_splits * n_random_states
    z_score = stats.norm.ppf(0.975)  # 95% CI
    
    explicitness_train_ci = z_score * (explicitness_train_std / np.sqrt(n_samples))
    explicitness_test_ci = z_score * (explicitness_test_std / np.sqrt(n_samples))

    scores.update({
        "explicitness_score_train": explicitness_train_mean,
        "explicitness_train_std": explicitness_train_std,
        "explicitness_train_ci": explicitness_train_ci, 
        "explicitness_train_values": explicitness_train_scores,
        "explicitness_score_test": explicitness_test_mean,
        "explicitness_test_std": explicitness_test_std,
        "explicitness_test_ci": explicitness_test_ci,
        "explicitness_test_values": explicitness_test_scores,
    })

    print("\nExplicitness Cross-validation Results:")
    print("------------------------------------------------")
    print(f"Explicitness (train): {explicitness_train_mean:.4f} ± {explicitness_train_ci:.4f} (95% CI)")
    print(f"Explicitness (test): {explicitness_test_mean:.4f} ± {explicitness_test_ci:.4f} (95% CI)")
    
    return scores    

  else:
    "No cross-validation, calculate explicitness on the pre-defined train/test sets"
    explicitness_score_train = np.zeros([ys_train.shape[0], 1])
    explicitness_score_test = np.zeros([ys_test.shape[0], 1])
    mus_train_norm, mean_mus, stddev_mus = disentanglement_eval.normalize_data(mus_train)
    mus_test_norm, _, _ = disentanglement_eval.normalize_data(mus_test, mean_mus, stddev_mus)
    for i in range(ys_train.shape[0]):
      explicitness_score_train[i], explicitness_score_test[i] = \
          explicitness_per_factor(mus_train_norm, ys_train[i, :],
                                  mus_test_norm, ys_test[i, :])
    scores["explicitness_score_train"] = (np.mean(explicitness_score_train) - 0.5) / (0.5)
    scores["explicitness_score_test"] = (np.mean(explicitness_score_test) - 0.5) / (0.5)
  
  return scores


def explicitness_per_factor(mus_train, y_train, mus_test, y_test):
  """Compute explicitness score for a factor as ROC-AUC of a classifier.

  Args:
    mus_train: Representation for training, (num_codes, num_points)-np array.
    y_train: Ground truth factors for training, (num_factors, num_points)-np
      array.
    mus_test: Representation for testing, (num_codes, num_points)-np array.
    y_test: Ground truth factors for testing, (num_factors, num_points)-np
      array.

  Returns:
    roc_train: ROC-AUC score of the classifier on training data.
    roc_test: ROC-AUC score of the classifier on testing data.
  """
  x_train = np.transpose(mus_train)
  x_test = np.transpose(mus_test)
  clf = linear_model.LogisticRegression().fit(x_train, y_train)
    
  test_classes = np.unique(y_test)
    
  # Filter predictions to only include classes present in test set
  y_pred_train = clf.predict_proba(x_train)
  y_pred_test = clf.predict_proba(x_test)
  
  # Get indices of classes present in test set
  class_indices = [np.where(clf.classes_ == c)[0][0] for c in test_classes]
  
  # Filter predictions to only include classes present in test set
  y_pred_train_filtered = y_pred_train[:, class_indices]
  y_pred_test_filtered = y_pred_test[:, class_indices]
  
  # Transform labels with only test classes
  mlb = preprocessing.MultiLabelBinarizer(classes=test_classes)
  y_train_bin_enc = mlb.fit_transform(np.expand_dims(y_train, 1))
  y_test_bin_enc = mlb.transform(np.expand_dims(y_test, 1))

  # Calculate ROC AUC scores
  roc_train = metrics.roc_auc_score(y_train_bin_enc, y_pred_train_filtered)
  roc_test = metrics.roc_auc_score(y_test_bin_enc, y_pred_test_filtered)
  
  return roc_train, roc_test


def modularity(mutual_information):
  """Computes the modularity from mutual information."""
  # Mutual information has shape [num_codes, num_factors].
  squared_mi = np.square(mutual_information)
  max_squared_mi = np.max(squared_mi, axis=1)
  numerator = np.sum(squared_mi, axis=1) - max_squared_mi
  denominator = max_squared_mi * (squared_mi.shape[1] -1.)
  delta = numerator / denominator
  modularity_score = 1. - delta
  index = (max_squared_mi == 0.)
  modularity_score[index] = 0.
  return np.mean(modularity_score)
