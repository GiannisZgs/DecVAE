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

"""Disentanglement metric suite on eval dumps.
Reproduces what compute_disentanglement_metrics does after its eval_dump_only point, for the
SimVowels path, by calling the same metric functions.
"""

import json
import os
import types
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from disentanglement_utils import dci, irs, modularity_explicitness, kl_distance_mi


def load_dump(path):
  """Load an eval dump written by eval_dump_only.

  Args:
    path: The .npz file, or its path without extension.
  Returns:
    mus (d, N), ys (F, N) in original order, n_train, and the sidecar dict.
  """
  stem = path[:-4] if path.endswith(".npz") else path
  d = np.load(stem + ".npz")
  with open(stem + ".json") as f:
    sidecar = json.load(f)
  mus = np.concatenate((d["mu_train"], d["mu_test"]), axis=1)
  ys = np.concatenate((d["y_train"], d["y_test"]), axis=1)
  return mus, ys, int(d["n_train"]), sidecar


def args_from_sidecar(sidecar, n_jobs=None):
  "The argument values the metrics read, as the data_training_args attributes they come from"
  return types.SimpleNamespace(
    train_data_percent=sidecar["train_data_percent"],
    dev_data_percent=sidecar["dev_data_percent"],
    disentanglement_num_workers=sidecar["disentanglement_num_workers"] if n_jobs is None else n_jobs,
  )


def split_row(target):
  "Row of ys the re-split is stratified on"
  if "speaker" in target[0]:
    return 0
  if len(target) > 1 and "speaker" in target[1]:
    return 1
  raise ValueError(f"The speaker re-split needs a speaker target, got {target}")


def resplit(mus, ys, target, args):
  """The stratified re-split of compute_disentanglement_metrics.

  Args:
    mus: (d, N) representations. ys: (F, N) factors.
  Returns:
    mu_train, y_train, mu_test, y_test, each (dims, points).
  """
  sss = StratifiedShuffleSplit(n_splits=1, test_size=args.dev_data_percent,
                               train_size=args.train_data_percent, random_state=42)
  train_index, dev_index = next(sss.split(mus.T, ys[split_row(target)]))
  return mus[:, train_index], ys[:, train_index], mus[:, dev_index], ys[:, dev_index]


def compute_metric_suite(mus, ys, n_train, target, args, random_state=0, unsupervised=True):
  """mus (d, N), ys (F, N) in original order. Returns the same keys as the results JSON.

  Args:
    n_train: Number of leading points that formed the train part of the dump.
    target: Factor names, one per row of ys.
    args: Object with train_data_percent, dev_data_percent and disentanglement_num_workers.
    random_state: Seed of the DCI gradient boosted trees.
    unsupervised: Whether to compute the Gaussian total correlation and mutual information.
  """
  mu_train, mu_test = mus[:, :n_train], mus[:, n_train:]
  y_train, y_test = ys[:, :n_train], ys[:, n_train:]
  mu_train, y_train, mu_test, y_test = resplit(np.concatenate((mu_train, mu_test), axis=1),
                                               np.concatenate((y_train, y_test), axis=1),
                                               target, args)
  n_jobs = args.disentanglement_num_workers

  unsupervised_scores = {}
  if unsupervised:
    unsupervised_scores = kl_distance_mi.unsupervised_metrics(np.concatenate((mu_train, mu_test), axis=1),
                                                              total_corr=True, wass_corr=False, n_jobs=n_jobs)

  dci_scores = dci.compute_dci(mu_train, y_train, mu_test, y_test, random_state=random_state)

  mus_combined = np.concatenate((mu_train, mu_test), axis=1)
  ys_combined = np.concatenate((y_train, y_test), axis=1)
  irs_score = {"IRS": irs.compute_irs(mus_combined, ys_combined, diff_quantile=0.99)["IRS"]}

  mod_expl_scores = {}
  if len(target) > 1:
    mod_expl_scores = modularity_explicitness.compute_modularity_explicitness(mu_train, y_train, mu_test, y_test,
                                                                              use_cv=False, n_jobs=n_jobs)

  return {**dci_scores, **irs_score, **mod_expl_scores, **unsupervised_scores}


def table_columns(results, gcn_key="gaussian_total_correlation"):
  "Results JSON keys renamed to the results-table columns"
  columns = {
    "MI": "mutual_info_score", "GCN": gcn_key,
    "DCI-D": "disentanglement", "DCI-C": "completeness", "DCI-I": "informativeness_test",
    "Modularity": "modularity_score", "Explicitness": "explicitness_score_test", "IRS": "IRS",
  }
  return {col: float(results[key]) for col, key in columns.items() if key in results}
