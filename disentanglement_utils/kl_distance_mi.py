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

"""Unsupervised scores based on code covariance and mutual information.
Adapted from the Google Research disentanglement_lib (https://github.com/google-research/disentanglement_lib) in accordance to Apache-2.0 License."""
from disentanglement_utils import disentanglement_eval
import numpy as np
import scipy


def unsupervised_metrics(mus_train,total_corr = True,wass_corr = True):
  """Computes unsupervised scores based on covariance and mutual information.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """

  scores = {}
  
  num_codes = mus_train.shape[0]
  cov_mus = np.cov(mus_train)
  assert num_codes == cov_mus.shape[0]

  if total_corr:
    # Gaussian total correlation.
    scores["gaussian_total_correlation"], scores["gaussian_total_correlation_norm"] = gaussian_total_correlation(cov_mus)
  
  if wass_corr:
  # Gaussian Wasserstein correlation.
    scores["gaussian_wasserstein_correlation"] = gaussian_wasserstein_correlation(
        cov_mus)
    scores["gaussian_wasserstein_correlation_norm"] = (
        scores["gaussian_wasserstein_correlation"] / np.sum(np.diag(cov_mus)))

  # Compute average mutual information between different factors.
  mus_discrete = disentanglement_eval.histogram_discretize(mus_train,num_bins=30)
  mutual_info_matrix = disentanglement_eval.discrete_mutual_info(mus_discrete, mus_discrete)
  np.fill_diagonal(mutual_info_matrix, 0)
  mutual_info_score = np.sum(mutual_info_matrix) / (num_codes**2 - num_codes)
  scores["mutual_info_score"] = mutual_info_score
  return scores


def kl_gaussians_numerically_unstable(mean_0, cov_0, mean_1, cov_1, k):
  """Unstable version used for testing gaussian_total_correlation."""
  det_0 = np.linalg.det(cov_0)
  det_1 = np.linalg.det(cov_1)
  inv_1 = np.linalg.inv(cov_1)
  return 0.5 * (
      np.trace(np.matmul(inv_1, cov_0)) + np.dot(mean_1 - mean_0,
                                                 np.dot(inv_1, mean_1 - mean_0))
      - k + np.log(det_1 / det_0))


def gaussian_total_correlation(cov):
  """Computes the total correlation of a Gaussian with covariance matrix cov.

  We use that the total correlation is the KL divergence between the Gaussian
  and the product of its marginals. By design, the means of these two Gaussians
  are zero and the covariance matrix of the second Gaussian is equal to the
  covariance matrix of the first Gaussian with off-diagonal entries set to zero.

  Args:
    cov: Numpy array with covariance matrix.

  Returns:
    Scalar with total correlation.
    Scalar with normalized (to account for the dimension) total correlation.
  """
  tc_raw = 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])

  dim = cov.shape[0]  # Latent space dimensionality
  avg_covariance = np.mean(np.diag(cov))  # Average variance
  
  tc_norm = tc_raw / (dim * np.abs(np.log(avg_covariance + 1e-8)))
    
  return tc_raw, tc_norm

  return

def gaussian_wasserstein_correlation(cov):
  """Wasserstein L2 distance between Gaussian and the product of its marginals.

  Args:
    cov: Numpy array with covariance matrix.

  Returns:
    Scalar with score.
  """
  sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
  return 2 * np.trace(cov) - 2 * np.trace(sqrtm)
