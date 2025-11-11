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

"""Compute latent disentanglement matrices based on mutual information. It can account for relations between generative factors 
and latent dimensions, as well as relations of latent dimensions tosingle instances of each factor (instance-level MI)."""

import matplotlib.pyplot as plt
import numpy as np
import os

def compute_disentanglement_matrices(latent, factor_dict, store_dir, latent_name, compute_3d_slices = False, save_visualizations=True):
    """
    Computes and returns both standard MI matrix and instance-level MI matrices for a latent.
    
    Args:
        latent: Latent representation tensor
        factor_dict: Dictionary of factors
        store_dir: Directory to save visualizations
        latent_name: Name of the latent space (e.g., 'X', 'OC1')
        save_visualizations: Whether to save visualization files
    
    Returns:
        Dictionary containing MI matrices and instance MI matrices
    """
    results = {}
    
    # Set filenames based on whether we want to save visualizations
    fname = os.path.join(store_dir, f"{latent_name}_mutual_info_matrix.jpg") if save_visualizations else None
    fname_instance = os.path.join(store_dir, f"{latent_name}_instance") if save_visualizations else None
    
    # Check if we're dealing with 2 or 3 factors
    if len(factor_dict) == 2:
        # Standard 2D case
        mi_matrix = compute_mi_matrix(latent, factor_dict, fname)
        instance_mi_matrices = compute_instance_mi_matrix(latent, factor_dict, fname_instance)
        
        results["mi_matrix"] = mi_matrix.tolist()
        results["instance_mi"] = {}
        for factor_name, matrix in instance_mi_matrices.items():
            results["instance_mi"][factor_name] = {
                "matrix": matrix.tolist(),
                "unique_values": np.unique(factor_dict[factor_name]).tolist()
            }
    elif len(factor_dict) == 3:
        # For 3 factors, compute pairwise 2D matrices using existing functions
        factor_names = list(factor_dict.keys())
        results["pairwise_mi"] = {}
        results["pairwise_instance_mi"] = {}
        
        # Skip speaker-king_stage combination for VOC_ALS dataset if needed
        skip_combination = None
        if "speaker" in factor_names and "king_stage" in factor_names:
            skip_combination = ("speaker", "king_stage")
        
        # Compute pairwise MI matrices
        for i, factor1 in enumerate(factor_names):
            for j, factor2 in enumerate(factor_names[i+1:], start=i+1):
                # Skip the specified combination if needed
                if skip_combination and ((factor1, factor2) == skip_combination or (factor2, factor1) == skip_combination):
                    print(f"Skipping {factor1}-{factor2} combination as requested")
                    continue
                
                # Create a subdictionary with just these two factors
                pair_dict = {
                    factor1: factor_dict[factor1],
                    factor2: factor_dict[factor2]
                }
                
                # Create filenames for this pair
                pair_key = f"{factor1}_{factor2}"
                pair_fname = os.path.join(store_dir, f"{latent_name}_{pair_key}_mi_matrix.jpg") if save_visualizations else None
                pair_fname_instance = os.path.join(store_dir, f"{latent_name}_{pair_key}_instance") if save_visualizations else None
                
                # Compute MI matrix and instance MI matrices for this pair using existing functions
                mi_matrix = compute_mi_matrix(latent, pair_dict, pair_fname)
                instance_mi_matrices = compute_instance_mi_matrix(latent, pair_dict, pair_fname_instance)
                
                # Store results
                results["pairwise_mi"][pair_key] = {
                    "mi_matrix": mi_matrix.tolist(),
                    "factors": [factor1, factor2]
                }
                
                # Store instance MI results
                results["pairwise_instance_mi"][pair_key] = {}
                for factor_name, matrix in instance_mi_matrices.items():
                    results["pairwise_instance_mi"][pair_key][factor_name] = {
                        "matrix": matrix.tolist(),
                        "unique_values": np.unique(factor_dict[factor_name]).tolist()
                    }
            
    else:
        raise ValueError(f"Unsupported number of factors: {len(factor_dict)}. Only 2 or 3 factors are supported.")
    
    if len(factor_dict) == 3 and compute_3d_slices:
        results["mi_matrix_3d"] = compute_mi_matrix_3d(latent, factor_dict, fname_instance)
        results["instance_mi_3d"] = compute_instance_mi_matrix_3d(latent, factor_dict, fname_instance)
    
    return results

def compute_mi_matrix(z, factor_dict,fname):
   
    z = z.cpu().numpy()
    factor_names = list(factor_dict.keys())
    latent_dim = z.shape[1]
    disent_matrix = np.zeros((latent_dim, len(factor_names)), dtype=float)

    for j, factor_name in enumerate(factor_names):
        y = np.array(factor_dict[factor_name])
        for i in range(latent_dim):
            discretized = np.round(z[:, i] * 50).astype(int)  # bin latents
            score = normalized_mutual_info_score(y, discretized)
            disent_matrix[i, j] = score
    
    if fname:
        plt.figure(figsize=(10, max(6, latent_dim * 0.5)))
        plt.imshow(disent_matrix, aspect='auto', cmap='viridis')
        plt.colorbar(label='R²')
        plt.xticks(np.arange(len(factor_names)), factor_names)
        plt.yticks(np.arange(latent_dim), [f"z[{i}]" for i in range(latent_dim)])
        plt.title("Disentanglement Matrix (R² scores)")
        plt.xlabel("Factor")
        plt.ylabel("Latent Dimension")
        plt.tight_layout()
        plt.savefig(fname)

    return disent_matrix

def compute_instance_mi_matrix(z, factor_dict, fname_prefix):
    """
    Computes mutual information between latent dimensions and specific instances of each factor.
    
    Args:
        z: Latent representations (numpy array or torch tensor)
        factor_dict: Dictionary with factor names as keys and pandas Series as values
        fname_prefix: Prefix for saving visualization files
    
    Returns:
        Dictionary of MI matrices for each factor, with instances as columns
    """
    if type(z) is not np.ndarray:
        z = z.cpu().numpy()

    mi_matrices = {}
    
    for factor_name, factor_values in factor_dict.items():
        # Get unique instances of this factor
        unique_instances = np.unique(factor_values)
        # Exclude -100 values which are padding/mask tokens
        if -100 in unique_instances:
            unique_instances = unique_instances[unique_instances != -100]
        
        latent_dim = z.shape[1]
        mi_matrix = np.zeros((latent_dim, len(unique_instances)), dtype=float)
        
        # For each unique instance, create a binary variable and compute MI
        for j, instance in enumerate(unique_instances):
            # Create binary variable (1 if factor value is this instance, 0 otherwise)
            binary_factor = (factor_values == instance).astype(int)
            
            for i in range(latent_dim):
                discretized = np.round(z[:, i] * 1000).astype(int)  # bin latents
                score = normalized_mutual_info_score(binary_factor, discretized)
                mi_matrix[i, j] = score
        
        # Store the matrix
        mi_matrices[factor_name] = mi_matrix
        
        if fname_prefix:
            # Visualize the matrix
            plt.figure(figsize=(max(8, len(unique_instances) * 0.4), max(6, latent_dim * 0.4)))
            plt.imshow(mi_matrix, aspect='auto', cmap='viridis')
            plt.colorbar(label='Normalized MI')
            plt.xticks(np.arange(len(unique_instances)), unique_instances, rotation=90)
            plt.yticks(np.arange(latent_dim), [f"z[{i}]" for i in range(latent_dim)])
            plt.title(f"MI between latents and {factor_name} instances")
            plt.xlabel(f"{factor_name} instances")
            plt.ylabel("Latent Dimension")
            plt.tight_layout()
            plt.savefig(f"{fname_prefix}_{factor_name}_instances_mi.jpg")
            plt.close()
    
    return mi_matrices
