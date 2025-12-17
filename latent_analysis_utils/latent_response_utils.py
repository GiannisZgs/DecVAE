"""
Utility functions for analyzing latent space responses to varying and fixed factors.
"""

import torch 
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import gzip


CANONICAL_FORMANT_FREQUENCIES = {
    'a':  [710, 1100, 2540],
    'e': [550, 1770, 2490],
    'I': [400, 1920, 2560],
    'aw': [590, 880, 2540],
    'u': [310, 870, 2250],
}

CANONICAL_FORMANT_FREQUENCIES_EXTENDED = {
    'i': [280, 2250, 2890],
    'I': [400, 1920, 2560],
    'e': [550, 1770, 2490],
    'ae': [690, 1660, 2490],
    'a': [710, 1100, 2540],
    'aw': [590, 880, 2540],
    'y': [450, 1030, 2380],
    'u': [310, 870, 2250],
}


VOWEL_TO_INT = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES.keys())}
INT_TO_VOWEL = {idx: vowel for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES.keys())}

VOWEL_TO_INT_EXTENDED = {vowel: idx for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES_EXTENDED.keys())}
INT_TO_VOWEL_EXTENDED = {idx: vowel for idx, vowel in enumerate(CANONICAL_FORMANT_FREQUENCIES_EXTENDED.keys())}

def analyze_latents_wrt_factor(mu, logvar, varying_factor_values, fixed_factor_values, latent_ids, fname, min_var_latents=1, varying="vowel"):
    """
    mu: latent space matrix
    logvar: log variance matrix
    varying_factor_values: values of the varying factor
    fixed_factor_values: values of the fixed factor
    varying: "vowel" or "speaker"
    """
    latent_dim = mu.shape[-1]
    if varying == "vowel":
        fixed_factor = "speaker"
    elif varying == "speaker":
        fixed_factor = "vowel"
        

    plt.figure(figsize=(16, latent_dim * 2))
    unique_factors = np.unique(varying_factor_values)
    unique_factors_fixed = np.unique(fixed_factor_values)
    colors = plt.cm.get_cmap('tab10', len(unique_factors))
    xaxis = np.linspace(min(fixed_factor_values).item(), max(fixed_factor_values).item(), len(unique_factors_fixed))
    mu_min, mu_max = torch.min(mu).item(), torch.max(mu).item()
   
    for i,z_id in enumerate(latent_ids):
        learnt_variance = logvar[:,i].exp().mean().item()
        plt.subplot(latent_dim, 1, i + 1)
        for j in range(len(unique_factors)):
            if varying == "vowel":
                if len(unique_factors) == 5:
                    varying_factor = INT_TO_VOWEL[unique_factors[j].item()]
                elif len(unique_factors) == 8:
                    varying_factor = INT_TO_VOWEL_EXTENDED[unique_factors[j].item()]
            elif varying == "speaker":
                varying_factor = np.round(unique_factors[j].item(),decimals = 3)
            factor_indices = np.where(varying_factor_values == unique_factors[j])[0]
            plt.plot(fixed_factor_values[factor_indices], mu[factor_indices, i], label=f"{varying}_{varying_factor}", color=colors(j))
            plt.ylim(mu_min - (mu_max-mu_min)/10 , mu_max + (mu_max-mu_min)/10)
            if varying == "vowel":
                xlabels = [f"SPK {k}" for k in range(len(unique_factors_fixed))]
                plt.xticks(ticks = xaxis,labels = xlabels, rotation=45)
            else:
                plt.xticks(ticks = xaxis,labels = unique_factors_fixed, rotation=45)
        if i >= len(latent_ids) - min_var_latents:
            plt.ylabel(f"$z_{{{z_id}}}$ - 'unused'")
        else:
            plt.ylabel(f"$z_{{{z_id}}}$")
        if i == 0:
            plt.title(f"Latent response across {'vowels' if varying=='vowel' else 'speakers'}")
        #plt.text(1.05, 0.5, f"Learned Variance(z_{z_id}): {learnt_variance:.3f}", transform=plt.gca().transAxes,
        #         fontsize=10, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel(fixed_factor)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 1])  # 0.95 to leave right side blank space
    plt.savefig(fname)

def visualize_latent_response(mu, factor, plottitle, fname,xlabel = None, ylabel = None):

    plt.figure(figsize=(6, 6))
    plt.plot(mu[:, 0], mu[:, 1], marker='o')
    for i, value in enumerate(factor):
        if type(value) == str:
            plt.text(mu[i, 0], mu[i, 1], value, fontsize=8, ha='right', va='bottom')
        else:
            plt.text(mu[i, 0], mu[i, 1], f"{np.round(value.item(),decimals=3)}", fontsize=8, ha='right', va='bottom')

    plt.title(plottitle)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(fname)

    return None

def calculate_variance_dimensions(mu, logvar, name):
    """
    Calculate and extract maximum and minimum variance dimensions for a latent space.
    
    Args:
        mu: The mean values tensor for the latent space
        logvar: The log variance tensor for the latent space
        name: Name of the latent space (for xlabel/ylabel)
    
    Returns:
        A dictionary containing all calculated values and dimensions
    """
    # Calculate dimensions with highest and lowest variance
    var_dims = np.argsort(-torch.var(mu, axis=0))[:7]
    min_var_dims = np.argsort(torch.var(mu, axis=0))[:3]
    
    # Extract reduced representations based on variance
    reduced = mu[:, var_dims]
    logvar_reduced = logvar[:, var_dims]
    reduced_min_var = mu[:, min_var_dims]
    logvar_reduced_min_var = logvar[:, min_var_dims]
    
    # Handle single dimension case
    if len(min_var_dims) == 1:
        reduced_min_var = reduced_min_var.unsqueeze(1)
        logvar_reduced_min_var = logvar_reduced_min_var.unsqueeze(1)
    
    # Concatenate max and min variance dimensions
    reduced_min_max_var = torch.cat((reduced, reduced_min_var), dim=1)
    logvar_reduced_min_max_var = torch.cat((logvar_reduced, logvar_reduced_min_var), dim=1)
    
    # Set labels for plotting
    xlabel, ylabel = f"z[{var_dims[0]}]", f"z[{var_dims[1]}]"
    
    # Combine variance dimensions
    if len(min_var_dims) == 1:
        var_dims = torch.cat((var_dims, min_var_dims.unsqueeze(0)))
    else:
        var_dims = torch.cat((var_dims, min_var_dims))
        
    #'reduced': reduced,
    #'logvar_reduced': logvar_reduced,
    #'reduced_min_var': reduced_min_var,
    #'logvar_reduced_min_var': logvar_reduced_min_var,
    return {
        'var_dims': var_dims,
        'min_var_dims': min_var_dims,
        'reduced_min_max_var': reduced_min_max_var,
        'logvar_reduced_min_max_var': logvar_reduced_min_max_var,
        'xlabel': xlabel,
        'ylabel': ylabel
    }

def save_latent_representation(latent_results, factors, fixed_factors, store_dir, latent_name, varying_factor, fixed_factor, dataset, num_vowels=None):
    """
    Save latent representation data to a JSON file.
    
    Args:
        latent_results: Dictionary containing latent space analysis results
        factors: The varying factor values (e.g., speaker_vt_factor or vowels)
        fixed_factors: The fixed factor values (e.g., vowels or speaker_vt_factor)
        store_dir: Directory to save the file
        latent_name: Name of the latent space (e.g., 'X', 'OC1', 'OCs_proj')
        varying_factor: What is varying ('speaker' or 'vowel')
        num_vowels: Number of vowels in the dataset (only for vowel experiments)
    
    Returns:
        Path to the saved file
    """
    fname = os.path.join(store_dir, f"{latent_name}_varying_{varying_factor}_fixed_{fixed_factor}.json")
    
    data_dict = {
        'mu': latent_results['reduced_min_max_var'].detach().cpu().numpy().tolist(),
        'logvar': latent_results['logvar_reduced_min_max_var'].detach().cpu().numpy().tolist(),
        'var_dims': latent_results['var_dims'].detach().cpu().numpy().tolist(),
        'min_var_latents': latent_results['min_var_dims'].detach().cpu().numpy().tolist(),
        'varying': [varying_factor],
        'latent': [latent_name]
    }

    if "vowels" in dataset:
        data_dict['speaker'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'speaker' else fixed_factors.detach().cpu().numpy().tolist()
        data_dict['vowel'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'vowel' else fixed_factors.detach().cpu().numpy().tolist()
    elif "timit" in dataset:
        data_dict['speaker'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'speaker' else fixed_factors.detach().cpu().numpy().tolist()
        data_dict['phoneme'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'phoneme' else fixed_factors.detach().cpu().numpy().tolist()
    elif "iemocap" in dataset:
        if 'emotion' in varying_factor or 'emotion' in fixed_factor:
            data_dict['emotion'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'emotion' else fixed_factors.detach().cpu().numpy().tolist()
        if 'speaker' in varying_factor or 'speaker' in fixed_factor:
            data_dict['speaker'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'speaker' else fixed_factors.detach().cpu().numpy().tolist()
        if varying_factor in ["phoneme","nonverbal"] or fixed_factor in ["phoneme","nonverbal"]:
            data_dict['phoneme'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'phoneme' else fixed_factors.detach().cpu().numpy().tolist()
    elif "VOC_ALS" in dataset:
        if 'phoneme' in varying_factor or 'phoneme' in fixed_factor:
            data_dict['phoneme'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'phoneme' else fixed_factors.detach().cpu().numpy().tolist()
        if 'king_stage' in varying_factor or 'king_stage' in fixed_factor:
            data_dict['king_stage'] = factors.detach().cpu().numpy().tolist() if varying_factor == 'king_stage' else fixed_factors.detach().cpu().numpy().tolist()
        
    if num_vowels is not None:
        data_dict['num_vowels'] = [num_vowels]
    
    with gzip.open(fname, "wt") as f:
        json.dump(data_dict, f)
        
    print(f"Saved latent representation {latent_name}_varying_{varying_factor}_fixed_{fixed_factor} to:", fname)
    
    return fname

def average_latent_representations(latent_spaces, factor_labels, experiment_type):
    """
    Average latent representations based on factor combinations.
    
    Args:
        latent_spaces: Dictionary mapping names to tensors {
            'X': (mu_X_z, logvar_X_z),
            'OC1': (mu_OC1_z, logvar_OC1_z),
            ...
        }
        factor_labels: Dictionary of factor label tensors {
            'phoneme': phonemes tensor,
            'speaker': speaker_id tensor,
            'emotion': emotion tensor (if applicable)
        }
        experiment_type: Type of experiment 
            (e.g., "fixed_phoneme", "fixed_emotion_phoneme_speaker")
        config: Model configuration with NoC and project_OCs settings
        
    Returns:
        Dictionary with averaged latent tensors and factor data
    """
    # Initialize results containers
    avg_mus = {}
    avg_logvars = {}
    
    # 2-factor case (like TIMIT, or simple IEMOCAP cases)
    if experiment_type in ["fixed_phoneme", "fixed_speaker", "fixed_phoneme_emotion", 
                           "fixed_speaker_emotion", "fixed_nonverbal_emotion", "fixed_kings_stage"]:
        
        # Determine fixed and varying factors based on experiment type
        if experiment_type == "fixed_phoneme":
            fixed_factor, varying_factor = "phoneme", "speaker"
            try: 
                fixed_values, varying_values = factor_labels["phoneme"], factor_labels["speaker"]
            except KeyError:
                fixed_factor, varying_factor = "phoneme", "king_stage"
                fixed_values, varying_values = factor_labels["phoneme"], factor_labels["king_stage"]
                
        elif experiment_type == "fixed_speaker":
            fixed_factor, varying_factor = "speaker", "phoneme"
            fixed_values, varying_values = factor_labels["speaker"], factor_labels["phoneme"]
        elif experiment_type == "fixed_phoneme_emotion":
            fixed_factor, varying_factor = "phoneme", "emotion" 
            fixed_values, varying_values = factor_labels["phoneme"], factor_labels["emotion"]
        elif experiment_type == "fixed_speaker_emotion":
            fixed_factor, varying_factor = "speaker", "emotion"
            fixed_values, varying_values = factor_labels["speaker"], factor_labels["emotion"]
        elif experiment_type == "fixed_nonverbal_emotion":
            fixed_factor, varying_factor = "nonverbal", "emotion"
            fixed_values, varying_values = factor_labels["phoneme"], factor_labels["emotion"]
        elif experiment_type == "fixed_kings_stage":
            fixed_factor, varying_factor = "king_stage", "phoneme"
            fixed_values, varying_values = factor_labels["king_stage"], factor_labels["phoneme"]
            
        # Remove padding values if needed
        if fixed_factor in ["phoneme", "nonverbal"]:
            unique_fixed = torch.unique(fixed_values)
            unique_fixed = unique_fixed[unique_fixed != -100]
        else:
            unique_fixed = torch.unique(fixed_values)
            
        unique_varying = torch.unique(varying_values)
        
        # Create pairs and map to indices
        unique_pairs = []
        pair_indices = {}
        
        for fixed_val in unique_fixed:
            for varying_val in unique_varying:
                pair = (fixed_val.item(), varying_val.item())
                mask = (fixed_values == fixed_val) & (varying_values == varying_val)
                if mask.sum() > 0:
                    indices = torch.where(mask)[0]
                    pair_indices[pair] = indices
                    unique_pairs.append(pair)
        
        # Calculate average representations for each latent space
        for latent_name, (mu, logvar) in latent_spaces.items():
            avg_mus[latent_name] = []
            avg_logvars[latent_name] = []
            
            for pair in unique_pairs:
                indices = pair_indices[pair]
                avg_mu = torch.mean(mu[indices], dim=0)
                avg_logvar = torch.mean(logvar[indices], dim=0)
                avg_mus[latent_name].append(avg_mu)
                avg_logvars[latent_name].append(avg_logvar)
        
        # Convert to tensors
        for latent_name in avg_mus.keys():
            if avg_mus[latent_name]:
                avg_mus[latent_name] = torch.stack(avg_mus[latent_name])
                avg_logvars[latent_name] = torch.stack(avg_logvars[latent_name])
        
        # Create tensors with factor values for each pair
        factor_tensors = {
            fixed_factor: torch.tensor([f for f, v in unique_pairs]),
            varying_factor: torch.tensor([v for f, v in unique_pairs])
        }
        
        return {
            'avg_mus': avg_mus,
            'avg_logvars': avg_logvars,
            'factor_tensors': factor_tensors,
            'unique_pairs': unique_pairs,
            'fixed_factor': fixed_factor,
            'varying_factor': varying_factor
        }
        
    # 3-factor case (IEMOCAP with fixed emotion, varying phoneme and speaker)
    elif experiment_type == "fixed_emotion_phoneme_speaker":
        # Get unique values for each factor
        all_emotions = torch.unique(factor_labels["emotion"])
        all_speakers = torch.unique(factor_labels["speaker"])
        all_phonemes = torch.unique(factor_labels["phoneme"])
        all_phonemes = all_phonemes[all_phonemes != -100]  # Remove padding value
        
        # Initialize dictionaries to store by emotion
        emotion_avg_mus = {}
        emotion_avg_logvars = {}
        emotion_phoneme_speaker_pairs = {}
        
        # Process each emotion
        for emotion_val in all_emotions:
            # Get indices for this emotion
            emotion_mask = (factor_labels["emotion"] == emotion_val)
            
            # Create pairs of (phoneme, speaker) for this emotion
            phoneme_speaker_pairs = []
            pair_indices = {}
            
            # Find all phoneme-speaker pairs for this emotion
            for p in all_phonemes:
                for s in all_speakers:
                    pair = (int(p.item()), s.item())
                    mask = emotion_mask & (factor_labels["phoneme"] == p) & (factor_labels["speaker"] == s)
                    if mask.sum() > 0:
                        indices = torch.where(mask)[0]
                        pair_indices[pair] = indices
                        phoneme_speaker_pairs.append(pair)
            
            # Store the pairs for this emotion
            emotion_phoneme_speaker_pairs[emotion_val.item()] = phoneme_speaker_pairs
            
            # Calculate average representations for each latent space
            emotion_avg_mus[emotion_val.item()] = {}
            emotion_avg_logvars[emotion_val.item()] = {}
            
            for latent_name, (mu, logvar) in latent_spaces.items():
                # Calculate average for each phoneme-speaker pair
                pair_mus = []
                pair_logvars = []
                
                for pair in phoneme_speaker_pairs:
                    indices = pair_indices[pair]
                    avg_mu = torch.mean(mu[indices], dim=0)
                    avg_logvar = torch.mean(logvar[indices], dim=0)
                    pair_mus.append(avg_mu)
                    pair_logvars.append(avg_logvar)
                
                if pair_mus:  # Check if we have any data
                    emotion_avg_mus[emotion_val.item()][latent_name] = torch.stack(pair_mus)
                    emotion_avg_logvars[emotion_val.item()][latent_name] = torch.stack(pair_logvars)
        
        return {
            'emotion_avg_mus': emotion_avg_mus,
            'emotion_avg_logvars': emotion_avg_logvars,
            'emotion_phoneme_speaker_pairs': emotion_phoneme_speaker_pairs,
            'all_emotions': all_emotions,
            'all_phonemes': all_phonemes,
            'all_speakers': all_speakers
        }
    
    else:
        raise ValueError(f"Unsupported experiment type: {experiment_type}")
