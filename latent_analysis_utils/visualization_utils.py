"""
Utilities for visualizing latent spaces in a 2D/3D manifold using manifold learning algorithms.    
"""
import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from transformers import is_wandb_available
import json
from sklearn.manifold import TSNE
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

ANGLES = [45, 135, 225, 315]  # angles for taking screenshots of the 3D rotating plots

VIS_METHODS = {
    'umap': umap.UMAP(n_components=2, random_state=42, metric = 'euclidean',n_neighbors=10,min_dist=0.1,densmap=False),
    'tsne': TSNE(n_components=2, random_state=42, learning_rate= 500, 
                  max_iter = 1000, perplexity=30, metric='euclidean',early_exaggeration=10,
                  init='pca'),
} 

def filter_yellow_colors(color_list):
    """Filter out colors that are too close to bright yellow"""
    filtered_colors = []
    for c in color_list:
        #bright yellow: (high R, high G, low B)
        if not (c[0] > 0.7 and c[1] > 0.7 and c[2] < 0.3):
            filtered_colors.append(c)
    return filtered_colors

def save_animation_screenshot(fig, ax, angle, save_path, dpi=600):
    """
    Save a single frame of a 3D animation at a specific viewing angle
    
    Args:
        fig: matplotlib figure object
        ax: matplotlib 3D axis
        angle: azimuth angle in degrees
        save_path: path to save the screenshot
        dpi: resolution of the saved image
    """
    # Set the viewing angle
    ax.view_init(elev=10., azim=angle)
    
    # Ensure the figure is rendered
    fig.canvas.draw()
    
    # Save the current view as an image
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight',pad_inches=1)
    
    #print(f"Screenshot saved at {save_path}")


def visualize(data_training_args, config,X,OCs,y_vec,z_or_h,data_set,target,manifold_dict = None, save_dir=None):
    """
    Utilities for visualizing latent spaces in a 2D/3D manifold using manifold learning algorithms.    
    Visualizations are created to showcase frequency-resonant latent subspaces and final latent space (prior approximation), as well as generative factors-related
    latent subspaces and final latent space (prior approximation).  
    This function handles data organization, label encoding, visualization and saving of plots.
    If data_training_args.vis_sphere is True, it will also sample 1000 points from a multivariate isotropic Gaussian distribution,
    project them along with the X,OCs data, and plot them together to visualize how the latent space aligns with a standard Gaussian prior.
    Args:
        data_training_args: Data and training related arguments.
        config: Model configuration object.
        X: Latent representations of the anchor subspace component that usually corresponds to the original signal - can be empty.
        OCs: Latent representations from decomposed latent subspaces - can be empty.
        y_vec: Target labels for coloring the plots.
        z_or_h: Indicates whether we are visualizing latent variables 'z' or hidden representations 'h'. Used for naming the saved plots.
        data_set: Dataset split being visualized ('train', 'dev', 'test').
        target: Target variable name for coloring the plots.
        manifold_dict: Dictionary of manifold learning methods (optional). If not provided a default will be used based on 
            which manifold method has been specified in data_training_args.vis_method.
        display_figures: If True, displays the generated figures.
        save_dir: Directory to save the generated plots (optional).
    """
    if "seq" in target:
        NoC = config.NoC_seq
    else:
        NoC = config.NoC

    if is_wandb_available() and data_training_args.with_wandb:
        import wandb
    if X is not None:
        if X.device == 'cuda':
            X = X.cpu()
    if OCs is not None:
        if OCs.device == 'cuda':
            OCs = OCs.cpu()
    if type(y_vec) == torch.Tensor:
        if y_vec.device == 'cuda':
            y_vec = y_vec.cpu()    

    scaler = StandardScaler()
    
    if X is not None:
        X_scaled = scaler.fit_transform(X)
        latent_dim = X.shape[1]
        colnames_X = ["X" + str(i) for i in range(latent_dim)]
        X = pd.DataFrame(data = X_scaled, columns = colnames_X)
    if OCs is not None:
        #Below scales OCs collectively
        #OCs_scaled = scaler.fit_transform(OCs.view(-1,OCs.shape[-1]))
        #latent_dim_OCs = OCs.shape[-1]
        #colnames_OCs = ["OC" + str(i) for i in range(latent_dim_OCs)]
        #OCs = pd.DataFrame(data = OCs_scaled, columns = colnames_OCs)
        OCs_scaled = []
        for oc in range(OCs.shape[0]):
            scaler_oc = StandardScaler()
            oc_scaled = scaler_oc.fit_transform(OCs[oc])
            OCs_scaled.append(oc_scaled)
        OCs_scaled = np.stack(OCs_scaled)
        latent_dim_OCs = OCs.shape[-1]
        colnames_OCs = ["OC" + str(i) for i in range(latent_dim_OCs)]
        OCs = pd.DataFrame(data = OCs_scaled.reshape(-1, latent_dim_OCs), columns = colnames_OCs)


    if "vowels" in data_training_args.dataset_name:        
        if data_training_args.sim_vowels_number == 5:
            int_to_vowel = {
                '0': 'a', '1': 'e', '2': 'I', '3': 'aw', '4': 'u'
            }           
        elif data_training_args.sim_vowels_number == 8:
             int_to_vowel = {'0':'i','1':'I','2':'e','3':'ae','4':'a','5':'aw','6':'y','7':'u'}
        "Convert vowels to strings / categorical"
        if "vowel" in target:
            vowels_categorical = [int_to_vowel[str(v.item())] for v in y_vec]
            labels = pd.DataFrame(data = vowels_categorical, columns = [target]).reset_index(drop=True)
    
        elif "speaker" in target:
            "Speakers - No need to discard overlaps"
            y_vec = y_vec.detach().cpu()
            sg_train = np.stack([(0.7,0.73),(0.78,0.81),(0.82,0.85),(0.86,0.89),(0.94,0.97),(1.02,1.05),(1.1,1.13),(1.14,1.17),(1.18,1.21),(1.26,1.29)])
            sg_dev = np.stack([(0.74,0.75),(0.9,0.91),(0.98,0.99),(1.06,1.07),(1.22,1.23)])
            sg_test = np.stack([(0.76,0.77),(0.92,0.93),(1.00,1.01),(1.08,1.09),(1.24,1.25)])
            speaker_groups = np.vstack([sg_train,sg_dev,sg_test])
            speakers_str = ['SP'+str(s+1) for s in range(speaker_groups.shape[0])]
            speakers_IDs = np.zeros_like(y_vec,dtype='object')
            for h,g in enumerate(speaker_groups):
                ix_L = np.where(y_vec >= g[0])[0]                      
                ix_U = np.where(y_vec < g[1])[0]
                ix0 = np.intersect1d(ix_L,ix_U)
                speakers_IDs[ix0] = speakers_str[h]
        
            labels = pd.DataFrame(data = speakers_IDs.astype('str'), columns = [target]).reset_index(drop=True)

        "Convert label categories to numericals"
        y_vec = np.array(labels)

    elif "timit" in data_training_args.dataset_name:  
        if type(y_vec) == torch.Tensor:
            y_vec = y_vec.detach().cpu()
        if y_vec.dtype == object:
            "In this case the below conversions to phoneme39 can be skipped"
            pass
        else:
            "First discard values not found"
            disc = np.where(y_vec == -100)[0]
            y_vec = np.delete(y_vec,disc)
            if X is not None:
                X = X.drop(disc).reset_index(drop=True)
            if OCs is not None:
                OCs = OCs.drop(disc).reset_index(drop=True)
            
            if 'phoneme48' in target:
                label_map_48_to_39 = {'cl':'sil',
                                        'vcl':'sil',
                                        'epi':'sil',
                                        'el':'l',
                                        'en':'n',
                                        'zh':'sh',
                                        'aa':'ao',
                                        'ih':'ix',
                                        'ah':'ax',                            
                                    } 
                
                with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'r') as json_file:
                    phoneme48_to_id = json.load(json_file)
                with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                    phoneme39_to_id = json.load(json_file)
                    id_to_phoneme39 = {v: k for k, v in phoneme39_to_id.items()}


                def id48_to_id39(y):
                    "Invert the dictionary"
                    id_to_phoneme48 = {v: k for k, v in phoneme48_to_id.items()}
                    "First map integers to phonemes48"
                    if type(y) == torch.Tensor:
                        y_str = [id_to_phoneme48[label.item()] for label in y]
                    else:
                        y_str = [id_to_phoneme48[label] for label in y]
                    "Then map phonemes48 to phonemes39"
                    y_39 = [a if a not in label_map_48_to_39 else label_map_48_to_39[a] for a in y_str]
                    "Then map phonemes39 to integers"
                    y_39_id = [phoneme39_to_id[label] for label in y_39]
                    return y_39_id

                y_39 = pd.DataFrame(data = np.array(id48_to_id39(y_vec)), columns=["phoneme39"])            
                y_39 = y_39.astype({'phoneme39': int})

            elif 'phoneme39' in target:
                with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
                    phoneme39_to_id = json.load(json_file)
                    id_to_phoneme39 = {v: k for k, v in phoneme39_to_id.items()}
                y_39_id = [phoneme39_to_id[label] for label in y_vec]
                y_39 = pd.DataFrame(data = np.array(y_39_id), columns=["phoneme39"])            
                y_39 = y_39.astype({'phoneme39': int})

            if type(y_vec) == torch.Tensor:
                y_vec = pd.DataFrame(data = y_vec.detach().cpu(), columns=[target])
                y_vec = y_vec.astype({target: int})

        #if 'phoneme' in target:
        #    y_for_map = pd.concat([y,y_39],axis=1)

    elif "iemocap" in data_training_args.dataset_name:  
        if type(y_vec) == torch.Tensor:
            y_vec = y_vec.detach().cpu()
        if y_vec.dtype == object:
            "In this case the below conversions to phoneme can be skipped"
            pass
        else:
            "First discard values not found"
            disc = np.where(y_vec == -100)[0]
            y_vec = np.delete(y_vec,disc)
            if X is not None:
                X = X.drop(disc).reset_index(drop=True)
            if OCs is not None:
                OCs = OCs.drop(disc).reset_index(drop=True)
            
            if 'phoneme' in target:
                
                with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
                    phoneme_to_id = json.load(json_file)
                id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}

            elif 'speaker' in target:
                with open(data_training_args.path_to_iemocap_speaker_dict_file, 'r') as json_file:
                    speaker_to_id = json.load(json_file)
                id_to_speaker = {v: k for k, v in speaker_to_id.items()}
                "Speaker is already in int format"
            
            elif 'emotion' in target:
                with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'r') as json_file:
                    emotion_to_id = json.load(json_file)
                id_to_emotion = {v: k for k, v in emotion_to_id.items()}
                "Emotion is already in int format"

            if type(y_vec) == torch.Tensor:
                y_vec = pd.DataFrame(data = y_vec.detach().cpu(), columns=[target])
                y_vec = y_vec.astype({target: int})


    elif "VOC_ALS" in data_training_args.dataset_name:
        "Load the encodings dictionary for the VOC ALS dataset"
        with open(data_training_args.path_to_voc_als_encodings, 'r') as json_file:
            encodings = json.load(json_file)
        
        if type(y_vec) == torch.Tensor:
            y_vec = y_vec.detach().cpu()

        if 'phoneme' in target:
            label_map = encodings['phoneme']
        elif 'speaker' in target:
            label_map = encodings['speaker_id']
        elif 'cantagallo' in target:
            label_map = encodings['Cantagallo_Questionnaire']
        elif 'group' in target:
            label_map = encodings['category']
        elif 'disease_duration' in target:
            label_map = encodings['DiseaseDuration']
            label_map.update({'-1':'No Disease'})
        elif 'king' in target:
            label_map = encodings['KingClinicalStage']
            label_map.update({'-1':'No Disease'})
        elif 'alsfrs_total' in target:
            label_map = encodings['ALSFRS-R_TotalScore']
            label_map.update({'-1':'No Disease'})

        if 'alsfrs_speech' in target:
            target_categorical = [str(v.item()) for v in y_vec]
        elif 'disease_duration' in target:
            target_categorical = [label_map[str(v.item())]+' year(s)' if v.item() != -1 else label_map[str(v.item())] for v in y_vec]
        else:
            target_categorical = [label_map[str(v.item())] for v in y_vec]
        y_vec = pd.DataFrame(data = target_categorical, columns = [target]).reset_index(drop=True)

        #y_vec = pd.DataFrame(data = y_vec.detach().cpu(), columns=[target])
        #y_vec = y_vec.astype({target: int})


    "These will be used for visualizing"
    if X is not None:
        X_original = torch.tensor(X.copy().values, device='cpu')
    if OCs is not None:
        OCs_original = torch.tensor(OCs.copy().values, device='cpu').reshape(NoC,-1,OCs.shape[-1])
    if "vowels" in data_training_args.dataset_name:
        y_original = y_vec.copy()
    
    elif "timit" in data_training_args.dataset_name:
        if type(y_vec) == np.ndarray:
            #y.dtype == object:
            "Just need to make a numerical encoding of the labels"
            le = LabelEncoder()
            y_str = y_vec.copy()
            y_original = le.fit_transform(y_vec)
        else:
            if "phoneme" in target:
                y_original = y_39.copy()
                #elif "speaker" in target:
            else:
                y_original = y_vec.copy()
    
    elif "iemocap" in data_training_args.dataset_name:
        if type(y_vec) == np.ndarray:
            "If str"
            #y.dtype == object:
            "Just need to make a numerical encoding of the labels"
            le = LabelEncoder()
            y_str = y_vec.copy()
            y_original = le.fit_transform(y_vec)
        else:
            y_original = y_vec.copy()

    elif "VOC_ALS" in data_training_args.dataset_name:
        le = LabelEncoder()
        y_str = y_vec.copy()
        y_original = le.fit_transform(y_vec)
    


    "Frequency labeling"
    if OCs is not None:
        OC_1 = OCs_original[0].clone()
        OC_2 = OCs_original[1].clone()
        labels_OC_1 = torch.ones(OC_1.size(0), dtype=torch.long)
        labels_OC_2 = torch.ones(OC_2.size(0), dtype=torch.long)*2
        combined_components_only = torch.cat((OC_1,OC_2),dim=0)
        combined_components_only_labels = torch.cat((labels_OC_1,labels_OC_2),dim=0)
    if X is not None:
        labels_X = torch.zeros(X_original.size(0), dtype=torch.long)
        if OCs is None and X is not None:
            original_features = X_original.clone()
    if OCs is not None and X is not None:
        combined_features = torch.cat((X_original,OC_1,OC_2),dim=0)
        combined_labels = torch.cat((labels_X,labels_OC_1,labels_OC_2),dim=0)
    
    if OCs is not None:
        if NoC >= 3 :
            OC_3 = OCs_original[2].clone()
            labels_OC_3 = torch.ones(OC_3.size(0), dtype=torch.long)*3
            if X is not None:
                combined_features = torch.cat((combined_features,OC_3),dim=0)
                combined_labels = torch.cat((combined_labels,labels_OC_3),dim=0)
            combined_components_only_labels = torch.cat((combined_components_only_labels,labels_OC_3),dim=0)
            combined_components_only = torch.cat((combined_components_only,OC_3),dim=0)
        if NoC >= 4:
            OC_4 = OCs_original[3].clone()
            labels_OC_4 = torch.ones(OC_4.size(0), dtype=torch.long)*4
            if X is not None:
                combined_features = torch.cat((combined_features,OC_4),dim=0)
                combined_labels = torch.cat((combined_labels,labels_OC_4),dim=0)
            combined_components_only_labels = torch.cat((combined_components_only_labels,labels_OC_4),dim=0)
            combined_components_only = torch.cat((combined_components_only,OC_4),dim=0)
        if NoC >= 5:
            OC_5 = OCs_original[4].clone()
            labels_OC_5 = torch.ones(OC_5.size(0), dtype=torch.long)*5
            if X is not None:
                combined_features = torch.cat((combined_features,OC_5),dim=0)
                combined_labels = torch.cat((combined_labels,labels_OC_5),dim=0)
            combined_components_only_labels = torch.cat((combined_components_only_labels,labels_OC_5),dim=0)
            combined_components_only = torch.cat((combined_components_only,OC_5),dim=0)
        if NoC >= 6:
            OC_6 = OCs_original[5].clone()
            labels_OC_6 = torch.ones(OC_6.size(0), dtype=torch.long)*6
            if X is not None:
                combined_features = torch.cat((combined_features,OC_6),dim=0)
                combined_labels = torch.cat((combined_labels,labels_OC_6),dim=0)
            combined_components_only_labels = torch.cat((combined_components_only_labels,labels_OC_6),dim=0)
            combined_components_only = torch.cat((combined_components_only,OC_6),dim=0)

    "Generate an isotropic Multivariate Gaussian ~ N(z; 0, I)"
    if data_training_args.vis_sphere: #z_or_h == 'h':
        if X is not None:
            mvgmean = np.zeros(X_original.size(1))
            mvgcov = np.eye(X_original.size(1))
        elif X is None and OCs is not None:
            mvgmean = np.zeros(OCs_original.size(1))
            mvgcov = np.eye(OCs_original.size(1))
        num_points = 1000
        if X is not None and OCs is not None:
            mvgpoints = torch.tensor(np.random.multivariate_normal(mvgmean, mvgcov, num_points)).to(combined_features.device).to(combined_features.dtype)
            combined_features = torch.cat((combined_features,mvgpoints),dim=0)
        elif X is None and OCs is not None:
            mvgpoints = torch.tensor(np.random.multivariate_normal(mvgmean, mvgcov, num_points)).to(combined_components_only.device).to(combined_components_only.dtype)
        elif X is not None and OCs is None:
            mvgpoints = torch.tensor(np.random.multivariate_normal(mvgmean, mvgcov, num_points)).to(X_original.device).to(X_original.dtype)
            original_features = torch.cat((original_features,mvgpoints),dim=0)
        if OCs is not None:
            combined_components_only = torch.cat((combined_components_only,mvgpoints),dim=0)

    "Initialize vis methods and get embeddings"
    if manifold_dict is None:
        reduction = VIS_METHODS[data_training_args.vis_method]
    else:
        reduction = manifold_dict[data_training_args.vis_method]
    if data_training_args.tsne_plot_2d_3d == "2d" or data_training_args.tsne_plot_2d_3d == "both":
        reduction.set_params(**{"n_components":2})
        if X is not None and OCs is not None:
            tsne_results = reduction.fit_transform(combined_features.detach().cpu().numpy())
        elif X is not None and OCs is None:
            tsne_results = reduction.fit_transform(original_features.detach().cpu().numpy())
        tsne_originals = tsne_results[:X_original.size(0),:]
        if OCs is not None:
            tsne_components_only_results = reduction.fit_transform(combined_components_only.detach().cpu().numpy())

    if data_training_args.tsne_plot_2d_3d == "3d" or data_training_args.tsne_plot_2d_3d == "both":
        reduction.set_params(**{"n_components":3})
        if X is not None and OCs is not None:
            tsne_results_3d = reduction.fit_transform(combined_features.detach().cpu().numpy())
        elif X is not None and OCs is None:
            tsne_results_3d = reduction.fit_transform(original_features.detach().cpu().numpy())
        tsne_originals_3d = tsne_results_3d[:X_original.size(0),:]
        if OCs is not None:
            tsne_components_only_3d_results = reduction.fit_transform(combined_components_only.detach().cpu().numpy())             
    
    if data_training_args.vis_sphere: 
        #Find std of multivariate Gaussian in reduced dimension
        if X is not None and OCs is not None:
            if data_training_args.tsne_plot_2d_3d == "2d" or data_training_args.tsne_plot_2d_3d == "both":
                tsne_mvG = tsne_results[combined_features.size(0)-num_points:,:]
                assert tsne_mvG.shape[0] == num_points
                tsne_results = tsne_results[:combined_features.size(0)-num_points,:]
                std_mvG_2d = np.std(tsne_mvG,axis=0)        
                center_mvG_2d = np.mean(tsne_mvG,axis=0)
            if data_training_args.tsne_plot_2d_3d == "3d" or data_training_args.tsne_plot_2d_3d == "both":
                tsne_mvG_3d = tsne_results_3d[combined_features.size(0)-num_points:,:]
                assert tsne_mvG_3d.shape[0] == num_points
                tsne_results_3d = tsne_results_3d[:combined_features.size(0)-num_points,:]
                std_mvG_3d = np.std(tsne_mvG_3d,axis=0)
                center_mvG_3d = np.mean(tsne_mvG_3d,axis=0)

        elif X is not None and OCs is None:
            if data_training_args.tsne_plot_2d_3d == "2d" or data_training_args.tsne_plot_2d_3d == "both":
                tsne_mvG = tsne_results[X_original.size(0):,:]
                tsne_results = tsne_results[:X_original.size(0),:]
                std_mvG_2d = np.std(tsne_mvG,axis=0)
                center_mvG_2d = np.mean(tsne_mvG,axis=0)
            if data_training_args.tsne_plot_2d_3d == "3d" or data_training_args.tsne_plot_2d_3d == "both":    
                tsne_mvG_3d = tsne_results_3d[X_original.size(0):,:]
                tsne_results_3d = tsne_results_3d[:X_original.size(0),:]            
                std_mvG_3d = np.std(tsne_mvG_3d,axis=0)        
                center_mvG_3d = np.mean(tsne_mvG_3d,axis=0)
    
        if OCs is not None:
            if data_training_args.tsne_plot_2d_3d == "2d" or data_training_args.tsne_plot_2d_3d == "both":    
                tsne_mvG_comps_only = tsne_components_only_results[combined_components_only.size(0)-num_points:,:]
                tsne_components_only_results = tsne_components_only_results[:combined_components_only.size(0)-num_points,:]
                std_mvG_2d_comps_only = np.std(tsne_mvG_comps_only,axis=0)
                center_mvG_2d_comps_only = np.mean(tsne_mvG_comps_only,axis=0)
                
            if data_training_args.tsne_plot_2d_3d == "3d" or data_training_args.tsne_plot_2d_3d == "both":    
                tsne_mvG_comps_only_3d = tsne_components_only_3d_results[combined_components_only.size(0)-num_points:,:]
                tsne_components_only_3d_results = tsne_components_only_3d_results[:combined_components_only.size(0)-num_points,:]
                std_mvG_3d_comps_only = np.std(tsne_mvG_comps_only_3d,axis=0)            
                center_mvG_3d_comps_only = np.mean(tsne_mvG_comps_only_3d,axis=0)

    colors_frequency = mpl.colormaps['Set1'].colors[:NoC+1]
    if data_training_args.dataset_name in ["sim_vowels","VOC_ALS","scRNA_seq"]:
        colors = (mpl.colormaps['Set1'].colors \
        + mpl.colormaps['Set2'].colors \
        + mpl.colormaps['Set3'].colors)
        colors = filter_yellow_colors(colors)[:len(np.unique(y_original))]

    elif data_training_args.dataset_name in ["timit", "iemocap"]:
        colors = (mpl.colormaps['Set1'].colors \
            + mpl.colormaps['Set2'].colors \
            + mpl.colormaps['Set3'].colors \
            + mpl.colormaps['Dark2'].colors \
            + mpl.colormaps['tab10'].colors   
            )
        colors = filter_yellow_colors(colors)[:len(np.unique(y_original))]

    if "vowels" in data_training_args.dataset_name:
        legend_labels = np.unique(y_original)

    elif "timit" in data_training_args.dataset_name:
        if type(y_vec) == np.ndarray: 
            #y.dtype == object:
            legend_labels = le.inverse_transform(np.unique(y_original))
        else:
            if 'phoneme' in target:
                legend_labels = id_to_phoneme39(np.unique(y_original))
                #elif 'speaker' in target:
            else:
                legend_labels = np.unique(y_original)

    elif "iemocap" in data_training_args.dataset_name:
        if type(y_vec) == np.ndarray:
            "If str" 
            #y.dtype == object:
            legend_labels = le.inverse_transform(np.unique(y_original))
        else:
            if 'phoneme' in target:
                legend_labels = [id_to_phoneme[i] for i in np.unique(y_original)]
            elif 'speaker' in target:
                legend_labels = [id_to_speaker[i] for i in np.unique(y_original)]
            elif 'emotion' in target:
                legend_labels = [id_to_emotion[i] for i in np.unique(y_original)]
    
    elif "VOC_ALS" in data_training_args.dataset_name:
        legend_labels = le.inverse_transform(np.unique(y_original))

    
    "Frequency plot"
    if data_training_args.frequency_vis:
        if data_training_args.tsne_plot_2d_3d == "2d" or data_training_args.tsne_plot_2d_3d == "both":
            "Original + all components"
            if X is not None and OCs is not None:
                fig, ax = plt.subplots(figsize=(16, 12))
                legend_labels_freq = ['Original Signal'] + [f'Component {i+1}' for i in range(NoC)] 
                for g in np.unique(combined_labels):
                    ix = np.where(combined_labels == g)
                    ax.scatter(tsne_results[ix, 0], tsne_results[ix, 1], color = colors_frequency[g], alpha=0.5,label = legend_labels_freq[g])
        
                if data_training_args.vis_sphere: #z_or_h == 'h':
                    legend_labels_freq = legend_labels_freq + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']
                    circle = plt.Circle(center_mvG_2d, std_mvG_2d.mean(), color='k', fill=False, label = 'Isotropic Multivariate Gaussian ~ N(z; 0, I)')
                    plt.gca().add_artist(circle)
                
                plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40)
                plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize =40)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                ax.tick_params(axis='both', which='major', labelsize=30)

                ax.legend(legend_labels_freq, loc = 'upper right') 
                #plt.tight_layout()
                if is_wandb_available() and data_training_args.with_wandb:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_{z_or_h}_X_OCs_frequency'
                    wandb.log({fname: wandb.Image(plt)})
                elif save_dir is not None:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_X_OCs_frequency'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, fname), 
                        dpi=600,
                        bbox_inches='tight')
                    
                plt.close(fig)

            "All components combined"
            if OCs is not None:
                fig, ax = plt.subplots(figsize=(16, 12))
                legend_labels_freq = [f'Component {i+1}' for i in range(NoC)] 
                for h,g in enumerate(np.unique(combined_components_only_labels)):
                    ix = np.where(combined_components_only_labels == g)
                    ax.scatter(tsne_components_only_results[ix, 0], tsne_components_only_results[ix, 1], color = colors_frequency[g], alpha=0.5,label = legend_labels_freq[h])
                if data_training_args.vis_sphere: #z_or_h == 'h':
                    legend_labels_freq = legend_labels_freq + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']
                    circle = plt.Circle(center_mvG_2d_comps_only, std_mvG_2d_comps_only.mean(), color='k', fill=False)
                    plt.gca().add_artist(circle)
                plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40)
                plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize =40)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                ax.tick_params(axis='both', which='major', labelsize=30)

                ax.legend(legend_labels_freq, loc='upper right')                

                plt.tight_layout()
                if is_wandb_available() and data_training_args.with_wandb:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_{z_or_h}_OCs_frequency'
                    wandb.log({fname: wandb.Image(plt)})
                elif save_dir is not None:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_OCs_frequency'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, fname), 
                        dpi=600,
                        bbox_inches='tight')
                    
                plt.close(fig)

        if data_training_args.tsne_plot_2d_3d == "3d" or data_training_args.tsne_plot_2d_3d == "both":
            "Initialize animation objects"
            def init():
                ax.view_init(elev=10., azim=0)
                return [scat]

            def animate(i):
                ax.view_init(elev=10., azim=i)
                return [scat]
            
            "Original + all components 3D"    
            if X is not None and OCs is not None:          
                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(projection='3d')
                legend_labels_freq = ['Original Signal'] + [f'Component {i+1}' for i in range(NoC)]
                for g in np.unique(combined_labels):
                    ix = np.where(combined_labels == g)
                    scat = ax.scatter(tsne_results_3d[ix, 0], tsne_results_3d[ix, 1], tsne_results_3d[ix,2], color = colors_frequency[g], alpha=0.5,label = legend_labels_freq[g])
                plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40, labelpad=30, fontweight='normal')
                plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize = 40, labelpad=30, fontweight='normal')
                ax.set_zlabel(f"{data_training_args.vis_method.upper()} 3", fontsize = 40, labelpad=30, fontweight='normal')
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)
                
                ax.tick_params(axis='z', which='major', labelsize=25)

                if data_training_args.vis_sphere: #z_or_h == 'h':                
                    # Create a sphere with radius std of the isotropic multivariate Gaussian
                    legend_labels_freq = legend_labels_freq + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    x = std_mvG_3d.mean() * np.outer(np.cos(u), np.sin(v)) + center_mvG_3d[0]
                    y = std_mvG_3d.mean() * np.outer(np.sin(u), np.sin(v)) + center_mvG_3d[1]
                    z = std_mvG_3d.mean() * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mvG_3d[2]
                    ax.plot_surface(x, y, z, color='gray', alpha=0.2,rstride=10, cstride=10, edgecolors='k', lw=0.6)
                    
                anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=360, interval=20, blit=True)
                fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_X_OCs_frequency_3d.gif'
                os.makedirs(save_dir, exist_ok=True)
                
                "Save a screenshot of the gif before saving the animation"
                if save_dir is not None:
                    # Save screenshots at specific angles
                    for angle in ANGLES:
                        screenshot_fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_X_OCs_frequency_3d_angle_{angle}.png'
                        save_animation_screenshot(fig, ax, angle, os.path.join(save_dir, screenshot_fname), dpi=600)

                writergif = animation.PillowWriter(fps=30)
                if save_dir is not None:
                    anim.save(os.path.join(save_dir, fname),writer=writergif)
                if is_wandb_available() and data_training_args.with_wandb:
                    fname = os.path.join(save_dir,f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_{z_or_h}_X_OCs_frequency_3d.mp4')
                    wandb.log({fname: wandb.Video(fname,fps = 30)})
                plt.close(fig)

            "Only components 3D"   
            if OCs is not None:    
                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(projection='3d')
                legend_labels_freq = [f'Component {i+1}' for i in range(NoC)] 
                for h,g in enumerate(np.unique(combined_components_only_labels)):
                    ix = np.where(combined_components_only_labels == g)
                    scat = ax.scatter(tsne_components_only_3d_results[ix, 0], tsne_components_only_3d_results[ix, 1], tsne_components_only_3d_results[ix,2], color = colors_frequency[g], alpha=0.5,label = legend_labels_freq[h])
                
                plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40, labelpad=30, fontweight='normal')
                plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize = 40, labelpad=30, fontweight='normal')
                ax.set_zlabel(f"{data_training_args.vis_method.upper()} 3", fontsize = 40, labelpad=30, fontweight='normal')
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)

                ax.tick_params(axis='z', which='major', labelsize=25)

                if data_training_args.vis_sphere: #z_or_h == 'h':
                    legend_labels_freq = legend_labels_freq + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']
                    # Create a sphere with radius std of the isotropic multivariate Gaussian
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    x = std_mvG_3d_comps_only.mean() * np.outer(np.cos(u), np.sin(v)) + center_mvG_3d_comps_only[0]
                    y = std_mvG_3d_comps_only.mean() * np.outer(np.sin(u), np.sin(v)) + center_mvG_3d_comps_only[1]
                    z = std_mvG_3d_comps_only.mean() * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mvG_3d_comps_only[2]
                    ax.plot_surface(x, y, z, color='gray', alpha=0.2,rstride=10, cstride=10, edgecolors='k', lw=0.6)

                anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=360, interval=20, blit=True)
                fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_OCs_frequency_3d.gif'
                os.makedirs(save_dir, exist_ok=True)
                
                "Save a screenshot of the gif before saving the animation"
                if save_dir is not None:
                    # Save screenshots at specific angles
                    for angle in ANGLES:
                        screenshot_fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_OCs_frequency_3d_angle_{angle}.png'
                        save_animation_screenshot(fig, ax, angle, os.path.join(save_dir, screenshot_fname), dpi=600)
                
                writergif = animation.PillowWriter(fps=30)
                if save_dir is not None:
                    anim.save(os.path.join(save_dir, fname),writer=writergif)
                if is_wandb_available() and data_training_args.with_wandb:
                    fname = os.path.join(save_dir,f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_{z_or_h}_OCs_frequency_3d.mp4')
                    wandb.log({fname: wandb.Video(fname,fps = 30)})
                plt.close(fig)
            

    "Generative Factors Plot - Phonetic Content and Speaker"
    if data_training_args.generative_factors_vis:
        if data_training_args.tsne_plot_2d_3d == "2d" or data_training_args.tsne_plot_2d_3d == "both":
            "All components combined"
            if OCs is not None:
                fig, ax = plt.subplots(figsize=(16, 12))
                legend_labels_gen = list(legend_labels.copy())
                popped = 0
                for h,g in enumerate(np.unique(y_original)):
                    ix0 = np.where(y_original == g)[0]
                    ix = ix0.copy()
                    for i in range(1,NoC):
                        ix = np.concatenate((ix,i*len(y_original) + ix0), axis = 0)
                    if len(ix)>0:
                        ax.scatter(tsne_components_only_results[ix, 0], tsne_components_only_results[ix, 1], color = colors[h], alpha=0.5,label = legend_labels_gen[h-popped])
                    else:
                        legend_labels_gen.pop(h-popped)
                        popped+=1
                
                if data_training_args.vis_sphere: #z_or_h == "h":
                    legend_labels_gen = legend_labels_gen + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']
                    circle = plt.Circle(center_mvG_2d_comps_only, std_mvG_2d_comps_only.mean(), color='k', fill=False)
                    plt.gca().add_artist(circle)
                
                plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40)
                plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize = 40)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                ax.tick_params(axis='both', which='major', labelsize=30)

                ax.legend(legend_labels_gen, ncol = 5, loc = 'upper right') 

                if is_wandb_available() and data_training_args.with_wandb:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_{z_or_h}_OCs_{target}'
                    wandb.log({fname: wandb.Image(plt)})
                elif save_dir is not None:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_OCs_{target}'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, fname), 
                        dpi=600,
                        bbox_inches='tight')
                    
                plt.close(fig)
            
            "Component by component visualization - Vowel/Speaker"
            if OCs is not None:
                for i in range(NoC):
                    fig, ax = plt.subplots(figsize=(16, 12))
                    legend_labels_gen = list(legend_labels.copy())
                    for h,g in enumerate(np.unique(y_original)):
                        ix0 = np.where(y_original == g)[0]
                        #for i in range(1,NoC):
                        ix = i*len(y_original) + ix0
                        ax.scatter(tsne_components_only_results[ix, 0], tsne_components_only_results[ix, 1], color = colors[h], alpha=0.5,label = legend_labels_gen[h])
                    
                    if data_training_args.vis_sphere: #z_or_h == 'h':
                        legend_labels_gen = legend_labels_gen + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']
                        circle = plt.Circle(center_mvG_2d_comps_only, std_mvG_2d_comps_only.mean(), color='k', fill=False)
                        plt.gca().add_artist(circle)
                    
                    plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40)
                    plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize = 40)
                    plt.xticks(fontsize=30)
                    plt.yticks(fontsize=30)
                    ax.tick_params(axis='both', which='major', labelsize=30)

                    ax.legend(legend_labels_gen, ncol = 5, loc = 'upper right')

                    if is_wandb_available() and data_training_args.with_wandb:
                        fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_{z_or_h}_OC{i+1}_{target}'
                        wandb.log({fname: wandb.Image(plt)})
                    elif save_dir is not None:
                        fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_OC{i+1}_{target}'
                        os.makedirs(save_dir, exist_ok=True)
                        plt.savefig(os.path.join(save_dir, fname), 
                            dpi=600,
                            bbox_inches='tight')

                    plt.close(fig)



            "Vowels/Speakers visualization with original signal latent"
            if X is not None:
                fig, ax = plt.subplots(figsize=(16, 12))
                legend_labels_gen = list(legend_labels.copy())
                popped = 0
                for h,g in enumerate(np.unique(y_original)):
                    ix = np.where(y_original == g)[0]
                    if len(ix) > 0:                            
                        ax.scatter(tsne_originals[ix, 0], tsne_originals[ix, 1], color = colors[h], alpha=0.5,label = legend_labels_gen[h-popped])
                    else:
                        legend_labels_gen.pop(h-popped)
                        popped += 1                
                if data_training_args.vis_sphere: #z_or_h == 'h':         
                    legend_labels_gen = legend_labels_gen + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']                 
                    circle = plt.Circle(center_mvG_2d, std_mvG_2d.mean(), color='k', fill=False)
                    plt.gca().add_artist(circle)
                
                plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40)
                plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize = 40)
                plt.xticks(fontsize=30)
                plt.yticks(fontsize=30)
                ax.tick_params(axis='both', which='major', labelsize=30)

                ax.legend(legend_labels_gen, ncol = 5, loc = 'upper right') 
                if is_wandb_available() and data_training_args.with_wandb:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_{z_or_h}_X_{target}'
                    wandb.log({fname: wandb.Image(plt)})
                elif save_dir is not None:
                    fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_X_{target}'
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(os.path.join(save_dir, fname), 
                        dpi=600,
                        bbox_inches='tight')

                plt.close(fig)

        "3D visualization of Phoneme/Speaker information allocation"
        if data_training_args.tsne_plot_2d_3d == "3d" or data_training_args.tsne_plot_2d_3d == "both":
            "Initialize animation objects"
            def init():
                ax.view_init(elev=10., azim=0)
                return [scat]

            def animate(i):
                ax.view_init(elev=10., azim=i)
                return [scat]
            
            "Original 3D - can be a joint embedding"    
            if X is not None: # and OCs is None:          
                fig = plt.figure(figsize=(16, 12))
                ax = fig.add_subplot(projection='3d')
                legend_labels_gen = list(legend_labels.copy())
                popped = 0
                for h,g in enumerate(np.unique(y_original)):
                    ix = np.where(y_original == g)[0]
                    if len(ix) > 0:                            
                        scat = ax.scatter(tsne_originals_3d[ix, 0], tsne_originals_3d[ix, 1], tsne_originals_3d[ix,2], color = colors[h], alpha=0.5,label = legend_labels_gen[h-popped])
                    else:
                        legend_labels_gen.pop(h-popped)
                        popped += 1       
                
                plt.xlabel(f"{data_training_args.vis_method.upper()} 1", fontsize = 40, labelpad=30, fontweight='normal')
                plt.ylabel(f"{data_training_args.vis_method.upper()} 2", fontsize = 40, labelpad=30, fontweight='normal')
                ax.set_zlabel(f"{data_training_args.vis_method.upper()} 3", fontsize = 40, labelpad=30, fontweight='normal')
                plt.xticks(fontsize=25)
                plt.yticks(fontsize=25)

                ax.tick_params(axis='z', which='major', labelsize=25)
                
                if data_training_args.vis_sphere: #z_or_h == 'h':                
                    # Create a sphere with radius std of the isotropic multivariate Gaussian
                    legend_labels_gen = legend_labels_gen + ['Isotropic Multivariate Gaussian ~ N(z; 0, I)']
                    u = np.linspace(0, 2 * np.pi, 100)
                    v = np.linspace(0, np.pi, 100)
                    x = std_mvG_3d.mean() * np.outer(np.cos(u), np.sin(v)) + center_mvG_3d[0]
                    y = std_mvG_3d.mean() * np.outer(np.sin(u), np.sin(v)) + center_mvG_3d[1]
                    z = std_mvG_3d.mean() * np.outer(np.ones(np.size(u)), np.cos(v)) + center_mvG_3d[2]
                    ax.plot_surface(x, y, z, color='gray', alpha=0.2,rstride=10, cstride=10, edgecolors='k', lw=0.6)
                    
    
                anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=360, interval=20, blit=True)
                fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_X_{target}.gif'
                os.makedirs(save_dir, exist_ok=True)
                
                "Save a screenshot of the gif before saving the animation"
                if save_dir is not None:
                    # Save screenshots at specific angles
                    for angle in ANGLES:
                        screenshot_fname = f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_X_{target}_3d_angle_{angle}.png'
                        save_animation_screenshot(fig, ax, angle, os.path.join(save_dir, screenshot_fname), dpi=600)
                
                writergif = animation.PillowWriter(fps=30)
                if save_dir is not None:
                    anim.save(os.path.join(save_dir, fname),writer=writergif)
                if is_wandb_available() and data_training_args.with_wandb:
                    fname = os.path.join(save_dir,f'{data_training_args.vis_method}_{data_set}_{config.decomp_to_perform}_NoC{NoC}_X_{target}.mp4')
                    wandb.log({fname: wandb.Video(fname,fps = 30)})
                plt.close(fig)

    return