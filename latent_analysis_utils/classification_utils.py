"""
Utilities for classification and clustering evaluation on latent representations.
Supports both unsupervised k-Means clustering and supervised classification through machine learning classifiers.
It performs train/dev/test splits if needed, handles different datasets, and logs results to Weights & Biases if enabled.
Has in-build k-fold cross-validation scheme and inner 3-fold-cross-validation for hyperparameter tuning.    
"""
import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit, LeaveOneGroupOut, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans

from tqdm.auto import tqdm
import time
from transformers import is_wandb_available
import json
from scipy import stats

import gc
import psutil
import os

CLASSIFIERS = {
    'LogisticRegression': {
        'classifier': [LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')],
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2',None],  
    },
    'RandomForest': {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [200],
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [2],
        'classifier__max_features': ['sqrt'],
    },
    'SVC': {
    'classifier': [SVC(random_state=42)],
    'classifier__C': [1],
    'classifier__kernel': ['rbf'],
    'classifier__decision_function_shape': ['ovr'],
    },
}

DUMMY_STRATEGIES = ['uniform', 'most_frequent']

def greedy_cluster_mapping(y_pred, y_true, num_clusters):
    """
    Map cluster labels to reference labels using the greedy Hungarian (Munkres) algorithm.
    """
    "Create a contingency matrix (cluster x label)"
    contingency_matrix = np.zeros((num_clusters, num_clusters), dtype=int)
    for cluster, label in zip(y_pred, y_true):
        contingency_matrix[cluster, int(label)] += 1

    "Greedy mapping"
    mapping = {}
    for _ in range(num_clusters):
        max_index = np.unravel_index(np.argmax(contingency_matrix, axis=None), contingency_matrix.shape)
        mapping[max_index[0]] = max_index[1]
        contingency_matrix[:, max_index[1]] = -1  # Mark this column as used
        contingency_matrix[max_index[0], :] = -1  # Mark this row as used

    # Map the predicted labels to the reference labels
    y_mapped = np.array([mapping[label] for label in y_pred])

    return y_mapped

def calculate_unweighted_accuracy(y_true, y_pred):
    """
    Calculate unweighted accuracy (UWA) for IEMOCAP or similar datasets.
    
    UWA is the average of per-class accuracies, which gives equal weight
    to all classes regardless of their frequency in the dataset.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        float: Unweighted accuracy score
    """

    if type(y_true) == pd.DataFrame:
        y_true = y_true.to_numpy()
    # Get unique classes
    classes = np.unique(y_true)
    
    # Calculate accuracy for each class
    per_class_accuracies = []
    for cls in classes:
        # Get indices for this class
        indices = np.where(y_true == cls)[0]
        if len(indices) > 0:
            # Calculate accuracy for this class
            class_acc = accuracy_score(y_true[indices], y_pred[indices])
            per_class_accuracies.append(class_acc)
    
    # Return mean of per-class accuracies
    return np.mean(per_class_accuracies) if per_class_accuracies else 0.0


"Goals of this analysis: perform clustering, perform classification"
def prediction_eval(data_training_args, config,X,X_test,y,y_test,checkpoint,latent_type,target = "vowel"):
    """
    Perform classification and clustering evaluation on latent representations.
    Supports both unsupervised k-Means clustering and supervised classification through machine learning classifiers.
    It performs train/dev/test splits if needed, handles different datasets, and logs results to Weights & Biases if enabled.
    Has in-build k-fold cross-validation scheme and inner 3-fold-cross-validation for hyperparameter tuning.    

    Args:
        data_training_args: Data and training related arguments
        config: Model configuration object.
        X: Latent representations for training data (torch.Tensor or pd.DataFrame)
        X_test: Latent representations for test data (torch.Tensor or pd.DataFrame)
        y: Labels for training data (torch.Tensor or np.array)
        y_test: Labels for test data (torch.Tensor or np.array)
        checkpoint: Checkpoint identifier for saving results; affects only the name of the saved results files
        latent_type: Refers to the type of latent aggregation of individual subspaces to obtain final latent prior approximation.
            Can be 'all', 'X', 'OCs_joint', 'OCs_proj'; affects only the name of the saved results files.
        target: Target variable for classification (default is "vowel")
    """



    def log_memory_usage(label):
        """Log memory usage at specific points in execution"""
        process = psutil.Process(os.getpid())
        print(f"Memory usage ({label}): {process.memory_info().rss / 1024 / 1024:.2f} MB")

    if is_wandb_available() and data_training_args.with_wandb:
        import wandb

    assert isinstance(target, str) or (isinstance(target, (list, tuple)) and len(target) <= 2), "Target must be a string or a list/tuple of max length 2"
    if len(target) > 1 and isinstance(target, (list, tuple)):
        "In case of stratified CV based on support variable"
        target_support = target[1]
        target = target[0]
    
    if X.device == 'cuda':
        X = X.cpu()
        if X_test is not None:   
            X_test = X_test.cpu()
    if y.device == 'cuda':
        y = y.cpu()
        if y_test is not None:   
            y_test = y_test.cpu()
    latent_dim = X.shape[1]
    colnames_X = ["X" + str(i) for i in range(latent_dim)]
    X = pd.DataFrame(data = X, columns = colnames_X)
    if X_test is not None:   
        X_test = pd.DataFrame(data = X_test, columns = colnames_X)

    if "vowels" in data_training_args.dataset_name:        
        if data_training_args.sim_vowels_number == 5:
            int_to_vowel = {
                '0': 'a', '1': 'e', '2': 'I', '3': 'aw', '4': 'u'
            }           
        elif data_training_args.sim_vowels_number == 8:
             int_to_vowel = {'0':'i','1':'I','2':'e','3':'ae','4':'a','5':'aw','6':'y','7':'u'}
        "Convert vowels to strings / categorical"
        if "vowel" in target:
            if data_training_args.discard_label_overlaps:
                vowels_categorical = [int_to_vowel[str(v.item())] for v in y if len(int_to_vowel[str(v.item())]) <= 2]
                vowels_categorical_test = [int_to_vowel[str(v.item())] for v in y_test if len(int_to_vowel[str(v.item())]) <= 2]
                corresp_inds = [i for i,v in enumerate(y) if len(int_to_vowel[str(v.item())]) <= 2]
                corresp_inds_test = [i for i,v in enumerate(y_test) if len(int_to_vowel[str(v.item())]) <= 2]
                "Discard the corresponding rows from the data"
                X = X.iloc[corresp_inds].reset_index(drop=True)
                X_test = X_test.iloc[corresp_inds_test].reset_index(drop=True)
            else:
                vowels_categorical = [int_to_vowel[str(v.item())] for v in y]
                vowels_categorical_test = [int_to_vowel[str(v.item())] for v in y_test]
            
            labels = pd.DataFrame(data = vowels_categorical, columns = [target]).reset_index(drop=True)
            labels_test = pd.DataFrame(data = vowels_categorical_test, columns = [target]).reset_index(drop=True)            
        elif "speaker" in target:
            "Speakers - No need to discard overlaps"
            y = y.detach().cpu()
            y_test = y_test.detach().cpu()
            sg_train = np.stack([(0.7,0.73),(0.78,0.81),(0.82,0.85),(0.86,0.89),(0.94,0.97),(1.02,1.05),(1.1,1.13),(1.14,1.17),(1.18,1.21),(1.26,1.29)])
            sg_dev = np.stack([(0.74,0.75),(0.9,0.91),(0.98,0.99),(1.06,1.07),(1.22,1.23)])
            sg_test = np.stack([(0.76,0.77),(0.92,0.93),(1.00,1.01),(1.08,1.09),(1.24,1.25)])
            speaker_groups = np.vstack([sg_train,sg_dev,sg_test])
            speakers_str = ['SP'+str(s+1) for s in range(speaker_groups.shape[0])]
            speakers_IDs = np.zeros_like(y,dtype='object')
            speakers_IDs_test = np.zeros_like(y_test,dtype='object')
            for h,g in enumerate(speaker_groups):
                ix_L = np.where(y >= g[0])[0]                      
                ix_U = np.where(y < g[1])[0]
                ix0 = np.intersect1d(ix_L,ix_U)
                speakers_IDs[ix0] = speakers_str[h]
                ix_L_test = np.where(y_test >= g[0])[0]
                ix_U_test = np.where(y_test < g[1])[0]
                ix0_test = np.intersect1d(ix_L_test,ix_U_test)
                speakers_IDs_test[ix0_test] = speakers_str[h]
        
            labels = pd.DataFrame(data = speakers_IDs.astype('str'), columns = [target]).reset_index(drop=True)
            labels_test = pd.DataFrame(data = speakers_IDs_test.astype('str'), columns = [target]).reset_index(drop=True)
        
        "Convert label categories to numericals - Separately for Supervised and Unsupervised Evaluation"
        "Unsupervised"
        le_y_unsup = LabelEncoder().fit(np.array(labels).ravel())
        le_y_test_unsup = LabelEncoder().fit(np.array(labels_test).ravel())
        y_unsup = pd.DataFrame(data = le_y_unsup.transform(np.array(labels).ravel()), columns=[target])
        y_test_unsup = pd.DataFrame(data = le_y_test_unsup.transform(np.array(labels_test).ravel()), columns=[target])

        "Supervised"
        le_y = LabelEncoder().fit(np.concatenate([labels,labels_test]).ravel())
        y = pd.DataFrame(data = le_y.transform(np.array(labels).ravel()), columns=[target])
        y_test = pd.DataFrame(data = le_y.transform(np.array(labels_test).ravel()), columns=[target])

    elif "timit" in data_training_args.dataset_name:  
        y = y.detach().cpu()
        y_test = y_test.detach().cpu()
        "First discard values not found"
        disc = np.where(y_test == -100)[0]
        y_test = np.delete(y_test,disc)
        X_test = X_test.drop(disc).reset_index(drop=True)
        disc = np.where(y == -100)[0]
        y = np.delete(y,disc)
        X = X.drop(disc).reset_index(drop=True)
        
        if 'phoneme' in target:
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

            y_39 = pd.DataFrame(data = np.array(id48_to_id39(y)), columns=["phoneme39"])
            y_39_test = pd.DataFrame(np.array(id48_to_id39(y_test)), columns=["phoneme39"])
            y_39 = y_39.astype({'phoneme39': int})
            y_39_test = y_39_test.astype({'phoneme39': int})

        "Convert label categories to numericals - Separately for Supervised and Unsupervised Evaluation"
        "Unsupervised"
        y = y.detach().cpu()
        y_test = y_test.detach().cpu()
        le_y_unsup = LabelEncoder().fit(np.array(y).ravel())
        le_y_test_unsup = LabelEncoder().fit(np.array(y_test).ravel())
        y_unsup = pd.DataFrame(data = le_y_unsup.transform(np.array(y).ravel()), columns=[target])
        y_test_unsup = pd.DataFrame(data = le_y_test_unsup.transform(np.array(y_test).ravel()), columns=[target])

        "Supervised"
        y = pd.DataFrame(data = y, columns=[target])
        y = y.astype({target: int})
        y_test = pd.DataFrame(data = y_test, columns=[target])
        y_test = y_test.astype({target: int})

        if 'phoneme' in target:
            y_for_map = pd.concat([y,y_39],axis=1)
            y_test_for_map = pd.concat([y_test,y_39_test],axis=1)

    elif "iemocap" in data_training_args.dataset_name:  
        y = y.detach().cpu()
        if len(y.shape) == 2:
            if y.shape[1] == 2:
                y_support = y[:,1]
                y = y[:,0]
        disc = np.where(y == -100)[0]
        y = np.delete(y,disc)
        if 'y_support' in locals():
            y_support = np.delete(y_support,disc)
        X = X.drop(disc).reset_index(drop=True)
        if y_test is not None:   
            y_test = y_test.detach().cpu()
            "First discard values not found"
            disc = np.where(y_test == -100)[0]
            y_test = np.delete(y_test,disc)
        if X_test is not None:   
            X_test = X_test.drop(disc).reset_index(drop=True)
                
        if 'phoneme' in target:
            with open(data_training_args.path_to_iemocap_phoneme_to_id_file, 'r') as json_file:
                phoneme_to_id = json.load(json_file)
                id_to_phoneme = {v: k for k, v in phoneme_to_id.items()}
        elif 'speaker' in target:
            with open(data_training_args.path_to_iemocap_speaker_dict_file, 'r') as json_file:
                speaker_to_id = json.load(json_file)
                id_to_speaker = {v: k for k, v in speaker_to_id.items()}
        elif 'emotion' in target:
            with open(data_training_args.path_to_iemocap_emotion_to_id_file, 'r') as json_file:
                emotion_to_id = json.load(json_file)
                id_to_emotion = {v: k for k, v in emotion_to_id.items()}

        "Convert label categories to numericals - Separately for Supervised and Unsupervised Evaluation"
        "Unsupervised"
        y = y.detach().cpu()
        if 'y_support' in locals():
            y_support = y_support.detach().cpu()
            
        if y_test is not None:   
            y_test = y_test.detach().cpu()
            le_y_test_unsup = LabelEncoder().fit(np.array(y_test).ravel())
            y_test_unsup = pd.DataFrame(data = le_y_test_unsup.transform(np.array(y_test).ravel()), columns=[target])
        else:
            y_test_unsup = None
        le_y_unsup = LabelEncoder().fit(np.array(y).ravel())   
        y_unsup = pd.DataFrame(data = le_y_unsup.transform(np.array(y).ravel()), columns=[target])
        "y_support not needed for unsupervised evaluation"

        "Supervised"
        y = pd.DataFrame(data = y, columns=[target])
        y = y.astype({target: int})
        if 'y_support' in locals():
            y_support = pd.DataFrame(data = y_support, columns=[target_support])
            y_support = y_support.astype({target_support: int})
        
        if y_test is not None:   
            y_test = pd.DataFrame(data = y_test, columns=[target])
            y_test = y_test.astype({target: int})
    
    elif "VOC_ALS" in data_training_args.dataset_name:
        y = y.detach().cpu()
        le_y_unsup = LabelEncoder().fit(np.array(y).ravel())
        y_unsup = pd.DataFrame(data = le_y_unsup.transform(np.array(y).ravel()), columns=[target])
        y = y_unsup.copy()
        if y_test is not None:
            le_y_test_unsup = LabelEncoder().fit(np.array(y_test).ravel())
            y_test_unsup = pd.DataFrame(data = le_y_test_unsup.transform(np.array(y_test).ravel()), columns=[target])                
            y_test = y_test_unsup.copy()
        else:
            y_test_unsup = None

    "Unsupervised Classification - K-Means"
    if data_training_args.unsup_eval:
        results_unsup = []

        "Create unsupervised evaluation pipeline"
        def _unsup_evaluation(X_train, y_train, X_test, y_test, num_clusters_train, num_clusters_test, rs = 42, feature_method=None,feat_params=None):
            """
            Perform K-means clustering with optional feature projection and evaluate using the greedy algorithm.
            """
            start_time = time.time()
            steps_train = [('scaler', StandardScaler())]
            steps_test = [('scaler', StandardScaler())]
            if feature_method is not None and feature_method != 'None':
                steps_train.append(('feature_method', feat_params['feature_method'][0]))
                steps_test.append(('feature_method', feat_params['feature_method'][0]))

            steps_train.append(('kmeans', KMeans(n_clusters=num_clusters_train, random_state=rs)))
            steps_test.append(('kmeans', KMeans(n_clusters=num_clusters_test, random_state=rs)))

            # Create the pipeline
            pipeline_train = Pipeline(steps_train)
            if y_test is not None:
                pipeline_test = Pipeline(steps_test)
            
            pipeline_train.fit(X_train)
            y_pred_train = pipeline_train.named_steps['kmeans'].labels_
            
            if y_test is not None:
                pipeline_test.fit(X_test)
                y_pred_test = pipeline_test.named_steps['kmeans'].labels_
            
            if len(y_train.shape) > 1:
                "Timit case: Modeling based on phonemes48, but mapping performance to phonemes39"
                y_pred_train_39 = id48_to_id39(y_pred_train)
                y_pred_test_39 = id48_to_id39(y_pred_test)

            # Compute accuracy and F1 score
            if len(y_train.shape) > 1:
                "Timit case: Modeling based on phonemes48, but mapping performance to phonemes39"
                y_mapped_train = greedy_cluster_mapping(y_pred_train, y_train['phoneme48'], num_clusters_train)
                y_mapped_test = greedy_cluster_mapping(y_pred_test, y_test['phoneme48'], num_clusters_test)
                y_mapped_train_39 = id48_to_id39(y_mapped_train)
                y_mapped_test_39 = id48_to_id39(y_mapped_test)
                train_accuracy = accuracy_score(y_train['phoneme39'], y_mapped_train_39)
                train_f1_score = f1_score(y_train['phoneme39'], y_mapped_train_39, average='weighted')
                train_f1_score_macro = f1_score(y_train['phoneme39'], y_mapped_train_39, average='macro')
                test_accuracy = accuracy_score(y_test['phoneme39'], y_mapped_test_39)
                test_f1_score = f1_score(y_test['phoneme39'], y_mapped_test_39, average='weighted')
                test_f1_score_macro = f1_score(y_test['phoneme39'], y_mapped_test_39, average='macro')
            elif len(y_train.shape) == 1 and 'emotion' in y_train.name:
                "In IEMOCAP, we will use the Weighted and Unweighted Accuracy metrics + F1 Score for emotions"
                y_mapped_train = greedy_cluster_mapping(y_pred_train, y_train, num_clusters_train)
                train_accuracy = accuracy_score(y_train, y_mapped_train)
                train_f1_score = f1_score(y_train, y_mapped_train, average='weighted')
                train_f1_score_macro = f1_score(y_train, y_mapped_train, average='macro')
                unweighted_train_accuracy = calculate_unweighted_accuracy(y_train, y_mapped_train)
                if y_test is not None:
                    y_mapped_test = greedy_cluster_mapping(y_pred_test, y_test, num_clusters_test)
                    test_accuracy = accuracy_score(y_test, y_mapped_test)
                    test_f1_score = f1_score(y_test, y_mapped_test, average='weighted')
                    test_f1_score_macro = f1_score(y_test, y_mapped_test, average='macro')
                    unweighted_test_accuracy = calculate_unweighted_accuracy(y_test, y_mapped_test)
            else:
                y_mapped_train = greedy_cluster_mapping(y_pred_train, y_train, num_clusters_train)
                train_accuracy = accuracy_score(y_train, y_mapped_train)
                train_f1_score = f1_score(y_train, y_mapped_train, average='weighted')
                train_f1_score_macro = f1_score(y_train, y_mapped_train, average='macro')
                if y_test is not None:
                    y_mapped_test = greedy_cluster_mapping(y_pred_test, y_test, num_clusters_test)
                    test_accuracy = accuracy_score(y_test, y_mapped_test)
                    test_f1_score = f1_score(y_test, y_mapped_test, average='weighted')
                    test_f1_score_macro = f1_score(y_test, y_mapped_test, average='macro')

            #print(f"Train Accuracy: {train_accuracy:.4f}")
            #print(f"Test Accuracy: {test_accuracy:.4f}")
            end_time = time.time()
            #print("--------------------------------------------------")

            if y_test is not None:
                results_unsup.append({
                    f'Test_Accuracy{rs}': test_accuracy,
                    f'Train_Accuracy{rs}': train_accuracy,
                    f'Train_F1_Score{rs}': train_f1_score,
                    f'Test_F1_Score{rs}': test_f1_score,
                    f'Train_F1_Score_Macro{rs}': train_f1_score_macro,
                    f'Test_F1_Score_Macro{rs}': test_f1_score_macro,
                    f'Fit_Time{rs}': end_time - start_time
                })
                if len(y_train.shape) == 1 and 'emotion' in y_train.name:
                    results_unsup[-1] = {**results_unsup[-1], **{
                        f'Train_Unweighted_Accuracy{rs}': unweighted_train_accuracy,
                        f'Test_Unweighted_Accuracy{rs}': unweighted_test_accuracy
                    }}
            else:
                results_unsup.append({
                    f'Train_Accuracy{rs}': train_accuracy,
                    f'Train_F1_Score{rs}': train_f1_score,
                    f'Train_F1_Score_Macro{rs}': train_f1_score_macro,
                    f'Fit_Time{rs}': end_time - start_time
                })
                if len(y_train.shape) == 1 and 'emotion' in y_train.name:
                    results_unsup[-1] = {**results_unsup[-1], **{
                        f'Train_Unweighted_Accuracy{rs}': unweighted_train_accuracy
                    }}

            return results_unsup
        
        def perform_unsupervised_evaluation_with_multiple_seeds(X, X_test, y, y_test, num_clusters, data_training_args, target):
            """
            Perform unsupervised evaluation across multiple random seeds without cross-validation.
            
            Args:
                X: Training features
                X_test: Test features
                y: Training labels
                y_test: Test labels
                num_clusters: Number of clusters for k-means
                data_training_args: Configuration arguments
                target: Target variable name
            """
            # Setup for tracking results across random states
            all_results = []
            
            # For each random state
            for rs in tqdm(range(data_training_args.random_states_unsup), desc="Evaluating random states"):
                feature_methods = {'None': {'feature_method': [None]}}
                
                # Call the unsupervised evaluation function
                if "timit" in data_training_args.dataset_name and 'phoneme48' in target:
                    #y_for_map = y  # Adjust based on your specific data structure
                    #y_test_for_map = y_test
                    
                    results = _unsup_evaluation(
                        X, y, X_test, y_test, 
                        num_clusters, num_clusters, rs=rs,
                        feature_method='None', feat_params=feature_methods['None']
                    )
                elif data_training_args.dataset_name in ["scRNA_seq", "VOC_ALS", "iemocap"]:
                    if y_test is None:
                        results = _unsup_evaluation(
                            X, y[target], X_test, y_test, 
                            num_clusters, num_clusters, rs=rs,
                            feature_method='None', feat_params=feature_methods['None']
                        )
                    else:
                        results = _unsup_evaluation(
                            X, y[target], X_test, y_test[target], 
                            num_clusters, num_clusters, rs=rs,
                            feature_method='None', feat_params=feature_methods['None']
                        )
                else:
                    results = _unsup_evaluation(
                        X, y[target], X_test, y_test[target], 
                        num_clusters, num_clusters, rs=rs,
                        feature_method='None', feat_params=feature_methods['None']
                    )
                
                # Extract train and test accuracy for this random state
                if y_test is not None:
                    test_accuracy = results[rs][f'Test_Accuracy{rs}']
                    test_f1_score = results[rs][f'Test_F1_Score{rs}']
                    test_f1_score_macro = results[rs][f'Test_F1_Score_Macro{rs}']
                train_accuracy = results[rs][f'Train_Accuracy{rs}']
                train_f1_score = results[rs][f'Train_F1_Score{rs}']
                train_f1_score_macro = results[rs][f'Train_F1_Score_Macro{rs}']
                fit_time = results[rs][f'Fit_Time{rs}']
                if 'emotion' in target:
                    unweighted_train_accuracy = results[rs][f'Train_Unweighted_Accuracy{rs}']
                    if y_test is not None:
                        unweighted_test_accuracy = results[rs][f'Test_Unweighted_Accuracy{rs}']

                # Store the results
                if y_test is not None:
                    all_results.append({
                        'Random_State': rs,
                        'Train_Accuracy': train_accuracy,
                        'Test_Accuracy': test_accuracy,
                        'Train_F1_Score': train_f1_score,
                        'Test_F1_Score': test_f1_score,
                        'Train_F1_Score_Macro': train_f1_score_macro,
                        'Test_F1_Score_Macro': test_f1_score_macro,
                        'Fit_Time': fit_time
                    })

                    print(f"Random state {rs}: Train Acc.={train_accuracy:.4f}, Test Acc.={test_accuracy:.4f}")
                    print(f"Random state {rs}: Train F1 Score= {train_f1_score:.4f}, Test F1 Score= {test_f1_score:.4f}")
                    print(f"Random state {rs}: Train F1 Score Macro= {train_f1_score_macro:.4f}, Test F1 Score Macro= {test_f1_score_macro:.4f}")
                    if 'emotion' in target:
                        all_results[-1]['Train_Unweighted_Accuracy'] = unweighted_train_accuracy
                        all_results[-1]['Test_Unweighted_Accuracy'] = unweighted_test_accuracy
                        print(f"Random state {rs}: Train Unweighted Acc.={unweighted_train_accuracy:.4f}, Test Unweighted Acc.={unweighted_test_accuracy:.4f}")
                else:
                    all_results.append({
                        'Random_State': rs,
                        'Train_Accuracy': train_accuracy,
                        'Train_F1_Score': train_f1_score,
                        'Train_F1_Score_Macro': train_f1_score_macro,
                        'Fit_Time': fit_time
                    })

                    print(f"Random state {rs}: Train Acc.={train_accuracy:.4f}")
                    print(f"Random state {rs}: Train F1 Score= {train_f1_score:.4f}")
                    print(f"Random state {rs}: Train F1 Score Macro= {train_f1_score_macro:.4f}")
                    if 'emotion' in target:
                        all_results[-1]['Train_Unweighted_Accuracy'] = unweighted_train_accuracy
                        print(f"Random state {rs}: Train Unweighted Acc.={unweighted_train_accuracy:.4f}")
            
            # Calculate statistics
            results_df = pd.DataFrame(all_results)
            
            train_mean_acc = results_df['Train_Accuracy'].mean()
            train_std_acc = results_df['Train_Accuracy'].std()
            if 'emotion' in target:
                train_mean_unweighted_acc = results_df['Train_Unweighted_Accuracy'].mean()
                train_std_unweighted_acc = results_df['Train_Unweighted_Accuracy'].std()
            if y_test is not None:
                test_mean_acc = results_df['Test_Accuracy'].mean()
                test_std_acc = results_df['Test_Accuracy'].std()
                test_mean_f1 = results_df['Test_F1_Score'].mean()
                test_std_f1 = results_df['Test_F1_Score'].std()
                test_mean_f1_macro = results_df['Test_F1_Score_Macro'].mean()
                test_std_f1_macro = results_df['Test_F1_Score_Macro'].std()
                if 'emotion' in target:
                    test_mean_unweighted_acc = results_df['Test_Unweighted_Accuracy'].mean()
                    test_std_unweighted_acc = results_df['Test_Unweighted_Accuracy'].std()
            train_mean_f1 = results_df['Train_F1_Score'].mean()
            train_std_f1 = results_df['Train_F1_Score'].std()
            train_mean_f1_macro = results_df['Train_F1_Score_Macro'].mean()
            train_std_f1_macro = results_df['Train_F1_Score_Macro'].std()
            

            # Calculate 95% confidence intervals
            confidence_level = 0.95
            n_samples = len(results_df)
            z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)
            train_ci_acc = z_score * (train_std_acc / np.sqrt(n_samples))
            train_ci_f1 = z_score * (train_std_f1 / np.sqrt(n_samples))
            train_ci_f1_macro = z_score * (train_std_f1_macro / np.sqrt(n_samples))
            if 'emotion' in target:
                train_ci_unweighted_acc = z_score * (train_std_unweighted_acc / np.sqrt(n_samples))
            if y_test is not None:
                test_ci_acc = z_score * (test_std_acc / np.sqrt(n_samples))
                test_ci_f1 = z_score * (test_std_f1 / np.sqrt(n_samples))
                test_ci_f1_macro = z_score * (test_std_f1_macro / np.sqrt(n_samples))
                if 'emotion' in target:
                    test_ci_unweighted_acc = z_score * (test_std_unweighted_acc / np.sqrt(n_samples))


            print("\n===== Results Summary =====")
            print(f"Train Accuracy: {train_mean_acc:.4f} ± {train_ci_acc:.4f}")
            print(f"Train F1 Score: {train_mean_f1:.4f} ± {train_ci_f1:.4f}")
            print(f"Train F1 Score (Macro): {train_f1_score_macro:.4f} ± {train_ci_f1_macro:.4f}")
            if 'emotion' in target:
                print(f"Train Unweighted Accuracy: {train_mean_unweighted_acc:.4f} ± {train_ci_unweighted_acc:.4f}")
            if y_test is not None:
                print(f"Test Accuracy: {test_mean_acc:.4f} ± {test_ci_acc:.4f}")
                print(f"Test F1 Score: {test_mean_f1:.4f} ± {test_ci_f1:.4f}")
                print(f"Test F1 Score (Macro): {test_mean_f1_macro:.4f} ± {test_ci_f1_macro:.4f}")
                if 'emotion' in target:
                    print(f"Test Unweighted Accuracy: {test_mean_unweighted_acc:.4f} ± {test_ci_unweighted_acc:.4f}")

            if y_test is not None:
                return_dict = {
                    'detailed_results': results_df,
                    'train_mean_acc': train_mean_acc,
                    'train_std_acc': train_std_acc,
                    'train_ci_acc': train_ci_acc,
                    'test_mean_acc': test_mean_acc,
                    'test_std_acc': test_std_acc,
                    'test_ci_acc': test_ci_acc,
                    'train_mean_f1': train_mean_f1,
                    'train_std_f1': train_std_f1,
                    'train_ci_f1': train_ci_f1,
                    'test_mean_f1': test_mean_f1,
                    'test_std_f1': test_std_f1,
                    'test_ci_f1': test_ci_f1,
                    'train_mean_f1_macro': train_mean_f1_macro,
                    'train_std_f1_macro': train_std_f1_macro,
                    'train_ci_f1_macro': train_ci_f1_macro,
                    'test_mean_f1_macro': test_mean_f1_macro,
                    'test_std_f1_macro': test_std_f1_macro,
                    'test_ci_f1_macro': test_ci_f1_macro,
                }
                if 'emotion' in target:
                    return_dict.update({
                        'train_mean_unweighted_acc': train_mean_unweighted_acc,
                        'train_std_unweighted_acc': train_std_unweighted_acc,
                        'train_ci_unweighted_acc': train_ci_unweighted_acc,
                        'test_mean_unweighted_acc': test_mean_unweighted_acc,
                        'test_std_unweighted_acc': test_std_unweighted_acc,
                        'test_ci_unweighted_acc': test_ci_unweighted_acc,
                    })
            else:
                return_dict = {
                    'detailed_results': results_df,
                    'train_mean_acc': train_mean_acc,
                    'train_std_acc': train_std_acc,
                    'train_ci_acc': train_ci_acc,
                    'train_mean_f1': train_mean_f1,
                    'train_std_f1': train_std_f1,
                    'train_ci_f1': train_ci_f1,
                    'train_mean_f1_macro': train_mean_f1_macro,
                    'train_std_f1_macro': train_std_f1_macro,
                    'train_ci_f1_macro': train_ci_f1_macro,
                }
                if 'emotion' in target:
                    return_dict.update({
                        'train_mean_unweighted_acc': train_mean_unweighted_acc,
                        'train_std_unweighted_acc': train_std_unweighted_acc,
                        'train_ci_unweighted_acc': train_ci_unweighted_acc,
                    })
            return return_dict

        if "timit" in data_training_args.dataset_name:
            if 'phoneme' in target:
                num_clusters_train = num_clusters_test = 48
            elif 'speaker' in target:
                num_clusters_train = 100 #50 speakers
                num_clusters_test = 48 # 24 speakers
        else:
            num_clusters_train = len(np.unique(y_unsup))
            if y_test is not None:
                num_clusters_test = len(np.unique(y_test_unsup))

        "Perform unsupervised evaluation with multiple random states"
        #data_training_args.random_states_unsup = 50
        if "timit" in data_training_args.dataset_name and 'phoneme48' in target:
            unsup_stats = perform_unsupervised_evaluation_with_multiple_seeds(
                X, X_test, y_for_map, y_test_for_map, 
                num_clusters_train, data_training_args, target
            )        
        else:
            unsup_stats = perform_unsupervised_evaluation_with_multiple_seeds(
                X, X_test, y_unsup, y_test_unsup, 
                num_clusters_train, data_training_args, target
            )

        if y_test is not None:
            dict_for_df = {
                'Classifier': 'KMeans',
                'Mean_Test_Accuracy': unsup_stats['test_mean_acc'],
                'Test_CI_95': unsup_stats['test_ci_acc'],
                'Std_Test_Accuracy': unsup_stats['test_std_acc'],
                'Mean_Train_Accuracy': unsup_stats['train_mean_acc'],
                'Train_CI_95': unsup_stats['train_ci_acc'],
                'Std_Train_Accuracy': unsup_stats['train_std_acc'],
                'Mean_Test_F1_Score': unsup_stats['test_mean_f1'],
                'Test_CI_95_F1': unsup_stats['test_ci_f1'],
                'Std_Test_F1_Score': unsup_stats['test_std_f1'],
                'Mean_Train_F1_Score': unsup_stats['train_mean_f1'],
                'Train_CI_95_F1': unsup_stats['train_ci_f1'],
                'Std_Train_F1_Score': unsup_stats['train_std_f1'],
                'Mean_Train_F1_Score_Macro': unsup_stats['train_mean_f1_macro'],
                'Train_CI_95_F1_Macro': unsup_stats['train_ci_f1_macro'],
                'Std_Train_F1_Score_Macro': unsup_stats['train_std_f1_macro'],
                'Mean_Test_F1_Score_Macro': unsup_stats['test_mean_f1_macro'],
                'Test_CI_95_F1_Macro': unsup_stats['test_ci_f1_macro'],
                'Std_Test_F1_Score_Macro': unsup_stats['test_std_f1_macro'],
                'Feature_Method': 'None'
            }
            if 'emotion' in target:
                dict_for_df.update({
                    'Mean_Train_Unweighted_Accuracy': unsup_stats['train_mean_unweighted_acc'],
                    'Std_Train_Unweighted_Accuracy': unsup_stats['train_std_unweighted_acc'],
                    'Train_CI_95_Unweighted': unsup_stats['train_ci_unweighted_acc'],
                    'Mean_Test_Unweighted_Accuracy': unsup_stats['test_mean_unweighted_acc'],
                    'Std_Test_Unweighted_Accuracy': unsup_stats['test_std_unweighted_acc'],
                    'Test_CI_95_Unweighted': unsup_stats['test_ci_unweighted_acc']
                })
        else:
            dict_for_df = {
                'Classifier': 'KMeans',
                'Mean_Train_Accuracy': unsup_stats['train_mean_acc'],
                'Train_CI_95': unsup_stats['train_ci_acc'],
                'Std_Train_Accuracy': unsup_stats['train_std_acc'],
                'Mean_Train_F1_Score': unsup_stats['train_mean_f1'],
                'Train_CI_95_F1': unsup_stats['train_ci_f1'],
                'Std_Train_F1_Score': unsup_stats['train_std_f1'],
                'Mean_Train_F1_Score_Macro': unsup_stats['train_mean_f1_macro'],
                'Train_CI_95_F1_Macro': unsup_stats['train_ci_f1_macro'],
                'Std_Train_F1_Score_Macro': unsup_stats['train_std_f1_macro'],
                'Feature_Method': 'None'
            }
            if 'emotion' in target:
                dict_for_df.update({
                    'Mean_Train_Unweighted_Accuracy': unsup_stats['train_mean_unweighted_acc'],
                    'Std_Train_Unweighted_Accuracy': unsup_stats['train_std_unweighted_acc'],
                    'Train_CI_95_Unweighted': unsup_stats['train_ci_unweighted_acc']
                })
        results_unsup_df = pd.DataFrame([dict_for_df])


        # Save all raw results
        detailed_results_df = unsup_stats['detailed_results']

        # Save results 
        current_result_dir = os.path.join(data_training_args.output_dir, checkpoint)
        if not os.path.exists(current_result_dir):
            os.makedirs(current_result_dir)

        # Create appropriate filenames
        if "vae" in config.model_type or "VAE" in config.model_type:
            model_type = "vae1d"
            base_fname = f'{model_type}_{latent_type}_{target}_z{config.vae_z_dim}_b{config.vae_beta}_{len(config.vae_kernel_sizes)}layers'
        elif config.dual_branched_latent:
            model_type = "dual"
            base_fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{int(config.beta_kl_prior_z)}_bs{int(config.beta_kl_prior_s)}_{model_type}_{latent_type}_{target}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}'
        elif config.only_z_branch:
            model_type = "single_z"
            base_fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{int(config.beta_kl_prior_z)}_{model_type}_{latent_type}_{target}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}'
        elif config.only_s_branch:
            model_type = "single_s"
            base_fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bs{int(config.beta_kl_prior_s)}_{model_type}_{latent_type}_{target}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}'
        
        # Save summary results
        results_unsup_df.to_csv(os.path.join(current_result_dir, f'{base_fname}_unsupervised_kmeans_summary.csv'), index=True)
        
        # Save detailed results by random state
        detailed_results_df.to_csv(os.path.join(current_result_dir, f'{base_fname}_unsupervised_kmeans_by_random_state.csv'), index=False)
        
        # Log to wandb if available
        if is_wandb_available() and data_training_args.with_wandb:
            label_prefix = f"kmeans_{target}_{latent_type}"
            if y_test is not None:
                wandb.log({
                    f"{label_prefix}_test_acc_mean": unsup_stats['test_mean_acc'],
                    f"{label_prefix}_test_acc_ci": unsup_stats['test_ci_acc'],
                    f"{label_prefix}_train_acc_mean": unsup_stats['train_mean_acc'],
                    f"{label_prefix}_train_acc_ci": unsup_stats['train_ci_acc'],
                    f"{label_prefix}_test_f1_mean": unsup_stats['test_mean_f1'],
                    f"{label_prefix}_test_f1_ci": unsup_stats['test_ci_f1'],
                    f"{label_prefix}_train_f1_mean": unsup_stats['train_mean_f1'],
                    f"{label_prefix}_train_f1_ci": unsup_stats['train_ci_f1'],
                    f"{label_prefix}_test_f1_macro_mean": unsup_stats['test_mean_f1_macro'],
                    f"{label_prefix}_test_f1_macro_ci": unsup_stats['test_ci_f1_macro'],
                    f"{label_prefix}_train_f1_macro_mean": unsup_stats['train_mean_f1_macro'],
                    f"{label_prefix}_train_f1_macro_ci": unsup_stats['train_ci_f1_macro']
                })
                if 'emotion' in target:
                    wandb.log({
                        f"{label_prefix}_train_unweighted_acc_mean": unsup_stats['train_mean_unweighted_acc'],
                        f"{label_prefix}_train_unweighted_acc_ci": unsup_stats['train_ci_unweighted_acc'],
                        f"{label_prefix}_test_unweighted_acc_mean": unsup_stats['test_mean_unweighted_acc'],
                        f"{label_prefix}_test_unweighted_acc_ci": unsup_stats['test_ci_unweighted_acc']
                    })
            else:
                wandb.log({
                    f"{label_prefix}_train_acc_mean": unsup_stats['train_mean_acc'],
                    f"{label_prefix}_train_acc_ci": unsup_stats['train_ci_acc'],
                    f"{label_prefix}_train_f1_mean": unsup_stats['train_mean_f1'],
                    f"{label_prefix}_train_f1_ci": unsup_stats['train_ci_f1'],
                    f"{label_prefix}_train_f1_macro_mean": unsup_stats['train_mean_f1_macro'],
                    f"{label_prefix}_train_f1_macro_ci": unsup_stats['train_ci_f1_macro']
                })
                if 'emotion' in target:
                    wandb.log({
                        f"{label_prefix}_train_unweighted_acc_mean": unsup_stats['train_mean_unweighted_acc'],
                        f"{label_prefix}_train_unweighted_acc_ci": unsup_stats['train_ci_unweighted_acc']
                    })

    "Frame Speaker/Phoneme Identification - Supervised"
    if data_training_args.sup_eval and (target != "speaker_seq" or data_training_args.dataset_name not in ["timit","VOC_ALS"]):
        all_cv_results = []

        if X_test is not None:
        # Merge training and test data
            X_merged = pd.concat([X, X_test], axis=0).reset_index(drop=True)
            y_merged = pd.concat([y, y_test], axis=0).reset_index(drop=True)
        else:
            X_merged = X.copy()
            y_merged = y.copy()    
        # For TIMIT phoneme case, also merge the 39-phoneme labels
        if "timit" in data_training_args.dataset_name and 'phoneme48' in target:
            y_39_merged = pd.concat([y_39, y_39_test], axis=0).reset_index(drop=True)

        for rs in range(data_training_args.random_states):
            print(f"\n=== Running CV with random state {rs} ===")
            
            if data_training_args.dataset_name == "iemocap" and 'emotion' in target:
                "Setup LOGO here"
                if target_support is not None:
                    # Extract speaker IDs from target_support 
                    speaker_ids = y_support[target_support].values
                    print(f"Using speaker information from '{target_support}' column for LOSO CV")
                                        
                    # Use LeaveOneGroupOut for speaker-independent evaluation
                    logo = LeaveOneGroupOut()
                    # Convert the iterator to a list of indices we can reuse
                    skf_list = list(logo.split(X_merged, y_merged[target], groups=speaker_ids))
                    print(f"Created {len(skf_list)} LOSO folds based on speaker groups")
                else:
                    print("Warning: No speaker information provided. Using standard stratification.")
                    skf = StratifiedKFold(n_splits=data_training_args.classif_eval_cv_splits, shuffle=True, random_state=rs)
                    skf_list = list(skf.split(X_merged, y_merged))
            else:
                # Setup 5-fold cross-validation with this random state
                skf = StratifiedKFold(n_splits=data_training_args.classif_eval_cv_splits, shuffle=True, random_state=rs)
                skf_list = list(skf.split(X_merged, y_merged))

            
            log_memory_usage(f"Start of random state {rs}")

            # For each fold
            for fold_idx, (train_index, test_index) in enumerate(skf_list):
                print(f"\n--- Fold {fold_idx+1}/5 ---")
                
                # Split data for this fold
                X_train = X_merged.iloc[train_index]
                y_train = y_merged.iloc[train_index]
                X_test = X_merged.iloc[test_index]
                y_test = y_merged.iloc[test_index]
                                
                #Reset indices
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)
                if 'y_support' in locals():
                    y_support_train = y_support.iloc[train_index] 
                    y_support_test = y_support.iloc[test_index] 
                    y_support_train = y_support_train.reset_index(drop=True) 
                    y_support_test = y_support_test.reset_index(drop=True)

                if "timit" in data_training_args.dataset_name and 'phoneme48' in target:
                    y_39_train = y_39_merged.iloc[train_index]
                    y_39_test = y_39_merged.iloc[test_index]
                

                "Catch split errors due to few samples of some classes"
                try:
                    if data_training_args.dataset_name == "iemocap" and 'emotion' in target:
                        "Setup again LOGO here"
                        if target_support is not None:
                            # Extract speaker IDs from target_support for the training set
                            train_speaker_ids = y_support_train[target_support].values
                            
                            # Use GroupShuffleSplit to ensure dev set has different speakers than train
                            gss = GroupShuffleSplit(n_splits=1, test_size=data_training_args.dev_data_percent, random_state=rs)
                            train_idx, dev_idx = next(gss.split(X_train, y_train[target], groups=train_speaker_ids))
                            
                            X_dev = X_train.iloc[dev_idx]
                            y_dev = y_train.iloc[dev_idx]
                            X_train_subset = X_train.iloc[train_idx]
                            y_train_subset = y_train.iloc[train_idx]
                            if 'y_support' in locals():
                                y_support_dev = y_support_train.iloc[dev_idx]
                                y_support_train_subset = y_support_train.iloc[train_idx]

                            # Print information about the split
                            train_speakers = set(train_speaker_ids[train_idx])
                            dev_speakers = set(train_speaker_ids[dev_idx])
                            test_speakers = set(y_support_test[target_support].values)
                            print(f"Train set: {len(train_idx)} samples, {len(train_speakers)} speakers")
                            print(f"Dev set: {len(dev_idx)} samples, {len(dev_speakers)} speakers")
                            print(f"Test set: {len(y_test)} samples, {len(test_speakers)} speakers")
                        else:
                            # Fall back to stratified split
                            sss = StratifiedShuffleSplit(n_splits=1, test_size=data_training_args.dev_data_percent,
                                        train_size=data_training_args.train_data_percent, random_state=rs)

                            for train_idx, dev_idx in sss.split(X_train, y_train[target]):
                                X_dev = X_train.iloc[dev_idx]
                                y_dev = y_train.iloc[dev_idx]
                                X_train_subset = X_train.iloc[train_idx]
                                y_train_subset = y_train.iloc[train_idx]

                    else:
                        # Stratified sampling: use subset of the dataset for dev and train
                        sss = StratifiedShuffleSplit(n_splits=1, test_size=data_training_args.dev_data_percent,
                                            train_size=data_training_args.train_data_percent, random_state=rs)
                
                        for train_idx, dev_idx in sss.split(X_train, y_train):
                            X_dev = X_train.iloc[dev_idx]
                            y_dev = y_train.iloc[dev_idx]
                            X_train_subset = X_train.iloc[train_idx]
                            y_train_subset = y_train.iloc[train_idx]
                            
                            if "timit" in data_training_args.dataset_name and 'phoneme48' in target:  
                                y_39_dev = y_39_train.iloc[dev_idx]
                                y_39_train_subset = y_39_train.iloc[train_idx]
                
                except ValueError as e:
                    if "The least populated class in y has only 1 member" in str(e):
                        print("Warning: Some classes have only 1 example. Using fallback sampling method.")
                        
                        # Count samples per class
                        class_counts = y_train[y_train.columns[0]].value_counts()
                        problematic_classes = class_counts[class_counts < 2].index.tolist()
                        
                        # Create dev set with a simple random split, but ensure problematic classes stay in training
                        dev_size = int(X_train.shape[0] * data_training_args.dev_data_percent)
                        train_size = int(X_train.shape[0]  * data_training_args.train_data_percent)
                        
                        # Get indices of samples from problematic classes
                        problematic_indices = y_train[y_train[y_train.columns[0]].isin(problematic_classes)].index
                        
                        # Get indices of samples from non-problematic classes
                        non_problematic_indices = y_train[~y_train[y_train.columns[0]].isin(problematic_classes)].index
                        
                        # Randomly sample from non-problematic classes
                        np.random.seed(rs)
                        if len(non_problematic_indices) > dev_size:
                            dev_idx = np.random.choice(non_problematic_indices, size=dev_size, replace=False)
                            remaining_indices = np.array([i for i in non_problematic_indices if i not in dev_idx])
                            
                            # Ensure we don't exceed requested train size
                            if len(remaining_indices) + len(problematic_indices) > train_size:
                                # Randomly sample from remaining indices to meet train_size
                                train_idx_non_problematic = np.random.choice(
                                    remaining_indices, 
                                    size=train_size - len(problematic_indices), 
                                    replace=False
                                )
                                train_idx = np.concatenate([train_idx_non_problematic, problematic_indices])
                            else:
                                # Use all remaining indices plus problematic indices
                                train_idx = np.concatenate([remaining_indices, problematic_indices])
                        else:
                            # If we don't have enough non-problematic samples for dev set, 
                            # use all available and reduce dev_size
                            dev_idx = non_problematic_indices
                            train_idx = problematic_indices
                            
                            print(f"Warning: Reduced dev set size to {len(dev_idx)} samples due to class distribution constraints")
                        
                        # Create the subsets
                        X_dev = X_train.iloc[dev_idx]
                        y_dev = y_train.iloc[dev_idx]
                        X_train_subset = X_train.iloc[train_idx]
                        y_train_subset = y_train.iloc[train_idx]
                        
                        if "timit" in data_training_args.dataset_name and 'phoneme48' in target:  
                            y_39_dev = y_39_train.iloc[dev_idx]
                            y_39_train_subset = y_39_train.iloc[train_idx]
                            
                        print(f"Created fallback split: train={len(train_idx)} samples, dev={len(dev_idx)} samples")
                    else:
                        # Re-raise the error if it's not the specific one we're handling
                        raise

                X_test_subset = X_test
                y_test_subset = y_test
                if "timit" in data_training_args.dataset_name and 'phoneme48' in target:
                    y_39_test_subset = y_39_test    

                # Run classifiers for this fold
                fold_results = []
                
                # Dummy Classifiers - Use the current random state
                for strategy in DUMMY_STRATEGIES:
                    dummy = DummyClassifier(strategy=strategy, random_state=rs)
                    dummy.fit(X_train_subset, y_train_subset)
                    eval_predictions = dummy.predict(X_test_subset)
                    
                    if "timit" in data_training_args.dataset_name and 'phoneme48' in target:
                        eval_predictions39 = id48_to_id39(eval_predictions)
                        test_accuracy = accuracy_score(y_39_test_subset, eval_predictions39)
                        test_f1 = f1_score(y_39_test_subset, eval_predictions39, average='weighted')
                        test_f1_macro = f1_score(y_39_test_subset, eval_predictions39, average='macro')
                    else:
                        test_accuracy = accuracy_score(y_test_subset, eval_predictions)
                        test_f1 = f1_score(y_test_subset, eval_predictions, average='weighted')
                        test_f1_macro = f1_score(y_test_subset, eval_predictions, average='macro')
                        if 'emotion' in target:
                            unweighted_test_accuracy = calculate_unweighted_accuracy(y_test_subset, eval_predictions)

                    
                    fold_results.append({
                        'Random_State': rs,
                        'Fold': fold_idx,
                        'Classifier': f'DummyClassifier_{strategy}', 
                        'Feature_Method': 'None', 
                        'Best_Params': None,
                        'Test_Accuracy': test_accuracy,
                        'Test_F1_Score': test_f1,
                        'Test_F1_Macro': test_f1_macro,
                        'Test_Unweighted_Accuracy': unweighted_test_accuracy if 'emotion' in target else None,
                    })          
                    print(f"DummyClassifier with strategy='{strategy}'")
                    print(f"Test Accuracy: {test_accuracy:.4f}")
                    print(f"Test F1 Score: {test_f1:.4f}\n")
                    print(f"Test F1 Macro: {test_f1_macro:.4f}\n")
                    print(f"Test Unweighted Accuracy: N/A\n" if 'emotion' not in target else f"Test Unweighted Accuracy: {unweighted_test_accuracy:.4f}\n")  

                # Main classifiers with proper random state
                feature_methods = {'None': {'feature_method': [None]}}
                
                # Create classifiers with the current random state
                current_classifiers = {
                    'LogisticRegression': {
                        'classifier': [LogisticRegression(max_iter=1000, random_state=rs, solver='lbfgs')],
                        'classifier__C': [0.01, 0.1, 1, 10],
                        'classifier__penalty': ['l2', None],  
                    },
                    'RandomForest': {
                        'classifier': [RandomForestClassifier(random_state=rs)],
                        'classifier__n_estimators': [200],
                        'classifier__max_depth': [None],
                        'classifier__min_samples_split': [2],
                        'classifier__max_features': ['sqrt'],
                    },
                    'SVC': {
                        'classifier': [SVC(random_state=rs)],
                        'classifier__C': [1],
                        'classifier__kernel': ['rbf'],
                        'classifier__decision_function_shape': ['ovr'],
                    }
                }
                
                if data_training_args.dataset_name in ["VOC_ALS","iemocap"]:
                    current_classifiers.pop('SVC', None)

                for cls_name, cls_params in current_classifiers.items():
                    for feat_name, feat_params in feature_methods.items():
                        try:
                            # Skip if classifier and feature selection method are the same
                            if cls_name == feat_name:
                                continue
                                
                            start_time = time.time()
                            target_col = y_train_subset.columns[0]
                            
                            # Create pipeline
                            steps = [('scaler', StandardScaler())]
                            if feat_name != 'None':
                                if feat_name == 'PCA':
                                    # Add random_state to PCA if used
                                    steps.append(('feature_method', PCA(random_state=rs)))
                                else:
                                    steps.append(('feature_method', feat_params['feature_method'][0]))
                            steps.append(('classifier', cls_params['classifier'][0]))
                            pipeline = Pipeline(steps)
                            
                            # Set parameters
                            params = {}
                            if feat_name != 'None':
                                feat_params_copy = feat_params.copy()
                                feat_params_copy.pop('feature_method')
                                params.update(feat_params_copy)
                            
                            cls_params_copy = cls_params.copy()
                            cls_params_copy.pop('classifier')
                            params.update(cls_params_copy)
                            
                            # Inner cross-validation for hyperparameter tuning
                            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)
                            grid_search = GridSearchCV(
                                estimator=pipeline,
                                param_grid=params,
                                cv=inner_cv,
                                scoring='accuracy',
                                n_jobs=data_training_args.classif_num_workers
                            )
                            
                            grid_search.fit(X_dev, y_dev[target_col])
                            best_params = grid_search.best_params_
                            del grid_search  

                            # Train final model with best parameters
                            final_pipeline = Pipeline(steps)
                            final_pipeline.set_params(**best_params)
                            final_pipeline.fit(X_train_subset, y_train_subset[target_col])
                            
                            # Evaluate
                            if "timit" in data_training_args.dataset_name and 'phoneme48' in target:
                                test_predictions39 = id48_to_id39(final_pipeline.predict(X_test_subset))
                                test_accuracy = accuracy_score(y_39_test_subset, test_predictions39)
                                test_f1 = f1_score(y_39_test_subset, test_predictions39, average='weighted')
                                test_f1_macro = f1_score(y_39_test_subset, test_predictions39, average='macro')
                            else:
                                test_predictions = final_pipeline.predict(X_test_subset)
                                test_accuracy = accuracy_score(y_test_subset[target_col], test_predictions)
                                test_f1 = f1_score(y_test_subset[target_col], test_predictions, average='weighted')
                                test_f1_macro = f1_score(y_test_subset[target_col], test_predictions, average='macro')
                                if 'emotion' in target:
                                    unweighted_test_accuracy = calculate_unweighted_accuracy(y_test_subset[target_col], test_predictions)

                            end_time = time.time()
                            
                            # Record results
                            fold_results.append({
                                'Random_State': rs,
                                'Fold': fold_idx,
                                'Classifier': cls_name,
                                'Feature_Method': feat_name,
                                'Best_Params': str(best_params),  # Convert to string for CSV saving
                                'Test_Accuracy': test_accuracy,
                                'Test_F1_Score': test_f1,
                                'Test_F1_Macro': test_f1_macro,
                                'Fit_Time': end_time - start_time
                            })
                            if 'emotion' in target:
                                fold_results[-1]['Test_Unweighted_Accuracy'] = unweighted_test_accuracy


                            print(f"Classifier: {cls_name}, Feature Method: {feat_name}, RS: {rs}")
                            print(f"Test Accuracy: {test_accuracy:.4f}")
                            print(f"Test F1 Score: {test_f1:.4f}")
                            print(f"Test F1 Macro: {test_f1_macro:.4f}")
                            if 'emotion' in target:
                                print(f"Test Unweighted Accuracy: {unweighted_test_accuracy:.4f}")
                            print(f"Time: {end_time - start_time:.2f}s\n")
                            
                        except Exception as e:
                            print(f"Failed to evaluate Classifier: {cls_name}, Feature Method: {feat_name}, RS: {rs}")
                            print(f"Error: {e}\n")
                
                # Add fold results to overall results
                all_cv_results.extend(fold_results)

                del X_train_subset, y_train_subset, X_test_subset, y_test_subset, X_dev, y_dev, fold_results
                if 'X_test' in locals():
                    del X_test, y_test
                # Force garbage collection
                gc.collect()
                log_memory_usage(f"After fold {fold_idx+1}, RS {rs}")

            log_memory_usage(f"End of random state {rs}")
            gc.collect()
        
        results_df = pd.DataFrame(all_cv_results)
    
        # Calculate statistics by classifier and feature method
        if 'emotion' in target:
            summary_df = results_df.groupby(['Classifier', 'Feature_Method']).agg({
                'Test_Accuracy': ['mean', 'std', 'min', 'max'],
                'Test_F1_Score': ['mean', 'std', 'min', 'max'],
                'Test_F1_Macro': ['mean', 'std', 'min', 'max'],
                'Test_Unweighted_Accuracy': ['mean', 'std', 'min', 'max'],
                'Fit_Time': ['mean']
            }).reset_index()
        else:
            summary_df = results_df.groupby(['Classifier', 'Feature_Method']).agg({
                'Test_Accuracy': ['mean', 'std', 'min', 'max'],
                'Test_F1_Score': ['mean', 'std', 'min', 'max'],
                'Test_F1_Macro': ['mean', 'std', 'min', 'max'],
                'Fit_Time': ['mean']
            }).reset_index()
        
        # Calculate 95% confidence intervals
        n_samples = data_training_args.classif_eval_cv_splits * data_training_args.random_states  # random states * 5 folds
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Add confidence intervals
        summary_df['Test_Accuracy_CI'] = z_score * (summary_df[('Test_Accuracy', 'std')] / np.sqrt(n_samples))
        summary_df['Test_F1_Score_CI'] = z_score * (summary_df[('Test_F1_Score', 'std')] / np.sqrt(n_samples))
        summary_df['Test_F1_Macro_CI'] = z_score * (summary_df[('Test_F1_Macro', 'std')] / np.sqrt(n_samples))
        if 'emotion' in target:
            summary_df['Test_Unweighted_Accuracy_CI'] = z_score * (summary_df[('Test_Unweighted_Accuracy', 'std')] / np.sqrt(n_samples))
        # Flatten column names
        summary_df.columns = ['_'.join(col).strip('_') for col in summary_df.columns.values]
        
        # Save results
        current_result_dir = os.path.join(data_training_args.output_dir, checkpoint)
        if not os.path.exists(current_result_dir):
            os.makedirs(current_result_dir)
        
        # Create filename based on model configuration
        if "vae" in config.model_type or "VAE" in config.model_type:
            model_type = "vae1d"
            base_fname = f'{model_type}_{latent_type}_{target}_z{config.vae_z_dim}_b{config.vae_beta}_{len(config.vae_kernel_sizes)}layers'
        elif config.dual_branched_latent:
            model_type = "dual"
            base_fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{config.beta_kl_prior_z}_bs{config.beta_kl_prior_s}_{model_type}_{latent_type}_{target}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}'
        elif config.only_z_branch:
            model_type = "single_z"
            base_fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bz{config.beta_kl_prior_z}_{model_type}_{latent_type}_{target}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}'
        elif config.only_s_branch:
            model_type = "single_s"
            base_fname = f'{config.decomp_to_perform}_NoC{config.NoC}_SNR{data_training_args.sim_snr_db}_bs{config.beta_kl_prior_s}_{model_type}_{latent_type}_{target}_z{config.z_latent_dim}_h{config.proj_codevector_dim_z}'
        
        # Save detailed results by fold
        results_df.to_csv(os.path.join(current_result_dir, f'{base_fname}_cv_details_supervised.csv'), index=False)
        
        # Save summary results
        summary_df.to_csv(os.path.join(current_result_dir, f'{base_fname}_cv_summary_supervised.csv'), index=False)
        
        # Log to wandb if available
        if is_wandb_available() and data_training_args.with_wandb:
            best_result = summary_df.loc[summary_df['Test_Accuracy_mean'].idxmax()]
            best_classifier = best_result['Classifier']
            if best_classifier == "LogisticRegression":
                best_classifier = 0
            elif best_classifier == "RandomForest":
                best_classifier = 1
            elif best_classifier == "SVC":
                best_classifier = 2
            best_accuracy = best_result['Test_Accuracy_mean']
            best_f1 = best_result['Test_F1_Score_mean']
            best_f1_ci = best_result['Test_F1_Score_CI']
            best_f1_macro = best_result['Test_F1_Macro_mean']
            best_f1_macro_ci = best_result['Test_F1_Macro_CI']
            if 'emotion' in target:
                best_unweighted_accuracy = best_result['Test_Unweighted_Accuracy_mean']
                best_ua_ci = best_result['Test_Unweighted_Accuracy_CI']
            best_wa_ci = best_result['Test_Accuracy_CI']
            
            wandb.log({
                f"supervised_cv_{target}_{latent_type}_best_classifier": best_classifier,
                f"supervised_cv_{target}_{latent_type}_best_accuracy": best_accuracy,
                f"supervised_cv_{target}_{latent_type}_best_accuracy_ci": best_wa_ci,
                f"supervised_cv_{target}_{latent_type}_best_f1": best_f1,
                f"supervised_cv_{target}_{latent_type}_best_f1_ci": best_f1_ci,
                f"supervised_cv_{target}_{latent_type}_best_f1_macro": best_f1_macro,
                f"supervised_cv_{target}_{latent_type}_best_f1_macro_ci": best_f1_macro_ci
            })
            if 'emotion' in target:
                wandb.log({
                    f"supervised_cv_{target}_{latent_type}_best_unweighted_accuracy": best_unweighted_accuracy,
                    f"supervised_cv_{target}_{latent_type}_best_unweighted_accuracy_ci": best_ua_ci
                })

    return
    
    

