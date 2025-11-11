"""
The purpose of this script is to split the data into smaller chunks for easier processing and storage.
These data are used to obtain Supplementary Information Figures 4 and 6.
"""
import os
import json

DECOMP = "filter"
NOC = 3
LEVEL = "frame"

def load_json_files(directory, features):
    data = {
        "sim_vowels": {},
        "timit": {}
    }
    
    # First load all parts
    for dataset in data.keys():
        concatenated_data = {feature: [] for feature in features}
        
        for part in range(1, 5):
            filename = f"{dataset}_dec_quality_{LEVEL}_NoC{NOC}_{DECOMP}_part{part}.json"
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r') as file:
                    part_data = json.load(file)
                    # Only keep specified features and extend their lists
                    for feature in part_data.keys():
                        if feature in features:
                            concatenated_data[feature].extend(part_data[feature])
                        # Free up memory
                        part_data[feature] = None                        
            else:
                print(f"File {filename} does not exist in the directory {directory}")
            part_data = None
            print(f"Finished loading part {part} of {dataset}")
        # Store concatenated data for each dataset
        data[dataset] = concatenated_data

    return data


directory = os.path.join("../data_for_figures/decomposition_quality/",DECOMP)
features = ["detected_frequencies","vowels","consonants","gender","speaker_id","overlap_mask"]
data = load_json_files(directory,features)

with open(os.path.join(directory,f'dec_quality_{LEVEL}_NoC{NOC}_{DECOMP}_det_freqs.json'), 'w') as f:
    json.dump(data, f)