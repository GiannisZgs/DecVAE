"""
Dataset loading and initial processing for the TIMIT dataset for different scenarios.
Returns DatasetDict objects for training, validation, and testing.
Also contains necessary functions for TIMIT dataset loading and processing.
"""
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
import os
import json

SEL_PHONEMES_LIST = ['iy','ey','ay','aw','ow','uh','f','l','s'] # 'b','d','uw', 'k'
MIN_ACCEPTABLE_LENGTH = 2000 
TARGET_EXAMPLES_PER_COMBINATION = 3

def load_timit(data_training_args):
    train_datasets_splits = []
    dev_datasets_splits = []
    test_datasets_splits = []

    global unique_characters
    global phoneme39_to_id
    global phoneme48_to_id
    global unique_speaker_ids
    global speaker_id_to_id

    timit = load_dataset("timit_asr",data_dir=data_training_args.data_dir)

    "Remove SA sentences"
    timit['train'] = timit['train'].filter(lambda example: 'SA' not in example['file'])
    timit['test'] = timit['test'].filter(lambda example: 'SA' not in example['file'])
    "Remove silent parts - Start and end is always silent"
    timit['train'] = timit['train'].map(_filter_audio_by_phonetic_detail)
    timit['test'] = timit['test'].map(_filter_audio_by_phonetic_detail)

    "Core test set list according to Kaldi recipes"
    with open(data_training_args.core_test_spkrs_list, 'r') as file:
        timit_core_test_speakers_list = [line.strip().upper() for line in file]
    "Development set holdout according to Kaldi recipe - https://github.com/kaldi-asr/kaldi/blob/master/egs/timit/s5/conf/dev_spk.list"
    with open(data_training_args.dev_spkrs_list, 'r') as file:
        timit_dev_speakers_list = [line.strip().upper() for line in file]
    filtered_files_core_test = [file for file in timit['test']['file'] if any(speaker in file for speaker in timit_core_test_speakers_list)]
    filtered_files_dev = [file for file in timit['test']['file'] if any(speaker in file for speaker in timit_dev_speakers_list)]
    timit['core_test'] = timit['test'].filter(lambda example: example['file'] in filtered_files_core_test)
    timit['test'] = timit['test'].filter(lambda example: example['file'] not in filtered_files_core_test)
    timit['dev'] = timit['test'].filter(lambda example: example['file'] in filtered_files_dev)
    timit['test'] = timit['test'].filter(lambda example: example['file'] not in filtered_files_dev)


    unique_characters = []
    "Identify the phonetic vocabulary"
    timit = timit.map(_collect_unique_characters)

    "Keep 48 from 61 phonemes"
    timit = timit.map(_keep_48_timit_phonemes)
    if not os.path.exists(data_training_args.path_to_timit_phoneme48_to_id_file):
        unique_characters = []
        timit = timit.map(_collect_unique_characters)
        "Create mappings from phonemes48 to integers"
        phoneme48_to_id = {phoneme: idx for idx, phoneme in enumerate(sorted(unique_characters))}
        with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'w') as json_file:
            json.dump(phoneme48_to_id, json_file, indent=4)
    else:
        with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'r') as json_file:
            phoneme48_to_id = json.load(json_file)
    
    "Map 48 to 39"
    timit = timit.map(_phoneme_map_48_to_39_timit)  
    if not os.path.exists(data_training_args.path_to_timit_phoneme39_to_id_file):
        unique_characters = []
        timit = timit.map(_collect_unique_characters)
        "Create mappings from phonemes39 to integers"
        phoneme39_to_id = {phoneme: idx for idx, phoneme in enumerate(sorted(unique_characters))}
        with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'w') as json_file:
            json.dump(phoneme39_to_id, json_file, indent=4)            
    else:
        with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
            phoneme39_to_id = json.load(json_file)

    "Encode phonemes to integers by applying mappings"
    timit = timit.map(_encode_phonemes)     

    if not os.path.exists(data_training_args.path_to_timit_speaker_dict_file):                  
        "Collect and encode unique speakers"
        unique_speaker_ids = set()

        timit = timit.map(_collect_unique_speaker_ids)

        # Create a mapping from speaker IDs to integers
        speaker_id_to_id = {speaker_id: idx for idx, speaker_id in enumerate(sorted(unique_speaker_ids))}

        with open(data_training_args.path_to_timit_speaker_dict_file, 'w') as json_file:
            json.dump(speaker_id_to_id, json_file, indent=4)
    else:
        with open(data_training_args.path_to_timit_speaker_dict_file, 'r') as json_file:
            speaker_id_to_id = json.load(json_file)


    "Encode speakers to integers"
    timit = timit.map(_encode_speaker_ids)
    
    train_datasets_splits.append(timit['train'])
    test_datasets_splits.append(timit['core_test'])
    dev_datasets_splits.append(timit['dev'])

    "Max samples in an utterance in TIMIT are 113967 - 7.12 seconds"
    "Uncomment below to calculate again"
    #max_samples_train,min_samples_train = find_min_max_samples(timit['train'], 'audio')
    #max_samples_test,min_samples_test = find_min_max_samples(timit['core_test'], 'audio')
    #max_samples_dev,min_samples_dev = find_min_max_samples(timit['dev'], 'audio')

    raw_datasets = DatasetDict()
    if len(train_datasets_splits) > 1:
        raw_datasets["train"] = concatenate_datasets(train_datasets_splits).shuffle(seed=data_training_args.seed)
    else:
        raw_datasets["train"] = train_datasets_splits[0]
    if len(test_datasets_splits) > 1:
        raw_datasets["test"] = concatenate_datasets(test_datasets_splits).shuffle(seed=data_training_args.seed)
    else:
        raw_datasets["test"] = test_datasets_splits[0]  
    raw_datasets["dev"] = dev_datasets_splits[0]

    if data_training_args.validation_split_percentage > 0:
        num_validation_samples = raw_datasets["train"].num_rows * data_training_args.validation_split_percentage // 100
        raw_datasets["validation"] = raw_datasets["train"].select(range(num_validation_samples))
        raw_datasets["train"] = raw_datasets["train"].select(range(num_validation_samples, raw_datasets["train"].num_rows))

    return raw_datasets

def load_traversal_subset_timit(data_training_args):
    """
    Load TIMIT dataset and filter it to create balanced traversal subsets for probing models.
    Creates two unified types of traversals using all available data:
    1. Fixed phoneme, varying speaker (multiple examples per speaker, same set of speakers across phonemes)
    2. Fixed speaker, varying phoneme (multiple examples per phoneme, same set of phonemes across speakers)
    
    Also tracks and reports min/max segment lengths for analysis.
    """
    global unique_characters
    global phoneme39_to_id
    global phoneme48_to_id
    global unique_speaker_ids
    global speaker_id_to_id

    # Initialize segment length tracking variables
    min_segment_length = float('inf')
    max_segment_length = 0
    segment_lengths = []

    timit = load_dataset("timit_asr", data_dir=data_training_args.data_dir)

    "Remove SA sentences"
    timit['train'] = timit['train'].filter(lambda example: 'SA' not in example['file'])
    timit['test'] = timit['test'].filter(lambda example: 'SA' not in example['file'])
    "Remove silent parts - Start and end is always silent"
    timit['train'] = timit['train'].map(_filter_audio_by_phonetic_detail)
    timit['test'] = timit['test'].map(_filter_audio_by_phonetic_detail)

    # Process phoneme mappings
    unique_characters = []
    timit = timit.map(_collect_unique_characters)
    timit = timit.map(_keep_48_timit_phonemes)
    
    # Load phoneme mappings
    with open(data_training_args.path_to_timit_phoneme48_to_id_file, 'r') as json_file:
        phoneme48_to_id = json.load(json_file)
    
    timit = timit.map(_phoneme_map_48_to_39_timit)
    
    with open(data_training_args.path_to_timit_phoneme39_to_id_file, 'r') as json_file:
        phoneme39_to_id = json.load(json_file)

    with open(data_training_args.path_to_timit_speaker_dict_file, 'r') as json_file:
        speaker_id_to_id = json.load(json_file)
    id_to_speaker_id = {v: k for k, v in speaker_id_to_id.items()}
    
    # Encode phonemes and speakers to integers
    timit = timit.map(_encode_phonemes)
    timit = timit.map(_encode_speaker_ids)
    
    # Get id to phoneme mapping for selected phonemes
    id_to_phoneme39 = {v: k for k, v in phoneme39_to_id.items()}
    selected_phoneme_ids = [phoneme39_to_id[p] for p in SEL_PHONEMES_LIST if p in phoneme39_to_id]
    
    # Function to filter examples containing a specific phoneme
    def contains_phoneme(example, phoneme_id):
        return phoneme_id in example['phonetic_detail']['utterance39']
    
    # Function to extract segments for a specific phoneme
    def extract_phoneme_segments(example, phoneme_id):
        segments = []
        for i, p_id in enumerate(example['phonetic_detail']['utterance39']):
            if p_id == phoneme_id:
                start = example['phonetic_detail']['start'][i]
                stop = example['phonetic_detail']['stop'][i]
                
                # Calculate segment length
                segment_length = stop - start
                
                # Skip segments that are too short
                if segment_length < MIN_ACCEPTABLE_LENGTH:
                    continue
                    
                # Create a new example with just this segment
                segment = example.copy()
                segment_audio = example['audio']['array'][start:stop]
                
                # Track segment length
                nonlocal min_segment_length, max_segment_length, segment_lengths
                min_segment_length = min(min_segment_length, segment_length)
                max_segment_length = max(max_segment_length, segment_length)
                segment_lengths.append(segment_length)
                
                segment['audio'] = {
                    'array': segment_audio,
                    'sampling_rate': example['audio']['sampling_rate']
                }
                segment['phonetic_detail'] = {
                    'utterance39': [p_id],
                    'utterance48': [example['phonetic_detail']['utterance48'][i]],
                    'start': [0],
                    'stop': [stop-start]
                }
                
                # Store segment length for reporting
                segment['segment_length'] = segment_length
                
                segments.append(segment)
        return segments    
    
    # Combine train and test sets for unified processing
    all_examples = []
    all_examples.extend([ex for ex in timit['train']])
    all_examples.extend([ex for ex in timit['test']])
    
    print(f"Processing {len(all_examples)} total examples from TIMIT")
    
    # First, collect all phoneme-speaker availability information
    phoneme_to_speakers = {phoneme_id: set() for phoneme_id in selected_phoneme_ids}
    speaker_to_phonemes = {}
    
    for example in all_examples:
        speaker_id = example['speaker_id']
        speaker_key = id_to_speaker_id.get(speaker_id)
        
        if speaker_key not in speaker_to_phonemes:
            speaker_to_phonemes[speaker_key] = set()
            
        for phoneme_id in selected_phoneme_ids:
            if contains_phoneme(example, phoneme_id):
                segments = extract_phoneme_segments(example, phoneme_id)
                if segments:
                    phoneme_to_speakers[phoneme_id].add(speaker_key)
                    speaker_to_phonemes[speaker_key].add(phoneme_id)
    
    # Find speakers who have all phonemes
    speakers_with_all_phonemes = set()
    for speaker, phonemes in speaker_to_phonemes.items():
        if len(phonemes) == len(selected_phoneme_ids):
            speakers_with_all_phonemes.add(speaker)
    
    # Find the intersection of speakers across all phonemes
    common_speakers = None
    for phoneme_id in selected_phoneme_ids:
        if common_speakers is None:
            common_speakers = phoneme_to_speakers[phoneme_id].copy()
        else:
            common_speakers &= phoneme_to_speakers[phoneme_id]
    
    if not common_speakers:
        print("Warning: No speakers found with examples for all phonemes!")
        common_speakers = set()
    
    print(f"Found {len(common_speakers)} speakers that appear in all selected phonemes")
    
    
    # 1. Fixed phoneme, varying speaker datasets (multiple samples per speaker, only common speakers)
    fixed_phoneme_datasets = []
    
    phoneme_min_lengths = {}
    phoneme_max_lengths = {}
    phoneme_speaker_counts = {}
    
    for phoneme_id in selected_phoneme_ids:
        phoneme_str = id_to_phoneme39[phoneme_id]
        phoneme_min_lengths[phoneme_str] = float('inf')
        phoneme_max_lengths[phoneme_str] = 0
        
        # Collect examples for this phoneme across common speakers only
        phoneme_examples = []
        speaker_counts = {}  # Track examples per speaker
        
        for example in all_examples:
            # Get speaker ID and check if it's in our common set
            speaker_id = example['speaker_id']
            speaker_key = id_to_speaker_id.get(speaker_id)
            
            if speaker_key not in common_speakers:
                continue
                
            # Skip if we already have enough examples for this speaker
            if speaker_counts.get(speaker_key, 0) >= TARGET_EXAMPLES_PER_COMBINATION:
                continue
                
            if contains_phoneme(example, phoneme_id):
                segments = extract_phoneme_segments(example, phoneme_id)
                if segments:
                    phoneme_examples.append(segments[0])
                    speaker_counts[speaker_key] = speaker_counts.get(speaker_key, 0) + 1
                    
                    # Update phoneme-specific length tracking
                    segment_length = segments[0]['segment_length']
                    phoneme_min_lengths[phoneme_str] = min(phoneme_min_lengths[phoneme_str], segment_length)
                    phoneme_max_lengths[phoneme_str] = max(phoneme_max_lengths[phoneme_str], segment_length)
        
        if phoneme_examples:
            # Count speakers with at least one example
            speakers_with_examples = len(speaker_counts)
            # Count total examples
            total_examples = sum(speaker_counts.values())
            
            phoneme_speaker_counts[phoneme_str] = speakers_with_examples
            fixed_phoneme_dataset = Dataset.from_dict({
                k: [example[k] for example in phoneme_examples]
                for k in phoneme_examples[0].keys()
            })
            fixed_phoneme_datasets.append({
                'dataset': fixed_phoneme_dataset,
                'fixed_factor': 'phoneme',
                'fixed_value': phoneme_str,
                'num_speakers': speakers_with_examples,
                'total_examples': total_examples,
                'speakers': sorted(list(speaker_counts.keys())),
                'speaker_counts': speaker_counts
            })
    
    # 2. Fixed speaker, varying phoneme datasets (multiple examples per phoneme, only common speakers)
    fixed_speaker_datasets = []
    
    speaker_min_lengths = {}
    speaker_max_lengths = {}
    speaker_phoneme_counts = {}
    
    for speaker in common_speakers:
        speaker_min_lengths[speaker] = float('inf')
        speaker_max_lengths[speaker] = 0
        
        # Get all examples from this speaker
        speaker_examples = []
        for example in all_examples:
            example_speaker = example['speaker_id']
            speaker_key = id_to_speaker_id.get(example_speaker)
            if speaker_key == speaker:
                speaker_examples.append(example)

        # For each phoneme, collect multiple examples
        all_phoneme_segments = []
        phoneme_counts = {}  # Track examples per phoneme
        
        for phoneme_id in selected_phoneme_ids:
            # Skip if we already have enough for this phoneme
            if phoneme_counts.get(phoneme_id, 0) >= TARGET_EXAMPLES_PER_COMBINATION:
                continue
                
            # Try to find examples for this phoneme
            for example in speaker_examples:
                if contains_phoneme(example, phoneme_id):
                    segments = extract_phoneme_segments(example, phoneme_id)
                    if segments and phoneme_counts.get(phoneme_id, 0) < TARGET_EXAMPLES_PER_COMBINATION:
                        segment = segments[0]
                        all_phoneme_segments.append(segment)
                        phoneme_counts[phoneme_id] = phoneme_counts.get(phoneme_id, 0) + 1
                        
                        # Update speaker-specific length tracking
                        segment_length = segment['segment_length']
                        speaker_min_lengths[speaker] = min(speaker_min_lengths[speaker], segment_length)
                        speaker_max_lengths[speaker] = max(speaker_max_lengths[speaker], segment_length)
        
        # Include speaker if we have examples for all phonemes
        if len(phoneme_counts) == len(selected_phoneme_ids):
            # Calculate phoneme coverage stats
            phonemes_with_examples = len(phoneme_counts)
            total_examples = sum(phoneme_counts.values())
            
            speaker_phoneme_counts[speaker] = total_examples
            
            fixed_speaker_dataset = Dataset.from_dict({
                k: [example[k] for example in all_phoneme_segments]
                for k in all_phoneme_segments[0].keys()
            })
            
            fixed_speaker_datasets.append({
                'dataset': fixed_speaker_dataset,
                'fixed_factor': 'speaker',
                'fixed_value': speaker,
                'num_phonemes': phonemes_with_examples,
                'total_examples': total_examples,
                'phoneme_counts': {id_to_phoneme39[p_id]: count for p_id, count in phoneme_counts.items()}
            })
    
    # Create the final datasets
    raw_datasets = DatasetDict()
    
    # Combine all fixed phoneme datasets
    if fixed_phoneme_datasets:
        all_fixed_phoneme = concatenate_datasets([d['dataset'] for d in fixed_phoneme_datasets])
        raw_datasets["fixed_phoneme"] = all_fixed_phoneme
        print(f"Created fixed_phoneme dataset with {all_fixed_phoneme.num_rows} examples")
    
    # Combine all fixed speaker datasets
    if fixed_speaker_datasets:
        all_fixed_speaker = concatenate_datasets([d['dataset'] for d in fixed_speaker_datasets])
        raw_datasets["fixed_speaker"] = all_fixed_speaker
        print(f"Created fixed_speaker dataset with {all_fixed_speaker.num_rows} examples")
    
    # Store metadata about the traversals and segment lengths
    raw_datasets.traversal_metadata = {
        'fixed_phoneme': fixed_phoneme_datasets,
        'fixed_speaker': fixed_speaker_datasets,
        'common_speakers': sorted(list(common_speakers)),
        'target_examples_per_combination': TARGET_EXAMPLES_PER_COMBINATION
    }
    
    # Store segment length statistics
    raw_datasets.segment_length_stats = {
        'overall_min_length': min_segment_length,
        'overall_max_length': max_segment_length,
        'phoneme_min_lengths': phoneme_min_lengths,
        'phoneme_max_lengths': phoneme_max_lengths,
        'speaker_min_lengths': speaker_min_lengths,
        'speaker_max_lengths': speaker_max_lengths,
    }
    
    # Report segment length statistics
    if min_segment_length != float('inf'):
        sample_rate = 16000  # TIMIT sample rate
        min_duration_ms = min_segment_length * 1000 / sample_rate
        max_duration_ms = max_segment_length * 1000 / sample_rate
        
        print("\n===== Audio Segment Length Statistics =====")
        print(f"Min segment length: {min_segment_length} samples ({min_duration_ms:.2f} ms)")
        print(f"Max segment length: {max_segment_length} samples ({max_duration_ms:.2f} ms)")
        
        print("\n----- Phoneme-specific Statistics -----")
        for phoneme, min_len in sorted(phoneme_min_lengths.items()):
            if min_len != float('inf'):
                min_dur_ms = min_len * 1000 / sample_rate
                max_dur_ms = phoneme_max_lengths[phoneme] * 1000 / sample_rate
                speakers = phoneme_speaker_counts.get(phoneme, 0)
                print(f"Phoneme '{phoneme}': {min_len}-{phoneme_max_lengths[phoneme]} samples ({min_dur_ms:.2f}-{max_dur_ms:.2f} ms), {speakers} speakers")
        
        print("\n----- Speaker Coverage Statistics -----")
        print(f"Common speakers across all phonemes: {len(common_speakers)}")
        print(f"Speakers with all phonemes included in dataset: {len(fixed_speaker_datasets)}")
        print(f"Target examples per combination: {TARGET_EXAMPLES_PER_COMBINATION}")
        
        # Calculate average examples per combination
        total_phoneme_examples = sum(d['total_examples'] for d in fixed_phoneme_datasets)
        total_phoneme_combinations = sum(d['num_speakers'] for d in fixed_phoneme_datasets)
        if total_phoneme_combinations > 0:
            avg_examples_per_phoneme_combo = total_phoneme_examples / total_phoneme_combinations
            print(f"Average examples per phoneme-speaker combination: {avg_examples_per_phoneme_combo:.2f}")
        
        total_speaker_examples = sum(d['total_examples'] for d in fixed_speaker_datasets)
        total_speaker_combinations = sum(d['num_phonemes'] for d in fixed_speaker_datasets)
        if total_speaker_combinations > 0:
            avg_examples_per_speaker_combo = total_speaker_examples / total_speaker_combinations
            print(f"Average examples per speaker-phoneme combination: {avg_examples_per_speaker_combo:.2f}")
        
        # Verify dataset consistency
        phoneme_speaker_sets = [set(d['speakers']) for d in fixed_phoneme_datasets]
        all_same_speakers = all(s == phoneme_speaker_sets[0] for s in phoneme_speaker_sets)
        if all_same_speakers:
            print("\n✓ All fixed_phoneme datasets use the same set of speakers")
        else:
            print("\nWARNING: Not all fixed_phoneme datasets use the same set of speakers")
    
    return raw_datasets

def _filter_audio_by_phonetic_detail(example):
    phonetic_detail = example['phonetic_detail']
    audio = example['audio']['array']
    start_index = None
    end_index = None

    # Find the end of the first "h#" label
    for i, label in enumerate(phonetic_detail['utterance']):
        if label == "h#":
            start_index = phonetic_detail['stop'][i]
            break

    # Find the start of the last "h#" label
    for i, label in reversed(list(enumerate(phonetic_detail['utterance']))):
        if label == "h#":
            end_index = phonetic_detail['start'][i]
            break

    # Filter the audio data
    if start_index is not None and end_index is not None and start_index < end_index:
        example['audio']['array'] = audio[start_index:end_index]
        example['phonetic_detail']['utterance'] = phonetic_detail['utterance'][1:-1]
        example['phonetic_detail']['start'] = phonetic_detail['start'][1:-1]
        new_start_point = example['phonetic_detail']['start'][0]
        example['phonetic_detail']['start'] = [e-new_start_point for e in example['phonetic_detail']['start']]
        example['phonetic_detail']['stop'] = phonetic_detail['stop'][1:-1]
        example['phonetic_detail']['stop'] = [e-new_start_point for e in example['phonetic_detail']['stop']]
    elif start_index is not None and end_index is None:
        example['audio']['array'] = audio[start_index:]
        example['phonetic_detail']['utterance'] = phonetic_detail['utterance'][1:]
        example['phonetic_detail']['start'] = phonetic_detail['start'][1:]
        new_start_point = example['phonetic_detail']['start'][0]
        example['phonetic_detail']['start'] = [e-new_start_point for e in example['phonetic_detail']['start']]
        example['phonetic_detail']['stop'] = phonetic_detail['stop'][1:]
        example['phonetic_detail']['stop'] = [e-new_start_point for e in example['phonetic_detail']['stop']]
    elif start_index is None and end_index is not None:
        example['audio']['array'] = audio[:end_index]
        example['phonetic_detail']['utterance'] = phonetic_detail['utterance'][:-1]
        example['phonetic_detail']['start'] = phonetic_detail['start'][:-1]
        example['phonetic_detail']['stop'] = phonetic_detail['stop'][:-1]

    return example

def _collect_unique_characters(example):
    phonetic_detail = example['phonetic_detail']
    if 'utterance' in phonetic_detail:
        utterance = phonetic_detail['utterance']
    elif 'utterance48' in phonetic_detail and 'utterance39' not in phonetic_detail:
        utterance = phonetic_detail['utterance48']
    elif 'utterance39' in phonetic_detail:
        utterance = phonetic_detail['utterance39']
    for char in utterance:
        if char not in unique_characters:
            unique_characters.append(char)
    return example

def _keep_48_timit_phonemes(sample):
    fold_map = {'eng':'ng',
                'ux':'uw',
                'axr':'er',
                'ax-h':'ax',
                'em':'m',
                'nx':'n',
                'hv':'hh',
                'pcl':'cl',
                'tcl':'cl',
                'kcl':'cl',
                'qcl':'cl',
                'bcl':'vcl',
                'dcl':'vcl',
                'gcl':'vcl',
                'h#':'sil',
                '#h':'sil',
                'pau':'sil',
            }
    phonetic_detail = sample['phonetic_detail']['utterance']
    to_discard = []
    for i, phoneme in enumerate(phonetic_detail):
        if phoneme == 'q':
            #Discard glottal stops
            to_discard.append(i)
        if phoneme in fold_map:
            sample['phonetic_detail']['utterance'][i] = fold_map[phoneme]
    
    sample['phonetic_detail']['utterance48'] = [phoneme for i, phoneme in enumerate(sample['phonetic_detail']['utterance']) if i not in to_discard]
    sample['phonetic_detail']['start'] = [start for i, start in enumerate(sample['phonetic_detail']['start']) if i not in to_discard]
    sample['phonetic_detail']['stop'] = [stop for i, stop in enumerate(sample['phonetic_detail']['stop']) if i not in to_discard]
    del sample['phonetic_detail']['utterance']

    assert len(sample['phonetic_detail']['utterance48']) == len(sample['phonetic_detail']['start']) == len(sample['phonetic_detail']['stop'])

    return sample

def _phoneme_map_48_to_39_timit(sample):
    fold_map = {'cl':'sil',
                'vcl':'sil',
                'epi':'sil',
                'el':'l',
                'en':'n',
                'zh':'sh',
                'aa':'ao',
                'ih':'ix',
                'ah':'ax',                            
            }
    phonetic_detail = sample['phonetic_detail']['utterance48']
    new_phonemes = []
    for i, phoneme in enumerate(phonetic_detail):
        if phoneme in fold_map:
            new_phonemes.append(fold_map[phoneme])
        else:
            new_phonemes.append(phoneme)
    sample['phonetic_detail']['utterance39'] = new_phonemes

    assert len(sample['phonetic_detail']['utterance48']) == len(sample['phonetic_detail']['utterance39'])
    return sample

def _encode_phonemes(example):
    example['phonetic_detail']['utterance39'] = [phoneme39_to_id[phoneme] for phoneme in example['phonetic_detail']['utterance39']]
    example['phonetic_detail']['utterance48'] = [phoneme48_to_id[phoneme] for phoneme in example['phonetic_detail']['utterance48']]
    return example

def _collect_unique_speaker_ids(example):
    unique_speaker_ids.add(example['speaker_id'])
    return example

def _encode_speaker_ids(example):
    example['speaker_id'] = speaker_id_to_id[example['speaker_id']]
    return example