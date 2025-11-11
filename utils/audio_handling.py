
def find_min_max_samples(dataset, audio_column_name):
    max_samples = 0
    min_samples = 1000000000
    for example in dataset:
        audio_length = len(example[audio_column_name]['array'])
        if audio_length > max_samples:
            max_samples = audio_length
        if audio_length < min_samples:
            min_samples = audio_length
    return max_samples, min_samples

def filter_audio_by_phonetic_detail(example):
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
