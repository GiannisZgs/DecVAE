"""Helpers for naming the .arrow cache files that hold the preprocessed inputs.

Feature extraction happens at the preprocessing step, so the cached files hold either the raw
decomposition or the extracted features. The two are not interchangeable, and the same dataset is
used with different input types, so the feature type is encoded in the file name.
"""


def cache_feature_type(input_type):
    """
    Map an input type onto the feature type stored in the cache. The ICA/PCA evaluations use
    variants that differ only in which decomposition components they read (e.g. 'mel_ocs' takes the
    components without the original signal, 'mel_all' takes all of them). All components are cached,
    so those variants share a single cache with the plain type they derive from.

    Args:
        input_type (str): The input type, e.g. 'mel', 'mel_ocs', 'waveform_all', 'fft'
    Returns:
        feature_type (str): The feature type held in the cache - 'mel', 'waveform' or 'fft'
    """
    if input_type is None:
        return "waveform"
    if input_type.startswith("mel"):
        return "mel"
    if input_type.startswith("waveform"):
        return "waveform"
    return input_type


def _add_feature_type(path, feature_type):
    if path is None:
        return None
    stem = path[:-len(".arrow")] if path.endswith(".arrow") else path
    return f"{stem}_{feature_type}.arrow"


def build_cache_file_names(data_training_args, input_type):
    """
    Build the cache_file_names mapping passed to Dataset.map(), with the feature type encoded in
    each file name so that caches holding different feature types cannot be mistaken for each other.

    Args:
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`): Holds
            the configured cache file names and the number of preprocessing workers
        input_type (str): The input type of the run, mapped through cache_feature_type
    Returns:
        cache_file_names (dict): Split name -> list of cache file paths, or None when not caching
    """
    feature_type = cache_feature_type(input_type)
    num_workers = data_training_args.preprocessing_num_workers

    names = {
        "train": getattr(data_training_args, "train_cache_file_name", None),
        "validation": getattr(data_training_args, "validation_cache_file_name", None),
        "test": getattr(data_training_args, "test_cache_file_name", None),
        "dev": getattr(data_training_args, "dev_cache_file_name", None),
    }

    if names["train"] is None and names["validation"] is None:
        return {"train": None, "validation": None}

    def shards(path):
        path = _add_feature_type(path, feature_type)
        if num_workers is not None and num_workers > 1:
            stem = path[:-len(".arrow")]
            return [f"{stem}_0000{i}_of_0000{num_workers}.arrow" for i in range(num_workers)]
        return [path]

    "A split whose path is not configured is left out entirely - the latent traversals for the"
    "simulated vowels, for instance, define a train file only"
    cache_file_names = {}
    for split in ("train", "validation", "test", "dev"):
        if names[split] is not None:
            cache_file_names[split] = shards(names[split])
    return cache_file_names


def build_map_cache_file_names(data_training_args, input_type):
    """
    Build the cache_file_names mapping passed to DatasetDict.map(), which writes the cache. Uses
    plain paths rather than the per-shard lists that reading needs, and must encode the feature
    type the same way build_cache_file_names does, or what is written is never read back.

    Args:
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`): Holds
            the configured cache file names
        input_type (str): The input type of the run, mapped through cache_feature_type
    Returns:
        cache_file_names (dict or None): Split name -> cache file path, for every configured split
    """
    feature_type = cache_feature_type(input_type)
    splits = (
        ("train", "train_cache_file_name"),
        ("validation", "validation_cache_file_name"),
        ("test", "test_cache_file_name"),
        ("dev", "dev_cache_file_name"),
    )
    names = {}
    for split, attr in splits:
        path = getattr(data_training_args, attr, None)
        if path is not None:
            names[split] = _add_feature_type(path, feature_type)
    return names or None
