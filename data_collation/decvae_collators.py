"""
Data collators for DecVAE models pretraining, fine-tuning, disentanglement evaluations and latent response evaluations.
Adapted from the HuggingFace Wav2Vec2 data collators under the Apache License 2.0.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2FeatureExtractor
from config_files.configuration_decVAE import DecVAEConfig
from args_configs.model_args import ModelArguments, ModelArgumentsPost
from args_configs.data_training_args import DataTrainingArguments, DataTrainingArgumentsPost
from models.dec_vae import DecVAEForPreTraining
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices
from feature_extraction import extract_mel_spectrogram, extract_fft_psd


@dataclass
class DataCollatorForDecVAEPretraining:
    """
    Data collator that will dynamically pad the inputs received, extract features, align them with labels and variables, 
    and prepare masked indices for self-supervised pretraining of a Variational Decomposition Autoencoder.

    Args:
        model (:class:`~DecVAEForPreTraining`):
            The DecVAE model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        dataset_name (:obj:`str`): The name of the dataset being used.
        model_args (:class:`~args_configs.model_args.ModelArguments`): The model arguments dictionary with necessary model-related parameters.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`): The data training arguments dictionary 
            with necessary data-related parameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object.
        model_name (:obj:`str`, `optional`, defaults to :obj:`"DecVAE"`): The name of the model being used.
        input_type (:obj:`str`, `optional`, defaults to :obj:`"waveform"`): The type of input features - 
            waveform, mel filterbank features, or fft features.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: DecVAEForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    dataset_name: str 
    model_args: ModelArguments
    data_training_args: DataTrainingArguments
    config: DecVAEConfig
    model_name: str = "DecVAE"
    input_type: str = "waveform"
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        if self.dataset_name in ["timit", "iemocap"]:
            if self.dataset_name == "timit":
                phonemes39 = [feature.pop("phonemes39") for feature in features]
                phonemes48 = [feature.pop("phonemes48") for feature in features]
            elif self.dataset_name == "iemocap":
                phonemes = [feature.pop("phonemes") for feature in features]
                emotions = [feature.pop("emotion_labels") for feature in features]
            start_phonemes = [feature.pop("start_phonemes") for feature in features]
            stop_phonemes = [feature.pop("stop_phonemes") for feature in features]
            overlap_mask = [feature.pop("overlap_mask") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            if self.dataset_name != "iemocap":
                [feature.pop("words") for feature in features];

        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        if batch.get("input_seq_values") is not None:
            "Pad input_seq_values"
            batch['input_seq_values'] = pad_sequence([input_seq_values.clone().detach().to(device) for input_seq_values in batch['input_seq_values']],batch_first=True, padding_value=0.0)

        "If input will be mel filterbank features, split sequence in frames"
        if self.input_type == 'mel':
            frame_len = batch["input_values"].shape[-1]
            if batch.get("input_seq_values") is not None:
                frames = batch["input_seq_values"].shape[-1]/frame_len
                new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames),frame_len),device = batch["input_seq_values"].device)
            
                # Split sequence into frames 
                for o in range(batch["input_values"].shape[1]):
                    sequence = batch["input_seq_values"][:,o,:].clone()
                    for f in range(int(frames)):
                        framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                        new_input_seq_values[:,o,f,:] = framed_sequence.clone()
                batch["input_seq_values"] = new_input_seq_values.clone()
        
            if len(self.model_args.conv_kernel) == 7:
                assert self.data_training_args.mel_hops == 4
            elif len(self.model_args.conv_kernel) == 5:
                assert self.data_training_args.mel_hops == 3
            for o in range(batch["input_values"].shape[1]):
                batch["input_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)
                if batch.get("input_seq_values") is not None:
                    batch["input_seq_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)

            if batch.get("input_seq_values") is not None:
                "Flatten sequence - Reverse framing"
                batch["input_seq_values"] = batch["input_seq_values"].reshape(batch["input_seq_values"].shape[0],batch["input_seq_values"].shape[1],-1)

        elif self.input_type == 'fft':                               
            # Apply fft using welch's power spectral density estimation
            batch["input_values"], batch["input_seq_values"] = extract_fft_psd(
                batch, 
                normalize=True, #self.data_training_args.mel_norm,
                device=batch["input_values"].device
            )
                                

        mask_indices_seq_length = batch["input_values"].shape[-2]
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)
        
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length) #length of z in embedding space after feature encoder

        if self.dataset_name in ["timit", "iemocap"]:
            "First pad to match length of largest phoneme vector"
            "Then concatenate to match the length of the input"
            if self.dataset_name == "timit":
                batch["phonemes39"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes39],batch_first=True, padding_value=-100) 
                batch["phonemes39"] = torch.cat(((batch["phonemes39"],-100*torch.ones((batch["phonemes39"].shape[0],mask_indices_seq_length-batch["phonemes39"].shape[1])))),dim=1)

                batch["phonemes48"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes48],batch_first=True, padding_value=-100) 
                batch["phonemes48"] = torch.cat(((batch["phonemes48"],-100*torch.ones((batch["phonemes48"].shape[0],mask_indices_seq_length-batch["phonemes48"].shape[1])))),dim=1)
            elif self.dataset_name == "iemocap":
                batch["phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes],batch_first=True, padding_value=-100) 
                batch["phonemes"] = torch.cat(((batch["phonemes"],-100*torch.ones((batch["phonemes"].shape[0],mask_indices_seq_length-batch["phonemes"].shape[1])))),dim=1)

                batch["emotion"] = torch.tensor(emotions, dtype=torch.long, device=device)
            
            batch["start_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in start_phonemes],batch_first=True, padding_value=-100) 
            batch["start_phonemes"] = torch.cat(((batch["start_phonemes"],-100*torch.ones((batch["start_phonemes"].shape[0],mask_indices_seq_length-batch["start_phonemes"].shape[1])))),dim=1)

            batch["stop_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in stop_phonemes],batch_first=True, padding_value=-100) 
            batch["stop_phonemes"] = torch.cat(((batch["stop_phonemes"],-100*torch.ones((batch["stop_phonemes"].shape[0],mask_indices_seq_length-batch["stop_phonemes"].shape[1])))),dim=1)

            batch["overlap_mask"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in overlap_mask],batch_first=True, padding_value=-100) 
            batch["overlap_mask"] = torch.cat(((batch["overlap_mask"],-100*torch.ones((batch["overlap_mask"].shape[0],mask_indices_seq_length-batch["overlap_mask"].shape[1])))),dim=1)

            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
        

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        "If last frame is not whole, it will be discarded as per our convolution configuration"
        if self.dataset_name == "iemocap":
            for i in range(batch["phonemes"].shape[0]):
                if (batch["phonemes"][i] != -100).sum() - batch["sub_attention_mask"][i].sum() == 1:
                    batch['phonemes'][i, batch["sub_attention_mask"][i].sum()] = -100
                    batch['overlap_mask'][i, batch["sub_attention_mask"][i].sum()] = -100
            assert (batch["phonemes"] != -100).sum() - batch["sub_attention_mask"].sum() == 0, "The number of phonemes does not match the number of frames in the input sequence. Please check your data preprocessing."
        #Sub-attention mask denotes the length of each segment in the batch - Useful in the case that the 
        #inputs are not padded
        features_shape = (batch_size, mask_indices_seq_length) #features in embedding space - z

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


@dataclass
class DataCollatorForDecVAE_SSL_FineTuning:
    """
    Data collator that will dynamically pad the inputs received, extract features, align them with labels and variables, 
    and prepare masked indices for self-supervised fine-tuning of a Variational Decomposition Autoencoder. Supports slightly
    different functionality to account for datasets that are used only for fine-tuning e.g. VOC-ALS.

    Args:
        model (:class:`~DecVAEForPreTraining`):
            The DecVAE model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        dataset_name (:obj:`str`): The name of the dataset being used.
        model_args (:class:`~args_configs.model_args.ModelArguments`): The model arguments dictionary with necessary model-related parameters.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArguments`): The data training arguments dictionary 
            with necessary data-related parameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object.
        model_name (:obj:`str`, `optional`, defaults to :obj:`"DecVAE"`): The name of the model being used.
        input_type (:obj:`str`, `optional`, defaults to :obj:`"waveform"`): The type of input features - 
            waveform, mel filterbank features, or fft features.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: DecVAEForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    dataset_name: str 
    model_args: ModelArguments
    data_training_args: DataTrainingArguments
    config: DecVAEConfig
    model_name: str = "DecVAE"
    input_type: str = "waveform"
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format

        if self.dataset_name in ["timit", "iemocap"]:
            if self.dataset_name == "timit":
                phonemes39 = [feature.pop("phonemes39") for feature in features]
                phonemes48 = [feature.pop("phonemes48") for feature in features]
            elif self.dataset_name == "iemocap":
                phonemes = [feature.pop("phonemes") for feature in features]
                emotions = [feature.pop("emotion_labels") for feature in features]
            start_phonemes = [feature.pop("start_phonemes") for feature in features]
            stop_phonemes = [feature.pop("stop_phonemes") for feature in features]
            overlap_mask = [feature.pop("overlap_mask") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            if self.dataset_name != "iemocap":
                [feature.pop("words") for feature in features];
        elif "VOC_ALS" in self.dataset_name:
            alsfrs_total = [feature.pop("alsfrs_total_enc") for feature in features]
            disease_duration = [feature.pop("disease_duration_enc") for feature in features]
            king_stage = [feature.pop("king_stage_enc") for feature in features]
            alsfrs_speech = [feature.pop("alsfrs_speech_enc") for feature in features]
            cantagallo = [feature.pop("cantagallo_enc") for feature in features]
            phonemes = [feature.pop("phonemes") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            group = [feature.pop("group") for feature in features]
        
        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        if batch.get("input_seq_values") is not None:
            "Pad input_seq_values"
            batch['input_seq_values'] = pad_sequence([input_seq_values.clone().detach().to(device) for input_seq_values in batch['input_seq_values']],batch_first=True, padding_value=0.0)

        "If input will be mel filterbank features, split sequence in frames"
        if self.input_type == 'mel':
            frame_len = batch["input_values"].shape[-1]
            if batch.get("input_seq_values") is not None:
                frames = batch["input_seq_values"].shape[-1]/frame_len
                new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames),frame_len),device = batch["input_seq_values"].device)
            
                # Split sequence into frames 
                for o in range(batch["input_values"].shape[1]):
                    sequence = batch["input_seq_values"][:,o,:].clone()
                    for f in range(int(frames)):
                        framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                        new_input_seq_values[:,o,f,:] = framed_sequence.clone()
                batch["input_seq_values"] = new_input_seq_values.clone()
        
            if len(self.model_args.conv_kernel) == 7:
                assert self.data_training_args.mel_hops == 4
            elif len(self.model_args.conv_kernel) == 5:
                assert self.data_training_args.mel_hops == 3
            for o in range(batch["input_values"].shape[1]):
                batch["input_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)
                if batch.get("input_seq_values") is not None:
                    batch["input_seq_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)

            if batch.get("input_seq_values") is not None:
                "Flatten sequence - Reverse framing"
                batch["input_seq_values"] = batch["input_seq_values"].reshape(batch["input_seq_values"].shape[0],batch["input_seq_values"].shape[1],-1)

        elif self.input_type == 'fft':                               
            # Apply fft using welch's power spectral density estimation
            batch["input_values"], batch["input_seq_values"] = extract_fft_psd(
                batch, 
                normalize=True, #self.data_training_args.mel_norm,
                device=batch["input_values"].device
            )
                                

        mask_indices_seq_length = batch["input_values"].shape[-2]
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length) #length of z in embedding space after feature encoder

        if self.dataset_name in ["timit", "librispeech_asr", "iemocap"]:
            "First pad to match length of largest phoneme vector"
            "Then concatenate to match the length of the input"
            if self.dataset_name == "timit":
                batch["phonemes39"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes39],batch_first=True, padding_value=-100) 
                batch["phonemes39"] = torch.cat(((batch["phonemes39"],-100*torch.ones((batch["phonemes39"].shape[0],mask_indices_seq_length-batch["phonemes39"].shape[1]), device=device))),dim=1)

                batch["phonemes48"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes48],batch_first=True, padding_value=-100) 
                batch["phonemes48"] = torch.cat(((batch["phonemes48"],-100*torch.ones((batch["phonemes48"].shape[0],mask_indices_seq_length-batch["phonemes48"].shape[1]), device=device))),dim=1)
            elif self.dataset_name == "iemocap":
                batch["phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes],batch_first=True, padding_value=-100) 
                batch["phonemes"] = torch.cat(((batch["phonemes"],-100*torch.ones((batch["phonemes"].shape[0],mask_indices_seq_length-batch["phonemes"].shape[1]), device=device))),dim=1)

                batch["emotion"] = torch.tensor(emotions, dtype=torch.long, device=device)
            
            batch["start_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in start_phonemes],batch_first=True, padding_value=-100) 
            batch["start_phonemes"] = torch.cat(((batch["start_phonemes"],-100*torch.ones((batch["start_phonemes"].shape[0],max(mask_indices_seq_length-batch["start_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["stop_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in stop_phonemes],batch_first=True, padding_value=-100) 
            batch["stop_phonemes"] = torch.cat(((batch["stop_phonemes"],-100*torch.ones((batch["stop_phonemes"].shape[0],max(mask_indices_seq_length-batch["stop_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["overlap_mask"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in overlap_mask],batch_first=True, padding_value=-100) 
            batch["overlap_mask"] = torch.cat(((batch["overlap_mask"],-100*torch.ones((batch["overlap_mask"].shape[0],mask_indices_seq_length-batch["overlap_mask"].shape[1]), device=device))),dim=1)

            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
        
        elif "VOC_ALS" in self.dataset_name:
            batch["alsfrs_total"] = torch.tensor(alsfrs_total, dtype=torch.long, device=device)
            batch["disease_duration"] = torch.tensor(disease_duration, dtype=torch.long, device=device)
            batch["king_stage"] = torch.tensor(king_stage, dtype=torch.long, device=device)
            batch["alsfrs_speech"] = torch.tensor(alsfrs_speech, dtype=torch.long, device=device)
            batch["cantagallo"] = torch.tensor(cantagallo, dtype=torch.long, device=device)
            batch["phonemes"] = torch.tensor(phonemes, dtype=torch.long, device=device)
            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
            batch["group"] = torch.tensor(group, dtype=torch.long, device=device)
            
        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        "If last frame is not whole, it will be discarded as per our convolution configuration"
        if self.dataset_name == "iemocap":
            for i in range(batch["phonemes"].shape[0]):
                if (batch["phonemes"][i] != -100).sum() - batch["sub_attention_mask"][i].sum() == 1:
                    batch['phonemes'][i, batch["sub_attention_mask"][i].sum()] = -100
                    batch['overlap_mask'][i, batch["sub_attention_mask"][i].sum()] = -100
            assert (batch["phonemes"] != -100).sum() - batch["sub_attention_mask"].sum() == 0, "The number of phonemes does not match the number of frames in the input sequence. Please check your data preprocessing."
        #Sub-attention mask denotes the length of each segment in the batch - Useful in the case that the 
        #inputs are not padded
        features_shape = (batch_size, mask_indices_seq_length) #features in embedding space - z

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


@dataclass
class DataCollatorForDecVAELatentPostAnalysis:
    """
    Data collator that will dynamically pad the inputs received, extract features, align them with labels and variables, 
    and prepare masked indices for the post-training evaluation of a Variational Decomposition Autoencoder. Supports slightly 
    different model and data training arguments through ModelArgumentsPost and DataTrainingArgumentsPost.

    Args:
        model (:class:`~DecVAEForPreTraining`):
            The DecVAE model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        dataset_name (:obj:`str`): The name of the dataset being used.
        model_args (:class:`~args_configs.model_args.ModelArgumentsPost`): The model arguments dictionary with necessary model-related parameters.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArgumentsPost`): The data training arguments dictionary
            with necessary data-related parameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object.
        model_name (:obj:`str`, `optional`, defaults to :obj:`"DecVAE"`): The name of the model being used.
        input_type (:obj:`str`, `optional`, defaults to :obj:`"waveform"`): The type of input features - 
            waveform, mel filterbank features, or fft features.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: DecVAEForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    dataset_name: str
    model_args: ModelArgumentsPost
    data_training_args: DataTrainingArgumentsPost
    config: DecVAEConfig
    model_name: str = "DecVAE"
    input_type: str = "waveform"
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        
        if self.dataset_name in ["timit", "iemocap"]:
            if self.dataset_name == "timit":
                phonemes39 = [feature.pop("phonemes39") for feature in features]
                phonemes48 = [feature.pop("phonemes48") for feature in features]
            elif self.dataset_name == "iemocap":
                phonemes = [feature.pop("phonemes") for feature in features]
                emotions = [feature.pop("emotion_labels") for feature in features]
            start_phonemes = [feature.pop("start_phonemes") for feature in features]
            stop_phonemes = [feature.pop("stop_phonemes") for feature in features]
            overlap_mask = [feature.pop("overlap_mask") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            if self.dataset_name != "iemocap":
                [feature.pop("words") for feature in features];
        elif "VOC_ALS" in self.dataset_name:
            alsfrs_total = [feature.pop("alsfrs_total_enc") for feature in features]
            disease_duration = [feature.pop("disease_duration_enc") for feature in features]
            king_stage = [feature.pop("king_stage_enc") for feature in features]
            alsfrs_speech = [feature.pop("alsfrs_speech_enc") for feature in features]
            cantagallo = [feature.pop("cantagallo_enc") for feature in features]
            phonemes = [feature.pop("phonemes") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            group = [feature.pop("group") for feature in features]

        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        if batch.get("input_seq_values") is not None:
            "Pad input_seq_values"
            batch['input_seq_values'] = pad_sequence([input_seq_values.clone().detach().to(device) for input_seq_values in batch['input_seq_values']],batch_first=True, padding_value=0.0)

        "If input will be mel filterbank features, split sequence in frames"
        if self.input_type == 'mel':
            frame_len = batch["input_values"].shape[-1]
            if batch.get("input_seq_values") is not None:
                frames = batch["input_seq_values"].shape[-1]/frame_len
                new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames),frame_len),device = batch["input_seq_values"].device)
            
                # Split sequence into frames 
                for o in range(batch["input_values"].shape[1]):
                    sequence = batch["input_seq_values"][:,o,:].clone()
                    for f in range(int(frames)):
                        framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                        new_input_seq_values[:,o,f,:] = framed_sequence.clone()
                batch["input_seq_values"] = new_input_seq_values.clone()
        
            if len(self.model_args.conv_kernel) == 7:
                assert self.data_training_args.mel_hops == 4
            elif len(self.model_args.conv_kernel) == 5:
                assert self.data_training_args.mel_hops == 3
            for o in range(batch["input_values"].shape[1]):
                batch["input_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)
                if batch.get("input_seq_values") is not None:
                    batch["input_seq_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)

            if batch.get("input_seq_values") is not None:
                "Flatten sequence - Reverse framing"
                batch["input_seq_values"] = batch["input_seq_values"].reshape(batch["input_seq_values"].shape[0],batch["input_seq_values"].shape[1],-1)


        mask_indices_seq_length = batch["input_values"].shape[-2]
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length) #length of z in embedding space after feature encoder


        if self.dataset_name in ["timit", "iemocap"]:
            "First pad to match length of largest phoneme vector"
            "Then concatenate to match the length of the input"
            if self.dataset_name == "timit":
                batch["phonemes39"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes39],batch_first=True, padding_value=-100) 
                batch["phonemes39"] = torch.cat(((batch["phonemes39"],-100*torch.ones((batch["phonemes39"].shape[0],mask_indices_seq_length-batch["phonemes39"].shape[1])))),dim=1)

                batch["phonemes48"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes48],batch_first=True, padding_value=-100) 
                batch["phonemes48"] = torch.cat(((batch["phonemes48"],-100*torch.ones((batch["phonemes48"].shape[0],mask_indices_seq_length-batch["phonemes48"].shape[1])))),dim=1)
            elif self.dataset_name == "iemocap":
                batch["phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes],batch_first=True, padding_value=-100) 
                batch["phonemes"] = torch.cat(((batch["phonemes"],-100*torch.ones((batch["phonemes"].shape[0],mask_indices_seq_length-batch["phonemes"].shape[1])))),dim=1)

                batch["emotion"] = torch.tensor(emotions, dtype=torch.long, device=device)

            batch["start_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in start_phonemes],batch_first=True, padding_value=-100) 
            batch["start_phonemes"] = torch.cat(((batch["start_phonemes"],-100*torch.ones((batch["start_phonemes"].shape[0],max(mask_indices_seq_length-batch["start_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["stop_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in stop_phonemes],batch_first=True, padding_value=-100) 
            batch["stop_phonemes"] = torch.cat(((batch["stop_phonemes"],-100*torch.ones((batch["stop_phonemes"].shape[0],max(mask_indices_seq_length-batch["stop_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["overlap_mask"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in overlap_mask],batch_first=True, padding_value=-100) 
            batch["overlap_mask"] = torch.cat(((batch["overlap_mask"],-100*torch.ones((batch["overlap_mask"].shape[0],mask_indices_seq_length-batch["overlap_mask"].shape[1])))),dim=1)

            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
        
        elif "VOC_ALS" in self.dataset_name:
            batch["alsfrs_total"] = torch.tensor(alsfrs_total, dtype=torch.long, device=device)
            batch["disease_duration"] = torch.tensor(disease_duration, dtype=torch.long, device=device)
            batch["king_stage"] = torch.tensor(king_stage, dtype=torch.long, device=device)
            batch["alsfrs_speech"] = torch.tensor(alsfrs_speech, dtype=torch.long, device=device)
            batch["cantagallo"] = torch.tensor(cantagallo, dtype=torch.long, device=device)
            batch["phonemes"] = torch.tensor(phonemes, dtype=torch.long, device=device)
            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
            batch["group"] = torch.tensor(group, dtype=torch.long, device=device)

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        "If last frame is not whole, it will be discarded as per our convolution configuration"
        if self.dataset_name == "iemocap":
            for i in range(batch["phonemes"].shape[0]):
                if (batch["phonemes"][i] != -100).sum() - batch["sub_attention_mask"][i].sum() == 1:
                    batch['phonemes'][i, batch["sub_attention_mask"][i].sum()] = -100
                    batch['overlap_mask'][i, batch["sub_attention_mask"][i].sum()] = -100
            assert (batch["phonemes"] != -100).sum() - batch["sub_attention_mask"].sum() == 0, "The number of phonemes does not match the number of frames in the input sequence. Please check your data preprocessing."

        #Sub-attention mask denotes the length of each segment in the batch - Useful in the case that the 
        #inputs are not padded
        features_shape = (batch_size, mask_indices_seq_length) #features in embedding space - z

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


@dataclass
class DataCollatorForDecVAELatentDisentanglement:
    """
    Data collator that will dynamically pad the inputs received, extract features, align them with labels and variables, 
    and prepare masked indices for prompting a Variational Decomposition Autoencoder for latent disentanglement visualization. Supports slightly 
    different model and data training arguments through ModelArgumentsPost and DataTrainingArgumentsPost.

    Args:
        model (:class:`~DecVAEForPreTraining`):
            The DecVAE model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        dataset_name (:obj:`str`): The name of the dataset being used.
        model_args (:class:`~args_configs.model_args.ModelArgumentsPost`): The model arguments dictionary with necessary model-related parameters.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArgumentsPost`): The data training arguments dictionary 
            with necessary data-related parameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object.
        model_name (:obj:`str`, `optional`, defaults to :obj:`"DecVAE"`): The name of the model being used.
        input_type (:obj:`str`, `optional`, defaults to :obj:`"waveform"`): The type of input features - 
            waveform, mel filterbank features, or fft features.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: DecVAEForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    dataset_name: str
    model_args: ModelArgumentsPost
    data_training_args: DataTrainingArgumentsPost
    config: DecVAEConfig
    model_name: str = "DecVAE"
    input_type: str = "waveform"
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        
        if self.dataset_name in ["timit", "iemocap"]:
            if self.dataset_name == "timit":
                phonemes39 = [feature.pop("phonemes39") for feature in features]
                phonemes48 = [feature.pop("phonemes48") for feature in features]
            elif self.dataset_name == "iemocap":
                phonemes = [feature.pop("phonemes") for feature in features]
                emotions = [feature.pop("emotion_labels") for feature in features]
            start_phonemes = [feature.pop("start_phonemes") for feature in features]
            stop_phonemes = [feature.pop("stop_phonemes") for feature in features]
            overlap_mask = [feature.pop("overlap_mask") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            if self.dataset_name != "iemocap":
                [feature.pop("words") for feature in features];
        elif "VOC_ALS" in self.dataset_name:
            alsfrs_total = [feature.pop("alsfrs_total_enc") for feature in features]
            disease_duration = [feature.pop("disease_duration_enc") for feature in features]
            king_stage = [feature.pop("king_stage_enc") for feature in features]
            alsfrs_speech = [feature.pop("alsfrs_speech_enc") for feature in features]
            cantagallo = [feature.pop("cantagallo_enc") for feature in features]
            phonemes = [feature.pop("phonemes") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            group = [feature.pop("group") for feature in features]


        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        if batch.get("input_seq_values") is not None:
            "Pad input_seq_values"
            batch['input_seq_values'] = pad_sequence([input_seq_values.clone().detach().to(device) for input_seq_values in batch['input_seq_values']],batch_first=True, padding_value=0.0)

        "If input will be mel filterbank features, split sequence in frames"
        if self.input_type == 'mel':
            frame_len = batch["input_values"].shape[-1]
            if batch.get("input_seq_values") is not None:
                frames = batch["input_seq_values"].shape[-1]/frame_len
                new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames),frame_len),device = batch["input_seq_values"].device)
            
                # Split sequence into frames 
                for o in range(batch["input_values"].shape[1]):
                    sequence = batch["input_seq_values"][:,o,:].clone()
                    for f in range(int(frames)):
                        framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                        new_input_seq_values[:,o,f,:] = framed_sequence.clone()
                batch["input_seq_values"] = new_input_seq_values.clone()
        
            if len(self.model_args.conv_kernel) == 7:
                assert self.data_training_args.mel_hops == 4
            elif len(self.model_args.conv_kernel) == 5:
                assert self.data_training_args.mel_hops == 3
            for o in range(batch["input_values"].shape[1]):
                batch["input_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)
                if batch.get("input_seq_values") is not None:
                    batch["input_seq_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)

            if batch.get("input_seq_values") is not None:
                "Flatten sequence - Reverse framing"
                batch["input_seq_values"] = batch["input_seq_values"].reshape(batch["input_seq_values"].shape[0],batch["input_seq_values"].shape[1],-1)

        mask_indices_seq_length = batch["input_values"].shape[-2]
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length) #length of z in embedding space after feature encoder


        if self.dataset_name in ["timit", "iemocap"]:
            "First pad to match length of largest phoneme vector"
            "Then concatenate to match the length of the input"
            if self.dataset_name == "timit":
                batch["phonemes39"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes39],batch_first=True, padding_value=-100) 
                batch["phonemes39"] = torch.cat(((batch["phonemes39"],-100*torch.ones((batch["phonemes39"].shape[0],mask_indices_seq_length-batch["phonemes39"].shape[1])))),dim=1)

                batch["phonemes48"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes48],batch_first=True, padding_value=-100) 
                batch["phonemes48"] = torch.cat(((batch["phonemes48"],-100*torch.ones((batch["phonemes48"].shape[0],mask_indices_seq_length-batch["phonemes48"].shape[1])))),dim=1)
            elif self.dataset_name == "iemocap":
                batch["phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes],batch_first=True, padding_value=-100) 
                batch["phonemes"] = torch.cat(((batch["phonemes"],-100*torch.ones((batch["phonemes"].shape[0],mask_indices_seq_length-batch["phonemes"].shape[1])))),dim=1)

                batch["emotion"] = torch.tensor(emotions, dtype=torch.long, device=device)

            batch["start_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in start_phonemes],batch_first=True, padding_value=-100) 
            batch["start_phonemes"] = torch.cat(((batch["start_phonemes"],-100*torch.ones((batch["start_phonemes"].shape[0],max(mask_indices_seq_length-batch["start_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["stop_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in stop_phonemes],batch_first=True, padding_value=-100) 
            batch["stop_phonemes"] = torch.cat(((batch["stop_phonemes"],-100*torch.ones((batch["stop_phonemes"].shape[0],max(mask_indices_seq_length-batch["stop_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["overlap_mask"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in overlap_mask],batch_first=True, padding_value=-100) 
            batch["overlap_mask"] = torch.cat(((batch["overlap_mask"],-100*torch.ones((batch["overlap_mask"].shape[0],mask_indices_seq_length-batch["overlap_mask"].shape[1])))),dim=1)

            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
        
        elif "VOC_ALS" in self.dataset_name:
            batch["alsfrs_total"] = torch.tensor(alsfrs_total, dtype=torch.long, device=device)
            batch["disease_duration"] = torch.tensor(disease_duration, dtype=torch.long, device=device)
            batch["king_stage"] = torch.tensor(king_stage, dtype=torch.long, device=device)
            batch["alsfrs_speech"] = torch.tensor(alsfrs_speech, dtype=torch.long, device=device)
            batch["cantagallo"] = torch.tensor(cantagallo, dtype=torch.long, device=device)
            batch["phonemes"] = torch.tensor(phonemes, dtype=torch.long, device=device)
            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
            batch["group"] = torch.tensor(group, dtype=torch.long, device=device)


        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        "If last frame is not whole, it will be discarded as per our convolution configuration"
        if self.dataset_name == "iemocap":
            for i in range(batch["phonemes"].shape[0]):
                if (batch["phonemes"][i] != -100).sum() - batch["sub_attention_mask"][i].sum() == 1:
                    batch['phonemes'][i, batch["sub_attention_mask"][i].sum()] = -100
                    batch['overlap_mask'][i, batch["sub_attention_mask"][i].sum()] = -100
            assert (batch["phonemes"] != -100).sum() - batch["sub_attention_mask"].sum() == 0, "The number of phonemes does not match the number of frames in the input sequence. Please check your data preprocessing."

        #Sub-attention mask denotes the length of each segment in the batch - Useful in the case that the 
        #inputs are not padded
        features_shape = (batch_size, mask_indices_seq_length) #features in embedding space - z

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


@dataclass
class DataCollatorForDecVAELatentTraversals:
    """
    Data collator that will dynamically pad the inputs received, extract features, align them with labels and variables, 
    and prepare masked indices for prompting a Variational Decomposition Autoencoder for latent disentanglement traversals extraction. Supports slightly 
    different model and data training arguments through ModelArgumentsPost and DataTrainingArgumentsPost.

    Args:
        model (:class:`~DecVAEForPreTraining`):
            The DecVAE model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        dataset_name (:obj:`str`): The name of the dataset being used.
        model_args (:class:`~args_configs.model_args.ModelArgumentsPost`): The model arguments dictionary with necessary model-related parameters.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArgumentsPost`): The data training arguments dictionary 
            with necessary data-related parameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object.
        model_name (:obj:`str`, `optional`, defaults to :obj:`"DecVAE"`): The name of the model being used.
        input_type (:obj:`str`, `optional`, defaults to :obj:`"waveform"`): The type of input features - 
            waveform, mel filterbank features, or fft features.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: DecVAEForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    dataset_name: str 
    model_args: ModelArgumentsPost
    data_training_args: DataTrainingArgumentsPost
    config: DecVAEConfig
    model_name: str = "DecVAE"
    input_type: str = "waveform"
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format

        if self.dataset_name in ["timit", "iemocap"]:
            if self.dataset_name == "timit":
                phonemes39 = [feature.pop("phonemes39") for feature in features]
                phonemes48 = [feature.pop("phonemes48") for feature in features]
            elif self.dataset_name == "iemocap":
                phonemes = [feature.pop("phonemes") for feature in features]
                emotions = [feature.pop("emotion_labels") for feature in features]
            start_phonemes = [feature.pop("start_phonemes") for feature in features]
            stop_phonemes = [feature.pop("stop_phonemes") for feature in features]
            overlap_mask = [feature.pop("overlap_mask") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            if self.dataset_name != "iemocap":
                [feature.pop("words") for feature in features];
        elif "VOC_ALS" in self.dataset_name:
            alsfrs_total = [feature.pop("alsfrs_total_enc") for feature in features]
            disease_duration = [feature.pop("disease_duration_enc") for feature in features]
            king_stage = [feature.pop("king_stage_enc") for feature in features]
            alsfrs_speech = [feature.pop("alsfrs_speech_enc") for feature in features]
            cantagallo = [feature.pop("cantagallo_enc") for feature in features]
            phonemes = [feature.pop("phonemes") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            group = [feature.pop("group") for feature in features]
        elif "vowels" in self.dataset_name:
            [feature.pop("overlap_mask") for feature in features]
            #[feature.pop("reconstruction_NRMSEs") for feature in features]
            #[feature.pop("reconstruction_NRMSE_seq") for feature in features]
            #[feature.pop("correlograms") for feature in features]
            #[feature.pop("correlogram_seq") for feature in features]
            for feature in features:
                feature["vowel_labels"] = [feature["vowel_labels"]]
                feature["speaker_vt_factor"] = [feature["speaker_vt_factor"]]

        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        if batch.get("input_seq_values") is not None:
            "Pad input_seq_values"
            batch['input_seq_values'] = pad_sequence([input_seq_values.clone().detach().to(device) for input_seq_values in batch['input_seq_values']],batch_first=True, padding_value=0.0)

        "If input will be mel filterbank features, split sequence in frames"
        if self.input_type == 'mel':
            frame_len = batch["input_values"].shape[-1]
            if batch.get("input_seq_values") is not None:
                frames = batch["input_seq_values"].shape[-1]/frame_len
                new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames),frame_len),device = batch["input_seq_values"].device)
            
                # Split sequence into frames 
                for o in range(batch["input_values"].shape[1]):
                    sequence = batch["input_seq_values"][:,o,:].clone()
                    for f in range(int(frames)):
                        framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                        new_input_seq_values[:,o,f,:] = framed_sequence.clone()
                batch["input_seq_values"] = new_input_seq_values.clone()
        
            if len(self.model_args.conv_kernel) == 7:
                assert self.data_training_args.mel_hops == 4
            elif len(self.model_args.conv_kernel) == 5:
                assert self.data_training_args.mel_hops == 3
            for o in range(batch["input_values"].shape[1]):
                if "vowels" in self.dataset_name:
                    mels, _ = extract_mel_spectrogram(batch["input_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)
                    batch["input_values"][:,o,...] = mels.reshape(mels.shape[0],-1).unsqueeze(1)
                else:
                    batch["input_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)
                if batch.get("input_seq_values") is not None:
                    batch["input_seq_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)

            if batch.get("input_seq_values") is not None:
                "Flatten sequence - Reverse framing"
                batch["input_seq_values"] = batch["input_seq_values"].reshape(batch["input_seq_values"].shape[0],batch["input_seq_values"].shape[1],-1)

        elif self.input_type == 'fft':                               
            # Apply fft using welch's power spectral density estimation
            batch["input_values"], batch["input_seq_values"] = extract_fft_psd(
                batch, 
                normalize=True, #self.data_training_args.mel_norm,
                device=batch["input_values"].device
            )
                                

        mask_indices_seq_length = batch["input_values"].shape[-2]
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length) #length of z in embedding space after feature encoder

        if self.dataset_name in ["timit", "iemocap"]:
            "First pad to match length of largest phoneme vector"
            "Then concatenate to match the length of the input"
            if self.dataset_name == "timit":
                batch["phonemes39"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes39],batch_first=True, padding_value=-100) 
                batch["phonemes39"] = torch.cat(((batch["phonemes39"],-100*torch.ones((batch["phonemes39"].shape[0],mask_indices_seq_length-batch["phonemes39"].shape[1]), device=device))),dim=1)

                batch["phonemes48"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes48],batch_first=True, padding_value=-100) 
                batch["phonemes48"] = torch.cat(((batch["phonemes48"],-100*torch.ones((batch["phonemes48"].shape[0],mask_indices_seq_length-batch["phonemes48"].shape[1]), device=device))),dim=1)
            elif self.dataset_name == "iemocap":
                batch["phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes],batch_first=True, padding_value=-100) 
                batch["phonemes"] = torch.cat(((batch["phonemes"],-100*torch.ones((batch["phonemes"].shape[0],mask_indices_seq_length-batch["phonemes"].shape[1]), device=device))),dim=1)

                batch["emotion"] = torch.tensor(emotions, dtype=torch.long, device=device)
            
            batch["start_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in start_phonemes],batch_first=True, padding_value=-100) 
            batch["start_phonemes"] = torch.cat(((batch["start_phonemes"],-100*torch.ones((batch["start_phonemes"].shape[0],max(mask_indices_seq_length-batch["start_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["stop_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in stop_phonemes],batch_first=True, padding_value=-100) 
            batch["stop_phonemes"] = torch.cat(((batch["stop_phonemes"],-100*torch.ones((batch["stop_phonemes"].shape[0],max(mask_indices_seq_length-batch["stop_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["overlap_mask"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in overlap_mask],batch_first=True, padding_value=-100) 
            batch["overlap_mask"] = torch.cat(((batch["overlap_mask"],-100*torch.ones((batch["overlap_mask"].shape[0],mask_indices_seq_length-batch["overlap_mask"].shape[1]), device=device))),dim=1)

            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
        
        elif "VOC_ALS" in self.dataset_name:
            batch["alsfrs_total"] = torch.tensor(alsfrs_total, dtype=torch.long, device=device)
            batch["disease_duration"] = torch.tensor(disease_duration, dtype=torch.long, device=device)
            batch["king_stage"] = torch.tensor(king_stage, dtype=torch.long, device=device)
            batch["alsfrs_speech"] = torch.tensor(alsfrs_speech, dtype=torch.long, device=device)
            batch["cantagallo"] = torch.tensor(cantagallo, dtype=torch.long, device=device)
            batch["phonemes"] = torch.tensor(phonemes, dtype=torch.long, device=device)
            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
            batch["group"] = torch.tensor(group, dtype=torch.long, device=device)
            

        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        "If last frame is not whole, it will be discarded as per our convolution configuration"
        if self.dataset_name == "iemocap":
            for i in range(batch["phonemes"].shape[0]):
                if (batch["phonemes"][i] != -100).sum() != batch["sub_attention_mask"][i].sum():
                    batch['phonemes'][i, batch["sub_attention_mask"][i].sum():] = -100
                    batch['overlap_mask'][i, batch["sub_attention_mask"][i].sum():] = -100
            assert (batch["phonemes"] != -100).sum() == batch["sub_attention_mask"].sum(), "The number of phonemes does not match the number of frames in the input sequence. Please check your data preprocessing."
        
        #Sub-attention mask denotes the length of each segment in the batch - Useful in the case that the 
        #inputs are not padded
        features_shape = (batch_size, mask_indices_seq_length) #features in embedding space - z

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch


@dataclass
class DataCollatorForDecVAELatentVisualization:
    """
    Data collator that will dynamically pad the inputs received, extract features, align them with labels and variables, 
    and prepare masked indices for prompting a Variational Decomposition Autoencoder for latent representations visualization. Supports slightly 
    different model and data training arguments through ModelArgumentsPost and DataTrainingArgumentsPost.

    Args:
        model (:class:`~DecVAEForPreTraining`):
            The DecVAE model used for pretraining. The data collator needs to have access
            to config and ``_get_feat_extract_output_lengths`` function for correct padding.
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`):
            The processor used for proccessing the data - used to pad the data.
        dataset_name (:obj:`str`): The name of the dataset being used.
        model_args (:class:`~args_configs.model_args.ModelArgumentsPost`): The model arguments dictionary with necessary model-related parameters.
        data_training_args (:class:`~args_configs.data_training_args.DataTrainingArgumentsPost`): The data training arguments dictionary 
            with necessary data-related parameters.
        config (:class:`~config_files.configuration_decVAE.DecVAEConfig`): The DecVAE configuration object.
        model_name (:obj:`str`, `optional`, defaults to :obj:`"DecVAE"`): The name of the model being used.
        input_type (:obj:`str`, `optional`, defaults to :obj:`"waveform"`): The type of input features - 
            waveform, mel filterbank features, or fft features.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        mask_time_prob (:obj:`float`, `optional`, defaults to :obj:`0.65`):
            Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked for the contrastive task.
            Note that overlap between masked sequences may decrease the actual percentage of masked vectors.
            The default value is taken from the original wav2vec 2.0 article (https://arxiv.org/abs/2006.11477),
            and results in about 49 percent of each sequence being masked on average.
        mask_time_length (:obj:`int`, `optional`, defaults to :obj:`10`):
            Length of each vector mask span to mask along the time axis in the contrastive task. The default value
            originates from the original wav2vec 2.0 article and corresponds to the ``M`` variable mentioned there.
    """

    model: DecVAEForPreTraining
    feature_extractor: Wav2Vec2FeatureExtractor
    dataset_name: str
    model_args: ModelArgumentsPost
    data_training_args: DataTrainingArgumentsPost
    config: DecVAEConfig
    model_name: str = "DecVAE"
    input_type: str = "waveform"
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    mask_time_prob: Optional[float] = 0.65
    mask_time_length: Optional[int] = 10

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # reformat list to dict and set to pytorch format
        
        if self.dataset_name in ["timit", "librispeech_asr", "iemocap"]:
            if self.dataset_name == "timit":
                phonemes39 = [feature.pop("phonemes39") for feature in features]
                phonemes48 = [feature.pop("phonemes48") for feature in features]
            elif self.dataset_name == "librispeech_asr":
                phonemes41 = [feature.pop("phonemes41") for feature in features]
            elif self.dataset_name == "iemocap":
                phonemes = [feature.pop("phonemes") for feature in features]
                emotions = [feature.pop("emotion_labels") for feature in features]
            start_phonemes = [feature.pop("start_phonemes") for feature in features]
            stop_phonemes = [feature.pop("stop_phonemes") for feature in features]
            overlap_mask = [feature.pop("overlap_mask") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            if self.dataset_name != "iemocap":
                [feature.pop("words") for feature in features];

        elif "VOC_ALS" in self.dataset_name:
            alsfrs_total = [feature.pop("alsfrs_total_enc") for feature in features]
            disease_duration = [feature.pop("disease_duration_enc") for feature in features]
            king_stage = [feature.pop("king_stage_enc") for feature in features]
            alsfrs_speech = [feature.pop("alsfrs_speech_enc") for feature in features]
            cantagallo = [feature.pop("cantagallo_enc") for feature in features]
            phonemes = [feature.pop("phonemes") for feature in features]
            speaker_id = [feature.pop("speaker") for feature in features]
            group = [feature.pop("group") for feature in features]
        elif "scRNA_seq" in self.dataset_name:
            celltype = [feature.pop("cell_type") for feature in features]
            cell_condition = [feature.pop("cell_condition") for feature in features]
            batch_information = [feature.pop("batch_information") for feature in features]
            [feature.pop("genes_count") for feature in features];
            [feature.pop("counts") for feature in features];
            [feature.pop("sz_factor") for feature in features];
            [feature.pop("id") for feature in features];



        batch = self.feature_extractor.pad(
            features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        device = batch["input_values"].device
        batch_size = batch["input_values"].shape[0]

        if batch.get("input_seq_values") is not None:
            "Pad input_seq_values"
            batch['input_seq_values'] = pad_sequence([input_seq_values.clone().detach().to(device) for input_seq_values in batch['input_seq_values']],batch_first=True, padding_value=0.0)

        "If input will be mel filterbank features, split sequence in frames"
        if self.input_type == 'mel':
            frame_len = batch["input_values"].shape[-1]
            if batch.get("input_seq_values") is not None:
                frames = batch["input_seq_values"].shape[-1]/frame_len
                new_input_seq_values = torch.zeros((batch_size,batch["input_values"].shape[1],int(frames),frame_len),device = batch["input_seq_values"].device)
            
                # Split sequence into frames 
                for o in range(batch["input_values"].shape[1]):
                    sequence = batch["input_seq_values"][:,o,:].clone()
                    for f in range(int(frames)):
                        framed_sequence = sequence[:,f*frame_len:(f+1)*frame_len]
                        new_input_seq_values[:,o,f,:] = framed_sequence.clone()
                batch["input_seq_values"] = new_input_seq_values.clone()
        
            if len(self.model_args.conv_kernel) == 7:
                assert self.data_training_args.mel_hops == 4
            elif len(self.model_args.conv_kernel) == 5:
                assert self.data_training_args.mel_hops == 3
            for o in range(batch["input_values"].shape[1]):
                batch["input_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)
                if batch.get("input_seq_values") is not None:
                    batch["input_seq_values"][:,o,...], _ = extract_mel_spectrogram(batch["input_seq_values"][:,o,...],self.config.fs,n_mels=self.data_training_args.n_mels, n_fft=int(self.data_training_args.mel_hops*self.config.receptive_field*self.config.fs), hop_length=int(((self.config.receptive_field*self.config.fs) + 1)/self.data_training_args.mel_hops), normalize=self.data_training_args.mel_norm, feature_length=frame_len)

            if batch.get("input_seq_values") is not None:
                "Flatten sequence - Reverse framing"
                batch["input_seq_values"] = batch["input_seq_values"].reshape(batch["input_seq_values"].shape[0],batch["input_seq_values"].shape[1],-1)

        elif self.input_type == 'fft':                               
            # Apply fft using welch's power spectral density estimation
            batch["input_values"], batch["input_seq_values"] = extract_fft_psd(
                batch, 
                normalize=True, #self.data_training_args.mel_norm,
                device=batch["input_values"].device
            )


        mask_indices_seq_length = batch["input_values"].shape[-2]
        batch["attention_mask"] = batch["attention_mask"].squeeze(1)
        # make sure masked sequence length is a Python scalar
        mask_indices_seq_length = int(mask_indices_seq_length) #length of z in embedding space after feature encoder


        if self.dataset_name in ["timit", "librispeech_asr", "iemocap"]:
            "First pad to match length of largest phoneme vector"
            "Then concatenate to match the length of the input"
            if self.dataset_name == "timit":
                batch["phonemes39"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes39],batch_first=True, padding_value=-100) 
                batch["phonemes39"] = torch.cat(((batch["phonemes39"],-100*torch.ones((batch["phonemes39"].shape[0],mask_indices_seq_length-batch["phonemes39"].shape[1])))),dim=1)

                batch["phonemes48"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes48],batch_first=True, padding_value=-100) 
                batch["phonemes48"] = torch.cat(((batch["phonemes48"],-100*torch.ones((batch["phonemes48"].shape[0],mask_indices_seq_length-batch["phonemes48"].shape[1])))),dim=1)
            elif self.dataset_name == "librispeech_asr":
                batch["phonemes41"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes41],batch_first=True, padding_value=-100) 
                batch["phonemes41"] = torch.cat(((batch["phonemes41"],-100*torch.ones((batch["phonemes41"].shape[0],mask_indices_seq_length-batch["phonemes41"].shape[1])))),dim=1)
            elif self.dataset_name == "iemocap":
                batch["phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in phonemes],batch_first=True, padding_value=-100) 
                batch["phonemes"] = torch.cat(((batch["phonemes"],-100*torch.ones((batch["phonemes"].shape[0],mask_indices_seq_length-batch["phonemes"].shape[1])))),dim=1)

                batch["emotion"] = torch.tensor(emotions, dtype=torch.long, device=device)
            
            batch["start_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in start_phonemes],batch_first=True, padding_value=-100) 
            batch["start_phonemes"] = torch.cat(((batch["start_phonemes"],-100*torch.ones((batch["start_phonemes"].shape[0],max(mask_indices_seq_length-batch["start_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["stop_phonemes"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in stop_phonemes],batch_first=True, padding_value=-100) 
            batch["stop_phonemes"] = torch.cat(((batch["stop_phonemes"],-100*torch.ones((batch["stop_phonemes"].shape[0],max(mask_indices_seq_length-batch["stop_phonemes"].shape[1],0)), device=device))),dim=1)

            batch["overlap_mask"] = pad_sequence([torch.tensor(phone, dtype=torch.long, device=device) for phone in overlap_mask],batch_first=True, padding_value=-100) 
            batch["overlap_mask"] = torch.cat(((batch["overlap_mask"],-100*torch.ones((batch["overlap_mask"].shape[0],mask_indices_seq_length-batch["overlap_mask"].shape[1])))),dim=1)

            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)

        elif "VOC_ALS" in self.dataset_name:
            batch["alsfrs_total"] = torch.tensor(alsfrs_total, dtype=torch.long, device=device)
            batch["disease_duration"] = torch.tensor(disease_duration, dtype=torch.long, device=device)
            batch["king_stage"] = torch.tensor(king_stage, dtype=torch.long, device=device)
            batch["alsfrs_speech"] = torch.tensor(alsfrs_speech, dtype=torch.long, device=device)
            batch["cantagallo"] = torch.tensor(cantagallo, dtype=torch.long, device=device)
            batch["phonemes"] = torch.tensor(phonemes, dtype=torch.long, device=device)
            batch["speaker_id"] = torch.tensor(speaker_id, dtype=torch.long, device=device)
            batch["group"] = torch.tensor(group, dtype=torch.long, device=device)

        elif "scRNA_seq" in self.dataset_name:
            batch["celltype"] = torch.tensor(celltype, dtype=torch.long, device=device)
            batch["cell_condition"] = torch.tensor(cell_condition, dtype=torch.long, device=device)
            batch["batch_information"] = torch.tensor(batch_information, dtype=torch.long, device=device)


        # make sure that no loss is computed on padded inputs
        if batch.get("attention_mask") is not None:
            # compute real output lengths according to convolution formula
            batch["sub_attention_mask"] = self.model._get_feature_vector_attention_mask(
                mask_indices_seq_length, batch["attention_mask"]
            )

        "If last frame is not whole, it will be discarded as per our convolution configuration"
        if self.dataset_name == "iemocap":
            for i in range(batch["phonemes"].shape[0]):
                if (batch["phonemes"][i] != -100).sum() - batch["sub_attention_mask"][i].sum() == 1:
                    batch['phonemes'][i, batch["sub_attention_mask"][i].sum()] = -100
                    batch['overlap_mask'][i, batch["sub_attention_mask"][i].sum()] = -100
            assert (batch["phonemes"] != -100).sum() - batch["sub_attention_mask"].sum() == 0, "The number of phonemes does not match the number of frames in the input sequence. Please check your data preprocessing."

        #Sub-attention mask denotes the length of each segment in the batch - Useful in the case that the 
        #inputs are not padded
        features_shape = (batch_size, mask_indices_seq_length) #features in embedding space - z

        # sample randomly masked indices
        mask_time_indices = _compute_mask_indices(
            features_shape,
            self.mask_time_prob,
            self.mask_time_length,
            attention_mask=batch.get("sub_attention_mask"),
        )

        # sample negative indices
        sampled_negative_indices = _sample_negative_indices(
            features_shape,
            self.model.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )
        batch["mask_time_indices"] = torch.tensor(mask_time_indices, dtype=torch.long, device=device)
        batch["sampled_negative_indices"] = torch.tensor(sampled_negative_indices, dtype=torch.long, device=device)

        return batch

