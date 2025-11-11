import jiwer
import transformers
import re
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
import json
from transformers import Wav2Vec2Processor
from transformers import Trainer
import torch
import numpy as np
from transformers import TrainingArguments

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower()
    return batch

def extract_all_chars(batch):
  all_text = " ".join(batch["text"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

#parser = argparse.ArgumentParser(description="Pre-train a self-supervised decomposition learning Wav2Vec2-like model")
#args, unknown = parser.parse_known_args() 
model_name_or_path = "patrickvonplaten/wav2vec2-base-v2"
pretrained_model_dir = "/home/giannis/Documents/ft_test" #DecSSL/pretraining_small/_complete_min_val_loss_min_val_contrastive_loss_min_val_decomposition_loss/" #ft_test
output_dir = "/home/giannis/Documents/ft_test_output"
timit = load_dataset("timit_asr",data_dir="/home/giannis/Documents/TIMIT")

timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

timit = timit.map(remove_special_characters)

vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])

vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))

with open('vocab_timit.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000,padding_side = "right", padding_value=0.0, do_normalize=True, return_attention_mask=True)
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)

tokenizer = Wav2Vec2CTCTokenizer("./vocab_timit.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

wer_metric = load_metric("wer",trust_remote_code=True)
#??? per_metric = load_metric("per")

model = Wav2Vec2ForCTC.from_pretrained(
    pretrained_model_dir, 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

n_params = count_parameters(model)

#Replace orig_layer_norm with torch.nn.Identity in Feature Extractor and Feature Projector
for child in model.wav2vec2.custom_feature_extractor.children():
    for sub_child in child:
        if hasattr(sub_child,'orig_layer_norm'):
            sub_child.orig_layer_norm = torch.nn.Identity()

for n,_ in model.wav2vec2.custom_feature_projection.named_children():
        if n == 'orig_layer_norm':
            model.wav2vec2.custom_feature_projection.orig_layer_norm = torch.nn.Identity()

from torchview import draw_graph

inputs = torch.randn(1, 80000)

model_graph = draw_graph(model, input_data=inputs)

model_graph.visual_graph

import graphviz
graphviz.set_jupyter_format('png')

model.freeze_feature_encoder()
#OR: model.freeze_base_model()
assert sum(p.numel() for p in model.wav2vec2.custom_feature_extractor.parameters() if p.requires_grad) == 0

training_args = TrainingArguments(
  output_dir=output_dir,
  group_by_length=False,
  per_device_train_batch_size=16,
  per_device_eval_batch_size=8,
  evaluation_strategy="steps",
  num_train_epochs=30,
  fp16=False,
  gradient_checkpointing=False, 
  save_steps=500,
  eval_steps=1,
  logging_steps=50,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=2,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit["train"],
    eval_dataset=timit["test"],
    tokenizer=processor.tokenizer,
)

trainer.train()

#Evaluation
