import logging
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import torch
import transformers
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from data import datasets, registry, types

def prepare_dataset(
    data_opts: List[types.DatasetConfig],
    data_args: types.VoiceDatasetArgs,
    max_samples: Optional[int] = None,
) -> datasets.SizedIterableDataset:
    data_sets = []
    for ds_config in data_opts:
        # Register if not already registered (or just ensure it's in the map)
        if ds_config.name not in registry.DATASET_MAP:
             registry.register_datasets([ds_config])
        
        ds = registry.create_dataset(ds_config.name, data_args, verbose=True)
        data_sets.append(ds)

    # Interleave if multiple datasets
    if len(data_sets) > 1:
        dataset = datasets.InterleaveDataset(data_sets)
    else:
        dataset = data_sets[0]

    # Apply range limit if specified
    if max_samples is not None:
        dataset = datasets.Range(dataset, max_samples)
    elif data_args.max_samples != -1:
        dataset = datasets.Range(dataset, data_args.max_samples)
        
    return dataset


class WhisperDataproc(datasets.Dataproc):
    def __init__(self, dataset, processor, max_label_length=448):
        super().__init__(dataset)
        self.processor = processor
        self.max_label_length = max_label_length
        self.normalizer = EnglishTextNormalizer({})

    def _process(self, sample):
        # Process audio
        audio = sample.audio
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features[0]
        
        # Process text
        # Filter empty normalized text
        if len(self.normalizer(sample.text)) == 0:
             return None

        labels = self.processor(text=sample.text).input_ids
        
        # Filter long transcriptions
        if len(labels) > self.max_label_length:
             logging.debug(f"Skipping sample with token length {len(labels)} > {self.max_label_length}")
             return None
        
        return {
            "audio": {"array": input_features},
            "text_input_ids": labels,
            "reference": sample.text,
        }


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor, return_references=False):
        self.processor = processor
        self.return_references = return_references

    def __call__(
        self, features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["audio"]["array"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["text_input_ids"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (
            labels[:, 0] == self.processor.tokenizer.bos_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        
        # Pass through references if present
        if self.return_references and "reference" in features[0]:
            batch["references"] = [feature["reference"] for feature in features]
            
        return batch
