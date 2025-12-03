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
    def __init__(self, dataset, processor):
        super().__init__(dataset)
        self.processor = processor

    def _process(self, sample):
        # Process audio
        audio = sample.audio
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features[0]
        
        # Process text
        labels = self.processor(text=sample.text).input_ids
        
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


def run_inference(
    model: torch.nn.Module,
    processor: WhisperProcessor,
    dataset: datasets.SizedIterableDataset,
    device: torch.device,
    batch_size: int = 1,
    language: Optional[str] = None,
    task: Optional[str] = None,
) -> Tuple[Dict[str, float], List[str], List[str]]:
    """
    Runs inference on the dataset and computes WER.
    Returns metrics, predictions, and references.
    """
    logging.info(f"Starting inference with batch_size={batch_size}...")
    
    # Force language and task if provided
    if language:
        model.generation_config.language = language
    if task:
        model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    model.eval()
    metric = evaluate.load("wer")
    normalizer = EnglishTextNormalizer({})
    
    predictions = []
    references = []
    
    # Prepare dataset and dataloader
    dataset_proc = WhisperDataproc(dataset, processor)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor, return_references=True)
    
    # Note: num_workers=0 is safer for iterable datasets sometimes, but >0 can help speed.
    # Since we are using an IterableDataset, we can't use shuffle=True (already handled)
    dataloader = torch.utils.data.DataLoader(
        dataset_proc,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0, 
    )
    
    from tqdm import tqdm
    
    for i, batch in enumerate(tqdm(dataloader, desc="Inference")):
        # Move inputs to device and cast to model's dtype
        input_features = batch["input_features"].to(device, dtype=model.dtype)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(input_features)
        
        # Decode
        transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Get references
        # If collator passed through 'references', use them.
        # Otherwise decode labels (fallback)
        if "references" in batch:
            batch_references = batch["references"]
        else:
            # Fallback: decode labels (replace -100 with pad_token_id)
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            batch_references = processor.batch_decode(labels, skip_special_tokens=True)

        # Normalization using Whisper English Normalizer
        transcriptions = [normalizer(t) for t in transcriptions]
        batch_references = [normalizer(r) for r in batch_references]
        
        predictions.extend(transcriptions)
        references.extend(batch_references)
        
        if i < 3:
            logging.info(f"Batch {i} Sample 0:")
            logging.info(f"  Ref: {batch_references[0]}")
            logging.info(f"  Pred: {transcriptions[0]}")

    wer = metric.compute(predictions=predictions, references=references)
    logging.info(f"Final Inference WER: {wer}")
    
    return {"wer": wer}, predictions, references
