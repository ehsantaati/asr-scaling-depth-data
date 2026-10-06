"""The fixed decoding path: one deterministic implementation shared by training
jobs, standalone inference, and the vanilla baseline.

It replaces three defects in ``inference.run_inference_map`` that made WERs
non-comparable across runs (that function is retained unchanged as the *legacy*
path so the old-vs-new delta can be measured — see inference.evaluate_datasets):

  1. ``model.generate(input_features)`` was called with no language/task, so
     multilingual Whisper performed per-utterance language identification.
  2. Samples with duration >= 30 s, or whose text failed ``format_asr_text``, were
     silently dropped -- 22.33% of the GigaSpeech test set in the archived runs --
     and the surviving denominator was never reported.
  3. Normalization used ``EnglishTextNormalizer({})``, i.e. an empty spelling map,
     rather than the model's own english.json.

Here: language and task are always forced, decoding is greedy by default, the
normalizer is the tokenizer's own, filtering happens only where it is configured
(``EvalDatasetArgs.max_audio_duration_secs``), and every count that goes into the
WER denominator is returned.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import torch
from tqdm.auto import tqdm

# Normalized reference strings that survive to the WER computation must be non-empty;
# jiwer raises on an empty reference.
_METRIC = None


def _wer_metric():
    global _METRIC
    if _METRIC is None:
        _METRIC = evaluate.load("wer")
    return _METRIC


@dataclasses.dataclass
class DecodeSpec:
    """Serializable decoding settings used for evaluation.

    The same spec is embedded in every eval result, so a fine-tuned run and the
    vanilla baseline can be shown to have been scored identically.
    """

    batch_size: int = 16
    num_beams: int = 1
    do_sample: bool = False
    language: str = "en"
    task: str = "transcribe"
    force_language: bool = True
    max_new_tokens: int = 200
    dtype: str = "fp16"
    normalizer: str = "WhisperTokenizer.normalize (model english.json spelling map)"
    path: str = "fixed"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def spec_from_config(config, path: str = "fixed") -> DecodeSpec:
    return DecodeSpec(
        batch_size=getattr(config, "eval_batch_size", 16),
        num_beams=getattr(config, "decode_num_beams", 1),
        do_sample=False,
        language=getattr(config, "language", "en"),
        task=getattr(config, "task", "transcribe"),
        force_language=getattr(config, "decode_force_language", True),
        max_new_tokens=getattr(config, "decode_max_new_tokens", 200),
        dtype=getattr(config, "decode_dtype", "fp16"),
        path=path,
    )


def torch_dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[name]


def _normalize(processor, text: str) -> str:
    """Whisper's own normalizer, including the english.json spelling map."""
    return processor.tokenizer.normalize(text)


@torch.inference_mode()
def transcribe_and_score(
    model,
    processor,
    dataset,
    spec: DecodeSpec,
    desc: str = "decode",
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """Decode ``dataset`` and score WER.

    ``dataset`` is one of our own SizedIterableDatasets yielding VoiceSample objects
    (i.e. exactly what ``utils.prepare_dataset`` returns) -- no HF-dataset unwrapping,
    so the text and audio a run is scored on come from the same pipeline that feeds
    training.

    Returns (metrics, normalized_predictions, normalized_references).
    """
    was_training = model.training
    model.eval()

    try:
        total = len(dataset)
    except TypeError:
        total = None

    raw_preds: List[str] = []
    raw_refs: List[str] = []

    batch_audio: List[Any] = []
    batch_text: List[str] = []
    n_source = 0

    gen_kwargs: Dict[str, Any] = {
        "num_beams": spec.num_beams,
        "do_sample": spec.do_sample,
        "max_new_tokens": spec.max_new_tokens,
    }
    if spec.force_language:
        # Supported API in transformers >= 4.39; forced_decoder_ids is deprecated.
        gen_kwargs["language"] = spec.language
        gen_kwargs["task"] = spec.task

    t0 = time.perf_counter()

    def _flush():
        if not batch_audio:
            return
        features = processor(
            batch_audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",
        ).input_features
        features = features.to(device=model.device, dtype=model.dtype)
        generated = model.generate(features, **gen_kwargs)
        texts = processor.batch_decode(generated, skip_special_tokens=True)
        raw_preds.extend(texts)
        raw_refs.extend(batch_text)
        batch_audio.clear()
        batch_text.clear()

    for sample in tqdm(dataset, total=total, desc=desc):
        n_source += 1
        batch_audio.append(sample.audio)
        batch_text.append(sample.text)
        if len(batch_audio) >= spec.batch_size:
            _flush()
    _flush()

    decode_wall_s = time.perf_counter() - t0

    # Normalize, then drop pairs whose reference normalizes to nothing. This is the
    # only sample loss in the fixed path, and it is counted and reported.
    preds: List[str] = []
    refs: List[str] = []
    n_dropped_empty_ref = 0
    for p, r in zip(raw_preds, raw_refs):
        rn = _normalize(processor, r)
        if len(rn.strip()) == 0:
            n_dropped_empty_ref += 1
            continue
        preds.append(_normalize(processor, p))
        refs.append(rn)

    if not refs:
        raise ValueError(
            f"No scorable utterances for '{getattr(dataset, 'name', dataset)}': "
            f"{n_source} samples yielded by the dataset, all references normalized to empty."
        )

    wer = _wer_metric().compute(predictions=preds, references=refs)

    # Counts accumulated by VoiceDataset.__iter__ during this pass, if the wrapper
    # chain exposes them (see data/datasets.py).
    filter_counts = _collect_filter_counts(dataset)

    metrics: Dict[str, Any] = {
        "wer": float(wer),
        "n_utts_scored": len(refs),
        "n_utts_decoded": len(raw_preds),
        "n_samples_from_dataset": n_source,
        "n_dropped_empty_ref": n_dropped_empty_ref,
        "n_ref_words": sum(len(r.split()) for r in refs),
        "decode_wall_s": round(decode_wall_s, 3),
        "spec": spec.to_dict(),
    }
    metrics.update(filter_counts)
    # The number the WER is actually an average over. Reported explicitly so that a
    # filtered evaluation set can never be mistaken for a complete one.
    metrics["denominator"] = len(refs)

    if was_training:
        model.train()
    return metrics, preds, refs


def _collect_filter_counts(dataset) -> Dict[str, Any]:
    """Pull the per-pass drop counters off the underlying VoiceDataset, if present."""
    node = dataset
    for _ in range(8):
        counts = getattr(node, "last_pass_counts", None)
        if counts:
            return {f"source_{k}": v for k, v in counts.items()}
        node = getattr(node, "_dataset", None)
        if node is None:
            break
    return {}
