import abc
import logging
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import datasets as hf_datasets
import numpy as np

import transformers
from torch.utils import data

from . import data_sample
from . import text_proc
from . import types


def _get_worker_info(length: int):
    """
    Calculate number of samples for this worker, accounting for max workers limit.
    Returns 0 if worker_id exceeds max allowed workers.
    """
    worker_id = 0
    num_workers = 1
    worker_info = data.get_worker_info()
    if worker_info is not None:
        worker_id = worker_info.id
        num_workers = worker_info.num_workers

    # Calculate samples for this worker
    worker_samples = length // num_workers
    extra_samples = length % num_workers

    # Workers with id < extra_samples get one extra sample
    if worker_id < extra_samples:
        worker_samples += 1

    return num_workers, worker_id, worker_samples


class SizedIterableDataset(abc.ABC, data.IterableDataset):
    """
    An interface for an IterableDataset that provides a length method.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


class VoiceDataset(SizedIterableDataset):
    """
    Base class for streaming voice datasets.
    Wraps a Hugging Face dataset.
    """

    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        super().__init__()
        self._args = args
        self._rng = np.random.default_rng(self._args.shuffle_seed)
        self._name = "[unset]"
        self._length = -1

    # num_samples is the total number of samples in the dataset
    def _init_dataset(
        self,
        dataset: data.Dataset,
        name: str,
        num_samples: int,
    ) -> None:
        self._dataset = dataset
        self._name = name
        self._length = num_samples

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def _load_hf_dataset(
        self,
        path: str,
        name: Optional[str] = None,
        *,
        split: Optional[str] = None,
        streaming: bool = True,
        audio_field: Optional[str] = None,
        features: Optional[hf_datasets.Features] = None,
    ) -> data.Dataset:
        # Handle slicing for streaming datasets manually
        # Syntax: split_name[start:stop]
        # Note: This only supports integer indices for now, not percentages
        slice_start = None
        slice_stop = None
        
        if streaming and split and "[" in split and "]" in split:
            import re
            # Match split_name[start:stop]
            match = re.match(r"^(.+)\[(\d*):(\d*)\]$", split)
            if match:
                split = match.group(1)
                start_str = match.group(2)
                stop_str = match.group(3)
                if start_str:
                    slice_start = int(start_str)
                if stop_str:
                    slice_stop = int(stop_str)
            else:
                 # Check for percentage which is not supported easily yet
                 if "%" in split:
                     raise ValueError(f"Percentage slicing is not supported for streaming datasets in internal logic: {split}")

        # HF datasets sometimes fails to download due to network issues, so retry a few times.
        dataset = hf_datasets.load_dataset(
            path,
            name,
            split=split,
            streaming=streaming,
            features=features,
            download_config=hf_datasets.DownloadConfig(max_retries=10),
            trust_remote_code=True,
        )
        
        if slice_start is not None:
            dataset = dataset.skip(slice_start)
        
        if slice_stop is not None:
            # For [start:stop], we want (stop - start) items if start is present
            # If start is None (0), we want stop items.
            # However, take() takes N items from current position.
            # If we skipped start, we are at start. We want to reach stop.
            # So we take (stop - start).
            
            take_count = slice_stop
            if slice_start is not None:
                take_count = slice_stop - slice_start
            
            # If take_count is negative, it means stop < start, which returns empty
            if take_count < 0:
                take_count = 0
                
            dataset = dataset.take(take_count)

        if audio_field is not None:
            dataset = dataset.cast_column(
                audio_field, hf_datasets.Audio(sampling_rate=data_sample.SAMPLE_RATE)
            )
        if self._args.shuffle:
            if streaming:
                dataset = dataset.shuffle(
                    seed=self._args.shuffle_seed,
                    buffer_size=self._args.shuffle_buffer_size,
                )
            else:
                dataset = dataset.shuffle(seed=self._args.shuffle_seed)
        return dataset



    def __iter__(self):
        num_workers, worker_id, worker_samples = _get_worker_info(self._length)
        
        dataset_iter = None
        # Handle sharding for map-style datasets (streaming=False)
        if num_workers > 1 and isinstance(self._dataset, hf_datasets.Dataset):
            # Calculate start/end indices for this worker
            base_chunk = self._length // num_workers
            remainder = self._length % num_workers
            start_idx = worker_id * base_chunk + min(worker_id, remainder)
            end_idx = start_idx + worker_samples
            
            # Create a subset for this worker
            # We use select() to get a slice of the dataset while preserving features
            dataset_shard = self._dataset.select(range(start_idx, end_idx))
            dataset_iter = iter(dataset_shard)
        else:
            if num_workers > 1:
                if not hasattr(self._dataset, "n_shards"):
                     logging.warning(f"{self._name} does not have n_shards attribute. Assuming custom sharding in builder.")
                elif self._dataset.n_shards < num_workers:
                     logging.warning(f"{self._name} has {self._dataset.n_shards} shards, which is less than the number of workers ({num_workers}).")
            dataset_iter = iter(self._dataset)

        actual_length = 0
        skipped_samples = 0
        bad_samples = 0
        for row in dataset_iter:
            actual_length += 1
            sample = self._get_sample(row)
            if sample is None:
                print(f"Sample is None in dataset {self.name} for row {row}")
                bad_samples += 1
                continue

            if len(sample.text.strip()) == 0:
                print(
                    f"Sample has empty text in dataset {self.name} for row {row}"
                )
                bad_samples += 1
                continue

            if sample.audio is None:
                print(f"Audio is None for sample {sample}")
                bad_samples += 1
                continue
            if sample.audio.shape[-1] == 0:
                print(f"Audio length is 0 for sample {sample}")
                bad_samples += 1
                continue
            if (
                self._args.max_audio_duration_secs > 0
                and sample.audio.shape[-1] / data_sample.SAMPLE_RATE
                > self._args.max_audio_duration_secs
            ):
                skipped_samples += 1
                continue

            yield sample

        logging.info(
            f"Extracted {actual_length} samples from {self.name} (total: {len(self)}), removed {bad_samples} bad samples, and skipped {skipped_samples} samples for exceeding max audio duration ({self._args.max_audio_duration_secs}s)."
        )

    @abc.abstractmethod
    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        """
        Converts a row from the dataset into a VoiceSample.
        Returns None if the sample should be skipped.
        """

    def _get_audio(
        self, row: transformers.BatchFeature, column_name: Optional[str] = "audio"
    ) -> np.ndarray:
        # Hugging Face datasets have an Audio object, with array and sampling_rate fields.

        if column_name in row:
            audio = row[column_name]["array"]
            sampling_rate = row[column_name]["sampling_rate"]
        elif f"{column_name}_array" in row:
            audio = row[f"{column_name}_array"]
            sampling_rate = row[f"{column_name}_sampling_rate"]
        else:
            raise ValueError("No audio field found in row.")
        assert sampling_rate == data_sample.SAMPLE_RATE
        return audio

    def _make_sample(
        self,
        text: str,
        audio: np.ndarray,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> data_sample.VoiceSample:
        return data_sample.VoiceSample(
            audio=audio,
            text=text,
            extra_kwargs=extra_kwargs,
        )


class GenericDataset(VoiceDataset):
    def __init__(
        self,
        args: types.VoiceDatasetArgs,
        config: types.DatasetConfig,
    ) -> None:
        assert config.splits is not None
        assert config.path is not None

        super().__init__(args)
        self._config = config
        dsets = []
        total_samples = 0
        for split in config.splits:
            if split.split == self._args.split:
                ds = self._load_hf_dataset(
                    config.path,
                    config.subset,
                    split=split.source_split or split.name,
                    streaming=(
                        config.streaming
                        if config.streaming is not None
                        else True
                    ),
                    audio_field=config.audio_field,
                    features=config.features,
                )
                dsets.append(ds)
                total_samples += split.num_samples
        assert (
            len(dsets) > 0
        ), f"The {config.name} dataset has no {self._args.split} splits."
        dataset = ds if len(dsets) == 1 else hf_datasets.concatenate_datasets(dsets)

        dataset_name = f"{config.name}.{self._args.split.value}"

        super()._init_dataset(dataset, dataset_name, total_samples)

    def __str__(self):
        return f"GenericDataset({self._config})"

    def _resolve_audio_path(self, relative_path: str) -> Path:
        path_value = Path(relative_path)
        if path_value.is_absolute() and path_value.exists():
            return path_value
        if path_value.exists():
            return path_value

        search_roots: List[str] = []
        if self._config.audio_root:
            search_roots.append(self._config.audio_root)
        env_roots = [
            "LIBRISQA_AUDIO_ROOT",
            "LIBRISPEECH_ROOT",
            "LIBRISPEECH_DATA_ROOT",
            "LIBRISPEECH_PATH",
            "LIBRI_SPEECH_ROOT",
        ]
        for env_var in env_roots:
            env_value = os.getenv(env_var)
            if env_value:
                search_roots.append(env_value)
        search_roots.append(".")

        for root in search_roots:
            candidate = Path(root).expanduser() / path_value
            if candidate.exists():
                return candidate
            if candidate.suffix.lower() == ".wav":
                alt = candidate.with_suffix(".flac")
                if alt.exists():
                    return alt

            if (
                path_value.parts
                and path_value.parts[0].lower() == "librispeech"
                and Path(root).expanduser().name.lower() == "librispeech"
            ):
                candidate = Path(root).expanduser() / Path(*path_value.parts[1:])
                if candidate.exists():
                    return candidate
                if candidate.suffix.lower() == ".wav":
                    alt = candidate.with_suffix(".flac")
                    if alt.exists():
                        return alt

        raise FileNotFoundError(
            f"Unable to locate audio file for path '{relative_path}'. "
            "Set LIBRISQA_AUDIO_ROOT or LIBRISPEECH_ROOT to the LibriSpeech directory."
        )

    def _load_audio_from_path(self, row: transformers.BatchFeature) -> Optional[np.ndarray]:
        if self._config.audio_path_field is None:
            raise ValueError("audio_path_field must be set to load audio from path.")
        path_value = row.get(self._config.audio_path_field)
        if path_value is None:
            logging.warning(
                "Missing audio path for dataset %s; skipping sample.", self._config.name
            )
            return None
        if isinstance(path_value, str) and len(path_value.strip()) == 0:
            logging.warning(
                "Empty audio path for dataset %s; skipping sample.", self._config.name
            )
            return None
        audio_path = self._resolve_audio_path(path_value)
        return data_sample.audio_from_file(str(audio_path))

    def _get_audio_from_config(self, row: transformers.BatchFeature) -> Optional[np.ndarray]:
        if self._config.audio_path_field is not None:
            return self._load_audio_from_path(row)
        if self._config.audio_field is not None:
            return self._get_audio(row, self._config.audio_field)
        raise ValueError(
            f"Dataset {self._config.name} does not define audio_field or audio_path_field."
        )

    def _get_sample(self, row) -> Optional[data_sample.VoiceSample]:

        # Setting up extra_kwargs for datasets like Voicebench
        extra_kwargs = None
        if (
            self._config.eval_config is not None
            and self._config.eval_config.extra_kwargs_map is not None
        ):
            extra_kwargs = {
                key: row.get(value)
                for key, value in self._config.eval_config.extra_kwargs_map.items()
            }

        # Get text field from config (default to 'text')
        text_field = self._config.transcript_field or "text"
        
        # Special handling for libriheavy which stores text in supervisions[0]['text']
        if text_field not in row and 'supervisions' in row:
            if len(row['supervisions']) > 0 and 'text' in row['supervisions'][0]:
                raw_text = row['supervisions'][0]['text']
            else:
                return None
        elif text_field not in row:
            raise ValueError(
                f"Text field '{text_field}' not found in dataset row. Available fields: {list(row.keys())}"
            )
        else:
            raw_text = row[text_field]
        
        # Apply text formatting
        try:
            text = text_proc.format_asr_text(raw_text)
        except Exception:
            # Skip samples with formatting errors (e.g., garbage tags)
            return None

        audio: Optional[np.ndarray] = self._get_audio_from_config(row)
        if audio is None:
            return None
        return self._make_sample(
            text=text,
            audio=audio,
            extra_kwargs=extra_kwargs,
        )

    def get_config(self):
        return self._config


class LibriSpeechDummyDataset(GenericDataset):
    def __init__(self, args: types.VoiceDatasetArgs) -> None:
        VoiceDataset.__init__(self, args)
        # This dataset doesn't support streaming.
        dataset = self._load_hf_dataset(
            "hf-internal-testing/librispeech_asr_dummy",
            "clean",
            split="validation",
            streaming=False,
        )
        self._init_dataset(dataset, "dummy", 73)

    def __str__(self):
        return "LibriSpeechDummyDataset"

    @property
    def name(self):
        return "dummy"

    def get_config(self):
        return types.DatasetConfig(
            name="dummy",
            path="hf-internal-testing/librispeech_asr_dummy",
        )

    def _get_sample(
        self, row: transformers.BatchFeature
    ) -> Optional[data_sample.VoiceSample]:
        text = text_proc.format_asr_text(row["text"])
        return self._make_sample(
            text=text,
            # some of our test models that use this dataset can only handle up to 4 seconds of audio
            audio=self._get_audio(row, "audio")[: 4 * data_sample.SAMPLE_RATE],
        )


class EmptyDataset(SizedIterableDataset):
    def __init__(self, length: int = 1) -> None:
        self._length = length

    def __iter__(self):
        return iter([])

    def __len__(self):
        return self._length

    def __str__(self):
        return f"EmptyDataset(length={self._length})"

    @property
    def name(self):
        return "empty"


class InterleaveDataset(SizedIterableDataset):
    """Interleaves multiple SizedIterableDataset objects based on normalized weights."""

    def __init__(
        self,
        datasets: Sequence[SizedIterableDataset],
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Args:
            datasets: A list of SizedIterableDataset objects.
            weights: An optional list of dataset weights, i.e., the number of times it should be repeated.
            seed: Optional seed for reproducibility.
        """
        self._datasets = datasets
        if weights is not None:
            assert len(weights) == len(datasets)
        else:
            weights = [1.0] * len(datasets)
        self._weights = weights
        self._weighted_samples = [int(w * len(d)) for w, d in zip(weights, datasets)]
        self._total_samples = sum(self._weighted_samples)

    def __iter__(self):
        ds_iters = [iter(ds) for ds in self._datasets]
        ds_pos = [0] * len(ds_iters)
        num_workers, worker_id, worker_samples = _get_worker_info(self._total_samples)
        # Find the iterator that is least far along and vend from it.
        for i in range(worker_samples):
            min_fraction = 1.0
            for j in range(len(ds_iters)):
                iter_fraction = ds_pos[j] / self._weighted_samples[j]
                if iter_fraction < min_fraction:
                    min_fraction = iter_fraction
                    iter_index = j
            try:
                yield next(ds_iters[iter_index])
            except StopIteration:
                ds_iters[iter_index] = iter(self._datasets[iter_index])
                try:
                    yield next(ds_iters[iter_index])
                except StopIteration:
                    warnings.warn(
                        f"Dataset {iter_index} is empty for worker {worker_id}/{num_workers}. num_workers is likely too high. Stopping iteration."
                    )
                    break
            ds_pos[iter_index] += 1

    def __len__(self):
        return self._total_samples

    def __str__(self):
        return "+".join([f"{d}:{w:.2f}" for w, d in zip(self._weights, self._datasets)])

    @property
    def name(self):
        return "+".join([ds.name for ds in self._datasets])


class Dataproc(SizedIterableDataset):
    """Base class to preprocess a dataset of VoiceSamples."""

    def __init__(self, dataset: SizedIterableDataset) -> None:
        self._dataset = dataset

    @abc.abstractmethod
    def _process(self, sample: data_sample.VoiceSample) -> Dict[str, Any]:
        pass

    def __iter__(self):
        # Replace generator expression with a regular function that yields items
        for sample in self._dataset:
            processed = self._process(sample)
            if processed is not None:
                yield processed

    def __len__(self):
        return len(self._dataset)

    def __str__(self):
        return f"Dataproc({self._dataset})"

    @property
    def name(self):
        return self._dataset.name


class Range(SizedIterableDataset):
    """Limits the number of samples from another dataset."""

    def __init__(
        self,
        dataset: SizedIterableDataset,
        num_samples: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        try:
            ds_len = len(dataset)
        except (ValueError, TypeError):
            ds_len = float("inf")

        self._length = num_samples or ds_len
        
        if ds_len != float("inf") and self._length > ds_len:
            warnings.warn(
                f"num_samples ({self._length}) exceeds dataset length ({ds_len}). Truncating to {ds_len}."
            )
            self._length = ds_len
        
        if self._length == float("inf"):
             # If we still have infinite length (no num_samples provided and dataset is infinite), 
             # we can't really set a length for Range. 
             # But Range is usually used to *limit* samples. 
             # If num_samples is None, we just pass through.
             # But we need an integer for __len__.
             # Let's default to a large number or keep it as is?
             # SizedIterableDataset requires int.
             # If we are here, it means we probably want to just iterate until exhaustion if num_samples is None.
             # But __len__ must return int.
             pass

        # If self._length is still inf, we cast to a large int for __len__ protocol if needed, 
        # or we accept that len(Range) might fail too if we don't set it.
        # But let's just use the logic above. 
        if self._length == float("inf"):
             # Fallback for __len__
             self._length = 2**63 - 1 

        self._name = f"{dataset.name}.{self._length}"

    def __iter__(self):
        num_workers, worker_id, worker_samples = _get_worker_info(self._length)
        if worker_samples == 0:
            return iter([])
        yielded_samples = 0
        try:
            for sample in self._dataset:
                yielded_samples += 1
                yield sample
                if yielded_samples == worker_samples:
                    break
        except Exception as e:
            logging.error(
                f"Worker {worker_id}/{num_workers} failed after yielding {yielded_samples}/{worker_samples} samples, out of {self._length} total samples with error: {e}"
            )
            raise e
        if yielded_samples < worker_samples:
            logging.warn(
                f"Worker {worker_id}/{num_workers} only yielded {yielded_samples} (expected {worker_samples}) samples, out of {self._length} total samples"
            )

    def __str__(self):
        return f"Range({self._dataset}%{len(self)})"

    def __len__(self):
        return self._length

    @property
    def name(self):
        return self._name

    def get_config(self):
        if isinstance(self._dataset, GenericDataset):
            return self._dataset.get_config()
        else:
            raise ValueError("Cannot get config for non-GenericDataset")
