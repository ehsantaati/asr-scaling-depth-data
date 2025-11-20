import torchaudio
import numpy as np
import torch
import re
import string
from collections import Counter
import math

def reformat_data(example):
  """Reformat data items"""
  example["audio"] = {
    "path": example["audio_path"],
    "array": example["array"],
    "sampling_rate": example["sampling_rate"],
  }
  example["text"] = example["text"]

  return example

def load_audio(example):
  """Load audio to array"""
  audio_path = example["audio_path"]
  array, sr = torchaudio.load(audio_path)
  example["array"] = array.numpy()
  example["sampling_rate"] = sr

  return example


def resample_audio(example,new_freq = 8000):
  """Resample audio to 8khz"""
  orig_freq = example["audio"]["sampling_rate"]
  array = example["audio"]["array"]
  tensor_array = torch.tensor(array)
  resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
  if orig_freq != new_freq:
    example["audio"]["array"] = resampler(tensor_array)
    example["audio"]["sampling_rate"] = new_freq
  return example


def get_audio_duration(example):
  """Calculate Audio Duration in seconds"""
  audio_path = example["audio"]["path"]
  info = torchaudio.info(audio_path)
  duration = info.num_frames / info.sample_rate
  example["input_length"] = duration
  return example


def get_dataset_duration(data):
  """Calculate the duration of dataset in seconds"""
  dataset_duration = [example["input_length"] for example in data]
  return sum(dataset_duration)

def encode2gsm(example):
  """encode audio into gsm"""
  array = example["audio"]["array"]
  tensor_array = torch.tensor(array).unsqueeze(0)
  sampling_rate = example["audio"]["sampling_rate"]
  encode = "gsm"
  __array__ = torchaudio.functional.apply_codec(
    tensor_array, sample_rate=sampling_rate, format=encode)
  example["audio"]["array"] = __array__
  example["audio"]["encode"] = encode
  return example

def is_audio_in_length_range(length,max_input_length):
  return length < max_input_length

def remove_punctuations(text: str) -> str:
    """
    Removes punctuations from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned text.
    """
    return re.sub(r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~‘’', "", text)

def calculate_compression_score(text):
  # Normalize the text: convert to lowercase and remove punctuation
  text = text.lower()
  text = text.translate(str.maketrans("", "", string.punctuation))

  # Tokenize the text into words
  words = text.split()

  # Count word frequencies
  word_frequencies = Counter(words)

  # Calculate the total number of words
  total_words = len(words)

  # Calculate the entropy
  entropy = 0
  for word, count in word_frequencies.items():
    probability = count / total_words
    entropy -= probability * math.log2(probability)

  # Normalize the entropy
  max_entropy = math.log2(len(word_frequencies))
  normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

  # Invert the score so that higher scores mean more repetition
  compression_score = 1 - normalized_entropy

  # Print the score
  # print(f"Compression-like Repetition Score: {compression_score:.4f}")

  return compression_score