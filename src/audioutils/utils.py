import math
import pathlib
import humanize
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from IPython.display import Audio, display
# import whisperx

# def resample_audio(audio_path, target_sample_rate=32000):
#   """Resample audio"""
#   waveform, sr = torchaudio.load(audio_path, normalize=True)
#   if sr != target_sample_rate:
#     resampler = torchaudio.transforms.Resample(
#         orig_freq=sr, new_freq=target_sample_rate)
#     waveform = resampler(waveform)
#   return waveform, target_sample_rate


def shorten_audio(waveform, sample_rate, traget_duration):
  """Shortens an audio file to the specified duration."""
  num_samples = int(traget_duration * sample_rate)

  shortened_waveform = waveform[:, :num_samples]
  return shortened_waveform

def create_clip(waveform, sample_rate, traget_duration):
  """create clips from audio"""

  start_index = int(sample_rate * traget_duration)
  end_index = start_index + int(sample_rate * traget_duration)

  clip_waveform = waveform[:, start_index:end_index]

  return clip_waveform

# def create_30s_clips(audio_path, model, output_dir):
#   """ Generate and store clips with a maximum duration of 30 seconds along with their corresponding texts."""
#   sampling_rate = 16000
#   tell_sampleing_rate = 8000
#   batch_size = 16
#   # create and modify paths to save clips and texts
#   if not isinstance(audio_path, pathlib.PosixPath):
#     audio_path = pathlib.Path(audio_path)
#   clips_dir = pathlib.PosixPath(output_dir) / "audios" / audio_path.stem
#   clips_dir.mkdir(parents=True, exist_ok=True)
#   text_clips_dir = pathlib.PosixPath(output_dir) / "texts" / audio_path.stem
#   text_clips_dir.mkdir(parents=True, exist_ok=True)
#   # load audio and transcribe
#   waveform = whisperx.load_audio(audio_path)
#   result = model.transcribe(waveform, batch_size=batch_size)
#   # identify clips and save
#   for idx, segment in enumerate(result['segments']):
#     text = segment['text']

#     start_time = segment['start']
#     end_time = segment['end']
#     start_idx = int(start_time * sampling_rate)
#     end_idx = int(end_time * sampling_rate)
#     clip = waveform[start_idx:end_idx]

#     clip = torch.tensor(clip).unsqueeze(0)  # convert array to tensor
#     resampler = torchaudio.transforms.Resample(
#       orig_freq=sampling_rate, new_freq=tell_sampleing_rate)
#     clip = resampler(clip)
#     output_f = clips_dir / f"seg_{idx}.wav"
#     torchaudio.save(output_f, clip, tell_sampleing_rate)

#     text_f = text_clips_dir / f"seg_{idx}.txt"
#     with open(text_f, "w") as f:
#       f.write(text)


def draw_spectorgram(audio_path,ax):
  """Draw Spectogram"""
  waveform, sr = librosa.load(audio_path)
  # Compute the spectrogram
  S = librosa.feature.melspectrogram(y=waveform, sr=sr)
  S_db = librosa.power_to_db(S, ref=np.max)
  librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel',ax=ax)

def get_audio_duration(audio_path):
  """Calculate Audio Duration in seconds"""

  info = torchaudio.info(audio_path)
  duration = info.num_frames / info.sample_rate

  return duration

def read_text(path: str) -> str:
  """Read files from text file"""
  with open(path, encoding="utf-8") as f:
      text = f.read()
      return text

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  """visualise waveform"""
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1,figsize=(5, 3))
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  """visualise spectogram from waveform"""
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1,figsize=(5, 3))
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=False)


def play_audio(waveform, sample_rate):
  """Play waveform"""
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  if num_channels == 1:
    display(Audio(waveform[0], rate=sample_rate))
  elif num_channels == 2:
    display(Audio((waveform[0], waveform[1]), rate=sample_rate))
  else:
    raise ValueError("Waveform with more than 2 channels are not supported.")

def _get_sample(path, resample=None):
  effects = [
    ["remix", "1"]
  ]
  if resample:
    effects.extend([
      ["lowpass", f"{resample // 2}"],
      ["rate", f'{resample}'],
    ])
  return torchaudio.sox_effects.apply_effects_file(path, effects=effects)

def get_rir_sample(*,sample_rir_path , resample=None, processed=False):
  rir_raw, sample_rate = _get_sample(sample_rir_path, resample=resample)
  if not processed:
    return rir_raw, sample_rate
  rir = rir_raw[:, int(sample_rate*1.01):int(sample_rate*1.3)]
  rir = rir / torch.norm(rir, p=2)
  rir = torch.flip(rir, [1])
  return rir, sample_rate


def simulate_phone_recording(
    speech_path, noise_path, snr_db=8, rir=False, sample_rate=16000,encode ="gsm"):
  """Augment clear speech to telephony call
  inspired by: https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#simulating-a-phone-recoding
  """

  # Read sudio
  speech, _ = resample_audio(speech_path, target_sample_rate=sample_rate)
  speech_len = speech.shape[1]

  # Apply room reverbration
  if rir:
    rir, _ = get_rir_sample(resample=sample_rate, processed=True)
    speech_ = torch.nn.functional.pad(speech, (rir.shape[1] - 1, 0))
    speech = torch.nn.functional.conv1d(speech_[None, ...], rir[None, ...])[0]

  speech_len = speech.shape[1]
  # apply noise
  noise, _ = resample_audio(noise_path, target_sample_rate=sample_rate)
  noise_len = noise.shape[1]
  ## to select random sectoin of noise
  # noise_len = noise.shape[1]
  # start_index = torch.randint(0, noise_len - speech_len + 1, size=(1, ))
  # noise = noise[:, start_index:start_index + speech_len]
  ## so select the nose from begining
  ## it happens that length noise is less than length of speech
  if noise_len < speech_len:
    # Calculate how many repetitions of noise are needed
    repetitions = int(speech_len / noise_len) + 1
    # Repeat the noise the required number of times
    noise = noise.repeat(1, repetitions)

  noise = noise[:, :speech_len]
  scale = math.exp(snr_db / 10) * noise.norm(p=2) / speech.norm(p=2)
  speech = (scale * speech + noise) / 2

  # Apply filtering and change sample rate
  speech, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
    speech,
    sample_rate,
    effects=[
      ["lowpass", "4000"],
      [
        "compand", "0.02,0.05", "-60,-60,-30,-10,-20,-8,-5,-8,-2,-8", "-8",
        "-7", "0.05"
      ],
      ["rate", "8000"],
    ],
  )

  if encode=="gsm":
    # Apply telephony codec
    speech = torchaudio.functional.apply_codec(speech, sample_rate, format="gsm")

  return speech, sample_rate

def get_trainable_parameters(model):
  """
  count and print number of trainable paraameters in the model
  """
  trainable_params = 0
  for param in model.parameters():
    if param.requires_grad:
      trainable_params += param.numel()
  all_params = sum(p.numel() for p in model.parameters())
  print(
    f"Total Trainable Parameters: {humanize.intword(trainable_params)} | {trainable_params/all_params *100:.2f}%")

def prepare_audio(example, target_sample_rate=8000, encode_gsm=True):
  """Load audio, resample and encode to gsm"""
  audio_path = example["path"]
  waveform, sr = torchaudio.load(audio_path)
  if sr != target_sample_rate:
    resampler = torchaudio.transforms.Resample(
      orig_freq=sr, new_freq=target_sample_rate)
    waveform = resampler(waveform)
  if encode_gsm:
    torchaudio.functional.apply_codec(
      waveform, target_sample_rate, format="gsm")

  example["audio"] = {"path":audio_path,"arrey":waveform, "sampling_rate":sr}
  return example

def extract_topic_info(model_version:str,topics, probs, topic_model):
  """ Reformat topic and extact label"""

  topic2label = {
    n: l
    for n, l in zip(
      topic_model.get_topic_info()["Topic"],
      topic_model.get_topic_info()["CustomName"],
      )
  }
  topic2label[-1] = "outlier"
  result = [
    {"model_version":[model_version], "topic": [topic], "prob": [prob], "label": [topic2label[topic]]}
    for topic, prob in zip(topics, probs)
  ]
  
  return result


def print_model_structure(module, level=0):
  print(" " * level, type(module).__name__, ":")
  for name, child in module.named_children():
    print_model_structure(child, level + 4)

def get_audio_duration(example):
  """Calculate Audio Duration in seconds"""
  audio_path = example["audio"]["path"]
  info = torchaudio.info(audio_path)
  duration = info.num_frames / info.sample_rate
  example["audio"]["length"] = duration
  return example

def get_dataset_duration(data):
  """Calculate the duration of dataset in seconds"""
  dataset_duration = [example["audio"]["length"] for example in data]
  return sum(dataset_duration)