import base64
import dataclasses
import io
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from numpy import typing as npt

SAMPLE_RATE = 16000


def audio_from_file(path: str) -> np.ndarray:
    """Load audio from a file, converting to float32 PCM @ 16 kHz."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE)
    assert audio.dtype == np.float32
    return audio


def audio_from_buf(buf: bytes) -> np.ndarray:
    """Load audio from a buffer, converting to float32 PCM @ 16 kHz."""
    audio, _ = librosa.load(io.BytesIO(buf), sr=SAMPLE_RATE)
    assert audio.dtype == np.float32
    return audio


def audio_to_wav(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert audio to WAV format, 16-bit PCM @ 16 kHz."""
    assert audio.dtype == np.float32
    with io.BytesIO() as buf:
        sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()


def audio_to_wav_base64(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio to a base64-encoded WAV file."""
    return base64.b64encode(audio_to_wav(audio, sample_rate)).decode("utf-8")


def audio_to_data_uri(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """Convert audio to a data URI."""
    return f"data:audio/wav;base64,{audio_to_wav_base64(audio, sample_rate)}"

@dataclasses.dataclass
class VoiceSample:
    """A sample containing audio and its transcript for ASR training."""
    
    audio: npt.NDArray[np.float32]
    """Audio data as float32 PCM @ `sample_rate`."""
    text: str
    """The transcript of the audio."""
    sample_rate: int = SAMPLE_RATE
    """Audio sample rate in Hz."""
    extra_kwargs: Optional[Dict[str, Any]] = None
    """For evaluations, extra columns from the sample."""

    @staticmethod
    def from_json(data: Dict[str, Any]) -> "VoiceSample":
        """Convert from JSON format; audio is expected as base64ed WAV."""
        audio_bytes = base64.b64decode(data["audio"])
        return VoiceSample(audio=audio_from_buf(audio_bytes), text=data["text"])

    @staticmethod
    def from_file(text: str, path: str) -> "VoiceSample":
        """Create a VoiceSample from text and an audio file."""
        return VoiceSample(audio=audio_from_file(path), text=text)

    @staticmethod
    def from_buf(text: str, buf: bytes) -> "VoiceSample":
        """Create a VoiceSample from text and an encoded audio buffer."""
        return VoiceSample(audio=audio_from_buf(buf), text=text)

    @staticmethod
    def from_raw(text: str, audio: np.ndarray, sample_rate: int) -> "VoiceSample":
        """Create a VoiceSample from text and raw audio data with sample rate."""
        return VoiceSample(audio=audio, text=text, sample_rate=sample_rate)

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format; audio is written as base64ed WAV."""
        return {
            "text": self.text,
            "audio": audio_to_wav_base64(self.audio, self.sample_rate)
        }

    def __post_init__(self):
        """Ensure audio is float32 PCM."""
        if self.audio is not None:
            if self.audio.dtype == np.float64:
                self.audio = self.audio.astype(np.float32)
            elif self.audio.dtype == np.int16:
                self.audio = self.audio.astype(np.float32) / np.float32(32768.0)
            elif self.audio.dtype == np.int32:
                self.audio = self.audio.astype(np.float32) / np.float32(2147483648.0)
            assert (
                self.audio.dtype == np.float32
            ), f"Unexpected audio dtype: {self.audio.dtype}"
            assert self.audio.ndim == 1, f"Unexpected audio shape: {self.audio.shape}"
