import sys
import os
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN)
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from data import datasets, types
from data.configs import spgispeech

# Mock args
args = types.VoiceDatasetArgs(
    split=types.DatasetSplit.VALIDATION,
    shuffle=False,
    shuffle_seed=42,
    max_audio_duration_secs=30.0,
)

# Use spgispeech-s config
config = spgispeech.SPGISPEECH_S_CONFIG

print(f"Attempting to load dataset: {config.name}...")
try:
    ds = datasets.GenericDataset(args, config)
    print("Dataset loaded successfully!")
    
    # Fetch a sample
    print("Fetching first sample...")
    iter_ds = iter(ds)
    sample = next(iter_ds)
    print("Sample fetched successfully!")
    print(f"Text: {sample.text}")
    print(f"Audio shape: {sample.audio.shape}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
