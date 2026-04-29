
import logging
from typing import Dict, Any, Optional, Callable
import data.text_proc as text_proc

def clean_text(raw_text: str) -> Optional[str]:
    """
    Applies text processing and returns cleaned text.
    Returns None if text is empty after processing or processing fails.
    """
    try:
        processed_text = text_proc.format_asr_text(raw_text)
        if len(processed_text) == 0:
            return None
        return processed_text
    except Exception:
        return None

def is_valid_sample(
    sample: Dict[str, Any], 
    max_duration: float = 30.0, 
    min_duration: float = 0.0,
    check_text: bool = True,
    text_key: Optional[str] = None
) -> bool:
    """
    Checks if a sample is valid for training/inference.
    
    Args:
        sample: The data sample dictionary.
        max_duration: Maximum allowed audio duration in seconds.
        min_duration: Minimum allowed audio duration.
        check_text: Whether to validate text content (non-empty after processing).
        
    Returns:
        True if valid, False otherwise.
    """
    sample_id = sample.get('segment_id', 'unknown')
    
    # 1. Check Audio Duration
    if "audio" in sample:
        audio_data = sample["audio"]
        dur = 0.0
        if "array" in audio_data:
            if audio_data.get("sampling_rate", 0) > 0:
                dur = len(audio_data["array"]) / audio_data["sampling_rate"]
        
        if dur > 0:
            if dur >= max_duration:
                return False
            if dur < min_duration:
                return False

    # 2. Check Text Validity
    if check_text:
        text_field = text_key
        if text_field is None:
            if "transcript" in sample:
                text_field = "transcript"
            elif "text" in sample:
                text_field = "text"
        
        if text_field:
            raw_text = sample[text_field]
            try:
                processed_text = text_proc.format_asr_text(raw_text)
                if len(processed_text) == 0:
                    return False
            except Exception:
                return False
        else:
            return False
            
    return True
