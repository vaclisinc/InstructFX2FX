"""Audio loading and formatting utilities for SocialFX dataset.

This module provides utilities for loading audio files with validation
and formatting dataset examples for few-shot prompting.
"""

import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from judge_system.data.models import SocialFXExample

logger = logging.getLogger(__name__)


def load_audio_sample(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file with validation.

    Loads an audio file using soundfile and validates its existence and
    sample rate. Issues a warning if the sample rate differs from the
    standard 44100 Hz.

    Args:
        path: Path to the audio file to load

    Returns:
        Tuple containing:
            - audio: NumPy array of audio samples
            - sr: Sample rate in Hz

    Raises:
        FileNotFoundError: If the audio file does not exist
        RuntimeError: If soundfile fails to read the file

    Example:
        >>> audio, sr = load_audio_sample("guitar.wav")
        >>> print(f"Loaded audio with shape {audio.shape} at {sr} Hz")
        Loaded audio with shape (220500,) at 44100 Hz
    """
    audio_path = Path(path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        audio, sr = sf.read(str(audio_path))
    except Exception as e:
        raise RuntimeError(f"Failed to read audio file {path}: {e}") from e

    # Validate sample rate
    if sr != 44100:
        logger.warning(
            f"Sample rate {sr} Hz != 44100 Hz for file {path}, "
            "resampling may be needed"
        )

    logger.debug(f"Loaded audio from {path}: shape={audio.shape}, sr={sr} Hz")

    return audio, sr


def format_example_for_prompt(example: SocialFXExample) -> str:
    """Format example for few-shot prompting.

    Converts a SocialFXExample into a formatted string suitable for
    inclusion in few-shot prompts. The format includes the description,
    instrument type, and parameters in JSON format with 2-space indentation.

    Args:
        example: SocialFXExample to format

    Returns:
        Formatted string representation of the example

    Example:
        >>> example = SocialFXExample(
        ...     id=1,
        ...     description="warm and intimate",
        ...     instrument="guitar",
        ...     effect_type="eq",
        ...     parameters={"bands": [{"frequency": 200, "gain": 3.0, "q": 0.7}]}
        ... )
        >>> print(format_example_for_prompt(example))
        Description: "warm and intimate"
        Instrument: guitar
        Parameters:
        {
          "bands": [
            {
              "frequency": 200,
              "gain": 3.0,
              "q": 0.7
            }
          ]
        }
    """
    formatted = f'''Description: "{example.description}"
Instrument: {example.instrument}
Parameters:
{json.dumps(example.parameters, indent=2)}'''

    return formatted
