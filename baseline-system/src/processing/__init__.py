"""Audio processing with parameters.

This package provides audio I/O operations, effect chain building,
and audio processing utilities for the baseline system.
"""

from src.processing.io import (
    AudioLoader,
    load_audio,
    save_audio,
    SUPPORTED_FORMATS,
)

from src.processing.exceptions import (
    AudioProcessingError,
    AudioValidationError,
    AudioLoadError,
    AudioSaveError,
    UnsupportedFormatError,
    AudioClippingError,
)


__all__ = [
    # Audio I/O
    "AudioLoader",
    "load_audio",
    "save_audio",
    "SUPPORTED_FORMATS",
    # Exceptions
    "AudioProcessingError",
    "AudioValidationError",
    "AudioLoadError",
    "AudioSaveError",
    "UnsupportedFormatError",
    "AudioClippingError",
]
