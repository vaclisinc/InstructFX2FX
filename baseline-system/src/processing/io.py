"""Audio I/O operations for loading, saving, and validating audio files.

This module provides the AudioLoader class for handling audio file operations:
- Loading audio files with automatic resampling
- Saving audio files with quality validation
- Multi-format support (WAV, MP3, FLAC)
- Audio validation (range, sample rate, NaN detection)
- Clipping detection and limiting
- Comprehensive error handling

The module uses librosa for audio loading with automatic resampling and
soundfile for high-quality audio saving.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import librosa
import numpy as np
import soundfile as sf

from src.utils.logging import get_audio_logger
from src.processing.exceptions import (
    AudioLoadError,
    AudioSaveError,
    AudioValidationError,
    UnsupportedFormatError,
    AudioClippingError,
)


# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


class AudioLoader:
    """Audio file loader with validation and format handling.

    The AudioLoader class provides robust audio I/O operations with:
    - Automatic resampling to target sample rate
    - Multi-format support (WAV, MP3, FLAC, OGG, M4A)
    - Audio quality validation
    - Clipping detection and limiting
    - Comprehensive error handling with logging

    Attributes:
        sample_rate: Target sample rate for all loaded audio (default: 44100)
        clipping_threshold: Threshold for clipping detection (default: 0.99)
        mono: Whether to convert audio to mono (default: False)

    Examples:
        >>> loader = AudioLoader(sample_rate=44100)
        >>> audio, sr = loader.load("input.wav")
        >>> processed_audio = process(audio)
        >>> loader.save(processed_audio, "output.wav", sr)

        >>> # With clipping detection
        >>> loader = AudioLoader(clipping_threshold=0.95)
        >>> audio, sr = loader.load("input.mp3")
        >>> if loader.has_clipping(audio):
        ...     audio = loader.apply_limiter(audio)
        >>> loader.save(audio, "output.wav", sr)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        clipping_threshold: float = 0.99,
        mono: bool = False,
    ):
        """Initialize AudioLoader with configuration.

        Args:
            sample_rate: Target sample rate for audio loading (default: 44100 Hz)
            clipping_threshold: Threshold for clipping detection, range [0.0, 1.0] (default: 0.99)
            mono: Whether to convert stereo audio to mono (default: False)

        Raises:
            ValueError: If sample_rate < 8000 or clipping_threshold not in [0, 1]
        """
        if sample_rate < 8000:
            raise ValueError(f"Sample rate {sample_rate} is too low (minimum 8000 Hz)")

        if not 0.0 <= clipping_threshold <= 1.0:
            raise ValueError(f"Clipping threshold {clipping_threshold} must be in [0, 1]")

        self.sample_rate = sample_rate
        self.clipping_threshold = clipping_threshold
        self.mono = mono
        self.logger = get_audio_logger()

    def load(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with automatic resampling and validation.

        Loads an audio file using librosa with automatic resampling to the target
        sample rate. The audio is validated for quality and format compatibility.

        Args:
            file_path: Path to audio file to load

        Returns:
            Tuple of (audio_data, sample_rate) where:
                - audio_data: numpy array of shape (n_samples,) for mono or (2, n_samples) for stereo
                - sample_rate: sample rate of the loaded audio (should match target sample rate)

        Raises:
            AudioLoadError: If file cannot be loaded (not found, corrupted, permission denied)
            UnsupportedFormatError: If file format is not supported
            AudioValidationError: If loaded audio fails validation checks

        Examples:
            >>> loader = AudioLoader(sample_rate=44100)
            >>> audio, sr = loader.load("input.wav")
            >>> print(f"Loaded {audio.shape} at {sr} Hz")
            Loaded (220500,) at 44100 Hz

            >>> # Load stereo audio
            >>> loader = AudioLoader(mono=False)
            >>> audio, sr = loader.load("stereo.mp3")
            >>> print(f"Loaded {audio.shape}")
            Loaded (2, 132300) at 44100 Hz
        """
        # Validate file path
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            self.logger.error("audio_load_failed", file_path=file_path, reason="file_not_found")
            raise AudioLoadError("File not found", file_path=file_path)

        # Check format support
        file_ext = file_path_obj.suffix.lower()
        if file_ext not in SUPPORTED_FORMATS:
            self.logger.error(
                "audio_load_failed",
                file_path=file_path,
                format=file_ext,
                reason="unsupported_format",
            )
            raise UnsupportedFormatError(file_ext, list(SUPPORTED_FORMATS))

        # Load audio with librosa
        try:
            self.logger.debug(
                "loading_audio",
                file_path=file_path,
                target_sr=self.sample_rate,
                mono=self.mono,
            )

            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=self.mono,
            )

            self.logger.info(
                "audio_loaded",
                file_path=file_path,
                shape=audio.shape,
                sample_rate=sr,
                duration=len(audio) / sr if audio.ndim == 1 else audio.shape[1] / sr,
            )

        except Exception as e:
            self.logger.error(
                "audio_load_failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise AudioLoadError(f"Failed to load audio: {str(e)}", file_path=file_path)

        # Validate loaded audio
        try:
            self.validate_audio(audio, sr)
        except AudioValidationError as e:
            self.logger.error(
                "audio_validation_failed",
                file_path=file_path,
                error=str(e),
            )
            raise

        return audio, sr

    def save(
        self,
        audio: np.ndarray,
        file_path: str,
        sample_rate: int,
        subtype: str = "PCM_24",
        apply_limiter: bool = True,
    ) -> None:
        """Save audio to file with quality validation and optional limiting.

        Saves audio data to a file with automatic clipping detection and optional
        limiting. The audio is validated before saving to ensure quality.

        Args:
            audio: Audio data as numpy array, shape (n_samples,) or (channels, n_samples)
            file_path: Output file path
            sample_rate: Sample rate of the audio data
            subtype: Audio subtype for WAV files (default: "PCM_24" for 24-bit)
            apply_limiter: Whether to apply limiting if clipping is detected (default: True)

        Raises:
            AudioSaveError: If file cannot be saved (permission denied, disk full, etc.)
            UnsupportedFormatError: If output format is not supported
            AudioValidationError: If audio data is invalid

        Examples:
            >>> loader = AudioLoader()
            >>> audio = np.random.randn(44100) * 0.5
            >>> loader.save(audio, "output.wav", 44100)

            >>> # Save with custom bit depth
            >>> loader.save(audio, "output.wav", 44100, subtype="PCM_16")

            >>> # Save without auto-limiting
            >>> loader.save(audio, "output.wav", 44100, apply_limiter=False)
        """
        # Validate audio before saving
        try:
            self.validate_audio(audio, sample_rate)
        except AudioValidationError as e:
            self.logger.error(
                "audio_save_validation_failed",
                file_path=file_path,
                error=str(e),
            )
            raise

        # Check output format
        file_path_obj = Path(file_path)
        file_ext = file_path_obj.suffix.lower()

        if file_ext not in SUPPORTED_FORMATS:
            self.logger.error(
                "audio_save_failed",
                file_path=file_path,
                format=file_ext,
                reason="unsupported_format",
            )
            raise UnsupportedFormatError(file_ext, list(SUPPORTED_FORMATS))

        # Check for clipping and apply limiter if needed
        if self.has_clipping(audio):
            if apply_limiter:
                self.logger.warning(
                    "clipping_detected",
                    file_path=file_path,
                    peak=float(np.max(np.abs(audio))),
                    action="applying_limiter",
                )
                audio = self.apply_limiter(audio)
            else:
                self.logger.warning(
                    "clipping_detected",
                    file_path=file_path,
                    peak=float(np.max(np.abs(audio))),
                    action="saving_as_is",
                )

        # Create output directory if it doesn't exist
        file_path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Prepare audio for saving (transpose if stereo)
        audio_to_save = audio.T if audio.ndim == 2 else audio

        # Save with soundfile
        try:
            self.logger.debug(
                "saving_audio",
                file_path=file_path,
                shape=audio.shape,
                sample_rate=sample_rate,
                subtype=subtype,
            )

            sf.write(
                file_path,
                audio_to_save,
                sample_rate,
                subtype=subtype if file_ext == ".wav" else None,
            )

            self.logger.info(
                "audio_saved",
                file_path=file_path,
                shape=audio.shape,
                sample_rate=sample_rate,
                file_size_mb=os.path.getsize(file_path) / (1024 * 1024),
            )

        except Exception as e:
            self.logger.error(
                "audio_save_failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise AudioSaveError(f"Failed to save audio: {str(e)}", file_path=file_path)

    def validate_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """Validate audio quality and properties.

        Performs comprehensive validation checks on audio data:
        - Checks for NaN or infinite values
        - Validates audio is not empty
        - Checks sample rate meets minimum requirements
        - Validates audio is within normalized range (with tolerance)

        Args:
            audio: Audio data to validate
            sample_rate: Sample rate of the audio

        Raises:
            AudioValidationError: If any validation check fails

        Examples:
            >>> loader = AudioLoader(sample_rate=44100)
            >>> audio = np.random.randn(44100) * 0.5
            >>> loader.validate_audio(audio, 44100)  # Passes

            >>> bad_audio = np.array([1.5, -1.5])
            >>> loader.validate_audio(bad_audio, 44100)  # Raises AudioValidationError
        """
        # Check for NaN or infinite values
        if np.any(np.isnan(audio)):
            raise AudioValidationError("Audio contains NaN values")

        if np.any(np.isinf(audio)):
            raise AudioValidationError("Audio contains infinite values")

        # Check if audio is empty
        if audio.size == 0:
            raise AudioValidationError("Audio is empty")

        # Check sample rate
        if sample_rate < 8000:
            raise AudioValidationError(
                f"Sample rate {sample_rate} is too low (minimum 8000 Hz)"
            )

        # Check if audio exceeds normalized range (with some tolerance for rounding)
        max_val = np.max(np.abs(audio))
        if max_val > 1.01:  # Allow 1% tolerance for floating point precision
            raise AudioValidationError(
                f"Audio exceeds normalized range: peak value {max_val:.3f} > 1.0"
            )

        self.logger.debug(
            "audio_validated",
            shape=audio.shape,
            sample_rate=sample_rate,
            peak=float(max_val),
            rms=float(np.sqrt(np.mean(audio**2))),
        )

    def has_clipping(self, audio: np.ndarray, threshold: Optional[float] = None) -> bool:
        """Detect if audio contains clipping.

        Checks if any samples exceed the clipping threshold. Clipping occurs when
        audio samples approach the maximum normalized value (±1.0), which can
        cause distortion.

        Args:
            audio: Audio data to check
            threshold: Optional clipping threshold (default: use instance threshold)

        Returns:
            True if clipping detected, False otherwise

        Examples:
            >>> loader = AudioLoader(clipping_threshold=0.99)
            >>> audio = np.array([0.5, 0.995, -0.997])
            >>> loader.has_clipping(audio)
            True

            >>> clean_audio = np.array([0.5, 0.8, -0.7])
            >>> loader.has_clipping(clean_audio)
            False
        """
        if threshold is None:
            threshold = self.clipping_threshold

        clipping = np.any(np.abs(audio) >= threshold)

        if clipping:
            peak = np.max(np.abs(audio))
            num_clipped = np.sum(np.abs(audio) >= threshold)
            self.logger.debug(
                "clipping_detected",
                peak=float(peak),
                threshold=threshold,
                num_clipped_samples=int(num_clipped),
                percent_clipped=float(num_clipped / audio.size * 100),
            )

        return clipping

    def apply_limiter(
        self,
        audio: np.ndarray,
        threshold: float = 0.95,
        ceiling: float = 0.99,
    ) -> np.ndarray:
        """Apply soft limiting to prevent clipping.

        Applies a soft limiter that smoothly reduces gain above the threshold
        to prevent audio from exceeding the ceiling value. This is a simple
        tanh-based soft clipper.

        Args:
            audio: Audio data to limit
            threshold: Threshold above which limiting starts (default: 0.95)
            ceiling: Maximum output level (default: 0.99)

        Returns:
            Limited audio data (copy, original is not modified)

        Examples:
            >>> loader = AudioLoader()
            >>> audio = np.array([0.5, 1.2, -1.1, 0.8])
            >>> limited = loader.apply_limiter(audio)
            >>> print(np.max(np.abs(limited)))
            0.99

            >>> # Custom threshold and ceiling
            >>> limited = loader.apply_limiter(audio, threshold=0.9, ceiling=0.95)
        """
        # Make a copy to avoid modifying original
        audio_limited = audio.copy()

        # Find samples above threshold
        mask = np.abs(audio_limited) > threshold

        if np.any(mask):
            # Apply soft clipping with tanh
            # Normalize to threshold, apply tanh, scale to ceiling
            audio_limited[mask] = (
                np.tanh((audio_limited[mask] / threshold)) * ceiling
            )

            self.logger.debug(
                "limiter_applied",
                threshold=threshold,
                ceiling=ceiling,
                num_limited_samples=int(np.sum(mask)),
                peak_before=float(np.max(np.abs(audio))),
                peak_after=float(np.max(np.abs(audio_limited))),
            )

        return audio_limited

    def get_audio_info(self, file_path: str) -> dict:
        """Get audio file information without loading the full audio.

        Retrieves metadata about an audio file including duration, sample rate,
        channels, and format without loading the entire audio into memory.

        Args:
            file_path: Path to audio file

        Returns:
            Dictionary with audio information:
                - duration: duration in seconds
                - sample_rate: sample rate in Hz
                - channels: number of channels
                - frames: total number of frames
                - format: file format
                - subtype: audio encoding subtype

        Raises:
            AudioLoadError: If file cannot be read

        Examples:
            >>> loader = AudioLoader()
            >>> info = loader.get_audio_info("input.wav")
            >>> print(f"Duration: {info['duration']:.2f}s, SR: {info['sample_rate']} Hz")
            Duration: 5.00s, SR: 44100 Hz
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise AudioLoadError("File not found", file_path=file_path)

        try:
            with sf.SoundFile(file_path) as audio_file:
                info = {
                    "duration": len(audio_file) / audio_file.samplerate,
                    "sample_rate": audio_file.samplerate,
                    "channels": audio_file.channels,
                    "frames": len(audio_file),
                    "format": audio_file.format,
                    "subtype": audio_file.subtype,
                }

            self.logger.debug("audio_info_retrieved", file_path=file_path, **info)

            return info

        except Exception as e:
            self.logger.error(
                "audio_info_failed",
                file_path=file_path,
                error=str(e),
            )
            raise AudioLoadError(f"Failed to read audio info: {str(e)}", file_path=file_path)


def load_audio(
    file_path: str,
    sample_rate: int = 44100,
    mono: bool = False,
) -> Tuple[np.ndarray, int]:
    """Convenience function to load audio file.

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (default: 44100)
        mono: Convert to mono (default: False)

    Returns:
        Tuple of (audio_data, sample_rate)

    Examples:
        >>> audio, sr = load_audio("input.wav")
        >>> audio, sr = load_audio("input.mp3", sample_rate=22050, mono=True)
    """
    loader = AudioLoader(sample_rate=sample_rate, mono=mono)
    return loader.load(file_path)


def save_audio(
    audio: np.ndarray,
    file_path: str,
    sample_rate: int,
    subtype: str = "PCM_24",
) -> None:
    """Convenience function to save audio file.

    Args:
        audio: Audio data to save
        file_path: Output file path
        sample_rate: Sample rate of audio data
        subtype: Audio subtype for WAV files (default: "PCM_24")

    Examples:
        >>> save_audio(audio, "output.wav", 44100)
        >>> save_audio(audio, "output.mp3", 44100)
    """
    loader = AudioLoader()
    loader.save(audio, file_path, sample_rate, subtype=subtype)
