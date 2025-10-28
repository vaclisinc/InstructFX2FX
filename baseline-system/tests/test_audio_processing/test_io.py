"""Unit tests for audio I/O operations.

This module tests the AudioLoader class and related functions for:
- Audio loading with various formats and configurations
- Audio saving with quality validation
- Audio validation (range, sample rate, NaN detection)
- Clipping detection and limiting
- Error handling for various failure scenarios
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.processing.io import AudioLoader, load_audio, save_audio, SUPPORTED_FORMATS
from src.processing.exceptions import (
    AudioLoadError,
    AudioSaveError,
    AudioValidationError,
    UnsupportedFormatError,
)
from src.utils.logging import configure_logging


# Configure logging for tests
configure_logging(level="INFO", format="console", console_output=True, file_output=False)


# Note: AudioLoader now uses soundfile + librosa.resample instead of librosa.load
# This avoids the Python 3.13 audioread/aifc compatibility issue
# All tests should now work on Python 3.13+


class TestAudioLoaderInit:
    """Test AudioLoader initialization and configuration."""

    def test_default_initialization(self):
        """Test AudioLoader with default parameters."""
        loader = AudioLoader()

        assert loader.sample_rate == 44100
        assert loader.clipping_threshold == 0.99
        assert loader.mono is False

    def test_custom_initialization(self):
        """Test AudioLoader with custom parameters."""
        loader = AudioLoader(
            sample_rate=48000,
            clipping_threshold=0.95,
            mono=True,
        )

        assert loader.sample_rate == 48000
        assert loader.clipping_threshold == 0.95
        assert loader.mono is True

    def test_invalid_sample_rate(self):
        """Test that sample rate below 8000 raises error."""
        with pytest.raises(ValueError, match="too low"):
            AudioLoader(sample_rate=4000)

    def test_invalid_clipping_threshold_high(self):
        """Test that clipping threshold > 1.0 raises error."""
        with pytest.raises(ValueError, match="must be in"):
            AudioLoader(clipping_threshold=1.5)

    def test_invalid_clipping_threshold_low(self):
        """Test that clipping threshold < 0.0 raises error."""
        with pytest.raises(ValueError, match="must be in"):
            AudioLoader(clipping_threshold=-0.1)


class TestAudioLoaderLoad:
    """Test audio loading functionality."""

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary WAV file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        # Create simple sine wave
        sample_rate = 44100
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * frequency * t)

        sf.write(temp_path, audio, sample_rate)

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    @pytest.fixture
    def temp_stereo_audio_file(self):
        """Create a temporary stereo WAV file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        # Create stereo sine wave
        sample_rate = 44100
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        left = 0.5 * np.sin(2 * np.pi * frequency * t)
        right = 0.5 * np.sin(2 * np.pi * frequency * 2 * t)
        audio = np.stack([left, right], axis=1)

        sf.write(temp_path, audio, sample_rate)

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_load_mono_audio(self, temp_audio_file):
        """Test loading mono audio file."""
        loader = AudioLoader(sample_rate=44100)
        audio, sr = loader.load(temp_audio_file)

        assert sr == 44100
        assert audio.ndim == 1
        assert len(audio) > 0
        assert np.max(np.abs(audio)) <= 1.0

    def test_load_stereo_audio(self, temp_stereo_audio_file):
        """Test loading stereo audio file."""
        loader = AudioLoader(sample_rate=44100, mono=False)
        audio, sr = loader.load(temp_stereo_audio_file)

        assert sr == 44100
        assert audio.ndim == 2
        assert audio.shape[0] == 2  # 2 channels
        assert np.max(np.abs(audio)) <= 1.0

    def test_load_with_resampling(self, temp_audio_file):
        """Test loading with automatic resampling."""
        loader = AudioLoader(sample_rate=22050)
        audio, sr = loader.load(temp_audio_file)

        assert sr == 22050
        assert len(audio) > 0

    def test_load_stereo_to_mono(self, temp_stereo_audio_file):
        """Test converting stereo to mono during load."""
        loader = AudioLoader(sample_rate=44100, mono=True)
        audio, sr = loader.load(temp_stereo_audio_file)

        assert sr == 44100
        assert audio.ndim == 1  # Converted to mono

    def test_load_nonexistent_file(self):
        """Test loading file that doesn't exist."""
        loader = AudioLoader()

        with pytest.raises(AudioLoadError, match="File not found"):
            loader.load("/nonexistent/path/audio.wav")

    def test_load_unsupported_format(self):
        """Test loading file with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            loader = AudioLoader()
            with pytest.raises(UnsupportedFormatError, match="Unsupported format"):
                loader.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_corrupted_file(self):
        """Test loading corrupted audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            f.write(b"not a valid wav file")

        try:
            loader = AudioLoader()
            with pytest.raises(AudioLoadError, match="Failed to load"):
                loader.load(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestAudioLoaderSave:
    """Test audio saving functionality."""

    @pytest.fixture
    def sample_audio(self):
        """Create sample audio data for testing."""
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        return 0.5 * np.sin(2 * np.pi * 440 * t)

    @pytest.fixture
    def sample_stereo_audio(self):
        """Create sample stereo audio data for testing."""
        sample_rate = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)
        return np.stack([left, right], axis=0)

    def test_save_mono_audio(self, sample_audio):
        """Test saving mono audio file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")

            loader = AudioLoader()
            loader.save(sample_audio, output_path, 44100)

            assert os.path.exists(output_path)

            # Verify saved audio
            loaded_audio, sr = sf.read(output_path)
            assert sr == 44100
            assert len(loaded_audio) == len(sample_audio)

    def test_save_stereo_audio(self, sample_stereo_audio):
        """Test saving stereo audio file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")

            loader = AudioLoader()
            loader.save(sample_stereo_audio, output_path, 44100)

            assert os.path.exists(output_path)

            # Verify saved audio
            loaded_audio, sr = sf.read(output_path)
            assert sr == 44100
            assert loaded_audio.shape[1] == 2  # 2 channels

    def test_save_with_custom_subtype(self, sample_audio):
        """Test saving with custom bit depth."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")

            loader = AudioLoader()
            loader.save(sample_audio, output_path, 44100, subtype="PCM_16")

            assert os.path.exists(output_path)

            # Verify file info
            info = sf.info(output_path)
            assert info.subtype == "PCM_16"

    def test_save_creates_directory(self, sample_audio):
        """Test that save creates output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "subdir", "output.wav")

            loader = AudioLoader()
            loader.save(sample_audio, output_path, 44100)

            assert os.path.exists(output_path)

    def test_save_with_clipping_detection(self):
        """Test save with clipping detection and limiting."""
        # Create audio with clipping (within tolerance for loading)
        audio = np.array([0.5, 0.999, -0.998, 0.8])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")

            loader = AudioLoader(clipping_threshold=0.95)
            loader.save(audio, output_path, 44100, apply_limiter=True)

            # Verify saved audio exists and is within bounds
            loaded_audio, _ = sf.read(output_path)
            assert np.max(np.abs(loaded_audio)) <= 1.0

    def test_save_without_limiter(self):
        """Test save without applying limiter on clipping audio."""
        # Create audio with slight clipping (within tolerance)
        audio = np.array([0.5, 1.0, -1.0, 0.8])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")

            loader = AudioLoader()
            loader.save(audio, output_path, 44100, apply_limiter=False)

            assert os.path.exists(output_path)

    def test_save_unsupported_format(self, sample_audio):
        """Test saving to unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.xyz")

            loader = AudioLoader()
            with pytest.raises(UnsupportedFormatError, match="Unsupported format"):
                loader.save(sample_audio, output_path, 44100)

    def test_save_invalid_audio(self):
        """Test saving invalid audio data raises validation error."""
        # Audio with values > 1.0 (beyond tolerance)
        audio = np.array([0.5, 2.0, -2.5, 0.8])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")

            loader = AudioLoader()
            with pytest.raises(AudioValidationError, match="exceeds normalized range"):
                loader.save(audio, output_path, 44100)


class TestAudioValidation:
    """Test audio validation functionality."""

    def test_validate_valid_audio(self):
        """Test validation passes for valid audio."""
        loader = AudioLoader()
        audio = np.array([0.5, -0.3, 0.8, -0.9])

        # Should not raise
        loader.validate_audio(audio, 44100)

    def test_validate_nan_values(self):
        """Test validation fails for NaN values."""
        loader = AudioLoader()
        audio = np.array([0.5, np.nan, 0.8])

        with pytest.raises(AudioValidationError, match="NaN"):
            loader.validate_audio(audio, 44100)

    def test_validate_infinite_values(self):
        """Test validation fails for infinite values."""
        loader = AudioLoader()
        audio = np.array([0.5, np.inf, 0.8])

        with pytest.raises(AudioValidationError, match="infinite"):
            loader.validate_audio(audio, 44100)

    def test_validate_empty_audio(self):
        """Test validation fails for empty audio."""
        loader = AudioLoader()
        audio = np.array([])

        with pytest.raises(AudioValidationError, match="empty"):
            loader.validate_audio(audio, 44100)

    def test_validate_low_sample_rate(self):
        """Test validation fails for low sample rate."""
        loader = AudioLoader()
        audio = np.array([0.5, -0.3, 0.8])

        with pytest.raises(AudioValidationError, match="too low"):
            loader.validate_audio(audio, 4000)

    def test_validate_exceeds_range(self):
        """Test validation fails for audio exceeding normalized range."""
        loader = AudioLoader()
        audio = np.array([0.5, 1.5, -1.5])

        with pytest.raises(AudioValidationError, match="exceeds normalized range"):
            loader.validate_audio(audio, 44100)

    def test_validate_allows_tolerance(self):
        """Test validation allows small tolerance for floating point precision."""
        loader = AudioLoader()
        audio = np.array([0.5, 1.005, -1.005])  # Within 1% tolerance

        # Should not raise
        loader.validate_audio(audio, 44100)


class TestClippingDetection:
    """Test clipping detection and limiting."""

    def test_has_clipping_positive(self):
        """Test clipping detection with clipping present."""
        loader = AudioLoader(clipping_threshold=0.99)
        # Use values at or above threshold
        audio = np.array([0.5, 0.99, -0.995, 0.8])

        assert loader.has_clipping(audio) == True

    def test_has_clipping_negative(self):
        """Test clipping detection with no clipping."""
        loader = AudioLoader(clipping_threshold=0.99)
        # Use values clearly below threshold
        audio = np.array([0.5, 0.8, -0.7, 0.6])

        assert loader.has_clipping(audio) == False

    def test_has_clipping_custom_threshold(self):
        """Test clipping detection with custom threshold."""
        loader = AudioLoader(clipping_threshold=0.99)
        # Use values between 0.95 and 0.99
        audio = np.array([0.5, 0.96, -0.97, 0.8])

        # Should not clip with default threshold (0.99)
        assert loader.has_clipping(audio) == False

        # Should clip with lower threshold (0.95)
        assert loader.has_clipping(audio, threshold=0.95) == True

    def test_apply_limiter_reduces_peaks(self):
        """Test limiter reduces peak values."""
        loader = AudioLoader()
        audio = np.array([0.5, 1.2, -1.1, 0.8])

        limited = loader.apply_limiter(audio, threshold=0.95, ceiling=0.99)

        assert np.max(np.abs(limited)) <= 0.99
        # Original audio should not be modified
        assert audio[1] == 1.2

    def test_apply_limiter_preserves_below_threshold(self):
        """Test limiter preserves values below threshold."""
        loader = AudioLoader()
        audio = np.array([0.5, 0.3, -0.4, 0.2])

        limited = loader.apply_limiter(audio, threshold=0.95)

        # Audio below threshold should be unchanged
        np.testing.assert_array_almost_equal(audio, limited)

    def test_apply_limiter_soft_knee(self):
        """Test limiter applies soft knee compression."""
        loader = AudioLoader()
        audio = np.array([0.5, 1.0, -1.0, 0.8])

        limited = loader.apply_limiter(audio, threshold=0.9, ceiling=0.95)

        # Should be limited but not hard-clipped
        assert np.max(np.abs(limited)) <= 0.95
        # Limiter should be gradual (tanh-based)
        assert limited[1] < audio[1]


class TestAudioInfo:
    """Test audio file information retrieval."""

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary WAV file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        # Create simple sine wave
        sample_rate = 44100
        duration = 2.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        sf.write(temp_path, audio, sample_rate, subtype="PCM_24")

        yield temp_path

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_get_audio_info(self, temp_audio_file):
        """Test retrieving audio file information."""
        loader = AudioLoader()
        info = loader.get_audio_info(temp_audio_file)

        assert "duration" in info
        assert "sample_rate" in info
        assert "channels" in info
        assert "frames" in info
        assert "format" in info
        assert "subtype" in info

        assert info["sample_rate"] == 44100
        assert info["channels"] == 1
        assert info["subtype"] == "PCM_24"
        assert 2.4 < info["duration"] < 2.6  # ~2.5 seconds

    def test_get_audio_info_nonexistent(self):
        """Test get_audio_info with nonexistent file."""
        loader = AudioLoader()

        with pytest.raises(AudioLoadError, match="File not found"):
            loader.get_audio_info("/nonexistent/file.wav")


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary WAV file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name

        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        sf.write(temp_path, audio, sample_rate)

        yield temp_path

        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_load_audio_convenience(self, temp_audio_file):
        """Test load_audio convenience function."""
        audio, sr = load_audio(temp_audio_file, sample_rate=44100)

        assert sr == 44100
        assert len(audio) > 0

    def test_load_audio_mono_conversion(self, temp_audio_file):
        """Test load_audio with mono conversion."""
        audio, sr = load_audio(temp_audio_file, sample_rate=22050, mono=True)

        assert sr == 22050
        assert audio.ndim == 1

    def test_save_audio_convenience(self):
        """Test save_audio convenience function."""
        # Use deterministic audio within range
        t = np.linspace(0, 1, 44100)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")
            save_audio(audio, output_path, 44100)

            assert os.path.exists(output_path)

    def test_save_audio_custom_subtype(self):
        """Test save_audio with custom subtype."""
        # Use deterministic audio within range
        t = np.linspace(0, 1, 44100)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "output.wav")
            save_audio(audio, output_path, 44100, subtype="PCM_16")

            info = sf.info(output_path)
            assert info.subtype == "PCM_16"


class TestSupportedFormats:
    """Test supported format constants."""

    def test_supported_formats_constant(self):
        """Test SUPPORTED_FORMATS constant is defined."""
        assert isinstance(SUPPORTED_FORMATS, set)
        assert len(SUPPORTED_FORMATS) > 0

    def test_supported_formats_includes_common(self):
        """Test SUPPORTED_FORMATS includes common formats."""
        assert ".wav" in SUPPORTED_FORMATS
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".flac" in SUPPORTED_FORMATS
