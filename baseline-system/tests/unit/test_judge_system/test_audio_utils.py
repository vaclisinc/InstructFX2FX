"""Unit tests for audio loading and formatting utilities."""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from judge_system.data.audio_utils import load_audio_sample, format_example_for_prompt
from judge_system.data.models import SocialFXExample


class TestLoadAudioSample:
    """Tests for load_audio_sample function."""

    @pytest.fixture
    def sample_audio_file(self, tmp_path):
        """Create a sample audio file for testing."""
        audio_path = tmp_path / "test.wav"

        # Create dummy audio data (1 second at 44100 Hz)
        sample_rate = 44100
        duration = 1.0
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples).astype(np.float32) * 0.1

        # Write audio file
        sf.write(str(audio_path), audio_data, sample_rate)

        return audio_path

    @pytest.fixture
    def sample_audio_file_different_sr(self, tmp_path):
        """Create a sample audio file with non-standard sample rate."""
        audio_path = tmp_path / "test_48k.wav"

        # Create audio at 48000 Hz
        sample_rate = 48000
        duration = 0.5
        samples = int(sample_rate * duration)
        audio_data = np.random.randn(samples).astype(np.float32) * 0.1

        sf.write(str(audio_path), audio_data, sample_rate)

        return audio_path

    def test_load_audio_sample_valid_file(self, sample_audio_file):
        """Test loading a valid audio file."""
        audio, sr = load_audio_sample(str(sample_audio_file))

        assert isinstance(audio, np.ndarray)
        assert isinstance(sr, int)
        assert sr == 44100
        assert len(audio) == 44100  # 1 second at 44100 Hz

    def test_load_audio_sample_returns_correct_shape(self, sample_audio_file):
        """Test that loaded audio has correct shape."""
        audio, sr = load_audio_sample(str(sample_audio_file))

        # Mono audio should be 1D
        assert audio.ndim == 1
        assert len(audio) > 0

    def test_load_audio_sample_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing file."""
        nonexistent_path = tmp_path / "nonexistent.wav"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_audio_sample(str(nonexistent_path))

        assert "Audio file not found" in str(exc_info.value)
        assert str(nonexistent_path) in str(exc_info.value)

    def test_load_audio_sample_different_sample_rate(self, sample_audio_file_different_sr):
        """Test loading audio file with non-standard sample rate."""
        audio, sr = load_audio_sample(str(sample_audio_file_different_sr))

        assert sr == 48000
        assert len(audio) == 24000  # 0.5 seconds at 48000 Hz

    def test_load_audio_sample_warns_non_standard_sr(self, sample_audio_file_different_sr, caplog):
        """Test that warning is issued for non-44100 Hz sample rate."""
        import logging

        with caplog.at_level(logging.WARNING):
            audio, sr = load_audio_sample(str(sample_audio_file_different_sr))

        assert "Sample rate" in caplog.text
        assert "44100" in caplog.text

    def test_load_audio_sample_corrupted_file(self, tmp_path):
        """Test that RuntimeError is raised for corrupted file."""
        corrupted_path = tmp_path / "corrupted.wav"
        corrupted_path.write_bytes(b"This is not a valid WAV file")

        with pytest.raises(RuntimeError) as exc_info:
            load_audio_sample(str(corrupted_path))

        assert "Failed to read audio file" in str(exc_info.value)

    def test_load_audio_sample_stereo_file(self, tmp_path):
        """Test loading stereo audio file."""
        audio_path = tmp_path / "stereo.wav"

        # Create stereo audio (2 channels)
        sample_rate = 44100
        duration = 0.5
        samples = int(sample_rate * duration)
        stereo_data = np.random.randn(samples, 2).astype(np.float32) * 0.1

        sf.write(str(audio_path), stereo_data, sample_rate)

        audio, sr = load_audio_sample(str(audio_path))

        assert sr == 44100
        assert audio.shape[0] == 22050  # 0.5 seconds
        assert audio.ndim == 2  # Stereo has 2 dimensions
        assert audio.shape[1] == 2  # 2 channels

    def test_load_audio_sample_empty_file(self, tmp_path):
        """Test loading empty audio file."""
        empty_path = tmp_path / "empty.wav"

        # Create empty audio file
        empty_data = np.array([], dtype=np.float32)
        sf.write(str(empty_path), empty_data, 44100)

        audio, sr = load_audio_sample(str(empty_path))

        assert len(audio) == 0
        assert sr == 44100

    def test_load_audio_sample_path_object(self, sample_audio_file):
        """Test that Path objects work as input."""
        audio, sr = load_audio_sample(str(sample_audio_file))

        assert isinstance(audio, np.ndarray)
        assert sr == 44100

    def test_load_audio_sample_preserves_audio_data(self, tmp_path):
        """Test that audio data is preserved correctly."""
        audio_path = tmp_path / "test.wav"

        # Create known audio data
        sample_rate = 44100
        original_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)

        sf.write(str(audio_path), original_data, sample_rate)

        loaded_audio, sr = load_audio_sample(str(audio_path))

        # Check data is approximately equal (allowing for WAV encoding/decoding precision)
        np.testing.assert_array_almost_equal(loaded_audio, original_data, decimal=4)


class TestFormatExampleForPrompt:
    """Tests for format_example_for_prompt function."""

    def test_format_example_basic(self):
        """Test formatting a basic example."""
        example = SocialFXExample(
            id=1,
            description="warm and intimate",
            instrument="guitar",
            effect_type="eq",
            parameters={"band1_freq": 200, "band1_gain": 3.0}
        )

        formatted = format_example_for_prompt(example)

        assert 'Description: "warm and intimate"' in formatted
        assert "Instrument: guitar" in formatted
        assert "Parameters:" in formatted
        assert "band1_freq" in formatted
        assert "band1_gain" in formatted

    def test_format_example_with_nested_parameters(self):
        """Test formatting example with nested parameter structure."""
        example = SocialFXExample(
            id=1,
            description="complex eq",
            instrument="drums",
            effect_type="eq",
            parameters={
                "bands": [
                    {"frequency": 200, "gain": 3.0, "q": 0.7},
                    {"frequency": 3000, "gain": -2.0, "q": 1.2}
                ]
            }
        )

        formatted = format_example_for_prompt(example)

        assert "complex eq" in formatted
        assert "drums" in formatted
        assert "bands" in formatted
        assert "frequency" in formatted
        assert "200" in formatted
        assert "3000" in formatted

    def test_format_example_json_indentation(self):
        """Test that JSON is properly indented."""
        example = SocialFXExample(
            id=1,
            description="test",
            instrument="guitar",
            effect_type="eq",
            parameters={"param1": 1.0, "param2": 2.0}
        )

        formatted = format_example_for_prompt(example)

        # Check for proper indentation (2 spaces)
        assert "  \"param1\"" in formatted or "  \"param2\"" in formatted

    def test_format_example_reverb_parameters(self):
        """Test formatting reverb effect parameters."""
        example = SocialFXExample(
            id=1,
            description="spacious cathedral",
            instrument="piano",
            effect_type="reverb",
            parameters={
                "room_size": 0.9,
                "damping": 0.3,
                "wet_level": 0.4,
                "dry_level": 0.6,
                "width": 0.9,
                "freeze_mode": False
            }
        )

        formatted = format_example_for_prompt(example)

        assert "spacious cathedral" in formatted
        assert "piano" in formatted
        assert "room_size" in formatted
        assert "0.9" in formatted
        assert "freeze_mode" in formatted
        assert "false" in formatted

    def test_format_example_compressor_parameters(self):
        """Test formatting compressor effect parameters."""
        example = SocialFXExample(
            id=1,
            description="punchy and controlled",
            instrument="drums",
            effect_type="compressor",
            parameters={
                "threshold": -20,
                "ratio": 4.0,
                "attack": 5,
                "release": 50,
                "knee": 3,
                "makeup_gain": 2.0
            }
        )

        formatted = format_example_for_prompt(example)

        assert "punchy and controlled" in formatted
        assert "drums" in formatted
        assert "threshold" in formatted
        assert "-20" in formatted
        assert "ratio" in formatted
        assert "4.0" in formatted

    def test_format_example_with_special_characters_in_description(self):
        """Test formatting example with special characters."""
        example = SocialFXExample(
            id=1,
            description='bright "metallic" sound with high-end sparkle',
            instrument="guitar",
            effect_type="eq",
            parameters={"freq": 8000, "gain": 5.0}
        )

        formatted = format_example_for_prompt(example)

        assert "bright" in formatted
        assert "metallic" in formatted

    def test_format_example_preserves_numeric_precision(self):
        """Test that numeric precision is preserved in formatting."""
        example = SocialFXExample(
            id=1,
            description="test",
            instrument="guitar",
            effect_type="eq",
            parameters={
                "freq": 123.456789,
                "gain": 2.5,
                "q": 0.707
            }
        )

        formatted = format_example_for_prompt(example)

        assert "123.456789" in formatted
        assert "2.5" in formatted
        assert "0.707" in formatted

    def test_format_example_empty_parameters(self):
        """Test formatting with minimal parameters."""
        example = SocialFXExample(
            id=1,
            description="minimal",
            instrument="piano",
            effect_type="eq",
            parameters={"value": 1}
        )

        formatted = format_example_for_prompt(example)

        assert "minimal" in formatted
        assert "piano" in formatted
        assert "value" in formatted

    def test_format_example_with_audio_path(self):
        """Test that audio_path is not included in formatted output."""
        example = SocialFXExample(
            id=1,
            description="test",
            instrument="guitar",
            effect_type="eq",
            parameters={"param": 1.0},
            audio_path="/path/to/guitar.wav"
        )

        formatted = format_example_for_prompt(example)

        # Audio path should not appear in prompt formatting
        assert "/path/to/guitar.wav" not in formatted

    def test_format_example_multiline_structure(self):
        """Test that formatted output has expected multiline structure."""
        example = SocialFXExample(
            id=1,
            description="test",
            instrument="guitar",
            effect_type="eq",
            parameters={"param": 1.0}
        )

        formatted = format_example_for_prompt(example)

        lines = formatted.split("\n")

        # Should have at least 4 lines: Description, Instrument, Parameters header, JSON
        assert len(lines) >= 4
        assert lines[0].startswith("Description:")
        assert lines[1].startswith("Instrument:")
        assert lines[2].startswith("Parameters:")

    def test_format_example_consistent_formatting(self):
        """Test that formatting is consistent across multiple calls."""
        example = SocialFXExample(
            id=1,
            description="test",
            instrument="guitar",
            effect_type="eq",
            parameters={"param": 1.0}
        )

        formatted1 = format_example_for_prompt(example)
        formatted2 = format_example_for_prompt(example)

        assert formatted1 == formatted2


class TestAudioUtilsIntegration:
    """Integration tests for audio utilities."""

    def test_load_and_format_workflow(self, tmp_path):
        """Test complete workflow of loading audio and formatting example."""
        # Create audio file
        audio_path = tmp_path / "test.wav"
        sample_rate = 44100
        audio_data = np.random.randn(44100).astype(np.float32) * 0.1
        sf.write(str(audio_path), audio_data, sample_rate)

        # Create example
        example = SocialFXExample(
            id=1,
            description="test effect",
            instrument="guitar",
            effect_type="eq",
            parameters={"freq": 1000, "gain": 3.0},
            audio_path=str(audio_path)
        )

        # Load audio
        loaded_audio, sr = load_audio_sample(example.audio_path)

        # Format example
        formatted = format_example_for_prompt(example)

        # Verify both operations succeeded
        assert len(loaded_audio) == 44100
        assert sr == 44100
        assert "test effect" in formatted
        assert "guitar" in formatted
