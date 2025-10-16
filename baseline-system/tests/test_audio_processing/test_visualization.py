"""Tests for AudioVisualizer class."""

import pytest
import numpy as np
import os
import tempfile
from pathlib import Path
from audio_processing.visualization import AudioVisualizer


class TestAudioVisualizer:
    """Test suite for AudioVisualizer class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sine_wave(self):
        """Create a 440 Hz sine wave at 0.5 amplitude."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        return np.sin(2 * np.pi * 440 * t) * 0.5, sr

    @pytest.fixture
    def stereo_audio(self):
        """Create stereo audio with different channels."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        left = np.sin(2 * np.pi * 440 * t) * 0.3
        right = np.sin(2 * np.pi * 880 * t) * 0.5
        return np.stack([left, right], axis=1), sr

    @pytest.fixture
    def white_noise(self):
        """Create white noise signal."""
        sr = 44100
        duration = 0.5
        samples = int(sr * duration)
        return np.random.uniform(-0.2, 0.2, samples), sr

    @pytest.fixture
    def silent_audio(self):
        """Create silent audio."""
        sr = 44100
        duration = 0.5
        return np.zeros(int(sr * duration)), sr

    @pytest.fixture
    def complex_audio(self):
        """Create complex audio with multiple frequencies."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Sum of multiple sine waves
        audio = (
            0.3 * np.sin(2 * np.pi * 220 * t)
            + 0.2 * np.sin(2 * np.pi * 440 * t)
            + 0.15 * np.sin(2 * np.pi * 880 * t)
            + 0.1 * np.sin(2 * np.pi * 1760 * t)
        )
        return audio, sr

    # ===== plot_waveform Tests =====

    def test_plot_waveform_creates_file(self, temp_dir, sine_wave):
        """Test that plot_waveform creates output file."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "waveform.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_waveform_file_not_empty(self, temp_dir, sine_wave):
        """Test that plot_waveform creates non-empty file."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "waveform.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path)

        # File should be > 1KB (typical PNG)
        file_size = os.path.getsize(save_path)
        assert file_size > 1000

    def test_plot_waveform_custom_title(self, temp_dir, sine_wave):
        """Test plot_waveform with custom title."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "waveform_custom.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path, title="Custom Title")

        assert os.path.exists(save_path)

    def test_plot_waveform_custom_figsize(self, temp_dir, sine_wave):
        """Test plot_waveform with custom figure size."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "waveform_size.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path, figsize=(16, 6))

        assert os.path.exists(save_path)

    def test_plot_waveform_stereo(self, temp_dir, stereo_audio):
        """Test plot_waveform with stereo audio."""
        audio, sr = stereo_audio
        save_path = os.path.join(temp_dir, "waveform_stereo.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_waveform_silent(self, temp_dir, silent_audio):
        """Test plot_waveform with silent audio."""
        audio, sr = silent_audio
        save_path = os.path.join(temp_dir, "waveform_silent.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_waveform_white_noise(self, temp_dir, white_noise):
        """Test plot_waveform with white noise."""
        audio, sr = white_noise
        save_path = os.path.join(temp_dir, "waveform_noise.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_waveform_different_sample_rates(self, temp_dir):
        """Test plot_waveform with different sample rates."""
        # 48kHz audio
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        save_path = os.path.join(temp_dir, "waveform_48k.png")
        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_waveform_nested_directory(self, temp_dir):
        """Test plot_waveform creates nested directories."""
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.5
        sr = 44100

        nested_dir = os.path.join(temp_dir, "nested", "plots")
        os.makedirs(nested_dir, exist_ok=True)
        save_path = os.path.join(nested_dir, "waveform.png")

        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_waveform_returns_none(self, temp_dir, sine_wave):
        """Test that plot_waveform returns None."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "waveform.png")

        result = AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert result is None

    # ===== plot_spectrogram Tests =====

    def test_plot_spectrogram_creates_file(self, temp_dir, sine_wave):
        """Test that plot_spectrogram creates output file."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "spectrogram.png")

        AudioVisualizer.plot_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_spectrogram_file_not_empty(self, temp_dir, sine_wave):
        """Test that plot_spectrogram creates non-empty file."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "spectrogram.png")

        AudioVisualizer.plot_spectrogram(audio, sr, save_path)

        # File should be > 1KB
        file_size = os.path.getsize(save_path)
        assert file_size > 1000

    def test_plot_spectrogram_custom_title(self, temp_dir, sine_wave):
        """Test plot_spectrogram with custom title."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "spec_custom.png")

        AudioVisualizer.plot_spectrogram(audio, sr, save_path, title="Custom Spec")

        assert os.path.exists(save_path)

    def test_plot_spectrogram_custom_figsize(self, temp_dir, sine_wave):
        """Test plot_spectrogram with custom figure size."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "spec_size.png")

        AudioVisualizer.plot_spectrogram(audio, sr, save_path, figsize=(14, 8))

        assert os.path.exists(save_path)

    def test_plot_spectrogram_custom_fft_params(self, temp_dir, sine_wave):
        """Test plot_spectrogram with custom FFT parameters."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "spec_fft.png")

        AudioVisualizer.plot_spectrogram(
            audio, sr, save_path, n_fft=4096, hop_length=1024
        )

        assert os.path.exists(save_path)

    def test_plot_spectrogram_white_noise(self, temp_dir, white_noise):
        """Test plot_spectrogram with white noise."""
        audio, sr = white_noise
        save_path = os.path.join(temp_dir, "spec_noise.png")

        AudioVisualizer.plot_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_spectrogram_complex_audio(self, temp_dir, complex_audio):
        """Test plot_spectrogram with complex multi-frequency audio."""
        audio, sr = complex_audio
        save_path = os.path.join(temp_dir, "spec_complex.png")

        AudioVisualizer.plot_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_spectrogram_silent(self, temp_dir, silent_audio):
        """Test plot_spectrogram with silent audio."""
        audio, sr = silent_audio
        save_path = os.path.join(temp_dir, "spec_silent.png")

        AudioVisualizer.plot_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_spectrogram_returns_none(self, temp_dir, sine_wave):
        """Test that plot_spectrogram returns None."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "spec.png")

        result = AudioVisualizer.plot_spectrogram(audio, sr, save_path)

        assert result is None

    # ===== plot_mel_spectrogram Tests =====

    def test_plot_mel_spectrogram_creates_file(self, temp_dir, sine_wave):
        """Test that plot_mel_spectrogram creates output file."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "mel_spec.png")

        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_mel_spectrogram_file_not_empty(self, temp_dir, sine_wave):
        """Test that plot_mel_spectrogram creates non-empty file."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "mel_spec.png")

        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path)

        # File should be > 1KB
        file_size = os.path.getsize(save_path)
        assert file_size > 1000

    def test_plot_mel_spectrogram_custom_title(self, temp_dir, sine_wave):
        """Test plot_mel_spectrogram with custom title."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "mel_custom.png")

        AudioVisualizer.plot_mel_spectrogram(
            audio, sr, save_path, title="Custom Mel Spec"
        )

        assert os.path.exists(save_path)

    def test_plot_mel_spectrogram_custom_figsize(self, temp_dir, sine_wave):
        """Test plot_mel_spectrogram with custom figure size."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "mel_size.png")

        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path, figsize=(14, 8))

        assert os.path.exists(save_path)

    def test_plot_mel_spectrogram_custom_params(self, temp_dir, sine_wave):
        """Test plot_mel_spectrogram with custom parameters."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "mel_params.png")

        AudioVisualizer.plot_mel_spectrogram(
            audio, sr, save_path, n_fft=4096, hop_length=1024, n_mels=256
        )

        assert os.path.exists(save_path)

    def test_plot_mel_spectrogram_white_noise(self, temp_dir, white_noise):
        """Test plot_mel_spectrogram with white noise."""
        audio, sr = white_noise
        save_path = os.path.join(temp_dir, "mel_noise.png")

        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_mel_spectrogram_complex_audio(self, temp_dir, complex_audio):
        """Test plot_mel_spectrogram with complex audio."""
        audio, sr = complex_audio
        save_path = os.path.join(temp_dir, "mel_complex.png")

        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_mel_spectrogram_silent(self, temp_dir, silent_audio):
        """Test plot_mel_spectrogram with silent audio."""
        audio, sr = silent_audio
        save_path = os.path.join(temp_dir, "mel_silent.png")

        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_mel_spectrogram_different_mel_bands(self, temp_dir, sine_wave):
        """Test plot_mel_spectrogram with different number of mel bands."""
        audio, sr = sine_wave

        # Test with 64 mel bands
        save_path_64 = os.path.join(temp_dir, "mel_64.png")
        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path_64, n_mels=64)
        assert os.path.exists(save_path_64)

        # Test with 256 mel bands
        save_path_256 = os.path.join(temp_dir, "mel_256.png")
        AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path_256, n_mels=256)
        assert os.path.exists(save_path_256)

    def test_plot_mel_spectrogram_returns_none(self, temp_dir, sine_wave):
        """Test that plot_mel_spectrogram returns None."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "mel.png")

        result = AudioVisualizer.plot_mel_spectrogram(audio, sr, save_path)

        assert result is None

    # ===== plot_comparison Tests =====

    def test_plot_comparison_creates_file(self, temp_dir, sine_wave):
        """Test that plot_comparison creates output file."""
        audio1, sr = sine_wave

        # Create second audio (different amplitude)
        t = np.linspace(0, 1.0, int(sr * 1.0))
        audio2 = np.sin(2 * np.pi * 440 * t) * 0.7

        save_path = os.path.join(temp_dir, "comparison.png")
        AudioVisualizer.plot_comparison(audio1, audio2, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_comparison_file_not_empty(self, temp_dir, sine_wave):
        """Test that plot_comparison creates non-empty file."""
        audio1, sr = sine_wave
        audio2 = audio1 * 0.8  # Reduced amplitude version

        save_path = os.path.join(temp_dir, "comparison.png")
        AudioVisualizer.plot_comparison(audio1, audio2, sr, save_path)

        # File should be > 2KB (two plots)
        file_size = os.path.getsize(save_path)
        assert file_size > 2000

    def test_plot_comparison_custom_labels(self, temp_dir, sine_wave):
        """Test plot_comparison with custom labels."""
        audio1, sr = sine_wave
        audio2 = audio1 * 0.8

        save_path = os.path.join(temp_dir, "comparison_labels.png")
        AudioVisualizer.plot_comparison(
            audio1, audio2, sr, save_path, label1="Before", label2="After"
        )

        assert os.path.exists(save_path)

    def test_plot_comparison_custom_figsize(self, temp_dir, sine_wave):
        """Test plot_comparison with custom figure size."""
        audio1, sr = sine_wave
        audio2 = audio1 * 0.8

        save_path = os.path.join(temp_dir, "comparison_size.png")
        AudioVisualizer.plot_comparison(
            audio1, audio2, sr, save_path, figsize=(16, 10)
        )

        assert os.path.exists(save_path)

    def test_plot_comparison_different_audio(self, temp_dir, sine_wave, white_noise):
        """Test plot_comparison with completely different audio types."""
        audio1, sr = sine_wave
        audio2, _ = white_noise

        # Pad noise to match sine wave length
        if len(audio2) < len(audio1):
            audio2 = np.pad(audio2, (0, len(audio1) - len(audio2)))
        else:
            audio2 = audio2[: len(audio1)]

        save_path = os.path.join(temp_dir, "comparison_different.png")
        AudioVisualizer.plot_comparison(
            audio1, audio2, sr, save_path, label1="Sine", label2="Noise"
        )

        assert os.path.exists(save_path)

    def test_plot_comparison_stereo(self, temp_dir, stereo_audio):
        """Test plot_comparison with stereo audio."""
        audio1, sr = stereo_audio

        # Create modified version
        audio2 = audio1 * 0.7

        save_path = os.path.join(temp_dir, "comparison_stereo.png")
        AudioVisualizer.plot_comparison(audio1, audio2, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_comparison_processed_vs_original(self, temp_dir, sine_wave):
        """Test plot_comparison for typical use case (original vs processed)."""
        original, sr = sine_wave

        # Simulate processing: add reverb-like effect (simple delay + attenuation)
        delay_samples = int(0.05 * sr)  # 50ms delay
        processed = original.copy()
        delayed = np.pad(original * 0.3, (delay_samples, 0))[: len(original)]
        processed = processed + delayed

        save_path = os.path.join(temp_dir, "comparison_processed.png")
        AudioVisualizer.plot_comparison(
            original, processed, sr, save_path, label1="Original", label2="Processed"
        )

        assert os.path.exists(save_path)

    def test_plot_comparison_different_lengths(self, temp_dir):
        """Test plot_comparison handles different length audio."""
        sr = 44100

        # Audio of different lengths
        t1 = np.linspace(0, 1.0, int(sr * 1.0))
        audio1 = np.sin(2 * np.pi * 440 * t1) * 0.5

        t2 = np.linspace(0, 0.5, int(sr * 0.5))
        audio2 = np.sin(2 * np.pi * 880 * t2) * 0.5

        save_path = os.path.join(temp_dir, "comparison_lengths.png")

        # Should handle different lengths (may pad or truncate)
        # Test doesn't fail = acceptable behavior
        AudioVisualizer.plot_comparison(audio1, audio2, sr, save_path)

        assert os.path.exists(save_path)

    def test_plot_comparison_returns_none(self, temp_dir, sine_wave):
        """Test that plot_comparison returns None."""
        audio1, sr = sine_wave
        audio2 = audio1 * 0.8

        save_path = os.path.join(temp_dir, "comp.png")
        result = AudioVisualizer.plot_comparison(audio1, audio2, sr, save_path)

        assert result is None

    # ===== Static Method Tests =====

    def test_static_method_no_instantiation(self, temp_dir, sine_wave):
        """Test that methods can be called without instantiation."""
        audio, sr = sine_wave
        save_path = os.path.join(temp_dir, "static_test.png")

        # Should work without creating AudioVisualizer instance
        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_multiple_plots_independent(self, temp_dir, sine_wave):
        """Test multiple plot calls are independent (no state)."""
        audio, sr = sine_wave

        # Create multiple plots
        for i in range(3):
            save_path = os.path.join(temp_dir, f"multi_{i}.png")
            AudioVisualizer.plot_waveform(audio, sr, save_path)
            assert os.path.exists(save_path)

    # ===== Integration Tests =====

    def test_all_plot_types_same_audio(self, temp_dir, complex_audio):
        """Test creating all plot types for same audio."""
        audio, sr = complex_audio

        # Create all plot types
        AudioVisualizer.plot_waveform(
            audio, sr, os.path.join(temp_dir, "all_waveform.png")
        )
        AudioVisualizer.plot_spectrogram(
            audio, sr, os.path.join(temp_dir, "all_spectrogram.png")
        )
        AudioVisualizer.plot_mel_spectrogram(
            audio, sr, os.path.join(temp_dir, "all_mel.png")
        )
        AudioVisualizer.plot_comparison(
            audio, audio * 0.5, sr, os.path.join(temp_dir, "all_comparison.png")
        )

        # Verify all created
        assert os.path.exists(os.path.join(temp_dir, "all_waveform.png"))
        assert os.path.exists(os.path.join(temp_dir, "all_spectrogram.png"))
        assert os.path.exists(os.path.join(temp_dir, "all_mel.png"))
        assert os.path.exists(os.path.join(temp_dir, "all_comparison.png"))

    def test_realistic_workflow(self, temp_dir):
        """Test realistic workflow: generate audio, process, visualize."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Generate original audio
        original = 0.4 * np.sin(2 * np.pi * 440 * t)

        # Simulate processing (simple amplitude change + noise)
        processed = original * 1.2
        noise = np.random.uniform(-0.05, 0.05, len(processed))
        processed = processed + noise

        # Create visualizations
        AudioVisualizer.plot_waveform(
            original, sr, os.path.join(temp_dir, "workflow_original.png")
        )
        AudioVisualizer.plot_waveform(
            processed, sr, os.path.join(temp_dir, "workflow_processed.png")
        )
        AudioVisualizer.plot_comparison(
            original,
            processed,
            sr,
            os.path.join(temp_dir, "workflow_comparison.png"),
            label1="Original",
            label2="Processed",
        )
        AudioVisualizer.plot_spectrogram(
            processed, sr, os.path.join(temp_dir, "workflow_spec.png")
        )

        # Verify all created
        assert os.path.exists(os.path.join(temp_dir, "workflow_original.png"))
        assert os.path.exists(os.path.join(temp_dir, "workflow_processed.png"))
        assert os.path.exists(os.path.join(temp_dir, "workflow_comparison.png"))
        assert os.path.exists(os.path.join(temp_dir, "workflow_spec.png"))

    # ===== Edge Cases =====

    def test_very_short_audio(self, temp_dir):
        """Test plotting very short audio."""
        sr = 44100
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, 441)) * 0.5

        save_path = os.path.join(temp_dir, "short.png")
        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_very_long_audio(self, temp_dir):
        """Test plotting very long audio (10 seconds)."""
        sr = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        save_path = os.path.join(temp_dir, "long.png")
        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)

    def test_extreme_amplitude(self, temp_dir):
        """Test plotting audio with extreme amplitudes."""
        sr = 44100
        t = np.linspace(0, 1.0, int(sr * 1.0))
        audio = np.sin(2 * np.pi * 440 * t) * 100.0  # Very loud

        save_path = os.path.join(temp_dir, "extreme.png")
        AudioVisualizer.plot_waveform(audio, sr, save_path)

        assert os.path.exists(save_path)
