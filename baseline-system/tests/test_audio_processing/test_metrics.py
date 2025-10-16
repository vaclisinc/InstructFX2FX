"""Tests for AudioMetrics class."""

import pytest
import numpy as np
from audio_processing.metrics import AudioMetrics


class TestAudioMetrics:
    """Test suite for AudioMetrics class."""

    @pytest.fixture
    def silent_audio(self):
        """Create silent audio (all zeros)."""
        return np.zeros(44100)  # 1 second of silence

    @pytest.fixture
    def sine_wave(self):
        """Create a 440 Hz sine wave at 0.5 amplitude."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        return np.sin(2 * np.pi * 440 * t) * 0.5, sr

    @pytest.fixture
    def full_scale_sine(self):
        """Create a sine wave at maximum amplitude (1.0)."""
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        return np.sin(2 * np.pi * 440 * t) * 1.0, sr

    @pytest.fixture
    def clipping_audio(self):
        """Create audio with clipping (values > 1.0)."""
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        # Sine wave at 1.5 amplitude (clipping)
        return np.sin(2 * np.pi * 440 * t) * 1.5, sr

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
        duration = 1.0
        samples = int(sr * duration)
        # White noise with amplitude 0.2
        return np.random.uniform(-0.2, 0.2, samples), sr

    # ===== compute_rms Tests =====

    def test_compute_rms_silent(self, silent_audio):
        """Test RMS of silent audio is zero."""
        rms = AudioMetrics.compute_rms(silent_audio)
        assert rms == 0.0

    def test_compute_rms_sine_wave(self, sine_wave):
        """Test RMS of sine wave approximates theoretical value."""
        audio, sr = sine_wave
        rms = AudioMetrics.compute_rms(audio)

        # RMS of sine wave = amplitude / sqrt(2)
        # For 0.5 amplitude: 0.5 / 1.414 ≈ 0.3536
        expected_rms = 0.5 / np.sqrt(2)
        assert abs(rms - expected_rms) < 0.001

    def test_compute_rms_positive(self, sine_wave):
        """Test RMS is always positive."""
        audio, sr = sine_wave
        rms = AudioMetrics.compute_rms(audio)
        assert rms >= 0.0

    def test_compute_rms_returns_float(self, sine_wave):
        """Test RMS returns float type."""
        audio, sr = sine_wave
        rms = AudioMetrics.compute_rms(audio)
        assert isinstance(rms, float)

    def test_compute_rms_stereo(self, stereo_audio):
        """Test RMS computation with stereo audio."""
        audio, sr = stereo_audio
        rms = AudioMetrics.compute_rms(audio)

        # Should compute RMS across entire array
        assert rms > 0.0
        assert isinstance(rms, float)

    def test_compute_rms_dc_offset(self):
        """Test RMS with DC offset (constant amplitude)."""
        audio = np.ones(44100) * 0.5  # Constant 0.5
        rms = AudioMetrics.compute_rms(audio)

        # RMS of constant = absolute value
        assert abs(rms - 0.5) < 0.001

    # ===== compute_peak Tests =====

    def test_compute_peak_silent(self, silent_audio):
        """Test peak of silent audio is zero."""
        peak = AudioMetrics.compute_peak(silent_audio)
        assert peak == 0.0

    def test_compute_peak_sine_wave(self, sine_wave):
        """Test peak of sine wave equals amplitude."""
        audio, sr = sine_wave
        peak = AudioMetrics.compute_peak(audio)

        # Peak should be 0.5 (the amplitude)
        assert abs(peak - 0.5) < 0.01

    def test_compute_peak_full_scale(self, full_scale_sine):
        """Test peak of full-scale sine wave."""
        audio, sr = full_scale_sine
        peak = AudioMetrics.compute_peak(audio)

        # Peak should be 1.0
        assert abs(peak - 1.0) < 0.01

    def test_compute_peak_clipping(self, clipping_audio):
        """Test peak detection with clipping audio."""
        audio, sr = clipping_audio
        peak = AudioMetrics.compute_peak(audio)

        # Peak should be 1.5 (clipping level)
        assert abs(peak - 1.5) < 0.01

    def test_compute_peak_positive(self, sine_wave):
        """Test peak is always positive."""
        audio, sr = sine_wave
        peak = AudioMetrics.compute_peak(audio)
        assert peak >= 0.0

    def test_compute_peak_returns_float(self, sine_wave):
        """Test peak returns float type."""
        audio, sr = sine_wave
        peak = AudioMetrics.compute_peak(audio)
        assert isinstance(peak, float)

    def test_compute_peak_stereo(self, stereo_audio):
        """Test peak computation with stereo audio."""
        audio, sr = stereo_audio
        peak = AudioMetrics.compute_peak(audio)

        # Peak should be from louder channel (right = 0.5)
        assert abs(peak - 0.5) < 0.01

    def test_compute_peak_negative_values(self):
        """Test peak handles negative values correctly."""
        audio = np.array([-0.8, 0.6, -0.3, 0.4])
        peak = AudioMetrics.compute_peak(audio)

        # Peak should be 0.8 (absolute value of -0.8)
        assert abs(peak - 0.8) < 0.01

    # ===== compute_spectral_centroid Tests =====

    def test_compute_spectral_centroid_sine_wave(self, sine_wave):
        """Test spectral centroid of sine wave matches frequency."""
        audio, sr = sine_wave
        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)

        # For 440 Hz sine wave, centroid should be near 440 Hz
        assert 400 < centroid < 500

    def test_compute_spectral_centroid_high_frequency(self):
        """Test spectral centroid with high frequency signal."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # 5000 Hz sine wave
        audio = np.sin(2 * np.pi * 5000 * t) * 0.5

        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)

        # Centroid should be near 5000 Hz
        assert 4500 < centroid < 5500

    def test_compute_spectral_centroid_low_frequency(self):
        """Test spectral centroid with low frequency signal."""
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        # 100 Hz sine wave
        audio = np.sin(2 * np.pi * 100 * t) * 0.5

        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)

        # Centroid should be near 100 Hz
        assert 50 < centroid < 150

    def test_compute_spectral_centroid_white_noise(self, white_noise):
        """Test spectral centroid of white noise."""
        audio, sr = white_noise
        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)

        # White noise should have centroid near middle of spectrum
        # (roughly sr/4 for uniform spectral distribution)
        assert 5000 < centroid < 15000

    def test_compute_spectral_centroid_positive(self, sine_wave):
        """Test spectral centroid is always positive."""
        audio, sr = sine_wave
        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)
        assert centroid > 0

    def test_compute_spectral_centroid_returns_float(self, sine_wave):
        """Test spectral centroid returns float type."""
        audio, sr = sine_wave
        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)
        assert isinstance(centroid, float)

    def test_compute_spectral_centroid_different_sample_rates(self):
        """Test spectral centroid with different sample rates."""
        # 48kHz sample rate
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5

        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)

        # Should still be near 440 Hz
        assert 400 < centroid < 500

    # ===== has_clipping Tests =====

    def test_has_clipping_no_clipping(self, sine_wave):
        """Test no clipping detected for normal audio."""
        audio, sr = sine_wave
        has_clip = AudioMetrics.has_clipping(audio)
        assert has_clip is False

    def test_has_clipping_with_clipping(self, clipping_audio):
        """Test clipping detected when present."""
        audio, sr = clipping_audio
        has_clip = AudioMetrics.has_clipping(audio)
        assert has_clip is True

    def test_has_clipping_at_threshold(self):
        """Test clipping detection at exact threshold."""
        audio = np.array([0.99, 0.5, -0.99])

        # At default threshold (0.99), should detect clipping
        has_clip = AudioMetrics.has_clipping(audio, threshold=0.99)
        assert has_clip is True

    def test_has_clipping_custom_threshold(self):
        """Test clipping with custom threshold."""
        audio = np.array([0.95, 0.5, -0.95])

        # With threshold 0.95, should detect clipping
        has_clip = AudioMetrics.has_clipping(audio, threshold=0.95)
        assert has_clip is True

        # With threshold 0.99, should not detect clipping
        has_clip = AudioMetrics.has_clipping(audio, threshold=0.99)
        assert has_clip is False

    def test_has_clipping_silent(self, silent_audio):
        """Test no clipping in silent audio."""
        has_clip = AudioMetrics.has_clipping(silent_audio)
        assert has_clip is False

    def test_has_clipping_returns_bool(self, sine_wave):
        """Test has_clipping returns boolean type."""
        audio, sr = sine_wave
        has_clip = AudioMetrics.has_clipping(audio)
        assert isinstance(has_clip, bool)

    def test_has_clipping_negative_values(self):
        """Test clipping detection with negative peaks."""
        audio = np.array([0.5, -1.5, 0.3])
        has_clip = AudioMetrics.has_clipping(audio)
        assert has_clip is True

    def test_has_clipping_stereo(self, stereo_audio):
        """Test clipping detection with stereo audio."""
        audio, sr = stereo_audio
        has_clip = AudioMetrics.has_clipping(audio)

        # Stereo audio at 0.5 max should not clip
        assert has_clip is False

    def test_has_clipping_single_sample(self):
        """Test clipping detection with single clipping sample."""
        # Mostly normal, one clipping sample
        audio = np.array([0.5, 0.6, 0.4, 1.0, 0.5])
        has_clip = AudioMetrics.has_clipping(audio, threshold=0.99)
        assert has_clip is True

    # ===== compute_all Tests =====

    def test_compute_all_returns_dict(self, sine_wave):
        """Test compute_all returns dictionary."""
        audio, sr = sine_wave
        metrics = AudioMetrics.compute_all(audio, sr)
        assert isinstance(metrics, dict)

    def test_compute_all_contains_all_metrics(self, sine_wave):
        """Test compute_all contains all expected metrics."""
        audio, sr = sine_wave
        metrics = AudioMetrics.compute_all(audio, sr)

        assert "rms" in metrics
        assert "peak" in metrics
        assert "spectral_centroid" in metrics
        assert "has_clipping" in metrics

    def test_compute_all_correct_values(self, sine_wave):
        """Test compute_all returns correct metric values."""
        audio, sr = sine_wave
        metrics = AudioMetrics.compute_all(audio, sr)

        # Check RMS
        expected_rms = 0.5 / np.sqrt(2)
        assert abs(metrics["rms"] - expected_rms) < 0.001

        # Check peak
        assert abs(metrics["peak"] - 0.5) < 0.01

        # Check spectral centroid
        assert 400 < metrics["spectral_centroid"] < 500

        # Check clipping
        assert metrics["has_clipping"] is False

    def test_compute_all_with_custom_threshold(self, sine_wave):
        """Test compute_all with custom clipping threshold."""
        audio, sr = sine_wave
        metrics = AudioMetrics.compute_all(audio, sr, clip_threshold=0.4)

        # With threshold 0.4, 0.5 amplitude sine should show clipping
        assert metrics["has_clipping"] is True

    def test_compute_all_silent(self, silent_audio):
        """Test compute_all with silent audio."""
        metrics = AudioMetrics.compute_all(silent_audio, 44100)

        assert metrics["rms"] == 0.0
        assert metrics["peak"] == 0.0
        assert metrics["spectral_centroid"] >= 0  # May vary
        assert metrics["has_clipping"] is False

    def test_compute_all_clipping_audio(self, clipping_audio):
        """Test compute_all with clipping audio."""
        audio, sr = clipping_audio
        metrics = AudioMetrics.compute_all(audio, sr)

        assert metrics["rms"] > 0
        assert metrics["peak"] > 1.0
        assert metrics["has_clipping"] is True

    def test_compute_all_consistency(self, sine_wave):
        """Test compute_all matches individual metric calls."""
        audio, sr = sine_wave

        # Compute individually
        rms_individual = AudioMetrics.compute_rms(audio)
        peak_individual = AudioMetrics.compute_peak(audio)
        centroid_individual = AudioMetrics.compute_spectral_centroid(audio, sr)
        clipping_individual = AudioMetrics.has_clipping(audio)

        # Compute all
        metrics = AudioMetrics.compute_all(audio, sr)

        # Should match
        assert abs(metrics["rms"] - rms_individual) < 1e-10
        assert abs(metrics["peak"] - peak_individual) < 1e-10
        assert abs(metrics["spectral_centroid"] - centroid_individual) < 1e-10
        assert metrics["has_clipping"] == clipping_individual

    # ===== Edge Cases =====

    def test_empty_audio(self):
        """Test metrics with empty audio array."""
        audio = np.array([])

        # RMS and peak should handle empty array
        # (numpy will return 0 or nan, depending on operation)
        try:
            rms = AudioMetrics.compute_rms(audio)
            assert rms >= 0 or np.isnan(rms)
        except (ValueError, RuntimeWarning):
            # Acceptable to raise error on empty audio
            pass

    def test_single_sample(self):
        """Test metrics with single sample."""
        audio = np.array([0.5])

        rms = AudioMetrics.compute_rms(audio)
        peak = AudioMetrics.compute_peak(audio)

        assert abs(rms - 0.5) < 0.01
        assert abs(peak - 0.5) < 0.01

    def test_very_short_audio(self):
        """Test spectral centroid with very short audio."""
        sr = 44100
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 0.01, 441)) * 0.5

        # Should still compute without error
        centroid = AudioMetrics.compute_spectral_centroid(audio, sr)
        assert centroid > 0

    def test_extreme_amplitudes(self):
        """Test metrics with extreme amplitude values."""
        audio = np.array([100.0, -50.0, 75.0])

        rms = AudioMetrics.compute_rms(audio)
        peak = AudioMetrics.compute_peak(audio)
        has_clip = AudioMetrics.has_clipping(audio)

        assert rms > 50
        assert peak == 100.0
        assert has_clip is True

    # ===== Static Method Tests =====

    def test_static_method_no_instantiation(self):
        """Test that metrics can be called without instantiation."""
        # Should work without creating AudioMetrics instance
        audio = np.array([0.5, 0.3, 0.7])

        rms = AudioMetrics.compute_rms(audio)
        peak = AudioMetrics.compute_peak(audio)

        assert isinstance(rms, float)
        assert isinstance(peak, float)

    def test_multiple_calls_independent(self, sine_wave):
        """Test multiple calls don't affect each other (no state)."""
        audio, sr = sine_wave

        # Call multiple times
        rms1 = AudioMetrics.compute_rms(audio)
        rms2 = AudioMetrics.compute_rms(audio)
        rms3 = AudioMetrics.compute_rms(audio)

        # Should return same result each time
        assert rms1 == rms2 == rms3
