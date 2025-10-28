"""Tests for AudioProcessor pipeline integration."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from audio_processing.processor import AudioProcessor
from audio_processing.types import ProcessingResult
from src.models.parameters.effect_chain import EffectChain
from src.models.parameters.reverb import ReverbParameters
from src.models.parameters.eq import EQParameters, EQBand
from src.models.parameters.compressor import CompressorParameters


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_audio_file(self, temp_dir):
        """Create a sample audio file for testing."""
        import soundfile as sf

        # Generate 1 second of test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

        # Save to temp file
        audio_path = os.path.join(temp_dir, "test_input.wav")
        sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

        return audio_path

    @pytest.fixture
    def sample_stereo_file(self, temp_dir):
        """Create a sample stereo audio file for testing."""
        import soundfile as sf

        # Generate 1 second of stereo test audio
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        left = 0.5 * np.sin(2 * np.pi * 440 * t)
        right = 0.5 * np.sin(2 * np.pi * 880 * t)
        audio = np.stack([left, right], axis=1)

        # Save to temp file
        audio_path = os.path.join(temp_dir, "test_stereo.wav")
        sf.write(audio_path, audio, sample_rate, subtype="PCM_16")

        return audio_path

    @pytest.fixture
    def simple_reverb_chain(self):
        """Create a simple reverb effect chain."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.3,
            dry_level=0.7,
            width=1.0,
            freeze_mode=False
        )
        return EffectChain(
            description="Simple reverb",
            effects=[reverb],
            order=["reverb"]
        )

    @pytest.fixture
    def complex_effect_chain(self):
        """Create a complex effect chain with multiple effects."""
        # EQ with 3 bands (minimum required)
        eq = EQParameters(
            eq_type="parametric",
            bands=[
                EQBand(frequency=200, gain=3.0, q=1.0),
                EQBand(frequency=1000, gain=1.0, q=1.2),
                EQBand(frequency=5000, gain=-2.0, q=1.5)
            ]
        )

        # Compressor
        compressor = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=100.0,
            knee=6.0,
            makeup_gain=3.0
        )

        # Reverb
        reverb = ReverbParameters(
            room_size=0.7,
            damping=0.4,
            wet_level=0.2,
            dry_level=0.8,
            width=1.0,
            freeze_mode=False
        )

        return EffectChain(
            description="Complex chain with EQ, compression, and reverb",
            effects=[eq, compressor, reverb],
            order=["eq", "compressor", "reverb"]
        )

    def test_processor_initialization(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(sample_rate=44100)
        assert processor.sample_rate == 44100
        assert processor.loader is not None
        assert processor.chain_builder is not None

    def test_processor_initialization_custom_sr(self):
        """Test AudioProcessor with custom sample rate."""
        processor = AudioProcessor(sample_rate=48000)
        assert processor.sample_rate == 48000
        assert processor.loader.sample_rate == 48000

    def test_process_basic(self, temp_dir, sample_audio_file, simple_reverb_chain):
        """Test basic audio processing with reverb."""
        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output.wav")

        result = processor.process(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=simple_reverb_chain
        )

        # Check result type
        assert isinstance(result, ProcessingResult)

        # Check output file exists
        assert os.path.exists(output_path)

        # Check metrics are reasonable
        assert result.input_rms > 0
        assert result.output_rms > 0
        assert result.peak_input > 0
        assert result.peak_output > 0
        assert isinstance(result.clipping_detected, bool)

    def test_process_complex_chain(self, temp_dir, sample_audio_file, complex_effect_chain):
        """Test processing with complex multi-effect chain."""
        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output_complex.wav")

        result = processor.process(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=complex_effect_chain
        )

        # Check processing succeeded
        assert isinstance(result, ProcessingResult)
        assert os.path.exists(output_path)

        # Verify metrics
        assert result.input_rms > 0
        assert result.output_rms > 0

    def test_process_stereo(self, temp_dir, sample_stereo_file, simple_reverb_chain):
        """Test processing stereo audio."""
        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output_stereo.wav")

        result = processor.process(
            input_path=sample_stereo_file,
            output_path=output_path,
            effect_chain=simple_reverb_chain
        )

        # Check processing succeeded
        assert isinstance(result, ProcessingResult)
        assert os.path.exists(output_path)

        # Verify output is also stereo
        import soundfile as sf
        audio, sr = sf.read(output_path)
        assert audio.ndim == 2  # Stereo
        assert audio.shape[1] == 2  # 2 channels

    def test_process_creates_output_directory(self, temp_dir, sample_audio_file, simple_reverb_chain):
        """Test that processor creates output directory if it doesn't exist."""
        processor = AudioProcessor(sample_rate=44100)
        output_subdir = os.path.join(temp_dir, "nested", "output")
        output_path = os.path.join(output_subdir, "output.wav")

        # Directory doesn't exist yet
        assert not os.path.exists(output_subdir)

        result = processor.process(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=simple_reverb_chain
        )

        # Directory was created
        assert os.path.exists(output_subdir)
        assert os.path.exists(output_path)

    def test_processing_result_metrics(self, temp_dir, sample_audio_file, simple_reverb_chain):
        """Test that ProcessingResult contains correct metrics."""
        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output.wav")

        result = processor.process(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=simple_reverb_chain
        )

        # Check RMS change calculation
        rms_change = result.get_rms_change_db()
        assert isinstance(rms_change, float)
        # Reverb typically increases RMS slightly due to added reflections
        assert -10 < rms_change < 10  # Reasonable range

        # Check peak change calculation
        peak_change = result.get_peak_change_db()
        assert isinstance(peak_change, float)
        assert -10 < peak_change < 10  # Reasonable range

    def test_process_with_monitoring(self, temp_dir, sample_audio_file, simple_reverb_chain):
        """Test process_with_monitoring method."""
        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output_monitored.wav")

        result, monitoring = processor.process_with_monitoring(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=simple_reverb_chain
        )

        # Check result
        assert isinstance(result, ProcessingResult)
        assert os.path.exists(output_path)

        # Check monitoring dict
        assert isinstance(monitoring, dict)
        assert "input_spectral_centroid" in monitoring
        assert "output_spectral_centroid" in monitoring
        assert "spectral_centroid_change" in monitoring

        # Check spectral centroids are reasonable
        assert monitoring["input_spectral_centroid"] > 0
        assert monitoring["output_spectral_centroid"] > 0

    def test_process_invalid_input(self, temp_dir, simple_reverb_chain):
        """Test processing with invalid input file."""
        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output.wav")

        with pytest.raises(Exception):  # AudioLoadError
            processor.process(
                input_path="nonexistent.wav",
                output_path=output_path,
                effect_chain=simple_reverb_chain
            )

    def test_process_preserves_sample_rate(self, temp_dir, sample_audio_file, simple_reverb_chain):
        """Test that processing preserves sample rate."""
        import soundfile as sf

        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output.wav")

        processor.process(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=simple_reverb_chain
        )

        # Check output sample rate
        info = sf.info(output_path)
        assert info.samplerate == 44100

    def test_process_eq_chain(self, temp_dir, sample_audio_file):
        """Test processing with EQ effect chain."""
        # Create EQ chain (minimum 3 bands required)
        eq = EQParameters(
            eq_type="parametric",
            bands=[
                EQBand(frequency=500, gain=3.0, q=1.0),
                EQBand(frequency=1000, gain=6.0, q=1.0),
                EQBand(frequency=4000, gain=-3.0, q=1.2)
            ]
        )
        eq_chain = EffectChain(
            description="Boost 1kHz",
            effects=[eq],
            order=["eq"]
        )

        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output_eq.wav")

        result = processor.process(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=eq_chain
        )

        assert isinstance(result, ProcessingResult)
        assert os.path.exists(output_path)

    def test_process_compressor_chain(self, temp_dir, sample_audio_file):
        """Test processing with compressor effect chain."""
        # Create compressor chain with all required fields
        compressor = CompressorParameters(
            threshold=-15.0,
            ratio=3.0,
            attack=10.0,
            release=100.0,
            knee=4.0,
            makeup_gain=2.0
        )
        comp_chain = EffectChain(
            description="Compress",
            effects=[compressor],
            order=["compressor"]
        )

        processor = AudioProcessor(sample_rate=44100)
        output_path = os.path.join(temp_dir, "output_comp.wav")

        result = processor.process(
            input_path=sample_audio_file,
            output_path=output_path,
            effect_chain=comp_chain
        )

        assert isinstance(result, ProcessingResult)
        assert os.path.exists(output_path)

    def test_high_sample_rate(self, temp_dir, simple_reverb_chain):
        """Test processing with high sample rate (48kHz)."""
        import soundfile as sf

        # Create 48kHz test file
        sample_rate = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        input_path = os.path.join(temp_dir, "input_48k.wav")
        sf.write(input_path, audio, sample_rate, subtype="PCM_16")

        # Process at 48kHz
        processor = AudioProcessor(sample_rate=48000)
        output_path = os.path.join(temp_dir, "output_48k.wav")

        result = processor.process(
            input_path=input_path,
            output_path=output_path,
            effect_chain=simple_reverb_chain
        )

        assert isinstance(result, ProcessingResult)

        # Verify output is 48kHz
        info = sf.info(output_path)
        assert info.samplerate == 48000
