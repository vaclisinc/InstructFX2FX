"""Tests for BatchProcessor class."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from audio_processing.processor import AudioProcessor
from audio_processing.batch import BatchProcessor
from audio_processing.types import ProcessingResult
from src.models.parameters.effect_chain import EffectChain
from src.models.parameters.reverb import ReverbParameters


class TestBatchProcessor:
    """Test suite for BatchProcessor class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_audio_files(self, temp_dir):
        """Create multiple sample audio files for batch testing."""
        import soundfile as sf

        sample_rate = 44100
        duration = 0.5
        files = []

        for i in range(3):
            t = np.linspace(0, duration, int(sample_rate * duration))
            # Different frequencies for each file
            freq = 440 * (i + 1)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)

            audio_path = os.path.join(temp_dir, f"test_input_{i}.wav")
            sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
            files.append(audio_path)

        return files

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

    def test_batch_processor_initialization(self):
        """Test BatchProcessor initialization."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)
        assert batch.processor is not None
        assert batch.processor == processor

    def test_process_batch_basic(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test basic batch processing."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "batch_output")

        results = batch.process_batch(
            input_files=sample_audio_files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        # Check results
        assert len(results) == len(sample_audio_files)
        assert all(isinstance(r, ProcessingResult) for r in results)

        # Check output directory was created
        assert os.path.exists(output_dir)

        # Check all output files exist
        for input_file in sample_audio_files:
            output_file = os.path.join(output_dir, os.path.basename(input_file))
            assert os.path.exists(output_file)

    def test_process_batch_creates_directory(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test that batch processor creates output directory."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "nested", "batch", "output")
        assert not os.path.exists(output_dir)

        results = batch.process_batch(
            input_files=sample_audio_files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        # Directory was created
        assert os.path.exists(output_dir)
        assert len(results) == len(sample_audio_files)

    def test_process_batch_empty_list(self, temp_dir, simple_reverb_chain):
        """Test that batch processor rejects empty file list."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "output")

        with pytest.raises(ValueError, match="input_files cannot be empty"):
            batch.process_batch(
                input_files=[],
                output_dir=output_dir,
                effect_chain=simple_reverb_chain
            )

    def test_process_batch_with_summary(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test batch processing with summary statistics."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "batch_output")

        results, summary = batch.process_batch_with_summary(
            input_files=sample_audio_files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        # Check results
        assert len(results) == len(sample_audio_files)

        # Check summary
        assert isinstance(summary, dict)
        assert summary["total_files"] == len(sample_audio_files)
        assert "avg_rms_change_db" in summary
        assert "avg_peak_change_db" in summary
        assert "files_with_clipping" in summary
        assert "clipping_rate" in summary

        # Check summary values are reasonable
        assert isinstance(summary["avg_rms_change_db"], float)
        assert isinstance(summary["avg_peak_change_db"], float)
        assert isinstance(summary["files_with_clipping"], int)
        assert isinstance(summary["clipping_rate"], float)
        assert 0 <= summary["clipping_rate"] <= 100

    def test_process_pairs(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test processing with explicit input/output pairs."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        # Create pairs with custom output paths
        pairs = [
            (sample_audio_files[0], os.path.join(temp_dir, "out1.wav")),
            (sample_audio_files[1], os.path.join(temp_dir, "subdir", "out2.wav")),
            (sample_audio_files[2], os.path.join(temp_dir, "out3.wav")),
        ]

        results = batch.process_pairs(
            file_pairs=pairs,
            effect_chain=simple_reverb_chain
        )

        # Check results
        assert len(results) == len(pairs)
        assert all(isinstance(r, ProcessingResult) for r in results)

        # Check all output files exist
        for input_path, output_path in pairs:
            assert os.path.exists(output_path)

    def test_process_pairs_empty_list(self, simple_reverb_chain):
        """Test that process_pairs rejects empty list."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        with pytest.raises(ValueError, match="file_pairs cannot be empty"):
            batch.process_pairs(
                file_pairs=[],
                effect_chain=simple_reverb_chain
            )

    def test_process_pairs_creates_directories(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test that process_pairs creates nested output directories."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        # Create pairs with nested output paths
        output_path = os.path.join(temp_dir, "nested", "dir", "output.wav")
        pairs = [(sample_audio_files[0], output_path)]

        assert not os.path.exists(os.path.dirname(output_path))

        results = batch.process_pairs(
            file_pairs=pairs,
            effect_chain=simple_reverb_chain
        )

        # Directory and file were created
        assert os.path.exists(os.path.dirname(output_path))
        assert os.path.exists(output_path)

    def test_batch_processing_resilience(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test that batch processing continues after failures."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        # Mix valid and invalid files
        mixed_files = [
            sample_audio_files[0],
            "nonexistent_file.wav",  # This will fail
            sample_audio_files[1],
        ]

        output_dir = os.path.join(temp_dir, "output")

        # Should not raise exception, but some files will fail
        results = batch.process_batch(
            input_files=mixed_files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        # Should have processed the valid files
        assert len(results) == 2  # Only 2 succeeded
        assert all(isinstance(r, ProcessingResult) for r in results)

    def test_batch_preserve_structure_false(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test batch processing with flat output structure."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "output")

        results = batch.process_batch(
            input_files=sample_audio_files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain,
            preserve_structure=False
        )

        # All files should be in output_dir directly
        for input_file in sample_audio_files:
            output_file = os.path.join(output_dir, os.path.basename(input_file))
            assert os.path.exists(output_file)

    def test_batch_different_sample_rates(self, temp_dir, simple_reverb_chain):
        """Test batch processing files with different sample rates."""
        import soundfile as sf

        # Create files with different sample rates
        files = []
        sample_rates = [44100, 48000, 22050]

        for i, sr in enumerate(sample_rates):
            t = np.linspace(0, 0.5, int(sr * 0.5))
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)

            audio_path = os.path.join(temp_dir, f"input_{sr}.wav")
            sf.write(audio_path, audio, sr, subtype="PCM_16")
            files.append(audio_path)

        # Process all at 44.1kHz
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "output")

        results = batch.process_batch(
            input_files=files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        # All should be processed and resampled to 44.1kHz
        assert len(results) == len(files)

        for input_file in files:
            output_file = os.path.join(output_dir, os.path.basename(input_file))
            info = sf.info(output_file)
            assert info.samplerate == 44100  # All resampled to target

    def test_batch_large_number_of_files(self, temp_dir, simple_reverb_chain):
        """Test batch processing with many files."""
        import soundfile as sf

        # Create 10 test files
        sample_rate = 44100
        duration = 0.2  # Short files for speed
        files = []

        for i in range(10):
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = 0.3 * np.sin(2 * np.pi * (400 + i * 50) * t)

            audio_path = os.path.join(temp_dir, f"input_{i:02d}.wav")
            sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
            files.append(audio_path)

        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "output")

        results = batch.process_batch(
            input_files=files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        assert len(results) == 10
        assert all(isinstance(r, ProcessingResult) for r in results)

    def test_batch_summary_with_clipping(self, temp_dir, simple_reverb_chain):
        """Test batch summary when some files have clipping."""
        import soundfile as sf

        # Create files with different levels (some will clip)
        sample_rate = 44100
        duration = 0.5
        files = []

        levels = [0.3, 0.95, 1.2]  # Last one will clip

        for i, level in enumerate(levels):
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = level * np.sin(2 * np.pi * 440 * t)

            audio_path = os.path.join(temp_dir, f"input_{i}.wav")
            sf.write(audio_path, audio, sample_rate, subtype="PCM_16")
            files.append(audio_path)

        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "output")

        results, summary = batch.process_batch_with_summary(
            input_files=files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        # Check clipping metrics
        assert summary["files_with_clipping"] >= 0
        assert summary["clipping_rate"] >= 0

    def test_batch_metrics_consistency(self, temp_dir, sample_audio_files, simple_reverb_chain):
        """Test that batch processing produces consistent metrics."""
        processor = AudioProcessor(sample_rate=44100)
        batch = BatchProcessor(processor)

        output_dir = os.path.join(temp_dir, "output")

        results = batch.process_batch(
            input_files=sample_audio_files,
            output_dir=output_dir,
            effect_chain=simple_reverb_chain
        )

        # All results should have valid metrics
        for result in results:
            assert result.input_rms > 0
            assert result.output_rms > 0
            assert result.peak_input > 0
            assert result.peak_output > 0
            assert isinstance(result.clipping_detected, bool)

            # RMS and peak changes should be calculable
            rms_change = result.get_rms_change_db()
            peak_change = result.get_peak_change_db()
            assert isinstance(rms_change, float)
            assert isinstance(peak_change, float)
