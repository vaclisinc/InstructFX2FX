"""Tests for ProcessingResult data type."""

import pytest
import numpy as np
from pydantic import ValidationError

from audio_processing.types import ProcessingResult


class TestProcessingResult:
    """Test suite for ProcessingResult data class."""

    def test_processing_result_creation(self):
        """Test creating a ProcessingResult."""
        result = ProcessingResult(
            input_rms=0.3,
            output_rms=0.4,
            peak_input=0.8,
            peak_output=0.9,
            clipping_detected=False
        )

        assert result.input_rms == 0.3
        assert result.output_rms == 0.4
        assert result.peak_input == 0.8
        assert result.peak_output == 0.9
        assert result.clipping_detected is False

    def test_processing_result_validation_positive_rms(self):
        """Test that RMS values must be non-negative."""
        # Valid: zero RMS
        result = ProcessingResult(
            input_rms=0.0,
            output_rms=0.0,
            peak_input=0.5,
            peak_output=0.5,
            clipping_detected=False
        )
        assert result.input_rms == 0.0

        # Invalid: negative RMS
        with pytest.raises(ValidationError):
            ProcessingResult(
                input_rms=-0.1,
                output_rms=0.4,
                peak_input=0.8,
                peak_output=0.9,
                clipping_detected=False
            )

    def test_processing_result_validation_peak_range(self):
        """Test that peak values are within valid range."""
        # Valid: within range
        result = ProcessingResult(
            input_rms=0.3,
            output_rms=0.4,
            peak_input=1.0,
            peak_output=1.5,
            clipping_detected=False
        )
        assert result.peak_input == 1.0

        # Invalid: too high
        with pytest.raises(ValidationError):
            ProcessingResult(
                input_rms=0.3,
                output_rms=0.4,
                peak_input=2.5,  # > 2.0
                peak_output=0.9,
                clipping_detected=False
            )

    def test_get_rms_change_db(self):
        """Test RMS change calculation in dB."""
        result = ProcessingResult(
            input_rms=0.5,
            output_rms=1.0,  # 2x increase
            peak_input=0.8,
            peak_output=0.9,
            clipping_detected=False
        )

        rms_change = result.get_rms_change_db()

        # 2x increase should be ~6 dB
        assert isinstance(rms_change, float)
        assert 5.9 < rms_change < 6.1

    def test_get_rms_change_db_decrease(self):
        """Test RMS change calculation for level decrease."""
        result = ProcessingResult(
            input_rms=1.0,
            output_rms=0.5,  # 0.5x decrease
            peak_input=0.8,
            peak_output=0.9,
            clipping_detected=False
        )

        rms_change = result.get_rms_change_db()

        # 0.5x decrease should be ~-6 dB
        assert isinstance(rms_change, float)
        assert -6.1 < rms_change < -5.9

    def test_get_rms_change_db_zero_input(self):
        """Test RMS change when input RMS is zero."""
        result = ProcessingResult(
            input_rms=0.0,
            output_rms=0.5,
            peak_input=0.8,
            peak_output=0.9,
            clipping_detected=False
        )

        rms_change = result.get_rms_change_db()

        # Should return 0 to avoid log(0)
        assert rms_change == 0.0

    def test_get_peak_change_db(self):
        """Test peak level change calculation in dB."""
        result = ProcessingResult(
            input_rms=0.3,
            output_rms=0.4,
            peak_input=0.5,
            peak_output=1.0,  # 2x increase
            clipping_detected=False
        )

        peak_change = result.get_peak_change_db()

        # 2x increase should be ~6 dB
        assert isinstance(peak_change, float)
        assert 5.9 < peak_change < 6.1

    def test_get_peak_change_db_zero_input(self):
        """Test peak change when input peak is zero."""
        result = ProcessingResult(
            input_rms=0.0,
            output_rms=0.0,
            peak_input=0.0,
            peak_output=0.5,
            clipping_detected=False
        )

        peak_change = result.get_peak_change_db()

        # Should return 0 to avoid log(0)
        assert peak_change == 0.0

    def test_processing_result_with_clipping(self):
        """Test ProcessingResult with clipping detected."""
        result = ProcessingResult(
            input_rms=0.3,
            output_rms=0.5,
            peak_input=0.8,
            peak_output=0.995,
            clipping_detected=True
        )

        assert result.clipping_detected is True

    def test_processing_result_serialization(self):
        """Test that ProcessingResult can be serialized."""
        result = ProcessingResult(
            input_rms=0.3,
            output_rms=0.4,
            peak_input=0.8,
            peak_output=0.9,
            clipping_detected=False
        )

        # Convert to dict
        result_dict = result.model_dump()

        assert isinstance(result_dict, dict)
        assert result_dict["input_rms"] == 0.3
        assert result_dict["output_rms"] == 0.4
        assert result_dict["peak_input"] == 0.8
        assert result_dict["peak_output"] == 0.9
        assert result_dict["clipping_detected"] is False

    def test_processing_result_from_dict(self):
        """Test creating ProcessingResult from dictionary."""
        data = {
            "input_rms": 0.3,
            "output_rms": 0.4,
            "peak_input": 0.8,
            "peak_output": 0.9,
            "clipping_detected": False
        }

        result = ProcessingResult(**data)

        assert result.input_rms == 0.3
        assert result.output_rms == 0.4
        assert result.peak_input == 0.8
        assert result.peak_output == 0.9
        assert result.clipping_detected is False

    def test_processing_result_realistic_values(self):
        """Test ProcessingResult with realistic audio processing values."""
        # Typical values from reverb processing
        result = ProcessingResult(
            input_rms=0.25,
            output_rms=0.28,  # Slight increase from reverb tail
            peak_input=0.92,
            peak_output=0.95,  # Slight increase
            clipping_detected=False
        )

        rms_change = result.get_rms_change_db()
        peak_change = result.get_peak_change_db()

        # Should be small positive changes
        assert 0 < rms_change < 2
        assert 0 < peak_change < 1

    def test_processing_result_compression_values(self):
        """Test ProcessingResult with typical compression values."""
        # Compression typically increases RMS but reduces peaks
        result = ProcessingResult(
            input_rms=0.2,
            output_rms=0.3,  # RMS increased
            peak_input=0.9,
            peak_output=0.7,  # Peak reduced
            clipping_detected=False
        )

        rms_change = result.get_rms_change_db()
        peak_change = result.get_peak_change_db()

        # RMS should increase, peak should decrease
        assert rms_change > 0
        assert peak_change < 0

    def test_processing_result_extreme_gain(self):
        """Test ProcessingResult with extreme gain changes."""
        # Very high gain (10x)
        result = ProcessingResult(
            input_rms=0.1,
            output_rms=1.0,
            peak_input=0.2,
            peak_output=2.0,  # At upper limit
            clipping_detected=True
        )

        rms_change = result.get_rms_change_db()

        # 10x should be 20 dB
        assert 19.9 < rms_change < 20.1

    def test_processing_result_no_change(self):
        """Test ProcessingResult when no processing change occurred."""
        result = ProcessingResult(
            input_rms=0.5,
            output_rms=0.5,
            peak_input=0.9,
            peak_output=0.9,
            clipping_detected=False
        )

        rms_change = result.get_rms_change_db()
        peak_change = result.get_peak_change_db()

        # Should be very close to 0 dB
        assert -0.1 < rms_change < 0.1
        assert -0.1 < peak_change < 0.1
