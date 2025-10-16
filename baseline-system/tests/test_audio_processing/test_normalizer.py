"""Tests for ParameterNormalizer class."""

import pytest
import math

from src.audio_processing.normalizer import ParameterNormalizer


class TestParameterNormalizer:
    """Test suite for ParameterNormalizer."""

    def test_normalize_eq_q_valid_range(self):
        """Test Q-factor normalization with valid values."""
        # Test minimum value
        result = ParameterNormalizer.normalize_eq_q(0.1)
        assert pytest.approx(result, 0.001) == 0.1 * 0.707

        # Test maximum value
        result = ParameterNormalizer.normalize_eq_q(10.0)
        assert pytest.approx(result, 0.001) == 10.0 * 0.707

        # Test middle value
        result = ParameterNormalizer.normalize_eq_q(5.0)
        assert pytest.approx(result, 0.001) == 5.0 * 0.707

        # Test common Q values
        result = ParameterNormalizer.normalize_eq_q(1.0)
        assert pytest.approx(result, 0.001) == 0.707

        result = ParameterNormalizer.normalize_eq_q(2.0)
        assert pytest.approx(result, 0.001) == 2.0 * 0.707

    def test_normalize_eq_q_invalid_range(self):
        """Test that Q-factor normalization raises error for invalid values."""
        # Below minimum
        with pytest.raises(ValueError, match="outside valid range"):
            ParameterNormalizer.normalize_eq_q(0.05)

        # Above maximum
        with pytest.raises(ValueError, match="outside valid range"):
            ParameterNormalizer.normalize_eq_q(15.0)

        # Negative value
        with pytest.raises(ValueError, match="outside valid range"):
            ParameterNormalizer.normalize_eq_q(-1.0)

    def test_normalize_reverb_room_size_valid_range(self):
        """Test room size normalization with valid values."""
        # Test minimum value (0 -> 0.1)
        result = ParameterNormalizer.normalize_reverb_room_size(0.0)
        assert pytest.approx(result, 0.001) == 0.1

        # Test maximum value (1 -> 0.9)
        result = ParameterNormalizer.normalize_reverb_room_size(1.0)
        assert pytest.approx(result, 0.001) == 0.9

        # Test middle value (0.5 -> 0.5)
        result = ParameterNormalizer.normalize_reverb_room_size(0.5)
        assert pytest.approx(result, 0.001) == 0.5

        # Test quarter value (0.25 -> 0.3)
        result = ParameterNormalizer.normalize_reverb_room_size(0.25)
        assert pytest.approx(result, 0.001) == 0.3

        # Test three-quarter value (0.75 -> 0.7)
        result = ParameterNormalizer.normalize_reverb_room_size(0.75)
        assert pytest.approx(result, 0.001) == 0.7

    def test_normalize_reverb_room_size_formula(self):
        """Test that room size normalization uses correct formula."""
        # Formula: output = input * 0.8 + 0.1
        test_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        for value in test_values:
            result = ParameterNormalizer.normalize_reverb_room_size(value)
            expected = value * 0.8 + 0.1
            assert pytest.approx(result, 0.001) == expected

    def test_normalize_reverb_room_size_invalid_range(self):
        """Test that room size normalization raises error for invalid values."""
        # Below minimum
        with pytest.raises(ValueError, match="outside valid range"):
            ParameterNormalizer.normalize_reverb_room_size(-0.1)

        # Above maximum
        with pytest.raises(ValueError, match="outside valid range"):
            ParameterNormalizer.normalize_reverb_room_size(1.5)

    def test_ms_to_seconds_conversion(self):
        """Test milliseconds to seconds conversion."""
        # Zero
        result = ParameterNormalizer.ms_to_seconds(0.0)
        assert pytest.approx(result, 0.001) == 0.0

        # Common values
        result = ParameterNormalizer.ms_to_seconds(1000.0)
        assert pytest.approx(result, 0.001) == 1.0

        result = ParameterNormalizer.ms_to_seconds(500.0)
        assert pytest.approx(result, 0.001) == 0.5

        result = ParameterNormalizer.ms_to_seconds(50.0)
        assert pytest.approx(result, 0.001) == 0.05

        result = ParameterNormalizer.ms_to_seconds(5.0)
        assert pytest.approx(result, 0.001) == 0.005

    def test_ms_to_seconds_invalid_value(self):
        """Test that negative time values raise error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            ParameterNormalizer.ms_to_seconds(-10.0)

    def test_seconds_to_ms_conversion(self):
        """Test seconds to milliseconds conversion."""
        # Zero
        result = ParameterNormalizer.seconds_to_ms(0.0)
        assert pytest.approx(result, 0.001) == 0.0

        # Common values
        result = ParameterNormalizer.seconds_to_ms(1.0)
        assert pytest.approx(result, 0.001) == 1000.0

        result = ParameterNormalizer.seconds_to_ms(0.5)
        assert pytest.approx(result, 0.001) == 500.0

        result = ParameterNormalizer.seconds_to_ms(0.05)
        assert pytest.approx(result, 0.001) == 50.0

        result = ParameterNormalizer.seconds_to_ms(0.005)
        assert pytest.approx(result, 0.001) == 5.0

    def test_seconds_to_ms_invalid_value(self):
        """Test that negative time values raise error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            ParameterNormalizer.seconds_to_ms(-1.0)

    def test_time_conversion_roundtrip(self):
        """Test that time conversions are reversible."""
        test_values = [0.0, 1.0, 10.0, 50.0, 100.0, 1000.0]

        for ms_value in test_values:
            # Convert to seconds and back
            seconds = ParameterNormalizer.ms_to_seconds(ms_value)
            ms_back = ParameterNormalizer.seconds_to_ms(seconds)
            assert pytest.approx(ms_back, 0.001) == ms_value

        for sec_value in [v / 1000.0 for v in test_values]:
            # Convert to milliseconds and back
            ms = ParameterNormalizer.seconds_to_ms(sec_value)
            sec_back = ParameterNormalizer.ms_to_seconds(ms)
            assert pytest.approx(sec_back, 0.001) == sec_value

    def test_db_to_linear_conversion(self):
        """Test decibel to linear conversion."""
        # 0 dB = 1.0 linear
        result = ParameterNormalizer.db_to_linear(0.0)
        assert pytest.approx(result, 0.001) == 1.0

        # +6 dB ≈ 2.0 linear
        result = ParameterNormalizer.db_to_linear(6.0)
        assert pytest.approx(result, 0.01) == 1.995

        # -6 dB ≈ 0.5 linear
        result = ParameterNormalizer.db_to_linear(-6.0)
        assert pytest.approx(result, 0.01) == 0.501

        # +20 dB = 10.0 linear
        result = ParameterNormalizer.db_to_linear(20.0)
        assert pytest.approx(result, 0.001) == 10.0

        # -20 dB = 0.1 linear
        result = ParameterNormalizer.db_to_linear(-20.0)
        assert pytest.approx(result, 0.001) == 0.1

    def test_linear_to_db_conversion(self):
        """Test linear to decibel conversion."""
        # 1.0 linear = 0 dB
        result = ParameterNormalizer.linear_to_db(1.0)
        assert pytest.approx(result, 0.001) == 0.0

        # 2.0 linear ≈ +6 dB
        result = ParameterNormalizer.linear_to_db(2.0)
        assert pytest.approx(result, 0.01) == 6.02

        # 0.5 linear ≈ -6 dB
        result = ParameterNormalizer.linear_to_db(0.5)
        assert pytest.approx(result, 0.01) == -6.02

        # 10.0 linear = +20 dB
        result = ParameterNormalizer.linear_to_db(10.0)
        assert pytest.approx(result, 0.001) == 20.0

        # 0.1 linear = -20 dB
        result = ParameterNormalizer.linear_to_db(0.1)
        assert pytest.approx(result, 0.001) == -20.0

    def test_linear_to_db_invalid_value(self):
        """Test that non-positive linear values raise error."""
        # Zero
        with pytest.raises(ValueError, match="must be positive"):
            ParameterNormalizer.linear_to_db(0.0)

        # Negative
        with pytest.raises(ValueError, match="must be positive"):
            ParameterNormalizer.linear_to_db(-1.0)

    def test_db_linear_conversion_roundtrip(self):
        """Test that dB/linear conversions are reversible."""
        test_db_values = [-20.0, -12.0, -6.0, 0.0, 6.0, 12.0, 20.0]

        for db_value in test_db_values:
            # Convert to linear and back
            linear = ParameterNormalizer.db_to_linear(db_value)
            db_back = ParameterNormalizer.linear_to_db(linear)
            assert pytest.approx(db_back, 0.001) == db_value

        test_linear_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        for linear_value in test_linear_values:
            # Convert to dB and back
            db = ParameterNormalizer.linear_to_db(linear_value)
            linear_back = ParameterNormalizer.db_to_linear(db)
            assert pytest.approx(linear_back, 0.001) == linear_value

    def test_clamp_within_range(self):
        """Test clamping values within valid range."""
        # Value within range - should remain unchanged
        result = ParameterNormalizer.clamp(5.0, 0.0, 10.0)
        assert result == 5.0

        result = ParameterNormalizer.clamp(0.5, 0.0, 1.0)
        assert result == 0.5

    def test_clamp_below_minimum(self):
        """Test clamping values below minimum."""
        result = ParameterNormalizer.clamp(-5.0, 0.0, 10.0)
        assert result == 0.0

        result = ParameterNormalizer.clamp(-100.0, -50.0, 50.0)
        assert result == -50.0

    def test_clamp_above_maximum(self):
        """Test clamping values above maximum."""
        result = ParameterNormalizer.clamp(15.0, 0.0, 10.0)
        assert result == 10.0

        result = ParameterNormalizer.clamp(100.0, -50.0, 50.0)
        assert result == 50.0

    def test_clamp_at_boundaries(self):
        """Test clamping values exactly at boundaries."""
        # At minimum
        result = ParameterNormalizer.clamp(0.0, 0.0, 10.0)
        assert result == 0.0

        # At maximum
        result = ParameterNormalizer.clamp(10.0, 0.0, 10.0)
        assert result == 10.0

    def test_clamp_invalid_range(self):
        """Test that invalid min/max raises error."""
        with pytest.raises(ValueError, match="cannot be greater than"):
            ParameterNormalizer.clamp(5.0, 10.0, 0.0)

    def test_clamp_negative_range(self):
        """Test clamping with negative ranges."""
        result = ParameterNormalizer.clamp(-15.0, -20.0, -10.0)
        assert result == -15.0

        result = ParameterNormalizer.clamp(-25.0, -20.0, -10.0)
        assert result == -20.0

        result = ParameterNormalizer.clamp(-5.0, -20.0, -10.0)
        assert result == -10.0

    def test_clamp_floating_point_precision(self):
        """Test clamping with floating point values."""
        result = ParameterNormalizer.clamp(0.12345, 0.1, 0.2)
        assert pytest.approx(result, 0.00001) == 0.12345

        result = ParameterNormalizer.clamp(0.05, 0.1, 0.2)
        assert pytest.approx(result, 0.00001) == 0.1

        result = ParameterNormalizer.clamp(0.25, 0.1, 0.2)
        assert pytest.approx(result, 0.00001) == 0.2

    def test_normalizer_all_static_methods(self):
        """Test that all methods are static and don't require instance."""
        # All methods should work without creating an instance
        assert ParameterNormalizer.normalize_eq_q(1.0) is not None
        assert ParameterNormalizer.normalize_reverb_room_size(0.5) is not None
        assert ParameterNormalizer.ms_to_seconds(1000.0) is not None
        assert ParameterNormalizer.seconds_to_ms(1.0) is not None
        assert ParameterNormalizer.db_to_linear(0.0) is not None
        assert ParameterNormalizer.linear_to_db(1.0) is not None
        assert ParameterNormalizer.clamp(5.0, 0.0, 10.0) is not None

    def test_normalize_eq_q_edge_cases(self):
        """Test Q-factor normalization edge cases."""
        # Test boundary values
        result = ParameterNormalizer.normalize_eq_q(0.1)
        assert result > 0  # Should remain positive

        result = ParameterNormalizer.normalize_eq_q(10.0)
        assert result < 10.0  # Should be scaled down

    def test_normalize_reverb_room_size_never_extreme(self):
        """Test that normalized room size never reaches extremes."""
        # Even at minimum input, output should be > 0
        result = ParameterNormalizer.normalize_reverb_room_size(0.0)
        assert result > 0.0

        # Even at maximum input, output should be < 1
        result = ParameterNormalizer.normalize_reverb_room_size(1.0)
        assert result < 1.0

    def test_comprehensive_parameter_pipeline(self):
        """Test realistic parameter normalization pipeline."""
        # Simulate normalizing parameters for a complete effect chain

        # EQ Q-factors
        q_values = [0.5, 1.0, 2.0, 3.0]
        normalized_qs = [ParameterNormalizer.normalize_eq_q(q) for q in q_values]
        assert all(0 < nq < 10 for nq in normalized_qs)

        # Reverb room sizes
        room_sizes = [0.0, 0.25, 0.5, 0.75, 1.0]
        normalized_rooms = [
            ParameterNormalizer.normalize_reverb_room_size(size)
            for size in room_sizes
        ]
        assert all(0.1 <= nr <= 0.9 for nr in normalized_rooms)

        # Time conversions
        attack_ms = 5.0
        release_ms = 50.0
        attack_sec = ParameterNormalizer.ms_to_seconds(attack_ms)
        release_sec = ParameterNormalizer.ms_to_seconds(release_ms)
        assert attack_sec < release_sec

        # Gain conversions
        makeup_gain_db = 6.0
        makeup_gain_linear = ParameterNormalizer.db_to_linear(makeup_gain_db)
        assert makeup_gain_linear > 1.0  # Positive gain = amplification
