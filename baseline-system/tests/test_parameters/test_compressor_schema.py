"""Tests for Compressor parameter schema validation.

Tests verify that CompressorParameters model correctly validates:
- Parameter ranges (threshold: -60 to 0, ratio: 1-20, attack: 0.1-100, release: 10-1000, knee: 0-12, makeup_gain: 0-24)
- Required fields
- Type validation
- Edge cases (min/max values)
- Timing relationships (attack < release)
"""

import pytest
from pydantic import ValidationError

from src.models.parameters.compressor import CompressorParameters
from tests.test_parameters.fixtures import (
    VALID_COMPRESSOR_MINIMAL,
    VALID_COMPRESSOR_EDGE_CASES,
    VALID_COMPRESSOR_ALL_MAX,
    VALID_COMPRESSOR_GENTLE,
    VALID_COMPRESSOR_AGGRESSIVE,
    INVALID_COMPRESSOR_THRESHOLD_TOO_LOW,
    INVALID_COMPRESSOR_THRESHOLD_TOO_HIGH,
    INVALID_COMPRESSOR_RATIO_TOO_LOW,
    INVALID_COMPRESSOR_RATIO_TOO_HIGH,
    INVALID_COMPRESSOR_ATTACK_TOO_LOW,
    INVALID_COMPRESSOR_ATTACK_TOO_HIGH,
    INVALID_COMPRESSOR_RELEASE_TOO_LOW,
    INVALID_COMPRESSOR_RELEASE_TOO_HIGH,
    INVALID_COMPRESSOR_KNEE_TOO_LOW,
    INVALID_COMPRESSOR_KNEE_TOO_HIGH,
    INVALID_COMPRESSOR_MAKEUP_GAIN_TOO_LOW,
    INVALID_COMPRESSOR_MAKEUP_GAIN_TOO_HIGH,
    INVALID_COMPRESSOR_MISSING_REQUIRED,
    INVALID_COMPRESSOR_WRONG_TYPE,
)


class TestCompressorParametersValidation:
    """Test CompressorParameters model validation."""

    def test_valid_compressor_minimal(self):
        """Valid compressor parameters should pass validation."""
        comp = CompressorParameters(**VALID_COMPRESSOR_MINIMAL)
        assert comp.threshold == -20.0
        assert comp.ratio == 4.0
        assert comp.attack == 5.0
        assert comp.release == 50.0
        assert comp.knee == 6.0
        assert comp.makeup_gain == 3.0
        print(f"✓ Valid compressor created: threshold={comp.threshold}, ratio={comp.ratio}:1")

    def test_valid_compressor_edge_cases(self):
        """Compressor with edge case values should pass validation."""
        comp = CompressorParameters(**VALID_COMPRESSOR_EDGE_CASES)
        assert comp.threshold == -60.0    # Min
        assert comp.ratio == 1.0          # Min
        assert comp.attack == 0.1         # Min
        assert comp.release == 10.0       # Min
        assert comp.knee == 0.0           # Min
        assert comp.makeup_gain == 0.0    # Min
        print(f"✓ Edge case values (all minimums) validated successfully")

    def test_valid_compressor_all_max(self):
        """Compressor with all maximum values should be valid."""
        comp = CompressorParameters(**VALID_COMPRESSOR_ALL_MAX)
        assert comp.threshold == 0.0
        assert comp.ratio == 20.0
        assert comp.attack == 100.0
        assert comp.release == 1000.0
        assert comp.knee == 12.0
        assert comp.makeup_gain == 24.0
        print(f"✓ All maximum values accepted")

    def test_valid_compressor_gentle(self):
        """Gentle compressor settings should be valid."""
        comp = CompressorParameters(**VALID_COMPRESSOR_GENTLE)
        assert comp.threshold == -10.0
        assert comp.ratio == 2.0
        assert comp.attack == 10.0
        assert comp.release == 100.0
        print(f"✓ Gentle compression settings (2:1 ratio) accepted")

    def test_valid_compressor_aggressive(self):
        """Aggressive compressor settings should be valid."""
        comp = CompressorParameters(**VALID_COMPRESSOR_AGGRESSIVE)
        assert comp.threshold == -30.0
        assert comp.ratio == 10.0
        assert comp.attack == 1.0
        assert comp.release == 20.0
        print(f"✓ Aggressive compression settings (10:1 ratio) accepted")

    # Threshold Tests
    def test_invalid_compressor_threshold_too_low(self):
        """Compressor with threshold below -60 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_THRESHOLD_TOO_LOW)

        error = exc_info.value
        print(f"✓ Threshold too low rejected (-70 < -60): {error}")
        assert "threshold" in str(error).lower()

    def test_invalid_compressor_threshold_too_high(self):
        """Compressor with threshold above 0 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_THRESHOLD_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Threshold too high rejected (5 > 0): {error}")
        assert "threshold" in str(error).lower()

    def test_compressor_threshold_boundary_min(self):
        """Compressor with threshold exactly at minimum (-60) should be valid."""
        comp = CompressorParameters(
            threshold=-60.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.threshold == -60.0
        print(f"✓ Threshold minimum boundary (-60 dB) accepted")

    def test_compressor_threshold_boundary_max(self):
        """Compressor with threshold exactly at maximum (0) should be valid."""
        comp = CompressorParameters(
            threshold=0.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.threshold == 0.0
        print(f"✓ Threshold maximum boundary (0 dB) accepted")

    # Ratio Tests
    def test_invalid_compressor_ratio_too_low(self):
        """Compressor with ratio below 1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_RATIO_TOO_LOW)

        error = exc_info.value
        print(f"✓ Ratio too low rejected (0.5 < 1): {error}")
        assert "ratio" in str(error).lower()

    def test_invalid_compressor_ratio_too_high(self):
        """Compressor with ratio above 20 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_RATIO_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Ratio too high rejected (25 > 20): {error}")
        assert "ratio" in str(error).lower()

    def test_compressor_ratio_boundary_min(self):
        """Compressor with ratio exactly at minimum (1:1) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=1.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.ratio == 1.0
        print(f"✓ Ratio minimum boundary (1:1) accepted")

    def test_compressor_ratio_boundary_max(self):
        """Compressor with ratio exactly at maximum (20:1) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=20.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.ratio == 20.0
        print(f"✓ Ratio maximum boundary (20:1) accepted")

    # Attack Tests
    def test_invalid_compressor_attack_too_low(self):
        """Compressor with attack below 0.1 ms should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_ATTACK_TOO_LOW)

        error = exc_info.value
        print(f"✓ Attack too low rejected (0.05 < 0.1): {error}")
        assert "attack" in str(error).lower()

    def test_invalid_compressor_attack_too_high(self):
        """Compressor with attack above 100 ms should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_ATTACK_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Attack too high rejected (150 > 100): {error}")
        assert "attack" in str(error).lower()

    def test_compressor_attack_boundary_min(self):
        """Compressor with attack exactly at minimum (0.1 ms) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=0.1,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.attack == 0.1
        print(f"✓ Attack minimum boundary (0.1 ms) accepted")

    def test_compressor_attack_boundary_max(self):
        """Compressor with attack exactly at maximum (100 ms) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=100.0,
            release=150.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.attack == 100.0
        print(f"✓ Attack maximum boundary (100 ms) accepted")

    # Release Tests
    def test_invalid_compressor_release_too_low(self):
        """Compressor with release below 10 ms should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_RELEASE_TOO_LOW)

        error = exc_info.value
        print(f"✓ Release too low rejected (5 < 10): {error}")
        assert "release" in str(error).lower()

    def test_invalid_compressor_release_too_high(self):
        """Compressor with release above 1000 ms should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_RELEASE_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Release too high rejected (1500 > 1000): {error}")
        assert "release" in str(error).lower()

    def test_compressor_release_boundary_min(self):
        """Compressor with release exactly at minimum (10 ms) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=10.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.release == 10.0
        print(f"✓ Release minimum boundary (10 ms) accepted")

    def test_compressor_release_boundary_max(self):
        """Compressor with release exactly at maximum (1000 ms) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=1000.0,
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.release == 1000.0
        print(f"✓ Release maximum boundary (1000 ms) accepted")

    # Knee Tests
    def test_invalid_compressor_knee_too_low(self):
        """Compressor with knee below 0 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_KNEE_TOO_LOW)

        error = exc_info.value
        print(f"✓ Knee too low rejected (-1 < 0): {error}")
        assert "knee" in str(error).lower()

    def test_invalid_compressor_knee_too_high(self):
        """Compressor with knee above 12 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_KNEE_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Knee too high rejected (15 > 12): {error}")
        assert "knee" in str(error).lower()

    def test_compressor_knee_hard(self):
        """Hard knee (0 dB) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=0.0,
            makeup_gain=3.0
        )
        assert comp.knee == 0.0
        print(f"✓ Hard knee (0 dB) accepted")

    def test_compressor_knee_soft(self):
        """Soft knee (12 dB) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=12.0,
            makeup_gain=3.0
        )
        assert comp.knee == 12.0
        print(f"✓ Soft knee (12 dB) accepted")

    # Makeup Gain Tests
    def test_invalid_compressor_makeup_gain_too_low(self):
        """Compressor with makeup_gain below 0 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_MAKEUP_GAIN_TOO_LOW)

        error = exc_info.value
        print(f"✓ Makeup gain too low rejected (-5 < 0): {error}")
        assert "makeup_gain" in str(error).lower()

    def test_invalid_compressor_makeup_gain_too_high(self):
        """Compressor with makeup_gain above 24 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_MAKEUP_GAIN_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Makeup gain too high rejected (30 > 24): {error}")
        assert "makeup_gain" in str(error).lower()

    def test_compressor_makeup_gain_boundary_min(self):
        """Compressor with makeup_gain exactly at minimum (0) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=0.0
        )
        assert comp.makeup_gain == 0.0
        print(f"✓ Makeup gain minimum boundary (0 dB) accepted")

    def test_compressor_makeup_gain_boundary_max(self):
        """Compressor with makeup_gain exactly at maximum (24) should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=24.0
        )
        assert comp.makeup_gain == 24.0
        print(f"✓ Makeup gain maximum boundary (24 dB) accepted")

    # Required Fields Tests
    def test_invalid_compressor_missing_threshold(self):
        """Compressor missing threshold field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(
                ratio=4.0,
                attack=5.0,
                release=50.0,
                knee=6.0,
                makeup_gain=3.0
            )

        error = exc_info.value
        print(f"✓ Missing threshold rejected: {error}")
        assert "threshold" in str(error).lower()

    def test_invalid_compressor_missing_ratio(self):
        """Compressor missing ratio field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(
                threshold=-20.0,
                attack=5.0,
                release=50.0,
                knee=6.0,
                makeup_gain=3.0
            )

        error = exc_info.value
        print(f"✓ Missing ratio rejected: {error}")
        assert "ratio" in str(error).lower()

    def test_invalid_compressor_missing_attack(self):
        """Compressor missing attack field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(
                threshold=-20.0,
                ratio=4.0,
                release=50.0,
                knee=6.0,
                makeup_gain=3.0
            )

        error = exc_info.value
        print(f"✓ Missing attack rejected: {error}")
        assert "attack" in str(error).lower()

    def test_invalid_compressor_missing_release(self):
        """Compressor missing release field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(
                threshold=-20.0,
                ratio=4.0,
                attack=5.0,
                knee=6.0,
                makeup_gain=3.0
            )

        error = exc_info.value
        print(f"✓ Missing release rejected: {error}")
        assert "release" in str(error).lower()

    def test_invalid_compressor_missing_knee(self):
        """Compressor missing knee field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(
                threshold=-20.0,
                ratio=4.0,
                attack=5.0,
                release=50.0,
                makeup_gain=3.0
            )

        error = exc_info.value
        print(f"✓ Missing knee rejected: {error}")
        assert "knee" in str(error).lower()

    def test_invalid_compressor_missing_makeup_gain(self):
        """Compressor missing makeup_gain field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(
                threshold=-20.0,
                ratio=4.0,
                attack=5.0,
                release=50.0,
                knee=6.0
            )

        error = exc_info.value
        print(f"✓ Missing makeup_gain rejected: {error}")
        assert "makeup_gain" in str(error).lower()

    # Type Validation Tests
    def test_invalid_compressor_wrong_type_threshold(self):
        """Compressor with string threshold should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(**INVALID_COMPRESSOR_WRONG_TYPE)

        error = exc_info.value
        print(f"✓ Wrong type for threshold rejected: {error}")
        assert "threshold" in str(error).lower()

    def test_invalid_compressor_wrong_type_ratio(self):
        """Compressor with string ratio should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            CompressorParameters(
                threshold=-20.0,
                ratio="4",
                attack=5.0,
                release=50.0,
                knee=6.0,
                makeup_gain=3.0
            )

        error = exc_info.value
        print(f"✓ Wrong type for ratio rejected: {error}")
        assert "ratio" in str(error).lower()

    # Serialization Tests
    def test_compressor_serialization(self):
        """Compressor parameters should be serializable to dict."""
        comp = CompressorParameters(**VALID_COMPRESSOR_MINIMAL)
        comp_dict = comp.model_dump()
        assert "threshold" in comp_dict
        assert "ratio" in comp_dict
        assert "attack" in comp_dict
        assert "release" in comp_dict
        assert "knee" in comp_dict
        assert "makeup_gain" in comp_dict
        print(f"✓ Compressor parameters serializable to dict: {list(comp_dict.keys())}")

    def test_compressor_values_accessible(self):
        """All compressor parameter values should be accessible."""
        comp = CompressorParameters(**VALID_COMPRESSOR_MINIMAL)
        assert comp.threshold == -20.0
        assert comp.ratio == 4.0
        assert comp.attack == 5.0
        assert comp.release == 50.0
        assert comp.knee == 6.0
        assert comp.makeup_gain == 3.0
        print(f"✓ All compressor values accessible")

    # Timing Relationship Tests
    def test_compressor_fast_attack_slow_release(self):
        """Fast attack with slow release should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=0.5,      # Fast attack
            release=500.0,   # Slow release
            knee=6.0,
            makeup_gain=3.0
        )
        assert comp.attack < comp.release
        print(f"✓ Fast attack ({comp.attack} ms) with slow release ({comp.release} ms) accepted")

    def test_compressor_slow_attack_fast_release(self):
        """Slow attack with fast release should be valid."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=50.0,     # Slow attack
            release=100.0,   # Fast release
            knee=6.0,
            makeup_gain=3.0
        )
        # Note: While unusual, this is technically valid
        print(f"✓ Slow attack ({comp.attack} ms) with fast release ({comp.release} ms) accepted")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
