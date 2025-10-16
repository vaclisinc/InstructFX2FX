"""Tests for EQ parameter schema validation.

Tests verify that EQBand and EQParameters models correctly validate:
- Parameter ranges (frequency, gain, q)
- List length constraints (3-10 bands)
- Required fields
- Type validation
- Edge cases (min/max values)
"""

import pytest
from pydantic import ValidationError

# Import will work once Stream A creates the models
try:
    from src.models.parameters.eq import EQBand, EQParameters
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    pytestmark = pytest.mark.skip(reason="EQ models not yet available from Stream A")

from tests.test_parameters.fixtures import (
    VALID_EQ_SINGLE_BAND,
    VALID_EQ_MINIMUM_BANDS,
    VALID_EQ_MAXIMUM_BANDS,
    VALID_EQ_EDGE_CASES,
    INVALID_EQ_TOO_FEW_BANDS,
    INVALID_EQ_TOO_MANY_BANDS,
    INVALID_EQ_FREQUENCY_TOO_LOW,
    INVALID_EQ_FREQUENCY_TOO_HIGH,
    INVALID_EQ_GAIN_TOO_LOW,
    INVALID_EQ_GAIN_TOO_HIGH,
    INVALID_EQ_Q_TOO_LOW,
    INVALID_EQ_Q_TOO_HIGH,
    INVALID_EQ_MISSING_REQUIRED_FIELD,
    INVALID_EQ_WRONG_TYPE,
)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestEQBandValidation:
    """Test EQBand model validation."""

    def test_valid_eq_band(self):
        """Valid EQ band parameters should pass validation."""
        band = EQBand(**VALID_EQ_SINGLE_BAND)
        assert band.frequency == 1000.0
        assert band.gain == 3.0
        assert band.q == 1.0
        print(f"✓ Valid EQ band created: frequency={band.frequency}, gain={band.gain}, q={band.q}")

    def test_eq_band_min_frequency(self):
        """EQ band with minimum frequency (20 Hz) should be valid."""
        band = EQBand(frequency=20.0, gain=0.0, q=1.0)
        assert band.frequency == 20.0
        print(f"✓ Minimum frequency accepted: {band.frequency} Hz")

    def test_eq_band_max_frequency(self):
        """EQ band with maximum frequency (20000 Hz) should be valid."""
        band = EQBand(frequency=20000.0, gain=0.0, q=1.0)
        assert band.frequency == 20000.0
        print(f"✓ Maximum frequency accepted: {band.frequency} Hz")

    def test_eq_band_frequency_below_min(self):
        """EQ band with frequency below 20 Hz should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=10.0, gain=0.0, q=1.0)

        error = exc_info.value
        print(f"✓ Frequency below minimum rejected: {error}")
        assert "frequency" in str(error).lower()

    def test_eq_band_frequency_above_max(self):
        """EQ band with frequency above 20000 Hz should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=25000.0, gain=0.0, q=1.0)

        error = exc_info.value
        print(f"✓ Frequency above maximum rejected: {error}")
        assert "frequency" in str(error).lower()

    def test_eq_band_min_gain(self):
        """EQ band with minimum gain (-12 dB) should be valid."""
        band = EQBand(frequency=1000.0, gain=-12.0, q=1.0)
        assert band.gain == -12.0
        print(f"✓ Minimum gain accepted: {band.gain} dB")

    def test_eq_band_max_gain(self):
        """EQ band with maximum gain (12 dB) should be valid."""
        band = EQBand(frequency=1000.0, gain=12.0, q=1.0)
        assert band.gain == 12.0
        print(f"✓ Maximum gain accepted: {band.gain} dB")

    def test_eq_band_gain_below_min(self):
        """EQ band with gain below -12 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, gain=-15.0, q=1.0)

        error = exc_info.value
        print(f"✓ Gain below minimum rejected: {error}")
        assert "gain" in str(error).lower()

    def test_eq_band_gain_above_max(self):
        """EQ band with gain above 12 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, gain=15.0, q=1.0)

        error = exc_info.value
        print(f"✓ Gain above maximum rejected: {error}")
        assert "gain" in str(error).lower()

    def test_eq_band_min_q(self):
        """EQ band with minimum Q factor (0.1) should be valid."""
        band = EQBand(frequency=1000.0, gain=0.0, q=0.1)
        assert band.q == 0.1
        print(f"✓ Minimum Q factor accepted: {band.q}")

    def test_eq_band_max_q(self):
        """EQ band with maximum Q factor (10) should be valid."""
        band = EQBand(frequency=1000.0, gain=0.0, q=10.0)
        assert band.q == 10.0
        print(f"✓ Maximum Q factor accepted: {band.q}")

    def test_eq_band_q_below_min(self):
        """EQ band with Q factor below 0.1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, gain=0.0, q=0.05)

        error = exc_info.value
        print(f"✓ Q factor below minimum rejected: {error}")
        assert "q" in str(error).lower()

    def test_eq_band_q_above_max(self):
        """EQ band with Q factor above 10 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, gain=0.0, q=15.0)

        error = exc_info.value
        print(f"✓ Q factor above maximum rejected: {error}")
        assert "q" in str(error).lower()

    def test_eq_band_missing_frequency(self):
        """EQ band missing frequency field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(gain=0.0, q=1.0)

        error = exc_info.value
        print(f"✓ Missing frequency rejected: {error}")
        assert "frequency" in str(error).lower()

    def test_eq_band_missing_gain(self):
        """EQ band missing gain field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, q=1.0)

        error = exc_info.value
        print(f"✓ Missing gain rejected: {error}")
        assert "gain" in str(error).lower()

    def test_eq_band_missing_q(self):
        """EQ band missing Q factor field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, gain=0.0)

        error = exc_info.value
        print(f"✓ Missing Q factor rejected: {error}")
        assert "q" in str(error).lower()

    def test_eq_band_wrong_type_frequency(self):
        """EQ band with string frequency should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency="1000", gain=0.0, q=1.0)

        error = exc_info.value
        print(f"✓ Wrong type for frequency rejected: {error}")
        assert "frequency" in str(error).lower()

    def test_eq_band_wrong_type_gain(self):
        """EQ band with string gain should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, gain="0", q=1.0)

        error = exc_info.value
        print(f"✓ Wrong type for gain rejected: {error}")
        assert "gain" in str(error).lower()

    def test_eq_band_wrong_type_q(self):
        """EQ band with string Q factor should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQBand(frequency=1000.0, gain=0.0, q="1.0")

        error = exc_info.value
        print(f"✓ Wrong type for Q factor rejected: {error}")
        assert "q" in str(error).lower()


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Models not available")
class TestEQParametersValidation:
    """Test EQParameters model validation."""

    def test_valid_eq_minimum_bands(self):
        """EQ with minimum bands (3) should pass validation."""
        eq = EQParameters(**VALID_EQ_MINIMUM_BANDS)
        assert len(eq.bands) == 3
        assert eq.type == "parametric"
        print(f"✓ EQ with {len(eq.bands)} bands (minimum) created successfully")

    def test_valid_eq_maximum_bands(self):
        """EQ with maximum bands (10) should pass validation."""
        eq = EQParameters(**VALID_EQ_MAXIMUM_BANDS)
        assert len(eq.bands) == 10
        assert eq.type == "parametric"
        print(f"✓ EQ with {len(eq.bands)} bands (maximum) created successfully")

    def test_valid_eq_edge_cases(self):
        """EQ with edge case values should pass validation."""
        eq = EQParameters(**VALID_EQ_EDGE_CASES)
        assert len(eq.bands) == 3
        # Verify edge case values
        assert eq.bands[0].frequency == 20.0      # Min frequency
        assert eq.bands[0].gain == -12.0          # Min gain
        assert eq.bands[0].q == 0.1               # Min q
        assert eq.bands[1].frequency == 20000.0   # Max frequency
        assert eq.bands[1].gain == 12.0           # Max gain
        assert eq.bands[1].q == 10.0              # Max q
        print(f"✓ EQ with edge case values validated successfully")

    def test_invalid_eq_too_few_bands(self):
        """EQ with fewer than 3 bands should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_TOO_FEW_BANDS)

        error = exc_info.value
        print(f"✓ Too few bands rejected (2 < 3 minimum): {error}")
        assert "bands" in str(error).lower()

    def test_invalid_eq_too_many_bands(self):
        """EQ with more than 10 bands should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_TOO_MANY_BANDS)

        error = exc_info.value
        print(f"✓ Too many bands rejected (11 > 10 maximum): {error}")
        assert "bands" in str(error).lower()

    def test_invalid_eq_frequency_too_low(self):
        """EQ with frequency below 20 Hz should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_FREQUENCY_TOO_LOW)

        error = exc_info.value
        print(f"✓ Frequency too low rejected: {error}")
        assert "frequency" in str(error).lower()

    def test_invalid_eq_frequency_too_high(self):
        """EQ with frequency above 20000 Hz should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_FREQUENCY_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Frequency too high rejected: {error}")
        assert "frequency" in str(error).lower()

    def test_invalid_eq_gain_too_low(self):
        """EQ with gain below -12 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_GAIN_TOO_LOW)

        error = exc_info.value
        print(f"✓ Gain too low rejected: {error}")
        assert "gain" in str(error).lower()

    def test_invalid_eq_gain_too_high(self):
        """EQ with gain above 12 dB should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_GAIN_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Gain too high rejected: {error}")
        assert "gain" in str(error).lower()

    def test_invalid_eq_q_too_low(self):
        """EQ with Q factor below 0.1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_Q_TOO_LOW)

        error = exc_info.value
        print(f"✓ Q factor too low rejected: {error}")
        assert "q" in str(error).lower()

    def test_invalid_eq_q_too_high(self):
        """EQ with Q factor above 10 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(**INVALID_EQ_Q_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Q factor too high rejected: {error}")
        assert "q" in str(error).lower()

    def test_invalid_eq_missing_bands(self):
        """EQ missing bands field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(type="parametric")

        error = exc_info.value
        print(f"✓ Missing bands field rejected: {error}")
        assert "bands" in str(error).lower()

    def test_invalid_eq_empty_bands(self):
        """EQ with empty bands list should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EQParameters(bands=[], type="parametric")

        error = exc_info.value
        print(f"✓ Empty bands list rejected: {error}")
        assert "bands" in str(error).lower()

    def test_eq_type_default_value(self):
        """EQ type should default to 'parametric' if not specified."""
        eq = EQParameters(bands=VALID_EQ_MINIMUM_BANDS["bands"])
        assert eq.type == "parametric"
        print(f"✓ EQ type defaults to 'parametric'")

    def test_eq_type_custom_value(self):
        """EQ should accept custom type value."""
        eq = EQParameters(
            bands=VALID_EQ_MINIMUM_BANDS["bands"],
            type="graphic"
        )
        assert eq.type == "graphic"
        print(f"✓ Custom EQ type accepted: {eq.type}")

    def test_eq_band_values_accessible(self):
        """Individual band values should be accessible."""
        eq = EQParameters(**VALID_EQ_MINIMUM_BANDS)
        first_band = eq.bands[0]
        assert first_band.frequency == 100.0
        assert first_band.gain == 2.0
        assert first_band.q == 0.7
        print(f"✓ Band values accessible: freq={first_band.frequency}, gain={first_band.gain}, q={first_band.q}")

    def test_eq_serialization(self):
        """EQ parameters should be serializable to dict."""
        eq = EQParameters(**VALID_EQ_MINIMUM_BANDS)
        eq_dict = eq.model_dump()
        assert "bands" in eq_dict
        assert "type" in eq_dict
        assert len(eq_dict["bands"]) == 3
        print(f"✓ EQ parameters serializable to dict: {list(eq_dict.keys())}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
