"""Tests for Reverb parameter schema validation.

Tests verify that ReverbParameters model correctly validates:
- Parameter ranges (room_size, damping, wet_level, dry_level, width: 0-1)
- Required fields
- Type validation (including boolean freeze_mode)
- Edge cases (min/max values)
"""

import pytest
from pydantic import ValidationError

from src.models.parameters.reverb import ReverbParameters
from tests.test_parameters.fixtures import (
    VALID_REVERB_MINIMAL,
    VALID_REVERB_EDGE_CASES,
    VALID_REVERB_ALL_MAX,
    VALID_REVERB_ALL_MIN,
    INVALID_REVERB_ROOM_SIZE_TOO_LOW,
    INVALID_REVERB_ROOM_SIZE_TOO_HIGH,
    INVALID_REVERB_DAMPING_TOO_LOW,
    INVALID_REVERB_DAMPING_TOO_HIGH,
    INVALID_REVERB_WET_LEVEL_TOO_LOW,
    INVALID_REVERB_WET_LEVEL_TOO_HIGH,
    INVALID_REVERB_DRY_LEVEL_TOO_LOW,
    INVALID_REVERB_DRY_LEVEL_TOO_HIGH,
    INVALID_REVERB_WIDTH_TOO_LOW,
    INVALID_REVERB_WIDTH_TOO_HIGH,
    INVALID_REVERB_MISSING_REQUIRED,
    INVALID_REVERB_WRONG_TYPE,
    INVALID_REVERB_FREEZE_WRONG_TYPE,
)


class TestReverbParametersValidation:
    """Test ReverbParameters model validation."""

    def test_valid_reverb_minimal(self):
        """Valid reverb parameters should pass validation."""
        reverb = ReverbParameters(**VALID_REVERB_MINIMAL)
        assert reverb.room_size == 0.5
        assert reverb.damping == 0.5
        assert reverb.wet_level == 0.33
        assert reverb.dry_level == 0.67
        assert reverb.width == 1.0
        assert reverb.freeze_mode is False
        print(f"✓ Valid reverb created: room_size={reverb.room_size}, damping={reverb.damping}")

    def test_valid_reverb_all_min(self):
        """Reverb with all minimum values (0.0) should be valid."""
        reverb = ReverbParameters(**VALID_REVERB_ALL_MIN)
        assert reverb.room_size == 0.0
        assert reverb.damping == 0.0
        assert reverb.wet_level == 0.0
        assert reverb.dry_level == 0.0
        assert reverb.width == 0.0
        assert reverb.freeze_mode is False
        print(f"✓ All minimum values accepted")

    def test_valid_reverb_all_max(self):
        """Reverb with all maximum values (1.0) should be valid."""
        reverb = ReverbParameters(**VALID_REVERB_ALL_MAX)
        assert reverb.room_size == 1.0
        assert reverb.damping == 1.0
        assert reverb.wet_level == 1.0
        assert reverb.dry_level == 1.0
        assert reverb.width == 1.0
        assert reverb.freeze_mode is True
        print(f"✓ All maximum values accepted")

    def test_valid_reverb_edge_cases(self):
        """Reverb with edge case values should pass validation."""
        reverb = ReverbParameters(**VALID_REVERB_EDGE_CASES)
        assert reverb.room_size == 0.0
        assert reverb.damping == 1.0
        assert reverb.wet_level == 0.0
        assert reverb.dry_level == 1.0
        assert reverb.width == 0.0
        print(f"✓ Edge case values validated successfully")

    # Room Size Tests
    def test_invalid_reverb_room_size_too_low(self):
        """Reverb with room_size below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_ROOM_SIZE_TOO_LOW)

        error = exc_info.value
        print(f"✓ Room size too low rejected (-0.1 < 0): {error}")
        assert "room_size" in str(error).lower()

    def test_invalid_reverb_room_size_too_high(self):
        """Reverb with room_size above 1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_ROOM_SIZE_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Room size too high rejected (1.5 > 1): {error}")
        assert "room_size" in str(error).lower()

    def test_reverb_room_size_boundary_min(self):
        """Reverb with room_size exactly at minimum (0) should be valid."""
        reverb = ReverbParameters(
            room_size=0.0,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0
        )
        assert reverb.room_size == 0.0
        print(f"✓ Room size minimum boundary (0.0) accepted")

    def test_reverb_room_size_boundary_max(self):
        """Reverb with room_size exactly at maximum (1) should be valid."""
        reverb = ReverbParameters(
            room_size=1.0,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0
        )
        assert reverb.room_size == 1.0
        print(f"✓ Room size maximum boundary (1.0) accepted")

    # Damping Tests
    def test_invalid_reverb_damping_too_low(self):
        """Reverb with damping below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_DAMPING_TOO_LOW)

        error = exc_info.value
        print(f"✓ Damping too low rejected (-0.1 < 0): {error}")
        assert "damping" in str(error).lower()

    def test_invalid_reverb_damping_too_high(self):
        """Reverb with damping above 1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_DAMPING_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Damping too high rejected (1.5 > 1): {error}")
        assert "damping" in str(error).lower()

    # Wet Level Tests
    def test_invalid_reverb_wet_level_too_low(self):
        """Reverb with wet_level below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_WET_LEVEL_TOO_LOW)

        error = exc_info.value
        print(f"✓ Wet level too low rejected (-0.1 < 0): {error}")
        assert "wet_level" in str(error).lower()

    def test_invalid_reverb_wet_level_too_high(self):
        """Reverb with wet_level above 1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_WET_LEVEL_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Wet level too high rejected (1.5 > 1): {error}")
        assert "wet_level" in str(error).lower()

    # Dry Level Tests
    def test_invalid_reverb_dry_level_too_low(self):
        """Reverb with dry_level below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_DRY_LEVEL_TOO_LOW)

        error = exc_info.value
        print(f"✓ Dry level too low rejected (-0.1 < 0): {error}")
        assert "dry_level" in str(error).lower()

    def test_invalid_reverb_dry_level_too_high(self):
        """Reverb with dry_level above 1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_DRY_LEVEL_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Dry level too high rejected (1.5 > 1): {error}")
        assert "dry_level" in str(error).lower()

    # Width Tests
    def test_invalid_reverb_width_too_low(self):
        """Reverb with width below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_WIDTH_TOO_LOW)

        error = exc_info.value
        print(f"✓ Width too low rejected (-0.1 < 0): {error}")
        assert "width" in str(error).lower()

    def test_invalid_reverb_width_too_high(self):
        """Reverb with width above 1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_WIDTH_TOO_HIGH)

        error = exc_info.value
        print(f"✓ Width too high rejected (1.5 > 1): {error}")
        assert "width" in str(error).lower()

    # Required Fields Tests
    def test_invalid_reverb_missing_room_size(self):
        """Reverb missing room_size field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(
                damping=0.5,
                wet_level=0.33,
                dry_level=0.67,
                width=1.0
            )

        error = exc_info.value
        print(f"✓ Missing room_size rejected: {error}")
        assert "room_size" in str(error).lower()

    def test_invalid_reverb_missing_damping(self):
        """Reverb missing damping field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(
                room_size=0.5,
                wet_level=0.33,
                dry_level=0.67,
                width=1.0
            )

        error = exc_info.value
        print(f"✓ Missing damping rejected: {error}")
        assert "damping" in str(error).lower()

    def test_invalid_reverb_missing_wet_level(self):
        """Reverb missing wet_level field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(
                room_size=0.5,
                damping=0.5,
                dry_level=0.67,
                width=1.0
            )

        error = exc_info.value
        print(f"✓ Missing wet_level rejected: {error}")
        assert "wet_level" in str(error).lower()

    def test_invalid_reverb_missing_dry_level(self):
        """Reverb missing dry_level field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(
                room_size=0.5,
                damping=0.5,
                wet_level=0.33,
                width=1.0
            )

        error = exc_info.value
        print(f"✓ Missing dry_level rejected: {error}")
        assert "dry_level" in str(error).lower()

    def test_invalid_reverb_missing_width(self):
        """Reverb missing width field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(
                room_size=0.5,
                damping=0.5,
                wet_level=0.33,
                dry_level=0.67
            )

        error = exc_info.value
        print(f"✓ Missing width rejected: {error}")
        assert "width" in str(error).lower()

    # Type Validation Tests
    def test_invalid_reverb_wrong_type_room_size(self):
        """Reverb with string room_size should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(
                room_size="0.5",
                damping=0.5,
                wet_level=0.33,
                dry_level=0.67,
                width=1.0
            )

        error = exc_info.value
        print(f"✓ Wrong type for room_size rejected: {error}")
        assert "room_size" in str(error).lower()

    def test_invalid_reverb_freeze_mode_wrong_type(self):
        """Reverb with string freeze_mode should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ReverbParameters(**INVALID_REVERB_FREEZE_WRONG_TYPE)

        error = exc_info.value
        print(f"✓ Wrong type for freeze_mode rejected: {error}")
        assert "freeze_mode" in str(error).lower()

    # Freeze Mode Tests
    def test_reverb_freeze_mode_default(self):
        """Reverb freeze_mode should default to False if not specified."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0
        )
        assert reverb.freeze_mode is False
        print(f"✓ Freeze mode defaults to False")

    def test_reverb_freeze_mode_true(self):
        """Reverb freeze_mode can be set to True."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0,
            freeze_mode=True
        )
        assert reverb.freeze_mode is True
        print(f"✓ Freeze mode accepts True")

    def test_reverb_freeze_mode_false(self):
        """Reverb freeze_mode can be explicitly set to False."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0,
            freeze_mode=False
        )
        assert reverb.freeze_mode is False
        print(f"✓ Freeze mode accepts False")

    # Serialization Tests
    def test_reverb_serialization(self):
        """Reverb parameters should be serializable to dict."""
        reverb = ReverbParameters(**VALID_REVERB_MINIMAL)
        reverb_dict = reverb.model_dump()
        assert "room_size" in reverb_dict
        assert "damping" in reverb_dict
        assert "wet_level" in reverb_dict
        assert "dry_level" in reverb_dict
        assert "width" in reverb_dict
        assert "freeze_mode" in reverb_dict
        print(f"✓ Reverb parameters serializable to dict: {list(reverb_dict.keys())}")

    def test_reverb_values_accessible(self):
        """All reverb parameter values should be accessible."""
        reverb = ReverbParameters(**VALID_REVERB_MINIMAL)
        assert reverb.room_size == 0.5
        assert reverb.damping == 0.5
        assert reverb.wet_level == 0.33
        assert reverb.dry_level == 0.67
        assert reverb.width == 1.0
        assert reverb.freeze_mode is False
        print(f"✓ All reverb values accessible")

    # Wet/Dry Balance Tests
    def test_reverb_wet_dry_balance_typical(self):
        """Typical wet/dry balance should be valid."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.3,   # 30% wet
            dry_level=0.7,   # 70% dry
            width=1.0
        )
        assert reverb.wet_level + reverb.dry_level == 1.0
        print(f"✓ Typical wet/dry balance accepted (30/70)")

    def test_reverb_all_wet(self):
        """All wet (100% wet, 0% dry) should be valid."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=1.0,
            dry_level=0.0,
            width=1.0
        )
        assert reverb.wet_level == 1.0
        assert reverb.dry_level == 0.0
        print(f"✓ All wet configuration accepted")

    def test_reverb_all_dry(self):
        """All dry (0% wet, 100% dry) should be valid."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.0,
            dry_level=1.0,
            width=1.0
        )
        assert reverb.wet_level == 0.0
        assert reverb.dry_level == 1.0
        print(f"✓ All dry configuration accepted")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
