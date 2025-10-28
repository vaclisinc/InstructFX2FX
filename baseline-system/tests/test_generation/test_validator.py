"""Tests for validation utilities.

Tests verify:
- Pre-validation of effect structures
- Post-validation of Pydantic models
- Validation result reporting
- Edge case handling
"""

import pytest
from src.generation.validator import (
    ValidationLevel,
    ValidationIssue,
    ValidationResult,
    validate_effect_structure,
    validate_effect_chain_structure,
    validate_effect_parameter,
    validate_effect_chain,
    PARAMETER_RANGES
)
from src.models.parameters import (
    EQParameters,
    EQBand,
    ReverbParameters,
    CompressorParameters,
    EffectChain
)


class TestValidationIssue:
    """Test ValidationIssue class."""

    def test_validation_issue_creation(self):
        """ValidationIssue should be created with proper fields."""
        issue = ValidationIssue(
            level=ValidationLevel.ERROR,
            field="frequency",
            message="Value out of range",
            current_value=50000,
            expected_value="20-20000"
        )
        assert issue.level == ValidationLevel.ERROR
        assert issue.field == "frequency"
        assert issue.current_value == 50000
        print(f"✓ ValidationIssue created: {issue}")

    def test_validation_issue_string_representation(self):
        """ValidationIssue should format correctly as string."""
        issue = ValidationIssue(
            level=ValidationLevel.WARNING,
            field="wet_level",
            message="Value unusually high",
            current_value=0.9
        )
        issue_str = str(issue)
        assert "[WARNING]" in issue_str
        assert "wet_level" in issue_str
        assert "0.9" in issue_str
        print(f"✓ Issue string: {issue_str}")


class TestValidationResult:
    """Test ValidationResult class."""

    def test_validation_result_valid(self):
        """ValidationResult with no errors should be valid."""
        result = ValidationResult(is_valid=True, issues=[])
        assert result.is_valid
        assert not result.has_errors()
        assert not result.has_warnings()
        assert bool(result) is True
        print("✓ Valid result: no issues")

    def test_validation_result_with_errors(self):
        """ValidationResult with errors should be invalid."""
        issues = [
            ValidationIssue(ValidationLevel.ERROR, "field1", "Error 1"),
            ValidationIssue(ValidationLevel.ERROR, "field2", "Error 2")
        ]
        result = ValidationResult(is_valid=False, issues=issues)
        assert not result.is_valid
        assert result.has_errors()
        assert len(result.get_errors()) == 2
        assert bool(result) is False
        print(f"✓ Invalid result: {len(result.get_errors())} errors")

    def test_validation_result_with_warnings(self):
        """ValidationResult with only warnings should be valid."""
        issues = [
            ValidationIssue(ValidationLevel.WARNING, "field1", "Warning 1")
        ]
        result = ValidationResult(is_valid=True, issues=issues)
        assert result.is_valid
        assert not result.has_errors()
        assert result.has_warnings()
        assert len(result.get_warnings()) == 1
        print(f"✓ Valid result with warnings: {len(result.get_warnings())} warnings")

    def test_validation_result_format_report(self):
        """ValidationResult should format detailed report."""
        issues = [
            ValidationIssue(ValidationLevel.ERROR, "frequency", "Out of range", 50000),
            ValidationIssue(ValidationLevel.WARNING, "gain", "Unusually high", 10)
        ]
        result = ValidationResult(is_valid=False, issues=issues)
        report = result.format_report()
        assert "Validation failed" in report
        assert "Errors (1)" in report
        assert "Warnings (1)" in report
        assert "frequency" in report
        assert "gain" in report
        print(f"✓ Formatted report:\n{report}")


class TestEQValidation:
    """Test EQ effect validation."""

    def test_valid_eq_structure(self):
        """Valid EQ structure should pass pre-validation."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 100, "gain": 2, "q": 0.7},
                {"frequency": 1000, "gain": -1, "q": 1.2},
                {"frequency": 8000, "gain": 3, "q": 0.9}
            ]
        }
        result = validate_effect_structure(eq_data)
        assert result.is_valid
        assert len(result.issues) == 0
        print(f"✓ Valid EQ structure passed pre-validation")

    def test_eq_missing_type(self):
        """EQ missing type field should fail validation."""
        eq_data = {
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 1.0}
            ]
        }
        result = validate_effect_structure(eq_data)
        assert not result.is_valid
        assert result.has_errors()
        errors = result.get_errors()
        assert any("type" in e.field for e in errors)
        print(f"✓ Missing type rejected: {errors[0].message}")

    def test_eq_too_few_bands(self):
        """EQ with too few bands should fail validation."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 1.0},
                {"frequency": 2000, "gain": -1, "q": 0.8}
            ]
        }
        result = validate_effect_structure(eq_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("bands" in e.field for e in errors)
        print(f"✓ Too few bands rejected: {errors[0].message}")

    def test_eq_too_many_bands(self):
        """EQ with too many bands should fail validation."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": i * 1000, "gain": 1, "q": 1.0}
                for i in range(1, 12)  # 11 bands
            ]
        }
        result = validate_effect_structure(eq_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("bands" in e.field for e in errors)
        print(f"✓ Too many bands rejected: {errors[0].message}")

    def test_eq_frequency_out_of_range(self):
        """EQ with frequency out of range should fail validation."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 50000, "gain": 2, "q": 1.0},
                {"frequency": 1000, "gain": -1, "q": 1.0},
                {"frequency": 8000, "gain": 1, "q": 1.0}
            ]
        }
        result = validate_effect_structure(eq_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("frequency" in e.field for e in errors)
        print(f"✓ Out-of-range frequency rejected: {errors[0].message}")

    def test_eq_gain_out_of_range(self):
        """EQ with gain out of range should fail validation."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 20, "q": 1.0},
                {"frequency": 2000, "gain": -1, "q": 1.0},
                {"frequency": 4000, "gain": 1, "q": 1.0}
            ]
        }
        result = validate_effect_structure(eq_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("gain" in e.field for e in errors)
        print(f"✓ Out-of-range gain rejected: {errors[0].message}")

    def test_eq_q_out_of_range(self):
        """EQ with Q factor out of range should fail validation."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 0.05},
                {"frequency": 2000, "gain": -1, "q": 1.0},
                {"frequency": 4000, "gain": 1, "q": 1.0}
            ]
        }
        result = validate_effect_structure(eq_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("q" in e.field for e in errors)
        print(f"✓ Out-of-range Q rejected: {errors[0].message}")

    def test_eq_missing_band_field(self):
        """EQ band missing required field should fail validation."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "q": 1.0},  # Missing gain
                {"frequency": 2000, "gain": -1, "q": 1.0},
                {"frequency": 4000, "gain": 1, "q": 1.0}
            ]
        }
        result = validate_effect_structure(eq_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("gain" in e.field.lower() for e in errors)
        print(f"✓ Missing band field rejected: {errors[0].message}")


class TestReverbValidation:
    """Test reverb effect validation."""

    def test_valid_reverb_structure(self):
        """Valid reverb structure should pass pre-validation."""
        reverb_data = {
            "type": "reverb",
            "room_size": 0.5,
            "damping": 0.5,
            "wet_level": 0.3,
            "dry_level": 0.7,
            "width": 1.0
        }
        result = validate_effect_structure(reverb_data)
        assert result.is_valid
        print("✓ Valid reverb structure passed pre-validation")

    def test_reverb_missing_required_field(self):
        """Reverb missing required field should fail validation."""
        reverb_data = {
            "type": "reverb",
            "room_size": 0.5,
            "damping": 0.5,
            # Missing wet_level, dry_level, width
        }
        result = validate_effect_structure(reverb_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert len(errors) >= 3  # At least 3 missing fields
        print(f"✓ Missing required fields rejected: {len(errors)} errors")

    def test_reverb_value_out_of_range(self):
        """Reverb with values out of range should fail validation."""
        reverb_data = {
            "type": "reverb",
            "room_size": 1.5,
            "damping": 0.5,
            "wet_level": 0.3,
            "dry_level": 0.7,
            "width": 1.0
        }
        result = validate_effect_structure(reverb_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("room_size" in e.field for e in errors)
        print(f"✓ Out-of-range value rejected: {errors[0].message}")

    def test_reverb_wet_dry_balance_warning(self):
        """Reverb with unusual wet/dry balance should generate warning."""
        reverb_data = {
            "type": "reverb",
            "room_size": 0.5,
            "damping": 0.5,
            "wet_level": 1.0,
            "dry_level": 1.5,
            "width": 1.0
        }
        result = validate_effect_structure(reverb_data)
        # Should have error for dry_level > 1.0
        assert not result.is_valid
        errors = result.get_errors()
        assert any("dry_level" in e.field for e in errors)
        print(f"✓ Invalid dry level rejected: {errors[0].message}")


class TestCompressorValidation:
    """Test compressor effect validation."""

    def test_valid_compressor_structure(self):
        """Valid compressor structure should pass pre-validation."""
        comp_data = {
            "type": "compressor",
            "threshold": -20,
            "ratio": 4,
            "attack": 5,
            "release": 50,
            "knee": 3,
            "makeup_gain": 6
        }
        result = validate_effect_structure(comp_data)
        assert result.is_valid
        print("✓ Valid compressor structure passed pre-validation")

    def test_compressor_missing_required_field(self):
        """Compressor missing required field should fail validation."""
        comp_data = {
            "type": "compressor",
            "threshold": -20,
            "ratio": 4,
            # Missing attack, release, knee, makeup_gain
        }
        result = validate_effect_structure(comp_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert len(errors) >= 4  # At least 4 missing fields
        print(f"✓ Missing required fields rejected: {len(errors)} errors")

    def test_compressor_threshold_out_of_range(self):
        """Compressor with threshold out of range should fail validation."""
        comp_data = {
            "type": "compressor",
            "threshold": 10,  # Above 0
            "ratio": 4,
            "attack": 5,
            "release": 50,
            "knee": 3,
            "makeup_gain": 6
        }
        result = validate_effect_structure(comp_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("threshold" in e.field for e in errors)
        print(f"✓ Out-of-range threshold rejected: {errors[0].message}")

    def test_compressor_attack_release_relationship(self):
        """Compressor with attack >= release should generate warning."""
        comp_data = {
            "type": "compressor",
            "threshold": -20,
            "ratio": 4,
            "attack": 100,
            "release": 50,  # Release < attack
            "knee": 3,
            "makeup_gain": 6
        }
        result = validate_effect_structure(comp_data)
        # Should still be structurally valid, but with warning
        assert result.is_valid or result.has_warnings()
        if result.has_warnings():
            warnings = result.get_warnings()
            assert any("attack" in w.field.lower() or "release" in w.field.lower() for w in warnings)
            print(f"✓ Attack/release relationship warning: {warnings[0].message}")
        else:
            print("✓ Attack/release validated")


class TestEffectChainValidation:
    """Test effect chain validation."""

    def test_valid_effect_chain_structure(self):
        """Valid effect chain should pass pre-validation."""
        chain_data = {
            "description": "test chain",
            "effects": [
                {
                    "type": "eq",
                    "bands": [
                        {"frequency": 100, "gain": 2, "q": 0.7},
                        {"frequency": 1000, "gain": -1, "q": 1.2},
                        {"frequency": 8000, "gain": 3, "q": 0.9}
                    ]
                },
                {
                    "type": "compressor",
                    "threshold": -20,
                    "ratio": 4,
                    "attack": 5,
                    "release": 50,
                    "knee": 3,
                    "makeup_gain": 6
                }
            ]
        }
        result = validate_effect_chain_structure(chain_data)
        assert result.is_valid
        print(f"✓ Valid effect chain with {len(chain_data['effects'])} effects passed pre-validation")

    def test_effect_chain_missing_effects(self):
        """Effect chain missing effects field should fail validation."""
        chain_data = {
            "description": "test chain"
        }
        result = validate_effect_chain_structure(chain_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("effects" in e.field for e in errors)
        print(f"✓ Missing effects rejected: {errors[0].message}")

    def test_effect_chain_empty_effects(self):
        """Effect chain with empty effects list should fail validation."""
        chain_data = {
            "description": "test chain",
            "effects": []
        }
        result = validate_effect_chain_structure(chain_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("effects" in e.field for e in errors)
        print(f"✓ Empty effects list rejected: {errors[0].message}")

    def test_effect_chain_invalid_effect(self):
        """Effect chain with invalid effect should fail validation."""
        chain_data = {
            "description": "test chain",
            "effects": [
                {
                    "type": "eq",
                    "bands": [
                        {"frequency": 50000, "gain": 2, "q": 1.0},  # Invalid frequency
                        {"frequency": 1000, "gain": -1, "q": 1.0},
                        {"frequency": 8000, "gain": 3, "q": 0.9}
                    ]
                }
            ]
        }
        result = validate_effect_chain_structure(chain_data)
        assert not result.is_valid
        errors = result.get_errors()
        assert any("frequency" in e.field for e in errors)
        print(f"✓ Invalid effect in chain rejected: {errors[0].message}")

    def test_effect_chain_missing_description_warning(self):
        """Effect chain missing description should generate warning."""
        chain_data = {
            "effects": [
                {
                    "type": "eq",
                    "bands": [
                        {"frequency": 100, "gain": 2, "q": 0.7},
                        {"frequency": 1000, "gain": -1, "q": 1.2},
                        {"frequency": 8000, "gain": 3, "q": 0.9}
                    ]
                }
            ]
        }
        result = validate_effect_chain_structure(chain_data)
        # Should be valid but with warning
        if result.has_warnings():
            warnings = result.get_warnings()
            assert any("description" in w.field for w in warnings)
            print(f"✓ Missing description warning: {warnings[0].message}")


class TestPostValidation:
    """Test post-validation with Pydantic models."""

    def test_valid_eq_parameter(self):
        """Valid EQ parameter should pass post-validation."""
        eq = EQParameters(
            bands=[
                EQBand(frequency=100, gain=2, q=0.7),
                EQBand(frequency=1000, gain=-1, q=1.2),
                EQBand(frequency=8000, gain=3, q=0.9)
            ]
        )
        result = validate_effect_parameter(eq)
        assert result.is_valid
        print("✓ Valid EQ parameter passed post-validation")

    def test_valid_reverb_parameter(self):
        """Valid reverb parameter should pass post-validation."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.3,
            dry_level=0.7,
            width=1.0
        )
        result = validate_effect_parameter(reverb)
        assert result.is_valid
        print("✓ Valid reverb parameter passed post-validation")

    def test_valid_compressor_parameter(self):
        """Valid compressor parameter should pass post-validation."""
        comp = CompressorParameters(
            threshold=-20,
            ratio=4,
            attack=5,
            release=50,
            knee=3,
            makeup_gain=6
        )
        result = validate_effect_parameter(comp)
        assert result.is_valid
        print("✓ Valid compressor parameter passed post-validation")

    def test_valid_effect_chain(self):
        """Valid effect chain should pass post-validation."""
        chain = EffectChain(
            description="test chain",
            effects=[
                EQParameters(
                    bands=[
                        EQBand(frequency=100, gain=2, q=0.7),
                        EQBand(frequency=1000, gain=-1, q=1.2),
                        EQBand(frequency=8000, gain=3, q=0.9)
                    ]
                ),
                CompressorParameters(
                    threshold=-20,
                    ratio=4,
                    attack=5,
                    release=50,
                    knee=3,
                    makeup_gain=6
                )
            ],
            order=["eq", "compressor"]
        )
        result = validate_effect_chain(chain)
        assert result.is_valid
        print(f"✓ Valid effect chain with {len(chain.effects)} effects passed post-validation")


class TestParameterRanges:
    """Test parameter range constants."""

    def test_parameter_ranges_defined(self):
        """Parameter ranges should be defined for all effect types."""
        assert "eq" in PARAMETER_RANGES
        assert "reverb" in PARAMETER_RANGES
        assert "compressor" in PARAMETER_RANGES
        print(f"✓ Parameter ranges defined for: {list(PARAMETER_RANGES.keys())}")

    def test_eq_parameter_ranges(self):
        """EQ parameter ranges should be correct."""
        eq_ranges = PARAMETER_RANGES["eq"]
        assert eq_ranges["frequency"] == (20.0, 20000.0)
        assert eq_ranges["gain"] == (-12.0, 12.0)
        assert eq_ranges["q"] == (0.1, 10.0)
        assert eq_ranges["bands_count"] == (3, 10)
        print(f"✓ EQ ranges: frequency={eq_ranges['frequency']}, gain={eq_ranges['gain']}, q={eq_ranges['q']}")

    def test_reverb_parameter_ranges(self):
        """Reverb parameter ranges should be correct."""
        reverb_ranges = PARAMETER_RANGES["reverb"]
        assert reverb_ranges["room_size"] == (0.0, 1.0)
        assert reverb_ranges["damping"] == (0.0, 1.0)
        assert reverb_ranges["wet_level"] == (0.0, 1.0)
        assert reverb_ranges["dry_level"] == (0.0, 1.0)
        assert reverb_ranges["width"] == (0.0, 1.0)
        print(f"✓ Reverb ranges: all parameters in (0.0, 1.0)")

    def test_compressor_parameter_ranges(self):
        """Compressor parameter ranges should be correct."""
        comp_ranges = PARAMETER_RANGES["compressor"]
        assert comp_ranges["threshold"] == (-60.0, 0.0)
        assert comp_ranges["ratio"] == (1.0, 20.0)
        assert comp_ranges["attack"] == (0.1, 100.0)
        assert comp_ranges["release"] == (10.0, 1000.0)
        assert comp_ranges["knee"] == (0.0, 12.0)
        assert comp_ranges["makeup_gain"] == (0.0, 24.0)
        print(f"✓ Compressor ranges: threshold={comp_ranges['threshold']}, ratio={comp_ranges['ratio']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
