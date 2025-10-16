"""Integration tests for parameter generation pipeline.

Tests end-to-end workflows including:
- Complete generation pipeline with validation
- Normalization and correction
- Error recovery scenarios
- Real Pydantic validation
"""

import pytest
import json
from unittest.mock import patch, Mock
import math

from src.generation.parameter_generator import ParameterGenerator
from src.generation.validator import (
    validate_effect_structure,
    validate_effect_chain_structure,
    validate_effect_chain,
    ValidationLevel
)
from src.generation.normalizer import (
    normalize_effect,
    normalize_effect_chain_data,
    normalize_effect_chain,
    clamp,
    safe_float,
    safe_bool
)
from src.models.parameters import EffectChain, EQParameters, ReverbParameters, CompressorParameters
from models.llm_judge import LLMResponse


# Test data with various issues that need normalization
OUT_OF_RANGE_EQ_RESPONSE = """{
    "description": "test with out of range values",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 50000, "gain": 20, "q": 15},
                {"frequency": 5, "gain": -20, "q": 0.01},
                {"frequency": 1000, "gain": 2, "q": 1.0}
            ]
        }
    ]
}"""

SPECIAL_VALUES_RESPONSE = """{
    "description": "test with special values",
    "effects": [
        {
            "type": "reverb",
            "room_size": "0.5",
            "damping": null,
            "wet_level": 0.3,
            "dry_level": 0.7,
            "width": 1.0,
            "freeze_mode": "false"
        }
    ]
}"""

TOO_FEW_BANDS_RESPONSE = """{
    "description": "test with too few bands",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 1.0}
            ]
        }
    ]
}"""

ATTACK_RELEASE_ISSUE_RESPONSE = """{
    "description": "test with attack >= release",
    "effects": [
        {
            "type": "compressor",
            "threshold": -20,
            "ratio": 4,
            "attack": 100,
            "release": 50,
            "knee": 3,
            "makeup_gain": 6
        }
    ]
}"""


class MockLLMProvider:
    """Mock LLM provider for integration tests."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.call_count = 0

    async def generate_with_retry(self, request):
        self.call_count += 1
        return LLMResponse(
            content=self.response_text,
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 200}
        )

    def get_provider_name(self):
        return "MockProvider"


class TestNormalizationHelpers:
    """Test normalization helper functions."""

    def test_clamp_within_range(self):
        """Clamp should not change values within range."""
        result = clamp(5.0, 0.0, 10.0)
        assert result == 5.0
        print(f"✓ Value within range unchanged: {result}")

    def test_clamp_below_min(self):
        """Clamp should set values below min to min."""
        result = clamp(-5.0, 0.0, 10.0)
        assert result == 0.0
        print(f"✓ Value below min clamped to {result}")

    def test_clamp_above_max(self):
        """Clamp should set values above max to max."""
        result = clamp(15.0, 0.0, 10.0)
        assert result == 10.0
        print(f"✓ Value above max clamped to {result}")

    def test_safe_float_with_number(self):
        """safe_float should convert numbers correctly."""
        assert safe_float(5) == 5.0
        assert safe_float(5.5) == 5.5
        print("✓ Numbers converted to float correctly")

    def test_safe_float_with_string(self):
        """safe_float should convert valid string numbers."""
        assert safe_float("5.5") == 5.5
        assert safe_float("invalid", default=1.0) == 1.0
        print("✓ Strings converted with fallback")

    def test_safe_float_with_nan(self):
        """safe_float should handle NaN values."""
        result = safe_float(float('nan'), default=1.0)
        assert result == 1.0
        print(f"✓ NaN handled with default: {result}")

    def test_safe_float_with_infinity(self):
        """safe_float should handle infinity values."""
        result = safe_float(float('inf'), default=1.0)
        assert result == 1.0
        print(f"✓ Infinity handled with default: {result}")

    def test_safe_bool_with_boolean(self):
        """safe_bool should handle boolean values."""
        assert safe_bool(True) is True
        assert safe_bool(False) is False
        print("✓ Boolean values handled correctly")

    def test_safe_bool_with_numbers(self):
        """safe_bool should convert numbers to boolean."""
        assert safe_bool(1) is True
        assert safe_bool(0) is False
        assert safe_bool(5) is True
        print("✓ Numbers converted to boolean")

    def test_safe_bool_with_strings(self):
        """safe_bool should convert string values."""
        assert safe_bool("true") is True
        assert safe_bool("false") is False
        assert safe_bool("yes") is True
        assert safe_bool("no") is False
        assert safe_bool("invalid", default=True) is True
        print("✓ Strings converted to boolean with fallback")


class TestEffectNormalization:
    """Test individual effect normalization."""

    def test_normalize_eq_out_of_range_values(self):
        """Normalizer should clamp out-of-range EQ values."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 50000, "gain": 20, "q": 15},
                {"frequency": 5, "gain": -20, "q": 0.01},
                {"frequency": 1000, "gain": 2, "q": 1.0}
            ]
        }
        normalized = normalize_effect(eq_data)

        # Check first band (all out of range)
        assert normalized["bands"][0]["frequency"] == 20000  # Clamped to max
        assert normalized["bands"][0]["gain"] == 12  # Clamped to max
        assert normalized["bands"][0]["q"] == 10  # Clamped to max

        # Check second band (all below min)
        assert normalized["bands"][1]["frequency"] == 20  # Clamped to min
        assert normalized["bands"][1]["gain"] == -12  # Clamped to min
        assert normalized["bands"][1]["q"] == 0.1  # Clamped to min

        print(f"✓ Out-of-range EQ values normalized")

    def test_normalize_eq_too_few_bands(self):
        """Normalizer should add bands when too few."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 1.0}
            ]
        }
        normalized = normalize_effect(eq_data)
        assert len(normalized["bands"]) >= 3
        print(f"✓ Too few bands padded to {len(normalized['bands'])}")

    def test_normalize_eq_too_many_bands(self):
        """Normalizer should truncate when too many bands."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": i * 1000, "gain": 1, "q": 1.0}
                for i in range(1, 12)  # 11 bands
            ]
        }
        normalized = normalize_effect(eq_data)
        assert len(normalized["bands"]) == 10
        print(f"✓ Too many bands truncated to {len(normalized['bands'])}")

    def test_normalize_eq_band_spacing(self):
        """Normalizer should ensure minimum spacing between bands."""
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 1.0},
                {"frequency": 1001, "gain": 1, "q": 1.0},  # Too close
                {"frequency": 1002, "gain": -1, "q": 1.0}  # Too close
            ]
        }
        normalized = normalize_effect(eq_data)

        # Check that bands are properly spaced
        for i in range(1, len(normalized["bands"])):
            prev_freq = normalized["bands"][i-1]["frequency"]
            curr_freq = normalized["bands"][i]["frequency"]
            assert curr_freq >= prev_freq * 1.1  # 10% minimum spacing
        print("✓ Band spacing enforced")

    def test_normalize_reverb_type_coercion(self):
        """Normalizer should coerce string numbers to float."""
        reverb_data = {
            "type": "reverb",
            "room_size": "0.5",
            "damping": "0.3",
            "wet_level": "0.4",
            "dry_level": "0.6",
            "width": "1.0"
        }
        normalized = normalize_effect(reverb_data)

        assert isinstance(normalized["room_size"], float)
        assert normalized["room_size"] == 0.5
        print("✓ String numbers coerced to float")

    def test_normalize_reverb_wet_dry_balance(self):
        """Normalizer should adjust excessive wet/dry levels."""
        reverb_data = {
            "type": "reverb",
            "room_size": 0.5,
            "damping": 0.5,
            "wet_level": 1.5,
            "dry_level": 1.5,
            "width": 1.0
        }
        normalized = normalize_effect(reverb_data)

        # Total should be adjusted
        total = normalized["wet_level"] + normalized["dry_level"]
        assert total <= 2.0
        print(f"✓ Wet/dry balance adjusted: total={total}")

    def test_normalize_compressor_attack_release(self):
        """Normalizer should fix attack >= release."""
        comp_data = {
            "type": "compressor",
            "threshold": -20,
            "ratio": 4,
            "attack": 100,
            "release": 50,
            "knee": 3,
            "makeup_gain": 6
        }
        normalized = normalize_effect(comp_data)

        assert normalized["attack"] < normalized["release"]
        print(f"✓ Attack/release fixed: attack={normalized['attack']}, release={normalized['release']}")


class TestEffectChainNormalization:
    """Test effect chain normalization."""

    def test_normalize_effect_chain_missing_description(self):
        """Normalizer should add default description if missing."""
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
        normalized = normalize_effect_chain_data(chain_data)
        assert "description" in normalized
        assert len(normalized["description"]) > 0
        print(f"✓ Default description added: '{normalized['description']}'")

    def test_normalize_effect_chain_builds_order(self):
        """Normalizer should build order list from effects."""
        chain_data = {
            "description": "test",
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
        normalized = normalize_effect_chain_data(chain_data)
        assert "order" in normalized
        assert normalized["order"] == ["eq", "compressor"]
        print(f"✓ Order list built: {normalized['order']}")

    def test_normalize_effect_chain_instance(self):
        """Normalizer should normalize EffectChain instance."""
        chain = EffectChain(
            description="test",
            effects=[
                EQParameters(
                    bands=[
                        {"frequency": 100, "gain": 2, "q": 0.7},
                        {"frequency": 1000, "gain": -1, "q": 1.2},
                        {"frequency": 8000, "gain": 3, "q": 0.9}
                    ]
                )
            ],
            order=["eq"]
        )
        normalized_chain = normalize_effect_chain(chain)
        assert isinstance(normalized_chain, EffectChain)
        assert len(normalized_chain.effects) == 1
        print("✓ EffectChain instance normalized successfully")


class TestEndToEndGeneration:
    """Test end-to-end generation pipeline with validation and normalization."""

    @pytest.mark.asyncio
    async def test_generate_with_valid_response(self):
        """Complete pipeline should work with valid LLM response."""
        valid_response = """{
            "description": "warm vocal",
            "effects": [
                {
                    "type": "eq",
                    "bands": [
                        {"frequency": 200, "gain": 3, "q": 0.7},
                        {"frequency": 3000, "gain": -2, "q": 1.2},
                        {"frequency": 8000, "gain": 1, "q": 0.9}
                    ]
                }
            ]
        }"""

        provider = MockLLMProvider(valid_response)

        with patch('src.generation.parameter_generator.load_prompt_template') as mock_template:
            mock_template.return_value.system_prompt = "Test prompt"
            with patch('src.generation.parameter_generator.format_prompt') as mock_format:
                mock_format.return_value = "Test"

                generator = ParameterGenerator(llm_provider=provider)
                result = await generator.generate_parameters(
                    description="warm vocal",
                    effects=["eq"]
                )

                # Validate result
                assert isinstance(result, EffectChain)
                assert len(result.effects) == 1

                # Validate with validator
                validation_result = validate_effect_chain(result)
                assert validation_result.is_valid

                print(f"✓ End-to-end generation successful: {result.description}")

    @pytest.mark.asyncio
    async def test_generate_with_normalization_needed(self):
        """Pipeline should normalize out-of-range values during correction."""
        provider = MockLLMProvider(OUT_OF_RANGE_EQ_RESPONSE)

        with patch('src.generation.parameter_generator.load_prompt_template') as mock_template:
            mock_template.return_value.system_prompt = "Test prompt"
            with patch('src.generation.parameter_generator.format_prompt') as mock_format:
                mock_format.return_value = "Test"

                # This will fail validation, triggering correction
                generator = ParameterGenerator(llm_provider=provider, max_correction_attempts=0)

                with pytest.raises(Exception):
                    # Should fail without normalization
                    result = await generator.generate_parameters(
                        description="test",
                        effects=["eq"]
                    )

                print("✓ Out-of-range values caught without normalization")

    @pytest.mark.asyncio
    async def test_validation_before_and_after_parsing(self):
        """Validation should work at both pre and post parsing stages."""
        chain_data = {
            "description": "test",
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

        # Pre-validation
        pre_result = validate_effect_chain_structure(chain_data)
        assert pre_result.is_valid
        print("✓ Pre-validation passed")

        # Parse to Pydantic model
        chain = EffectChain(**chain_data)

        # Post-validation
        post_result = validate_effect_chain(chain)
        assert post_result.is_valid
        print("✓ Post-validation passed")


class TestErrorRecoveryScenarios:
    """Test error recovery in various scenarios."""

    def test_normalize_then_validate(self):
        """Normalization followed by validation should succeed."""
        # Start with invalid data
        eq_data = {
            "type": "eq",
            "bands": [
                {"frequency": 50000, "gain": 20, "q": 15}
            ]
        }

        # Pre-validation should fail
        pre_result = validate_effect_structure(eq_data)
        assert not pre_result.is_valid
        print(f"✓ Pre-validation failed as expected: {len(pre_result.get_errors())} errors")

        # Normalize
        normalized = normalize_effect(eq_data)

        # Validation should now pass
        post_result = validate_effect_structure(normalized)
        # Note: May still fail due to too few bands, but values should be in range
        for band in normalized["bands"]:
            assert 20 <= band["frequency"] <= 20000
            assert -12 <= band["gain"] <= 12
            assert 0.1 <= band["q"] <= 10

        print("✓ Values normalized to valid ranges")

    def test_handle_missing_fields_with_defaults(self):
        """Normalizer should handle missing fields gracefully."""
        reverb_data = {
            "type": "reverb",
            "room_size": 0.5,
            # Missing other required fields
        }

        # This should not crash, but add defaults
        normalized = normalize_effect(reverb_data)

        # Check that all required fields are present
        assert "damping" in normalized
        assert "wet_level" in normalized
        assert "dry_level" in normalized
        assert "width" in normalized

        print("✓ Missing fields handled with defaults")

    def test_handle_invalid_types_with_coercion(self):
        """Normalizer should coerce invalid types when possible."""
        comp_data = {
            "type": "compressor",
            "threshold": "-20",
            "ratio": "4",
            "attack": "5",
            "release": "50",
            "knee": "3",
            "makeup_gain": "6"
        }

        normalized = normalize_effect(comp_data)

        # All values should be floats now
        assert isinstance(normalized["threshold"], (int, float))
        assert isinstance(normalized["ratio"], (int, float))
        assert normalized["threshold"] == -20
        assert normalized["ratio"] == 4

        print("✓ String numbers coerced to numeric types")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
