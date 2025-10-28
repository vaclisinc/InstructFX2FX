"""Tests for EffectChain schema validation.

Tests verify that EffectChain model correctly validates:
- Union type handling (EQ, Reverb, Compressor)
- Effect order validation
- Description field requirement
- Empty vs non-empty effect lists
- Valid effect combinations
- Order consistency with effects list
"""

import pytest
from pydantic import ValidationError

from src.models.parameters.effect_chain import EffectChain
from src.models.parameters.eq import EQParameters
from src.models.parameters.reverb import ReverbParameters
from src.models.parameters.compressor import CompressorParameters
from tests.test_parameters.fixtures import (
    VALID_EFFECT_CHAIN_SINGLE,
    VALID_EFFECT_CHAIN_MULTIPLE,
    VALID_EFFECT_CHAIN_ALL_EFFECTS,
    INVALID_EFFECT_CHAIN_EMPTY_EFFECTS,
    INVALID_EFFECT_CHAIN_MISSING_DESCRIPTION,
    INVALID_EFFECT_CHAIN_MISSING_ORDER,
    INVALID_EFFECT_CHAIN_MISMATCHED_ORDER,
)


class TestEffectChainValidation:
    """Test EffectChain model validation."""

    def test_valid_effect_chain_single_eq(self):
        """Effect chain with single EQ should pass validation."""
        # Create programmatically to ensure correct structure
        eq = EQParameters(
            bands=[
                {"frequency": 200.0, "gain": 3.0, "q": 0.7},
                {"frequency": 3000.0, "gain": -2.0, "q": 1.2},
                {"frequency": 8000.0, "gain": 1.0, "q": 0.9}
            ]
        )
        chain = EffectChain(
            description="warm and intimate vocal sound",
            effects=[eq],
            order=["eq"]
        )
        assert chain.description == "warm and intimate vocal sound"
        assert len(chain.effects) == 1
        assert len(chain.order) == 1
        assert chain.order[0] == "eq"
        print(f"✓ Single EQ effect chain created: {chain.description}")

    def test_valid_effect_chain_eq_compressor(self):
        """Effect chain with EQ + Compressor should pass validation."""
        eq = EQParameters(
            bands=[
                {"frequency": 5000.0, "gain": 4.0, "q": 1.0},
                {"frequency": 10000.0, "gain": 2.0, "q": 0.8},
                {"frequency": 1000.0, "gain": -1.0, "q": 1.2}
            ]
        )
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=3.0,
            makeup_gain=6.0
        )
        chain = EffectChain(
            description="bright and energetic guitar sound",
            effects=[eq, comp],
            order=["eq", "compressor"]
        )
        assert len(chain.effects) == 2
        assert len(chain.order) == 2
        assert chain.order == ["eq", "compressor"]
        print(f"✓ EQ + Compressor chain created: {chain.description}")

    def test_valid_effect_chain_all_effects(self):
        """Effect chain with all three effect types should pass validation."""
        eq = EQParameters(
            bands=[
                {"frequency": 100.0, "gain": 2.0, "q": 0.7},
                {"frequency": 1000.0, "gain": -1.0, "q": 1.2},
                {"frequency": 8000.0, "gain": 3.0, "q": 0.9}
            ]
        )
        comp = CompressorParameters(
            threshold=-15.0,
            ratio=3.0,
            attack=10.0,
            release=100.0,
            knee=6.0,
            makeup_gain=4.0
        )
        reverb = ReverbParameters(
            room_size=0.3,
            damping=0.7,
            wet_level=0.2,
            dry_level=0.8,
            width=1.0,
            freeze_mode=False
        )
        chain = EffectChain(
            description="full processing chain",
            effects=[eq, comp, reverb],
            order=["eq", "compressor", "reverb"]
        )
        assert len(chain.effects) == 3
        assert len(chain.order) == 3
        assert chain.order == ["eq", "compressor", "reverb"]
        print(f"✓ Full effect chain (EQ + Compressor + Reverb) created")

    def test_valid_effect_chain_different_order(self):
        """Effect chain with different effect order should pass validation."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 5000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 8000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        chain = EffectChain(
            description="compressor before eq",
            effects=[comp, eq],
            order=["compressor", "eq"]
        )
        assert chain.order == ["compressor", "eq"]
        print(f"✓ Compressor → EQ order accepted")

    def test_valid_effect_chain_single_compressor(self):
        """Effect chain with single Compressor should pass validation."""
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        chain = EffectChain(
            description="compression only",
            effects=[comp],
            order=["compressor"]
        )
        assert len(chain.effects) == 1
        assert chain.order[0] == "compressor"
        print(f"✓ Single compressor effect chain created")

    def test_valid_effect_chain_single_reverb(self):
        """Effect chain with single Reverb should pass validation."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0,
            freeze_mode=False
        )
        chain = EffectChain(
            description="reverb only",
            effects=[reverb],
            order=["reverb"]
        )
        assert len(chain.effects) == 1
        assert chain.order[0] == "reverb"
        print(f"✓ Single reverb effect chain created")

    def test_invalid_effect_chain_empty_effects(self):
        """Effect chain with empty effects list should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            EffectChain(**INVALID_EFFECT_CHAIN_EMPTY_EFFECTS)

        error = exc_info.value
        print(f"✓ Empty effects list rejected: {error}")
        # Check that error mentions effects list
        error_str = str(error).lower()
        assert "effects" in error_str or "empty" in error_str or "min" in error_str

    def test_invalid_effect_chain_missing_description(self):
        """Effect chain missing description field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            eq = EQParameters(
                bands=[
                    {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                    {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                    {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
                ]
            )
            EffectChain(
                effects=[eq],
                order=["eq"]
            )

        error = exc_info.value
        print(f"✓ Missing description rejected: {error}")
        assert "description" in str(error).lower()

    def test_invalid_effect_chain_missing_order(self):
        """Effect chain missing order field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            eq = EQParameters(
                bands=[
                    {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                    {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                    {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
                ]
            )
            EffectChain(
                description="missing order",
                effects=[eq]
            )

        error = exc_info.value
        print(f"✓ Missing order field rejected: {error}")
        assert "order" in str(error).lower()

    def test_invalid_effect_chain_empty_description(self):
        """Effect chain with empty description string should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            eq = EQParameters(
                bands=[
                    {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                    {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                    {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
                ]
            )
            EffectChain(
                description="",
                effects=[eq],
                order=["eq"]
            )

        error = exc_info.value
        print(f"✓ Empty description string rejected: {error}")
        assert "description" in str(error).lower()

    def test_effect_chain_description_accessible(self):
        """Effect chain description should be accessible."""
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        chain = EffectChain(
            description="test description",
            effects=[eq],
            order=["eq"]
        )
        assert chain.description == "test description"
        print(f"✓ Description accessible: '{chain.description}'")

    def test_effect_chain_effects_accessible(self):
        """Individual effects in chain should be accessible."""
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        chain = EffectChain(
            description="test chain",
            effects=[eq, comp],
            order=["eq", "compressor"]
        )
        assert isinstance(chain.effects[0], EQParameters)
        assert isinstance(chain.effects[1], CompressorParameters)
        print(f"✓ Effects accessible by index: {[type(e).__name__ for e in chain.effects]}")

    def test_effect_chain_order_accessible(self):
        """Effect order should be accessible and correct."""
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        chain = EffectChain(
            description="test chain",
            effects=[eq, comp],
            order=["eq", "compressor"]
        )
        assert chain.order == ["eq", "compressor"]
        print(f"✓ Order accessible: {chain.order}")

    def test_effect_chain_serialization(self):
        """Effect chain should be serializable to dict."""
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        chain = EffectChain(
            description="test chain",
            effects=[eq],
            order=["eq"]
        )
        chain_dict = chain.model_dump()
        assert "description" in chain_dict
        assert "effects" in chain_dict
        assert "order" in chain_dict
        assert len(chain_dict["effects"]) == 1
        print(f"✓ Effect chain serializable to dict: {list(chain_dict.keys())}")

    def test_effect_chain_multiple_same_type(self):
        """Effect chain with multiple instances of same effect type should be valid."""
        eq1 = EQParameters(
            bands=[
                {"frequency": 100.0, "gain": 3.0, "q": 0.7},
                {"frequency": 500.0, "gain": -2.0, "q": 1.2},
                {"frequency": 2000.0, "gain": 1.0, "q": 0.9}
            ]
        )
        eq2 = EQParameters(
            bands=[
                {"frequency": 5000.0, "gain": 2.0, "q": 1.0},
                {"frequency": 10000.0, "gain": 3.0, "q": 0.8},
                {"frequency": 15000.0, "gain": -1.0, "q": 1.2}
            ]
        )
        chain = EffectChain(
            description="double eq processing",
            effects=[eq1, eq2],
            order=["eq", "eq"]
        )
        assert len(chain.effects) == 2
        assert isinstance(chain.effects[0], EQParameters)
        assert isinstance(chain.effects[1], EQParameters)
        print(f"✓ Multiple effects of same type accepted")

    def test_effect_chain_long_description(self):
        """Effect chain with long description should be valid."""
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        long_desc = "This is a very detailed description of the audio processing chain that includes multiple clauses and explains the intended sonic characteristics in great detail."
        chain = EffectChain(
            description=long_desc,
            effects=[eq],
            order=["eq"]
        )
        assert chain.description == long_desc
        print(f"✓ Long description accepted: {len(long_desc)} characters")

    def test_effect_chain_reverb_eq_compressor_order(self):
        """Effect chain with reverb → eq → compressor order should be valid."""
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0
        )
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        chain = EffectChain(
            description="reverb first processing",
            effects=[reverb, eq, comp],
            order=["reverb", "eq", "compressor"]
        )
        assert chain.order == ["reverb", "eq", "compressor"]
        print(f"✓ Reverb → EQ → Compressor order accepted")

    def test_effect_chain_type_checking(self):
        """Effect chain should correctly identify effect types."""
        eq = EQParameters(
            bands=[
                {"frequency": 1000.0, "gain": 3.0, "q": 1.0},
                {"frequency": 2000.0, "gain": -2.0, "q": 0.8},
                {"frequency": 4000.0, "gain": 2.0, "q": 1.2}
            ]
        )
        comp = CompressorParameters(
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=6.0,
            makeup_gain=3.0
        )
        reverb = ReverbParameters(
            room_size=0.5,
            damping=0.5,
            wet_level=0.33,
            dry_level=0.67,
            width=1.0
        )
        chain = EffectChain(
            description="type checking test",
            effects=[eq, comp, reverb],
            order=["eq", "compressor", "reverb"]
        )
        assert isinstance(chain.effects[0], EQParameters)
        assert isinstance(chain.effects[1], CompressorParameters)
        assert isinstance(chain.effects[2], ReverbParameters)
        print(f"✓ Effect types correctly identified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
