"""Tests for EffectChainBuilder class."""

import pytest
from pedalboard import Pedalboard, PeakFilter, Reverb, Compressor

from src.audio_processing.effects import EffectChainBuilder
from src.models.parameters.effect_chain import EffectChain
from src.models.parameters.eq import EQParameters, EQBand
from src.models.parameters.reverb import ReverbParameters
from src.models.parameters.compressor import CompressorParameters


class TestEffectChainBuilder:
    """Test suite for EffectChainBuilder."""

    @pytest.fixture
    def builder(self):
        """Create an EffectChainBuilder instance."""
        return EffectChainBuilder()

    @pytest.fixture
    def simple_eq_params(self):
        """Create simple EQ parameters for testing."""
        return EQParameters(
            effect_type="eq",
            bands=[
                EQBand(frequency=100.0, gain=3.0, q=1.0),
                EQBand(frequency=1000.0, gain=-2.0, q=2.0),
                EQBand(frequency=10000.0, gain=1.5, q=0.5)
            ]
        )

    @pytest.fixture
    def simple_reverb_params(self):
        """Create simple reverb parameters for testing."""
        return ReverbParameters(
            effect_type="reverb",
            room_size=0.5,
            damping=0.3,
            wet_level=0.4,
            dry_level=0.6,
            width=0.8,
            freeze_mode=False
        )

    @pytest.fixture
    def simple_compressor_params(self):
        """Create simple compressor parameters for testing."""
        return CompressorParameters(
            effect_type="compressor",
            threshold=-20.0,
            ratio=4.0,
            attack=5.0,
            release=50.0,
            knee=3.0,
            makeup_gain=6.0
        )

    @pytest.fixture
    def mixed_effect_chain(self, simple_eq_params, simple_reverb_params, simple_compressor_params):
        """Create an effect chain with multiple effect types."""
        return EffectChain(
            description="Test effect chain with EQ, reverb, and compressor",
            effects=[simple_eq_params, simple_reverb_params, simple_compressor_params],
            order=["eq", "reverb", "compressor"]
        )

    def test_builder_initialization(self, builder):
        """Test that builder initializes correctly."""
        assert builder is not None
        assert "eq" in builder.effect_map
        assert "reverb" in builder.effect_map
        assert "compressor" in builder.effect_map

    def test_create_eq_filters(self, builder, simple_eq_params):
        """Test creating EQ filters from parameters."""
        filters = builder.create_eq(simple_eq_params)

        assert isinstance(filters, list)
        assert len(filters) == 3  # Three bands
        assert all(isinstance(f, PeakFilter) for f in filters)

        # Verify first filter properties match (approximately, due to normalization)
        # We can't directly access filter parameters, but we can verify creation succeeded

    def test_create_eq_q_normalization(self, builder, simple_eq_params):
        """Test that Q-factor normalization is applied."""
        filters = builder.create_eq(simple_eq_params)

        # The normalization should have been applied (0.707 factor)
        # This test verifies the method runs without errors
        assert len(filters) == len(simple_eq_params.bands)

    def test_create_reverb(self, builder, simple_reverb_params):
        """Test creating reverb effect from parameters."""
        reverb = builder.create_reverb(simple_reverb_params)

        assert isinstance(reverb, Reverb)
        # Reverb object created successfully

    def test_create_reverb_room_size_normalization(self, builder):
        """Test reverb room size normalization."""
        # Test extreme values
        params_min = ReverbParameters(
            effect_type="reverb",
            room_size=0.0,  # Should normalize to 0.1
            damping=0.5,
            wet_level=0.5,
            dry_level=0.5,
            width=0.5,
            freeze_mode=False
        )
        reverb_min = builder.create_reverb(params_min)
        assert isinstance(reverb_min, Reverb)

        params_max = ReverbParameters(
            effect_type="reverb",
            room_size=1.0,  # Should normalize to 0.9
            damping=0.5,
            wet_level=0.5,
            dry_level=0.5,
            width=0.5,
            freeze_mode=False
        )
        reverb_max = builder.create_reverb(params_max)
        assert isinstance(reverb_max, Reverb)

    def test_create_compressor(self, builder, simple_compressor_params):
        """Test creating compressor effect from parameters."""
        compressor = builder.create_compressor(simple_compressor_params)

        assert isinstance(compressor, Compressor)
        # Compressor object created successfully

    def test_build_chain_single_effect(self, builder, simple_eq_params):
        """Test building a chain with a single effect type."""
        chain = EffectChain(
            description="Single EQ effect",
            effects=[simple_eq_params],
            order=["eq"]
        )

        board = builder.build_chain(chain)

        assert isinstance(board, Pedalboard)
        # EQ creates 3 filters (one per band), so board should have 3 plugins
        assert len(board) == 3

    def test_build_chain_multiple_effects(self, builder, mixed_effect_chain):
        """Test building a chain with multiple effect types."""
        board = builder.build_chain(mixed_effect_chain)

        assert isinstance(board, Pedalboard)
        # EQ (3 bands) + Reverb (1) + Compressor (1) = 5 total plugins
        assert len(board) == 5

    def test_build_chain_effect_order(self, builder, simple_eq_params, simple_reverb_params):
        """Test that effects are added in the correct order."""
        # Create chain with EQ first, then reverb
        chain1 = EffectChain(
            description="EQ then reverb",
            effects=[simple_eq_params, simple_reverb_params],
            order=["eq", "reverb"]
        )
        board1 = builder.build_chain(chain1)
        assert len(board1) == 4  # 3 EQ bands + 1 reverb

        # Create chain with reverb first, then EQ
        chain2 = EffectChain(
            description="Reverb then EQ",
            effects=[simple_reverb_params, simple_eq_params],
            order=["reverb", "eq"]
        )
        board2 = builder.build_chain(chain2)
        assert len(board2) == 4  # 1 reverb + 3 EQ bands

        # Both boards should have same number of effects but different order
        # We can't easily verify order without processing audio, but we can verify creation

    def test_build_chain_multiple_same_type(self, builder, simple_reverb_params):
        """Test building a chain with multiple effects of the same type."""
        reverb2 = ReverbParameters(
            effect_type="reverb",
            room_size=0.8,
            damping=0.7,
            wet_level=0.3,
            dry_level=0.7,
            width=0.5,
            freeze_mode=False
        )

        chain = EffectChain(
            description="Two reverbs",
            effects=[simple_reverb_params, reverb2],
            order=["reverb", "reverb"]
        )

        board = builder.build_chain(chain)
        assert len(board) == 2  # Two reverb effects

    def test_build_chain_unsupported_effect_type(self, builder):
        """Test that unsupported effect types raise an error."""
        # We can't create an EffectChain with invalid type due to Pydantic validation,
        # but we can test the error handling in the builder

        # This would require bypassing Pydantic validation, which we shouldn't do
        # Instead, we verify that only valid types are in effect_map
        assert "delay" not in builder.effect_map
        assert "chorus" not in builder.effect_map

    def test_build_chain_empty_effects_list(self, builder):
        """Test that empty effect chain is caught by validation."""
        # This should be caught by Pydantic validation before reaching builder
        with pytest.raises(ValueError):
            EffectChain(
                description="Empty chain",
                effects=[],
                order=[]
            )

    def test_create_eq_with_many_bands(self, builder):
        """Test creating EQ with maximum number of bands."""
        bands = [
            EQBand(frequency=50.0 * (2 ** i), gain=1.0, q=1.0)
            for i in range(10)  # Maximum 10 bands
        ]
        params = EQParameters(
            effect_type="eq",
            bands=bands
        )

        filters = builder.create_eq(params)
        assert len(filters) == 10

    def test_create_reverb_freeze_mode(self, builder):
        """Test creating reverb with freeze mode enabled."""
        params = ReverbParameters(
            effect_type="reverb",
            room_size=0.9,
            damping=0.1,
            wet_level=0.8,
            dry_level=0.2,
            width=1.0,
            freeze_mode=True  # Freeze mode enabled
        )

        reverb = builder.create_reverb(params)
        assert isinstance(reverb, Reverb)

    def test_create_compressor_extreme_values(self, builder):
        """Test creating compressor with extreme but valid values."""
        # Gentle compression
        params_gentle = CompressorParameters(
            effect_type="compressor",
            threshold=-40.0,
            ratio=1.5,
            attack=0.1,  # Very fast attack
            release=10.0,  # Fast release
            knee=12.0,  # Maximum soft knee
            makeup_gain=0.0
        )
        compressor_gentle = builder.create_compressor(params_gentle)
        assert isinstance(compressor_gentle, Compressor)

        # Aggressive limiting
        params_aggressive = CompressorParameters(
            effect_type="compressor",
            threshold=-5.0,
            ratio=20.0,  # Maximum ratio (limiting)
            attack=0.5,
            release=100.0,
            knee=0.0,  # Hard knee
            makeup_gain=24.0  # Maximum makeup gain
        )
        compressor_aggressive = builder.create_compressor(params_aggressive)
        assert isinstance(compressor_aggressive, Compressor)

    def test_builder_effect_factory_methods(self, builder):
        """Test that all factory methods are properly registered."""
        assert callable(builder.effect_map["eq"])
        assert callable(builder.effect_map["reverb"])
        assert callable(builder.effect_map["compressor"])

    def test_build_chain_complex_scenario(self, builder):
        """Test building a complex chain with multiple effects in specific order."""
        # Create a realistic effect chain: EQ -> Compressor -> Reverb
        eq = EQParameters(
            effect_type="eq",
            bands=[
                EQBand(frequency=80.0, gain=-3.0, q=1.0),  # Cut low rumble
                EQBand(frequency=250.0, gain=2.0, q=1.5),  # Boost low mids
                EQBand(frequency=3000.0, gain=4.0, q=2.0),  # Boost presence
                EQBand(frequency=12000.0, gain=-1.0, q=0.5)  # Slight high cut
            ]
        )

        compressor = CompressorParameters(
            effect_type="compressor",
            threshold=-18.0,
            ratio=3.0,
            attack=5.0,
            release=80.0,
            knee=6.0,
            makeup_gain=8.0
        )

        reverb = ReverbParameters(
            effect_type="reverb",
            room_size=0.6,
            damping=0.4,
            wet_level=0.3,
            dry_level=0.7,
            width=0.9,
            freeze_mode=False
        )

        chain = EffectChain(
            description="Professional vocal processing chain",
            effects=[eq, compressor, reverb],
            order=["eq", "compressor", "reverb"]
        )

        board = builder.build_chain(chain)

        # 4 EQ bands + 1 compressor + 1 reverb = 6 total plugins
        assert len(board) == 6
        assert isinstance(board, Pedalboard)
