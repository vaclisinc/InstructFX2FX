"""Unit tests for SocialFX data models."""

import pytest
from pydantic import ValidationError
from judge_system.data.models import SocialFXExample, DatasetMetadata


class TestSocialFXExample:
    """Tests for SocialFXExample model."""

    def test_valid_example_creation(self):
        """Test creating a valid SocialFXExample."""
        example = SocialFXExample(
            id=1,
            description="warm and intimate",
            instrument="guitar",
            effect_type="eq",
            parameters={"band1_freq": 200, "band1_gain": 3.0},
            audio_path="/path/to/guitar.wav"
        )

        assert example.id == 1
        assert example.description == "warm and intimate"
        assert example.instrument == "guitar"
        assert example.effect_type == "eq"
        assert example.parameters == {"band1_freq": 200, "band1_gain": 3.0}
        assert example.audio_path == "/path/to/guitar.wav"

    def test_valid_example_without_audio_path(self):
        """Test creating a valid SocialFXExample without audio_path."""
        example = SocialFXExample(
            id=2,
            description="bright and aggressive",
            instrument="drums",
            effect_type="compressor",
            parameters={"threshold": -20, "ratio": 4.0}
        )

        assert example.audio_path is None

    def test_all_valid_instruments(self):
        """Test that all valid instruments are accepted."""
        valid_instruments = ['guitar', 'drums', 'piano']

        for instrument in valid_instruments:
            example = SocialFXExample(
                id=1,
                description="test",
                instrument=instrument,
                effect_type="eq",
                parameters={"param": 1.0}
            )
            assert example.instrument == instrument

    def test_invalid_instrument(self):
        """Test that invalid instrument raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SocialFXExample(
                id=1,
                description="test",
                instrument="violin",  # Invalid instrument
                effect_type="eq",
                parameters={"param": 1.0}
            )

        assert "Instrument must be one of" in str(exc_info.value)

    def test_empty_parameters_raises_error(self):
        """Test that empty parameters dictionary raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SocialFXExample(
                id=1,
                description="test",
                instrument="guitar",
                effect_type="eq",
                parameters={}  # Empty parameters
            )

        assert "Parameters dictionary cannot be empty" in str(exc_info.value)

    def test_negative_id_raises_error(self):
        """Test that negative id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SocialFXExample(
                id=-1,  # Negative id
                description="test",
                instrument="guitar",
                effect_type="eq",
                parameters={"param": 1.0}
            )

        # Check for validation error on id field
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('id',) for error in errors)

    def test_empty_description_raises_error(self):
        """Test that empty description raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SocialFXExample(
                id=1,
                description="",  # Empty description
                instrument="guitar",
                effect_type="eq",
                parameters={"param": 1.0}
            )

        # Check for validation error on description field
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('description',) for error in errors)

    def test_empty_effect_type_raises_error(self):
        """Test that empty effect_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SocialFXExample(
                id=1,
                description="test",
                instrument="guitar",
                effect_type="",  # Empty effect_type
                parameters={"param": 1.0}
            )

        # Check for validation error on effect_type field
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('effect_type',) for error in errors)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            SocialFXExample(
                id=1,
                description="test",
                instrument="guitar",
                effect_type="eq",
                parameters={"param": 1.0},
                extra_field="not allowed"  # Extra field
            )

        # Check for validation error about extra fields
        errors = exc_info.value.errors()
        assert any('extra_forbidden' in error['type'] for error in errors)


class TestDatasetMetadata:
    """Tests for DatasetMetadata model."""

    def test_valid_metadata_creation(self):
        """Test creating valid DatasetMetadata."""
        metadata = DatasetMetadata(
            total_examples=100,
            instruments=["guitar", "drums", "piano"],
            effect_types=["eq", "reverb", "compressor"],
            description_count={"eq": 30, "reverb": 35, "compressor": 35},
            parameter_ranges={
                "eq": {
                    "band1_freq": (20.0, 20000.0),
                    "band1_gain": (-24.0, 24.0)
                }
            }
        )

        assert metadata.total_examples == 100
        assert metadata.instruments == ["guitar", "drums", "piano"]
        assert metadata.effect_types == ["eq", "reverb", "compressor"]
        assert metadata.description_count == {"eq": 30, "reverb": 35, "compressor": 35}
        assert "eq" in metadata.parameter_ranges

    def test_valid_metadata_without_parameter_ranges(self):
        """Test creating valid DatasetMetadata without parameter_ranges."""
        metadata = DatasetMetadata(
            total_examples=50,
            instruments=["guitar"],
            effect_types=["eq"],
            description_count={"eq": 50}
        )

        assert metadata.parameter_ranges == {}

    def test_negative_total_examples_raises_error(self):
        """Test that negative total_examples raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                total_examples=-10,  # Negative total_examples
                instruments=["guitar"],
                effect_types=["eq"],
                description_count={"eq": 10}
            )

        # Check for validation error on total_examples field
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('total_examples',) for error in errors)

    def test_empty_instruments_raises_error(self):
        """Test that empty instruments list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                total_examples=10,
                instruments=[],  # Empty instruments list
                effect_types=["eq"],
                description_count={"eq": 10}
            )

        # Check for validation error on instruments field
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('instruments',) for error in errors)

    def test_empty_effect_types_raises_error(self):
        """Test that empty effect_types list raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                total_examples=10,
                instruments=["guitar"],
                effect_types=[],  # Empty effect_types list
                description_count={"eq": 10}
            )

        # Check for validation error on effect_types field
        errors = exc_info.value.errors()
        assert any(error['loc'] == ('effect_types',) for error in errors)

    def test_negative_description_count_raises_error(self):
        """Test that negative description count raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                total_examples=10,
                instruments=["guitar"],
                effect_types=["eq"],
                description_count={"eq": -5}  # Negative count
            )

        assert "must be non-negative" in str(exc_info.value)

    def test_invalid_parameter_range_raises_error(self):
        """Test that invalid parameter range (min > max) raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                total_examples=10,
                instruments=["guitar"],
                effect_types=["eq"],
                description_count={"eq": 10},
                parameter_ranges={
                    "eq": {
                        "band1_freq": (20000.0, 20.0)  # min > max
                    }
                }
            )

        assert "Invalid range" in str(exc_info.value)
        assert "min" in str(exc_info.value)
        assert "max" in str(exc_info.value)

    def test_valid_parameter_ranges_with_equal_min_max(self):
        """Test that parameter range with min == max is valid."""
        metadata = DatasetMetadata(
            total_examples=10,
            instruments=["guitar"],
            effect_types=["eq"],
            description_count={"eq": 10},
            parameter_ranges={
                "eq": {
                    "band1_freq": (1000.0, 1000.0)  # min == max is valid
                }
            }
        )

        assert metadata.parameter_ranges["eq"]["band1_freq"] == (1000.0, 1000.0)

    def test_multiple_effect_types_with_ranges(self):
        """Test metadata with multiple effect types and parameter ranges."""
        metadata = DatasetMetadata(
            total_examples=90,
            instruments=["guitar", "drums", "piano"],
            effect_types=["eq", "reverb", "compressor"],
            description_count={"eq": 30, "reverb": 30, "compressor": 30},
            parameter_ranges={
                "eq": {
                    "band1_freq": (20.0, 20000.0),
                    "band1_gain": (-24.0, 24.0)
                },
                "reverb": {
                    "room_size": (0.0, 1.0),
                    "damping": (0.0, 1.0)
                },
                "compressor": {
                    "threshold": (-60.0, 0.0),
                    "ratio": (1.0, 20.0)
                }
            }
        )

        assert len(metadata.parameter_ranges) == 3
        assert "eq" in metadata.parameter_ranges
        assert "reverb" in metadata.parameter_ranges
        assert "compressor" in metadata.parameter_ranges

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetMetadata(
                total_examples=10,
                instruments=["guitar"],
                effect_types=["eq"],
                description_count={"eq": 10},
                extra_field="not allowed"  # Extra field
            )

        # Check for validation error about extra fields
        errors = exc_info.value.errors()
        assert any('extra_forbidden' in error['type'] for error in errors)
