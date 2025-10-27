"""Unit tests for SocialFX CSV parsers."""

import pytest
import pandas as pd
from pathlib import Path
from judge_system.data.parsers import (
    _load_eq_parameters,
    _load_reverb_parameters,
    _load_compressor_parameters
)
from judge_system.data.models import SocialFXExample


class TestEQParameterParsing:
    """Tests for EQ parameter CSV parsing."""

    def test_load_eq_parameters_valid_csv(self, tmp_path):
        """Test loading valid EQ parameters CSV."""
        # Create test CSV
        csv_content = """id,description,instrument,band1_freq,band1_gain,band1_q,band2_freq,band2_gain,band2_q
1,warm and intimate,guitar,200,3.0,0.7,3000,-2.0,1.2
2,bright and aggressive,drums,5000,4.0,1.0,10000,2.0,0.8"""

        csv_path = tmp_path / "eq_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        # Load parameters
        examples = _load_eq_parameters(csv_path, audio_dir)

        assert len(examples) == 2
        assert examples[0].id == 1
        assert examples[0].description == "warm and intimate"
        assert examples[0].instrument == "guitar"
        assert examples[0].effect_type == "eq"
        assert len(examples[0].parameters["bands"]) == 2
        assert examples[0].parameters["bands"][0]["frequency"] == 200
        assert examples[0].parameters["bands"][0]["gain"] == 3.0
        assert examples[0].parameters["bands"][0]["q"] == 0.7
        assert examples[0].audio_path == str(audio_dir / "guitar.wav")

    def test_load_eq_parameters_single_band(self, tmp_path):
        """Test loading EQ parameters with single band."""
        csv_content = """id,description,instrument,band1_freq,band1_gain,band1_q
1,simple eq,piano,1000,2.5,0.9"""

        csv_path = tmp_path / "eq_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_eq_parameters(csv_path, audio_dir)

        assert len(examples) == 1
        assert len(examples[0].parameters["bands"]) == 1
        assert examples[0].parameters["bands"][0]["frequency"] == 1000

    def test_load_eq_parameters_multiple_bands(self, tmp_path):
        """Test loading EQ parameters with multiple bands."""
        csv_content = """id,description,instrument,band1_freq,band1_gain,band1_q,band2_freq,band2_gain,band2_q,band3_freq,band3_gain,band3_q,band4_freq,band4_gain,band4_q
1,complex eq,guitar,100,1.0,0.5,500,2.0,0.7,2000,-1.5,1.0,8000,3.0,0.8"""

        csv_path = tmp_path / "eq_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_eq_parameters(csv_path, audio_dir)

        assert len(examples[0].parameters["bands"]) == 4

    def test_load_eq_parameters_file_not_found(self, tmp_path):
        """Test that missing CSV file raises FileNotFoundError."""
        csv_path = tmp_path / "nonexistent.csv"
        audio_dir = tmp_path / "audio"

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_eq_parameters(csv_path, audio_dir)

        assert "EQ parameters CSV not found" in str(exc_info.value)

    def test_load_eq_parameters_empty_csv(self, tmp_path):
        """Test that empty CSV raises EmptyDataError."""
        csv_path = tmp_path / "eq_params.csv"
        csv_path.write_text("")

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(pd.errors.EmptyDataError) as exc_info:
            _load_eq_parameters(csv_path, audio_dir)

        assert "EQ parameters CSV is empty" in str(exc_info.value)

    def test_load_eq_parameters_missing_required_columns(self, tmp_path):
        """Test that missing required columns raises KeyError."""
        csv_content = """id,description,band1_freq,band1_gain,band1_q
1,missing instrument,200,3.0,0.7"""

        csv_path = tmp_path / "eq_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(KeyError) as exc_info:
            _load_eq_parameters(csv_path, audio_dir)

        assert "missing required columns" in str(exc_info.value)
        assert "instrument" in str(exc_info.value)

    def test_load_eq_parameters_invalid_instrument(self, tmp_path):
        """Test that invalid instrument raises ValidationError."""
        csv_content = """id,description,instrument,band1_freq,band1_gain,band1_q
1,test,violin,200,3.0,0.7"""

        csv_path = tmp_path / "eq_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(ValueError) as exc_info:
            _load_eq_parameters(csv_path, audio_dir)

        assert "Instrument must be one of" in str(exc_info.value)

    def test_load_eq_parameters_type_conversion(self, tmp_path):
        """Test that numeric values are properly converted to float."""
        csv_content = """id,description,instrument,band1_freq,band1_gain,band1_q
1,test,guitar,200.5,3.2,0.75"""

        csv_path = tmp_path / "eq_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_eq_parameters(csv_path, audio_dir)

        assert isinstance(examples[0].parameters["bands"][0]["frequency"], float)
        assert examples[0].parameters["bands"][0]["frequency"] == 200.5


class TestReverbParameterParsing:
    """Tests for Reverb parameter CSV parsing."""

    def test_load_reverb_parameters_valid_csv(self, tmp_path):
        """Test loading valid reverb parameters CSV."""
        csv_content = """id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,spacious cathedral,piano,0.9,0.3,0.4,0.6,0.9,false
2,tight room,drums,0.2,0.8,0.2,0.8,0.5,true"""

        csv_path = tmp_path / "reverb_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_reverb_parameters(csv_path, audio_dir)

        assert len(examples) == 2
        assert examples[0].id == 1
        assert examples[0].description == "spacious cathedral"
        assert examples[0].instrument == "piano"
        assert examples[0].effect_type == "reverb"
        assert examples[0].parameters["room_size"] == 0.9
        assert examples[0].parameters["damping"] == 0.3
        assert examples[0].parameters["freeze_mode"] is False
        assert examples[1].parameters["freeze_mode"] is True

    def test_load_reverb_parameters_boolean_variations(self, tmp_path):
        """Test that various boolean representations are handled correctly."""
        csv_content = """id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,test1,guitar,0.5,0.5,0.5,0.5,0.5,true
2,test2,guitar,0.5,0.5,0.5,0.5,0.5,True
3,test3,guitar,0.5,0.5,0.5,0.5,0.5,1
4,test4,guitar,0.5,0.5,0.5,0.5,0.5,yes
5,test5,guitar,0.5,0.5,0.5,0.5,0.5,false
6,test6,guitar,0.5,0.5,0.5,0.5,0.5,False
7,test7,guitar,0.5,0.5,0.5,0.5,0.5,0
8,test8,guitar,0.5,0.5,0.5,0.5,0.5,no"""

        csv_path = tmp_path / "reverb_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_reverb_parameters(csv_path, audio_dir)

        # True variations
        assert examples[0].parameters["freeze_mode"] is True
        assert examples[1].parameters["freeze_mode"] is True
        assert examples[2].parameters["freeze_mode"] is True
        assert examples[3].parameters["freeze_mode"] is True

        # False variations
        assert examples[4].parameters["freeze_mode"] is False
        assert examples[5].parameters["freeze_mode"] is False
        assert examples[6].parameters["freeze_mode"] is False
        assert examples[7].parameters["freeze_mode"] is False

    def test_load_reverb_parameters_invalid_boolean(self, tmp_path):
        """Test that invalid boolean value raises ValueError."""
        csv_content = """id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,test,guitar,0.5,0.5,0.5,0.5,0.5,maybe"""

        csv_path = tmp_path / "reverb_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(ValueError) as exc_info:
            _load_reverb_parameters(csv_path, audio_dir)

        assert "Invalid freeze_mode value" in str(exc_info.value)

    def test_load_reverb_parameters_file_not_found(self, tmp_path):
        """Test that missing CSV file raises FileNotFoundError."""
        csv_path = tmp_path / "nonexistent.csv"
        audio_dir = tmp_path / "audio"

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_reverb_parameters(csv_path, audio_dir)

        assert "Reverb parameters CSV not found" in str(exc_info.value)

    def test_load_reverb_parameters_empty_csv(self, tmp_path):
        """Test that empty CSV raises EmptyDataError."""
        csv_path = tmp_path / "reverb_params.csv"
        csv_path.write_text("")

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(pd.errors.EmptyDataError):
            _load_reverb_parameters(csv_path, audio_dir)

    def test_load_reverb_parameters_missing_required_columns(self, tmp_path):
        """Test that missing required columns raises KeyError."""
        csv_content = """id,description,instrument,room_size,damping
1,incomplete,guitar,0.5,0.5"""

        csv_path = tmp_path / "reverb_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(KeyError) as exc_info:
            _load_reverb_parameters(csv_path, audio_dir)

        assert "missing required columns" in str(exc_info.value)


class TestCompressorParameterParsing:
    """Tests for Compressor parameter CSV parsing."""

    def test_load_compressor_parameters_valid_csv(self, tmp_path):
        """Test loading valid compressor parameters CSV."""
        csv_content = """id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,punchy and controlled,drums,-20,4.0,5,50,3,2.0
2,smooth leveling,guitar,-30,2.5,10,100,6,3.5"""

        csv_path = tmp_path / "compressor_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_compressor_parameters(csv_path, audio_dir)

        assert len(examples) == 2
        assert examples[0].id == 1
        assert examples[0].description == "punchy and controlled"
        assert examples[0].instrument == "drums"
        assert examples[0].effect_type == "compressor"
        assert examples[0].parameters["threshold"] == -20
        assert examples[0].parameters["ratio"] == 4.0
        assert examples[0].parameters["attack"] == 5
        assert examples[0].parameters["release"] == 50
        assert examples[0].parameters["knee"] == 3
        assert examples[0].parameters["makeup_gain"] == 2.0

    def test_load_compressor_parameters_negative_values(self, tmp_path):
        """Test that negative threshold values are handled correctly."""
        csv_content = """id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,test,piano,-50,8.0,2,30,5,4.0"""

        csv_path = tmp_path / "compressor_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_compressor_parameters(csv_path, audio_dir)

        assert examples[0].parameters["threshold"] == -50

    def test_load_compressor_parameters_file_not_found(self, tmp_path):
        """Test that missing CSV file raises FileNotFoundError."""
        csv_path = tmp_path / "nonexistent.csv"
        audio_dir = tmp_path / "audio"

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_compressor_parameters(csv_path, audio_dir)

        assert "Compressor parameters CSV not found" in str(exc_info.value)

    def test_load_compressor_parameters_empty_csv(self, tmp_path):
        """Test that empty CSV raises EmptyDataError."""
        csv_path = tmp_path / "compressor_params.csv"
        csv_path.write_text("")

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(pd.errors.EmptyDataError):
            _load_compressor_parameters(csv_path, audio_dir)

    def test_load_compressor_parameters_missing_required_columns(self, tmp_path):
        """Test that missing required columns raises KeyError."""
        csv_content = """id,description,instrument,threshold,ratio
1,incomplete,guitar,-20,4.0"""

        csv_path = tmp_path / "compressor_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with pytest.raises(KeyError) as exc_info:
            _load_compressor_parameters(csv_path, audio_dir)

        assert "missing required columns" in str(exc_info.value)

    def test_load_compressor_parameters_type_conversion(self, tmp_path):
        """Test that all numeric values are properly converted to float."""
        csv_content = """id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,test,guitar,-25.5,3.2,7.5,80.3,4.1,2.8"""

        csv_path = tmp_path / "compressor_params.csv"
        csv_path.write_text(csv_content)

        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        examples = _load_compressor_parameters(csv_path, audio_dir)

        params = examples[0].parameters
        assert isinstance(params["threshold"], float)
        assert isinstance(params["ratio"], float)
        assert isinstance(params["attack"], float)
        assert isinstance(params["release"], float)
        assert isinstance(params["knee"], float)
        assert isinstance(params["makeup_gain"], float)
