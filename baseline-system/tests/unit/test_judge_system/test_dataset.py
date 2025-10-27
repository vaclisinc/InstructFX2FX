"""Unit tests for SocialFXDataset loader."""

import pytest
from pathlib import Path
from judge_system.data.dataset import SocialFXDataset
from judge_system.data.models import SocialFXExample, DatasetMetadata


class TestSocialFXDatasetInitialization:
    """Tests for SocialFXDataset initialization."""

    def test_init_with_default_directory(self):
        """Test initialization with default data directory."""
        dataset = SocialFXDataset()

        assert dataset.data_dir == Path("data/socialfx")
        assert dataset.audio_dir == Path("data/socialfx/audio")
        assert dataset.params_dir == Path("data/socialfx/parameters")
        assert dataset.examples == []
        assert dataset.metadata is None

    def test_init_with_custom_directory(self):
        """Test initialization with custom data directory."""
        custom_path = "/custom/path/to/data"
        dataset = SocialFXDataset(data_dir=custom_path)

        assert dataset.data_dir == Path(custom_path)
        assert dataset.audio_dir == Path(custom_path) / "audio"
        assert dataset.params_dir == Path(custom_path) / "parameters"


class TestSocialFXDatasetStructureVerification:
    """Tests for dataset directory structure verification."""

    def test_verify_structure_all_files_present(self, tmp_path):
        """Test that structure verification passes when all files present."""
        # Create directory structure
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        # Create required audio files
        (audio_dir / "guitar.wav").touch()
        (audio_dir / "drums.wav").touch()
        (audio_dir / "piano.wav").touch()

        # Create required CSV files
        (params_dir / "eq_params.csv").write_text("id,description,instrument\n")
        (params_dir / "reverb_params.csv").write_text("id,description,instrument\n")
        (params_dir / "compressor_params.csv").write_text("id,description,instrument\n")

        dataset = SocialFXDataset(data_dir=str(tmp_path))
        # Should not raise any exception
        dataset._verify_structure()

    def test_verify_structure_missing_audio_file(self, tmp_path):
        """Test that verification fails with missing audio file."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        # Missing guitar.wav
        (audio_dir / "drums.wav").touch()
        (audio_dir / "piano.wav").touch()

        (params_dir / "eq_params.csv").write_text("id,description,instrument\n")
        (params_dir / "reverb_params.csv").write_text("id,description,instrument\n")
        (params_dir / "compressor_params.csv").write_text("id,description,instrument\n")

        dataset = SocialFXDataset(data_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError) as exc_info:
            dataset._verify_structure()

        assert "Missing required files" in str(exc_info.value)
        assert "guitar.wav" in str(exc_info.value)

    def test_verify_structure_missing_csv_file(self, tmp_path):
        """Test that verification fails with missing CSV file."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        (audio_dir / "guitar.wav").touch()
        (audio_dir / "drums.wav").touch()
        (audio_dir / "piano.wav").touch()

        # Missing reverb_params.csv
        (params_dir / "eq_params.csv").write_text("id,description,instrument\n")
        (params_dir / "compressor_params.csv").write_text("id,description,instrument\n")

        dataset = SocialFXDataset(data_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError) as exc_info:
            dataset._verify_structure()

        assert "Missing required files" in str(exc_info.value)
        assert "reverb_params.csv" in str(exc_info.value)

    def test_verify_structure_missing_audio_directory(self, tmp_path):
        """Test that verification fails when audio directory missing."""
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        (params_dir / "eq_params.csv").write_text("id,description,instrument\n")
        (params_dir / "reverb_params.csv").write_text("id,description,instrument\n")
        (params_dir / "compressor_params.csv").write_text("id,description,instrument\n")

        dataset = SocialFXDataset(data_dir=str(tmp_path))

        with pytest.raises(FileNotFoundError) as exc_info:
            dataset._verify_structure()

        assert "Missing required files" in str(exc_info.value)


class TestSocialFXDatasetLoading:
    """Tests for dataset loading functionality."""

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path):
        """Create a complete sample dataset directory."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        # Create audio files
        (audio_dir / "guitar.wav").touch()
        (audio_dir / "drums.wav").touch()
        (audio_dir / "piano.wav").touch()

        # Create EQ CSV
        eq_csv = """id,description,instrument,band1_freq,band1_gain,band1_q,band2_freq,band2_gain,band2_q
1,warm,guitar,200,3.0,0.7,3000,-2.0,1.2
2,bright,drums,5000,4.0,1.0,10000,2.0,0.8
3,balanced,piano,500,1.5,0.9,4000,-1.0,1.1"""
        (params_dir / "eq_params.csv").write_text(eq_csv)

        # Create Reverb CSV
        reverb_csv = """id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,spacious,piano,0.9,0.3,0.4,0.6,0.9,false
2,tight,drums,0.2,0.8,0.2,0.8,0.5,true"""
        (params_dir / "reverb_params.csv").write_text(reverb_csv)

        # Create Compressor CSV
        comp_csv = """id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,punchy,drums,-20,4.0,5,50,3,2.0
2,smooth,guitar,-30,2.5,10,100,6,3.5"""
        (params_dir / "compressor_params.csv").write_text(comp_csv)

        return tmp_path

    def test_load_complete_dataset(self, sample_dataset_dir):
        """Test loading a complete dataset."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        # Should have 3 EQ + 2 Reverb + 2 Compressor = 7 examples
        assert len(dataset.examples) == 7
        assert dataset.metadata is not None
        assert dataset.metadata.total_examples == 7

    def test_load_generates_metadata(self, sample_dataset_dir):
        """Test that loading generates proper metadata."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        assert dataset.metadata is not None
        assert set(dataset.metadata.instruments) == {"guitar", "drums", "piano"}
        assert set(dataset.metadata.effect_types) == {"eq", "reverb", "compressor"}
        assert dataset.metadata.description_count["eq"] == 3
        assert dataset.metadata.description_count["reverb"] == 2
        assert dataset.metadata.description_count["compressor"] == 2

    def test_load_separates_effect_types(self, sample_dataset_dir):
        """Test that examples are correctly separated by effect type."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        eq_examples = [e for e in dataset.examples if e.effect_type == "eq"]
        reverb_examples = [e for e in dataset.examples if e.effect_type == "reverb"]
        comp_examples = [e for e in dataset.examples if e.effect_type == "compressor"]

        assert len(eq_examples) == 3
        assert len(reverb_examples) == 2
        assert len(comp_examples) == 2


class TestSocialFXDatasetFiltering:
    """Tests for dataset filtering functionality."""

    @pytest.fixture
    def loaded_dataset(self, sample_dataset_dir):
        """Create and load a sample dataset."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()
        return dataset

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path):
        """Create a complete sample dataset directory."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        (audio_dir / "guitar.wav").touch()
        (audio_dir / "drums.wav").touch()
        (audio_dir / "piano.wav").touch()

        eq_csv = """id,description,instrument,band1_freq,band1_gain,band1_q
1,warm,guitar,200,3.0,0.7
2,bright,drums,5000,4.0,1.0
3,balanced,piano,500,1.5,0.9
4,crisp,guitar,8000,2.0,0.8"""
        (params_dir / "eq_params.csv").write_text(eq_csv)

        reverb_csv = """id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,spacious,piano,0.9,0.3,0.4,0.6,0.9,false
2,tight,drums,0.2,0.8,0.2,0.8,0.5,true"""
        (params_dir / "reverb_params.csv").write_text(reverb_csv)

        comp_csv = """id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,punchy,drums,-20,4.0,5,50,3,2.0"""
        (params_dir / "compressor_params.csv").write_text(comp_csv)

        return tmp_path

    def test_get_examples_no_filter(self, loaded_dataset):
        """Test getting all examples without filtering."""
        examples = loaded_dataset.get_examples()

        assert len(examples) == 7  # 4 EQ + 2 Reverb + 1 Compressor

    def test_get_examples_filter_by_instrument(self, loaded_dataset):
        """Test filtering examples by instrument."""
        guitar_examples = loaded_dataset.get_examples(instrument="guitar")
        drums_examples = loaded_dataset.get_examples(instrument="drums")
        piano_examples = loaded_dataset.get_examples(instrument="piano")

        assert len(guitar_examples) == 2  # 2 EQ guitar examples
        assert len(drums_examples) == 3  # 1 EQ + 1 Reverb + 1 Compressor
        assert len(piano_examples) == 2  # 1 EQ + 1 Reverb
        assert all(e.instrument == "guitar" for e in guitar_examples)

    def test_get_examples_filter_by_effect_type(self, loaded_dataset):
        """Test filtering examples by effect type."""
        eq_examples = loaded_dataset.get_examples(effect_type="eq")
        reverb_examples = loaded_dataset.get_examples(effect_type="reverb")
        comp_examples = loaded_dataset.get_examples(effect_type="compressor")

        assert len(eq_examples) == 4
        assert len(reverb_examples) == 2
        assert len(comp_examples) == 1
        assert all(e.effect_type == "eq" for e in eq_examples)

    def test_get_examples_filter_by_both(self, loaded_dataset):
        """Test filtering by both instrument and effect type."""
        guitar_eq = loaded_dataset.get_examples(instrument="guitar", effect_type="eq")

        assert len(guitar_eq) == 2
        assert all(e.instrument == "guitar" and e.effect_type == "eq" for e in guitar_eq)

    def test_get_examples_with_limit(self, loaded_dataset):
        """Test limiting the number of returned examples."""
        examples = loaded_dataset.get_examples(limit=3)

        assert len(examples) == 3

    def test_get_examples_filter_and_limit(self, loaded_dataset):
        """Test combining filter and limit."""
        examples = loaded_dataset.get_examples(effect_type="eq", limit=2)

        assert len(examples) == 2
        assert all(e.effect_type == "eq" for e in examples)


class TestSocialFXDatasetFewShotSelection:
    """Tests for few-shot example selection."""

    @pytest.fixture
    def loaded_dataset(self, sample_dataset_dir):
        """Create and load a sample dataset."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()
        return dataset

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path):
        """Create a sample dataset with diverse examples."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        (audio_dir / "guitar.wav").touch()
        (audio_dir / "drums.wav").touch()
        (audio_dir / "piano.wav").touch()

        # Create multiple examples per instrument for EQ
        eq_csv = """id,description,instrument,band1_freq,band1_gain,band1_q
1,warm,guitar,200,3.0,0.7
2,bright,drums,5000,4.0,1.0
3,balanced,piano,500,1.5,0.9
4,crisp,guitar,8000,2.0,0.8
5,punchy,drums,100,2.5,0.6
6,smooth,piano,3000,-1.0,1.2"""
        (params_dir / "eq_params.csv").write_text(eq_csv)

        reverb_csv = """id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,spacious,piano,0.9,0.3,0.4,0.6,0.9,false"""
        (params_dir / "reverb_params.csv").write_text(reverb_csv)

        comp_csv = """id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,punchy,drums,-20,4.0,5,50,3,2.0"""
        (params_dir / "compressor_params.csv").write_text(comp_csv)

        return tmp_path

    def test_get_few_shot_examples_default(self, loaded_dataset):
        """Test getting few-shot examples with default settings."""
        examples = loaded_dataset.get_few_shot_examples(effect_type="eq", n_examples=3)

        assert len(examples) == 3
        assert all(e.effect_type == "eq" for e in examples)

    def test_get_few_shot_examples_diverse(self, loaded_dataset):
        """Test that diverse=True selects across instruments."""
        examples = loaded_dataset.get_few_shot_examples(
            effect_type="eq",
            n_examples=3,
            diverse=True
        )

        assert len(examples) == 3
        instruments = [e.instrument for e in examples]
        # Should have representation from different instruments
        assert len(set(instruments)) >= 2

    def test_get_few_shot_examples_not_diverse(self, loaded_dataset):
        """Test that diverse=False returns first n examples."""
        examples = loaded_dataset.get_few_shot_examples(
            effect_type="eq",
            n_examples=3,
            diverse=False
        )

        assert len(examples) == 3
        # Should be first 3 examples in order
        all_eq = loaded_dataset.get_examples(effect_type="eq")
        assert examples == all_eq[:3]

    def test_get_few_shot_examples_more_than_available(self, loaded_dataset):
        """Test requesting more examples than available."""
        examples = loaded_dataset.get_few_shot_examples(
            effect_type="compressor",
            n_examples=10,
            diverse=True
        )

        # Should only return available examples (1 compressor example)
        assert len(examples) <= 10

    def test_select_diverse_examples_round_robin(self, loaded_dataset):
        """Test that diverse selection uses round-robin across instruments."""
        eq_examples = loaded_dataset.get_examples(effect_type="eq")
        selected = loaded_dataset._select_diverse_examples(eq_examples, 6)

        assert len(selected) == 6

        # Count instruments
        instruments = [e.instrument for e in selected]
        # Should have even distribution (2 of each: guitar, drums, piano)
        from collections import Counter
        instrument_counts = Counter(instruments)
        assert instrument_counts["guitar"] == 2
        assert instrument_counts["drums"] == 2
        assert instrument_counts["piano"] == 2


class TestSocialFXDatasetMetadataGeneration:
    """Tests for metadata generation."""

    @pytest.fixture
    def sample_dataset_dir(self, tmp_path):
        """Create a sample dataset."""
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        params_dir = tmp_path / "parameters"
        params_dir.mkdir()

        (audio_dir / "guitar.wav").touch()
        (audio_dir / "drums.wav").touch()
        (audio_dir / "piano.wav").touch()

        eq_csv = """id,description,instrument,band1_freq,band1_gain,band1_q
1,warm,guitar,200,3.0,0.7
2,bright,guitar,5000,5.0,0.5"""
        (params_dir / "eq_params.csv").write_text(eq_csv)

        reverb_csv = """id,description,instrument,room_size,damping,wet_level,dry_level,width,freeze_mode
1,spacious,piano,0.9,0.8,0.4,0.6,0.9,false
2,tight,drums,0.1,0.2,0.1,0.9,0.2,true"""
        (params_dir / "reverb_params.csv").write_text(reverb_csv)

        comp_csv = """id,description,instrument,threshold,ratio,attack,release,knee,makeup_gain
1,punchy,drums,-50,8.0,1,200,2,10.0"""
        (params_dir / "compressor_params.csv").write_text(comp_csv)

        return tmp_path

    def test_calculate_parameter_ranges_eq(self, sample_dataset_dir):
        """Test parameter range calculation for EQ."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        eq_ranges = dataset.metadata.parameter_ranges.get("eq", {})

        assert "frequency" in eq_ranges
        assert "gain" in eq_ranges
        assert "q" in eq_ranges

        # Check ranges
        assert eq_ranges["frequency"] == (200, 5000)
        assert eq_ranges["gain"] == (3.0, 5.0)
        assert eq_ranges["q"] == (0.5, 0.7)

    def test_calculate_parameter_ranges_reverb(self, sample_dataset_dir):
        """Test parameter range calculation for Reverb."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        reverb_ranges = dataset.metadata.parameter_ranges.get("reverb", {})

        assert "room_size" in reverb_ranges
        assert "damping" in reverb_ranges
        assert "wet_level" in reverb_ranges
        assert "dry_level" in reverb_ranges
        assert "width" in reverb_ranges

        assert reverb_ranges["room_size"] == (0.1, 0.9)
        assert reverb_ranges["damping"] == (0.2, 0.8)

    def test_calculate_parameter_ranges_compressor(self, sample_dataset_dir):
        """Test parameter range calculation for Compressor."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        comp_ranges = dataset.metadata.parameter_ranges.get("compressor", {})

        assert "threshold" in comp_ranges
        assert "ratio" in comp_ranges
        assert "attack" in comp_ranges
        assert "release" in comp_ranges
        assert "knee" in comp_ranges
        assert "makeup_gain" in comp_ranges

    def test_metadata_instruments_sorted(self, sample_dataset_dir):
        """Test that instruments list is sorted."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        assert dataset.metadata.instruments == ["drums", "guitar", "piano"]

    def test_metadata_effect_types_sorted(self, sample_dataset_dir):
        """Test that effect_types list is sorted."""
        dataset = SocialFXDataset(data_dir=str(sample_dataset_dir))
        dataset.load()

        assert dataset.metadata.effect_types == ["compressor", "eq", "reverb"]


class TestSocialFXDatasetErrorHandling:
    """Tests for error handling in dataset operations."""

    def test_load_with_missing_directory(self):
        """Test that loading fails gracefully with missing directory."""
        dataset = SocialFXDataset(data_dir="/nonexistent/path")

        with pytest.raises(FileNotFoundError):
            dataset.load()

    def test_get_examples_before_loading(self):
        """Test that get_examples returns empty list before loading."""
        dataset = SocialFXDataset()
        examples = dataset.get_examples()

        assert examples == []

    def test_get_few_shot_examples_before_loading(self):
        """Test that few-shot selection works with empty dataset."""
        dataset = SocialFXDataset()
        examples = dataset.get_few_shot_examples(effect_type="eq", n_examples=3)

        assert examples == []
