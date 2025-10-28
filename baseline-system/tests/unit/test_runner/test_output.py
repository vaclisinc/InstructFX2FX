"""Tests for OutputManager class."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.runner.output import OutputManager


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "test_outputs"
    yield output_dir
    # Cleanup
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.fixture
def output_manager(temp_output_dir):
    """Create OutputManager instance."""
    return OutputManager(temp_output_dir)


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    audio_file = tmp_path / "sample.wav"
    # Create 1 second of silence at 44100 Hz
    audio_data = np.zeros(44100, dtype=np.float32)
    sf.write(audio_file, audio_data, 44100)
    return audio_file


class TestOutputManagerInitialization:
    """Test OutputManager initialization."""

    def test_directory_structure_created(self, temp_output_dir):
        """Test that directory structure is created on initialization."""
        manager = OutputManager(temp_output_dir)

        assert temp_output_dir.exists()
        assert manager.audio_dir.exists()
        assert manager.parameters_dir.exists()
        assert manager.scores_dir.exists()
        assert manager.logs_dir.exists()

    def test_metadata_file_created(self, output_manager):
        """Test that metadata file is created with correct structure."""
        assert output_manager.metadata_file.exists()

        with open(output_manager.metadata_file, "r") as f:
            metadata = json.load(f)

        assert "created_at" in metadata
        assert "experiments_completed" in metadata
        assert "experiments_failed" in metadata
        assert "last_updated" in metadata
        assert metadata["experiments_completed"] == 0
        assert metadata["experiments_failed"] == 0

    def test_metadata_loaded_on_existing_directory(self, temp_output_dir):
        """Test that existing metadata is loaded when directory exists."""
        # Create first manager and save some metadata
        manager1 = OutputManager(temp_output_dir)
        manager1.metadata["experiments_completed"] = 5

        # Manually save to simulate previous run
        with open(manager1.metadata_file, "w") as f:
            json.dump(manager1.metadata, f, indent=2)

        # Create second manager - should load existing metadata
        manager2 = OutputManager(temp_output_dir)

        assert manager2.metadata["experiments_completed"] == 5


class TestSaveAudio:
    """Test save_audio method."""

    def test_save_audio_success(self, output_manager, sample_audio_file):
        """Test successful audio file saving."""
        experiment_id = "test_exp_001"

        saved_path = output_manager.save_audio(sample_audio_file, experiment_id)

        assert saved_path.exists()
        assert saved_path.parent == output_manager.audio_dir
        assert saved_path.name == f"{experiment_id}.wav"

        # Verify audio content is copied correctly
        original_audio, _ = sf.read(sample_audio_file)
        saved_audio, _ = sf.read(saved_path)
        np.testing.assert_array_equal(original_audio, saved_audio)

    def test_save_audio_preserves_extension(self, output_manager, tmp_path):
        """Test that audio file extension is preserved."""
        # Create MP3-like file (just for testing naming)
        mp3_file = tmp_path / "test.mp3"
        # Create a wav file but name it mp3 for testing
        audio_data = np.zeros(44100, dtype=np.float32)
        sf.write(mp3_file, audio_data, 44100)

        experiment_id = "test_exp_002"
        saved_path = output_manager.save_audio(mp3_file, experiment_id)

        assert saved_path.suffix == ".mp3"

    def test_save_audio_file_not_found(self, output_manager):
        """Test that FileNotFoundError is raised for missing file."""
        nonexistent_file = Path("/nonexistent/audio.wav")
        experiment_id = "test_exp_003"

        with pytest.raises(FileNotFoundError) as exc_info:
            output_manager.save_audio(nonexistent_file, experiment_id)

        assert "not found" in str(exc_info.value).lower()


class TestSaveParameters:
    """Test save_parameters method."""

    def test_save_parameters_success(self, output_manager):
        """Test successful parameter saving."""
        experiment_id = "test_exp_001"
        parameters = {
            "reverb": {
                "decay": 0.8,
                "wet_dry": 0.5,
            },
            "eq": {
                "gain": 2.0,
                "frequency": 1000,
            },
        }

        saved_path = output_manager.save_parameters(parameters, experiment_id)

        assert saved_path.exists()
        assert saved_path.parent == output_manager.parameters_dir
        assert saved_path.name == f"{experiment_id}.json"

        # Verify parameters are saved correctly
        with open(saved_path, "r") as f:
            loaded_params = json.load(f)

        assert loaded_params == parameters

    def test_save_parameters_empty_dict(self, output_manager):
        """Test saving empty parameters dictionary."""
        experiment_id = "test_exp_002"
        parameters = {}

        saved_path = output_manager.save_parameters(parameters, experiment_id)

        assert saved_path.exists()

        with open(saved_path, "r") as f:
            loaded_params = json.load(f)

        assert loaded_params == {}

    def test_save_parameters_complex_structure(self, output_manager):
        """Test saving parameters with nested complex structure."""
        experiment_id = "test_exp_003"
        parameters = {
            "reverb": {
                "decay": 0.8,
                "parameters": [1, 2, 3],
                "nested": {
                    "deep": {
                        "value": 42,
                    }
                },
            },
        }

        saved_path = output_manager.save_parameters(parameters, experiment_id)

        with open(saved_path, "r") as f:
            loaded_params = json.load(f)

        assert loaded_params == parameters


class TestSaveScore:
    """Test save_score method."""

    def test_save_score_success(self, output_manager):
        """Test successful score saving."""
        experiment_id = "test_exp_001"
        score = {
            "overall": 85.3,
            "dimensions": {
                "warmth": 90.0,
                "clarity": 80.0,
            },
        }

        saved_path = output_manager.save_score(score, experiment_id)

        assert saved_path.exists()
        assert saved_path.parent == output_manager.scores_dir
        assert saved_path.name == f"{experiment_id}.json"

        # Verify score is saved correctly
        with open(saved_path, "r") as f:
            loaded_score = json.load(f)

        assert loaded_score["overall"] == score["overall"]
        assert loaded_score["dimensions"] == score["dimensions"]
        assert "scored_at" in loaded_score

    def test_save_score_adds_timestamp(self, output_manager):
        """Test that timestamp is automatically added to score."""
        experiment_id = "test_exp_002"
        score = {"overall": 75.0}

        saved_path = output_manager.save_score(score, experiment_id)

        with open(saved_path, "r") as f:
            loaded_score = json.load(f)

        assert "scored_at" in loaded_score
        # Timestamp should be ISO format
        from datetime import datetime
        timestamp = datetime.fromisoformat(loaded_score["scored_at"])
        assert isinstance(timestamp, datetime)


class TestRecordMethods:
    """Test record_success and record_failure methods."""

    def test_record_success_increments_counter(self, output_manager):
        """Test that record_success increments completed counter."""
        experiment_id = "test_exp_001"

        assert output_manager.metadata["experiments_completed"] == 0

        output_manager.record_success(experiment_id)

        assert output_manager.metadata["experiments_completed"] == 1

        # Test multiple successes
        output_manager.record_success("test_exp_002")
        assert output_manager.metadata["experiments_completed"] == 2

    def test_record_success_persists_metadata(self, temp_output_dir):
        """Test that record_success persists metadata to disk."""
        manager = OutputManager(temp_output_dir)
        manager.record_success("test_exp_001")

        # Create new manager - should load persisted metadata
        manager2 = OutputManager(temp_output_dir)
        assert manager2.metadata["experiments_completed"] == 1

    def test_record_failure_increments_counter(self, output_manager):
        """Test that record_failure increments failed counter."""
        experiment_id = "test_exp_001"
        error = "API timeout"

        assert output_manager.metadata["experiments_failed"] == 0

        output_manager.record_failure(experiment_id, error)

        assert output_manager.metadata["experiments_failed"] == 1

    def test_record_failure_stores_error_details(self, output_manager):
        """Test that record_failure stores error details."""
        experiment_id = "test_exp_001"
        error = "API timeout after 30s"

        output_manager.record_failure(experiment_id, error)

        assert "failures" in output_manager.metadata
        assert len(output_manager.metadata["failures"]) == 1

        failure = output_manager.metadata["failures"][0]
        assert failure["experiment_id"] == experiment_id
        assert failure["error"] == error
        assert "timestamp" in failure


class TestGetExperimentSummary:
    """Test get_experiment_summary method."""

    def test_summary_with_no_experiments(self, output_manager):
        """Test summary generation with no experiments."""
        summary = output_manager.get_experiment_summary()

        assert summary["experiments_completed"] == 0
        assert summary["experiments_failed"] == 0
        assert summary["file_counts"]["audio"] == 0
        assert summary["file_counts"]["parameters"] == 0
        assert summary["file_counts"]["scores"] == 0
        assert summary["scores"]["count"] == 0
        assert summary["scores"]["average"] == 0

    def test_summary_with_experiments(self, output_manager, sample_audio_file):
        """Test summary generation with multiple experiments."""
        # Create some experiments
        for i in range(3):
            exp_id = f"exp_{i:03d}"
            output_manager.save_audio(sample_audio_file, exp_id)
            output_manager.save_parameters({"param": i}, exp_id)
            output_manager.save_score({"overall": 70.0 + i * 10}, exp_id)
            output_manager.record_success(exp_id)

        summary = output_manager.get_experiment_summary()

        assert summary["experiments_completed"] == 3
        assert summary["file_counts"]["audio"] == 3
        assert summary["file_counts"]["parameters"] == 3
        assert summary["file_counts"]["scores"] == 3
        assert summary["scores"]["count"] == 3
        assert summary["scores"]["average"] == 80.0  # (70 + 80 + 90) / 3
        assert summary["scores"]["min"] == 70.0
        assert summary["scores"]["max"] == 90.0

    def test_summary_calculates_total_size(self, output_manager, sample_audio_file):
        """Test that summary includes total file size."""
        exp_id = "exp_001"
        output_manager.save_audio(sample_audio_file, exp_id)
        output_manager.save_parameters({"param": 1}, exp_id)
        output_manager.save_score({"overall": 85.0}, exp_id)

        summary = output_manager.get_experiment_summary()

        assert "total_size_mb" in summary
        assert summary["total_size_mb"] > 0


class TestListExperiments:
    """Test list_experiments method."""

    def test_list_experiments_empty(self, output_manager):
        """Test listing experiments when none exist."""
        experiments = output_manager.list_experiments()

        assert experiments == []

    def test_list_experiments_with_scores(self, output_manager):
        """Test listing experiments based on score files."""
        # Create experiments
        for i in range(5):
            exp_id = f"exp_{i:03d}"
            output_manager.save_score({"overall": 80.0}, exp_id)

        experiments = output_manager.list_experiments()

        assert len(experiments) == 5
        assert "exp_000" in experiments
        assert "exp_004" in experiments
        # Should be sorted
        assert experiments == sorted(experiments)


class TestGetExperimentFiles:
    """Test get_experiment_files method."""

    def test_get_experiment_files_all_present(self, output_manager, sample_audio_file):
        """Test retrieving all files for an experiment."""
        exp_id = "exp_001"
        output_manager.save_audio(sample_audio_file, exp_id)
        output_manager.save_parameters({"param": 1}, exp_id)
        output_manager.save_score({"overall": 85.0}, exp_id)

        files = output_manager.get_experiment_files(exp_id)

        assert files["audio"] is not None
        assert files["audio"].exists()
        assert files["parameters"] is not None
        assert files["parameters"].exists()
        assert files["score"] is not None
        assert files["score"].exists()

    def test_get_experiment_files_partial(self, output_manager):
        """Test retrieving files when only some exist."""
        exp_id = "exp_002"
        output_manager.save_score({"overall": 75.0}, exp_id)

        files = output_manager.get_experiment_files(exp_id)

        assert files["audio"] is None
        assert files["parameters"] is None
        assert files["score"] is not None

    def test_get_experiment_files_none(self, output_manager):
        """Test retrieving files for non-existent experiment."""
        exp_id = "nonexistent"

        files = output_manager.get_experiment_files(exp_id)

        assert files["audio"] is None
        assert files["parameters"] is None
        assert files["score"] is None


class TestGetLogsDir:
    """Test get_logs_dir method."""

    def test_get_logs_dir_returns_path(self, output_manager):
        """Test that get_logs_dir returns correct path."""
        logs_dir = output_manager.get_logs_dir()

        assert logs_dir == output_manager.logs_dir
        assert logs_dir.exists()
        assert logs_dir.is_dir()
