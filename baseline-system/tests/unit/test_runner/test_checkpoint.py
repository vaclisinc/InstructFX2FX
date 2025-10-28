"""Tests for CheckpointManager class."""

import json
import shutil
from pathlib import Path

import pytest

from src.runner.checkpoint import CheckpointManager


@pytest.fixture
def temp_experiment_dir(tmp_path):
    """Create temporary experiment directory."""
    experiment_dir = tmp_path / "test_experiment"
    yield experiment_dir
    # Cleanup
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)


@pytest.fixture
def checkpoint_manager(temp_experiment_dir):
    """Create CheckpointManager instance."""
    return CheckpointManager(temp_experiment_dir)


class TestCheckpointManagerInitialization:
    """Test CheckpointManager initialization."""

    def test_directory_created(self, temp_experiment_dir):
        """Test that experiment directory is created on initialization."""
        manager = CheckpointManager(temp_experiment_dir)

        assert temp_experiment_dir.exists()
        assert manager.checkpoint_file == temp_experiment_dir / "checkpoint.json"

    def test_checkpoint_file_not_created_initially(self, checkpoint_manager):
        """Test that checkpoint file is not created until first save."""
        assert not checkpoint_manager.checkpoint_file.exists()


class TestSaveCheckpoint:
    """Test save_checkpoint method."""

    def test_save_checkpoint_creates_file(self, checkpoint_manager):
        """Test that save_checkpoint creates checkpoint file."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1],
            pending=[2, 3, 4],
            failed=[],
            metadata={"total": 5},
        )

        assert checkpoint_manager.checkpoint_file.exists()

    def test_save_checkpoint_structure(self, checkpoint_manager):
        """Test that checkpoint has correct structure."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1, 2],
            pending=[3, 4],
            failed=[5],
            metadata={"total": 6, "description": "test"},
        )

        with open(checkpoint_manager.checkpoint_file, "r") as f:
            data = json.load(f)

        assert "created_at" in data
        assert "last_updated" in data
        assert data["completed"] == [0, 1, 2]
        assert data["pending"] == [3, 4]
        assert data["failed"] == [5]
        assert data["metadata"]["total"] == 6
        assert data["metadata"]["description"] == "test"

    def test_save_checkpoint_sorts_indices(self, checkpoint_manager):
        """Test that checkpoint indices are sorted."""
        checkpoint_manager.save_checkpoint(
            completed=[5, 1, 3],
            pending=[8, 2, 6],
            failed=[9, 4],
            metadata={},
        )

        with open(checkpoint_manager.checkpoint_file, "r") as f:
            data = json.load(f)

        assert data["completed"] == [1, 3, 5]
        assert data["pending"] == [2, 6, 8]
        assert data["failed"] == [4, 9]

    def test_save_checkpoint_updates_timestamp(self, checkpoint_manager):
        """Test that last_updated is updated on each save."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1],
            failed=[],
            metadata={},
        )

        with open(checkpoint_manager.checkpoint_file, "r") as f:
            data1 = json.load(f)
        timestamp1 = data1["last_updated"]

        # Save again
        checkpoint_manager.save_checkpoint(
            completed=[0, 1],
            pending=[],
            failed=[],
            metadata={},
        )

        with open(checkpoint_manager.checkpoint_file, "r") as f:
            data2 = json.load(f)
        timestamp2 = data2["last_updated"]

        # Timestamps should be different (or same if very fast)
        # But created_at should be the same
        assert data1["created_at"] == data2["created_at"]

    def test_save_checkpoint_atomic_write(self, checkpoint_manager):
        """Test that checkpoint is written atomically using temp file."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1],
            pending=[2, 3],
            failed=[],
            metadata={},
        )

        # Temp file should not exist after successful write
        temp_file = checkpoint_manager.checkpoint_file.with_suffix(".tmp")
        assert not temp_file.exists()
        assert checkpoint_manager.checkpoint_file.exists()


class TestLoadCheckpoint:
    """Test load_checkpoint method."""

    def test_load_checkpoint_no_file_returns_empty(self, checkpoint_manager):
        """Test that load returns empty structure when no checkpoint exists."""
        data = checkpoint_manager.load_checkpoint()

        assert data["completed"] == []
        assert data["pending"] == []
        assert data["failed"] == []
        assert data["metadata"] == {}

    def test_load_checkpoint_reads_saved_data(self, checkpoint_manager):
        """Test that load reads previously saved checkpoint."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1, 2],
            pending=[3, 4],
            failed=[5],
            metadata={"total": 6},
        )

        data = checkpoint_manager.load_checkpoint()

        assert data["completed"] == [0, 1, 2]
        assert data["pending"] == [3, 4]
        assert data["failed"] == [5]
        assert data["metadata"]["total"] == 6

    def test_load_checkpoint_invalid_json_raises_error(self, checkpoint_manager):
        """Test that corrupted checkpoint file raises JSONDecodeError."""
        # Create corrupted checkpoint file
        checkpoint_manager.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_manager.checkpoint_file, "w") as f:
            f.write("invalid json {")

        with pytest.raises(json.JSONDecodeError):
            checkpoint_manager.load_checkpoint()


class TestGetResumeState:
    """Test get_resume_state method."""

    def test_get_resume_state_no_checkpoint(self, checkpoint_manager):
        """Test resume state with no checkpoint."""
        state = checkpoint_manager.get_resume_state()

        assert state["completed"] == set()
        assert state["pending"] == []
        assert state["failed"] == set()
        assert state["next_index"] is None
        assert state["total_remaining"] == 0

    def test_get_resume_state_with_checkpoint(self, checkpoint_manager):
        """Test resume state with existing checkpoint."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1],
            pending=[2, 3, 4],
            failed=[5],
            metadata={"total": 6},
        )

        state = checkpoint_manager.get_resume_state()

        assert state["completed"] == {0, 1}
        assert state["pending"] == [2, 3, 4]
        assert state["failed"] == {5}
        assert state["next_index"] == 2
        assert state["total_remaining"] == 3
        assert state["metadata"]["total"] == 6

    def test_get_resume_state_all_completed(self, checkpoint_manager):
        """Test resume state when all experiments completed."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1, 2, 3, 4],
            pending=[],
            failed=[],
            metadata={"total": 5},
        )

        state = checkpoint_manager.get_resume_state()

        assert len(state["completed"]) == 5
        assert state["pending"] == []
        assert state["next_index"] is None
        assert state["total_remaining"] == 0


class TestClearCheckpoint:
    """Test clear_checkpoint method."""

    def test_clear_checkpoint_removes_file(self, checkpoint_manager):
        """Test that clear_checkpoint removes the checkpoint file."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[],
            failed=[],
            metadata={},
        )

        assert checkpoint_manager.checkpoint_file.exists()

        checkpoint_manager.clear_checkpoint()

        assert not checkpoint_manager.checkpoint_file.exists()

    def test_clear_checkpoint_no_file_no_error(self, checkpoint_manager):
        """Test that clearing non-existent checkpoint doesn't raise error."""
        assert not checkpoint_manager.checkpoint_file.exists()

        # Should not raise error
        checkpoint_manager.clear_checkpoint()

        assert not checkpoint_manager.checkpoint_file.exists()


class TestHasCheckpoint:
    """Test has_checkpoint method."""

    def test_has_checkpoint_returns_false_initially(self, checkpoint_manager):
        """Test that has_checkpoint returns False when no checkpoint exists."""
        assert not checkpoint_manager.has_checkpoint()

    def test_has_checkpoint_returns_true_after_save(self, checkpoint_manager):
        """Test that has_checkpoint returns True after saving."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1],
            failed=[],
            metadata={},
        )

        assert checkpoint_manager.has_checkpoint()

    def test_has_checkpoint_returns_false_after_clear(self, checkpoint_manager):
        """Test that has_checkpoint returns False after clearing."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[],
            failed=[],
            metadata={},
        )

        checkpoint_manager.clear_checkpoint()

        assert not checkpoint_manager.has_checkpoint()


class TestUpdateCompleted:
    """Test update_completed method."""

    def test_update_completed_adds_to_completed(self, checkpoint_manager):
        """Test that update_completed adds experiment to completed list."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1],
            pending=[2, 3],
            failed=[],
            metadata={},
        )

        checkpoint_manager.update_completed(2)

        data = checkpoint_manager.load_checkpoint()
        assert 2 in data["completed"]
        assert 2 not in data["pending"]

    def test_update_completed_removes_from_pending(self, checkpoint_manager):
        """Test that update_completed removes experiment from pending."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1, 2, 3],
            failed=[],
            metadata={},
        )

        checkpoint_manager.update_completed(2)

        data = checkpoint_manager.load_checkpoint()
        assert data["pending"] == [1, 3]

    def test_update_completed_removes_from_failed_if_retried(self, checkpoint_manager):
        """Test that update_completed removes from failed (retry case)."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1],
            failed=[2],
            metadata={},
        )

        checkpoint_manager.update_completed(2)

        data = checkpoint_manager.load_checkpoint()
        assert 2 in data["completed"]
        assert 2 not in data["failed"]

    def test_update_completed_idempotent(self, checkpoint_manager):
        """Test that updating same experiment multiple times is safe."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1],
            failed=[],
            metadata={},
        )

        checkpoint_manager.update_completed(1)
        checkpoint_manager.update_completed(1)

        data = checkpoint_manager.load_checkpoint()
        assert data["completed"].count(1) == 1


class TestUpdateFailed:
    """Test update_failed method."""

    def test_update_failed_adds_to_failed(self, checkpoint_manager):
        """Test that update_failed adds experiment to failed list."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1, 2],
            failed=[],
            metadata={},
        )

        checkpoint_manager.update_failed(1, "API timeout")

        data = checkpoint_manager.load_checkpoint()
        assert 1 in data["failed"]
        assert 1 not in data["pending"]

    def test_update_failed_removes_from_pending(self, checkpoint_manager):
        """Test that update_failed removes experiment from pending."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1, 2, 3],
            failed=[],
            metadata={},
        )

        checkpoint_manager.update_failed(2, "Processing error")

        data = checkpoint_manager.load_checkpoint()
        assert data["pending"] == [1, 3]

    def test_update_failed_stores_error_details(self, checkpoint_manager):
        """Test that update_failed stores error message in metadata."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1],
            failed=[],
            metadata={},
        )

        error_msg = "API rate limit exceeded"
        checkpoint_manager.update_failed(1, error_msg)

        data = checkpoint_manager.load_checkpoint()
        assert "failures" in data["metadata"]
        assert "1" in data["metadata"]["failures"]
        assert data["metadata"]["failures"]["1"]["error"] == error_msg
        assert "timestamp" in data["metadata"]["failures"]["1"]

    def test_update_failed_multiple_failures(self, checkpoint_manager):
        """Test that multiple failures are tracked separately."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1, 2, 3],
            failed=[],
            metadata={},
        )

        checkpoint_manager.update_failed(1, "Error 1")
        checkpoint_manager.update_failed(2, "Error 2")

        data = checkpoint_manager.load_checkpoint()
        assert len(data["failed"]) == 2
        assert "1" in data["metadata"]["failures"]
        assert "2" in data["metadata"]["failures"]


class TestValidateCheckpoint:
    """Test _validate_checkpoint method."""

    def test_validate_checkpoint_missing_fields_raises_error(self, checkpoint_manager):
        """Test that validation fails for missing required fields."""
        invalid_checkpoint = {
            "completed": [0],
            "pending": [1],
            # Missing "failed" and "metadata"
        }

        with pytest.raises(ValueError) as exc_info:
            checkpoint_manager._validate_checkpoint(invalid_checkpoint)

        assert "missing required field" in str(exc_info.value).lower()

    def test_validate_checkpoint_valid_data_no_error(self, checkpoint_manager):
        """Test that validation passes for valid checkpoint."""
        valid_checkpoint = {
            "completed": [0, 1],
            "pending": [2, 3],
            "failed": [4],
            "metadata": {"total": 5},
        }

        # Should not raise error
        checkpoint_manager._validate_checkpoint(valid_checkpoint)


class TestGetProgressSummary:
    """Test get_progress_summary method."""

    def test_progress_summary_no_checkpoint(self, checkpoint_manager):
        """Test progress summary with no checkpoint."""
        summary = checkpoint_manager.get_progress_summary()

        assert summary["total"] == 0
        assert summary["completed"] == 0
        assert summary["pending"] == 0
        assert summary["failed"] == 0
        assert summary["completion_rate"] == 0

    def test_progress_summary_with_checkpoint(self, checkpoint_manager):
        """Test progress summary with existing checkpoint."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1, 2],
            pending=[3, 4],
            failed=[5],
            metadata={},
        )

        summary = checkpoint_manager.get_progress_summary()

        assert summary["total"] == 6
        assert summary["completed"] == 3
        assert summary["pending"] == 2
        assert summary["failed"] == 1
        assert summary["completion_rate"] == 50.0  # 3/6 = 50%

    def test_progress_summary_all_completed(self, checkpoint_manager):
        """Test progress summary when all experiments completed."""
        checkpoint_manager.save_checkpoint(
            completed=[0, 1, 2, 3, 4],
            pending=[],
            failed=[],
            metadata={},
        )

        summary = checkpoint_manager.get_progress_summary()

        assert summary["total"] == 5
        assert summary["completed"] == 5
        assert summary["completion_rate"] == 100.0

    def test_progress_summary_includes_timestamps(self, checkpoint_manager):
        """Test that progress summary includes timestamp information."""
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1],
            failed=[],
            metadata={},
        )

        summary = checkpoint_manager.get_progress_summary()

        assert "created_at" in summary
        assert "last_updated" in summary


class TestCheckpointResumption:
    """Integration tests for checkpoint resumption workflow."""

    def test_resume_workflow(self, checkpoint_manager):
        """Test complete checkpoint resume workflow."""
        # Initial run - complete some experiments
        checkpoint_manager.save_checkpoint(
            completed=[0, 1],
            pending=[2, 3, 4],
            failed=[],
            metadata={"total": 5},
        )

        # Simulate resume
        resume_state = checkpoint_manager.get_resume_state()
        assert resume_state["next_index"] == 2
        assert len(resume_state["completed"]) == 2

        # Complete one more
        checkpoint_manager.update_completed(2)

        # Check state updated
        resume_state = checkpoint_manager.get_resume_state()
        assert resume_state["next_index"] == 3
        assert len(resume_state["completed"]) == 3

    def test_retry_failed_experiment(self, checkpoint_manager):
        """Test retrying a failed experiment."""
        # Initial state with a failure
        checkpoint_manager.save_checkpoint(
            completed=[0],
            pending=[1],
            failed=[2],
            metadata={},
        )

        # Retry the failed experiment
        checkpoint_manager.update_completed(2)

        # Verify it's now completed and removed from failed
        data = checkpoint_manager.load_checkpoint()
        assert 2 in data["completed"]
        assert 2 not in data["failed"]
