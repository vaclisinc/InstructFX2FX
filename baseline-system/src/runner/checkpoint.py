"""Checkpoint management for resumable experiments.

This module provides the CheckpointManager class for saving and loading
experiment checkpoint state:
- Tracking completed, pending, and failed experiments
- Saving checkpoint state with metadata
- Loading checkpoint state for resuming
- Clearing checkpoints after successful completion
- Validating checkpoint integrity
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

from src.utils.logging import get_logger


class CheckpointManager:
    """Manage experiment checkpoints for resume capability.

    The CheckpointManager enables long-running experiments to be interrupted
    and resumed by saving experiment state at regular intervals. It tracks
    which experiments have been completed, which are pending, and which have failed.

    Checkpoint file structure:
        {
            "created_at": "2024-10-28T12:00:00",
            "last_updated": "2024-10-28T12:30:00",
            "completed": [0, 1, 2],
            "pending": [3, 4, 5, 6],
            "failed": [7],
            "metadata": {
                "total_experiments": 10,
                "checkpoint_interval": 5,
                ...
            }
        }

    Attributes:
        experiment_dir: Root directory for experiment
        checkpoint_file: Path to checkpoint JSON file

    Examples:
        >>> checkpoint_mgr = CheckpointManager(Path("./outputs/exp_001"))
        >>>
        >>> # Save checkpoint after completing some experiments
        >>> checkpoint_mgr.save_checkpoint(
        ...     completed=[0, 1, 2],
        ...     pending=[3, 4, 5],
        ...     failed=[],
        ...     metadata={"total": 6}
        ... )
        >>>
        >>> # Load checkpoint to resume
        >>> state = checkpoint_mgr.load_checkpoint()
        >>> resume_state = checkpoint_mgr.get_resume_state()
        >>> print(f"Resume from: {resume_state['next_index']}")
        >>>
        >>> # Clear checkpoint after completion
        >>> checkpoint_mgr.clear_checkpoint()
    """

    def __init__(self, experiment_dir: Path):
        """Initialize CheckpointManager.

        Args:
            experiment_dir: Root directory for experiment outputs
        """
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_file = self.experiment_dir / "checkpoint.json"
        self.logger = get_logger("runner")

        # Ensure experiment directory exists
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        completed: List[int],
        pending: List[int],
        failed: List[int],
        metadata: Dict[str, Any],
    ) -> None:
        """Save checkpoint state to disk.

        Saves current experiment progress including completed, pending, and
        failed experiment indices along with arbitrary metadata.

        Args:
            completed: List of completed experiment indices
            pending: List of pending experiment indices
            failed: List of failed experiment indices
            metadata: Additional metadata (e.g., config, timestamps, counters)

        Raises:
            OSError: If checkpoint file cannot be written

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> checkpoint_mgr.save_checkpoint(
            ...     completed=[0, 1, 2],
            ...     pending=[3, 4, 5],
            ...     failed=[],
            ...     metadata={
            ...         "total_experiments": 6,
            ...         "batch_size": 10,
            ...         "description": "baseline experiment"
            ...     }
            ... )
        """
        checkpoint_data = {
            "created_at": self._get_created_at(),
            "last_updated": datetime.now().isoformat(),
            "completed": sorted(completed),
            "pending": sorted(pending),
            "failed": sorted(failed),
            "metadata": metadata,
        }

        try:
            # Write atomically by writing to temp file then renaming
            temp_file = self.checkpoint_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_file.replace(self.checkpoint_file)

            self.logger.info(
                "checkpoint_saved",
                checkpoint_file=str(self.checkpoint_file),
                completed_count=len(completed),
                pending_count=len(pending),
                failed_count=len(failed),
            )

        except Exception as e:
            self.logger.error(
                "checkpoint_save_failed",
                checkpoint_file=str(self.checkpoint_file),
                error=str(e),
                exc_info=True,
            )
            raise

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint state from disk.

        Reads and validates checkpoint file, returning the full checkpoint state.

        Returns:
            Dictionary with checkpoint data including completed, pending, failed,
            and metadata. Returns empty structure if no checkpoint exists.

        Raises:
            OSError: If checkpoint file cannot be read
            json.JSONDecodeError: If checkpoint file is corrupted

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> state = checkpoint_mgr.load_checkpoint()
            >>> if state["completed"]:
            ...     print(f"Already completed: {len(state['completed'])} experiments")
            >>> if state["failed"]:
            ...     print(f"Failed experiments: {state['failed']}")
        """
        if not self.checkpoint_file.exists():
            self.logger.info(
                "no_checkpoint_found",
                checkpoint_file=str(self.checkpoint_file),
            )
            return {
                "completed": [],
                "pending": [],
                "failed": [],
                "metadata": {},
            }

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint_data = json.load(f)

            # Validate checkpoint structure
            self._validate_checkpoint(checkpoint_data)

            self.logger.info(
                "checkpoint_loaded",
                checkpoint_file=str(self.checkpoint_file),
                completed_count=len(checkpoint_data["completed"]),
                pending_count=len(checkpoint_data["pending"]),
                failed_count=len(checkpoint_data["failed"]),
            )

            return checkpoint_data

        except json.JSONDecodeError as e:
            self.logger.error(
                "checkpoint_corrupted",
                checkpoint_file=str(self.checkpoint_file),
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error(
                "checkpoint_load_failed",
                checkpoint_file=str(self.checkpoint_file),
                error=str(e),
                exc_info=True,
            )
            raise

    def get_resume_state(self) -> Dict[str, Any]:
        """Get state needed to resume experiment.

        Analyzes checkpoint data to determine what work remains and provides
        a resume state that can be used to continue execution.

        Returns:
            Dictionary with resume information:
                - completed: Set of completed indices
                - pending: List of pending indices (sorted)
                - failed: Set of failed indices
                - next_index: Next experiment index to process
                - total_remaining: Number of experiments remaining
                - metadata: Original checkpoint metadata

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> resume_state = checkpoint_mgr.get_resume_state()
            >>>
            >>> if resume_state["completed"]:
            ...     print(f"Resuming after {len(resume_state['completed'])} completed")
            >>>
            >>> for idx in resume_state["pending"]:
            ...     if idx not in resume_state["failed"]:
            ...         process_experiment(idx)
        """
        checkpoint_data = self.load_checkpoint()

        completed_set = set(checkpoint_data["completed"])
        pending_list = checkpoint_data["pending"]
        failed_set = set(checkpoint_data["failed"])

        # Calculate next index to process
        next_index = None
        if pending_list:
            next_index = pending_list[0]

        resume_state = {
            "completed": completed_set,
            "pending": pending_list,
            "failed": failed_set,
            "next_index": next_index,
            "total_remaining": len(pending_list),
            "metadata": checkpoint_data["metadata"],
        }

        self.logger.info(
            "resume_state_generated",
            completed_count=len(completed_set),
            pending_count=len(pending_list),
            failed_count=len(failed_set),
            next_index=next_index,
        )

        return resume_state

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion.

        Deletes the checkpoint file, typically called after all experiments
        have completed successfully.

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> # After all experiments complete
            >>> checkpoint_mgr.clear_checkpoint()
        """
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
                self.logger.info(
                    "checkpoint_cleared",
                    checkpoint_file=str(self.checkpoint_file),
                )
            except Exception as e:
                self.logger.error(
                    "checkpoint_clear_failed",
                    checkpoint_file=str(self.checkpoint_file),
                    error=str(e),
                )
                raise
        else:
            self.logger.debug(
                "checkpoint_already_cleared",
                checkpoint_file=str(self.checkpoint_file),
            )

    def has_checkpoint(self) -> bool:
        """Check if checkpoint file exists.

        Returns:
            True if checkpoint file exists, False otherwise

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> if checkpoint_mgr.has_checkpoint():
            ...     print("Resuming from checkpoint...")
            ...     state = checkpoint_mgr.load_checkpoint()
            ... else:
            ...     print("Starting fresh experiment...")
        """
        exists = self.checkpoint_file.exists()

        self.logger.debug(
            "checkpoint_exists_check",
            checkpoint_file=str(self.checkpoint_file),
            exists=exists,
        )

        return exists

    def update_completed(self, experiment_index: int) -> None:
        """Update checkpoint with newly completed experiment.

        Loads current checkpoint, adds the experiment to completed list,
        removes it from pending, and saves the updated checkpoint.

        Args:
            experiment_index: Index of completed experiment

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> checkpoint_mgr.update_completed(3)
        """
        checkpoint_data = self.load_checkpoint()

        # Add to completed
        if experiment_index not in checkpoint_data["completed"]:
            checkpoint_data["completed"].append(experiment_index)

        # Remove from pending if present
        if experiment_index in checkpoint_data["pending"]:
            checkpoint_data["pending"].remove(experiment_index)

        # Remove from failed if it was retried successfully
        if experiment_index in checkpoint_data["failed"]:
            checkpoint_data["failed"].remove(experiment_index)

        # Save updated checkpoint
        self.save_checkpoint(
            completed=checkpoint_data["completed"],
            pending=checkpoint_data["pending"],
            failed=checkpoint_data["failed"],
            metadata=checkpoint_data["metadata"],
        )

        self.logger.info(
            "checkpoint_updated_completed",
            experiment_index=experiment_index,
            total_completed=len(checkpoint_data["completed"]),
        )

    def update_failed(self, experiment_index: int, error: str) -> None:
        """Update checkpoint with failed experiment.

        Loads current checkpoint, adds the experiment to failed list,
        removes it from pending, and saves the updated checkpoint.

        Args:
            experiment_index: Index of failed experiment
            error: Error message or description

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> checkpoint_mgr.update_failed(5, "API timeout")
        """
        checkpoint_data = self.load_checkpoint()

        # Add to failed
        if experiment_index not in checkpoint_data["failed"]:
            checkpoint_data["failed"].append(experiment_index)

        # Remove from pending if present
        if experiment_index in checkpoint_data["pending"]:
            checkpoint_data["pending"].remove(experiment_index)

        # Store error details in metadata
        if "failures" not in checkpoint_data["metadata"]:
            checkpoint_data["metadata"]["failures"] = {}

        checkpoint_data["metadata"]["failures"][str(experiment_index)] = {
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

        # Save updated checkpoint
        self.save_checkpoint(
            completed=checkpoint_data["completed"],
            pending=checkpoint_data["pending"],
            failed=checkpoint_data["failed"],
            metadata=checkpoint_data["metadata"],
        )

        self.logger.error(
            "checkpoint_updated_failed",
            experiment_index=experiment_index,
            error=error,
            total_failed=len(checkpoint_data["failed"]),
        )

    def _validate_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """Validate checkpoint data structure.

        Args:
            checkpoint_data: Checkpoint data to validate

        Raises:
            ValueError: If checkpoint structure is invalid
        """
        required_fields = ["completed", "pending", "failed", "metadata"]
        for field in required_fields:
            if field not in checkpoint_data:
                raise ValueError(f"Checkpoint missing required field: {field}")

        # Validate that indices don't overlap
        completed_set = set(checkpoint_data["completed"])
        pending_set = set(checkpoint_data["pending"])
        failed_set = set(checkpoint_data["failed"])

        overlap_completed_pending = completed_set & pending_set
        if overlap_completed_pending:
            self.logger.warning(
                "checkpoint_validation_warning",
                issue="completed_pending_overlap",
                indices=list(overlap_completed_pending),
            )

        # Note: An index can be in both failed and pending (for retry)
        # but should not be in both failed and completed
        overlap_completed_failed = completed_set & failed_set
        if overlap_completed_failed:
            self.logger.warning(
                "checkpoint_validation_warning",
                issue="completed_failed_overlap",
                indices=list(overlap_completed_failed),
            )

    def _get_created_at(self) -> str:
        """Get creation timestamp from existing checkpoint or create new one.

        Returns:
            ISO format timestamp
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    data = json.load(f)
                    return data.get("created_at", datetime.now().isoformat())
            except Exception:
                pass

        return datetime.now().isoformat()

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of checkpoint progress.

        Returns:
            Dictionary with progress statistics:
                - total: Total experiments
                - completed: Number completed
                - pending: Number pending
                - failed: Number failed
                - completion_rate: Percentage completed
                - created_at: Checkpoint creation time
                - last_updated: Last update time

        Examples:
            >>> checkpoint_mgr = CheckpointManager(Path("./outputs"))
            >>> summary = checkpoint_mgr.get_progress_summary()
            >>> print(f"Progress: {summary['completion_rate']:.1f}%")
            >>> print(f"Completed: {summary['completed']}/{summary['total']}")
        """
        checkpoint_data = self.load_checkpoint()

        completed_count = len(checkpoint_data["completed"])
        pending_count = len(checkpoint_data["pending"])
        failed_count = len(checkpoint_data["failed"])
        total = completed_count + pending_count + failed_count

        completion_rate = (completed_count / total * 100) if total > 0 else 0

        summary = {
            "total": total,
            "completed": completed_count,
            "pending": pending_count,
            "failed": failed_count,
            "completion_rate": completion_rate,
            "created_at": checkpoint_data.get("created_at"),
            "last_updated": checkpoint_data.get("last_updated"),
        }

        self.logger.info(
            "progress_summary_generated",
            **summary,
        )

        return summary
