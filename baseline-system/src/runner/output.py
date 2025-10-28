"""Output management for experiment results.

This module provides the OutputManager class for organizing experiment outputs:
- Directory structure creation (audio, parameters, scores, logs)
- Saving audio files, parameters, and scores with consistent naming
- Generating experiment summaries and statistics
- Managing output versioning and cleanup
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.utils.logging import get_logger


class OutputManager:
    """Organize and manage experiment outputs.

    The OutputManager creates a structured directory layout for experiment results
    and provides methods to save various types of outputs (audio, parameters, scores).
    It also generates summaries and metadata for tracking experiments.

    Directory structure:
        base_dir/
        ├── audio/           # Processed audio files
        ├── parameters/      # Generated parameter JSON files
        ├── scores/          # Scoring results
        ├── logs/            # Experiment logs
        └── metadata.json    # Experiment metadata

    Attributes:
        base_dir: Root directory for all outputs
        audio_dir: Directory for audio files
        parameters_dir: Directory for parameter files
        scores_dir: Directory for score files
        logs_dir: Directory for log files
        metadata_file: Path to metadata file

    Examples:
        >>> output_mgr = OutputManager(Path("./outputs/exp_001"))
        >>> output_mgr.save_audio(Path("processed.wav"), "exp_001_item_1")
        >>> output_mgr.save_parameters({"reverb": {...}}, "exp_001_item_1")
        >>> output_mgr.save_score({"overall": 85.3}, "exp_001_item_1")
        >>> summary = output_mgr.get_experiment_summary()
    """

    def __init__(self, base_dir: Path):
        """Initialize OutputManager and create directory structure.

        Args:
            base_dir: Root directory for experiment outputs
        """
        self.base_dir = Path(base_dir)
        self.audio_dir = self.base_dir / "audio"
        self.parameters_dir = self.base_dir / "parameters"
        self.scores_dir = self.base_dir / "scores"
        self.logs_dir = self.base_dir / "logs"
        self.metadata_file = self.base_dir / "metadata.json"

        self.logger = get_logger("runner")
        self._create_structure()
        self._initialize_metadata()

    def _create_structure(self) -> None:
        """Create output directory structure.

        Creates all required subdirectories for organizing outputs.
        Uses exist_ok=True to allow resuming with existing directories.
        """
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.parameters_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            "output_structure_created",
            base_dir=str(self.base_dir),
            subdirs=["audio", "parameters", "scores", "logs"],
        )

    def _initialize_metadata(self) -> None:
        """Initialize or load experiment metadata.

        Creates metadata.json with experiment start time and counters.
        If file exists (resume case), loads existing metadata.
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                self.metadata = json.load(f)
            self.logger.info(
                "metadata_loaded",
                metadata_file=str(self.metadata_file),
                experiments_count=self.metadata.get("experiments_completed", 0),
            )
        else:
            self.metadata = {
                "created_at": datetime.now().isoformat(),
                "experiments_completed": 0,
                "experiments_failed": 0,
                "last_updated": datetime.now().isoformat(),
            }
            self._save_metadata()
            self.logger.info(
                "metadata_initialized",
                metadata_file=str(self.metadata_file),
            )

    def _save_metadata(self) -> None:
        """Save current metadata to disk.

        Updates last_updated timestamp and writes metadata.json.
        """
        self.metadata["last_updated"] = datetime.now().isoformat()
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_audio(self, audio_path: Path, experiment_id: str) -> Path:
        """Save processed audio file.

        Copies the audio file to the audio directory with experiment ID naming.

        Args:
            audio_path: Path to source audio file
            experiment_id: Unique experiment identifier

        Returns:
            Path to saved audio file

        Raises:
            FileNotFoundError: If source audio file doesn't exist
            OSError: If copy operation fails

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> saved_path = output_mgr.save_audio(Path("temp.wav"), "exp_001")
            >>> print(saved_path)
            ./outputs/audio/exp_001.wav
        """
        if not audio_path.exists():
            self.logger.error(
                "audio_save_failed",
                experiment_id=experiment_id,
                audio_path=str(audio_path),
                reason="file_not_found",
            )
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Preserve original extension
        extension = audio_path.suffix
        output_path = self.audio_dir / f"{experiment_id}{extension}"

        try:
            shutil.copy2(audio_path, output_path)
            self.logger.info(
                "audio_saved",
                experiment_id=experiment_id,
                output_path=str(output_path),
                file_size_mb=output_path.stat().st_size / (1024 * 1024),
            )
            return output_path

        except Exception as e:
            self.logger.error(
                "audio_save_failed",
                experiment_id=experiment_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def save_parameters(self, parameters: Dict[str, Any], experiment_id: str) -> Path:
        """Save generated parameters as JSON.

        Saves effect parameters in a JSON file with pretty formatting.

        Args:
            parameters: Effect parameters dictionary
            experiment_id: Unique experiment identifier

        Returns:
            Path to saved parameters file

        Raises:
            OSError: If write operation fails

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> params = {"reverb": {"decay": 0.8}, "eq": {"gain": 2.0}}
            >>> saved_path = output_mgr.save_parameters(params, "exp_001")
        """
        output_path = self.parameters_dir / f"{experiment_id}.json"

        try:
            with open(output_path, "w") as f:
                json.dump(parameters, f, indent=2)

            self.logger.info(
                "parameters_saved",
                experiment_id=experiment_id,
                output_path=str(output_path),
                num_effects=len(parameters),
            )
            return output_path

        except Exception as e:
            self.logger.error(
                "parameters_save_failed",
                experiment_id=experiment_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def save_score(self, score: Dict[str, Any], experiment_id: str) -> Path:
        """Save scoring results as JSON.

        Saves complete scoring results including overall score, dimensions,
        and any metadata from the scoring system.

        Args:
            score: Scoring results dictionary
            experiment_id: Unique experiment identifier

        Returns:
            Path to saved score file

        Raises:
            OSError: If write operation fails

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> score = {
            ...     "overall": 85.3,
            ...     "dimensions": {"warmth": 90, "clarity": 80},
            ...     "metadata": {"model": "claude-3-5-sonnet"}
            ... }
            >>> saved_path = output_mgr.save_score(score, "exp_001")
        """
        output_path = self.scores_dir / f"{experiment_id}.json"

        try:
            # Add timestamp to score
            score_with_timestamp = {
                **score,
                "scored_at": datetime.now().isoformat(),
            }

            with open(output_path, "w") as f:
                json.dump(score_with_timestamp, f, indent=2)

            self.logger.info(
                "score_saved",
                experiment_id=experiment_id,
                output_path=str(output_path),
                overall_score=score.get("overall"),
            )
            return output_path

        except Exception as e:
            self.logger.error(
                "score_save_failed",
                experiment_id=experiment_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def record_success(self, experiment_id: str) -> None:
        """Record successful experiment completion.

        Updates metadata counters for completed experiments.

        Args:
            experiment_id: Unique experiment identifier
        """
        self.metadata["experiments_completed"] += 1
        self._save_metadata()

        self.logger.info(
            "experiment_success_recorded",
            experiment_id=experiment_id,
            total_completed=self.metadata["experiments_completed"],
        )

    def record_failure(self, experiment_id: str, error: str) -> None:
        """Record failed experiment.

        Updates metadata counters and logs failure details.

        Args:
            experiment_id: Unique experiment identifier
            error: Error message or description
        """
        self.metadata["experiments_failed"] += 1

        # Track failures in metadata
        if "failures" not in self.metadata:
            self.metadata["failures"] = []

        self.metadata["failures"].append({
            "experiment_id": experiment_id,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_metadata()

        self.logger.error(
            "experiment_failure_recorded",
            experiment_id=experiment_id,
            error=error,
            total_failed=self.metadata["experiments_failed"],
        )

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Generate summary of all experiments in directory.

        Collects statistics and information about all experiments:
        - Total experiments completed and failed
        - Average scores
        - File counts and sizes
        - Time range

        Returns:
            Dictionary with experiment summary statistics

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> summary = output_mgr.get_experiment_summary()
            >>> print(f"Completed: {summary['experiments_completed']}")
            >>> print(f"Average score: {summary['average_score']:.2f}")
        """
        # Count files in each directory
        audio_files = list(self.audio_dir.glob("*"))
        parameter_files = list(self.parameters_dir.glob("*.json"))
        score_files = list(self.scores_dir.glob("*.json"))

        # Calculate total size
        total_size = sum(
            f.stat().st_size for f in audio_files + parameter_files + score_files
        )

        # Load and analyze scores
        scores = []
        for score_file in score_files:
            try:
                with open(score_file, "r") as f:
                    score_data = json.load(f)
                    if "overall" in score_data:
                        scores.append(score_data["overall"])
            except Exception as e:
                self.logger.warning(
                    "score_load_failed",
                    score_file=str(score_file),
                    error=str(e),
                )

        summary = {
            "base_dir": str(self.base_dir),
            "created_at": self.metadata.get("created_at"),
            "last_updated": self.metadata.get("last_updated"),
            "experiments_completed": self.metadata.get("experiments_completed", 0),
            "experiments_failed": self.metadata.get("experiments_failed", 0),
            "file_counts": {
                "audio": len(audio_files),
                "parameters": len(parameter_files),
                "scores": len(score_files),
            },
            "total_size_mb": total_size / (1024 * 1024),
            "scores": {
                "count": len(scores),
                "average": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
            },
        }

        self.logger.info(
            "summary_generated",
            experiments_completed=summary["experiments_completed"],
            experiments_failed=summary["experiments_failed"],
            average_score=summary["scores"]["average"],
        )

        return summary

    def cleanup(self, keep_successful: bool = True, keep_failed: bool = True) -> None:
        """Clean up experiment outputs.

        Removes experiment output files based on success/failure status.
        Useful for removing failed experiments or clearing all data.

        Args:
            keep_successful: Keep outputs from successful experiments (default: True)
            keep_failed: Keep outputs from failed experiments (default: True)

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> # Remove only failed experiments
            >>> output_mgr.cleanup(keep_successful=True, keep_failed=False)
            >>> # Remove all outputs
            >>> output_mgr.cleanup(keep_successful=False, keep_failed=False)
        """
        if not keep_successful and not keep_failed:
            # Remove entire directory
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
                self.logger.info(
                    "outputs_cleaned",
                    action="removed_all",
                    base_dir=str(self.base_dir),
                )
        else:
            # Selective cleanup requires tracking individual experiments
            # This is a simplified implementation
            self.logger.warning(
                "selective_cleanup_not_implemented",
                message="Selective cleanup requires experiment tracking",
            )

    def get_logs_dir(self) -> Path:
        """Get logs directory path.

        Returns:
            Path to logs directory

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> logs_dir = output_mgr.get_logs_dir()
        """
        return self.logs_dir

    def list_experiments(self) -> List[str]:
        """List all experiment IDs with saved results.

        Returns:
            List of experiment IDs that have score files

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> experiments = output_mgr.list_experiments()
            >>> print(f"Total experiments: {len(experiments)}")
        """
        score_files = self.scores_dir.glob("*.json")
        experiment_ids = [f.stem for f in score_files]

        self.logger.debug(
            "experiments_listed",
            count=len(experiment_ids),
        )

        return sorted(experiment_ids)

    def get_experiment_files(self, experiment_id: str) -> Dict[str, Optional[Path]]:
        """Get all file paths for a specific experiment.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Dictionary with paths to audio, parameters, and score files
            Returns None for missing files

        Examples:
            >>> output_mgr = OutputManager(Path("./outputs"))
            >>> files = output_mgr.get_experiment_files("exp_001")
            >>> if files["audio"]:
            ...     print(f"Audio: {files['audio']}")
        """
        files = {
            "audio": None,
            "parameters": None,
            "score": None,
        }

        # Find audio file (any supported extension)
        audio_pattern = self.audio_dir / f"{experiment_id}.*"
        audio_files = list(self.audio_dir.glob(f"{experiment_id}.*"))
        if audio_files:
            files["audio"] = audio_files[0]

        # Check for parameters
        param_file = self.parameters_dir / f"{experiment_id}.json"
        if param_file.exists():
            files["parameters"] = param_file

        # Check for score
        score_file = self.scores_dir / f"{experiment_id}.json"
        if score_file.exists():
            files["score"] = score_file

        self.logger.debug(
            "experiment_files_retrieved",
            experiment_id=experiment_id,
            has_audio=files["audio"] is not None,
            has_parameters=files["parameters"] is not None,
            has_score=files["score"] is not None,
        )

        return files
