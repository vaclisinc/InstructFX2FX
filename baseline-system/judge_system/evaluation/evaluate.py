"""Pipeline evaluator for baseline system evaluation and experiment tracking.

This module provides the PipelineEvaluator class for running evaluation experiments,
collecting metrics, and tracking results with structured logging.
"""

import subprocess
import structlog
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from .metrics import ExperimentMetrics, MetricsCollector
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest, ScoringResponse

# Configure structlog for experiment tracking
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger()


def generate_experiment_id() -> str:
    """Generate unique experiment ID with timestamp and random suffix.

    Format: exp_{timestamp}_{random}
    Example: exp_20251027_183045_a7f3

    Returns:
        Unique experiment ID string
    """
    import random
    import string

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"exp_{timestamp}_{random_suffix}"


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash for reproducibility tracking.

    Returns:
        Git commit hash (short format) or None if not in git repository
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        logger.warning("Failed to get git commit hash")
        return None


class PipelineEvaluator:
    """Evaluate baseline pipeline performance with experiment tracking.

    This class orchestrates evaluation of the baseline system by running the
    pipeline on test inputs, collecting metrics, and tracking experiments with
    structured logging. It integrates with the scoring system and metrics collector
    to provide comprehensive evaluation capabilities.

    Attributes:
        config: Evaluation configuration dictionary
        metrics_collector: MetricsCollector instance for aggregating results
        scoring_system: Optional ScoringSystem instance for scoring evaluations
        logger: Structured logger bound with experiment context
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline evaluator.

        Args:
            config: Configuration dictionary with keys:
                - output_dir: Directory path for output files (required)
                - scoring_system: Optional ScoringSystem instance
                - git_tracking: Whether to track git commit hash (default: True)
                - log_level: Logging level (default: "info")

        Raises:
            ValueError: If required config keys are missing
        """
        if "output_dir" not in config:
            raise ValueError("Configuration must include 'output_dir'")

        self.config = config
        self.output_dir = Path(config["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics collector
        self.metrics_collector = MetricsCollector(str(self.output_dir))

        # Get optional scoring system
        self.scoring_system: Optional[ScoringSystem] = config.get("scoring_system")

        # Configuration flags
        self.git_tracking_enabled = config.get("git_tracking", True)

        # Bind logger with evaluator context
        self.logger = logger.bind(
            evaluator="PipelineEvaluator",
            output_dir=str(self.output_dir)
        )

        self.logger.info(
            "PipelineEvaluator initialized",
            git_tracking=self.git_tracking_enabled,
            has_scoring_system=self.scoring_system is not None
        )

    async def evaluate_single(
        self,
        description: str,
        audio_path: str,
        parameters: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ExperimentMetrics:
        """Run pipeline and collect metrics for single input.

        This method evaluates a single audio sample with a description, optionally
        using provided parameters. If parameters are not provided, the pipeline
        should generate them (this requires integration with generation module).

        Args:
            description: Text description of desired audio effect
            audio_path: Path to input audio file
            parameters: Optional pre-generated parameters. If None, pipeline generates them.
            metadata: Optional additional metadata to track with experiment

        Returns:
            ExperimentMetrics containing all collected metrics and results

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If description is empty
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        audio_path_obj = Path(audio_path)
        if not audio_path_obj.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Generate experiment ID
        experiment_id = generate_experiment_id()

        # Get git commit hash if tracking enabled
        git_hash = get_git_commit_hash() if self.git_tracking_enabled else None

        # Bind experiment context to logger
        exp_logger = self.logger.bind(
            experiment_id=experiment_id,
            description=description,
            audio_path=str(audio_path),
            git_hash=git_hash
        )

        exp_logger.info("Starting single experiment evaluation")

        # Initialize scores and audio_metrics
        scores: Dict[str, float] = {}
        audio_metrics: Dict[str, float] = {}

        # TODO: If parameters is None, integrate with generation module to generate them
        # For now, require parameters to be provided
        if parameters is None:
            raise NotImplementedError(
                "Parameter generation not yet integrated. Please provide parameters explicitly."
            )

        exp_logger.info("Using provided parameters", parameter_count=len(parameters))

        # Run scoring if scoring system available
        if self.scoring_system:
            exp_logger.info("Running scoring evaluation")

            try:
                scoring_request = ScoringRequest(
                    description=description,
                    parameters=parameters,
                    iteration=0
                )

                # Score parameters
                scoring_response: ScoringResponse = await self.scoring_system.score_parameters(
                    scoring_request
                )

                # Extract scores from response
                scores["overall_score"] = scoring_response.overall_score
                scores["confidence"] = scoring_response.confidence

                # Add dimension scores
                for dim in scoring_response.dimensions:
                    scores[f"dim_{dim.name}"] = dim.score

                exp_logger.info(
                    "Scoring completed",
                    overall_score=scoring_response.overall_score,
                    confidence=scoring_response.confidence
                )

            except Exception as e:
                exp_logger.error("Scoring failed", error=str(e), error_type=type(e).__name__)
                # Continue with empty scores rather than failing entire evaluation
                scores["error"] = -1.0

        else:
            exp_logger.warning("No scoring system available, skipping scoring")

        # TODO: Extract audio metrics (loudness, spectral features, etc.)
        # This requires integration with audio processing module
        # For now, leave audio_metrics empty or extract basic features
        exp_logger.info("Audio metrics extraction placeholder")

        # Build metadata
        full_metadata = {
            "git_hash": git_hash,
            "audio_file": str(audio_path),
            "timestamp": datetime.now().isoformat(),
        }

        # Add user-provided metadata
        if metadata:
            full_metadata.update(metadata)

        # Create experiment metrics
        experiment_metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            description=description,
            parameters=parameters,
            scores=scores,
            audio_metrics=audio_metrics,
            metadata=full_metadata
        )

        # Collect metrics
        self.metrics_collector.collect(experiment_metrics)

        exp_logger.info(
            "Single experiment evaluation completed",
            scores=scores,
            audio_metrics=audio_metrics
        )

        return experiment_metrics

    async def evaluate_batch(
        self,
        descriptions: List[str],
        audio_paths: List[str],
        parameters_list: Optional[List[Dict[str, Any]]] = None,
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[ExperimentMetrics]:
        """Run evaluation on batch of inputs.

        Evaluates multiple audio samples with corresponding descriptions in sequence.
        Collects metrics for each sample and aggregates results.

        Args:
            descriptions: List of text descriptions
            audio_paths: List of audio file paths
            parameters_list: Optional list of pre-generated parameters for each sample.
                           If None, pipeline generates them (not yet implemented).
            metadata_list: Optional list of metadata dicts for each sample

        Returns:
            List of ExperimentMetrics for each evaluated sample

        Raises:
            ValueError: If input lists have mismatched lengths
        """
        # Validate input lengths
        if len(descriptions) != len(audio_paths):
            raise ValueError(
                f"Length mismatch: {len(descriptions)} descriptions vs "
                f"{len(audio_paths)} audio paths"
            )

        if parameters_list is not None and len(parameters_list) != len(descriptions):
            raise ValueError(
                f"Length mismatch: {len(parameters_list)} parameters vs "
                f"{len(descriptions)} descriptions"
            )

        if metadata_list is not None and len(metadata_list) != len(descriptions):
            raise ValueError(
                f"Length mismatch: {len(metadata_list)} metadata items vs "
                f"{len(descriptions)} descriptions"
            )

        batch_size = len(descriptions)
        self.logger.info("Starting batch evaluation", batch_size=batch_size)

        results: List[ExperimentMetrics] = []

        for i in range(batch_size):
            description = descriptions[i]
            audio_path = audio_paths[i]
            parameters = parameters_list[i] if parameters_list else None
            metadata = metadata_list[i] if metadata_list else None

            self.logger.info(
                f"Evaluating sample {i+1}/{batch_size}",
                description=description[:50] + "..." if len(description) > 50 else description
            )

            try:
                metrics = await self.evaluate_single(
                    description=description,
                    audio_path=audio_path,
                    parameters=parameters,
                    metadata=metadata
                )
                results.append(metrics)

            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate sample {i+1}/{batch_size}",
                    error=str(e),
                    error_type=type(e).__name__,
                    description=description,
                    audio_path=audio_path
                )
                # Continue with remaining samples rather than failing entire batch
                continue

        self.logger.info(
            "Batch evaluation completed",
            total_samples=batch_size,
            successful_evaluations=len(results),
            failed_evaluations=batch_size - len(results)
        )

        return results

    def generate_report(self, format: str = "text") -> str:
        """Generate evaluation report with all metrics.

        Creates a comprehensive report summarizing all collected experiments,
        including aggregate statistics, score distributions, and key findings.

        Args:
            format: Report format - "text", "markdown", or "json" (default: "text")

        Returns:
            Formatted report string

        Raises:
            ValueError: If no experiments have been collected or invalid format
        """
        if self.metrics_collector.get_experiments_count() == 0:
            raise ValueError("No experiments collected. Cannot generate report.")

        valid_formats = ["text", "markdown", "json"]
        if format not in valid_formats:
            raise ValueError(f"Invalid format '{format}'. Must be one of {valid_formats}")

        self.logger.info("Generating evaluation report", format=format)

        # Compute statistics
        try:
            stats = self.metrics_collector.compute_statistics()
        except Exception as e:
            self.logger.error("Failed to compute statistics", error=str(e))
            raise

        if format == "json":
            import json
            report_data = {
                "total_experiments": stats["total_experiments"],
                "statistics": stats,
                "experiments": [exp.to_dict() for exp in self.metrics_collector.experiments]
            }
            return json.dumps(report_data, indent=2)

        # Generate text or markdown report
        lines = []

        if format == "markdown":
            lines.append("# Evaluation Report")
            lines.append("")
            lines.append(f"**Generated**: {datetime.now().isoformat()}")
            lines.append(f"**Total Experiments**: {stats['total_experiments']}")
            lines.append("")
            lines.append("## Score Statistics")
            lines.append("")
        else:
            lines.append("=" * 60)
            lines.append("EVALUATION REPORT")
            lines.append("=" * 60)
            lines.append(f"Generated: {datetime.now().isoformat()}")
            lines.append(f"Total Experiments: {stats['total_experiments']}")
            lines.append("")
            lines.append("SCORE STATISTICS")
            lines.append("-" * 60)

        # Add score statistics
        for metric_name, metric_stats in stats["scores"].items():
            if format == "markdown":
                lines.append(f"### {metric_name}")
                lines.append("")
                lines.append(f"- **Mean**: {metric_stats['mean']:.2f}")
                lines.append(f"- **Std**: {metric_stats['std']:.2f}")
                lines.append(f"- **Min**: {metric_stats['min']:.2f}")
                lines.append(f"- **Max**: {metric_stats['max']:.2f}")
                lines.append(f"- **Count**: {metric_stats['count']}")
                ci_lower, ci_upper = metric_stats['ci_95']
                lines.append(f"- **95% CI**: [{ci_lower:.2f}, {ci_upper:.2f}]")
                lines.append("")
            else:
                lines.append(f"  {metric_name}:")
                lines.append(f"    Mean:     {metric_stats['mean']:.2f}")
                lines.append(f"    Std:      {metric_stats['std']:.2f}")
                lines.append(f"    Min:      {metric_stats['min']:.2f}")
                lines.append(f"    Max:      {metric_stats['max']:.2f}")
                lines.append(f"    Count:    {metric_stats['count']}")
                ci_lower, ci_upper = metric_stats['ci_95']
                lines.append(f"    95% CI:   [{ci_lower:.2f}, {ci_upper:.2f}]")
                lines.append("")

        # Add audio metrics statistics
        if stats["audio_metrics"]:
            if format == "markdown":
                lines.append("## Audio Metrics Statistics")
                lines.append("")
            else:
                lines.append("AUDIO METRICS STATISTICS")
                lines.append("-" * 60)

            for metric_name, metric_stats in stats["audio_metrics"].items():
                if format == "markdown":
                    lines.append(f"### {metric_name}")
                    lines.append("")
                    lines.append(f"- **Mean**: {metric_stats['mean']:.2f}")
                    lines.append(f"- **Std**: {metric_stats['std']:.2f}")
                    lines.append(f"- **Min**: {metric_stats['min']:.2f}")
                    lines.append(f"- **Max**: {metric_stats['max']:.2f}")
                    lines.append("")
                else:
                    lines.append(f"  {metric_name}:")
                    lines.append(f"    Mean:     {metric_stats['mean']:.2f}")
                    lines.append(f"    Std:      {metric_stats['std']:.2f}")
                    lines.append(f"    Min:      {metric_stats['min']:.2f}")
                    lines.append(f"    Max:      {metric_stats['max']:.2f}")
                    lines.append("")

        if format == "text":
            lines.append("=" * 60)

        report = "\n".join(lines)

        self.logger.info("Report generated", format=format, length=len(report))

        return report

    def export_results(self, json_path: Optional[str] = None, csv_path: Optional[str] = None) -> Dict[str, str]:
        """Export collected metrics to JSON and/or CSV formats.

        Args:
            json_path: Optional path for JSON output. If None, uses default location.
            csv_path: Optional path for CSV output. If None, uses default location.

        Returns:
            Dictionary with paths to exported files: {"json": path, "csv": path}

        Raises:
            ValueError: If no experiments have been collected
        """
        if self.metrics_collector.get_experiments_count() == 0:
            raise ValueError("No experiments collected. Cannot export results.")

        exported_files = {}

        if json_path is not None or csv_path is None:
            # Export JSON (default if nothing specified)
            json_file = self.metrics_collector.export_json(json_path)
            exported_files["json"] = json_file
            self.logger.info("Exported JSON results", path=json_file)

        if csv_path is not None:
            # Export CSV
            csv_file = self.metrics_collector.export_csv(csv_path)
            exported_files["csv"] = csv_file
            self.logger.info("Exported CSV results", path=csv_file)

        return exported_files

    def clear_experiments(self) -> None:
        """Clear all collected experiments.

        Useful for starting fresh evaluation without creating new evaluator instance.
        """
        self.metrics_collector.clear()
        self.logger.info("Cleared all collected experiments")

    def get_experiment_count(self) -> int:
        """Get total number of collected experiments.

        Returns:
            Number of experiments in collection
        """
        return self.metrics_collector.get_experiments_count()


__all__ = [
    "PipelineEvaluator",
    "generate_experiment_id",
    "get_git_commit_hash",
]
