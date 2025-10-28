"""Metrics collection and aggregation for experiment evaluation.

This module provides data structures and utilities for collecting, aggregating,
and exporting experiment metrics from the baseline pipeline.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import csv
import numpy as np
from datetime import datetime


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics.

    Attributes:
        experiment_id: Unique identifier for the experiment
        timestamp: ISO 8601 formatted timestamp
        description: Text description of the audio effect
        parameters: Generated effect parameters dictionary
        scores: Scoring metrics (e.g., cosine_similarity, total_score)
        audio_metrics: Audio quality metrics (e.g., loudness, spectral features)
        metadata: Additional metadata (model, instrument, effect_type, etc.)
    """

    experiment_id: str
    timestamp: str
    description: str
    parameters: Dict[str, Any]
    scores: Dict[str, float]
    audio_metrics: Dict[str, float]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format.

        Returns:
            Dictionary representation of metrics
        """
        return asdict(self)


class MetricsCollector:
    """Collect and aggregate metrics from experiments.

    This class provides functionality to collect experiment metrics,
    compute aggregate statistics, and export results in various formats.

    Attributes:
        output_dir: Directory path for output files
        experiments: List of collected experiment metrics
    """

    def __init__(self, output_dir: str):
        """Initialize metrics collector.

        Args:
            output_dir: Directory path where output files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[ExperimentMetrics] = []

    def collect(self, experiment: ExperimentMetrics) -> None:
        """Add experiment metrics to collection.

        Args:
            experiment: ExperimentMetrics instance to add to collection
        """
        self.experiments.append(experiment)

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute aggregate statistics across experiments.

        Computes mean, standard deviation, min, max, and 95% confidence intervals
        for all numeric scores and audio metrics across collected experiments.

        Returns:
            Dictionary containing statistics for each metric:
            {
                "scores": {
                    "metric_name": {
                        "mean": float,
                        "std": float,
                        "min": float,
                        "max": float,
                        "count": int,
                        "ci_95": (lower, upper)
                    },
                    ...
                },
                "audio_metrics": { ... },
                "total_experiments": int
            }

        Raises:
            ValueError: If no experiments have been collected
        """
        if not self.experiments:
            raise ValueError("No experiments collected. Cannot compute statistics.")

        # Aggregate all scores and audio metrics
        score_values: Dict[str, List[float]] = {}
        audio_metric_values: Dict[str, List[float]] = {}

        for exp in self.experiments:
            for metric_name, value in exp.scores.items():
                if isinstance(value, (int, float)):
                    if metric_name not in score_values:
                        score_values[metric_name] = []
                    score_values[metric_name].append(float(value))

            for metric_name, value in exp.audio_metrics.items():
                if isinstance(value, (int, float)):
                    if metric_name not in audio_metric_values:
                        audio_metric_values[metric_name] = []
                    audio_metric_values[metric_name].append(float(value))

        # Compute statistics for each metric
        def compute_metric_stats(values: List[float]) -> Dict[str, Any]:
            """Compute statistics for a list of values."""
            arr = np.array(values)
            n = len(arr)

            # Compute mean and std
            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

            # Compute 95% confidence interval (1.96 * stderr)
            stderr = std / np.sqrt(n) if n > 1 else 0.0
            ci_95_lower = mean - 1.96 * stderr
            ci_95_upper = mean + 1.96 * stderr

            return {
                "mean": mean,
                "std": std,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "count": n,
                "ci_95": (ci_95_lower, ci_95_upper),
            }

        # Compute statistics for all scores
        score_stats = {
            name: compute_metric_stats(values)
            for name, values in score_values.items()
        }

        # Compute statistics for all audio metrics
        audio_metric_stats = {
            name: compute_metric_stats(values)
            for name, values in audio_metric_values.items()
        }

        return {
            "scores": score_stats,
            "audio_metrics": audio_metric_stats,
            "total_experiments": len(self.experiments),
        }

    def export_json(self, filepath: Optional[str] = None) -> str:
        """Export metrics to JSON format.

        Args:
            filepath: Optional path for output file. If not provided,
                     defaults to {output_dir}/metrics.json

        Returns:
            Path to the exported JSON file

        Raises:
            ValueError: If no experiments have been collected
        """
        if not self.experiments:
            raise ValueError("No experiments collected. Cannot export.")

        if filepath is None:
            filepath = str(self.output_dir / "metrics.json")
        else:
            filepath = str(Path(filepath))

        # Convert experiments to dictionaries
        data = {
            "experiments": [exp.to_dict() for exp in self.experiments],
            "statistics": self.compute_statistics(),
            "exported_at": datetime.now().isoformat(),
        }

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return filepath

    def export_csv(self, filepath: Optional[str] = None) -> str:
        """Export metrics to CSV format.

        Exports a flattened view of metrics with one row per experiment.
        Complex nested structures (parameters, metadata) are serialized as JSON strings.

        Args:
            filepath: Optional path for output file. If not provided,
                     defaults to {output_dir}/metrics.csv

        Returns:
            Path to the exported CSV file

        Raises:
            ValueError: If no experiments have been collected
        """
        if not self.experiments:
            raise ValueError("No experiments collected. Cannot export.")

        if filepath is None:
            filepath = str(self.output_dir / "metrics.csv")
        else:
            filepath = str(Path(filepath))

        # Collect all unique score and audio metric keys
        all_score_keys = set()
        all_audio_metric_keys = set()

        for exp in self.experiments:
            all_score_keys.update(exp.scores.keys())
            all_audio_metric_keys.update(exp.audio_metrics.keys())

        # Sort keys for consistent column ordering
        score_keys = sorted(all_score_keys)
        audio_metric_keys = sorted(all_audio_metric_keys)

        # Define CSV columns
        base_columns = ["experiment_id", "timestamp", "description"]
        score_columns = [f"score_{key}" for key in score_keys]
        audio_columns = [f"audio_{key}" for key in audio_metric_keys]
        complex_columns = ["parameters", "metadata"]

        fieldnames = base_columns + score_columns + audio_columns + complex_columns

        # Write CSV file
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for exp in self.experiments:
                row = {
                    "experiment_id": exp.experiment_id,
                    "timestamp": exp.timestamp,
                    "description": exp.description,
                    "parameters": json.dumps(exp.parameters),
                    "metadata": json.dumps(exp.metadata),
                }

                # Add scores
                for key in score_keys:
                    row[f"score_{key}"] = exp.scores.get(key, "")

                # Add audio metrics
                for key in audio_metric_keys:
                    row[f"audio_{key}"] = exp.audio_metrics.get(key, "")

                writer.writerow(row)

        return filepath

    def clear(self) -> None:
        """Clear all collected experiments.

        Useful for starting fresh collection without creating a new instance.
        """
        self.experiments.clear()

    def get_experiments_by_metadata(
        self, key: str, value: Any
    ) -> List[ExperimentMetrics]:
        """Filter experiments by metadata field.

        Args:
            key: Metadata field key to filter by
            value: Value to match

        Returns:
            List of experiments matching the metadata criteria
        """
        return [
            exp
            for exp in self.experiments
            if exp.metadata.get(key) == value
        ]

    def get_experiments_count(self) -> int:
        """Get total number of collected experiments.

        Returns:
            Number of experiments in collection
        """
        return len(self.experiments)
