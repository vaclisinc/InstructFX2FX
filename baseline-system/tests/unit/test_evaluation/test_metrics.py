"""Unit tests for metrics collection and aggregation."""

import pytest
import json
import csv
from pathlib import Path
from datetime import datetime
from judge_system.evaluation.metrics import ExperimentMetrics, MetricsCollector


class TestExperimentMetrics:
    """Test suite for ExperimentMetrics dataclass."""

    def test_creation_with_valid_data(self):
        """Test creating ExperimentMetrics with valid data."""
        metrics = ExperimentMetrics(
            experiment_id="exp_001",
            timestamp="2025-10-27T12:00:00",
            description="Test description",
            parameters={"reverb": {"decay": 0.5}},
            scores={"cosine_similarity": 0.85},
            audio_metrics={"loudness": -12.5},
            metadata={"instrument": "guitar"},
        )

        assert metrics.experiment_id == "exp_001"
        assert metrics.timestamp == "2025-10-27T12:00:00"
        assert metrics.description == "Test description"
        assert metrics.parameters == {"reverb": {"decay": 0.5}}
        assert metrics.scores == {"cosine_similarity": 0.85}
        assert metrics.audio_metrics == {"loudness": -12.5}
        assert metrics.metadata == {"instrument": "guitar"}

    def test_to_dict_conversion(self):
        """Test converting ExperimentMetrics to dictionary."""
        metrics = ExperimentMetrics(
            experiment_id="exp_002",
            timestamp="2025-10-27T13:00:00",
            description="Another test",
            parameters={"eq": {"freq": 1000}},
            scores={"total_score": 75.5},
            audio_metrics={"spectral_centroid": 2500.0},
            metadata={"effect_type": "eq"},
        )

        result = metrics.to_dict()

        assert isinstance(result, dict)
        assert result["experiment_id"] == "exp_002"
        assert result["parameters"] == {"eq": {"freq": 1000}}
        assert result["scores"] == {"total_score": 75.5}
        assert result["audio_metrics"] == {"spectral_centroid": 2500.0}

    def test_empty_parameters_and_metrics(self):
        """Test creating metrics with empty dicts."""
        metrics = ExperimentMetrics(
            experiment_id="exp_003",
            timestamp="2025-10-27T14:00:00",
            description="Empty metrics test",
            parameters={},
            scores={},
            audio_metrics={},
            metadata={},
        )

        assert metrics.parameters == {}
        assert metrics.scores == {}
        assert metrics.audio_metrics == {}
        assert metrics.metadata == {}


class TestMetricsCollector:
    """Test suite for MetricsCollector class."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        return str(tmp_path / "metrics_output")

    @pytest.fixture
    def collector(self, temp_output_dir):
        """Create MetricsCollector instance."""
        return MetricsCollector(temp_output_dir)

    @pytest.fixture
    def sample_metrics(self):
        """Create sample experiment metrics."""
        return [
            ExperimentMetrics(
                experiment_id="exp_001",
                timestamp="2025-10-27T10:00:00",
                description="First experiment",
                parameters={"reverb": {"decay": 0.5, "wet_dry": 0.3}},
                scores={"cosine_similarity": 0.85, "total_score": 85.0},
                audio_metrics={"loudness": -12.5, "spectral_centroid": 2500.0},
                metadata={"instrument": "guitar", "effect_type": "reverb"},
            ),
            ExperimentMetrics(
                experiment_id="exp_002",
                timestamp="2025-10-27T10:05:00",
                description="Second experiment",
                parameters={"eq": {"freq": 1000, "gain": 3.0}},
                scores={"cosine_similarity": 0.78, "total_score": 78.0},
                audio_metrics={"loudness": -14.2, "spectral_centroid": 2200.0},
                metadata={"instrument": "drums", "effect_type": "eq"},
            ),
            ExperimentMetrics(
                experiment_id="exp_003",
                timestamp="2025-10-27T10:10:00",
                description="Third experiment",
                parameters={"compressor": {"threshold": -20, "ratio": 4.0}},
                scores={"cosine_similarity": 0.92, "total_score": 92.0},
                audio_metrics={"loudness": -10.8, "spectral_centroid": 2800.0},
                metadata={"instrument": "piano", "effect_type": "compressor"},
            ),
        ]

    def test_initialization(self, collector, temp_output_dir):
        """Test MetricsCollector initialization."""
        assert collector.output_dir == Path(temp_output_dir)
        assert collector.output_dir.exists()
        assert collector.experiments == []

    def test_collect_single_experiment(self, collector, sample_metrics):
        """Test collecting a single experiment."""
        collector.collect(sample_metrics[0])

        assert len(collector.experiments) == 1
        assert collector.experiments[0].experiment_id == "exp_001"

    def test_collect_multiple_experiments(self, collector, sample_metrics):
        """Test collecting multiple experiments."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        assert len(collector.experiments) == 3
        assert collector.experiments[0].experiment_id == "exp_001"
        assert collector.experiments[1].experiment_id == "exp_002"
        assert collector.experiments[2].experiment_id == "exp_003"

    def test_compute_statistics_empty_collection(self, collector):
        """Test computing statistics with no experiments."""
        with pytest.raises(ValueError, match="No experiments collected"):
            collector.compute_statistics()

    def test_compute_statistics_single_experiment(self, collector, sample_metrics):
        """Test computing statistics with single experiment."""
        collector.collect(sample_metrics[0])
        stats = collector.compute_statistics()

        assert stats["total_experiments"] == 1
        assert "scores" in stats
        assert "audio_metrics" in stats

        # Check score statistics
        assert "cosine_similarity" in stats["scores"]
        assert stats["scores"]["cosine_similarity"]["mean"] == 0.85
        assert stats["scores"]["cosine_similarity"]["count"] == 1
        assert stats["scores"]["cosine_similarity"]["min"] == 0.85
        assert stats["scores"]["cosine_similarity"]["max"] == 0.85

        # Check audio metric statistics
        assert "loudness" in stats["audio_metrics"]
        assert stats["audio_metrics"]["loudness"]["mean"] == -12.5

    def test_compute_statistics_multiple_experiments(self, collector, sample_metrics):
        """Test computing statistics with multiple experiments."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        stats = collector.compute_statistics()

        assert stats["total_experiments"] == 3

        # Check cosine_similarity statistics
        cs_stats = stats["scores"]["cosine_similarity"]
        assert cs_stats["count"] == 3
        assert cs_stats["mean"] == pytest.approx((0.85 + 0.78 + 0.92) / 3)
        assert cs_stats["min"] == 0.78
        assert cs_stats["max"] == 0.92
        assert cs_stats["std"] > 0

        # Check confidence interval exists
        assert "ci_95" in cs_stats
        assert isinstance(cs_stats["ci_95"], tuple)
        assert len(cs_stats["ci_95"]) == 2

        # Check total_score statistics
        ts_stats = stats["scores"]["total_score"]
        assert ts_stats["mean"] == pytest.approx((85.0 + 78.0 + 92.0) / 3)

        # Check loudness statistics
        loudness_stats = stats["audio_metrics"]["loudness"]
        assert loudness_stats["mean"] == pytest.approx((-12.5 - 14.2 - 10.8) / 3)

    def test_confidence_interval_calculation(self, collector, sample_metrics):
        """Test that confidence intervals are calculated correctly."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        stats = collector.compute_statistics()
        cs_stats = stats["scores"]["cosine_similarity"]

        # CI should contain the mean
        ci_lower, ci_upper = cs_stats["ci_95"]
        assert ci_lower <= cs_stats["mean"] <= ci_upper

        # CI should be wider than zero for multiple samples
        assert ci_upper > ci_lower

    def test_export_json_empty_collection(self, collector):
        """Test exporting JSON with no experiments."""
        with pytest.raises(ValueError, match="No experiments collected"):
            collector.export_json()

    def test_export_json_default_filepath(self, collector, sample_metrics):
        """Test JSON export with default filepath."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        filepath = collector.export_json()

        assert Path(filepath).exists()
        assert Path(filepath).name == "metrics.json"

        # Verify JSON content
        with open(filepath, "r") as f:
            data = json.load(f)

        assert "experiments" in data
        assert "statistics" in data
        assert "exported_at" in data
        assert len(data["experiments"]) == 3

    def test_export_json_custom_filepath(
        self, collector, sample_metrics, temp_output_dir
    ):
        """Test JSON export with custom filepath."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        custom_path = str(Path(temp_output_dir) / "custom_metrics.json")
        filepath = collector.export_json(custom_path)

        assert filepath == custom_path
        assert Path(filepath).exists()

        # Verify content structure
        with open(filepath, "r") as f:
            data = json.load(f)

        assert len(data["experiments"]) == 3
        assert data["experiments"][0]["experiment_id"] == "exp_001"

    def test_export_json_content_validation(self, collector, sample_metrics):
        """Test that exported JSON contains all required fields."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        filepath = collector.export_json()

        with open(filepath, "r") as f:
            data = json.load(f)

        # Check experiment structure
        exp = data["experiments"][0]
        assert "experiment_id" in exp
        assert "timestamp" in exp
        assert "description" in exp
        assert "parameters" in exp
        assert "scores" in exp
        assert "audio_metrics" in exp
        assert "metadata" in exp

        # Check statistics structure
        stats = data["statistics"]
        assert "total_experiments" in stats
        assert "scores" in stats
        assert "audio_metrics" in stats

    def test_export_csv_empty_collection(self, collector):
        """Test exporting CSV with no experiments."""
        with pytest.raises(ValueError, match="No experiments collected"):
            collector.export_csv()

    def test_export_csv_default_filepath(self, collector, sample_metrics):
        """Test CSV export with default filepath."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        filepath = collector.export_csv()

        assert Path(filepath).exists()
        assert Path(filepath).name == "metrics.csv"

        # Verify CSV content
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3

    def test_export_csv_custom_filepath(
        self, collector, sample_metrics, temp_output_dir
    ):
        """Test CSV export with custom filepath."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        custom_path = str(Path(temp_output_dir) / "custom_metrics.csv")
        filepath = collector.export_csv(custom_path)

        assert filepath == custom_path
        assert Path(filepath).exists()

    def test_export_csv_column_structure(self, collector, sample_metrics):
        """Test that CSV has correct columns and structure."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        filepath = collector.export_csv()

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check base columns
        assert "experiment_id" in rows[0]
        assert "timestamp" in rows[0]
        assert "description" in rows[0]
        assert "parameters" in rows[0]
        assert "metadata" in rows[0]

        # Check score columns
        assert "score_cosine_similarity" in rows[0]
        assert "score_total_score" in rows[0]

        # Check audio metric columns
        assert "audio_loudness" in rows[0]
        assert "audio_spectral_centroid" in rows[0]

    def test_export_csv_content_validation(self, collector, sample_metrics):
        """Test that CSV contains correct data."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        filepath = collector.export_csv()

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check first row
        assert rows[0]["experiment_id"] == "exp_001"
        assert rows[0]["description"] == "First experiment"
        assert rows[0]["score_cosine_similarity"] == "0.85"
        assert rows[0]["audio_loudness"] == "-12.5"

        # Check parameters are JSON strings
        params = json.loads(rows[0]["parameters"])
        assert "reverb" in params

    def test_clear_experiments(self, collector, sample_metrics):
        """Test clearing all collected experiments."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        assert len(collector.experiments) == 3

        collector.clear()

        assert len(collector.experiments) == 0

    def test_get_experiments_by_metadata(self, collector, sample_metrics):
        """Test filtering experiments by metadata."""
        for metrics in sample_metrics:
            collector.collect(metrics)

        # Filter by instrument
        guitar_exps = collector.get_experiments_by_metadata("instrument", "guitar")
        assert len(guitar_exps) == 1
        assert guitar_exps[0].experiment_id == "exp_001"

        # Filter by effect_type
        reverb_exps = collector.get_experiments_by_metadata("effect_type", "reverb")
        assert len(reverb_exps) == 1
        assert reverb_exps[0].experiment_id == "exp_001"

        # Filter with no matches
        no_match = collector.get_experiments_by_metadata("instrument", "vocals")
        assert len(no_match) == 0

    def test_get_experiments_count(self, collector, sample_metrics):
        """Test getting count of collected experiments."""
        assert collector.get_experiments_count() == 0

        collector.collect(sample_metrics[0])
        assert collector.get_experiments_count() == 1

        for metrics in sample_metrics[1:]:
            collector.collect(metrics)
        assert collector.get_experiments_count() == 3

        collector.clear()
        assert collector.get_experiments_count() == 0

    def test_mixed_metric_types(self, collector):
        """Test handling experiments with varying metrics."""
        # First experiment has metric A and B
        exp1 = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp="2025-10-27T10:00:00",
            description="Test 1",
            parameters={},
            scores={"metric_a": 1.0, "metric_b": 2.0},
            audio_metrics={},
            metadata={},
        )

        # Second experiment has metric B and C
        exp2 = ExperimentMetrics(
            experiment_id="exp_2",
            timestamp="2025-10-27T10:01:00",
            description="Test 2",
            parameters={},
            scores={"metric_b": 3.0, "metric_c": 4.0},
            audio_metrics={},
            metadata={},
        )

        collector.collect(exp1)
        collector.collect(exp2)

        stats = collector.compute_statistics()

        # metric_a should only have 1 sample
        assert stats["scores"]["metric_a"]["count"] == 1
        assert stats["scores"]["metric_a"]["mean"] == 1.0

        # metric_b should have 2 samples
        assert stats["scores"]["metric_b"]["count"] == 2
        assert stats["scores"]["metric_b"]["mean"] == 2.5

        # metric_c should only have 1 sample
        assert stats["scores"]["metric_c"]["count"] == 1
        assert stats["scores"]["metric_c"]["mean"] == 4.0

    def test_non_numeric_values_ignored(self, collector):
        """Test that non-numeric score values are ignored in statistics."""
        exp = ExperimentMetrics(
            experiment_id="exp_1",
            timestamp="2025-10-27T10:00:00",
            description="Test",
            parameters={},
            scores={
                "numeric_score": 5.0,
                "string_score": "not_a_number",
                "none_score": None,
            },
            audio_metrics={"loudness": -12.0},
            metadata={},
        )

        collector.collect(exp)
        stats = collector.compute_statistics()

        # Only numeric_score should appear in statistics
        assert "numeric_score" in stats["scores"]
        assert "string_score" not in stats["scores"]
        assert "none_score" not in stats["scores"]

    def test_output_directory_creation(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        nested_path = tmp_path / "level1" / "level2" / "level3"
        collector = MetricsCollector(str(nested_path))

        assert nested_path.exists()
        assert nested_path.is_dir()
