"""Unit tests for ConfigurationComparator."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from judge_system.evaluation.compare import ConfigurationComparator
from judge_system.evaluation.metrics import ExperimentMetrics


class TestConfigurationComparator:
    """Tests for ConfigurationComparator class."""

    @pytest.fixture
    def temp_experiment_dirs(self, tmp_path):
        """Create temporary experiment directories with mock data."""
        # Create two configurations
        config_a_dir = tmp_path / "config_a"
        config_b_dir = tmp_path / "config_b"
        config_a_dir.mkdir()
        config_b_dir.mkdir()

        # Create experiments for config_a (higher scores)
        for i in range(5):
            experiment = ExperimentMetrics(
                experiment_id=f"exp_a_{i}",
                timestamp=f"2025-10-27T10:{i:02d}:00Z",
                description=f"Test experiment A {i}",
                parameters={"param1": 0.5 + i * 0.1, "param2": 1.0},
                scores={
                    "cosine_similarity": 0.85 + i * 0.02,
                    "embedding_distance": 0.15 - i * 0.02
                },
                audio_metrics={
                    "loudness": -14.0 + i * 0.5,
                    "spectral_centroid": 2000.0 + i * 100
                },
                metadata={"config": "a", "version": "1.0"}
            )

            exp_file = config_a_dir / f"experiment_{i}.json"
            with open(exp_file, 'w') as f:
                # Convert to dict manually since ExperimentMetrics may not have asdict
                exp_dict = {
                    "experiment_id": experiment.experiment_id,
                    "timestamp": experiment.timestamp,
                    "description": experiment.description,
                    "parameters": experiment.parameters,
                    "scores": experiment.scores,
                    "audio_metrics": experiment.audio_metrics,
                    "metadata": experiment.metadata
                }
                json.dump(exp_dict, f)

        # Create experiments for config_b (lower scores)
        for i in range(5):
            experiment = ExperimentMetrics(
                experiment_id=f"exp_b_{i}",
                timestamp=f"2025-10-27T11:{i:02d}:00Z",
                description=f"Test experiment B {i}",
                parameters={"param1": 0.3 + i * 0.1, "param2": 2.0},
                scores={
                    "cosine_similarity": 0.75 + i * 0.02,
                    "embedding_distance": 0.25 - i * 0.02
                },
                audio_metrics={
                    "loudness": -16.0 + i * 0.5,
                    "spectral_centroid": 1800.0 + i * 100
                },
                metadata={"config": "b", "version": "1.0"}
            )

            exp_file = config_b_dir / f"experiment_{i}.json"
            with open(exp_file, 'w') as f:
                exp_dict = {
                    "experiment_id": experiment.experiment_id,
                    "timestamp": experiment.timestamp,
                    "description": experiment.description,
                    "parameters": experiment.parameters,
                    "scores": experiment.scores,
                    "audio_metrics": experiment.audio_metrics,
                    "metadata": experiment.metadata
                }
                json.dump(exp_dict, f)

        return {"config_a": str(config_a_dir), "config_b": str(config_b_dir)}

    @pytest.fixture
    def comparator_with_data(self, temp_experiment_dirs):
        """Create comparator with loaded experiment data."""
        comparator = ConfigurationComparator()
        comparator.load_experiments([
            temp_experiment_dirs["config_a"],
            temp_experiment_dirs["config_b"]
        ])
        return comparator

    def test_initialization(self):
        """Test ConfigurationComparator initialization."""
        comparator = ConfigurationComparator()
        assert comparator.experiments == {}
        assert comparator.config_names == []

    def test_load_experiments_success(self, temp_experiment_dirs):
        """Test successfully loading experiments from directories."""
        comparator = ConfigurationComparator()
        comparator.load_experiments([
            temp_experiment_dirs["config_a"],
            temp_experiment_dirs["config_b"]
        ])

        assert len(comparator.config_names) == 2
        assert "config_a" in comparator.config_names
        assert "config_b" in comparator.config_names
        assert len(comparator.experiments["config_a"]) == 5
        assert len(comparator.experiments["config_b"]) == 5

    def test_load_experiments_empty_list_raises_error(self):
        """Test that empty directory list raises ValueError."""
        comparator = ConfigurationComparator()
        with pytest.raises(ValueError) as exc_info:
            comparator.load_experiments([])

        assert "Must provide at least one experiment directory" in str(exc_info.value)

    def test_load_experiments_nonexistent_directory_raises_error(self):
        """Test that nonexistent directory raises FileNotFoundError."""
        comparator = ConfigurationComparator()
        with pytest.raises(FileNotFoundError) as exc_info:
            comparator.load_experiments(["/nonexistent/path"])

        assert "Experiment directory not found" in str(exc_info.value)

    def test_load_experiments_file_instead_of_directory_raises_error(self, tmp_path):
        """Test that providing a file instead of directory raises ValueError."""
        # Create a file instead of directory
        test_file = tmp_path / "not_a_directory.txt"
        test_file.write_text("test")

        comparator = ConfigurationComparator()
        with pytest.raises(ValueError) as exc_info:
            comparator.load_experiments([str(test_file)])

        assert "Path is not a directory" in str(exc_info.value)

    def test_load_experiments_empty_directory_raises_error(self, tmp_path):
        """Test that directory with no valid experiments raises ValueError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        comparator = ConfigurationComparator()
        with pytest.raises(ValueError) as exc_info:
            comparator.load_experiments([str(empty_dir)])

        assert "No valid experiments found" in str(exc_info.value)

    def test_load_experiments_skips_invalid_json(self, tmp_path):
        """Test that invalid JSON files are skipped but loading continues."""
        test_dir = tmp_path / "mixed"
        test_dir.mkdir()

        # Create valid experiment
        valid_exp = test_dir / "valid.json"
        with open(valid_exp, 'w') as f:
            json.dump({
                "experiment_id": "test",
                "timestamp": "2025-10-27T10:00:00Z",
                "description": "test",
                "parameters": {"p": 1},
                "scores": {"s": 0.5},
                "audio_metrics": {"a": 1.0},
                "metadata": {}
            }, f)

        # Create invalid JSON
        invalid_exp = test_dir / "invalid.json"
        invalid_exp.write_text("not valid json {")

        comparator = ConfigurationComparator()
        # Should load successfully, skipping invalid file
        comparator.load_experiments([str(test_dir)])

        assert len(comparator.experiments["mixed"]) == 1

    def test_compare_metrics_cosine_similarity(self, comparator_with_data):
        """Test comparing cosine_similarity metric across configurations."""
        result = comparator_with_data.compare_metrics("cosine_similarity")

        assert result["metric_name"] == "cosine_similarity"
        assert "config_a" in result["config_stats"]
        assert "config_b" in result["config_stats"]

        # Config A should have higher mean (0.85-0.93)
        # Config B should have lower mean (0.75-0.83)
        assert result["config_stats"]["config_a"]["mean"] > 0.85
        assert result["config_stats"]["config_b"]["mean"] > 0.75
        assert result["config_stats"]["config_a"]["mean"] > result["config_stats"]["config_b"]["mean"]

        # Best config should be config_a
        assert result["best_config"] == "config_a"
        assert result["worst_config"] == "config_b"

    def test_compare_metrics_audio_metrics(self, comparator_with_data):
        """Test comparing audio metrics like loudness."""
        result = comparator_with_data.compare_metrics("loudness")

        assert result["metric_name"] == "loudness"
        assert "config_a" in result["config_stats"]
        assert "config_b" in result["config_stats"]

        # Check statistics are computed
        stats_a = result["config_stats"]["config_a"]
        assert "mean" in stats_a
        assert "std" in stats_a
        assert "min" in stats_a
        assert "max" in stats_a
        assert "median" in stats_a
        assert stats_a["count"] == 5

    def test_compare_metrics_no_experiments_raises_error(self):
        """Test that comparing metrics without loaded experiments raises ValueError."""
        comparator = ConfigurationComparator()
        with pytest.raises(ValueError) as exc_info:
            comparator.compare_metrics("cosine_similarity")

        assert "No experiments loaded" in str(exc_info.value)

    def test_compare_metrics_nonexistent_metric_raises_error(self, comparator_with_data):
        """Test that comparing nonexistent metric raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            comparator_with_data.compare_metrics("nonexistent_metric")

        assert "not found in any experiments" in str(exc_info.value)

    def test_statistical_significance_single_metric(self, comparator_with_data):
        """Test statistical significance test for single metric."""
        result = comparator_with_data.statistical_significance(
            "config_a", "config_b", "cosine_similarity"
        )

        assert result["metric_name"] == "cosine_similarity"
        assert result["config_a"] == "config_a"
        assert result["config_b"] == "config_b"
        assert "config_a_mean" in result
        assert "config_b_mean" in result
        assert "t_statistic" in result
        assert "p_value" in result
        assert "effect_size" in result
        assert "significant" in result
        assert "interpretation" in result

        # Config A mean should be higher
        assert result["config_a_mean"] > result["config_b_mean"]

        # Should be statistically significant (large difference)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_statistical_significance_all_metrics(self, comparator_with_data):
        """Test statistical significance for all common metrics."""
        result = comparator_with_data.statistical_significance(
            "config_a", "config_b"
        )

        assert result["config_a"] == "config_a"
        assert result["config_b"] == "config_b"
        assert "metrics_tested" in result
        assert "results" in result

        # Should test both score and audio metrics
        assert len(result["metrics_tested"]) > 0
        assert "cosine_similarity" in result["metrics_tested"]

        # Check structure of individual metric results
        for metric in result["metrics_tested"]:
            assert metric in result["results"]
            test_result = result["results"][metric]
            assert "p_value" in test_result
            assert "t_statistic" in test_result
            assert "effect_size" in test_result

    def test_statistical_significance_no_experiments_raises_error(self):
        """Test that significance test without experiments raises ValueError."""
        comparator = ConfigurationComparator()
        with pytest.raises(ValueError) as exc_info:
            comparator.statistical_significance("config_a", "config_b", "metric")

        assert "No experiments loaded" in str(exc_info.value)

    def test_statistical_significance_nonexistent_config_raises_error(self, comparator_with_data):
        """Test that significance test with nonexistent config raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            comparator_with_data.statistical_significance(
                "nonexistent", "config_b", "cosine_similarity"
            )

        assert "Configuration 'nonexistent' not found" in str(exc_info.value)

    def test_statistical_significance_insufficient_data_raises_error(self, tmp_path):
        """Test that insufficient data for t-test raises ValueError."""
        # Create config with only 1 experiment (need at least 2)
        test_dir = tmp_path / "insufficient"
        test_dir.mkdir()

        exp_file = test_dir / "exp.json"
        with open(exp_file, 'w') as f:
            json.dump({
                "experiment_id": "test",
                "timestamp": "2025-10-27T10:00:00Z",
                "description": "test",
                "parameters": {"p": 1},
                "scores": {"metric": 0.5},
                "audio_metrics": {},
                "metadata": {}
            }, f)

        # Create another config with enough data
        test_dir2 = tmp_path / "sufficient"
        test_dir2.mkdir()
        for i in range(3):
            exp_file = test_dir2 / f"exp_{i}.json"
            with open(exp_file, 'w') as f:
                json.dump({
                    "experiment_id": f"test_{i}",
                    "timestamp": f"2025-10-27T10:{i:02d}:00Z",
                    "description": "test",
                    "parameters": {"p": 1},
                    "scores": {"metric": 0.6 + i * 0.1},
                    "audio_metrics": {},
                    "metadata": {}
                }, f)

        comparator = ConfigurationComparator()
        comparator.load_experiments([str(test_dir), str(test_dir2)])

        with pytest.raises(ValueError) as exc_info:
            comparator.statistical_significance(
                "insufficient", "sufficient", "metric"
            )

        assert "Insufficient data for statistical test" in str(exc_info.value)

    def test_effect_size_interpretation(self):
        """Test Cohen's d effect size interpretation."""
        comparator = ConfigurationComparator()

        assert comparator._interpret_effect_size(0.1) == "negligible"
        assert comparator._interpret_effect_size(0.3) == "small"
        assert comparator._interpret_effect_size(0.6) == "medium"
        assert comparator._interpret_effect_size(1.0) == "large"
        assert comparator._interpret_effect_size(-0.6) == "medium"  # Negative values

    def test_cohens_d_calculation(self):
        """Test Cohen's d calculation."""
        comparator = ConfigurationComparator()

        # Two samples with clear difference
        sample_a = [10.0, 11.0, 12.0, 13.0, 14.0]
        sample_b = [5.0, 6.0, 7.0, 8.0, 9.0]

        d = comparator._cohens_d(sample_a, sample_b)

        # Should be positive and large (samples differ by ~1 pooled SD)
        assert d > 0
        assert abs(d) > 2.0  # Large effect

    def test_cohens_d_identical_samples(self):
        """Test Cohen's d with identical samples."""
        comparator = ConfigurationComparator()

        sample = [5.0, 5.0, 5.0, 5.0, 5.0]
        d = comparator._cohens_d(sample, sample)

        assert d == 0.0

    def test_t_test_calculation(self):
        """Test independent t-test calculation."""
        comparator = ConfigurationComparator()

        # Two samples with difference
        sample_a = [10.0, 11.0, 12.0, 13.0, 14.0]
        sample_b = [5.0, 6.0, 7.0, 8.0, 9.0]

        t_stat, p_value = comparator._independent_t_test(sample_a, sample_b)

        # Should have high t-statistic and low p-value
        assert abs(t_stat) > 2.0
        assert p_value < 0.05

    def test_t_test_identical_samples(self):
        """Test t-test with identical samples."""
        comparator = ConfigurationComparator()

        sample = [5.0, 5.0, 5.0, 5.0, 5.0]
        t_stat, p_value = comparator._independent_t_test(sample, sample)

        assert t_stat == 0.0
        assert p_value > 0.05  # Not significant

    def test_generate_comparison_report(self, comparator_with_data):
        """Test generating comparison report."""
        report = comparator_with_data.generate_comparison_report()

        # Check report contains expected sections
        assert "CONFIGURATION COMPARISON REPORT" in report
        assert "SUMMARY" in report
        assert "METRIC COMPARISONS" in report
        assert "STATISTICAL SIGNIFICANCE TESTS" in report

        # Check configurations are listed
        assert "config_a" in report
        assert "config_b" in report

        # Check metrics are listed
        assert "cosine_similarity" in report
        assert "loudness" in report

    def test_generate_comparison_report_saves_to_file(self, comparator_with_data, tmp_path):
        """Test saving comparison report to file."""
        output_file = tmp_path / "report.txt"
        report = comparator_with_data.generate_comparison_report(str(output_file))

        # Check file was created
        assert output_file.exists()

        # Check file contents match returned report
        with open(output_file, 'r') as f:
            file_contents = f.read()
        assert file_contents == report

    def test_generate_comparison_report_no_experiments_raises_error(self):
        """Test that generating report without experiments raises ValueError."""
        comparator = ConfigurationComparator()
        with pytest.raises(ValueError) as exc_info:
            comparator.generate_comparison_report()

        assert "No experiments loaded" in str(exc_info.value)

    def test_export_comparison_json(self, comparator_with_data, tmp_path):
        """Test exporting comparison data to JSON."""
        output_file = tmp_path / "comparison.json"
        comparator_with_data.export_comparison_json(str(output_file))

        # Check file was created
        assert output_file.exists()

        # Load and verify JSON structure
        with open(output_file, 'r') as f:
            data = json.load(f)

        assert "configurations" in data
        assert "experiment_counts" in data
        assert "metrics" in data

        assert "config_a" in data["configurations"]
        assert "config_b" in data["configurations"]

        assert data["experiment_counts"]["config_a"] == 5
        assert data["experiment_counts"]["config_b"] == 5

        # Check metrics data
        assert "cosine_similarity" in data["metrics"]
        assert "loudness" in data["metrics"]

    def test_export_comparison_json_no_experiments_raises_error(self):
        """Test that exporting JSON without experiments raises ValueError."""
        comparator = ConfigurationComparator()
        with pytest.raises(ValueError) as exc_info:
            comparator.export_comparison_json("/tmp/test.json")

        assert "No experiments loaded" in str(exc_info.value)

    def test_export_comparison_json_creates_directory(self, comparator_with_data, tmp_path):
        """Test that export creates parent directories if needed."""
        output_file = tmp_path / "subdir" / "comparison.json"
        comparator_with_data.export_comparison_json(str(output_file))

        # Check file was created with parent directory
        assert output_file.exists()
        assert output_file.parent.exists()

    def test_multiple_configurations(self, tmp_path):
        """Test comparing more than 2 configurations."""
        # Create 3 configurations
        configs = {}
        for config_name in ["config_x", "config_y", "config_z"]:
            config_dir = tmp_path / config_name
            config_dir.mkdir()
            configs[config_name] = str(config_dir)

            # Create experiments
            for i in range(3):
                exp_file = config_dir / f"exp_{i}.json"
                with open(exp_file, 'w') as f:
                    json.dump({
                        "experiment_id": f"{config_name}_{i}",
                        "timestamp": f"2025-10-27T10:{i:02d}:00Z",
                        "description": "test",
                        "parameters": {"p": 1},
                        "scores": {"metric": 0.5 + i * 0.1},
                        "audio_metrics": {},
                        "metadata": {}
                    }, f)

        comparator = ConfigurationComparator()
        comparator.load_experiments(list(configs.values()))

        assert len(comparator.config_names) == 3

        # Generate report should handle all pairwise comparisons
        report = comparator.generate_comparison_report()
        assert "config_x" in report
        assert "config_y" in report
        assert "config_z" in report

    def test_normal_cdf_approximation(self):
        """Test normal CDF approximation."""
        comparator = ConfigurationComparator()

        # Test some known values
        assert 0.4 < comparator._normal_cdf(0) < 0.6  # ~0.5
        assert 0.8 < comparator._normal_cdf(1) < 0.9  # ~0.84
        assert 0.9 < comparator._normal_cdf(2) < 1.0  # ~0.97

    def test_extract_metric_values(self, comparator_with_data):
        """Test extracting metric values from configuration."""
        values = comparator_with_data._extract_metric_values(
            "config_a", "cosine_similarity"
        )

        assert len(values) == 5
        assert all(isinstance(v, float) for v in values)
        assert all(0.85 <= v <= 0.95 for v in values)

    def test_extract_metric_values_nonexistent_metric_raises_error(self, comparator_with_data):
        """Test that extracting nonexistent metric raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            comparator_with_data._extract_metric_values(
                "config_a", "nonexistent"
            )

        assert "not found in configuration" in str(exc_info.value)

    def test_find_common_metrics(self, comparator_with_data):
        """Test finding common metrics between configurations."""
        common = comparator_with_data._find_common_metrics("config_a", "config_b")

        # Both configs have same metrics
        assert "cosine_similarity" in common
        assert "embedding_distance" in common
        assert "loudness" in common
        assert "spectral_centroid" in common

    def test_find_common_metrics_no_overlap(self, tmp_path):
        """Test finding common metrics when there's no overlap."""
        # Create two configs with different metrics
        config_a_dir = tmp_path / "config_a"
        config_b_dir = tmp_path / "config_b"
        config_a_dir.mkdir()
        config_b_dir.mkdir()

        # Config A has metric1
        with open(config_a_dir / "exp.json", 'w') as f:
            json.dump({
                "experiment_id": "test",
                "timestamp": "2025-10-27T10:00:00Z",
                "description": "test",
                "parameters": {"p": 1},
                "scores": {"metric1": 0.5},
                "audio_metrics": {},
                "metadata": {}
            }, f)

        # Config B has metric2
        with open(config_b_dir / "exp.json", 'w') as f:
            json.dump({
                "experiment_id": "test",
                "timestamp": "2025-10-27T10:00:00Z",
                "description": "test",
                "parameters": {"p": 1},
                "scores": {"metric2": 0.6},
                "audio_metrics": {},
                "metadata": {}
            }, f)

        comparator = ConfigurationComparator()
        comparator.load_experiments([str(config_a_dir), str(config_b_dir)])

        common = comparator._find_common_metrics("config_a", "config_b")
        assert len(common) == 0

    def test_comparison_with_different_experiment_counts(self, tmp_path):
        """Test comparison works with different numbers of experiments per config."""
        config_a_dir = tmp_path / "config_a"
        config_b_dir = tmp_path / "config_b"
        config_a_dir.mkdir()
        config_b_dir.mkdir()

        # Config A: 3 experiments
        for i in range(3):
            with open(config_a_dir / f"exp_{i}.json", 'w') as f:
                json.dump({
                    "experiment_id": f"a_{i}",
                    "timestamp": f"2025-10-27T10:{i:02d}:00Z",
                    "description": "test",
                    "parameters": {"p": 1},
                    "scores": {"metric": 0.8 + i * 0.05},
                    "audio_metrics": {},
                    "metadata": {}
                }, f)

        # Config B: 7 experiments
        for i in range(7):
            with open(config_b_dir / f"exp_{i}.json", 'w') as f:
                json.dump({
                    "experiment_id": f"b_{i}",
                    "timestamp": f"2025-10-27T11:{i:02d}:00Z",
                    "description": "test",
                    "parameters": {"p": 1},
                    "scores": {"metric": 0.7 + i * 0.03},
                    "audio_metrics": {},
                    "metadata": {}
                }, f)

        comparator = ConfigurationComparator()
        comparator.load_experiments([str(config_a_dir), str(config_b_dir)])

        # Should successfully compare despite different counts
        result = comparator.compare_metrics("metric")
        assert result["config_stats"]["config_a"]["count"] == 3
        assert result["config_stats"]["config_b"]["count"] == 7

        # Statistical test should still work
        sig_result = comparator.statistical_significance(
            "config_a", "config_b", "metric"
        )
        assert "t_statistic" in sig_result
        assert "p_value" in sig_result
