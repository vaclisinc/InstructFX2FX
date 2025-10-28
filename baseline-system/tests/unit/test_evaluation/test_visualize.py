"""Unit tests for visualization module.

Tests the ResultVisualizer class and all its plotting methods using
realistic mock ExperimentMetrics data.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json

from judge_system.evaluation.metrics import ExperimentMetrics
from judge_system.evaluation.visualize import ResultVisualizer


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_metrics():
    """Create realistic sample experiment metrics for testing."""
    metrics = []

    # Create 20 realistic experiments with varying scores
    for i in range(20):
        experiment = ExperimentMetrics(
            experiment_id=f"exp_{i:03d}",
            timestamp=datetime.now().isoformat(),
            description=f"Test experiment {i}",
            parameters={
                "reverb": {
                    "delay_time": 0.03 + i * 0.001,
                    "decay": 0.5 + i * 0.02,
                    "wet_gain": 1.0 + i * 0.1,
                },
                "eq": {
                    "low_gain": -2.0 + i * 0.2,
                    "mid_gain": 0.0 + i * 0.1,
                    "high_gain": 1.0 + i * 0.15,
                }
            },
            scores={
                "cosine_similarity": 0.6 + i * 0.015,
                "total_score": 70.0 + i * 1.5,
                "quality_score": 75.0 + i * 1.0,
            },
            audio_metrics={
                "loudness": -20.0 + i * 0.5,
                "spectral_centroid": 2000.0 + i * 50,
                "rms_energy": 0.1 + i * 0.01,
            },
            metadata={
                "model": "claude-3-5-sonnet-20241022",
                "instrument": "synth",
                "effect_type": "reverb",
            }
        )
        metrics.append(experiment)

    return metrics


@pytest.fixture
def minimal_metrics():
    """Create minimal metrics with just 2 experiments."""
    return [
        ExperimentMetrics(
            experiment_id="exp_001",
            timestamp="2025-01-01T10:00:00",
            description="Test 1",
            parameters={"gain": 0.5},
            scores={"cosine_similarity": 0.75},
            audio_metrics={"loudness": -18.0},
            metadata={"model": "test"}
        ),
        ExperimentMetrics(
            experiment_id="exp_002",
            timestamp="2025-01-01T11:00:00",
            description="Test 2",
            parameters={"gain": 0.8},
            scores={"cosine_similarity": 0.82},
            audio_metrics={"loudness": -16.0},
            metadata={"model": "test"}
        )
    ]


class TestResultVisualizerInit:
    """Test ResultVisualizer initialization."""

    def test_init_creates_output_directory(self, temp_output_dir):
        """Test that initialization creates output directory."""
        output_dir = Path(temp_output_dir) / "visualizations"
        visualizer = ResultVisualizer(str(output_dir))

        assert output_dir.exists()
        assert output_dir.is_dir()
        assert visualizer.output_dir == output_dir

    def test_init_with_custom_style(self, temp_output_dir):
        """Test initialization with custom matplotlib style."""
        visualizer = ResultVisualizer(temp_output_dir, style="ggplot")
        assert visualizer.style == "ggplot"

    def test_init_with_custom_figsize(self, temp_output_dir):
        """Test initialization with custom figure size."""
        visualizer = ResultVisualizer(temp_output_dir, figsize=(10, 6))
        assert visualizer.figsize == (10, 6)


class TestPlotScoreDistribution:
    """Test score distribution plotting."""

    def test_plot_score_distribution_creates_file(self, temp_output_dir, sample_metrics):
        """Test that score distribution plot is created successfully."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.plot_score_distribution(sample_metrics)

        assert Path(output_path).exists()
        assert Path(output_path).suffix == ".png"
        assert Path(output_path).stat().st_size > 0

    def test_plot_score_distribution_custom_filepath(self, temp_output_dir, sample_metrics):
        """Test score distribution with custom filepath."""
        visualizer = ResultVisualizer(temp_output_dir)
        custom_path = str(Path(temp_output_dir) / "custom_scores.png")
        output_path = visualizer.plot_score_distribution(sample_metrics, filepath=custom_path)

        assert output_path == custom_path
        assert Path(custom_path).exists()

    def test_plot_score_distribution_empty_metrics_raises_error(self, temp_output_dir):
        """Test that empty metrics list raises ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)

        with pytest.raises(ValueError, match="Cannot plot empty metrics list"):
            visualizer.plot_score_distribution([])

    def test_plot_score_distribution_no_scores_raises_error(self, temp_output_dir):
        """Test that metrics with no scores raise ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)
        metrics = [
            ExperimentMetrics(
                experiment_id="exp_001",
                timestamp="2025-01-01T10:00:00",
                description="Test",
                parameters={},
                scores={},  # Empty scores
                audio_metrics={},
                metadata={}
            )
        ]

        with pytest.raises(ValueError, match="No score metrics found"):
            visualizer.plot_score_distribution(metrics)

    def test_plot_score_distribution_single_score_type(self, temp_output_dir):
        """Test plotting with single score type."""
        visualizer = ResultVisualizer(temp_output_dir)
        metrics = [
            ExperimentMetrics(
                experiment_id=f"exp_{i:03d}",
                timestamp="2025-01-01T10:00:00",
                description=f"Test {i}",
                parameters={},
                scores={"total_score": 70.0 + i * 5},
                audio_metrics={},
                metadata={}
            )
            for i in range(10)
        ]

        output_path = visualizer.plot_score_distribution(metrics)
        assert Path(output_path).exists()

    def test_plot_score_distribution_multiple_score_types(self, temp_output_dir, sample_metrics):
        """Test plotting with multiple score types creates subplots."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.plot_score_distribution(sample_metrics)

        # Verify file is larger (contains multiple subplots)
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 10000  # Reasonable size for multi-plot


class TestPlotParameterAnalysis:
    """Test parameter analysis and correlation plotting."""

    def test_plot_parameter_analysis_creates_file(self, temp_output_dir, sample_metrics):
        """Test that parameter analysis plot is created successfully."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.plot_parameter_analysis(sample_metrics)

        assert Path(output_path).exists()
        assert Path(output_path).suffix == ".png"
        assert Path(output_path).stat().st_size > 0

    def test_plot_parameter_analysis_custom_filepath(self, temp_output_dir, sample_metrics):
        """Test parameter analysis with custom filepath."""
        visualizer = ResultVisualizer(temp_output_dir)
        custom_path = str(Path(temp_output_dir) / "custom_params.png")
        output_path = visualizer.plot_parameter_analysis(sample_metrics, filepath=custom_path)

        assert output_path == custom_path
        assert Path(custom_path).exists()

    def test_plot_parameter_analysis_empty_metrics_raises_error(self, temp_output_dir):
        """Test that empty metrics list raises ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)

        with pytest.raises(ValueError, match="Cannot plot empty metrics list"):
            visualizer.plot_parameter_analysis([])

    def test_plot_parameter_analysis_no_numeric_params_raises_error(self, temp_output_dir):
        """Test that metrics with no numeric parameters raise ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)
        metrics = [
            ExperimentMetrics(
                experiment_id="exp_001",
                timestamp="2025-01-01T10:00:00",
                description="Test",
                parameters={"name": "string_param"},  # Non-numeric
                scores={},
                audio_metrics={},
                metadata={}
            )
        ]

        with pytest.raises(ValueError, match="No numeric parameters"):
            visualizer.plot_parameter_analysis(metrics)

    def test_plot_parameter_analysis_nested_parameters(self, temp_output_dir, sample_metrics):
        """Test that nested parameters are properly flattened and analyzed."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.plot_parameter_analysis(sample_metrics)

        # Should handle nested parameters like reverb.delay_time
        assert Path(output_path).exists()

    def test_plot_parameter_analysis_insufficient_data_raises_error(self, temp_output_dir):
        """Test that insufficient numeric data raises ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)
        metrics = [
            ExperimentMetrics(
                experiment_id="exp_001",
                timestamp="2025-01-01T10:00:00",
                description="Test",
                parameters={"gain": 0.5},  # Only 1 numeric field
                scores={},
                audio_metrics={},
                metadata={}
            )
        ]

        with pytest.raises(ValueError, match="Need at least 2 numeric"):
            visualizer.plot_parameter_analysis(metrics)


class TestPlotTimeSeries:
    """Test time series plotting."""

    def test_plot_time_series_creates_file(self, temp_output_dir, sample_metrics):
        """Test that time series plot is created successfully."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.plot_time_series(sample_metrics)

        assert Path(output_path).exists()
        assert Path(output_path).suffix == ".png"
        assert Path(output_path).stat().st_size > 0

    def test_plot_time_series_custom_filepath(self, temp_output_dir, sample_metrics):
        """Test time series with custom filepath."""
        visualizer = ResultVisualizer(temp_output_dir)
        custom_path = str(Path(temp_output_dir) / "custom_timeseries.png")
        output_path = visualizer.plot_time_series(sample_metrics, filepath=custom_path)

        assert output_path == custom_path
        assert Path(custom_path).exists()

    def test_plot_time_series_empty_metrics_raises_error(self, temp_output_dir):
        """Test that empty metrics list raises ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)

        with pytest.raises(ValueError, match="Cannot plot empty metrics list"):
            visualizer.plot_time_series([])

    def test_plot_time_series_no_metrics_raises_error(self, temp_output_dir):
        """Test that metrics with no scores/audio_metrics raise ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)
        metrics = [
            ExperimentMetrics(
                experiment_id="exp_001",
                timestamp="2025-01-01T10:00:00",
                description="Test",
                parameters={},
                scores={},
                audio_metrics={},
                metadata={}
            )
        ]

        with pytest.raises(ValueError, match="No metrics found to plot"):
            visualizer.plot_time_series(metrics)

    def test_plot_time_series_sorts_by_timestamp(self, temp_output_dir):
        """Test that time series properly sorts metrics by timestamp."""
        visualizer = ResultVisualizer(temp_output_dir)

        # Create metrics with out-of-order timestamps
        metrics = [
            ExperimentMetrics(
                experiment_id="exp_002",
                timestamp="2025-01-01T11:00:00",
                description="Second",
                parameters={},
                scores={"score": 80.0},
                audio_metrics={},
                metadata={}
            ),
            ExperimentMetrics(
                experiment_id="exp_001",
                timestamp="2025-01-01T10:00:00",
                description="First",
                parameters={},
                scores={"score": 70.0},
                audio_metrics={},
                metadata={}
            ),
        ]

        output_path = visualizer.plot_time_series(metrics)
        assert Path(output_path).exists()


class TestCreateComparisonTable:
    """Test comparison table generation."""

    def test_create_comparison_table_returns_html(self, temp_output_dir, sample_metrics):
        """Test that comparison table returns valid HTML."""
        visualizer = ResultVisualizer(temp_output_dir)

        configs = ["config_a", "config_b"]
        metrics_by_config = {
            "config_a": sample_metrics[:10],
            "config_b": sample_metrics[10:],
        }

        html = visualizer.create_comparison_table(configs, metrics_by_config)

        assert isinstance(html, str)
        assert "<table" in html
        assert "</table>" in html
        assert "config_a" in html
        assert "config_b" in html

    def test_create_comparison_table_includes_metrics(self, temp_output_dir, sample_metrics):
        """Test that comparison table includes all score metrics."""
        visualizer = ResultVisualizer(temp_output_dir)

        configs = ["config_a"]
        metrics_by_config = {"config_a": sample_metrics}

        html = visualizer.create_comparison_table(configs, metrics_by_config)

        # Should include score names from sample_metrics
        assert "cosine_similarity" in html
        assert "total_score" in html
        assert "quality_score" in html

    def test_create_comparison_table_shows_statistics(self, temp_output_dir, sample_metrics):
        """Test that comparison table shows mean ± std."""
        visualizer = ResultVisualizer(temp_output_dir)

        configs = ["config_a"]
        metrics_by_config = {"config_a": sample_metrics}

        html = visualizer.create_comparison_table(configs, metrics_by_config)

        # Should contain statistical notation
        assert "±" in html

    def test_create_comparison_table_empty_configs_raises_error(self, temp_output_dir):
        """Test that empty configs raises ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)

        with pytest.raises(ValueError, match="No configurations provided"):
            visualizer.create_comparison_table([], {})

    def test_create_comparison_table_missing_config_raises_error(self, temp_output_dir, sample_metrics):
        """Test that missing config in metrics_by_config raises ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)

        configs = ["config_a", "config_b"]
        metrics_by_config = {"config_a": sample_metrics}  # Missing config_b

        with pytest.raises(ValueError, match="All configs must have corresponding metrics"):
            visualizer.create_comparison_table(configs, metrics_by_config)

    def test_create_comparison_table_handles_na_values(self, temp_output_dir):
        """Test that comparison table handles missing metrics with N/A."""
        visualizer = ResultVisualizer(temp_output_dir)

        configs = ["config_a", "config_b"]
        metrics_by_config = {
            "config_a": [
                ExperimentMetrics(
                    experiment_id="exp_001",
                    timestamp="2025-01-01T10:00:00",
                    description="Test",
                    parameters={},
                    scores={"score_a": 75.0},
                    audio_metrics={},
                    metadata={}
                )
            ],
            "config_b": [
                ExperimentMetrics(
                    experiment_id="exp_002",
                    timestamp="2025-01-01T11:00:00",
                    description="Test",
                    parameters={},
                    scores={"score_b": 80.0},  # Different score key
                    audio_metrics={},
                    metadata={}
                )
            ],
        }

        html = visualizer.create_comparison_table(configs, metrics_by_config)
        assert "N/A" in html


class TestGenerateDashboard:
    """Test HTML dashboard generation."""

    def test_generate_dashboard_creates_file(self, temp_output_dir, sample_metrics):
        """Test that dashboard HTML file is created successfully."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.generate_dashboard(sample_metrics)

        assert Path(output_path).exists()
        assert Path(output_path).suffix == ".html"
        assert Path(output_path).stat().st_size > 0

    def test_generate_dashboard_custom_path(self, temp_output_dir, sample_metrics):
        """Test dashboard generation with custom output path."""
        visualizer = ResultVisualizer(temp_output_dir)
        custom_path = str(Path(temp_output_dir) / "custom_dashboard.html")
        output_path = visualizer.generate_dashboard(sample_metrics, output_path=custom_path)

        assert output_path == custom_path
        assert Path(custom_path).exists()

    def test_generate_dashboard_contains_html_structure(self, temp_output_dir, sample_metrics):
        """Test that dashboard contains valid HTML structure."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.generate_dashboard(sample_metrics)

        with open(output_path, 'r') as f:
            content = f.read()

        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content
        assert "<head>" in content
        assert "<body>" in content

    def test_generate_dashboard_includes_plots(self, temp_output_dir, sample_metrics):
        """Test that dashboard references generated plot files."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.generate_dashboard(sample_metrics)

        with open(output_path, 'r') as f:
            content = f.read()

        # Should reference plot image files
        assert "score_distribution.png" in content
        assert "time_series.png" in content
        assert '<img' in content

    def test_generate_dashboard_includes_statistics(self, temp_output_dir, sample_metrics):
        """Test that dashboard includes summary statistics."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.generate_dashboard(sample_metrics)

        with open(output_path, 'r') as f:
            content = f.read()

        # Should include stats section
        assert "Summary Statistics" in content
        assert "Total Experiments" in content
        assert str(len(sample_metrics)) in content

    def test_generate_dashboard_empty_metrics_raises_error(self, temp_output_dir):
        """Test that empty metrics list raises ValueError."""
        visualizer = ResultVisualizer(temp_output_dir)

        with pytest.raises(ValueError, match="Cannot generate dashboard from empty metrics"):
            visualizer.generate_dashboard([])

    def test_generate_dashboard_creates_all_plots(self, temp_output_dir, sample_metrics):
        """Test that dashboard generation creates all plot files."""
        visualizer = ResultVisualizer(temp_output_dir)
        output_path = visualizer.generate_dashboard(sample_metrics)

        # Check that individual plot files exist
        output_dir = Path(output_path).parent
        assert (output_dir / "score_distribution.png").exists()
        assert (output_dir / "time_series.png").exists()

    def test_generate_dashboard_handles_missing_parameter_analysis(self, temp_output_dir, minimal_metrics):
        """Test dashboard generation when parameter analysis fails."""
        visualizer = ResultVisualizer(temp_output_dir)

        # minimal_metrics may not have enough data for correlation
        output_path = visualizer.generate_dashboard(minimal_metrics)

        # Dashboard should still be created
        assert Path(output_path).exists()

        with open(output_path, 'r') as f:
            content = f.read()

        # Should still have other sections
        assert "Summary Statistics" in content
        assert "Score Distributions" in content


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_visualization_workflow(self, temp_output_dir, sample_metrics):
        """Test complete visualization workflow with all methods."""
        visualizer = ResultVisualizer(temp_output_dir)

        # Generate all plots
        score_path = visualizer.plot_score_distribution(sample_metrics)
        param_path = visualizer.plot_parameter_analysis(sample_metrics)
        time_path = visualizer.plot_time_series(sample_metrics)

        # Create comparison table
        configs = ["config_a", "config_b"]
        metrics_by_config = {
            "config_a": sample_metrics[:10],
            "config_b": sample_metrics[10:],
        }
        table_html = visualizer.create_comparison_table(configs, metrics_by_config)

        # Generate dashboard
        dashboard_path = visualizer.generate_dashboard(sample_metrics)

        # Verify all outputs exist
        assert Path(score_path).exists()
        assert Path(param_path).exists()
        assert Path(time_path).exists()
        assert isinstance(table_html, str)
        assert Path(dashboard_path).exists()

    def test_visualizations_with_real_file_checks(self, temp_output_dir, sample_metrics):
        """Test that generated files are valid and non-empty."""
        visualizer = ResultVisualizer(temp_output_dir)

        # Generate visualizations
        score_path = visualizer.plot_score_distribution(sample_metrics)
        param_path = visualizer.plot_parameter_analysis(sample_metrics)
        time_path = visualizer.plot_time_series(sample_metrics)
        dashboard_path = visualizer.generate_dashboard(sample_metrics)

        # Check file sizes (should be substantial)
        assert Path(score_path).stat().st_size > 5000
        assert Path(param_path).stat().st_size > 5000
        assert Path(time_path).stat().st_size > 5000
        assert Path(dashboard_path).stat().st_size > 1000
