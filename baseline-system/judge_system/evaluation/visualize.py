"""Visualization tools for experiment results and metrics.

This module provides comprehensive visualization capabilities for analyzing
experiment metrics, including score distributions, parameter correlations,
time series analysis, and interactive HTML dashboards.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from judge_system.evaluation.metrics import ExperimentMetrics


class ResultVisualizer:
    """Visualize experiment results and metrics.

    This class provides methods for creating various visualizations of experiment
    data including histograms, heatmaps, time series plots, and comprehensive
    HTML dashboards with embedded plots.

    Attributes:
        output_dir: Directory path where visualizations will be saved
        style: Matplotlib style to use for plots (default: 'seaborn-v0_8')
        figsize: Default figure size for plots (width, height)
    """

    def __init__(
        self,
        output_dir: str,
        style: str = 'seaborn-v0_8',
        figsize: tuple = (12, 8)
    ):
        """Initialize result visualizer.

        Args:
            output_dir: Directory path where visualizations will be saved
            style: Matplotlib style to use for plots
            figsize: Default figure size (width, height) in inches
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        self.figsize = figsize

        # Set seaborn style for better-looking plots
        sns.set_theme(style="whitegrid")

    def plot_score_distribution(
        self,
        metrics: List[ExperimentMetrics],
        filepath: Optional[str] = None
    ) -> str:
        """Plot distribution of scores across experiments.

        Creates histograms showing the distribution of each score metric
        across all experiments. Multiple score types are displayed in
        subplots for easy comparison.

        Args:
            metrics: List of ExperimentMetrics to visualize
            filepath: Optional path for output file. If not provided,
                     defaults to {output_dir}/score_distribution.png

        Returns:
            Path to the saved plot file

        Raises:
            ValueError: If metrics list is empty
        """
        if not metrics:
            raise ValueError("Cannot plot empty metrics list")

        if filepath is None:
            filepath = str(self.output_dir / "score_distribution.png")

        # Collect all score types
        score_keys = set()
        for exp in metrics:
            score_keys.update(exp.scores.keys())
        score_keys = sorted(score_keys)

        if not score_keys:
            raise ValueError("No score metrics found in experiments")

        # Determine subplot layout
        n_scores = len(score_keys)
        n_cols = min(3, n_scores)
        n_rows = (n_scores + n_cols - 1) // n_cols

        # Create figure
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(self.figsize[0], self.figsize[1] * n_rows / 2)
        )

        # Ensure axes is always a list
        if n_scores == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        # Plot each score type
        for idx, score_key in enumerate(score_keys):
            ax = axes[idx]

            # Extract scores for this metric
            values = [
                exp.scores[score_key]
                for exp in metrics
                if score_key in exp.scores
            ]

            if not values:
                continue

            # Create histogram
            ax.hist(values, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{score_key} Distribution', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            mean_val = np.mean(values)
            std_val = np.std(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.legend(fontsize=9)

        # Hide unused subplots
        for idx in range(n_scores, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(
            f'Score Distributions (n={len(metrics)} experiments)',
            fontsize=14,
            fontweight='bold',
            y=1.0
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_parameter_analysis(
        self,
        metrics: List[ExperimentMetrics],
        filepath: Optional[str] = None
    ) -> str:
        """Analyze parameter ranges and their impact on scores.

        Creates correlation heatmaps showing relationships between
        generated parameters and resulting scores. This helps identify
        which parameters most strongly influence the outcomes.

        Args:
            metrics: List of ExperimentMetrics to analyze
            filepath: Optional path for output file. If not provided,
                     defaults to {output_dir}/parameter_analysis.png

        Returns:
            Path to the saved plot file

        Raises:
            ValueError: If metrics list is empty or no numeric parameters found
        """
        if not metrics:
            raise ValueError("Cannot plot empty metrics list")

        if filepath is None:
            filepath = str(self.output_dir / "parameter_analysis.png")

        # Extract numeric parameters and scores
        data_dict: Dict[str, List[float]] = {}

        for exp in metrics:
            # Flatten parameters
            for param_key, param_value in exp.parameters.items():
                if isinstance(param_value, dict):
                    # Nested parameters (e.g., reverb.delay_time)
                    for nested_key, nested_value in param_value.items():
                        if isinstance(nested_value, (int, float)):
                            key = f"{param_key}.{nested_key}"
                            if key not in data_dict:
                                data_dict[key] = []
                            data_dict[key].append(float(nested_value))
                elif isinstance(param_value, (int, float)):
                    # Flat parameters
                    if param_key not in data_dict:
                        data_dict[param_key] = []
                    data_dict[param_key].append(float(param_value))

            # Add scores
            for score_key, score_value in exp.scores.items():
                if isinstance(score_value, (int, float)):
                    key = f"score.{score_key}"
                    if key not in data_dict:
                        data_dict[key] = []
                    data_dict[key].append(float(score_value))

        if not data_dict:
            raise ValueError("No numeric parameters or scores found in experiments")

        # Ensure all lists have the same length
        expected_len = len(metrics)
        data_dict = {
            key: values for key, values in data_dict.items()
            if len(values) == expected_len
        }

        if len(data_dict) < 2:
            raise ValueError("Need at least 2 numeric parameters/scores for correlation analysis")

        # Create correlation matrix
        import pandas as pd
        df = pd.DataFrame(data_dict)
        correlation_matrix = df.corr()

        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )

        ax.set_title(
            'Parameter-Score Correlation Matrix',
            fontsize=14,
            fontweight='bold',
            pad=20
        )

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def plot_time_series(
        self,
        metrics: List[ExperimentMetrics],
        filepath: Optional[str] = None
    ) -> str:
        """Plot metrics over time/experiment sequence.

        Creates line plots showing how scores and audio metrics evolve
        over the course of experiments. Useful for tracking improvement
        or detecting drift over time.

        Args:
            metrics: List of ExperimentMetrics to visualize
            filepath: Optional path for output file. If not provided,
                     defaults to {output_dir}/time_series.png

        Returns:
            Path to the saved plot file

        Raises:
            ValueError: If metrics list is empty
        """
        if not metrics:
            raise ValueError("Cannot plot empty metrics list")

        if filepath is None:
            filepath = str(self.output_dir / "time_series.png")

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)

        # Collect all score and audio metric types
        score_keys = set()
        audio_keys = set()
        for exp in sorted_metrics:
            score_keys.update(exp.scores.keys())
            audio_keys.update(exp.audio_metrics.keys())

        score_keys = sorted(score_keys)
        audio_keys = sorted(audio_keys)

        if not score_keys and not audio_keys:
            raise ValueError("No metrics found to plot")

        # Create subplots
        n_plots = min(2, int(bool(score_keys)) + int(bool(audio_keys)))
        fig, axes = plt.subplots(n_plots, 1, figsize=(self.figsize[0], self.figsize[1] * 0.7))

        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # Plot scores over time
        if score_keys:
            ax = axes[plot_idx]
            plot_idx += 1

            for score_key in score_keys:
                values = [
                    exp.scores.get(score_key, np.nan)
                    for exp in sorted_metrics
                ]
                ax.plot(range(len(values)), values, marker='o', label=score_key, linewidth=2)

            ax.set_xlabel('Experiment Index', fontsize=10)
            ax.set_ylabel('Score Value', fontsize=10)
            ax.set_title('Scores Over Time', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        # Plot audio metrics over time
        if audio_keys:
            ax = axes[plot_idx] if n_plots > 1 else axes[0]

            for audio_key in audio_keys[:5]:  # Limit to top 5 for readability
                values = [
                    exp.audio_metrics.get(audio_key, np.nan)
                    for exp in sorted_metrics
                ]
                ax.plot(range(len(values)), values, marker='s', label=audio_key, linewidth=2)

            ax.set_xlabel('Experiment Index', fontsize=10)
            ax.set_ylabel('Metric Value', fontsize=10)
            ax.set_title('Audio Metrics Over Time', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f'Metrics Time Series (n={len(sorted_metrics)} experiments)',
            fontsize=14,
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    def create_comparison_table(
        self,
        configs: List[str],
        metrics_by_config: Dict[str, List[ExperimentMetrics]]
    ) -> str:
        """Create comparison table for different configurations.

        Generates an HTML table comparing aggregate statistics across
        different experimental configurations.

        Args:
            configs: List of configuration names
            metrics_by_config: Dictionary mapping config names to their metrics

        Returns:
            HTML string containing the comparison table

        Raises:
            ValueError: If configs is empty or metrics_by_config doesn't match
        """
        if not configs:
            raise ValueError("No configurations provided")

        if not all(config in metrics_by_config for config in configs):
            raise ValueError("All configs must have corresponding metrics")

        # Build table header
        html = ['<table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif;">']
        html.append('<thead>')
        html.append('<tr style="background-color: #4CAF50; color: white;">')
        html.append('<th style="border: 1px solid #ddd; padding: 12px; text-align: left;">Metric</th>')

        for config in configs:
            html.append(f'<th style="border: 1px solid #ddd; padding: 12px; text-align: center;">{config}</th>')

        html.append('</tr>')
        html.append('</thead>')
        html.append('<tbody>')

        # Collect all unique score keys
        all_score_keys = set()
        for metrics_list in metrics_by_config.values():
            for exp in metrics_list:
                all_score_keys.update(exp.scores.keys())

        # Add rows for each metric
        for score_key in sorted(all_score_keys):
            html.append('<tr>')
            html.append(f'<td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">{score_key}</td>')

            for config in configs:
                metrics_list = metrics_by_config[config]
                values = [
                    exp.scores.get(score_key)
                    for exp in metrics_list
                    if score_key in exp.scores
                ]

                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    cell_text = f'{mean_val:.3f} ± {std_val:.3f}'
                else:
                    cell_text = 'N/A'

                html.append(f'<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">{cell_text}</td>')

            html.append('</tr>')

        # Add experiment count row
        html.append('<tr style="background-color: #f2f2f2;">')
        html.append('<td style="border: 1px solid #ddd; padding: 12px; font-weight: bold;">Experiments</td>')

        for config in configs:
            count = len(metrics_by_config[config])
            html.append(f'<td style="border: 1px solid #ddd; padding: 12px; text-align: center;">{count}</td>')

        html.append('</tr>')
        html.append('</tbody>')
        html.append('</table>')

        return '\n'.join(html)

    def generate_dashboard(
        self,
        metrics: List[ExperimentMetrics],
        output_path: Optional[str] = None
    ) -> str:
        """Generate HTML dashboard with all visualizations.

        Creates a comprehensive HTML dashboard that embeds all plots
        and statistics for easy viewing and sharing.

        Args:
            metrics: List of ExperimentMetrics to visualize
            output_path: Optional path for output HTML file. If not provided,
                        defaults to {output_dir}/dashboard.html

        Returns:
            Path to the saved dashboard file

        Raises:
            ValueError: If metrics list is empty
        """
        if not metrics:
            raise ValueError("Cannot generate dashboard from empty metrics list")

        if output_path is None:
            output_path = str(self.output_dir / "dashboard.html")

        # Generate all plots
        score_dist_path = self.plot_score_distribution(metrics)
        time_series_path = self.plot_time_series(metrics)

        # Try to generate parameter analysis (may fail if insufficient data)
        param_analysis_path = None
        try:
            param_analysis_path = self.plot_parameter_analysis(metrics)
        except ValueError:
            pass

        # Compute statistics
        score_stats = {}
        for exp in metrics:
            for score_key, score_value in exp.scores.items():
                if isinstance(score_value, (int, float)):
                    if score_key not in score_stats:
                        score_stats[score_key] = []
                    score_stats[score_key].append(float(score_value))

        # Build HTML
        html = ['<!DOCTYPE html>']
        html.append('<html lang="en">')
        html.append('<head>')
        html.append('<meta charset="UTF-8">')
        html.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html.append('<title>Experiment Dashboard</title>')
        html.append('<style>')
        html.append('body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }')
        html.append('h1 { color: #333; text-align: center; }')
        html.append('h2 { color: #555; border-bottom: 2px solid #4CAF50; padding-bottom: 5px; }')
        html.append('.container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }')
        html.append('.section { margin: 30px 0; }')
        html.append('.plot { text-align: center; margin: 20px 0; }')
        html.append('.plot img { max-width: 100%; height: auto; border: 1px solid #ddd; }')
        html.append('.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }')
        html.append('.stat-card { background-color: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }')
        html.append('.stat-title { font-weight: bold; color: #555; margin-bottom: 5px; }')
        html.append('.stat-value { font-size: 24px; color: #333; }')
        html.append('</style>')
        html.append('</head>')
        html.append('<body>')
        html.append('<div class="container">')

        # Header
        html.append('<h1>Experiment Evaluation Dashboard</h1>')
        html.append(f'<p style="text-align: center; color: #777;">Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>')

        # Summary statistics
        html.append('<div class="section">')
        html.append('<h2>Summary Statistics</h2>')
        html.append('<div class="stats">')

        html.append('<div class="stat-card">')
        html.append('<div class="stat-title">Total Experiments</div>')
        html.append(f'<div class="stat-value">{len(metrics)}</div>')
        html.append('</div>')

        for score_key, values in sorted(score_stats.items())[:4]:  # Show top 4 metrics
            mean_val = np.mean(values)
            html.append('<div class="stat-card">')
            html.append(f'<div class="stat-title">{score_key} (mean)</div>')
            html.append(f'<div class="stat-value">{mean_val:.3f}</div>')
            html.append('</div>')

        html.append('</div>')
        html.append('</div>')

        # Score distribution plot
        html.append('<div class="section">')
        html.append('<h2>Score Distributions</h2>')
        html.append('<div class="plot">')
        html.append(f'<img src="{Path(score_dist_path).name}" alt="Score Distribution">')
        html.append('</div>')
        html.append('</div>')

        # Parameter analysis plot (if available)
        if param_analysis_path:
            html.append('<div class="section">')
            html.append('<h2>Parameter-Score Correlations</h2>')
            html.append('<div class="plot">')
            html.append(f'<img src="{Path(param_analysis_path).name}" alt="Parameter Analysis">')
            html.append('</div>')
            html.append('</div>')

        # Time series plot
        html.append('<div class="section">')
        html.append('<h2>Metrics Over Time</h2>')
        html.append('<div class="plot">')
        html.append(f'<img src="{Path(time_series_path).name}" alt="Time Series">')
        html.append('</div>')
        html.append('</div>')

        html.append('</div>')
        html.append('</body>')
        html.append('</html>')

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(html))

        return output_path
