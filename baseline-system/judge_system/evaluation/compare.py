"""Configuration comparison and statistical analysis for experiments."""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict

from judge_system.evaluation.metrics import ExperimentMetrics


class ConfigurationComparator:
    """Compare different pipeline configurations with statistical analysis."""

    def __init__(self):
        """Initialize the comparator."""
        self.experiments: Dict[str, List[ExperimentMetrics]] = {}
        self.config_names: List[str] = []

    def load_experiments(self, experiment_dirs: List[str]) -> None:
        """Load experiments from multiple directories.

        Args:
            experiment_dirs: List of directory paths containing experiment JSON files

        Raises:
            FileNotFoundError: If any directory doesn't exist
            ValueError: If no valid experiments found in directories
        """
        if not experiment_dirs:
            raise ValueError("Must provide at least one experiment directory")

        loaded_count = 0
        for exp_dir in experiment_dirs:
            dir_path = Path(exp_dir)
            if not dir_path.exists():
                raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

            if not dir_path.is_dir():
                raise ValueError(f"Path is not a directory: {exp_dir}")

            # Use directory name as configuration name
            config_name = dir_path.name
            self.experiments[config_name] = []

            # Load all JSON files from directory
            json_files = list(dir_path.glob("*.json"))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        experiment = ExperimentMetrics(**data)
                        self.experiments[config_name].append(experiment)
                        loaded_count += 1
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    # Skip invalid files but continue loading others
                    print(f"Warning: Failed to load {json_file}: {e}")
                    continue

            if self.experiments[config_name]:
                self.config_names.append(config_name)
            else:
                # Remove empty configuration
                del self.experiments[config_name]

        if loaded_count == 0:
            raise ValueError("No valid experiments found in any directory")

    def compare_metrics(self, metric_name: str) -> Dict[str, Any]:
        """Compare specific metric across configurations.

        Args:
            metric_name: Name of metric to compare (e.g., 'cosine_similarity')

        Returns:
            Dictionary containing comparison statistics:
            - config_stats: Per-config mean, std, min, max, median
            - best_config: Configuration with highest mean
            - worst_config: Configuration with lowest mean
            - overall_mean: Mean across all configurations
            - overall_std: Std across all configurations

        Raises:
            ValueError: If no configurations loaded or metric not found
        """
        if not self.experiments:
            raise ValueError("No experiments loaded. Call load_experiments() first.")

        # Collect metric values for each configuration
        config_values: Dict[str, List[float]] = defaultdict(list)

        for config_name, experiments in self.experiments.items():
            for exp in experiments:
                # Try to find metric in scores or audio_metrics
                value = None
                if metric_name in exp.scores:
                    value = exp.scores[metric_name]
                elif metric_name in exp.audio_metrics:
                    value = exp.audio_metrics[metric_name]

                if value is not None:
                    config_values[config_name].append(value)

        if not any(config_values.values()):
            raise ValueError(f"Metric '{metric_name}' not found in any experiments")

        # Compute statistics for each configuration
        config_stats = {}
        all_values = []

        for config_name in self.config_names:
            values = config_values[config_name]
            if not values:
                continue

            all_values.extend(values)
            config_stats[config_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'count': len(values)
            }

        # Identify best and worst configurations
        means = {name: stats['mean'] for name, stats in config_stats.items()}
        best_config = max(means.items(), key=lambda x: x[1])[0] if means else None
        worst_config = min(means.items(), key=lambda x: x[1])[0] if means else None

        return {
            'metric_name': metric_name,
            'config_stats': config_stats,
            'best_config': best_config,
            'worst_config': worst_config,
            'overall_mean': float(np.mean(all_values)) if all_values else 0.0,
            'overall_std': float(np.std(all_values)) if all_values else 0.0
        }

    def statistical_significance(
        self,
        config_a: str,
        config_b: str,
        metric_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Test statistical significance of differences between two configurations.

        Uses independent t-test to determine if differences are statistically significant.

        Args:
            config_a: First configuration name
            config_b: Second configuration name
            metric_name: Specific metric to test (if None, tests all common metrics)

        Returns:
            Dictionary containing:
            - metric_name: Name of metric tested
            - config_a_mean: Mean for config A
            - config_b_mean: Mean for config B
            - difference: Absolute difference between means
            - t_statistic: T-test statistic
            - p_value: Two-tailed p-value
            - significant: Boolean (p < 0.05)
            - effect_size: Cohen's d effect size

        Raises:
            ValueError: If configurations not found or insufficient data
        """
        if not self.experiments:
            raise ValueError("No experiments loaded. Call load_experiments() first.")

        if config_a not in self.experiments:
            raise ValueError(f"Configuration '{config_a}' not found")
        if config_b not in self.experiments:
            raise ValueError(f"Configuration '{config_b}' not found")

        if metric_name:
            # Test single metric
            return self._test_metric_significance(config_a, config_b, metric_name)
        else:
            # Test all common metrics
            return self._test_all_metrics_significance(config_a, config_b)

    def _test_metric_significance(
        self,
        config_a: str,
        config_b: str,
        metric_name: str
    ) -> Dict[str, Any]:
        """Test significance for a single metric."""
        # Extract metric values for both configurations
        values_a = self._extract_metric_values(config_a, metric_name)
        values_b = self._extract_metric_values(config_b, metric_name)

        if len(values_a) < 2 or len(values_b) < 2:
            raise ValueError(
                f"Insufficient data for statistical test. "
                f"Need at least 2 samples per config. "
                f"Got {len(values_a)} for '{config_a}' and {len(values_b)} for '{config_b}'"
            )

        # Compute t-test
        t_stat, p_value = self._independent_t_test(values_a, values_b)

        # Compute effect size (Cohen's d)
        effect_size = self._cohens_d(values_a, values_b)

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))

        return {
            'metric_name': metric_name,
            'config_a': config_a,
            'config_b': config_b,
            'config_a_mean': mean_a,
            'config_b_mean': mean_b,
            'config_a_std': float(np.std(values_a)),
            'config_b_std': float(np.std(values_b)),
            'config_a_count': len(values_a),
            'config_b_count': len(values_b),
            'difference': abs(mean_a - mean_b),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'effect_size': float(effect_size),
            'interpretation': self._interpret_effect_size(effect_size)
        }

    def _test_all_metrics_significance(
        self,
        config_a: str,
        config_b: str
    ) -> Dict[str, Any]:
        """Test significance for all common metrics between two configs."""
        # Find all common metrics
        common_metrics = self._find_common_metrics(config_a, config_b)

        if not common_metrics:
            raise ValueError(
                f"No common metrics found between '{config_a}' and '{config_b}'"
            )

        results = {}
        for metric in common_metrics:
            try:
                results[metric] = self._test_metric_significance(
                    config_a, config_b, metric
                )
            except ValueError:
                # Skip metrics with insufficient data
                continue

        return {
            'config_a': config_a,
            'config_b': config_b,
            'metrics_tested': list(results.keys()),
            'results': results
        }

    def _extract_metric_values(self, config_name: str, metric_name: str) -> List[float]:
        """Extract all values for a specific metric from a configuration."""
        values = []
        for exp in self.experiments[config_name]:
            value = None
            if metric_name in exp.scores:
                value = exp.scores[metric_name]
            elif metric_name in exp.audio_metrics:
                value = exp.audio_metrics[metric_name]

            if value is not None:
                values.append(value)

        if not values:
            raise ValueError(
                f"Metric '{metric_name}' not found in configuration '{config_name}'"
            )

        return values

    def _find_common_metrics(self, config_a: str, config_b: str) -> List[str]:
        """Find metrics that exist in both configurations."""
        metrics_a = set()
        metrics_b = set()

        for exp in self.experiments[config_a]:
            metrics_a.update(exp.scores.keys())
            metrics_a.update(exp.audio_metrics.keys())

        for exp in self.experiments[config_b]:
            metrics_b.update(exp.scores.keys())
            metrics_b.update(exp.audio_metrics.keys())

        return sorted(list(metrics_a.intersection(metrics_b)))

    def _independent_t_test(
        self,
        sample_a: List[float],
        sample_b: List[float]
    ) -> Tuple[float, float]:
        """Perform independent samples t-test.

        Args:
            sample_a: First sample values
            sample_b: Second sample values

        Returns:
            Tuple of (t_statistic, p_value)
        """
        n_a = len(sample_a)
        n_b = len(sample_b)

        mean_a = np.mean(sample_a)
        mean_b = np.mean(sample_b)

        var_a = np.var(sample_a, ddof=1)
        var_b = np.var(sample_b, ddof=1)

        # Pooled standard error
        pooled_se = np.sqrt(var_a / n_a + var_b / n_b)

        # T-statistic
        if pooled_se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean_a - mean_b) / pooled_se

        # Degrees of freedom (Welch's approximation)
        if var_a == 0 and var_b == 0:
            df = n_a + n_b - 2
        else:
            numerator = (var_a / n_a + var_b / n_b) ** 2
            denominator = (
                (var_a / n_a) ** 2 / (n_a - 1) +
                (var_b / n_b) ** 2 / (n_b - 1)
            )
            df = numerator / denominator if denominator > 0 else n_a + n_b - 2

        # Two-tailed p-value using t-distribution approximation
        p_value = self._t_distribution_p_value(abs(t_stat), df)

        return t_stat, p_value

    def _t_distribution_p_value(self, t_stat: float, df: float) -> float:
        """Approximate two-tailed p-value for t-distribution.

        Uses approximation since scipy is not available.
        For production, should use scipy.stats.t.sf()
        """
        # Simple approximation using normal distribution for large df
        # For df > 30, t-distribution approximates normal distribution
        if df > 30:
            # Use normal approximation
            z = t_stat
            # Two-tailed p-value for standard normal
            p_value = 2 * (1 - self._normal_cdf(abs(z)))
        else:
            # Rough approximation for small df
            # This is simplified and should be replaced with scipy.stats.t
            z_equiv = t_stat * np.sqrt(df / (df + t_stat ** 2))
            p_value = 2 * (1 - self._normal_cdf(abs(z_equiv)))

        return float(np.clip(p_value, 0, 1))

    def _normal_cdf(self, x: float) -> float:
        """Approximate cumulative distribution function for standard normal.

        Using error function approximation.
        """
        return (1.0 + np.tanh(x * np.sqrt(2 / np.pi))) / 2.0

    def _cohens_d(self, sample_a: List[float], sample_b: List[float]) -> float:
        """Calculate Cohen's d effect size.

        Args:
            sample_a: First sample values
            sample_b: Second sample values

        Returns:
            Cohen's d effect size
        """
        n_a = len(sample_a)
        n_b = len(sample_b)

        mean_a = np.mean(sample_a)
        mean_b = np.mean(sample_b)

        var_a = np.var(sample_a, ddof=1)
        var_b = np.var(sample_b, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_std

    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size.

        Args:
            d: Cohen's d value

        Returns:
            Interpretation string
        """
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def generate_comparison_report(self, output_path: Optional[str] = None) -> str:
        """Generate detailed comparison report.

        Args:
            output_path: Optional path to save report as text file

        Returns:
            Formatted comparison report as string

        Raises:
            ValueError: If no experiments loaded
        """
        if not self.experiments:
            raise ValueError("No experiments loaded. Call load_experiments() first.")

        lines = []
        lines.append("=" * 80)
        lines.append("CONFIGURATION COMPARISON REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Configurations: {len(self.config_names)}")
        for config in self.config_names:
            count = len(self.experiments[config])
            lines.append(f"  - {config}: {count} experiments")
        lines.append("")

        # Find all unique metrics across all configurations
        all_metrics = set()
        for experiments in self.experiments.values():
            for exp in experiments:
                all_metrics.update(exp.scores.keys())
                all_metrics.update(exp.audio_metrics.keys())

        # Compare each metric
        lines.append("METRIC COMPARISONS")
        lines.append("-" * 80)

        for metric in sorted(all_metrics):
            try:
                comparison = self.compare_metrics(metric)
                lines.append(f"\nMetric: {metric}")
                lines.append(f"  Overall Mean: {comparison['overall_mean']:.4f}")
                lines.append(f"  Overall Std:  {comparison['overall_std']:.4f}")
                lines.append(f"  Best Config:  {comparison['best_config']}")
                lines.append(f"  Worst Config: {comparison['worst_config']}")
                lines.append("")

                lines.append("  Per-Configuration Statistics:")
                for config in self.config_names:
                    if config in comparison['config_stats']:
                        stats = comparison['config_stats'][config]
                        lines.append(
                            f"    {config:20s} "
                            f"mean={stats['mean']:7.4f} "
                            f"std={stats['std']:7.4f} "
                            f"min={stats['min']:7.4f} "
                            f"max={stats['max']:7.4f} "
                            f"n={stats['count']}"
                        )
            except ValueError:
                # Skip metrics not present in all configs
                continue

        # Statistical significance tests (pairwise)
        if len(self.config_names) >= 2:
            lines.append("")
            lines.append("STATISTICAL SIGNIFICANCE TESTS")
            lines.append("-" * 80)

            # Test all pairs
            for i, config_a in enumerate(self.config_names):
                for config_b in self.config_names[i+1:]:
                    lines.append(f"\nComparing: {config_a} vs {config_b}")
                    lines.append("-" * 40)

                    try:
                        result = self.statistical_significance(config_a, config_b)

                        for metric, test in result['results'].items():
                            sig_marker = "***" if test['significant'] else "   "
                            lines.append(
                                f"  {metric:25s} "
                                f"{sig_marker} "
                                f"p={test['p_value']:.4f} "
                                f"d={test['effect_size']:+.3f} "
                                f"({test['interpretation']})"
                            )
                    except ValueError as e:
                        lines.append(f"  Error: {e}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("*** p < 0.05 (statistically significant)")
        lines.append("Effect sizes: |d| < 0.2 (negligible), < 0.5 (small), "
                    "< 0.8 (medium), >= 0.8 (large)")
        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(report)

        return report

    def export_comparison_json(self, output_path: str) -> None:
        """Export comparison data to JSON format.

        Args:
            output_path: Path to save JSON file
        """
        if not self.experiments:
            raise ValueError("No experiments loaded. Call load_experiments() first.")

        # Find all metrics
        all_metrics = set()
        for experiments in self.experiments.values():
            for exp in experiments:
                all_metrics.update(exp.scores.keys())
                all_metrics.update(exp.audio_metrics.keys())

        # Build comparison data
        comparison_data = {
            'configurations': self.config_names,
            'experiment_counts': {
                config: len(exps) for config, exps in self.experiments.items()
            },
            'metrics': {}
        }

        for metric in sorted(all_metrics):
            try:
                comparison_data['metrics'][metric] = self.compare_metrics(metric)
            except ValueError:
                continue

        # Save to JSON
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
