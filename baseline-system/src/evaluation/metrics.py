"""
Shared evaluation metrics and utilities for retrieval tasks.

Provides common functions used across evaluation scripts to avoid code duplication.
"""

from typing import List, Dict, Tuple
import numpy as np


def calculate_hit_rates(
    predicted_labels: List[str],
    ground_truth: List[str],
    k_values: List[int]
) -> Dict[int, bool]:
    """
    Calculate hit rates for different k values.

    Args:
        predicted_labels: List of predicted labels (ordered by score)
        ground_truth: List of ground truth labels
        k_values: List of k values to evaluate (e.g., [1, 5, 10])

    Returns:
        Dictionary mapping k -> hit (boolean)
    """
    hits = {}
    for k in k_values:
        top_k = predicted_labels[:k]
        hits[k] = any(gt in top_k for gt in ground_truth)
    return hits


def calculate_reciprocal_rank(
    predicted_labels: List[str],
    ground_truth: List[str]
) -> float:
    """
    Calculate reciprocal rank for a single prediction.

    Args:
        predicted_labels: List of predicted labels (ordered by score)
        ground_truth: List of ground truth labels

    Returns:
        Reciprocal rank (1/rank if found, 0 otherwise)
    """
    for rank, label in enumerate(predicted_labels, start=1):
        if label in ground_truth:
            return 1.0 / rank
    return 0.0


def calculate_mean_reciprocal_rank(reciprocal_ranks: List[float]) -> float:
    """
    Calculate mean reciprocal rank from a list of reciprocal ranks.

    Args:
        reciprocal_ranks: List of reciprocal ranks

    Returns:
        Mean reciprocal rank
    """
    if not reciprocal_ranks:
        return 0.0
    return float(np.mean(reciprocal_ranks))


def aggregate_hit_rates(
    sample_hits: List[Dict[int, bool]],
    total_samples: int
) -> Dict[int, float]:
    """
    Aggregate hit rates across all samples.

    Args:
        sample_hits: List of hit dictionaries from calculate_hit_rates
        total_samples: Total number of samples

    Returns:
        Dictionary mapping k -> hit rate (0-1)
    """
    if total_samples == 0:
        return {}

    # Get all k values from first sample
    k_values = list(sample_hits[0].keys()) if sample_hits else []

    # Count hits for each k
    hit_counts = {k: 0 for k in k_values}
    for hits in sample_hits:
        for k, hit in hits.items():
            if hit:
                hit_counts[k] += 1

    # Convert to rates
    return {k: count / total_samples for k, count in hit_counts.items()}


def format_metrics_summary(
    hit_rates: Dict[int, float],
    mrr: float,
    mean_score_top1: float = None
) -> str:
    """
    Format metrics into a readable summary string.

    Args:
        hit_rates: Dictionary of k -> hit rate
        mrr: Mean reciprocal rank
        mean_score_top1: Optional mean score of top-1 predictions

    Returns:
        Formatted string
    """
    lines = ["Metrics Summary:", "=" * 40]

    for k in sorted(hit_rates.keys()):
        lines.append(f"  Hit@{k}: {hit_rates[k]:.3f}")

    lines.append(f"  MRR: {mrr:.3f}")

    if mean_score_top1 is not None:
        lines.append(f"  Mean Top-1 Score: {mean_score_top1:.3f}")

    return "\n".join(lines)
