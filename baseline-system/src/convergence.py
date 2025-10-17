"""Convergence detection for refinement loop score plateaus.

This module provides convergence detection logic to identify when score
improvements have plateaued during iterative refinement, enabling early
stopping and efficient iteration management.
"""

from typing import List


class ConvergenceDetector:
    """Detects convergence in score sequences during refinement loops.

    Uses a sliding window approach to detect when score improvements have
    plateaued below a minimum threshold, indicating convergence.

    Attributes:
        window_size: Number of recent scores to analyze for convergence
        min_improvement: Minimum score range within window to continue refinement
    """

    def __init__(
        self,
        window_size: int = 3,
        min_improvement: float = 2.0
    ):
        """Initialize convergence detector.

        Args:
            window_size: Number of recent scores to check for plateau (must be >= 2)
            min_improvement: Minimum score range to avoid convergence detection

        Raises:
            ValueError: If window_size < 2 or min_improvement < 0
        """
        if window_size < 2:
            raise ValueError(f"window_size must be at least 2, got {window_size}")
        if min_improvement < 0:
            raise ValueError(f"min_improvement must be non-negative, got {min_improvement}")

        self.window_size = window_size
        self.min_improvement = min_improvement

    def has_converged(self, scores: List[float]) -> bool:
        """Check if score sequence has converged (plateaued).

        Analyzes the most recent scores within the window to determine if
        the score range (max - min) is below the minimum improvement threshold.

        Args:
            scores: List of scores from refinement iterations (chronological order)

        Returns:
            True if scores have converged (plateaued), False otherwise

        Raises:
            ValueError: If any score is negative
        """
        # Validate scores
        if any(score < 0 for score in scores):
            raise ValueError("All scores must be non-negative")

        # Need enough scores to analyze
        if len(scores) < self.window_size:
            return False

        # Get recent scores within window
        recent_scores = scores[-self.window_size:]

        # Calculate score range (spread) in window
        score_range = max(recent_scores) - min(recent_scores)

        # Converged if range is below minimum improvement threshold
        return score_range < self.min_improvement

    def predict_convergence(
        self,
        scores: List[float],
        lookahead: int = 3
    ) -> int:
        """Estimate iterations until convergence based on score trend.

        Uses moving average of score improvements to estimate how many
        additional iterations are needed to reach convergence or max score.

        Args:
            scores: List of scores from refinement iterations (chronological order)
            lookahead: Maximum number of iterations to predict

        Returns:
            Estimated iterations until convergence:
            - -1 if insufficient data (< 2 scores)
            - 0 if already converged or negative trend
            - 1 to lookahead for positive trend predictions

        Raises:
            ValueError: If any score is negative or lookahead < 1
        """
        # Validate inputs
        if any(score < 0 for score in scores):
            raise ValueError("All scores must be non-negative")
        if lookahead < 1:
            raise ValueError(f"lookahead must be at least 1, got {lookahead}")

        # Need at least 2 scores to calculate trend
        if len(scores) < 2:
            return -1

        # Calculate score improvements between consecutive iterations
        improvements = [
            scores[i] - scores[i-1]
            for i in range(1, len(scores))
        ]

        # Calculate average improvement rate
        avg_improvement = sum(improvements) / len(improvements)

        # If no improvement or negative trend, convergence is immediate
        if avg_improvement <= 0:
            return 0

        # Estimate iterations based on remaining gap to perfect score (100)
        remaining_gap = 100 - scores[-1]

        # Calculate estimated iterations to close gap
        estimated_iterations = int(remaining_gap / avg_improvement)

        # Cap at lookahead limit
        return min(estimated_iterations, lookahead)
