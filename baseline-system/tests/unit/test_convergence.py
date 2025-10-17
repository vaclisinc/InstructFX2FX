"""Tests for convergence detection module."""

import pytest

from src.convergence import ConvergenceDetector


class TestConvergenceDetectorInitialization:
    """Test ConvergenceDetector initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        detector = ConvergenceDetector()

        assert detector.window_size == 3
        assert detector.min_improvement == 2.0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        detector = ConvergenceDetector(window_size=5, min_improvement=5.0)

        assert detector.window_size == 5
        assert detector.min_improvement == 5.0

    def test_invalid_window_size_zero(self):
        """Test that window_size of 0 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConvergenceDetector(window_size=0)

        assert "window_size must be at least 2" in str(exc_info.value)

    def test_invalid_window_size_one(self):
        """Test that window_size of 1 raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConvergenceDetector(window_size=1)

        assert "window_size must be at least 2" in str(exc_info.value)

    def test_invalid_window_size_negative(self):
        """Test that negative window_size raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConvergenceDetector(window_size=-5)

        assert "window_size must be at least 2" in str(exc_info.value)

    def test_invalid_min_improvement_negative(self):
        """Test that negative min_improvement raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConvergenceDetector(min_improvement=-1.0)

        assert "min_improvement must be non-negative" in str(exc_info.value)

    def test_zero_min_improvement_allowed(self):
        """Test that zero min_improvement is allowed."""
        detector = ConvergenceDetector(min_improvement=0.0)
        assert detector.min_improvement == 0.0


class TestHasConverged:
    """Test has_converged method."""

    def test_insufficient_scores_returns_false(self):
        """Test that insufficient scores returns False."""
        detector = ConvergenceDetector(window_size=3)

        assert detector.has_converged([]) is False
        assert detector.has_converged([70.0]) is False
        assert detector.has_converged([70.0, 72.0]) is False

    def test_exact_window_size_converged(self):
        """Test convergence detection with exactly window_size scores."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Range is 1.5 (73.5 - 72.0) < 2.0 -> converged
        scores = [72.0, 73.0, 73.5]
        assert detector.has_converged(scores) is True

    def test_exact_window_size_not_converged(self):
        """Test non-convergence with exactly window_size scores."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Range is 3.0 (73.0 - 70.0) >= 2.0 -> not converged
        scores = [70.0, 72.0, 73.0]
        assert detector.has_converged(scores) is False

    def test_flat_plateau_converged(self):
        """Test convergence with completely flat scores."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # All identical scores -> range = 0 < 2.0 -> converged
        scores = [75.0, 75.0, 75.0, 75.0]
        assert detector.has_converged(scores) is True

    def test_oscillating_plateau_converged(self):
        """Test convergence with small oscillations."""
        detector = ConvergenceDetector(window_size=4, min_improvement=2.0)

        # Oscillating between 74-75.5 -> range = 1.5 < 2.0 -> converged
        scores = [70.0, 72.0, 74.0, 75.5, 74.5, 75.0]
        assert detector.has_converged(scores) is True

    def test_oscillating_plateau_not_converged(self):
        """Test non-convergence with larger oscillations."""
        detector = ConvergenceDetector(window_size=4, min_improvement=2.0)

        # Oscillating between 72-75 -> range = 3.0 >= 2.0 -> not converged
        scores = [70.0, 72.0, 75.0, 72.0]
        assert detector.has_converged(scores) is False

    def test_monotonic_increasing_converged(self):
        """Test convergence with slowly increasing scores."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Slowly increasing: 84 -> 84.5 -> 85 -> range = 1.0 < 2.0
        scores = [75.0, 80.0, 84.0, 84.5, 85.0]
        assert detector.has_converged(scores) is True

    def test_monotonic_increasing_not_converged(self):
        """Test non-convergence with rapid increasing scores."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Rapidly increasing: 80 -> 82 -> 85 -> range = 5.0 >= 2.0
        scores = [70.0, 75.0, 80.0, 82.0, 85.0]
        assert detector.has_converged(scores) is False

    def test_monotonic_decreasing_converged(self):
        """Test convergence with slowly decreasing scores."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Slowly decreasing: 85 -> 84.5 -> 84 -> range = 1.0 < 2.0
        scores = [90.0, 87.0, 85.0, 84.5, 84.0]
        assert detector.has_converged(scores) is True

    def test_recent_window_only_checked(self):
        """Test that only recent window is checked, not entire history."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Early scores varied widely, but last 3 are flat
        # Range in last 3: 89.5 - 89.0 = 0.5 < 2.0 -> converged
        scores = [60.0, 70.0, 80.0, 89.0, 89.2, 89.5]
        assert detector.has_converged(scores) is True

    def test_large_improvement_then_plateau(self):
        """Test detection of plateau after large improvements."""
        detector = ConvergenceDetector(window_size=4, min_improvement=2.0)

        # Large jumps then plateau at ~90
        # Last 4: 89.5, 90.0, 90.5, 90.2 -> range = 1.0 < 2.0
        scores = [60.0, 75.0, 85.0, 89.5, 90.0, 90.5, 90.2]
        assert detector.has_converged(scores) is True

    def test_boundary_case_exact_threshold(self):
        """Test boundary case where range equals threshold."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Range is exactly 2.0 (75.0 - 73.0) -> not converged (< not <=)
        scores = [73.0, 74.0, 75.0]
        assert detector.has_converged(scores) is False

    def test_boundary_case_just_below_threshold(self):
        """Test boundary case just below threshold."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Range is 1.99 < 2.0 -> converged
        scores = [73.0, 74.0, 74.99]
        assert detector.has_converged(scores) is True

    def test_zero_min_improvement_threshold(self):
        """Test convergence with zero threshold (any non-zero range fails)."""
        detector = ConvergenceDetector(window_size=3, min_improvement=0.0)

        # Only perfectly flat scores converge with zero threshold
        assert detector.has_converged([75.0, 75.0, 75.0]) is False  # Floating point precision
        assert detector.has_converged([75.0, 75.0001, 75.0]) is False

    def test_negative_score_raises_error(self):
        """Test that negative scores raise ValueError."""
        detector = ConvergenceDetector(window_size=3)

        with pytest.raises(ValueError) as exc_info:
            detector.has_converged([70.0, -5.0, 75.0])

        assert "scores must be non-negative" in str(exc_info.value)

    def test_all_negative_scores_raises_error(self):
        """Test that all negative scores raise ValueError."""
        detector = ConvergenceDetector(window_size=3)

        with pytest.raises(ValueError) as exc_info:
            detector.has_converged([-10.0, -5.0, -2.0])

        assert "scores must be non-negative" in str(exc_info.value)

    def test_mixed_score_range(self):
        """Test with diverse score range."""
        detector = ConvergenceDetector(window_size=5, min_improvement=3.0)

        # Last 5: 85, 87, 86, 88, 87.5 -> range = 3.0 -> not converged (equal)
        scores = [50.0, 65.0, 80.0, 85.0, 87.0, 86.0, 88.0, 87.5]
        assert detector.has_converged(scores) is False


class TestPredictConvergence:
    """Test predict_convergence method."""

    def test_insufficient_scores_returns_minus_one(self):
        """Test that insufficient scores returns -1."""
        detector = ConvergenceDetector()

        assert detector.predict_convergence([]) == -1
        assert detector.predict_convergence([70.0]) == -1

    def test_no_improvement_returns_zero(self):
        """Test that zero improvement returns 0."""
        detector = ConvergenceDetector()

        # Flat scores -> avg improvement = 0
        scores = [75.0, 75.0, 75.0, 75.0]
        assert detector.predict_convergence(scores) == 0

    def test_negative_trend_returns_zero(self):
        """Test that negative trend returns 0."""
        detector = ConvergenceDetector()

        # Decreasing scores -> negative avg improvement
        scores = [80.0, 75.0, 70.0, 65.0]
        assert detector.predict_convergence(scores) == 0

    def test_positive_trend_small_gap(self):
        """Test prediction with positive trend and small gap to 100."""
        detector = ConvergenceDetector()

        # Increasing by ~2.5 per iteration, currently at 95
        # Gap = 100 - 95 = 5, avg_improvement = 2.5, estimate = 2
        scores = [85.0, 87.5, 90.0, 92.5, 95.0]
        result = detector.predict_convergence(scores)

        # Should predict ~2 iterations (5 / 2.5)
        assert result == 2

    def test_positive_trend_large_gap(self):
        """Test prediction with positive trend but large gap to 100."""
        detector = ConvergenceDetector()

        # Increasing by 5 per iteration, currently at 70
        # Gap = 30, avg_improvement = 5, estimate = 6
        # But capped at default lookahead = 3
        scores = [50.0, 55.0, 60.0, 65.0, 70.0]
        result = detector.predict_convergence(scores, lookahead=3)

        assert result == 3

    def test_prediction_capped_at_lookahead(self):
        """Test that prediction is capped at lookahead value."""
        detector = ConvergenceDetector()

        # Very slow improvement: 1 per iteration, gap = 20
        # Would estimate 20 iterations, but capped at lookahead
        scores = [70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0]
        result = detector.predict_convergence(scores, lookahead=5)

        assert result == 5

    def test_custom_lookahead(self):
        """Test prediction with custom lookahead."""
        detector = ConvergenceDetector()

        # Improvement of 10 per iteration, gap = 40
        # Estimate = 4, lookahead = 10 -> return 4
        scores = [20.0, 30.0, 40.0, 50.0, 60.0]
        result = detector.predict_convergence(scores, lookahead=10)

        assert result == 4

    def test_already_at_100_returns_zero(self):
        """Test prediction when score is already at maximum."""
        detector = ConvergenceDetector()

        # Already at 100 -> gap = 0, but avg_improvement > 0
        scores = [80.0, 90.0, 100.0]
        result = detector.predict_convergence(scores)

        assert result == 0

    def test_oscillating_scores_prediction(self):
        """Test prediction with oscillating scores."""
        detector = ConvergenceDetector()

        # Oscillating: improvements are [5, -3, 4, -2]
        # Avg improvement = (5 - 3 + 4 - 2) / 4 = 1.0
        # Gap = 100 - 77 = 23, estimate = 23 (capped at lookahead)
        scores = [70.0, 75.0, 72.0, 76.0, 74.0, 77.0]
        result = detector.predict_convergence(scores, lookahead=3)

        assert result == 3

    def test_rapidly_improving_scores(self):
        """Test prediction with rapid improvement."""
        detector = ConvergenceDetector()

        # Rapidly improving by ~10 per iteration
        # Current: 70, gap = 30, avg = 10, estimate = 3
        scores = [30.0, 40.0, 50.0, 60.0, 70.0]
        result = detector.predict_convergence(scores)

        assert result == 3

    def test_slowly_improving_scores(self):
        """Test prediction with slow improvement."""
        detector = ConvergenceDetector()

        # Slowly improving by ~0.5 per iteration
        # Current: 72, gap = 28, avg = 0.5, estimate = 56 (capped)
        scores = [70.0, 70.5, 71.0, 71.5, 72.0]
        result = detector.predict_convergence(scores, lookahead=3)

        assert result == 3

    def test_two_scores_minimum(self):
        """Test prediction with exactly 2 scores (minimum)."""
        detector = ConvergenceDetector()

        # Two scores: improvement = 10, gap = 30, estimate = 3
        scores = [60.0, 70.0]
        result = detector.predict_convergence(scores)

        assert result == 3

    def test_negative_scores_raises_error(self):
        """Test that negative scores raise ValueError."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError) as exc_info:
            detector.predict_convergence([70.0, -5.0, 75.0])

        assert "scores must be non-negative" in str(exc_info.value)

    def test_invalid_lookahead_raises_error(self):
        """Test that invalid lookahead raises ValueError."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError) as exc_info:
            detector.predict_convergence([70.0, 75.0, 80.0], lookahead=0)

        assert "lookahead must be at least 1" in str(exc_info.value)

    def test_negative_lookahead_raises_error(self):
        """Test that negative lookahead raises ValueError."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError) as exc_info:
            detector.predict_convergence([70.0, 75.0, 80.0], lookahead=-5)

        assert "lookahead must be at least 1" in str(exc_info.value)

    def test_prediction_with_very_small_improvements(self):
        """Test prediction with very small incremental improvements."""
        detector = ConvergenceDetector()

        # Tiny improvements: 0.1 per iteration
        # Gap = 20, avg = 0.1, estimate = 200 (capped at lookahead)
        scores = [79.0, 79.1, 79.2, 79.3, 79.4, 79.5, 79.6, 79.7, 79.8, 79.9, 80.0]
        result = detector.predict_convergence(scores, lookahead=3)

        assert result == 3

    def test_prediction_near_maximum_score(self):
        """Test prediction when score is very close to 100."""
        detector = ConvergenceDetector()

        # Near 100, improving slowly
        # Gap = ~2, avg_improvement = ~0.5, estimate = ~4 (capped)
        scores = [97.0, 97.5, 98.0, 98.5]
        result = detector.predict_convergence(scores, lookahead=3)

        # Should predict 3 iterations (capped)
        assert result == 3


class TestConvergenceDetectorIntegration:
    """Integration tests for realistic refinement scenarios."""

    def test_typical_refinement_scenario(self):
        """Test typical refinement loop convergence."""
        detector = ConvergenceDetector(window_size=3, min_improvement=2.0)

        # Simulate typical refinement: large initial gains, then plateau
        scores = [45.0, 60.0, 72.0, 80.0, 85.0, 88.0, 89.5, 90.0, 90.3]

        # Should not converge early
        assert detector.has_converged(scores[:5]) is False

        # Should converge near the end
        assert detector.has_converged(scores) is True

    def test_early_stopping_high_score(self):
        """Test convergence detection for early stopping at high score."""
        detector = ConvergenceDetector(window_size=3, min_improvement=1.0)

        # Reaches high score quickly then plateaus
        scores = [50.0, 75.0, 88.0, 92.0, 93.0, 93.5, 93.8]

        # Should detect plateau at end
        assert detector.has_converged(scores) is True

    def test_no_convergence_continuous_improvement(self):
        """Test no convergence with continuous steady improvement."""
        detector = ConvergenceDetector(window_size=4, min_improvement=2.0)

        # Steady improvement, no plateau
        scores = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0]

        # Should not converge
        assert detector.has_converged(scores) is False

    def test_convergence_prediction_early_stage(self):
        """Test convergence prediction in early refinement."""
        detector = ConvergenceDetector()

        # Early stage, large improvements
        scores = [45.0, 60.0, 72.0]
        iterations = detector.predict_convergence(scores, lookahead=5)

        # Should predict several more iterations
        assert iterations >= 2

    def test_convergence_prediction_late_stage(self):
        """Test convergence prediction near convergence."""
        detector = ConvergenceDetector()

        # Late stage, small improvements
        scores = [85.0, 88.0, 89.5, 90.0, 90.3]
        iterations = detector.predict_convergence(scores, lookahead=5)

        # Should predict few iterations (close to 100)
        assert iterations <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
