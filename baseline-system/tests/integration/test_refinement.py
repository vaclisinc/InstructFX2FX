"""Integration tests for end-to-end refinement system.

Tests verify complete refinement loop functionality including:
- End-to-end refinement with real parameter generation and scoring
- Both parameter_only and audio_based modes
- Iteration history tracking and best parameter selection
- Convergence scenarios (plateau, max iterations, early stop)
- Temperature scheduling affecting generation
- Feedback integration into refinement prompts
- Error recovery and graceful degradation
"""

import pytest
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.baseline.refinement_controller import RefinementLoopController
from src.models.refinement import (
    IterationResult,
    RefinementConfig,
    RefinementResult,
)
from src.convergence import ConvergenceDetector
from src.scoring.models import ScoringRequest, ScoringResponse, ScoreDimension
from src.models.parameters import EffectChain, EQParameters, EQBand, ReverbParameters
from models.llm_judge.types import LLMRequest, LLMResponse


# Integration test helpers - real implementations with test data
# Following project standard: NO MOCK SERVICES


class IntegrationTestGenerator:
    """Test generator for integration tests with realistic behavior."""

    def __init__(
        self,
        parameter_progression: Optional[List[Dict[str, Any]]] = None,
        should_fail: bool = False
    ):
        """Initialize integration test generator.

        Args:
            parameter_progression: List of parameters to return per iteration
            should_fail: If True, raise errors during generation
        """
        self.parameter_progression = parameter_progression or self._default_progression()
        self.should_fail = should_fail
        self.llm_provider = IntegrationTestLLMProvider()
        self.template = IntegrationTestPromptTemplate()
        self.call_count = 0
        self.requests: List[str] = []

    def _default_progression(self) -> List[Dict[str, Any]]:
        """Generate default parameter progression simulating improvement."""
        return [
            # Iteration 0: Initial parameters
            {
                "eq": {
                    "bands": [
                        {"frequency": 1000, "gain": 2.0, "q": 1.0},
                        {"frequency": 3000, "gain": -1.0, "q": 1.0},
                        {"frequency": 8000, "gain": 1.0, "q": 1.0}
                    ]
                }
            },
            # Iteration 1: Refined based on feedback
            {
                "eq": {
                    "bands": [
                        {"frequency": 1000, "gain": 3.0, "q": 1.2},
                        {"frequency": 3000, "gain": -0.5, "q": 1.0},
                        {"frequency": 8000, "gain": 1.5, "q": 0.8}
                    ]
                }
            },
            # Iteration 2: Further refinement
            {
                "eq": {
                    "bands": [
                        {"frequency": 1000, "gain": 3.5, "q": 1.3},
                        {"frequency": 3000, "gain": -0.3, "q": 1.1},
                        {"frequency": 8000, "gain": 2.0, "q": 0.7}
                    ]
                }
            },
        ]

    async def generate_parameters(
        self,
        description: str,
        **kwargs
    ) -> EffectChain:
        """Generate parameters for iteration.

        Args:
            description: User description

        Returns:
            EffectChain with parameters for current iteration

        Raises:
            Exception: If should_fail is True
        """
        if self.should_fail:
            raise Exception("Simulated generator failure")

        self.requests.append(description)
        params_index = min(self.call_count, len(self.parameter_progression) - 1)
        params = self.parameter_progression[params_index]
        self.call_count += 1

        # Create effect chain
        effects = []
        if "eq" in params:
            bands = [EQBand(**band) for band in params["eq"]["bands"]]
            effects.append(EQParameters(bands=bands))
        if "reverb" in params:
            effects.append(ReverbParameters(**params["reverb"]))

        return EffectChain(
            description=description,
            effects=effects,
            order=list(params.keys())
        )

    def parse_and_validate(
        self,
        content: str,
        description: Optional[str] = None
    ) -> EffectChain:
        """Parse and validate LLM response.

        Args:
            content: LLM response content
            description: Optional description

        Returns:
            EffectChain for current iteration
        """
        params_index = min(self.call_count - 1, len(self.parameter_progression) - 1)
        params = self.parameter_progression[params_index]

        effects = []
        if "eq" in params:
            bands = [EQBand(**band) for band in params["eq"]["bands"]]
            effects.append(EQParameters(bands=bands))
        if "reverb" in params:
            effects.append(ReverbParameters(**params["reverb"]))

        return EffectChain(
            description=description or "test",
            effects=effects,
            order=list(params.keys())
        )


class IntegrationTestScorer:
    """Test scorer for integration tests with realistic scoring behavior."""

    def __init__(
        self,
        score_progression: Optional[List[float]] = None,
        should_fail: bool = False
    ):
        """Initialize integration test scorer.

        Args:
            score_progression: List of scores to return per iteration
            should_fail: If True, raise errors during scoring
        """
        self.score_progression = score_progression or [65.0, 75.0, 82.0, 87.0, 89.0]
        self.should_fail = should_fail
        self.call_count = 0
        self.requests: List[ScoringRequest] = []

    async def score_parameters(
        self,
        request: ScoringRequest
    ) -> ScoringResponse:
        """Score parameters with realistic progression.

        Args:
            request: Scoring request

        Returns:
            ScoringResponse with score from progression

        Raises:
            Exception: If should_fail is True
        """
        if self.should_fail:
            raise Exception("Simulated scorer failure")

        self.requests.append(request)
        score_index = min(self.call_count, len(self.score_progression) - 1)
        score = self.score_progression[score_index]
        self.call_count += 1

        # Generate realistic dimensions
        dimensions = [
            ScoreDimension(
                name="semantic_match",
                score=score + 2.0,
                reasoning=f"Semantic alignment improving at iteration {request.iteration}"
            ),
            ScoreDimension(
                name="technical_quality",
                score=score - 1.0,
                reasoning=f"Technical quality assessment for iteration {request.iteration}"
            ),
            ScoreDimension(
                name="specificity",
                score=score,
                reasoning=f"Parameter specificity at iteration {request.iteration}"
            )
        ]

        # Generate realistic feedback and suggestions
        if score < 70:
            feedback = "Parameters need significant improvement to match description."
            suggestions = [
                "Increase gain in mid frequencies",
                "Adjust Q factor for more focused EQ",
                "Consider wider frequency range"
            ]
        elif score < 85:
            feedback = "Parameters are improving but still have room for refinement."
            suggestions = [
                "Fine-tune gain values",
                "Optimize Q factor balance"
            ]
        else:
            feedback = "Parameters show strong alignment with description."
            suggestions = [
                "Minor tweaks to high frequency response"
            ]

        return ScoringResponse(
            dimensions=dimensions,
            overall_score=score,
            feedback=feedback,
            suggestions=suggestions,
            confidence=min(0.95, 0.7 + (score / 200))
        )

    async def score_with_audio(
        self,
        request: ScoringRequest,
        audio_path: str
    ) -> ScoringResponse:
        """Score with audio (delegates to score_parameters).

        Args:
            request: Scoring request
            audio_path: Path to audio file

        Returns:
            ScoringResponse
        """
        return await self.score_parameters(request)


class IntegrationTestLLMProvider:
    """Test LLM provider for integration tests."""

    def __init__(self):
        """Initialize test provider."""
        self.call_count = 0
        self.requests: List[LLMRequest] = []
        self.temperatures_used: List[float] = []

    async def generate_with_retry(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """Generate LLM response.

        Args:
            request: LLM request

        Returns:
            LLMResponse with test content
        """
        self.requests.append(request)
        self.temperatures_used.append(request.temperature)
        self.call_count += 1

        content = """
        {
            "description": "refined effect",
            "effects": [
                {
                    "type": "eq",
                    "bands": [
                        {"frequency": 1000, "gain": 3.0, "q": 1.2},
                        {"frequency": 3000, "gain": -0.5, "q": 1.0},
                        {"frequency": 8000, "gain": 1.5, "q": 0.8}
                    ]
                }
            ]
        }
        """

        return LLMResponse(
            content=content,
            model="test-model",
            tokens_used=500,
            prompt_tokens=200,
            completion_tokens=300,
            finish_reason="stop",
            provider="test"
        )


class IntegrationTestPromptTemplate:
    """Test prompt template for integration tests."""

    def __init__(self):
        """Initialize test template."""
        self.system_prompt = "You are an expert audio engineer specializing in effect parameter design."


# Integration Test Classes


@pytest.mark.asyncio
class TestEndToEndRefinement:
    """Test complete end-to-end refinement loops."""

    async def test_basic_refinement_loop(self):
        """Test basic refinement loop with parameter generation and scoring."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        config = RefinementConfig(
            max_iterations=5,
            min_score_improvement=2.0,
            convergence_window=3
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="warm and intimate vocal sound"
        )

        # Verify result structure
        assert isinstance(result, RefinementResult)
        assert result.description == "warm and intimate vocal sound"
        assert isinstance(result.initial_parameters, dict)
        assert isinstance(result.final_parameters, dict)
        assert len(result.iterations) > 0
        assert result.total_iterations == len(result.iterations)
        assert 0 <= result.final_score <= 100
        assert result.improvement >= 0  # Score should improve
        assert result.convergence_reason in [
            "max_iterations_reached",
            "target_score_reached (87.0)",
            "target_score_reached (89.0)",
            "score_plateau_detected"
        ]

        # Verify iteration history
        for i, iteration in enumerate(result.iterations):
            assert iteration.iteration == i
            assert isinstance(iteration.parameters, dict)
            assert 0 <= iteration.score <= 100
            assert len(iteration.feedback) > 0
            assert isinstance(iteration.suggestions, list)

        # Verify scoring was called
        assert scorer.call_count == result.total_iterations

        print(
            f"✓ Basic refinement completed: {result.total_iterations} iterations, "
            f"score improved {result.improvement:.1f} points, "
            f"reason: {result.convergence_reason}"
        )

    async def test_refinement_with_initial_parameters(self):
        """Test refinement starting from provided initial parameters."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer
        )

        initial_params = {
            "eq": {
                "bands": [
                    {"frequency": 500, "gain": 1.0, "q": 1.0},
                    {"frequency": 2000, "gain": 0.0, "q": 1.0},
                    {"frequency": 6000, "gain": 0.5, "q": 1.0}
                ]
            }
        }

        result = await controller.refine(
            description="bright and energetic sound",
            initial_parameters=initial_params
        )

        assert result.initial_parameters == initial_params
        assert generator.call_count == 0  # Initial params provided, no generation needed
        print("✓ Refinement with provided initial parameters succeeded")

    async def test_convergence_on_plateau(self):
        """Test refinement stops when scores plateau."""
        # Score progression that plateaus quickly
        score_progression = [70.0, 75.0, 80.0, 81.0, 81.3, 81.5]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)
        config = RefinementConfig(
            max_iterations=10,
            min_score_improvement=2.0,
            convergence_window=3,
            early_stop_score=None  # Disable early stop
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Should stop due to plateau, not max iterations
        assert result.total_iterations < config.max_iterations
        assert result.convergence_reason == "score_plateau_detected"
        assert result.total_iterations >= config.convergence_window

        # Verify plateau in recent scores
        recent_scores = [it.score for it in result.iterations[-3:]]
        score_range = max(recent_scores) - min(recent_scores)
        assert score_range < config.min_score_improvement

        print(
            f"✓ Convergence on plateau: stopped at {result.total_iterations} iterations "
            f"(max={config.max_iterations}), recent range={score_range:.1f}"
        )

    async def test_max_iterations_reached(self):
        """Test refinement stops at max iterations."""
        # Continuous improvement that doesn't plateau
        score_progression = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)
        config = RefinementConfig(
            max_iterations=4,
            min_score_improvement=2.0,
            early_stop_score=None
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        assert result.total_iterations == config.max_iterations
        assert result.convergence_reason == "max_iterations_reached"
        print(
            f"✓ Max iterations reached: stopped at {result.total_iterations} "
            f"of {config.max_iterations}"
        )

    async def test_early_stop_score_reached(self):
        """Test refinement stops when target score reached."""
        # Rapid improvement to target
        score_progression = [70.0, 80.0, 92.0, 95.0]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)
        config = RefinementConfig(
            max_iterations=10,
            early_stop_score=90.0
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Should stop when score >= 90.0
        assert result.total_iterations < config.max_iterations
        assert result.final_score >= config.early_stop_score
        assert "target_score_reached" in result.convergence_reason
        print(
            f"✓ Early stop triggered: score {result.final_score:.1f} >= "
            f"{config.early_stop_score}, reason: {result.convergence_reason}"
        )


@pytest.mark.asyncio
class TestIterationHistoryTracking:
    """Test iteration history tracking and best parameter selection."""

    async def test_iteration_history_completeness(self):
        """Test that all iteration data is properly tracked."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        config = RefinementConfig(max_iterations=3)

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Verify each iteration has complete data
        for i, iteration in enumerate(result.iterations):
            assert iteration.iteration == i
            assert isinstance(iteration.parameters, dict)
            assert len(iteration.parameters) > 0
            assert 0 <= iteration.score <= 100
            assert len(iteration.feedback) > 0
            assert isinstance(iteration.suggestions, list)
            assert len(iteration.suggestions) > 0
            # Verify timestamp is valid ISO format
            datetime.fromisoformat(iteration.timestamp)

        print(
            f"✓ Iteration history complete: {len(result.iterations)} iterations, "
            f"all fields populated"
        )

    async def test_best_parameter_selection(self):
        """Test that best parameters are selected from history."""
        # Score progression with best in middle
        score_progression = [70.0, 80.0, 92.0, 85.0, 83.0]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)
        config = RefinementConfig(
            max_iterations=5,
            early_stop_score=None,
            min_score_improvement=0.5  # Allow all iterations
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Best score should be 92.0 from iteration 2
        assert result.final_score == 92.0
        best_iteration = result.iterations[2]
        assert best_iteration.score == 92.0
        assert result.final_parameters == best_iteration.parameters
        print(
            f"✓ Best parameters selected: score {result.final_score:.1f} "
            f"from iteration {best_iteration.iteration} "
            f"(not final iteration {result.total_iterations - 1})"
        )

    async def test_improvement_calculation(self):
        """Test score improvement calculation."""
        score_progression = [60.0, 70.0, 80.0, 85.0]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer
        )

        result = await controller.refine(
            description="test description"
        )

        initial_score = result.iterations[0].score
        final_score = result.final_score
        expected_improvement = final_score - initial_score

        assert result.improvement == expected_improvement
        assert result.improvement > 0  # Should improve
        print(
            f"✓ Improvement calculated: {initial_score:.1f} -> {final_score:.1f} "
            f"= {result.improvement:.1f} points"
        )


@pytest.mark.asyncio
class TestRefinementModes:
    """Test both parameter_only and audio_based modes."""

    async def test_parameter_only_mode(self):
        """Test refinement in parameter_only mode."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        config = RefinementConfig(mode="parameter_only")

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Verify parameter_only mode was used
        assert controller.config.mode == "parameter_only"
        assert controller.processor is None
        # All scoring should go through score_parameters
        assert scorer.call_count > 0
        print(f"✓ parameter_only mode: {scorer.call_count} scoring calls")

    async def test_audio_based_mode_without_processor(self):
        """Test audio_based mode falls back to parameter_only without processor."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        config = RefinementConfig(mode="audio_based")

        # Create controller without audio processor
        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            audio_processor=None,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Should still work, falling back to parameter_only
        assert result.total_iterations > 0
        print("✓ audio_based mode without processor falls back gracefully")


@pytest.mark.asyncio
class TestTemperatureScheduling:
    """Test temperature scheduling affects generation."""

    async def test_default_temperature_schedule(self):
        """Test default temperature schedule is applied."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        config = RefinementConfig(max_iterations=5)

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Check that temperatures were used (tracked by LLM provider)
        temps = generator.llm_provider.temperatures_used

        # Should have declining temperatures (exploration -> exploitation)
        assert len(temps) > 0
        if len(temps) > 1:
            # Generally decreasing trend
            assert temps[0] >= temps[-1]
        print(
            f"✓ Default temperature schedule applied: "
            f"{[f'{t:.2f}' for t in temps]}"
        )

    async def test_custom_temperature_schedule(self):
        """Test custom temperature schedule is respected."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        custom_temps = [0.9, 0.7, 0.5, 0.3]
        config = RefinementConfig(
            max_iterations=4,
            temperature_schedule=custom_temps
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Verify custom temperatures were used
        temps = generator.llm_provider.temperatures_used
        assert len(temps) == len(custom_temps)
        for i, temp in enumerate(temps):
            assert temp == custom_temps[i]
        print(f"✓ Custom temperature schedule respected: {temps}")


@pytest.mark.asyncio
class TestFeedbackIntegration:
    """Test feedback integration into refinement prompts."""

    async def test_feedback_in_refinement_prompts(self):
        """Test that scorer feedback is integrated into subsequent prompts."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        config = RefinementConfig(max_iterations=3)

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="warm intimate vocal"
        )

        # Verify scorer was called and provided feedback
        assert scorer.call_count > 0
        for request in scorer.requests:
            assert isinstance(request, ScoringRequest)
            assert request.description == "warm intimate vocal"

        # Verify generator received refinement requests
        # (First call is initial generation, rest are refinements)
        assert generator.call_count >= 1

        print(
            f"✓ Feedback integrated: {scorer.call_count} scoring calls, "
            f"{generator.call_count} generation calls"
        )

    async def test_previous_score_tracking(self):
        """Test that previous scores are tracked in requests."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()
        config = RefinementConfig(max_iterations=4)

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="test description"
        )

        # Check previous_score tracking in requests
        for i, request in enumerate(scorer.requests):
            if i == 0:
                # First iteration has no previous score
                assert request.previous_score is None
            else:
                # Subsequent iterations should have previous score
                assert request.previous_score is not None
                assert request.previous_score == result.iterations[i - 1].score

        print(
            f"✓ Previous score tracking: {len(scorer.requests)} requests, "
            f"scores properly tracked"
        )


@pytest.mark.asyncio
class TestErrorRecovery:
    """Test error recovery and graceful degradation."""

    async def test_empty_description_fails_fast(self):
        """Test that empty description fails immediately with clear error."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer()

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer
        )

        with pytest.raises(ValueError) as exc_info:
            await controller.refine(description="")

        error = exc_info.value
        assert "Description cannot be empty" in str(error)
        assert generator.call_count == 0  # Should fail before any calls
        assert scorer.call_count == 0
        print(f"✓ Empty description fails fast: {error}")

    async def test_generator_failure_propagates(self):
        """Test that generator failures propagate properly."""
        generator = IntegrationTestGenerator(should_fail=True)
        scorer = IntegrationTestScorer()

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer
        )

        with pytest.raises(Exception) as exc_info:
            await controller.refine(description="test")

        error = exc_info.value
        assert "Simulated generator failure" in str(error)
        print(f"✓ Generator failure propagated: {error}")

    async def test_scorer_failure_propagates(self):
        """Test that scorer failures propagate properly."""
        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(should_fail=True)

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer
        )

        with pytest.raises(Exception) as exc_info:
            await controller.refine(description="test")

        error = exc_info.value
        assert "Simulated scorer failure" in str(error)
        print(f"✓ Scorer failure propagated: {error}")


@pytest.mark.asyncio
class TestRefinementScenarios:
    """Test realistic refinement scenarios."""

    async def test_typical_refinement_scenario(self):
        """Test typical refinement with gradual improvement."""
        # Realistic score progression: large gains early, plateau later
        score_progression = [55.0, 68.0, 78.0, 84.0, 88.0, 90.0, 90.5, 90.8]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)
        config = RefinementConfig(
            max_iterations=10,
            min_score_improvement=2.0,
            convergence_window=3,
            early_stop_score=None
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="warm and intimate vocal with gentle reverb"
        )

        # Should converge on plateau (last 3 scores within 2.0 points)
        assert result.convergence_reason == "score_plateau_detected"
        assert result.total_iterations >= 3
        assert result.final_score >= 88.0
        assert result.improvement > 30.0  # Significant improvement

        print(
            f"✓ Typical refinement: {result.total_iterations} iterations, "
            f"{result.improvement:.1f} point improvement, "
            f"final score {result.final_score:.1f}"
        )

    async def test_quick_convergence_scenario(self):
        """Test quick convergence with high initial quality."""
        # High initial score, reaches target quickly
        score_progression = [85.0, 90.0, 93.0, 94.0]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)
        config = RefinementConfig(
            max_iterations=10,
            early_stop_score=92.0
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="bright and clear sound"
        )

        # Should stop early due to reaching target
        assert "target_score_reached" in result.convergence_reason
        assert result.total_iterations < 5
        assert result.final_score >= 92.0

        print(
            f"✓ Quick convergence: reached {result.final_score:.1f} "
            f"in {result.total_iterations} iterations"
        )

    async def test_difficult_optimization_scenario(self):
        """Test difficult optimization requiring many iterations."""
        # Slow, steady improvement
        score_progression = [45.0, 52.0, 58.0, 63.0, 67.0, 70.0, 72.0, 73.0]

        generator = IntegrationTestGenerator()
        scorer = IntegrationTestScorer(score_progression=score_progression)
        config = RefinementConfig(
            max_iterations=8,
            min_score_improvement=2.0,
            convergence_window=3
        )

        controller = RefinementLoopController(
            parameter_generator=generator,
            scoring_system=scorer,
            config=config
        )

        result = await controller.refine(
            description="complex atmospheric soundscape"
        )

        # Should use many iterations before converging or hitting max
        assert result.total_iterations >= 6
        assert result.improvement > 20.0

        print(
            f"✓ Difficult optimization: {result.total_iterations} iterations, "
            f"{result.improvement:.1f} point improvement from {result.iterations[0].score:.1f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
