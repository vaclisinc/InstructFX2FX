"""Refinement loop controller for iterative parameter optimization.

This module implements the RefinementLoopController that orchestrates the
closed-loop refinement system between parameter generation and scoring.
It manages the refinement cycle, integrates score-based feedback, tracks
iteration history, and implements convergence detection.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from src.models.refinement import (
    IterationResult,
    RefinementConfig,
    RefinementResult,
)
from src.convergence import ConvergenceDetector
from src.generation.parameter_generator import ParameterGenerator
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest, ScoringResponse
from models.llm_judge.types import LLMRequest


logger = logging.getLogger(__name__)


class RefinementLoopController:
    """Controller for iterative refinement of audio effect parameters.

    This class orchestrates the closed-loop refinement system:
    1. Generates initial parameters from description
    2. Scores parameters using ScoringSystem
    3. Integrates score feedback into refinement prompts
    4. Generates improved parameters
    5. Repeats until convergence or max iterations

    The controller tracks full iteration history, implements convergence
    detection, and supports both parameter-only and audio-based refinement modes.

    Attributes:
        generator: ParameterGenerator instance for parameter creation
        scorer: ScoringSystem instance for parameter evaluation
        processor: Optional audio processor for audio-based mode
        config: RefinementConfig with loop behavior settings
        history: List of iteration results (populated during refinement)
        convergence_detector: ConvergenceDetector for early stopping
    """

    def __init__(
        self,
        parameter_generator: ParameterGenerator,
        scoring_system: ScoringSystem,
        audio_processor: Optional[Any] = None,
        config: Optional[RefinementConfig] = None
    ):
        """Initialize refinement loop controller.

        Args:
            parameter_generator: ParameterGenerator instance
            scoring_system: ScoringSystem instance
            audio_processor: Optional audio processor for audio-based mode
            config: RefinementConfig (default: RefinementConfig())

        Raises:
            ValueError: If parameter_generator or scoring_system is None
        """
        if parameter_generator is None:
            raise ValueError("parameter_generator cannot be None")
        if scoring_system is None:
            raise ValueError("scoring_system cannot be None")

        self.generator = parameter_generator
        self.scorer = scoring_system
        self.processor = audio_processor
        self.config = config or RefinementConfig()
        self.history: List[IterationResult] = []

        # Initialize convergence detector from config
        self.convergence_detector = ConvergenceDetector(
            window_size=self.config.convergence_window,
            min_improvement=self.config.min_score_improvement
        )

        logger.info(
            f"RefinementLoopController initialized: max_iterations={self.config.max_iterations}, "
            f"mode={self.config.mode}, convergence_window={self.config.convergence_window}"
        )

    async def refine(
        self,
        description: str,
        initial_parameters: Optional[Dict[str, Any]] = None
    ) -> RefinementResult:
        """Run iterative refinement loop.

        This is the main entry point for refinement. It orchestrates the
        complete refinement cycle from initial generation through convergence.

        Args:
            description: User's high-level description of desired audio effect
            initial_parameters: Optional starting parameters (generated if None)

        Returns:
            RefinementResult with best parameters and full iteration history

        Raises:
            ValueError: If description is empty
            Exception: If generation or scoring fails

        Example:
            >>> controller = RefinementLoopController(generator, scorer)
            >>> result = await controller.refine(
            ...     description="warm and intimate vocal sound"
            ... )
            >>> print(f"Final score: {result.final_score}")
            >>> print(f"Iterations: {result.total_iterations}")
        """
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")

        logger.info(f"Starting refinement for: '{description}'")
        self.history = []

        # Generate initial parameters if not provided
        if initial_parameters is None:
            logger.info("Generating initial parameters...")
            effect_chain = await self.generator.generate_parameters(
                description=description
            )
            initial_parameters = effect_chain.to_dict()
            logger.info(f"Initial parameters generated: {list(initial_parameters.keys())}")

        current_parameters = initial_parameters

        # Refinement loop
        for iteration in range(self.config.max_iterations):
            logger.info(f"--- Iteration {iteration} ---")

            # Score current parameters
            score_request = ScoringRequest(
                description=description,
                parameters=current_parameters,
                iteration=iteration,
                previous_score=self.history[-1].score if self.history else None
            )

            # Choose scoring mode
            if self.config.mode == "audio_based" and self.processor:
                logger.info("Scoring with audio analysis...")
                # Process audio and score
                audio_path = await self.processor.apply_effects(current_parameters)
                score_response = await self.scorer.score_with_audio(
                    score_request,
                    audio_path
                )
            else:
                logger.info("Scoring parameters only...")
                score_response = await self.scorer.score_parameters(score_request)

            logger.info(
                f"Iteration {iteration} score: {score_response.overall_score:.1f}/100 "
                f"(confidence: {score_response.confidence:.2f})"
            )

            # Record iteration result
            iteration_result = IterationResult(
                iteration=iteration,
                parameters=current_parameters,
                score=score_response.overall_score,
                feedback=score_response.feedback,
                suggestions=score_response.suggestions,
                timestamp=datetime.now().isoformat()
            )
            self.history.append(iteration_result)

            # Check convergence
            if self.should_stop(score_response.overall_score):
                logger.info(f"Convergence detected after {iteration + 1} iterations")
                break

            # Generate refined parameters for next iteration
            logger.info("Generating refined parameters...")
            current_parameters = await self.generate_refinement(
                description=description,
                current_parameters=current_parameters,
                score_response=score_response,
                iteration=iteration
            )

        # Select best parameters from history
        best_result = max(self.history, key=lambda r: r.score)
        logger.info(
            f"Refinement complete: {len(self.history)} iterations, "
            f"best score: {best_result.score:.1f}, "
            f"improvement: {best_result.score - self.history[0].score:.1f}"
        )

        return RefinementResult(
            description=description,
            initial_parameters=initial_parameters,
            final_parameters=best_result.parameters,
            iterations=self.history,
            total_iterations=len(self.history),
            final_score=best_result.score,
            improvement=best_result.score - self.history[0].score,
            convergence_reason=self.get_convergence_reason()
        )

    async def generate_refinement(
        self,
        description: str,
        current_parameters: Dict[str, Any],
        score_response: ScoringResponse,
        iteration: int
    ) -> Dict[str, Any]:
        """Generate refined parameters based on score feedback.

        This method builds a refinement prompt that includes:
        - Original description
        - Current parameters
        - Current score
        - Detailed feedback
        - Specific suggestions
        - Dimension-level reasoning

        The LLM uses this context to generate improved parameters that
        address the identified weaknesses while preserving strengths.

        Args:
            description: Original user description
            current_parameters: Current effect parameters
            score_response: Scoring result with feedback and suggestions
            iteration: Current iteration number

        Returns:
            Dictionary of refined effect parameters

        Raises:
            Exception: If parameter generation fails
        """
        # Build refinement prompt with comprehensive feedback
        refinement_prompt = f"""ORIGINAL DESCRIPTION: {description}

CURRENT PARAMETERS:
{json.dumps(current_parameters, indent=2)}

CURRENT SCORE: {score_response.overall_score}/100

FEEDBACK: {score_response.feedback}

SUGGESTIONS:
{chr(10).join(f'- {s}' for s in score_response.suggestions)}

DIMENSION SCORES:
{chr(10).join(f'- {d.name}: {d.score}/100 - {d.reasoning}' for d in score_response.dimensions)}

TASK: Generate improved parameters that address the feedback and suggestions.
Focus on the lowest-scoring dimensions while maintaining what works well.
Ensure the parameters better match the original description.
"""

        logger.debug(f"Refinement prompt length: {len(refinement_prompt)} chars")

        # Adjust temperature based on iteration
        temperature = self.get_temperature_for_iteration(iteration)
        logger.debug(f"Using temperature: {temperature}")

        # Generate refined parameters using LLM
        response = await self.generator.llm_provider.generate_with_retry(
            LLMRequest(
                prompt=refinement_prompt,
                system_prompt=self.generator.template.system_prompt,
                temperature=temperature,
                max_tokens=2048
            )
        )

        # Parse and validate refined parameters
        refined_effect_chain = self.generator.parse_and_validate(
            response.content,
            description=description
        )

        refined_parameters = refined_effect_chain.to_dict()
        logger.info(f"Refined parameters generated: {list(refined_parameters.keys())}")

        return refined_parameters

    def should_stop(self, current_score: float) -> bool:
        """Determine if refinement should stop.

        Checks multiple stopping conditions:
        1. Early stop score threshold reached
        2. Convergence detected (score plateau)

        Args:
            current_score: Current iteration score

        Returns:
            True if refinement should stop, False to continue
        """
        # Early stopping if target score reached
        if (self.config.early_stop_score is not None and
            current_score >= self.config.early_stop_score):
            logger.info(
                f"Early stop: score {current_score:.1f} >= "
                f"target {self.config.early_stop_score}"
            )
            return True

        # Check convergence using ConvergenceDetector
        if len(self.history) < self.config.convergence_window:
            return False

        scores = [r.score for r in self.history]
        has_converged = self.convergence_detector.has_converged(scores)

        if has_converged:
            recent_scores = scores[-self.config.convergence_window:]
            logger.info(
                f"Convergence detected: recent scores {recent_scores} "
                f"have range < {self.config.min_score_improvement}"
            )

        return has_converged

    def get_temperature_for_iteration(self, iteration: int) -> float:
        """Get temperature for current iteration.

        Uses temperature schedule if configured, otherwise implements
        a default schedule that starts high (exploration) and decreases
        over iterations (exploitation).

        Args:
            iteration: Current iteration number (0-indexed)

        Returns:
            Temperature value (0.0-2.0)
        """
        if self.config.temperature_schedule:
            # Use configured schedule, clamping to last value if needed
            idx = min(iteration, len(self.config.temperature_schedule) - 1)
            temp = self.config.temperature_schedule[idx]
            logger.debug(f"Using scheduled temperature[{idx}]: {temp}")
            return temp

        # Default schedule: high exploration early, low exploitation later
        # Start at 0.9, decrease by 0.06 per iteration, minimum 0.3
        temp = max(0.3, 0.9 - (iteration * 0.06))
        logger.debug(f"Using default temperature for iteration {iteration}: {temp}")
        return temp

    def get_convergence_reason(self) -> str:
        """Explain why refinement stopped.

        Analyzes the final state to determine the convergence reason:
        - max_iterations_reached
        - target_score_reached (score)
        - score_plateau_detected

        Returns:
            String describing why refinement stopped
        """
        if not self.history:
            return "no_iterations_completed"

        if len(self.history) >= self.config.max_iterations:
            return "max_iterations_reached"

        last_score = self.history[-1].score
        if (self.config.early_stop_score is not None and
            last_score >= self.config.early_stop_score):
            return f"target_score_reached ({last_score:.1f})"

        return "score_plateau_detected"


__all__ = [
    "RefinementLoopController",
]
