"""Core scoring system for evaluating audio effect parameters.

This module implements the ScoringSystem class that uses LLMs to evaluate
generated audio effect parameters against descriptions. It supports both
parameter-only scoring (fast mode) and audio-based scoring (comprehensive mode).
"""

import re
import json
import logging
from typing import Dict, Any, Optional
from pydantic import ValidationError

from .models import ScoringRequest, ScoringResponse, ScoreDimension
from .prompts import (
    format_scoring_prompt,
    format_audio_scoring_prompt,
    get_scoring_system_prompt,
)
from .config import ScoringConfig, RetryContext
from .exceptions import (
    ScoringError,
    MalformedResponseError,
    ScoreOutOfRangeError,
)

# Import LLM provider types
from models.llm_judge.base import LLMProvider
from models.llm_judge.types import LLMRequest


logger = logging.getLogger(__name__)


class ScoringSystem:
    """LLM-based scoring system for audio effect parameter evaluation.

    This system evaluates generated audio effect parameters against descriptions
    using LLM-based reasoning. It extracts structured scores across multiple
    dimensions (semantic match, technical quality, specificity) and provides
    actionable feedback for parameter refinement.

    Supports two scoring modes:
    - Parameter-only (fast): Evaluates parameters without audio processing
    - Audio-based (comprehensive): Includes audio feature analysis

    Attributes:
        llm_provider: LLM provider instance for generating scores
        config: Scoring configuration (dimensions, weights, temperature, etc.)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[ScoringConfig] = None
    ):
        """Initialize scoring system.

        Args:
            llm_provider: LLM provider instance (e.g., ClaudeProvider, OpenRouterProvider)
            config: Scoring configuration. If None, uses default configuration.

        Raises:
            ValueError: If llm_provider is None
        """
        if llm_provider is None:
            raise ValueError("llm_provider cannot be None")

        self.llm_provider = llm_provider
        self.config = config or ScoringConfig({
            "mode": "parameter_only",
            "dimensions": ["semantic_match", "technical_quality", "specificity"],
            "weights": {
                "semantic_match": 0.5,
                "technical_quality": 0.3,
                "specificity": 0.2
            },
            "temperature": 0.3,
            "retry": {
                "max_attempts": 3,
                "correction_prompt": True
            }
        })

        logger.info(
            f"ScoringSystem initialized with mode={self.config.mode}, "
            f"temperature={self.config.temperature}"
        )

    async def score_parameters(
        self,
        request: ScoringRequest
    ) -> ScoringResponse:
        """Score generated parameters against description (parameter-only mode).

        This method evaluates how well the generated parameters match the
        description without processing audio. It's faster and suitable for
        rapid iteration in the refinement loop.

        Args:
            request: Scoring request containing description and parameters

        Returns:
            ScoringResponse with scores, feedback, and suggestions

        Raises:
            ScoringError: If scoring fails after all retry attempts
            MalformedResponseError: If LLM response cannot be parsed
            ValidationError: If response doesn't match expected schema
        """
        logger.info(
            f"Scoring parameters (iteration={request.iteration}, "
            f"prev_score={request.previous_score})"
        )

        # Build scoring prompt
        prompt = format_scoring_prompt(
            description=request.description,
            parameters=request.parameters,
            previous_score=request.previous_score,
            iteration=request.iteration
        )

        # Get system prompt for consistent scoring behavior
        system_prompt = get_scoring_system_prompt()

        # Use retry context for robust score extraction
        with RetryContext(
            max_attempts=self.config.max_retry_attempts,
            correction_prompt_enabled=self.config.use_correction_prompt
        ) as retry:
            for attempt in retry:
                try:
                    # Create LLM request
                    llm_request = LLMRequest(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=self.config.temperature,
                        max_tokens=4096
                    )

                    # Get LLM evaluation
                    logger.debug(f"Attempt {attempt}: Requesting LLM scoring")
                    response = await self.llm_provider.generate_with_retry(llm_request)

                    # Parse and validate score
                    score_response = self.parse_score_response(response.content)

                    # Validate and adjust scores if needed
                    score_response = self._validate_and_adjust_scores(score_response)

                    logger.info(
                        f"Scoring completed: overall={score_response.overall_score:.1f}, "
                        f"confidence={score_response.confidence:.2f}"
                    )

                    return score_response

                except MalformedResponseError as e:
                    logger.warning(f"Attempt {attempt} failed to parse score: {e}")

                    if attempt >= retry.max_attempts:
                        # Final attempt failed
                        retry.raise_exhausted()

                    # Get correction prompt for next attempt
                    correction = retry.get_correction_prompt(e)
                    if correction:
                        # Append correction to prompt
                        prompt = f"{prompt}\n\n{correction}"

                    continue

        # Should never reach here due to retry.raise_exhausted()
        raise ScoringError("Unexpected error in retry logic")

    async def score_with_audio(
        self,
        request: ScoringRequest,
        audio_path: str
    ) -> ScoringResponse:
        """Score parameters including audio analysis (audio-based mode).

        This method evaluates both the parameters and the actual sonic result
        by extracting and analyzing audio features. More accurate but slower
        than parameter-only scoring.

        Args:
            request: Scoring request containing description and parameters
            audio_path: Path to audio file for feature extraction

        Returns:
            ScoringResponse with scores, feedback, and suggestions

        Raises:
            ScoringError: If scoring fails after all retry attempts
            FileNotFoundError: If audio file doesn't exist
        """
        logger.info(f"Scoring with audio analysis: {audio_path}")

        # Extract audio features
        features = await self.extract_audio_features(audio_path)

        # Update request with audio features
        request.audio_features = features

        # Build enhanced scoring prompt with audio context
        prompt = format_audio_scoring_prompt(
            description=request.description,
            parameters=request.parameters,
            audio_features=features,
            previous_score=request.previous_score,
            iteration=request.iteration
        )

        # Get system prompt
        system_prompt = get_scoring_system_prompt()

        # Use retry context for robust score extraction
        with RetryContext(
            max_attempts=self.config.max_retry_attempts,
            correction_prompt_enabled=self.config.use_correction_prompt
        ) as retry:
            for attempt in retry:
                try:
                    # Create LLM request
                    llm_request = LLMRequest(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=self.config.temperature,
                        max_tokens=4096
                    )

                    # Get LLM evaluation
                    response = await self.llm_provider.generate_with_retry(llm_request)

                    # Parse and validate score
                    score_response = self.parse_score_response(response.content)

                    # Validate and adjust scores
                    score_response = self._validate_and_adjust_scores(score_response)

                    logger.info(
                        f"Audio-based scoring completed: overall={score_response.overall_score:.1f}"
                    )

                    return score_response

                except MalformedResponseError as e:
                    logger.warning(f"Attempt {attempt} failed: {e}")

                    if attempt >= retry.max_attempts:
                        retry.raise_exhausted()

                    # Get correction prompt
                    correction = retry.get_correction_prompt(e)
                    if correction:
                        prompt = f"{prompt}\n\n{correction}"

                    continue

        raise ScoringError("Unexpected error in retry logic")

    def parse_score_response(self, content: str) -> ScoringResponse:
        """Extract structured scores from LLM response.

        This method parses the LLM's text response to extract JSON-formatted
        scores. It handles various response formats and validates the structure.

        Args:
            content: Raw LLM response content

        Returns:
            Parsed and validated ScoringResponse

        Raises:
            MalformedResponseError: If JSON cannot be extracted or parsed
            ValidationError: If parsed data doesn't match expected schema
        """
        try:
            # Extract JSON from response using regex
            # Look for content between curly braces (supports multiline)
            json_match = re.search(r'\{.*\}', content, re.DOTALL)

            if not json_match:
                raise MalformedResponseError(
                    "No JSON found in LLM response",
                    raw_output=content
                )

            json_str = json_match.group(0)

            # Parse JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise MalformedResponseError(
                    "Failed to parse JSON from LLM response",
                    raw_output=json_str,
                    parse_error=e
                )

            # Validate against Pydantic model
            try:
                score_response = ScoringResponse(**data)
                return score_response
            except ValidationError as e:
                raise MalformedResponseError(
                    "Response JSON doesn't match expected schema",
                    raw_output=json_str,
                    parse_error=e
                )

        except MalformedResponseError:
            # Re-raise as-is
            raise
        except Exception as e:
            # Catch any unexpected errors
            raise MalformedResponseError(
                f"Unexpected error parsing score response: {e}",
                raw_output=content,
                parse_error=e
            )

    def compute_weighted_score(
        self,
        dimensions: list[ScoreDimension]
    ) -> float:
        """Compute weighted average score across dimensions.

        Uses dimension weights from configuration to calculate overall score.
        Falls back to equal weights if a dimension is missing from config.

        Args:
            dimensions: List of scored dimensions with individual scores

        Returns:
            Weighted average score (0-100)

        Raises:
            ValueError: If dimensions list is empty
            ScoreOutOfRangeError: If computed score is invalid
        """
        if not dimensions:
            raise ValueError("Cannot compute weighted score from empty dimensions list")

        total_score = 0.0
        total_weight = 0.0

        for dim in dimensions:
            # Get weight from config, default to 1.0 if not found
            weight = self.config.weights.get(dim.name, 1.0)
            total_score += dim.score * weight
            total_weight += weight

        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")

        weighted_score = total_score / total_weight

        # Validate range
        if weighted_score < 0 or weighted_score > 100:
            raise ScoreOutOfRangeError(
                "Computed weighted score is out of valid range",
                score=weighted_score,
                valid_range=(0, 100)
            )

        return weighted_score

    def get_system_prompt(self) -> str:
        """Get system prompt for scoring.

        Returns:
            System prompt string for LLM context
        """
        return get_scoring_system_prompt()

    async def extract_audio_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract audio features for scoring context.

        This is a placeholder for audio feature extraction. In the full
        implementation, this would use librosa or similar to extract:
        - Spectral features (centroid, rolloff, bandwidth)
        - Temporal features (RMS energy, zero crossing rate)
        - Harmonic features (harmonic ratio)
        - Room characteristics (estimated reverb time)

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary of extracted audio features

        Raises:
            FileNotFoundError: If audio file doesn't exist
        """
        # Placeholder implementation
        # TODO: Implement actual audio feature extraction using librosa
        logger.warning(
            "extract_audio_features is placeholder - returning empty features"
        )

        import os
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Return placeholder features
        return {
            "spectral_centroid": 2500.0,
            "spectral_rolloff": 5000.0,
            "spectral_bandwidth": 1500.0,
            "rms_energy": 0.05,
            "zero_crossing_rate": 0.1,
            "harmonic_ratio": 0.7,
        }

    def add_audio_context(
        self,
        prompt: str,
        features: Dict[str, Any]
    ) -> str:
        """Add audio features to scoring prompt.

        Args:
            prompt: Base scoring prompt
            features: Extracted audio features

        Returns:
            Enhanced prompt with audio context
        """
        from .prompts import format_audio_features

        audio_context = format_audio_features(features)
        return f"""{prompt}

AUDIO ANALYSIS:
{audio_context}

Consider both the parameters AND the actual audio characteristics when scoring.
"""

    def _validate_and_adjust_scores(
        self,
        response: ScoringResponse
    ) -> ScoringResponse:
        """Validate and adjust scores to ensure they're within valid ranges.

        Clamps out-of-range scores to [0, 100] and logs warnings.

        Args:
            response: Raw scoring response

        Returns:
            Validated and adjusted scoring response
        """
        adjusted = False

        # Check dimension scores
        for dim in response.dimensions:
            if dim.score < 0:
                logger.warning(
                    f"Dimension '{dim.name}' score {dim.score} < 0, clamping to 0"
                )
                dim.score = 0.0
                adjusted = True
            elif dim.score > 100:
                logger.warning(
                    f"Dimension '{dim.name}' score {dim.score} > 100, clamping to 100"
                )
                dim.score = 100.0
                adjusted = True

        # Check overall score
        if response.overall_score < 0:
            logger.warning(
                f"Overall score {response.overall_score} < 0, clamping to 0"
            )
            response.overall_score = 0.0
            adjusted = True
        elif response.overall_score > 100:
            logger.warning(
                f"Overall score {response.overall_score} > 100, clamping to 100"
            )
            response.overall_score = 100.0
            adjusted = True

        # Check confidence
        if response.confidence < 0:
            logger.warning(
                f"Confidence {response.confidence} < 0, clamping to 0"
            )
            response.confidence = 0.0
            adjusted = True
        elif response.confidence > 1:
            logger.warning(
                f"Confidence {response.confidence} > 1, clamping to 1"
            )
            response.confidence = 1.0
            adjusted = True

        if adjusted:
            logger.info("Scores were adjusted to fit valid ranges")

        return response


__all__ = [
    "ScoringSystem",
]
