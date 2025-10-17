"""Scoring system for audio evaluation."""

from .models import (
    ScoringRequest,
    ScoringResponse,
    ScoreDimension,
)

from .config import (
    ScoringConfig,
    load_scoring_config,
    generate_correction_prompt,
    retry_with_correction,
    RetryContext,
)

from .exceptions import (
    ScoringError,
    MalformedResponseError,
    ScoreOutOfRangeError,
    RetryExhaustedError,
    ConfigurationError,
)

from .prompts import (
    load_scoring_template,
    format_scoring_prompt,
    format_audio_scoring_prompt,
    format_audio_features,
    get_scoring_system_prompt,
    format_correction_prompt,
    format_refinement_prompt,
)

__all__ = [
    # Models
    "ScoringRequest",
    "ScoringResponse",
    "ScoreDimension",
    # Configuration
    "ScoringConfig",
    "load_scoring_config",
    "generate_correction_prompt",
    "retry_with_correction",
    "RetryContext",
    # Exceptions
    "ScoringError",
    "MalformedResponseError",
    "ScoreOutOfRangeError",
    "RetryExhaustedError",
    "ConfigurationError",
    # Prompts
    "load_scoring_template",
    "format_scoring_prompt",
    "format_audio_scoring_prompt",
    "format_audio_features",
    "get_scoring_system_prompt",
    "format_correction_prompt",
    "format_refinement_prompt",
]
