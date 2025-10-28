"""Configuration management for scoring system.

This module handles loading and validation of scoring configuration from YAML files,
including dimension weights, retry settings, and scoring modes.
"""

import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Callable, TypeVar
from functools import wraps

from .exceptions import ConfigurationError, RetryExhaustedError, MalformedResponseError

T = TypeVar('T')


class ScoringConfig:
    """Configuration for scoring system.

    Attributes:
        mode: Scoring mode (parameter_only or audio_based)
        dimensions: List of dimension names to evaluate
        weights: Dictionary mapping dimension names to weights
        temperature: Temperature setting for LLM scoring
        max_retry_attempts: Maximum retry attempts for score extraction
        use_correction_prompt: Whether to use correction prompts on retry
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize scoring configuration from dictionary.

        Args:
            config_dict: Configuration dictionary from YAML

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            self.mode = config_dict.get("mode", "parameter_only")
            self.dimensions = config_dict.get("dimensions", [
                "semantic_match",
                "technical_quality",
                "specificity"
            ])

            # Load dimension weights
            weights_config = config_dict.get("weights", {})
            self.weights = {
                "semantic_match": weights_config.get("semantic_match", 0.5),
                "technical_quality": weights_config.get("technical_quality", 0.3),
                "specificity": weights_config.get("specificity", 0.2)
            }

            # Load retry configuration
            retry_config = config_dict.get("retry", {})
            self.max_retry_attempts = retry_config.get("max_attempts", 3)
            self.use_correction_prompt = retry_config.get("correction_prompt", True)

            # Temperature setting
            self.temperature = config_dict.get("temperature", 0.3)

            # Validate configuration
            self._validate()

        except KeyError as e:
            raise ConfigurationError(
                f"Missing required configuration field: {e}",
                config_error=e
            )

    def _validate(self):
        """Validate configuration values.

        Raises:
            ConfigurationError: If validation fails
        """
        # Validate mode
        valid_modes = ["parameter_only", "audio_based"]
        if self.mode not in valid_modes:
            raise ConfigurationError(
                f"Invalid scoring mode: {self.mode}. Must be one of {valid_modes}"
            )

        # Validate dimensions
        if not self.dimensions:
            raise ConfigurationError("At least one dimension must be specified")

        # Validate weights sum to approximately 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ConfigurationError(
                f"Dimension weights must sum to 1.0, got {total_weight}"
            )

        # Validate weight coverage
        for dim in self.dimensions:
            if dim not in self.weights:
                raise ConfigurationError(
                    f"Missing weight for dimension: {dim}"
                )

        # Validate temperature range
        if not 0.0 <= self.temperature <= 2.0:
            raise ConfigurationError(
                f"Temperature must be between 0.0 and 2.0, got {self.temperature}"
            )

        # Validate retry attempts
        if self.max_retry_attempts < 1:
            raise ConfigurationError(
                f"max_retry_attempts must be at least 1, got {self.max_retry_attempts}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "mode": self.mode,
            "dimensions": self.dimensions,
            "weights": self.weights,
            "temperature": self.temperature,
            "retry": {
                "max_attempts": self.max_retry_attempts,
                "correction_prompt": self.use_correction_prompt
            }
        }


def load_scoring_config(config_path: Optional[str] = None) -> ScoringConfig:
    """Load scoring configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, uses default config.

    Returns:
        ScoringConfig object

    Raises:
        ConfigurationError: If config file cannot be loaded or is invalid
    """
    # Use default config path if not specified
    if config_path is None:
        # Look for scoring.yaml in configs directory
        default_path = Path(__file__).parent.parent.parent / "configs" / "scoring.yaml"
        if default_path.exists():
            config_path = str(default_path)
        else:
            # Use embedded defaults if no config file exists
            return ScoringConfig({
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

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract scoring section if it exists
        if 'scoring' in config_dict:
            config_dict = config_dict['scoring']

        return ScoringConfig(config_dict)

    except FileNotFoundError as e:
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            config_path=config_path,
            config_error=e
        )
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML configuration",
            config_path=config_path,
            config_error=e
        )
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load configuration",
            config_path=config_path,
            config_error=e
        )


def generate_correction_prompt(original_error: str, raw_output: str) -> str:
    """Generate a correction prompt for malformed LLM responses.

    Args:
        original_error: Description of the parsing error
        raw_output: The raw output that failed to parse

    Returns:
        Correction prompt text
    """
    return f"""Your previous response could not be parsed correctly.

Error: {original_error}

Your response was:
{raw_output[:500]}{"..." if len(raw_output) > 500 else ""}

Please provide the scoring response in STRICT JSON format. The response must be:
1. Valid JSON (no extra text before or after)
2. All scores must be numbers between 0 and 100
3. Include all required fields: dimensions, overall_score, feedback, suggestions, confidence

Format:
{{
  "dimensions": [
    {{"name": "semantic_match", "score": 85, "reasoning": "..."}},
    {{"name": "technical_quality", "score": 90, "reasoning": "..."}},
    {{"name": "specificity", "score": 75, "reasoning": "..."}}
  ],
  "overall_score": 83,
  "feedback": "Brief overall feedback...",
  "suggestions": ["Suggestion 1", "Suggestion 2"],
  "confidence": 0.85
}}

Please respond with only the JSON, no additional text."""


def retry_with_correction(
    max_attempts: int = 3,
    correction_prompt_enabled: bool = True,
    backoff_factor: float = 1.5,
    initial_delay: float = 0.5
) -> Callable:
    """Decorator for retrying functions with exponential backoff and correction prompts.

    Args:
        max_attempts: Maximum number of retry attempts
        correction_prompt_enabled: Whether to generate correction prompts
        backoff_factor: Multiplier for exponential backoff
        initial_delay: Initial delay before first retry (seconds)

    Returns:
        Decorator function

    Example:
        @retry_with_correction(max_attempts=3, correction_prompt_enabled=True)
        async def parse_score(response):
            # Function that might raise MalformedResponseError
            return parse_json(response)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_error = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except MalformedResponseError as e:
                    last_error = e

                    if attempt >= max_attempts:
                        # Max attempts reached
                        break

                    # Apply exponential backoff
                    time.sleep(delay)
                    delay *= backoff_factor

                    # If correction prompts are enabled, inject correction
                    if correction_prompt_enabled and hasattr(func, '_inject_correction'):
                        correction = generate_correction_prompt(
                            str(e.parse_error or "Malformed JSON"),
                            e.raw_output or ""
                        )
                        kwargs['correction_prompt'] = correction

                    continue

            # All retries exhausted
            raise RetryExhaustedError(
                "Failed to parse score after maximum retry attempts",
                max_attempts=max_attempts,
                attempts_made=max_attempts,
                last_error=last_error
            )

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_error = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except MalformedResponseError as e:
                    last_error = e

                    if attempt >= max_attempts:
                        # Max attempts reached
                        break

                    # Apply exponential backoff
                    time.sleep(delay)
                    delay *= backoff_factor

                    # If correction prompts are enabled, inject correction
                    if correction_prompt_enabled and hasattr(func, '_inject_correction'):
                        correction = generate_correction_prompt(
                            str(e.parse_error or "Malformed JSON"),
                            e.raw_output or ""
                        )
                        kwargs['correction_prompt'] = correction

                    continue

            # All retries exhausted
            raise RetryExhaustedError(
                "Failed to parse score after maximum retry attempts",
                max_attempts=max_attempts,
                attempts_made=max_attempts,
                last_error=last_error
            )

        # Return appropriate wrapper based on whether function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


class RetryContext:
    """Context manager for retry logic with correction prompts.

    Example:
        with RetryContext(max_attempts=3) as retry:
            for attempt in retry:
                try:
                    result = parse_score(response)
                    break
                except MalformedResponseError as e:
                    if attempt < retry.max_attempts:
                        correction = retry.get_correction_prompt(e)
                        # Use correction in next attempt
                    else:
                        raise
    """

    def __init__(
        self,
        max_attempts: int = 3,
        correction_prompt_enabled: bool = True,
        backoff_factor: float = 1.5,
        initial_delay: float = 0.5
    ):
        """Initialize retry context.

        Args:
            max_attempts: Maximum retry attempts
            correction_prompt_enabled: Enable correction prompts
            backoff_factor: Exponential backoff multiplier
            initial_delay: Initial delay in seconds
        """
        self.max_attempts = max_attempts
        self.correction_prompt_enabled = correction_prompt_enabled
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.current_attempt = 0
        self.last_error: Optional[Exception] = None

    def __enter__(self):
        """Enter retry context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit retry context."""
        return False

    def __iter__(self):
        """Iterate through retry attempts."""
        for attempt in range(1, self.max_attempts + 1):
            self.current_attempt = attempt

            if attempt > 1:
                # Apply backoff delay
                delay = self.initial_delay * (self.backoff_factor ** (attempt - 2))
                time.sleep(delay)

            yield attempt

    def get_correction_prompt(self, error: MalformedResponseError) -> Optional[str]:
        """Generate correction prompt for error.

        Args:
            error: The malformed response error

        Returns:
            Correction prompt text or None if disabled
        """
        self.last_error = error

        if not self.correction_prompt_enabled:
            return None

        return generate_correction_prompt(
            str(error.parse_error or "Malformed JSON"),
            error.raw_output or ""
        )

    def raise_exhausted(self):
        """Raise RetryExhaustedError with current state."""
        raise RetryExhaustedError(
            "Retry attempts exhausted",
            max_attempts=self.max_attempts,
            attempts_made=self.current_attempt,
            last_error=self.last_error
        )


__all__ = [
    "ScoringConfig",
    "load_scoring_config",
    "generate_correction_prompt",
    "retry_with_correction",
    "RetryContext",
]
