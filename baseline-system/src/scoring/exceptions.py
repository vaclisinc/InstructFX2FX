"""Custom exceptions for scoring module.

This module defines specialized exceptions for scoring errors,
providing clear error handling and debugging information.
"""


class ScoringError(Exception):
    """Base exception for scoring errors.

    All scoring exceptions inherit from this base class,
    allowing catch-all error handling when needed.

    Attributes:
        message: Human-readable error message
        details: Optional additional error details
    """

    def __init__(self, message: str, details: str = None):
        """Initialize scoring error.

        Args:
            message: Error message
            details: Optional additional details
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        """Format error message with details."""
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class MalformedResponseError(ScoringError):
    """Exception raised when LLM scoring response cannot be parsed.

    This occurs when the LLM generates malformed JSON or includes
    non-JSON text in the scoring response.

    Attributes:
        raw_output: The raw LLM output that failed to parse
        parse_error: The original parsing exception
    """

    def __init__(self, message: str, raw_output: str = None, parse_error: Exception = None):
        """Initialize malformed response error.

        Args:
            message: Error message
            raw_output: The raw LLM output
            parse_error: Original parsing exception
        """
        self.raw_output = raw_output
        self.parse_error = parse_error

        details = []
        if raw_output:
            # Truncate long outputs for readability
            truncated = raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
            details.append(f"Raw output: {truncated}")
        if parse_error:
            details.append(f"Parse error: {str(parse_error)}")

        super().__init__(message, "\n".join(details) if details else None)


class ScoreOutOfRangeError(ScoringError):
    """Exception raised when score values are outside valid range.

    This occurs when the LLM generates scores outside the 0-100 range
    or when computed scores are invalid.

    Attributes:
        score: The invalid score value
        dimension: The dimension name (if applicable)
        valid_range: The expected valid range (min, max)
    """

    def __init__(
        self,
        message: str,
        score: float = None,
        dimension: str = None,
        valid_range: tuple = (0, 100)
    ):
        """Initialize score out of range error.

        Args:
            message: Error message
            score: The invalid score value
            dimension: Dimension name
            valid_range: Valid range tuple (min, max)
        """
        self.score = score
        self.dimension = dimension
        self.valid_range = valid_range

        details = []
        if score is not None:
            details.append(f"Invalid score: {score}")
        if dimension:
            details.append(f"Dimension: {dimension}")
        if valid_range:
            details.append(f"Valid range: {valid_range[0]}-{valid_range[1]}")

        super().__init__(message, "\n".join(details) if details else None)


class RetryExhaustedError(ScoringError):
    """Exception raised when maximum retry attempts are exhausted.

    This occurs when score extraction fails repeatedly after multiple
    retry attempts with correction prompts.

    Attributes:
        max_attempts: Maximum number of retry attempts allowed
        attempts_made: Number of attempts actually made
        last_error: The final error that triggered exhaustion
    """

    def __init__(
        self,
        message: str,
        max_attempts: int = None,
        attempts_made: int = None,
        last_error: Exception = None
    ):
        """Initialize retry exhausted error.

        Args:
            message: Error message
            max_attempts: Maximum attempts allowed
            attempts_made: Attempts made
            last_error: Final error
        """
        self.max_attempts = max_attempts
        self.attempts_made = attempts_made
        self.last_error = last_error

        details = []
        if max_attempts is not None:
            details.append(f"Max attempts: {max_attempts}")
        if attempts_made is not None:
            details.append(f"Attempts made: {attempts_made}")
        if last_error:
            details.append(f"Last error: {str(last_error)}")

        super().__init__(message, "\n".join(details) if details else None)


class ConfigurationError(ScoringError):
    """Exception raised when scoring configuration is invalid.

    This occurs when the scoring YAML configuration is missing required
    fields, has invalid values, or cannot be loaded.

    Attributes:
        config_path: Path to the configuration file
        config_error: Original configuration error
    """

    def __init__(
        self,
        message: str,
        config_path: str = None,
        config_error: Exception = None
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            config_path: Configuration file path
            config_error: Original error
        """
        self.config_path = config_path
        self.config_error = config_error

        details = []
        if config_path:
            details.append(f"Config path: {config_path}")
        if config_error:
            details.append(f"Error: {str(config_error)}")

        super().__init__(message, "\n".join(details) if details else None)


__all__ = [
    "ScoringError",
    "MalformedResponseError",
    "ScoreOutOfRangeError",
    "RetryExhaustedError",
    "ConfigurationError",
]
