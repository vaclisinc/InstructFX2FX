"""Custom exception classes for LLM provider error handling.

This module defines a hierarchy of exceptions for better error handling
and classification throughout the LLM provider abstraction layer.
"""


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors.

    All exceptions raised by the LLM provider layer should inherit from this.
    """
    pass


class ConfigurationError(LLMProviderError):
    """Raised when provider configuration is invalid or missing.

    Examples:
        - Missing API key
        - Invalid model name
        - Malformed configuration dictionary
    """
    pass


class AuthenticationError(LLMProviderError):
    """Raised when API authentication fails.

    Examples:
        - Invalid API key
        - Expired credentials
        - Insufficient permissions
    """
    pass


class RateLimitError(LLMProviderError):
    """Raised when API rate limits are exceeded.

    This is a retryable error that should trigger exponential backoff.

    Attributes:
        retry_after: Optional number of seconds to wait before retrying
    """

    def __init__(self, message: str, retry_after: float = None):
        """Initialize rate limit error.

        Args:
            message: Error description
            retry_after: Suggested wait time in seconds before retry
        """
        super().__init__(message)
        self.retry_after = retry_after


class APIError(LLMProviderError):
    """Raised when API returns an error response.

    This is a general API error that may or may not be retryable
    depending on the status code.

    Attributes:
        status_code: HTTP status code (if applicable)
        response_body: Raw response body from API
    """

    def __init__(self, message: str, status_code: int = None, response_body: str = None):
        """Initialize API error.

        Args:
            message: Error description
            status_code: HTTP status code
            response_body: Raw API response body
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body

    def is_retryable(self) -> bool:
        """Check if this error should be retried.

        Returns:
            True if error is retryable (5xx, 429), False otherwise
        """
        if not self.status_code:
            return False

        # Retry on server errors and rate limits
        return self.status_code >= 500 or self.status_code == 429


class TimeoutError(LLMProviderError):
    """Raised when API request times out.

    This is a retryable error.

    Attributes:
        timeout_seconds: The timeout duration that was exceeded
    """

    def __init__(self, message: str, timeout_seconds: float = None):
        """Initialize timeout error.

        Args:
            message: Error description
            timeout_seconds: The timeout duration that was exceeded
        """
        super().__init__(message)
        self.timeout_seconds = timeout_seconds


class NetworkError(LLMProviderError):
    """Raised when network connectivity issues occur.

    This is a retryable error.

    Examples:
        - Connection refused
        - DNS resolution failure
        - Network unreachable
    """
    pass


class InvalidRequestError(LLMProviderError):
    """Raised when request parameters are invalid.

    This is NOT a retryable error - the request must be fixed.

    Examples:
        - Empty prompt
        - Invalid temperature range
        - Unsupported model name
    """
    pass


class ResponseParsingError(LLMProviderError):
    """Raised when API response cannot be parsed.

    This may be retryable if it's a temporary API issue.

    Attributes:
        raw_response: The unparseable response body
    """

    def __init__(self, message: str, raw_response: str = None):
        """Initialize response parsing error.

        Args:
            message: Error description
            raw_response: The raw response that couldn't be parsed
        """
        super().__init__(message)
        self.raw_response = raw_response


class MaxRetriesExceededError(LLMProviderError):
    """Raised when all retry attempts have been exhausted.

    This wraps the underlying error that caused retries to fail.

    Attributes:
        attempts: Number of attempts made
        last_error: The final error that caused failure
    """

    def __init__(self, message: str, attempts: int, last_error: Exception = None):
        """Initialize max retries exceeded error.

        Args:
            message: Error description
            attempts: Number of attempts made
            last_error: The underlying error
        """
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def is_retryable_error(error: Exception) -> bool:
    """Check if an error should trigger a retry.

    This is a utility function that classifies errors as retryable or not.

    Retryable errors:
        - RateLimitError
        - TimeoutError
        - NetworkError
        - APIError with 5xx or 429 status
        - ResponseParsingError (might be temporary)

    Non-retryable errors:
        - AuthenticationError
        - ConfigurationError
        - InvalidRequestError
        - APIError with 4xx status (except 429)

    Args:
        error: The exception to classify

    Returns:
        True if error should be retried, False otherwise
    """
    # Explicitly retryable
    if isinstance(error, (RateLimitError, TimeoutError, NetworkError, ResponseParsingError)):
        return True

    # Check APIError status code
    if isinstance(error, APIError):
        return error.is_retryable()

    # Explicitly non-retryable
    if isinstance(error, (AuthenticationError, ConfigurationError, InvalidRequestError)):
        return False

    # Unknown errors - be conservative, don't retry
    return False


__all__ = [
    "LLMProviderError",
    "ConfigurationError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    "TimeoutError",
    "NetworkError",
    "InvalidRequestError",
    "ResponseParsingError",
    "MaxRetriesExceededError",
    "is_retryable_error",
]
