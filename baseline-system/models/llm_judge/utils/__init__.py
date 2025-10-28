"""Utility modules for LLM provider error handling and retry logic.

This package provides:
- Custom exception hierarchy for better error classification
- Retry utilities with exponential backoff
- Advanced rate limiting utilities
"""

from .exceptions import (
    LLMProviderError,
    ConfigurationError,
    AuthenticationError,
    RateLimitError,
    APIError,
    TimeoutError,
    NetworkError,
    InvalidRequestError,
    ResponseParsingError,
    MaxRetriesExceededError,
    is_retryable_error,
)

from .retry import (
    calculate_exponential_backoff,
    retry_async,
    retry_sync,
    with_retry,
    with_retry_sync,
)

from .rate_limiter import (
    TokenBucketRateLimiter,
    AdaptiveRateLimiter,
    MultiProviderRateLimiter,
)


__all__ = [
    # Exceptions
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
    # Retry utilities
    "calculate_exponential_backoff",
    "retry_async",
    "retry_sync",
    "with_retry",
    "with_retry_sync",
    # Rate limiting
    "TokenBucketRateLimiter",
    "AdaptiveRateLimiter",
    "MultiProviderRateLimiter",
]
