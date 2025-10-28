"""Abstract base class for LLM providers.

This module defines the LLMProvider abstract base class that all concrete
provider implementations must inherit from. It provides:
- Standard interface for LLM generation
- Built-in retry logic with exponential backoff
- Rate limiting capabilities
- Request/response logging
- Token usage tracking
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, Any, Optional
import random

from .types import LLMRequest, LLMResponse, RetryConfig, RateLimitConfig


logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API requests.

    Implements a simple sliding window rate limiter to prevent
    exceeding API quota limits.
    """

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.requests: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request.

        Blocks if rate limit would be exceeded, waiting until
        a request slot becomes available.
        """
        if not self.config.enabled:
            return

        async with self._lock:
            now = time.time()
            window_start = now - 60.0  # 60 seconds window

            # Remove requests outside the window
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()

            # Check if we've hit the limit
            if len(self.requests) >= self.config.requests_per_minute:
                # Wait until oldest request falls outside window
                sleep_time = self.requests[0] - window_start + 0.1
                logger.debug(
                    f"Rate limit reached, sleeping for {sleep_time:.2f}s"
                )
                await asyncio.sleep(sleep_time)

                # Clean up again after sleep
                now = time.time()
                window_start = now - 60.0
                while self.requests and self.requests[0] < window_start:
                    self.requests.popleft()

            # Record this request
            self.requests.append(now)


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    All concrete provider implementations (ClaudeProvider, OpenRouterProvider, etc.)
    must inherit from this class and implement the abstract methods.

    The base class provides:
    - Retry logic with exponential backoff
    - Rate limiting
    - Request/response logging
    - Token usage tracking

    Attributes:
        config: Provider configuration dictionary
        retry_config: Retry behavior configuration
        rate_limit_config: Rate limiting configuration
        rate_limiter: Rate limiter instance
    """

    def __init__(
        self,
        config: Dict[str, Any],
        retry_config: Optional[RetryConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """Initialize LLM provider.

        Args:
            config: Provider-specific configuration dictionary
            retry_config: Optional retry configuration, uses defaults if not provided
            rate_limit_config: Optional rate limit config, uses defaults if not provided
        """
        self.config = config
        self.retry_config = retry_config or RetryConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.rate_limiter = RateLimiter(self.rate_limit_config)

        # Validate configuration on initialization
        if not self.validate_config():
            raise ValueError(f"Invalid configuration for {self.__class__.__name__}")

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from LLM.

        This method must be implemented by concrete provider classes.
        It should:
        1. Format the request according to provider's API
        2. Make the API call
        3. Parse the response
        4. Return an LLMResponse object

        Args:
            request: LLM request parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ValueError: If request is invalid
            RuntimeError: If API call fails after retries
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration.

        This method must be implemented by concrete provider classes.
        It should check that all required configuration values are present
        and valid (e.g., API keys, model names, endpoints).

        Returns:
            True if configuration is valid, False otherwise
        """
        pass

    async def generate_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Generate completion with retry logic and rate limiting.

        This is the main entry point for making LLM requests. It:
        1. Applies rate limiting
        2. Logs the request
        3. Calls generate() with retry logic
        4. Logs the response and token usage

        Args:
            request: LLM request parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            RuntimeError: If all retry attempts fail
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()

        # Log request (sanitized)
        logger.info(
            f"LLM request: model={request.model}, "
            f"temp={request.temperature}, "
            f"max_tokens={request.max_tokens}, "
            f"prompt_length={len(request.prompt)}"
        )
        logger.debug(f"Prompt preview: {request.prompt[:100]}...")

        attempt = 0
        last_exception = None

        while attempt < self.retry_config.max_attempts:
            try:
                start_time = time.time()
                response = await self.generate(request)
                elapsed = time.time() - start_time

                # Log response
                logger.info(
                    f"LLM response: model={response.model}, "
                    f"tokens={response.tokens_used}, "
                    f"finish_reason={response.finish_reason}, "
                    f"elapsed={elapsed:.2f}s, "
                    f"provider={response.provider}"
                )

                return response

            except Exception as e:
                last_exception = e
                attempt += 1

                if attempt >= self.retry_config.max_attempts:
                    logger.error(
                        f"All {self.retry_config.max_attempts} retry attempts failed",
                        exc_info=True
                    )
                    break

                # Calculate backoff delay
                delay = self._calculate_backoff_delay(attempt)

                # Check if error is retryable
                if not self._is_retryable_error(e):
                    logger.error(f"Non-retryable error: {e}", exc_info=True)
                    break

                logger.warning(
                    f"Attempt {attempt} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        raise RuntimeError(
            f"LLM generation failed after {attempt} attempts: {last_exception}"
        ) from last_exception

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with optional jitter.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Delay in seconds
        """
        delay = min(
            self.retry_config.initial_delay *
            (self.retry_config.exponential_base ** (attempt - 1)),
            self.retry_config.max_delay
        )

        if self.retry_config.jitter:
            # Add ±20% random jitter
            jitter = delay * 0.2 * (random.random() * 2 - 1)
            delay += jitter

        return max(0.1, delay)  # Ensure minimum 0.1s delay

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable.

        Retryable errors include:
        - Network timeouts
        - 429 (rate limit) errors
        - 5xx server errors
        - Connection errors

        Non-retryable errors include:
        - 4xx client errors (except 429)
        - Authentication errors
        - Invalid request format

        Args:
            error: The exception to check

        Returns:
            True if error should be retried, False otherwise
        """
        error_str = str(error).lower()

        # Check for retryable patterns
        retryable_patterns = [
            "timeout",
            "429",
            "rate limit",
            "too many requests",
            "5",  # 5xx errors
            "connection",
            "network",
            "temporary",
        ]

        # Check for non-retryable patterns
        non_retryable_patterns = [
            "401",
            "403",
            "authentication",
            "invalid api key",
            "bad request",
            "400",
        ]

        # First check non-retryable (takes precedence)
        for pattern in non_retryable_patterns:
            if pattern in error_str:
                return False

        # Then check retryable
        for pattern in retryable_patterns:
            if pattern in error_str:
                return True

        # Default to not retrying unknown errors
        return False

    def get_provider_name(self) -> str:
        """Get the name of this provider.

        Returns:
            Provider name (e.g., 'anthropic', 'openrouter')
        """
        return self.__class__.__name__.replace("Provider", "").lower()


__all__ = [
    "LLMProvider",
    "RateLimiter",
]
