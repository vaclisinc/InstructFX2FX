"""Rate limiting utilities for LLM provider operations.

This module provides additional rate limiting utilities that complement the
RateLimiter class in base.py. These utilities can be used for more complex
rate limiting scenarios.
"""

import asyncio
import time
from collections import deque
from typing import Optional, Dict
import logging


logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Token bucket rate limiter with configurable capacity and refill rate.

    This is an alternative to the sliding window rate limiter in base.py,
    providing more granular control over burst behavior.

    The token bucket algorithm allows for bursts up to the bucket capacity,
    while maintaining an average rate over time.

    Attributes:
        capacity: Maximum number of tokens (requests) in the bucket
        refill_rate: Number of tokens added per second
        tokens: Current number of tokens available
        last_refill: Timestamp of last token refill
    """

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket rate limiter.

        Args:
            capacity: Maximum tokens (burst size)
            refill_rate: Tokens added per second (average rate)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the bucket, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire (default 1)
        """
        async with self._lock:
            while True:
                self._refill()

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return

                # Not enough tokens, calculate wait time
                deficit = tokens - self.tokens
                wait_time = deficit / self.refill_rate

                logger.debug(
                    f"Rate limit: need {tokens} tokens, have {self.tokens:.2f}. "
                    f"Waiting {wait_time:.2f}s"
                )

                await asyncio.sleep(wait_time)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def available_tokens(self) -> float:
        """Get current number of available tokens.

        Returns:
            Number of tokens currently available
        """
        self._refill()
        return self.tokens


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on API feedback.

    This rate limiter automatically reduces rate when encountering 429 errors
    and gradually increases it when successful.

    Attributes:
        initial_rate: Starting requests per minute
        min_rate: Minimum allowed rate (safety floor)
        max_rate: Maximum allowed rate (safety ceiling)
        current_rate: Current requests per minute
        decrease_factor: Multiplier to reduce rate on 429 (e.g., 0.5 = halve rate)
        increase_factor: Multiplier to increase rate on success (e.g., 1.1 = +10%)
    """

    def __init__(
        self,
        initial_rate: float = 50.0,
        min_rate: float = 5.0,
        max_rate: float = 100.0,
        decrease_factor: float = 0.5,
        increase_factor: float = 1.05
    ):
        """Initialize adaptive rate limiter.

        Args:
            initial_rate: Starting requests per minute
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
            decrease_factor: Rate reduction multiplier on failure
            increase_factor: Rate increase multiplier on success
        """
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.current_rate = initial_rate
        self.decrease_factor = decrease_factor
        self.increase_factor = increase_factor

        # Use token bucket internally
        self._limiter = TokenBucketRateLimiter(
            capacity=int(initial_rate),
            refill_rate=initial_rate / 60.0  # Convert to per-second
        )

        self._lock = asyncio.Lock()
        self._success_count = 0
        self._increase_threshold = 10  # Increase rate after N successes

    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        await self._limiter.acquire()

    async def report_success(self) -> None:
        """Report successful request to potentially increase rate."""
        async with self._lock:
            self._success_count += 1

            # Increase rate after threshold successes
            if self._success_count >= self._increase_threshold:
                old_rate = self.current_rate
                self.current_rate = min(
                    self.max_rate,
                    self.current_rate * self.increase_factor
                )

                if self.current_rate != old_rate:
                    logger.info(
                        f"Rate limit increased: {old_rate:.2f} -> {self.current_rate:.2f} req/min"
                    )
                    self._update_limiter()

                self._success_count = 0

    async def report_rate_limit(self, retry_after: Optional[float] = None) -> None:
        """Report rate limit error to decrease rate.

        Args:
            retry_after: Optional seconds to wait (from Retry-After header)
        """
        async with self._lock:
            old_rate = self.current_rate
            self.current_rate = max(
                self.min_rate,
                self.current_rate * self.decrease_factor
            )

            logger.warning(
                f"Rate limit hit. Decreasing rate: {old_rate:.2f} -> {self.current_rate:.2f} req/min"
            )

            self._update_limiter()
            self._success_count = 0  # Reset success counter

            # If API provided retry_after, wait that long
            if retry_after:
                logger.info(f"Waiting {retry_after}s as suggested by API")
                await asyncio.sleep(retry_after)

    def _update_limiter(self) -> None:
        """Update internal limiter with new rate."""
        self._limiter = TokenBucketRateLimiter(
            capacity=int(self.current_rate),
            refill_rate=self.current_rate / 60.0
        )

    def get_current_rate(self) -> float:
        """Get current rate limit in requests per minute.

        Returns:
            Current rate limit
        """
        return self.current_rate


class MultiProviderRateLimiter:
    """Rate limiter that manages limits across multiple providers.

    This is useful when you have multiple LLM providers and want to track
    rate limits independently for each.

    Attributes:
        limiters: Dictionary mapping provider names to their rate limiters
    """

    def __init__(self):
        """Initialize multi-provider rate limiter."""
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = asyncio.Lock()

    async def add_provider(
        self,
        provider_name: str,
        capacity: int,
        refill_rate: float
    ) -> None:
        """Add a rate limiter for a provider.

        Args:
            provider_name: Name of the provider (e.g., 'anthropic', 'openrouter')
            capacity: Maximum burst size for this provider
            refill_rate: Tokens per second for this provider
        """
        async with self._lock:
            self.limiters[provider_name] = TokenBucketRateLimiter(
                capacity=capacity,
                refill_rate=refill_rate
            )
            logger.info(
                f"Added rate limiter for {provider_name}: "
                f"capacity={capacity}, rate={refill_rate}/s"
            )

    async def acquire(self, provider_name: str, tokens: int = 1) -> None:
        """Acquire tokens for a specific provider.

        Args:
            provider_name: Name of the provider
            tokens: Number of tokens to acquire

        Raises:
            KeyError: If provider not registered
        """
        if provider_name not in self.limiters:
            raise KeyError(
                f"Provider '{provider_name}' not registered. "
                f"Available providers: {list(self.limiters.keys())}"
            )

        await self.limiters[provider_name].acquire(tokens)

    def get_limiter(self, provider_name: str) -> TokenBucketRateLimiter:
        """Get rate limiter for a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Rate limiter instance for the provider

        Raises:
            KeyError: If provider not registered
        """
        if provider_name not in self.limiters:
            raise KeyError(f"Provider '{provider_name}' not registered")

        return self.limiters[provider_name]

    def list_providers(self) -> list:
        """List all registered providers.

        Returns:
            List of provider names
        """
        return list(self.limiters.keys())


__all__ = [
    "TokenBucketRateLimiter",
    "AdaptiveRateLimiter",
    "MultiProviderRateLimiter",
]
