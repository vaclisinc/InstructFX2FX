"""Retry utilities for LLM provider operations.

This module provides additional retry utilities that complement the base
retry logic in LLMProvider. These utilities can be used for other operations
beyond LLM generation.
"""

import asyncio
import logging
import random
import time
from functools import wraps
from typing import Callable, TypeVar, Any, Optional

from .exceptions import MaxRetriesExceededError, is_retryable_error


logger = logging.getLogger(__name__)

T = TypeVar('T')


def calculate_exponential_backoff(
    attempt: int,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with optional jitter.

    Args:
        attempt: Current attempt number (1-indexed)
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential calculation
        jitter: Whether to add random jitter (±20%)

    Returns:
        Delay in seconds with jitter applied
    """
    delay = min(
        initial_delay * (exponential_base ** (attempt - 1)),
        max_delay
    )

    if jitter:
        # Add ±20% random jitter
        jitter_amount = delay * 0.2 * (random.random() * 2 - 1)
        delay += jitter_amount

    return max(0.1, delay)  # Ensure minimum 0.1s delay


async def retry_async(
    func: Callable[..., T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> T:
    """Retry an async function with exponential backoff.

    This is a generic retry utility that can be used for any async operation,
    not just LLM generation.

    Args:
        func: Async function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial retry delay in seconds
        max_delay: Maximum retry delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retryable_exceptions: Tuple of exception types to retry on (None = retry all)
        on_retry: Optional callback called on each retry (exception, attempt)

    Returns:
        Result of successful function call

    Raises:
        MaxRetriesExceededError: If all attempts fail
        Exception: Last exception if non-retryable
    """
    attempt = 0
    last_exception = None

    while attempt < max_attempts:
        attempt += 1
        try:
            return await func()
        except Exception as e:
            last_exception = e

            # Check if we should retry this exception
            should_retry = False
            if retryable_exceptions is None:
                should_retry = is_retryable_error(e)
            else:
                should_retry = isinstance(e, retryable_exceptions)

            if not should_retry:
                logger.error(f"Non-retryable error on attempt {attempt}: {e}")
                raise

            if attempt >= max_attempts:
                break

            # Calculate backoff delay
            delay = calculate_exponential_backoff(
                attempt,
                initial_delay,
                max_delay,
                exponential_base,
                jitter
            )

            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            # Call retry callback if provided
            if on_retry:
                on_retry(e, attempt)

            await asyncio.sleep(delay)

    # All attempts exhausted
    raise MaxRetriesExceededError(
        f"Operation failed after {max_attempts} attempts",
        attempts=max_attempts,
        last_error=last_exception
    ) from last_exception


def retry_sync(
    func: Callable[..., T],
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
) -> T:
    """Retry a sync function with exponential backoff.

    Synchronous version of retry_async for blocking operations.

    Args:
        func: Sync function to retry
        max_attempts: Maximum number of attempts
        initial_delay: Initial retry delay in seconds
        max_delay: Maximum retry delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retryable_exceptions: Tuple of exception types to retry on (None = retry all)
        on_retry: Optional callback called on each retry (exception, attempt)

    Returns:
        Result of successful function call

    Raises:
        MaxRetriesExceededError: If all attempts fail
        Exception: Last exception if non-retryable
    """
    attempt = 0
    last_exception = None

    while attempt < max_attempts:
        attempt += 1
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Check if we should retry this exception
            should_retry = False
            if retryable_exceptions is None:
                should_retry = is_retryable_error(e)
            else:
                should_retry = isinstance(e, retryable_exceptions)

            if not should_retry:
                logger.error(f"Non-retryable error on attempt {attempt}: {e}")
                raise

            if attempt >= max_attempts:
                break

            # Calculate backoff delay
            delay = calculate_exponential_backoff(
                attempt,
                initial_delay,
                max_delay,
                exponential_base,
                jitter
            )

            logger.warning(
                f"Attempt {attempt}/{max_attempts} failed: {e}. "
                f"Retrying in {delay:.2f}s..."
            )

            # Call retry callback if provided
            if on_retry:
                on_retry(e, attempt)

            time.sleep(delay)

    # All attempts exhausted
    raise MaxRetriesExceededError(
        f"Operation failed after {max_attempts} attempts",
        attempts=max_attempts,
        last_error=last_exception
    ) from last_exception


def with_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = None
):
    """Decorator to add retry logic to async functions.

    Example:
        @with_retry(max_attempts=5, initial_delay=2.0)
        async def fetch_data():
            # Your code here
            pass

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial retry delay in seconds
        max_delay: Maximum retry delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async def call_func():
                return await func(*args, **kwargs)

            return await retry_async(
                call_func,
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions
            )

        return wrapper
    return decorator


def with_retry_sync(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = None
):
    """Decorator to add retry logic to sync functions.

    Example:
        @with_retry_sync(max_attempts=5, initial_delay=2.0)
        def fetch_data():
            # Your code here
            pass

    Args:
        max_attempts: Maximum number of attempts
        initial_delay: Initial retry delay in seconds
        max_delay: Maximum retry delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter to delays
        retryable_exceptions: Tuple of exception types to retry on

    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            def call_func():
                return func(*args, **kwargs)

            return retry_sync(
                call_func,
                max_attempts=max_attempts,
                initial_delay=initial_delay,
                max_delay=max_delay,
                exponential_base=exponential_base,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions
            )

        return wrapper
    return decorator


__all__ = [
    "calculate_exponential_backoff",
    "retry_async",
    "retry_sync",
    "with_retry",
    "with_retry_sync",
]
