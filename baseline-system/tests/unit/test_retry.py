"""Tests for retry utilities."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock

from models.llm_judge.utils.retry import (
    calculate_exponential_backoff,
    retry_async,
    retry_sync,
    with_retry,
    with_retry_sync,
)
from models.llm_judge.utils.exceptions import (
    MaxRetriesExceededError,
    NetworkError,
    RateLimitError,
    AuthenticationError,
)


class TestCalculateExponentialBackoff:
    """Test exponential backoff calculation."""

    def test_basic_exponential_backoff(self):
        """Test basic exponential backoff without jitter."""
        # Attempt 1: 1.0 * 2^0 = 1.0
        delay1 = calculate_exponential_backoff(1, initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert delay1 == 1.0

        # Attempt 2: 1.0 * 2^1 = 2.0
        delay2 = calculate_exponential_backoff(2, initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert delay2 == 2.0

        # Attempt 3: 1.0 * 2^2 = 4.0
        delay3 = calculate_exponential_backoff(3, initial_delay=1.0, exponential_base=2.0, jitter=False)
        assert delay3 == 4.0

    def test_max_delay_cap(self):
        """Test that delay is capped at max_delay."""
        # Attempt 10 would give 1.0 * 2^9 = 512, but max_delay=30
        delay = calculate_exponential_backoff(
            10,
            initial_delay=1.0,
            max_delay=30.0,
            exponential_base=2.0,
            jitter=False
        )
        assert delay == 30.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds randomness to delay."""
        delays = [
            calculate_exponential_backoff(3, initial_delay=1.0, jitter=True)
            for _ in range(10)
        ]

        # With jitter, delays should vary
        assert len(set(delays)) > 1

        # But all should be within ±20% of base delay (4.0)
        base_delay = 4.0
        for delay in delays:
            assert 4.0 * 0.8 <= delay <= 4.0 * 1.2

    def test_minimum_delay(self):
        """Test that delay is never less than 0.1s."""
        # Even with very small initial delay
        delay = calculate_exponential_backoff(1, initial_delay=0.01, jitter=False)
        assert delay >= 0.1

    def test_custom_exponential_base(self):
        """Test with different exponential base."""
        # With base 3: 1.0 * 3^2 = 9.0
        delay = calculate_exponential_backoff(
            3,
            initial_delay=1.0,
            exponential_base=3.0,
            jitter=False
        )
        assert delay == 9.0


class TestRetryAsync:
    """Test async retry function."""

    @pytest.mark.asyncio
    async def test_successful_first_attempt(self):
        """Test that successful call on first attempt doesn't retry."""
        call_count = 0

        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_async(successful_func, max_attempts=3)

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Test retry on retryable errors."""
        call_count = 0

        async def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Network error")
            return "success"

        result = await retry_async(
            failing_then_success,
            max_attempts=5,
            initial_delay=0.1,
            max_delay=1.0
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_retryable_error(self):
        """Test that non-retryable errors fail immediately."""
        call_count = 0

        async def auth_error_func():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Invalid API key")

        with pytest.raises(AuthenticationError):
            await retry_async(auth_error_func, max_attempts=3)

        # Should only be called once (no retries)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test MaxRetriesExceededError when all attempts fail."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise RateLimitError("Rate limit")

        with pytest.raises(MaxRetriesExceededError) as exc_info:
            await retry_async(
                always_fails,
                max_attempts=3,
                initial_delay=0.1
            )

        assert call_count == 3
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_error, RateLimitError)

    @pytest.mark.asyncio
    async def test_retry_with_specific_exceptions(self):
        """Test retry only on specific exception types."""
        call_count = 0

        async def network_error_func():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Network error")

        # Should not retry NetworkError when only ValueError is retryable
        with pytest.raises(NetworkError):
            await retry_async(
                network_error_func,
                max_attempts=3,
                retryable_exceptions=(ValueError,)
            )

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """Test that on_retry callback is called."""
        callback_calls = []

        def on_retry_callback(error, attempt):
            callback_calls.append((error, attempt))

        call_count = 0

        async def failing_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError(f"Attempt {call_count}")
            return "success"

        result = await retry_async(
            failing_twice,
            max_attempts=5,
            initial_delay=0.1,
            on_retry=on_retry_callback
        )

        assert result == "success"
        assert len(callback_calls) == 2
        assert callback_calls[0][1] == 1  # First attempt
        assert callback_calls[1][1] == 2  # Second attempt


class TestRetrySyncVersions:
    """Test synchronous retry function."""

    def test_retry_sync_successful(self):
        """Test sync retry with successful call."""
        call_count = 0

        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = retry_sync(successful_func, max_attempts=3)

        assert result == "success"
        assert call_count == 1

    def test_retry_sync_with_retries(self):
        """Test sync retry with failures then success."""
        call_count = 0

        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Network error")
            return "success"

        result = retry_sync(
            failing_then_success,
            max_attempts=5,
            initial_delay=0.1
        )

        assert result == "success"
        assert call_count == 3


class TestWithRetryDecorator:
    """Test retry decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Test basic decorator usage."""
        call_count = 0

        @with_retry(max_attempts=3, initial_delay=0.1)
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NetworkError("Network error")
            return "success"

        result = await decorated_func()

        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_decorator_with_args(self):
        """Test decorator with function arguments."""
        @with_retry(max_attempts=3, initial_delay=0.1)
        async def decorated_func(x, y):
            return x + y

        result = await decorated_func(5, 3)
        assert result == 8

    @pytest.mark.asyncio
    async def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""
        @with_retry(max_attempts=3, initial_delay=0.1)
        async def decorated_func(name, greeting="Hello"):
            return f"{greeting}, {name}!"

        result = await decorated_func("World", greeting="Hi")
        assert result == "Hi, World!"


class TestWithRetrySyncDecorator:
    """Test sync retry decorator."""

    def test_sync_decorator_basic(self):
        """Test basic sync decorator usage."""
        call_count = 0

        @with_retry_sync(max_attempts=3, initial_delay=0.1)
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise NetworkError("Network error")
            return "success"

        result = decorated_func()

        assert result == "success"
        assert call_count == 2

    def test_sync_decorator_with_args(self):
        """Test sync decorator with arguments."""
        @with_retry_sync(max_attempts=3, initial_delay=0.1)
        def decorated_func(x, y):
            return x * y

        result = decorated_func(4, 5)
        assert result == 20


class TestRetryTiming:
    """Test retry timing behavior."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test that exponential backoff waits correct amount of time."""
        call_count = 0
        start_time = time.time()

        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Network error")
            return "success"

        result = await retry_async(
            failing_func,
            max_attempts=3,
            initial_delay=0.1,
            exponential_base=2.0,
            jitter=False
        )

        elapsed = time.time() - start_time

        # Should wait: 0.1s (1st retry) + 0.2s (2nd retry) = 0.3s
        # Plus some tolerance for execution time
        assert elapsed >= 0.3
        assert elapsed < 0.5  # Should not take too long

        assert result == "success"
        assert call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
