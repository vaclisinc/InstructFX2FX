"""Tests for base LLMProvider class."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from models.llm_judge.base import LLMProvider, RateLimiter
from models.llm_judge.types import LLMRequest, LLMResponse, RetryConfig, RateLimitConfig
from models.llm_judge.utils.exceptions import (
    RateLimitError,
    APIError,
    MaxRetriesExceededError
)


class MockProvider(LLMProvider):
    """Mock provider for testing base class functionality."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize mock provider."""
        super().__init__(config)
        self.generate_call_count = 0
        self.validate_call_count = 0

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Mock generate method."""
        self.generate_call_count += 1
        return LLMResponse(
            content="Mock response",
            model="mock-model",
            tokens_used=10,
            prompt_tokens=5,
            completion_tokens=5,
            finish_reason="stop",
            provider="mock"
        )

    async def validate_config(self) -> bool:
        """Mock validate_config method."""
        self.validate_call_count += 1
        return True

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "mock"


class TestLLMProviderBase:
    """Test base LLMProvider functionality."""

    def test_abstract_base_cannot_instantiate(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider({})

    def test_mock_provider_initialization(self):
        """Test mock provider initialization."""
        config = {
            "api_key": "test-key",
            "retry": {"max_attempts": 3},
            "rate_limit": {"requests_per_minute": 60}
        }
        provider = MockProvider(config)

        assert provider.config == config
        assert provider.retry_config.max_attempts == 3
        assert provider.rate_limit_config.requests_per_minute == 60

    def test_default_retry_config(self):
        """Test default retry configuration."""
        provider = MockProvider({})

        assert provider.retry_config.max_attempts == 3
        assert provider.retry_config.initial_delay == 1.0
        assert provider.retry_config.max_delay == 30.0
        assert provider.retry_config.exponential_base == 2.0
        assert provider.retry_config.jitter is True

    def test_default_rate_limit_config(self):
        """Test default rate limit configuration."""
        provider = MockProvider({})

        assert provider.rate_limit_config.enabled is True
        assert provider.rate_limit_config.requests_per_minute == 50

    @pytest.mark.asyncio
    async def test_generate_with_retry_success(self):
        """Test generate_with_retry succeeds on first try."""
        provider = MockProvider({})
        request = LLMRequest(prompt="Test")

        response = await provider.generate_with_retry(request)

        assert response.content == "Mock response"
        assert provider.generate_call_count == 1

    @pytest.mark.asyncio
    async def test_generate_with_retry_transient_error(self):
        """Test generate_with_retry retries on transient errors."""
        provider = MockProvider({})
        provider.generate = AsyncMock(
            side_effect=[
                RateLimitError("Rate limited", retry_after=0.01),
                LLMResponse(
                    content="Success",
                    model="test",
                    tokens_used=10,
                    prompt_tokens=5,
                    completion_tokens=5,
                    finish_reason="stop",
                    provider="mock"
                )
            ]
        )

        request = LLMRequest(prompt="Test")
        response = await provider.generate_with_retry(request)

        assert response.content == "Success"
        assert provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_with_retry_max_attempts_exceeded(self):
        """Test generate_with_retry fails after max attempts."""
        config = {"retry": {"max_attempts": 2, "initial_delay": 0.001}}
        provider = MockProvider(config)

        # Always fail with retryable error
        provider.generate = AsyncMock(
            side_effect=APIError("Server error", status_code=500)
        )

        request = LLMRequest(prompt="Test")
        with pytest.raises(MaxRetriesExceededError) as exc_info:
            await provider.generate_with_retry(request)

        assert exc_info.value.attempts == 2
        assert provider.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_with_retry_non_retryable_error(self):
        """Test generate_with_retry doesn't retry non-retryable errors."""
        provider = MockProvider({})

        # Fail with non-retryable error
        provider.generate = AsyncMock(
            side_effect=ValueError("Invalid input")
        )

        request = LLMRequest(prompt="Test")
        with pytest.raises(ValueError):
            await provider.generate_with_retry(request)

        # Should not retry
        assert provider.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limiting_applied(self):
        """Test that rate limiting is applied when enabled."""
        config = {
            "rate_limit": {
                "enabled": True,
                "requests_per_minute": 60
            }
        }
        provider = MockProvider(config)

        # Mock the rate limiter
        provider.rate_limiter.acquire = AsyncMock()

        request = LLMRequest(prompt="Test")
        await provider.generate_with_retry(request)

        # Verify rate limiter was called
        provider.rate_limiter.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting_disabled(self):
        """Test that rate limiting is not applied when disabled."""
        config = {
            "rate_limit": {
                "enabled": False
            }
        }
        provider = MockProvider(config)

        # Mock the rate limiter
        provider.rate_limiter.acquire = AsyncMock()

        request = LLMRequest(prompt="Test")
        await provider.generate_with_retry(request)

        # Verify rate limiter was not called
        provider.rate_limiter.acquire.assert_not_called()

    @pytest.mark.asyncio
    async def test_logging_on_success(self):
        """Test that successful requests are logged."""
        provider = MockProvider({})

        with patch('models.llm_judge.base.logger') as mock_logger:
            request = LLMRequest(prompt="Test prompt")
            await provider.generate_with_retry(request)

            # Check that info logging occurred
            mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_logging_on_error(self):
        """Test that errors are logged."""
        provider = MockProvider({})
        provider.generate = AsyncMock(side_effect=ValueError("Test error"))

        with patch('models.llm_judge.base.logger') as mock_logger:
            request = LLMRequest(prompt="Test")

            with pytest.raises(ValueError):
                await provider.generate_with_retry(request)

            # Check that error logging occurred
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_is_retryable_error(self):
        """Test error classification for retryability."""
        provider = MockProvider({})

        # Retryable errors
        assert provider._is_retryable_error(RateLimitError("Rate limited"))
        assert provider._is_retryable_error(APIError("Server error", status_code=500))
        assert provider._is_retryable_error(RuntimeError("Connection failed"))

        # Non-retryable errors
        assert not provider._is_retryable_error(ValueError("Invalid input"))
        assert not provider._is_retryable_error(TypeError("Wrong type"))
        assert not provider._is_retryable_error(APIError("Bad request", status_code=400))


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_burst(self):
        """Test rate limiter allows burst up to limit."""
        limiter = RateLimiter(requests_per_minute=60)

        # Should allow rapid succession up to burst limit
        for _ in range(5):
            allowed = await limiter.acquire()
            assert allowed is True

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_when_exhausted(self):
        """Test rate limiter blocks when tokens exhausted."""
        # Very low rate to test blocking
        limiter = RateLimiter(requests_per_minute=60)
        limiter.tokens = 0  # Exhaust tokens
        limiter.last_update = asyncio.get_event_loop().time()

        # Should block until tokens refill
        start_time = asyncio.get_event_loop().time()
        allowed = await limiter.acquire()
        end_time = asyncio.get_event_loop().time()

        assert allowed is True
        # Should have waited some time
        assert end_time > start_time

    @pytest.mark.asyncio
    async def test_rate_limiter_refills_tokens(self):
        """Test rate limiter refills tokens over time."""
        limiter = RateLimiter(requests_per_minute=60)
        initial_tokens = limiter.tokens

        # Use some tokens
        await limiter.acquire()
        await limiter.acquire()
        assert limiter.tokens < initial_tokens

        # Wait for refill
        await asyncio.sleep(0.1)
        await limiter.acquire()

        # Tokens should have refilled partially
        assert limiter.tokens > initial_tokens - 3

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization with different rates."""
        limiter = RateLimiter(requests_per_minute=120)
        assert limiter.rate == 2.0  # 120 per minute = 2 per second
        assert limiter.capacity == 120
        assert limiter.tokens == 120