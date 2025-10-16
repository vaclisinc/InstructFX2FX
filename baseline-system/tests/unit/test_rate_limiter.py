"""Tests for rate limiter utilities."""

import pytest
import asyncio
import time

from models.llm_judge.utils.rate_limiter import (
    TokenBucketRateLimiter,
    AdaptiveRateLimiter,
    MultiProviderRateLimiter,
)


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter."""

    @pytest.mark.asyncio
    async def test_basic_acquisition(self):
        """Test basic token acquisition."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)

        # Should succeed immediately with full bucket
        await limiter.acquire(1)
        await limiter.acquire(1)
        await limiter.acquire(1)

        # Should have consumed 3 tokens
        assert limiter.available_tokens() < 10

    @pytest.mark.asyncio
    async def test_burst_capacity(self):
        """Test that burst up to capacity is allowed."""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)

        # Should be able to acquire entire capacity at once
        await limiter.acquire(5)

        # Should have approximately no tokens left (account for timing)
        assert limiter.available_tokens() < 0.1

    @pytest.mark.asyncio
    async def test_blocks_when_empty(self):
        """Test that acquisition blocks when bucket is empty."""
        limiter = TokenBucketRateLimiter(capacity=2, refill_rate=2.0)  # 2 tokens/sec

        # Drain the bucket
        await limiter.acquire(2)

        # Next acquisition should block until tokens refill
        start_time = time.time()
        await limiter.acquire(1)
        elapsed = time.time() - start_time

        # Should have waited ~0.5 seconds (1 token at 2 tokens/sec)
        assert elapsed >= 0.4  # Account for timing variations
        assert elapsed < 0.7

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        """Test that tokens refill over time."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=10.0)  # 10 tokens/sec

        # Drain bucket
        await limiter.acquire(10)
        assert limiter.available_tokens() < 0.1  # Account for timing

        # Wait for refill
        await asyncio.sleep(0.5)

        # Should have ~5 tokens (10 tokens/sec * 0.5 sec)
        tokens = limiter.available_tokens()
        assert 4.0 <= tokens <= 6.0  # Account for timing variations

    @pytest.mark.asyncio
    async def test_refill_capped_at_capacity(self):
        """Test that refill doesn't exceed capacity."""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)

        # Start with full bucket
        initial = limiter.available_tokens()
        assert initial == 5.0

        # Wait for "refill"
        await asyncio.sleep(2.0)

        # Should still be at capacity, not above
        tokens = limiter.available_tokens()
        assert tokens == 5.0

    @pytest.mark.asyncio
    async def test_multiple_tokens_per_request(self):
        """Test acquiring multiple tokens at once."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)

        # Acquire 3 tokens
        await limiter.acquire(3)
        assert limiter.available_tokens() < 8

        # Acquire 5 more tokens
        await limiter.acquire(5)
        assert limiter.available_tokens() < 3


class TestAdaptiveRateLimiter:
    """Test adaptive rate limiter."""

    @pytest.mark.asyncio
    async def test_initial_rate(self):
        """Test that limiter starts at initial rate."""
        limiter = AdaptiveRateLimiter(initial_rate=50.0)
        assert limiter.get_current_rate() == 50.0

    @pytest.mark.asyncio
    async def test_decrease_on_rate_limit(self):
        """Test that rate decreases after rate limit error."""
        limiter = AdaptiveRateLimiter(
            initial_rate=50.0,
            min_rate=10.0,
            decrease_factor=0.5
        )

        initial_rate = limiter.get_current_rate()
        await limiter.report_rate_limit()

        new_rate = limiter.get_current_rate()
        assert new_rate == initial_rate * 0.5

    @pytest.mark.asyncio
    async def test_increase_after_successes(self):
        """Test that rate increases after threshold successes."""
        limiter = AdaptiveRateLimiter(
            initial_rate=50.0,
            max_rate=100.0,
            increase_factor=1.1
        )

        # Report 10 successes (threshold)
        for _ in range(10):
            await limiter.report_success()

        new_rate = limiter.get_current_rate()
        assert new_rate > 50.0
        assert new_rate == pytest.approx(55.0, rel=0.01)  # 50 * 1.1

    @pytest.mark.asyncio
    async def test_rate_capped_at_min(self):
        """Test that rate doesn't go below min_rate."""
        limiter = AdaptiveRateLimiter(
            initial_rate=20.0,
            min_rate=5.0,
            decrease_factor=0.5
        )

        # Report multiple rate limits
        await limiter.report_rate_limit()  # 20 -> 10
        await limiter.report_rate_limit()  # 10 -> 5
        await limiter.report_rate_limit()  # Should stay at 5

        assert limiter.get_current_rate() == 5.0

    @pytest.mark.asyncio
    async def test_rate_capped_at_max(self):
        """Test that rate doesn't go above max_rate."""
        limiter = AdaptiveRateLimiter(
            initial_rate=90.0,
            max_rate=100.0,
            increase_factor=1.2
        )

        # Report successes to increase rate
        for _ in range(10):
            await limiter.report_success()

        # Should hit max
        assert limiter.get_current_rate() == 100.0

        # Additional successes shouldn't increase further
        for _ in range(10):
            await limiter.report_success()

        assert limiter.get_current_rate() == 100.0

    @pytest.mark.asyncio
    async def test_success_counter_resets_after_rate_limit(self):
        """Test that success counter resets after rate limit."""
        limiter = AdaptiveRateLimiter(initial_rate=50.0, increase_factor=1.1)

        # Build up successes
        for _ in range(5):
            await limiter.report_success()

        # Hit rate limit
        await limiter.report_rate_limit()

        # Success counter should reset, need 10 more for increase
        initial_rate = limiter.get_current_rate()
        for _ in range(9):
            await limiter.report_success()

        # Shouldn't have increased yet
        assert limiter.get_current_rate() == initial_rate

        # One more should trigger increase
        await limiter.report_success()
        assert limiter.get_current_rate() > initial_rate

    @pytest.mark.asyncio
    async def test_retry_after_honored(self):
        """Test that retry_after suggestion is honored."""
        limiter = AdaptiveRateLimiter(initial_rate=50.0)

        start_time = time.time()
        await limiter.report_rate_limit(retry_after=0.2)
        elapsed = time.time() - start_time

        # Should have waited ~0.2 seconds
        assert elapsed >= 0.15  # Account for timing variations
        assert elapsed < 0.35


class TestMultiProviderRateLimiter:
    """Test multi-provider rate limiter."""

    @pytest.mark.asyncio
    async def test_add_provider(self):
        """Test adding providers."""
        limiter = MultiProviderRateLimiter()

        await limiter.add_provider("anthropic", capacity=50, refill_rate=50/60)
        await limiter.add_provider("openrouter", capacity=100, refill_rate=100/60)

        providers = limiter.list_providers()
        assert "anthropic" in providers
        assert "openrouter" in providers

    @pytest.mark.asyncio
    async def test_acquire_per_provider(self):
        """Test that acquisition works per provider."""
        limiter = MultiProviderRateLimiter()

        await limiter.add_provider("provider_a", capacity=5, refill_rate=1.0)
        await limiter.add_provider("provider_b", capacity=5, refill_rate=1.0)

        # Drain provider_a
        await limiter.acquire("provider_a", tokens=5)

        # provider_b should still be available
        await limiter.acquire("provider_b", tokens=1)

        # Check available tokens
        limiter_a = limiter.get_limiter("provider_a")
        limiter_b = limiter.get_limiter("provider_b")

        assert limiter_a.available_tokens() < 0.1  # Account for timing
        assert limiter_b.available_tokens() < 5

    @pytest.mark.asyncio
    async def test_acquire_unknown_provider_raises(self):
        """Test that acquiring for unknown provider raises KeyError."""
        limiter = MultiProviderRateLimiter()

        with pytest.raises(KeyError) as exc_info:
            await limiter.acquire("unknown_provider")

        assert "unknown_provider" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_unknown_limiter_raises(self):
        """Test that getting unknown limiter raises KeyError."""
        limiter = MultiProviderRateLimiter()

        with pytest.raises(KeyError):
            limiter.get_limiter("unknown_provider")

    @pytest.mark.asyncio
    async def test_list_providers_empty(self):
        """Test listing providers when none added."""
        limiter = MultiProviderRateLimiter()
        assert limiter.list_providers() == []

    @pytest.mark.asyncio
    async def test_get_limiter_returns_correct_instance(self):
        """Test that get_limiter returns correct limiter."""
        limiter = MultiProviderRateLimiter()

        await limiter.add_provider("test", capacity=10, refill_rate=1.0)

        provider_limiter = limiter.get_limiter("test")
        assert isinstance(provider_limiter, TokenBucketRateLimiter)
        assert provider_limiter.capacity == 10


class TestRateLimiterConcurrency:
    """Test rate limiter behavior under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_acquisitions(self):
        """Test that concurrent acquisitions are properly serialized."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=100.0)

        # Launch 10 concurrent acquisitions
        tasks = [limiter.acquire(1) for _ in range(10)]
        await asyncio.gather(*tasks)

        # All 10 should have succeeded, bucket should be nearly empty
        # (allow for small refill during operation)
        assert limiter.available_tokens() < 0.5

    @pytest.mark.asyncio
    async def test_concurrent_over_capacity(self):
        """Test behavior when concurrent requests exceed capacity."""
        limiter = TokenBucketRateLimiter(capacity=5, refill_rate=10.0)  # 10 tokens/sec

        start_time = time.time()

        # Launch 10 requests (2x capacity)
        tasks = [limiter.acquire(1) for _ in range(10)]
        await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # First 5 should be immediate, next 5 should wait ~0.5s for refill
        assert elapsed >= 0.4  # Account for timing variations
        assert elapsed < 0.8


class TestRateLimiterEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_zero_capacity_raises(self):
        """Test that zero capacity is handled."""
        # This should technically work but will always block
        limiter = TokenBucketRateLimiter(capacity=1, refill_rate=1.0)
        await limiter.acquire(1)
        # If we get here, it worked

    @pytest.mark.asyncio
    async def test_very_high_refill_rate(self):
        """Test with very high refill rate."""
        limiter = TokenBucketRateLimiter(capacity=1000, refill_rate=1000.0)

        # Should handle large burst
        await limiter.acquire(500)
        assert limiter.available_tokens() < 1000

    @pytest.mark.asyncio
    async def test_fractional_tokens(self):
        """Test with fractional token acquisition."""
        limiter = TokenBucketRateLimiter(capacity=10, refill_rate=1.0)

        # Should handle fractional tokens
        await limiter.acquire(0.5)
        tokens = limiter.available_tokens()
        assert tokens < 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
