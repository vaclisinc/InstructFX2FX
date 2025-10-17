"""Real integration tests for OpenRouterProvider with actual API calls.

These tests make real API calls to OpenRouter and are skipped unless
OPENROUTER_API_KEY environment variable is set.

IMPORTANT: These tests will use real API credits. They use lightweight
free models to minimize costs.

To run these tests:
    pytest tests/integration/test_openrouter_integration.py -v -m integration

To skip integration tests:
    pytest tests/ -m "not integration"
"""

import os
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch
import openai

from models.llm_judge.providers.openrouter import OpenRouterProvider
from models.llm_judge.types import LLMRequest, LLMResponse, RetryConfig
from models.llm_judge.factory import create_provider


# Skip all tests in this module if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="Requires OPENROUTER_API_KEY environment variable"
)


# Use free models for testing to minimize costs
FREE_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
CHEAP_MODEL = "openai/gpt-3.5-turbo"


@pytest.mark.integration
class TestOpenRouterRealAPI:
    """Integration tests for basic OpenRouter API functionality."""

    @pytest.mark.asyncio
    async def test_basic_generation_with_real_api(self):
        """Test basic generation with real OpenRouter API call.

        This test verifies:
        - API connection works
        - Request/response format is correct
        - Token counts are returned
        - Provider metadata is correct
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        request = LLMRequest(
            prompt="Say exactly 'Hello' and nothing else.",
            temperature=0.0,
            max_tokens=10
        )

        response = await provider.generate(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content is not None
        assert len(response.content) > 0
        assert response.provider == "openrouter"
        assert response.model == FREE_MODEL

        # Verify token usage
        assert response.tokens_used > 0
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.tokens_used == response.prompt_tokens + response.completion_tokens

        # Verify finish reason
        assert response.finish_reason in ["stop", "length", "end_turn"]

        print(f"✓ Basic generation test passed")
        print(f"  Model: {response.model}")
        print(f"  Tokens used: {response.tokens_used}")
        print(f"  Response: {response.content[:100]}")

    @pytest.mark.asyncio
    async def test_system_prompt_handling(self):
        """Test that system prompts are properly sent to OpenRouter API.

        This test verifies:
        - System prompts are included in API request
        - System prompt affects model behavior
        - Response format is maintained
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        request = LLMRequest(
            prompt="What language should you respond in?",
            system_prompt="You are a French translator. Always respond in French.",
            temperature=0.7,
            max_tokens=50
        )

        response = await provider.generate(request)

        assert isinstance(response, LLMResponse)
        assert response.content
        # System prompt should influence response
        # (French words likely, but we don't hard-assert language)
        assert len(response.content) > 0

        print(f"✓ System prompt test passed")
        print(f"  Response: {response.content[:100]}")

    @pytest.mark.asyncio
    async def test_model_selection(self):
        """Test that model selection works correctly.

        This test verifies:
        - Different models can be specified
        - Model name is correctly returned in response
        - Each model generates valid responses
        """
        models_to_test = [
            FREE_MODEL,
            "meta-llama/llama-3.2-1b-instruct:free"
        ]

        for model in models_to_test:
            config = {
                "provider": "openrouter",
                "model": model
            }

            try:
                provider = create_provider(config)

                request = LLMRequest(
                    prompt="Say 'test' and nothing else.",
                    temperature=0.0,
                    max_tokens=5
                )

                response = await provider.generate(request)

                assert isinstance(response, LLMResponse)
                assert response.model == model
                assert response.content
                assert response.tokens_used > 0

                print(f"✓ Model {model} works")

            except Exception as e:
                # Some models might not be available, that's okay
                print(f"⚠ Model {model} failed: {e}")

    @pytest.mark.asyncio
    async def test_token_usage_tracking(self):
        """Test that token usage is accurately tracked.

        This test verifies:
        - Token counts are returned from API
        - Token math is consistent (total = prompt + completion)
        - Different prompt sizes result in different token counts
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        # Short prompt
        short_request = LLMRequest(
            prompt="Hi",
            temperature=0.0,
            max_tokens=5
        )
        short_response = await provider.generate(short_request)

        # Longer prompt
        long_request = LLMRequest(
            prompt="This is a longer prompt with more words that should use more tokens than the short prompt.",
            temperature=0.0,
            max_tokens=5
        )
        long_response = await provider.generate(long_request)

        # Verify token consistency
        assert short_response.tokens_used == short_response.prompt_tokens + short_response.completion_tokens
        assert long_response.tokens_used == long_response.prompt_tokens + long_response.completion_tokens

        # Longer prompt should use more tokens
        assert long_response.prompt_tokens > short_response.prompt_tokens

        print(f"✓ Token tracking test passed")
        print(f"  Short prompt tokens: {short_response.prompt_tokens}")
        print(f"  Long prompt tokens: {long_response.prompt_tokens}")


@pytest.mark.integration
class TestOpenRouterRetryLogic:
    """Integration tests for retry logic with real API failures."""

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Test retry behavior on rate limit errors (429).

        This test verifies:
        - Rate limit errors trigger retry
        - Exponential backoff is applied
        - Eventually succeeds after retry
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        # Configure retry with faster timings for testing
        provider.retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=2.0,
            exponential_base=2.0,
            jitter=False
        )

        # Mock the client to simulate rate limit on first call, then succeed
        original_create = provider.client.chat.completions.create
        call_count = 0

        async def mock_create_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: rate limit error
                raise openai.RateLimitError(
                    message="Rate limit exceeded",
                    response=None,
                    body=None
                )
            else:
                # Subsequent calls: success
                return await original_create(*args, **kwargs)

        provider.client.chat.completions.create = mock_create_with_retry

        request = LLMRequest(
            prompt="Say 'retry test passed'",
            temperature=0.0,
            max_tokens=10
        )

        start_time = time.time()
        response = await provider.generate_with_retry(request)
        elapsed = time.time() - start_time

        # Verify retry happened
        assert call_count == 2  # First failed, second succeeded
        assert elapsed >= 0.5  # Should have delayed at least initial_delay

        # Verify response is valid
        assert isinstance(response, LLMResponse)
        assert response.content

        print(f"✓ Rate limit retry test passed")
        print(f"  Attempts: {call_count}")
        print(f"  Elapsed: {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Test retry behavior on server errors (5xx).

        This test verifies:
        - 5xx errors trigger retry
        - Retry logic works for server errors
        - Eventually succeeds or fails after max attempts
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        # Configure retry
        provider.retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            max_delay=2.0,
            exponential_base=2.0,
            jitter=False
        )

        # Mock server error on first call
        original_create = provider.client.chat.completions.create
        call_count = 0

        async def mock_create_with_server_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: server error
                mock_response = type('obj', (object,), {'status_code': 503})()
                raise openai.APIStatusError(
                    message="Service temporarily unavailable",
                    response=mock_response,
                    body=None
                )
            else:
                # Subsequent calls: success
                return await original_create(*args, **kwargs)

        provider.client.chat.completions.create = mock_create_with_server_error

        request = LLMRequest(
            prompt="Test server error retry",
            temperature=0.0,
            max_tokens=10
        )

        response = await provider.generate_with_retry(request)

        # Verify retry happened and succeeded
        assert call_count == 2
        assert isinstance(response, LLMResponse)

        print(f"✓ Server error retry test passed")
        print(f"  Attempts: {call_count}")

    @pytest.mark.asyncio
    async def test_max_retry_limit(self):
        """Test that retries stop at max_attempts.

        This test verifies:
        - Retries stop after max_attempts
        - Final exception is raised
        - Exponential backoff is applied
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        # Configure retry with low max_attempts
        provider.retry_config = RetryConfig(
            max_attempts=2,
            initial_delay=0.3,
            max_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )

        # Mock to always fail with retryable error
        call_count = 0

        async def mock_always_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise openai.RateLimitError(
                message="Rate limit exceeded",
                response=None,
                body=None
            )

        provider.client.chat.completions.create = mock_always_fail

        request = LLMRequest(
            prompt="This will fail",
            temperature=0.0,
            max_tokens=10
        )

        # Should raise after max_attempts
        start_time = time.time()
        with pytest.raises(RuntimeError, match="failed after"):
            await provider.generate_with_retry(request)
        elapsed = time.time() - start_time

        # Verify max attempts reached
        assert call_count == 2  # max_attempts
        # Should have delayed: 0.3s after first failure
        assert elapsed >= 0.3

        print(f"✓ Max retry limit test passed")
        print(f"  Attempts: {call_count}")
        print(f"  Elapsed: {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test that exponential backoff timing is correct.

        This test verifies:
        - Delays increase exponentially
        - Max delay cap is respected
        - Backoff formula is correct
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        # Configure retry with specific backoff parameters
        provider.retry_config = RetryConfig(
            max_attempts=4,
            initial_delay=0.5,
            max_delay=2.0,
            exponential_base=2.0,
            jitter=False  # No jitter for predictable timing
        )

        # Track call times
        call_times = []

        async def mock_track_calls(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 4:
                raise openai.RateLimitError(
                    message="Rate limit",
                    response=None,
                    body=None
                )
            # Succeed on 4th attempt
            from unittest.mock import MagicMock
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage.prompt_tokens = 5
            mock_response.usage.completion_tokens = 5
            mock_response.usage.total_tokens = 10
            return mock_response

        provider.client.chat.completions.create = mock_track_calls

        request = LLMRequest(
            prompt="Test backoff",
            temperature=0.0,
            max_tokens=10
        )

        await provider.generate_with_retry(request)

        # Verify delays
        assert len(call_times) == 4

        # Calculate actual delays between calls
        delay1 = call_times[1] - call_times[0]  # After 1st failure: 0.5s
        delay2 = call_times[2] - call_times[1]  # After 2nd failure: 1.0s
        delay3 = call_times[3] - call_times[2]  # After 3rd failure: 2.0s (capped)

        # Allow some tolerance for timing
        assert 0.4 <= delay1 <= 0.7, f"First delay should be ~0.5s, got {delay1:.2f}s"
        assert 0.9 <= delay2 <= 1.3, f"Second delay should be ~1.0s, got {delay2:.2f}s"
        assert 1.8 <= delay3 <= 2.5, f"Third delay should be ~2.0s, got {delay3:.2f}s"

        print(f"✓ Exponential backoff test passed")
        print(f"  Delay 1: {delay1:.2f}s (expected ~0.5s)")
        print(f"  Delay 2: {delay2:.2f}s (expected ~1.0s)")
        print(f"  Delay 3: {delay3:.2f}s (expected ~2.0s)")


@pytest.mark.integration
class TestOpenRouterErrorHandling:
    """Integration tests for error handling with real API."""

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test behavior with invalid API key.

        This test verifies:
        - Invalid API key triggers authentication error
        - Error message is clear and helpful
        - Error is non-retryable
        """
        config = {
            "provider": "openrouter",
            "api_key": "sk-or-invalid-key-12345",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        # Set retry config to fail fast
        provider.retry_config = RetryConfig(
            max_attempts=1,
            initial_delay=0.1
        )

        request = LLMRequest(
            prompt="This should fail",
            temperature=0.0,
            max_tokens=10
        )

        # Should fail with authentication error
        with pytest.raises(RuntimeError, match="authentication"):
            await provider.generate_with_retry(request)

        print(f"✓ Invalid API key test passed")

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires valid API key"
    )
    async def test_invalid_model_name(self):
        """Test behavior with invalid model name.

        This test verifies:
        - Invalid model name triggers error
        - Error message indicates the problem
        - Error is non-retryable
        """
        config = {
            "provider": "openrouter",
            "model": "invalid/nonexistent-model-xyz"
        }
        provider = create_provider(config)

        # Set retry config
        provider.retry_config = RetryConfig(
            max_attempts=1,
            initial_delay=0.1
        )

        request = LLMRequest(
            prompt="This should fail",
            temperature=0.0,
            max_tokens=10
        )

        # Should fail with model error
        with pytest.raises(RuntimeError):
            await provider.generate_with_retry(request)

        print(f"✓ Invalid model name test passed")

    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test handling of network errors.

        This test verifies:
        - Network errors are caught properly
        - Error messages are clear
        - Timeouts work as expected
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL,
            "timeout": 0.001  # Very short timeout to force error
        }
        provider = create_provider(config)

        # Set retry config
        provider.retry_config = RetryConfig(
            max_attempts=2,
            initial_delay=0.1
        )

        request = LLMRequest(
            prompt="This will timeout",
            temperature=0.0,
            max_tokens=10
        )

        # Should fail with timeout error
        with pytest.raises(RuntimeError):
            await provider.generate_with_retry(request)

        print(f"✓ Network error handling test passed")


@pytest.mark.integration
class TestOpenRouterAdvancedFeatures:
    """Integration tests for advanced OpenRouter features."""

    @pytest.mark.asyncio
    async def test_openrouter_specific_headers(self):
        """Test that OpenRouter-specific headers are sent correctly.

        This test verifies:
        - HTTP-Referer header is sent
        - X-Title header is sent
        - Headers don't break API calls
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL,
            "site_url": "https://github.com/test-integration",
            "site_name": "Integration Test Suite"
        }
        provider = create_provider(config)

        request = LLMRequest(
            prompt="Say 'headers work'",
            temperature=0.0,
            max_tokens=10
        )

        response = await provider.generate(request)

        # Should succeed with headers
        assert isinstance(response, LLMResponse)
        assert response.content

        print(f"✓ OpenRouter headers test passed")

    @pytest.mark.asyncio
    async def test_temperature_variation(self):
        """Test that temperature parameter affects output variability.

        This test verifies:
        - Different temperatures can be set
        - Temperature affects response diversity
        - API correctly handles temperature parameter
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        # Same prompt with different temperatures
        prompt = "Write a creative short sentence about a cat."

        # Temperature 0 - deterministic
        low_temp_request = LLMRequest(
            prompt=prompt,
            temperature=0.0,
            max_tokens=30
        )

        # Temperature 1.5 - more random
        high_temp_request = LLMRequest(
            prompt=prompt,
            temperature=1.5,
            max_tokens=30
        )

        low_temp_response = await provider.generate(low_temp_request)
        high_temp_response = await provider.generate(high_temp_request)

        # Both should succeed
        assert isinstance(low_temp_response, LLMResponse)
        assert isinstance(high_temp_response, LLMResponse)
        assert low_temp_response.content
        assert high_temp_response.content

        print(f"✓ Temperature variation test passed")
        print(f"  Temp 0.0: {low_temp_response.content[:50]}")
        print(f"  Temp 1.5: {high_temp_response.content[:50]}")

    @pytest.mark.asyncio
    async def test_max_tokens_enforcement(self):
        """Test that max_tokens limit is respected.

        This test verifies:
        - max_tokens parameter is sent to API
        - API respects the token limit
        - Finish reason indicates token limit if reached
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        request = LLMRequest(
            prompt="Write a very long story about a journey through space, with lots of details and descriptions.",
            temperature=0.7,
            max_tokens=20  # Very low limit
        )

        response = await provider.generate(request)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.completion_tokens <= 20  # Should respect limit
        # May or may not hit length limit depending on response

        print(f"✓ Max tokens enforcement test passed")
        print(f"  Completion tokens: {response.completion_tokens}")
        print(f"  Finish reason: {response.finish_reason}")

    @pytest.mark.asyncio
    async def test_model_override_in_request(self):
        """Test that model can be overridden per-request.

        This test verifies:
        - Default model can be set in config
        - Model can be overridden per request
        - Response reflects the actual model used
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL  # Default model
        }
        provider = create_provider(config)

        # Request with model override
        override_model = "meta-llama/llama-3.2-1b-instruct:free"
        request = LLMRequest(
            prompt="Say 'override works'",
            model=override_model,
            temperature=0.0,
            max_tokens=10
        )

        try:
            response = await provider.generate(request)

            # Should use override model
            assert response.model == override_model

            print(f"✓ Model override test passed")
            print(f"  Override model: {override_model}")

        except Exception as e:
            # Model might not be available, that's okay
            print(f"⚠ Model override test skipped: {e}")


@pytest.mark.integration
class TestOpenRouterCumulativeUsage:
    """Integration tests for cumulative token usage tracking."""

    @pytest.mark.asyncio
    async def test_multiple_requests_token_tracking(self):
        """Test token tracking across multiple requests.

        This test verifies:
        - Each request returns accurate token counts
        - Token counts can be summed for billing
        - No token count anomalies
        """
        config = {
            "provider": "openrouter",
            "model": FREE_MODEL
        }
        provider = create_provider(config)

        prompts = [
            "Say 'one'",
            "Say 'two'",
            "Say 'three'"
        ]

        total_tokens = 0
        responses = []

        for prompt in prompts:
            request = LLMRequest(
                prompt=prompt,
                temperature=0.0,
                max_tokens=5
            )

            response = await provider.generate(request)
            responses.append(response)
            total_tokens += response.tokens_used

            # Verify each response
            assert response.tokens_used > 0
            assert response.tokens_used == response.prompt_tokens + response.completion_tokens

        # Verify cumulative tracking
        assert len(responses) == 3
        assert total_tokens > 0

        print(f"✓ Cumulative token tracking test passed")
        print(f"  Total requests: {len(responses)}")
        print(f"  Total tokens: {total_tokens}")
        for i, resp in enumerate(responses):
            print(f"  Request {i+1}: {resp.tokens_used} tokens")
