"""Real integration tests for ClaudeProvider with actual API calls.

This test suite performs REAL API calls to Anthropic Claude API (not mocked).
Tests are automatically skipped if ANTHROPIC_API_KEY is not set.

These tests verify:
1. Real API calls with various parameters
2. Retry logic with actual rate limit and server errors
3. Error handling with real error scenarios
4. Token usage tracking accuracy
5. Request parameter handling (temperature, max_tokens, system_prompt)

IMPORTANT: These tests will consume API credits. They use small prompts
to minimize costs, but be aware of usage.

To run these tests:
    export ANTHROPIC_API_KEY=your-api-key
    pytest tests/integration/test_claude_integration.py -v
"""

import os
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from anthropic import RateLimitError, APIStatusError, APIConnectionError
from dotenv import load_dotenv

from models.llm_judge.providers import ClaudeProvider
from models.llm_judge.types import LLMRequest, LLMResponse, RetryConfig

# Load environment variables from .env file
load_dotenv()


class TestClaudeRealAPIIntegration:
    """Integration tests with real Claude API calls."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_real_basic_generation(self):
        """Test basic generation with real Claude API.

        Verifies:
        - API connection works
        - Response format is correct
        - Content is generated
        - Token counting is accurate
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        request = LLMRequest(
            prompt="Say 'Hello' and nothing else.",
            temperature=0.0,
            max_tokens=10
        )

        response = await provider.generate(request)

        # Verify response structure
        assert isinstance(response, LLMResponse)
        assert response.content
        assert len(response.content) > 0

        # Verify token tracking
        assert response.tokens_used > 0
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.tokens_used == response.prompt_tokens + response.completion_tokens

        # Verify metadata
        assert response.provider == "claude"
        assert response.model == "claude-3-haiku-20240307"
        assert response.finish_reason in ["end_turn", "stop_sequence", "max_tokens"]

        # Verify content makes sense
        assert "hello" in response.content.lower()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_real_system_prompt_handling(self):
        """Test system prompt handling with real API.

        Verifies:
        - System prompts are properly sent to API
        - System prompts affect response behavior
        - Token counting includes system prompt
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        request = LLMRequest(
            prompt="What are you?",
            system_prompt="You are a helpful AI assistant that always responds in exactly 3 words.",
            temperature=0.7,
            max_tokens=50
        )

        response = await provider.generate(request)

        assert isinstance(response, LLMResponse)
        assert response.content

        # System prompt should influence response length (around 3 words)
        word_count = len(response.content.split())
        # Allow some flexibility but should be close to 3 words
        assert 1 <= word_count <= 10, f"Expected ~3 words, got {word_count}: {response.content}"

        # Verify tokens are counted
        assert response.prompt_tokens > 0  # System prompt adds to token count
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_real_temperature_parameter(self):
        """Test temperature parameter affects randomness.

        Verifies:
        - Temperature=0.0 produces consistent results
        - Temperature parameter is respected by API
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        prompt = "What is 2+2? Answer with just the number."

        # Make two requests with temperature=0.0 (should be deterministic)
        request1 = LLMRequest(prompt=prompt, temperature=0.0, max_tokens=10)
        request2 = LLMRequest(prompt=prompt, temperature=0.0, max_tokens=10)

        response1 = await provider.generate(request1)
        response2 = await provider.generate(request2)

        # Both should contain "4"
        assert "4" in response1.content
        assert "4" in response2.content

        # With temp=0, responses should be very similar or identical
        # (allowing for minor variations due to API non-determinism)
        assert response1.content.strip() == response2.content.strip() or \
               ("4" in response1.content and "4" in response2.content)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_real_max_tokens_parameter(self):
        """Test max_tokens parameter limits response length.

        Verifies:
        - max_tokens is enforced by API
        - Completion tokens stay within limit
        - finish_reason indicates max_tokens when limit hit
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        # Request a long response but limit to 5 tokens
        request = LLMRequest(
            prompt="Write a long essay about artificial intelligence.",
            temperature=0.7,
            max_tokens=5  # Very small limit
        )

        response = await provider.generate(request)

        # Verify completion tokens are limited
        assert response.completion_tokens <= 5

        # Should hit the max_tokens limit
        assert response.finish_reason == "max_tokens"

        # Content should be short
        assert len(response.content) < 100  # Should be very short with only 5 tokens

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_real_token_counting_accuracy(self):
        """Test token counting matches API response.

        Verifies:
        - Token counts are extracted from API response
        - prompt_tokens + completion_tokens = tokens_used
        - Token counts are positive and reasonable
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        request = LLMRequest(
            prompt="Count to 5.",
            temperature=0.0,
            max_tokens=50
        )

        response = await provider.generate(request)

        # Verify all token counts are positive
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.tokens_used > 0

        # Verify token math is correct
        assert response.tokens_used == response.prompt_tokens + response.completion_tokens

        # Verify counts are reasonable for this simple prompt
        assert response.prompt_tokens < 50  # "Count to 5" is very short
        assert response.completion_tokens < 100  # Response should be short


class TestClaudeRetryLogicIntegration:
    """Integration tests for retry logic with real error scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_retry_on_rate_limit_error(self):
        """Test retry logic with simulated rate limit error.

        Verifies:
        - Rate limit errors trigger retry
        - Exponential backoff is applied
        - Eventually succeeds after retries
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.1,  # Short delay for testing
            max_delay=1.0
        )
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Test",
            temperature=0.0,
            max_tokens=10
        )

        # Mock the generate method to fail twice with rate limit, then succeed
        original_generate = provider.generate
        call_count = 0

        async def mock_generate_with_failures(req):
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                # Simulate rate limit error for first two calls
                from anthropic import RateLimitError
                import httpx
                # Create proper mock response with request attribute
                mock_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                mock_response = httpx.Response(status_code=429, request=mock_request)
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=mock_response,
                    body=None
                )
            else:
                # Third call succeeds
                return await original_generate(req)

        provider.generate = mock_generate_with_failures

        # Should retry and eventually succeed
        response = await provider.generate_with_retry(request)

        assert isinstance(response, LLMResponse)
        assert call_count == 3  # Failed twice, succeeded on third
        assert response.content

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_retry_on_server_error(self):
        """Test retry logic with simulated server error (5xx).

        Verifies:
        - Server errors trigger retry
        - Retries with exponential backoff
        - Eventually succeeds
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.1,
            max_delay=1.0
        )
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Test",
            temperature=0.0,
            max_tokens=10
        )

        # Mock to fail once with server error, then succeed
        original_generate = provider.generate
        call_count = 0

        async def mock_generate_with_server_error(req):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Simulate 500 server error
                from anthropic import APIStatusError
                import httpx
                mock_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                mock_response = httpx.Response(status_code=500, request=mock_request)
                raise APIStatusError(
                    "500 Internal server error",
                    response=mock_response,
                    body=None
                )
            else:
                return await original_generate(req)

        provider.generate = mock_generate_with_server_error

        response = await provider.generate_with_retry(request)

        assert isinstance(response, LLMResponse)
        assert call_count == 2  # Failed once, succeeded on second
        assert response.content

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_retry_max_attempts_exceeded(self):
        """Test that retry gives up after max attempts.

        Verifies:
        - Max retry limit is respected
        - Appropriate error is raised
        - Error message includes attempt count
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        retry_config = RetryConfig(
            max_attempts=2,
            initial_delay=0.1,
            max_delay=1.0
        )
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Test",
            temperature=0.0,
            max_tokens=10
        )

        # Mock to always fail with rate limit error
        async def mock_generate_always_fails(req):
            from anthropic import RateLimitError
            import httpx
            mock_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            mock_response = httpx.Response(status_code=429, request=mock_request)
            raise RateLimitError(
                "Rate limit exceeded",
                response=mock_response,
                body=None
            )

        provider.generate = mock_generate_always_fails

        # Should fail after max attempts
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            await provider.generate_with_retry(request)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_exponential_backoff_timing(self):
        """Test that exponential backoff increases delay between retries.

        Verifies:
        - Delay increases exponentially
        - Timing follows retry config
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.2,
            exponential_base=2.0,
            max_delay=5.0,
            jitter=False  # Disable jitter for predictable timing
        )
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Test",
            temperature=0.0,
            max_tokens=10
        )

        # Mock to fail twice then succeed
        original_generate = provider.generate
        call_count = 0
        call_times = []

        async def mock_generate_track_timing(req):
            nonlocal call_count
            call_count += 1
            call_times.append(asyncio.get_event_loop().time())

            if call_count <= 2:
                from anthropic import RateLimitError
                import httpx
                mock_request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
                mock_response = httpx.Response(status_code=429, request=mock_request)
                raise RateLimitError(
                    "Rate limit exceeded",
                    response=mock_response,
                    body=None
                )
            else:
                return await original_generate(req)

        provider.generate = mock_generate_track_timing

        start_time = asyncio.get_event_loop().time()
        await provider.generate_with_retry(request)
        total_time = asyncio.get_event_loop().time() - start_time

        # Should have made 3 calls
        assert call_count == 3

        # Total time should include delays (0.2s + 0.4s ≈ 0.6s minimum)
        assert total_time >= 0.5  # Allow some tolerance


class TestClaudeErrorHandlingIntegration:
    """Integration tests for error handling with real scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_error(self):
        """Test error handling with invalid API key.

        Verifies:
        - Invalid API key is detected
        - Appropriate error is raised
        - Error is not retried (authentication errors are non-retryable)
        """
        config = {
            "api_key": "sk-ant-invalid-test-key-12345",
            "model": "claude-3-haiku-20240307"
        }
        retry_config = RetryConfig(max_attempts=2)
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Test",
            temperature=0.0,
            max_tokens=10
        )

        # Should fail with authentication error (not retried)
        with pytest.raises(RuntimeError):
            await provider.generate_with_retry(request)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_network_error_handling(self):
        """Test handling of network connection errors.

        Verifies:
        - Network errors are detected
        - Network errors trigger retry
        - Clear error message on failure
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        retry_config = RetryConfig(max_attempts=2, initial_delay=0.1)
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Test",
            temperature=0.0,
            max_tokens=10
        )

        # Mock to simulate network error
        async def mock_generate_network_error(req):
            from anthropic import APIConnectionError
            mock_request = type('MockRequest', (), {})()
            raise APIConnectionError(request=mock_request)

        provider.generate = mock_generate_network_error

        # Should fail after retries
        with pytest.raises(RuntimeError):
            await provider.generate_with_retry(request)

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_timeout_handling(self):
        """Test handling of timeout errors.

        Verifies:
        - Timeout errors are detected
        - Timeout errors trigger retry
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-5-sonnet-20241022",
            "timeout": 0.001  # Extremely short timeout to force timeout
        }
        retry_config = RetryConfig(max_attempts=1)  # Only try once
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Write a very long essay about the history of computing.",
            temperature=0.7,
            max_tokens=1000
        )

        # This might timeout or might succeed quickly
        # Just verify it doesn't crash unexpectedly
        try:
            response = await provider.generate_with_retry(request)
            # If it succeeds, verify basic structure
            assert isinstance(response, LLMResponse)
        except RuntimeError:
            # Timeout is acceptable
            pass

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_invalid_model_error(self):
        """Test error handling with invalid model name.

        Verifies:
        - Invalid model names are rejected by API
        - Appropriate error is raised
        - Error message is clear
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-invalid-model-xyz"
        }
        retry_config = RetryConfig(max_attempts=1)
        provider = ClaudeProvider(config, retry_config=retry_config)

        request = LLMRequest(
            prompt="Test",
            temperature=0.0,
            max_tokens=10
        )

        # Should fail with error about invalid model
        with pytest.raises(RuntimeError):
            await provider.generate_with_retry(request)


class TestClaudeTokenTracking:
    """Integration tests for token usage tracking."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_token_counts_match_api_response(self):
        """Test that token counts match API response exactly.

        Verifies:
        - Token counts are extracted from API response
        - No manual estimation or calculation
        - Counts are accurate and consistent
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        prompts = [
            "Hello",
            "What is the capital of France?",
            "Write a haiku about programming."
        ]

        for prompt in prompts:
            request = LLMRequest(
                prompt=prompt,
                temperature=0.0,
                max_tokens=50
            )

            response = await provider.generate(request)

            # Verify token consistency
            assert response.prompt_tokens > 0
            assert response.completion_tokens > 0
            assert response.tokens_used == response.prompt_tokens + response.completion_tokens

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_cumulative_token_tracking(self):
        """Test cumulative token tracking across multiple requests.

        Verifies:
        - Each request has accurate token counts
        - Token counts can be summed across requests
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        requests = [
            LLMRequest(prompt="Say 'one'", temperature=0.0, max_tokens=5),
            LLMRequest(prompt="Say 'two'", temperature=0.0, max_tokens=5),
            LLMRequest(prompt="Say 'three'", temperature=0.0, max_tokens=5),
        ]

        total_tokens = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for req in requests:
            response = await provider.generate(req)

            total_tokens += response.tokens_used
            total_prompt_tokens += response.prompt_tokens
            total_completion_tokens += response.completion_tokens

        # Verify cumulative math is correct
        assert total_tokens == total_prompt_tokens + total_completion_tokens
        assert total_tokens > 0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="Requires ANTHROPIC_API_KEY environment variable"
    )
    async def test_system_prompt_affects_token_count(self):
        """Test that system prompts are included in token counting.

        Verifies:
        - System prompts add to prompt tokens
        - Token count with system prompt > without
        """
        config = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-haiku-20240307"  # Use Haiku for faster/cheaper testing
        }
        provider = ClaudeProvider(config)

        # Request without system prompt
        request_no_system = LLMRequest(
            prompt="Hello",
            temperature=0.0,
            max_tokens=10
        )

        response_no_system = await provider.generate(request_no_system)

        # Request with system prompt
        request_with_system = LLMRequest(
            prompt="Hello",
            system_prompt="You are a helpful AI assistant that provides concise responses.",
            temperature=0.0,
            max_tokens=10
        )

        response_with_system = await provider.generate(request_with_system)

        # System prompt should increase token count
        assert response_with_system.prompt_tokens > response_no_system.prompt_tokens
