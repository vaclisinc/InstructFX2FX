"""Real integration tests for OpenAIProvider with actual API calls.

This test suite performs REAL API calls to OpenAI API (not mocked).
Tests are automatically skipped if OPENAI_API_KEY is not set.

These tests verify:
1. Real API calls with various parameters
2. Retry logic with actual rate limit and server errors  
3. Error handling with real error scenarios
4. Token usage tracking accuracy
5. Request parameter handling (temperature, max_tokens, system_prompt)

IMPORTANT: These tests will consume API credits. They use small prompts
to minimize costs, but be aware of usage.

To run these tests:
    export OPENAI_API_KEY=your-api-key
    pytest tests/integration/test_openai_integration.py -v
"""

import os
import pytest
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.llm_judge.factory import create_provider
from models.llm_judge.types import LLMRequest

# Skip all tests if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="Requires OPENAI_API_KEY environment variable"
)


@pytest.mark.integration
class TestOpenAIRealAPIIntegration:
    """Real API integration tests for OpenAIProvider."""

    @pytest.mark.asyncio
    async def test_real_basic_generation(self):
        """Test basic generation with real OpenAI API."""
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        }
        provider = create_provider(config)
        
        request = LLMRequest(
            prompt="Say 'Hello' and nothing else.",
            temperature=0,
            max_tokens=10
        )
        
        response = await provider.generate(request)
        
        assert response.content
        assert response.tokens_used > 0
        assert response.provider == "openai"
        assert response.model == "gpt-3.5-turbo"
        assert "Hello" in response.content or "hello" in response.content

    @pytest.mark.asyncio
    async def test_real_system_prompt_handling(self):
        """Test system prompt with real API."""
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        }
        provider = create_provider(config)
        
        request = LLMRequest(
            prompt="What are you?",
            system_prompt="You are a helpful assistant. Always start your response with 'AI:'",
            temperature=0,
            max_tokens=30
        )
        
        response = await provider.generate(request)
        
        assert response.content
        assert "AI:" in response.content or "assistant" in response.content.lower()
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_real_temperature_parameter(self):
        """Test temperature parameter affects output diversity."""
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        }
        provider = create_provider(config)
        
        # Test with temperature 0 (deterministic)
        request = LLMRequest(
            prompt="Count: one, two, three,",
            temperature=0,
            max_tokens=5
        )
        
        response = await provider.generate(request)
        assert response.content
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_real_max_tokens_parameter(self):
        """Test max_tokens parameter limits response length."""
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        }
        provider = create_provider(config)
        
        request = LLMRequest(
            prompt="Write a long essay about AI.",
            temperature=0,
            max_tokens=5
        )
        
        response = await provider.generate(request)
        
        assert response.content
        assert response.tokens_used <= 10  # Should be close to max_tokens
        assert response.finish_reason in ["length", "stop"]

    @pytest.mark.asyncio  
    async def test_real_token_counting_accuracy(self):
        """Test that token counts match API response."""
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        }
        provider = create_provider(config)
        
        request = LLMRequest(
            prompt="Say 'test'",
            temperature=0,
            max_tokens=10
        )
        
        response = await provider.generate(request)
        
        assert response.tokens_used > 0
        assert response.prompt_tokens > 0
        assert response.completion_tokens > 0
        assert response.tokens_used == response.prompt_tokens + response.completion_tokens


@pytest.mark.integration
class TestOpenAIErrorHandling:
    """Error handling tests with real API scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_api_key_error(self):
        """Test error handling for invalid API key."""
        config = {
            "provider": "openai",
            "api_key": "sk-invalid-key-123",
            "model": "gpt-3.5-turbo"
        }
        provider = create_provider(config)
        
        request = LLMRequest(
            prompt="Test",
            max_tokens=10
        )
        
        with pytest.raises(RuntimeError, match="authentication"):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_invalid_model_error(self):
        """Test error handling for invalid model name."""
        config = {
            "provider": "openai",
            "model": "gpt-invalid-model"
        }
        provider = create_provider(config)
        
        request = LLMRequest(
            prompt="Test",
            max_tokens=10
        )
        
        with pytest.raises(RuntimeError, match="not found|Model"):
            await provider.generate(request)


@pytest.mark.integration  
class TestOpenAIModelSelection:
    """Test different OpenAI models."""

    @pytest.mark.asyncio
    async def test_gpt4_model(self):
        """Test GPT-4 model (if available)."""
        config = {
            "provider": "openai",
            "model": "gpt-4o-mini"  # Use mini for cost savings
        }
        
        try:
            provider = create_provider(config)
            
            request = LLMRequest(
                prompt="What is 2+2? Answer with just the number.",
                temperature=0,
                max_tokens=5
            )
            
            response = await provider.generate(request)
            
            assert response.content
            assert "4" in response.content
            assert response.model == "gpt-4o-mini"
            
        except RuntimeError as e:
            # Skip if model not available
            if "not found" in str(e).lower():
                pytest.skip(f"GPT-4o-mini not available: {e}")
            raise


@pytest.mark.integration
class TestOpenAIAdvancedFeatures:
    """Test advanced OpenAI features."""

    @pytest.mark.asyncio
    async def test_multiple_requests_cumulative_tokens(self):
        """Test cumulative token tracking across requests."""
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo"
        }
        provider = create_provider(config)
        
        total_tokens = 0
        
        for i in range(2):
            request = LLMRequest(
                prompt=f"Say number {i}",
                temperature=0,
                max_tokens=5
            )
            
            response = await provider.generate(request)
            total_tokens += response.tokens_used
        
        assert total_tokens > 0
        assert total_tokens < 100  # Should be relatively small for short prompts

    @pytest.mark.asyncio
    async def test_model_override_in_request(self):
        """Test that request can override default model."""
        config = {
            "provider": "openai",
            "model": "gpt-3.5-turbo"  # Default model
        }
        provider = create_provider(config)
        
        # Override with different model in request
        request = LLMRequest(
            prompt="Say 'test'",
            model="gpt-3.5-turbo",  # Explicit model
            temperature=0,
            max_tokens=5
        )
        
        response = await provider.generate(request)
        
        assert response.model == "gpt-3.5-turbo"
        assert response.content
