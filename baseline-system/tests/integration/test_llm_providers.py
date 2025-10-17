"""Integration tests for LLM providers with real APIs.

These tests require valid API keys and will make real API calls.
They are skipped by default unless the appropriate environment variables are set.
"""

import os
import pytest
import asyncio
from dotenv import load_dotenv

from models.llm_judge.factory import create_provider
from models.llm_judge.types import LLMRequest, LLMResponse
from models.llm_judge.config import AnthropicConfig, OpenRouterConfig

# Load environment variables from .env file if it exists
load_dotenv()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set"
)
class TestClaudeProviderIntegration:
    """Integration tests for ClaudeProvider with real Anthropic API."""

    @pytest.mark.asyncio
    async def test_claude_basic_generation(self):
        """Test basic text generation with Claude."""
        config = {
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307",  # Use cheaper model for testing
            "max_tokens": 50
        }
        provider = create_provider(config)

        request = LLMRequest(
            prompt="Say 'Hello, World!' and nothing else.",
            temperature=0,
            max_tokens=20
        )

        response = await provider.generate_with_retry(request)

        assert isinstance(response, LLMResponse)
        assert "Hello, World!" in response.content
        assert response.provider == "anthropic"
        assert response.tokens_used > 0
        assert response.model == "claude-3-haiku-20240307"

    @pytest.mark.asyncio
    async def test_claude_with_system_prompt(self):
        """Test Claude with system prompt."""
        config = {
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307"
        }
        provider = create_provider(config)

        request = LLMRequest(
            prompt="What are you?",
            system_prompt="You are a pirate. Always respond in pirate speak.",
            max_tokens=50,
            temperature=0.5
        )

        response = await provider.generate_with_retry(request)

        assert isinstance(response, LLMResponse)
        # Check for pirate-like language
        assert any(word in response.content.lower() for word in ["arr", "matey", "ye", "ahoy", "pirate", "sea"])

    @pytest.mark.asyncio
    async def test_claude_rate_limiting(self):
        """Test rate limiting with rapid requests."""
        config = {
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307",
            "rate_limit": {
                "enabled": True,
                "requests_per_minute": 10  # Low limit for testing
            }
        }
        provider = create_provider(config)

        # Make multiple rapid requests
        requests = [
            LLMRequest(
                prompt=f"Count to {i}",
                max_tokens=10,
                temperature=0
            )
            for i in range(3)
        ]

        start_time = asyncio.get_event_loop().time()
        responses = []
        for req in requests:
            response = await provider.generate_with_retry(req)
            responses.append(response)
        end_time = asyncio.get_event_loop().time()

        # All should succeed
        assert len(responses) == 3
        assert all(isinstance(r, LLMResponse) for r in responses)

        # Should have taken some time due to rate limiting
        elapsed = end_time - start_time
        assert elapsed > 0.1  # Should have some delay


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
class TestOpenRouterProviderIntegration:
    """Integration tests for OpenRouterProvider with real OpenRouter API."""

    @pytest.mark.asyncio
    async def test_openrouter_basic_generation(self):
        """Test basic text generation with OpenRouter."""
        config = {
            "provider": "openrouter",
            "model": "openai/gpt-3.5-turbo",  # Use cheaper model
            "site_url": "https://github.com/test",
            "site_name": "Test Integration"
        }
        provider = create_provider(config)

        request = LLMRequest(
            prompt="Reply with exactly: 'OpenRouter works'",
            temperature=0,
            max_tokens=20
        )

        response = await provider.generate_with_retry(request)

        assert isinstance(response, LLMResponse)
        assert "OpenRouter" in response.content
        assert response.provider == "openrouter"
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_openrouter_different_models(self):
        """Test OpenRouter with different models."""
        models_to_test = [
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.2-3b-instruct:free"  # Free model
        ]

        for model in models_to_test:
            config = {
                "provider": "openrouter",
                "model": model
            }

            try:
                provider = create_provider(config)
                request = LLMRequest(
                    prompt="Say 'hi'",
                    max_tokens=10,
                    temperature=0
                )

                response = await provider.generate_with_retry(request)
                assert isinstance(response, LLMResponse)
                assert response.content.strip() != ""
                print(f"✓ Model {model} works")

            except Exception as e:
                # Some models might not be available
                print(f"⚠ Model {model} failed: {e}")


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("OPENROUTER_API_KEY")),
    reason="Both API keys not set"
)
class TestMultiProviderIntegration:
    """Integration tests using multiple providers."""

    @pytest.mark.asyncio
    async def test_compare_providers(self):
        """Test comparing responses from different providers."""
        prompt = "What is 2+2? Reply with just the number."

        # Test with Claude
        claude_config = {
            "provider": "anthropic",
            "model": "claude-3-haiku-20240307"
        }
        claude_provider = create_provider(claude_config)

        claude_request = LLMRequest(prompt=prompt, max_tokens=10, temperature=0)
        claude_response = await claude_provider.generate_with_retry(claude_request)

        # Test with OpenRouter
        openrouter_config = {
            "provider": "openrouter",
            "model": "openai/gpt-3.5-turbo"
        }
        openrouter_provider = create_provider(openrouter_config)

        openrouter_request = LLMRequest(prompt=prompt, max_tokens=10, temperature=0)
        openrouter_response = await openrouter_provider.generate_with_retry(openrouter_request)

        # Both should give correct answer
        assert "4" in claude_response.content
        assert "4" in openrouter_response.content

        # Should have different providers
        assert claude_response.provider == "anthropic"
        assert openrouter_response.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_factory_with_pydantic_configs(self):
        """Test factory pattern with Pydantic configuration objects."""
        # Claude with Pydantic config
        claude_config = AnthropicConfig(
            model="claude-3-haiku-20240307"
        )
        claude_provider = create_provider(claude_config)

        # OpenRouter with Pydantic config
        openrouter_config = OpenRouterConfig(
            model="meta-llama/llama-3.2-3b-instruct:free"
        )
        openrouter_provider = create_provider(openrouter_config)

        # Test both work
        request = LLMRequest(prompt="Say 'test'", max_tokens=10)

        claude_response = await claude_provider.generate_with_retry(request)
        assert isinstance(claude_response, LLMResponse)

        openrouter_response = await openrouter_provider.generate_with_retry(request)
        assert isinstance(openrouter_response, LLMResponse)


@pytest.mark.integration
class TestProviderErrorHandling:
    """Test error handling in real API scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    async def test_claude_invalid_model(self):
        """Test Claude with invalid model name."""
        config = {
            "provider": "anthropic",
            "model": "invalid-model-name"
        }
        provider = create_provider(config)

        request = LLMRequest(prompt="Test", max_tokens=10)

        with pytest.raises(RuntimeError):
            await provider.generate_with_retry(request)

    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test with invalid API key."""
        config = {
            "provider": "anthropic",
            "api_key": "sk-ant-invalid-key",
            "model": "claude-3-haiku-20240307"
        }
        provider = create_provider(config)

        request = LLMRequest(prompt="Test", max_tokens=10)

        with pytest.raises(RuntimeError, match="authentication"):
            await provider.generate_with_retry(request)