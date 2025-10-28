"""Tests for OpenRouterProvider."""

import os
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import openai

from models.llm_judge.providers.openrouter import OpenRouterProvider
from models.llm_judge.types import LLMRequest, LLMResponse


class TestOpenRouterProviderConfiguration:
    """Test configuration and initialization of OpenRouterProvider."""

    def test_init_with_api_key_in_config(self):
        """Test initialization with API key in config."""
        config = {
            "api_key": "sk-or-test-key",
            "model": "openai/gpt-4"
        }
        provider = OpenRouterProvider(config)
        assert provider.api_key == "sk-or-test-key"
        assert provider.model == "openai/gpt-4"

    def test_init_with_api_key_from_env(self, monkeypatch):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        config = {"model": "claude-3-opus"}
        provider = OpenRouterProvider(config)
        assert provider.api_key == "sk-or-env-key"

    def test_init_with_default_model(self):
        """Test initialization with default model."""
        config = {"api_key": "sk-or-test"}
        provider = OpenRouterProvider(config)
        assert provider.model == "openai/gpt-4o"

    def test_init_with_site_config(self):
        """Test initialization with site URL and name."""
        config = {
            "api_key": "sk-or-test",
            "site_url": "https://example.com",
            "site_name": "Example App"
        }
        provider = OpenRouterProvider(config)
        assert provider.site_url == "https://example.com"
        assert provider.site_name == "Example App"

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        config = {
            "api_key": "sk-or-test",
            "timeout": 120
        }
        provider = OpenRouterProvider(config)
        assert provider.client.timeout == 120

    def test_init_without_api_key_raises_error(self):
        """Test initialization without API key raises error."""
        config = {}
        with pytest.raises(ValueError, match="API key not found"):
            OpenRouterProvider(config)

    def test_get_provider_name(self):
        """Test get_provider_name returns correct name."""
        config = {"api_key": "sk-or-test"}
        provider = OpenRouterProvider(config)
        assert provider.get_provider_name() == "openrouter"


class TestOpenRouterProviderGeneration:
    """Test generation methods of OpenRouterProvider."""

    @pytest.mark.asyncio
    async def test_generate_basic_request(self):
        """Test basic generation request."""
        config = {"api_key": "sk-or-test"}
        provider = OpenRouterProvider(config)

        # Mock the OpenAI client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        request = LLMRequest(prompt="Test prompt")
        response = await provider.generate(request)

        assert response.content == "Test response"
        assert response.tokens_used == 30
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 20
        assert response.provider == "openrouter"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generation with system prompt."""
        config = {"api_key": "sk-or-test"}
        provider = OpenRouterProvider(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 15
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 40

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        request = LLMRequest(
            prompt="Test prompt",
            system_prompt="You are a helpful assistant"
        )
        response = await provider.generate(request)

        # Verify the call included system message
        call_args = provider.client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_model_override(self):
        """Test generation with model override in request."""
        config = {"api_key": "sk-or-test", "model": "default-model"}
        provider = OpenRouterProvider(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 30
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        request = LLMRequest(prompt="Test", model="claude-3-opus")
        await provider.generate(request)

        # Verify the override model was used
        call_args = provider.client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "claude-3-opus"

    @pytest.mark.asyncio
    async def test_generate_with_openrouter_headers(self):
        """Test generation includes OpenRouter-specific headers."""
        config = {
            "api_key": "sk-or-test",
            "site_url": "https://myapp.com",
            "site_name": "My App"
        }
        provider = OpenRouterProvider(config)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.total_tokens = 10
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5

        provider.client.chat.completions.create = AsyncMock(return_value=mock_response)

        request = LLMRequest(prompt="Test")
        await provider.generate(request)

        # Verify headers were included
        call_args = provider.client.chat.completions.create.call_args
        headers = call_args.kwargs["extra_headers"]
        assert headers["HTTP-Referer"] == "https://myapp.com"
        assert headers["X-Title"] == "My App"


class TestOpenRouterProviderErrorHandling:
    """Test error handling in OpenRouterProvider."""

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test handling of authentication errors."""
        config = {"api_key": "invalid-key"}
        provider = OpenRouterProvider(config)

        error = openai.AuthenticationError(
            message="Invalid API key",
            response=None,
            body=None
        )
        provider.client.chat.completions.create = AsyncMock(side_effect=error)

        request = LLMRequest(prompt="Test")
        with pytest.raises(RuntimeError, match="authentication failed"):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test handling of rate limit errors."""
        config = {"api_key": "sk-or-test"}
        provider = OpenRouterProvider(config)

        error = openai.RateLimitError(
            message="Rate limit exceeded",
            response=None,
            body=None
        )
        provider.client.chat.completions.create = AsyncMock(side_effect=error)

        request = LLMRequest(prompt="Test")
        with pytest.raises(RuntimeError, match="rate limit exceeded"):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_api_status_error_400(self):
        """Test handling of 400 status errors."""
        config = {"api_key": "sk-or-test"}
        provider = OpenRouterProvider(config)

        # Create mock response for status error
        mock_response = Mock()
        mock_response.status_code = 400

        error = openai.APIStatusError(
            message="Bad request",
            response=mock_response,
            body=None
        )
        error.status_code = 400

        provider.client.chat.completions.create = AsyncMock(side_effect=error)

        request = LLMRequest(prompt="Test")
        with pytest.raises(RuntimeError, match="invalid request"):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_timeout_error(self):
        """Test handling of timeout errors."""
        config = {"api_key": "sk-or-test"}
        provider = OpenRouterProvider(config)

        error = openai.APITimeoutError(request=None)
        provider.client.chat.completions.create = AsyncMock(side_effect=error)

        request = LLMRequest(prompt="Test")
        with pytest.raises(RuntimeError, match="timed out"):
            await provider.generate(request)


class TestOpenRouterProviderValidation:
    """Test validation methods of OpenRouterProvider."""

    @pytest.mark.asyncio
    async def test_validate_config_valid(self):
        """Test validation of valid configuration."""
        config = {
            "api_key": "sk-or-test",
            "model": "openai/gpt-4",
            "timeout": 60
        }
        provider = OpenRouterProvider(config)
        assert await provider.validate_config() is True

    @pytest.mark.asyncio
    async def test_validate_config_invalid_timeout(self):
        """Test validation with invalid timeout."""
        config = {
            "api_key": "sk-or-test",
            "timeout": -1
        }
        provider = OpenRouterProvider(config)
        with pytest.raises(ValueError, match="positive number"):
            await provider.validate_config()

    @pytest.mark.asyncio
    async def test_validate_config_no_api_key(self):
        """Test validation without API key."""
        # Have to set api_key to None after init since init validates
        config = {"api_key": "temp"}
        provider = OpenRouterProvider(config)
        provider.api_key = None
        with pytest.raises(ValueError, match="API key is required"):
            await provider.validate_config()


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
class TestOpenRouterProviderIntegration:
    """Integration tests for OpenRouterProvider with real API."""

    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test real API call to OpenRouter."""
        config = {
            "model": "openai/gpt-3.5-turbo"  # Use cheaper model for testing
        }
        provider = OpenRouterProvider(config)

        request = LLMRequest(
            prompt="Say 'test successful' and nothing else",
            max_tokens=10,
            temperature=0
        )

        response = await provider.generate_with_retry(request)

        assert response.content.lower().strip() in ["test successful", "test successful."]
        assert response.provider == "openrouter"
        assert response.tokens_used > 0