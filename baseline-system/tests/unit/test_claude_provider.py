"""Unit tests for ClaudeProvider.

These tests verify the ClaudeProvider implementation including:
- Configuration validation
- Request formatting
- Response parsing
- Error handling
- Token counting
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from models.llm_judge.providers import ClaudeProvider
from models.llm_judge.types import LLMRequest, LLMResponse
from anthropic import APIConnectionError, RateLimitError, APIStatusError


class TestClaudeProviderConfiguration:
    """Test ClaudeProvider configuration and initialization."""

    def test_init_with_api_key_in_config(self):
        """Test initialization with API key in config."""
        config = {
            "api_key": "sk-ant-test-key",
            "model": "claude-3-5-sonnet-20241022"
        }
        provider = ClaudeProvider(config)

        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.client is not None

    def test_init_with_api_key_in_env(self, monkeypatch):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")

        config = {
            "model": "claude-3-5-sonnet-20241022"
        }
        provider = ClaudeProvider(config)

        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.client is not None

    def test_init_with_default_model(self):
        """Test initialization uses default model when not specified."""
        config = {
            "api_key": "sk-ant-test-key"
        }
        provider = ClaudeProvider(config)

        assert provider.model == ClaudeProvider.DEFAULT_MODEL

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        config = {
            "api_key": "sk-ant-test-key",
            "timeout": 120.0
        }
        provider = ClaudeProvider(config)

        assert provider.client is not None

    def test_init_without_api_key_raises_error(self, monkeypatch):
        """Test initialization fails without API key."""
        # Clear environment variable if set
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        config = {
            "model": "claude-3-5-sonnet-20241022"
        }

        with pytest.raises(ValueError, match="Invalid configuration"):
            ClaudeProvider(config)

    def test_validate_config_success(self):
        """Test successful config validation."""
        config = {
            "api_key": "sk-ant-test-key",
            "model": "claude-3-5-sonnet-20241022",
            "timeout": 60.0
        }
        provider = ClaudeProvider(config)

        assert provider.validate_config() is True

    def test_validate_config_empty_api_key(self, monkeypatch):
        """Test config validation fails with empty API key."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        config = {
            "api_key": "",
            "model": "claude-3-5-sonnet-20241022"
        }

        # Should raise during init
        with pytest.raises(ValueError):
            ClaudeProvider(config)

    def test_validate_config_empty_model(self):
        """Test config validation fails with empty model."""
        config = {
            "api_key": "sk-ant-test-key",
            "model": ""
        }

        # Empty model should fail validation during init
        with pytest.raises(ValueError, match="Invalid configuration"):
            ClaudeProvider(config)

    def test_validate_config_invalid_timeout(self):
        """Test config validation fails with invalid timeout."""
        config = {
            "api_key": "sk-ant-test-key",
            "model": "claude-3-5-sonnet-20241022",
            "timeout": -10
        }

        # Invalid timeout should fail validation during init
        with pytest.raises(ValueError, match="Invalid configuration"):
            ClaudeProvider(config)

    def test_get_provider_name(self):
        """Test provider name extraction."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        assert provider.get_provider_name() == "claude"


class TestClaudeProviderGeneration:
    """Test ClaudeProvider message generation."""

    @pytest.mark.asyncio
    async def test_generate_basic_request(self):
        """Test basic message generation."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock the API response
        mock_content_block = MagicMock()
        mock_content_block.text = "This is a test response"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_message = MagicMock()
        mock_message.content = [mock_content_block]
        mock_message.usage = mock_usage
        mock_message.model = "claude-3-5-sonnet-20241022"
        mock_message.stop_reason = "end_turn"

        provider.client.messages.create = AsyncMock(return_value=mock_message)

        # Make request
        request = LLMRequest(
            prompt="Hello, Claude!",
            temperature=0.7,
            max_tokens=1000
        )

        response = await provider.generate(request)

        # Verify response
        assert isinstance(response, LLMResponse)
        assert response.content == "This is a test response"
        assert response.model == "claude-3-5-sonnet-20241022"
        assert response.tokens_used == 15
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5
        assert response.finish_reason == "end_turn"
        assert response.provider == "claude"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test message generation with system prompt."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock the API response
        mock_content_block = MagicMock()
        mock_content_block.text = "Response text"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 20
        mock_usage.output_tokens = 10

        mock_message = MagicMock()
        mock_message.content = [mock_content_block]
        mock_message.usage = mock_usage
        mock_message.model = "claude-3-5-sonnet-20241022"
        mock_message.stop_reason = "end_turn"

        create_mock = AsyncMock(return_value=mock_message)
        provider.client.messages.create = create_mock

        # Make request with system prompt
        request = LLMRequest(
            prompt="What is 2+2?",
            system_prompt="You are a helpful math tutor.",
            temperature=0.7,
            max_tokens=1000
        )

        response = await provider.generate(request)

        # Verify system prompt was passed
        call_args = create_mock.call_args
        assert "system" in call_args.kwargs
        assert call_args.kwargs["system"] == "You are a helpful math tutor."

        # Verify response
        assert response.content == "Response text"
        assert response.tokens_used == 30

    @pytest.mark.asyncio
    async def test_generate_with_stop_sequences(self):
        """Test message generation with stop sequences."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock the API response
        mock_content_block = MagicMock()
        mock_content_block.text = "Response"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_message = MagicMock()
        mock_message.content = [mock_content_block]
        mock_message.usage = mock_usage
        mock_message.model = "claude-3-5-sonnet-20241022"
        mock_message.stop_reason = "stop_sequence"

        create_mock = AsyncMock(return_value=mock_message)
        provider.client.messages.create = create_mock

        # Make request with stop sequences
        request = LLMRequest(
            prompt="Count to 10",
            stop_sequences=["5", "STOP"],
            temperature=0.7,
            max_tokens=1000
        )

        response = await provider.generate(request)

        # Verify stop sequences were passed
        call_args = create_mock.call_args
        assert "stop_sequences" in call_args.kwargs
        assert call_args.kwargs["stop_sequences"] == ["5", "STOP"]

        # Verify response
        assert response.finish_reason == "stop_sequence"

    @pytest.mark.asyncio
    async def test_generate_with_model_override(self):
        """Test message generation with model override."""
        config = {
            "api_key": "sk-ant-test-key",
            "model": "claude-3-5-sonnet-20241022"
        }
        provider = ClaudeProvider(config)

        # Mock the API response
        mock_content_block = MagicMock()
        mock_content_block.text = "Response"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 5

        mock_message = MagicMock()
        mock_message.content = [mock_content_block]
        mock_message.usage = mock_usage
        mock_message.model = "claude-3-opus-20240229"
        mock_message.stop_reason = "end_turn"

        create_mock = AsyncMock(return_value=mock_message)
        provider.client.messages.create = create_mock

        # Make request with different model
        request = LLMRequest(
            prompt="Hello",
            model="claude-3-opus-20240229",
            temperature=0.7,
            max_tokens=1000
        )

        response = await provider.generate(request)

        # Verify correct model was used
        call_args = create_mock.call_args
        assert call_args.kwargs["model"] == "claude-3-opus-20240229"
        assert response.model == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_generate_multiple_content_blocks(self):
        """Test message generation with multiple content blocks."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock multiple content blocks
        mock_block1 = MagicMock()
        mock_block1.text = "First part "

        mock_block2 = MagicMock()
        mock_block2.text = "second part"

        mock_usage = MagicMock()
        mock_usage.input_tokens = 10
        mock_usage.output_tokens = 8

        mock_message = MagicMock()
        mock_message.content = [mock_block1, mock_block2]
        mock_message.usage = mock_usage
        mock_message.model = "claude-3-5-sonnet-20241022"
        mock_message.stop_reason = "end_turn"

        provider.client.messages.create = AsyncMock(return_value=mock_message)

        # Make request
        request = LLMRequest(
            prompt="Test",
            temperature=0.7,
            max_tokens=1000
        )

        response = await provider.generate(request)

        # Verify content concatenation
        assert response.content == "First part second part"


class TestClaudeProviderErrorHandling:
    """Test ClaudeProvider error handling."""

    @pytest.mark.asyncio
    async def test_generate_connection_error(self):
        """Test handling of connection errors."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock connection error - APIConnectionError requires request parameter
        mock_request = MagicMock()
        provider.client.messages.create = AsyncMock(
            side_effect=APIConnectionError(request=mock_request)
        )

        request = LLMRequest(
            prompt="Test",
            temperature=0.7,
            max_tokens=1000
        )

        with pytest.raises(APIConnectionError):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self):
        """Test handling of rate limit errors."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock rate limit error
        mock_response = MagicMock()
        mock_response.status_code = 429

        provider.client.messages.create = AsyncMock(
            side_effect=RateLimitError(
                "Rate limit exceeded",
                response=mock_response,
                body=None
            )
        )

        request = LLMRequest(
            prompt="Test",
            temperature=0.7,
            max_tokens=1000
        )

        with pytest.raises(RateLimitError):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_generate_api_status_error(self):
        """Test handling of API status errors."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock status error
        mock_response = MagicMock()
        mock_response.status_code = 500

        provider.client.messages.create = AsyncMock(
            side_effect=APIStatusError(
                "Server error",
                response=mock_response,
                body=None
            )
        )

        request = LLMRequest(
            prompt="Test",
            temperature=0.7,
            max_tokens=1000
        )

        with pytest.raises(APIStatusError):
            await provider.generate(request)

    @pytest.mark.asyncio
    async def test_generate_unexpected_error(self):
        """Test handling of unexpected errors."""
        config = {"api_key": "sk-ant-test-key"}
        provider = ClaudeProvider(config)

        # Mock unexpected error
        provider.client.messages.create = AsyncMock(
            side_effect=Exception("Unexpected error")
        )

        request = LLMRequest(
            prompt="Test",
            temperature=0.7,
            max_tokens=1000
        )

        with pytest.raises(RuntimeError, match="Unexpected error"):
            await provider.generate(request)


class TestClaudeProviderIntegration:
    """Integration tests requiring API key (optional)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    async def test_real_api_call(self):
        """Test actual API call (requires valid API key)."""
        config = {
            "model": "claude-3-5-sonnet-20241022"
        }
        provider = ClaudeProvider(config)

        request = LLMRequest(
            prompt="Say 'Hello, World!' and nothing else.",
            temperature=0.0,
            max_tokens=50
        )

        response = await provider.generate_with_retry(request)

        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0
        assert response.tokens_used > 0
        assert response.provider == "claude"
        assert "Hello" in response.content or "hello" in response.content
