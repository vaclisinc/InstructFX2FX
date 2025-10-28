"""Tests for provider factory."""

import pytest
from unittest.mock import patch, MagicMock

from models.llm_judge.factory import (
    create_provider,
    create_provider_from_llm_config,
    validate_provider_config,
    get_supported_providers,
    get_provider_info,
    ProviderNotFoundError,
    ProviderInstantiationError
)
from models.llm_judge.config import LLMConfig, AnthropicConfig, OpenRouterConfig
from models.llm_judge.base import LLMProvider


class TestProviderFactory:
    """Test provider factory functions."""

    def test_get_supported_providers(self):
        """Test getting list of supported providers."""
        providers = get_supported_providers()
        assert "anthropic" in providers
        assert "openrouter" in providers
        assert len(providers) == 2

    def test_get_provider_info_anthropic(self):
        """Test getting Anthropic provider info."""
        info = get_provider_info("anthropic")
        assert info["name"] == "anthropic"
        assert info["display_name"] == "Anthropic Claude"
        assert "claude-3" in info["description"]
        assert "claude-3-5-sonnet" in info["models"]

    def test_get_provider_info_openrouter(self):
        """Test getting OpenRouter provider info."""
        info = get_provider_info("openrouter")
        assert info["name"] == "openrouter"
        assert info["display_name"] == "OpenRouter"
        assert "unified API" in info["description"]
        assert "openai/gpt-4" in info["models"]

    def test_get_provider_info_unknown(self):
        """Test getting info for unknown provider."""
        with pytest.raises(ProviderNotFoundError, match="Unknown provider"):
            get_provider_info("unknown")

    @patch('models.llm_judge.factory.ClaudeProvider')
    def test_create_provider_anthropic_dict(self, mock_claude):
        """Test creating Anthropic provider from dict config."""
        mock_instance = MagicMock(spec=LLMProvider)
        mock_claude.return_value = mock_instance

        config = {
            "provider": "anthropic",
            "api_key": "sk-ant-test",
            "model": "claude-3-5-sonnet"
        }

        provider = create_provider(config)

        mock_claude.assert_called_once_with(config)
        assert provider == mock_instance

    @patch('models.llm_judge.factory.OpenRouterProvider')
    def test_create_provider_openrouter_dict(self, mock_openrouter):
        """Test creating OpenRouter provider from dict config."""
        mock_instance = MagicMock(spec=LLMProvider)
        mock_openrouter.return_value = mock_instance

        config = {
            "provider": "openrouter",
            "api_key": "sk-or-test",
            "model": "openai/gpt-4"
        }

        provider = create_provider(config)

        mock_openrouter.assert_called_once_with(config)
        assert provider == mock_instance

    @patch('models.llm_judge.factory.ClaudeProvider')
    def test_create_provider_with_pydantic_config(self, mock_claude):
        """Test creating provider from Pydantic config object."""
        mock_instance = MagicMock(spec=LLMProvider)
        mock_claude.return_value = mock_instance

        config = AnthropicConfig(
            api_key="sk-ant-test",
            model="claude-3-5-sonnet"
        )

        provider = create_provider(config)

        # Should convert to dict
        mock_claude.assert_called_once()
        call_args = mock_claude.call_args[0][0]
        assert call_args["api_key"] == "sk-ant-test"
        assert call_args["model"] == "claude-3-5-sonnet"

    def test_create_provider_default_anthropic(self):
        """Test creating provider defaults to Anthropic."""
        with patch('models.llm_judge.factory.ClaudeProvider') as mock_claude:
            mock_instance = MagicMock(spec=LLMProvider)
            mock_claude.return_value = mock_instance

            config = {"api_key": "sk-test"}
            provider = create_provider(config)

            mock_claude.assert_called_once()
            assert provider == mock_instance

    def test_create_provider_unknown_type(self):
        """Test creating provider with unknown type raises error."""
        config = {
            "provider": "unknown",
            "api_key": "sk-test"
        }

        with pytest.raises(ProviderNotFoundError, match="Unknown provider: unknown"):
            create_provider(config)

    @patch('models.llm_judge.factory.ClaudeProvider')
    def test_create_provider_instantiation_error(self, mock_claude):
        """Test handling of provider instantiation errors."""
        mock_claude.side_effect = ValueError("Invalid config")

        config = {
            "provider": "anthropic",
            "api_key": "invalid"
        }

        with pytest.raises(ProviderInstantiationError, match="Failed to create anthropic provider"):
            create_provider(config)

    @patch('models.llm_judge.factory.ClaudeProvider')
    def test_create_provider_from_llm_config(self, mock_claude):
        """Test creating provider from LLMConfig wrapper."""
        mock_instance = MagicMock(spec=LLMProvider)
        mock_claude.return_value = mock_instance

        llm_config = LLMConfig(
            provider_config=AnthropicConfig(
                api_key="sk-ant-test",
                model="claude-3-5-sonnet"
            )
        )

        provider = create_provider_from_llm_config(llm_config)

        mock_claude.assert_called_once()
        assert provider == mock_instance

    @patch('models.llm_judge.factory.ClaudeProvider')
    def test_validate_provider_config_valid(self, mock_claude):
        """Test validating valid provider configuration."""
        mock_instance = MagicMock(spec=LLMProvider)
        mock_instance.validate_config.return_value = True
        mock_claude.return_value = mock_instance

        config = {
            "provider": "anthropic",
            "api_key": "sk-ant-test"
        }

        result = validate_provider_config(config)

        assert result is True
        mock_instance.validate_config.assert_called_once()

    @patch('models.llm_judge.factory.ClaudeProvider')
    def test_validate_provider_config_invalid(self, mock_claude):
        """Test validating invalid provider configuration."""
        mock_instance = MagicMock(spec=LLMProvider)
        mock_instance.validate_config.side_effect = ValueError("Invalid API key")
        mock_claude.return_value = mock_instance

        config = {
            "provider": "anthropic",
            "api_key": ""
        }

        result = validate_provider_config(config)

        assert result is False

    def test_validate_provider_config_unknown_provider(self):
        """Test validating config for unknown provider."""
        config = {
            "provider": "unknown",
            "api_key": "sk-test"
        }

        result = validate_provider_config(config)
        assert result is False

    @patch('models.llm_judge.factory.ClaudeProvider')
    @patch('models.llm_judge.factory.OpenRouterProvider')
    def test_create_multiple_providers(self, mock_openrouter, mock_claude):
        """Test creating multiple providers in sequence."""
        mock_claude_instance = MagicMock(spec=LLMProvider)
        mock_openrouter_instance = MagicMock(spec=LLMProvider)
        mock_claude.return_value = mock_claude_instance
        mock_openrouter.return_value = mock_openrouter_instance

        # Create Anthropic provider
        anthropic_config = {
            "provider": "anthropic",
            "api_key": "sk-ant-test"
        }
        anthropic_provider = create_provider(anthropic_config)
        assert anthropic_provider == mock_claude_instance

        # Create OpenRouter provider
        openrouter_config = {
            "provider": "openrouter",
            "api_key": "sk-or-test"
        }
        openrouter_provider = create_provider(openrouter_config)
        assert openrouter_provider == mock_openrouter_instance

        # Verify both were called
        mock_claude.assert_called_once()
        mock_openrouter.assert_called_once()

    def test_factory_with_empty_config(self):
        """Test factory behavior with empty config."""
        with pytest.raises((ProviderInstantiationError, ValueError)):
            create_provider({})

    def test_factory_with_none_config(self):
        """Test factory behavior with None config."""
        with pytest.raises((TypeError, AttributeError)):
            create_provider(None)