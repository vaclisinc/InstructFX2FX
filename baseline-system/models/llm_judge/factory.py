"""Factory pattern for creating LLM provider instances.

This module provides factory functions to instantiate the appropriate
LLM provider based on configuration. It handles:
- Provider type detection
- Configuration validation
- Provider instantiation with proper error handling
- Support for both dict and Pydantic config objects
"""

import logging
from typing import Dict, Any, Union

from .base import LLMProvider
from .config import ProviderConfig, AnthropicConfig, OpenRouterConfig, LLMConfig


logger = logging.getLogger(__name__)


class ProviderNotFoundError(Exception):
    """Raised when an unknown provider type is requested."""
    pass


class ProviderInstantiationError(Exception):
    """Raised when provider instantiation fails."""
    pass


def create_provider(config: Union[Dict[str, Any], ProviderConfig]) -> LLMProvider:
    """Factory function to instantiate LLM provider from configuration.

    This is the main entry point for creating provider instances. It:
    1. Validates the configuration
    2. Determines the provider type
    3. Instantiates the appropriate provider class
    4. Returns a ready-to-use provider instance

    Args:
        config: Provider configuration as dict or ProviderConfig object.
                Must contain 'provider' key indicating type.

    Returns:
        LLMProvider instance ready for use

    Raises:
        ProviderNotFoundError: If provider type is unknown
        ProviderInstantiationError: If provider creation fails
        ValueError: If configuration is invalid

    Example:
        >>> config = {
        ...     "provider": "anthropic",
        ...     "api_key": "sk-ant-...",
        ...     "model": "claude-3-5-sonnet-20241022"
        ... }
        >>> provider = create_provider(config)
        >>> # Use provider for generation
    """
    # Handle dict config
    if isinstance(config, dict):
        provider_type = config.get("provider", "anthropic")
        logger.debug(f"Creating provider from dict config: type={provider_type}")
        # Keep as dict for OpenRouterProvider compatibility
        config_dict = config
    elif isinstance(config, ProviderConfig):
        provider_type = config.provider
        config_dict = config.to_dict()
    else:
        raise TypeError(
            f"Config must be dict or ProviderConfig, got {type(config).__name__}"
        )

    model = config_dict.get("model", "unknown")
    logger.info(f"Instantiating {provider_type} provider with model={model}")

    # Import and instantiate the appropriate provider
    # Note: Provider implementations are being developed in parallel streams
    # For now, we raise an error with helpful message
    try:
        if provider_type == "anthropic":
            # Import ClaudeProvider when available
            try:
                from .providers.claude import ClaudeProvider
                return ClaudeProvider(config=config_dict)
            except ImportError:
                raise ProviderInstantiationError(
                    f"ClaudeProvider not yet implemented. "
                    f"Implementation is in Stream C (parallel development)."
                )

        elif provider_type == "openrouter":
            # Import OpenRouterProvider
            try:
                from .providers.openrouter import OpenRouterProvider
                return OpenRouterProvider(config=config_dict)
            except ImportError as e:
                raise ProviderInstantiationError(
                    f"OpenRouterProvider import failed: {e}"
                )

        elif provider_type == "openai":
            # Import OpenAIProvider
            try:
                from .providers.openai import OpenAIProvider
                return OpenAIProvider(config=config_dict)
            except ImportError as e:
                raise ProviderInstantiationError(
                    f"OpenAIProvider import failed: {e}"
                )

        else:
            # This should never happen due to earlier validation
            raise ProviderNotFoundError(
                f"Unknown provider type: '{provider_type}'"
            )

    except ProviderInstantiationError:
        # Re-raise provider instantiation errors
        raise
    except Exception as e:
        # Catch any other errors during instantiation
        raise ProviderInstantiationError(
            f"Failed to instantiate {provider_type} provider: {e}"
        ) from e


def create_provider_from_llm_config(llm_config: Union[Dict[str, Any], LLMConfig]) -> LLMProvider:
    """Create provider from top-level LLMConfig.

    Convenience function that extracts provider_config from LLMConfig
    and creates the provider.

    Args:
        llm_config: LLMConfig object or dict with 'provider_config' key

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If llm_config is invalid
        ProviderNotFoundError: If provider type is unknown
        ProviderInstantiationError: If provider creation fails

    Example:
        >>> llm_config = {
        ...     "provider_config": {
        ...         "provider": "anthropic",
        ...         "api_key": "sk-ant-...",
        ...         "model": "claude-3-5-sonnet-20241022"
        ...     }
        ... }
        >>> provider = create_provider_from_llm_config(llm_config)
    """
    # Convert dict to LLMConfig if needed
    if isinstance(llm_config, dict):
        try:
            llm_config = LLMConfig.from_dict(llm_config)
        except Exception as e:
            raise ValueError(f"Invalid LLMConfig: {e}") from e
    elif not isinstance(llm_config, LLMConfig):
        raise TypeError(
            f"llm_config must be dict or LLMConfig, got {type(llm_config).__name__}"
        )

    # Extract provider config and create provider
    return create_provider(llm_config.provider_config)


def validate_provider_config(config: Union[Dict[str, Any], ProviderConfig]) -> bool:
    """Validate provider configuration without instantiating.

    Useful for configuration validation during startup or in tests.

    Args:
        config: Provider configuration as dict or ProviderConfig

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid
        ProviderNotFoundError: If provider type is unknown
    """
    # Convert to ProviderConfig if needed
    if isinstance(config, dict):
        provider_type = config.get("provider", "anthropic")

        if provider_type == "anthropic":
            AnthropicConfig(**config)
        elif provider_type == "openai":
            # OpenAI uses similar config to OpenRouter
            pass  # Basic validation, detailed validation in provider
        elif provider_type == "openrouter":
            OpenRouterConfig(**config)
        else:
            raise ProviderNotFoundError(
                f"Unknown provider type: '{provider_type}'"
            )
    elif isinstance(config, ProviderConfig):
        # Already validated by Pydantic
        pass
    else:
        raise TypeError(
            f"Config must be dict or ProviderConfig, got {type(config).__name__}"
        )

    return True


def get_supported_providers() -> list[str]:
    """Get list of supported provider types.

    Returns:
        List of provider type strings
    """
    return ["anthropic", "openai", "openrouter"]


def get_provider_info(provider_type: str) -> Dict[str, Any]:
    """Get information about a specific provider.

    Args:
        provider_type: Provider type ('anthropic' or 'openrouter')

    Returns:
        Dictionary with provider information

    Raises:
        ProviderNotFoundError: If provider type is unknown
    """
    provider_info_map = {
        "anthropic": {
            "name": "Anthropic Claude",
            "config_class": "AnthropicConfig",
            "provider_class": "ClaudeProvider",
            "module": "models.llm_judge.providers.claude",
            "default_model": "claude-4-sonnet",
            "supports_streaming": True,
            "supports_vision": True,
        },
        "openai": {
            "name": "OpenAI",
            "config_class": "OpenAIConfig",
            "provider_class": "OpenAIProvider",
            "module": "models.llm_judge.providers.openai",
            "default_model": "gpt-4o",
            "supports_streaming": True,
            "supports_vision": True,
        },
        "openrouter": {
            "name": "OpenRouter",
            "config_class": "OpenRouterConfig",
            "provider_class": "OpenRouterProvider",
            "module": "models.llm_judge.providers.openrouter",
            "default_model": "openai/gpt-4o",
            "supports_streaming": True,
            "supports_vision": False,  # Depends on underlying model
        },
    }

    if provider_type not in provider_info_map:
        raise ProviderNotFoundError(
            f"Unknown provider type: '{provider_type}'. "
            f"Supported providers: {', '.join(get_supported_providers())}"
        )

    return provider_info_map[provider_type]


__all__ = [
    "create_provider",
    "create_provider_from_llm_config",
    "validate_provider_config",
    "get_supported_providers",
    "get_provider_info",
    "ProviderNotFoundError",
    "ProviderInstantiationError",
]
