"""LLM Provider Abstraction Layer.

This package provides a flexible abstraction layer for Large Language Model providers,
supporting both Anthropic Claude and OpenRouter APIs with a unified interface.

Key Features:
- Unified LLMProvider interface for multiple providers
- Pydantic-based request/response data models
- Built-in retry logic with exponential backoff
- Rate limiting to prevent quota exhaustion
- Request/response logging for debugging
- Token usage tracking and reporting

Usage:
    from models.llm_judge import LLMRequest, LLMResponse, LLMProvider
    from models.llm_judge.providers import ClaudeProvider  # Concrete implementation

    # Create provider
    provider = ClaudeProvider(config)

    # Make request
    request = LLMRequest(
        prompt="Explain quantum computing",
        temperature=0.7,
        max_tokens=1000
    )

    response = await provider.generate_with_retry(request)
    print(response.content)

Module Structure:
- types: Data models (LLMRequest, LLMResponse, RetryConfig, RateLimitConfig)
- base: Abstract LLMProvider base class with retry and rate limiting
- config: Configuration schemas (ProviderConfig, AnthropicConfig, OpenRouterConfig, LLMConfig)
- factory: Provider factory for instantiation from config
- providers.claude: ClaudeProvider implementation (Stream C)
- providers.openrouter: OpenRouterProvider implementation (Stream D)
"""

from .types import (
    LLMRequest,
    LLMResponse,
    RetryConfig,
    RateLimitConfig,
)

from .base import (
    LLMProvider,
    RateLimiter,
)

from .config import (
    ProviderConfig,
    AnthropicConfig,
    OpenRouterConfig,
    LLMConfig,
)

from .factory import (
    create_provider,
    create_provider_from_llm_config,
    validate_provider_config,
    get_supported_providers,
    get_provider_info,
    ProviderNotFoundError,
    ProviderInstantiationError,
)

# Provider implementations
from .providers import ClaudeProvider
# from .providers import OpenRouterProvider  # To be implemented


__version__ = "0.1.0"

__all__ = [
    # Data models
    "LLMRequest",
    "LLMResponse",
    "RetryConfig",
    "RateLimitConfig",
    # Base classes
    "LLMProvider",
    "RateLimiter",
    # Configuration
    "ProviderConfig",
    "AnthropicConfig",
    "OpenRouterConfig",
    "LLMConfig",
    # Factory
    "create_provider",
    "create_provider_from_llm_config",
    "validate_provider_config",
    "get_supported_providers",
    "get_provider_info",
    "ProviderNotFoundError",
    "ProviderInstantiationError",
    # Provider implementations
    "ClaudeProvider",
    # "OpenRouterProvider",  # To be implemented
]
