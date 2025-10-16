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
    from models.llm_judge.claude import ClaudeProvider  # Concrete implementation

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
- claude: ClaudeProvider implementation (to be implemented in Issue #6)
- openrouter: OpenRouterProvider implementation (to be implemented in Issue #7)
- factory: Provider factory for instantiation from config (to be implemented)
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
]
