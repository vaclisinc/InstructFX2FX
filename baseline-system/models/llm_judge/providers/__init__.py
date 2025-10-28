"""LLM provider implementations.

This package contains concrete implementations of the LLMProvider interface
for various LLM services.

Available providers:
    - OpenRouterProvider: Unified API for hundreds of models via OpenRouter
"""

from .openrouter import OpenRouterProvider


__all__ = [
    "OpenRouterProvider",
]
