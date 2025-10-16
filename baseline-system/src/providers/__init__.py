"""LLM provider abstractions."""

import os
import httpx
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def get_anthropic_client(api_key: Optional[str] = None):
    """Get Anthropic client with proper configuration.

    Args:
        api_key: Optional API key, defaults to environment variable

    Returns:
        Configured Anthropic client
    """
    import anthropic

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

    # Create httpx client without proxy to avoid configuration issues
    http_client = httpx.Client(trust_env=False)

    return anthropic.Anthropic(
        api_key=api_key,
        http_client=http_client
    )


def get_openai_client(api_key: Optional[str] = None,
                      base_url: Optional[str] = None):
    """Get OpenAI client with proper configuration.

    Args:
        api_key: Optional API key, defaults to environment variable
        base_url: Optional base URL for API (e.g., for OpenRouter)

    Returns:
        Configured OpenAI client
    """
    from openai import OpenAI

    api_key = api_key or os.getenv("OPENAI_API_KEY")

    # Create httpx client without proxy to avoid configuration issues
    http_client = httpx.Client(trust_env=False)

    kwargs = {
        "api_key": api_key,
        "http_client": http_client
    }

    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def get_openrouter_client(api_key: Optional[str] = None):
    """Get OpenRouter client (OpenAI-compatible).

    Args:
        api_key: Optional API key, defaults to environment variable

    Returns:
        Configured OpenRouter client
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")

    return get_openai_client(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )


# Export convenience functions
__all__ = [
    "get_anthropic_client",
    "get_openai_client",
    "get_openrouter_client"
]
