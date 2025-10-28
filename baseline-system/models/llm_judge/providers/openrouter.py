"""OpenRouter provider implementation using OpenAI-compatible API.

This module provides the OpenRouterProvider class that implements the LLMProvider
interface for accessing models through OpenRouter's unified API.

OpenRouter provides access to hundreds of AI models through a single endpoint
using the OpenAI-compatible API format.
"""

import logging
import os
from typing import Dict, Any, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from ..base import LLMProvider
from ..types import LLMRequest, LLMResponse


logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """OpenRouter LLM provider using OpenAI-compatible API.

    Uses the OpenAI SDK with a custom base_url to access OpenRouter's
    unified API endpoint. Supports hundreds of models from various providers.

    Configuration keys:
        - api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
        - model: Default model to use (e.g., "openai/gpt-4o")
        - site_url: Optional site URL for OpenRouter rankings
        - site_name: Optional site name for OpenRouter rankings
        - timeout: Request timeout in seconds (default: 60)

    Example:
        ```python
        config = {
            "api_key": "sk-or-v1-...",
            "model": "openai/gpt-4o",
            "site_url": "https://myapp.com",
            "site_name": "My App",
        }
        provider = OpenRouterProvider(config)
        response = await provider.generate(request)
        ```
    """

    def __init__(
        self,
        config: Dict[str, Any],
        retry_config: Optional[Any] = None,
        rate_limit_config: Optional[Any] = None,
    ):
        """Initialize OpenRouter provider.

        Args:
            config: Provider configuration dictionary
            retry_config: Optional retry configuration
            rate_limit_config: Optional rate limit configuration

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config, retry_config, rate_limit_config)

        # Get API key from config or environment
        api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key must be provided in config or "
                "OPENROUTER_API_KEY environment variable"
            )

        # Set up extra headers for OpenRouter
        extra_headers = {}
        if "site_url" in config:
            extra_headers["HTTP-Referer"] = config["site_url"]
        if "site_name" in config:
            extra_headers["X-Title"] = config["site_name"]

        # Initialize OpenAI client with OpenRouter endpoint
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=config.get("timeout", 60.0),
            default_headers=extra_headers if extra_headers else None,
        )

        # Store default model
        self.default_model = config.get("model", "openai/gpt-4o")

        logger.info(
            f"Initialized OpenRouterProvider with model={self.default_model}"
        )

    def validate_config(self) -> bool:
        """Validate provider configuration.

        Checks that:
        - API key is present (in config or environment)
        - Model name is specified
        - Optional fields are properly formatted

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check API key
        api_key = self.config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error("OpenRouter API key not found in config or environment")
            return False

        # Check model
        if not self.config.get("model"):
            logger.warning("No default model specified, will use 'openai/gpt-4o'")

        # Validate timeout if provided
        timeout = self.config.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                logger.error(f"Invalid timeout value: {timeout}")
                return False

        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from OpenRouter.

        Converts the LLMRequest to OpenAI message format, makes the API call,
        and converts the response back to LLMResponse format.

        Args:
            request: LLM request parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ValueError: If request is invalid
            RuntimeError: If API call fails
        """
        # Determine which model to use
        model = request.model or self.default_model

        # Build messages array
        messages = []
        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt,
            })
        messages.append({
            "role": "user",
            "content": request.prompt,
        })

        # Build API request parameters
        api_params = {
            "model": model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # Add stop sequences if provided
        if request.stop_sequences:
            api_params["stop"] = request.stop_sequences

        try:
            logger.debug(f"Calling OpenRouter API with model={model}")

            # Make API call
            completion: ChatCompletion = await self.client.chat.completions.create(
                **api_params
            )

            # Extract response data
            choice = completion.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "unknown"

            # Extract token usage
            usage = completion.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            # Build LLMResponse
            response = LLMResponse(
                content=content,
                model=completion.model,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                provider=self.get_provider_name(),
            )

            logger.debug(
                f"OpenRouter response: {len(content)} chars, "
                f"{total_tokens} tokens, "
                f"finish_reason={finish_reason}"
            )

            return response

        except Exception as e:
            # Map OpenRouter/OpenAI errors to more descriptive messages
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                raise RuntimeError(
                    "OpenRouter authentication failed. Check your API key."
                ) from e
            elif "429" in error_msg or "rate limit" in error_msg.lower():
                raise RuntimeError(
                    "OpenRouter rate limit exceeded. Please try again later."
                ) from e
            elif "400" in error_msg or "invalid" in error_msg.lower():
                raise ValueError(
                    f"Invalid request to OpenRouter: {error_msg}"
                ) from e
            elif "timeout" in error_msg.lower():
                raise RuntimeError(
                    "OpenRouter request timed out. Please try again."
                ) from e
            else:
                raise RuntimeError(
                    f"OpenRouter API error: {error_msg}"
                ) from e


__all__ = ["OpenRouterProvider"]
