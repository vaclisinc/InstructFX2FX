"""OpenRouter provider implementation using OpenAI-compatible API.

This module provides an implementation of the LLMProvider interface for
accessing various language models through OpenRouter's unified API.
"""

import os
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import openai

from ..base import LLMProvider
from ..types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenRouterProvider(LLMProvider):
    """Provider implementation for OpenRouter API.

    OpenRouter provides a unified API for accessing multiple language models
    including GPT-4, Claude, PaLM, and others through an OpenAI-compatible interface.

    Attributes:
        client: AsyncOpenAI client configured for OpenRouter
        model: Default model to use for generation
        site_url: Optional URL for OpenRouter rankings
        site_name: Optional site name for OpenRouter rankings
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenRouter provider.

        Args:
            config: Configuration dictionary with the following keys:
                - api_key: OpenRouter API key (or use OPENROUTER_API_KEY env var)
                - model: Default model name (e.g., "openai/gpt-4")
                - site_url: Optional URL for OpenRouter rankings
                - site_name: Optional site name for OpenRouter rankings
                - timeout: Optional timeout in seconds (default: 60)
        """
        # Get API key from config or environment BEFORE calling super().__init__()
        # because super().__init__() calls validate_config() which needs self.api_key
        self.api_key = config.get("api_key") or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not found in config or OPENROUTER_API_KEY environment variable")

        # Set default model BEFORE super().__init__()
        self.model = config.get("model", "openai/gpt-4o")

        # Optional OpenRouter-specific configuration
        self.site_url = config.get("site_url")
        self.site_name = config.get("site_name")

        # Now call super().__init__() which will call validate_config()
        super().__init__(config)

        # Initialize OpenAI client with OpenRouter base URL
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=config.get("timeout", 60)
        )

        logger.info(f"Initialized OpenRouterProvider with model: {self.model}")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using OpenRouter API.

        Args:
            request: The LLM request containing prompt and parameters

        Returns:
            LLMResponse with the generated content

        Raises:
            openai.AuthenticationError: If API key is invalid
            openai.RateLimitError: If rate limit is exceeded
            openai.APIError: For other API errors
        """
        try:
            # Prepare messages
            messages = []

            # Add system message if provided
            if request.system_prompt:
                messages.append({
                    "role": "system",
                    "content": request.system_prompt
                })

            # Add user message
            messages.append({
                "role": "user",
                "content": request.prompt
            })

            # Prepare extra headers for OpenRouter
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                extra_headers["X-Title"] = self.site_name

            # Use model from request or default
            model = request.model or self.model

            # Make the API call
            logger.debug(f"Sending request to OpenRouter with model: {model}")

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences,
                extra_headers=extra_headers if extra_headers else None
            )

            # Extract response content
            content = response.choices[0].message.content or ""

            # Extract usage information
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else (prompt_tokens + completion_tokens)

            # Create response
            llm_response = LLMResponse(
                content=content,
                model=model,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=response.choices[0].finish_reason or "stop",
                provider="openrouter"
            )

            logger.info(f"OpenRouter response generated successfully using {model} ({total_tokens} tokens)")
            return llm_response

        except openai.AuthenticationError as e:
            logger.error(f"OpenRouter authentication failed: {e}")
            raise RuntimeError(f"OpenRouter authentication failed. Check your API key.") from e
        except openai.RateLimitError as e:
            logger.warning(f"OpenRouter rate limit exceeded: {e}")
            raise RuntimeError(f"OpenRouter rate limit exceeded. Please try again later.") from e
        except openai.APIStatusError as e:
            if e.status_code == 400:
                logger.error(f"OpenRouter invalid request: {e}")
                raise RuntimeError(f"OpenRouter invalid request: {e.message}") from e
            else:
                logger.error(f"OpenRouter API error (status {e.status_code}): {e}")
                raise RuntimeError(f"OpenRouter API error: {e.message}") from e
        except openai.APITimeoutError as e:
            logger.error(f"OpenRouter request timed out: {e}")
            raise RuntimeError(f"OpenRouter request timed out") from e
        except Exception as e:
            logger.error(f"Unexpected error in OpenRouter generation: {e}")
            raise RuntimeError(f"Unexpected error: {str(e)}") from e

    def validate_config(self) -> bool:
        """Validate the provider configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if not self.api_key:
            raise ValueError("API key is required")

        if self.config.get("timeout") is not None:
            timeout = self.config.get("timeout")
            if not isinstance(timeout, (int, float)) or timeout <= 0:
                raise ValueError("Timeout must be a positive number")

        # Warn if no model specified (will use default)
        if not self.model:
            logger.warning("No default model specified, will use 'openai/gpt-4o'")

        return True

    def get_provider_name(self) -> str:
        """Get the provider name.

        Returns:
            The provider name
        """
        return "openrouter"