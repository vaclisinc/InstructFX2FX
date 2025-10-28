"""OpenAI provider implementation using official OpenAI API.

This module provides a direct implementation of the LLMProvider interface for
accessing OpenAI's GPT models through their official API.
"""

import os
import logging
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
import openai

from ..base import LLMProvider
from ..types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """Provider implementation for OpenAI API.

    This provider uses the official OpenAI API to access GPT models directly,
    including GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, and other OpenAI models.

    Attributes:
        client: AsyncOpenAI client configured for OpenAI API
        model: Default model to use for generation
    """

    DEFAULT_MODEL = "gpt-4o"
    DEFAULT_TIMEOUT = 60.0

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider.

        Args:
            config: Configuration dictionary with the following keys:
                - api_key: OpenAI API key (or use OPENAI_API_KEY env var)
                - model: Default model name (e.g., "gpt-4o", "gpt-3.5-turbo")
                - timeout: Optional timeout in seconds (default: 60)
                - organization: Optional OpenAI organization ID
        """
        # Get API key from config or environment BEFORE calling super().__init__()
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found in config or OPENAI_API_KEY environment variable"
            )

        # Set default model BEFORE super().__init__()
        self.model = config.get("model", self.DEFAULT_MODEL)

        # Optional organization ID
        self.organization = config.get("organization") or os.getenv("OPENAI_ORGANIZATION")

        # Now call super().__init__() which will call validate_config()
        super().__init__(config)

        # Initialize OpenAI client
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": config.get("timeout", self.DEFAULT_TIMEOUT)
        }

        if self.organization:
            client_kwargs["organization"] = self.organization

        self.client = AsyncOpenAI(**client_kwargs)

        logger.info(f"Initialized OpenAIProvider with model: {self.model}")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response using OpenAI API.

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

            # Use model from request or default
            model = request.model or self.model

            # Make the API call
            logger.debug(f"Sending request to OpenAI with model: {model}")

            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stop=request.stop_sequences
            )

            # Extract response content
            content = response.choices[0].message.content or ""

            # Extract usage information
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else (
                prompt_tokens + completion_tokens
            )

            # Create response
            llm_response = LLMResponse(
                content=content,
                model=model,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=response.choices[0].finish_reason or "stop",
                provider="openai"
            )

            logger.info(
                f"OpenAI response generated successfully using {model} ({total_tokens} tokens)"
            )
            return llm_response

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI authentication failed: {e}")
            raise RuntimeError(
                f"OpenAI authentication failed. Check your API key."
            ) from e
        except openai.RateLimitError as e:
            logger.warning(f"OpenAI rate limit exceeded: {e}")
            raise RuntimeError(
                f"OpenAI rate limit exceeded. Please try again later."
            ) from e
        except openai.APIStatusError as e:
            if e.status_code == 400:
                logger.error(f"OpenAI invalid request: {e}")
                raise RuntimeError(f"OpenAI invalid request: {e.message}") from e
            elif e.status_code == 404:
                logger.error(f"OpenAI model not found: {e}")
                raise RuntimeError(
                    f"Model '{model}' not found. Check model name."
                ) from e
            else:
                logger.error(f"OpenAI API error (status {e.status_code}): {e}")
                raise RuntimeError(f"OpenAI API error: {e.message}") from e
        except openai.APITimeoutError as e:
            logger.error(f"OpenAI request timed out: {e}")
            raise RuntimeError(f"OpenAI request timed out") from e
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI generation: {e}")
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

        # Validate model name format
        if self.model and not isinstance(self.model, str):
            raise ValueError("Model must be a string")

        return True

    def get_provider_name(self) -> str:
        """Get the provider name.

        Returns:
            The provider name
        """
        return "openai"
