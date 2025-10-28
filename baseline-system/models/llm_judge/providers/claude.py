"""Anthropic Claude provider implementation.

This module implements the ClaudeProvider class, which provides access to
Anthropic's Claude models using the official anthropic SDK.

Features:
- Support for all Claude models (claude-3.5-sonnet, claude-3-opus, etc.)
- Async message generation with streaming support
- Proper token counting using API response data
- System prompt handling via system parameter
- Error mapping to appropriate exceptions
"""

import logging
import os
from typing import Dict, Any, Optional

from anthropic import AsyncAnthropic, APIError, APIConnectionError, RateLimitError, APIStatusError

from ..base import LLMProvider
from ..types import LLMRequest, LLMResponse, RetryConfig, RateLimitConfig


logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider implementation.

    This provider uses the official Anthropic SDK to interact with Claude models.
    It supports async message generation, proper error handling, and token tracking.

    Configuration:
        api_key: Anthropic API key (required, can be set via ANTHROPIC_API_KEY env var)
        model: Default model name (optional, defaults to claude-3-5-sonnet-20241022)
        timeout: Request timeout in seconds (optional, default 60)

    Example:
        config = {
            "api_key": "sk-ant-...",
            "model": "claude-3-5-sonnet-20241022",
            "timeout": 60
        }
        provider = ClaudeProvider(config)

        request = LLMRequest(
            prompt="Explain quantum computing",
            system_prompt="You are a helpful physics professor",
            temperature=0.7,
            max_tokens=1000
        )

        response = await provider.generate_with_retry(request)
        print(response.content)
    """

    DEFAULT_MODEL = "claude-4-sonnet-20250514"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        config: Dict[str, Any],
        retry_config: Optional[RetryConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
    ):
        """Initialize Claude provider.

        Args:
            config: Provider configuration with api_key, model, timeout
            retry_config: Optional retry configuration
            rate_limit_config: Optional rate limit configuration

        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config, retry_config, rate_limit_config)

        # Get API key from config or environment
        api_key = config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set 'api_key' in config or "
                "ANTHROPIC_API_KEY environment variable."
            )

        # Get model and timeout
        self.model = config.get("model", self.DEFAULT_MODEL)
        timeout = config.get("timeout", self.DEFAULT_TIMEOUT)

        # Initialize Anthropic client
        # Note: Only pass supported parameters to avoid compatibility issues
        self.client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout
        )

        logger.info(
            f"Initialized ClaudeProvider with model={self.model}, "
            f"timeout={timeout}s"
        )

    def validate_config(self) -> bool:
        """Validate provider configuration.

        Checks that:
        1. API key is present (in config or environment)
        2. Model name is specified
        3. Timeout is positive

        Returns:
            True if configuration is valid, False otherwise
        """
        # Check API key
        api_key = self.config.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key or not api_key.strip():
            logger.error("No API key found in config or ANTHROPIC_API_KEY env var")
            return False

        # Check model
        model = self.config.get("model", self.DEFAULT_MODEL)
        if not model or not model.strip():
            logger.error("Model name is empty")
            return False

        # Check timeout
        timeout = self.config.get("timeout", self.DEFAULT_TIMEOUT)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            logger.error(f"Invalid timeout: {timeout}")
            return False

        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from Claude.

        This method:
        1. Formats the request for Anthropic's API
        2. Makes the async API call
        3. Parses the response
        4. Returns an LLMResponse object

        Args:
            request: LLM request parameters

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ValueError: If request is invalid
            APIConnectionError: If network connection fails
            RateLimitError: If rate limit is exceeded (429)
            APIStatusError: For other API errors (4xx, 5xx)
            RuntimeError: For other unexpected errors
        """
        try:
            # Determine which model to use (request override or default)
            model = request.model or self.model

            # Build messages list (Claude expects a list of message dicts)
            messages = [
                {
                    "role": "user",
                    "content": request.prompt,
                }
            ]

            # Build request parameters
            params: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            # Add system prompt if provided
            if request.system_prompt:
                params["system"] = request.system_prompt

            # Add stop sequences if provided
            if request.stop_sequences:
                params["stop_sequences"] = request.stop_sequences

            # Make API call
            logger.debug(f"Calling Claude API with model={model}")
            message = await self.client.messages.create(**params)

            # Extract text content from response
            # Claude returns content as a list of content blocks
            content = ""
            for block in message.content:
                if hasattr(block, 'text'):
                    content += block.text

            # Get token usage
            usage = message.usage
            prompt_tokens = usage.input_tokens
            completion_tokens = usage.output_tokens
            total_tokens = prompt_tokens + completion_tokens

            # Get stop reason
            finish_reason = message.stop_reason or "unknown"

            # Build response
            response = LLMResponse(
                content=content,
                model=message.model,
                tokens_used=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                provider=self.get_provider_name(),
            )

            logger.debug(
                f"Claude response: tokens={total_tokens} "
                f"(prompt={prompt_tokens}, completion={completion_tokens}), "
                f"finish_reason={finish_reason}"
            )

            return response

        except APIConnectionError as e:
            logger.error(f"Claude API connection error: {e}")
            raise
        except RateLimitError as e:
            logger.warning(f"Claude API rate limit exceeded: {e}")
            raise
        except APIStatusError as e:
            logger.error(
                f"Claude API status error: status={e.status_code}, "
                f"response={e.response}"
            )
            raise
        except APIError as e:
            logger.error(f"Claude API error: {e}")
            raise RuntimeError(f"Claude API error: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in Claude provider: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error: {e}") from e


__all__ = [
    "ClaudeProvider",
]
