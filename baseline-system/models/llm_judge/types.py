"""Data models for LLM provider abstraction layer.

This module defines the core data structures used for LLM interactions:
- LLMRequest: Input parameters for LLM generation
- LLMResponse: Output data from LLM generation
- RetryConfig: Configuration for retry logic
- RateLimitConfig: Configuration for rate limiting
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, model_validator


class LLMRequest(BaseModel):
    """Request parameters for LLM generation.

    Attributes:
        prompt: The main prompt text to send to the LLM
        system_prompt: Optional system prompt to set context/behavior
        temperature: Sampling temperature (0.0-1.0), controls randomness
        max_tokens: Maximum number of tokens to generate
        stop_sequences: Optional list of sequences that stop generation
        model: Optional model name override (provider-specific)
    """

    prompt: str = Field(..., min_length=1, description="Main prompt text")
    system_prompt: Optional[str] = Field(
        None,
        description="System prompt for context/behavior"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Maximum tokens to generate"
    )
    stop_sequences: Optional[List[str]] = Field(
        None,
        description="Sequences that stop generation"
    )
    model: Optional[str] = Field(
        None,
        description="Model name override (provider-specific)"
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """Ensure prompt is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v

    class Config:
        """Pydantic configuration."""
        frozen = False
        validate_assignment = True


class LLMResponse(BaseModel):
    """Response data from LLM generation.

    Attributes:
        content: Generated text content
        model: Model name that generated the response
        tokens_used: Total number of tokens used (prompt + completion)
        prompt_tokens: Number of tokens in the prompt
        completion_tokens: Number of tokens in the completion
        finish_reason: Reason generation stopped (e.g., 'stop', 'length', 'error')
        provider: Name of the LLM provider (e.g., 'anthropic', 'openrouter')
    """

    content: str = Field(..., description="Generated text content")
    model: str = Field(..., description="Model name")
    tokens_used: int = Field(..., ge=0, description="Total tokens used")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Completion tokens")
    finish_reason: str = Field(..., description="Generation stop reason")
    provider: str = Field(..., description="LLM provider name")

    @model_validator(mode='after')
    def validate_tokens_consistency(self):
        """Ensure tokens_used equals prompt_tokens + completion_tokens if both are set."""
        if self.prompt_tokens > 0 and self.completion_tokens > 0:
            expected = self.prompt_tokens + self.completion_tokens
            if self.tokens_used != expected:
                raise ValueError(
                    f"tokens_used ({self.tokens_used}) should equal "
                    f"prompt_tokens + completion_tokens ({expected})"
                )
        return self

    class Config:
        """Pydantic configuration."""
        frozen = True
        validate_assignment = True


class RetryConfig(BaseModel):
    """Configuration for retry logic with exponential backoff.

    Attributes:
        max_attempts: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
    """

    max_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum retry attempts"
    )
    initial_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Initial delay in seconds"
    )
    max_delay: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay in seconds"
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.1,
        le=10.0,
        description="Exponential backoff base"
    )
    jitter: bool = Field(
        default=True,
        description="Add random jitter to delays"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting.

    Attributes:
        requests_per_minute: Maximum requests allowed per minute
        enabled: Whether rate limiting is enabled
    """

    requests_per_minute: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Max requests per minute"
    )
    enabled: bool = Field(
        default=True,
        description="Enable rate limiting"
    )

    class Config:
        """Pydantic configuration."""
        frozen = True


__all__ = [
    "LLMRequest",
    "LLMResponse",
    "RetryConfig",
    "RateLimitConfig",
]
