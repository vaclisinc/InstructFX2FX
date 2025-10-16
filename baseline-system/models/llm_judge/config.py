"""Configuration schema for LLM provider system.

This module defines the configuration structures for:
- Provider selection and initialization
- Retry logic and error handling
- Rate limiting
- Timeout settings
- Model-specific parameters

All configuration uses Pydantic for validation and type safety.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from .types import RetryConfig, RateLimitConfig


class ProviderConfig(BaseModel):
    """Base configuration for LLM providers.

    Attributes:
        provider: Provider type ('anthropic' or 'openrouter')
        api_key: API key for authentication
        model: Model name to use (provider-specific)
        timeout: Request timeout in seconds
        retry: Retry configuration
        rate_limit: Rate limiting configuration
        extra: Additional provider-specific settings
    """

    provider: Literal["anthropic", "openrouter"] = Field(
        default="anthropic",
        description="LLM provider type"
    )
    api_key: str = Field(
        ...,
        min_length=1,
        description="API key for authentication"
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model name (provider-specific)"
    )
    timeout: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Request timeout in seconds"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retry configuration"
    )
    rate_limit: RateLimitConfig = Field(
        default_factory=RateLimitConfig,
        description="Rate limiting configuration"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific settings"
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        v = v.strip()
        if not v:
            raise ValueError("API key cannot be empty or whitespace only")
        if len(v) < 10:
            raise ValueError("API key appears too short to be valid")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model name format."""
        v = v.strip()
        if not v:
            raise ValueError("Model name cannot be empty or whitespace only")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary format.

        Returns:
            Dictionary representation of configuration
        """
        return self.model_dump()

    class Config:
        """Pydantic configuration."""
        frozen = False
        validate_assignment = True


class AnthropicConfig(ProviderConfig):
    """Configuration specific to Anthropic Claude provider.

    Attributes:
        provider: Fixed as 'anthropic'
        model: Claude model name (e.g., 'claude-3-5-sonnet-20241022')
        anthropic_version: API version string
        max_tokens_default: Default max tokens for requests
    """

    provider: Literal["anthropic"] = Field(
        default="anthropic",
        frozen=True,
        description="Provider type (fixed as 'anthropic')"
    )
    model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Claude model name"
    )
    anthropic_version: str = Field(
        default="2023-06-01",
        description="Anthropic API version"
    )
    max_tokens_default: int = Field(
        default=4096,
        ge=1,
        le=200000,
        description="Default max tokens for requests"
    )

    @field_validator("model")
    @classmethod
    def validate_claude_model(cls, v: str) -> str:
        """Validate Claude model name."""
        v = v.strip().lower()
        if not v.startswith("claude-"):
            raise ValueError("Anthropic model name must start with 'claude-'")
        return v

    class Config:
        """Pydantic configuration."""
        frozen = False
        validate_assignment = True


class OpenRouterConfig(ProviderConfig):
    """Configuration specific to OpenRouter provider.

    Attributes:
        provider: Fixed as 'openrouter'
        model: Model identifier (e.g., 'anthropic/claude-3.5-sonnet')
        base_url: OpenRouter API base URL
        site_url: Optional site URL for OpenRouter dashboard
        app_name: Optional app name for OpenRouter dashboard
    """

    provider: Literal["openrouter"] = Field(
        default="openrouter",
        frozen=True,
        description="Provider type (fixed as 'openrouter')"
    )
    model: str = Field(
        ...,
        description="OpenRouter model identifier"
    )
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    site_url: Optional[str] = Field(
        None,
        description="Site URL for OpenRouter dashboard"
    )
    app_name: Optional[str] = Field(
        None,
        description="App name for OpenRouter dashboard"
    )

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate base URL format."""
        v = v.strip()
        if not v.startswith("http"):
            raise ValueError("Base URL must start with http:// or https://")
        return v.rstrip("/")

    class Config:
        """Pydantic configuration."""
        frozen = False
        validate_assignment = True


class LLMConfig(BaseModel):
    """Top-level configuration for LLM system.

    This is the main configuration object that can be loaded from
    files or environment variables.

    Attributes:
        provider_config: Provider-specific configuration
        log_requests: Whether to log request details
        log_responses: Whether to log response details
        log_tokens: Whether to log token usage
    """

    provider_config: ProviderConfig = Field(
        ...,
        description="Provider configuration"
    )
    log_requests: bool = Field(
        default=True,
        description="Log request details"
    )
    log_responses: bool = Field(
        default=True,
        description="Log response details"
    )
    log_tokens: bool = Field(
        default=True,
        description="Log token usage"
    )

    @model_validator(mode='after')
    def validate_provider_config_type(self):
        """Ensure provider_config matches its declared provider type."""
        provider_type = self.provider_config.provider

        if provider_type == "anthropic" and not isinstance(self.provider_config, AnthropicConfig):
            # Convert to AnthropicConfig if needed
            config_dict = self.provider_config.to_dict()
            self.provider_config = AnthropicConfig(**config_dict)
        elif provider_type == "openrouter" and not isinstance(self.provider_config, OpenRouterConfig):
            # Convert to OpenRouterConfig if needed
            config_dict = self.provider_config.to_dict()
            self.provider_config = OpenRouterConfig(**config_dict)

        return self

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LLMConfig":
        """Create LLMConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            LLMConfig instance

        Raises:
            ValueError: If configuration is invalid
        """
        provider_type = config.get("provider_config", {}).get("provider", "anthropic")

        # Create appropriate provider config
        provider_data = config.get("provider_config", {})
        if provider_type == "anthropic":
            provider_config = AnthropicConfig(**provider_data)
        elif provider_type == "openrouter":
            provider_config = OpenRouterConfig(**provider_data)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

        # Create main config
        return cls(
            provider_config=provider_config,
            log_requests=config.get("log_requests", True),
            log_responses=config.get("log_responses", True),
            log_tokens=config.get("log_tokens", True),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns:
            Dictionary representation
        """
        return self.model_dump()

    class Config:
        """Pydantic configuration."""
        frozen = False
        validate_assignment = True


__all__ = [
    "ProviderConfig",
    "AnthropicConfig",
    "OpenRouterConfig",
    "LLMConfig",
]
