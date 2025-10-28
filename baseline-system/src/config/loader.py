"""Configuration loader with YAML support and pydantic validation.

This module provides a configuration management system that:
- Loads YAML configuration files
- Validates configuration using pydantic models
- Supports environment variable overrides
- Enables multiple configuration profiles (dev, test, prod)
- Provides hot-reloading capabilities
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    sample_rate: int = Field(default=44100, ge=8000, le=192000)
    audio_dir: Path = Field(default=Path("./audio_samples"))
    max_duration: float = Field(default=30.0, gt=0)

    @field_validator("audio_dir")
    @classmethod
    def validate_audio_dir(cls, v: Path) -> Path:
        """Ensure audio directory is a Path object."""
        return Path(v) if not isinstance(v, Path) else v


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = Field(default="anthropic")
    model: str = Field(default="claude-3-5-sonnet-20241022")
    api_key: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate LLM provider."""
        allowed = ["anthropic", "openai", "openrouter"]
        if v.lower() not in allowed:
            raise ValueError(f"Provider must be one of {allowed}")
        return v.lower()


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="INFO")
    format: str = Field(default="json")
    output_dir: Path = Field(default=Path("./logs"))
    console_output: bool = Field(default=True)
    file_output: bool = Field(default=True)

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        """Validate log level."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.upper()

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate log format."""
        allowed = ["json", "console"]
        if v.lower() not in allowed:
            raise ValueError(f"Log format must be one of {allowed}")
        return v.lower()

    @field_validator("output_dir")
    @classmethod
    def validate_output_dir(cls, v: Path) -> Path:
        """Ensure output directory is a Path object."""
        return Path(v) if not isinstance(v, Path) else v


class ExperimentConfig(BaseModel):
    """Experiment-specific configuration."""

    name: str = Field(default="baseline_experiment")
    description: str = Field(default="")
    dataset: str = Field(default="CAL500")
    batch_size: int = Field(default=8, ge=1)
    num_iterations: int = Field(default=5, ge=1)
    save_results: bool = Field(default=True)
    results_dir: Path = Field(default=Path("./results"))

    @field_validator("results_dir")
    @classmethod
    def validate_results_dir(cls, v: Path) -> Path:
        """Ensure results directory is a Path object."""
        return Path(v) if not isinstance(v, Path) else v


class Config(BaseModel):
    """Main configuration model."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    profile: str = Field(default="dev")

    @field_validator("profile")
    @classmethod
    def validate_profile(cls, v: str) -> str:
        """Validate configuration profile."""
        allowed = ["dev", "test", "prod"]
        if v.lower() not in allowed:
            raise ValueError(f"Profile must be one of {allowed}")
        return v.lower()


class ConfigLoader:
    """Configuration loader with hot-reloading support."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader.

        Args:
            config_path: Path to configuration file. If None, uses CONFIG_PATH env var
                        or defaults to 'configs/default.yaml'
        """
        self._config_path = self._resolve_config_path(config_path)
        self._config: Optional[Config] = None
        self._last_mtime: Optional[float] = None

    @staticmethod
    def _resolve_config_path(config_path: Optional[str] = None) -> Path:
        """Resolve configuration file path.

        Priority:
        1. Provided config_path parameter
        2. CONFIG_PATH environment variable
        3. Default: configs/default.yaml

        Args:
            config_path: Optional path to config file

        Returns:
            Resolved Path object
        """
        if config_path:
            return Path(config_path)

        env_path = os.getenv("CONFIG_PATH")
        if env_path:
            return Path(env_path)

        return Path("configs/default.yaml")

    def load(self, force_reload: bool = False) -> Config:
        """Load configuration from YAML file.

        Args:
            force_reload: Force reload even if file hasn't changed

        Returns:
            Validated Config object

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValidationError: If config validation fails
            yaml.YAMLError: If YAML parsing fails
        """
        if not self._config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self._config_path}")

        current_mtime = self._config_path.stat().st_mtime

        # Return cached config if file hasn't changed
        if not force_reload and self._config and self._last_mtime == current_mtime:
            return self._config

        # Load YAML file
        with open(self._config_path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        # Apply environment variable overrides
        data = self._apply_env_overrides(data)

        # Validate and create config
        try:
            self._config = Config.model_validate(data)
            self._last_mtime = current_mtime
            return self._config
        except ValidationError as e:
            raise ValidationError.from_exception_data(
                title="Configuration validation failed",
                line_errors=e.errors()
            )

    def _apply_env_overrides(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to configuration.

        Environment variables are prefixed with the section name:
        - AUDIO_SAMPLE_RATE overrides audio.sample_rate
        - LLM_API_KEY overrides llm.api_key
        - LOGGING_LEVEL overrides logging.level

        Args:
            data: Configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        # Audio overrides
        if "audio" not in data:
            data["audio"] = {}
        if sample_rate := os.getenv("AUDIO_SAMPLE_RATE"):
            data["audio"]["sample_rate"] = int(sample_rate)
        if audio_dir := os.getenv("AUDIO_DIR"):
            data["audio"]["audio_dir"] = audio_dir

        # LLM overrides
        if "llm" not in data:
            data["llm"] = {}
        if api_key := os.getenv("ANTHROPIC_API_KEY"):
            data["llm"]["api_key"] = api_key
        elif api_key := os.getenv("OPENROUTER_API_KEY"):
            data["llm"]["api_key"] = api_key
        if provider := os.getenv("LLM_PROVIDER"):
            data["llm"]["provider"] = provider
        if model := os.getenv("LLM_MODEL"):
            data["llm"]["model"] = model
        if temperature := os.getenv("LLM_TEMPERATURE"):
            data["llm"]["temperature"] = float(temperature)

        # Logging overrides
        if "logging" not in data:
            data["logging"] = {}
        if log_level := os.getenv("LOG_LEVEL"):
            data["logging"]["level"] = log_level
        if log_format := os.getenv("LOG_FORMAT"):
            data["logging"]["format"] = log_format
        if output_dir := os.getenv("LOG_OUTPUT_DIR"):
            data["logging"]["output_dir"] = output_dir

        return data

    def reload(self) -> Config:
        """Force reload configuration from file.

        Returns:
            Newly loaded Config object
        """
        return self.load(force_reload=True)

    def has_changed(self) -> bool:
        """Check if configuration file has been modified.

        Returns:
            True if file has been modified since last load
        """
        if not self._config_path.exists():
            return False

        if self._last_mtime is None:
            return True

        current_mtime = self._config_path.stat().st_mtime
        return current_mtime != self._last_mtime

    @property
    def config(self) -> Optional[Config]:
        """Get current configuration without reloading.

        Returns:
            Current Config object or None if not loaded
        """
        return self._config


# Global configuration instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """Get or create global configuration loader instance.

    Args:
        config_path: Optional path to config file

    Returns:
        ConfigLoader instance
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def load_config(config_path: Optional[str] = None, force_reload: bool = False) -> Config:
    """Load configuration using global loader.

    Args:
        config_path: Optional path to config file
        force_reload: Force reload even if file hasn't changed

    Returns:
        Validated Config object
    """
    loader = get_config_loader(config_path)
    return loader.load(force_reload=force_reload)


def reload_config() -> Config:
    """Reload configuration from file.

    Returns:
        Newly loaded Config object
    """
    if _config_loader is None:
        raise RuntimeError("Configuration not initialized. Call load_config() first.")
    return _config_loader.reload()
