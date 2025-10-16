"""Parameter generation module.

This module provides functionality for generating audio effect parameters
from high-level textual descriptions using Large Language Models.

Key Components:
- ParameterGenerator: Main class for parameter generation
- Custom exceptions for error handling

Usage:
    from src.generation import ParameterGenerator
    from models.llm_judge import ClaudeProvider

    # Create provider and generator
    provider = ClaudeProvider(config)
    generator = ParameterGenerator(provider)

    # Generate parameters
    effect_chain = await generator.generate_parameters(
        description="warm and intimate vocal sound",
        effects=["eq", "reverb"]
    )
"""

from .parameter_generator import ParameterGenerator
from .exceptions import (
    ParameterGenerationError,
    JSONParseError,
    ValidationError,
    LLMProviderError,
    PromptTemplateError
)

__all__ = [
    "ParameterGenerator",
    "ParameterGenerationError",
    "JSONParseError",
    "ValidationError",
    "LLMProviderError",
    "PromptTemplateError",
]
