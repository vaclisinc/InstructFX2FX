"""Parameter generation module.

This module provides functionality for generating audio effect parameters
from high-level textual descriptions using Large Language Models.

Key Components:
- ParameterGenerator: Main class for parameter generation
- Validator: Pre and post-validation utilities
- Normalizer: Parameter normalization and correction
- Custom exceptions for error handling

Usage:
    from src.generation import ParameterGenerator, normalize_effect_chain
    from models.llm_judge import ClaudeProvider

    # Create provider and generator
    provider = ClaudeProvider(config)
    generator = ParameterGenerator(provider)

    # Generate parameters
    effect_chain = await generator.generate_parameters(
        description="warm and intimate vocal sound",
        effects=["eq", "reverb"]
    )

    # Normalize if needed
    normalized_chain = normalize_effect_chain(effect_chain)
"""

from .parameter_generator import ParameterGenerator
from .exceptions import (
    ParameterGenerationError,
    JSONParseError,
    ValidationError,
    LLMProviderError,
    PromptTemplateError
)
from .validator import (
    ValidationLevel,
    ValidationIssue,
    ValidationResult,
    validate_effect_structure,
    validate_effect_chain_structure,
    validate_effect_parameter,
    validate_effect_chain,
    PARAMETER_RANGES
)
from .normalizer import (
    clamp,
    safe_float,
    safe_bool,
    normalize_effect,
    normalize_effect_chain_data,
    normalize_effect_chain,
)

__all__ = [
    # Core classes
    "ParameterGenerator",
    # Exceptions
    "ParameterGenerationError",
    "JSONParseError",
    "ValidationError",
    "LLMProviderError",
    "PromptTemplateError",
    # Validation
    "ValidationLevel",
    "ValidationIssue",
    "ValidationResult",
    "validate_effect_structure",
    "validate_effect_chain_structure",
    "validate_effect_parameter",
    "validate_effect_chain",
    "PARAMETER_RANGES",
    # Normalization
    "clamp",
    "safe_float",
    "safe_bool",
    "normalize_effect",
    "normalize_effect_chain_data",
    "normalize_effect_chain",
]
