"""Data models for SocialFX dataset loading and validation.

This module defines Pydantic models for representing SocialFX dataset examples
and metadata, with validation for instrument types and parameter ranges.
"""

from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, field_validator


class SocialFXExample(BaseModel):
    """Represents a single example from the SocialFX dataset.

    Each example includes an audio effect description, the instrument it applies to,
    the effect type (eq, reverb, compressor), and the actual effect parameters.

    Attributes:
        id: Unique identifier for this example
        description: Text description of the desired audio effect (e.g., "warm and intimate")
        instrument: Type of instrument - must be guitar, drums, or piano
        effect_type: Type of audio effect (eq, reverb, compressor)
        parameters: Dictionary containing effect-specific parameters
        audio_path: Optional path to the reference audio file
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    id: int
    description: str
    instrument: str
    effect_type: str
    parameters: Dict
    audio_path: Optional[str] = None

    @field_validator('instrument')
    @classmethod
    def validate_instrument(cls, v: str) -> str:
        """Validate that instrument is one of the allowed types.

        Args:
            v: Instrument name to validate

        Returns:
            The validated instrument name

        Raises:
            ValueError: If instrument is not guitar, drums, or piano
        """
        valid = ['guitar', 'drums', 'piano']
        if v not in valid:
            raise ValueError(f"Instrument must be one of {valid}, got '{v}'")
        return v

    @field_validator('parameters')
    @classmethod
    def validate_parameters(cls, v: Dict) -> Dict:
        """Validate that parameters dictionary is not empty.

        Args:
            v: Parameters dictionary to validate

        Returns:
            The validated parameters dictionary

        Raises:
            ValueError: If parameters dictionary is empty
        """
        if not v:
            raise ValueError("Parameters dictionary cannot be empty")
        return v


class DatasetMetadata(BaseModel):
    """Metadata and statistics about the SocialFX dataset.

    This model captures high-level information about the dataset including
    counts, supported instruments and effects, and parameter value ranges.

    Attributes:
        total_examples: Total number of examples in the dataset
        instruments: List of instrument types present in the dataset
        effect_types: List of effect types present in the dataset
        description_count: Count of examples per effect type
        parameter_ranges: Min/max ranges for each parameter by effect type
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    total_examples: int
    instruments: List[str]
    effect_types: List[str]
    description_count: Dict[str, int]
    parameter_ranges: Dict[str, Dict[str, Tuple[float, float]]]

    @field_validator('total_examples')
    @classmethod
    def validate_total_examples(cls, v: int) -> int:
        """Validate that total_examples is non-negative.

        Args:
            v: Total examples count

        Returns:
            The validated count

        Raises:
            ValueError: If count is negative
        """
        if v < 0:
            raise ValueError(f"total_examples must be non-negative, got {v}")
        return v

    @field_validator('instruments')
    @classmethod
    def validate_instruments(cls, v: List[str]) -> List[str]:
        """Validate that instruments list is not empty.

        Args:
            v: List of instruments

        Returns:
            The validated list

        Raises:
            ValueError: If list is empty
        """
        if not v:
            raise ValueError("instruments list cannot be empty")
        return v

    @field_validator('effect_types')
    @classmethod
    def validate_effect_types(cls, v: List[str]) -> List[str]:
        """Validate that effect_types list is not empty.

        Args:
            v: List of effect types

        Returns:
            The validated list

        Raises:
            ValueError: If list is empty
        """
        if not v:
            raise ValueError("effect_types list cannot be empty")
        return v

    @field_validator('description_count')
    @classmethod
    def validate_description_count(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Validate that all description counts are non-negative.

        Args:
            v: Dictionary of effect type to count

        Returns:
            The validated dictionary

        Raises:
            ValueError: If any count is negative
        """
        for effect_type, count in v.items():
            if count < 0:
                raise ValueError(
                    f"description_count for '{effect_type}' must be non-negative, got {count}"
                )
        return v

    @field_validator('parameter_ranges')
    @classmethod
    def validate_parameter_ranges(cls, v: Dict[str, Dict[str, Tuple[float, float]]]) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Validate that parameter ranges have min <= max.

        Args:
            v: Dictionary of effect type to parameter ranges

        Returns:
            The validated dictionary

        Raises:
            ValueError: If any range has min > max
        """
        for effect_type, params in v.items():
            for param_name, (min_val, max_val) in params.items():
                if min_val > max_val:
                    raise ValueError(
                        f"parameter_ranges for '{effect_type}.{param_name}' has min > max: "
                        f"{min_val} > {max_val}"
                    )
        return v
