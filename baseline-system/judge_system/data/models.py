"""Data models for SocialFX dataset and metadata."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Optional, Any, Tuple


class SocialFXExample(BaseModel):
    """Model for a single SocialFX dataset example.

    Attributes:
        id: Unique identifier for the example
        description: Text description of the desired audio effect
        instrument: Type of instrument (guitar, drums, or piano)
        effect_type: Type of effect applied (eq, reverb, or compressor)
        parameters: Dictionary containing effect-specific parameters
        audio_path: Optional path to the corresponding audio file
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    id: int = Field(
        ge=0,
        description="Unique identifier for the example"
    )
    description: str = Field(
        min_length=1,
        description="Text description of the desired audio effect"
    )
    instrument: str = Field(
        description="Type of instrument (guitar, drums, or piano)"
    )
    effect_type: str = Field(
        min_length=1,
        description="Type of effect applied (eq, reverb, or compressor)"
    )
    parameters: Dict[str, Any] = Field(
        description="Dictionary containing effect-specific parameters"
    )
    audio_path: Optional[str] = Field(
        default=None,
        description="Optional path to the corresponding audio file"
    )

    @field_validator('instrument')
    @classmethod
    def validate_instrument(cls, v: str) -> str:
        """Validate instrument is one of the supported types.

        Args:
            v: Instrument type to validate

        Returns:
            Validated instrument type

        Raises:
            ValueError: If instrument is not guitar, drums, or piano
        """
        valid = ['guitar', 'drums', 'piano']
        if v not in valid:
            raise ValueError(f"Instrument must be one of {valid}, got '{v}'")
        return v

    @field_validator('parameters')
    @classmethod
    def validate_parameters_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameters dictionary is not empty.

        Args:
            v: Parameters dictionary to validate

        Returns:
            Validated parameters dictionary

        Raises:
            ValueError: If parameters dictionary is empty
        """
        if not v:
            raise ValueError("Parameters dictionary cannot be empty")
        return v


class DatasetMetadata(BaseModel):
    """Metadata and statistics for the SocialFX dataset.

    Attributes:
        total_examples: Total number of examples in the dataset
        instruments: List of all instrument types in the dataset
        effect_types: List of all effect types in the dataset
        description_count: Count of examples per effect type
        parameter_ranges: Min/max ranges for numeric parameters by effect type
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    total_examples: int = Field(
        ge=0,
        description="Total number of examples in the dataset"
    )
    instruments: List[str] = Field(
        min_length=1,
        description="List of all instrument types in the dataset"
    )
    effect_types: List[str] = Field(
        min_length=1,
        description="List of all effect types in the dataset"
    )
    description_count: Dict[str, int] = Field(
        description="Count of examples per effect type"
    )
    parameter_ranges: Dict[str, Dict[str, Tuple[float, float]]] = Field(
        default_factory=dict,
        description="Min/max ranges for numeric parameters by effect type"
    )

    @field_validator('total_examples')
    @classmethod
    def validate_total_examples(cls, v: int) -> int:
        """Ensure total_examples is non-negative.

        Args:
            v: Total examples count to validate

        Returns:
            Validated total examples count

        Raises:
            ValueError: If total examples is negative
        """
        if v < 0:
            raise ValueError(f"Total examples must be non-negative, got {v}")
        return v

    @field_validator('instruments')
    @classmethod
    def validate_instruments_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure instruments list is not empty.

        Args:
            v: Instruments list to validate

        Returns:
            Validated instruments list

        Raises:
            ValueError: If instruments list is empty
        """
        if not v:
            raise ValueError("Instruments list cannot be empty")
        return v

    @field_validator('effect_types')
    @classmethod
    def validate_effect_types_not_empty(cls, v: List[str]) -> List[str]:
        """Ensure effect types list is not empty.

        Args:
            v: Effect types list to validate

        Returns:
            Validated effect types list

        Raises:
            ValueError: If effect types list is empty
        """
        if not v:
            raise ValueError("Effect types list cannot be empty")
        return v

    @field_validator('description_count')
    @classmethod
    def validate_description_count(cls, v: Dict[str, int]) -> Dict[str, int]:
        """Ensure all counts are non-negative.

        Args:
            v: Description count dictionary to validate

        Returns:
            Validated description count dictionary

        Raises:
            ValueError: If any count is negative
        """
        for effect_type, count in v.items():
            if count < 0:
                raise ValueError(
                    f"Count for effect type '{effect_type}' must be non-negative, got {count}"
                )
        return v

    @field_validator('parameter_ranges')
    @classmethod
    def validate_parameter_ranges(
        cls,
        v: Dict[str, Dict[str, Tuple[float, float]]]
    ) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Ensure all parameter ranges are valid (min <= max).

        Args:
            v: Parameter ranges dictionary to validate

        Returns:
            Validated parameter ranges dictionary

        Raises:
            ValueError: If any range has min > max
        """
        for effect_type, params in v.items():
            for param_name, (min_val, max_val) in params.items():
                if min_val > max_val:
                    raise ValueError(
                        f"Invalid range for {effect_type}.{param_name}: "
                        f"min ({min_val}) > max ({max_val})"
                    )
        return v
