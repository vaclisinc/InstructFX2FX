"""EQ effect parameter models."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal
from .base import BaseEffectParameters


class EQBand(BaseModel):
    """Single EQ band parameters.

    Attributes:
        frequency: Center frequency in Hz (20-20000)
        gain: Gain adjustment in dB (-12 to +12)
        q: Q factor/bandwidth (0.1 to 10)
    """

    frequency: float = Field(
        ge=20,
        le=20000,
        description="Center frequency in Hz"
    )
    gain: float = Field(
        ge=-12,
        le=12,
        description="Gain in dB"
    )
    q: float = Field(
        ge=0.1,
        le=10,
        description="Q factor (bandwidth)"
    )

    @field_validator('frequency')
    @classmethod
    def validate_frequency(cls, v: float) -> float:
        """Ensure frequency is within audible range."""
        if v < 20 or v > 20000:
            raise ValueError(f"Frequency {v} Hz is outside audible range (20-20000 Hz)")
        return v


class EQParameters(BaseEffectParameters):
    """EQ effect parameters.

    Attributes:
        effect_type: Must be "eq"
        bands: List of EQ bands (3-10 bands)
        eq_type: Type of EQ (parametric, graphic, shelving)
    """

    effect_type: Literal["eq"] = "eq"
    bands: List[EQBand] = Field(
        min_length=3,
        max_length=10,
        description="List of EQ bands"
    )
    eq_type: Literal["parametric", "graphic", "shelving"] = Field(
        default="parametric",
        description="Type of EQ"
    )

    @field_validator('bands')
    @classmethod
    def validate_bands(cls, v: List[EQBand]) -> List[EQBand]:
        """Ensure bands are sorted by frequency and don't overlap excessively."""
        if len(v) < 3:
            raise ValueError("EQ must have at least 3 bands")
        if len(v) > 10:
            raise ValueError("EQ cannot have more than 10 bands")

        # Sort bands by frequency
        sorted_bands = sorted(v, key=lambda b: b.frequency)

        # Check for excessive overlap
        for i in range(len(sorted_bands) - 1):
            freq_ratio = sorted_bands[i + 1].frequency / sorted_bands[i].frequency
            if freq_ratio < 1.1:
                raise ValueError(
                    f"EQ bands at {sorted_bands[i].frequency} Hz and "
                    f"{sorted_bands[i + 1].frequency} Hz are too close"
                )

        return sorted_bands
