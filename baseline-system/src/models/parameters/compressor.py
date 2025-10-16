"""Compressor effect parameter models."""

from pydantic import Field, field_validator
from typing import Literal
from .base import BaseEffectParameters


class CompressorParameters(BaseEffectParameters):
    """Compressor effect parameters.

    Attributes:
        effect_type: Must be "compressor"
        threshold: Threshold in dB (-60 to 0)
        ratio: Compression ratio (1 to 20, 1=no compression, 20=limiting)
        attack: Attack time in milliseconds (0.1 to 100)
        release: Release time in milliseconds (10 to 1000)
        knee: Knee width in dB (0 to 12, 0=hard knee, 12=soft knee)
        makeup_gain: Makeup gain in dB (0 to 24)
    """

    effect_type: Literal["compressor"] = "compressor"

    threshold: float = Field(
        ge=-60,
        le=0,
        description="Threshold in dB"
    )
    ratio: float = Field(
        ge=1,
        le=20,
        description="Compression ratio (1=no compression, 20=limiting)"
    )
    attack: float = Field(
        ge=0.1,
        le=100,
        description="Attack time in milliseconds"
    )
    release: float = Field(
        ge=10,
        le=1000,
        description="Release time in milliseconds"
    )
    knee: float = Field(
        ge=0,
        le=12,
        description="Knee width in dB (0=hard knee, 12=soft knee)"
    )
    makeup_gain: float = Field(
        ge=0,
        le=24,
        description="Makeup gain in dB"
    )

    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        """Ensure threshold is within valid range."""
        if v < -60 or v > 0:
            raise ValueError(f"Threshold {v} dB must be between -60 and 0 dB")
        return v

    @field_validator('ratio')
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        """Ensure ratio is valid."""
        if v < 1:
            raise ValueError(f"Ratio {v} must be at least 1:1 (no compression)")
        if v > 20:
            raise ValueError(f"Ratio {v} exceeds maximum of 20:1")
        return v

    @field_validator('attack')
    @classmethod
    def validate_attack(cls, v: float) -> float:
        """Ensure attack time is reasonable."""
        if v < 0.1:
            raise ValueError(f"Attack time {v} ms is too fast (minimum 0.1 ms)")
        if v > 100:
            raise ValueError(f"Attack time {v} ms is too slow (maximum 100 ms)")
        return v

    @field_validator('release')
    @classmethod
    def validate_release(cls, v: float) -> float:
        """Ensure release time is reasonable."""
        if v < 10:
            raise ValueError(f"Release time {v} ms is too fast (minimum 10 ms)")
        if v > 1000:
            raise ValueError(f"Release time {v} ms is too slow (maximum 1000 ms)")
        return v

    def __init__(self, **data):
        """Initialize compressor parameters with validation."""
        super().__init__(**data)

        # Validate attack/release relationship
        if self.attack >= self.release:
            raise ValueError(
                f"Attack time ({self.attack} ms) should be shorter than "
                f"release time ({self.release} ms) for natural compression"
            )
