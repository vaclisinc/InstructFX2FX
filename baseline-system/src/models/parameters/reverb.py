"""Reverb effect parameter models."""

from pydantic import Field, field_validator
from typing import Literal
from .base import BaseEffectParameters


class ReverbParameters(BaseEffectParameters):
    """Reverb effect parameters.

    Attributes:
        effect_type: Must be "reverb"
        room_size: Room size (0-1, 0=small, 1=large)
        damping: High frequency damping (0-1, 0=none, 1=full)
        wet_level: Wet signal level (0-1)
        dry_level: Dry signal level (0-1)
        width: Stereo width (0-1, 0=mono, 1=full stereo)
        freeze_mode: Freeze mode (infinite reverb tail)
    """

    effect_type: Literal["reverb"] = "reverb"

    room_size: float = Field(
        ge=0,
        le=1,
        description="Room size (0=small, 1=large)"
    )
    damping: float = Field(
        ge=0,
        le=1,
        description="High frequency damping (0=none, 1=full)"
    )
    wet_level: float = Field(
        ge=0,
        le=1,
        description="Wet signal level"
    )
    dry_level: float = Field(
        ge=0,
        le=1,
        description="Dry signal level"
    )
    width: float = Field(
        ge=0,
        le=1,
        description="Stereo width (0=mono, 1=full stereo)"
    )
    freeze_mode: bool = Field(
        default=False,
        description="Freeze mode (infinite reverb tail)"
    )

    @field_validator('wet_level', 'dry_level')
    @classmethod
    def validate_levels(cls, v: float) -> float:
        """Ensure levels are within valid range."""
        if v < 0 or v > 1:
            raise ValueError(f"Level {v} must be between 0 and 1")
        return v

    def __init__(self, **data):
        """Initialize reverb parameters with validation."""
        super().__init__(**data)

        # Validate wet/dry balance
        if self.wet_level + self.dry_level > 2.0:
            raise ValueError(
                f"Combined wet ({self.wet_level}) and dry ({self.dry_level}) "
                "levels should not exceed 2.0 for reasonable mixing"
            )
