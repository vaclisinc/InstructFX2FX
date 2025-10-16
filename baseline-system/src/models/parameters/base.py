"""Base models for effect parameters."""

from pydantic import BaseModel, ConfigDict
from typing import Literal


class BaseEffectParameters(BaseModel):
    """Base class for all effect parameters."""

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    effect_type: str

    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        return self.model_dump()
