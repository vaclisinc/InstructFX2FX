"""Effect chain model for combining multiple effects."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Union, Literal
from .eq import EQParameters
from .reverb import ReverbParameters
from .compressor import CompressorParameters


# Type alias for any effect parameter
EffectParameter = Union[EQParameters, ReverbParameters, CompressorParameters]


class EffectChain(BaseModel):
    """Chain of audio effects to be applied in sequence.

    Attributes:
        description: Human-readable description of the effect chain
        effects: List of effect parameters
        order: Effect execution order (list of effect types)
    """

    description: str = Field(
        min_length=1,
        description="Description of the desired audio characteristics"
    )
    effects: List[EffectParameter] = Field(
        min_length=1,
        max_length=10,
        description="List of effect parameters"
    )
    order: List[Literal["eq", "reverb", "compressor"]] = Field(
        min_length=1,
        max_length=10,
        description="Effect execution order"
    )

    @field_validator('effects')
    @classmethod
    def validate_effects(cls, v: List[EffectParameter]) -> List[EffectParameter]:
        """Ensure effects list is not empty and not too long."""
        if len(v) == 0:
            raise ValueError("Effect chain must contain at least one effect")
        if len(v) > 10:
            raise ValueError("Effect chain cannot contain more than 10 effects")
        return v

    @field_validator('order')
    @classmethod
    def validate_order(cls, v: List[str]) -> List[str]:
        """Ensure order list contains valid effect types."""
        valid_types = {"eq", "reverb", "compressor"}
        for effect_type in v:
            if effect_type not in valid_types:
                raise ValueError(
                    f"Invalid effect type '{effect_type}'. "
                    f"Must be one of: {', '.join(valid_types)}"
                )
        return v

    def __init__(self, **data):
        """Initialize effect chain with validation."""
        super().__init__(**data)

        # Validate that order matches effects
        if len(self.order) != len(self.effects):
            raise ValueError(
                f"Order length ({len(self.order)}) must match "
                f"effects length ({len(self.effects)})"
            )

        # Validate that effect types in order match actual effects
        for i, (effect, effect_type) in enumerate(zip(self.effects, self.order)):
            if effect.effect_type != effect_type:
                raise ValueError(
                    f"Effect at position {i} has type '{effect.effect_type}' "
                    f"but order specifies '{effect_type}'"
                )

    def get_effect_by_type(self, effect_type: str) -> List[EffectParameter]:
        """Get all effects of a specific type.

        Args:
            effect_type: Type of effect to retrieve

        Returns:
            List of effects matching the specified type
        """
        return [e for e in self.effects if e.effect_type == effect_type]

    def to_dict(self) -> dict:
        """Convert effect chain to dictionary.

        Returns:
            Dictionary representation of the effect chain
        """
        return {
            "description": self.description,
            "effects": [e.to_dict() for e in self.effects],
            "order": self.order
        }
