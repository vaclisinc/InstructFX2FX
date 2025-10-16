"""Parameter models for audio effects.

This module provides Pydantic models for validating audio effect parameters
including EQ, Reverb, and Compressor effects. All models include comprehensive
validation to ensure parameters are within valid ranges.

Classes:
    BaseEffectParameters: Base class for all effect parameters
    EQBand: Single EQ band parameters
    EQParameters: EQ effect parameters with multiple bands
    ReverbParameters: Reverb effect parameters
    CompressorParameters: Compressor effect parameters
    EffectChain: Chain of multiple effects with execution order

Example:
    >>> from src.models.parameters import EQParameters, EQBand
    >>> eq = EQParameters(
    ...     bands=[
    ...         EQBand(frequency=1000, gain=3, q=1.0),
    ...         EQBand(frequency=5000, gain=-2, q=0.7),
    ...         EQBand(frequency=10000, gain=1, q=1.2)
    ...     ]
    ... )
    >>> print(eq.to_dict())
"""

from .base import BaseEffectParameters
from .eq import EQBand, EQParameters
from .reverb import ReverbParameters
from .compressor import CompressorParameters
from .effect_chain import EffectChain, EffectParameter

__all__ = [
    "BaseEffectParameters",
    "EQBand",
    "EQParameters",
    "ReverbParameters",
    "CompressorParameters",
    "EffectChain",
    "EffectParameter",
]
