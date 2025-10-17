"""Models module for baseline system.

This module provides data models for the baseline system including
parameter models for audio effects and refinement system models.
"""

from .parameters import (
    BaseEffectParameters,
    EQBand,
    EQParameters,
    ReverbParameters,
    CompressorParameters,
    EffectChain,
    EffectParameter,
)
from .refinement import (
    IterationResult,
    RefinementConfig,
    RefinementResult,
)

__all__ = [
    "BaseEffectParameters",
    "EQBand",
    "EQParameters",
    "ReverbParameters",
    "CompressorParameters",
    "EffectChain",
    "EffectParameter",
    "IterationResult",
    "RefinementConfig",
    "RefinementResult",
]
