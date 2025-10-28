"""Baseline system for iterative audio effect parameter refinement.

This module provides the core refinement loop controller that orchestrates
the closed-loop system between parameter generation and scoring.
"""

from .refinement_controller import RefinementLoopController

__all__ = [
    "RefinementLoopController",
]
