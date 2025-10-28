"""Refinement system data models.

This module provides Pydantic models for the iterative refinement system,
including iteration results, refinement configuration, and final results.

Classes:
    IterationResult: Individual iteration result with parameters and feedback
    RefinementConfig: Configuration for refinement loop behavior
    RefinementResult: Complete refinement result with full iteration history

Example:
    >>> from src.models.refinement import RefinementConfig, IterationResult
    >>> config = RefinementConfig(
    ...     max_iterations=10,
    ...     min_score_improvement=2.0,
    ...     early_stop_score=90.0
    ... )
    >>> iteration = IterationResult(
    ...     iteration=0,
    ...     parameters={"reverb": {"decay": 0.5}},
    ...     score=75.0,
    ...     feedback="Good reverb decay",
    ...     suggestions=["Increase wet/dry mix"],
    ...     timestamp="2025-10-17T12:00:00Z"
    ... )
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Dict, List, Optional, Any, Literal


class IterationResult(BaseModel):
    """Individual iteration result in refinement loop.

    Attributes:
        iteration: Iteration number (0-indexed)
        parameters: Generated effect parameters as dictionary
        score: Overall score for this iteration (0-100 scale)
        feedback: Qualitative feedback about the parameters
        suggestions: List of specific suggestions for improvement
        timestamp: ISO 8601 timestamp of iteration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    iteration: int = Field(
        ge=0,
        description="Iteration number (0-indexed)"
    )
    parameters: Dict[str, Any] = Field(
        description="Generated effect parameters as dictionary"
    )
    score: float = Field(
        ge=0,
        le=100,
        description="Overall score for this iteration (0-100 scale)"
    )
    feedback: str = Field(
        min_length=1,
        description="Qualitative feedback about the parameters"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="List of specific suggestions for improvement"
    )
    timestamp: str = Field(
        min_length=1,
        description="ISO 8601 timestamp of iteration"
    )

    @field_validator('parameters')
    @classmethod
    def validate_parameters_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameters dictionary is not empty."""
        if not v:
            raise ValueError("Parameters dictionary cannot be empty")
        return v

    @field_validator('score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure score is within valid range [0, 100]."""
        if v < 0 or v > 100:
            raise ValueError(f"Score {v} must be between 0 and 100")
        return v


class RefinementConfig(BaseModel):
    """Configuration for refinement loop behavior.

    Attributes:
        max_iterations: Maximum number of refinement iterations
        min_score_improvement: Minimum score improvement to continue (0-100 scale)
        convergence_window: Number of recent iterations to check for convergence
        mode: Refinement mode - "parameter_only" or "audio_based"
        temperature_schedule: Optional list of temperature values per iteration
        early_stop_score: Optional score threshold to stop refinement early (0-100 scale)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    max_iterations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of refinement iterations"
    )
    min_score_improvement: float = Field(
        default=2.0,
        ge=0,
        le=100,
        description="Minimum score improvement to continue (0-100 scale)"
    )
    convergence_window: int = Field(
        default=3,
        ge=2,
        le=10,
        description="Number of recent iterations to check for convergence"
    )
    mode: Literal["parameter_only", "audio_based"] = Field(
        default="parameter_only",
        description="Refinement mode - parameter_only or audio_based"
    )
    temperature_schedule: Optional[List[float]] = Field(
        default=None,
        description="Optional list of temperature values per iteration"
    )
    early_stop_score: Optional[float] = Field(
        default=90.0,
        ge=0,
        le=100,
        description="Optional score threshold to stop refinement early (0-100 scale)"
    )

    @field_validator('max_iterations')
    @classmethod
    def validate_max_iterations(cls, v: int) -> int:
        """Ensure max_iterations is reasonable."""
        if v < 1:
            raise ValueError("max_iterations must be at least 1")
        if v > 100:
            raise ValueError("max_iterations cannot exceed 100")
        return v

    @field_validator('convergence_window')
    @classmethod
    def validate_convergence_window(cls, v: int) -> int:
        """Ensure convergence_window is valid."""
        if v < 2:
            raise ValueError("convergence_window must be at least 2")
        if v > 10:
            raise ValueError("convergence_window cannot exceed 10")
        return v

    @field_validator('temperature_schedule')
    @classmethod
    def validate_temperature_schedule(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Ensure temperature values are valid."""
        if v is not None:
            if not v:
                raise ValueError("temperature_schedule cannot be empty if provided")
            for temp in v:
                if temp < 0 or temp > 2.0:
                    raise ValueError(f"Temperature {temp} must be between 0 and 2.0")
        return v

    @field_validator('early_stop_score')
    @classmethod
    def validate_early_stop_score(cls, v: Optional[float]) -> Optional[float]:
        """Ensure early_stop_score is within valid range."""
        if v is not None:
            if v < 0 or v > 100:
                raise ValueError(f"early_stop_score {v} must be between 0 and 100")
        return v


class RefinementResult(BaseModel):
    """Complete refinement result with full iteration history.

    Attributes:
        description: Original user description of desired audio effect
        initial_parameters: Starting parameters before refinement
        final_parameters: Best parameters found during refinement
        iterations: List of all iteration results
        total_iterations: Total number of iterations performed
        final_score: Best score achieved (0-100 scale)
        improvement: Score improvement from initial to final (can be negative)
        convergence_reason: Reason why refinement stopped
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    description: str = Field(
        min_length=1,
        description="Original user description of desired audio effect"
    )
    initial_parameters: Dict[str, Any] = Field(
        description="Starting parameters before refinement"
    )
    final_parameters: Dict[str, Any] = Field(
        description="Best parameters found during refinement"
    )
    iterations: List[IterationResult] = Field(
        min_length=1,
        description="List of all iteration results"
    )
    total_iterations: int = Field(
        ge=1,
        description="Total number of iterations performed"
    )
    final_score: float = Field(
        ge=0,
        le=100,
        description="Best score achieved (0-100 scale)"
    )
    improvement: float = Field(
        ge=-100,
        le=100,
        description="Score improvement from initial to final (can be negative)"
    )
    convergence_reason: str = Field(
        min_length=1,
        description="Reason why refinement stopped"
    )

    @field_validator('initial_parameters', 'final_parameters')
    @classmethod
    def validate_parameters_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameters dictionaries are not empty."""
        if not v:
            raise ValueError("Parameters dictionary cannot be empty")
        return v

    @field_validator('iterations')
    @classmethod
    def validate_iterations_not_empty(cls, v: List[IterationResult]) -> List[IterationResult]:
        """Ensure iterations list is not empty."""
        if not v:
            raise ValueError("Iterations list cannot be empty")
        return v

    @field_validator('total_iterations')
    @classmethod
    def validate_total_iterations_matches(cls, v: int, info) -> int:
        """Ensure total_iterations matches iterations list length."""
        if 'iterations' in info.data:
            iterations = info.data['iterations']
            if v != len(iterations):
                raise ValueError(
                    f"total_iterations {v} must match iterations list length {len(iterations)}"
                )
        return v

    @field_validator('final_score')
    @classmethod
    def validate_final_score_range(cls, v: float) -> float:
        """Ensure final_score is within valid range [0, 100]."""
        if v < 0 or v > 100:
            raise ValueError(f"final_score {v} must be between 0 and 100")
        return v
