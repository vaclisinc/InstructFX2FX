"""Scoring system data models."""

from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Dict, List, Any


class ScoringRequest(BaseModel):
    """Request model for scoring audio effect parameters.

    Attributes:
        description: Original user description of desired audio effect
        parameters: Generated effect parameters as dictionary
        audio_features: Optional audio analysis features (spectral, temporal, etc.)
        previous_score: Optional score from previous iteration for tracking improvement
        iteration: Current iteration number in refinement loop (0-indexed)
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
    parameters: Dict[str, Any] = Field(
        description="Generated effect parameters as dictionary"
    )
    audio_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional audio analysis features (spectral, temporal, etc.)"
    )
    previous_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Optional score from previous iteration for tracking improvement"
    )
    iteration: int = Field(
        default=0,
        ge=0,
        description="Current iteration number in refinement loop (0-indexed)"
    )

    @field_validator('parameters')
    @classmethod
    def validate_parameters_not_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure parameters dictionary is not empty."""
        if not v:
            raise ValueError("Parameters dictionary cannot be empty")
        return v


class ScoreDimension(BaseModel):
    """Individual scoring dimension with score and reasoning.

    Attributes:
        name: Name of the scoring dimension (e.g., "semantic_match", "technical_quality")
        score: Numeric score for this dimension (0-100 scale)
        reasoning: Detailed explanation of why this score was assigned
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    name: str = Field(
        min_length=1,
        description="Name of the scoring dimension"
    )
    score: float = Field(
        ge=0,
        le=100,
        description="Numeric score for this dimension (0-100 scale)"
    )
    reasoning: str = Field(
        min_length=1,
        description="Detailed explanation of why this score was assigned"
    )

    @field_validator('score')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure score is within valid range [0, 100]."""
        if v < 0 or v > 100:
            raise ValueError(f"Score {v} must be between 0 and 100")
        return v


class ScoringResponse(BaseModel):
    """Response model containing evaluation scores and feedback.

    Attributes:
        overall_score: Overall weighted score across all dimensions (0-100 scale)
        dimensions: List of individual dimension scores with reasoning
        feedback: Qualitative feedback about the generated parameters
        suggestions: List of specific suggestions for improvement
        confidence: Confidence level in the scoring (0-1 scale, 0=no confidence, 1=full confidence)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        strict=True,
        extra="forbid"
    )

    overall_score: float = Field(
        ge=0,
        le=100,
        description="Overall weighted score across all dimensions (0-100 scale)"
    )
    dimensions: List[ScoreDimension] = Field(
        min_length=1,
        description="List of individual dimension scores with reasoning"
    )
    feedback: str = Field(
        min_length=1,
        description="Qualitative feedback about the generated parameters"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="List of specific suggestions for improvement"
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence level in the scoring (0=no confidence, 1=full confidence)"
    )

    @field_validator('overall_score')
    @classmethod
    def validate_overall_score_range(cls, v: float) -> float:
        """Ensure overall score is within valid range [0, 100]."""
        if v < 0 or v > 100:
            raise ValueError(f"Overall score {v} must be between 0 and 100")
        return v

    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v: float) -> float:
        """Ensure confidence is within valid range [0, 1]."""
        if v < 0 or v > 1:
            raise ValueError(f"Confidence {v} must be between 0 and 1")
        return v

    @field_validator('dimensions')
    @classmethod
    def validate_dimensions_not_empty(cls, v: List[ScoreDimension]) -> List[ScoreDimension]:
        """Ensure dimensions list is not empty."""
        if not v:
            raise ValueError("Dimensions list cannot be empty")
        return v
