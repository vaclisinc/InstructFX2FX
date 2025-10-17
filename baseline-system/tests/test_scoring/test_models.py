"""Tests for scoring system data models.

Tests verify that scoring models correctly validate:
- Score ranges (0-100 for dimensions and overall_score)
- Confidence ranges (0-1)
- Required fields and types
- Edge cases (min/max values, empty lists/dicts)
- Validation logic for all constraints
"""

import pytest
from pydantic import ValidationError

from src.scoring.models import (
    ScoringRequest,
    ScoreDimension,
    ScoringResponse,
)


class TestScoringRequestValidation:
    """Test ScoringRequest model validation."""

    def test_valid_scoring_request_minimal(self):
        """Valid minimal scoring request should pass validation."""
        request = ScoringRequest(
            description="warm and cozy reverb",
            parameters={"reverb": {"room_size": 0.5, "damping": 0.7}}
        )
        assert request.description == "warm and cozy reverb"
        assert request.parameters == {"reverb": {"room_size": 0.5, "damping": 0.7}}
        assert request.audio_features is None
        assert request.previous_score is None
        assert request.iteration == 0
        print(f"✓ Valid minimal scoring request created: iteration={request.iteration}")

    def test_valid_scoring_request_full(self):
        """Valid full scoring request with all fields should pass validation."""
        request = ScoringRequest(
            description="bright and crisp with subtle echo",
            parameters={
                "reverb": {"room_size": 0.3, "damping": 0.2},
                "eq": {"low": 0.5, "mid": 0.8, "high": 0.9}
            },
            audio_features={
                "spectral_centroid": 2500.0,
                "rms_energy": 0.3
            },
            previous_score=75.5,
            iteration=3
        )
        assert request.description == "bright and crisp with subtle echo"
        assert "reverb" in request.parameters
        assert "eq" in request.parameters
        assert request.audio_features["spectral_centroid"] == 2500.0
        assert request.previous_score == 75.5
        assert request.iteration == 3
        print(f"✓ Valid full scoring request created: iteration={request.iteration}, previous_score={request.previous_score}")

    def test_scoring_request_iteration_default(self):
        """Iteration should default to 0 if not specified."""
        request = ScoringRequest(
            description="test",
            parameters={"test": "value"}
        )
        assert request.iteration == 0
        print(f"✓ Iteration defaults to 0")

    def test_invalid_scoring_request_empty_description(self):
        """Empty description should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringRequest(
                description="",
                parameters={"test": "value"}
            )
        error = exc_info.value
        print(f"✓ Empty description rejected: {error}")
        assert "description" in str(error).lower()

    def test_invalid_scoring_request_empty_parameters(self):
        """Empty parameters dictionary should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringRequest(
                description="test description",
                parameters={}
            )
        error = exc_info.value
        print(f"✓ Empty parameters rejected: {error}")
        assert "empty" in str(error).lower() or "parameters" in str(error).lower()

    def test_invalid_scoring_request_missing_description(self):
        """Missing description field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringRequest(parameters={"test": "value"})
        error = exc_info.value
        print(f"✓ Missing description rejected: {error}")
        assert "description" in str(error).lower()

    def test_invalid_scoring_request_missing_parameters(self):
        """Missing parameters field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringRequest(description="test description")
        error = exc_info.value
        print(f"✓ Missing parameters rejected: {error}")
        assert "parameters" in str(error).lower()

    def test_invalid_scoring_request_negative_iteration(self):
        """Negative iteration should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringRequest(
                description="test",
                parameters={"test": "value"},
                iteration=-1
            )
        error = exc_info.value
        print(f"✓ Negative iteration rejected: {error}")
        assert "iteration" in str(error).lower()

    def test_invalid_scoring_request_previous_score_too_low(self):
        """Previous score below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringRequest(
                description="test",
                parameters={"test": "value"},
                previous_score=-5.0
            )
        error = exc_info.value
        print(f"✓ Previous score too low rejected (-5.0 < 0): {error}")
        assert "previous_score" in str(error).lower()

    def test_invalid_scoring_request_previous_score_too_high(self):
        """Previous score above 100 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringRequest(
                description="test",
                parameters={"test": "value"},
                previous_score=150.0
            )
        error = exc_info.value
        print(f"✓ Previous score too high rejected (150.0 > 100): {error}")
        assert "previous_score" in str(error).lower()

    def test_valid_scoring_request_previous_score_boundary_min(self):
        """Previous score exactly at minimum (0) should be valid."""
        request = ScoringRequest(
            description="test",
            parameters={"test": "value"},
            previous_score=0.0
        )
        assert request.previous_score == 0.0
        print(f"✓ Previous score minimum boundary (0.0) accepted")

    def test_valid_scoring_request_previous_score_boundary_max(self):
        """Previous score exactly at maximum (100) should be valid."""
        request = ScoringRequest(
            description="test",
            parameters={"test": "value"},
            previous_score=100.0
        )
        assert request.previous_score == 100.0
        print(f"✓ Previous score maximum boundary (100.0) accepted")

    def test_scoring_request_serialization(self):
        """ScoringRequest should be serializable to dict."""
        request = ScoringRequest(
            description="test",
            parameters={"test": "value"},
            iteration=2
        )
        request_dict = request.model_dump()
        assert "description" in request_dict
        assert "parameters" in request_dict
        assert "iteration" in request_dict
        print(f"✓ ScoringRequest serializable to dict: {list(request_dict.keys())}")


class TestScoreDimensionValidation:
    """Test ScoreDimension model validation."""

    def test_valid_score_dimension(self):
        """Valid score dimension should pass validation."""
        dimension = ScoreDimension(
            name="semantic_match",
            score=85.5,
            reasoning="Parameters closely align with the warm and cozy description"
        )
        assert dimension.name == "semantic_match"
        assert dimension.score == 85.5
        assert "warm and cozy" in dimension.reasoning
        print(f"✓ Valid score dimension created: name={dimension.name}, score={dimension.score}")

    def test_valid_score_dimension_min_score(self):
        """Score dimension with minimum score (0) should be valid."""
        dimension = ScoreDimension(
            name="technical_quality",
            score=0.0,
            reasoning="Parameters are technically incorrect"
        )
        assert dimension.score == 0.0
        print(f"✓ Minimum score (0.0) accepted")

    def test_valid_score_dimension_max_score(self):
        """Score dimension with maximum score (100) should be valid."""
        dimension = ScoreDimension(
            name="specificity",
            score=100.0,
            reasoning="Parameters are perfectly precise and purposeful"
        )
        assert dimension.score == 100.0
        print(f"✓ Maximum score (100.0) accepted")

    def test_invalid_score_dimension_score_too_low(self):
        """Score below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreDimension(
                name="test_dimension",
                score=-10.0,
                reasoning="test reasoning"
            )
        error = exc_info.value
        print(f"✓ Score too low rejected (-10.0 < 0): {error}")
        assert "score" in str(error).lower()

    def test_invalid_score_dimension_score_too_high(self):
        """Score above 100 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreDimension(
                name="test_dimension",
                score=150.0,
                reasoning="test reasoning"
            )
        error = exc_info.value
        print(f"✓ Score too high rejected (150.0 > 100): {error}")
        assert "score" in str(error).lower()

    def test_invalid_score_dimension_empty_name(self):
        """Empty name should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreDimension(
                name="",
                score=50.0,
                reasoning="test reasoning"
            )
        error = exc_info.value
        print(f"✓ Empty name rejected: {error}")
        assert "name" in str(error).lower()

    def test_invalid_score_dimension_empty_reasoning(self):
        """Empty reasoning should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreDimension(
                name="test_dimension",
                score=50.0,
                reasoning=""
            )
        error = exc_info.value
        print(f"✓ Empty reasoning rejected: {error}")
        assert "reasoning" in str(error).lower()

    def test_invalid_score_dimension_missing_name(self):
        """Missing name field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreDimension(
                score=50.0,
                reasoning="test reasoning"
            )
        error = exc_info.value
        print(f"✓ Missing name rejected: {error}")
        assert "name" in str(error).lower()

    def test_invalid_score_dimension_missing_score(self):
        """Missing score field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreDimension(
                name="test_dimension",
                reasoning="test reasoning"
            )
        error = exc_info.value
        print(f"✓ Missing score rejected: {error}")
        assert "score" in str(error).lower()

    def test_invalid_score_dimension_missing_reasoning(self):
        """Missing reasoning field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoreDimension(
                name="test_dimension",
                score=50.0
            )
        error = exc_info.value
        print(f"✓ Missing reasoning rejected: {error}")
        assert "reasoning" in str(error).lower()

    def test_score_dimension_serialization(self):
        """ScoreDimension should be serializable to dict."""
        dimension = ScoreDimension(
            name="test_dimension",
            score=75.0,
            reasoning="test reasoning"
        )
        dimension_dict = dimension.model_dump()
        assert "name" in dimension_dict
        assert "score" in dimension_dict
        assert "reasoning" in dimension_dict
        print(f"✓ ScoreDimension serializable to dict: {list(dimension_dict.keys())}")


class TestScoringResponseValidation:
    """Test ScoringResponse model validation."""

    def test_valid_scoring_response_minimal(self):
        """Valid minimal scoring response should pass validation."""
        response = ScoringResponse(
            overall_score=82.3,
            dimensions=[
                ScoreDimension(
                    name="semantic_match",
                    score=85.0,
                    reasoning="Good alignment"
                )
            ],
            feedback="Overall parameters match well",
            confidence=0.85
        )
        assert response.overall_score == 82.3
        assert len(response.dimensions) == 1
        assert response.dimensions[0].name == "semantic_match"
        assert response.feedback == "Overall parameters match well"
        assert len(response.suggestions) == 0
        assert response.confidence == 0.85
        print(f"✓ Valid minimal scoring response created: overall_score={response.overall_score}, confidence={response.confidence}")

    def test_valid_scoring_response_full(self):
        """Valid full scoring response with all fields should pass validation."""
        response = ScoringResponse(
            overall_score=78.5,
            dimensions=[
                ScoreDimension(
                    name="semantic_match",
                    score=80.0,
                    reasoning="Parameters generally align with description"
                ),
                ScoreDimension(
                    name="technical_quality",
                    score=85.0,
                    reasoning="Musically appropriate parameters"
                ),
                ScoreDimension(
                    name="specificity",
                    score=70.0,
                    reasoning="Some parameters could be more precise"
                )
            ],
            feedback="Parameters show good overall quality with room for refinement",
            suggestions=[
                "Consider increasing reverb room_size slightly",
                "Try reducing high frequency emphasis",
                "Adjust wet/dry balance for more natural sound"
            ],
            confidence=0.92
        )
        assert response.overall_score == 78.5
        assert len(response.dimensions) == 3
        assert response.dimensions[0].name == "semantic_match"
        assert response.dimensions[1].name == "technical_quality"
        assert response.dimensions[2].name == "specificity"
        assert len(response.suggestions) == 3
        assert response.confidence == 0.92
        print(f"✓ Valid full scoring response created: {len(response.dimensions)} dimensions, {len(response.suggestions)} suggestions")

    def test_valid_scoring_response_empty_suggestions(self):
        """Empty suggestions list should be valid (perfect score case)."""
        response = ScoringResponse(
            overall_score=98.0,
            dimensions=[
                ScoreDimension(
                    name="semantic_match",
                    score=98.0,
                    reasoning="Excellent alignment"
                )
            ],
            feedback="Nearly perfect parameters",
            suggestions=[],
            confidence=0.95
        )
        assert len(response.suggestions) == 0
        print(f"✓ Empty suggestions list accepted (perfect score case)")

    def test_valid_scoring_response_min_overall_score(self):
        """Overall score at minimum (0) should be valid."""
        response = ScoringResponse(
            overall_score=0.0,
            dimensions=[
                ScoreDimension(
                    name="semantic_match",
                    score=0.0,
                    reasoning="No alignment"
                )
            ],
            feedback="Parameters completely miss the mark",
            confidence=0.8
        )
        assert response.overall_score == 0.0
        print(f"✓ Minimum overall score (0.0) accepted")

    def test_valid_scoring_response_max_overall_score(self):
        """Overall score at maximum (100) should be valid."""
        response = ScoringResponse(
            overall_score=100.0,
            dimensions=[
                ScoreDimension(
                    name="semantic_match",
                    score=100.0,
                    reasoning="Perfect alignment"
                )
            ],
            feedback="Perfect parameters",
            confidence=1.0
        )
        assert response.overall_score == 100.0
        print(f"✓ Maximum overall score (100.0) accepted")

    def test_valid_scoring_response_min_confidence(self):
        """Confidence at minimum (0) should be valid."""
        response = ScoringResponse(
            overall_score=50.0,
            dimensions=[
                ScoreDimension(
                    name="semantic_match",
                    score=50.0,
                    reasoning="Uncertain evaluation"
                )
            ],
            feedback="Low confidence in evaluation",
            confidence=0.0
        )
        assert response.confidence == 0.0
        print(f"✓ Minimum confidence (0.0) accepted")

    def test_valid_scoring_response_max_confidence(self):
        """Confidence at maximum (1) should be valid."""
        response = ScoringResponse(
            overall_score=85.0,
            dimensions=[
                ScoreDimension(
                    name="semantic_match",
                    score=85.0,
                    reasoning="Clear evaluation"
                )
            ],
            feedback="High confidence in evaluation",
            confidence=1.0
        )
        assert response.confidence == 1.0
        print(f"✓ Maximum confidence (1.0) accepted")

    def test_invalid_scoring_response_overall_score_too_low(self):
        """Overall score below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=-10.0,
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=50.0,
                        reasoning="test"
                    )
                ],
                feedback="test feedback",
                confidence=0.8
            )
        error = exc_info.value
        print(f"✓ Overall score too low rejected (-10.0 < 0): {error}")
        assert "overall_score" in str(error).lower()

    def test_invalid_scoring_response_overall_score_too_high(self):
        """Overall score above 100 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=150.0,
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=50.0,
                        reasoning="test"
                    )
                ],
                feedback="test feedback",
                confidence=0.8
            )
        error = exc_info.value
        print(f"✓ Overall score too high rejected (150.0 > 100): {error}")
        assert "overall_score" in str(error).lower()

    def test_invalid_scoring_response_confidence_too_low(self):
        """Confidence below 0 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=75.0,
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=75.0,
                        reasoning="test"
                    )
                ],
                feedback="test feedback",
                confidence=-0.5
            )
        error = exc_info.value
        print(f"✓ Confidence too low rejected (-0.5 < 0): {error}")
        assert "confidence" in str(error).lower()

    def test_invalid_scoring_response_confidence_too_high(self):
        """Confidence above 1 should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=75.0,
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=75.0,
                        reasoning="test"
                    )
                ],
                feedback="test feedback",
                confidence=1.5
            )
        error = exc_info.value
        print(f"✓ Confidence too high rejected (1.5 > 1): {error}")
        assert "confidence" in str(error).lower()

    def test_invalid_scoring_response_empty_dimensions(self):
        """Empty dimensions list should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=75.0,
                dimensions=[],
                feedback="test feedback",
                confidence=0.8
            )
        error = exc_info.value
        print(f"✓ Empty dimensions list rejected: {error}")
        assert "dimensions" in str(error).lower()

    def test_invalid_scoring_response_empty_feedback(self):
        """Empty feedback should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=75.0,
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=75.0,
                        reasoning="test"
                    )
                ],
                feedback="",
                confidence=0.8
            )
        error = exc_info.value
        print(f"✓ Empty feedback rejected: {error}")
        assert "feedback" in str(error).lower()

    def test_invalid_scoring_response_missing_overall_score(self):
        """Missing overall_score field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=75.0,
                        reasoning="test"
                    )
                ],
                feedback="test feedback",
                confidence=0.8
            )
        error = exc_info.value
        print(f"✓ Missing overall_score rejected: {error}")
        assert "overall_score" in str(error).lower()

    def test_invalid_scoring_response_missing_dimensions(self):
        """Missing dimensions field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=75.0,
                feedback="test feedback",
                confidence=0.8
            )
        error = exc_info.value
        print(f"✓ Missing dimensions rejected: {error}")
        assert "dimensions" in str(error).lower()

    def test_invalid_scoring_response_missing_feedback(self):
        """Missing feedback field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=75.0,
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=75.0,
                        reasoning="test"
                    )
                ],
                confidence=0.8
            )
        error = exc_info.value
        print(f"✓ Missing feedback rejected: {error}")
        assert "feedback" in str(error).lower()

    def test_invalid_scoring_response_missing_confidence(self):
        """Missing confidence field should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            ScoringResponse(
                overall_score=75.0,
                dimensions=[
                    ScoreDimension(
                        name="test",
                        score=75.0,
                        reasoning="test"
                    )
                ],
                feedback="test feedback"
            )
        error = exc_info.value
        print(f"✓ Missing confidence rejected: {error}")
        assert "confidence" in str(error).lower()

    def test_scoring_response_serialization(self):
        """ScoringResponse should be serializable to dict."""
        response = ScoringResponse(
            overall_score=80.0,
            dimensions=[
                ScoreDimension(
                    name="test",
                    score=80.0,
                    reasoning="test reasoning"
                )
            ],
            feedback="test feedback",
            suggestions=["suggestion 1", "suggestion 2"],
            confidence=0.9
        )
        response_dict = response.model_dump()
        assert "overall_score" in response_dict
        assert "dimensions" in response_dict
        assert "feedback" in response_dict
        assert "suggestions" in response_dict
        assert "confidence" in response_dict
        print(f"✓ ScoringResponse serializable to dict: {list(response_dict.keys())}")

    def test_scoring_response_multiple_dimensions(self):
        """ScoringResponse should handle multiple dimensions correctly."""
        dimensions = [
            ScoreDimension(
                name=f"dimension_{i}",
                score=70.0 + i * 5,
                reasoning=f"Reasoning for dimension {i}"
            )
            for i in range(5)
        ]
        response = ScoringResponse(
            overall_score=80.0,
            dimensions=dimensions,
            feedback="Multiple dimensions evaluated",
            confidence=0.85
        )
        assert len(response.dimensions) == 5
        assert response.dimensions[0].score == 70.0
        assert response.dimensions[4].score == 90.0
        print(f"✓ Multiple dimensions ({len(response.dimensions)}) handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
