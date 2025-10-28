"""Unit tests for scoring system core functionality.

Tests verify the parsing, validation, and computation logic of the ScoringSystem,
particularly focusing on edge cases, malformed responses, and error handling.
"""

import pytest
import json
from pydantic import ValidationError

from src.scoring.scorer import ScoringSystem
from src.scoring.models import (
    ScoringRequest,
    ScoringResponse,
    ScoreDimension,
)
from src.scoring.exceptions import (
    MalformedResponseError,
    ScoreOutOfRangeError,
)
from src.scoring.config import ScoringConfig
from models.llm_judge.base import LLMProvider
from models.llm_judge.types import LLMRequest, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API calls."""

    def __init__(self):
        """Initialize mock provider with minimal config."""
        config = {
            "provider": "mock",
            "model": "mock-model",
            "api_key": "mock-key"
        }
        super().__init__(config)

    def validate_config(self) -> bool:
        """Mock config validation."""
        return True

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Mock generation - should not be called in unit tests."""
        raise NotImplementedError("Mock provider should not generate in unit tests")


class TestParseScoreResponse:
    """Test parse_score_response method with various edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = ScoringSystem(
            llm_provider=MockLLMProvider(),
            config=ScoringConfig({
                "mode": "parameter_only",
                "dimensions": ["semantic_match", "technical_quality", "specificity"],
                "weights": {
                    "semantic_match": 0.5,
                    "technical_quality": 0.3,
                    "specificity": 0.2
                },
                "temperature": 0.3,
                "retry": {
                    "max_attempts": 3,
                    "correction_prompt": True
                }
            })
        )

    def test_parse_valid_json_response(self):
        """Parse valid JSON response successfully."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 85.5, "reasoning": "Good alignment"},
            {"name": "technical_quality", "score": 90.0, "reasoning": "Excellent quality"},
            {"name": "specificity", "score": 75.0, "reasoning": "Could be more precise"}
          ],
          "overall_score": 83.5,
          "feedback": "Parameters show strong overall quality",
          "suggestions": ["Increase room_size", "Adjust wet/dry balance"],
          "confidence": 0.85
        }
        """

        result = self.scorer.parse_score_response(response_content)

        assert isinstance(result, ScoringResponse)
        assert result.overall_score == 83.5
        assert len(result.dimensions) == 3
        assert result.dimensions[0].name == "semantic_match"
        assert result.dimensions[0].score == 85.5
        assert result.confidence == 0.85
        assert len(result.suggestions) == 2
        print(f"✓ Valid JSON parsed: overall_score={result.overall_score}, dimensions={len(result.dimensions)}")

    def test_parse_json_with_text_before(self):
        """Parse JSON even when preceded by text."""
        response_content = """
        Here is my evaluation of the parameters:

        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Reasonable match"}
          ],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": 0.75
        }
        """

        result = self.scorer.parse_score_response(response_content)

        assert isinstance(result, ScoringResponse)
        assert result.overall_score == 80.0
        print(f"✓ JSON parsed with preceding text")

    def test_parse_json_with_text_after(self):
        """Parse JSON even when followed by text."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 75.0, "reasoning": "Decent match"}
          ],
          "overall_score": 75.0,
          "feedback": "Acceptable parameters",
          "suggestions": ["Improve reverb"],
          "confidence": 0.70
        }

        I hope this helps with your audio processing!
        """

        result = self.scorer.parse_score_response(response_content)

        assert isinstance(result, ScoringResponse)
        assert result.overall_score == 75.0
        print(f"✓ JSON parsed with following text")

    def test_parse_malformed_json_missing_brace(self):
        """Fail gracefully on JSON missing closing brace."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "feedback": "Good",
          "suggestions": [],
          "confidence": 0.8
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert "json" in str(error).lower()
        print(f"✓ Missing brace detected: {error}")

    def test_parse_malformed_json_invalid_syntax(self):
        """Fail gracefully on JSON with invalid syntax."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good",}
          ],
          "overall_score": 80.0,
          "feedback": "Good",
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert error.raw_output is not None
        print(f"✓ Invalid JSON syntax detected: {error}")

    def test_parse_missing_dimensions_field(self):
        """Fail when dimensions field is missing."""
        response_content = """
        {
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": ["Improve"],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert error.parse_error is not None
        print(f"✓ Missing dimensions field detected: {error}")

    def test_parse_missing_overall_score_field(self):
        """Fail when overall_score field is missing."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good"}
          ],
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Missing overall_score field detected: {error}")

    def test_parse_missing_feedback_field(self):
        """Fail when feedback field is missing."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Missing feedback field detected: {error}")

    def test_parse_missing_confidence_field(self):
        """Fail when confidence field is missing."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": []
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Missing confidence field detected: {error}")

    def test_parse_empty_dimensions_list(self):
        """Fail when dimensions list is empty."""
        response_content = """
        {
          "dimensions": [],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Empty dimensions list detected: {error}")

    def test_parse_dimension_score_out_of_range_negative(self):
        """Accept negative dimension score (will be clamped later)."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": -10.0, "reasoning": "Very poor"}
          ],
          "overall_score": 0.0,
          "feedback": "Poor parameters",
          "suggestions": ["Complete redesign needed"],
          "confidence": 0.9
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert "score" in str(error).lower()
        print(f"✓ Negative dimension score detected: {error}")

    def test_parse_dimension_score_out_of_range_high(self):
        """Accept high dimension score (will be clamped later)."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 150.0, "reasoning": "Exceptional"}
          ],
          "overall_score": 100.0,
          "feedback": "Perfect parameters",
          "suggestions": [],
          "confidence": 0.9
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert "score" in str(error).lower()
        print(f"✓ High dimension score detected: {error}")

    def test_parse_overall_score_negative(self):
        """Accept negative overall score (will be clamped later)."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 0.0, "reasoning": "Poor"}
          ],
          "overall_score": -5.0,
          "feedback": "Very poor parameters",
          "suggestions": ["Start over"],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Negative overall score detected: {error}")

    def test_parse_overall_score_too_high(self):
        """Accept high overall score (will be clamped later)."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 100.0, "reasoning": "Perfect"}
          ],
          "overall_score": 110.0,
          "feedback": "Beyond perfect parameters",
          "suggestions": [],
          "confidence": 0.9
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ High overall score detected: {error}")

    def test_parse_confidence_negative(self):
        """Fail when confidence is negative."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": -0.5
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert "confidence" in str(error).lower()
        print(f"✓ Negative confidence detected: {error}")

    def test_parse_confidence_too_high(self):
        """Fail when confidence is above 1."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": 1.5
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert "confidence" in str(error).lower()
        print(f"✓ High confidence detected: {error}")

    def test_parse_empty_feedback(self):
        """Fail when feedback is empty string."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0, "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "feedback": "",
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Empty feedback detected: {error}")

    def test_parse_empty_suggestions_list(self):
        """Accept empty suggestions list (valid for perfect scores)."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 98.0, "reasoning": "Nearly perfect"}
          ],
          "overall_score": 98.0,
          "feedback": "Excellent parameters",
          "suggestions": [],
          "confidence": 0.95
        }
        """

        result = self.scorer.parse_score_response(response_content)

        assert isinstance(result, ScoringResponse)
        assert len(result.suggestions) == 0
        print(f"✓ Empty suggestions list accepted (perfect score case)")

    def test_parse_no_json_in_response(self):
        """Fail when response contains no JSON."""
        response_content = """
        I cannot provide a score for these parameters because they are invalid.
        Please provide valid audio effect parameters.
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        assert "No JSON found" in str(error)
        print(f"✓ No JSON in response detected: {error}")

    def test_parse_dimension_missing_name(self):
        """Fail when dimension is missing name field."""
        response_content = """
        {
          "dimensions": [
            {"score": 80.0, "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Dimension missing name detected: {error}")

    def test_parse_dimension_missing_score(self):
        """Fail when dimension is missing score field."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "reasoning": "Good"}
          ],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Dimension missing score detected: {error}")

    def test_parse_dimension_missing_reasoning(self):
        """Fail when dimension is missing reasoning field."""
        response_content = """
        {
          "dimensions": [
            {"name": "semantic_match", "score": 80.0}
          ],
          "overall_score": 80.0,
          "feedback": "Good parameters",
          "suggestions": [],
          "confidence": 0.8
        }
        """

        with pytest.raises(MalformedResponseError) as exc_info:
            self.scorer.parse_score_response(response_content)

        error = exc_info.value
        print(f"✓ Dimension missing reasoning detected: {error}")


class TestComputeWeightedScore:
    """Test compute_weighted_score method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = ScoringSystem(
            llm_provider=MockLLMProvider(),
            config=ScoringConfig({
                "mode": "parameter_only",
                "dimensions": ["semantic_match", "technical_quality", "specificity"],
                "weights": {
                    "semantic_match": 0.5,
                    "technical_quality": 0.3,
                    "specificity": 0.2
                },
                "temperature": 0.3,
                "retry": {
                    "max_attempts": 3,
                    "correction_prompt": True
                }
            })
        )

    def test_compute_weighted_score_standard(self):
        """Compute weighted score with standard weights."""
        dimensions = [
            ScoreDimension(name="semantic_match", score=80.0, reasoning="Good"),
            ScoreDimension(name="technical_quality", score=90.0, reasoning="Excellent"),
            ScoreDimension(name="specificity", score=70.0, reasoning="Acceptable")
        ]

        weighted = self.scorer.compute_weighted_score(dimensions)

        # Expected: (80*0.5 + 90*0.3 + 70*0.2) / 1.0 = (40 + 27 + 14) = 81.0
        expected = 81.0
        assert abs(weighted - expected) < 0.01
        print(f"✓ Weighted score computed: {weighted} (expected {expected})")

    def test_compute_weighted_score_all_equal(self):
        """Compute weighted score when all dimension scores are equal."""
        dimensions = [
            ScoreDimension(name="semantic_match", score=75.0, reasoning="Good"),
            ScoreDimension(name="technical_quality", score=75.0, reasoning="Good"),
            ScoreDimension(name="specificity", score=75.0, reasoning="Good")
        ]

        weighted = self.scorer.compute_weighted_score(dimensions)

        # When all scores are equal, weighted average equals the score
        assert abs(weighted - 75.0) < 0.01
        print(f"✓ Weighted score with equal dimensions: {weighted}")

    def test_compute_weighted_score_empty_dimensions(self):
        """Fail when dimensions list is empty."""
        dimensions = []

        with pytest.raises(ValueError) as exc_info:
            self.scorer.compute_weighted_score(dimensions)

        error = exc_info.value
        assert "empty" in str(error).lower()
        print(f"✓ Empty dimensions error: {error}")

    def test_compute_weighted_score_unknown_dimension(self):
        """Handle unknown dimension by using default weight of 1.0."""
        dimensions = [
            ScoreDimension(name="semantic_match", score=80.0, reasoning="Good"),
            ScoreDimension(name="unknown_dimension", score=60.0, reasoning="New")
        ]

        weighted = self.scorer.compute_weighted_score(dimensions)

        # semantic_match: 0.5, unknown_dimension: 1.0 (default)
        # Expected: (80*0.5 + 60*1.0) / (0.5 + 1.0) = (40 + 60) / 1.5 = 66.67
        expected = 66.67
        assert abs(weighted - expected) < 0.1
        print(f"✓ Weighted score with unknown dimension: {weighted} (expected ~{expected})")

    def test_compute_weighted_score_boundary_min(self):
        """Compute weighted score with all dimensions at minimum (0)."""
        dimensions = [
            ScoreDimension(name="semantic_match", score=0.0, reasoning="Poor"),
            ScoreDimension(name="technical_quality", score=0.0, reasoning="Poor"),
            ScoreDimension(name="specificity", score=0.0, reasoning="Poor")
        ]

        weighted = self.scorer.compute_weighted_score(dimensions)

        assert weighted == 0.0
        print(f"✓ Weighted score at minimum boundary: {weighted}")

    def test_compute_weighted_score_boundary_max(self):
        """Compute weighted score with all dimensions at maximum (100)."""
        dimensions = [
            ScoreDimension(name="semantic_match", score=100.0, reasoning="Perfect"),
            ScoreDimension(name="technical_quality", score=100.0, reasoning="Perfect"),
            ScoreDimension(name="specificity", score=100.0, reasoning="Perfect")
        ]

        weighted = self.scorer.compute_weighted_score(dimensions)

        assert weighted == 100.0
        print(f"✓ Weighted score at maximum boundary: {weighted}")

    def test_compute_weighted_score_single_dimension(self):
        """Compute weighted score with single dimension."""
        dimensions = [
            ScoreDimension(name="semantic_match", score=85.0, reasoning="Good")
        ]

        weighted = self.scorer.compute_weighted_score(dimensions)

        # Single dimension: score * weight / weight = score
        assert weighted == 85.0
        print(f"✓ Weighted score with single dimension: {weighted}")


class TestValidateAndAdjustScores:
    """Test _validate_and_adjust_scores method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = ScoringSystem(
            llm_provider=MockLLMProvider(),
            config=ScoringConfig({
                "mode": "parameter_only",
                "dimensions": ["semantic_match", "technical_quality", "specificity"],
                "weights": {
                    "semantic_match": 0.5,
                    "technical_quality": 0.3,
                    "specificity": 0.2
                },
                "temperature": 0.3,
                "retry": {
                    "max_attempts": 3,
                    "correction_prompt": True
                }
            })
        )

    def test_validate_valid_scores(self):
        """Valid scores should pass through unchanged."""
        response = ScoringResponse(
            overall_score=85.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=80.0, reasoning="Good"),
                ScoreDimension(name="technical_quality", score=90.0, reasoning="Excellent")
            ],
            feedback="Good parameters",
            suggestions=["Improve X"],
            confidence=0.85
        )

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.overall_score == 85.0
        assert validated.dimensions[0].score == 80.0
        assert validated.dimensions[1].score == 90.0
        assert validated.confidence == 0.85
        print(f"✓ Valid scores passed through unchanged")

    def test_validate_clamp_dimension_score_negative(self):
        """Negative dimension scores should be clamped to 0."""
        response = ScoringResponse(
            overall_score=50.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=50.0, reasoning="Poor")
            ],
            feedback="Poor parameters",
            suggestions=["Fix everything"],
            confidence=0.7
        )

        # Bypass Pydantic validation to test clamping
        response.dimensions[0].__dict__['score'] = -10.0

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.dimensions[0].score == 0.0
        print(f"✓ Negative dimension score clamped to 0")

    def test_validate_clamp_dimension_score_high(self):
        """High dimension scores should be clamped to 100."""
        response = ScoringResponse(
            overall_score=95.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=80.0, reasoning="Good")
            ],
            feedback="Good parameters",
            suggestions=[],
            confidence=0.9
        )

        # Bypass Pydantic validation to test clamping
        response.dimensions[0].__dict__['score'] = 150.0

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.dimensions[0].score == 100.0
        print(f"✓ High dimension score clamped to 100")

    def test_validate_clamp_overall_score_negative(self):
        """Negative overall score should be clamped to 0."""
        response = ScoringResponse(
            overall_score=50.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=50.0, reasoning="OK")
            ],
            feedback="OK parameters",
            suggestions=["Improve"],
            confidence=0.7
        )

        # Bypass Pydantic validation to test clamping
        response.__dict__['overall_score'] = -5.0

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.overall_score == 0.0
        print(f"✓ Negative overall score clamped to 0")

    def test_validate_clamp_overall_score_high(self):
        """High overall score should be clamped to 100."""
        response = ScoringResponse(
            overall_score=95.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=95.0, reasoning="Excellent")
            ],
            feedback="Excellent parameters",
            suggestions=[],
            confidence=0.95
        )

        # Bypass Pydantic validation to test clamping
        response.__dict__['overall_score'] = 110.0

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.overall_score == 100.0
        print(f"✓ High overall score clamped to 100")

    def test_validate_clamp_confidence_negative(self):
        """Negative confidence should be clamped to 0."""
        response = ScoringResponse(
            overall_score=75.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=75.0, reasoning="OK")
            ],
            feedback="OK parameters",
            suggestions=[],
            confidence=0.8
        )

        # Bypass Pydantic validation to test clamping
        response.__dict__['confidence'] = -0.5

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.confidence == 0.0
        print(f"✓ Negative confidence clamped to 0")

    def test_validate_clamp_confidence_high(self):
        """High confidence should be clamped to 1."""
        response = ScoringResponse(
            overall_score=90.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=90.0, reasoning="Great")
            ],
            feedback="Great parameters",
            suggestions=[],
            confidence=0.9
        )

        # Bypass Pydantic validation to test clamping
        response.__dict__['confidence'] = 1.5

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.confidence == 1.0
        print(f"✓ High confidence clamped to 1")

    def test_validate_clamp_multiple_values(self):
        """Multiple out-of-range values should all be clamped."""
        response = ScoringResponse(
            overall_score=80.0,
            dimensions=[
                ScoreDimension(name="semantic_match", score=80.0, reasoning="Good"),
                ScoreDimension(name="technical_quality", score=90.0, reasoning="Excellent")
            ],
            feedback="Good parameters",
            suggestions=[],
            confidence=0.85
        )

        # Bypass Pydantic validation to test clamping
        response.dimensions[0].__dict__['score'] = -20.0
        response.dimensions[1].__dict__['score'] = 120.0
        response.__dict__['overall_score'] = 105.0
        response.__dict__['confidence'] = 1.8

        validated = self.scorer._validate_and_adjust_scores(response)

        assert validated.dimensions[0].score == 0.0
        assert validated.dimensions[1].score == 100.0
        assert validated.overall_score == 100.0
        assert validated.confidence == 1.0
        print(f"✓ Multiple out-of-range values all clamped correctly")


class TestScoringSystemInitialization:
    """Test ScoringSystem initialization and configuration."""

    def test_init_with_mock_provider(self):
        """Initialize with mock LLM provider."""
        provider = MockLLMProvider()
        scorer = ScoringSystem(llm_provider=provider)

        assert scorer.llm_provider is provider
        assert scorer.config is not None
        assert scorer.config.mode == "parameter_only"
        print(f"✓ ScoringSystem initialized with mock provider")

    def test_init_with_custom_config(self):
        """Initialize with custom configuration."""
        config = ScoringConfig({
            "mode": "audio_based",
            "dimensions": ["semantic_match", "technical_quality"],
            "weights": {
                "semantic_match": 0.6,
                "technical_quality": 0.4,
                "specificity": 0.0
            },
            "temperature": 0.5,
            "retry": {
                "max_attempts": 5,
                "correction_prompt": False
            }
        })

        scorer = ScoringSystem(
            llm_provider=MockLLMProvider(),
            config=config
        )

        assert scorer.config.mode == "audio_based"
        assert scorer.config.temperature == 0.5
        assert scorer.config.max_retry_attempts == 5
        assert not scorer.config.use_correction_prompt
        print(f"✓ ScoringSystem initialized with custom config")

    def test_init_none_provider_fails(self):
        """Initialization should fail with None provider."""
        with pytest.raises(ValueError) as exc_info:
            ScoringSystem(llm_provider=None)

        error = exc_info.value
        assert "provider" in str(error).lower()
        print(f"✓ None provider rejected: {error}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
