"""Tests for ParameterGenerator class.

Tests verify parameter generation functionality including:
- Successful generation with mocked LLM responses
- Error handling and retry logic
- Correction prompts
- Prompt versioning
- JSON parsing and validation
"""

import pytest
import json
from unittest.mock import AsyncMock, Mock, patch
from pydantic import ValidationError

from src.generation.parameter_generator import ParameterGenerator
from src.generation.exceptions import (
    ParameterGenerationError,
    JSONParseError,
    ValidationError as GenValidationError,
    LLMProviderError,
    PromptTemplateError
)
from models.llm_judge import LLMProvider, LLMRequest, LLMResponse
from src.models.parameters import EffectChain


# Mock LLM responses
VALID_EQ_RESPONSE = """{
    "description": "warm and intimate vocal sound",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 200, "gain": 3, "q": 0.7},
                {"frequency": 3000, "gain": -2, "q": 1.2},
                {"frequency": 8000, "gain": 1, "q": 0.9}
            ]
        }
    ]
}"""

VALID_MULTI_EFFECT_RESPONSE = """{
    "description": "bright and energetic guitar sound",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 5000, "gain": 4, "q": 1.0},
                {"frequency": 10000, "gain": 2, "q": 0.8},
                {"frequency": 1000, "gain": -1, "q": 1.2}
            ]
        },
        {
            "type": "compressor",
            "threshold": -20,
            "ratio": 4,
            "attack": 5,
            "release": 50,
            "knee": 3,
            "makeup_gain": 6
        }
    ]
}"""

VALID_REVERB_RESPONSE = """{
    "description": "spacious ambient atmosphere",
    "effects": [
        {
            "type": "reverb",
            "room_size": 0.8,
            "damping": 0.4,
            "wet_level": 0.6,
            "dry_level": 0.4,
            "width": 1.0,
            "freeze_mode": false
        }
    ]
}"""

MARKDOWN_WRAPPED_RESPONSE = """Here's the JSON output:

```json
{
    "description": "warm vocal",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 1.0},
                {"frequency": 3000, "gain": -1, "q": 1.0},
                {"frequency": 8000, "gain": 1, "q": 1.0}
            ]
        }
    ]
}
```

This will create a warm sound.
"""

INVALID_JSON_RESPONSE = """{
    "description": "broken json",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "gain": 2, "q": 1.0},
    ]
}"""  # Missing closing brackets

INVALID_PARAMETERS_RESPONSE = """{
    "description": "out of range parameters",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 50000, "gain": 50, "q": 100},
                {"frequency": 1000, "gain": 2, "q": 1.0},
                {"frequency": 8000, "gain": 1, "q": 1.0}
            ]
        }
    ]
}"""

MISSING_REQUIRED_FIELD_RESPONSE = """{
    "description": "missing required field",
    "effects": [
        {
            "type": "eq",
            "bands": [
                {"frequency": 1000, "q": 1.0},
                {"frequency": 3000, "gain": -1, "q": 1.0},
                {"frequency": 8000, "gain": 1, "q": 1.0}
            ]
        }
    ]
}"""


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses=None, should_fail=False):
        """Initialize mock provider.

        Args:
            responses: List of response strings to return in sequence
            should_fail: Whether to raise errors
        """
        self.responses = responses or []
        self.should_fail = should_fail
        self.call_count = 0
        self.requests = []

    async def generate_with_retry(self, request: LLMRequest) -> LLMResponse:
        """Mock generate with retry."""
        self.requests.append(request)
        self.call_count += 1

        if self.should_fail:
            raise Exception("Mock LLM provider error")

        if not self.responses:
            raise Exception("No mock responses available")

        # Cycle through responses
        response_text = self.responses[(self.call_count - 1) % len(self.responses)]

        return LLMResponse(
            content=response_text,
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 200}
        )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "MockProvider"


# Fixtures

@pytest.fixture
def mock_llm_provider():
    """Create mock LLM provider."""
    return MockLLMProvider(responses=[VALID_EQ_RESPONSE])


@pytest.fixture
def mock_prompt_template():
    """Mock prompt template loading."""
    with patch('src.generation.parameter_generator.load_prompt_template') as mock:
        mock_template = Mock()
        mock_template.system_prompt = "You are an audio engineer."
        mock.return_value = mock_template
        yield mock


@pytest.fixture
def generator(mock_llm_provider, mock_prompt_template):
    """Create ParameterGenerator instance with mocked dependencies."""
    with patch('src.generation.parameter_generator.format_prompt') as mock_format:
        mock_format.return_value = "Test prompt"
        gen = ParameterGenerator(
            llm_provider=mock_llm_provider,
            prompt_version="v1",
            max_correction_attempts=3
        )
        yield gen


# Tests

class TestParameterGeneratorInitialization:
    """Test ParameterGenerator initialization."""

    def test_initialization_with_defaults(self, mock_llm_provider, mock_prompt_template):
        """Generator should initialize with default settings."""
        generator = ParameterGenerator(llm_provider=mock_llm_provider)
        assert generator.llm_provider == mock_llm_provider
        assert generator.prompt_version == "v1"
        assert generator.max_correction_attempts == 3
        print(f"✓ Generator initialized with defaults: version={generator.prompt_version}, max_attempts={generator.max_correction_attempts}")

    def test_initialization_with_custom_settings(self, mock_llm_provider, mock_prompt_template):
        """Generator should accept custom settings."""
        generator = ParameterGenerator(
            llm_provider=mock_llm_provider,
            prompt_version="v2",
            max_correction_attempts=5
        )
        assert generator.prompt_version == "v2"
        assert generator.max_correction_attempts == 5
        print(f"✓ Generator initialized with custom settings: version={generator.prompt_version}, max_attempts={generator.max_correction_attempts}")

    def test_initialization_template_error(self, mock_llm_provider):
        """Generator should raise PromptTemplateError if template loading fails."""
        with patch('src.generation.parameter_generator.load_prompt_template') as mock:
            mock.side_effect = Exception("Template not found")
            with pytest.raises(PromptTemplateError) as exc_info:
                ParameterGenerator(llm_provider=mock_llm_provider)

            error = exc_info.value
            print(f"✓ Template error raised: {error.message}")
            assert "Failed to load prompt template" in error.message


class TestSuccessfulGeneration:
    """Test successful parameter generation."""

    @pytest.mark.asyncio
    async def test_generate_single_eq_effect(self, mock_prompt_template):
        """Generator should successfully generate single EQ effect."""
        provider = MockLLMProvider(responses=[VALID_EQ_RESPONSE])
        generator = ParameterGenerator(llm_provider=provider)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"

            result = await generator.generate_parameters(
                description="warm and intimate vocal sound",
                effects=["eq"]
            )

            assert isinstance(result, EffectChain)
            assert len(result.effects) == 1
            assert result.effects[0].effect_type == "eq"
            assert len(result.effects[0].bands) == 3
            assert result.description == "warm and intimate vocal sound"
            print(f"✓ Generated EQ effect with {len(result.effects[0].bands)} bands")

    @pytest.mark.asyncio
    async def test_generate_multiple_effects(self, mock_prompt_template):
        """Generator should successfully generate multiple effects."""
        provider = MockLLMProvider(responses=[VALID_MULTI_EFFECT_RESPONSE])
        generator = ParameterGenerator(llm_provider=provider)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"

            result = await generator.generate_parameters(
                description="bright and energetic guitar sound",
                effects=["eq", "compressor"]
            )

            assert isinstance(result, EffectChain)
            assert len(result.effects) == 2
            assert result.effects[0].effect_type == "eq"
            assert result.effects[1].effect_type == "compressor"
            assert result.order == ["eq", "compressor"]
            print(f"✓ Generated {len(result.effects)} effects: {result.order}")

    @pytest.mark.asyncio
    async def test_generate_reverb_effect(self, mock_prompt_template):
        """Generator should successfully generate reverb effect."""
        provider = MockLLMProvider(responses=[VALID_REVERB_RESPONSE])
        generator = ParameterGenerator(llm_provider=provider)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"

            result = await generator.generate_parameters(
                description="spacious ambient atmosphere",
                effects=["reverb"]
            )

            assert isinstance(result, EffectChain)
            assert len(result.effects) == 1
            assert result.effects[0].effect_type == "reverb"
            reverb = result.effects[0]
            assert 0 <= reverb.room_size <= 1
            assert 0 <= reverb.wet_level <= 1
            print(f"✓ Generated reverb: room_size={reverb.room_size}, wet={reverb.wet_level}")

    @pytest.mark.asyncio
    async def test_generate_with_markdown_wrapped_json(self, mock_prompt_template):
        """Generator should extract JSON from markdown code blocks."""
        provider = MockLLMProvider(responses=[MARKDOWN_WRAPPED_RESPONSE])
        generator = ParameterGenerator(llm_provider=provider)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"

            result = await generator.generate_parameters(
                description="warm vocal",
                effects=["eq"]
            )

            assert isinstance(result, EffectChain)
            assert len(result.effects) == 1
            print("✓ Successfully extracted JSON from markdown code block")

    @pytest.mark.asyncio
    async def test_generate_with_custom_temperature(self, mock_prompt_template):
        """Generator should accept custom temperature parameter."""
        provider = MockLLMProvider(responses=[VALID_EQ_RESPONSE])
        generator = ParameterGenerator(llm_provider=provider)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"

            result = await generator.generate_parameters(
                description="test",
                effects=["eq"],
                temperature=0.9,
                max_tokens=1000
            )

            assert isinstance(result, EffectChain)
            # Check that request was made with correct parameters
            assert len(provider.requests) == 1
            request = provider.requests[0]
            assert request.temperature == 0.9
            assert request.max_tokens == 1000
            print(f"✓ Generated with custom parameters: temp={request.temperature}, max_tokens={request.max_tokens}")


class TestErrorHandling:
    """Test error handling in parameter generation."""

    @pytest.mark.asyncio
    async def test_invalid_effect_type(self, generator):
        """Generator should raise ValueError for invalid effect types."""
        with pytest.raises(ValueError) as exc_info:
            await generator.generate_parameters(
                description="test",
                effects=["invalid_effect"]
            )

        error = exc_info.value
        print(f"✓ Invalid effect type rejected: {error}")
        assert "Invalid effect type" in str(error)

    @pytest.mark.asyncio
    async def test_llm_provider_failure(self, mock_prompt_template):
        """Generator should raise LLMProviderError when provider fails."""
        provider = MockLLMProvider(should_fail=True)
        generator = ParameterGenerator(llm_provider=provider)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"

            with pytest.raises(LLMProviderError) as exc_info:
                await generator.generate_parameters(
                    description="test",
                    effects=["eq"]
                )

            error = exc_info.value
            print(f"✓ LLM provider error caught: {error.message}")
            assert "LLM provider failed" in error.message

    @pytest.mark.asyncio
    async def test_invalid_json_response(self, mock_prompt_template):
        """Generator should handle invalid JSON with retry."""
        # First response is invalid, second is valid
        provider = MockLLMProvider(responses=[
            INVALID_JSON_RESPONSE,
            VALID_EQ_RESPONSE
        ])
        generator = ParameterGenerator(llm_provider=provider, max_correction_attempts=1)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"
            with patch('src.generation.parameter_generator.format_correction_prompt') as mock_correct:
                mock_correct.return_value = "Correction prompt"

                result = await generator.generate_parameters(
                    description="test",
                    effects=["eq"]
                )

                assert isinstance(result, EffectChain)
                assert provider.call_count == 2  # Initial + 1 correction
                print(f"✓ Recovered from invalid JSON after {provider.call_count} attempts")

    @pytest.mark.asyncio
    async def test_invalid_parameters_with_correction(self, mock_prompt_template):
        """Generator should handle out-of-range parameters with retry."""
        # First response has invalid params, second is valid
        provider = MockLLMProvider(responses=[
            INVALID_PARAMETERS_RESPONSE,
            VALID_EQ_RESPONSE
        ])
        generator = ParameterGenerator(llm_provider=provider, max_correction_attempts=1)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"
            with patch('src.generation.parameter_generator.format_correction_prompt') as mock_correct:
                mock_correct.return_value = "Correction prompt"

                result = await generator.generate_parameters(
                    description="test",
                    effects=["eq"]
                )

                assert isinstance(result, EffectChain)
                assert provider.call_count == 2
                print(f"✓ Corrected invalid parameters after {provider.call_count} attempts")

    @pytest.mark.asyncio
    async def test_max_correction_attempts_exceeded(self, mock_prompt_template):
        """Generator should fail after max correction attempts."""
        # All responses are invalid
        provider = MockLLMProvider(responses=[INVALID_JSON_RESPONSE])
        generator = ParameterGenerator(llm_provider=provider, max_correction_attempts=2)

        with patch('src.generation.parameter_generator.format_prompt') as mock_format:
            mock_format.return_value = "Test prompt"
            with patch('src.generation.parameter_generator.format_correction_prompt') as mock_correct:
                mock_correct.return_value = "Correction prompt"

                with pytest.raises(GenValidationError) as exc_info:
                    await generator.generate_parameters(
                        description="test",
                        effects=["eq"]
                    )

                error = exc_info.value
                print(f"✓ Failed after max attempts: {error.message}")
                assert "after 2 correction attempts" in error.message
                assert provider.call_count == 3  # Initial + 2 corrections


class TestJSONParsing:
    """Test JSON parsing and extraction."""

    def test_extract_json_from_plain_text(self, generator):
        """Parser should extract JSON from plain response."""
        json_str = '{"test": "value"}'
        result = generator._extract_json(json_str)
        assert result == json_str
        print("✓ Plain JSON extracted correctly")

    def test_extract_json_from_markdown_json_block(self, generator):
        """Parser should extract JSON from ```json blocks."""
        text = "```json\n{\"test\": \"value\"}\n```"
        result = generator._extract_json(text)
        assert result == '{"test": "value"}'
        print("✓ JSON extracted from markdown json block")

    def test_extract_json_from_generic_code_block(self, generator):
        """Parser should extract JSON from generic ``` blocks."""
        text = "```\n{\"test\": \"value\"}\n```"
        result = generator._extract_json(text)
        assert result == '{"test": "value"}'
        print("✓ JSON extracted from generic code block")

    def test_extract_json_from_text_with_explanation(self, generator):
        """Parser should extract JSON when surrounded by explanatory text."""
        text = "Here's the result:\n{\"test\": \"value\"}\nHope this helps!"
        result = generator._extract_json(text)
        assert result == '{"test": "value"}'
        print("✓ JSON extracted from text with explanations")


class TestPromptVersioning:
    """Test prompt versioning support."""

    def test_different_prompt_versions(self, mock_llm_provider):
        """Generator should support different prompt versions."""
        with patch('src.generation.parameter_generator.load_prompt_template') as mock_template:
            mock_template.return_value.system_prompt = "Test system prompt"

            gen_v1 = ParameterGenerator(mock_llm_provider, prompt_version="v1")
            gen_v2 = ParameterGenerator(mock_llm_provider, prompt_version="v2")

            assert gen_v1.prompt_version == "v1"
            assert gen_v2.prompt_version == "v2"
            print(f"✓ Multiple prompt versions supported: v1, v2")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
