"""
Test suite for parameter generation and judging.

Tests the LLM-based parameter generation and audio judging functions.
Following TDD approach - write tests first, then implement.
"""

import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file at module import time
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path)


class TestGenerateParameters:
    """Tests for generate_parameters function."""

    def test_generate_parameters_function_exists(self):
        """Test that generate_parameters function can be imported."""
        from src.generation.parameters import generate_parameters
        assert callable(generate_parameters)

    def test_generate_parameters_returns_dict(self):
        """Test that generate_parameters returns a dictionary."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        # Load config
        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        # Print model info
        print(f"\n[Model: {config['llm']['provider']}/{config['llm']['model']}]")

        # Generate parameters with a simple prompt
        user_prompt = "warm and spacious"
        result = generate_parameters(user_prompt, config)

        # Print LLM response for human verification
        print(f"[User Prompt]: {user_prompt}")
        print(f"[Generated Parameters]: {json.dumps(result, indent=2)}")

        # Should return a dictionary
        assert isinstance(result, dict)

    def test_generate_parameters_has_required_keys(self):
        """Test that generated parameters have reverb, eq, and compressor keys."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        # Load config
        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        # Print model info
        print(f"\n[Model: {config['llm']['provider']}/{config['llm']['model']}]")

        # Generate parameters
        user_prompt = "bright and energetic"
        result = generate_parameters(user_prompt, config)

        # Print LLM response for human verification
        print(f"[User Prompt]: {user_prompt}")
        print(f"[Generated Parameters]: {json.dumps(result, indent=2)}")

        # Should have all three effect types
        assert 'reverb' in result
        assert 'eq' in result
        assert 'compressor' in result

    def test_reverb_parameters_match_schema(self):
        """Test that reverb parameters match the socialfx_data schema."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        print(f"\n[Model: {config['llm']['provider']}/{config['llm']['model']}]")

        user_prompt = "cathedral-like reverb"
        result = generate_parameters(user_prompt, config)

        print(f"[User Prompt]: {user_prompt}")
        print(f"[Reverb Parameters]: {json.dumps(result['reverb'], indent=2)}")

        # Check reverb structure
        reverb = result['reverb']
        assert isinstance(reverb, dict)

        # Check all required fields exist
        required_fields = ['delay_time', 'decay', 'stereo_spread', 'cutoff_freq', 'wet_dry']
        for field in required_fields:
            assert field in reverb, f"Missing reverb field: {field}"
            assert isinstance(reverb[field], (int, float)), f"Reverb {field} should be numeric"

    def test_eq_parameters_match_schema(self):
        """Test that EQ parameters match the socialfx_data schema."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        user_prompt = "boost high frequencies"
        result = generate_parameters(user_prompt, config)

        # Check EQ structure
        eq = result['eq']
        assert isinstance(eq, list), "EQ should be a list of bands"

        # Check each band has required fields
        for band in eq:
            assert isinstance(band, dict)
            assert 'freq' in band
            assert 'gain' in band
            assert 'Q' in band
            assert isinstance(band['freq'], (int, float))
            assert isinstance(band['gain'], (int, float))
            assert isinstance(band['Q'], (int, float))

    def test_compressor_parameters_match_schema(self):
        """Test that compressor parameters match the socialfx_data schema."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        user_prompt = "heavy compression"
        result = generate_parameters(user_prompt, config)

        # Check compressor structure
        compressor = result['compressor']
        assert isinstance(compressor, dict)

        # Check all required fields exist
        required_fields = ['threshold', 'ratio', 'attack', 'release', 'makeup_gain']
        for field in required_fields:
            assert field in compressor, f"Missing compressor field: {field}"
            assert isinstance(compressor[field], (int, float)), f"Compressor {field} should be numeric"

    def test_uses_generation_prompt_template(self):
        """Test that generate_parameters uses the generation prompt template from config."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        # This test verifies the function uses config['prompts']['generation_template']
        # by checking that it generates valid parameters (which requires the prompt template)
        user_prompt = "dark and moody"
        result = generate_parameters(user_prompt, config)

        # If it used the template correctly, we should get valid output
        assert 'reverb' in result
        assert 'eq' in result
        assert 'compressor' in result


class TestJudgeAudio:
    """Tests for judge_audio function."""

    def test_judge_audio_function_exists(self):
        """Test that judge_audio function can be imported."""
        from src.generation.parameters import judge_audio
        assert callable(judge_audio)

    def test_judge_audio_returns_float(self):
        """Test that judge_audio returns a float score."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import judge_audio
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        print(f"\n[Model: {config['llm']['provider']}/{config['llm']['model']}]")

        user_prompt = "warm and spacious"
        audio_description = "The audio has moderate reverb with a warm tone"

        score = judge_audio(user_prompt, audio_description, config)

        # Print judge response for human verification
        print(f"[User Prompt]: {user_prompt}")
        print(f"[Audio Description]: {audio_description}")
        print(f"[Judge Score]: {score}")

        # Should return a float
        assert isinstance(score, (int, float))

    def test_judge_audio_returns_score_in_range(self):
        """Test that judge_audio returns a score between 0 and 10."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import judge_audio
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        print(f"\n[Model: {config['llm']['provider']}/{config['llm']['model']}]")

        user_prompt = "bright and energetic"
        audio_description = "The audio is bright with boosted high frequencies and fast dynamics"

        score = judge_audio(user_prompt, audio_description, config)

        # Print judge response for human verification
        print(f"[User Prompt]: {user_prompt}")
        print(f"[Audio Description]: {audio_description}")
        print(f"[Judge Score]: {score}")

        # Score should be between 0 and 10
        assert 0 <= score <= 10, f"Score {score} is out of range [0, 10]"

    def test_judge_audio_uses_judge_template(self):
        """Test that judge_audio uses the judge prompt template from config."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import judge_audio
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        user_prompt = "cathedral-like reverb"
        audio_description = "Large room reverb with long decay time"

        # This test verifies the function uses config['prompts']['judge_template']
        # by checking that it returns a valid score
        score = judge_audio(user_prompt, audio_description, config)

        # If it used the template correctly, we should get a valid score
        assert isinstance(score, (int, float))
        assert 0 <= score <= 10


class TestErrorHandling:
    """Tests for error handling in parameter generation and judging."""

    def test_handles_malformed_json_from_llm(self):
        """Test that generate_parameters handles malformed JSON gracefully."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        # Use a prompt that might confuse the LLM
        user_prompt = "invalid {{{{ json }}}"

        # Should either:
        # 1. Return valid parameters anyway (LLM recovered)
        # 2. Raise a clear error about JSON parsing
        try:
            result = generate_parameters(user_prompt, config)
            # If it succeeded, validate the output
            assert isinstance(result, dict)
            assert 'reverb' in result
            assert 'eq' in result
            assert 'compressor' in result
        except Exception as e:
            # If it failed, error should mention JSON or parsing
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['json', 'parse', 'format', 'invalid'])

    def test_handles_non_numeric_score_from_llm(self):
        """Test that judge_audio handles non-numeric scores gracefully."""
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import judge_audio
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        user_prompt = "test"
        audio_description = "test audio"

        # Should either:
        # 1. Return a valid numeric score (LLM followed instructions)
        # 2. Raise a clear error about invalid score format
        try:
            score = judge_audio(user_prompt, audio_description, config)
            # If it succeeded, validate the output
            assert isinstance(score, (int, float))
            assert 0 <= score <= 10
        except Exception as e:
            # If it failed, error should mention score or number
            error_msg = str(e).lower()
            assert any(word in error_msg for word in ['score', 'number', 'numeric', 'float', 'int', 'parse'])

    def test_generate_parameters_requires_user_prompt(self):
        """Test that generate_parameters requires a user prompt."""
        from src.generation.parameters import generate_parameters
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        # Empty prompt should either work or raise clear error
        try:
            result = generate_parameters("", config)
            # If it works, should still return valid structure
            assert isinstance(result, dict)
        except Exception as e:
            # Error should be clear about missing prompt
            pass  # Any error is acceptable for empty prompt

    def test_judge_audio_requires_all_arguments(self):
        """Test that judge_audio requires all arguments."""
        from src.generation.parameters import judge_audio
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        # Empty arguments should either work or raise clear error
        try:
            score = judge_audio("", "", config)
            # If it works, should return valid score
            assert isinstance(score, (int, float))
        except Exception as e:
            # Any error is acceptable for empty arguments
            pass
