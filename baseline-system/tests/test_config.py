"""
Test suite for configuration system.

Tests the YAML config loading, environment variable loading, and validation.
Following TDD approach - these tests should fail initially.
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path


class TestConfigLoading:
    """Tests for YAML configuration file loading."""

    def test_load_config_reads_yaml_file(self):
        """Test that load_config successfully reads a YAML file."""
        # This will fail until we implement load_config
        from src.config.loader import load_config

        # Create a temporary config file with all required sections
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {
                    'provider': 'openrouter',
                    'model': 'anthropic/claude-3-sonnet'
                },
                'prompts': {
                    'generation_template': 'prompts/generation.txt',
                    'judge_template': 'prompts/judge.txt',
                    'refinement_template': 'prompts/refinement.txt'
                },
                'refinement': {
                    'max_iterations': 5,
                    'convergence_threshold': 0.1
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            config = load_config(config_path)
            assert isinstance(config, dict)
            assert 'llm' in config
            assert config['llm']['provider'] == 'openrouter'
        finally:
            os.unlink(config_path)

    def test_load_config_validates_required_keys(self):
        """Test that load_config validates required configuration keys."""
        from src.config.loader import load_config

        # Create config with missing keys
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {'llm': {'provider': 'openrouter'}}  # Missing model
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(KeyError) as exc_info:
                load_config(config_path)
            assert 'model' in str(exc_info.value).lower() or 'required' in str(exc_info.value).lower()
        finally:
            os.unlink(config_path)

    def test_load_config_has_all_required_structure(self):
        """Test that default config has all required keys from spec."""
        from src.config.loader import load_config

        # This tests against the actual default.yaml that should exist
        config_path = Path('configs/default.yaml')

        # Skip if file doesn't exist (it shouldn't during RED phase)
        if not config_path.exists():
            pytest.skip("default.yaml not yet created")

        config = load_config(str(config_path))

        # Validate structure according to Task 1.1 specs
        assert 'llm' in config
        assert 'provider' in config['llm']
        assert 'model' in config['llm']

        assert 'prompts' in config
        assert 'generation_template' in config['prompts']
        assert 'judge_template' in config['prompts']
        assert 'refinement_template' in config['prompts']

        assert 'refinement' in config
        assert 'max_iterations' in config['refinement']
        assert 'convergence_threshold' in config['refinement']

    def test_load_config_raises_error_for_nonexistent_file(self):
        """Test that load_config raises clear error for missing file."""
        from src.config.loader import load_config

        with pytest.raises(FileNotFoundError) as exc_info:
            load_config('nonexistent_config.yaml')
        assert 'not found' in str(exc_info.value).lower() or 'nonexistent' in str(exc_info.value).lower()


class TestEnvLoading:
    """Tests for environment variable loading."""

    def test_load_env_reads_dotenv_file(self):
        """Test that load_env loads variables from .env file."""
        from src.config.loader import load_env

        # Create temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('TEST_API_KEY=test_value_123\n')
            f.write('ANOTHER_KEY=another_value\n')
            env_path = f.name

        try:
            # Load the env file
            load_env(env_path)

            # Verify variables are in environment
            assert os.getenv('TEST_API_KEY') == 'test_value_123'
            assert os.getenv('ANOTHER_KEY') == 'another_value'
        finally:
            # Cleanup
            os.unlink(env_path)
            # Clean environment
            if 'TEST_API_KEY' in os.environ:
                del os.environ['TEST_API_KEY']
            if 'ANOTHER_KEY' in os.environ:
                del os.environ['ANOTHER_KEY']

    def test_load_env_loads_api_keys(self):
        """Test that API keys can be loaded from .env file."""
        from src.config.loader import load_env

        # Save original API keys
        original_keys = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
            'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
            'OPENROUTER_API_KEY': os.getenv('OPENROUTER_API_KEY')
        }

        # Create .env with API keys
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('OPENAI_API_KEY=sk-test-openai\n')
            f.write('ANTHROPIC_API_KEY=sk-ant-test\n')
            f.write('OPENROUTER_API_KEY=sk-or-test\n')
            env_path = f.name

        try:
            load_env(env_path)

            # Verify API keys are loaded
            assert os.getenv('OPENAI_API_KEY') == 'sk-test-openai'
            assert os.getenv('ANTHROPIC_API_KEY') == 'sk-ant-test'
            assert os.getenv('OPENROUTER_API_KEY') == 'sk-or-test'
        finally:
            os.unlink(env_path)
            # Restore original API keys (don't delete them!)
            for key, value in original_keys.items():
                if value is not None:
                    os.environ[key] = value
                elif key in os.environ:
                    del os.environ[key]

    def test_load_env_with_default_path(self):
        """Test that load_env can use default .env path."""
        from src.config.loader import load_env

        # Save current directory
        original_dir = os.getcwd()

        # Create temporary directory with .env
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            env_file = Path(tmpdir) / '.env'
            env_file.write_text('DEFAULT_TEST_KEY=default_value\n')

            try:
                # Should load from current directory by default
                load_env()
                assert os.getenv('DEFAULT_TEST_KEY') == 'default_value'
            finally:
                os.chdir(original_dir)
                if 'DEFAULT_TEST_KEY' in os.environ:
                    del os.environ['DEFAULT_TEST_KEY']


class TestConfigValidation:
    """Tests for configuration validation and error handling."""

    def test_missing_llm_provider_raises_error(self):
        """Test that missing LLM provider raises clear error."""
        from src.config.loader import load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {
                    'model': 'gpt-4'
                },
                'prompts': {
                    'generation_template': 'test',
                    'judge_template': 'test',
                    'refinement_template': 'test'
                },
                'refinement': {
                    'max_iterations': 5,
                    'convergence_threshold': 0.1
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(KeyError) as exc_info:
                load_config(config_path)
            assert 'provider' in str(exc_info.value).lower()
        finally:
            os.unlink(config_path)

    def test_missing_prompts_section_raises_error(self):
        """Test that missing prompts section raises clear error."""
        from src.config.loader import load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {
                    'provider': 'openai',
                    'model': 'gpt-4'
                },
                'refinement': {
                    'max_iterations': 5,
                    'convergence_threshold': 0.1
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(KeyError) as exc_info:
                load_config(config_path)
            assert 'prompts' in str(exc_info.value).lower()
        finally:
            os.unlink(config_path)

    def test_missing_refinement_settings_raises_error(self):
        """Test that missing refinement settings raises clear error."""
        from src.config.loader import load_config

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                'llm': {
                    'provider': 'openai',
                    'model': 'gpt-4'
                },
                'prompts': {
                    'generation_template': 'test',
                    'judge_template': 'test',
                    'refinement_template': 'test'
                }
            }
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(KeyError) as exc_info:
                load_config(config_path)
            assert 'refinement' in str(exc_info.value).lower() or 'max_iterations' in str(exc_info.value).lower()
        finally:
            os.unlink(config_path)


class TestPromptTemplates:
    """Tests for prompt template files."""

    def test_prompt_templates_exist(self):
        """Test that prompt template files exist in prompts/ directory."""
        prompts_dir = Path('prompts')

        # These should fail until we create the files
        assert (prompts_dir / 'generation.txt').exists(), "generation.txt not found"
        assert (prompts_dir / 'judge.txt').exists(), "judge.txt not found"
        assert (prompts_dir / 'refinement.txt').exists(), "refinement.txt not found"

    def test_prompt_templates_are_readable(self):
        """Test that prompt templates can be read."""
        prompts_dir = Path('prompts')

        for template_name in ['generation.txt', 'judge.txt', 'refinement.txt']:
            template_path = prompts_dir / template_name
            if template_path.exists():
                content = template_path.read_text()
                assert len(content) > 0, f"{template_name} is empty"


class TestEnvExample:
    """Tests for .env.example file."""

    def test_env_example_exists(self):
        """Test that .env.example file exists."""
        env_example = Path('.env.example')
        assert env_example.exists(), ".env.example file not found"

    def test_env_example_has_required_keys(self):
        """Test that .env.example contains all required API keys."""
        env_example = Path('.env.example')

        if not env_example.exists():
            pytest.skip(".env.example not yet created")

        content = env_example.read_text()

        # Check for all three required API keys
        assert 'OPENAI_API_KEY' in content, "OPENAI_API_KEY not in .env.example"
        assert 'ANTHROPIC_API_KEY' in content, "ANTHROPIC_API_KEY not in .env.example"
        assert 'OPENROUTER_API_KEY' in content, "OPENROUTER_API_KEY not in .env.example"
