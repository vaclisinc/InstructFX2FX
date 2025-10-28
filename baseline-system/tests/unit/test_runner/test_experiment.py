"""Tests for ExperimentRunner class."""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import pytest
import yaml

from src.runner.experiment import (
    ExperimentConfig,
    ExperimentRunner,
    load_config,
    validate_config,
)


@pytest.fixture
def temp_experiment_dir(tmp_path):
    """Create temporary experiment directory."""
    experiment_dir = tmp_path / "test_experiment"
    yield experiment_dir
    # Cleanup
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)


@pytest.fixture
def sample_config_file(tmp_path):
    """Create sample YAML configuration file."""
    config_data = {
        'llm': {
            'provider': 'anthropic',
            'model': 'claude-3-5-sonnet-20241022',
            'temperature': 0.7,
            'max_tokens': 4096,
        },
        'audio': {
            'sample_rate': 44100,
            'effects': ['reverb', 'eq'],
            'validation': True,
        },
        'scoring': {
            'method': 'embedding',
            'mode': 'parameter_only',
            'dimensions': ['semantic_match', 'technical_quality'],
            'weights': {
                'semantic_match': 0.6,
                'technical_quality': 0.4,
            },
        },
        'execution': {
            'batch_size': 10,
            'checkpoint_interval': 5,
            'max_retries': 3,
            'timeout': 300,
            'workers': 1,
        },
        'output': {
            'base_dir': './outputs',
            'save_audio': True,
            'save_parameters': True,
            'save_logs': True,
        },
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    return config_file


@pytest.fixture
def experiment_config():
    """Create sample ExperimentConfig."""
    return ExperimentConfig(
        llm_provider='anthropic',
        llm_model='claude-3-5-sonnet-20241022',
        audio_config={'sample_rate': 44100},
        scoring_config={
            'mode': 'parameter_only',
            'dimensions': ['semantic_match', 'technical_quality'],
            'weights': {'semantic_match': 0.6, 'technical_quality': 0.4},
        },
        execution_config={'timeout': 300, 'checkpoint_interval': 5},
        output_config={'save_audio': True},
        temperature=0.7,
        max_tokens=4096,
    )


@pytest.fixture
def sample_audio_file(tmp_path):
    """Create a sample audio file for testing."""
    import numpy as np
    import soundfile as sf

    audio_file = tmp_path / "test_audio.wav"
    # Create 1 second of silence at 44100 Hz
    audio_data = np.zeros(44100, dtype=np.float32)
    sf.write(audio_file, audio_data, 44100)
    return audio_file


class TestLoadConfig:
    """Test load_config function."""

    def test_load_config_success(self, sample_config_file):
        """Test successful config loading from YAML file."""
        config = load_config(sample_config_file)

        assert isinstance(config, ExperimentConfig)
        assert config.llm_provider == 'anthropic'
        assert config.llm_model == 'claude-3-5-sonnet-20241022'
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.audio_config['sample_rate'] == 44100
        assert config.scoring_config['method'] == 'embedding'
        assert config.execution_config['timeout'] == 300

    def test_load_config_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised for missing config."""
        nonexistent_file = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError) as exc_info:
            load_config(nonexistent_file)

        assert str(nonexistent_file) in str(exc_info.value)

    def test_load_config_defaults_applied(self, tmp_path):
        """Test that default values are applied when not specified."""
        minimal_config = {
            'llm': {},
            'audio': {},
            'scoring': {},
            'execution': {},
            'output': {},
        }

        config_file = tmp_path / "minimal.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(minimal_config, f)

        config = load_config(config_file)

        # Check defaults
        assert config.llm_provider == 'anthropic'
        assert config.llm_model == 'claude-3-5-sonnet-20241022'
        assert config.temperature == 0.7
        assert config.max_tokens == 4096

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test that invalid YAML raises error."""
        invalid_file = tmp_path / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: syntax: {")

        with pytest.raises(yaml.YAMLError):
            load_config(invalid_file)


class TestValidateConfig:
    """Test validate_config function."""

    def test_validate_valid_config(self, experiment_config):
        """Test validation of valid configuration."""
        result = validate_config(experiment_config)

        assert result['valid'] is True
        assert isinstance(result['warnings'], list)

    def test_validate_unknown_provider_warning(self):
        """Test that unknown provider generates warning."""
        config = ExperimentConfig(
            llm_provider='unknown_provider',
            llm_model='test-model',
            audio_config={},
            scoring_config={'method': 'embedding'},
            execution_config={'workers': 1},
            output_config={},
        )

        result = validate_config(config)

        assert result['valid'] is True
        assert any('unknown_provider' in w.lower() for w in result['warnings'])

    def test_validate_unknown_scoring_method_warning(self):
        """Test that unknown scoring method generates warning."""
        config = ExperimentConfig(
            llm_provider='anthropic',
            llm_model='test-model',
            audio_config={},
            scoring_config={'method': 'unknown_method'},
            execution_config={'workers': 1},
            output_config={},
        )

        result = validate_config(config)

        assert any('unknown_method' in w.lower() for w in result['warnings'])

    def test_validate_unusual_sample_rate_warning(self):
        """Test that unusual sample rate generates warning."""
        config = ExperimentConfig(
            llm_provider='anthropic',
            llm_model='test-model',
            audio_config={'sample_rate': 11025},
            scoring_config={'method': 'embedding'},
            execution_config={'workers': 1},
            output_config={},
        )

        result = validate_config(config)

        assert any('11025' in str(w) for w in result['warnings'])

    def test_validate_invalid_workers_raises_error(self):
        """Test that workers < 1 raises ValueError."""
        config = ExperimentConfig(
            llm_provider='anthropic',
            llm_model='test-model',
            audio_config={},
            scoring_config={'method': 'embedding'},
            execution_config={'workers': 0},
            output_config={},
        )

        with pytest.raises(ValueError) as exc_info:
            validate_config(config)

        assert "workers must be >= 1" in str(exc_info.value)

    def test_validate_high_worker_count_warning(self):
        """Test that very high worker count generates warning."""
        config = ExperimentConfig(
            llm_provider='anthropic',
            llm_model='test-model',
            audio_config={},
            scoring_config={'method': 'embedding'},
            execution_config={'workers': 32},
            output_config={},
        )

        result = validate_config(config)

        assert any('32' in str(w) and 'worker' in w.lower() for w in result['warnings'])


class TestExperimentRunnerInitialization:
    """Test ExperimentRunner initialization."""

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    def test_initialization_creates_directories(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir
    ):
        """Test that initialization creates output directory."""
        mock_provider.return_value = Mock()
        mock_audio.return_value = None
        mock_scorer.return_value = Mock()

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        assert temp_experiment_dir.exists()
        assert runner.output_dir == temp_experiment_dir
        assert runner.config == experiment_config

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    def test_initialization_creates_managers(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir
    ):
        """Test that initialization creates OutputManager and CheckpointManager."""
        mock_provider.return_value = Mock()
        mock_audio.return_value = None
        mock_scorer.return_value = Mock()

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        assert runner.output_manager is not None
        assert runner.checkpoint_manager is not None

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    def test_initialization_calls_init_methods(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir
    ):
        """Test that initialization calls all init methods."""
        mock_provider.return_value = Mock()
        mock_audio.return_value = None
        mock_scorer.return_value = Mock()

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        mock_provider.assert_called_once()
        mock_audio.assert_called_once()
        mock_scorer.assert_called_once()


class TestInitProvider:
    """Test _init_provider method."""

    @patch('src.runner.experiment.ClaudeProvider')
    def test_init_provider_anthropic(self, mock_claude, experiment_config, temp_experiment_dir):
        """Test initialization of Anthropic Claude provider."""
        with patch('src.runner.experiment.ExperimentRunner._init_audio_processor'):
            with patch('src.runner.experiment.ExperimentRunner._init_scorer'):
                runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        mock_claude.assert_called_once()
        call_args = mock_claude.call_args[0][0]
        assert call_args['model'] == 'claude-3-5-sonnet-20241022'
        assert call_args['temperature'] == 0.7
        assert call_args['timeout'] == 300

    @patch('src.runner.experiment.OpenAIProvider')
    def test_init_provider_openai(self, mock_openai, temp_experiment_dir):
        """Test initialization of OpenAI provider."""
        config = ExperimentConfig(
            llm_provider='openai',
            llm_model='gpt-4',
            audio_config={},
            scoring_config={},
            execution_config={'timeout': 300},
            output_config={},
        )

        with patch('src.runner.experiment.ExperimentRunner._init_audio_processor'):
            with patch('src.runner.experiment.ExperimentRunner._init_scorer'):
                runner = ExperimentRunner(config, temp_experiment_dir)

        mock_openai.assert_called_once()

    @patch('src.runner.experiment.OpenRouterProvider')
    def test_init_provider_openrouter(self, mock_openrouter, temp_experiment_dir):
        """Test initialization of OpenRouter provider."""
        config = ExperimentConfig(
            llm_provider='openrouter',
            llm_model='anthropic/claude-3-opus',
            audio_config={},
            scoring_config={},
            execution_config={'timeout': 300},
            output_config={},
        )

        with patch('src.runner.experiment.ExperimentRunner._init_audio_processor'):
            with patch('src.runner.experiment.ExperimentRunner._init_scorer'):
                runner = ExperimentRunner(config, temp_experiment_dir)

        mock_openrouter.assert_called_once()

    def test_init_provider_unsupported_raises_error(self, temp_experiment_dir):
        """Test that unsupported provider raises ValueError."""
        config = ExperimentConfig(
            llm_provider='unsupported_provider',
            llm_model='test-model',
            audio_config={},
            scoring_config={},
            execution_config={'timeout': 300},
            output_config={},
        )

        with patch('src.runner.experiment.ExperimentRunner._init_audio_processor'):
            with patch('src.runner.experiment.ExperimentRunner._init_scorer'):
                with pytest.raises(ValueError) as exc_info:
                    ExperimentRunner(config, temp_experiment_dir)

        assert 'unsupported provider' in str(exc_info.value).lower()


class TestInitScorer:
    """Test _init_scorer method."""

    @patch('src.runner.experiment.ScoringSystem')
    @patch('src.runner.experiment.ScoringConfig')
    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    def test_init_scorer_creates_scoring_system(
        self, mock_audio, mock_provider, mock_scoring_config, mock_scoring_system,
        experiment_config, temp_experiment_dir
    ):
        """Test that scorer is initialized with correct config."""
        mock_provider.return_value = Mock()
        mock_audio.return_value = None
        mock_config_instance = Mock()
        mock_scoring_config.return_value = mock_config_instance

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        mock_scoring_config.assert_called_once_with(experiment_config.scoring_config)
        mock_scoring_system.assert_called_once()
        call_kwargs = mock_scoring_system.call_args[1]
        assert 'llm_provider' in call_kwargs
        assert call_kwargs['config'] == mock_config_instance


class TestRunSingle:
    """Test run_single method."""

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    async def test_run_single_success(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir, sample_audio_file
    ):
        """Test successful single experiment run."""
        # Setup mocks
        mock_provider_instance = Mock()
        mock_provider_instance.generate_with_retry = AsyncMock(
            return_value=Mock(content='{"reverb": {"decay": 0.8, "wet_dry": 0.5}}')
        )
        mock_provider.return_value = mock_provider_instance

        mock_audio.return_value = None

        mock_scorer_instance = Mock()
        mock_scorer_instance.score_parameters = AsyncMock(
            return_value=Mock(
                overall_score=85.5,
                confidence=0.9,
                dimensions=[
                    Mock(name='semantic_match', score=90.0, reasoning='Good match'),
                    Mock(name='technical_quality', score=80.0, reasoning='Solid quality'),
                ],
                feedback='Great parameters',
                suggestions=['Try adjusting reverb'],
            )
        )
        mock_scorer.return_value = mock_scorer_instance

        # Create runner
        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        # Run experiment
        result = runner.run_single(
            description="warm cathedral atmosphere",
            audio_path=sample_audio_file
        )

        # Verify result structure
        assert 'experiment_id' in result
        assert 'score' in result
        assert 'audio_path' in result
        assert 'parameters_path' in result
        assert 'score_path' in result
        assert 'status' in result
        assert 'description' in result

        # Verify values
        assert result['status'] == 'completed'
        assert result['score'] == 85.5
        assert result['description'] == "warm cathedral atmosphere"

        # Verify files were created
        assert Path(result['audio_path']).exists()
        assert Path(result['parameters_path']).exists()
        assert Path(result['score_path']).exists()

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    def test_run_single_parameter_generation_failure(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir, sample_audio_file
    ):
        """Test that parameter generation failure is handled."""
        # Setup mocks to fail
        mock_provider_instance = Mock()
        mock_provider_instance.generate_with_retry = AsyncMock(
            side_effect=Exception("API timeout")
        )
        mock_provider.return_value = mock_provider_instance
        mock_audio.return_value = None
        mock_scorer.return_value = Mock()

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        # Should raise exception
        with pytest.raises(Exception) as exc_info:
            runner.run_single(
                description="test description",
                audio_path=sample_audio_file
            )

        assert "API timeout" in str(exc_info.value)

        # Verify failure was recorded
        assert runner.output_manager.metadata['experiments_failed'] == 1

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    def test_run_single_scoring_failure(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir, sample_audio_file
    ):
        """Test that scoring failure is handled."""
        # Setup mocks
        mock_provider_instance = Mock()
        mock_provider_instance.generate_with_retry = AsyncMock(
            return_value=Mock(content='{"reverb": {"decay": 0.8}}')
        )
        mock_provider.return_value = mock_provider_instance

        mock_audio.return_value = None

        mock_scorer_instance = Mock()
        mock_scorer_instance.score_parameters = AsyncMock(
            side_effect=Exception("Scoring failed")
        )
        mock_scorer.return_value = mock_scorer_instance

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        with pytest.raises(Exception) as exc_info:
            runner.run_single(
                description="test description",
                audio_path=sample_audio_file
            )

        assert "Scoring failed" in str(exc_info.value)


class TestGenerateParameters:
    """Test _generate_parameters method."""

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    @pytest.mark.asyncio
    async def test_generate_parameters_success(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir, sample_audio_file
    ):
        """Test successful parameter generation."""
        mock_provider_instance = Mock()
        mock_response = Mock(
            content='{"reverb": {"decay": 0.8, "wet_dry": 0.5}, "eq": {"gain": 2.0}}'
        )
        mock_provider_instance.generate_with_retry = AsyncMock(return_value=mock_response)
        mock_provider.return_value = mock_provider_instance

        mock_audio.return_value = None
        mock_scorer.return_value = Mock()

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        parameters = await runner._generate_parameters(
            description="warm jazz club",
            audio_path=sample_audio_file
        )

        assert 'reverb' in parameters
        assert parameters['reverb']['decay'] == 0.8
        assert parameters['reverb']['wet_dry'] == 0.5
        assert 'eq' in parameters
        assert parameters['eq']['gain'] == 2.0

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    @pytest.mark.asyncio
    async def test_generate_parameters_invalid_json(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir, sample_audio_file
    ):
        """Test that invalid JSON response raises error."""
        mock_provider_instance = Mock()
        mock_response = Mock(content='This is not JSON')
        mock_provider_instance.generate_with_retry = AsyncMock(return_value=mock_response)
        mock_provider.return_value = mock_provider_instance

        mock_audio.return_value = None
        mock_scorer.return_value = Mock()

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        with pytest.raises(ValueError) as exc_info:
            await runner._generate_parameters(
                description="test",
                audio_path=sample_audio_file
            )

        assert "failed to extract json" in str(exc_info.value).lower()


class TestScoreParameters:
    """Test _score_parameters method."""

    @patch('src.runner.experiment.ExperimentRunner._init_provider')
    @patch('src.runner.experiment.ExperimentRunner._init_audio_processor')
    @patch('src.runner.experiment.ExperimentRunner._init_scorer')
    @pytest.mark.asyncio
    async def test_score_parameters_success(
        self, mock_scorer, mock_audio, mock_provider,
        experiment_config, temp_experiment_dir
    ):
        """Test successful parameter scoring."""
        mock_provider.return_value = Mock()
        mock_audio.return_value = None

        mock_scorer_instance = Mock()
        mock_scorer_instance.score_parameters = AsyncMock(
            return_value=Mock(
                overall_score=85.0,
                confidence=0.9,
                dimensions=[
                    Mock(name='semantic_match', score=90.0, reasoning='Excellent'),
                    Mock(name='technical_quality', score=80.0, reasoning='Good'),
                ],
                feedback='Nice work',
                suggestions=['Improve reverb'],
            )
        )
        mock_scorer.return_value = mock_scorer_instance

        runner = ExperimentRunner(experiment_config, temp_experiment_dir)

        score_dict = await runner._score_parameters(
            description="warm atmosphere",
            parameters={"reverb": {"decay": 0.8}}
        )

        assert score_dict['overall_score'] == 85.0
        assert score_dict['confidence'] == 0.9
        assert len(score_dict['dimensions']) == 2
        assert score_dict['dimensions'][0]['name'] == 'semantic_match'
        assert score_dict['dimensions'][0]['score'] == 90.0
        assert score_dict['feedback'] == 'Nice work'
        assert len(score_dict['suggestions']) == 1
