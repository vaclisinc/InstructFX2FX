"""Integration tests for CLI commands."""

import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import pytest
import yaml
from click.testing import CliRunner

from src.main import cli, run_single, run_batch, resume, validate


@pytest.fixture
def cli_runner():
    """Create Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create temporary test directory structure."""
    test_dir = tmp_path / "cli_test"
    test_dir.mkdir()

    # Create subdirectories
    (test_dir / "configs").mkdir()
    (test_dir / "audio").mkdir()
    (test_dir / "outputs").mkdir()

    yield test_dir

    # Cleanup
    if test_dir.exists():
        shutil.rmtree(test_dir)


@pytest.fixture
def sample_config(temp_test_dir):
    """Create sample configuration file."""
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
            'checkpoint_interval': 5,
            'timeout': 300,
            'workers': 1,
        },
        'output': {
            'save_audio': True,
            'save_parameters': True,
        },
    }

    config_file = temp_test_dir / "configs" / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    return config_file


@pytest.fixture
def sample_audio(temp_test_dir):
    """Create sample audio file."""
    import numpy as np
    import soundfile as sf

    audio_file = temp_test_dir / "audio" / "test.wav"
    audio_data = np.zeros(44100, dtype=np.float32)
    sf.write(audio_file, audio_data, 44100)

    return audio_file


@pytest.fixture
def sample_descriptions_file(temp_test_dir):
    """Create sample descriptions file."""
    descriptions = [
        "warm cathedral atmosphere",
        "bright outdoor space",
        "intimate jazz club",
    ]

    desc_file = temp_test_dir / "descriptions.txt"
    with open(desc_file, 'w') as f:
        f.write('\n'.join(descriptions))

    return desc_file


@pytest.fixture
def multiple_audio_files(temp_test_dir):
    """Create multiple audio files."""
    import numpy as np
    import soundfile as sf

    audio_dir = temp_test_dir / "audio"

    for i in range(5):
        audio_file = audio_dir / f"test_{i:02d}.wav"
        audio_data = np.zeros(44100, dtype=np.float32)
        sf.write(audio_file, audio_data, 44100)

    return audio_dir


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_group_exists(self, cli_runner):
        """Test that CLI group is accessible."""
        result = cli_runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'Baseline System' in result.output
        assert 'run-single' in result.output
        assert 'run-batch' in result.output
        assert 'resume' in result.output
        assert 'validate' in result.output

    def test_cli_version(self, cli_runner):
        """Test that version flag works."""
        result = cli_runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert '0.1.0' in result.output


class TestRunSingleCommand:
    """Test run-single command."""

    def test_run_single_help(self, cli_runner):
        """Test run-single help message."""
        result = cli_runner.invoke(run_single, ['--help'])

        assert result.exit_code == 0
        assert '--config' in result.output
        assert '--description' in result.output
        assert '--audio' in result.output
        assert '--output-dir' in result.output
        assert '--dry-run' in result.output

    def test_run_single_missing_required_args(self, cli_runner):
        """Test that missing required arguments are caught."""
        result = cli_runner.invoke(run_single, [])

        assert result.exit_code != 0
        assert 'Missing option' in result.output or 'required' in result.output.lower()

    def test_run_single_config_not_found(self, cli_runner, sample_audio):
        """Test error when config file doesn't exist."""
        result = cli_runner.invoke(run_single, [
            '--config', '/nonexistent/config.yaml',
            '--description', 'test',
            '--audio', str(sample_audio),
        ])

        assert result.exit_code != 0
        # Click catches FileNotFoundError for path validation

    def test_run_single_audio_not_found(self, cli_runner, sample_config):
        """Test error when audio file doesn't exist."""
        result = cli_runner.invoke(run_single, [
            '--config', str(sample_config),
            '--description', 'test',
            '--audio', '/nonexistent/audio.wav',
        ])

        assert result.exit_code != 0
        # Click validates path existence

    def test_run_single_dry_run(self, cli_runner, sample_config, sample_audio):
        """Test dry-run mode."""
        result = cli_runner.invoke(run_single, [
            '--config', str(sample_config),
            '--description', 'warm cathedral',
            '--audio', str(sample_audio),
            '--dry-run',
        ])

        assert result.exit_code == 0
        assert 'Configuration is valid' in result.output
        assert 'Audio file exists' in result.output
        assert 'Dry run complete' in result.output

    @patch('src.main.ExperimentRunner')
    def test_run_single_success(
        self, mock_runner_class, cli_runner, sample_config, sample_audio, temp_test_dir
    ):
        """Test successful single experiment run."""
        # Setup mock
        mock_runner = Mock()
        mock_runner.run_single.return_value = {
            'experiment_id': 'exp_001',
            'score': 85.5,
            'audio_path': str(temp_test_dir / 'outputs' / 'audio' / 'exp_001.wav'),
            'parameters_path': str(temp_test_dir / 'outputs' / 'parameters' / 'exp_001.json'),
            'results_path': str(temp_test_dir / 'outputs' / 'scores' / 'exp_001.json'),
        }
        mock_runner_class.return_value = mock_runner

        result = cli_runner.invoke(run_single, [
            '--config', str(sample_config),
            '--description', 'warm cathedral',
            '--audio', str(sample_audio),
            '--output-dir', str(temp_test_dir / 'outputs'),
        ])

        assert result.exit_code == 0
        assert 'Experiment completed successfully' in result.output
        assert 'Score: 85.5' in result.output or 'Score: 85.50' in result.output

        # Verify runner was called correctly
        mock_runner_class.assert_called_once()
        mock_runner.run_single.assert_called_once()

    @patch('src.main.ExperimentRunner')
    def test_run_single_experiment_failure(
        self, mock_runner_class, cli_runner, sample_config, sample_audio
    ):
        """Test handling of experiment failure."""
        # Setup mock to fail
        mock_runner = Mock()
        mock_runner.run_single.side_effect = Exception("API timeout")
        mock_runner_class.return_value = mock_runner

        result = cli_runner.invoke(run_single, [
            '--config', str(sample_config),
            '--description', 'test',
            '--audio', str(sample_audio),
        ])

        assert result.exit_code != 0
        assert 'API timeout' in result.output


class TestRunBatchCommand:
    """Test run-batch command."""

    def test_run_batch_help(self, cli_runner):
        """Test run-batch help message."""
        result = cli_runner.invoke(run_batch, ['--help'])

        assert result.exit_code == 0
        assert '--config' in result.output
        assert '--descriptions' in result.output
        assert '--audio-dir' in result.output
        assert '--workers' in result.output
        assert '--checkpoint' in result.output

    def test_run_batch_missing_required_args(self, cli_runner):
        """Test that missing required arguments are caught."""
        result = cli_runner.invoke(run_batch, [])

        assert result.exit_code != 0

    @patch('src.main.BatchRunner')
    def test_run_batch_success(
        self, mock_batch_class, cli_runner, sample_config,
        sample_descriptions_file, multiple_audio_files, temp_test_dir
    ):
        """Test successful batch processing."""
        # Setup mock
        mock_batch = Mock()
        mock_batch.run_batch.return_value = {
            'total': 3,
            'completed': 3,
            'failed': 0,
            'avg_score': 82.5,
            'output_dir': str(temp_test_dir / 'outputs'),
        }
        mock_batch_class.return_value = mock_batch

        result = cli_runner.invoke(run_batch, [
            '--config', str(sample_config),
            '--descriptions', str(sample_descriptions_file),
            '--audio-dir', str(multiple_audio_files),
            '--output-dir', str(temp_test_dir / 'outputs'),
        ])

        assert result.exit_code == 0
        assert 'Batch processing complete' in result.output
        assert 'Total: 3' in result.output
        assert 'Completed: 3' in result.output
        assert 'Failed: 0' in result.output
        assert '82.5' in result.output or '82.50' in result.output

        # Verify batch runner was called
        mock_batch_class.assert_called_once()
        mock_batch.run_batch.assert_called_once()

    @patch('src.main.BatchRunner')
    def test_run_batch_with_workers(
        self, mock_batch_class, cli_runner, sample_config,
        sample_descriptions_file, multiple_audio_files
    ):
        """Test batch processing with multiple workers."""
        mock_batch = Mock()
        mock_batch.run_batch.return_value = {
            'total': 3,
            'completed': 3,
            'failed': 0,
            'avg_score': 80.0,
            'output_dir': './outputs',
        }
        mock_batch_class.return_value = mock_batch

        result = cli_runner.invoke(run_batch, [
            '--config', str(sample_config),
            '--descriptions', str(sample_descriptions_file),
            '--audio-dir', str(multiple_audio_files),
            '--workers', '4',
        ])

        assert result.exit_code == 0

        # Verify workers parameter was passed
        call_kwargs = mock_batch_class.call_args[1]
        assert call_kwargs['workers'] == 4

    @patch('src.main.BatchRunner')
    def test_run_batch_max_experiments_limit(
        self, mock_batch_class, cli_runner, sample_config,
        sample_descriptions_file, multiple_audio_files
    ):
        """Test batch processing with max_experiments limit."""
        mock_batch = Mock()
        mock_batch.run_batch.return_value = {
            'total': 2,
            'completed': 2,
            'failed': 0,
            'avg_score': 80.0,
            'output_dir': './outputs',
        }
        mock_batch_class.return_value = mock_batch

        result = cli_runner.invoke(run_batch, [
            '--config', str(sample_config),
            '--descriptions', str(sample_descriptions_file),
            '--audio-dir', str(multiple_audio_files),
            '--max-experiments', '2',
        ])

        assert result.exit_code == 0
        assert 'Limited to 2 experiments' in result.output

        # Verify only 2 descriptions were passed
        call_args = mock_batch.run_batch.call_args
        descriptions_arg = call_args[1]['descriptions']
        assert len(descriptions_arg) == 2

    @patch('src.main.BatchRunner')
    def test_run_batch_partial_failures(
        self, mock_batch_class, cli_runner, sample_config,
        sample_descriptions_file, multiple_audio_files
    ):
        """Test batch processing with some failures."""
        mock_batch = Mock()
        mock_batch.run_batch.return_value = {
            'total': 3,
            'completed': 2,
            'failed': 1,
            'avg_score': 85.0,
            'output_dir': './outputs',
        }
        mock_batch_class.return_value = mock_batch

        result = cli_runner.invoke(run_batch, [
            '--config', str(sample_config),
            '--descriptions', str(sample_descriptions_file),
            '--audio-dir', str(multiple_audio_files),
        ])

        assert result.exit_code == 0
        assert 'Failed: 1' in result.output


class TestResumeCommand:
    """Test resume command."""

    def test_resume_help(self, cli_runner):
        """Test resume help message."""
        result = cli_runner.invoke(resume, ['--help'])

        assert result.exit_code == 0
        assert '--experiment-dir' in result.output

    def test_resume_missing_checkpoint(self, cli_runner, temp_test_dir):
        """Test error when checkpoint file doesn't exist."""
        result = cli_runner.invoke(resume, [
            '--experiment-dir', str(temp_test_dir / 'outputs'),
        ])

        assert result.exit_code != 0
        assert 'No checkpoint found' in result.output

    @patch('src.main.BatchRunner.from_checkpoint')
    def test_resume_success(self, mock_from_checkpoint, cli_runner, temp_test_dir):
        """Test successful resume."""
        # Create checkpoint file
        checkpoint_data = {
            'completed': [0, 1],
            'pending': [2, 3],
            'failed': [],
            'metadata': {},
        }

        checkpoint_file = temp_test_dir / 'outputs' / 'checkpoint.json'
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        # Setup mock
        mock_batch = Mock()
        mock_batch.resume.return_value = {
            'new_completions': 2,
            'total_completed': 4,
            'failed': 0,
        }
        mock_from_checkpoint.return_value = mock_batch

        result = cli_runner.invoke(resume, [
            '--experiment-dir', str(temp_test_dir / 'outputs'),
        ])

        assert result.exit_code == 0
        assert 'Completed: 2' in result.output
        assert 'Pending: 2' in result.output
        assert 'Resuming 2 experiments' in result.output
        assert 'Resume complete' in result.output
        assert 'Newly completed: 2' in result.output

    @patch('src.main.BatchRunner.from_checkpoint')
    def test_resume_no_pending(self, mock_from_checkpoint, cli_runner, temp_test_dir):
        """Test resume when all experiments are complete."""
        # Create checkpoint with no pending
        checkpoint_data = {
            'completed': [0, 1, 2],
            'pending': [],
            'failed': [],
            'metadata': {},
        }

        checkpoint_file = temp_test_dir / 'outputs' / 'checkpoint.json'
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        result = cli_runner.invoke(resume, [
            '--experiment-dir', str(temp_test_dir / 'outputs'),
        ])

        assert result.exit_code == 0
        assert 'No pending experiments' in result.output
        assert 'all complete' in result.output

    @patch('src.main.BatchRunner.from_checkpoint')
    def test_resume_with_failures(self, mock_from_checkpoint, cli_runner, temp_test_dir):
        """Test resume with some previously failed experiments."""
        checkpoint_data = {
            'completed': [0],
            'pending': [1, 2],
            'failed': [3],
            'metadata': {},
        }

        checkpoint_file = temp_test_dir / 'outputs' / 'checkpoint.json'
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f)

        mock_batch = Mock()
        mock_batch.resume.return_value = {
            'new_completions': 2,
            'total_completed': 3,
            'failed': 1,
        }
        mock_from_checkpoint.return_value = mock_batch

        result = cli_runner.invoke(resume, [
            '--experiment-dir', str(temp_test_dir / 'outputs'),
        ])

        assert result.exit_code == 0
        assert 'Failed: 1' in result.output


class TestValidateCommand:
    """Test validate command."""

    def test_validate_help(self, cli_runner):
        """Test validate help message."""
        result = cli_runner.invoke(validate, ['--help'])

        assert result.exit_code == 0
        assert '--config' in result.output

    def test_validate_valid_config(self, cli_runner, sample_config):
        """Test validation of valid configuration."""
        result = cli_runner.invoke(validate, [
            '--config', str(sample_config),
        ])

        assert result.exit_code == 0
        assert 'Configuration is valid' in result.output
        assert 'Provider: anthropic' in result.output
        assert 'Model: claude-3-5-sonnet-20241022' in result.output

    def test_validate_config_with_warnings(self, cli_runner, temp_test_dir):
        """Test validation with warnings."""
        # Create config with unusual values
        config_data = {
            'llm': {
                'provider': 'unknown_provider',
                'model': 'test-model',
            },
            'audio': {
                'sample_rate': 11025,  # Unusual sample rate
            },
            'scoring': {
                'method': 'unknown_method',
            },
            'execution': {
                'workers': 1,
            },
            'output': {},
        }

        config_file = temp_test_dir / "warn_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        result = cli_runner.invoke(validate, [
            '--config', str(config_file),
        ])

        assert result.exit_code == 0
        assert 'Configuration is valid' in result.output
        assert 'Warnings:' in result.output

    def test_validate_invalid_config(self, cli_runner, temp_test_dir):
        """Test validation of invalid configuration."""
        # Create config with invalid workers value
        config_data = {
            'llm': {'provider': 'anthropic'},
            'audio': {},
            'scoring': {},
            'execution': {'workers': 0},  # Invalid!
            'output': {},
        }

        config_file = temp_test_dir / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        result = cli_runner.invoke(validate, [
            '--config', str(config_file),
        ])

        assert result.exit_code != 0
        assert 'workers must be >= 1' in result.output


class TestCLIEndToEnd:
    """End-to-end CLI workflow tests."""

    @patch('src.main.ExperimentRunner')
    @patch('src.main.BatchRunner')
    def test_full_workflow_single_to_batch(
        self, mock_batch_class, mock_runner_class,
        cli_runner, sample_config, sample_audio,
        sample_descriptions_file, multiple_audio_files, temp_test_dir
    ):
        """Test complete workflow from validation to single run to batch."""
        # Step 1: Validate config
        result = cli_runner.invoke(validate, [
            '--config', str(sample_config),
        ])
        assert result.exit_code == 0

        # Step 2: Run single experiment
        mock_runner = Mock()
        mock_runner.run_single.return_value = {
            'experiment_id': 'exp_001',
            'score': 85.0,
            'audio_path': 'out/audio/exp_001.wav',
            'parameters_path': 'out/params/exp_001.json',
            'results_path': 'out/scores/exp_001.json',
        }
        mock_runner_class.return_value = mock_runner

        result = cli_runner.invoke(run_single, [
            '--config', str(sample_config),
            '--description', 'test',
            '--audio', str(sample_audio),
        ])
        assert result.exit_code == 0

        # Step 3: Run batch
        mock_batch = Mock()
        mock_batch.run_batch.return_value = {
            'total': 3,
            'completed': 3,
            'failed': 0,
            'avg_score': 80.0,
            'output_dir': str(temp_test_dir / 'outputs'),
        }
        mock_batch_class.return_value = mock_batch

        result = cli_runner.invoke(run_batch, [
            '--config', str(sample_config),
            '--descriptions', str(sample_descriptions_file),
            '--audio-dir', str(multiple_audio_files),
        ])
        assert result.exit_code == 0

    def test_invalid_yaml_syntax(self, cli_runner, temp_test_dir):
        """Test handling of invalid YAML syntax."""
        invalid_config = temp_test_dir / "invalid.yaml"
        with open(invalid_config, 'w') as f:
            f.write("invalid: yaml: {")

        result = cli_runner.invoke(validate, [
            '--config', str(invalid_config),
        ])

        assert result.exit_code != 0
