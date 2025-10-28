"""Main CLI entry point for baseline system experiments.

This module provides a command-line interface for running baseline system experiments,
including single experiments, batch processing, and resuming interrupted runs.
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path to allow imports from src
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import structlog

logger = structlog.get_logger()


@click.group()
@click.version_option(version='0.1.0', prog_name='baseline-system')
def cli():
    """Baseline System - LLM-as-Music-Judge Research

    A research framework for evaluating LLM-generated audio effect parameters
    through iterative refinement and multi-dimensional scoring.

    Examples:
        Run single experiment:
        $ baseline-system run-single --config configs/experiment.yaml \\
            --description "after rain campus in October" \\
            --audio samples/input.wav

        Run batch experiments:
        $ baseline-system run-batch --config configs/experiment.yaml \\
            --descriptions prompts.txt --audio-dir samples/

        Resume interrupted experiment:
        $ baseline-system resume --experiment-dir outputs/exp_001/
    """
    pass


@cli.command()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to YAML configuration file'
)
@click.option(
    '--description',
    type=str,
    required=True,
    help='Audio description prompt (e.g., "warm jazz club atmosphere")'
)
@click.option(
    '--audio',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to input audio file'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default='./outputs',
    show_default=True,
    help='Directory for experiment outputs'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Validate configuration without executing experiment'
)
def run_single(
    config: Path,
    description: str,
    audio: Path,
    output_dir: Path,
    dry_run: bool
):
    """Run a single experiment with one audio description.

    This command executes a complete baseline experiment pipeline:
    1. Load configuration and validate settings
    2. Generate audio effect parameters from description
    3. Apply effects to input audio
    4. Score the result using the judge system
    5. Save outputs (audio, parameters, scores, logs)

    The experiment results are organized in the output directory with
    timestamped subdirectories for each run.

    Examples:
        Basic usage:
        $ baseline-system run-single \\
            --config configs/experiment.yaml \\
            --description "cathedral with stone walls" \\
            --audio samples/dry_vocal.wav

        Dry run to validate config:
        $ baseline-system run-single \\
            --config configs/experiment.yaml \\
            --description "test" \\
            --audio test.wav \\
            --dry-run
    """
    from src.runner.experiment import ExperimentRunner, load_config

    logger.info(
        "Starting single experiment",
        description=description,
        audio=str(audio),
        config=str(config),
        dry_run=dry_run
    )

    # Load and validate configuration
    try:
        experiment_config = load_config(config)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error("Failed to load configuration", error=str(e))
        raise click.ClickException(f"Configuration error: {e}")

    if dry_run:
        click.echo("✓ Configuration is valid")
        click.echo(f"✓ Audio file exists: {audio}")
        click.echo(f"✓ Output directory: {output_dir}")
        click.echo("\nDry run complete - no experiment executed")
        return

    # Initialize and run experiment
    try:
        runner = ExperimentRunner(experiment_config, output_dir)
        result = runner.run_single(description, audio)

        click.echo("\n✓ Experiment completed successfully")
        click.echo(f"  Score: {result['score']:.2f}")
        click.echo(f"  Output audio: {result['audio_path']}")
        click.echo(f"  Parameters: {result['parameters_path']}")
        click.echo(f"  Results: {result['results_path']}")

    except Exception as e:
        logger.error("Experiment failed", error=str(e))
        raise click.ClickException(f"Experiment error: {e}")


@cli.command()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to YAML configuration file'
)
@click.option(
    '--descriptions',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Text file with descriptions (one per line)'
)
@click.option(
    '--audio-dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help='Directory containing input audio files'
)
@click.option(
    '--output-dir',
    type=click.Path(path_type=Path),
    default='./outputs',
    show_default=True,
    help='Directory for batch experiment outputs'
)
@click.option(
    '--checkpoint',
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help='Resume from checkpoint file'
)
@click.option(
    '--workers',
    type=int,
    default=1,
    show_default=True,
    help='Number of parallel workers (1=sequential)'
)
@click.option(
    '--max-experiments',
    type=int,
    default=None,
    help='Maximum number of experiments to run (for testing)'
)
def run_batch(
    config: Path,
    descriptions: Path,
    audio_dir: Path,
    output_dir: Path,
    checkpoint: Optional[Path],
    workers: int,
    max_experiments: Optional[int]
):
    """Run batch experiments with multiple audio descriptions.

    This command processes multiple descriptions from a text file, pairing each
    with audio files from the specified directory. It supports:

    - Parallel execution with multiple workers
    - Automatic checkpointing for resume capability
    - Progress tracking with ETA estimates
    - Retry logic for transient failures
    - Comprehensive result aggregation

    The descriptions file should contain one description per line. Audio files
    are matched to descriptions by order or by naming convention (if using
    structured filenames).

    Examples:
        Basic batch processing:
        $ baseline-system run-batch \\
            --config configs/experiment.yaml \\
            --descriptions prompts/test_set.txt \\
            --audio-dir samples/clean/

        Parallel execution:
        $ baseline-system run-batch \\
            --config configs/experiment.yaml \\
            --descriptions prompts/full_dataset.txt \\
            --audio-dir samples/ \\
            --workers 4

        Resume from checkpoint:
        $ baseline-system run-batch \\
            --config configs/experiment.yaml \\
            --descriptions prompts/test_set.txt \\
            --audio-dir samples/ \\
            --checkpoint outputs/exp_batch/checkpoint.json
    """
    from src.runner.experiment import ExperimentRunner, load_config
    from src.runner.batch import BatchRunner

    logger.info(
        "Starting batch experiments",
        descriptions=str(descriptions),
        audio_dir=str(audio_dir),
        config=str(config),
        workers=workers,
        checkpoint=str(checkpoint) if checkpoint else None
    )

    # Load configuration
    try:
        experiment_config = load_config(config)
    except Exception as e:
        logger.error("Failed to load configuration", error=str(e))
        raise click.ClickException(f"Configuration error: {e}")

    # Read descriptions
    try:
        with open(descriptions, 'r') as f:
            description_list = [line.strip() for line in f if line.strip()]

        if max_experiments:
            description_list = description_list[:max_experiments]
            click.echo(f"Limited to {max_experiments} experiments")

        click.echo(f"Loaded {len(description_list)} descriptions")
    except Exception as e:
        raise click.ClickException(f"Failed to read descriptions: {e}")

    # Initialize batch runner
    try:
        batch_runner = BatchRunner(
            experiment_config=experiment_config,
            output_dir=output_dir,
            workers=workers,
            checkpoint_path=checkpoint
        )

        # Run batch experiments
        results = batch_runner.run_batch(
            descriptions=description_list,
            audio_dir=audio_dir
        )

        # Display summary
        click.echo("\n✓ Batch processing complete")
        click.echo(f"  Total: {results['total']}")
        click.echo(f"  Completed: {results['completed']}")
        click.echo(f"  Failed: {results['failed']}")
        click.echo(f"  Average score: {results['avg_score']:.2f}")
        click.echo(f"  Results directory: {results['output_dir']}")

    except Exception as e:
        logger.error("Batch processing failed", error=str(e))
        raise click.ClickException(f"Batch error: {e}")


@cli.command()
@click.option(
    '--experiment-dir',
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help='Directory containing interrupted experiment'
)
def resume(experiment_dir: Path):
    """Resume an interrupted experiment from checkpoint.

    This command locates the checkpoint file in the experiment directory
    and resumes processing from where it was interrupted. It:

    - Loads the checkpoint state (completed, pending, failed)
    - Restores experiment configuration
    - Continues processing pending experiments
    - Updates results with new completions

    The checkpoint is automatically saved during batch processing at
    regular intervals defined in the configuration.

    Examples:
        Resume interrupted batch:
        $ baseline-system resume \\
            --experiment-dir outputs/exp_batch_20241028_120000/
    """
    from src.runner.checkpoint import CheckpointManager
    from src.runner.batch import BatchRunner

    logger.info("Resuming experiment", experiment_dir=str(experiment_dir))

    # Load checkpoint
    checkpoint_manager = CheckpointManager(experiment_dir)

    try:
        checkpoint_data = checkpoint_manager.load_checkpoint()
    except FileNotFoundError:
        raise click.ClickException(
            f"No checkpoint found in {experiment_dir}. "
            "Ensure this is a valid experiment directory with checkpoint.json"
        )
    except Exception as e:
        raise click.ClickException(f"Failed to load checkpoint: {e}")

    click.echo(f"Loaded checkpoint from {experiment_dir}")
    click.echo(f"  Completed: {len(checkpoint_data['completed'])}")
    click.echo(f"  Pending: {len(checkpoint_data['pending'])}")
    click.echo(f"  Failed: {len(checkpoint_data['failed'])}")

    if not checkpoint_data['pending']:
        click.echo("\nNo pending experiments to resume - all complete!")
        return

    click.echo(f"\nResuming {len(checkpoint_data['pending'])} experiments...")

    # Resume batch processing
    try:
        batch_runner = BatchRunner.from_checkpoint(checkpoint_data, experiment_dir)
        results = batch_runner.resume()

        click.echo("\n✓ Resume complete")
        click.echo(f"  Newly completed: {results['new_completions']}")
        click.echo(f"  Total completed: {results['total_completed']}")
        click.echo(f"  Still failed: {results['failed']}")

    except Exception as e:
        logger.error("Resume failed", error=str(e))
        raise click.ClickException(f"Resume error: {e}")


@cli.command()
@click.option(
    '--config',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to YAML configuration file to validate'
)
def validate(config: Path):
    """Validate a configuration file without running experiments.

    This command checks:
    - YAML syntax and structure
    - Required fields and data types
    - Provider and model availability
    - Path existence for referenced files
    - Logical consistency of parameters

    Examples:
        $ baseline-system validate --config configs/experiment.yaml
    """
    from src.runner.experiment import load_config, validate_config

    click.echo(f"Validating configuration: {config}")

    try:
        experiment_config = load_config(config)
        validation_results = validate_config(experiment_config)

        click.echo("\n✓ Configuration is valid")
        click.echo(f"  Provider: {experiment_config.llm_provider}")
        click.echo(f"  Model: {experiment_config.llm_model}")
        click.echo(f"  Audio sample rate: {experiment_config.audio_config.get('sample_rate', 44100)} Hz")
        click.echo(f"  Scoring method: {experiment_config.scoring_config.get('method', 'embedding')}")

        if validation_results.get('warnings'):
            click.echo("\nWarnings:")
            for warning in validation_results['warnings']:
                click.echo(f"  ⚠ {warning}")

    except Exception as e:
        logger.error("Configuration validation failed", error=str(e))
        raise click.ClickException(f"Validation error: {e}")


if __name__ == '__main__':
    cli()
