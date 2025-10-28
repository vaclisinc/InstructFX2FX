"""Batch experiment runner for processing multiple experiments.

This module provides the BatchRunner class for executing multiple experiments
in sequence or parallel, with checkpointing and progress tracking.

Note: This is a stub implementation. Full implementation will be completed
in Stream C (Experiment Runner Implementation).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import structlog

from src.runner.experiment import ExperimentConfig, ExperimentRunner
from src.runner.checkpoint import CheckpointManager
from src.runner.output import OutputManager

logger = structlog.get_logger()


class BatchRunner:
    """Execute batch experiments with checkpointing and progress tracking.

    This class manages multiple experiment runs, providing:
    - Parallel execution with worker pools
    - Automatic checkpointing for resume capability
    - Progress tracking and ETA estimates
    - Result aggregation and reporting

    Note: This is a stub implementation that will be fully implemented
    in Stream C (Experiment Runner Implementation).
    """

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        output_dir: Path,
        workers: int = 1,
        checkpoint_path: Optional[Path] = None
    ):
        """Initialize batch runner.

        Args:
            experiment_config: Configuration for experiments
            output_dir: Directory for batch outputs
            workers: Number of parallel workers (1 = sequential)
            checkpoint_path: Path to existing checkpoint for resume
        """
        self.config = experiment_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.checkpoint_path = checkpoint_path

        logger.info(
            "Initializing batch runner",
            workers=workers,
            output_dir=str(output_dir)
        )

        # Initialize components
        self.experiment_runner = ExperimentRunner(experiment_config, output_dir)
        self.checkpoint_manager = self.experiment_runner.checkpoint_manager
        self.output_manager = self.experiment_runner.output_manager

        logger.info("Batch runner initialized successfully")

    def run_batch(
        self,
        descriptions: List[str],
        audio_dir: Path
    ) -> Dict[str, Any]:
        """Execute batch of experiments.

        Args:
            descriptions: List of audio descriptions to process
            audio_dir: Directory containing input audio files

        Returns:
            Dictionary with batch results:
            - total: Total number of experiments
            - completed: Number successfully completed
            - failed: Number that failed
            - avg_score: Average score across completed experiments
            - output_dir: Path to results directory
        """
        import time
        from datetime import datetime

        logger.info(
            "Running batch experiments",
            num_descriptions=len(descriptions),
            audio_dir=str(audio_dir),
            workers=self.workers
        )

        # Get list of audio files
        audio_files = self._get_audio_files(audio_dir)

        # Ensure we have enough audio files
        if len(audio_files) < len(descriptions):
            logger.warning(
                "Not enough audio files for all descriptions",
                descriptions=len(descriptions),
                audio_files=len(audio_files)
            )

        # Match descriptions to audio files
        experiments = [
            {
                'index': i,
                'description': descriptions[i],
                'audio_path': audio_files[i % len(audio_files)]  # Cycle if not enough files
            }
            for i in range(len(descriptions))
        ]

        # Initialize checkpoint with all experiments as pending
        checkpoint_interval = self.config.execution_config.get('checkpoint_interval', 5)
        self.checkpoint_manager.save_checkpoint(
            completed=[],
            pending=list(range(len(experiments))),
            failed=[],
            metadata={
                'total_experiments': len(experiments),
                'started_at': datetime.now().isoformat(),
                'checkpoint_interval': checkpoint_interval,
                'descriptions': descriptions,
                'audio_dir': str(audio_dir)
            }
        )

        # Execute experiments
        completed = []
        failed = []
        scores = []
        start_time = time.time()

        for i, exp in enumerate(experiments):
            try:
                logger.info(
                    "Processing experiment",
                    index=exp['index'],
                    progress=f"{i+1}/{len(experiments)}",
                    description=exp['description'][:50]
                )

                result = self.experiment_runner.run_single(
                    description=exp['description'],
                    audio_path=exp['audio_path']
                )

                completed.append(exp['index'])
                scores.append(result['score'])

                # Update checkpoint at intervals
                if (i + 1) % checkpoint_interval == 0:
                    self._update_checkpoint(completed, experiments, failed)

                logger.info(
                    "Experiment completed",
                    index=exp['index'],
                    score=result['score']
                )

            except Exception as e:
                logger.error(
                    "Experiment failed",
                    index=exp['index'],
                    error=str(e),
                    exc_info=True
                )
                failed.append(exp['index'])

                # Update checkpoint after failure
                self._update_checkpoint(completed, experiments, failed)

        # Final checkpoint update
        self._update_checkpoint(completed, experiments, failed, final=True)

        # Calculate statistics
        elapsed = time.time() - start_time
        avg_score = sum(scores) / len(scores) if scores else 0.0

        result = {
            'total': len(experiments),
            'completed': len(completed),
            'failed': len(failed),
            'avg_score': avg_score,
            'output_dir': str(self.output_dir),
            'elapsed_seconds': elapsed,
            'status': 'completed'
        }

        logger.info(
            "Batch processing complete",
            **result
        )

        return result

    def _get_audio_files(self, audio_dir: Path) -> List[Path]:
        """Get list of audio files from directory.

        Args:
            audio_dir: Directory containing audio files

        Returns:
            List of audio file paths
        """
        audio_dir = Path(audio_dir)
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

        audio_files = []
        for ext in extensions:
            audio_files.extend(audio_dir.glob(f'*{ext}'))

        audio_files = sorted(audio_files)

        logger.info(
            "Found audio files",
            count=len(audio_files),
            directory=str(audio_dir)
        )

        return audio_files

    def _update_checkpoint(
        self,
        completed: List[int],
        experiments: List[Dict[str, Any]],
        failed: List[int],
        final: bool = False
    ) -> None:
        """Update checkpoint with current progress.

        Args:
            completed: List of completed experiment indices
            experiments: All experiments
            failed: List of failed experiment indices
            final: Whether this is the final checkpoint
        """
        pending = [
            exp['index'] for exp in experiments
            if exp['index'] not in completed and exp['index'] not in failed
        ]

        self.checkpoint_manager.save_checkpoint(
            completed=completed,
            pending=pending,
            failed=failed,
            metadata={
                'total_experiments': len(experiments),
                'final': final
            }
        )

        if final:
            logger.info("Final checkpoint saved")
        else:
            logger.info(
                "Checkpoint updated",
                completed=len(completed),
                pending=len(pending),
                failed=len(failed)
            )

    def resume(self) -> Dict[str, Any]:
        """Resume batch processing from checkpoint.

        Returns:
            Dictionary with resume results:
            - new_completions: Number of newly completed experiments
            - total_completed: Total completed (including previous)
            - failed: Number that failed
        """
        import time

        logger.info("Resuming batch experiments from checkpoint")

        # Load checkpoint state
        resume_state = self.checkpoint_manager.get_resume_state()

        if not resume_state['pending']:
            logger.info("No pending experiments to resume")
            return {
                'new_completions': 0,
                'total_completed': len(resume_state['completed']),
                'failed': len(resume_state['failed']),
                'status': 'already_complete'
            }

        # Get original experiment data from checkpoint metadata
        metadata = resume_state['metadata']
        descriptions = metadata.get('descriptions', [])
        audio_dir = Path(metadata.get('audio_dir', self.output_dir / 'audio'))

        # Get audio files
        audio_files = self._get_audio_files(audio_dir)

        # Reconstruct pending experiments
        pending_experiments = [
            {
                'index': idx,
                'description': descriptions[idx],
                'audio_path': audio_files[idx % len(audio_files)]
            }
            for idx in resume_state['pending']
        ]

        logger.info(
            "Resuming experiments",
            pending=len(pending_experiments),
            already_completed=len(resume_state['completed']),
            previously_failed=len(resume_state['failed'])
        )

        # Execute pending experiments
        completed = list(resume_state['completed'])
        failed = list(resume_state['failed'])
        scores = []
        start_time = time.time()
        checkpoint_interval = metadata.get('checkpoint_interval', 5)

        new_completions = 0
        for i, exp in enumerate(pending_experiments):
            try:
                logger.info(
                    "Processing resumed experiment",
                    index=exp['index'],
                    progress=f"{i+1}/{len(pending_experiments)}",
                    description=exp['description'][:50]
                )

                result = self.experiment_runner.run_single(
                    description=exp['description'],
                    audio_path=exp['audio_path']
                )

                completed.append(exp['index'])
                scores.append(result['score'])
                new_completions += 1

                # Update checkpoint at intervals
                if (i + 1) % checkpoint_interval == 0:
                    self.checkpoint_manager.save_checkpoint(
                        completed=completed,
                        pending=[e['index'] for e in pending_experiments[i+1:]],
                        failed=failed,
                        metadata=metadata
                    )

                logger.info(
                    "Resumed experiment completed",
                    index=exp['index'],
                    score=result['score']
                )

            except Exception as e:
                logger.error(
                    "Resumed experiment failed",
                    index=exp['index'],
                    error=str(e),
                    exc_info=True
                )
                failed.append(exp['index'])

                # Update checkpoint after failure
                self.checkpoint_manager.save_checkpoint(
                    completed=completed,
                    pending=[e['index'] for e in pending_experiments[i+1:]],
                    failed=failed,
                    metadata=metadata
                )

        # Final checkpoint update
        self.checkpoint_manager.save_checkpoint(
            completed=completed,
            pending=[],
            failed=failed,
            metadata={**metadata, 'final': True}
        )

        elapsed = time.time() - start_time

        result = {
            'new_completions': new_completions,
            'total_completed': len(completed),
            'failed': len(failed),
            'elapsed_seconds': elapsed,
            'status': 'completed'
        }

        logger.info(
            "Resume complete",
            **result
        )

        return result

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_data: Dict[str, Any],
        experiment_dir: Path
    ) -> 'BatchRunner':
        """Create BatchRunner from checkpoint data.

        Args:
            checkpoint_data: Loaded checkpoint state
            experiment_dir: Directory containing experiment

        Returns:
            Configured BatchRunner ready to resume
        """
        logger.info(
            "Creating batch runner from checkpoint",
            experiment_dir=str(experiment_dir)
        )

        # Load the original experiment config from checkpoint metadata
        # For now, we'll create a minimal config - in production this should be
        # stored in the checkpoint metadata
        from src.runner.experiment import ExperimentConfig

        # Create default config (this should ideally be saved in checkpoint)
        config = ExperimentConfig(
            llm_provider='anthropic',
            llm_model='claude-3-5-sonnet-20241022',
            audio_config={},
            scoring_config={
                'mode': 'parameter_only',
                'dimensions': ['semantic_match', 'technical_quality', 'specificity'],
                'weights': {
                    'semantic_match': 0.5,
                    'technical_quality': 0.3,
                    'specificity': 0.2
                }
            },
            execution_config={'checkpoint_interval': 5, 'timeout': 300},
            output_config={}
        )

        # Create BatchRunner instance
        batch_runner = cls(
            experiment_config=config,
            output_dir=experiment_dir,
            workers=1
        )

        logger.info("Batch runner created from checkpoint successfully")

        return batch_runner
