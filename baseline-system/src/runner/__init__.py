"""Experiment runner module for orchestrating baseline experiments.

This module provides:
- OutputManager: Organize experiment outputs (audio, parameters, scores, logs)
- CheckpointManager: Save/load checkpoint state for resuming experiments
- ExperimentRunner: Orchestrate complete baseline experiment pipeline
- BatchRunner: Execute batch experiments with parallel processing
- Configuration loading and validation utilities

These utilities enable robust, resumable experiment execution with proper
organization of outputs and comprehensive error handling.
"""

from src.runner.checkpoint import CheckpointManager
from src.runner.output import OutputManager
from src.runner.experiment import ExperimentRunner, load_config, validate_config
from src.runner.batch import BatchRunner

__all__ = [
    "CheckpointManager",
    "OutputManager",
    "ExperimentRunner",
    "load_config",
    "validate_config",
    "BatchRunner",
]
