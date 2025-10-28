"""Experiment runner for baseline system.

This module provides the core ExperimentRunner class that orchestrates
complete baseline experiments end-to-end, integrating LLM providers,
audio processing, and scoring systems.

Note: This is a stub implementation. Full implementation will be completed
in Stream C (Experiment Runner Implementation).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
import yaml

logger = structlog.get_logger()


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution.

    This dataclass holds all configuration needed to run experiments,
    loaded from YAML configuration files.
    """
    llm_provider: str
    llm_model: str
    audio_config: Dict[str, Any]
    scoring_config: Dict[str, Any]
    execution_config: Dict[str, Any]
    output_config: Dict[str, Any]
    temperature: float = 0.7
    max_tokens: int = 4096


def load_config(config_path: Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ExperimentConfig object with loaded settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or missing required fields
    """
    logger.info("Loading configuration", path=str(config_path))

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Extract required configuration sections
    try:
        llm_config = config_data.get('llm', {})
        audio_config = config_data.get('audio', {})
        scoring_config = config_data.get('scoring', {})
        execution_config = config_data.get('execution', {})
        output_config = config_data.get('output', {})

        experiment_config = ExperimentConfig(
            llm_provider=llm_config.get('provider', 'anthropic'),
            llm_model=llm_config.get('model', 'claude-3-5-sonnet-20241022'),
            audio_config=audio_config,
            scoring_config=scoring_config,
            execution_config=execution_config,
            output_config=output_config,
            temperature=llm_config.get('temperature', 0.7),
            max_tokens=llm_config.get('max_tokens', 4096)
        )

        logger.info(
            "Configuration loaded",
            provider=experiment_config.llm_provider,
            model=experiment_config.llm_model
        )

        return experiment_config

    except KeyError as e:
        raise ValueError(f"Missing required configuration field: {e}")


def validate_config(config: ExperimentConfig) -> Dict[str, Any]:
    """Validate experiment configuration.

    Checks configuration for completeness and logical consistency.

    Args:
        config: ExperimentConfig to validate

    Returns:
        Dictionary with validation results and any warnings

    Raises:
        ValueError: If configuration is invalid
    """
    warnings = []

    # Validate provider
    valid_providers = ['anthropic', 'openai', 'openrouter']
    if config.llm_provider not in valid_providers:
        warnings.append(
            f"Unknown provider '{config.llm_provider}'. "
            f"Valid providers: {', '.join(valid_providers)}"
        )

    # Validate scoring method
    scoring_method = config.scoring_config.get('method', 'embedding')
    valid_methods = ['embedding', 'llm_judge']
    if scoring_method not in valid_methods:
        warnings.append(
            f"Unknown scoring method '{scoring_method}'. "
            f"Valid methods: {', '.join(valid_methods)}"
        )

    # Validate audio sample rate
    sample_rate = config.audio_config.get('sample_rate', 44100)
    if sample_rate not in [22050, 44100, 48000]:
        warnings.append(
            f"Unusual sample rate {sample_rate}. "
            f"Common rates: 22050, 44100, 48000"
        )

    # Validate execution parameters
    workers = config.execution_config.get('workers', 1)
    if workers < 1:
        raise ValueError("workers must be >= 1")

    if workers > 16:
        warnings.append(
            f"High worker count ({workers}). "
            "Consider rate limits and resource usage."
        )

    return {
        'valid': True,
        'warnings': warnings
    }


class ExperimentRunner:
    """Orchestrate complete baseline experiment pipeline.

    This class integrates LLM providers, audio processing, and scoring
    to run complete end-to-end experiments.
    """

    def __init__(self, config: ExperimentConfig, output_dir: Path):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            output_dir: Directory for experiment outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Initializing experiment runner",
            provider=config.llm_provider,
            model=config.llm_model,
            output_dir=str(output_dir)
        )

        # Initialize pipeline components
        self.provider = self._init_provider()
        self.audio_processor = self._init_audio_processor()
        self.scorer = self._init_scorer()

        # Initialize output and checkpoint managers
        from src.runner.output import OutputManager
        from src.runner.checkpoint import CheckpointManager

        self.output_manager = OutputManager(output_dir)
        self.checkpoint_manager = CheckpointManager(output_dir)

        logger.info(
            "Experiment runner initialized successfully",
            provider=config.llm_provider,
            model=config.llm_model
        )

    def run_single(self, description: str, audio_path: Path) -> Dict[str, Any]:
        """Execute single experiment end-to-end.

        This method orchestrates the complete pipeline:
        1. Generate parameters from description using LLM
        2. Process audio with generated parameters (placeholder)
        3. Score the result using judge system
        4. Save all outputs

        Args:
            description: Audio description prompt
            audio_path: Path to input audio file

        Returns:
            Dictionary with experiment results including:
            - score: Final score from judge system
            - audio_path: Path to processed audio
            - parameters_path: Path to saved parameters JSON
            - results_path: Path to detailed results
        """
        import uuid
        import asyncio
        from datetime import datetime

        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Running single experiment",
            experiment_id=experiment_id,
            description=description,
            audio=str(audio_path)
        )

        try:
            # Step 1: Generate parameters using LLM provider
            logger.info("Step 1: Generating parameters from description")
            parameters = asyncio.run(self._generate_parameters(description, audio_path))

            # Step 2: Process audio (placeholder - no audio processing for now)
            logger.info("Step 2: Audio processing (placeholder)")
            # For now, we'll skip actual audio processing and use the input audio
            # In the future, this would apply the parameters to the audio
            processed_audio_path = audio_path

            # Step 3: Score the result
            logger.info("Step 3: Scoring parameters")
            score_result = asyncio.run(self._score_parameters(description, parameters))

            # Step 4: Save outputs
            logger.info("Step 4: Saving outputs")
            saved_audio = self.output_manager.save_audio(processed_audio_path, experiment_id)
            saved_params = self.output_manager.save_parameters(parameters, experiment_id)
            saved_score = self.output_manager.save_score(score_result, experiment_id)

            # Record success
            self.output_manager.record_success(experiment_id)

            result = {
                'experiment_id': experiment_id,
                'score': score_result.get('overall_score', 0.0),
                'audio_path': str(saved_audio),
                'parameters_path': str(saved_params),
                'score_path': str(saved_score),
                'status': 'completed',
                'description': description
            }

            logger.info(
                "Experiment completed successfully",
                experiment_id=experiment_id,
                score=result['score']
            )

            return result

        except Exception as e:
            logger.error(
                "Experiment failed",
                experiment_id=experiment_id,
                error=str(e),
                exc_info=True
            )
            self.output_manager.record_failure(experiment_id, str(e))
            raise

    def _init_provider(self):
        """Initialize LLM provider from config.

        Instantiates the appropriate provider (Anthropic, OpenAI, OpenRouter)
        based on configuration.

        Returns:
            Configured LLM provider instance

        Raises:
            ValueError: If provider type is not supported
        """
        from models.llm_judge.providers import ClaudeProvider, OpenAIProvider, OpenRouterProvider

        provider_type = self.config.llm_provider.lower()

        # Prepare provider config
        provider_config = {
            'model': self.config.llm_model,
            'temperature': self.config.temperature,
            'timeout': self.config.execution_config.get('timeout', 300)
        }

        if provider_type == 'anthropic':
            logger.info("Initializing Anthropic Claude provider")
            return ClaudeProvider(provider_config)
        elif provider_type == 'openai':
            logger.info("Initializing OpenAI provider")
            return OpenAIProvider(provider_config)
        elif provider_type == 'openrouter':
            logger.info("Initializing OpenRouter provider")
            return OpenRouterProvider(provider_config)
        else:
            raise ValueError(
                f"Unsupported provider: {provider_type}. "
                f"Supported providers: anthropic, openai, openrouter"
            )

    def _init_audio_processor(self):
        """Initialize audio processor from config.

        Note: Audio processing is not yet fully implemented in the baseline system.
        This returns None as a placeholder. Future implementation will integrate
        with Issue #7 audio processing components.

        Returns:
            None (placeholder)
        """
        logger.info("Audio processor initialization (placeholder - not yet implemented)")
        # TODO: Integrate with audio processing from Issue #7
        return None

    def _init_scorer(self):
        """Initialize scoring system from config.

        Instantiates the scoring system with the configured LLM provider
        and scoring configuration.

        Returns:
            Configured ScoringSystem instance
        """
        from src.scoring import ScoringSystem, ScoringConfig

        logger.info("Initializing scoring system")

        # Create scoring config from experiment config
        scoring_config = ScoringConfig(self.config.scoring_config)

        # Initialize scorer with LLM provider
        scorer = ScoringSystem(
            llm_provider=self.provider,
            config=scoring_config
        )

        logger.info("Scoring system initialized successfully")
        return scorer

    async def _generate_parameters(self, description: str, audio_path: Path) -> Dict[str, Any]:
        """Generate audio effect parameters from description using LLM.

        Args:
            description: Audio description prompt
            audio_path: Path to input audio file (for context)

        Returns:
            Dictionary of generated audio effect parameters
        """
        from models.llm_judge.types import LLMRequest

        # Build prompt for parameter generation
        prompt = f"""Generate audio effect parameters for the following description:

Description: "{description}"

Output a JSON object containing audio effect parameters suitable for achieving this sound.
The parameters should include effects like reverb, EQ, delay, etc.

Example format:
{{
  "reverb": {{
    "decay": 0.8,
    "wet_dry": 0.5,
    "room_size": 0.7
  }},
  "eq": {{
    "low_gain": 0.0,
    "mid_gain": 2.0,
    "high_gain": -1.0
  }}
}}

Output only the JSON object, no additional text."""

        system_prompt = """You are an expert audio engineer. Generate audio effect parameters that match the given description."""

        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            model=self.config.llm_model
        )

        response = await self.provider.generate_with_retry(request)

        # Parse JSON from response
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if not json_match:
            raise ValueError("Failed to extract JSON from LLM response")

        parameters = json.loads(json_match.group(0))
        logger.info("Generated parameters successfully", num_effects=len(parameters))

        return parameters

    async def _score_parameters(self, description: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Score generated parameters against description.

        Args:
            description: Original audio description
            parameters: Generated effect parameters

        Returns:
            Dictionary with scoring results
        """
        from src.scoring import ScoringRequest

        # Create scoring request
        request = ScoringRequest(
            description=description,
            parameters=parameters,
            iteration=1,
            previous_score=None
        )

        # Score parameters
        score_response = await self.scorer.score_parameters(request)

        # Convert to dictionary
        score_dict = {
            'overall_score': score_response.overall_score,
            'confidence': score_response.confidence,
            'dimensions': [
                {
                    'name': dim.name,
                    'score': dim.score,
                    'reasoning': dim.reasoning
                }
                for dim in score_response.dimensions
            ],
            'feedback': score_response.feedback,
            'suggestions': score_response.suggestions
        }

        return score_dict
