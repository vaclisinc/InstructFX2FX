# API Reference

Complete API documentation for extending and using the baseline system programmatically.

## Core Modules

### models.llm_judge.base

Base LLM provider abstraction.

#### `LLMProvider`

Abstract base class for all LLM providers.

```python
from models.llm_judge.base import LLMProvider
from models.llm_judge.types import LLMRequest, LLMResponse

class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    def __init__(
        self,
        config: Dict[str, Any],
        retry_config: Optional[RetryConfig] = None,
        rate_limit_config: Optional[RateLimitConfig] = None
    ):
        """Initialize provider with configuration."""
        pass

    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate completion from LLM.

        Args:
            request: LLM request with prompt, temperature, etc.

        Returns:
            LLMResponse with generated content and metadata

        Raises:
            ValueError: If request is invalid
            RuntimeError: If API call fails after retries
        """
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate provider configuration.

        Returns:
            True if config is valid, False otherwise
        """
        pass

    async def generate_with_retry(
        self,
        request: LLMRequest
    ) -> LLMResponse:
        """Generate with automatic retry logic.

        Applies rate limiting, retries failed requests with
        exponential backoff.

        Args:
            request: LLM request parameters

        Returns:
            LLMResponse with generated content

        Raises:
            RuntimeError: If all retry attempts fail
        """
        pass
```

**Example:**
```python
from models.llm_judge.anthropic import AnthropicProvider
from models.llm_judge.types import LLMRequest

# Initialize provider
provider = AnthropicProvider({
    "api_key": os.getenv("ANTHROPIC_API_KEY"),
    "model": "claude-3-5-sonnet-20241022"
})

# Generate
request = LLMRequest(
    prompt="Generate reverb parameters for 'warm natural decay'",
    temperature=0.7,
    max_tokens=2000
)

response = await provider.generate_with_retry(request)
print(response.content)
```

### src.scoring.scorer

Audio scoring and evaluation.

#### `ScoringSystem`

Main scoring system for evaluating audio against descriptions.

```python
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest, ScoringResponse

class ScoringSystem:
    """LLM-based scoring system for audio evaluation."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: Optional[ScoringConfig] = None
    ):
        """Initialize scoring system.

        Args:
            llm_provider: LLM provider for judge scoring
            config: Scoring configuration (dimensions, weights, etc.)
        """
        pass

    async def score_parameters(
        self,
        request: ScoringRequest
    ) -> ScoringResponse:
        """Score generated parameters (parameter-only mode).

        Args:
            request: Scoring request with description and parameters

        Returns:
            ScoringResponse with scores, feedback, suggestions

        Raises:
            ScoringError: If scoring fails after retries
            MalformedResponseError: If LLM response is invalid
        """
        pass

    async def score_with_audio(
        self,
        request: ScoringRequest,
        audio_path: str
    ) -> ScoringResponse:
        """Score with audio analysis (audio-based mode).

        Args:
            request: Scoring request
            audio_path: Path to processed audio file

        Returns:
            ScoringResponse with scores and reasoning
        """
        pass

    def parse_score_response(
        self,
        content: str
    ) -> ScoringResponse:
        """Parse LLM response into structured scores.

        Args:
            content: Raw LLM response content

        Returns:
            Parsed and validated ScoringResponse

        Raises:
            MalformedResponseError: If parsing fails
        """
        pass
```

**Example:**
```python
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest
from tests.mocks.mock_provider import MockLLMProvider

# Initialize
scorer = ScoringSystem(llm_provider=MockLLMProvider())

# Score parameters
request = ScoringRequest(
    description="warm reverb with natural decay",
    parameters={
        "reverb": {
            "delay_time": 0.03,
            "decay": 0.7,
            "wet_dry": 0.6
        }
    },
    iteration=1
)

response = await scorer.score_parameters(request)
print(f"Score: {response.overall_score}")
print(f"Feedback: {response.feedback}")
```

### src.runner.experiment

Experiment execution and orchestration.

#### `ExperimentRunner`

Runs single experiments end-to-end.

```python
from src.runner.experiment import ExperimentRunner, ExperimentConfig

class ExperimentRunner:
    """Run single experiment: generation → processing → scoring."""

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment runner.

        Args:
            config: Complete experiment configuration
        """
        pass

    async def run_single(
        self,
        description: str,
        audio_path: Path
    ) -> Dict[str, Any]:
        """Run single experiment.

        Args:
            description: Text description of desired effect
            audio_path: Path to input audio file

        Returns:
            Dictionary with:
                - parameters: Generated effect parameters
                - audio_path: Path to processed audio
                - score: Evaluation score
                - metadata: Experiment metadata

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If experiment fails
        """
        pass
```

#### `BatchRunner`

Runs batch experiments with checkpointing.

```python
from src.runner.experiment import BatchRunner

class BatchRunner:
    """Run batch experiments with parallel processing."""

    def __init__(
        self,
        config: ExperimentConfig,
        checkpoint_manager: CheckpointManager
    ):
        """Initialize batch runner.

        Args:
            config: Experiment configuration
            checkpoint_manager: For saving/loading state
        """
        pass

    async def run_batch(
        self,
        descriptions: List[str],
        audio_paths: List[Path],
        output_dir: Path
    ) -> List[Dict[str, Any]]:
        """Run batch of experiments.

        Args:
            descriptions: List of text descriptions
            audio_paths: List of input audio paths
            output_dir: Directory for all outputs

        Returns:
            List of experiment results
        """
        pass

    async def resume(
        self,
        experiment_dir: Path
    ) -> List[Dict[str, Any]]:
        """Resume interrupted batch from checkpoint.

        Args:
            experiment_dir: Directory with checkpoint.json

        Returns:
            List of all results (completed + resumed)
        """
        pass
```

**Example:**
```python
from src.runner.experiment import ExperimentRunner, ExperimentConfig
from pathlib import Path

# Configure
config = ExperimentConfig(
    llm_provider="anthropic",
    llm_model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    output_dir=Path("outputs/test")
)

# Run
runner = ExperimentRunner(config)
result = await runner.run_single(
    description="warm reverb with natural decay",
    audio_path=Path("audio/test.wav")
)

print(f"Score: {result['score']}")
print(f"Output: {result['audio_path']}")
```

### src.runner.cli

Command-line interface.

#### CLI Commands

```python
import click
from src.runner.cli import cli, run_single, run_batch, resume

@cli.command()
@click.option('--description', required=True)
@click.option('--audio', required=True, type=click.Path(exists=True))
@click.option('--output-dir', default='./outputs')
def run_single(description: str, audio: str, output_dir: str):
    """Run single experiment."""
    pass

@cli.command()
@click.option('--descriptions', required=True, type=click.Path(exists=True))
@click.option('--audio', required=True, type=click.Path(exists=True))
@click.option('--output-dir', default='./outputs')
def run_batch(descriptions: str, audio: str, output_dir: str):
    """Run batch experiments."""
    pass

@cli.command()
@click.option('--experiment-dir', required=True, type=click.Path(exists=True))
def resume(experiment_dir: str):
    """Resume interrupted experiment."""
    pass
```

**Usage:**
```bash
# Command line
python -m src.runner.cli run-single \
  --description "warm reverb" \
  --audio test.wav \
  --output-dir outputs/exp1

# Programmatic
from src.runner.cli import run_single_programmatic

result = run_single_programmatic(
    description="warm reverb",
    audio_path="test.wav",
    output_dir="outputs/exp1",
    config_path=None
)
```

## Data Models

### LLMRequest

```python
from pydantic import BaseModel

class LLMRequest(BaseModel):
    """Request to LLM provider."""
    prompt: str
    system_prompt: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    model: Optional[str] = None
```

### LLMResponse

```python
class LLMResponse(BaseModel):
    """Response from LLM provider."""
    content: str
    model: str
    tokens_used: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: str
    provider: str
```

### ScoringRequest

```python
class ScoringRequest(BaseModel):
    """Request for scoring evaluation."""
    description: str
    parameters: Dict[str, Any]
    iteration: int
    previous_score: Optional[float] = None
    audio_features: Optional[Dict[str, float]] = None
```

### ScoringResponse

```python
class ScoringResponse(BaseModel):
    """Response from scoring evaluation."""
    overall_score: float  # 0-100
    confidence: float  # 0-1
    dimensions: List[ScoreDimension]
    feedback: str
    suggestions: List[str]
```

## Extending the System

### Add New LLM Provider

```python
from models.llm_judge.base import LLMProvider
from models.llm_judge.types import LLMRequest, LLMResponse

class CustomProvider(LLMProvider):
    """Custom LLM provider implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.endpoint = config.get("endpoint")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # Implement API call
        response = await self.call_api(request)
        return LLMResponse(
            content=response["text"],
            model=request.model,
            tokens_used=response["tokens"],
            finish_reason="stop",
            provider="custom"
        )

    def validate_config(self) -> bool:
        return self.api_key is not None
```

### Add New Scoring Method

```python
from src.scoring.scorer import ScoringSystem

class CustomScorer(ScoringSystem):
    """Custom scoring implementation."""

    async def score_custom(
        self,
        audio_path: Path,
        description: str
    ) -> float:
        # Implement custom scoring logic
        # e.g., use different embedding model
        embedding = self.custom_encoder.encode(audio_path)
        text_emb = self.custom_encoder.encode_text(description)
        return cosine_similarity(embedding, text_emb)
```

### Add New Audio Effect

```python
from pedalboard import Pedalboard, Reverb, Gain

def apply_custom_effect(audio, params):
    """Apply custom audio effect."""
    board = Pedalboard([
        YourCustomEffect(**params),
        Gain(gain_db=params.get("gain", 0.0))
    ])
    return board(audio, sample_rate=44100)
```

## Utility Functions

### Audio Processing

```python
import librosa
import soundfile as sf

# Load audio
audio, sr = librosa.load("input.wav", sr=44100)

# Save audio
sf.write("output.wav", audio, sr)

# Resample
audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=48000)
```

### Configuration

```python
import yaml

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Save config
with open("config.yaml", "w") as f:
    yaml.dump(config, f)
```

## Testing Utilities

### Mock Provider

```python
from tests.mocks.mock_provider import MockLLMProvider

# Create mock for testing
mock = MockLLMProvider(response_mode="valid")

# Test without API calls
response = await mock.generate(request)
assert mock.call_count == 1
```

### Test Fixtures

```python
import pytest

@pytest.fixture
def test_audio_path(tmp_path):
    """Generate test audio file."""
    from tests.fixtures.conftest import test_audio_path
    return test_audio_path

@pytest.fixture
def sample_parameters():
    """Sample effect parameters."""
    return {
        "reverb": {"delay_time": 0.03, "decay": 0.7},
        "eq": {"low_gain": 0.0, "mid_gain": 2.0}
    }
```

## See Also

- [Architecture](architecture.md) - System design overview
- [Usage Guide](usage.md) - How to use the system
- [Examples](examples.md) - Practical code examples
- [Configuration](configuration.md) - Config options
