# System Architecture

Technical design and architecture of the baseline system.

## Overview

The baseline system implements a closed-loop architecture enabling LLMs to iteratively refine audio effect parameters based on self-evaluated scores.

```
┌─────────────┐
│    User     │
│ Description │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  LLM Provider   │  ← Temperature, Model
│  (Claude/OR)    │
└────────┬────────┘
         │ JSON Parameters
         ▼
┌─────────────────┐
│ Audio Processor │  ← Effects (Reverb, EQ)
│  (Pedalboard)   │
└────────┬────────┘
         │ Processed Audio
         ▼
┌─────────────────┐
│  Score Judge    │  ← CLAP/LLM Judge
│   (Evaluate)    │
└────────┬────────┘
         │ Score + Feedback
         │
         └──────► (Loop back for refinement)
```

## Core Components

### 1. LLM Provider Abstraction

**Purpose:** Generate audio effect parameters from text descriptions

**Design:**
```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        pass
```

**Implementations:**
- `AnthropicProvider` - Uses Claude API
- `OpenRouterProvider` - Uses OpenRouter API
- `MockLLMProvider` - For testing (no API calls)

**Key Features:**
- Async generation for better performance
- Built-in retry logic with exponential backoff
- Rate limiting to respect API quotas
- Token usage tracking
- Request/response logging

**Location:** `models/llm_judge/`

### 2. Audio Processing Pipeline

**Purpose:** Apply audio effects based on generated parameters

**Stack:**
- **librosa** - Audio loading/analysis
- **soundfile** - File I/O
- **pedalboard** - High-quality audio effects (Spotify's library)

**Pipeline:**
```
Load Audio → Validate → Apply Reverb → Apply EQ → Normalize → Save
```

**Effects:**
- **Reverb**: `delay_time`, `decay`, `stereo_spread`, `cutoff_freq`, `wet_dry`
- **EQ**: `low_gain`, `mid_gain`, `high_gain` + frequency bands

**Why Pedalboard?**
- Professional-grade DSP
- Fast C++ implementation
- Python bindings
- Used in production at Spotify

**Location:** `src/processing/`

### 3. Scoring System

**Purpose:** Evaluate how well processed audio matches description

**Methods:**

#### Embedding Similarity (Default)
```python
embedding_desc = clap.encode_text(description)
embedding_audio = clap.encode_audio(processed_audio)
score = cosine_similarity(embedding_desc, embedding_audio)
```

- **Model**: CLAP (Contrastive Language-Audio Pretraining)
- **Speed**: Fast (~100ms per score)
- **Range**: 0-1 (higher = better match)
- **Use**: Default for most experiments

#### LLM Judge (Experimental)
```python
prompt = f"Rate how well this audio matches: {description}"
response = llm.generate(prompt + audio_features)
score = parse_score(response)  # With reasoning
```

- **Model**: Same LLM as parameter generation
- **Speed**: Slow (~2-5s per score)
- **Range**: 0-100 with detailed reasoning
- **Use**: High-stakes evaluation, research analysis

**Location:** `src/scoring/`

### 4. Experiment Runner

**Purpose:** Orchestrate complete experiments with checkpointing

**Components:**

#### ExperimentRunner
- Single experiment execution
- Parameter generation → Audio processing → Scoring
- Error handling and retries
- Metadata collection

#### BatchRunner
- Multiple experiments from prompt list
- Parallel execution (configurable workers)
- Automatic checkpointing
- Resume capability

#### CheckpointManager
- Save state every N experiments
- Enable resume on failure
- Track completed/failed experiments

#### OutputManager
- Organize experiment outputs
- Save audio, parameters, metadata
- Generate logs
- Create structured directories

**Location:** `src/runner/`

### 5. CLI Interface

**Purpose:** User-friendly command-line interface

**Commands:**
- `run-single` - One experiment
- `run-batch` - Multiple experiments
- `resume` - Continue interrupted batch
- `evaluate` - Analyze results

**Built with:** Click framework

**Location:** `src/runner/cli.py`

## Data Flow

### Single Experiment Flow

```
1. User Input
   ├─ Description: "warm reverb with natural decay"
   ├─ Audio file: test.wav
   └─ Config: temperature=0.7

2. Parameter Generation
   ├─ Build prompt with description
   ├─ Call LLM API
   ├─ Parse JSON response
   ├─ Validate parameters
   └─ Retry if needed

3. Audio Processing
   ├─ Load audio (librosa)
   ├─ Apply reverb (pedalboard)
   │   ├─ delay_time: 0.03
   │   ├─ decay: 0.7
   │   └─ wet_dry: 0.6
   ├─ Apply EQ (pedalboard)
   ├─ Normalize levels
   └─ Save output.wav

4. Scoring
   ├─ Load CLAP model
   ├─ Encode description → embedding_desc
   ├─ Encode audio → embedding_audio
   ├─ Compute similarity → score
   └─ Save to metadata.json

5. Output
   ├─ outputs/experiment1/
   │   ├─ input.wav
   │   ├─ output.wav
   │   ├─ parameters.json
   │   ├─ metadata.json
   │   └─ logs/experiment.log
   └─ Return score
```

### Batch Experiment Flow

```
1. Load Batch Config
   ├─ Read prompts.txt (N descriptions)
   ├─ Configure parallel workers
   └─ Set checkpoint interval

2. Initialize Workers
   ├─ Create worker pool
   ├─ Load checkpoint (if resume)
   └─ Distribute work

3. Process Experiments (Parallel)
   ├─ Worker 1: Experiment 1
   ├─ Worker 2: Experiment 2
   ├─ Worker 3: Experiment 3
   └─ Worker 4: Experiment 4
   (Each runs full single experiment flow)

4. Checkpointing
   ├─ Every 5 experiments (configurable)
   ├─ Save: completed IDs, current state
   └─ Enable resume on interruption

5. Aggregation
   ├─ Collect all results
   ├─ Generate summary statistics
   └─ Create comparison reports
```

## Design Decisions

### Why Provider Pattern?

**Problem:** Need to support multiple LLM services (Anthropic, OpenRouter, future providers)

**Solution:** Abstract base class with concrete implementations

**Benefits:**
- Easy to add new providers
- Swap providers without changing core logic
- Consistent interface across providers
- Testable with mock provider

### Why Async/Await?

**Problem:** LLM API calls are I/O bound (waiting for network)

**Solution:** Use Python's asyncio for concurrent operations

**Benefits:**
- Better performance for batch processing
- Non-blocking API calls
- Efficient resource utilization
- Scalable to many parallel requests

### Why Checkpoint System?

**Problem:** Long-running experiments can fail (API issues, power loss, etc.)

**Solution:** Periodic state saving with resume capability

**Benefits:**
- No lost work on interruption
- Resume from exact same point
- Efficient for large datasets
- Confidence for overnight runs

### Why Separate Scoring Methods?

**Problem:** Different use cases need different evaluation approaches

**Solution:** Pluggable scoring strategies

**Benefits:**
- Fast embedding for iteration/development
- Detailed LLM judge for final evaluation
- Easy to add new scoring methods
- Compare scoring approaches

### Why Configuration Files?

**Problem:** Experiments need to be reproducible

**Solution:** YAML configuration with version control

**Benefits:**
- Document exact experiment parameters
- Share configurations with team
- Reproduce results later
- A/B test different settings

## Error Handling Strategy

### Fail Fast
- Invalid config → Error on startup
- Missing API keys → Error before first request
- Bad audio file → Error during validation

### Retry with Backoff
- Network timeouts → 3 retries with exponential backoff
- Rate limits → Wait and retry
- Transient errors → Automatic recovery

### Graceful Degradation
- One experiment fails → Continue batch
- Missing optional feature → Log warning, continue
- Scoring fails → Save parameters anyway

### User-Friendly Messages
- Clear error messages
- Actionable solutions
- Link to troubleshooting guide

## Performance Considerations

### Bottlenecks
1. **LLM API calls** - ~1-3s per request (unavoidable)
2. **Audio processing** - ~0.1-0.5s (fast with pedalboard)
3. **Scoring** - ~0.1s (embedding) or ~2s (LLM judge)
4. **I/O operations** - ~0.05s (minimal)

### Optimizations
- **Parallel batch processing** - 4x speedup with 4 workers
- **Async API calls** - Non-blocking I/O
- **Efficient audio processing** - C++ backend (pedalboard)
- **Caching** - Reuse loaded models
- **Streaming** - Process audio in chunks (future)

### Scalability
- **Current**: 10-20 experiments/minute (single worker)
- **Parallel**: 40-80 experiments/minute (4 workers)
- **Bottleneck**: LLM API rate limits (not system design)

## Testing Architecture

### Unit Tests
- Each component tested in isolation
- Mock external dependencies (LLM, audio files)
- Fast execution (<30 seconds)
- 80%+ code coverage

### Integration Tests
- Full pipeline execution
- Real audio processing
- Mock LLM (no API costs)
- Verify end-to-end flow

### Performance Tests
- Benchmark critical operations
- Ensure no regressions
- Monitor resource usage

**Location:** `tests/`

## Future Enhancements

### Planned
- Iterative refinement loop (feedback to LLM)
- More audio effects (compression, delay, distortion)
- Real-time parameter adjustment
- Web interface for experiments
- Distributed processing for large datasets

### Research Directions
- NIMA-style preference modeling
- Vibe-based evaluation dimensions
- Human-in-the-loop refinement
- Multi-modal scoring (audio + visual)

## Module Dependency Graph

```
src/runner/cli.py
    └── src/runner/experiment.py
            ├── models/llm_judge/
            │       ├── base.py
            │       ├── anthropic.py
            │       └── openrouter.py
            ├── src/processing/effects.py
            │       └── pedalboard (external)
            ├── src/scoring/scorer.py
            │       └── transformers (CLAP)
            └── src/runner/checkpoint.py
```

## Configuration Flow

```
1. Default Values (hardcoded)
2. Config File (YAML)
3. Environment Variables (.env)
4. CLI Arguments
   └── (later overrides earlier)
```

## See Also

- [Usage Guide](usage.md) - How to use the system
- [Configuration](configuration.md) - All config options
- [API Reference](api.md) - Code documentation
- [Examples](examples.md) - Practical examples
