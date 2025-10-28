# Configuration Reference

Complete reference for all configuration options.

## Configuration File Structure

Configuration files use YAML format:

```yaml
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 2000
  system_prompt: null  # Optional custom prompt file

audio:
  sample_rate: 44100
  format: wav
  effects:
    - reverb
    - eq
  validation: true
  max_duration: 300  # seconds

scoring:
  method: embedding  # embedding or llm_judge
  model: clap-htsat-fused
  threshold: 0.7
  batch_size: 32

execution:
  batch_size: 10
  checkpoint_interval: 5
  max_retries: 3
  timeout: 300  # seconds per experiment
  parallel_workers: 1

output:
  base_dir: ./outputs
  save_audio: true
  save_parameters: true
  save_metadata: true
  save_logs: true
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR
```

## LLM Configuration

### provider
- **Type**: string
- **Options**: `anthropic`, `openrouter`
- **Default**: `anthropic`
- **Description**: Which LLM service to use

### model
- **Type**: string
- **Default**: `claude-3-5-sonnet-20241022`
- **Options**:
  - Anthropic: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
  - OpenRouter: `anthropic/claude-3.5-sonnet`, `meta-llama/llama-3.1-70b`
- **Description**: Specific model identifier

### temperature
- **Type**: float
- **Range**: 0.0 - 1.0
- **Default**: 0.7
- **Description**: Controls randomness in responses
  - 0.0-0.3: Very deterministic, consistent outputs
  - 0.4-0.7: Balanced creativity and consistency
  - 0.8-1.0: More creative, varied outputs

### max_tokens
- **Type**: integer
- **Default**: 2000
- **Description**: Maximum response length from LLM

### system_prompt
- **Type**: string or null
- **Default**: null (uses default prompt)
- **Description**: Path to custom system prompt file

## Audio Configuration

### sample_rate
- **Type**: integer
- **Default**: 44100
- **Common Values**: 44100, 48000, 96000
- **Description**: Audio sample rate in Hz

### format
- **Type**: string
- **Default**: wav
- **Options**: `wav`, `mp3`, `flac`
- **Description**: Output audio format

### effects
- **Type**: list of strings
- **Default**: `[reverb, eq]`
- **Available**: `reverb`, `eq`, `compressor`, `delay`
- **Description**: Which audio effects to enable

### validation
- **Type**: boolean
- **Default**: true
- **Description**: Validate audio files before processing

### max_duration
- **Type**: integer
- **Default**: 300 (5 minutes)
- **Description**: Maximum audio duration in seconds

## Scoring Configuration

### method
- **Type**: string
- **Options**: `embedding`, `llm_judge`
- **Default**: `embedding`
- **Description**: How to evaluate audio quality
  - `embedding`: Fast, uses CLAP similarity
  - `llm_judge`: Slow, uses LLM reasoning

### model
- **Type**: string
- **Default**: `clap-htsat-fused`
- **Description**: Model for embeddings (if method=embedding)

### threshold
- **Type**: float
- **Range**: 0.0 - 1.0
- **Default**: 0.7
- **Description**: Minimum acceptable score

### batch_size
- **Type**: integer
- **Default**: 32
- **Description**: Batch size for embedding computation

## Execution Configuration

### batch_size
- **Type**: integer
- **Default**: 10
- **Description**: Number of experiments per batch

### checkpoint_interval
- **Type**: integer
- **Default**: 5
- **Description**: Save checkpoint every N experiments

### max_retries
- **Type**: integer
- **Default**: 3
- **Description**: Retry failed experiments N times

### timeout
- **Type**: integer
- **Default**: 300 seconds
- **Description**: Timeout per experiment

### parallel_workers
- **Type**: integer
- **Default**: 1
- **Range**: 1-4 (recommended)
- **Description**: Number of parallel workers for batch processing

## Output Configuration

### base_dir
- **Type**: string
- **Default**: `./outputs`
- **Description**: Base directory for all outputs

### save_audio
- **Type**: boolean
- **Default**: true
- **Description**: Save processed audio files

### save_parameters
- **Type**: boolean
- **Default**: true
- **Description**: Save generated parameters as JSON

### save_metadata
- **Type**: boolean
- **Default**: true
- **Description**: Save experiment metadata

### save_logs
- **Type**: boolean
- **Default**: true
- **Description**: Save detailed logs

### log_level
- **Type**: string
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`
- **Default**: `INFO`
- **Description**: Logging verbosity level

## Environment Variables

Required API keys (in `.env` file):

```bash
# Anthropic API
ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter API
OPENROUTER_API_KEY=sk-or-...

# Optional: Custom endpoints
ANTHROPIC_BASE_URL=https://api.anthropic.com
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Example Configurations

### High Quality Research

```yaml
llm:
  model: claude-3-opus-20240229
  temperature: 0.5

audio:
  sample_rate: 96000
  effects: [reverb, eq]

scoring:
  method: llm_judge
  threshold: 0.8

output:
  log_level: DEBUG
```

### Fast Iteration

```yaml
llm:
  model: claude-3-5-sonnet-20241022
  temperature: 0.7

audio:
  sample_rate: 44100

scoring:
  method: embedding
  threshold: 0.6

execution:
  parallel_workers: 4
```

### Cost Optimization

```yaml
llm:
  provider: openrouter
  model: meta-llama/llama-3.1-70b
  temperature: 0.7

scoring:
  method: embedding  # No LLM scoring

output:
  save_audio: false  # Save disk space
```

## Configuration Priority

Settings are loaded in this order (later overrides earlier):

1. Default values (hardcoded)
2. Configuration file (--config)
3. Environment variables
4. Command-line arguments

Example:
```bash
# Config file sets temperature=0.5
# CLI overrides to 0.8
python -m src.runner.cli run-single \
  --config my_config.yaml \
  --temperature 0.8 \
  --description "test"
```

## Validation

Configuration is validated on startup:

- API keys presence
- Valid parameter ranges
- File paths exist
- Model availability

Use `--dry-run` to validate without execution:

```bash
python -m src.runner.cli run-single \
  --config my_config.yaml \
  --description "test" \
  --audio test.wav \
  --dry-run
```

## See Also

- [Usage Guide](usage.md) - How to run experiments
- [Examples](examples.md) - Example configurations
- [Troubleshooting](troubleshooting.md) - Common config issues
