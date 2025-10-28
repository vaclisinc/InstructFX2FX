# Usage Guide

Complete guide for running experiments with the baseline system.

## Single Experiment

Run one experiment with a single description:

```bash
cd baseline-system

# Activate virtual environment
source venv/bin/activate

# Run single experiment
python -m src.runner.cli run-single \
  --description "warm reverb with natural decay" \
  --audio audio_samples/test.wav \
  --output-dir ./outputs/experiment1
```

### Options

- `--description`: Text description of desired audio effect (required)
- `--audio`: Path to input audio file (required)
- `--output-dir`: Directory for outputs (default: ./outputs)
- `--provider`: LLM provider - `anthropic` or `openrouter` (default: anthropic)
- `--model`: Model name (default: claude-3-5-sonnet-20241022)
- `--temperature`: LLM temperature 0-1 (default: 0.7)
- `--config`: Path to YAML config file (optional)

### Output Structure

```
outputs/experiment1/
├── input.wav              # Original input audio
├── output.wav             # Processed audio with effects
├── parameters.json        # Generated effect parameters
├── metadata.json          # Experiment metadata
└── logs/
    └── experiment.log     # Detailed logs
```

## Batch Experiments

Run multiple experiments from a file:

```bash
# Create prompts file
cat > prompts.txt << EOF
warm reverb with natural decay
bright EQ boost emphasizing highs
dark atmospheric with long reverb tail
clean and transparent processing
vintage analog warmth
EOF

# Run batch
python -m src.runner.cli run-batch \
  --descriptions prompts.txt \
  --audio audio_samples/test.wav \
  --output-dir ./outputs/batch1
```

### Batch Options

- `--descriptions`: Path to file with one description per line (required)
- `--audio`: Single audio file to use for all (required)
- `--audio-dir`: Directory of audio files (alternative to --audio)
- `--output-dir`: Base directory for all outputs
- `--parallel`: Number of parallel workers (default: 1)
- `--checkpoint-interval`: Save checkpoint every N experiments (default: 5)

### Checkpoint & Resume

Experiments automatically checkpoint. Resume interrupted runs:

```bash
python -m src.runner.cli resume \
  --experiment-dir ./outputs/batch1
```

## Evaluation

Analyze experiment results:

```bash
python -m src.evaluation.compare \
  --experiment-dirs ./outputs/batch1 ./outputs/batch2 \
  --output comparison_report.html
```

### Evaluation Options

- `--experiment-dirs`: Paths to experiment directories (space-separated)
- `--output`: Output report file (HTML or JSON)
- `--metrics`: Metrics to compute (all, embedding, audio_quality)

## Configuration File

Use YAML config for reproducible experiments:

```yaml
# config/my_experiment.yaml
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.7
  max_tokens: 2000

audio:
  sample_rate: 44100
  effects:
    - reverb
    - eq

scoring:
  method: embedding
  model: clap-htsat-fused

execution:
  batch_size: 10
  checkpoint_interval: 5
  max_retries: 3
```

Run with config:

```bash
python -m src.runner.cli run-single \
  --config config/my_experiment.yaml \
  --description "warm reverb" \
  --audio test.wav
```

## Common Workflows

### Reverb Study

Test different reverb characteristics:

```bash
cat > reverb_study.txt << EOF
small room reverb
cathedral reverb with long decay
warm plate reverb
spring reverb with vintage character
tight chamber reverb
EOF

python -m src.runner.cli run-batch \
  --descriptions reverb_study.txt \
  --audio audio_samples/dry_vocals.wav \
  --output-dir outputs/reverb_study
```

### EQ Comparison

Test frequency shaping:

```bash
cat > eq_study.txt << EOF
bright and airy
dark and moody
warm and full
crisp and clear
scooped mid range
EOF

python -m src.runner.cli run-batch \
  --descriptions eq_study.txt \
  --audio audio_samples/music.wav \
  --output-dir outputs/eq_study
```

### Parameter Tuning

Test same description with different temperatures:

```bash
# Low temperature (more deterministic)
python -m src.runner.cli run-single \
  --description "warm reverb" \
  --audio test.wav \
  --temperature 0.3 \
  --output-dir outputs/temp_03

# High temperature (more creative)
python -m src.runner.cli run-single \
  --description "warm reverb" \
  --audio test.wav \
  --temperature 0.9 \
  --output-dir outputs/temp_09
```

## Scoring Methods

### Embedding Similarity (Default)

Uses CLAP model for audio-text alignment:

```bash
python -m src.runner.cli run-single \
  --description "bright reverb" \
  --audio test.wav \
  --scoring-method embedding
```

Score ranges from 0-1 (higher = better match).

### LLM Judge (Experimental)

Uses LLM to evaluate quality:

```bash
python -m src.runner.cli run-single \
  --description "warm vintage reverb" \
  --audio test.wav \
  --scoring-method llm_judge
```

Provides detailed reasoning along with score.

## Tips

1. **Start Simple**: Test with single experiments before batch
2. **Check Outputs**: Listen to processed audio to validate
3. **Monitor Logs**: Check `experiment.log` for debugging
4. **Use Checkpoints**: Enable for long batch runs
5. **Experiment with Temperature**: 0.3-0.5 for consistency, 0.7-0.9 for creativity
6. **Save Configs**: Document successful configurations in YAML files

## Next Steps

- See [Configuration Reference](configuration.md) for all options
- See [Examples](examples.md) for more experiment ideas
- See [Troubleshooting](troubleshooting.md) for common issues
