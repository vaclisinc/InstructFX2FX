# Example Experiments

Practical examples demonstrating system capabilities.

## Quick Start Example

Simplest possible experiment:

```bash
cd baseline-system
source venv/bin/activate

python -m src.runner.cli run-single \
  --description "warm reverb with natural decay" \
  --audio audio_samples/test.wav \
  --output-dir outputs/quick_test
```

## Example 1: Reverb Study

Compare different reverb characteristics:

```bash
# Run batch with reverb prompts
python -m src.runner.cli run-batch \
  --descriptions examples/reverb_prompts.txt \
  --audio audio_samples/vocals.wav \
  --output-dir outputs/reverb_study

# Evaluate results
python -m src.evaluation.compare \
  --experiment-dirs outputs/reverb_study \
  --output reverb_comparison.html
```

**Prompts** (`examples/reverb_prompts.txt`):
- `small room reverb`
- `cathedral reverb with long decay`
- `warm plate reverb`
- `spring reverb with vintage character`

**Expected Results:**
- Small room: Short delay (10-20ms), quick decay (<1s)
- Cathedral: Long delay (50-100ms), extended decay (3-5s)
- Plate: Medium delay (30ms), warm character
- Spring: Metallic resonance, vintage character

## Example 2: EQ Shaping Study

Test frequency balance descriptions:

```bash
python -m src.runner.cli run-batch \
  --descriptions examples/eq_prompts.txt \
  --audio audio_samples/music.wav \
  --output-dir outputs/eq_study
```

**Prompts** (`examples/eq_prompts.txt`):
- `bright and airy` → High frequency boost
- `dark and moody` → Low frequency emphasis
- `warm and full` → Mid-low boost
- `crisp and clear` → Presence boost

## Example 3: Complex Descriptions

Test abstract, evocative descriptions:

```bash
python -m src.runner.cli run-batch \
  --descriptions examples/complex_prompts.txt \
  --audio audio_samples/ambient.wav \
  --output-dir outputs/complex_study \
  --temperature 0.8  # More creative interpretation
```

**Prompts** (`examples/complex_prompts.txt`):
- `after rain campus in October`
- `distant thunder approaching`
- `morning coffee shop atmosphere`
- `late night city ambience`

These test the LLM's ability to translate abstract concepts into audio parameters.

## Example 4: Temperature Comparison

Compare LLM creativity levels:

```bash
# Conservative (deterministic)
python -m src.runner.cli run-single \
  --description "warm reverb" \
  --audio test.wav \
  --temperature 0.3 \
  --output-dir outputs/temp_low

# Balanced
python -m src.runner.cli run-single \
  --description "warm reverb" \
  --audio test.wav \
  --temperature 0.7 \
  --output-dir outputs/temp_mid

# Creative
python -m src.runner.cli run-single \
  --description "warm reverb" \
  --audio test.wav \
  --temperature 0.9 \
  --output-dir outputs/temp_high
```

Listen to outputs and compare parameter consistency.

## Example 5: High Quality Research Run

Use optimized config for publication-quality results:

```bash
python -m src.runner.cli run-batch \
  --config configs/high_quality.yaml \
  --descriptions examples/reverb_prompts.txt \
  --audio audio_samples/reference.wav \
  --output-dir outputs/paper_results
```

This uses:
- Claude Opus (most capable model)
- 96kHz sample rate
- LLM judge scoring
- Strict quality threshold (0.8)

## Example 6: Fast Iteration

Rapid testing during development:

```bash
python -m src.runner.cli run-batch \
  --config configs/fast_iteration.yaml \
  --descriptions my_test_prompts.txt \
  --audio test.wav \
  --output-dir outputs/dev_test
```

This uses:
- Parallel workers (4x speedup)
- Fast embedding scoring
- Standard quality audio

## Example 7: Model Comparison

Compare different LLM providers:

```bash
# Anthropic Claude
python -m src.runner.cli run-single \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  --description "warm reverb" \
  --audio test.wav \
  --output-dir outputs/claude

# OpenRouter (alternative model)
python -m src.runner.cli run-single \
  --provider openrouter \
  --model anthropic/claude-3.5-sonnet \
  --description "warm reverb" \
  --audio test.wav \
  --output-dir outputs/openrouter
```

Compare parameter quality and cost.

## Example 8: Scoring Method Comparison

Test both scoring approaches:

```bash
# Embedding scoring (fast)
python -m src.runner.cli run-single \
  --description "bright reverb" \
  --audio test.wav \
  --scoring-method embedding \
  --output-dir outputs/scoring_embedding

# LLM judge (detailed)
python -m src.runner.cli run-single \
  --description "bright reverb" \
  --audio test.wav \
  --scoring-method llm_judge \
  --output-dir outputs/scoring_llm
```

Check `metadata.json` for scoring details.

## Research Workflows

### Prompt Engineering Study

Test variations of the same intent:

```bash
cat > prompt_variations.txt << EOF
add reverb
apply reverb effect
warm reverb processing
reverb with natural decay
subtle room ambience
EOF

python -m src.runner.cli run-batch \
  --descriptions prompt_variations.txt \
  --audio test.wav \
  --output-dir outputs/prompt_study
```

Analyze which descriptions produce best results.

### Parameter Sensitivity Analysis

Test same prompt multiple times:

```bash
for i in {1..10}; do
  python -m src.runner.cli run-single \
    --description "warm reverb" \
    --audio test.wav \
    --output-dir outputs/sensitivity/run_$i
done
```

Analyze parameter distribution and consistency.

### Dataset Evaluation

Run on entire dataset:

```bash
python -m src.runner.cli run-batch \
  --descriptions dataset/prompts.txt \
  --audio-dir dataset/audio/ \
  --output-dir outputs/dataset_eval \
  --parallel 4 \
  --checkpoint-interval 5
```

## Output Analysis

### Individual Experiment

```bash
# Listen to result
afplay outputs/experiment1/output.wav

# View parameters
cat outputs/experiment1/parameters.json | jq .

# Check score
cat outputs/experiment1/metadata.json | jq .score
```

### Batch Analysis

```bash
# Generate comparison report
python -m src.evaluation.compare \
  --experiment-dirs outputs/reverb_study outputs/eq_study \
  --output comparison.html

# Open in browser
open comparison.html
```

## Next Steps

- Modify prompts in `examples/` for your research
- Create custom configs in `configs/`
- See [Usage Guide](usage.md) for more CLI options
- See [Configuration](configuration.md) for tuning parameters
