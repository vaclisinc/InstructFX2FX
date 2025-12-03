# Text2Preset MVP: LLM-initialized Text2FX Refinement

## Overview

This MVP demonstrates an iterative audio parameter refinement system that combines:
- **LLM**: Generates initial audio effect parameters from text descriptions
- **CLAP**: Encodes audio and text into a shared embedding space
- **DDSP**: Applies differentiable audio effects
- **Gradient Descent**: Refines parameters based on user feedback

## System Architecture

```
User Input: "it's too A, it should be B"
         ↓
    [LLM Generation]
         ↓
  Initial Parameters
         ↓
    [Apply to Audio] ← Reference Audio
         ↓
   Processed Audio
         ↓
  [CLAP Audio Encoder] ──┐
                         ├─→ [Directional Loss]
  [CLAP Text Encoder] ───┘         ↓
         ↑              [Gradient Descent]
    User Prompt                ↓
                      Refined Parameters
```

## Three Experiments

### Experiment 1: A → not A
- **Text Direction**: "it's too A" → "it should be not A"
- **Hypothesis**: CLAP understands negation and reversal
- **Example**: "too bright" → "not bright" (reduce high frequencies)

### Experiment 2: not B → B
- **Text Direction**: "it's not B" → "it should be B"
- **Hypothesis**: CLAP understands addition and enhancement
- **Example**: "not warm" → "warm" (boost low frequencies)

### Experiment 3: A → B (Full Reprompt)
- **Text Direction**: "it's too A" → "it should be B"
- **Hypothesis**: CLAP handles complex bidirectional adjustments
- **Example**: "too harsh" → "should be smooth" (reduce A, increase B)

## Key Innovation

Unlike Text2FX (random initialization), we use **LLM-generated parameters as initialization** for:
1. Faster convergence
2. Semantically meaningful starting point
3. Better final quality

## Quick Start

```bash
# Open in Google Colab
Open demo.ipynb

# Or run locally
pip install -r requirements.txt
python experiments/exp1_A_to_notA.py
```

## Files

- `demo.ipynb`: Interactive Colab notebook
- `experiments/`: Three experimental scripts
- `src/`: Core system components
- `data/`: Reference audio and prompts
- `outputs/`: Generated results

## Dependencies

- PyTorch
- CLAP (laion-ai/clap)
- dasp_pytorch
- anthropic/openai (for LLM)
- librosa, soundfile

## Results

Results will be saved to `outputs/results/` with:
- Optimized audio files
- Parameter evolution plots
- Loss curves
- Comparison metrics
