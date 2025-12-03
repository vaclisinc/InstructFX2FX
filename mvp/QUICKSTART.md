# Quick Start Guide

## Installation

### Option 1: Colab (Recommended for Demo)

1. Open `demo.ipynb` in Google Colab
2. Run the first cell to install dependencies
3. Set your API key
4. Follow the interactive demo

### Option 2: Local Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Running Experiments

### Experiment 1: A → not A

Test whether CLAP understands negation.

```bash
python experiments/exp1_A_to_notA.py \
  --audio data/reference_audio/drums.wav \
  --attribute bright \
  --iterations 100
```

### Experiment 2: not B → B

Test whether CLAP understands enhancement.

```bash
python experiments/exp2_notB_to_B.py \
  --audio data/reference_audio/drums.wav \
  --attribute warm \
  --iterations 100
```

### Experiment 3: A → B

Test complex bidirectional adjustment.

```bash
python experiments/exp3_A_to_B.py \
  --audio data/reference_audio/drums.wav \
  --from-attr harsh \
  --to-attr smooth \
  --iterations 100
```

## File Structure

```
mvp/
├── demo.ipynb              # Interactive Colab demo (START HERE)
├── experiments/            # Python scripts for each experiment
│   ├── exp1_A_to_notA.py
│   ├── exp2_notB_to_B.py
│   └── exp3_A_to_B.py
├── src/                    # Core modules (clean & readable)
│   ├── clap.py            # CLAP model wrapper
│   ├── ddsp.py            # Differentiable audio FX
│   ├── llm.py             # LLM parameter generation
│   ├── refine.py          # Text2FX refinement loop
│   └── utils.py           # Helper functions
├── data/
│   ├── reference_audio/   # Put your audio files here
│   └── prompts.json       # Pre-defined test prompts
└── outputs/               # Results will be saved here
```

## Understanding the Code

### 1. LLM Generation (`src/llm.py`)

```python
# Generate initial parameters from text
params = generate_initial_params(
    llm_client=client,
    prompt="make this sound warm"
)
# Returns: {'eq': [...], 'compressor': [...], 'reverb': [...]}
```

### 2. CLAP Encoding (`src/clap.py`)

```python
# Encode audio and text to same embedding space
audio_emb = clap_model.get_audio_embedding(audio)
text_emb = clap_model.get_text_embedding("this sound is warm")
```

### 3. Directional Loss (`src/refine.py`)

```python
# Loss encourages audio direction to match text direction
loss = directional_loss(
    audio_anchor=original_audio_emb,
    audio_effected=processed_audio_emb,
    text_anchor=embedding("too bright"),
    text_target=embedding("not bright")
)
```

### 4. Full Pipeline

```python
# 1. LLM generates initial params
params_init = generate_initial_params(llm, "warm")

# 2. Convert to tensor
params_tensor = params_dict_to_tensor(params_init, fx_chain)

# 3. Refine with gradient descent
params_refined, history = refine_with_directional_loss(
    audio=audio,
    fx_chain=fx_chain,
    initial_params=params_tensor,
    text_anchor="not warm",
    text_target="warm",
    clap_model=clap,
    n_iterations=100
)

# 4. Generate final audio
audio_refined = fx_chain(audio, torch.sigmoid(params_refined))
```

## Expected Results

After running an experiment, you'll get:

```
outputs/results/exp1_A_to_notA/bright/
├── original.wav              # Original audio
├── llm_init.wav              # After LLM params
├── refined.wav               # After Text2FX refinement
├── parameters.json           # Initial & refined params
├── history.json              # Loss at each iteration
├── optimization_curve.png    # Loss curve plot
└── experiment_info.json      # Metadata
```

## Troubleshooting

### CUDA out of memory
- Use `--device cpu`
- Reduce audio length
- Use smaller batch size

### ImportError for dasp_pytorch
```bash
pip install git+https://github.com/csteinmetz1/dasp-pytorch.git
```

### API key not found
```bash
export ANTHROPIC_API_KEY="your-key"
# or create a .env file:
echo "ANTHROPIC_API_KEY=your-key" > .env
```

## Next Steps

1. ✅ Run the Colab demo
2. ✅ Test on your own audio files
3. ✅ Compare LLM init vs random init
4. ✅ Analyze parameter changes
5. ✅ Try different text prompts

## Key Innovation

Unlike Text2FX (random init), we use **LLM-generated parameters as initialization**:

- ✅ Faster convergence
- ✅ Better semantic alignment
- ✅ More interpretable results

The ablation study in `demo.ipynb` shows the comparison!
