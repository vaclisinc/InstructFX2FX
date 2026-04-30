# InstructFX2FX

Text-driven audio effect parameter optimization. Describe a sound quality (e.g. "warm", "bright", "spacious") and the system selects appropriate audio effects, initializes parameters via LLM, and optimizes them using CLAP-guided gradient descent or Bayesian optimization.

## Architecture

3-layer pipeline orchestrated by `src/pipeline/orchestrator.py`:

```
User prompt ("bright")
        │
        ▼
┌──────────────────────────────┐
│  Layer 1: FX Selector Agent  │  GPT-4o via OpenRouter tool-calling
│  src/agents/fx_selector.py   │  → ordered FX chain ["eq", "rev"]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Layer 2: Parser             │  3-case session router
│  src/agents/parser.py        │  LLM-init missing params, then optimize
└──────────────┬───────────────┘
               │
     ┌─────────┴─────────┐
     ▼                   ▼
┌──────────┐      ┌──────────────┐
│ Layer 3A │      │  Layer 3B    │
│ DASP GD  │      │ Pedalboard   │
│ (eq,rev) │─────▶│ Bayesian BO  │
│ trainer  │audio │ fxsearcher   │
└──────────┘      └──────────────┘
```

**Layer 1 — FX Selector Agent**: Each available FX is an OpenAI tool. The LLM calls them in signal-chain order.

**Layer 2 — Parser**: Routes based on session state:
- Case 1: All FX exist in session → optimize from existing params
- Case 2: None exist → LLM-initialize, then optimize
- Case 3: Mixed → LLM-initialize missing, merge with existing, then optimize

**Layer 3 — Dual-Track Optimization** (sequential):
- 3A: DASP effects (eq, rev) → CLAP gradient descent via `move_in_CLAP()`
- 3B: Pedalboard effects (comp, dist, delay, pitchshift, bitcrush) → Bayesian BO via `fxsearcher()`
- DASP output audio feeds as input to the Pedalboard stage

**Session** (`src/session/session.py`): Accumulates params across prompts for multi-turn interaction.

## Available Effects

| Canonical Name | Backend | Optimization | Params |
|---------------|---------|-------------|--------|
| `eq` | DASP | Gradient Descent | 18 (6-band parametric EQ) |
| `rev` | DASP | Gradient Descent | 25 (noise-shaped reverb) |
| `comp` | Pedalboard | Bayesian BO | Compressor |
| `dist` | Pedalboard | Bayesian BO | Distortion |
| `delay` | Pedalboard | Bayesian BO | Delay |
| `pitchshift` | Pedalboard | Bayesian BO | Pitch Shift |
| `bitcrush` | Pedalboard | Bayesian BO | Bitcrusher |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_api_key
```

## Quick Start

```python
import sys
sys.path.insert(0, "src")

from llms.llmclient import LLMClient
from embeddings.clap import CLAPWrapper
from session.session import Session
from pipeline.orchestrator import Orchestrator

llm = LLMClient()
clap = CLAPWrapper()
session = Session(available_fx=["eq", "comp", "rev", "dist"])

orch = Orchestrator(llm, clap, device="cpu")

# First prompt
out = orch.run("bright", session, dry_audio_tensor)
# Second prompt — session remembers previous FX
out = orch.run("add warmth", session, dry_audio_tensor)
```

## Project Structure

```
src/
├── agents/
│   ├── fx_selector.py      # Layer 1: LLM FX selection via tool calls
│   └── parser.py            # Layer 2: session routing + optimization dispatch
├── configurations/
│   └── config.py            # Config dataclass, enums (LossFunction, OptimizationMethod)
├── effects/
│   └── fx.py                # Effect adapters, FXChainFactory, param ranges
├── embeddings/
│   └── clap.py              # CLAP wrapper (audio/text embeddings)
├── FxSearcher/
│   └── fxsearcher.py        # Bayesian optimization for Pedalboard FX
├── llms/
│   └── llmclient.py         # OpenRouter GPT-4o client
├── metrics/                 # Audio quality and CLAP similarity metrics
├── pipeline/
│   └── orchestrator.py      # Main entry point
├── prompts/
│   ├── prompt.py            # PromptFactory for DASP/Pedalboard init prompts
│   └── instruction.py       # Instruction templates
├── session/
│   └── session.py           # Session state management
├── training/
│   ├── trainer.py           # move_in_CLAP() — gradient descent in CLAP space
│   ├── loss.py              # Loss functions (directional, semantic similarity, guided)
│   └── parameterengine.py   # Parameter tensor ↔ dict conversion
├── utilities/
│   ├── audio_processing.py
│   ├── fx_processing.py     # fx_initial_params_to_tensor, fx_tensor_to_params_dict
│   └── text_processing.py   # JSON extraction from LLM output
└── visualization/
    └── plotting.py
dry_audio/                   # Test audio files (piano, violin, oboe)
```

## Running Tests

```bash
python tests/test_pipeline.py --quick   # fast, skip optimization
python tests/test_pipeline.py           # full suite with optimization
```
