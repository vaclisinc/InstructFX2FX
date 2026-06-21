# InstructFX2FX

[![InstructFX2FX demo preview](https://raw.githubusercontent.com/vaclisinc/InstructFX2FX/32e0b65052e5f163f245eeca8f989803416f3ce0/presentation/demo/assets/cover.jpg)](https://instructfx2fx.vaclis.net/)

**A Multi-turn Text-to-Preset Demo for Iterative Audio Effect Refinement** — DAFx26, Demo Track.

[Live demo](https://instructfx2fx.vaclis.net/) · [Paper](presentation/demo/assets/paper.pdf) · UC Berkeley, CNMAT

Real audio mixing is iterative: a sequence of small corrections, not one descriptor. We study that multi-turn setting, which we call **sequential FX refinement**:

> Given an existing FX parameter set **P** and a sequence of natural-language instructions **{I₁, I₂, …}**, update the effect parameters so the rendered audio tracks the user's evolving intent while preserving relevant structure from the previous state.

InstructFX2FX is a hybrid: an LLM plans and orders the FX chain and proposes initial parameters; CLAP-guided optimization then refines them perceptually, turn after turn — gradient descent for differentiable effects (EQ, reverb) and Bayesian optimization for the rest.

## Architecture

3-layer pipeline orchestrated by `src/pipeline/orchestrator.py`:

```
Instruction ("make it brighter")  +  session state (P, history)
        │
        ▼
┌──────────────────────────────┐
│  Layer 1: FX Selector Agent  │  GPT-4o via OpenRouter tool-calling
│  src/agents/fx_selector.py   │  → ordered FX chain ["eq", "rev"]
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Layer 2: Parser / Router    │  routes the update (3 modes, below)
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

**Layer 2 — Parser / Router**: Each instruction takes one of three modes, based on session state:

| Mode | Session state | Action |
|------|---------------|--------|
| **Initialize** | no FX exist yet | LLM-initialize a fresh chain (no optimization this turn) |
| **Extend** | some FX exist | LLM-initialize the new effects, merge with existing, then optimize all |
| **Reuse & optimize** | all FX already exist | re-optimize from the existing params, in place |

**Layer 3 — Dual-track optimization** (sequential):
- 3A: DASP effects (`eq`, `rev`) → CLAP gradient descent via `move_in_CLAP()`
- 3B: Pedalboard effects (`comp`, `dist`, `delay`, `pitchshift`, `bitcrush`) → Bayesian BO via `fxsearcher()`
- DASP output audio feeds as input to the Pedalboard stage

**Session** (`src/session/session.py`): Accumulates params across instructions for multi-turn interaction. This persistent state is what makes refinement iterative.

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

## Results

CLAP-guided refinement lowers target-directed MMD on **9 of 10** directed descriptor pairs versus an LLM-only re-prompting baseline (0.45 → 0.34, a 24% reduction). Listen to the gradient-descent sessions, scrub the optimization trajectories, and A/B dry vs. result on the [live demo](https://instructfx2fx.vaclis.net/).

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

# First instruction
out = orch.run("bright", session, dry_audio_tensor)
# Second instruction — session remembers previous FX
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

## Demo

**Live site** ([instructfx2fx.vaclis.net](https://instructfx2fx.vaclis.net/)) — a static page with the waveform players, dry/result A/B, and gradient-descent trajectory scrubbing. Source lives in `presentation/demo/` (plain HTML/CSS/JS, no build step):

```bash
cd presentation/demo
python -m http.server          # then open http://localhost:8000
```

**Interactive app** — separate React frontend + FastAPI backend:

```bash
# frontend
cd apps/web_frontend && npm run dev

# backend
source .venv/bin/activate
uvicorn apps.web_api.app.main:app --reload
```

## Cite

```bibtex
@inproceedings{instructfx2fx2026,
  title     = {InstructFX2FX: A Multi-turn Text-to-Preset Demo for
               Iterative Audio Effect Refinement},
  author    = {Yu, Song-Ze and Liessens Dujardin, Milan and
               Cai, Yuxuan and Zhang, Wantong},
  booktitle = {Proc. 29th Int. Conf. Digital Audio Effects (DAFx)},
  year      = {2026}
}
```
