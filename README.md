<div align="center">

# InstructFX2FX

### A Multi-turn Text-to-Preset Demo for Iterative Audio Effect Refinement

<span style="white-space:nowrap;">Song-Ze Yu</span>&nbsp;·
<span style="white-space:nowrap;">Milan Liessens Dujardin</span>&nbsp;·
<span style="white-space:nowrap;">Yuxuan Cai</span>&nbsp;·
<span style="white-space:nowrap;">Wantong Zhang</span>

<sub>Center for New Music and Audio Technologies (CNMAT) · University of California, Berkeley</sub>

<br/>

[![arXiv](https://img.shields.io/badge/arXiv-2606.22005-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.22005v2)
[![DAFx 2026](https://img.shields.io/badge/DAFx_2026-Demo_Track-success.svg)](https://dafx26.mit.edu/)
[![Live demo](https://img.shields.io/badge/Live_demo-instructfx2fx.vaclis.net-0071e3.svg)](https://instructfx2fx.vaclis.net/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

</div>

<p align="center">
  <a href="https://instructfx2fx.vaclis.net/">
    <img src="https://raw.githubusercontent.com/vaclisinc/InstructFX2FX/32e0b65052e5f163f245eeca8f989803416f3ce0/presentation/demo/assets/cover.jpg" alt="InstructFX2FX demo preview" width="100%">
  </a>
</p>

---

> **TL;DR.** Real audio mixing is iterative — a sequence of small corrections, not one descriptor. **InstructFX2FX** treats this as *sequential FX refinement*: an LLM plans and orders the FX chain and proposes the initial parameter state, then CLAP-guided optimization refines it perceptually, turn after turn — gradient descent for differentiable effects (EQ, reverb) and Bayesian optimization for the rest. On SocialFX descriptor transitions it lowers target-directed MMD on **9 of 10** directed pairs versus an LLM-only re-prompting baseline.

## Highlights

| | |
|---|---|
| **Problem** | Sequential, multi-turn FX refinement — update an existing chain and parameter state, not regenerate a preset from scratch |
| **Method** | LLM planner (select + order + initialize) → CLAP-guided optimization (perceptual refinement) over a persistent session state |
| **Backends** | Gradient descent (differentiable: EQ, reverb) · Bayesian optimization (Pedalboard: comp, dist, delay, pitch, crush) |
| **Result** | Target-directed MMD ↓ on **9 / 10** directed pairs vs. LLM-only re-prompting (0.45 → 0.34, a 24% reduction) |
| **Interaction** | Human-in-the-loop: audition the saved optimization checkpoints each turn, then prompt the next correction |

---

## Sequential FX Refinement

We study the multi-turn setting we call **sequential FX refinement**:

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

Create a `.env` file in the project root:

```
OPENROUTER_API_KEY=your_api_key
```

## Quick Start

Run the multi-turn pipeline end-to-end on the bundled audio (`dry_audio/piano/piano.wav`). Set `OPENROUTER_API_KEY` in `.env` first; CLAP weights auto-download on first run:

```bash
python tests/test_pipeline.py
```

It drives a few multi-turn instruction sequences through the orchestrator and writes the rendered audio and parameters for each turn to `tests/outputs/`.

The core API is one call per instruction, with `Session` carrying state across turns:

```python
orch = Orchestrator(llm, clap, device="cpu")
session = Session(available_fx=["eq", "comp", "rev", "dist"])

orch.run("bright", session, audio)        # turn 1
orch.run("add warmth", session, audio)    # turn 2 — Session remembers turn 1
```

See [`tests/test_pipeline.py`](tests/test_pipeline.py) for the full runnable example (audio loading and model setup).

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

## Citation

```bibtex
@misc{yu2026instructfx2fxmultiturntexttopresetdemo,
      title={InstructFX2FX: A Multi-turn Text-to-Preset Demo for Iterative Audio Effect Refinement},
      author={Song-Ze Yu and Milan Liessens Dujardin and Yuxuan Cai and Wantong Zhang},
      year={2026},
      eprint={2606.22005},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2606.22005},
}
```

## Acknowledgements

InstructFX2FX builds on [LAION-CLAP](https://github.com/LAION-AI/CLAP) for audio–text embeddings, [`dasp`](https://github.com/csteinmetz1/dasp-pytorch) for differentiable audio effects, and [Pedalboard](https://github.com/spotify/pedalboard) for non-differentiable plugins. Our CLAP-guided objectives follow prior text-guided FX work — [Text2FX](https://arxiv.org/abs/2409.18847) and [FxSearcher](https://arxiv.org/abs/2511.14138) — and the LLM-initialization idea follows [LLM2Fx](https://arxiv.org/abs/2505.20770). Descriptor data is drawn from [SocialFX](https://dl.acm.org/doi/10.1145/2964284.2967207).
