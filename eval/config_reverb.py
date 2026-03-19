"""
eval/config_reverb.py — Constants for the InstructFX2FX reverb evaluation pipeline.

All paths are absolute and derived from the project root so this file is
importable from any working directory.
"""

import os
import torch
from typing import List, Tuple

# ── Project root (eval/ lives one level below) ─────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Word pairs ──────────────────────────────────────────────────────────────
# 7 final Reverb words: small, spacious, echo, warm, dark, hall, bright
#
# Selection rationale:
#   - small, spacious, hall: spatial size dimension (small room → large hall)
#   - echo: temporal dimension (long reflections / decay)
#   - warm, bright, dark: timbral color of the reverb (surface material / damping)
#   - These 7 words cover the main perceptual axes of reverb:
#     size (small↔spacious↔hall), decay (echo), and color (warm↔bright↔dark)
#
# A → B means "transform audio_A to sound more like B"
WORD_PAIRS: List[Tuple[str, str]] = [
    # Control group: near-opposite pairs (expected to partially cancel)
    ("small",    "spacious"),
    ("spacious", "small"),
    # Main experiment: pairs across different reverb dimensions
    ("dark",     "bright"),
    ("warm",     "bright"),
    ("hall",     "small"),
    ("echo",     "small"),
    ("dark",     "spacious"),
    ("warm",     "hall"),
    ("bright",   "echo"),
    ("small",    "dark"),
]

# ── Instruments & dry audio ─────────────────────────────────────────────────
INSTRUMENTS: List[str] = ["piano", "violin"]

DRY_AUDIO_DIR: str = os.path.join(ROOT, "dry_audio")

# ── Cache / output directories ──────────────────────────────────────────────
GT_BANK_DIR: str = os.path.join(ROOT, "eval", "gt_cache_reverb")
SYSTEM_RESULTS_DIR: str = os.path.join(ROOT, "eval", "system_outputs_reverb")
RESULTS_DIR: str = os.path.join(ROOT, "eval", "results_reverb")

# ── Node / apply_reverb.js (TODO: implement reverb processor) ──────────────
APPLY_REVERB_JS_PATH: str = os.path.join(ROOT, "eval", "apply_reverb", "apply_reverb.js")

# ── FX type ─────────────────────────────────────────────────────────────────
FX_EFFECTS: List[str] = ["reverb"]

# ── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 44100
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── GT bank sampling ────────────────────────────────────────────────────────
GT_BANK_SAMPLES: int = 50

# ── System hyperparams ──────────────────────────────────────────────────────
N_GRAD_ITER: int = 1000
SAVE_INTERVAL: int = 50
NR_RUNS_PER_FILE: int = 20

# ── Init mode ────────────────────────────────────────────────────────────────
FIXED_INIT: bool = False