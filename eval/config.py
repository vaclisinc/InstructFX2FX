"""
eval/config.py — Constants for the InstructFX2FX evaluation pipeline.

All paths are absolute and derived from the project root so this file is
importable from any working directory.
"""

import os
import torch
from typing import List, Tuple

# ── Project root (eval/ lives one level below) ─────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Word pairs ──────────────────────────────────────────────────────────────
# 7 final EQ words: warm, bright, soft, loud, harsh, calm, heavy
# See eval/select_eq_pairs.md for selection rationale.
# A → B means "transform audio_A to sound more like B"
WORD_PAIRS: List[Tuple[str, str]] = [
    # Control group: near-opposite pairs (expected to partially cancel in sequential GT)
    ("warm",   "bright"),
    ("bright", "warm"),
    # Main experiment: pairs affecting different or non-canceling frequency regions
    ("heavy",  "calm"),
    ("heavy",  "harsh"),
    ("harsh",  "soft"),
    ("harsh",  "calm"),
    ("warm",   "heavy"),
    ("soft",   "loud"),
    ("calm",   "loud"),
    ("loud",   "heavy"),
]

# ── Instruments & dry audio ─────────────────────────────────────────────────
# Instrument names must match subdirectory names under DRY_AUDIO_DIR.
# Only violin currently has WAV files in dry_audio/.
INSTRUMENTS: List[str] = ["piano", "violin"]

DRY_AUDIO_DIR: str = os.path.join(ROOT, "dry_audio")
# Expected structure: DRY_AUDIO_DIR/{instrument}/*.wav

# ── Cache / output directories ──────────────────────────────────────────────
GT_BANK_DIR_SEQUENTIAL: str = os.path.join(ROOT, "eval", "gt_cache", "sequential")
GT_BANK_DIR_ONESHOT: str = os.path.join(ROOT, "eval", "gt_cache", "one-shot")
SYSTEM_RESULTS_DIR: str = os.path.join(ROOT, "eval", "system_outputs")
RESULTS_DIR: str = os.path.join(ROOT, "eval", "results")

# ── Node / apply_eq.js ──────────────────────────────────────────────────────
APPLY_EQ_JS_PATH: str = os.path.join(ROOT, "eval", "apply_eq", "apply_eq.js")

# ── FX type ─────────────────────────────────────────────────────────────────
FX_EFFECTS: List[str] = ["eq"]

# ── Audio ───────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 44100
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── GT bank sampling ────────────────────────────────────────────────────────
GT_BANK_SAMPLES: int = 50   # random (i, j) pairs sampled per dry clip

# ── System hyperparams ──────────────────────────────────────────────────────
N_GRAD_ITER: int = 1000
SAVE_INTERVAL: int = 50    # snapshot_interval for trajectory (Exp 2)
NR_RUNS_PER_FILE: int = 20  # nr_of_experiments_per_file passed to run_LLMLLM_vs_InstructFX2FX

# ── Init mode ────────────────────────────────────────────────────────────────
# True  (Option B): call LLM once per audio file before the runs loop;
#                   ALL runs (and both methods) share that single fixed init.
# False (Option A): each run calls LLM fresh for init (stochastic across runs);
#                   LLM_LLM captures the init and InstructFX2FX does not reuse it within
#                   the same run.
FIXED_INIT: bool = False
