# test how CLAP deals best with composite instructions


from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm

from src.configurations.config import OptimizationMethod

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import eval.config as cfg
from eval.data.socialfx_loader import load_all_params_for_word
from eval.exp.exp1_sequential import _dry_paths
from eval.gt_bank import build_word_gt_bank

from src.metrics.dsp_feature_metrics import (
    TIMBRE_FEATURE_NAMES,
    extract_timbre_features,
)

# ── Labels ─────────────────────────────────────────────────────────────

SOURCE_GT = "SocialFX GT"
SOURCE_LLM = "LLM Init"
SOURCE_CLAP = "CLAP"

# ── Output directories (under eval/) ─────────────────────────────────────────

_EXP4_SYSTEM_RESULTS_DIR = os.path.join(cfg.SYSTEM_RESULTS_DIR, "exp4")
_EXP4_RESULTS_DIR = os.path.join(cfg.RESULTS_DIR, "exp4")

_EXP4_CLAP_DIR = os.path.join(_EXP4_SYSTEM_RESULTS_DIR, "clap")


def run_exp4(
    llm_client,
    clap,
    fx_type: str = "eq",
    device: str = cfg.DEVICE,
    n_clap_calls: int = 100,
    effects: List[str] = None,
    words: Optional[List[str]] = None,
) -> Dict:
    pass