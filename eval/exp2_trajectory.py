"""
eval/exp2_trajectory.py — Phase 5: Experiment 2 (Trajectory MMD).

Measures how MMD evolves across optimization steps for InstructFX2FX (OURS),
relative to two single-word GT distributions:
    - word_A GT bank  (where the audio starts)
    - word_B GT bank  (where the audio should end up)

BASELINE (LLM+LLM) has no trajectory — only a single final MMD value is recorded.

No new metric logic — everything reuses:
    - build_word_gt_bank()               from eval/gt_bank.py
    - _compute_mmd_for_pair_instrument() from eval/exp1_sequential.py
    - collect_trajectory_audio_paths()   from eval/run_system.py
    - collect_final_audio_paths()        from eval/run_system.py

Functions:
    run_exp2  — end-to-end trajectory MMD
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import eval.config as cfg
from eval.data.socialfx_loader import load_all_params_for_word
from eval.exp1_sequential import _compute_mmd_for_pair_instrument, _dry_paths, _latest_experiment_dir
from eval.gt_bank import build_word_gt_bank
from eval.run_system import collect_final_audio_paths, collect_trajectory_audio_paths

try:
    from experimentation.experiment import Method
except ModuleNotFoundError:
    sys.path.insert(0, str(_ROOT / "src"))
    from experimentation.experiment import Method


# ── Helpers ──────────────────────────────────────────────────────────────────

def _word_gt_paths(word: str, instrument: str) -> List[str]:
    pattern = os.path.join(cfg.GT_BANK_DIR, word, instrument, "*.wav")
    return sorted(glob.glob(pattern))


def _iter_num(step: str) -> Optional[int]:
    """Parse numeric iteration index from a step label like 'iter_100_in_gradient_descent'.

    Returns None for 'start' and 'end'.
    """
    if step in ("start", "end"):
        return None
    m = re.search(r"\d+", step)
    return int(m.group()) if m else None



def _trajectory_mmd(
    trajectory: Dict[str, List[str]],
    word_A_paths: List[str],
    word_B_paths: List[str],
) -> Dict:
    """Compute MMD toward word_A and word_B at each trajectory step.

    Returns:
        {
            "steps":       [step_name, ...],
            "iter_nums":   [int_or_null, ...],
            "mmd_toward_A": [float, ...],
            "mmd_toward_B": [float, ...],
        }
    """
    steps, iter_nums, mmd_A, mmd_B = [], [], [], []
    for step, step_paths in trajectory.items():
        if not step_paths:
            continue
        steps.append(step)
        iter_nums.append(_iter_num(step))
        mmd_A.append(_compute_mmd_for_pair_instrument(step_paths, word_A_paths))
        mmd_B.append(_compute_mmd_for_pair_instrument(step_paths, word_B_paths))
    return {"steps": steps, "iter_nums": iter_nums, "mmd_toward_A": mmd_A, "mmd_toward_B": mmd_B}


# ── Main experiment function ──────────────────────────────────────────────────

def run_exp2() -> Dict:
    """End-to-end Experiment 2: build word GT banks → load trajectories → compute MMD.

    All pairs, instruments, paths, and hyperparams are taken from eval/config.py.
    System outputs must already exist on disk (produced by run_exp1 / run_system).

    Returns:
        {
            "pairs": {
                "warm_to_bright": {
                    "violin": {
                        "ours": {
                            "steps": [...],
                            "iter_nums": [...],
                            "mmd_toward_A": [...],
                            "mmd_toward_B": [...]
                        },
                        "baseline": {
                            "mmd_toward_A": float,
                            "mmd_toward_B": float
                        }
                    }
                },
                ...
            }
        }
    """
    # ── Step 1: Build single-word GT banks (idempotent) ───────────────────────
    print("[exp2] === Step 1: Build single-word GT banks ===")
    unique_words = sorted({w for pair in cfg.WORD_PAIRS for w in pair})
    for word in unique_words:
        params = load_all_params_for_word(word)
        for instrument in cfg.INSTRUMENTS:
            dry = _dry_paths(instrument)
            if not dry:
                print(f"[exp2] SKIP word GT {word}/{instrument}: no dry paths")
                continue
            paths = build_word_gt_bank(
                word=word,
                instrument=instrument,
                dry_paths=dry,
                params_list=params,
                cache_dir=cfg.GT_BANK_DIR,
                sr=cfg.SAMPLE_RATE,
            )
            print(f"[exp2] word GT {word}/{instrument}: {len(paths)} files cached")

    # ── Step 2: Compute trajectory MMD per pair × instrument ─────────────────
    print("\n[exp2] === Step 2: Compute trajectory MMD ===")
    pair_results: Dict = {}

    for word_A, word_B in cfg.WORD_PAIRS:
        pair_key = f"{word_A}_to_{word_B}"
        pair_entry: Dict = {}

        for instrument in cfg.INSTRUMENTS:
            word_A_paths = _word_gt_paths(word_A, instrument)
            word_B_paths = _word_gt_paths(word_B, instrument)
            if not word_A_paths or not word_B_paths:
                print(f"[exp2] SKIP {pair_key}/{instrument}: missing word GT")
                continue

            exp_dir = _latest_experiment_dir(word_A, word_B, instrument)
            if exp_dir is None:
                print(f"[exp2] SKIP {pair_key}/{instrument}: no experiment dir")
                continue

            inst_entry: Dict = {}

            # ── OURS: optimization_steps trajectory ─────────────────────────
            ours_traj = collect_trajectory_audio_paths(exp_dir)
            if ours_traj:
                ours_data = _trajectory_mmd(ours_traj, word_A_paths, word_B_paths)
                inst_entry["ours"] = ours_data
                print(f"[exp2] OURS  {pair_key}/{instrument}: {len(ours_data['steps'])} steps")
            else:
                print(f"[exp2] OURS  {pair_key}/{instrument}: no optimization_steps found")

            # ── BASELINE: single final MMD (no trajectory) ──────────────────
            baseline_finals = collect_final_audio_paths(exp_dir, Method.LLM_LLM)
            if baseline_finals:
                inst_entry["baseline"] = {
                    "mmd_toward_A": _compute_mmd_for_pair_instrument(baseline_finals, word_A_paths),
                    "mmd_toward_B": _compute_mmd_for_pair_instrument(baseline_finals, word_B_paths),
                }
                print(f"[exp2] BASE  {pair_key}/{instrument}: MMD computed ({len(baseline_finals)} files)")
            else:
                print(f"[exp2] BASE  {pair_key}/{instrument}: no final WAVs found")

            if inst_entry:
                pair_entry[instrument] = inst_entry

        if pair_entry:
            pair_results[pair_key] = pair_entry

    results = {"pairs": pair_results}

    # ── Step 3: Save results ──────────────────────────────────────────────────
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(cfg.RESULTS_DIR, "exp2_trajectory.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[exp2] saved → {out_path}")

    return results
