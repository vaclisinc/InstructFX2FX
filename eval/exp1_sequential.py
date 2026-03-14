"""
eval/exp1_sequential.py — Phase 4: Experiment 1 (Sequential MMD).

Drives the full pipeline for one or more methods:
    1. Build sequential GT bank (Phase 2 logic, idempotent)
    2. Run system for each pair × instrument (Phase 3 logic, idempotent)
    3. Compute MMD(system final output, sequential GT) per instrument → aggregate

All config (pairs, instruments, paths, hyperparams) is read from eval/config.py.
Callers only supply model objects and method identifiers.

Functions:
    _compute_mmd_for_pair_instrument  — thin wrapper: extract features → cal_mmd_score
    run_exp1                          — end-to-end: build GT → run system → compute MMD
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.metrics import cal_mmd_score, extract_dsp_feature

import eval.config as cfg
from eval.data.socialfx_loader import load_all_params_for_word
from eval.gt_bank import build_sequential_gt_bank
from eval.run_system import collect_final_audio_paths, run_system_for_pair


# ── Core metric helper ───────────────────────────────────────────────────────

def _compute_mmd_for_pair_instrument(
    system_audio_paths: List[str],
    gt_audio_paths: List[str],
) -> float:
    """Extract 35-D DSP features and compute Gaussian-kernel MMD.

    Returns:
        MMD score (float ≥ 0).  Lower means distributions are closer.
    """
    X = np.array([extract_dsp_feature(p) for p in gt_audio_paths])
    Y = np.array([extract_dsp_feature(p) for p in system_audio_paths])
    return float(cal_mmd_score(X, Y))


# ── Path helpers ─────────────────────────────────────────────────────────────

def _dry_paths(instrument: str) -> List[str]:
    d = os.path.join(cfg.DRY_AUDIO_DIR, instrument)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav"))


def _seq_gt_paths(word_A: str, word_B: str, instrument: str) -> List[str]:
    pattern = os.path.join(cfg.GT_BANK_DIR, f"{word_A}_to_{word_B}", instrument, "*.wav")
    return sorted(glob.glob(pattern))


def _latest_experiment_dir(word_A: str, word_B: str, instrument: str):
    pair_inst_dir = os.path.join(cfg.SYSTEM_RESULTS_DIR, f"{word_A}_to_{word_B}", instrument)
    candidates = sorted(glob.glob(os.path.join(pair_inst_dir, "experiment_*")))
    return candidates[-1] if candidates else None


# ── Main experiment function ─────────────────────────────────────────────────

def run_exp1(
    methods: list,
    method_names: List[str],
    llm_client,
    clap,
) -> List[Dict]:
    """End-to-end Experiment 1: build GT → run system → compute sequential MMD.

    All pairs, instruments, paths, and hyperparams are taken from eval/config.py.

    Args:
        methods:       List of Method enums, e.g. [Method.InstructFX2FX, Method.LLM_LLM].
        method_names:  Matching human-readable labels, e.g. ["ours", "baseline"].
        llm_client:    LLMClient instance.
        clap:          CLAPWrapper instance.

    Returns:
        List of result dicts, one per method:
        {
            "method": str,
            "pairs": {
                "warm_to_bright": {"violin": float, ..., "avg": float},
                ...
            },
            "overall_avg": float
        }
    """
    assert len(methods) == len(method_names)

    # ── Step 1: Build sequential GT bank (idempotent) ─────────────────────────
    print("[exp1] === Step 1: Build sequential GT bank ===")
    for word_A, word_B in cfg.WORD_PAIRS:
        pair_key = f"{word_A}_to_{word_B}"
        params_A = load_all_params_for_word(word_A)
        params_B = load_all_params_for_word(word_B)
        for instrument in cfg.INSTRUMENTS:
            dry_paths = _dry_paths(instrument)
            if not dry_paths:
                print(f"[exp1] SKIP GT {pair_key}/{instrument}: no dry paths")
                continue
            paths = build_sequential_gt_bank(
                word_A=word_A, word_B=word_B,
                instrument=instrument,
                dry_paths=dry_paths,
                params_A=params_A,
                params_B=params_B,
                cache_dir=cfg.GT_BANK_DIR,
                sr=cfg.SAMPLE_RATE,
            )
            print(f"[exp1] GT {pair_key}/{instrument}: {len(paths)} files cached")

    # ── Step 2: Run system (idempotent — skip if experiment dir exists) ────────
    print("\n[exp1] === Step 2: Run system ===")
    for word_A, word_B in cfg.WORD_PAIRS:
        pair_key = f"{word_A}_to_{word_B}"
        for instrument in cfg.INSTRUMENTS:
            dry_paths = _dry_paths(instrument)
            if not dry_paths:
                print(f"[exp1] SKIP sys {pair_key}/{instrument}: no dry paths")
                continue
            if _latest_experiment_dir(word_A, word_B, instrument):
                print(f"[exp1] sys {pair_key}/{instrument}: already exists, skipping")
                continue
            print(f"[exp1] sys {pair_key}/{instrument}: running …")
            run_system_for_pair(
                word_A=word_A, word_B=word_B,
                instrument=instrument,
                dry_paths=dry_paths,
                methods=methods,
                llm_client=llm_client,
                clap=clap,
                output_dir=cfg.SYSTEM_RESULTS_DIR,
                n_iter=cfg.N_GRAD_ITER,
                save_interval=cfg.SAVE_INTERVAL,
                nr_runs=cfg.NR_RUNS_PER_FILE,
            )

    # ── Step 3: Compute MMD per method ────────────────────────────────────────
    print("\n[exp1] === Step 3: Compute MMD ===")
    all_results = []

    for method, method_name in zip(methods, method_names):
        all_mmds: List[float] = []
        pair_results: Dict[str, Dict] = {}

        for word_A, word_B in cfg.WORD_PAIRS:
            pair_key = f"{word_A}_to_{word_B}"
            pair_entry: Dict = {}
            per_inst_mmds: List[float] = []

            for instrument in cfg.INSTRUMENTS:
                gt_paths = _seq_gt_paths(word_A, word_B, instrument)
                if not gt_paths:
                    print(f"[exp1] SKIP MMD {pair_key}/{instrument}: no GT")
                    continue

                exp_dir = _latest_experiment_dir(word_A, word_B, instrument)
                if exp_dir is None:
                    print(f"[exp1] SKIP MMD {pair_key}/{instrument}: no experiment dir")
                    continue

                sys_paths = collect_final_audio_paths(exp_dir, method)
                if not sys_paths:
                    print(f"[exp1] SKIP MMD {pair_key}/{instrument}: no final WAVs for {method_name}")
                    continue

                mmd = _compute_mmd_for_pair_instrument(sys_paths, gt_paths)
                pair_entry[instrument] = mmd
                per_inst_mmds.append(mmd)
                all_mmds.append(mmd)
                print(f"[exp1] {method_name} | {pair_key}/{instrument}: MMD={mmd:.6f} "
                      f"(n_sys={len(sys_paths)}, n_gt={len(gt_paths)})")

            pair_entry["avg"] = float(np.mean(per_inst_mmds)) if per_inst_mmds else float("nan")
            pair_results[pair_key] = pair_entry

        overall_avg = float(np.mean(all_mmds)) if all_mmds else float("nan")
        results = {"method": method_name, "pairs": pair_results, "overall_avg": overall_avg}

        os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(cfg.RESULTS_DIR, f"exp1_{method_name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[exp1] {method_name}: overall_avg={overall_avg:.6f} → {out_path}")

        all_results.append(results)

    return all_results
