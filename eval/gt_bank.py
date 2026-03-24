"""
eval/gt_bank.py — Pre-render and cache GT audio to disk.

Two functions:
    build_word_gt_bank       — renders every param × dry clip for one word
    build_sequential_gt_bank — renders sequential (params_A then params_B) GT for Exp 1

Both are idempotent: files that already exist are skipped.
"""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import List

import numpy as np
import soundfile as sf

from eval.apply_eq.get_gt import get_gt
from eval.config import APPLY_EQ_JS_PATH, GT_BANK_SAMPLES, SAMPLE_RATE
from eval.data.socialfx_loader import load_all_params_for_word
from eval.helpers import _dry_paths

import eval.config as cfg


def build_word_gt_bank_per_instance(
    word: str,
    instrument: str,
    dry_paths: List[str],
    params_list: List[List[float]],
    cache_dir: str,
    sr: int = SAMPLE_RATE,
) -> List[str]:
    """
    Render GT audio for a single word: every param set × every dry clip.
    Used for Exp 2 reference distributions (word_X_dist).

    Output paths: cache_dir/{word}/{instrument}/{dry_idx}_{param_idx}.wav
    Skips files that already exist (idempotent).

    Returns list of all cached WAV paths (len = len(params_list) * len(dry_paths)).
    """
    out_dir = os.path.join(cache_dir, word, instrument)

    # if os.path.exists(out_dir):
    #     print(f"Output directory already exists. SKIP.")
    #     return []
    os.makedirs(out_dir, exist_ok=True)

    paths: List[str] = []
    for dry_idx, dry_path in enumerate(dry_paths):
        if len(params_list) == len(os.listdir(out_dir)):
            print(f"All param sets already rendered for dry clip {dry_idx}. SKIP.")
            continue

        dry_audio, file_sr = sf.read(dry_path, dtype="float32", always_2d=False)
        use_sr = file_sr  # honour the file's native sample rate

        for param_idx, params in enumerate(params_list):
            out_path = os.path.join(out_dir, f"{dry_idx}_{param_idx}.wav")
            paths.append(out_path)

            if os.path.exists(out_path):
                continue  # idempotent — skip

            wet = get_gt("eq", params, dry_audio, sr=use_sr, apply_eq_js_path=APPLY_EQ_JS_PATH)
            sf.write(out_path, wet, use_sr, subtype="FLOAT")

    return paths


def build_sequential_gt_bank_per_instance(
    word_A: str,
    word_B: str,
    instrument: str,
    dry_paths: List[str],
    params_A: List[List[float]],
    params_B: List[List[float]],
    cache_dir: str,
    sr: int = SAMPLE_RATE,
) -> List[str]:
    """
    Render sequential GT for Exp 1:
        audio_A  = get_gt("eq", params_A[i], dry)
        audio_AB = get_gt("eq", params_B[j], audio_A)   ← sequential

    Output paths: cache_dir/{word_A}_to_{word_B}/{instrument}/{dry_idx}_{i}_{j}.wav
    Skips files that already exist (idempotent).

    Returns list of all cached WAV paths
    (len = len(dry_paths) * len(params_A) * len(params_B)).
    """
    pair_name = f"{word_A}_to_{word_B}"
    out_dir = os.path.join(cache_dir, pair_name, instrument)
    os.makedirs(out_dir, exist_ok=True)

    # If the directory already contains WAV files, return them as-is.
    # Avoids re-sampling a different random subset on subsequent calls.
    existing = sorted(
        os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.endswith(".wav")
    )
    if existing:
        return existing

    all_pairs = [(i, j) for i in range(len(params_A)) for j in range(len(params_B))]
    sampled_pairs = random.sample(all_pairs, min(GT_BANK_SAMPLES, len(all_pairs)))

    paths: List[str] = []
    for dry_idx, dry_path in enumerate(dry_paths):
        dry_audio, file_sr = sf.read(dry_path, dtype="float32", always_2d=False)
        use_sr = file_sr  # honour the file's native sample rate

        # group by i to avoid recomputing audio_A for the same params_A
        pairs_by_i: dict = defaultdict(list)
        for i, j in sampled_pairs:
            pairs_by_i[i].append(j)

        for i, js in pairs_by_i.items():
            audio_A = get_gt("eq", params_A[i], dry_audio, sr=use_sr, apply_eq_js_path=APPLY_EQ_JS_PATH)
            for j in js:
                out_path = os.path.join(out_dir, f"{dry_idx}_{i}_{j}.wav")
                paths.append(out_path)

                if os.path.exists(out_path):
                    continue  # idempotent — skip

                audio_AB = get_gt("eq", params_B[j], audio_A, sr=use_sr, apply_eq_js_path=APPLY_EQ_JS_PATH)
                sf.write(out_path, audio_AB, use_sr, subtype="FLOAT")

    return paths

def build_word_gt_bank_full(words, instruments):
    for word in words:
        params = load_all_params_for_word(word)
        for instrument in instruments:
            dry = _dry_paths(instrument)
            if not dry:
                print(f"[exp2] SKIP word GT {word}/{instrument}: no dry paths")
                continue
            paths = build_word_gt_bank_per_instance(
                word=word,
                instrument=instrument,
                dry_paths=dry,
                params_list=params,
                cache_dir=cfg.GT_BANK_DIR_ONESHOT,
                sr=cfg.SAMPLE_RATE,
            )
            print(f"[exp2] word GT {word}/{instrument}: {len(paths)} files cached")


def build_sequential_gt_bank_full(pairs, instruments):
    for word_A, word_B in pairs:
        pair_key = f"{word_A}_to_{word_B}"
        params_A = load_all_params_for_word(word_A)
        params_B = load_all_params_for_word(word_B)
        for instrument in cfg.INSTRUMENTS:
            dry_paths = _dry_paths(instrument)
            if not dry_paths:
                print(f"[exp1] SKIP GT {pair_key}/{instrument}: no dry paths")
                continue
            paths = build_sequential_gt_bank_per_instance(
                word_A=word_A, word_B=word_B,
                instrument=instrument,
                dry_paths=dry_paths,
                params_A=params_A,
                params_B=params_B,
                cache_dir=cfg.GT_BANK_DIR_SEQUENTIAL,
                sr=cfg.SAMPLE_RATE,
            )
            print(f"[exp1] GT {pair_key}/{instrument}: {len(paths)} files cached")
