"""
eval/gt_bank.py — Pre-render and cache GT audio to disk.

Two functions:
    build_word_gt_bank       — renders every param × dry clip for one word
    build_sequential_gt_bank — renders sequential (params_A then params_B) GT for Exp 1

Both are idempotent: files that already exist are skipped.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import soundfile as sf

from eval.apply_eq.get_gt import get_gt
from eval.config import APPLY_EQ_JS_PATH, SAMPLE_RATE


def build_word_gt_bank(
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
    os.makedirs(out_dir, exist_ok=True)

    paths: List[str] = []
    for dry_idx, dry_path in enumerate(dry_paths):
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


def build_sequential_gt_bank(
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

    paths: List[str] = []
    for dry_idx, dry_path in enumerate(dry_paths):
        dry_audio, file_sr = sf.read(dry_path, dtype="float32", always_2d=False)
        use_sr = file_sr  # honour the file's native sample rate

        for i, pA in enumerate(params_A):
            audio_A = get_gt("eq", pA, dry_audio, sr=use_sr, apply_eq_js_path=APPLY_EQ_JS_PATH)

            for j, pB in enumerate(params_B):
                out_path = os.path.join(out_dir, f"{dry_idx}_{i}_{j}.wav")
                paths.append(out_path)

                if os.path.exists(out_path):
                    continue  # idempotent — skip

                audio_AB = get_gt("eq", pB, audio_A, sr=use_sr, apply_eq_js_path=APPLY_EQ_JS_PATH)
                sf.write(out_path, audio_AB, use_sr, subtype="FLOAT")

    return paths
