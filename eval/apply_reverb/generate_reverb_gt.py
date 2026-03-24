"""
generate_reverb_gt.py — Generate reverb ground truth for a word
===============================================================

Loads SocialFX reverb data, loops through all parameter sets for a word,
applies each one to the input audio via apply_reverb.js, saves results.

Usage:
    python generate_reverb_gt.py

To change the word or paths, edit the config section below.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# >>>  CONFIG — edit these  <<<
# ============================================================================

INPUT_WAV = "/Users/yuxuancai/cnmat2026/text2preset-1/data/audio/piano.wav"
OUTPUT_DIR = "/Users/yuxuancai/cnmat2026/text2preset-1/src/metrics/apply_eq/warm_gt_reverb"
WORD = "warm"
FX_SPLIT = "reverb"
MIN_CONSISTENCY = 0.0   # 0.0 = keep all rows, raise to filter

# Path to apply_reverb.js (same directory as this script by default)
APPLY_REVERB_JS = Path(__file__).parent / "apply_reverb.js"


# ============================================================================
# Data loading
# ============================================================================

def load_reverb_data(min_consistency: float = 0.0) -> pd.DataFrame:
    """Load SocialFX reverb split from HuggingFace."""
    url = "hf://datasets/seungheondoh/socialfx-original/data/reverb-00000-of-00001.parquet"
    print("Loading SocialFX reverb data from HuggingFace...")
    df = pd.read_parquet(url)
    df["word"] = df["text"].str.lower().str.strip()
    def _consistency(x):
        try:
            d = ast.literal_eval(x) if isinstance(x, str) else x
            return float(d.get("ratings_consistency", 0.0))
        except Exception:
            return 0.0

    df["consistency"] = df["extra"].apply(_consistency)
    if min_consistency > 0:
        df = df[df["consistency"] >= min_consistency]
    print(f"  {len(df)} rows, {df['word'].nunique()} unique words")
    return df


# ============================================================================
# Core: apply one reverb parameter set via Node.js
# ============================================================================

def apply_reverb(
    input_wav: str,
    output_wav: str,
    param_values: list | np.ndarray,
) -> str:
    """
    Apply one 40-param SocialFX reverb to audio.

    Args:
        input_wav:    Path to input wav file
        output_wav:   Path to write processed wav
        param_values: 40-element list/array (one row from SocialFX reverb)

    Returns:
        output_wav path
    """
    params = list(np.asarray(param_values, dtype=float))
    assert len(params) >= 5, f"Expected at least 5 reverb params [delay_time, decay, stereo_spread, cutoff_freq, wet_gain], got {len(params)}"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(params, f)
        params_path = f.name

    try:
        subprocess.run(
            ["node", str(APPLY_REVERB_JS), input_wav, output_wav, params_path],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(APPLY_REVERB_JS.parent),
        )
        return output_wav
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e.stderr}")
        raise
    finally:
        os.unlink(params_path)


# ============================================================================
# Main: loop through all param sets for a word
# ============================================================================

def main():
    df = load_reverb_data(MIN_CONSISTENCY)

    rows = df[df["word"] == WORD]
    if len(rows) == 0:
        available = sorted(df["word"].unique().tolist())
        print(f"Word '{WORD}' not found. Available: {available[:30]}...")
        return

    print(f"\nWord: '{WORD}'")
    print(f"  Parameter sets: {len(rows)}")
    print(f"  IDs: {rows['id'].tolist()[:10]}...")
    print(f"  Input: {INPUT_WAV}")
    print(f"  Output dir: {OUTPUT_DIR}")

    if not os.path.exists(INPUT_WAV):
        print(f"\n  ERROR: Input file not found: {INPUT_WAV}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nProcessing {len(rows)} parameter sets...\n")
    for i, (_, row) in enumerate(rows.iterrows()):
        row_id = row["id"]
        params = row["param_values"]

        output_path = os.path.join(OUTPUT_DIR, f"{row_id}.wav")
        print(f"  [{i+1}/{len(rows)}] {row_id}", end="", flush=True)

        apply_reverb(INPUT_WAV, output_path, params)
        print(f" -> {row_id}.wav")

    print(f"\nDone! Saved {len(rows)} files to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
