"""
eval/apply_eq/get_gt.py — Apply Audealize FX to dry audio via apply_eq.js.

Calls the Node.js subprocess `apply_eq.js` with temp WAV files.

CLI signature (discovered in Phase 0):
    node apply_eq.js eq <input.wav> <output.wav> <params.json> [--range 1.0]

Usage:
    from eval.apply_eq.get_gt import get_gt
    wet = get_gt("eq", params_list_40, dry_audio_array)
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import List

import numpy as np
import soundfile as sf

# Default path — can be overridden per-call
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # eval/apply_eq/.. → eval/.. → root
_DEFAULT_APPLY_EQ_JS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "apply_eq.js"
)


def get_gt(
    fx_type: str,
    params: List[float],
    dry_audio: np.ndarray,
    sr: int = 44100,
    apply_eq_js_path: str = _DEFAULT_APPLY_EQ_JS,
) -> np.ndarray:
    """
    Apply Audealize FX to dry audio by calling apply_eq.js via subprocess.
    Writes temp WAV files, calls Node.js, reads back output.

    Input:
        fx_type: str            "eq" (reverb not yet supported in apply_eq.js)
        params: List[float]     length 40, raw Audealize EQ curve
        dry_audio: np.ndarray   shape (T,) or (1, T) or (T, C), float32, any SR
        sr: int                 sample rate (44100)
        apply_eq_js_path: str   path to apply_eq.js

    Output:
        processed: np.ndarray   shape (T,), float32  — wet audio

    Raises:
        RuntimeError  if apply_eq.js exits non-zero
        ValueError    if fx_type is not "eq"
    """
    if fx_type != "eq":
        raise ValueError(f"Only fx_type='eq' is supported; got '{fx_type}'")

    # Normalise dry audio to mono float32 1-D
    audio = np.array(dry_audio, dtype=np.float32)
    if audio.ndim == 2:
        # (C, T) or (T, C) — detect by shorter axis
        if audio.shape[0] <= audio.shape[1]:
            audio = audio[0]      # take first channel from (C, T)
        else:
            audio = audio[:, 0]   # take first channel from (T, C)
    if audio.ndim != 1:
        raise ValueError(f"Cannot interpret dry_audio with shape {audio.shape}")

    with tempfile.TemporaryDirectory() as tmp:
        in_wav = os.path.join(tmp, "dry.wav")
        out_wav = os.path.join(tmp, "wet.wav")
        params_json = os.path.join(tmp, "params.json")

        # Write input WAV
        sf.write(in_wav, audio, sr, subtype="FLOAT")

        # Write params JSON
        with open(params_json, "w") as f:
            json.dump([float(v) for v in params], f)

        # Call apply_eq.js
        result = subprocess.run(
            ["node", apply_eq_js_path, fx_type, in_wav, out_wav, params_json],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"apply_eq.js exited {result.returncode}:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

        if not os.path.exists(out_wav):
            raise FileNotFoundError(
                f"apply_eq.js exited 0 but output WAV not found at {out_wav}"
            )

        # Read back output — soundfile returns (T,) for mono or (T, C) for stereo
        wet, _ = sf.read(out_wav, dtype="float32", always_2d=False)
        if wet.ndim == 2:
            wet = wet[:, 0]  # take first channel

    return wet
