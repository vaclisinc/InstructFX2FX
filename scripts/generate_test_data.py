"""
Generate a few ground-truth and predicted audio pairs for testing LLM2Fx metrics.
Creates data/source, data/gt, data/pred and JSON metadata.

Run from repo root:
  python scripts/generate_test_data.py
Then run MMD evaluation:
  python -m src.run_mmd_demo --gt_dir data/gt --pred_dir data/pred --sr 22050
"""

import json
import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLE_RATE = 44100
DURATION = 3  # seconds, keep short for quick tests
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
N_FILES = 3  # minimal set for testing


def generate_signal(name: str, t: np.ndarray) -> np.ndarray:
    if "guitar" in name:
        return 0.5 * np.sin(2 * np.pi * 110 * t) + 0.3 * np.sin(2 * np.pi * 220 * t)
    if "vocal" in name:
        return 0.4 * np.sin(2 * np.pi * 250 * t) * (1 + 0.2 * np.sin(2 * np.pi * 5 * t))
    # default: simple tone
    return 0.5 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 0.5)


def apply_fx(audio_path: str, output_path: str, fx_params: dict) -> None:
    audio, sr = sf.read(audio_path)
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    try:
        from pedalboard import (
            Pedalboard, Reverb, Compressor, Gain,
            LowShelfFilter, HighShelfFilter,
        )
        effects = []
        if "eq_low_gain_db" in fx_params:
            effects.append(LowShelfFilter(
                cutoff_frequency_hz=fx_params.get("eq_low_freq", 200),
                gain_db=fx_params["eq_low_gain_db"],
            ))
        if "eq_high_gain_db" in fx_params:
            effects.append(HighShelfFilter(
                cutoff_frequency_hz=fx_params.get("eq_high_freq", 4000),
                gain_db=fx_params["eq_high_gain_db"],
            ))
        if "compressor_threshold" in fx_params:
            effects.append(Compressor(
                threshold_db=fx_params.get("compressor_threshold", -20),
                ratio=fx_params.get("compressor_ratio", 4.0),
            ))
        if "reverb_room_size" in fx_params:
            effects.append(Reverb(
                room_size=fx_params.get("reverb_room_size", 0.5),
                damping=fx_params.get("reverb_damping", 0.5),
                wet_level=fx_params.get("reverb_wet", 0.3),
            ))
        if "gain_db" in fx_params:
            effects.append(Gain(gain_db=fx_params["gain_db"]))
        board = Pedalboard(effects)
        processed = board(audio.T, sr).T
        processed = np.clip(processed, -1.0, 1.0)
    except ImportError:
        # No pedalboard: simulate with gain only (so metrics can still run)
        gain_db = fx_params.get("gain_db", 0.0)
        gain_lin = 10.0 ** (gain_db / 20.0)
        processed = np.clip(audio * gain_lin, -1.0, 1.0)
    sf.write(output_path, processed, sr)


def add_noise(params: dict, scale: float = 0.1) -> dict:
    out = {}
    for k, v in params.items():
        if "room_size" in k or "damping" in k or "wet" in k:
            out[k] = float(np.clip(v + np.random.uniform(-scale, scale), 0, 1))
        elif "ratio" in k:
            out[k] = float(max(v + np.random.uniform(-0.5, 0.5), 1.0))
        else:
            out[k] = v + np.random.uniform(-scale * abs(v), scale * abs(v)) if v != 0 else v
    return out


def main():
    (OUTPUT_DIR / "source").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "gt").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "pred").mkdir(parents=True, exist_ok=True)

    names = ["guitar_clean.wav", "vocal_dry.wav", "piano_close.wav"]
    t = np.linspace(0, DURATION, SAMPLE_RATE * DURATION, dtype=np.float32)

    gt_config = {
        "guitar_clean.wav": {
            "prompt": "warm and full",
            "params": {
                "eq_low_gain_db": 4.0, "eq_low_freq": 250,
                "eq_high_gain_db": -3.0, "eq_high_freq": 5000,
                "reverb_room_size": 0.4, "reverb_damping": 0.6,
                "reverb_wet": 0.25, "gain_db": 1.0,
            },
        },
        "vocal_dry.wav": {
            "prompt": "bright and present",
            "params": {
                "eq_low_gain_db": -2.0, "eq_low_freq": 150,
                "eq_high_gain_db": 3.0, "eq_high_freq": 8000,
                "compressor_threshold": -18, "compressor_ratio": 3.0,
                "reverb_room_size": 0.3, "reverb_damping": 0.7,
                "reverb_wet": 0.15, "gain_db": 2.0,
            },
        },
        "piano_close.wav": {
            "prompt": "spacious and lush",
            "params": {
                "eq_low_gain_db": 1.0, "eq_low_freq": 200,
                "eq_high_gain_db": 1.5, "eq_high_freq": 6000,
                "reverb_room_size": 0.7, "reverb_damping": 0.4,
                "reverb_wet": 0.45, "gain_db": 0.5,
            },
        },
    }

    np.random.seed(42)
    prompts, gt_params_out, pred_params_out = {}, {}, {}

    for fname in names[:N_FILES]:
        # source
        audio = generate_signal(fname, t).astype(np.float32)
        audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
        src_path = OUTPUT_DIR / "source" / fname
        sf.write(str(src_path), audio, SAMPLE_RATE)

        cfg = gt_config[fname]
        prompts[fname] = cfg["prompt"]
        gt_params_out[fname] = cfg["params"]
        pred_params_out[fname] = add_noise(cfg["params"])

        apply_fx(str(src_path), str(OUTPUT_DIR / "gt" / fname), cfg["params"])
        apply_fx(str(src_path), str(OUTPUT_DIR / "pred" / fname), pred_params_out[fname])
        print(f"  {fname}: '{cfg['prompt']}'")

    for name, data in [
        ("prompts.json", prompts),
        ("gt_params.json", gt_params_out),
        ("pred_params.json", pred_params_out),
    ]:
        with open(OUTPUT_DIR / name, "w") as f:
            json.dump(data, f, indent=2, default=float)

    print(f"\nCreated {N_FILES} pairs in {OUTPUT_DIR}")
    print("Run MMD: python -m src.run_mmd_demo --gt_dir data/gt --pred_dir data/pred --sr 22050")


if __name__ == "__main__":
    main()
