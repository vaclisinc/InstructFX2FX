"""
Embedding Sensitivity Test
==========================
Question: How sensitive are audio embeddings to EQ changes?

Three models compared:
  - CLAP:         text-audio cosine similarity (higher = more like the word)
  - FX-encoder++: audio-audio cosine distance from DRY (higher = more EQ change detected)
  - MusicCoCa:    text-audio cosine similarity (optional, requires magenta-realtime)

Usage:
    .venv/bin/python scripts/sensitivity_test.py [--model clap|fxenc|musiccoca|all]

Install MusicCoCa (optional):
    .venv/bin/pip install magenta-realtime
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pedalboard import HighShelfFilter, LowShelfFilter, PeakFilter, Pedalboard

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # needed for `from embeddings.clap import ...`

DRY_AUDIO = ROOT / "dry_audio" / "violin" / "violin 0.wav"

WORDS = ["bright", "warm", "harsh", "soft", "heavy", "calm", "loud"]

EQ_PRESETS = {
    "bright":  [HighShelfFilter(cutoff_frequency_hz=4000,  gain_db=+6.0)],
    "warm":    [LowShelfFilter(cutoff_frequency_hz=300,    gain_db=+4.0),
                HighShelfFilter(cutoff_frequency_hz=4000,  gain_db=-3.0)],
    "harsh":   [PeakFilter(cutoff_frequency_hz=3000, gain_db=+8.0, q=2.0)],
    "soft":    [HighShelfFilter(cutoff_frequency_hz=4000,  gain_db=-6.0),
                PeakFilter(cutoff_frequency_hz=3000, gain_db=-4.0, q=1.5)],
    "heavy":   [LowShelfFilter(cutoff_frequency_hz=200,    gain_db=+6.0)],
    "calm":    [HighShelfFilter(cutoff_frequency_hz=6000,  gain_db=-4.0),
                LowShelfFilter(cutoff_frequency_hz=200,    gain_db=-2.0)],
    "loud":    [PeakFilter(cutoff_frequency_hz=1000, gain_db=+5.0, q=0.8)],
}


# ── helpers ──────────────────────────────────────────────────────────────────

def apply_eq(audio_mono: np.ndarray, sr: int, plugins: list) -> np.ndarray:
    board = Pedalboard(plugins)
    return board(audio_mono[np.newaxis, :], sr)[0]


def load_dry() -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(DRY_AUDIO))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr


# ── CLAP scorer (text-audio similarity) ──────────────────────────────────────

def make_clap_scorer():
    from src.metrics.clap_metrics import compute_clap_score_from_array

    def score(audio: np.ndarray, sr: int, word: str) -> float:
        return compute_clap_score_from_array(audio, sr, word)

    return score


# ── FX-encoder++ scorer (audio-audio distance from DRY) ──────────────────────

def make_fxenc_scorer():
    from fxencoder_plusplus import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("default", device=device)
    model.eval()

    def get_emb(audio: np.ndarray, sr: int) -> torch.Tensor:
        import librosa
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        stereo = np.stack([audio, audio], axis=0)           # [2, T]
        wav = torch.from_numpy(stereo).float().unsqueeze(0).to(device)  # [1, 2, T]
        with torch.no_grad():
            emb = model.get_fx_embedding(wav)               # [1, D], L2-normalized
        return emb.squeeze(0)                               # [D]

    # FX-encoder++ has no text side → measure cosine DISTANCE from dry embedding
    # (higher distance = embedding changed more = more sensitive to EQ)
    dry_emb: list[torch.Tensor] = []   # mutable cell so inner func can write it

    def score(audio: np.ndarray, sr: int, word: str) -> float:
        """Returns cosine distance from DRY embedding (0=identical, 2=opposite)."""
        emb = get_emb(audio, sr)
        if not dry_emb:
            # first call is always DRY → distance = 0 by definition
            dry_emb.append(emb)
            return 0.0
        cos_sim = F.cosine_similarity(emb.unsqueeze(0), dry_emb[0].unsqueeze(0)).item()
        return 1.0 - cos_sim   # distance: 0=same, larger=more different

    return score, get_emb, dry_emb


# ── MusicCoCa scorer (text-audio similarity) ─────────────────────────────────

def make_musiccoca_scorer():
    try:
        from magenta_rt.musiccoca import MusicCoCaV212F
        from magenta_rt.audio import Waveform
    except ImportError:
        raise ImportError(
            "MusicCoCa not installed. Run: pip install -e /tmp/magenta-realtime"
        )

    print("  Loading MusicCoCa (downloads weights on first run)...", flush=True)
    model = MusicCoCaV212F()

    def score(audio: np.ndarray, sr: int, word: str) -> float:
        import librosa
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        # Waveform expects [T, channels]
        stereo = np.stack([audio, audio], axis=1)   # [T, 2]
        wav = Waveform(stereo.astype(np.float32), 16000)

        a_emb = model.embed(wav)    # [768]
        t_emb = model.embed(word)   # [768]

        cos = np.dot(a_emb, t_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(t_emb) + 1e-8)
        return float(cos)

    return score


# ── printing ──────────────────────────────────────────────────────────────────

def print_clap_table(model_name: str, results: dict, words: list):
    """Table for text-audio models: rows=EQ preset, cols=words, values=cos_sim."""
    col = 10
    print(f"\n{'='*65}")
    print(f"  {model_name}  (text-audio cosine similarity, higher = closer to word)")
    print(f"{'='*65}")
    header = f"{'EQ preset':<12}" + "".join(f"{w:>{col}}" for w in words)
    print(header)
    print("-" * len(header))
    dry = results["DRY"]
    for preset, scores in results.items():
        row = f"{preset:<12}"
        for w in words:
            s = scores[w]
            if preset == "DRY":
                row += f"{s:>{col}.4f}"
            else:
                delta = s - dry[w]
                marker = "↑" if delta > 0.003 else ("↓" if delta < -0.003 else " ")
                row += f"{s:>{col-1}.4f}{marker}"
        print(row)
    print("-" * len(header))
    # diagonal sensitivity: does preset X increase score for word X?
    print(f"{'diag Δ':<12}", end="")
    for w in words:
        if w in results:
            delta = results[w][w] - dry[w]
            print(f"{delta:>+{col}.4f}", end="")
        else:
            print(f"{'N/A':>{col}}", end="")
    print("\n")


def print_fxenc_table(results: dict):
    """Table for FX-encoder++: rows=EQ preset, single col=distance from DRY."""
    print(f"\n{'='*40}")
    print(f"  FX-encoder++  (cosine distance from DRY)")
    print(f"{'='*40}")
    print(f"{'EQ preset':<12}  {'dist_from_dry':>14}")
    print("-" * 30)
    for preset, dist in results.items():
        if preset == "DRY":
            print(f"{'DRY':<12}  {dist:>14.4f}  (baseline)")
        else:
            bar = "█" * int(dist * 500)
            print(f"{preset:<12}  {dist:>14.4f}  {bar}")
    print()


# ── runners ───────────────────────────────────────────────────────────────────

def run_text_audio(scorer, model_name: str):
    audio, sr = load_dry()
    results = {}
    print(f"\nComputing {model_name}...", flush=True)
    results["DRY"] = {w: scorer(audio, sr, w) for w in WORDS}
    print("  DRY done")
    for preset_name, plugins in EQ_PRESETS.items():
        processed = apply_eq(audio, sr, plugins)
        results[preset_name] = {w: scorer(processed, sr, w) for w in WORDS}
        print(f"  {preset_name} done")
    print_clap_table(model_name, results, WORDS)


def run_fxenc():
    scorer, get_emb, dry_cache = make_fxenc_scorer()
    audio, sr = load_dry()
    results = {}
    print("\nComputing FX-encoder++...", flush=True)

    # DRY baseline — scorer initializes dry_cache on first call
    results["DRY"] = scorer(audio, sr, "")   # word arg unused
    print("  DRY done")

    for preset_name, plugins in EQ_PRESETS.items():
        processed = apply_eq(audio, sr, plugins)
        results[preset_name] = scorer(processed, sr, "")
        print(f"  {preset_name} done")

    print_fxenc_table(results)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["clap", "fxenc", "musiccoca", "all"],
        default="all",
    )
    args = parser.parse_args()

    if args.model in ("clap", "all"):
        run_text_audio(make_clap_scorer(), "CLAP")

    if args.model in ("fxenc", "all"):
        run_fxenc()

    if args.model in ("musiccoca", "all"):
        try:
            run_text_audio(make_musiccoca_scorer(), "MusicCoCa")
        except ImportError as e:
            print(f"\n[MusicCoCa skipped] {e}")


if __name__ == "__main__":
    main()
