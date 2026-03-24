"""
src/metrics/run_metrics.py — Orchestrate all evaluation metrics.

Two entry points:

    run_metrics_against_gt(gt_folder, technique_folder)
        Compares a technique's outputs to ground-truth audio.
        Metrics: MMD on DSP features, avg DSP-feature distance to GT centroid,
                 CLAP text-audio similarity.

    run_metrics_for_comparison({name: folder, ...})
        Compares multiple techniques side-by-side (no GT needed).
        Metrics: CLAP similarity, FAD, loudness (LUFS).

Both expect folders structured as:  {word}/{instrument}/*.wav
The word subfolder name is used as the text prompt for CLAP scoring.
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import glob
import tqdm
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist

from metrics.dsp_feature_metrics import (
    extract_dsp_features,
    extract_dsp_features_from_array,
    extract_features_batch,
)
from metrics.mmd_metric import compute_mmd
from metrics.clap_metrics import compute_clap_score
from metrics.audio_quality_metrics import FxSearcherFAD, FxSearcherIntegratedLUFS
from metrics.deep_embedding_metrics import get_fx_embedding


# ── Helpers ──────────────────────────────────────────────────────────────────

def _discover_words_instruments(folder: str) -> List[Tuple[str, str]]:
    """Return (word, instrument) pairs found under folder/{word}/{instrument}/...

    Supports both flat layout ({word}/{instrument}/*.wav) and the experiment
    layout ({word_pair}/{instrument}/experiment_*/…/run_*/final_audio.wav).
    """
    pairs = []
    if not os.path.isdir(folder):
        return pairs
    for word in sorted(os.listdir(folder)):
        word_dir = os.path.join(folder, word)
        if not os.path.isdir(word_dir):
            continue
        for instrument in sorted(os.listdir(word_dir)):
            inst_dir = os.path.join(word_dir, instrument)
            if not os.path.isdir(inst_dir):
                continue
            # Check flat WAVs first, then deep experiment structure
            if (any(f.endswith(".wav") for f in os.listdir(inst_dir))
                    or glob.glob(os.path.join(inst_dir, "**", "final_audio.wav"), recursive=True)):
                pairs.append((word, instrument))
    return pairs


def _wav_paths(folder: str, word: str, instrument: str) -> List[str]:
    """Sorted .wav paths under folder/word/instrument/.

    Looks for flat WAVs first; if none, searches recursively for
    final_audio.wav (experiment layout).
    """
    d = os.path.join(folder, word, instrument)
    if not os.path.isdir(d):
        return []
    flat = sorted(glob.glob(os.path.join(d, "*.wav")))
    if flat:
        return flat
    return sorted(glob.glob(os.path.join(d, "**", "final_audio.wav"), recursive=True))


# ── run_metrics_against_gt ───────────────────────────────────────────────────

def run_metrics_against_gt(
    gt_folder: str,
    test_folders: Dict[str, str],
    sr: int = 22050,
) -> Dict[str, Any]:
    """Compare a technique's outputs against ground-truth audio.

    Both folders must have structure:  {word}/{instrument}/*.wav

    Returns dict with:
        mmd_dsp_per_word      – {word: mmd_score}
        mmd_dsp_overall       – single MMD across all words
        dsp_distance_per_word – {word: avg_euclidean_distance_to_gt_centroid}
        dsp_distance_overall  – grand avg
        clap_per_word         – {word: avg_clap_score}
        clap_overall          – grand avg CLAP score
    """
    print(f"\n{'='*60}")
    print("  Metrics against GT")
    print(f"{'='*60}")
    print(f"  GT folder:        {gt_folder}")
    print(f"  Test folders:      {test_folders}")

    # Accumulators
    gt_feats_by_word: Dict[str, List[np.ndarray]] = {}
    gt_fxenc_by_word: Dict[str, List[np.ndarray]] = {}
    gt_pairs = _discover_words_instruments(gt_folder)
    gt_wavs_covered = {}
    gt_fxenc_covered = {}

    results: Dict[str, Dict[str, Any]] = {}

    for approach_name, test_folder in test_folders.items():
        print(f"\n  ── {approach_name} ──")
        tech_pairs = _discover_words_instruments(test_folder)
        common = sorted(set(gt_pairs) & set(tech_pairs))
        print(f"    Common (word, instrument) pairs: {len(common)}")
        if not common:
            print("[run_metrics] No overlapping (word, instrument) pairs found.")
            return {}

        tech_feats_by_word: Dict[str, List[np.ndarray]] = {}
        tech_fxenc_by_word: Dict[str, List[np.ndarray]] = {}

        for word, instrument in tqdm.tqdm(common):
            gt_wavs = _wav_paths(gt_folder, word, instrument)
            tech_wavs = _wav_paths(test_folder, word, instrument)
            if not gt_wavs or not tech_wavs:
                continue

            # DSP features + FXEnc embeddings for GT
            for p in gt_wavs:
                if p in gt_wavs_covered:
                    f = gt_wavs_covered[p]
                else:
                    f = extract_dsp_features(p, sr=sr)
                    gt_wavs_covered[p] = f
                gt_feats_by_word.setdefault(word, []).append(f)

                if p in gt_fxenc_covered:
                    emb = gt_fxenc_covered[p]
                else:
                    emb = get_fx_embedding(p).detach().cpu().numpy().flatten()
                    gt_fxenc_covered[p] = emb
                gt_fxenc_by_word.setdefault(word, []).append(emb)

            # DSP features + FXEnc embeddings for test
            for p in tech_wavs:
                f = extract_dsp_features(p, sr=sr)
                tech_feats_by_word.setdefault(word, []).append(f)

                emb = get_fx_embedding(p).detach().cpu().numpy().flatten()
                tech_fxenc_by_word.setdefault(word, []).append(emb)

        # ── Per-word MMD on DSP features ─────────────────────────────────────
        mmd_per_word: Dict[str, float] = {}
        for word in sorted(gt_feats_by_word):
            if word not in tech_feats_by_word:
                continue
            X = np.array(gt_feats_by_word[word])
            Y = np.array(tech_feats_by_word[word])
            mmd_per_word[word] = compute_mmd(X, Y)

        # Overall MMD
        all_gt = np.concatenate([np.array(v) for v in gt_feats_by_word.values()]) if gt_feats_by_word else np.empty((0, 0))
        all_tech = np.concatenate([np.array(v) for v in tech_feats_by_word.values()]) if tech_feats_by_word else np.empty((0, 0))
        mmd_overall = compute_mmd(all_gt, all_tech) if all_gt.size and all_tech.size else float("nan")

        # ── Per-word avg Euclidean distance to GT centroid ────────────────────
        dsp_dist_per_word: Dict[str, float] = {}
        for word in sorted(gt_feats_by_word):
            if word not in tech_feats_by_word:
                continue
            gt_centroid = np.mean(gt_feats_by_word[word], axis=0)
            dists = [float(np.linalg.norm(f - gt_centroid)) for f in tech_feats_by_word[word]]
            dsp_dist_per_word[word] = float(np.mean(dists))

        dsp_dist_overall = float(np.mean(list(dsp_dist_per_word.values()))) if dsp_dist_per_word else float("nan")

        # ── Per-word cosine distance between avg FXEnc embeddings ─────────
        fxenc_dist_per_word: Dict[str, float] = {}
        for word in sorted(gt_fxenc_by_word):
            if word not in tech_fxenc_by_word:
                continue
            gt_avg = np.mean(gt_fxenc_by_word[word], axis=0)
            tech_avg = np.mean(tech_fxenc_by_word[word], axis=0)
            fxenc_dist_per_word[word] = float(cosine_dist(gt_avg, tech_avg))

        fxenc_dist_overall = float(np.mean(list(fxenc_dist_per_word.values()))) if fxenc_dist_per_word else float("nan")

        results[approach_name] = {
            "mmd_dsp_per_word": mmd_per_word,
            "mmd_dsp_overall": mmd_overall,
            "dsp_distance_per_word": dsp_dist_per_word,
            "dsp_distance_overall": dsp_dist_overall,
            "fxenc_distance_per_word": fxenc_dist_per_word,
            "fxenc_distance_overall": fxenc_dist_overall,
        }

    # Print summary
    for approach_name, metrics in results.items():
        print(f"\n  ── {approach_name} ──")
        print(f"    MMD (DSP, overall): {metrics['mmd_dsp_overall']:.4f}")
        for w, v in metrics['mmd_dsp_per_word'].items():
            print(f"      {w:12s}: {v:.4f}")
        print(f"    DSP distance to GT centroid (overall): {metrics['dsp_distance_overall']:.4f}")
        for w, v in metrics['dsp_distance_per_word'].items():
            print(f"      {w:12s}: {v:.4f}")
        print(f"    FXEnc cosine distance (overall): {metrics['fxenc_distance_overall']:.4f}")
        for w, v in metrics['fxenc_distance_per_word'].items():
            print(f"      {w:12s}: {v:.4f}")

    return results


# ── run_metrics_for_comparison ───────────────────────────────────────────────

def run_metrics_for_comparison(
    technique_folders: Dict[str, str],
    sr: int = 22050,
) -> Dict[str, Dict[str, Any]]:
    """Compare multiple techniques side-by-side (no GT reference).

    Args:
        technique_folders: {technique_name: folder_path}
            Each folder has structure:  {word}/{instrument}/*.wav

    Returns dict keyed by technique name, each containing:
        clap_per_word  – {word: avg_clap_score}
        clap_overall   – grand avg
        lufs_per_word  – {word: avg_lufs}
        lufs_overall   – grand avg
    """
    lufs_metric = FxSearcherIntegratedLUFS()

    print(f"\n{'='*60}")
    print("  Comparative Metrics (no GT)")
    print(f"{'='*60}")
    print(f"  Techniques: {list(technique_folders.keys())}")

    results: Dict[str, Dict[str, Any]] = {}

    for name, folder in technique_folders.items():
        print(f"\n  ── {name} ──")
        pairs = _discover_words_instruments(folder)
        if not pairs:
            print(f"    No (word, instrument) pairs found in {folder}")
            results[name] = {}
            continue

        clap_by_word: Dict[str, List[float]] = {}
        lufs_by_word: Dict[str, List[float]] = {}

        for word, instrument in pairs:
            wavs = _wav_paths(folder, word, instrument)
            for p in wavs:
                # CLAP
                score = compute_clap_score(p, word)
                clap_by_word.setdefault(word, []).append(score)

                # LUFS
                lufs = lufs_metric.compute(None, p)
                lufs_by_word.setdefault(word, []).append(lufs)

        clap_per_word = {w: float(np.mean(s)) for w, s in clap_by_word.items()}
        clap_overall = float(np.mean(list(clap_per_word.values()))) if clap_per_word else float("nan")

        lufs_per_word = {w: float(np.mean(s)) for w, s in lufs_by_word.items()}
        lufs_overall = float(np.mean(list(lufs_per_word.values()))) if lufs_per_word else float("nan")

        results[name] = {
            "clap_per_word": clap_per_word,
            "clap_overall": clap_overall,
            "lufs_per_word": lufs_per_word,
            "lufs_overall": lufs_overall,
        }

        print(f"    CLAP (overall): {clap_overall:.4f}")
        print(f"    LUFS (overall): {lufs_overall:.1f}")

    # # ── Pairwise FAD between techniques ──────────────────────────────────
    # names = list(technique_folders.keys())
    # if len(names) >= 2:
    #     fad = FxSearcherFAD()
    #     print(f"\n  ── Pairwise FAD ──")
    #     for i, n1 in enumerate(names):
    #         for n2 in names[i + 1:]:
    #             fad_score = fad.compute(technique_folders[n1], technique_folders[n2])
    #             print(f"    {n1} vs {n2}: {fad_score:.4f}")
    #             results.setdefault("_pairwise_fad", {})[f"{n1}_vs_{n2}"] = fad_score

    # return results




if __name__ == "__main__":

    run_metrics_against_gt(
        gt_folder="../../eval/gt_cache/one-shot",
        test_folders={
            "llm": "../../eval/system_outputs/exp3/llm",
            "clap_loss": "../../eval/system_outputs/exp3/clap_loss",
        },
        sr=22050,
    )
