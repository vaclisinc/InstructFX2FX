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
import re
from datetime import datetime
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

def _save_timestamped_reports(prefix: str, payload: Dict[str, Any], summary_text: str) -> Tuple[str, str]:
    """Save JSON+TXT reports as eval/results/archive/{prefix}_{YYYY-mm-dd_HH-MM-SS}.{json,txt}."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join("eval", "results", "archive", f"{prefix}_{ts}")
    json_path = f"{base}.json"
    txt_path = f"{base}.txt"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    with open(txt_path, "w") as f:
        f.write(summary_text)
    return json_path, txt_path

def _discover_words_instruments(folder: str, word_position_in_test_folder_name: Optional[int] = None) -> List[Tuple[str, str]]:
    """Return (word, instrument) pairs found under folder/{word}/{instrument}/...

    Supports both flat layout ({word}/{instrument}/*.wav) and the experiment
    layout ({word_pair}/{instrument}/experiment_*/…/run_*/final_audio.wav).
    """
    pairs = []
    if not os.path.isdir(folder):
        print(f"[run_metrics] Warning: folder {folder} does not exist or is not a directory.")
        return pairs
    for word in sorted(os.listdir(folder)):
        part = None
        if word_position_in_test_folder_name is not None:
            parts = word.split("_")
            if len(parts) > word_position_in_test_folder_name:
                part = parts[word_position_in_test_folder_name]
            else:
                raise ValueError(f"word_position_in_test_folder_name={word_position_in_test_folder_name} is out of range for folder name '{word}'")

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
                pairs.append((word, instrument, part))
    return pairs


def _wav_paths(
    folder: str,
    word: str,
    instrument: str,
    search_pattern: Optional[str] = None,
) -> List[str]:
    """Sorted .wav paths under folder/word/instrument/.

    Looks for flat WAVs first; if none, searches recursively for
    final_audio.wav (experiment layout).
    """
    if search_pattern is not None:
        pattern_path = os.path.join(folder, word, instrument, search_pattern)
        if any(ch in search_pattern for ch in "*?[]"):
            return sorted(glob.glob(pattern_path, recursive=True))
        return [pattern_path] if os.path.isfile(pattern_path) else []

    else:
        d = os.path.join(folder, word, instrument)
    if not os.path.isdir(d):
        return []
    flat = sorted(glob.glob(os.path.join(d, "*.wav")))
    if flat:
        return flat
    return sorted(glob.glob(os.path.join(d, "**", "final_audio.wav"), recursive=True))


_ITER_FILE_RE = re.compile(r"iter_(\d+)\.wav$")


def _extract_iteration_from_path(path: str) -> Optional[int]:
    m = _ITER_FILE_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def run_iteration_improvement_metrics(
    test_folders: Dict[str, str],
    search_patterns: Dict[str, str],
    gt_folder: Optional[str] = None,
    baseline_pattern: str = "intermediate/optimization_steps/start.wav",
    improvement_epsilon: float = 1e-6,
) -> Dict[str, Dict[str, Any]]:
    """Estimate iterations needed to improve over the LLM baseline on average.

    For each (word, instrument), this expects trajectory snapshots matched by
    `search_patterns[name]` such as `intermediate/optimization_steps/iter_*.wav`.
    Baseline CLAP score is taken from `baseline_pattern` (default: start.wav).

    This function reuses the existing metric runners:
      - run_metrics_solo: CLAP trajectory and first CLAP improvement.
      - run_metrics_against_gt (optional): first GT-distance improvement.
    """
    print(f"\n{'='*60}")
    print("  Iterations-to-Improvement Metrics")
    print(f"{'='*60}")

    results: Dict[str, Dict[str, Any]] = {}

    baseline_search = {name: baseline_pattern for name in test_folders}
    baseline_solo = run_metrics_solo(
        test_folders=test_folders,
        sr=22050,
        search_patterns=baseline_search,
        save_report=False,
        verbose=False,
    )

    baseline_gt: Dict[str, Dict[str, Any]] = {}
    if gt_folder:
        baseline_gt = run_metrics_against_gt(
            gt_folder=gt_folder,
            test_folders=test_folders,
            sr=22050,
            search_patterns=baseline_search,
            save_report=False,
            verbose=False,
        )

    for name, folder in test_folders.items():
        pattern = search_patterns.get(name)
        if not pattern:
            print(f"  [{name}] No search pattern provided, skipping.")
            results[name] = {}
            continue

        pairs = _discover_words_instruments(folder)
        if not pairs:
            print(f"  [{name}] No (word, instrument, part) pairs found.")
            results[name] = {}
            continue

        iteration_ids: List[int] = sorted(
            {
                it
                for word, instrument, part in pairs
                for p in _wav_paths(folder, word, instrument, search_pattern=pattern)
                for it in [_extract_iteration_from_path(p)]
                if it is not None
            }
        )
        if not iteration_ids:
            print(f"  [{name}] No iter_*.wav snapshots found for pattern '{pattern}'.")
            results[name] = {}
            continue

        baseline_clap_per_word = baseline_solo.get(name, {}).get("clap_per_word", {})
        if not baseline_clap_per_word:
            print(f"  [{name}] Could not compute baseline CLAP with '{baseline_pattern}'.")
            results[name] = {}
            continue

        baseline_gt_dsp_per_word: Dict[str, float] = {}
        baseline_gt_mmd_overall: Optional[float] = None
        baseline_gt_fxenc_overall: Optional[float] = None
        if gt_folder:
            baseline_gt_dsp_per_word = baseline_gt.get(name, {}).get("dsp_distance_per_word", {})
            baseline_gt_mmd_overall = baseline_gt.get(name, {}).get("mmd_dsp_overall")
            baseline_gt_fxenc_overall = baseline_gt.get(name, {}).get("fxenc_distance_overall")

        first_iter_by_word: Dict[str, Optional[int]] = {w: None for w in baseline_clap_per_word}
        first_iter_gt_by_word: Dict[str, Optional[int]] = {w: None for w in baseline_clap_per_word}
        first_iter_improved_mmd: Optional[int] = None
        first_iter_improved_fxenc: Optional[int] = None
        best_iter = iteration_ids[0]
        best_clap = float("-inf")
        trajectory: List[Dict[str, Any]] = []

        for it in iteration_ids:
            iter_pattern = pattern.replace("*", str(it))
            iter_solo = run_metrics_solo(
                test_folders={name: folder},
                sr=22050,
                search_patterns={name: iter_pattern},
                save_report=False,
                verbose=False,
            )
            iter_clap = iter_solo.get(name, {}).get("clap_overall")
            iter_clap_per_word = iter_solo.get(name, {}).get("clap_per_word", {})
            iter_gt_dsp = None
            iter_gt_mmd = None
            iter_gt_fxenc = None
            iter_gt_dsp_per_word: Dict[str, float] = {}

            if gt_folder:
                iter_gt = run_metrics_against_gt(
                    gt_folder=gt_folder,
                    test_folders={name: folder},
                    sr=22050,
                    search_patterns={name: iter_pattern},
                    save_report=False,
                    verbose=False,
                )
                iter_gt_dsp = iter_gt.get(name, {}).get("dsp_distance_overall")
                iter_gt_mmd = iter_gt.get(name, {}).get("mmd_dsp_overall")
                iter_gt_fxenc = iter_gt.get(name, {}).get("fxenc_distance_overall")
                iter_gt_dsp_per_word = iter_gt.get(name, {}).get("dsp_distance_per_word", {})

            if iter_clap is None or np.isnan(iter_clap):
                continue

            if iter_clap > best_clap:
                best_clap = float(iter_clap)
                best_iter = it

            for word, baseline_word_clap in baseline_clap_per_word.items():
                word_iter_clap = iter_clap_per_word.get(word)
                if (
                    first_iter_by_word[word] is None
                    and word_iter_clap is not None
                    and word_iter_clap > baseline_word_clap + improvement_epsilon
                ):
                    first_iter_by_word[word] = it

            if gt_folder:
                for word, baseline_word_dsp in baseline_gt_dsp_per_word.items():
                    word_iter_dsp = iter_gt_dsp_per_word.get(word)
                    if (
                        word in first_iter_gt_by_word
                        and first_iter_gt_by_word[word] is None
                        and word_iter_dsp is not None
                        and word_iter_dsp < baseline_word_dsp - improvement_epsilon
                    ):
                        first_iter_gt_by_word[word] = it

                if (
                    first_iter_improved_mmd is None
                    and baseline_gt_mmd_overall is not None
                    and iter_gt_mmd is not None
                    and not np.isnan(baseline_gt_mmd_overall)
                    and not np.isnan(iter_gt_mmd)
                    and iter_gt_mmd < baseline_gt_mmd_overall - improvement_epsilon
                ):
                    first_iter_improved_mmd = it

                if (
                    first_iter_improved_fxenc is None
                    and baseline_gt_fxenc_overall is not None
                    and iter_gt_fxenc is not None
                    and not np.isnan(baseline_gt_fxenc_overall)
                    and not np.isnan(iter_gt_fxenc)
                    and iter_gt_fxenc < baseline_gt_fxenc_overall - improvement_epsilon
                ):
                    first_iter_improved_fxenc = it

            trajectory.append(
                {
                    "iteration": it,
                    "clap_overall": float(iter_clap),
                    "mmd_dsp_overall": float(iter_gt_mmd) if iter_gt_mmd is not None else None,
                    "dsp_distance_overall": float(iter_gt_dsp) if iter_gt_dsp is not None else None,
                    "fxenc_distance_overall": float(iter_gt_fxenc) if iter_gt_fxenc is not None else None,
                }
            )

        improved_iters = [i for i in first_iter_by_word.values() if i is not None]
        improved_words = len(improved_iters)
        total_words = len(first_iter_by_word)

        results[name] = {
            "baseline_clap_overall": float(baseline_solo.get(name, {}).get("clap_overall", float("nan"))),
            "baseline_mmd_dsp_overall": (
                float(baseline_gt_mmd_overall)
                if baseline_gt_mmd_overall is not None and not np.isnan(baseline_gt_mmd_overall)
                else None
            ),
            "baseline_dsp_distance_overall": (
                float(baseline_gt.get(name, {}).get("dsp_distance_overall"))
                if gt_folder and baseline_gt.get(name, {}).get("dsp_distance_overall") is not None
                else None
            ),
            "baseline_fxenc_distance_overall": (
                float(baseline_gt_fxenc_overall)
                if baseline_gt_fxenc_overall is not None and not np.isnan(baseline_gt_fxenc_overall)
                else None
            ),
            "average_iterations_to_improve": float(np.mean(improved_iters)) if improved_iters else None,
            "median_iterations_to_improve": float(np.median(improved_iters)) if improved_iters else None,
            "std_iterations_to_improve": float(np.std(improved_iters)) if improved_iters else None,
            "first_iter_improved_clap_per_word": first_iter_by_word,
            "first_iter_improved_dsp_distance_per_word": first_iter_gt_by_word if gt_folder else None,
            "first_iter_improved_mmd_dsp_overall": first_iter_improved_mmd if gt_folder else None,
            "first_iter_improved_fxenc_distance_overall": first_iter_improved_fxenc if gt_folder else None,
            "best_iter": int(best_iter),
            "best_clap": float(best_clap) if best_clap != float("-inf") else None,
            "improved_pairs": improved_words,
            "total_pairs": total_words,
            "improvement_rate": (float(improved_words / total_words) if total_words else None),
            "trajectory": trajectory,
        }

    build_summary_text_iterations(results)
    return results


def build_summary_text_iterations(results: Dict[str, Dict[str, Any]]) -> str:
    summary_lines: List[str] = []
    for name, metrics in results.items():
        if not metrics:
            continue
        summary_lines.append(f"\n  ── {name} ──")
        summary_lines.append(
            f"    Avg iterations to improve CLAP over baseline: {metrics['average_iterations_to_improve']}"
        )
        summary_lines.append(
            f"    Median iterations to improve: {metrics['median_iterations_to_improve']}"
        )
        summary_lines.append(
            f"    Improvement rate: {metrics['improved_pairs']}/{metrics['total_pairs']} ({metrics['improvement_rate']})"
        )
        if metrics.get("baseline_mmd_dsp_overall") is not None:
            summary_lines.append(
                f"    Baseline MMD (DSP): {metrics['baseline_mmd_dsp_overall']:.4f}"
            )
            summary_lines.append(
                f"    First iter improved MMD: {metrics.get('first_iter_improved_mmd_dsp_overall')}"
            )
        if metrics.get("baseline_dsp_distance_overall") is not None:
            summary_lines.append(
                f"    Baseline DSP distance: {metrics['baseline_dsp_distance_overall']:.4f}"
            )
        if metrics.get("baseline_fxenc_distance_overall") is not None:
            summary_lines.append(
                f"    Baseline FXEnc distance: {metrics['baseline_fxenc_distance_overall']:.4f}"
            )
            summary_lines.append(
                f"    First iter improved FXEnc: {metrics.get('first_iter_improved_fxenc_distance_overall')}"
            )

    valid_results = [(n, m) for n, m in results.items() if m and m.get("average_iterations_to_improve") is not None]
    if valid_results:
        summary_lines.append("\n  Ranking by average iterations to improve (lower is better):")
        ranked = sorted(valid_results, key=lambda kv: kv[1]["average_iterations_to_improve"])
        for i, (name, m) in enumerate(ranked, start=1):
            summary_lines.append(
                f"      {i}. {name:12s} | avg_iter={m['average_iterations_to_improve']:.2f}, improve_rate={m['improvement_rate']:.2%}"
            )

    summary_text = "\n".join(summary_lines)
    if summary_text:
        print(summary_text)

    results.update({
        "context": "this is an automatically saved report",
        "timestamp": datetime.now().isoformat(),
        "user": os.getenv("USER", "unknown")
    })

    report_txt_extension = "\n\n\n: This is an automatically saved report. Timestamp: " + datetime.now().isoformat() + ". User: " + os.getenv("USER", "unknown") + "\n"

    report_json, report_txt = _save_timestamped_reports("metrics_iterations", results, summary_text + report_txt_extension)
    print(f"\n  Report saved (json): {report_json}")
    print(f"  Report saved (txt):  {report_txt}")
    return summary_text


# ── run_metrics_against_gt ───────────────────────────────────────────────────

def run_metrics_against_gt(
    gt_folder: str,
    test_folders: Dict[str, str],
    sr: int = 22050,
    search_patterns: Optional[Dict[str, str]] = None,
    word_position_in_test_folder_name: Optional[int] = None,
    save_report: bool = True,
    verbose: bool = True,
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
    if verbose:
        print(f"\n{'='*60}")
        print("  Metrics against GT")
        print(f"{'='*60}")
        print(f"  GT folder:        {gt_folder}")
        print(f"  Test folders:      {test_folders}")

    # Accumulators
    gt_feats_by_word: Dict[str, List[np.ndarray]] = {}
    gt_fxenc_by_word: Dict[str, List[np.ndarray]] = {}
    gt_words = _discover_words_instruments(gt_folder)
    gt_wavs_covered = {}
    gt_fxenc_covered = {}

    results: Dict[str, Dict[str, Any]] = {}

    search_patterns = search_patterns or {}

    for approach_name, test_folder in test_folders.items():
        if verbose:
            print(f"\n  ── {approach_name} ──")
        search_pattern = search_patterns.get(approach_name)
        tech_words = _discover_words_instruments(test_folder)
        # Only compare (word, instrument), ignore 'part' for set intersection
        gt_pairs_set = set((w, i) for w, i, _ in gt_words)
        tech_pairs_set = set((w, i) for w, i, _ in tech_words)
        common = sorted(gt_pairs_set & tech_pairs_set)
        if verbose:
            print(f"    Common (word, instrument) pairs: {len(common)}")
        if not common:
            if verbose:
                print("[run_metrics] No overlapping (word, instrument) pairs found.")
                print('GT folder', os.listdir(gt_folder))
                print('Test folder', os.listdir(test_folder))
                print('GT pairs:', gt_words)
                print('Test pairs:', tech_words)
            return {}

        tech_feats_by_word: Dict[str, List[np.ndarray]] = {}
        tech_fxenc_by_word: Dict[str, List[np.ndarray]] = {}

        for word, instrument in tqdm.tqdm(common):
            # Find all matching (word, instrument, part) in GT and test
            gt_matches = [t for t in gt_words if t[0] == word and t[1] == instrument]
            tech_matches = [t for t in tech_words if t[0] == word and t[1] == instrument]
            # Use the first part (or None) for each, for compatibility
            gt_part = gt_matches[0][2] if gt_matches else None
            tech_part = tech_matches[0][2] if tech_matches else None
            gt_wavs = _wav_paths(gt_folder, word, instrument)
            tech_wavs = _wav_paths(test_folder, word, instrument, search_pattern=search_pattern)
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

    if save_report:
        build_summary_text_gt(results)

    return results


def build_summary_text_gt(results):
    # Print summary + build report text
    summary_lines: List[str] = []
    for approach_name, metrics in results.items():
        summary_lines.append(f"\n  ── {approach_name} ──")
        summary_lines.append(f"    MMD (DSP, overall): {metrics['mmd_dsp_overall']:.4f}")
        for w, v in metrics['mmd_dsp_per_word'].items():
            summary_lines.append(f"      {w:12s}: {v:.4f}")
        summary_lines.append(f"    DSP distance to GT centroid (overall): {metrics['dsp_distance_overall']:.4f}")
        for w, v in metrics['dsp_distance_per_word'].items():
            summary_lines.append(f"      {w:12s}: {v:.4f}")
        summary_lines.append(f"    FXEnc cosine distance (overall): {metrics['fxenc_distance_overall']:.4f}")
        for w, v in metrics['fxenc_distance_per_word'].items():
            summary_lines.append(f"      {w:12s}: {v:.4f}")

    summary_lines.append("\n  Comparison ordering by metric (against GT):")

    mmd_ranking = sorted(
        results.items(),
        key=lambda kv: kv[1].get("mmd_dsp_overall", float("inf")),
    )
    summary_lines.append("    MMD (DSP, lower is better):")
    for i, (name, m) in enumerate(mmd_ranking, start=1):
        summary_lines.append(f"      {i}. {name:12s} | {m['mmd_dsp_overall']:.4f}")

    dsp_ranking = sorted(
        results.items(),
        key=lambda kv: kv[1].get("dsp_distance_overall", float("inf")),
    )
    summary_lines.append("    DSP distance (lower is better):")
    for i, (name, m) in enumerate(dsp_ranking, start=1):
        summary_lines.append(f"      {i}. {name:12s} | {m['dsp_distance_overall']:.4f}")

    fxenc_ranking = sorted(
        results.items(),
        key=lambda kv: kv[1].get("fxenc_distance_overall", float("inf")),
    )
    summary_lines.append("    FXEnc cosine distance (lower is better):")
    for i, (name, m) in enumerate(fxenc_ranking, start=1):
        summary_lines.append(f"      {i}. {name:12s} | {m['fxenc_distance_overall']:.4f}")

    summary_text = "\n".join(summary_lines)
    if summary_text:
        print(summary_text)

    results.update({
        "context": "this is an automatically saved report",
        "timestamp": datetime.now().isoformat(),
        "user": os.getenv("USER", "unknown")
    })

    report_txt_extension = "\n\n\n: This is an automatically saved report. Timestamp: " + datetime.now().isoformat() + ". User: " + os.getenv("USER", "unknown") + "\n"
    report_json, report_txt = _save_timestamped_reports("metrics_against_gt", results, summary_text + report_txt_extension)
    print(f"\n  Report saved (json): {report_json}")
    print(f"  Report saved (txt):  {report_txt}")






# ── run_metrics_solo ───────────────────────────────────────────────

def run_metrics_solo(
    test_folders: Dict[str, str],
    sr: int = 22050,
    search_patterns: Optional[Dict[str, str]] = None,
    save_report: bool = True,
    verbose: bool = True,
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

    if verbose:
        print(f"\n{'='*60}")
        print("  Comparative Metrics (no GT)")
        print(f"{'='*60}")
        print(f"  Techniques: {list(test_folders.keys())}")

    results: Dict[str, Dict[str, Any]] = {}

    search_patterns = search_patterns or {}

    for name, folder in test_folders.items():
        search_pattern = search_patterns.get(name, None)
        pairs = _discover_words_instruments(folder)
        if not pairs:
            if verbose:
                print(f"    No (word, instrument, part) pairs found in {folder}")
            results[name] = {}
            continue

        clap_by_word: Dict[str, List[float]] = {}
        lufs_by_word: Dict[str, List[float]] = {}

        for word, instrument, part in pairs:
            wavs = _wav_paths(folder, word, instrument, search_pattern=search_pattern)
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
    if save_report:
        build_summary_text_solo(results)
    return results

def build_summary_text_solo(results: Dict[str, Dict[str, Any]]) -> str:
    summary_lines: List[str] = []
    for name, metrics in results.items():
        if not metrics:
            continue
        summary_lines.append(f"\n  ── {name} ──")
        summary_lines.append(f"    CLAP (overall): {metrics['clap_overall']:.4f}")
        summary_lines.append(f"    LUFS (overall): {metrics['lufs_overall']:.1f}")

    valid_results = [(n, m) for n, m in results.items() if m]
    summary_lines.append("\n  Comparison ordering by metric (solo):")

    clap_ranking = sorted(
        valid_results,
        key=lambda kv: kv[1].get("clap_overall", float("-inf")),
        reverse=True,
    )
    summary_lines.append("    CLAP (higher is better):")
    for i, (name, m) in enumerate(clap_ranking, start=1):
        summary_lines.append(f"      {i}. {name:12s} | {m['clap_overall']:.4f}")

    lufs_ranking = sorted(
        valid_results,
        key=lambda kv: abs(kv[1].get("lufs_overall", 0.0) + 14.0),
    )
    summary_lines.append("    LUFS (closer to -14.0 is better):")
    for i, (name, m) in enumerate(lufs_ranking, start=1):
        summary_lines.append(
            f"      {i}. {name:12s} | {m['lufs_overall']:.1f} (|delta|={abs(m['lufs_overall'] + 14.0):.1f})"
        )

    summary_text = "\n".join(summary_lines)
    if summary_text:
        print(summary_text)

    results.update({
        "context": "this is an automatically saved report",
        "timestamp": datetime.now().isoformat(),
        "user": os.getenv("USER", "unknown")
    })

    report_txt_extension = "\n\n\n: This is an automatically saved report. Timestamp: " + datetime.now().isoformat() + ". User: " + os.getenv("USER", "unknown") + "\n"
    report_json, report_txt = _save_timestamped_reports("metrics_solo", results, summary_text + report_txt_extension)
    print(f"\n  Report saved (json): {report_json}")
    print(f"  Report saved (txt):  {report_txt}")


def run_all_metrics(gt_folder, test_folders):
    run_metrics_solo(
        test_folders=test_folders,
        sr=22050,
    )

    run_metrics_against_gt(
        test_folders=test_folders,
        gt_folder= gt_folder
    )

if __name__ == "__main__":

    # run_metrics_against_gt(
    #     gt_folder="eval/gt_cache/one-shot",
    #     test_folders={
    #         "llm": "eval/system_outputs/exp3/llm",
    #         "clap_loss": "eval/system_outputs/exp3/clap_loss",
    #     },
    #     sr=22050,
    # )

    run_metrics_solo(
        test_folders={
            "llm": "eval/system_outputs/exp3/llm",
            "clap_loss": "eval/system_outputs/exp3/clap_loss",
        },
        search_patterns={
            "llm_clap": "intermediate/optimization_steps/iter_30.wav"
        },
        sr=22050,
    )

    run_iteration_improvement_metrics(
        test_folders={
            "llm_clap": "eval/system_outputs/exp5/llm_clap",
        },
        search_patterns={
            "llm_clap": "intermediate/optimization_steps/iter_*.wav",
        },
        baseline_pattern="intermediate/optimization_steps/start.wav",
    )
