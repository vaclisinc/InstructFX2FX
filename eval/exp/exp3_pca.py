"""
eval/exp/exp3_pca.py — Experiment 3: PCA on timbre features.

Reproduces the Social-FX timbre PCA experiment for **three sources**:
    1. SocialFX ground-truth audio (rendered from dataset EQ params)
    2. Single LLM call (parameter initialisation only, no refinement)
    3. Single CLAP forward-loss run (FxSearcher Bayesian optimisation)

Pipeline:
    Step 1  Build single-word GT banks from SocialFX params  (idempotent)
    Step 2  Single LLM init per word × instrument × dry clip  (idempotent)
    Step 3  Single CLAP forward-loss per word × instrument × dry clip (idempotent)
    Step 4  Extract 19-D timbre features, compute feature diffs (wet − dry)
    Step 5  PCA, plot word centroids + per-sample scatter in PC1–PC2 space
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tqdm

from src.configurations.config import OptimizationMethod

_ROOT = Path(__file__).resolve().parent.parent.parent   # project root
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

import eval.config as cfg
from eval.helpers import _dry_paths
from eval.gt_bank import build_word_gt_bank_full

from src.metrics.dsp_feature_metrics import (
    TIMBRE_FEATURE_NAMES,
    extract_timbre_features,
)

# ── Source labels ─────────────────────────────────────────────────────────────

SOURCE_GT = "SocialFX GT"
SOURCE_LLM = "Single LLM Init"
SOURCE_CLAP = "CLAP Forward Loss"

# ── Output directories (under eval/) ─────────────────────────────────────────

_EXP3_SYSTEM_RESULTS_DIR = os.path.join(cfg.SYSTEM_RESULTS_DIR, "exp3")
_EXP3_RESULTS_DIR = os.path.join(cfg.RESULTS_DIR, "exp3")
_EXP3_LLM_DIR = os.path.join(_EXP3_SYSTEM_RESULTS_DIR, "llm")
_EXP3_CLAP_DIR = os.path.join(_EXP3_SYSTEM_RESULTS_DIR, "clap_loss")

# ── Path helpers ─────────────────────────────────────────────────────────────

def _word_gt_paths(word: str, instrument: str) -> List[str]:
    """WAV paths for the single-word GT bank."""
    d = os.path.join(cfg.GT_BANK_DIR_ONESHOT, word, instrument)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".wav"))


# ── Feature extraction helpers ───────────────────────────────────────────────

def _extract_feature_diffs(
    wet_paths: List[str],
    dry_paths: List[str],
    sr: int = cfg.SAMPLE_RATE,
) -> np.ndarray:
    """Compute timbre-feature differences (wet − dry) for all wet paths.

    Each wet file is paired with dry files round-robin (matching the GT bank
    convention where dry_idx is encoded in the filename).

    Returns array of shape (len(wet_paths), 19).
    """
    # Pre-compute dry features
    dry_feats = np.array([extract_timbre_features(p, sr=sr) for p in dry_paths])

    diffs = []
    for idx, wp in enumerate(wet_paths):
        wet_feat = extract_timbre_features(wp, sr=sr)
        # Match dry clip: GT bank names start with "{dry_idx}_…"
        stem = Path(wp).stem
        try:
            dry_idx = int(stem.split("_")[0])
        except (ValueError, IndexError):
            dry_idx = idx  # fallback: round-robin
        dry_feat = dry_feats[dry_idx % len(dry_feats)]
        diffs.append(wet_feat - dry_feat)

    return np.array(diffs, dtype=np.float32)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_exp3(
    llm_client,
    clap,
    fx_type: str = "eq",
    device: str = cfg.DEVICE,
    n_clap_calls: int = 100,
    effects: List[str] = None,
    words: Optional[List[str]] = None,
) -> Dict:
    """End-to-end Experiment 3: timbre PCA across three sources.

    Sources compared:
        1. SocialFX GT — audio rendered from dataset EQ params.
        2. Single LLM init — one LLM call to generate params, no refinement.
        3. CLAP forward loss — FxSearcher Bayesian optimisation with
           SEMANTIC_SIMILARITY_LOSS ("This sound is {word}").

    Args:
        llm_client:   LLMClient for the single LLM init.
        clap:         CLAPWrapper for CLAP-loss runs.
        fx_type:      SocialFX split ("eq" or "reverb").
        device:       torch device string.
        n_clap_calls: Bayesian optimisation iterations for CLAP-loss runs.
        effects:      Effect list (defaults to cfg.FX_EFFECTS).

    Returns:
        Dict with keys: "feature_names", "words" and per-source result blocks.
    """
    import torch
    import soundfile as sf
    import librosa
    from src.configurations.config import (
        Config,
        LossFunction,
        ParameterInitializationMethod,
    )
    from src.training.parameterengine import ParameterEngine
    from src.effects.fx import FXChainFactory
    from src.prompts.prompt import Prompt, PromptFactory
    from src.utilities.audio_processing import _ensure_bct

    if effects is None:
        effects = cfg.FX_EFFECTS

    unique_words = sorted(words)
    parameter_engine = ParameterEngine()

    fx_chain = FXChainFactory.create_fx_chain_from_effects(
        effects, sample_rate=cfg.SAMPLE_RATE, device=device,
    )

    # ── Step 1: Build single-word GT banks (idempotent) ──────────────────
    print("[exp3] === Step 1: Build single-word GT banks ===")
    build_word_gt_bank_full(unique_words, cfg.INSTRUMENTS)

    # ── Step 2: Single LLM init per word × instrument × dry clip ─────────
    print("\n[exp3] === Step 2: Single LLM init ===")
    for word in unique_words:
        for instrument in cfg.INSTRUMENTS:
            dry = _dry_paths(instrument)
            if not dry:
                continue
            out_dir = os.path.join(_EXP3_LLM_DIR, word, instrument)
            # Idempotent: skip if WAVs already exist
            if os.path.isdir(out_dir) and any(f.endswith(".wav") for f in os.listdir(out_dir)):
                print(f"[exp3] SKIP LLM {word}/{instrument}: exists")
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f"[exp3] RUN  LLM {word}/{instrument}")

            for dry_idx, dry_path in enumerate(dry):
                audio_np, sr = sf.read(dry_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != cfg.SAMPLE_RATE:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
                audio_t = torch.from_numpy(audio_np).float().to(device)
                audio_t = _ensure_bct(audio_t)

                # Single LLM call: init only, no loss function → returns immediately

                llm_config = Config(
                    prompt=PromptFactory.LLM_PARAMETER_INITIALIZATION_PROMPT_DASP(
                        fx_chain=fx_chain,
                        instruction=f"Make this {instrument} sound {word}.",
                        effects=effects,
                    ),
                    initialization_method=ParameterInitializationMethod.LLM,
                    loss_function=None,
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                )
                params_tensor, params_dict, _, _, _ = parameter_engine.get_params(audio_t, llm_config)

                name = instrument.replace(" ", "_")

                # Render audio through FX chain and save
                output_audio = fx_chain(audio_t, params_tensor) # No SIGMOID
                output_np = output_audio.squeeze().detach().cpu().numpy()
                wav_path = os.path.join(out_dir, f"{name}_llm.wav")
                sf.write(wav_path, output_np, cfg.SAMPLE_RATE)
                with open(os.path.join(out_dir, f"{name}_llm_params.json"), "w") as f:
                    json.dump(params_dict, f, indent=2)

            print(f"[exp3] LLM {word}/{instrument}: {len(dry)} files saved")

    # ── Step 3: Single CLAP forward-loss per word × instrument × dry clip
    print("\n[exp3] === Step 3: CLAP forward-loss ===")
    for word in tqdm.tqdm(unique_words):
        for instrument in cfg.INSTRUMENTS:
            dry = _dry_paths(instrument)
            if not dry:
                continue
            out_dir = os.path.join(_EXP3_CLAP_DIR, word, instrument)
            if os.path.isdir(out_dir) and any(f.endswith(".wav") for f in os.listdir(out_dir)):
                print(f"[exp3] SKIP CLAP {word}/{instrument}: exists")
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f"[exp3] RUN  CLAP {word}/{instrument}")

            for dry_idx, dry_path in enumerate(dry):
                audio_np, sr = sf.read(dry_path)
                if audio_np.ndim > 1:
                    audio_np = audio_np.mean(axis=1)
                if sr != cfg.SAMPLE_RATE:
                    audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
                audio_t = torch.from_numpy(audio_np).float().to(device)
                audio_t = _ensure_bct(audio_t)

                clap_config = Config(
                    initialization_method=ParameterInitializationMethod.RANDOM,
                    loss_function=LossFunction.SEMANTIC_SIMILARITY_LOSS,
                    llmclient=llm_client,
                    fx_chain=fx_chain,
                    embedding=clap,
                    device=device,
                    num_iterations=n_clap_calls,
                    text_target=f"This sound is {word}.",
                    optimization_method=OptimizationMethod.GRADIENT_DESCENT,
                )
                # fxsearcher saves best.wav to its own subdir;
                # also copy the result into our flat output dir
                final_params, _, result_audio, history, audios = parameter_engine.get_params(audio_t, clap_config)
                if result_audio is not None:
                    result_np = result_audio.squeeze().detach().cpu().numpy()
                    wav_path = os.path.join(out_dir, f"{dry_idx}_clap.wav")
                    sf.write(wav_path, result_np, cfg.SAMPLE_RATE)

            print(f"[exp3] CLAP {word}/{instrument}: {len(dry)} files saved")

    # ── Step 4: Extract timbre features & compute diffs ──────────────────
    print("\n[exp3] === Step 4: Extract features ===")

    # source_label → { word → feature_diffs array (n_samples, 19) }
    all_diffs: Dict[str, Dict[str, np.ndarray]] = {
        SOURCE_GT: {},
        SOURCE_LLM: {},
        SOURCE_CLAP: {},
    }

    for word in tqdm.tqdm(unique_words):
        print(f"Analyzing features for {word}")
        for instrument in cfg.INSTRUMENTS:
            dry = _dry_paths(instrument)
            if not dry:
                continue

            # GT
            gt_wavs = _word_gt_paths(word, instrument)
            if gt_wavs:
                d = _extract_feature_diffs(gt_wavs, dry)
                all_diffs[SOURCE_GT].setdefault(word, []).append(d)

            # LLM
            llm_dir = os.path.join(_EXP3_LLM_DIR, word, instrument)
            llm_wavs = sorted(glob.glob(os.path.join(llm_dir, "*.wav")))
            if llm_wavs:
                d = _extract_feature_diffs(llm_wavs, dry)
                all_diffs[SOURCE_LLM].setdefault(word, []).append(d)

            # CLAP
            clap_dir = os.path.join(_EXP3_CLAP_DIR, word, instrument)
            clap_wavs = sorted(glob.glob(os.path.join(clap_dir, "*.wav")))
            if clap_wavs:
                d = _extract_feature_diffs(clap_wavs, dry)
                all_diffs[SOURCE_CLAP].setdefault(word, []).append(d)

    # Concatenate across instruments
    for source in all_diffs:
        for word in list(all_diffs[source]):
            all_diffs[source][word] = np.concatenate(all_diffs[source][word], axis=0)

    # ── Step 5: PCA + plot ───────────────────────────────────────────────
    print("\n[exp3] === Step 5: PCA + visualisation ===")

    results: Dict = {
        "feature_names": TIMBRE_FEATURE_NAMES,
        "words": unique_words,
        "sources": {},
    }

    pca_socialfx = None
    for source in [SOURCE_GT, SOURCE_LLM, SOURCE_CLAP]:
        word_diffs = all_diffs[source]
        if not word_diffs:
            print(f"[exp3] SKIP plot for {source}: no data")
            continue

        if pca_socialfx is not None: # also plot the pca without socialfx
            alone = source + "_alone"
            results["sources"][alone], _ = _pca_and_plot(alone, word_diffs, unique_words, None)

        results["sources"][source], pca = _pca_and_plot(source, word_diffs, unique_words, pca_socialfx)

        if source == SOURCE_GT:
            pca_socialfx = pca

    # Save JSON results
    os.makedirs(_EXP3_RESULTS_DIR, exist_ok=True)
    out_json = os.path.join(_EXP3_RESULTS_DIR, "exp3_pca_results.json")
    diff_json = os.path.join(_EXP3_RESULTS_DIR, "exp3_pca_data.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(diff_json, "w") as f:
        json.dump(all_diffs, f, indent=2, default=str)
    print(f"\n[exp3] Results saved to {out_json}")

    return results


# ── PCA + plotting ───────────────────────────────────────────────────────────

def _pca_and_plot(
    source_label: str,
    word_diffs: Dict[str, np.ndarray],
    all_words: List[str],
    pca_socialfx = None
) -> Dict:
    """Fit PCA on feature-diffs, produce scatter + centroid plot, return stats."""
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Stack all diffs into one matrix for PCA fit
    labels: List[str] = []
    X_parts: List[np.ndarray] = []
    for word in all_words:
        if word in word_diffs:
            X_parts.append(word_diffs[word])
            labels.extend([word] * len(word_diffs[word]))
    X = np.concatenate(X_parts, axis=0)  # (N_total, 19)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # fit on all samples

    if pca_socialfx is None:
        pca = PCA(min(X_scaled.shape[1], X_scaled.shape[0]))
        Z = pca.fit_transform(X_scaled)  # (N_total, n_components)
    else:
        pca = pca_socialfx
        Z = pca.transform(X_scaled)


    # Feature correlations with PC1 and PC2
    loadings = pca.components_  # (n_components, 19)
    feature_names = TIMBRE_FEATURE_NAMES
    pc1_corr = {fn: float(loadings[0, i]) for i, fn in enumerate(feature_names)}
    pc2_corr = {fn: float(loadings[1, i]) for i, fn in enumerate(feature_names)} if loadings.shape[0] > 1 else {}

    explained = pca.explained_variance_ratio_.tolist()

    # Per-word centroids and confidence (1/variance)
    word_stats: Dict[str, Dict] = {}
    for word in all_words:
        if word not in word_diffs:
            continue
        mask = np.array([l == word for l in labels])
        pts = Z[mask, :2]
        centroid = pts.mean(axis=0).tolist()
        variance = float(np.mean(np.var(pts, axis=0))) + 1e-10
        confidence = 1.0 / variance
        word_stats[word] = {
            "centroid_pc1": centroid[0],
            "centroid_pc2": centroid[1] if len(centroid) > 1 else 0.0,
            "confidence": confidence,
            "n_samples": int(pts.shape[0]),
        }

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.get_cmap("tab20", len(all_words))
    word_to_color = {w: cmap(i) for i, w in enumerate(all_words)}

    # Scatter per-sample points
    for word in all_words:
        if word not in word_diffs:
            continue
        mask = np.array([l == word for l in labels])
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[word_to_color[word]],
            alpha=0.3, s=15,
            label=word,
        )

    # Centroids with word labels, sized by confidence
    max_conf = max(ws["confidence"] for ws in word_stats.values()) if word_stats else 1.0
    for word, ws in word_stats.items():
        rel_size = ws["confidence"] / max_conf
        fontsize = 9 + 10 * rel_size
        ax.annotate(
            word,
            (ws["centroid_pc1"], ws["centroid_pc2"]),
            fontsize=fontsize,
            fontweight="bold",
            ha="center", va="center",
            color=word_to_color.get(word, "black"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.7),
        )

    ax.set_xlabel(f"PC 1 ({explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({explained[1]*100:.1f}% variance)" if len(explained) > 1 else "PC 2")
    ax.set_title(f"Timbre PCA — {source_label}")
    ax.legend(loc="upper right", fontsize=7, ncol=2, markerscale=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(_EXP3_RESULTS_DIR,f"pca_{source_label.replace(' ', '_').lower()}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[exp3] Saved plot: {plot_path}")

    return {
        "explained_variance": explained,
        "pc1_correlations": pc1_corr,
        "pc2_correlations": pc2_corr,
        "word_stats": word_stats,
        "plot_path": plot_path,
    }, pca