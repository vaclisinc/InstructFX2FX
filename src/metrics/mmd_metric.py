from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .dsp_feature_metrics import extract_features_batch, extract_dsp_features_from_array
from .metric import Metric


def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """
    Gaussian-kernel MMD between two feature matrices.

    - Jointly standardises features over X ∪ Y
    - If sigma is None, uses the median pairwise distance heuristic
    - Returns square-rooted unbiased estimator (LLM2Fx-style)
    """
    from scipy.spatial.distance import cdist

    combined = np.vstack([X, Y])
    mu, sd = combined.mean(0), combined.std(0) + 1e-8
    Zn = (combined - mu) / sd

    n, m = len(X), len(Y)

    D = cdist(Zn, Zn, "sqeuclidean")

    if sigma is None:
        off_diag = D[D > 0]
        sigma_sq = max(float(np.median(off_diag)) if len(off_diag) else 1.0, 1e-8)
    else:
        sigma_sq = sigma**2

    K = np.exp(-D / (2 * sigma_sq))
    K_xx = K[:n, :n]
    K_yy = K[n:, n:]
    K_xy = K[:n, n:]

    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    mmd_sq = (
        K_xx.sum() / max(n * (n - 1), 1)
        - 2.0 * K_xy.sum() / max(n * m, 1)
        + K_yy.sum() / max(m * (m - 1), 1)
    )
    return float(np.sqrt(max(mmd_sq, 0.0)))


def run_mmd_evaluation(gt_dir: str, pred_dir: str, sr: int = 22050) -> float:
    """
    Folder-based MMD evaluation convenience helper.

    Extracts DSP features from both folders and returns the MMD value.
    """
    print("\n==============================================================")
    print("  PWFX — MMD Evaluation (DSP features)")
    print("==============================================================")

    print("\n[1/2] Extracting DSP features...")
    gt_f, _ = extract_features_batch(gt_dir, sr=sr)
    pr_f, _ = extract_features_batch(pred_dir, sr=sr)
    print(f"  GT:   {gt_f.shape[0]} files x {gt_f.shape[1]} features")
    print(f"  Pred: {pr_f.shape[0]} files x {pr_f.shape[1]} features")

    print("\n[2/2] Computing MMD...")
    mmd = compute_mmd(gt_f, pr_f)
    print(f"  MMD (DSP features) = {mmd:.4f}")
    print("  Lower is better; 0 means identical feature distributions.")

    return mmd


@dataclass
class AudioFeaturesMMD(Metric):
    """
    LLM2Fx-style MMD over DSP features for a single pair of audio items.

    This mirrors the original LLM2FxMMD class in llm2fx.py, but with
    the DSP feature logic factored out to dsp_features.py.
    """

    sr: int = 22050

    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        gt_feat = extract_dsp_features_from_array(
            np.asarray(target_audio), sr=self.sr
        )
        pred_feat = extract_dsp_features_from_array(
            np.asarray(original_audio), sr=self.sr
        )
        return compute_mmd(gt_feat[np.newaxis, :], pred_feat[np.newaxis, :])


__all__ = ["compute_mmd", "run_mmd_evaluation", "LLM2FxMMD"]
