from __future__ import annotations

import numpy as np

from .dsp_features import extract_features_batch, extract_dsp_features_from_array


def compute_mmd(X: np.ndarray, Y: np.ndarray, sigma: float | None = None) -> float:
    """
    Gaussian-kernel MMD between two feature matrices.

    - Jointly standardises features over X ∪ Y
    - If sigma is None, uses the median pairwise distance heuristic
    - Returns square-rooted unbiased estimator (LLM2Fx-style)
    """
    from scipy.spatial.distance import pdist, cdist

    combined = np.vstack([X, Y])
    mu, sd = combined.mean(0), combined.std(0) + 1e-8
    Xn, Yn = (X - mu) / sd, (Y - mu) / sd

    if sigma is None:
        sq = pdist(np.vstack([Xn, Yn]), "sqeuclidean")
        sigma_sq = max(float(np.median(sq)) if len(sq) else 1.0, 1e-8)
    else:
        sigma_sq = sigma**2

    K_xx = np.exp(-cdist(Xn, Xn, "sqeuclidean") / (2 * sigma_sq))
    K_yy = np.exp(-cdist(Yn, Yn, "sqeuclidean") / (2 * sigma_sq))
    K_xy = np.exp(-cdist(Xn, Yn, "sqeuclidean") / (2 * sigma_sq))

    n, m = len(Xn), len(Yn)
    np.fill_diagonal(K_xx, 0.0)
    np.fill_diagonal(K_yy, 0.0)

    mmd_sq = (
        K_xx.sum() / max(n * (n - 1), 1)
        - 2.0 * K_xy.sum() / max(n * m, 1)
        + K_yy.sum() / max(m * (m - 1), 1)
    )
    return float(np.sqrt(max(mmd_sq, 0.0)))


def cal_mmd_score(matrix_gt: np.ndarray, matrix_ours: np.ndarray) -> float:
    """
    MMD score between ground-truth and our feature matrices.

    Args:
        matrix_gt:   (n, D) array of DSP feature vectors (e.g. from GT audio).
        matrix_ours: (m, D) array of DSP feature vectors (e.g. from our output).

    Returns:
        Single float. Lower = closer distributions; 0 = identical.
    """
    return compute_mmd(matrix_gt, matrix_ours)


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


__all__ = ["compute_mmd", "cal_mmd_score", "run_mmd_evaluation"]

