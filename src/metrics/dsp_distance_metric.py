from __future__ import annotations

from typing import Tuple, Dict

import numpy as np

from .dsp_features import extract_features_batch


def compute_dsp_feature_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Euclidean distance between mean feature vectors.

    Reference numbers (from LLM2Fx paper):
      - LLM2Fx-Tools ~ 8.29
      - No FX       ~ 14.82
    """
    return float(np.linalg.norm(gt.mean(0) - pred.mean(0)))


def run_dsp_distance_evaluation(gt_dir: str, pred_dir: str, sr: int = 22050) -> float:
    """
    Compute DSP feature distance between two folders of audio.

    - Extract 35-D DSP features for each file in gt_dir and pred_dir.
    - Average features per set.
    - Return Euclidean distance between the two mean vectors.
    """
    print("\n==============================================================")
    print("  PWFX — DSP Feature Distance")
    print("==============================================================")

    print("\n[1/2] Extracting DSP features...")
    gt_f, _ = extract_features_batch(gt_dir, sr=sr)
    pr_f, _ = extract_features_batch(pred_dir, sr=sr)
    print(f"  GT:   {gt_f.shape[0]} files x {gt_f.shape[1]} features")
    print(f"  Pred: {pr_f.shape[0]} files x {pr_f.shape[1]} features")

    print("\n[2/2] Computing DSP feature distance...")
    dist = compute_dsp_feature_distance(gt_f, pr_f)
    print(f"  DSP feature distance = {dist:.4f}")
    print("  Lower is better; 0 would mean identical average features.")

    return dist


__all__ = ["compute_dsp_feature_distance", "run_dsp_distance_evaluation"]

