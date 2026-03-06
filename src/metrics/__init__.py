"""
Metrics: DSP features and MMD score.

Simple API:

    from src.metrics import extract_dsp_feature, cal_mmd_score

    # One clip → 35-D feature list
    features = extract_dsp_feature(audio)   # audio = path (str) or array
    # or with sample rate when using array:
    features = extract_dsp_feature(audio_array, sr=22050)

    # Two feature matrices → single MMD score (lower = closer)
    score = cal_mmd_score(matrix_gt, matrix_ours)
"""

from .dsp_features import extract_dsp_feature
from .mmd_metric import cal_mmd_score

__all__ = ["extract_dsp_feature", "cal_mmd_score"]
