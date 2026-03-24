from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

# 35-D feature definition (shared with LLM2Fx paper)
FEATURE_NAMES = [
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_flatness_mean",
    "spectral_flatness_std",
    "spectral_contrast_0",
    "spectral_contrast_1",
    "spectral_contrast_2",
    "spectral_contrast_3",
    "spectral_contrast_4",
    "spectral_contrast_5",
    "spectral_contrast_6",
    "mfcc_0",
    "mfcc_1",
    "mfcc_2",
    "mfcc_3",
    "mfcc_4",
    "mfcc_5",
    "mfcc_6",
    "mfcc_7",
    "mfcc_8",
    "mfcc_9",
    "mfcc_10",
    "mfcc_11",
    "mfcc_12",
    "rms_mean",
    "rms_std",
    "zcr_mean",
    "zcr_std",
    "crest_factor",
    "brightness",
    "loudness_db",
]


def extract_dsp_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    """
    Extract 35 DSP features from a single audio file on disk.

    This matches the feature definition used in the original LLM2Fx code
    so that metrics are comparable across scripts.
    """
    import librosa

    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    if np.max(np.abs(y)) < 1e-8:
        return np.zeros(35, dtype=np.float32)

    feats: list[float] = []

    v = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats.extend([np.mean(v), np.std(v)])

    v = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats.extend([np.mean(v), np.std(v)])

    v = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats.extend([np.mean(v), np.std(v)])

    v = librosa.feature.spectral_flatness(y=y)[0]
    feats.extend([np.mean(v), np.std(v)])

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats.append(float(np.mean(contrast[i])))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats.append(float(np.mean(mfcc[i])))

    rms = librosa.feature.rms(y=y)[0]
    feats.extend([float(np.mean(rms)), float(np.std(rms))])

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats.extend([float(np.mean(zcr)), float(np.std(zcr))])

    rms_val = float(np.sqrt(np.mean(y**2)))
    feats.append(float(np.max(np.abs(y)) / (rms_val + 1e-8)))

    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    feats.append(float(np.sum(S[freqs >= 1500, :]) / (np.sum(S) + 1e-8)))

    feats.append(20.0 * float(np.log10(rms_val + 1e-8)))

    return np.asarray(feats, dtype=np.float32)


def extract_dsp_features_from_array(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """
    Extract 35 DSP features from an in-memory audio array.

    Args:
        audio: (samples,) or (channels, samples); float in [-1, 1].
    """
    import librosa

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    if np.max(np.abs(audio)) < 1e-8:
        return np.zeros(35, dtype=np.float32)

    y = audio.astype(np.float64)
    feats: list[float] = []

    v = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats.extend([np.mean(v), np.std(v)])

    v = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    feats.extend([np.mean(v), np.std(v)])

    v = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats.extend([np.mean(v), np.std(v)])

    v = librosa.feature.spectral_flatness(y=y)[0]
    feats.extend([np.mean(v), np.std(v)])

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats.append(float(np.mean(contrast[i])))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        feats.append(float(np.mean(mfcc[i])))

    rms = librosa.feature.rms(y=y)[0]
    feats.extend([float(np.mean(rms)), float(np.std(rms))])

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats.extend([float(np.mean(zcr)), float(np.std(zcr))])

    rms_val = float(np.sqrt(np.mean(y**2)))
    feats.append(float(np.max(np.abs(y)) / (rms_val + 1e-8)))

    S = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=sr)
    feats.append(float(np.sum(S[freqs >= 1500, :]) / (np.sum(S) + 1e-8)))

    feats.append(20.0 * float(np.log10(rms_val + 1e-8)))

    return np.asarray(feats, dtype=np.float32)


def extract_features_batch(audio_dir: str, sr: int = 22050) -> Tuple[np.ndarray, List[str]]:
    """
    Extract DSP features from all audio files in a directory.

    Returns:
        features: (n_files, 35)
        filenames: list of filenames (sorted)
    """
    from tqdm import tqdm

    audio_path = Path(audio_dir)
    if not audio_path.is_dir():
        raise FileNotFoundError(
            f"Directory not found: {audio_dir!r}. "
            "Generate test data first: python scripts/generate_test_data.py"
        )

    exts = {".wav", ".mp3", ".flac", ".ogg"}
    files = sorted([f for f in audio_path.iterdir() if f.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No audio files in {audio_dir}")

    features, names = [], []
    for f in tqdm(files, desc=f"  Features [{audio_path.name}]"):
        features.append(extract_dsp_features(str(f), sr=sr))
        names.append(f.name)

    return np.asarray(features), names


# ---------------------------------------------------------------------------
# Timbre PCA features (19-D)
#
# Two principal components from timbre perception research:
#   PC 1: Irregularity, Irregularity_p, Kurtosis_p, Skew_s, Irregularity_h,
#          Kurtosis_h, Skew_p, Std, RMS, Skew_h, Kurtosis_s, Var
#   PC 2: Centroid_p, Centroid_h, Rolloff_s, Std_h, Std_p, Centroid_s, Slope_s
#
# Subscripts:  s = magnitude spectrum,  p = peak spectrum,
#              h = harmonic peak spectrum,  (none) = temporal / abstract
# Spectral irregularity uses the method described by Krimphoff et al.
# ---------------------------------------------------------------------------

TIMBRE_FEATURE_NAMES = [
    # Temporal / abstract (no subscript)
    "irregularity",
    "std",
    "rms",
    "var",
    # Magnitude spectrum (_s)
    "skew_s",
    "kurtosis_s",
    "centroid_s",
    "rolloff_s",
    "slope_s",
    # Peak spectrum (_p)
    "irregularity_p",
    "kurtosis_p",
    "skew_p",
    "centroid_p",
    "std_p",
    # Harmonic peak spectrum (_h)
    "irregularity_h",
    "kurtosis_h",
    "skew_h",
    "centroid_h",
    "std_h",
]


# ── Spectral helpers ─────────────────────────────────────────────────────────

def _irregularity_krimphoff(amplitudes: np.ndarray) -> float:
    """Spectral irregularity (Krimphoff et al.):
    sum of squared differences between adjacent amplitude values."""
    if len(amplitudes) < 2:
        return 0.0
    return float(np.sum(np.diff(amplitudes) ** 2))


def _weighted_centroid(amps: np.ndarray, freqs: np.ndarray) -> float:
    total = np.sum(amps)
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * amps) / total)


def _weighted_spread(amps: np.ndarray, freqs: np.ndarray) -> float:
    """Amplitude-weighted spectral standard deviation (spread)."""
    centroid = _weighted_centroid(amps, freqs)
    total = np.sum(amps)
    if total < 1e-10:
        return 0.0
    return float(np.sqrt(np.sum(amps * (freqs - centroid) ** 2) / total))


def _weighted_skewness(amps: np.ndarray, freqs: np.ndarray) -> float:
    centroid = _weighted_centroid(amps, freqs)
    spread = _weighted_spread(amps, freqs)
    total = np.sum(amps)
    if total < 1e-10 or spread < 1e-10:
        return 0.0
    return float(np.sum(amps * ((freqs - centroid) / spread) ** 3) / total)


def _weighted_kurtosis(amps: np.ndarray, freqs: np.ndarray) -> float:
    centroid = _weighted_centroid(amps, freqs)
    spread = _weighted_spread(amps, freqs)
    total = np.sum(amps)
    if total < 1e-10 or spread < 1e-10:
        return 0.0
    return float(np.sum(amps * ((freqs - centroid) / spread) ** 4) / total)


def _spectral_rolloff_custom(amps: np.ndarray, freqs: np.ndarray, threshold: float = 0.85) -> float:
    """Frequency below which *threshold* fraction of spectral energy lies."""
    total = np.sum(amps)
    if total < 1e-10:
        return 0.0
    cumsum = np.cumsum(amps)
    idx = int(np.searchsorted(cumsum, threshold * total))
    return float(freqs[min(idx, len(freqs) - 1)])


def _spectral_slope(amps: np.ndarray, freqs: np.ndarray) -> float:
    """Linear regression slope of amplitude vs. frequency."""
    if len(freqs) < 2:
        return 0.0
    f_mean = np.mean(freqs)
    a_mean = np.mean(amps)
    denom = np.sum((freqs - f_mean) ** 2)
    if abs(denom) < 1e-10:
        return 0.0
    return float(np.sum((freqs - f_mean) * (amps - a_mean)) / denom)


def _find_spectral_peaks(magnitude: np.ndarray, freqs: np.ndarray):
    """Return (peak_amps, peak_freqs) from the magnitude spectrum."""
    from scipy.signal import find_peaks as _find_peaks

    indices, _ = _find_peaks(magnitude, height=0)
    if len(indices) == 0:
        return magnitude, freqs  # fallback: treat whole spectrum as peaks
    return magnitude[indices], freqs[indices]


def _find_harmonic_peaks(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int = 20,
    tolerance_hz: float = 30.0,
):
    """Return (harmonic_amps, harmonic_freqs) at multiples of f0."""
    if f0 <= 0 or len(freqs) == 0:
        return np.array([]), np.array([])
    target_freqs = f0 * np.arange(1, n_harmonics + 1)
    target_freqs = target_freqs[target_freqs < freqs[-1]]
    amps, matched_freqs = [], []
    for hf in target_freqs:
        idx = int(np.argmin(np.abs(freqs - hf)))
        if abs(freqs[idx] - hf) <= tolerance_hz:
            amps.append(magnitude[idx])
            matched_freqs.append(freqs[idx])
    if not amps:
        return np.array([]), np.array([])
    return np.array(amps), np.array(matched_freqs)


def _estimate_f0(y: np.ndarray, sr: int) -> float:
    """Robust f0 estimate via librosa.pyin (median of voiced frames)."""
    import librosa

    f0_arr, voiced, _ = librosa.pyin(
        y, fmin=50, fmax=4000, sr=sr, frame_length=2048
    )
    voiced_f0 = f0_arr[voiced]
    if len(voiced_f0) == 0:
        return 0.0
    return float(np.median(voiced_f0))


# ── Main extraction functions ────────────────────────────────────────────────

def _compute_timbre_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Compute the 19-D timbre feature vector from an audio signal array."""
    n_features = len(TIMBRE_FEATURE_NAMES)
    if np.max(np.abs(y)) < 1e-8:
        return np.zeros(n_features, dtype=np.float32)

    # ── Temporal / abstract (no subscript) ────────────────────────────────
    sig_rms = float(np.sqrt(np.mean(y ** 2)))
    sig_std = float(np.std(y))
    sig_var = float(np.var(y))

    # Magnitude spectrum (average over frames)
    S = np.abs(np.fft.rfft(y))
    import librosa
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr).astype(np.float64)

    # Irregularity on the full magnitude spectrum (Krimphoff)
    irregularity = _irregularity_krimphoff(S)

    # ── Magnitude spectrum features (_s) ──────────────────────────────────
    skew_s = _weighted_skewness(S, freqs)
    kurtosis_s = _weighted_kurtosis(S, freqs)
    centroid_s = _weighted_centroid(S, freqs)
    rolloff_s = _spectral_rolloff_custom(S, freqs)
    slope_s = _spectral_slope(S, freqs)

    # ── Peak spectrum features (_p) ───────────────────────────────────────
    peak_amps, peak_freqs = _find_spectral_peaks(S, freqs)
    irregularity_p = _irregularity_krimphoff(peak_amps)
    kurtosis_p = _weighted_kurtosis(peak_amps, peak_freqs)
    skew_p = _weighted_skewness(peak_amps, peak_freqs)
    centroid_p = _weighted_centroid(peak_amps, peak_freqs)
    std_p = _weighted_spread(peak_amps, peak_freqs)

    # ── Harmonic peak spectrum features (_h) ──────────────────────────────
    f0 = _estimate_f0(y, sr)
    harm_amps, harm_freqs = _find_harmonic_peaks(S, freqs, f0)

    if len(harm_amps) >= 2:
        irregularity_h = _irregularity_krimphoff(harm_amps)
        kurtosis_h = _weighted_kurtosis(harm_amps, harm_freqs)
        skew_h = _weighted_skewness(harm_amps, harm_freqs)
        centroid_h = _weighted_centroid(harm_amps, harm_freqs)
        std_h = _weighted_spread(harm_amps, harm_freqs)
    else:
        irregularity_h = kurtosis_h = skew_h = centroid_h = std_h = 0.0

    return np.asarray(
        [
            # No subscript
            irregularity, sig_std, sig_rms, sig_var,
            # _s
            skew_s, kurtosis_s, centroid_s, rolloff_s, slope_s,
            # _p
            irregularity_p, kurtosis_p, skew_p, centroid_p, std_p,
            # _h
            irregularity_h, kurtosis_h, skew_h, centroid_h, std_h,
        ],
        dtype=np.float32,
    )


def extract_timbre_features(audio_path: str, sr: int = 22050) -> np.ndarray:
    """Extract the 19-D timbre PCA feature vector from an audio file on disk."""
    import librosa

    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    return _compute_timbre_features(y, sr)


def extract_timbre_features_from_array(audio: np.ndarray, sr: int = 22050) -> np.ndarray:
    """Extract the 19-D timbre PCA feature vector from an in-memory audio array."""
    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)
    return _compute_timbre_features(audio.astype(np.float64), sr)


def compute_dsp_feature_distance(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    Euclidean distance between mean feature vectors.

    Reference numbers (from LLM2Fx paper):
      - LLM2Fx-Tools ~ 8.29
      - No FX       ~ 14.82 # FIXME: this is complete hallucination: LLM2FX could not include LLM2FX-Tools
    """
    return float(np.linalg.norm(gt.mean(0) - pred.mean(0)))





__all__ = ["compute_dsp_feature_distance", "run_dsp_distance_evaluation"]

class DSPFeatureDistance():
    def compute(gt_dir: str, pred_dir: str, sr: int = 22050) -> float: # FIXME why sr 22050?
        """
        Compute DSP feature distance between two folders of audio.

        - Extract 35-D DSP features for each file in gt_dir and pred_dir.
        - Average features per set.
        - Return Euclidean distance between the two mean vectors.
        """
        print("\n==============================================================")
        print("  DSP Feature Distance")
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