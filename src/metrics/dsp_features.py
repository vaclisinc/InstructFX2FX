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

