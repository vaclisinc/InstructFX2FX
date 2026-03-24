from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
import torch
import soundfile as sf
from scipy.spatial.distance import cosine as cosine_dist

from .metric import Metric
from embeddings.clap import CLAPWrapper


_clap_wrapper: CLAPWrapper | None = None


def _get_clap() -> CLAPWrapper:
    """Lazy-load a shared CLAPWrapper singleton."""
    global _clap_wrapper
    if _clap_wrapper is None:
        _clap_wrapper = CLAPWrapper()
    return _clap_wrapper


def _load_audio_tensor(audio_path: str, target_sr: int = 48000) -> torch.Tensor:
    """Load a WAV file and return a [1, 1, T] tensor for CLAPWrapper."""
    import librosa

    wav, sr = sf.read(audio_path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]


def _array_to_tensor(audio: np.ndarray, sr: int, target_sr: int = 48000) -> torch.Tensor:
    """Convert a numpy waveform to a [1, 1, T] tensor, resampling if needed."""
    import librosa
    if isinstance(audio, torch.Tensor):
        return audio.float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    if audio.ndim > 1:
        audio = audio.mean(axis=0) if audio.shape[0] < audio.shape[1] else audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)


def compute_clap_score(audio_path: str, text_prompt: str) -> float:
    """
    CLAP cosine similarity S(audio, text).

    Reference Text2FX numbers:
      - Text2FX   ≈ 0.527
      - FxSearcher≈ 0.447
      - LLM2Fx    ≈ 0.232 (speech)
    """
    clap = _get_clap()
    audio_t = _load_audio_tensor(audio_path, target_sr=clap.sample_rate)
    a = clap.get_audio_embedding(audio_t).detach().cpu().numpy().flatten()
    t = clap.get_text_embedding(f"this sound is {text_prompt}").detach().cpu().numpy().flatten()
    return float(1.0 - cosine_dist(a, t))


def compute_clap_score_from_array(
    audio: np.ndarray, sr: int, text_prompt: str
) -> float:
    """
    CLAP score from in-memory audio.

    Args:
        audio: (samples,) or (channels, samples)
        sr: sample rate
    """
    clap = _get_clap()
    audio_t = _array_to_tensor(audio, sr, target_sr=clap.sample_rate)
    a = clap.get_audio_embedding(audio_t).detach().cpu().numpy().flatten()
    t = clap.get_text_embedding(f"this sound is {text_prompt}").detach().cpu().numpy().flatten()
    return float(1.0 - cosine_dist(a, t))


def compute_guided_clap_score(
    audio_path: str,
    target_prompt: str,
    guide_prompt: str = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound",
) -> Tuple[float, float, float]:
    """
    Guided CLAP score used in FxSearcher / Text2FX:

      S_final = S(audio, target_prompt) - S(audio, guide_prompt)

    Returns:
      (S_target, S_guide, S_final)
    """
    clap = _get_clap()
    audio_t = _load_audio_tensor(audio_path, target_sr=clap.sample_rate)
    a = clap.get_audio_embedding(audio_t).detach().cpu().numpy().flatten()
    tt = clap.get_text_embedding(f"this sound is {target_prompt}").detach().cpu().numpy().flatten()
    tg = clap.get_text_embedding(guide_prompt).detach().cpu().numpy().flatten()
    st = float(1.0 - cosine_dist(a, tt))
    sg = float(1.0 - cosine_dist(a, tg))
    return st, sg, st - sg

def compute_guided_clap_score_from_array(
    audio: np.ndarray,
    sr: int,
    target_prompt: str,
    guide_prompt: str = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound",
) -> Tuple[float, float, float]:
    """
    Guided CLAP score from in-memory audio.

    Returns:
      (S_target, S_guide, S_final)
    """
    clap = _get_clap()
    audio_t = _array_to_tensor(audio, sr, target_sr=clap.sample_rate)
    a = clap.get_audio_embedding(audio_t).detach().cpu().numpy().flatten()
    tt = clap.get_text_embedding(f"this sound is {target_prompt}").detach().cpu().numpy().flatten()
    tg = clap.get_text_embedding(guide_prompt).detach().cpu().numpy().flatten()
    st = float(1.0 - cosine_dist(a, tt))
    sg = float(1.0 - cosine_dist(a, tg))
    return st, sg, st - sg

@dataclass
class CLAPSimilarityScore(Metric):
    """
    CLAP text–audio similarity metric (higher is better).
    """

    sr: int = 44100

    def compute(
        self,
        original_audio: Any,
        text_target: str,
    ) -> float:
        if not text_target:
            return float("nan")
        if isinstance(original_audio, str):
            audio = _load_audio_tensor(original_audio, target_sr=self.sr).squeeze(0).cpu().numpy()
        else:
            audio = np.asarray(original_audio)
        return compute_clap_score_from_array(audio, self.sr, text_target)


@dataclass
class GuidedCLAPSimilarityScore(Metric):
    """
    Guided CLAP (target - guide). Higher is better.
    """

    sr: int = 44100
    guide_prompt: str = (
        "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"
    )

    def compute(
        self,
        original_audio: Any,
        text_target: str,
        guide_prompt: str | None = None,
    ) -> float:
        if not text_target:
            return float("nan")

        if isinstance(original_audio, str):
            audio = _load_audio_tensor(original_audio, target_sr=self.sr).squeeze(0).cpu().numpy()
        else:
            audio = np.asarray(original_audio)

        if audio.ndim == 2:
            audio = audio.mean(axis=0)

        gp = guide_prompt if guide_prompt is not None else self.guide_prompt
        st, sg, sf = compute_guided_clap_score_from_array(audio, self.sr, text_target, gp)
        return sf


__all__ = [
    "compute_clap_score",
    "compute_clap_score_from_array",
    "compute_guided_clap_score",
    "CLAPSimilarityScore",
    "GuidedCLAPSimilarityScore",
]
