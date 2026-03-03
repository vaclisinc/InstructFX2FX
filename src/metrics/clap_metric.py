from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from .metric import Metric


_clap_model = None


def _get_clap():
    """
    Lazy-load the laion_clap model once.
    """
    global _clap_model
    if _clap_model is None:
        import laion_clap

        _clap_model = laion_clap.CLAP_Module(enable_fusion=False)
        _clap_model.load_ckpt()
    return _clap_model


def compute_clap_score(audio_path: str, text_prompt: str) -> float:
    """
    CLAP cosine similarity S(audio, text).

    Reference Text2FX numbers:
      - Text2FX   ≈ 0.527
      - FxSearcher≈ 0.447
      - LLM2Fx    ≈ 0.232 (speech)
    """
    from scipy.spatial.distance import cosine

    m = _get_clap()
    a = m.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)
    t = m.get_text_embedding([f"this sound is {text_prompt}"], use_tensor=False)
    return float(1.0 - cosine(a.flatten(), t.flatten()))


def compute_clap_score_from_array(
    audio: np.ndarray, sr: int, text_prompt: str
) -> float:
    """
    CLAP score from in-memory audio.

    Args:
        audio: (samples,) or (channels, samples)
        sr: sample rate
    """
    import tempfile
    import soundfile as sf

    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
        sf.write(f.name, audio, sr)
        return compute_clap_score(f.name, text_prompt)


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
    from scipy.spatial.distance import cosine

    m = _get_clap()
    a = m.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False).flatten()
    tt = m.get_text_embedding(
        [f"this sound is {target_prompt}"], use_tensor=False
    ).flatten()
    tg = m.get_text_embedding([guide_prompt], use_tensor=False).flatten()
    st = float(1.0 - cosine(a, tt))
    sg = float(1.0 - cosine(a, tg))
    return st, sg, st - sg


def _prompt_text(prompt: Any) -> str:
    """Get text from Prompt dataclass or plain string."""
    if prompt is None:
        return ""
    if hasattr(prompt, "instruction"):
        return getattr(prompt, "instruction") or ""
    if hasattr(prompt, "text"):
        return getattr(prompt, "text") or ""
    return str(prompt)


@dataclass
class LLM2FxCLAP(Metric):
    """
    CLAP text–audio similarity metric (higher is better).

    Uses laion_clap under the hood but works on in-memory audio arrays
    for easy integration inside experiments.
    """

    sr: int = 48000

    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        text = _prompt_text(prompt)
        if not text:
            return float("nan")
        audio = np.asarray(original_audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        return compute_clap_score_from_array(audio, self.sr, text)


@dataclass
class LLM2FxGuidedCLAP(Metric):
    """
    Guided CLAP (target - guide). Higher is better.
    """

    sr: int = 48000
    guide_prompt: str = (
        "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"
    )

    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        import tempfile
        import soundfile as sf

        text = _prompt_text(prompt)
        if not text:
            return float("nan")

        audio = np.asarray(original_audio)
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            sf.write(f.name, audio, self.sr)
            _, _, s_final = compute_guided_clap_score(
                f.name, text, self.guide_prompt
            )
        return s_final


__all__ = [
    "compute_clap_score",
    "compute_clap_score_from_array",
    "compute_guided_clap_score",
    "LLM2FxCLAP",
    "LLM2FxGuidedCLAP",
]

