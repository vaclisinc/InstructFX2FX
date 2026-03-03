from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union, Sequence

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import ClapModel, ClapProcessor
except Exception as e:
    ClapModel = None
    ClapProcessor = None


@dataclass(frozen=True)
class ClapFxSearcherConfig:
    """
    FxSearcher-style CLAP config.
    - model_name: 'laion/clap-htsat-unfused' is the standard for this framework.
    - target_sr: 48kHz is required by the CLAP processor.
    - max_seconds: 10s truncation is standard to match the training window.
    """
    model_name: str = "laion/clap-htsat-unfused"
    target_sr: int = 48000
    max_seconds: float = 10.0


class CLAPFxSearcherWrapper:
    """
    HuggingFace CLAP wrapper optimized for FxSearcher.
    
    This wrapper explicitly handles:
    1. Robust mono-conversion and 48kHz resampling.
    2. Prompt templating ("this sound is {text}") for better alignment.
    3. Guided scoring (Target Score - Artifact Score).
    """

    def __init__(self, device: str = "cpu", config: Optional[ClapFxSearcherConfig] = None):
        if ClapModel is None or ClapProcessor is None:
            raise RuntimeError(
                "CLAPFxSearcherWrapper requires 'transformers' library."
            )

        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.config = config or ClapFxSearcherConfig()

        self.model = ClapModel.from_pretrained(self.config.model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(self.config.model_name)
        self.model.eval()

        # Disable gradients as this is used for inference/scoring only
        for p in self.model.parameters():
            p.requires_grad = False

    def _to_mono_1d_np(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Standardizes audio to mono float32 1D numpy array."""
        if isinstance(audio, torch.Tensor):
            x = audio.detach().cpu().float()
            if x.dim() == 3: x = x.squeeze(0) # [B,C,T] -> [C,T]
            if x.dim() == 2 and x.size(0) <= 2: x = x.mean(dim=0) # [C,T] -> [T]
            return x.numpy().astype(np.float32)

        x = np.asarray(audio, dtype=np.float32)
        if x.ndim == 2:
            # Handle [C, T] or [T, C]
            axis = 0 if x.shape[0] < x.shape[1] else 1
            x = x.mean(axis=axis)
        return x.flatten()

    def _resample_and_truncate(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Resamples to 48kHz and truncates to max_seconds as required by CLAP."""
        target_sr = self.config.target_sr
        max_len = int(self.config.max_seconds * target_sr)

        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)

        if len(wav) > max_len:
            wav = wav[:max_len]

        return wav.astype(np.float32)

    def get_text_embedding(self, text: Union[str, Sequence[str]]) -> torch.Tensor:
        """
        Returns normalized text embeddings. 
        Prepends 'this sound is ' to match FxSearcher/LAION-CLAP training distribution.
        """
        if isinstance(text, str):
            text_list = [f"this sound is {text}"]
        else:
            text_list = [f"this sound is {t}" for t in text]

        inputs = self.processor(text=text_list, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            z = F.normalize(out, dim=-1)
        return z

    def get_audio_embedding(self, audio: Union[np.ndarray, torch.Tensor], sr: int) -> torch.Tensor:
        """Processes audio and returns a normalized embedding [1, D]."""
        wav = self._to_mono_1d_np(audio)
        wav = self._resample_and_truncate(wav, sr)
        
        inputs = self.processor(audio=wav, sampling_rate=self.config.target_sr, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.get_audio_features(**inputs)
            z = F.normalize(out, dim=-1)
        return z

    def cosine_score(self, audio: Union[np.ndarray, torch.Tensor], sr: int, text: str) -> float:
        """Computes similarity between audio and text prompt."""
        za = self.get_audio_embedding(audio, sr) 
        zt = self.get_text_embedding(text)       
        return float((za * zt).sum(dim=-1).item())

    def guided_cosine_score(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        sr: int,
        positive_text: str,
        negative_text: str = "A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound",
    ) -> float:
        """
        Guided Score = S(audio, target) - S(audio, artifacts).
        This is the primary objective function for the FxSearcher Bayesian Search.
        """
        za = self.get_audio_embedding(audio, sr)
        zpos = self.get_text_embedding(positive_text)
        zneg = self.get_text_embedding(negative_text)
        
        spos = (za * zpos).sum(dim=-1).item()
        sneg = (za * zneg).sum(dim=-1).item()
        return float(spos - sneg)