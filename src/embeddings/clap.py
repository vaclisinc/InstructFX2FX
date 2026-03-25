import os
import io
import contextlib
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import laion_clap
from abc import ABC, abstractmethod


class EmbeddingWrapper(ABC):
    @abstractmethod
    def get_audio_embedding(self, audio, enable_grad=False):
        pass

    @abstractmethod
    def get_text_embedding(self, text):
        pass

# ========== CLAP Model ==========

class CLAPWrapper(EmbeddingWrapper):
    """CLAP model for encoding audio and text using laion_clap."""

    _MODEL_CACHE: dict[tuple[str, str], laion_clap.CLAP_Module] = {}

    @staticmethod
    def _cache_key(device: str, ckpt_path: str | None) -> tuple[str, str]:
        ckpt = "" if ckpt_path is None else str(Path(ckpt_path).expanduser().resolve())
        return str(device), ckpt

    def __init__(self, device='cpu', ckpt_path: str | None = None):
        self.device = device
        key = self._cache_key(device, ckpt_path)

        cached_model = self._MODEL_CACHE.get(key)
        if cached_model is not None:
            self.model = cached_model
            self.sample_rate = 48000
            return

        self.model = laion_clap.CLAP_Module(enable_fusion=False, device=device)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if ckpt_path:
                ckpt = Path(ckpt_path).expanduser().resolve()
                # if ckpt.exists() and ckpt.is_file():
                #     self.model.load_ckpt(ckpt=str(ckpt), verbose=False)
                # else:
                #     # Treat path as a cache directory (existing or to-be-created).
                #     ckpt.mkdir(parents=True, exist_ok=True)
                #     os.environ["TORCH_HOME"] = str(ckpt)
                #     self.model.load_ckpt(verbose=False)
            else:
                self.model.load_ckpt(verbose=False)  # falls back to download/cache behavior

        # Freeze CLAP parameters - we only optimize through it, not the model itself

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.sample_rate = 48000
        # self._MODEL_CACHE[key] = self.model

    def get_audio_embedding(self, audio, enable_grad=False):
        """audio: [B, C, T] → embedding: [B, D]

        audio should be torch tensor in shape [B, C, T] where:
        - B is batch size
        - C is channels (1 or 2)
        - T is time samples
        """
        # Convert to mono if stereo
        if audio.shape[1] == 2:
            audio = audio.mean(dim=1, keepdim=False)
        elif audio.shape[1] == 1:
            audio = audio.squeeze(1)

        # Resample if needed (laion_clap expects 48kHz)
        if audio.shape[-1] > self.sample_rate:  # 10 seconds at 48kHz
            audio = audio[..., :self.sample_rate]

        # Use laion_clap's get_audio_embedding_from_data with use_tensor=True
        # This keeps everything as tensors and preserves gradients
        result = self.model.get_audio_embedding_from_data(x=audio, use_tensor=True)

        # Ensure it's a torch tensor
        if not isinstance(result, torch.Tensor):
            result = torch.from_numpy(result).to(self.device)

        return result

    def get_text_embedding(self, text):
        """text: str or list → embedding: [B, D]
        Always returns torch tensor on the correct device.
        """
        if isinstance(text, str):
            text = [text]

        # laion_clap returns numpy array, convert to torch tensor
        emb = self.model.get_text_embedding(text)

        # Convert to tensor and move to device
        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).to(self.device)
        elif isinstance(emb, torch.Tensor):
            emb = emb.to(self.device)

        return emb

    def __call__(self, value, enable_grad=False):
        if value is not None and isinstance(value, torch.Tensor):
            return self.get_audio_embedding(value, enable_grad)
        elif value is not None and isinstance(value, str):
            return self.get_text_embedding(value)
        else:
            raise ValueError("Must provide either audio or text input to CLAPWrapper.")