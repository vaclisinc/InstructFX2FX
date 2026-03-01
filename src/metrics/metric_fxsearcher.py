from __future__ import annotations

import abc
import os
import time
import tempfile
import shutil
from dataclasses import dataclass
from typing import Any, Optional, Union, Tuple

import numpy as np
import torch
import librosa
import soundfile as sf

import whisper
import jiwer
from pesq import pesq
from frechet_audio_distance import FrechetAudioDistance
import pyloudnorm as pyln

from metrics.metric import Metric
from prompts.prompt import Prompt
from embeddings.clap_fxsearcher import CLAPFxSearcherWrapper

@dataclass(frozen=True)
class AudioItem:
    """Carrier so Metric.compute can receive waveform + sr."""
    source: Any
    sr: Optional[int] = None

def _load_waveform_mono(x: Any) -> np.ndarray:
    """Convert input into mono float32 numpy waveform [T]."""
    if isinstance(x, str):
        wav, _sr = librosa.load(x, sr=None, mono=True)
        return wav.astype(np.float32)

    if isinstance(x, torch.Tensor):
        t = x.detach().cpu().float()
        if t.dim() == 1: 
            return t.numpy()
        if t.dim() == 2: 
            return t.mean(dim=0).numpy()
        if t.dim() == 3: 
            return _load_waveform_mono(t.squeeze(0))
        
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1: return a
    if a.ndim == 2: return a.mean(axis=0) if a.shape[0] < a.shape[1] else a.mean(axis=1)
    return a

def _extract_audio_and_sr(obj: Any) -> Tuple[Any, int]:
    """Returns (waveform_or_path, sr)."""
    if isinstance(obj, AudioItem):
        if isinstance(obj.source, str):
            return obj.source, int(sf.info(obj.source).samplerate)
        return obj.source, int(obj.sr)
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[0], int(obj[1])
    if isinstance(obj, str):
        return obj, int(sf.info(obj).samplerate)
    raise ValueError("Audio must be AudioItem, (waveform, sr), or filepath.")

class FxSearcherCLAPScore(Metric):
    """Text-Audio Alignment (Standard CLAP)"""
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.clap = CLAPFxSearcherWrapper(device=device)

    def compute(self, original_audio: Any, target_audio: Any, prompt: Optional[Prompt] = None):
        wav_obj, sr = _extract_audio_and_sr(target_audio)
        wav = _load_waveform_mono(wav_obj)
        return float(self.clap.cosine_score(wav, sr, prompt.instruction))

class FxSearcherGuidedCLAPScore(Metric):
    """Text-Audio Alignment (Guided to avoid artifacts)"""
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.clap = CLAPFxSearcherWrapper(device=device)

    def compute(self, original_audio: Any, target_audio: Any, prompt: Optional[Prompt] = None):
        wav_obj, sr = _extract_audio_and_sr(target_audio)
        wav = _load_waveform_mono(wav_obj)
        return float(self.clap.guided_cosine_score(wav, sr, prompt.instruction))

class FxSearcherWER(Metric):
    """Speech Intelligibility (Word Error Rate) via Whisper-large-v3"""
    def __init__(self, device="cuda"):
        self.model = whisper.load_model("large-v3", device=device)

    def compute(self, original_audio: Any, target_audio: Any, prompt: Optional[Prompt] = None):
        wav_orig, sr_o = _extract_audio_and_sr(original_audio)
        wav_target, sr_t = _extract_audio_and_sr(target_audio)
        
        # Whisper transcribes from path or ndarray
        ref = self.model.transcribe(_load_waveform_mono(wav_orig))["text"]
        hyp = self.model.transcribe(_load_waveform_mono(wav_target))["text"]
        return float(jiwer.wer(ref, hyp))

class FxSearcherPESQ(Metric):
    """Perceptual Speech Quality (WB-PESQ)"""
    def compute(self, original_audio: Any, target_audio: Any, prompt=None):
        wav_o_raw, sr_o = _extract_audio_and_sr(original_audio)
        wav_t_raw, sr_t = _extract_audio_and_sr(target_audio)
        
        # PESQ strictly requires 16kHz
        ref = librosa.resample(_load_waveform_mono(wav_o_raw), orig_sr=sr_o, target_sr=16000)
        deg = librosa.resample(_load_waveform_mono(wav_t_raw), orig_sr=sr_t, target_sr=16000)
        return float(pesq(16000, ref, deg, 'wb'))

class FxSearcherFAD(Metric):
    """Fréchet Audio Distance (Distribution similarity)"""
    def __init__(self, model_name: str = "vggish"):
        self.fad = FrechetAudioDistance(model_name=model_name, use_pca=False, verbose=False)

    def compute(self, original_audio: str, target_audio: str, prompt: Optional[Prompt] = None):
        # original_audio and target_audio should be folder paths for a true FAD
        return float(self.fad.score(original_audio, target_audio))

class FxSearcherIntegratedLUFS(Metric):
    """Loudness (BS.1770)"""
    def compute(self, original_audio: Any, target_audio: Any, prompt: Optional[Prompt] = None):
        wav_obj, sr = _extract_audio_and_sr(target_audio)
        wav = _load_waveform_mono(wav_obj)
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(wav))

class FxSearcherInferenceTime(Metric):
    """Seconds per sample (from metadata)"""
    def compute(self, original_audio: Any, target_audio: Any, prompt: Optional[Prompt] = None):
        if isinstance(target_audio, dict):
            return float(target_audio.get('search_time_seconds', 0.0))
        return 0.0

class AIJudgeQwen(Metric):
    """Qwen2.5-omni-7B: Absolute 1-5 Alignment Rating"""
    def compute(self, target_audio: Any, prompt: Prompt):
        # Implementation: Send audio + prompt to Qwen-Omni
        # Return numeric score 1.0 - 5.0
        return 0.0 

class AIJudgeGemini(Metric):
    """Gemini 2.5 Flash: Pairwise preference (Win Rate)"""
    def compute(self, audio_a: Any, audio_b: Any, prompt: Prompt):
        # Implementation: Ask Gemini which audio matches prompt better
        # Return "A", "B", or "Tie"
        return "A"