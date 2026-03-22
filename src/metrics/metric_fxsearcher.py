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
# from pesq import pesq
from frechet_audio_distance import FrechetAudioDistance
import pyloudnorm as pyln

import json
from typing import Optional, Dict, Any, List

from src.metrics.metric import Metric
from src.prompts.prompt import Prompt
from drafts.clap_fxsearcher import CLAPFxSearcherWrapper # TODO: Change to CLAPWrapper to have the same CLAP for both text2fx and fxsearcher

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
    def compute(self, original_audio: Any, target_audio: Any, prompt=None):

        try:
            from pesq import pesq
        except Exception:
            return None

        wav_o_raw, sr_o = _extract_audio_and_sr(original_audio)
        wav_t_raw, sr_t = _extract_audio_and_sr(target_audio)

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

def run_fxsearcher_evaluation(
    pred_dir: str,
    gt_dir: Optional[str] = None,
    prompts_map: Optional[Dict[str, str]] = None,
    target_sr: int = 48000,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:

    clap_metric = FxSearcherCLAPScore()
    guided_clap_metric = FxSearcherGuidedCLAPScore()
    lufs_metric = FxSearcherIntegratedLUFS()

    # wer_metric = FxSearcherWER()
    pesq_metric = FxSearcherPESQ()

    per_file = {}

    fnames = sorted([f for f in os.listdir(pred_dir) if f.endswith(".wav")])

    if max_files is not None:
        fnames = fnames[:max_files]

    for fname in fnames:

        audio_path = os.path.join(pred_dir, fname)

        prompt_obj = None
        if prompts_map is not None and fname in prompts_map:
            prompt_obj = Prompt(instruction=prompts_map[fname])

        item = AudioItem(audio_path)

        metrics = {}

        if prompt_obj is not None:
            metrics["clap"] = clap_metric.compute(None, item, prompt_obj)
            metrics["guided_clap"] = guided_clap_metric.compute(None, item, prompt_obj)

        metrics["lufs"] = lufs_metric.compute(None, item)

        # try:
        #     metrics["wer"] = wer_metric.compute(item, item)
        # except Exception:
        #     metrics["wer"] = None
        metrics["wer"] = None

        try:
            metrics["pesq"] = pesq_metric.compute(item, item)
        except Exception:
            metrics["pesq"] = None

        per_file[fname] = metrics

    summary = {}

    def safe_mean(values):
        vals = [v for v in values if v is not None]
        if len(vals) == 0:
            return None
        return float(sum(vals) / len(vals))

    clap_vals = [v.get("clap") for v in per_file.values()]
    guided_vals = [v.get("guided_clap") for v in per_file.values()]
    lufs_vals = [v.get("lufs") for v in per_file.values()]

    summary["clap_mean"] = safe_mean(clap_vals)
    summary["guided_clap_mean"] = safe_mean(guided_vals)
    summary["lufs_mean"] = safe_mean(lufs_vals)

    if gt_dir is not None:
        try:
            fad_metric = FxSearcherFAD()
            summary["fad"] = fad_metric.compute(gt_dir, pred_dir)
        except Exception:
            summary["fad"] = None

    return {
        "pred_dir": pred_dir,
        "gt_dir": gt_dir,
        "summary": summary,
        "per_file": per_file,
    }