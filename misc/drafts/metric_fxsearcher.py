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






class FxSearcherInferenceTime(Metric):
    """Seconds per sample (from metadata)"""
    def compute(self, original_audio: Any, target_audio: Any, prompt: Optional[Prompt] = None):
        if isinstance(target_audio, dict):
            return float(target_audio.get('search_time_seconds', 0.0))
        return 0.0



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