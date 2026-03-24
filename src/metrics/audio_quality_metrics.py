from .metric import Metric
from typing import Any, Optional, Dict
from frechet_audio_distance import FrechetAudioDistance
import torch
import numpy as np
import librosa
import pyloudnorm as pyln

from utilities.audio_processing import _load_waveform_mono

class FxSearcherFAD(Metric):
    """Fréchet Audio Distance (Distribution similarity)"""
    def __init__(self, model_name: str = "vggish"):
        self.fad = FrechetAudioDistance(model_name=model_name, use_pca=False, verbose=False)

    def compute(self, original_audio: str, target_audio: str, prompt: Optional[Any] = None):
        # original_audio and target_audio should be folder paths for a true FAD
        return float(self.fad.score(original_audio, target_audio))


class FxSearcherIntegratedLUFS(Metric):
    """Loudness (BS.1770)"""
    def compute(self, original_audio: Any, target_audio: Any, prompt: Optional[Any] = None):
        if isinstance(target_audio, str):
            wav, sr = librosa.load(target_audio, sr=None, mono=True)
        else:
            wav = _load_waveform_mono(target_audio)
            sr = 44100
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(wav))