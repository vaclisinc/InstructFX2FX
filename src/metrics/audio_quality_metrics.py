from .metric import Metric
from typing import Any, Optional, Dict
from frechet_audio_distance import FrechetAudioDistance
import pyloudnorm as pyln
import torch

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