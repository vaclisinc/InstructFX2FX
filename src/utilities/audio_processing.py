import torch
from typing import Any, Dict, Tuple
import torchaudio
import librosa
import numpy as np
from pathlib import Path
import json
import soundfile as sf


def load_and_preprocess_audio(file_path, device, sample_rate=44100, max_seconds=10):
    """Load audio file, resample to 44.1kHz if needed, and limit length."""
    audio, sr = torchaudio.load(file_path)

    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)
        sr = sample_rate

    # Limit length (max_seconds seconds max)
    max_samples = max_seconds * sr
    if audio.shape[-1] > max_samples:
        audio = audio[..., :max_samples]

    audio = audio.unsqueeze(0).to(device)  # Add batch dimension
    return audio

def play_audio(audio, sample_rate=44100):
    """Utility to play audio in Jupyter notebooks."""
    from IPython.display import Audio
    audio_np = audio.detach().squeeze().cpu().numpy().squeeze()
    return Audio(audio_np, rate=sample_rate)

def play_audio_from_file(file_path, sample_rate=44100):
    """Utility to play audio from a file in Jupyter notebooks."""
    from IPython.display import Audio
    return Audio(filename=file_path, rate=sample_rate)

def resample_audio(audio, sr):
    print("current sample rate:", sr)

    if sr != 48000:
        print('resampling audio to 48kHz for CLAP processing...')
        audio = librosa.resample(
            audio.astype(np.float32),
            orig_sr=sr,
            target_sr=48000,
            res_type="kaiser_best",
        )
    else:
        print('Audio is already at 48kHz, no resampling needed.')
    print("new sample rate: 48000 as type ", type(audio))
    return audio

def save_audio_batch(audio_list, output_dir, sample_rate=44100):
    """Save a batch of audio tensors to individual WAV files using variable names.

    Args:
        audio_list: List of audio tensors/arrays to save (e.g., [audio1, audio2, audio3])
        output_dir: Path to output directory (will be created if it doesn't exist)
        sample_rate: Sample rate for saving (default 44100)

    Returns:
        List of saved file paths

    Example:
        save_audio_batch([audio, audio1, audio2, audio3], "../results/comparison")
    """
    import os
    import torch
    import inspect

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    # Get variable names from caller's scope using inspection
    frame = inspect.currentframe().f_back
    caller_locals = frame.f_locals

    # Map object IDs to variable names
    id_to_name = {}
    for var_name, var_value in caller_locals.items():
        id_to_name[id(var_value)] = var_name

    for audio in audio_list:
        # Get variable name from the caller's scope
        audio_id = id(audio)
        var_name = id_to_name.get(audio_id, f"audio_{len(saved_paths)}")

        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().squeeze().numpy()
        else:
            audio_np = np.asarray(audio).squeeze()

        # Ensure 1D or 2D (channels, samples)
        if audio_np.ndim == 3:
            audio_np = audio_np.squeeze(0)

        # Normalize filename
        safe_name = str(var_name).replace('/', '_').replace(' ', '_')
        if not safe_name.endswith('.wav'):
            safe_name += '.wav'

        file_path = os.path.join(output_dir, safe_name)

        # Save using torchaudio
        audio_tensor = torch.from_numpy(audio_np.astype(np.float32))
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        torchaudio.save(file_path, audio_tensor, sample_rate)
        saved_paths.append(file_path)
        print(f"✓ Saved: {var_name} → {file_path}")

    return saved_paths




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


def _process_and_save_audio_and_params(
    audio_tensor: torch.Tensor,
    params_tensor: torch.Tensor,
    params_dict: Dict[str, Any],
    fx_chain,
    sample_rate: int,
    output_dir: Path,
    name: str,
    details: Dict[str, Any] = None,
) -> Tuple[str, str]:
    """
    Process audio through FX chain and save both audio and parameters.

    Returns:
        Tuple of (audio_file_path, params_file_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process audio
    if params_tensor is not None:
        output_audio = fx_chain(audio_tensor, params_tensor)
        output_audio_np = output_audio.squeeze().detach().cpu().numpy()
    else:
        output_audio_np = audio_tensor.squeeze().detach().cpu().numpy()

    # Save audio
    audio_path = output_dir / f"{name}.wav"
    sf.write(audio_path, output_audio_np, sample_rate)

    # Save parameters
    params_path = output_dir / f"{name}_params.json"
    with open(params_path, "w") as f:
        json.dump(
            {
                "params_dict": params_dict or {},
                "params_tensor": params_tensor.detach().cpu().numpy().tolist() if params_tensor is not None else [],
                "details": details or {},
            },
            f,
            indent=2,
        )

    return str(audio_path), str(params_path)


def _ensure_bct(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Ensure audio is shaped as [batch, channels, time] for dasp_pytorch."""
    if audio_tensor.dim() == 1:
        return audio_tensor.unsqueeze(0).unsqueeze(0)
    if audio_tensor.dim() == 2:
        return audio_tensor.unsqueeze(0)
    if audio_tensor.dim() == 3:
        return audio_tensor
    raise ValueError(f"Expected audio tensor with 1-3 dims, got shape {tuple(audio_tensor.shape)}")


def _process_and_save_audio_param_history(audio_param_history, stages_dir, audio, fx_chain, sample_rate):
    intermediate_dir = stages_dir / "optimization_steps"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    print(f"saving the intermediate audio and params for {len(audio_param_history)} optimization steps to: {intermediate_dir}")

    if isinstance(audio_param_history, dict):
        history_items = list(audio_param_history.items())
    else:
        history_items = [(f"iter_{idx:04d}", item) for idx, item in enumerate(audio_param_history)]

    saved_count = 0
    for item_key, item_value in history_items:
        stage_name = str(item_key)
        snap_params_tensor = None
        effected = None

        if isinstance(item_value, tuple) and len(item_value) >= 2:
            if isinstance(item_value[0], torch.Tensor):
                effected = item_value[0]
            snap_params_tensor = item_value[1]
        elif isinstance(item_value, dict):
            if "audio" in item_value:
                effected = item_value['audio']
            snap_params_tensor = item_value.get("params")

        if effected is None:
            if snap_params_tensor is None:
                continue

            if not isinstance(snap_params_tensor, torch.Tensor):
                try:
                    snap_params_tensor = torch.tensor(snap_params_tensor, dtype=audio.dtype, device=audio.device)
                except Exception:
                    continue

            if snap_params_tensor.dim() == 1:
                snap_params_tensor = snap_params_tensor.unsqueeze(0)
            snap_params_tensor = snap_params_tensor.to(device=audio.device, dtype=audio.dtype)

            try:
                # Apply param tensor values directly to the original audio.
                effected = fx_chain(audio, snap_params_tensor)
            except Exception:
                continue

        effected_np = effected.squeeze().detach().cpu().numpy()
        snap_path = intermediate_dir / f"{stage_name}.wav"
        sf.write(snap_path, effected_np, sample_rate)

        snap_params_path = intermediate_dir / f"{stage_name}_params.json"
        with open(snap_params_path, "w") as f:
            json.dump(
                {
                    "params_tensor": snap_params_tensor.detach().cpu().tolist(),
                },
                f,
                indent=2,
            )
        saved_count += 1

        print(f"✓ Saved {saved_count} intermediate snapshots")