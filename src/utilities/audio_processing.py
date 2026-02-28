import torchaudio
import librosa
import numpy as np

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