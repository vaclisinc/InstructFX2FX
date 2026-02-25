import torchaudio

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