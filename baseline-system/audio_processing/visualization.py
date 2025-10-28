"""Audio visualization utilities for waveform and spectrogram plotting."""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Optional


class AudioVisualizer:
    """Utility class for generating audio visualizations.

    All methods are static and can be called without instantiation.
    Provides waveform plots and spectrograms for audio analysis.
    """

    @staticmethod
    def plot_waveform(
        audio: np.ndarray,
        sr: int,
        save_path: str,
        title: Optional[str] = "Waveform",
        figsize: tuple = (12, 4),
    ) -> None:
        """Generate and save a waveform plot of the audio signal.

        Creates a time-domain visualization showing amplitude over time.
        Useful for identifying transients, dynamics, and overall shape.

        Args:
            audio: Audio signal as numpy array (mono or stereo)
            sr: Sample rate in Hz
            save_path: Path where plot image will be saved (e.g., "waveform.png")
            title: Title for the plot (default: "Waveform")
            figsize: Figure size as (width, height) tuple (default: (12, 4))

        Example:
            >>> import librosa
            >>> audio, sr = librosa.load("audio.wav")
            >>> AudioVisualizer.plot_waveform(audio, sr, "waveform.png")
        """
        plt.figure(figsize=figsize)
        librosa.display.waveshow(audio, sr=sr)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_spectrogram(
        audio: np.ndarray,
        sr: int,
        save_path: str,
        title: Optional[str] = "Spectrogram",
        figsize: tuple = (12, 6),
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> None:
        """Generate and save a spectrogram plot of the audio signal.

        Creates a frequency-domain visualization showing frequency content
        over time. Brightness indicates amplitude at each frequency.

        Args:
            audio: Audio signal as numpy array (mono recommended)
            sr: Sample rate in Hz
            save_path: Path where plot image will be saved (e.g., "spectrogram.png")
            title: Title for the plot (default: "Spectrogram")
            figsize: Figure size as (width, height) tuple (default: (12, 6))
            n_fft: FFT window size (default: 2048)
            hop_length: Number of samples between successive frames (default: 512)

        Example:
            >>> import librosa
            >>> audio, sr = librosa.load("audio.wav")
            >>> AudioVisualizer.plot_spectrogram(audio, sr, "spectrogram.png")
        """
        plt.figure(figsize=figsize)

        # Compute STFT and convert to dB scale
        D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Display spectrogram
        img = librosa.display.specshow(
            D_db, sr=sr, x_axis="time", y_axis="hz", hop_length=hop_length
        )

        plt.title(title)
        plt.colorbar(img, format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_mel_spectrogram(
        audio: np.ndarray,
        sr: int,
        save_path: str,
        title: Optional[str] = "Mel Spectrogram",
        figsize: tuple = (12, 6),
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ) -> None:
        """Generate and save a mel-scaled spectrogram plot.

        Creates a perceptually-scaled frequency-domain visualization where
        frequency bins are spaced according to the mel scale (mimicking
        human hearing). Useful for music and speech analysis.

        Args:
            audio: Audio signal as numpy array (mono recommended)
            sr: Sample rate in Hz
            save_path: Path where plot image will be saved (e.g., "mel_spec.png")
            title: Title for the plot (default: "Mel Spectrogram")
            figsize: Figure size as (width, height) tuple (default: (12, 6))
            n_fft: FFT window size (default: 2048)
            hop_length: Number of samples between successive frames (default: 512)
            n_mels: Number of mel bands (default: 128)

        Example:
            >>> import librosa
            >>> audio, sr = librosa.load("audio.wav")
            >>> AudioVisualizer.plot_mel_spectrogram(audio, sr, "mel_spec.png")
        """
        plt.figure(figsize=figsize)

        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Display mel spectrogram
        img = librosa.display.specshow(
            mel_spec_db, sr=sr, x_axis="time", y_axis="mel", hop_length=hop_length
        )

        plt.title(title)
        plt.colorbar(img, format="%+2.0f dB")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_comparison(
        audio1: np.ndarray,
        audio2: np.ndarray,
        sr: int,
        save_path: str,
        label1: str = "Original",
        label2: str = "Processed",
        figsize: tuple = (12, 8),
    ) -> None:
        """Generate side-by-side waveform comparison of two audio signals.

        Useful for comparing original vs processed audio, or A/B testing
        different effect settings.

        Args:
            audio1: First audio signal as numpy array
            audio2: Second audio signal as numpy array
            sr: Sample rate in Hz
            save_path: Path where plot image will be saved
            label1: Label for first audio (default: "Original")
            label2: Label for second audio (default: "Processed")
            figsize: Figure size as (width, height) tuple (default: (12, 8))

        Example:
            >>> import librosa
            >>> original, sr = librosa.load("original.wav")
            >>> processed, sr = librosa.load("processed.wav")
            >>> AudioVisualizer.plot_comparison(
            ...     original, processed, sr, "comparison.png"
            ... )
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        # Plot first audio
        librosa.display.waveshow(audio1, sr=sr, ax=axes[0])
        axes[0].set_title(label1)
        axes[0].set_ylabel("Amplitude")
        axes[0].grid(True, alpha=0.3)

        # Plot second audio
        librosa.display.waveshow(audio2, sr=sr, ax=axes[1])
        axes[1].set_title(label2)
        axes[1].set_ylabel("Amplitude")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
