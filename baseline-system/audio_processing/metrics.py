"""Audio metrics computation for quality assessment and analysis."""

import numpy as np
import librosa
from typing import Union


class AudioMetrics:
    """Utility class for computing audio quality metrics.

    All methods are static and can be called without instantiation.
    Provides metrics for RMS level, peak level, spectral characteristics,
    and audio quality issues like clipping.
    """

    @staticmethod
    def compute_rms(audio: np.ndarray) -> float:
        """Compute Root Mean Square (RMS) level of audio signal.

        RMS represents the average power of the signal and is useful for
        measuring overall loudness and comparing processed vs unprocessed audio.

        Args:
            audio: Audio signal as numpy array (mono or stereo)

        Returns:
            RMS level as float (0.0 to 1.0 for normalized audio)

        Example:
            >>> audio, sr = librosa.load("audio.wav")
            >>> rms = AudioMetrics.compute_rms(audio)
            >>> print(f"RMS: {rms:.4f}")
        """
        return float(np.sqrt(np.mean(audio**2)))

    @staticmethod
    def compute_peak(audio: np.ndarray) -> float:
        """Compute peak level (maximum absolute amplitude) of audio signal.

        Peak level identifies the loudest sample and is critical for
        detecting clipping or headroom issues.

        Args:
            audio: Audio signal as numpy array (mono or stereo)

        Returns:
            Peak level as float (0.0 to 1.0 for normalized audio)

        Example:
            >>> audio, sr = librosa.load("audio.wav")
            >>> peak = AudioMetrics.compute_peak(audio)
            >>> print(f"Peak: {peak:.4f}")
        """
        return float(np.max(np.abs(audio)))

    @staticmethod
    def compute_spectral_centroid(audio: np.ndarray, sr: int) -> float:
        """Compute spectral centroid (brightness measure) of audio signal.

        The spectral centroid indicates where the "center of mass" of the
        spectrum is located. Higher values indicate brighter, more high-frequency
        content. Lower values indicate darker, more bass-heavy content.

        Args:
            audio: Audio signal as numpy array (mono recommended)
            sr: Sample rate in Hz

        Returns:
            Mean spectral centroid in Hz

        Example:
            >>> audio, sr = librosa.load("audio.wav")
            >>> centroid = AudioMetrics.compute_spectral_centroid(audio, sr)
            >>> print(f"Spectral Centroid: {centroid:.2f} Hz")
        """
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        return float(np.mean(centroid))

    @staticmethod
    def has_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
        """Detect clipping (signal exceeding valid range) in audio.

        Clipping occurs when the audio signal exceeds the maximum representable
        amplitude, causing distortion. This checks if any samples approach or
        exceed the normalized maximum.

        Args:
            audio: Audio signal as numpy array (mono or stereo)
            threshold: Amplitude threshold for clipping detection (default 0.99)
                      Values closer to 1.0 are more strict

        Returns:
            True if clipping detected, False otherwise

        Example:
            >>> audio, sr = librosa.load("audio.wav")
            >>> if AudioMetrics.has_clipping(audio):
            ...     print("Warning: Audio contains clipping!")
        """
        return bool(np.any(np.abs(audio) >= threshold))

    @staticmethod
    def compute_all(audio: np.ndarray, sr: int, clip_threshold: float = 0.99) -> dict:
        """Compute all available metrics for the audio signal.

        Convenience method to compute all metrics in one call.

        Args:
            audio: Audio signal as numpy array
            sr: Sample rate in Hz
            clip_threshold: Threshold for clipping detection (default 0.99)

        Returns:
            Dictionary containing all computed metrics:
                - rms: RMS level
                - peak: Peak level
                - spectral_centroid: Spectral centroid in Hz
                - has_clipping: Boolean indicating clipping presence

        Example:
            >>> audio, sr = librosa.load("audio.wav")
            >>> metrics = AudioMetrics.compute_all(audio, sr)
            >>> print(f"RMS: {metrics['rms']:.4f}")
            >>> print(f"Peak: {metrics['peak']:.4f}")
            >>> print(f"Spectral Centroid: {metrics['spectral_centroid']:.2f} Hz")
            >>> print(f"Clipping: {metrics['has_clipping']}")
        """
        return {
            "rms": AudioMetrics.compute_rms(audio),
            "peak": AudioMetrics.compute_peak(audio),
            "spectral_centroid": AudioMetrics.compute_spectral_centroid(audio, sr),
            "has_clipping": AudioMetrics.has_clipping(audio, clip_threshold),
        }
