"""Audio processing module for metrics computation, visualization, and pipeline integration."""

from audio_processing.metrics import AudioMetrics
from audio_processing.visualization import AudioVisualizer
from audio_processing.processor import AudioProcessor
from audio_processing.batch import BatchProcessor
from audio_processing.types import ProcessingResult

__all__ = [
    "AudioMetrics",
    "AudioVisualizer",
    "AudioProcessor",
    "BatchProcessor",
    "ProcessingResult",
]
