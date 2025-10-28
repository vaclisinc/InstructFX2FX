"""Pytest fixtures for test data and common test resources.

This module provides reusable test fixtures including:
- Test audio files (generated programmatically)
- Sample effect parameters
- Sample descriptions
- Test directories

These fixtures use appropriate pytest scopes to optimize test performance.
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, Any


@pytest.fixture(scope="session")
def test_audio_dir(tmp_path_factory):
    """Create directory with test audio files.

    This fixture creates a temporary directory that persists for the entire
    test session. Use this for storing generated test audio files.

    Args:
        tmp_path_factory: Pytest's temporary path factory

    Returns:
        Path to test audio directory
    """
    audio_dir = tmp_path_factory.mktemp("audio")
    return audio_dir


@pytest.fixture(scope="session")
def test_audio_path(test_audio_dir):
    """Generate test audio file with white noise.

    Creates a 1-second WAV file with white noise for testing audio processing.

    Args:
        test_audio_dir: Test audio directory fixture

    Returns:
        Path to generated test audio file
    """
    # Generate 1 second of white noise at 44.1kHz
    sample_rate = 44100
    duration = 1.0
    samples = int(sample_rate * duration)
    audio = np.random.randn(samples).astype(np.float32) * 0.1  # Scale to avoid clipping

    # Write to file
    path = test_audio_dir / "test.wav"
    sf.write(path, audio, sample_rate)

    return path


@pytest.fixture(scope="session")
def test_audio_stereo_path(test_audio_dir):
    """Generate stereo test audio file.

    Creates a 1-second stereo WAV file with slightly different noise in each channel.

    Args:
        test_audio_dir: Test audio directory fixture

    Returns:
        Path to generated stereo test audio file
    """
    sample_rate = 44100
    duration = 1.0
    samples = int(sample_rate * duration)

    # Generate stereo audio (2 channels)
    left = np.random.randn(samples).astype(np.float32) * 0.1
    right = np.random.randn(samples).astype(np.float32) * 0.1
    audio_stereo = np.column_stack((left, right))

    path = test_audio_dir / "test_stereo.wav"
    sf.write(path, audio_stereo, sample_rate)

    return path


@pytest.fixture
def sample_parameters() -> Dict[str, Any]:
    """Sample effect parameters for testing.

    Returns:
        Dictionary of effect parameters matching expected structure
    """
    return {
        "reverb": {
            "delay_time": 0.03,
            "decay": 0.7,
            "stereo_spread": 0.0,
            "cutoff_freq": 10000,
            "wet_gain": 0.0,
            "wet_dry": 0.6
        },
        "eq": {
            "low_gain": 0.0,
            "mid_gain": 2.0,
            "high_gain": -1.0,
            "low_freq": 100,
            "mid_freq": 1000,
            "high_freq": 8000
        }
    }


@pytest.fixture
def sample_parameters_minimal() -> Dict[str, Any]:
    """Minimal effect parameters for testing edge cases.

    Returns:
        Dictionary with minimal/neutral effect parameters
    """
    return {
        "reverb": {
            "delay_time": 0.01,
            "decay": 0.0,
            "stereo_spread": 0.0,
            "cutoff_freq": 20000,
            "wet_gain": 0.0,
            "wet_dry": 0.0
        },
        "eq": {
            "low_gain": 0.0,
            "mid_gain": 0.0,
            "high_gain": 0.0,
            "low_freq": 100,
            "mid_freq": 1000,
            "high_freq": 8000
        }
    }


@pytest.fixture
def sample_description() -> str:
    """Sample description for testing.

    Returns:
        Example audio effect description string
    """
    return "warm reverb with natural decay"


@pytest.fixture
def sample_descriptions() -> list[str]:
    """Multiple sample descriptions for batch testing.

    Returns:
        List of audio effect description strings
    """
    return [
        "warm reverb with natural decay",
        "bright EQ boost with presence",
        "dark atmospheric with long reverb",
        "clean and transparent processing",
        "vintage analog warmth"
    ]


@pytest.fixture
def sample_audio_features() -> Dict[str, float]:
    """Sample audio features for testing audio analysis.

    Returns:
        Dictionary of mock audio feature values
    """
    return {
        "spectral_centroid": 2500.0,
        "spectral_rolloff": 5000.0,
        "spectral_bandwidth": 1500.0,
        "rms_energy": 0.05,
        "zero_crossing_rate": 0.1,
        "harmonic_ratio": 0.7,
    }


@pytest.fixture
def sample_scoring_response() -> Dict[str, Any]:
    """Sample scoring response for testing.

    Returns:
        Dictionary matching ScoringResponse schema
    """
    return {
        "overall_score": 75.0,
        "confidence": 0.85,
        "dimensions": [
            {
                "name": "semantic_match",
                "score": 80.0,
                "reasoning": "Parameters align well with description"
            },
            {
                "name": "technical_quality",
                "score": 70.0,
                "reasoning": "Technically sound but could be refined"
            },
            {
                "name": "specificity",
                "score": 75.0,
                "reasoning": "Good level of detail in parameters"
            }
        ],
        "feedback": "The parameters show good alignment with the target description",
        "suggestions": [
            "Consider increasing decay time for more natural reverb",
            "Adjust wet/dry mix for better balance"
        ]
    }


__all__ = [
    "test_audio_dir",
    "test_audio_path",
    "test_audio_stereo_path",
    "sample_parameters",
    "sample_parameters_minimal",
    "sample_description",
    "sample_descriptions",
    "sample_audio_features",
    "sample_scoring_response",
]
