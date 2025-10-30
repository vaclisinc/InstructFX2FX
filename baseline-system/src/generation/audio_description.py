"""
Audio description generation using CLAP (Contrastive Language-Audio Pretraining).

Uses LAION CLAP model to generate textual descriptions of audio files.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import laion_clap
import torch


# Global model instance (lazy loaded)
_clap_model = None
_device = None


def _get_clap_model():
    """
    Get or initialize the CLAP model (singleton pattern).

    Returns:
        tuple: (model, device)
    """
    global _clap_model, _device

    if _clap_model is None:
        # Determine device
        _device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[CLAP] Initializing model on {_device}...")

        # Initialize CLAP model
        # Using music_audioset model which is trained on music and general audio
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, device=_device)
        _clap_model.load_ckpt()  # Load default pretrained checkpoint

        print("[CLAP] Model loaded successfully")

    return _clap_model, _device


def generate_audio_description_with_clap(
    audio_path: str,
    candidate_descriptions: Optional[list[str]] = None
) -> str:
    """
    Generate a textual description of audio using CLAP model.

    If candidate_descriptions are provided, selects the best matching description.
    Otherwise, uses CLAP embeddings to generate a description.

    Args:
        audio_path: Path to audio file
        candidate_descriptions: Optional list of candidate descriptions to choose from

    Returns:
        String description of the audio

    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If CLAP model fails
    """
    # Check if audio file exists
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        # Get CLAP model
        model, device = _get_clap_model()

        # Load and encode audio
        audio_file = str(audio_path_obj.absolute())
        audio_embeddings = model.get_audio_embedding_from_filelist(x=[audio_file], use_tensor=True)

        if candidate_descriptions:
            # Encode candidate descriptions
            text_embeddings = model.get_text_embedding(candidate_descriptions, use_tensor=True)

            # Compute similarities
            similarities = audio_embeddings @ text_embeddings.T
            best_idx = similarities.argmax().item()

            return candidate_descriptions[best_idx]
        else:
            # Generate generic description based on audio characteristics
            # Use a set of candidate descriptions covering common audio characteristics
            default_candidates = [
                "The audio has a bright, high-frequency tone with clear articulation",
                "The audio has a warm, rich tone with pronounced low frequencies",
                "The audio has a balanced frequency response with moderate dynamics",
                "The audio has heavy reverberation creating a spacious atmosphere",
                "The audio has tight, dry sound with minimal reverb",
                "The audio has moderate reverb with a natural room sound",
                "The audio has heavy compression with controlled dynamics",
                "The audio has natural dynamics with light compression",
                "The audio has a dark, mellow tone with soft high frequencies",
                "The audio has enhanced mid-range frequencies with presence",
                "The audio has a sharp, aggressive tone with boosted highs",
                "The audio has a smooth, polished sound with subtle effects"
            ]

            # Encode candidate descriptions
            text_embeddings = model.get_text_embedding(default_candidates, use_tensor=True)

            # Compute similarities
            similarities = audio_embeddings @ text_embeddings.T
            best_idx = similarities.argmax().item()

            return default_candidates[best_idx]

    except Exception as e:
        raise Exception(f"CLAP audio description failed: {str(e)}")


def generate_audio_description_from_params(params: dict) -> str:
    """
    Generate a textual description based on parameters (fallback method).

    This is the original parameter-based approach, kept as a fallback
    when actual audio files are not available.

    Args:
        params: Dictionary with 'reverb', 'eq', and 'compressor' parameters

    Returns:
        String description of the audio characteristics
    """
    description_parts = []

    # Describe reverb
    if 'reverb' in params:
        reverb = params['reverb']
        wet_dry = reverb.get('wet_dry', 0.5)
        decay = reverb.get('decay', 0.5)

        if wet_dry > 0.7:
            description_parts.append("heavy reverb")
        elif wet_dry > 0.4:
            description_parts.append("moderate reverb")
        else:
            description_parts.append("light reverb")

        if decay > 0.7:
            description_parts.append("with long decay")
        elif decay > 0.4:
            description_parts.append("with medium decay")
        else:
            description_parts.append("with short decay")

    # Describe EQ
    if 'eq' in params and isinstance(params['eq'], list) and len(params['eq']) > 0:
        eq_bands = params['eq']
        high_boost = any(band.get('freq', 0) > 5000 and band.get('gain', 0) > 3 for band in eq_bands)
        low_boost = any(band.get('freq', 0) < 500 and band.get('gain', 0) > 3 for band in eq_bands)

        if high_boost and low_boost:
            description_parts.append("enhanced highs and lows")
        elif high_boost:
            description_parts.append("bright with boosted high frequencies")
        elif low_boost:
            description_parts.append("warm with boosted low frequencies")
        else:
            description_parts.append("balanced EQ")

    # Describe compression
    if 'compressor' in params:
        compressor = params['compressor']
        ratio = compressor.get('ratio', 1.0)

        if ratio > 8:
            description_parts.append("heavy compression")
        elif ratio > 4:
            description_parts.append("moderate compression")
        elif ratio > 2:
            description_parts.append("light compression")

    # Combine description parts
    if description_parts:
        return "The audio has " + ", ".join(description_parts) + "."
    else:
        return "The audio has neutral processing applied."
