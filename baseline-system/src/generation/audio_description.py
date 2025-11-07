"""
Audio description generation using CLAP (Contrastive Language-Audio Pretraining).

Uses LAION CLAP model to generate textual descriptions of audio files.
"""

import os
import json
from pathlib import Path
from typing import Optional
import numpy as np
import laion_clap
import torch


# Global model instance (lazy loaded)
_clap_model = None
_device = None
_socialfx_labels = None


def _get_clap_model(device: str = 'cuda:0'):
    """
    Get or initialize the CLAP model (singleton pattern).

    Args:
        device: Device to use (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu')
                If 'cuda:X' is specified but CUDA is not available, falls back to MPS if available, else CPU.

    Returns:
        tuple: (model, device)
    """
    global _clap_model, _device

    if _clap_model is None:
        # Determine the actual device to use
        if device.startswith('cuda'):
            if torch.cuda.is_available():
                _device = device
                # Extract GPU index if specified (e.g., 'cuda:1' -> 1)
                if ':' in device:
                    gpu_index = int(device.split(':')[1])
                    torch.cuda.set_device(gpu_index)
            elif torch.backends.mps.is_available():
                print(f"[CLAP] CUDA not available, falling back to MPS")
                _device = 'mps'
            else:
                print(f"[CLAP] CUDA not available, falling back to CPU")
                _device = 'cpu'
        elif device == 'mps':
            if torch.backends.mps.is_available():
                _device = 'mps'
            else:
                print(f"[CLAP] MPS not available, falling back to CPU")
                _device = 'cpu'
        else:
            _device = device

        print(f"[CLAP] Initializing model on {_device}...")

        # Clean up any existing GPU memory
        if _device.startswith('cuda'):
            torch.cuda.empty_cache()

        # Set random seeds for deterministic behavior
        torch.manual_seed(42)
        if _device.startswith('cuda'):
            torch.cuda.manual_seed(42)

        # Initialize CLAP model
        # Using music_audioset model which is trained on music and general audio
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, device=_device)
        _clap_model.load_ckpt()  # Load default pretrained checkpoint

        # Set to eval mode to disable dropout and make it deterministic
        _clap_model.eval()

        # Disable cudnn benchmark for determinism (CUDA only)
        if _device.startswith('cuda'):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print(f"[CLAP] Model loaded successfully on {_device}")

    return _clap_model, _device


def load_socialfx_labels(labels_path: str = None, effect_type: str = None) -> list[str]:
    """
    Load Social FX labels from JSON file (singleton pattern).

    Args:
        labels_path: Path to socialfx_labels.json (optional, uses default if not provided)
        effect_type: Filter by effect type: 'reverb', 'eq', 'compressor', or None for all (default: None)

    Returns:
        List of Social FX labels (filtered by effect_type if specified)
    """
    global _socialfx_labels

    if _socialfx_labels is None:
        # Determine default path if not provided
        if labels_path is None:
            # Try to find the labels file relative to this file
            current_file = Path(__file__)
            possible_paths = [
                current_file.parent.parent.parent / 'data' / 'socialfx_labels.json',
                Path('baseline-system/data/socialfx_labels.json'),
                Path('data/socialfx_labels.json'),
            ]

            for path in possible_paths:
                if path.exists():
                    labels_path = str(path)
                    break

        if labels_path is None or not Path(labels_path).exists():
            print("[Warning] Social FX labels not found, using fallback")
            return None

        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                _socialfx_labels = json.load(f)

            total_labels = len(_socialfx_labels.get('all', []))
            print(f"[Social FX] Loaded {total_labels} labels from {labels_path}")
            print(f"  Reverb: {len(_socialfx_labels.get('reverb', []))} labels")
            print(f"  EQ: {len(_socialfx_labels.get('eq', []))} labels")
            print(f"  Compressor: {len(_socialfx_labels.get('compressor', []))} labels")

        except Exception as e:
            print(f"[Warning] Failed to load Social FX labels: {e}")
            return None

    # Filter by effect type if specified
    if effect_type is not None:
        if effect_type not in ['reverb', 'eq', 'compressor']:
            raise ValueError(f"Invalid effect_type: {effect_type}. Must be 'reverb', 'eq', 'compressor', or None")

        filtered_labels = _socialfx_labels.get(effect_type, [])
        if not filtered_labels:
            print(f"[Warning] No labels found for effect type: {effect_type}")
        return filtered_labels

    # Return all labels
    return _socialfx_labels.get('all', [])


def generate_audio_description_with_clap_topk(
    audio_path: str,
    candidate_descriptions: list[str],
    k: int = 5,
    batch_size: int = 128,
    device: str = 'cuda:0'
) -> list[tuple[str, float]]:
    """
    Generate top-k textual descriptions of audio using CLAP model.
    Uses batch processing to manage GPU memory efficiently.

    Args:
        audio_path: Path to audio file
        candidate_descriptions: List of candidate descriptions to rank
        k: Number of top matches to return (default: 5)
        batch_size: Batch size for text encoding (default: 128)
        device: Device to use (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu'). Default: 'cuda:0'

    Returns:
        List of (label, score) tuples, sorted by score descending

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If k > len(candidate_descriptions)
    """
    # Check if audio file exists
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Validate k
    if k > len(candidate_descriptions):
        raise ValueError(
            f"k ({k}) cannot be greater than number of candidates ({len(candidate_descriptions)})"
        )

    try:
        # Get CLAP model
        model, active_device = _get_clap_model(device)

        # Load and encode audio
        audio_file = str(audio_path_obj.absolute())
        print(f"[CLAP] Encoding audio on {active_device}...")
        audio_embeddings = model.get_audio_embedding_from_filelist(x=[audio_file], use_tensor=True)

        # Process text in batches to avoid OOM
        num_candidates = len(candidate_descriptions)
        all_similarities = []

        print(f"[CLAP] Encoding {num_candidates} text labels in batches of {batch_size}...")
        num_batches = (num_candidates + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_candidates)
            batch_texts = candidate_descriptions[start_idx:end_idx]

            # Encode batch
            with torch.no_grad():  # Don't track gradients to save memory
                text_embeddings = model.get_text_embedding(batch_texts, use_tensor=True)

                # Compute similarities for this batch
                batch_similarities = audio_embeddings @ text_embeddings.T

                # Move to CPU immediately to free memory
                all_similarities.append(batch_similarities.cpu())

                # Clean up
                del text_embeddings, batch_similarities
                if active_device.startswith('cuda'):
                    torch.cuda.empty_cache()

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Processed {batch_idx + 1}/{num_batches} batches")

        # Concatenate all similarities on CPU
        similarities = torch.cat(all_similarities, dim=1)
        similarities_np = similarities.numpy().flatten()

        # Get top-k indices
        top_k_indices = np.argsort(similarities_np)[-k:][::-1]

        # Build result list
        results = []
        for idx in top_k_indices:
            label = candidate_descriptions[idx]
            score = float(similarities_np[idx])
            results.append((label, score))

        print(f"[CLAP] Top-{k} matching complete")
        return results

    except Exception as e:
        raise Exception(f"CLAP top-k description failed: {str(e)}")


def generate_audio_description_with_clap(
    audio_path: str,
    candidate_descriptions: Optional[list[str]] = None,
    k: int = 5,
    effect_type: Optional[str] = None,
    device: str = 'cuda:0'
) -> str:
    """
    Generate a textual description of audio using CLAP model with top-k matching.

    If candidate_descriptions are provided, uses them. Otherwise, loads Social FX labels.
    Can optionally filter by effect type (reverb, eq, compressor).
    Returns a formatted string with top-k best matching labels.

    Args:
        audio_path: Path to audio file
        candidate_descriptions: Optional list of candidate descriptions to choose from
        k: Number of top labels to return (default: 5)
        effect_type: Filter labels by effect type: 'reverb', 'eq', 'compressor', or None (default: None)
        device: Device to use (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu'). Default: 'cuda:0'

    Returns:
        String description of the audio (e.g., "The audio is: spacious, warm, echo, deep, hollow")

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If invalid effect_type specified
        Exception: If CLAP model fails
    """
    # Check if audio file exists
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Determine which candidates to use
    if candidate_descriptions is None:
        # Load Social FX labels (optionally filtered by effect type)
        socialfx_labels = load_socialfx_labels(effect_type=effect_type)

        if socialfx_labels:
            candidate_descriptions = socialfx_labels
            if effect_type:
                print(f"[Info] Using {len(candidate_descriptions)} {effect_type} labels")
        else:
            # Fallback to hardcoded candidates if Social FX labels not available
            print("[Warning] Using fallback hardcoded candidates")
            candidate_descriptions = [
                "bright", "warm", "balanced", "spacious", "dry", "natural",
                "compressed", "dynamic", "dark", "present", "aggressive", "smooth"
            ]

    try:
        # Use top-k matching
        top_k_results = generate_audio_description_with_clap_topk(
            audio_path,
            candidate_descriptions,
            k=min(k, len(candidate_descriptions)),
            device=device
        )

        # Format as simple comma-separated list
        labels = [label for label, score in top_k_results]
        description = f"The audio is: {', '.join(labels)}"

        return description

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
