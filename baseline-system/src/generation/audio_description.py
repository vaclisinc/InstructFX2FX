"""
Audio description generation using CLAP (Contrastive Language-Audio Pretraining) and MERT.

Uses LAION CLAP model and optionally MERT (Music Enhanced Representation Transformer)
to generate textual descriptions of audio files.

Supported retrieval modes:
- 'clap': Pure CLAP audio-text matching (default)
- 'mert': MERT audio embeddings with CLAP text embeddings
- 'hybrid': Weighted combination of CLAP and MERT scores
"""

import os
import json
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import laion_clap
import torch
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor, AutoModel


# Global model instances (lazy loaded)
_clap_model = None
_mert_model = None
_mert_processor = None
_device = None
_socialfx_labels = None


def _determine_device(device: str, model_name: str = "Model") -> str:
    """
    Determine the actual device to use with fallback logic.

    Args:
        device: Requested device (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu')
        model_name: Name of the model for logging purposes

    Returns:
        Actual device string to use
    """
    if device.startswith('cuda'):
        if torch.cuda.is_available():
            # Extract GPU index if specified (e.g., 'cuda:1' -> 1)
            if ':' in device:
                gpu_index = int(device.split(':')[1])
                torch.cuda.set_device(gpu_index)
            return device
        elif torch.backends.mps.is_available():
            print(f"[{model_name}] CUDA not available, falling back to MPS")
            return 'mps'
        else:
            print(f"[{model_name}] CUDA not available, falling back to CPU")
            return 'cpu'
    elif device == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
        else:
            print(f"[{model_name}] MPS not available, falling back to CPU")
            return 'cpu'
    else:
        return device


def _setup_deterministic_behavior(device: str):
    """
    Setup deterministic behavior for reproducible results.

    Args:
        device: Device string (e.g., 'cuda:0', 'mps', 'cpu')
    """
    # Set random seeds
    torch.manual_seed(42)
    if device.startswith('cuda'):
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _cleanup_gpu_memory(device: str):
    """
    Clean up GPU memory if using CUDA.

    Args:
        device: Device string (e.g., 'cuda:0', 'mps', 'cpu')
    """
    if device.startswith('cuda'):
        torch.cuda.empty_cache()


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
        _device = _determine_device(device, "CLAP")
        print(f"[CLAP] Initializing model on {_device}...")

        _cleanup_gpu_memory(_device)
        _setup_deterministic_behavior(_device)

        # Initialize CLAP model
        # Using music_audioset model which is trained on music and general audio
        _clap_model = laion_clap.CLAP_Module(enable_fusion=False, device=_device)
        _clap_model.load_ckpt()  # Load default pretrained checkpoint

        # Set to eval mode to disable dropout and make it deterministic
        _clap_model.eval()

        print(f"[CLAP] Model loaded successfully on {_device}")

    return _clap_model, _device


def _get_mert_model(device: str = 'cuda:0'):
    """
    Get or initialize the MERT model (singleton pattern).

    Args:
        device: Device to use (e.g., 'cuda:0', 'cuda:1', 'mps', 'cpu')
                If 'cuda:X' is specified but CUDA is not available, falls back to MPS if available, else CPU.

    Returns:
        tuple: (model, processor, device)
    """
    global _mert_model, _mert_processor, _device

    if _mert_model is None:
        _device = _determine_device(device, "MERT")
        print(f"[MERT] Initializing model on {_device}...")

        _cleanup_gpu_memory(_device)
        _setup_deterministic_behavior(_device)

        # Initialize MERT model (using 95M parameter version for efficiency)
        # Available models: m-a-p/MERT-v1-95M, m-a-p/MERT-v1-330M
        model_name = "m-a-p/MERT-v1-95M"

        print(f"[MERT] Loading {model_name}...")
        _mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        _mert_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        _mert_model.to(_device)

        # Set to eval mode
        _mert_model.eval()

        print(f"[MERT] Model loaded successfully on {_device}")

    return _mert_model, _mert_processor, _device


def get_mert_embeddings(audio_path: str, device: str = 'cuda:0', layer: int = -1) -> np.ndarray:
    """
    Extract MERT embeddings from audio file.

    Args:
        audio_path: Path to audio file
        device: Device to use
        layer: Which layer to extract embeddings from (-1 for last layer)

    Returns:
        numpy array of shape (embedding_dim,) - mean-pooled across time
    """
    model, processor, active_device = _get_mert_model(device)

    # Load audio
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio using soundfile (more reliable than torchaudio)
    waveform, sample_rate = sf.read(str(audio_path_obj))

    # Convert to numpy array and handle channels
    if len(waveform.shape) > 1:
        # Convert stereo to mono by averaging channels
        waveform = waveform.mean(axis=1)

    # Resample if needed using scipy
    if sample_rate != processor.sampling_rate:
        from scipy import signal
        # Calculate resampling ratio
        num_samples = int(len(waveform) * processor.sampling_rate / sample_rate)
        waveform = signal.resample(waveform, num_samples)
        sample_rate = processor.sampling_rate

    # Process audio
    inputs = processor(waveform, sampling_rate=processor.sampling_rate, return_tensors="pt")
    inputs = {k: v.to(active_device) for k, v in inputs.items()}

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

        # Get hidden states from specified layer
        if layer == -1:
            hidden_states = outputs.last_hidden_state  # (batch, time, hidden_dim)
        else:
            hidden_states = outputs.hidden_states[layer]

        # Mean pool across time dimension
        embeddings = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        embeddings = embeddings.cpu().numpy().squeeze()  # (hidden_dim,)

    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings)

    return embeddings


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


def _project_embedding(source_emb: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Project embedding to target dimension using truncate/pad strategy.

    Args:
        source_emb: Source embedding (normalized)
        target_dim: Target dimension size

    Returns:
        Projected embedding (normalized)
    """
    source_dim = source_emb.shape[0]

    if source_dim > target_dim:
        # Truncate (simple approach - could use PCA for better projection)
        projected = source_emb[:target_dim]
    elif source_dim < target_dim:
        # Pad with zeros
        padding = np.zeros(target_dim - source_dim)
        projected = np.concatenate([source_emb, padding])
    else:
        projected = source_emb

    # Renormalize after projection
    return projected / np.linalg.norm(projected)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize scores to [0, 1] range using min-max normalization.

    Args:
        scores: Array of scores

    Returns:
        Normalized scores in [0, 1] range
    """
    score_min, score_max = scores.min(), scores.max()

    if score_max > score_min:
        return (scores - score_min) / (score_max - score_min)
    else:
        # All scores are the same - return uniform distribution
        return np.ones_like(scores)


def _encode_text_in_batches(
    clap_model,
    candidate_descriptions: list[str],
    batch_size: int,
    device: str
) -> np.ndarray:
    """
    Encode text descriptions in batches to avoid OOM.

    Args:
        clap_model: CLAP model instance
        candidate_descriptions: List of text descriptions
        batch_size: Batch size for encoding
        device: Device string for GPU cleanup

    Returns:
        numpy array of text embeddings (num_candidates, embedding_dim)
    """
    num_candidates = len(candidate_descriptions)
    all_text_embeddings = []

    print(f"[CLAP] Encoding {num_candidates} text labels in batches of {batch_size}...")
    num_batches = (num_candidates + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_candidates)
        batch_texts = candidate_descriptions[start_idx:end_idx]

        with torch.no_grad():
            text_embeddings = clap_model.get_text_embedding(batch_texts, use_tensor=True)
            all_text_embeddings.append(text_embeddings.cpu())

            del text_embeddings
            _cleanup_gpu_memory(device)

        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            print(f"  Processed {batch_idx + 1}/{num_batches} batches")

    # Concatenate all batches
    return torch.cat(all_text_embeddings, dim=0)


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

        # Encode text in batches
        text_embeddings = _encode_text_in_batches(model, candidate_descriptions, batch_size, active_device)

        # Compute all similarities at once
        with torch.no_grad():
            similarities = audio_embeddings @ text_embeddings.T
            similarities_np = similarities.cpu().numpy().flatten()

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


def generate_audio_description_hybrid(
    audio_path: str,
    candidate_descriptions: list[str],
    k: int = 5,
    mode: Literal['clap', 'mert', 'hybrid'] = 'clap',
    mert_weight: float = 0.5,
    batch_size: int = 128,
    device: str = 'cuda:0'
) -> list[tuple[str, float]]:
    """
    Generate top-k textual descriptions using CLAP, MERT, or hybrid approach.

    Retrieval modes:
    - 'clap': Standard CLAP audio-text matching (default)
    - 'mert': MERT audio embeddings vs CLAP text embeddings
    - 'hybrid': Weighted combination of CLAP and MERT scores

    Args:
        audio_path: Path to audio file
        candidate_descriptions: List of candidate descriptions to rank
        k: Number of top matches to return
        mode: Retrieval mode ('clap', 'mert', 'hybrid')
        mert_weight: Weight for MERT scores in hybrid mode (0-1), CLAP gets (1-mert_weight)
        batch_size: Batch size for text encoding
        device: Device to use

    Returns:
        List of (label, score) tuples, sorted by score descending

    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If invalid parameters
    """
    # Validate inputs
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if k > len(candidate_descriptions):
        raise ValueError(f"k ({k}) cannot be greater than number of candidates ({len(candidate_descriptions)})")

    if not 0 <= mert_weight <= 1:
        raise ValueError(f"mert_weight must be between 0 and 1, got {mert_weight}")

    try:
        if mode == 'clap':
            # Standard CLAP retrieval
            return generate_audio_description_with_clap_topk(
                audio_path, candidate_descriptions, k, batch_size, device
            )

        elif mode == 'mert':
            # MERT mode: MERT audio embeddings → project to CLAP audio space → match with CLAP text
            print(f"[MERT] Extracting MERT audio embeddings...")
            mert_audio_emb = get_mert_embeddings(audio_path, device)  # (mert_dim,)

            # Get CLAP model
            clap_model, active_device = _get_clap_model(device)

            # Get CLAP audio embedding for the same audio (as reference/target space)
            audio_file = str(audio_path_obj.absolute())
            print(f"[CLAP] Getting CLAP audio embedding (target space)...")
            clap_audio_emb = clap_model.get_audio_embedding_from_filelist(x=[audio_file], use_tensor=False)
            clap_audio_emb = clap_audio_emb.flatten()  # (clap_audio_dim,)

            # Project MERT embedding to CLAP audio space
            # TODO: Train a proper projection layer with paired MERT-CLAP embeddings
            mert_dim = mert_audio_emb.shape[0]
            clap_dim = clap_audio_emb.shape[0]
            print(f"[MERT] Projecting MERT embedding ({mert_dim}D) to CLAP audio space ({clap_dim}D)...")
            projected_audio_emb = _project_embedding(mert_audio_emb, clap_dim)

            # Encode text in batches
            text_embeddings = _encode_text_in_batches(clap_model, candidate_descriptions, batch_size, active_device)

            # Compute similarities using projected MERT audio embedding
            with torch.no_grad():
                projected_audio_tensor = torch.from_numpy(projected_audio_emb).unsqueeze(0).to(text_embeddings.device)
                similarities = (projected_audio_tensor @ text_embeddings.T).cpu().numpy().flatten()

            # Get top-k
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            results = [(candidate_descriptions[idx], float(similarities[idx])) for idx in top_k_indices]
            print(f"[MERT->CLAP] Top-{k} matching complete")
            return results

        elif mode == 'hybrid':
            # Combine CLAP and MERT scores
            print(f"[Hybrid] Computing combined CLAP + MERT scores (MERT weight={mert_weight})...")

            # Get CLAP model and audio embedding
            clap_model, active_device = _get_clap_model(device)
            audio_file = str(audio_path_obj.absolute())

            print(f"[CLAP] Encoding audio...")
            clap_audio_emb = clap_model.get_audio_embedding_from_filelist(x=[audio_file], use_tensor=True)

            # Get MERT audio embedding
            print(f"[MERT] Extracting audio embeddings...")
            mert_audio_emb = get_mert_embeddings(audio_path, device)

            # Encode text in batches
            text_embeddings = _encode_text_in_batches(clap_model, candidate_descriptions, batch_size, active_device)

            # Compute CLAP similarities
            with torch.no_grad():
                clap_similarities = (clap_audio_emb @ text_embeddings.T).cpu().numpy().flatten()

            # Compute MERT similarities (project to text embedding space)
            text_emb_np = text_embeddings.cpu().numpy()
            text_emb_norm = text_emb_np / np.linalg.norm(text_emb_np, axis=1, keepdims=True)

            # Project MERT to text embedding dimension
            text_dim = text_emb_norm.shape[1]
            mert_audio_projected = _project_embedding(mert_audio_emb, text_dim)
            mert_similarities = text_emb_norm @ mert_audio_projected

            # Normalize scores to [0, 1] for fair combination
            clap_normalized = _normalize_scores(clap_similarities)
            mert_normalized = _normalize_scores(mert_similarities)

            # Weighted combination
            combined_scores = (1 - mert_weight) * clap_normalized + mert_weight * mert_normalized

            # Get top-k
            top_k_indices = np.argsort(combined_scores)[-k:][::-1]

            results = [(candidate_descriptions[idx], float(combined_scores[idx])) for idx in top_k_indices]
            print(f"[Hybrid] Top-{k} matching complete")
            return results

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'clap', 'mert', or 'hybrid'")

    except Exception as e:
        raise Exception(f"Hybrid audio description failed: {str(e)}")


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
