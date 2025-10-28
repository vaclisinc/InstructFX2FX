"""Parameter normalization utilities.

This module provides utilities for normalizing and correcting parameter values
that are outside valid ranges or have type issues. It attempts to fix common
issues while preserving the intent of the generated parameters.
"""

import logging
import math
from typing import Dict, Any, List, Union, Tuple
from copy import deepcopy

from src.models.parameters import (
    EQParameters,
    EQBand,
    ReverbParameters,
    CompressorParameters,
    EffectChain,
    EffectParameter
)
from .validator import PARAMETER_RANGES


logger = logging.getLogger(__name__)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val].

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float with fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default

    if isinstance(value, (int, float)):
        # Handle special cases
        if math.isnan(value):
            logger.warning(f"NaN value encountered, using default: {default}")
            return default
        if math.isinf(value):
            logger.warning(f"Infinity value encountered, using default: {default}")
            return default
        return float(value)

    if isinstance(value, str):
        try:
            result = float(value)
            if math.isnan(result) or math.isinf(result):
                return default
            return result
        except (ValueError, TypeError):
            logger.warning(f"Cannot convert '{value}' to float, using default: {default}")
            return default

    logger.warning(f"Cannot convert {type(value).__name__} to float, using default: {default}")
    return default


def safe_bool(value: Any, default: bool = False) -> bool:
    """Safely convert value to bool with fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Bool value or default
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    if isinstance(value, str):
        lower = value.lower().strip()
        if lower in ("true", "yes", "1", "on"):
            return True
        if lower in ("false", "no", "0", "off"):
            return False

    logger.warning(f"Cannot convert {type(value).__name__} to bool, using default: {default}")
    return default


def normalize_eq_band(band_data: Dict[str, Any], ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Normalize single EQ band parameters.

    Args:
        band_data: Band data dictionary
        ranges: Parameter ranges

    Returns:
        Normalized band data
    """
    normalized = {}

    # Normalize frequency
    freq = safe_float(band_data.get("frequency"), default=1000.0)
    min_freq, max_freq = ranges["frequency"]
    normalized["frequency"] = clamp(freq, min_freq, max_freq)
    if normalized["frequency"] != freq:
        logger.debug(f"Clamped frequency {freq} -> {normalized['frequency']}")

    # Normalize gain
    gain = safe_float(band_data.get("gain"), default=0.0)
    min_gain, max_gain = ranges["gain"]
    normalized["gain"] = clamp(gain, min_gain, max_gain)
    if normalized["gain"] != gain:
        logger.debug(f"Clamped gain {gain} -> {normalized['gain']}")

    # Normalize Q factor
    q = safe_float(band_data.get("q"), default=1.0)
    min_q, max_q = ranges["q"]
    normalized["q"] = clamp(q, min_q, max_q)
    if normalized["q"] != q:
        logger.debug(f"Clamped Q {q} -> {normalized['q']}")

    return normalized


def normalize_eq_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize EQ effect parameters.

    Args:
        data: EQ effect data

    Returns:
        Normalized EQ data with valid EffectChain
    """
    ranges = PARAMETER_RANGES["eq"]
    normalized = deepcopy(data)

    # Get parameters from either nested or flat structure
    if "parameters" in data:
        params = data["parameters"]
    else:
        params = {k: v for k, v in data.items() if k != "type"}

    # Normalize bands
    if "bands" not in params or not isinstance(params["bands"], list):
        logger.warning("Missing or invalid bands, using default 3-band EQ")
        params["bands"] = [
            {"frequency": 100, "gain": 0, "q": 1.0},
            {"frequency": 1000, "gain": 0, "q": 1.0},
            {"frequency": 10000, "gain": 0, "q": 1.0},
        ]

    bands = params["bands"]

    # Normalize band count
    min_bands, max_bands = ranges["bands_count"]
    if len(bands) < min_bands:
        logger.warning(f"Too few bands ({len(bands)}), padding to {min_bands}")
        # Add default bands
        default_freqs = [100, 1000, 10000, 5000, 200]
        while len(bands) < min_bands:
            freq = default_freqs[len(bands) % len(default_freqs)]
            bands.append({"frequency": freq, "gain": 0, "q": 1.0})

    if len(bands) > max_bands:
        logger.warning(f"Too many bands ({len(bands)}), truncating to {max_bands}")
        bands = bands[:max_bands]

    # Normalize each band
    normalized_bands = []
    for i, band_data in enumerate(bands):
        if not isinstance(band_data, dict):
            logger.warning(f"Band {i} is not a dict, using default")
            band_data = {"frequency": 1000, "gain": 0, "q": 1.0}
        normalized_bands.append(normalize_eq_band(band_data, ranges))

    # Sort bands by frequency to avoid overlap issues
    normalized_bands.sort(key=lambda b: b["frequency"])

    # Ensure minimum spacing between bands
    adjusted_bands = [normalized_bands[0]]
    for i in range(1, len(normalized_bands)):
        prev_freq = adjusted_bands[-1]["frequency"]
        curr_freq = normalized_bands[i]["frequency"]

        # Require 10% minimum spacing
        min_spacing = prev_freq * 1.1
        if curr_freq < min_spacing:
            curr_freq = min(min_spacing, ranges["frequency"][1])
            logger.debug(f"Adjusted band {i} frequency to maintain spacing: {curr_freq}")

        adjusted_bands.append({
            **normalized_bands[i],
            "frequency": curr_freq
        })

    # Build normalized structure
    result = {
        "bands": adjusted_bands,
        "eq_type": params.get("eq_type", "parametric")
    }

    return result


def normalize_reverb_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize reverb effect parameters.

    Args:
        data: Reverb effect data

    Returns:
        Normalized reverb data
    """
    ranges = PARAMETER_RANGES["reverb"]

    # Get parameters
    if "parameters" in data:
        params = data["parameters"]
    else:
        params = {k: v for k, v in data.items() if k != "type"}

    normalized = {}

    # Normalize each parameter
    for field, (min_val, max_val) in ranges.items():
        value = safe_float(params.get(field), default=(min_val + max_val) / 2)
        normalized[field] = clamp(value, min_val, max_val)
        if normalized[field] != value:
            logger.debug(f"Clamped {field} {value} -> {normalized[field]}")

    # Normalize freeze mode
    normalized["freeze_mode"] = safe_bool(params.get("freeze_mode"), default=False)

    # Adjust wet/dry balance if needed
    total = normalized["wet_level"] + normalized["dry_level"]
    if total > 2.0:
        logger.warning(f"Wet/dry total ({total}) > 2.0, normalizing")
        scale = 1.8 / total  # Scale to 1.8 for safety margin
        normalized["wet_level"] *= scale
        normalized["dry_level"] *= scale

    return normalized


def normalize_compressor_parameters(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize compressor effect parameters.

    Args:
        data: Compressor effect data

    Returns:
        Normalized compressor data
    """
    ranges = PARAMETER_RANGES["compressor"]

    # Get parameters
    if "parameters" in data:
        params = data["parameters"]
    else:
        params = {k: v for k, v in data.items() if k != "type"}

    normalized = {}

    # Normalize each parameter
    for field, (min_val, max_val) in ranges.items():
        value = safe_float(params.get(field), default=(min_val + max_val) / 2)
        normalized[field] = clamp(value, min_val, max_val)
        if normalized[field] != value:
            logger.debug(f"Clamped {field} {value} -> {normalized[field]}")

    # Ensure attack < release
    if normalized["attack"] >= normalized["release"]:
        logger.warning(f"Attack ({normalized['attack']}) >= release ({normalized['release']}), adjusting")
        # Set attack to 80% of release
        normalized["attack"] = min(normalized["attack"], normalized["release"] * 0.8)
        # Ensure attack is still in valid range
        normalized["attack"] = clamp(normalized["attack"], ranges["attack"][0], ranges["attack"][1])

    return normalized


def normalize_effect(effect_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize effect parameters based on effect type.

    Args:
        effect_data: Effect data dictionary with 'type' field

    Returns:
        Normalized effect data

    Raises:
        ValueError: If effect type is invalid
    """
    effect_type = effect_data.get("type")
    if not effect_type:
        raise ValueError("Effect data missing 'type' field")

    if effect_type == "eq":
        normalized = normalize_eq_parameters(effect_data)
        return {"type": "eq", **normalized}
    elif effect_type == "reverb":
        normalized = normalize_reverb_parameters(effect_data)
        return {"type": "reverb", **normalized}
    elif effect_type == "compressor":
        normalized = normalize_compressor_parameters(effect_data)
        return {"type": "compressor", **normalized}
    else:
        raise ValueError(f"Unknown effect type: {effect_type}")


def normalize_effect_chain_data(chain_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize effect chain data dictionary.

    Args:
        chain_data: Effect chain data dictionary

    Returns:
        Normalized effect chain data
    """
    normalized = deepcopy(chain_data)

    # Ensure description exists
    if "description" not in normalized or not normalized["description"]:
        normalized["description"] = "Generated audio effects"

    # Ensure effects list exists
    if "effects" not in normalized or not isinstance(normalized["effects"], list):
        raise ValueError("Effect chain must have 'effects' list")

    if len(normalized["effects"]) == 0:
        raise ValueError("Effect chain cannot have empty effects list")

    # Normalize each effect
    normalized_effects = []
    for i, effect_data in enumerate(normalized["effects"]):
        if not isinstance(effect_data, dict):
            logger.warning(f"Effect {i} is not a dict, skipping")
            continue

        try:
            normalized_effect = normalize_effect(effect_data)
            normalized_effects.append(normalized_effect)
        except Exception as e:
            logger.error(f"Failed to normalize effect {i}: {e}")
            # Skip invalid effects

    if len(normalized_effects) == 0:
        raise ValueError("No valid effects after normalization")

    normalized["effects"] = normalized_effects

    # Build order list
    order = [effect["type"] for effect in normalized_effects]
    normalized["order"] = order

    return normalized


def normalize_effect_chain(chain: EffectChain) -> EffectChain:
    """Normalize an EffectChain instance.

    This converts the chain to dict, normalizes, and reconstructs.

    Args:
        chain: EffectChain to normalize

    Returns:
        New normalized EffectChain instance
    """
    # Convert to dict
    chain_data = chain.model_dump()

    # Normalize
    normalized_data = normalize_effect_chain_data(chain_data)

    # Reconstruct
    return EffectChain(**normalized_data)


__all__ = [
    "clamp",
    "safe_float",
    "safe_bool",
    "normalize_eq_parameters",
    "normalize_reverb_parameters",
    "normalize_compressor_parameters",
    "normalize_effect",
    "normalize_effect_chain_data",
    "normalize_effect_chain",
]
