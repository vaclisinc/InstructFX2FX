"""CSV parsers for loading SocialFX dataset parameters.

This module provides functions to parse CSV files containing EQ, Reverb, and
Compressor parameters into SocialFXExample models with proper type conversion
and validation.
"""

from pathlib import Path
from typing import List
import pandas as pd

from judge_system.data.models import SocialFXExample


def _load_eq_parameters(csv_path: Path, audio_dir: Path) -> List[SocialFXExample]:
    """Load EQ parameters from CSV file.

    Parses a CSV file containing multi-band EQ parameters. The CSV should have
    columns: id, description, instrument, and dynamic band parameters
    (band1_freq, band1_gain, band1_q, band2_freq, band2_gain, band2_q, ...).

    Args:
        csv_path: Path to the EQ parameters CSV file
        audio_dir: Path to directory containing audio samples

    Returns:
        List of SocialFXExample objects with effect_type="eq"

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
        pd.errors.ParserError: If CSV file is corrupted or malformed
        KeyError: If required columns are missing from CSV
        ValueError: If model validation fails (e.g., invalid instrument)
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"EQ parameters CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"EQ parameters CSV is empty: {csv_path}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse EQ parameters CSV: {csv_path}") from e

    # Validate required columns
    required_cols = ['id', 'description', 'instrument']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"EQ parameters CSV missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    examples = []
    for _, row in df.iterrows():
        # Parse band parameters dynamically
        bands = []
        band_idx = 1
        while f"band{band_idx}_freq" in row:
            # Extract band parameters with proper type conversion
            freq_col = f"band{band_idx}_freq"
            gain_col = f"band{band_idx}_gain"
            q_col = f"band{band_idx}_q"

            # Check if all band columns exist
            if freq_col not in row or gain_col not in row or q_col not in row:
                break

            bands.append({
                "frequency": float(row[freq_col]),
                "gain": float(row[gain_col]),
                "q": float(row[q_col])
            })
            band_idx += 1

        # Construct audio path
        audio_path = str(audio_dir / f"{row['instrument']}.wav")

        example = SocialFXExample(
            id=int(row["id"]),
            description=str(row["description"]),
            instrument=str(row["instrument"]),
            effect_type="eq",
            parameters={"bands": bands},
            audio_path=audio_path
        )
        examples.append(example)

    return examples


def _load_reverb_parameters(csv_path: Path, audio_dir: Path) -> List[SocialFXExample]:
    """Load reverb parameters from CSV file.

    Parses a CSV file containing reverb effect parameters. The CSV should have
    columns: id, description, instrument, room_size, damping, wet_level,
    dry_level, width, freeze_mode.

    Args:
        csv_path: Path to the reverb parameters CSV file
        audio_dir: Path to directory containing audio samples

    Returns:
        List of SocialFXExample objects with effect_type="reverb"

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
        pd.errors.ParserError: If CSV file is corrupted or malformed
        KeyError: If required columns are missing from CSV
        ValueError: If model validation fails or boolean conversion fails
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Reverb parameters CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Reverb parameters CSV is empty: {csv_path}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse reverb parameters CSV: {csv_path}") from e

    # Validate required columns
    required_cols = [
        'id', 'description', 'instrument', 'room_size', 'damping',
        'wet_level', 'dry_level', 'width', 'freeze_mode'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Reverb parameters CSV missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    examples = []
    for _, row in df.iterrows():
        # Convert freeze_mode to boolean
        freeze_mode = row["freeze_mode"]
        if isinstance(freeze_mode, str):
            # Handle string representations: "true", "false", "True", "False", "1", "0"
            freeze_mode_lower = freeze_mode.lower().strip()
            if freeze_mode_lower in ("true", "1", "yes"):
                freeze_mode = True
            elif freeze_mode_lower in ("false", "0", "no"):
                freeze_mode = False
            else:
                raise ValueError(
                    f"Invalid freeze_mode value: '{freeze_mode}'. "
                    f"Expected boolean or 'true'/'false' string"
                )
        else:
            # Handle numeric or boolean types
            freeze_mode = bool(freeze_mode)

        # Construct audio path
        audio_path = str(audio_dir / f"{row['instrument']}.wav")

        example = SocialFXExample(
            id=int(row["id"]),
            description=str(row["description"]),
            instrument=str(row["instrument"]),
            effect_type="reverb",
            parameters={
                "room_size": float(row["room_size"]),
                "damping": float(row["damping"]),
                "wet_level": float(row["wet_level"]),
                "dry_level": float(row["dry_level"]),
                "width": float(row["width"]),
                "freeze_mode": freeze_mode
            },
            audio_path=audio_path
        )
        examples.append(example)

    return examples


def _load_compressor_parameters(csv_path: Path, audio_dir: Path) -> List[SocialFXExample]:
    """Load compressor parameters from CSV file.

    Parses a CSV file containing compressor effect parameters. The CSV should
    have columns: id, description, instrument, threshold, ratio, attack,
    release, knee, makeup_gain.

    Args:
        csv_path: Path to the compressor parameters CSV file
        audio_dir: Path to directory containing audio samples

    Returns:
        List of SocialFXExample objects with effect_type="compressor"

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        pd.errors.EmptyDataError: If CSV file is empty
        pd.errors.ParserError: If CSV file is corrupted or malformed
        KeyError: If required columns are missing from CSV
        ValueError: If model validation fails or type conversion fails
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Compressor parameters CSV not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Compressor parameters CSV is empty: {csv_path}") from e
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Failed to parse compressor parameters CSV: {csv_path}") from e

    # Validate required columns
    required_cols = [
        'id', 'description', 'instrument', 'threshold', 'ratio',
        'attack', 'release', 'knee', 'makeup_gain'
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Compressor parameters CSV missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    examples = []
    for _, row in df.iterrows():
        # Construct audio path
        audio_path = str(audio_dir / f"{row['instrument']}.wav")

        example = SocialFXExample(
            id=int(row["id"]),
            description=str(row["description"]),
            instrument=str(row["instrument"]),
            effect_type="compressor",
            parameters={
                "threshold": float(row["threshold"]),
                "ratio": float(row["ratio"]),
                "attack": float(row["attack"]),
                "release": float(row["release"]),
                "knee": float(row["knee"]),
                "makeup_gain": float(row["makeup_gain"])
            },
            audio_path=audio_path
        )
        examples.append(example)

    return examples
