"""Main dataset loader for SocialFX dataset with filtering and few-shot selection."""

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from judge_system.data.models import SocialFXExample, DatasetMetadata


logger = logging.getLogger(__name__)


class SocialFXDataset:
    """Main dataset class for loading and managing SocialFX examples.

    This class handles:
    - Directory structure verification
    - Loading CSV files containing EQ, Reverb, and Compressor parameters
    - Filtering examples by instrument and effect type
    - Selecting diverse examples for few-shot prompting
    - Generating dataset metadata and statistics

    Attributes:
        data_dir: Root directory containing the SocialFX dataset
        audio_dir: Directory containing audio files (guitar.wav, drums.wav, piano.wav)
        params_dir: Directory containing parameter CSV files
        examples: List of all loaded examples
        metadata: Dataset metadata and statistics
    """

    def __init__(self, data_dir: str = "data/socialfx"):
        """Initialize dataset loader with directory paths.

        Args:
            data_dir: Path to root directory containing SocialFX dataset.
                     Default is "data/socialfx".
        """
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.params_dir = self.data_dir / "parameters"

        self.examples: List[SocialFXExample] = []
        self.metadata: Optional[DatasetMetadata] = None
        self._cache: Dict = {}

    def load(self) -> None:
        """Load entire dataset into memory.

        This method:
        1. Verifies directory structure exists
        2. Loads all CSV files (EQ, Reverb, Compressor)
        3. Generates dataset metadata

        Raises:
            FileNotFoundError: If required files or directories are missing
            ValueError: If CSV files are malformed or contain invalid data
        """
        logger.info(f"Loading SocialFX dataset from {self.data_dir}")

        # Verify directory structure
        self._verify_structure()

        # Load all CSV files
        eq_examples = self._load_eq_parameters()
        reverb_examples = self._load_reverb_parameters()
        comp_examples = self._load_compressor_parameters()

        self.examples = eq_examples + reverb_examples + comp_examples

        # Generate metadata
        self.metadata = self._generate_metadata()

        logger.info(f"Loaded {len(self.examples)} examples")
        logger.info(
            f"Instruments: {self.metadata.instruments}, "
            f"Effect types: {self.metadata.effect_types}"
        )

    def _verify_structure(self) -> None:
        """Verify dataset directory structure exists.

        Checks for:
        - Audio directory with guitar.wav, drums.wav, piano.wav
        - Parameters directory with eq_params.csv, reverb_params.csv, compressor_params.csv

        Raises:
            FileNotFoundError: If any required files or directories are missing
        """
        required_files = [
            self.audio_dir / "guitar.wav",
            self.audio_dir / "drums.wav",
            self.audio_dir / "piano.wav",
            self.params_dir / "eq_params.csv",
            self.params_dir / "reverb_params.csv",
            self.params_dir / "compressor_params.csv"
        ]

        missing = [f for f in required_files if not f.exists()]
        if missing:
            missing_str = ", ".join(str(f) for f in missing)
            raise FileNotFoundError(
                f"Missing required files: {missing_str}"
            )

        logger.debug("Directory structure verified successfully")

    def _load_eq_parameters(self) -> List[SocialFXExample]:
        """Load EQ parameters from CSV.

        Parses eq_params.csv and extracts band parameters (frequency, gain, Q).
        Supports multiple bands with naming pattern: band1_freq, band1_gain, band1_q, etc.

        Returns:
            List of SocialFXExample instances with effect_type='eq'

        Raises:
            ValueError: If CSV format is invalid or contains invalid data
        """
        csv_path = self.params_dir / "eq_params.csv"
        logger.debug(f"Loading EQ parameters from {csv_path}")

        df = pd.read_csv(csv_path)

        examples = []
        for _, row in df.iterrows():
            # Parse band parameters
            bands = []
            band_idx = 1
            while f"band{band_idx}_freq" in row:
                bands.append({
                    "frequency": float(row[f"band{band_idx}_freq"]),
                    "gain": float(row[f"band{band_idx}_gain"]),
                    "q": float(row[f"band{band_idx}_q"])
                })
                band_idx += 1

            example = SocialFXExample(
                id=int(row["id"]),
                description=str(row["description"]),
                instrument=str(row["instrument"]),
                effect_type="eq",
                parameters={"bands": bands},
                audio_path=str(self.audio_dir / f"{row['instrument']}.wav")
            )
            examples.append(example)

        logger.debug(f"Loaded {len(examples)} EQ examples")
        return examples

    def _load_reverb_parameters(self) -> List[SocialFXExample]:
        """Load reverb parameters from CSV.

        Parses reverb_params.csv and extracts reverb parameters
        (room_size, damping, wet_level, dry_level, width, freeze_mode).

        Returns:
            List of SocialFXExample instances with effect_type='reverb'

        Raises:
            ValueError: If CSV format is invalid or contains invalid data
        """
        csv_path = self.params_dir / "reverb_params.csv"
        logger.debug(f"Loading reverb parameters from {csv_path}")

        df = pd.read_csv(csv_path)

        examples = []
        for _, row in df.iterrows():
            # Parse freeze_mode as boolean
            freeze_mode = row["freeze_mode"]
            if isinstance(freeze_mode, str):
                freeze_mode = freeze_mode.lower() in ['true', '1', 'yes']
            else:
                freeze_mode = bool(freeze_mode)

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
                audio_path=str(self.audio_dir / f"{row['instrument']}.wav")
            )
            examples.append(example)

        logger.debug(f"Loaded {len(examples)} reverb examples")
        return examples

    def _load_compressor_parameters(self) -> List[SocialFXExample]:
        """Load compressor parameters from CSV.

        Parses compressor_params.csv and extracts compressor parameters
        (threshold, ratio, attack, release, knee, makeup_gain).

        Returns:
            List of SocialFXExample instances with effect_type='compressor'

        Raises:
            ValueError: If CSV format is invalid or contains invalid data
        """
        csv_path = self.params_dir / "compressor_params.csv"
        logger.debug(f"Loading compressor parameters from {csv_path}")

        df = pd.read_csv(csv_path)

        examples = []
        for _, row in df.iterrows():
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
                audio_path=str(self.audio_dir / f"{row['instrument']}.wav")
            )
            examples.append(example)

        logger.debug(f"Loaded {len(examples)} compressor examples")
        return examples

    def get_examples(
        self,
        instrument: Optional[str] = None,
        effect_type: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[SocialFXExample]:
        """Get filtered examples based on criteria.

        Args:
            instrument: Filter by instrument type (guitar, drums, piano).
                       None means no filtering.
            effect_type: Filter by effect type (eq, reverb, compressor).
                        None means no filtering.
            limit: Maximum number of examples to return. None means return all.

        Returns:
            List of examples matching the filter criteria
        """
        filtered = self.examples

        if instrument:
            filtered = [e for e in filtered if e.instrument == instrument]

        if effect_type:
            filtered = [e for e in filtered if e.effect_type == effect_type]

        if limit:
            filtered = filtered[:limit]

        logger.debug(
            f"Filtered {len(filtered)} examples "
            f"(instrument={instrument}, effect_type={effect_type}, limit={limit})"
        )
        return filtered

    def get_few_shot_examples(
        self,
        effect_type: str,
        n_examples: int = 3,
        diverse: bool = True
    ) -> List[SocialFXExample]:
        """Get examples for few-shot prompting.

        Args:
            effect_type: Effect type to get examples for (eq, reverb, compressor)
            n_examples: Number of examples to return. Default is 3.
            diverse: If True, select diverse examples across instruments.
                    If False, return first n_examples. Default is True.

        Returns:
            List of examples suitable for few-shot prompting
        """
        examples = self.get_examples(effect_type=effect_type)

        if diverse:
            # Select diverse examples across instruments and descriptions
            selected = self._select_diverse_examples(examples, n_examples)
        else:
            selected = examples[:n_examples]

        logger.debug(
            f"Selected {len(selected)} few-shot examples for {effect_type} "
            f"(diverse={diverse})"
        )
        return selected

    def _select_diverse_examples(
        self,
        examples: List[SocialFXExample],
        n: int
    ) -> List[SocialFXExample]:
        """Select diverse examples across instruments using round-robin.

        This ensures few-shot examples cover multiple instruments rather than
        being heavily biased toward a single instrument.

        Args:
            examples: Pool of examples to select from
            n: Number of examples to select

        Returns:
            List of n diverse examples selected via round-robin across instruments
        """
        selected = []
        instruments = list(set(e.instrument for e in examples))

        if not instruments:
            return []

        # Round-robin across instruments
        for i in range(n):
            instrument = instruments[i % len(instruments)]
            instrument_examples = [
                e for e in examples
                if e.instrument == instrument and e not in selected
            ]
            if instrument_examples:
                selected.append(instrument_examples[0])

            # If we've exhausted all instruments, stop early
            if len(selected) < i + 1:
                break

        return selected[:n]

    def _generate_metadata(self) -> DatasetMetadata:
        """Generate dataset statistics and metadata.

        Computes:
        - Total example count
        - Unique instruments
        - Unique effect types
        - Example count per effect type
        - Parameter ranges per effect type

        Returns:
            DatasetMetadata instance with computed statistics
        """
        instruments = sorted(list(set(e.instrument for e in self.examples)))
        effect_types = sorted(list(set(e.effect_type for e in self.examples)))

        description_count = {
            effect: len([e for e in self.examples if e.effect_type == effect])
            for effect in effect_types
        }

        parameter_ranges = self._calculate_parameter_ranges()

        metadata = DatasetMetadata(
            total_examples=len(self.examples),
            instruments=instruments,
            effect_types=effect_types,
            description_count=description_count,
            parameter_ranges=parameter_ranges
        )

        logger.debug(f"Generated metadata: {metadata.total_examples} examples")
        return metadata

    def _calculate_parameter_ranges(self) -> Dict[str, Dict[str, tuple]]:
        """Calculate min/max ranges for each numeric parameter by effect type.

        Analyzes all examples to determine the range of values for each parameter.
        Useful for validation and understanding parameter distributions.

        Returns:
            Dictionary mapping effect_type -> parameter_name -> (min, max)
            Example: {"eq": {"gain": (-10.0, 10.0)}, "reverb": {"room_size": (0.0, 1.0)}}
        """
        ranges: Dict[str, Dict[str, tuple]] = {}

        for effect_type in set(e.effect_type for e in self.examples):
            effect_examples = [e for e in self.examples if e.effect_type == effect_type]
            ranges[effect_type] = {}

            if effect_type == "eq":
                # For EQ, analyze band parameters
                all_freqs = []
                all_gains = []
                all_qs = []

                for example in effect_examples:
                    bands = example.parameters.get("bands", [])
                    for band in bands:
                        all_freqs.append(band["frequency"])
                        all_gains.append(band["gain"])
                        all_qs.append(band["q"])

                if all_freqs:
                    ranges[effect_type]["frequency"] = (min(all_freqs), max(all_freqs))
                if all_gains:
                    ranges[effect_type]["gain"] = (min(all_gains), max(all_gains))
                if all_qs:
                    ranges[effect_type]["q"] = (min(all_qs), max(all_qs))

            elif effect_type == "reverb":
                # For reverb, analyze each parameter
                params_to_track = ["room_size", "damping", "wet_level", "dry_level", "width"]

                for param in params_to_track:
                    values = [
                        float(e.parameters[param])
                        for e in effect_examples
                        if param in e.parameters
                    ]
                    if values:
                        ranges[effect_type][param] = (min(values), max(values))

            elif effect_type == "compressor":
                # For compressor, analyze each parameter
                params_to_track = ["threshold", "ratio", "attack", "release", "knee", "makeup_gain"]

                for param in params_to_track:
                    values = [
                        float(e.parameters[param])
                        for e in effect_examples
                        if param in e.parameters
                    ]
                    if values:
                        ranges[effect_type][param] = (min(values), max(values))

        logger.debug(f"Calculated parameter ranges for {len(ranges)} effect types")
        return ranges
