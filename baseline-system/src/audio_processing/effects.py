"""Effect chain builder for pedalboard integration."""

from typing import List
from pedalboard import (
    Pedalboard,
    Reverb,
    Compressor,
    PeakFilter,
)
import structlog

from ..models.parameters.effect_chain import EffectChain
from ..models.parameters.eq import EQParameters
from ..models.parameters.reverb import ReverbParameters
from ..models.parameters.compressor import CompressorParameters
from .normalizer import ParameterNormalizer

logger = structlog.get_logger(__name__)


class EffectChainBuilder:
    """Builds pedalboard effect chains from parameter models.

    This class converts high-level effect parameters (from Pydantic models)
    into actual pedalboard effect objects that can process audio.
    """

    def __init__(self):
        """Initialize the effect chain builder."""
        self.effect_map = {
            "eq": self.create_eq,
            "reverb": self.create_reverb,
            "compressor": self.create_compressor
        }
        logger.info("EffectChainBuilder initialized", effect_types=list(self.effect_map.keys()))

    def build_chain(self, effect_chain: EffectChain) -> Pedalboard:
        """Build pedalboard effect chain from parameters.

        Args:
            effect_chain: EffectChain model containing effect parameters

        Returns:
            Pedalboard object with effects in specified order

        Raises:
            ValueError: If effect type is not supported
        """
        logger.info(
            "Building effect chain",
            num_effects=len(effect_chain.effects),
            order=effect_chain.order
        )

        board = Pedalboard()

        for i, effect_params in enumerate(effect_chain.effects):
            effect_type = effect_params.effect_type

            if effect_type not in self.effect_map:
                raise ValueError(
                    f"Unsupported effect type '{effect_type}'. "
                    f"Supported types: {list(self.effect_map.keys())}"
                )

            logger.debug(
                "Adding effect to chain",
                position=i,
                effect_type=effect_type
            )

            # Get the factory method for this effect type
            factory = self.effect_map[effect_type]

            # Create the effect(s) and add to board
            effects = factory(effect_params)

            # Handle single effect or list of effects (e.g., EQ bands)
            if isinstance(effects, list):
                for effect in effects:
                    board.append(effect)
            else:
                board.append(effects)

        logger.info(
            "Effect chain built successfully",
            total_plugins=len(board)
        )

        return board

    def create_eq(self, params: EQParameters) -> List[PeakFilter]:
        """Create EQ filters from parameters.

        Args:
            params: EQ parameters including bands

        Returns:
            List of PeakFilter objects, one per band
        """
        logger.debug(
            "Creating EQ filters",
            num_bands=len(params.bands),
            eq_type=params.eq_type
        )

        filters = []
        for i, band in enumerate(params.bands):
            # Normalize Q-factor for pedalboard
            normalized_q = ParameterNormalizer.normalize_eq_q(band.q)

            filter = PeakFilter(
                cutoff_frequency_hz=band.frequency,
                gain_db=band.gain,
                q=normalized_q
            )

            logger.debug(
                "Created EQ band",
                band_index=i,
                frequency=band.frequency,
                gain=band.gain,
                q_original=band.q,
                q_normalized=normalized_q
            )

            filters.append(filter)

        return filters

    def create_reverb(self, params: ReverbParameters) -> Reverb:
        """Create reverb from parameters.

        Args:
            params: Reverb parameters

        Returns:
            Reverb effect object
        """
        # Normalize room size for pedalboard
        normalized_room_size = ParameterNormalizer.normalize_reverb_room_size(
            params.room_size
        )

        logger.debug(
            "Creating reverb effect",
            room_size_original=params.room_size,
            room_size_normalized=normalized_room_size,
            damping=params.damping,
            wet_level=params.wet_level,
            dry_level=params.dry_level,
            width=params.width,
            freeze_mode=params.freeze_mode
        )

        return Reverb(
            room_size=normalized_room_size,
            damping=params.damping,
            wet_level=params.wet_level,
            dry_level=params.dry_level,
            width=params.width,
            freeze_mode=params.freeze_mode
        )

    def create_compressor(self, params: CompressorParameters) -> Compressor:
        """Create compressor from parameters.

        Args:
            params: Compressor parameters

        Returns:
            Compressor effect object
        """
        logger.debug(
            "Creating compressor effect",
            threshold=params.threshold,
            ratio=params.ratio,
            attack=params.attack,
            release=params.release
        )

        return Compressor(
            threshold_db=params.threshold,
            ratio=params.ratio,
            attack_ms=params.attack,
            release_ms=params.release
        )
