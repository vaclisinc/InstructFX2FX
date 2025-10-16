"""Parameter normalizer for JSON-to-pedalboard conversion."""

import structlog

logger = structlog.get_logger(__name__)


class ParameterNormalizer:
    """Normalizes effect parameters from JSON schema to pedalboard format.

    Some parameters need conversion between our JSON schema and pedalboard's
    expected ranges or formats. This class provides static methods for these
    conversions to ensure compatibility.
    """

    @staticmethod
    def normalize_eq_q(q: float) -> float:
        """Normalize Q-factor for pedalboard.

        Our schema uses Q in range [0.1, 10.0], which maps well to pedalboard's
        expectations, but we apply a slight scaling factor for optimal behavior.

        The Q-factor in EQ controls the bandwidth of the filter. A higher Q means
        a narrower bandwidth (more selective). The 0.707 factor helps align our
        scale with pedalboard's internal processing.

        Args:
            q: Q-factor from our schema (0.1 to 10.0)

        Returns:
            Normalized Q-factor for pedalboard

        Raises:
            ValueError: If Q is outside valid range
        """
        if q < 0.1 or q > 10.0:
            raise ValueError(f"Q-factor {q} is outside valid range [0.1, 10.0]")

        # Apply scaling factor for pedalboard compatibility
        # This factor (0.707) aligns our Q scale with pedalboard's processing
        normalized = q * 0.707

        logger.debug(
            "Normalized EQ Q-factor",
            original_q=q,
            normalized_q=normalized,
            scaling_factor=0.707
        )

        return normalized

    @staticmethod
    def normalize_reverb_room_size(size: float) -> float:
        """Map room_size to pedalboard range.

        Our schema uses room_size in [0, 1], but pedalboard's reverb works best
        when avoiding extreme values. We map [0, 1] to [0.1, 0.9] to prevent
        potential artifacts at the extremes.

        Args:
            size: Room size from our schema (0.0 to 1.0)

        Returns:
            Normalized room size for pedalboard

        Raises:
            ValueError: If room size is outside valid range
        """
        if size < 0.0 or size > 1.0:
            raise ValueError(f"Room size {size} is outside valid range [0.0, 1.0]")

        # Map [0, 1] to [0.1, 0.9] to avoid extremes
        # Formula: output = input * 0.8 + 0.1
        # When input=0: output=0.1
        # When input=1: output=0.9
        normalized = size * 0.8 + 0.1

        logger.debug(
            "Normalized reverb room size",
            original_size=size,
            normalized_size=normalized
        )

        return normalized

    @staticmethod
    def ms_to_seconds(ms: float) -> float:
        """Convert milliseconds to seconds.

        Some pedalboard effects may require time values in seconds instead of
        milliseconds. This is a simple utility for that conversion.

        Args:
            ms: Time in milliseconds

        Returns:
            Time in seconds

        Raises:
            ValueError: If time is negative
        """
        if ms < 0:
            raise ValueError(f"Time {ms} cannot be negative")

        seconds = ms / 1000.0

        logger.debug(
            "Converted time units",
            milliseconds=ms,
            seconds=seconds
        )

        return seconds

    @staticmethod
    def seconds_to_ms(seconds: float) -> float:
        """Convert seconds to milliseconds.

        Some pedalboard effects may provide time values in seconds that need
        to be converted back to milliseconds for our schema.

        Args:
            seconds: Time in seconds

        Returns:
            Time in milliseconds

        Raises:
            ValueError: If time is negative
        """
        if seconds < 0:
            raise ValueError(f"Time {seconds} cannot be negative")

        ms = seconds * 1000.0

        logger.debug(
            "Converted time units",
            seconds=seconds,
            milliseconds=ms
        )

        return ms

    @staticmethod
    def db_to_linear(db: float) -> float:
        """Convert decibel value to linear gain.

        Useful for converting between dB (logarithmic) and linear scales.

        Args:
            db: Value in decibels

        Returns:
            Linear gain value
        """
        import math

        linear = math.pow(10, db / 20.0)

        logger.debug(
            "Converted dB to linear",
            db=db,
            linear=linear
        )

        return linear

    @staticmethod
    def linear_to_db(linear: float) -> float:
        """Convert linear gain to decibel value.

        Useful for converting between linear and dB (logarithmic) scales.

        Args:
            linear: Linear gain value

        Returns:
            Value in decibels

        Raises:
            ValueError: If linear value is not positive
        """
        import math

        if linear <= 0:
            raise ValueError(f"Linear value {linear} must be positive")

        db = 20.0 * math.log10(linear)

        logger.debug(
            "Converted linear to dB",
            linear=linear,
            db=db
        )

        return db

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value to a specified range.

        Utility function to ensure a value stays within bounds.

        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Clamped value

        Raises:
            ValueError: If min_val > max_val
        """
        if min_val > max_val:
            raise ValueError(f"min_val {min_val} cannot be greater than max_val {max_val}")

        clamped = max(min_val, min(max_val, value))

        if clamped != value:
            logger.debug(
                "Value clamped",
                original=value,
                clamped=clamped,
                min=min_val,
                max=max_val
            )

        return clamped
