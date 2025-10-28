"""Data types for audio processing pipeline."""

from pydantic import BaseModel, Field


class ProcessingResult(BaseModel):
    """Result of audio processing operation.

    Contains metrics about the input and output audio to assess
    the quality and impact of the applied effects.

    Attributes:
        input_rms: RMS level of input audio
        output_rms: RMS level of output audio
        peak_input: Peak amplitude of input audio
        peak_output: Peak amplitude of output audio
        clipping_detected: Whether clipping was detected in output
    """

    input_rms: float = Field(
        ge=0.0,
        description="Root mean square level of input audio"
    )
    output_rms: float = Field(
        ge=0.0,
        description="Root mean square level of output audio"
    )
    peak_input: float = Field(
        ge=0.0,
        le=2.0,  # Allow some headroom for pre-limited signals
        description="Peak amplitude of input audio"
    )
    peak_output: float = Field(
        ge=0.0,
        le=2.0,
        description="Peak amplitude of output audio"
    )
    clipping_detected: bool = Field(
        description="Whether clipping was detected in output audio"
    )

    def get_rms_change_db(self) -> float:
        """Calculate RMS change in decibels.

        Returns:
            RMS change in dB (positive means louder, negative means quieter)
        """
        import numpy as np
        if self.input_rms == 0:
            return 0.0
        return float(20 * np.log10(self.output_rms / self.input_rms))

    def get_peak_change_db(self) -> float:
        """Calculate peak level change in decibels.

        Returns:
            Peak change in dB (positive means louder, negative means quieter)
        """
        import numpy as np
        if self.peak_input == 0:
            return 0.0
        return float(20 * np.log10(self.peak_output / self.peak_input))
