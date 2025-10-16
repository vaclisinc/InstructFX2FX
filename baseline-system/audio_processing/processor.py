"""Main audio processing pipeline integrating loader, effects, and metrics."""

import numpy as np
import structlog

from src.processing.io import AudioLoader
from src.audio_processing.effects import EffectChainBuilder
from src.models.parameters.effect_chain import EffectChain
from .metrics import AudioMetrics
from .types import ProcessingResult


logger = structlog.get_logger(__name__)


class AudioProcessor:
    """Main audio processing pipeline.

    Integrates audio loading, effect chain processing, and metrics computation
    into a single end-to-end workflow. Handles the complete processing pipeline
    from loading an audio file to saving the processed result with quality validation.

    Attributes:
        loader: AudioLoader instance for I/O operations
        chain_builder: EffectChainBuilder for creating effect chains
        sample_rate: Target sample rate for audio processing

    Example:
        >>> from src.models.parameters.effect_chain import EffectChain
        >>> processor = AudioProcessor(sample_rate=44100)
        >>> result = processor.process(
        ...     input_path="input.wav",
        ...     output_path="output.wav",
        ...     effect_chain=my_effect_chain
        ... )
        >>> print(f"RMS change: {result.get_rms_change_db():.2f} dB")
    """

    def __init__(self, sample_rate: int = 44100):
        """Initialize audio processor.

        Args:
            sample_rate: Target sample rate for audio processing (default: 44100 Hz)
        """
        self.sample_rate = sample_rate
        self.loader = AudioLoader(sample_rate=sample_rate)
        self.chain_builder = EffectChainBuilder()

        logger.info(
            "AudioProcessor initialized",
            sample_rate=sample_rate
        )

    def process(
        self,
        input_path: str,
        output_path: str,
        effect_chain: EffectChain,
    ) -> ProcessingResult:
        """Process audio file with effect chain.

        Complete pipeline that:
        1. Loads audio using AudioLoader
        2. Builds effect chain using EffectChainBuilder
        3. Processes audio with pedalboard
        4. Validates and saves using AudioLoader
        5. Returns processing metrics

        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            effect_chain: EffectChain model with effect parameters

        Returns:
            ProcessingResult with input/output metrics

        Raises:
            AudioLoadError: If input file cannot be loaded
            AudioSaveError: If output file cannot be saved
            AudioValidationError: If audio validation fails
            ValueError: If effect chain is invalid

        Example:
            >>> from src.models.parameters.effect_chain import EffectChain
            >>> from src.models.parameters.reverb import ReverbParameters
            >>>
            >>> # Create effect chain
            >>> reverb = ReverbParameters(
            ...     room_size=0.7,
            ...     damping=0.5,
            ...     wet_level=0.3,
            ...     dry_level=0.7
            ... )
            >>> chain = EffectChain(
            ...     description="Add reverb",
            ...     effects=[reverb],
            ...     order=["reverb"]
            ... )
            >>>
            >>> # Process audio
            >>> processor = AudioProcessor()
            >>> result = processor.process("input.wav", "output.wav", chain)
            >>> print(f"Processing complete: {result.clipping_detected=}")
        """
        logger.info(
            "Starting audio processing",
            input_path=input_path,
            output_path=output_path,
            num_effects=len(effect_chain.effects),
            effect_order=effect_chain.order
        )

        # Step 1: Load audio
        logger.debug("Loading audio", input_path=input_path)
        audio, sr = self.loader.load(input_path)

        # Compute input metrics
        input_rms = AudioMetrics.compute_rms(audio)
        peak_input = AudioMetrics.compute_peak(audio)

        logger.debug(
            "Input audio metrics",
            rms=input_rms,
            peak=peak_input,
            shape=audio.shape
        )

        # Step 2: Build effect chain
        logger.debug("Building effect chain")
        board = self.chain_builder.build_chain(effect_chain)

        # Step 3: Process audio with pedalboard
        logger.debug("Applying effects")
        try:
            # Transpose if stereo (pedalboard expects channels-last)
            if audio.ndim == 2:
                audio_for_processing = audio.T
            else:
                audio_for_processing = audio

            # Process with pedalboard
            processed = board(audio_for_processing, sr)

            # Transpose back if stereo
            if processed.ndim == 2:
                processed = processed.T

            logger.debug(
                "Effects applied",
                output_shape=processed.shape
            )

        except Exception as e:
            logger.error(
                "Effect processing failed",
                error=str(e),
                exc_info=True
            )
            raise RuntimeError(f"Failed to apply effects: {str(e)}")

        # Compute output metrics
        output_rms = AudioMetrics.compute_rms(processed)
        peak_output = AudioMetrics.compute_peak(processed)
        clipping_detected = AudioMetrics.has_clipping(processed)

        logger.debug(
            "Output audio metrics",
            rms=output_rms,
            peak=peak_output,
            clipping_detected=clipping_detected
        )

        # Step 4: Save processed audio
        logger.debug("Saving processed audio", output_path=output_path)
        self.loader.save(processed, output_path, sr)

        # Step 5: Create and return result
        result = ProcessingResult(
            input_rms=input_rms,
            output_rms=output_rms,
            peak_input=peak_input,
            peak_output=peak_output,
            clipping_detected=clipping_detected
        )

        logger.info(
            "Audio processing complete",
            input_path=input_path,
            output_path=output_path,
            rms_change_db=result.get_rms_change_db(),
            peak_change_db=result.get_peak_change_db(),
            clipping_detected=clipping_detected
        )

        return result

    def process_with_monitoring(
        self,
        input_path: str,
        output_path: str,
        effect_chain: EffectChain,
    ) -> tuple[ProcessingResult, dict]:
        """Process audio with additional monitoring metrics.

        Same as process() but also computes additional spectral metrics
        for detailed analysis.

        Args:
            input_path: Path to input audio file
            output_path: Path to save processed audio
            effect_chain: EffectChain model with effect parameters

        Returns:
            Tuple of (ProcessingResult, monitoring_dict) where monitoring_dict contains:
                - input_spectral_centroid: Spectral centroid of input
                - output_spectral_centroid: Spectral centroid of output
                - spectral_centroid_change: Change in spectral centroid

        Example:
            >>> processor = AudioProcessor()
            >>> result, monitoring = processor.process_with_monitoring(
            ...     "input.wav", "output.wav", effect_chain
            ... )
            >>> print(f"Brightness change: {monitoring['spectral_centroid_change']:.2f} Hz")
        """
        logger.info(
            "Starting audio processing with monitoring",
            input_path=input_path,
            output_path=output_path
        )

        # Load audio
        audio, sr = self.loader.load(input_path)

        # Compute input spectral metrics
        input_centroid = AudioMetrics.compute_spectral_centroid(audio, sr)

        # Process audio (same as process method)
        board = self.chain_builder.build_chain(effect_chain)

        if audio.ndim == 2:
            audio_for_processing = audio.T
        else:
            audio_for_processing = audio

        processed = board(audio_for_processing, sr)

        if processed.ndim == 2:
            processed = processed.T

        # Compute output spectral metrics
        output_centroid = AudioMetrics.compute_spectral_centroid(processed, sr)

        # Compute standard metrics
        input_rms = AudioMetrics.compute_rms(audio)
        output_rms = AudioMetrics.compute_rms(processed)
        peak_input = AudioMetrics.compute_peak(audio)
        peak_output = AudioMetrics.compute_peak(processed)
        clipping_detected = AudioMetrics.has_clipping(processed)

        # Save
        self.loader.save(processed, output_path, sr)

        # Create result
        result = ProcessingResult(
            input_rms=input_rms,
            output_rms=output_rms,
            peak_input=peak_input,
            peak_output=peak_output,
            clipping_detected=clipping_detected
        )

        # Create monitoring dict
        monitoring = {
            "input_spectral_centroid": input_centroid,
            "output_spectral_centroid": output_centroid,
            "spectral_centroid_change": output_centroid - input_centroid
        }

        logger.info(
            "Audio processing with monitoring complete",
            rms_change_db=result.get_rms_change_db(),
            spectral_centroid_change=monitoring["spectral_centroid_change"]
        )

        return result, monitoring
