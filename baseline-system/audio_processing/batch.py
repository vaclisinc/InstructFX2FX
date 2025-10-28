"""Batch processing for multiple audio files."""

import os
from pathlib import Path
from typing import List
import structlog

from src.models.parameters.effect_chain import EffectChain
from .processor import AudioProcessor
from .types import ProcessingResult


logger = structlog.get_logger(__name__)


class BatchProcessor:
    """Batch processor for multiple audio files.

    Handles processing multiple audio files with the same effect chain,
    with automatic output directory creation and progress logging.

    Attributes:
        processor: AudioProcessor instance for single-file processing

    Example:
        >>> processor = AudioProcessor(sample_rate=44100)
        >>> batch = BatchProcessor(processor)
        >>> results = batch.process_batch(
        ...     input_files=["audio1.wav", "audio2.wav"],
        ...     output_dir="processed/",
        ...     effect_chain=my_effect_chain
        ... )
        >>> print(f"Processed {len(results)} files")
    """

    def __init__(self, processor: AudioProcessor):
        """Initialize batch processor.

        Args:
            processor: AudioProcessor instance to use for processing
        """
        self.processor = processor
        logger.info("BatchProcessor initialized")

    def process_batch(
        self,
        input_files: List[str],
        output_dir: str,
        effect_chain: EffectChain,
        preserve_structure: bool = False,
    ) -> List[ProcessingResult]:
        """Process multiple files with same effect chain.

        Processes a list of audio files, saving them to the output directory
        with the same filenames. Creates the output directory if it doesn't exist.

        Args:
            input_files: List of input file paths
            output_dir: Directory to save processed files
            effect_chain: EffectChain to apply to all files
            preserve_structure: If True, preserve subdirectory structure from input paths
                               (default: False, all outputs flat in output_dir)

        Returns:
            List of ProcessingResult objects, one per input file

        Raises:
            ValueError: If input_files is empty
            AudioLoadError: If any input file cannot be loaded
            AudioSaveError: If any output file cannot be saved

        Example:
            >>> import glob
            >>> processor = AudioProcessor()
            >>> batch = BatchProcessor(processor)
            >>>
            >>> # Process all WAV files in a directory
            >>> input_files = glob.glob("inputs/*.wav")
            >>> results = batch.process_batch(
            ...     input_files=input_files,
            ...     output_dir="outputs/",
            ...     effect_chain=my_chain
            ... )
            >>>
            >>> # Check results
            >>> for i, result in enumerate(results):
            ...     print(f"File {i}: RMS change {result.get_rms_change_db():.2f} dB")
        """
        if not input_files:
            raise ValueError("input_files cannot be empty")

        logger.info(
            "Starting batch processing",
            num_files=len(input_files),
            output_dir=output_dir,
            preserve_structure=preserve_structure
        )

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.debug("Output directory ready", path=str(output_path))

        results = []
        failed_files = []

        for i, input_file in enumerate(input_files):
            logger.info(
                "Processing file",
                index=i + 1,
                total=len(input_files),
                input_file=input_file
            )

            try:
                # Determine output path
                if preserve_structure:
                    # Preserve directory structure
                    input_path = Path(input_file)
                    rel_path = input_path.name  # Just filename for now (can enhance)
                    output_file = output_path / rel_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                else:
                    # Flat output structure
                    output_file = output_path / os.path.basename(input_file)

                # Process the file
                result = self.processor.process(
                    input_path=input_file,
                    output_path=str(output_file),
                    effect_chain=effect_chain
                )

                results.append(result)

                logger.info(
                    "File processed successfully",
                    index=i + 1,
                    input_file=input_file,
                    output_file=str(output_file),
                    rms_change_db=result.get_rms_change_db()
                )

            except Exception as e:
                logger.error(
                    "Failed to process file",
                    index=i + 1,
                    input_file=input_file,
                    error=str(e),
                    exc_info=True
                )
                failed_files.append((input_file, str(e)))

                # Continue processing remaining files rather than failing completely
                continue

        # Log summary
        logger.info(
            "Batch processing complete",
            total_files=len(input_files),
            successful=len(results),
            failed=len(failed_files)
        )

        if failed_files:
            logger.warning(
                "Some files failed to process",
                failed_files=[f[0] for f in failed_files]
            )

        return results

    def process_batch_with_summary(
        self,
        input_files: List[str],
        output_dir: str,
        effect_chain: EffectChain,
    ) -> tuple[List[ProcessingResult], dict]:
        """Process batch and return summary statistics.

        Same as process_batch() but also computes aggregate statistics
        across all processed files.

        Args:
            input_files: List of input file paths
            output_dir: Directory to save processed files
            effect_chain: EffectChain to apply to all files

        Returns:
            Tuple of (results_list, summary_dict) where summary_dict contains:
                - total_files: Total number of files processed
                - avg_rms_change_db: Average RMS change in dB
                - avg_peak_change_db: Average peak change in dB
                - files_with_clipping: Number of files with clipping
                - clipping_rate: Percentage of files with clipping

        Example:
            >>> batch = BatchProcessor(AudioProcessor())
            >>> results, summary = batch.process_batch_with_summary(
            ...     input_files=["a.wav", "b.wav"],
            ...     output_dir="out/",
            ...     effect_chain=chain
            ... )
            >>> print(f"Average RMS change: {summary['avg_rms_change_db']:.2f} dB")
            >>> print(f"Clipping rate: {summary['clipping_rate']:.1f}%")
        """
        # Process batch
        results = self.process_batch(
            input_files=input_files,
            output_dir=output_dir,
            effect_chain=effect_chain
        )

        if not results:
            logger.warning("No files successfully processed")
            return results, {
                "total_files": 0,
                "avg_rms_change_db": 0.0,
                "avg_peak_change_db": 0.0,
                "files_with_clipping": 0,
                "clipping_rate": 0.0
            }

        # Compute summary statistics
        rms_changes = [r.get_rms_change_db() for r in results]
        peak_changes = [r.get_peak_change_db() for r in results]
        files_with_clipping = sum(1 for r in results if r.clipping_detected)

        summary = {
            "total_files": len(results),
            "avg_rms_change_db": sum(rms_changes) / len(rms_changes),
            "avg_peak_change_db": sum(peak_changes) / len(peak_changes),
            "files_with_clipping": files_with_clipping,
            "clipping_rate": (files_with_clipping / len(results)) * 100
        }

        logger.info(
            "Batch summary computed",
            **summary
        )

        return results, summary

    def process_pairs(
        self,
        file_pairs: List[tuple[str, str]],
        effect_chain: EffectChain,
    ) -> List[ProcessingResult]:
        """Process files with explicit input/output path pairs.

        Useful when you want full control over output paths rather than
        using a common output directory.

        Args:
            file_pairs: List of (input_path, output_path) tuples
            effect_chain: EffectChain to apply to all files

        Returns:
            List of ProcessingResult objects, one per pair

        Example:
            >>> pairs = [
            ...     ("input1.wav", "custom_output1.wav"),
            ...     ("input2.wav", "different_dir/output2.wav"),
            ... ]
            >>> batch = BatchProcessor(AudioProcessor())
            >>> results = batch.process_pairs(pairs, effect_chain)
        """
        if not file_pairs:
            raise ValueError("file_pairs cannot be empty")

        logger.info(
            "Starting paired batch processing",
            num_pairs=len(file_pairs)
        )

        results = []

        for i, (input_path, output_path) in enumerate(file_pairs):
            logger.info(
                "Processing pair",
                index=i + 1,
                total=len(file_pairs),
                input_path=input_path,
                output_path=output_path
            )

            try:
                # Create output directory if needed
                output_dir = Path(output_path).parent
                output_dir.mkdir(parents=True, exist_ok=True)

                # Process
                result = self.processor.process(
                    input_path=input_path,
                    output_path=output_path,
                    effect_chain=effect_chain
                )

                results.append(result)

                logger.info(
                    "Pair processed successfully",
                    index=i + 1,
                    rms_change_db=result.get_rms_change_db()
                )

            except Exception as e:
                logger.error(
                    "Failed to process pair",
                    index=i + 1,
                    input_path=input_path,
                    output_path=output_path,
                    error=str(e),
                    exc_info=True
                )
                # Continue processing remaining pairs
                continue

        logger.info(
            "Paired batch processing complete",
            total_pairs=len(file_pairs),
            successful=len(results)
        )

        return results
