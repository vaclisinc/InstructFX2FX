"""Custom exceptions for audio processing.

This module defines custom exception classes used throughout the audio processing pipeline.
Each exception provides specific information about the type of error that occurred.
"""


class AudioProcessingError(Exception):
    """Base exception for all audio processing errors.

    All custom exceptions in the audio processing module inherit from this class,
    making it easy to catch all audio-related errors with a single except clause.
    """
    pass


class AudioValidationError(AudioProcessingError):
    """Exception raised when audio validation fails.

    This exception is raised when audio data fails validation checks such as:
    - Audio exceeds normalized range [-1.0, 1.0]
    - Sample rate is below minimum requirements
    - Audio contains NaN or infinite values
    - Audio shape is invalid (e.g., empty arrays)

    Args:
        message: Description of the validation failure

    Examples:
        >>> raise AudioValidationError("Audio exceeds normalized range")
        >>> raise AudioValidationError(f"Sample rate {sr} below minimum 44100")
    """
    pass


class AudioLoadError(AudioProcessingError):
    """Exception raised when audio file loading fails.

    This exception is raised when:
    - File does not exist
    - File format is not supported
    - File is corrupted or malformed
    - Insufficient permissions to read file

    Args:
        message: Description of the load failure
        file_path: Path to the file that failed to load

    Examples:
        >>> raise AudioLoadError("File not found", file_path="/path/to/audio.wav")
        >>> raise AudioLoadError("Unsupported format", file_path="/path/to/audio.xyz")
    """

    def __init__(self, message: str, file_path: str = None):
        """Initialize AudioLoadError with message and optional file path.

        Args:
            message: Error description
            file_path: Optional path to the file that failed to load
        """
        self.file_path = file_path
        full_message = f"{message}"
        if file_path:
            full_message = f"{message}: {file_path}"
        super().__init__(full_message)


class AudioSaveError(AudioProcessingError):
    """Exception raised when audio file saving fails.

    This exception is raised when:
    - Unable to write to output path
    - Insufficient disk space
    - Invalid output format
    - Insufficient permissions to write file

    Args:
        message: Description of the save failure
        file_path: Path where save was attempted

    Examples:
        >>> raise AudioSaveError("Permission denied", file_path="/protected/output.wav")
        >>> raise AudioSaveError("Disk full", file_path="/path/to/output.wav")
    """

    def __init__(self, message: str, file_path: str = None):
        """Initialize AudioSaveError with message and optional file path.

        Args:
            message: Error description
            file_path: Optional path where save was attempted
        """
        self.file_path = file_path
        full_message = f"{message}"
        if file_path:
            full_message = f"{message}: {file_path}"
        super().__init__(full_message)


class UnsupportedFormatError(AudioProcessingError):
    """Exception raised when audio format is not supported.

    This exception is raised when attempting to load or save audio in a format
    that is not supported by the audio processing pipeline.

    Args:
        format: The unsupported format
        supported_formats: List of supported formats

    Examples:
        >>> raise UnsupportedFormatError(".xyz", [".wav", ".mp3", ".flac"])
    """

    def __init__(self, format: str, supported_formats: list[str] = None):
        """Initialize UnsupportedFormatError with format information.

        Args:
            format: The unsupported format (e.g., ".xyz")
            supported_formats: Optional list of supported formats
        """
        self.format = format
        self.supported_formats = supported_formats or [".wav", ".mp3", ".flac"]

        message = f"Unsupported format: {format}. Supported formats: {', '.join(self.supported_formats)}"
        super().__init__(message)


class AudioClippingError(AudioProcessingError):
    """Exception raised when audio contains severe clipping.

    This exception can be raised when audio contains clipping that cannot be
    recovered through limiting or other processing techniques.

    Args:
        message: Description of the clipping issue
        peak_value: Maximum absolute value found in the audio

    Examples:
        >>> raise AudioClippingError("Severe clipping detected", peak_value=1.5)
    """

    def __init__(self, message: str, peak_value: float = None):
        """Initialize AudioClippingError with clipping information.

        Args:
            message: Error description
            peak_value: Optional maximum absolute value that was detected
        """
        self.peak_value = peak_value
        full_message = f"{message}"
        if peak_value is not None:
            full_message = f"{message} (peak: {peak_value:.3f})"
        super().__init__(full_message)
