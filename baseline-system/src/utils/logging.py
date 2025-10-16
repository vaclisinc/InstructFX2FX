"""Structured logging setup using structlog.

This module provides:
- Structured JSON logging for production
- Pretty console logging for development
- Context-aware logging for experiment tracking
- Separate logs for API calls, audio processing, and scoring
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import structlog
from structlog.types import Processor


def configure_logging(
    level: str = "INFO",
    format: str = "json",
    output_dir: Optional[Path] = None,
    console_output: bool = True,
    file_output: bool = True,
) -> None:
    """Configure structured logging with structlog.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ('json' for structured, 'console' for pretty)
        output_dir: Directory for log files (required if file_output=True)
        console_output: Enable console logging
        file_output: Enable file logging

    Raises:
        ValueError: If file_output is True but output_dir is None
    """
    if file_output and output_dir is None:
        raise ValueError("output_dir must be provided when file_output is True")

    # Convert level string to logging level
    log_level = getattr(logging, level.upper())

    # Create output directory if needed
    if file_output and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout if console_output else None,
        level=log_level,
    )

    # Add file handlers if needed
    if file_output and output_dir:
        _setup_file_handlers(output_dir, log_level)

    # Define shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Add callsite information (useful for debugging)
    shared_processors.append(
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            }
        )
    )

    # Choose renderer based on format
    if format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _setup_file_handlers(output_dir: Path, log_level: int) -> None:
    """Set up file handlers for different log categories.

    Creates separate log files for:
    - general.log: All logs
    - api.log: LLM API calls
    - audio.log: Audio processing
    - scoring.log: Scoring system

    Args:
        output_dir: Directory for log files
        log_level: Logging level
    """
    root_logger = logging.getLogger()

    # General log file
    general_handler = logging.FileHandler(output_dir / "general.log")
    general_handler.setLevel(log_level)
    general_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(general_handler)

    # API calls log
    api_logger = logging.getLogger("api")
    api_handler = logging.FileHandler(output_dir / "api.log")
    api_handler.setLevel(log_level)
    api_handler.setFormatter(logging.Formatter("%(message)s"))
    api_logger.addHandler(api_handler)
    api_logger.propagate = False  # Don't propagate to root

    # Audio processing log
    audio_logger = logging.getLogger("audio")
    audio_handler = logging.FileHandler(output_dir / "audio.log")
    audio_handler.setLevel(log_level)
    audio_handler.setFormatter(logging.Formatter("%(message)s"))
    audio_logger.addHandler(audio_handler)
    audio_logger.propagate = False

    # Scoring system log
    scoring_logger = logging.getLogger("scoring")
    scoring_handler = logging.FileHandler(output_dir / "scoring.log")
    scoring_handler.setLevel(log_level)
    scoring_handler.setFormatter(logging.Formatter("%(message)s"))
    scoring_logger.addHandler(scoring_handler)
    scoring_logger.propagate = False


def get_logger(name: Optional[str] = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance.

    Args:
        name: Logger name (e.g., 'api', 'audio', 'scoring')
        **initial_values: Initial context values to bind to logger

    Returns:
        Configured BoundLogger instance

    Examples:
        >>> log = get_logger("api")
        >>> log.info("request_started", method="POST", endpoint="/generate")

        >>> log = get_logger("audio", experiment="baseline_001")
        >>> log.debug("processing_audio", file="sample.wav", duration=3.5)

        >>> log = get_logger("scoring")
        >>> log.info("score_calculated", audio_id="abc123", score=85.3)
    """
    logger = structlog.get_logger(name)
    if initial_values:
        logger = logger.bind(**initial_values)
    return logger


def bind_context(**kwargs: Any) -> None:
    """Bind context values that will be included in all subsequent logs.

    Context values are thread-local and will be automatically included
    in all log entries within the same thread/async context.

    Args:
        **kwargs: Context key-value pairs

    Examples:
        >>> bind_context(experiment_id="exp_001", user_id="user_123")
        >>> log = get_logger()
        >>> log.info("started")  # Will include experiment_id and user_id

        >>> bind_context(iteration=1)
        >>> log.info("iteration_complete")  # Will include iteration
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Remove specific keys from the context.

    Args:
        *keys: Context keys to remove

    Examples:
        >>> unbind_context("iteration")
        >>> unbind_context("experiment_id", "user_id")
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context values.

    Examples:
        >>> clear_context()  # Removes all bound context
    """
    structlog.contextvars.clear_contextvars()


class LogContext:
    """Context manager for temporary log context.

    Automatically binds context on entry and unbinds on exit.

    Examples:
        >>> with LogContext(experiment="exp_001", iteration=1):
        ...     log = get_logger()
        ...     log.info("processing")  # Includes experiment and iteration
        >>> # Context automatically cleared after exiting
    """

    def __init__(self, **context: Any):
        """Initialize log context.

        Args:
            **context: Context key-value pairs
        """
        self.context = context

    def __enter__(self) -> "LogContext":
        """Enter context and bind values."""
        bind_context(**self.context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context and unbind values."""
        unbind_context(*self.context.keys())


def log_function_call(logger: Optional[structlog.stdlib.BoundLogger] = None):
    """Decorator to automatically log function calls.

    Args:
        logger: Optional logger instance. If None, creates logger from function module.

    Examples:
        >>> @log_function_call()
        ... def process_audio(file_path: str, sample_rate: int = 44100):
        ...     # Process audio
        ...     return result

        >>> @log_function_call(get_logger("api"))
        ... def call_llm(prompt: str):
        ...     # Call LLM
        ...     return response
    """
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)

        def wrapper(*args, **kwargs):
            logger.debug(
                "function_called",
                function=func.__name__,
                args=args,
                kwargs=kwargs,
            )
            try:
                result = func(*args, **kwargs)
                logger.debug(
                    "function_completed",
                    function=func.__name__,
                )
                return result
            except Exception as e:
                logger.error(
                    "function_failed",
                    function=func.__name__,
                    error=str(e),
                    exc_info=True,
                )
                raise

        return wrapper
    return decorator


# Category-specific logger factories
def get_api_logger(**context: Any) -> structlog.stdlib.BoundLogger:
    """Get logger for API calls.

    Args:
        **context: Initial context values

    Returns:
        Configured logger for API category
    """
    return get_logger("api", **context)


def get_audio_logger(**context: Any) -> structlog.stdlib.BoundLogger:
    """Get logger for audio processing.

    Args:
        **context: Initial context values

    Returns:
        Configured logger for audio category
    """
    return get_logger("audio", **context)


def get_scoring_logger(**context: Any) -> structlog.stdlib.BoundLogger:
    """Get logger for scoring system.

    Args:
        **context: Initial context values

    Returns:
        Configured logger for scoring category
    """
    return get_logger("scoring", **context)
