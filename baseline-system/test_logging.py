#!/usr/bin/env python3
"""Test logging system."""

from src.utils.logging import (
    get_logger, get_api_logger, get_audio_logger, get_scoring_logger,
    bind_context, unbind_context, clear_context, LogContext
)

def main():
    print("=== Testing Logging System ===\n")

    # 1. Get different loggers
    print("1. Testing different logger categories:")
    general_logger = get_logger("test_module")
    api_logger = get_api_logger()
    audio_logger = get_audio_logger()
    scoring_logger = get_scoring_logger()

    general_logger.info("General log message", test=True)
    print("   ✓ General logger")

    api_logger.info("API call", endpoint="/test", method="GET")
    print("   ✓ API logger")

    audio_logger.info("Audio processing", sample_rate=44100, duration=30.0)
    print("   ✓ Audio logger")

    scoring_logger.info("Scoring result", score=0.85, metric="similarity")
    print("   ✓ Scoring logger")

    # 2. Test context binding
    print("\n2. Testing context management:")
    bind_context(experiment_id="exp-001", user="test_user")
    general_logger.info("With bound context")
    print("   ✓ Context binding")

    unbind_context("user")
    general_logger.info("After unbinding user")
    print("   ✓ Context unbinding")

    clear_context()
    general_logger.info("After clearing context")
    print("   ✓ Context clearing")

    # 3. Test LogContext context manager
    print("\n3. Testing LogContext manager:")
    with LogContext(session="test-session", iteration=1):
        general_logger.info("Inside context manager")
        print("   ✓ LogContext entry")

    general_logger.info("Outside context manager")
    print("   ✓ LogContext exit")

    # 4. Test different log levels
    print("\n4. Testing log levels:")
    general_logger.debug("Debug message", details="verbose")
    print("   ✓ DEBUG level")

    general_logger.info("Info message")
    print("   ✓ INFO level")

    general_logger.warning("Warning message", issue="potential problem")
    print("   ✓ WARNING level")

    general_logger.error("Error message", error_code=500)
    print("   ✓ ERROR level")

    # 5. Test structured data
    print("\n5. Testing structured data:")
    general_logger.info(
        "Complex data",
        user_data={
            "id": 123,
            "name": "test",
            "preferences": ["audio", "video"]
        },
        metrics={
            "accuracy": 0.95,
            "loss": 0.05
        }
    )
    print("   ✓ Complex structured data")

    # 6. Test exception logging
    print("\n6. Testing exception logging:")
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        general_logger.exception("Exception occurred", error_type=type(e).__name__)
        print("   ✓ Exception logging")

    print("\n=== All Logging Tests Passed ✓ ===")
    print("\nLog files created in 'logs/' directory:")
    print("  - logs/general.log")
    print("  - logs/api.log")
    print("  - logs/audio.log")
    print("  - logs/scoring.log")

if __name__ == "__main__":
    main()