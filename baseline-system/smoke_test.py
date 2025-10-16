#!/usr/bin/env python
"""Smoke test to verify environment setup is working correctly.

This script performs basic checks to ensure:
- All required dependencies can be imported
- Configuration loader works correctly
- Logging system initializes properly
- API keys are loaded from .env file
- Directory structure is in place

Run this after initial setup to verify everything is configured correctly.

Usage:
    python smoke_test.py
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(check_name: str, passed: bool, details: str = "") -> None:
    """Print a formatted test result."""
    status = "✓" if passed else "✗"
    status_text = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"

    print(f"{color}{status} {check_name:50} [{status_text}]{reset}")
    if details:
        print(f"  {details}")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version meets requirements."""
    version = sys.version_info
    required_version = (3, 9)

    if version >= required_version:
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)"


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if all required dependencies can be imported."""
    required_modules = [
        "anthropic",
        "openai",
        "librosa",
        "soundfile",
        "pedalboard",
        "numpy",
        "pydantic",
        "yaml",
        "structlog",
        "datasets",
        "pytest",
        "dotenv",
    ]

    missing_modules = []

    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)

    passed = len(missing_modules) == 0
    return passed, missing_modules


def check_directory_structure() -> Tuple[bool, List[str]]:
    """Check if required directories exist."""
    required_dirs = [
        "src",
        "src/config",
        "src/providers",
        "src/generation",
        "src/processing",
        "src/scoring",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/fixtures",
        "configs",
        "audio_samples",
    ]

    missing_dirs = []

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)

    passed = len(missing_dirs) == 0
    return passed, missing_dirs


def check_config_files() -> Tuple[bool, List[str]]:
    """Check if required configuration files exist."""
    required_files = [
        "configs/default.yaml",
        "configs/experiment.yaml",
        ".env.template",
        "requirements.txt",
        "setup.py",
        "README.md",
    ]

    missing_files = []

    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    passed = len(missing_files) == 0
    return passed, missing_files


def check_env_file() -> Tuple[bool, str]:
    """Check if .env file exists and is configured."""
    if not Path(".env").exists():
        return False, ".env file not found (copy from .env.template)"

    return True, ".env file exists"


def check_api_keys() -> Tuple[bool, List[str]]:
    """Check if API keys are loaded from environment."""
    # Load .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        return False, ["python-dotenv not installed"]

    api_keys = {
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }

    configured_keys = [key for key, value in api_keys.items() if value and value != f"sk-{'ant' if 'ANTHROPIC' in key else 'or' if 'OPENROUTER' in key else ''}-your-key-here"]
    missing_keys = [key for key, value in api_keys.items() if not value or value.endswith("-your-key-here")]

    if len(configured_keys) == 0:
        return False, missing_keys

    return True, missing_keys


def check_config_loader() -> Tuple[bool, str]:
    """Check if configuration loader works."""
    try:
        from src.config.loader import load_config

        config = load_config()

        # Verify key attributes exist
        assert hasattr(config, 'audio')
        assert hasattr(config, 'llm')
        assert hasattr(config, 'logging')
        assert hasattr(config, 'experiment')

        # Verify some default values
        assert config.audio.sample_rate > 0
        assert config.llm.provider in ['anthropic', 'openai', 'openrouter']

        return True, f"Configuration loaded successfully from {config.profile} profile"
    except Exception as e:
        return False, f"Error: {str(e)}"


def check_logging_system() -> Tuple[bool, str]:
    """Check if logging system initializes correctly."""
    try:
        from src.utils.logging import configure_logging, get_logger
        import tempfile
        import shutil

        # Create temporary log directory
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # Configure logging
            configure_logging(
                level="INFO",
                format="console",
                output_dir=temp_dir,
                console_output=False,  # Disable console for test
                file_output=True
            )

            # Get logger and test logging
            log = get_logger("smoke_test")
            log.info("test_log_entry", test=True)

            # Verify log files were created
            log_files = list(temp_dir.glob("*.log"))
            if len(log_files) == 0:
                return False, "No log files created"

            return True, f"Logging system initialized, {len(log_files)} log files created"
        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    except Exception as e:
        return False, f"Error: {str(e)}"


def check_package_installation() -> Tuple[bool, str]:
    """Check if package is installed in development mode."""
    try:
        import src

        # Check if package is installed in editable mode
        src_path = Path(src.__file__).parent
        expected_path = Path.cwd() / "src"

        if src_path.resolve() == expected_path.resolve():
            return True, "Package installed in development mode"
        else:
            return False, "Package not installed in development mode (run: pip install -e .)"

    except ImportError:
        return False, "Package not installed (run: pip install -e .)"


def run_smoke_tests() -> bool:
    """Run all smoke tests and return overall pass/fail status."""
    print_header("LLM-as-Music-Judge Baseline System - Smoke Test")

    all_passed = True

    # Check Python version
    print("\n--- Python Environment ---")
    passed, details = check_python_version()
    print_result("Python version", passed, details)
    all_passed = all_passed and passed

    # Check dependencies
    passed, missing = check_dependencies()
    details = f"All dependencies installed" if passed else f"Missing: {', '.join(missing)}"
    print_result("Dependencies", passed, details)
    all_passed = all_passed and passed

    # Check package installation
    passed, details = check_package_installation()
    print_result("Package installation", passed, details)
    all_passed = all_passed and passed

    # Check directory structure
    print("\n--- Project Structure ---")
    passed, missing = check_directory_structure()
    details = f"All required directories exist" if passed else f"Missing: {', '.join(missing)}"
    print_result("Directory structure", passed, details)
    all_passed = all_passed and passed

    # Check config files
    passed, missing = check_config_files()
    details = f"All required files exist" if passed else f"Missing: {', '.join(missing)}"
    print_result("Configuration files", passed, details)
    all_passed = all_passed and passed

    # Check .env file
    passed, details = check_env_file()
    print_result("Environment file", passed, details)
    env_exists = passed

    # Check API keys (only if .env exists)
    print("\n--- API Configuration ---")
    if env_exists:
        passed, missing = check_api_keys()
        if passed:
            details = "At least one API key configured"
        else:
            details = f"No API keys configured. Add one of: {', '.join(missing[:1])}"
        print_result("API keys", passed, details)
        # Don't fail overall test if API keys not configured (might be intentional for some tests)
        if not passed:
            print("  Note: API keys are required for LLM integration")
    else:
        print_result("API keys", False, "Skipped (no .env file)")

    # Check configuration loader
    print("\n--- Core Systems ---")
    passed, details = check_config_loader()
    print_result("Configuration loader", passed, details)
    all_passed = all_passed and passed

    # Check logging system
    passed, details = check_logging_system()
    print_result("Logging system", passed, details)
    all_passed = all_passed and passed

    # Print summary
    print_header("Summary")
    if all_passed:
        print("\n✓ All critical checks passed!")
        print("\nYour environment is ready for development.")
        print("\nNext steps:")
        print("  1. Ensure API keys are configured in .env file")
        print("  2. Run tests: pytest")
        print("  3. Start experimenting with the baseline system")
    else:
        print("\n✗ Some checks failed.")
        print("\nPlease fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Install package: pip install -e .")
        print("  - Create .env file: cp .env.template .env")
        print("  - Add API keys to .env file")

    print(f"\n{'='*70}\n")

    return all_passed


def main():
    """Main entry point for smoke test."""
    try:
        # Change to script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)

        # Run tests
        success = run_smoke_tests()

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nSmoke test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during smoke test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
