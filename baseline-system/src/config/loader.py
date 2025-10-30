"""
Configuration loading and validation.

Provides functions to load YAML configuration files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


def load_config(path: str) -> dict:
    """
    Load and validate YAML configuration file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required configuration keys are missing
    """
    config_path = Path(path)

    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required keys
    required_keys = {
        'llm': ['provider', 'model'],
        'prompts': ['generation_template', 'judge_template', 'refinement_template'],
        'refinement': ['max_iterations', 'convergence_threshold']
    }

    # Check top-level sections
    for section in required_keys.keys():
        if section not in config:
            raise KeyError(f"Missing required configuration section: {section}")

    # Check nested keys
    for section, keys in required_keys.items():
        for key in keys:
            if key not in config[section]:
                raise KeyError(f"Missing required key '{key}' in section '{section}'")

    return config


def load_env(path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.

    Uses python-dotenv to load variables into os.environ.

    Args:
        path: Path to .env file. If None, uses default .env in current directory

    Note:
        This function loads variables into the environment but doesn't return them.
        Use os.getenv() to access the variables after loading.
    """
    if path is None:
        # Use default .env in current directory
        path = '.env'

    # Load the .env file
    # override=True means .env values override existing environment variables
    load_dotenv(path, override=True)
