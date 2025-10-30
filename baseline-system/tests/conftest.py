"""
Pytest configuration for test suite.

Provides helper functions and fixtures for tests.
"""

import os
import pytest
from pathlib import Path
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env_vars():
    """
    Session-wide fixture that loads .env file once before any tests run.
    Autouse=True means this runs automatically without being requested.
    """
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)


def require_api_key(key_name):
    """
    Helper function to skip tests if required API key is not set.

    Usage in tests:
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')
    """
    if not os.getenv(key_name):
        pytest.skip(f"{key_name} not set in environment")
