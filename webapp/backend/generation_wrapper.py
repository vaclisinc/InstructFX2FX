"""
Simplified wrapper for baseline-system parameter generation.
Avoids importing audio_description dependencies (CLAP, torch) for demo.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any
import sys

# Add baseline-system to path
baseline_path = Path(__file__).parent.parent.parent / "baseline-system"
sys.path.insert(0, str(baseline_path))

from src.llm.client import call_llm


def _load_prompt_template(template_path_str: str) -> str:
    """Load a prompt template from file."""
    template_path = Path(template_path_str)

    # Handle relative paths
    if not template_path.is_absolute():
        possible_paths = [
            template_path,
            Path('baseline-system') / template_path,
            Path(__file__).parent.parent.parent / 'baseline-system' / template_path
        ]

        for path in possible_paths:
            if path.exists():
                template_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find prompt template")

    with open(template_path, 'r') as f:
        return f.read()


def generate_parameters(user_prompt: str, config: dict) -> tuple[dict, str]:
    """
    Generate audio effect parameters from user's text description.

    Simplified version for webapp demo - doesn't use audio description.

    Args:
        user_prompt: User's textual description
        config: Configuration dictionary

    Returns:
        Tuple of (parameters dict, system_prompt string)
    """
    # Load generation prompt template
    template = _load_prompt_template(config['prompts']['generation_template'])
    prompt = template.replace('{user_prompt}', user_prompt)

    # Call LLM
    try:
        response = call_llm(prompt, config)

        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object directly
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in LLM response")

        params = json.loads(json_str)

        # Validate structure
        required_keys = ['reverb', 'eq', 'compressor']
        if not all(key in params for key in required_keys):
            raise ValueError(f"Missing required keys. Got: {params.keys()}")

        return params, prompt  # Return both parameters and the system prompt

    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse LLM JSON response: {e}")
    except Exception as e:
        raise Exception(f"Parameter generation failed: {e}")
