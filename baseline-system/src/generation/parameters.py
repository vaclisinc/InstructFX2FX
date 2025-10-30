"""
Parameter generation and audio judging functions.

Uses LLM to generate audio effect parameters from text descriptions
and judge the quality of processed audio.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any
from src.llm.client import call_llm


def _load_prompt_template(template_path_str: str) -> str:
    """
    Load a prompt template from file, handling relative paths.

    Args:
        template_path_str: Path to template file (relative or absolute)

    Returns:
        Contents of template file

    Raises:
        FileNotFoundError: If template file cannot be found
    """
    template_path = Path(template_path_str)

    # Handle relative paths
    if not template_path.is_absolute():
        # Try multiple possible locations
        possible_paths = [
            template_path,
            Path('baseline-system') / template_path,
            Path(__file__).parent.parent.parent / template_path
        ]

        for path in possible_paths:
            if path.exists():
                template_path = path
                break
        else:
            raise FileNotFoundError(f"Could not find prompt template at any of: {possible_paths}")

    with open(template_path, 'r') as f:
        return f.read()


def generate_parameters(user_prompt: str, config: dict) -> dict:
    """
    Generate audio effect parameters from a user's text description.

    Args:
        user_prompt: User's textual description of desired audio effect
        config: Configuration dictionary containing prompt templates and LLM settings

    Returns:
        Dictionary with 'reverb', 'eq', and 'compressor' parameters matching socialfx_data schema

    Raises:
        Exception: If LLM call fails or returns invalid JSON
    """
    # Load and fill generation prompt template
    template = _load_prompt_template(config['prompts']['generation_template'])
    prompt = template.replace('{user_prompt}', user_prompt)

    # Call LLM to generate parameters
    try:
        response = call_llm(prompt, config)

        # Parse JSON response
        # LLM might return JSON wrapped in markdown code blocks, so extract it
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response

        # Parse JSON
        try:
            parameters = json.loads(json_str.strip())
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse JSON from LLM response: {str(e)}\nResponse: {response}")

        # Validate structure
        if not isinstance(parameters, dict):
            raise Exception(f"LLM returned non-dict: {type(parameters)}")

        required_keys = ['reverb', 'eq', 'compressor']
        for key in required_keys:
            if key not in parameters:
                raise Exception(f"Missing required key '{key}' in LLM response")

        return parameters

    except Exception as e:
        raise Exception(f"Parameter generation failed: {str(e)}")


def judge_audio(user_prompt: str, audio_description: str, config: dict) -> float:
    """
    Judge how well processed audio matches the user's intent.

    Args:
        user_prompt: User's original textual description
        audio_description: Description of the processed audio characteristics
        config: Configuration dictionary containing prompt templates and LLM settings

    Returns:
        Float score between 0 and 10

    Raises:
        Exception: If LLM call fails or returns invalid score
    """
    # Load and fill judge prompt template
    template = _load_prompt_template(config['prompts']['judge_template'])
    prompt = template.replace('{user_prompt}', user_prompt)
    prompt = prompt.replace('{audio_description}', audio_description)

    # Call LLM to get judgment
    try:
        response = call_llm(prompt, config)

        # Extract numeric score from response
        # LLM should return just a number, but might include extra text
        # Try to find a number in the response
        numbers = re.findall(r'\b\d+\.?\d*\b', response)

        if not numbers:
            raise Exception(f"No numeric score found in LLM response: {response}")

        # Take the first number found
        try:
            score = float(numbers[0])
        except ValueError as e:
            raise Exception(f"Failed to parse score as float: {numbers[0]}")

        # Validate range
        if not (0 <= score <= 10):
            raise Exception(f"Score {score} is out of valid range [0, 10]")

        return score

    except Exception as e:
        raise Exception(f"Audio judging failed: {str(e)}")
