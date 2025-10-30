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
from src.generation.audio_description import (
    generate_audio_description_with_clap,
    generate_audio_description_from_params
)


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


def _parse_json_from_llm_response(response: str) -> dict:
    """
    Parse JSON from LLM response, handling markdown code blocks.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON as dictionary

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed
    """
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

    return json.loads(json_str.strip())


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
        try:
            parameters = _parse_json_from_llm_response(response)
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


def _generate_audio_description(audio_path: str, params: dict) -> str:
    """
    Generate a textual description of audio.

    Uses CLAP model if audio file exists, otherwise falls back to parameter-based description.

    Args:
        audio_path: Path to audio file
        params: Dictionary with 'reverb', 'eq', and 'compressor' parameters

    Returns:
        String description of the audio characteristics
    """
    # Check if audio file exists
    audio_file = Path(audio_path)
    if audio_file.exists():
        try:
            # Use CLAP to generate description from actual audio
            description = generate_audio_description_with_clap(str(audio_file))
            print(f"[CLAP Description]: {description}")
            return description
        except Exception as e:
            # Fall back to parameter-based description if CLAP fails
            print(f"[Warning] CLAP description failed: {e}, using parameter-based description")
            return generate_audio_description_from_params(params)
    else:
        # Use parameter-based description as fallback
        print(f"[Info] Audio file not found: {audio_path}, using parameter-based description")
        return generate_audio_description_from_params(params)


def refine_loop(user_prompt: str, audio_path: str, config: dict) -> dict:
    """
    Iteratively refine audio effect parameters based on judge feedback.

    Orchestrates the full refinement loop:
    1. Generate initial parameters from user prompt
    2. Generate audio description using CLAP (if audio exists) or parameters (fallback)
    3. Judge the result using LLM
    4. If not converged, refine parameters based on feedback
    5. Repeat until max iterations or convergence

    Args:
        user_prompt: User's textual description of desired audio effect
        audio_path: Path to input audio file (used for CLAP description if exists)
        config: Configuration dictionary with LLM settings and refinement config

    Returns:
        Dictionary with:
            - 'best_params': Parameters with the highest score
            - 'history': List of dicts with 'iteration', 'params', 'score' for each iteration

    Raises:
        Exception: If parameter generation or judging fails
    """
    # Get refinement configuration
    max_iterations = config['refinement']['max_iterations']
    convergence_threshold = config['refinement'].get('convergence_threshold', 0.1)
    target_score = config['refinement'].get('target_score', 8.0)

    # Initialize tracking
    history = []
    best_score = -1
    best_params = None

    # Generate initial parameters
    current_params = generate_parameters(user_prompt, config)

    # Refinement loop
    for iteration in range(max_iterations):
        # Generate audio description using CLAP (if audio file exists) or parameters (fallback)
        audio_description = _generate_audio_description(audio_path, current_params)

        # Judge the current parameters
        score = judge_audio(user_prompt, audio_description, config)

        # Track history
        history.append({
            'iteration': iteration + 1,
            'params': current_params.copy(),
            'score': score
        })

        # Update best parameters if this is better
        if score > best_score:
            best_score = score
            best_params = current_params.copy()

        # Check stopping conditions
        # 1. Reached target score
        if score >= target_score:
            break

        # 2. Score plateaued (convergence)
        if iteration > 0:
            previous_score = history[-2]['score']
            score_improvement = score - previous_score

            if abs(score_improvement) < convergence_threshold:
                break

        # 3. Last iteration - don't generate new params
        if iteration >= max_iterations - 1:
            break

        # Refine parameters for next iteration
        # Load and fill refinement prompt template
        template = _load_prompt_template(config['prompts']['refinement_template'])
        prompt = template.replace('{user_prompt}', user_prompt)
        prompt = prompt.replace('{previous_params}', json.dumps(current_params, indent=2))
        prompt = prompt.replace('{score}', str(score))

        # Call LLM to get refined parameters
        try:
            response = call_llm(prompt, config)
            current_params = _parse_json_from_llm_response(response)

        except Exception as e:
            # If refinement fails, keep current params and stop
            print(f"Warning: Refinement failed at iteration {iteration + 1}: {str(e)}")
            break

    return {
        'best_params': best_params,
        'history': history
    }
