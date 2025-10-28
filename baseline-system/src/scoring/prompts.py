"""Scoring prompt templates and utilities.

This module provides functions for formatting scoring prompts that evaluate
generated audio effect parameters against descriptions. It supports both
parameter-only and audio-based scoring modes.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_scoring_template(template_name: str = "scoring_template") -> str:
    """Load scoring prompt template from file.

    Args:
        template_name: Name of the template file (without .txt extension)

    Returns:
        Template content as string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    # Look for template in src/prompts directory
    template_path = Path(__file__).parent.parent / "prompts" / f"{template_name}.txt"

    if not template_path.exists():
        raise FileNotFoundError(f"Scoring template not found: {template_path}")

    with open(template_path, 'r') as f:
        return f.read()


def format_scoring_prompt(
    description: str,
    parameters: Dict[str, Any],
    previous_score: Optional[float] = None,
    iteration: int = 0,
    template_name: str = "scoring_template"
) -> str:
    """Format a scoring prompt for parameter evaluation.

    This creates a prompt that asks the LLM to score how well the generated
    parameters match the description. Used for parameter-only scoring (fast mode).

    Args:
        description: Original user description of desired audio characteristics
        parameters: Generated effect parameters to evaluate
        previous_score: Previous iteration's score (for tracking improvement)
        iteration: Current iteration number
        template_name: Template file to use

    Returns:
        Formatted scoring prompt string

    Example:
        >>> prompt = format_scoring_prompt(
        ...     description="warm and intimate vocal sound",
        ...     parameters={"eq": {...}, "reverb": {...}}
        ... )
    """
    template = load_scoring_template(template_name)

    # Format parameters as JSON
    parameters_json = json.dumps(parameters, indent=2)

    # Build context about previous iterations
    iteration_context = ""
    if iteration > 0 and previous_score is not None:
        iteration_context = f"\n\nITERATION CONTEXT:\nThis is iteration {iteration}. Previous score: {previous_score:.1f}/100"

    # Format template
    prompt = template.format(
        description=description,
        parameters_json=parameters_json,
        iteration_context=iteration_context
    )

    return prompt


def format_audio_scoring_prompt(
    description: str,
    parameters: Dict[str, Any],
    audio_features: Dict[str, Any],
    previous_score: Optional[float] = None,
    iteration: int = 0,
    template_name: str = "scoring_template"
) -> str:
    """Format a scoring prompt including audio analysis.

    This creates an enhanced prompt that includes audio features for more
    accurate evaluation. Used for audio-based scoring (comprehensive mode).

    Args:
        description: Original user description
        parameters: Generated effect parameters
        audio_features: Extracted audio features (spectral, temporal, etc.)
        previous_score: Previous iteration's score
        iteration: Current iteration number
        template_name: Template file to use

    Returns:
        Formatted scoring prompt with audio context

    Example:
        >>> prompt = format_audio_scoring_prompt(
        ...     description="warm and intimate vocal sound",
        ...     parameters={"eq": {...}},
        ...     audio_features={"spectral_centroid": 2500, ...}
        ... )
    """
    # Start with base prompt
    base_prompt = format_scoring_prompt(
        description=description,
        parameters=parameters,
        previous_score=previous_score,
        iteration=iteration,
        template_name=template_name
    )

    # Add audio analysis section
    audio_context = format_audio_features(audio_features)

    enhanced_prompt = f"""{base_prompt}

AUDIO ANALYSIS:
{audio_context}

Consider both the parameters AND the actual audio characteristics when scoring.
"""

    return enhanced_prompt


def format_audio_features(features: Dict[str, Any]) -> str:
    """Format audio features for inclusion in scoring prompt.

    Args:
        features: Dictionary of audio features

    Returns:
        Formatted string describing audio features
    """
    feature_lines = []

    # Spectral features
    if "spectral_centroid" in features:
        feature_lines.append(f"- Spectral Centroid: {features['spectral_centroid']:.1f} Hz (brightness)")

    if "spectral_rolloff" in features:
        feature_lines.append(f"- Spectral Rolloff: {features['spectral_rolloff']:.1f} Hz (high frequency content)")

    if "spectral_bandwidth" in features:
        feature_lines.append(f"- Spectral Bandwidth: {features['spectral_bandwidth']:.1f} Hz (frequency spread)")

    # Temporal features
    if "rms_energy" in features:
        feature_lines.append(f"- RMS Energy: {features['rms_energy']:.4f} (loudness)")

    if "zero_crossing_rate" in features:
        feature_lines.append(f"- Zero Crossing Rate: {features['zero_crossing_rate']:.4f} (noisiness)")

    # Harmonic features
    if "harmonic_ratio" in features:
        feature_lines.append(f"- Harmonic Ratio: {features['harmonic_ratio']:.2f} (tonality)")

    # Room characteristics
    if "reverb_time" in features:
        feature_lines.append(f"- Estimated Reverb Time: {features['reverb_time']:.2f}s (space)")

    if not feature_lines:
        return "No audio features available"

    return "\n".join(feature_lines)


def get_scoring_system_prompt() -> str:
    """Get system prompt for consistent scoring behavior.

    Returns:
        System prompt string that instructs LLM on scoring methodology
    """
    return """You are an expert audio engineer evaluating audio effect parameters.

Your task is to score how well generated audio effect parameters match a given description.

SCORING METHODOLOGY:
- Use a 0-100 scale for all scores
- Be objective and consistent across evaluations
- Consider both technical correctness and semantic alignment
- Provide specific, actionable feedback
- Support your scores with clear reasoning

OUTPUT REQUIREMENTS:
- Return strictly valid JSON
- Include all required fields
- Ensure scores are within 0-100 range
- Keep confidence scores between 0-1
- Make suggestions specific and implementable"""


def format_correction_prompt(
    original_description: str,
    parameters: Dict[str, Any],
    invalid_response: str,
    error_message: str
) -> str:
    """Format a correction prompt for invalid scoring responses.

    Used when the LLM generates invalid JSON or out-of-range scores.

    Args:
        original_description: Original user description
        parameters: The parameters that were being scored
        invalid_response: The invalid response that was generated
        error_message: Description of what went wrong

    Returns:
        Formatted correction prompt
    """
    parameters_json = json.dumps(parameters, indent=2)

    return f"""=== CORRECTION NEEDED ===

Your previous scoring response was invalid. Please correct it.

Original Description: {original_description}

Parameters Being Scored:
```json
{parameters_json}
```

Previous Invalid Response:
```
{invalid_response}
```

Error:
{error_message}

Please provide a valid scoring response that:
1. Contains valid JSON matching the exact schema
2. Has all scores within 0-100 range
3. Has confidence value between 0-1
4. Includes all required fields (dimensions, overall_score, feedback, suggestions, confidence)

Required JSON format:
{{
  "dimensions": [
    {{"name": "semantic_match", "score": 85, "reasoning": "..."}},
    {{"name": "technical_quality", "score": 90, "reasoning": "..."}},
    {{"name": "specificity", "score": 75, "reasoning": "..."}}
  ],
  "overall_score": 83,
  "feedback": "Clear explanation of overall assessment...",
  "suggestions": ["Specific suggestion 1", "Specific suggestion 2"],
  "confidence": 0.85
}}

Generate corrected scoring response now:
"""


def format_refinement_prompt(
    description: str,
    current_parameters: Dict[str, Any],
    score: float,
    feedback: str,
    suggestions: List[str],
    iteration: int
) -> str:
    """Format a refinement prompt for parameter improvement.

    This prompt is used to guide the LLM in improving parameters based on
    scoring feedback from a previous iteration.

    Args:
        description: Original user description
        current_parameters: Current parameters that need improvement
        score: Current score (0-100)
        feedback: Feedback from scoring evaluation
        suggestions: List of specific improvement suggestions
        iteration: Current iteration number

    Returns:
        Formatted refinement prompt
    """
    parameters_json = json.dumps(current_parameters, indent=2)
    suggestions_text = "\n".join(f"- {s}" for s in suggestions)

    return f"""=== PARAMETER REFINEMENT ===

TASK: Improve the audio effect parameters based on scoring feedback.

ORIGINAL DESCRIPTION: {description}

ITERATION: {iteration}
CURRENT SCORE: {score}/100

CURRENT PARAMETERS:
```json
{parameters_json}
```

FEEDBACK:
{feedback}

IMPROVEMENT SUGGESTIONS:
{suggestions_text}

INSTRUCTIONS:
1. Analyze the feedback and suggestions carefully
2. Adjust parameters to better match the description
3. Focus on the lowest-scoring dimensions
4. Make targeted, incremental improvements
5. Maintain technical validity of all parameters

Generate improved parameters in the same JSON format:
"""
