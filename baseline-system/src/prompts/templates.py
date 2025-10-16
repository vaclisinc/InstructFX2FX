"""Prompt formatting utilities for parameter generation.

This module provides functions for formatting prompts with templates,
few-shot examples, and variable substitution for LLM parameter generation.
"""

from typing import List, Optional
from .loader import PromptTemplate, load_prompt_template


def format_prompt(
    description: str,
    effects: Optional[List[str]] = None,
    template_version: str = "v1",
    template_name: str = "parameter_generation",
    include_examples: bool = True,
    num_examples: Optional[int] = None
) -> str:
    """Format a complete prompt for parameter generation.

    This function combines the system prompt, few-shot examples, and user prompt
    into a single formatted string ready for LLM input.

    Args:
        description: User's description of desired audio characteristics
        effects: List of effect types to generate (e.g., ["eq", "reverb", "compressor"])
                If None, defaults to all three effects
        template_version: Prompt template version to use
        template_name: Name of the template
        include_examples: Whether to include few-shot examples
        num_examples: Number of examples to include (None = all)

    Returns:
        Formatted prompt string ready for LLM

    Example:
        >>> prompt = format_prompt(
        ...     description="warm and intimate vocal sound",
        ...     effects=["eq", "reverb"]
        ... )
        >>> print(prompt)
    """
    # Load template
    template = load_prompt_template(
        version=template_version,
        template_name=template_name
    )

    # Default to all effects if not specified
    if effects is None:
        effects = ["eq", "reverb", "compressor"]

    # Format effects list as comma-separated string
    effects_str = ", ".join(effects)

    # Build prompt components
    prompt_parts = []

    # 1. System prompt
    prompt_parts.append("=== SYSTEM INSTRUCTIONS ===")
    prompt_parts.append(template.system_prompt)
    prompt_parts.append("")

    # 2. Few-shot examples (if requested)
    if include_examples and template.few_shot_examples:
        prompt_parts.append("=== EXAMPLES ===")
        prompt_parts.append("")

        examples = template.few_shot_examples
        if num_examples is not None:
            examples = examples[:num_examples]

        examples_text = []
        for i, example in enumerate(examples, 1):
            if example.content:
                # Format example with description
                example_text = f"Example {i}: {example.description}\n"

                # Add the JSON content
                import json
                example_json = json.dumps(example.content, indent=2)
                example_text += f"```json\n{example_json}\n```"

                examples_text.append(example_text)

        prompt_parts.append("\n\n".join(examples_text))
        prompt_parts.append("")

    # 3. User prompt
    prompt_parts.append("=== YOUR TASK ===")
    user_prompt = template.format_user_prompt(
        description=description,
        effects=effects_str
    )
    prompt_parts.append(user_prompt)

    # Join all parts
    return "\n".join(prompt_parts)


def format_system_prompt(
    template_version: str = "v1",
    template_name: str = "parameter_generation"
) -> str:
    """Get only the system prompt from a template.

    Args:
        template_version: Prompt template version
        template_name: Template name

    Returns:
        System prompt string
    """
    template = load_prompt_template(
        version=template_version,
        template_name=template_name
    )
    return template.system_prompt


def format_user_prompt(
    description: str,
    effects: Optional[List[str]] = None,
    template_version: str = "v1",
    template_name: str = "parameter_generation"
) -> str:
    """Format only the user prompt (without system prompt or examples).

    Args:
        description: User's description of desired audio characteristics
        effects: List of effect types to generate
        template_version: Prompt template version
        template_name: Template name

    Returns:
        Formatted user prompt string
    """
    template = load_prompt_template(
        version=template_version,
        template_name=template_name
    )

    if effects is None:
        effects = ["eq", "reverb", "compressor"]

    effects_str = ", ".join(effects)

    return template.format_user_prompt(
        description=description,
        effects=effects_str
    )


def get_few_shot_examples(
    template_version: str = "v1",
    template_name: str = "parameter_generation",
    num_examples: Optional[int] = None
) -> str:
    """Get formatted few-shot examples from a template.

    Args:
        template_version: Prompt template version
        template_name: Template name
        num_examples: Number of examples to include (None = all)

    Returns:
        Formatted examples string
    """
    template = load_prompt_template(
        version=template_version,
        template_name=template_name
    )

    examples = template.few_shot_examples
    if num_examples is not None:
        examples = examples[:num_examples]

    import json
    examples_text = []
    for i, example in enumerate(examples, 1):
        if example.content:
            example_text = f"Example {i}: {example.description}\n"
            example_json = json.dumps(example.content, indent=2)
            example_text += f"```json\n{example_json}\n```"
            examples_text.append(example_text)

    return "\n\n".join(examples_text)


def format_correction_prompt(
    original_description: str,
    invalid_output: str,
    error_message: str,
    effects: Optional[List[str]] = None,
    template_version: str = "v1"
) -> str:
    """Format a correction prompt for invalid LLM output.

    Used when the LLM generates invalid JSON or parameters outside valid ranges.

    Args:
        original_description: Original user description
        invalid_output: The invalid output that was generated
        error_message: Description of what went wrong
        effects: List of effect types
        template_version: Template version to use

    Returns:
        Formatted correction prompt
    """
    template = load_prompt_template(version=template_version)

    if effects is None:
        effects = ["eq", "reverb", "compressor"]

    effects_str = ", ".join(effects)

    correction = f"""=== CORRECTION NEEDED ===

Your previous output was invalid. Please correct it.

Original Description: {original_description}
Requested Effects: {effects_str}

Previous Invalid Output:
```
{invalid_output}
```

Error:
{error_message}

Please generate valid JSON output that:
1. Matches the exact schema specified in the system instructions
2. Has all parameter values within the specified valid ranges
3. Uses appropriate effect settings for the description

{template.system_prompt}

Generate corrected parameters now:
"""

    return correction


def format_simple_prompt(
    description: str,
    effects: Optional[List[str]] = None
) -> str:
    """Format a simple prompt without template system.

    Useful for quick testing or when template system is not needed.

    Args:
        description: Audio characteristics description
        effects: List of effect types

    Returns:
        Simple formatted prompt
    """
    if effects is None:
        effects = ["eq", "reverb", "compressor"]

    effects_str = ", ".join(effects)

    return f"""Generate audio effect parameters for the following description:

Description: {description}
Effects to generate: {effects_str}

Output valid JSON with the following structure:
{{
  "description": "...",
  "effects": [
    {{
      "type": "eq|reverb|compressor",
      "parameters": {{ /* effect-specific parameters */ }}
    }}
  ]
}}
"""
