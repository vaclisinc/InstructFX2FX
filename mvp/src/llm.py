"""
LLM Client for Parameter Generation

Generates initial audio effect parameters from text descriptions.
"""

import json
import re
from typing import Literal


def setup_llm_client(
    provider: Literal['anthropic', 'openai'] = 'anthropic',
    api_key: str = None
):
    """
    Setup LLM client.

    Args:
        provider: LLM provider ('anthropic' or 'openai')
        api_key: API key (if None, reads from environment)

    Returns:
        LLM client instance
    """
    if provider == 'anthropic':
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        print(f"✓ Anthropic client initialized")
        return client

    elif provider == 'openai':
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        print(f"✓ OpenAI client initialized")
        return client

    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_initial_params(
    llm_client,
    prompt: str,
    model: str = None
) -> dict:
    """
    Generate initial audio effect parameters using LLM.

    Args:
        llm_client: LLM client instance
        prompt: Text description (e.g., "make this sound warm")
        model: Model name (optional, uses default if None)

    Returns:
        Dictionary with 'eq', 'compressor', 'reverb' parameters
    """
    # Construct generation prompt
    system_prompt = """You are an audio engineer assistant that generates audio effect parameters from text descriptions.

Given a text description, generate parameters for:
- 6-band Parametric EQ (18 parameters: 6 bands × 3 params each)
  - Each band has: frequency (Hz), gain (dB), Q factor
- Compressor (parameters as needed)
- Reverb (parameters as needed)

Return ONLY a JSON object with this structure:
{
  "eq": [list of 18 numbers for 6-band EQ],
  "compressor": [list of compressor params],
  "reverb": [list of reverb params]
}

Parameter ranges:
- EQ frequency: 20-20000 Hz (log scale)
- EQ gain: -12 to +12 dB
- EQ Q: 0.5 to 4.0
- All params should be normalized to [0, 1] range

Be thoughtful about the text description and choose parameters that match the intent."""

    user_message = f"Generate audio effect parameters for: {prompt}"

    # Call LLM based on client type
    if hasattr(llm_client, 'messages'):  # Anthropic
        if model is None:
            model = "claude-3-5-sonnet-20241022"

        response = llm_client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        response_text = response.content[0].text

    elif hasattr(llm_client, 'chat'):  # OpenAI
        if model is None:
            model = "gpt-4o"

        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=1024
        )
        response_text = response.choices[0].message.content

    else:
        raise ValueError("Unknown LLM client type")

    # Parse JSON from response
    params = _parse_json_from_response(response_text)

    # Validate structure
    required_keys = ['eq', 'compressor', 'reverb']
    for key in required_keys:
        if key not in params:
            raise ValueError(f"LLM response missing required key: {key}")

    return params


def _parse_json_from_response(response: str) -> dict:
    """
    Extract and parse JSON from LLM response.

    Handles cases where JSON is wrapped in markdown code blocks.

    Args:
        response: Raw LLM response text

    Returns:
        Parsed JSON dictionary
    """
    # Try to extract JSON from markdown code block
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response

    # Parse JSON
    try:
        return json.loads(json_str.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from LLM response: {e}\n\nResponse:\n{response}")


# Example usage for testing
if __name__ == "__main__":
    import os

    # Test with Anthropic
    client = setup_llm_client(provider='anthropic')

    # Generate params
    params = generate_initial_params(
        llm_client=client,
        prompt="make this sound warm and cozy"
    )

    print("\nGenerated Parameters:")
    print(json.dumps(params, indent=2))
