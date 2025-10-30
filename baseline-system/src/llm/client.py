"""
LLM client supporting multiple providers.

Provides a unified interface for calling OpenRouter, OpenAI, and Claude APIs.
"""

import os
from typing import Dict, Any


def call_llm(prompt: str, config: dict) -> str:
    """
    Call an LLM with the given prompt using the specified provider.

    Args:
        prompt: The text prompt to send to the LLM
        config: Configuration dictionary containing:
            - config['llm']['provider']: Provider name ('openrouter', 'openai', 'claude')
            - config['llm']['model']: Model identifier

    Returns:
        String response from the LLM

    Raises:
        ValueError: If provider is not supported
        Exception: If API key is missing or API call fails
    """
    # Extract provider and model from config
    provider = config['llm']['provider']
    model = config['llm']['model']

    # Route to appropriate provider
    if provider == 'openrouter':
        return _call_openrouter(prompt, model)
    elif provider == 'openai':
        return _call_openai(prompt, model)
    elif provider == 'claude':
        return _call_claude(prompt, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _call_openrouter(prompt: str, model: str) -> str:
    """
    Call OpenRouter API.

    Args:
        prompt: Text prompt
        model: Model identifier (e.g., 'anthropic/claude-3-haiku')

    Returns:
        String response from the model

    Raises:
        Exception: If API key is missing or API call fails
    """
    import requests

    # Get API key from environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise Exception("OPENROUTER_API_KEY not found in environment. Please set it in your .env file.")

    # OpenRouter API endpoint
    url = "https://openrouter.ai/api/v1/chat/completions"

    # Prepare request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    try:
        # Make API request with timeout
        response = requests.post(url, json=data, headers=headers, timeout=30)

        # Check for errors
        if response.status_code == 401:
            raise Exception("Invalid API key or unauthorized access to OpenRouter API")
        elif response.status_code == 429:
            raise Exception("Rate limit exceeded for OpenRouter API")
        elif response.status_code != 200:
            raise Exception(f"OpenRouter API error (status {response.status_code}): {response.text}")

        # Extract response text
        result = response.json()
        return result['choices'][0]['message']['content']

    except requests.exceptions.Timeout:
        raise Exception("Request to OpenRouter API timed out")
    except requests.exceptions.RequestException as e:
        raise Exception(f"OpenRouter API request failed: {str(e)}")


def _call_openai(prompt: str, model: str) -> str:
    """
    Call OpenAI API.

    Args:
        prompt: Text prompt
        model: Model identifier (e.g., 'gpt-4o-mini')

    Returns:
        String response from the model

    Raises:
        Exception: If API key is missing or API call fails
    """
    from openai import OpenAI, AuthenticationError, RateLimitError, APITimeoutError

    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment. Please set it in your .env file.")

    try:
        # Create client
        client = OpenAI(api_key=api_key, timeout=30.0)

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract response text
        return response.choices[0].message.content

    except AuthenticationError:
        raise Exception("Invalid API key or unauthorized access to OpenAI API")
    except RateLimitError:
        raise Exception("Rate limit exceeded for OpenAI API")
    except APITimeoutError:
        raise Exception("Request to OpenAI API timed out")
    except Exception as e:
        raise Exception(f"OpenAI API request failed: {str(e)}")


def _call_claude(prompt: str, model: str) -> str:
    """
    Call Anthropic Claude API directly.

    Args:
        prompt: Text prompt
        model: Model identifier (e.g., 'claude-3-haiku-20240307')

    Returns:
        String response from the model

    Raises:
        Exception: If API key is missing or API call fails
    """
    from anthropic import Anthropic, AuthenticationError, RateLimitError, APITimeoutError

    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY not found in environment. Please set it in your .env file.")

    try:
        # Create client
        client = Anthropic(api_key=api_key, timeout=30.0)

        # Make API call
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract response text
        return response.content[0].text

    except AuthenticationError:
        raise Exception("Invalid API key or unauthorized access to Anthropic API")
    except RateLimitError:
        raise Exception("Rate limit exceeded for Anthropic API")
    except APITimeoutError:
        raise Exception("Request to Anthropic API timed out")
    except Exception as e:
        raise Exception(f"Anthropic API request failed: {str(e)}")
