#!/usr/bin/env python3
"""Example script demonstrating parameter generation usage.

IMPORTANT: Before running this script, create a .env file with your API keys:

    ANTHROPIC_API_KEY=sk-ant-...
    # OR
    OPENROUTER_API_KEY=sk-or-...

Never commit API keys to git!

This script demonstrates:
1. Basic parameter generation from descriptions
2. Single effect generation
3. Effect chain generation
4. Error handling
5. Different prompt versions

Usage:
    python examples/generate_parameters.py
    python examples/generate_parameters.py --description "warm and intimate"
    python examples/generate_parameters.py --effects eq reverb
    python examples/generate_parameters.py --provider openrouter
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

from models.llm_judge import (
    ClaudeProvider,
    create_provider,
    AnthropicConfig,
    OpenRouterConfig
)
from src.generation import (
    ParameterGenerator,
    ParameterGenerationError,
    JSONParseError,
    ValidationError,
    LLMProviderError
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example descriptions
EXAMPLE_DESCRIPTIONS = [
    {
        "name": "Warm and Intimate",
        "description": "warm and intimate vocal sound with subtle room ambience",
        "effects": ["eq", "reverb"]
    },
    {
        "name": "Bright and Energetic",
        "description": "bright and energetic guitar sound with punchy dynamics",
        "effects": ["eq", "compressor"]
    },
    {
        "name": "Dark and Atmospheric",
        "description": "dark atmospheric pad with deep reverb and low-end presence",
        "effects": ["eq", "reverb", "compressor"]
    },
    {
        "name": "Clean and Natural",
        "description": "clean and natural acoustic sound with minimal processing",
        "effects": ["eq"]
    },
    {
        "name": "Aggressive and Punchy",
        "description": "aggressive punchy drum sound with tight compression",
        "effects": ["eq", "compressor"]
    }
]


def print_section(title: str, width: int = 80):
    """Print a section header."""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_effect_chain(chain):
    """Pretty print an effect chain."""
    print(f"\nDescription: {chain.description}")
    print(f"Effect Order: {' → '.join(chain.order)}")
    print(f"\nGenerated Parameters:")

    for i, effect in enumerate(chain.effects, 1):
        effect_type = effect.effect_type.upper()
        print(f"\n{i}. {effect_type}")
        print("-" * 40)

        effect_dict = effect.to_dict()
        params = {k: v for k, v in effect_dict.items() if k != "effect_type"}

        print(json.dumps(params, indent=2))


async def example_basic_generation(generator: ParameterGenerator):
    """Example 1: Basic parameter generation."""
    print_section("Example 1: Basic Parameter Generation")

    description = "warm and intimate vocal sound"
    print(f"\nGenerating parameters for: '{description}'")

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=["eq", "reverb"]
        )

        print_effect_chain(chain)

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Parameter generation failed: {e}")
        return None


async def example_single_effect(generator: ParameterGenerator):
    """Example 2: Generate single effect."""
    print_section("Example 2: Single Effect Generation")

    description = "boost high frequencies for brightness and air"
    print(f"\nGenerating EQ for: '{description}'")

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=["eq"],
            temperature=0.5  # Lower temperature for more focused output
        )

        print_effect_chain(chain)

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Parameter generation failed: {e}")
        return None


async def example_effect_chain(generator: ParameterGenerator):
    """Example 3: Generate full effect chain."""
    print_section("Example 3: Full Effect Chain")

    description = "professional broadcast voice with clarity and presence"
    print(f"\nGenerating effect chain for: '{description}'")

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=["eq", "compressor", "reverb"],
            temperature=0.7,
            include_examples=True
        )

        print_effect_chain(chain)

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Parameter generation failed: {e}")
        return None


async def example_error_handling(generator: ParameterGenerator):
    """Example 4: Error handling demonstration."""
    print_section("Example 4: Error Handling")

    # Try various error scenarios
    print("\nTesting error handling...")

    # Test 1: Empty description
    print("\n1. Testing empty description:")
    try:
        chain = await generator.generate_parameters(
            description="",
            effects=["eq"]
        )
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    # Test 2: Invalid effect type
    print("\n2. Testing invalid effect type:")
    try:
        chain = await generator.generate_parameters(
            description="test",
            effects=["invalid_effect"]
        )
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    print("\n✓ Error handling tests complete")


async def run_example_suite(generator: ParameterGenerator):
    """Run all examples in sequence."""
    print_section("Parameter Generation Example Suite", width=80)

    print("\nRunning demonstration examples...")
    print("This may take a minute as we call the LLM API multiple times.")

    # Run examples
    await example_basic_generation(generator)
    await asyncio.sleep(1)  # Rate limiting

    await example_single_effect(generator)
    await asyncio.sleep(1)

    await example_effect_chain(generator)
    await asyncio.sleep(1)

    await example_error_handling(generator)

    print_section("Examples Complete", width=80)


async def run_custom_generation(
    generator: ParameterGenerator,
    description: str,
    effects: List[str]
):
    """Run custom parameter generation."""
    print_section("Custom Parameter Generation")

    print(f"\nDescription: {description}")
    print(f"Effects: {', '.join(effects)}")

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=effects
        )

        print_effect_chain(chain)

        # Save to file
        output_file = project_root / "output" / "generated_parameters.json"
        output_file.parent.mkdir(exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(chain.to_dict(), f, indent=2)

        print(f"\n✓ Parameters saved to: {output_file}")

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Parameter generation failed: {e}")
        return None


def create_llm_provider(provider_name: str = "claude"):
    """Create and configure LLM provider.

    Args:
        provider_name: Provider to use ("claude" or "openrouter")

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If API key is missing
    """
    if provider_name == "claude":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found in environment.\n"
                "Please create a .env file with:\n"
                "ANTHROPIC_API_KEY=sk-ant-..."
            )

        config = AnthropicConfig(
            api_key=api_key,
            model="claude-3-5-sonnet-20241022",
            retry={"max_attempts": 3},
            rate_limit={"requests_per_minute": 50}
        )

        return create_provider("anthropic", config.model_dump())

    elif provider_name == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment.\n"
                "Please create a .env file with:\n"
                "OPENROUTER_API_KEY=sk-or-..."
            )

        config = OpenRouterConfig(
            api_key=api_key,
            model="anthropic/claude-3.5-sonnet",
            retry={"max_attempts": 3},
            rate_limit={"requests_per_minute": 30}
        )

        return create_provider("openrouter", config.model_dump())

    else:
        raise ValueError(f"Unknown provider: {provider_name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate audio effect parameters from descriptions using LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/generate_parameters.py
  python examples/generate_parameters.py --description "warm vocal" --effects eq reverb
  python examples/generate_parameters.py --provider openrouter
  python examples/generate_parameters.py --examples

IMPORTANT: Create a .env file with your API key before running:
  ANTHROPIC_API_KEY=sk-ant-...
  # OR
  OPENROUTER_API_KEY=sk-or-...
        """
    )

    parser.add_argument(
        "--description",
        type=str,
        help="Audio characteristic description"
    )
    parser.add_argument(
        "--effects",
        nargs="+",
        choices=["eq", "reverb", "compressor"],
        help="Effect types to generate"
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "openrouter"],
        default="claude",
        help="LLM provider to use (default: claude)"
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        help="Prompt template version (default: v1)"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Run full example suite"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load environment variables
    load_dotenv()

    # Check for API keys
    print("\n" + "=" * 80)
    print(" Parameter Generation Demo")
    print("=" * 80)
    print("\n⚠️  IMPORTANT: This script requires API keys!")
    print("Please ensure you have created a .env file with:")
    print("  ANTHROPIC_API_KEY=sk-ant-...")
    print("  # OR")
    print("  OPENROUTER_API_KEY=sk-or-...")
    print("\nNever commit .env files to git!")
    print("=" * 80)

    # Create provider
    try:
        print(f"\nInitializing {args.provider} provider...")
        provider = create_llm_provider(args.provider)
        print(f"✓ Provider initialized: {provider.get_provider_name()}")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return 1

    # Create generator
    print(f"Loading prompt template version {args.prompt_version}...")
    try:
        generator = ParameterGenerator(
            llm_provider=provider,
            prompt_version=args.prompt_version
        )
        print(f"✓ Generator ready")
    except Exception as e:
        print(f"\n❌ Error initializing generator: {e}")
        return 1

    # Run examples or custom generation
    try:
        if args.examples:
            asyncio.run(run_example_suite(generator))
        elif args.description:
            effects = args.effects or ["eq", "reverb", "compressor"]
            asyncio.run(run_custom_generation(generator, args.description, effects))
        else:
            # Default: run a simple demo
            print("\nRunning simple demo (use --examples for full suite)...")
            description = "warm and intimate vocal sound"
            effects = ["eq", "reverb"]
            asyncio.run(run_custom_generation(generator, description, effects))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
