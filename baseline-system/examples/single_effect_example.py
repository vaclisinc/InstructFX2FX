#!/usr/bin/env python3
"""
Single Effect Generation Example

This script demonstrates generating individual audio effects (EQ, Reverb, or
Compressor) from textual descriptions using the parameter generation system.

IMPORTANT: Before running this script, create a .env file with your API keys:

    ANTHROPIC_API_KEY=sk-ant-...
    # OR
    OPENROUTER_API_KEY=sk-or-...

Never commit API keys to git!

This example shows:
1. Generating EQ-only parameters
2. Generating Reverb-only parameters
3. Generating Compressor-only parameters
4. Parameter validation
5. Error handling
6. Output formatting

Usage:
    python examples/single_effect_example.py
    python examples/single_effect_example.py --effect eq
    python examples/single_effect_example.py --description "bright sound"
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from models.llm_judge import create_provider, AnthropicConfig, OpenRouterConfig
from src.generation import (
    ParameterGenerator,
    ParameterGenerationError,
    ValidationError
)
from src.generation.validator import validate_effect_parameter


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_effect_details(effect):
    """Print detailed effect parameters."""
    effect_type = effect.effect_type.upper()
    print(f"\n{effect_type} Parameters:")
    print("-" * 70)

    if effect.effect_type == "eq":
        print(f"EQ Type: {effect.eq_type}")
        print(f"Number of Bands: {len(effect.bands)}")
        print("\nBands:")
        for i, band in enumerate(effect.bands, 1):
            print(f"  {i}. {band.frequency:>6.1f} Hz: {band.gain:>+5.1f} dB (Q: {band.q:.1f})")

    elif effect.effect_type == "reverb":
        print(f"Room Size:    {effect.room_size:.2f}")
        print(f"Damping:      {effect.damping:.2f}")
        print(f"Wet Level:    {effect.wet_level:.2f}")
        print(f"Dry Level:    {effect.dry_level:.2f}")
        print(f"Width:        {effect.width:.2f}")
        print(f"Freeze Mode:  {effect.freeze_mode}")

    elif effect.effect_type == "compressor":
        print(f"Threshold:    {effect.threshold:>6.1f} dB")
        print(f"Ratio:        {effect.ratio:>6.1f}:1")
        print(f"Attack:       {effect.attack:>6.1f} ms")
        print(f"Release:      {effect.release:>6.1f} ms")
        print(f"Knee:         {effect.knee:>6.1f} dB")
        print(f"Makeup Gain:  {effect.makeup_gain:>6.1f} dB")

    # Show JSON representation
    print("\nJSON Output:")
    print(json.dumps(effect.to_dict(), indent=2))


async def generate_eq_example(generator: ParameterGenerator):
    """Example: Generate EQ parameters only."""
    print_header("Example 1: EQ Generation")

    description = "boost high frequencies for brightness and air"
    print(f"\nDescription: {description}")
    print("Requested Effect: EQ only")

    try:
        # Generate EQ parameters
        chain = await generator.generate_parameters(
            description=description,
            effects=["eq"],
            temperature=0.5  # Lower temperature for focused output
        )

        # Extract the EQ effect
        eq = chain.effects[0]
        print_effect_details(eq)

        # Validate
        result = validate_effect_parameter(eq)
        if result.is_valid:
            print("\n✓ Validation: PASSED")
        else:
            print("\n✗ Validation: FAILED")
            print(result.format_report())

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Generation failed: {e}")
        return None


async def generate_reverb_example(generator: ParameterGenerator):
    """Example: Generate Reverb parameters only."""
    print_header("Example 2: Reverb Generation")

    description = "large cathedral space with long decay"
    print(f"\nDescription: {description}")
    print("Requested Effect: Reverb only")

    try:
        # Generate Reverb parameters
        chain = await generator.generate_parameters(
            description=description,
            effects=["reverb"],
            temperature=0.6
        )

        # Extract the Reverb effect
        reverb = chain.effects[0]
        print_effect_details(reverb)

        # Validate
        result = validate_effect_parameter(reverb)
        if result.is_valid:
            print("\n✓ Validation: PASSED")
        else:
            print("\n✗ Validation: FAILED")
            print(result.format_report())

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Generation failed: {e}")
        return None


async def generate_compressor_example(generator: ParameterGenerator):
    """Example: Generate Compressor parameters only."""
    print_header("Example 3: Compressor Generation")

    description = "heavy compression for aggressive punch and sustain"
    print(f"\nDescription: {description}")
    print("Requested Effect: Compressor only")

    try:
        # Generate Compressor parameters
        chain = await generator.generate_parameters(
            description=description,
            effects=["compressor"],
            temperature=0.7
        )

        # Extract the Compressor effect
        compressor = chain.effects[0]
        print_effect_details(compressor)

        # Validate
        result = validate_effect_parameter(compressor)
        if result.is_valid:
            print("\n✓ Validation: PASSED")
        else:
            print("\n✗ Validation: FAILED")
            print(result.format_report())

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Generation failed: {e}")
        return None


async def error_handling_example(generator: ParameterGenerator):
    """Example: Demonstrate error handling."""
    print_header("Example 4: Error Handling")

    print("\nTesting various error scenarios...\n")

    # Test 1: Invalid effect type
    print("1. Invalid effect type:")
    try:
        chain = await generator.generate_parameters(
            description="test",
            effects=["invalid_effect"]
        )
        print("   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Caught expected error: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {type(e).__name__}: {e}")

    # Test 2: Empty description
    print("\n2. Empty description:")
    try:
        # This will likely succeed but generate generic parameters
        chain = await generator.generate_parameters(
            description="",
            effects=["eq"]
        )
        if chain:
            print("   ⚠ Generated parameters for empty description (may be generic)")
    except ValueError as e:
        print(f"   ✓ Caught error: {e}")
    except Exception as e:
        print(f"   ⚠ Unexpected behavior: {type(e).__name__}: {e}")

    # Test 3: Extremely long description
    print("\n3. Very long description:")
    long_desc = "warm " * 100  # Very repetitive description
    try:
        chain = await generator.generate_parameters(
            description=long_desc,
            effects=["eq"],
            max_tokens=1024  # Limit tokens
        )
        if chain:
            print("   ✓ Handled long description successfully")
    except Exception as e:
        print(f"   ⚠ Error with long description: {type(e).__name__}: {e}")

    print("\n✓ Error handling tests complete")


async def save_example(generator: ParameterGenerator, description: str, effect_type: str):
    """Generate and save parameters to file."""
    print_header(f"Generating and Saving {effect_type.upper()}")

    print(f"\nDescription: {description}")
    print(f"Effect Type: {effect_type}")

    try:
        # Generate parameters
        chain = await generator.generate_parameters(
            description=description,
            effects=[effect_type]
        )

        effect = chain.effects[0]
        print_effect_details(effect)

        # Save to file
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"{effect_type}_parameters.json"
        with open(output_file, 'w') as f:
            json.dump(effect.to_dict(), f, indent=2)

        print(f"\n✓ Parameters saved to: {output_file}")

        return chain

    except Exception as e:
        logger.error(f"Failed to generate or save: {e}")
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


async def run_all_examples(generator: ParameterGenerator):
    """Run all example demonstrations."""
    print("\n" + "=" * 70)
    print("  Single Effect Generation Examples")
    print("=" * 70)
    print("\nThis demo shows how to generate individual effect parameters")
    print("from textual descriptions using the parameter generation system.")

    # Run examples with delays for rate limiting
    await generate_eq_example(generator)
    await asyncio.sleep(1)

    await generate_reverb_example(generator)
    await asyncio.sleep(1)

    await generate_compressor_example(generator)
    await asyncio.sleep(1)

    await error_handling_example(generator)

    print("\n" + "=" * 70)
    print("  Examples Complete!")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate single audio effect parameters from descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/single_effect_example.py
  python examples/single_effect_example.py --effect eq --description "bright sound"
  python examples/single_effect_example.py --effect reverb --save
  python examples/single_effect_example.py --provider openrouter

IMPORTANT: Create a .env file with your API key before running:
  ANTHROPIC_API_KEY=sk-ant-...
  # OR
  OPENROUTER_API_KEY=sk-or-...
        """
    )

    parser.add_argument(
        "--effect",
        choices=["eq", "reverb", "compressor"],
        help="Specific effect type to generate"
    )
    parser.add_argument(
        "--description",
        type=str,
        help="Custom description for effect generation"
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "openrouter"],
        default="claude",
        help="LLM provider to use (default: claude)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save generated parameters to file"
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

    # Print API key reminder
    print("\n" + "=" * 70)
    print("  IMPORTANT: API Key Required")
    print("=" * 70)
    print("\nPlease ensure you have created a .env file with:")
    print("  ANTHROPIC_API_KEY=sk-ant-...")
    print("  # OR")
    print("  OPENROUTER_API_KEY=sk-or-...")
    print("\nNever commit .env files to git!")
    print("=" * 70)

    # Create provider
    try:
        print(f"\nInitializing {args.provider} provider...")
        provider = create_llm_provider(args.provider)
        print(f"✓ Provider ready: {provider.get_provider_name()}")
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return 1

    # Create generator
    try:
        print("Loading parameter generator...")
        generator = ParameterGenerator(
            llm_provider=provider,
            prompt_version="v1"
        )
        print("✓ Generator ready")
    except Exception as e:
        print(f"\n❌ Error initializing generator: {e}")
        return 1

    # Run examples
    try:
        if args.effect and args.description:
            # Generate specific effect with custom description
            if args.save:
                asyncio.run(save_example(generator, args.description, args.effect))
            else:
                async def run():
                    chain = await generator.generate_parameters(
                        description=args.description,
                        effects=[args.effect]
                    )
                    print_effect_details(chain.effects[0])

                asyncio.run(run())

        elif args.effect:
            # Generate specific effect with default description
            descriptions = {
                "eq": "boost high frequencies for brightness",
                "reverb": "large spacious reverb",
                "compressor": "gentle transparent compression"
            }
            description = descriptions[args.effect]

            if args.save:
                asyncio.run(save_example(generator, description, args.effect))
            else:
                async def run():
                    chain = await generator.generate_parameters(
                        description=description,
                        effects=[args.effect]
                    )
                    print_effect_details(chain.effects[0])

                asyncio.run(run())

        else:
            # Run full example suite
            asyncio.run(run_all_examples(generator))

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
