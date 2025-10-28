#!/usr/bin/env python3
"""
Effect Chain Generation Example

This script demonstrates generating complete effect chains (multiple effects in
sequence) from textual descriptions using the parameter generation system.

IMPORTANT: Before running this script, create a .env file with your API keys:

    ANTHROPIC_API_KEY=sk-ant-...
    # OR
    OPENROUTER_API_KEY=sk-or-...

Never commit API keys to git!

This example shows:
1. Generating multi-effect chains
2. Effect ordering and signal flow
3. Validation and normalization
4. Real-world creative descriptions
5. Saving parameter sets to JSON
6. Comparing different effect combinations

Usage:
    python examples/effect_chain_example.py
    python examples/effect_chain_example.py --description "warm vintage vocal"
    python examples/effect_chain_example.py --effects eq compressor reverb
    python examples/effect_chain_example.py --save output/my_chain.json
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from models.llm_judge import create_provider, AnthropicConfig, OpenRouterConfig
from src.generation import (
    ParameterGenerator,
    ParameterGenerationError
)
from src.generation.validator import validate_effect_chain
from src.generation.normalizer import normalize_effect_chain


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Real-world effect chain descriptions
EXAMPLE_CHAINS = [
    {
        "name": "Warm Vintage Vocal",
        "description": "warm and vintage vocal sound with analog character for soul music",
        "effects": ["eq", "compressor", "reverb"]
    },
    {
        "name": "Bright Modern Guitar",
        "description": "bright and modern electric guitar with crisp high-end and tight dynamics",
        "effects": ["eq", "compressor"]
    },
    {
        "name": "Cinematic Pad",
        "description": "dark atmospheric pad with deep reverb and evolving texture for film score",
        "effects": ["eq", "reverb", "compressor"]
    },
    {
        "name": "Punchy Electronic Drums",
        "description": "punchy and aggressive drum sound with maximum impact for EDM",
        "effects": ["eq", "compressor"]
    },
    {
        "name": "Natural Acoustic",
        "description": "clean and natural acoustic guitar with subtle room ambience",
        "effects": ["eq", "reverb"]
    },
    {
        "name": "Broadcast Voice",
        "description": "professional broadcast voice with clarity, presence, and controlled dynamics",
        "effects": ["eq", "compressor", "reverb"]
    }
]


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_chain_overview(chain):
    """Print high-level overview of effect chain."""
    print(f"\nDescription: {chain.description}")
    print(f"Number of Effects: {len(chain.effects)}")
    print(f"Signal Flow: {' → '.join([e.upper() for e in chain.order])}")


def print_chain_details(chain):
    """Print detailed parameters for all effects in chain."""
    print_chain_overview(chain)

    for i, effect in enumerate(chain.effects, 1):
        effect_type = effect.effect_type.upper()
        print(f"\n{i}. {effect_type}")
        print("-" * 80)

        if effect.effect_type == "eq":
            print(f"   EQ Type: {effect.eq_type}")
            print(f"   Bands: {len(effect.bands)}")
            for j, band in enumerate(effect.bands, 1):
                print(f"      {j}. {band.frequency:>6.0f} Hz: {band.gain:>+5.1f} dB (Q: {band.q:.1f})")

        elif effect.effect_type == "reverb":
            print(f"   Room Size:    {effect.room_size:.2f}")
            print(f"   Damping:      {effect.damping:.2f}")
            print(f"   Wet Level:    {effect.wet_level:.2f}")
            print(f"   Dry Level:    {effect.dry_level:.2f}")
            print(f"   Width:        {effect.width:.2f}")

        elif effect.effect_type == "compressor":
            print(f"   Threshold:    {effect.threshold:>6.1f} dB")
            print(f"   Ratio:        {effect.ratio:>6.1f}:1")
            print(f"   Attack:       {effect.attack:>6.1f} ms")
            print(f"   Release:      {effect.release:>6.1f} ms")
            print(f"   Knee:         {effect.knee:>6.1f} dB")
            print(f"   Makeup Gain:  {effect.makeup_gain:>6.1f} dB")


async def generate_basic_chain(generator: ParameterGenerator):
    """Example 1: Generate a basic effect chain."""
    print_header("Example 1: Basic Effect Chain")

    description = "warm and intimate vocal sound"
    effects = ["eq", "reverb"]

    print(f"\nDescription: {description}")
    print(f"Requested Effects: {', '.join(effects)}")

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=effects
        )

        print_chain_details(chain)

        # Validate
        result = validate_effect_chain(chain)
        if result.is_valid:
            print("\n✓ Validation: PASSED")
        else:
            print("\n✗ Validation: FAILED")
            print(result.format_report())

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Generation failed: {e}")
        return None


async def generate_full_chain(generator: ParameterGenerator):
    """Example 2: Generate a complete effect chain with all three effects."""
    print_header("Example 2: Complete Effect Chain (EQ → Compressor → Reverb)")

    description = "professional broadcast voice with clarity and presence"
    effects = ["eq", "compressor", "reverb"]

    print(f"\nDescription: {description}")
    print(f"Requested Effects: {', '.join(effects)}")
    print("Note: Effects are applied in the order specified")

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=effects,
            temperature=0.7,
            include_examples=True
        )

        print_chain_details(chain)

        # Validate
        result = validate_effect_chain(chain)
        if result.is_valid:
            print("\n✓ Validation: PASSED")
        else:
            print("\n✗ Validation: FAILED")
            print(result.format_report())

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Generation failed: {e}")
        return None


async def generate_custom_order(generator: ParameterGenerator):
    """Example 3: Generate chain with custom effect order."""
    print_header("Example 3: Custom Effect Order")

    description = "heavy compression first, then EQ for tonal shaping"
    effects = ["compressor", "eq"]  # Unconventional order

    print(f"\nDescription: {description}")
    print(f"Requested Effects: {', '.join(effects)}")
    print("Note: Compressor before EQ (unconventional but sometimes useful)")

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=effects
        )

        print_chain_details(chain)

        print("\n⚠ Note: Typical order is EQ → Compressor → Reverb")
        print("   This example shows custom ordering is possible")

        return chain

    except ParameterGenerationError as e:
        logger.error(f"Generation failed: {e}")
        return None


async def generate_and_validate(generator: ParameterGenerator, description: str, effects: List[str]):
    """Generate chain with comprehensive validation."""
    print_header("Example 4: Generation with Validation & Normalization")

    print(f"\nDescription: {description}")
    print(f"Effects: {', '.join(effects)}")

    try:
        # Generate
        print("\nStep 1: Generating parameters...")
        chain = await generator.generate_parameters(
            description=description,
            effects=effects
        )
        print(f"✓ Generated {len(chain.effects)} effects")

        # Validate
        print("\nStep 2: Validating parameters...")
        result = validate_effect_chain(chain)

        if result.is_valid:
            print("✓ Validation passed - all parameters valid")
        else:
            print("⚠ Validation found issues:")
            print(result.format_report())

        # Show warnings even if valid
        if result.has_warnings():
            print("\nWarnings:")
            for warning in result.get_warnings():
                print(f"  - {warning}")

        # Normalize (ensures all values in valid ranges)
        print("\nStep 3: Normalizing parameters...")
        normalized_chain = normalize_effect_chain(chain)
        print("✓ Parameters normalized")

        # Re-validate normalized chain
        norm_result = validate_effect_chain(normalized_chain)
        assert norm_result.is_valid, "Normalized chain should always be valid"
        print("✓ Normalized chain validation passed")

        print_chain_details(normalized_chain)

        return normalized_chain

    except Exception as e:
        logger.error(f"Error during generation/validation: {e}")
        return None


async def save_chain(generator: ParameterGenerator, description: str, effects: List[str], output_file: str):
    """Generate and save effect chain to JSON file."""
    print_header(f"Generating and Saving Effect Chain")

    print(f"\nDescription: {description}")
    print(f"Effects: {', '.join(effects)}")
    print(f"Output File: {output_file}")

    try:
        # Generate chain
        chain = await generator.generate_parameters(
            description=description,
            effects=effects
        )

        print_chain_details(chain)

        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(chain.to_dict(), f, indent=2)

        print(f"\n✓ Effect chain saved to: {output_path}")
        print(f"   File size: {output_path.stat().st_size} bytes")

        # Show how to load it back
        print("\nTo load this chain later:")
        print(f"  with open('{output_path}', 'r') as f:")
        print(f"      data = json.load(f)")
        print(f"      chain = EffectChain(**data)")

        return chain

    except Exception as e:
        logger.error(f"Failed to generate or save chain: {e}")
        return None


async def compare_effect_combinations(generator: ParameterGenerator):
    """Example: Compare different effect combinations for same description."""
    print_header("Example 5: Comparing Effect Combinations")

    description = "warm and smooth vocal"
    combinations = [
        ["eq"],
        ["eq", "reverb"],
        ["eq", "compressor"],
        ["eq", "compressor", "reverb"]
    ]

    print(f"\nDescription: {description}")
    print("Comparing different effect combinations:\n")

    results = {}

    for effects in combinations:
        combo_name = " + ".join([e.upper() for e in effects])
        print(f"Generating: {combo_name}...")

        try:
            chain = await generator.generate_parameters(
                description=description,
                effects=effects
            )

            results[combo_name] = chain
            print(f"  ✓ Generated {len(chain.effects)} effects")

            # Small delay for rate limiting
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[combo_name] = None

    # Display comparison
    print("\n" + "-" * 80)
    print("Comparison Results:")
    print("-" * 80)

    for combo_name, chain in results.items():
        if chain:
            print(f"\n{combo_name}:")
            print_chain_overview(chain)
        else:
            print(f"\n{combo_name}: FAILED")

    return results


async def generate_example_library(generator: ParameterGenerator):
    """Generate a library of example chains from presets."""
    print_header("Example 6: Generate Example Chain Library")

    print("\nGenerating multiple example chains from presets...")
    print(f"Total presets: {len(EXAMPLE_CHAINS)}\n")

    library = {}

    for i, preset in enumerate(EXAMPLE_CHAINS, 1):
        print(f"{i}/{len(EXAMPLE_CHAINS)}: {preset['name']}")
        print(f"   Description: {preset['description']}")
        print(f"   Effects: {', '.join(preset['effects'])}")

        try:
            chain = await generator.generate_parameters(
                description=preset['description'],
                effects=preset['effects']
            )

            library[preset['name']] = chain
            print(f"   ✓ Generated successfully")

            # Rate limiting
            await asyncio.sleep(1)

        except Exception as e:
            print(f"   ✗ Failed: {e}")
            library[preset['name']] = None

    # Save library
    output_dir = project_root / "output" / "chain_library"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving library to {output_dir}...")

    for name, chain in library.items():
        if chain:
            safe_name = name.lower().replace(" ", "_")
            output_file = output_dir / f"{safe_name}.json"

            with open(output_file, 'w') as f:
                json.dump(chain.to_dict(), f, indent=2)

            print(f"  ✓ Saved: {output_file.name}")

    print(f"\n✓ Library saved: {len([c for c in library.values() if c])} chains")

    return library


def create_llm_provider(provider_name: str = "claude"):
    """Create and configure LLM provider."""
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
    print("\n" + "=" * 80)
    print("  Effect Chain Generation Examples")
    print("=" * 80)
    print("\nThis demo shows how to generate multi-effect chains from")
    print("textual descriptions using the parameter generation system.")

    # Run examples with delays
    await generate_basic_chain(generator)
    await asyncio.sleep(1)

    await generate_full_chain(generator)
    await asyncio.sleep(1)

    await generate_custom_order(generator)
    await asyncio.sleep(1)

    await generate_and_validate(
        generator,
        "dark atmospheric pad",
        ["eq", "reverb", "compressor"]
    )
    await asyncio.sleep(1)

    await compare_effect_combinations(generator)

    print("\n" + "=" * 80)
    print("  Examples Complete!")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multi-effect audio chains from descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/effect_chain_example.py
  python examples/effect_chain_example.py --description "warm vocal"
  python examples/effect_chain_example.py --effects eq compressor reverb
  python examples/effect_chain_example.py --save output/my_chain.json
  python examples/effect_chain_example.py --library

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
        help="Effect types to include in chain"
    )
    parser.add_argument(
        "--save",
        type=str,
        metavar="FILE",
        help="Save generated chain to JSON file"
    )
    parser.add_argument(
        "--library",
        action="store_true",
        help="Generate complete example library"
    )
    parser.add_argument(
        "--provider",
        choices=["claude", "openrouter"],
        default="claude",
        help="LLM provider to use (default: claude)"
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
    print("\n" + "=" * 80)
    print("  IMPORTANT: API Key Required")
    print("=" * 80)
    print("\nPlease ensure you have created a .env file with:")
    print("  ANTHROPIC_API_KEY=sk-ant-...")
    print("  # OR")
    print("  OPENROUTER_API_KEY=sk-or-...")
    print("\nNever commit .env files to git!")
    print("=" * 80)

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
        if args.library:
            # Generate example library
            asyncio.run(generate_example_library(generator))

        elif args.description and args.effects:
            # Custom chain generation
            if args.save:
                asyncio.run(save_chain(generator, args.description, args.effects, args.save))
            else:
                async def run():
                    chain = await generator.generate_parameters(
                        description=args.description,
                        effects=args.effects
                    )
                    print_chain_details(chain)

                asyncio.run(run())

        elif args.description:
            # Generate with default effects
            effects = ["eq", "reverb", "compressor"]
            if args.save:
                asyncio.run(save_chain(generator, args.description, effects, args.save))
            else:
                async def run():
                    chain = await generator.generate_parameters(
                        description=args.description,
                        effects=effects
                    )
                    print_chain_details(chain)

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
