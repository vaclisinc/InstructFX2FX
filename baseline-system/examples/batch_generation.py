#!/usr/bin/env python3
"""
Batch Parameter Generation Example

This script demonstrates batch processing of multiple descriptions to generate
audio effect parameters in parallel, with progress tracking, error recovery,
and comparison of different prompt versions.

IMPORTANT: Before running this script, create a .env file with your API keys:

    ANTHROPIC_API_KEY=sk-ant-...
    # OR
    OPENROUTER_API_KEY=sk-or-...

Never commit API keys to git!

This example shows:
1. Parallel batch generation from description lists
2. Progress tracking and reporting
3. Error recovery for failed generations
4. Exporting results to JSON files
5. Comparing different prompt versions (A/B testing)
6. Performance measurement and optimization

Usage:
    python examples/batch_generation.py
    python examples/batch_generation.py --input descriptions.txt
    python examples/batch_generation.py --output results/
    python examples/batch_generation.py --compare-prompts v1 v2
    python examples/batch_generation.py --parallel 5
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from models.llm_judge import create_provider, AnthropicConfig, OpenRouterConfig
from src.generation import (
    ParameterGenerator,
    ParameterGenerationError,
    ValidationError,
    JSONParseError
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of a single parameter generation."""
    description: str
    effects: List[str]
    success: bool
    chain: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0
    prompt_version: str = "v1"

    def to_dict(self):
        """Convert to dictionary."""
        return asdict(self)


# Example descriptions for batch processing
BATCH_DESCRIPTIONS = [
    {"description": "warm and intimate vocal", "effects": ["eq", "reverb"]},
    {"description": "bright energetic guitar", "effects": ["eq", "compressor"]},
    {"description": "dark atmospheric pad", "effects": ["eq", "reverb", "compressor"]},
    {"description": "punchy aggressive drums", "effects": ["eq", "compressor"]},
    {"description": "smooth vintage tone", "effects": ["eq", "compressor", "reverb"]},
    {"description": "clean natural acoustic", "effects": ["eq", "reverb"]},
    {"description": "heavy compressed bass", "effects": ["eq", "compressor"]},
    {"description": "spacious ambient texture", "effects": ["reverb"]},
    {"description": "tight controlled dynamics", "effects": ["compressor"]},
    {"description": "broadcast quality voice", "effects": ["eq", "compressor", "reverb"]},
]


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


async def generate_single(
    generator: ParameterGenerator,
    description: str,
    effects: List[str],
    prompt_version: str = "v1"
) -> GenerationResult:
    """Generate parameters for a single description with error handling."""
    start_time = time.time()

    try:
        chain = await generator.generate_parameters(
            description=description,
            effects=effects
        )

        duration = time.time() - start_time

        return GenerationResult(
            description=description,
            effects=effects,
            success=True,
            chain=chain.to_dict(),
            duration=duration,
            prompt_version=prompt_version
        )

    except (ParameterGenerationError, ValidationError, JSONParseError) as e:
        duration = time.time() - start_time
        logger.error(f"Generation failed for '{description}': {e}")

        return GenerationResult(
            description=description,
            effects=effects,
            success=False,
            error=str(e),
            duration=duration,
            prompt_version=prompt_version
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Unexpected error for '{description}': {e}")

        return GenerationResult(
            description=description,
            effects=effects,
            success=False,
            error=f"Unexpected error: {str(e)}",
            duration=duration,
            prompt_version=prompt_version
        )


async def batch_generate(
    generator: ParameterGenerator,
    batch: List[Dict[str, Any]],
    max_concurrent: int = 3,
    prompt_version: str = "v1"
) -> List[GenerationResult]:
    """Generate parameters for multiple descriptions in parallel.

    Args:
        generator: ParameterGenerator instance
        batch: List of dicts with 'description' and 'effects' keys
        max_concurrent: Maximum concurrent generations
        prompt_version: Prompt version being used

    Returns:
        List of GenerationResult objects
    """
    print_header("Batch Parameter Generation")

    print(f"\nBatch size: {len(batch)} descriptions")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Prompt version: {prompt_version}\n")

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_semaphore(item, index):
        async with semaphore:
            print(f"[{index + 1}/{len(batch)}] Generating: {item['description'][:50]}...")

            result = await generate_single(
                generator,
                item['description'],
                item.get('effects', ["eq", "reverb", "compressor"]),
                prompt_version
            )

            status = "✓" if result.success else "✗"
            print(f"[{index + 1}/{len(batch)}] {status} {result.description[:50]} ({result.duration:.1f}s)")

            return result

    # Create tasks for all generations
    tasks = [
        generate_with_semaphore(item, i)
        for i, item in enumerate(batch)
    ]

    # Execute with progress tracking
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_duration = time.time() - start_time

    # Print summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful

    print("\n" + "-" * 80)
    print(f"Batch Complete:")
    print(f"  Total: {len(results)}")
    print(f"  Successful: {successful} ({successful / len(results) * 100:.1f}%)")
    print(f"  Failed: {failed}")
    print(f"  Total Time: {total_duration:.1f}s")
    print(f"  Average Time: {total_duration / len(results):.1f}s per generation")
    print("-" * 80)

    return results


async def batch_from_file(
    generator: ParameterGenerator,
    input_file: str,
    max_concurrent: int = 3
) -> List[GenerationResult]:
    """Load descriptions from file and generate in batch.

    File format (one per line):
        description | effect1,effect2,effect3
        OR
        description

    Example:
        warm vocal | eq,reverb
        bright guitar | eq,compressor
        atmospheric pad
    """
    print_header(f"Batch Generation from File: {input_file}")

    # Load descriptions
    descriptions = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue  # Skip empty lines and comments

            if '|' in line:
                desc, effects_str = line.split('|', 1)
                effects = [e.strip() for e in effects_str.split(',')]
            else:
                desc = line
                effects = ["eq", "reverb", "compressor"]

            descriptions.append({
                "description": desc.strip(),
                "effects": effects
            })

    print(f"\nLoaded {len(descriptions)} descriptions from {input_file}")

    # Generate
    results = await batch_generate(generator, descriptions, max_concurrent)

    return results


async def compare_prompt_versions(
    provider,
    descriptions: List[Dict[str, Any]],
    versions: List[str]
) -> Dict[str, List[GenerationResult]]:
    """Compare generation results across different prompt versions.

    This is useful for A/B testing prompt improvements.
    """
    print_header("Prompt Version Comparison")

    print(f"\nComparing {len(versions)} prompt versions:")
    for v in versions:
        print(f"  - {v}")

    print(f"\nTest set: {len(descriptions)} descriptions\n")

    all_results = {}

    for version in versions:
        print(f"\n{'=' * 80}")
        print(f"Testing Prompt Version: {version}")
        print('=' * 80)

        # Create generator for this version
        generator = ParameterGenerator(
            llm_provider=provider,
            prompt_version=version
        )

        # Generate batch
        results = await batch_generate(
            generator,
            descriptions,
            max_concurrent=3,
            prompt_version=version
        )

        all_results[version] = results

        # Small delay between versions
        await asyncio.sleep(2)

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    for version in versions:
        results = all_results[version]
        successful = sum(1 for r in results if r.success)
        avg_duration = sum(r.duration for r in results) / len(results)

        print(f"\n{version}:")
        print(f"  Success Rate: {successful}/{len(results)} ({successful / len(results) * 100:.1f}%)")
        print(f"  Avg Duration: {avg_duration:.2f}s")

        # Show failures
        failures = [r for r in results if not r.success]
        if failures:
            print(f"  Failures ({len(failures)}):")
            for r in failures:
                print(f"    - {r.description[:40]}: {r.error}")

    return all_results


def save_results(results: List[GenerationResult], output_dir: str):
    """Save batch results to JSON files.

    Creates:
    - summary.json: Overview and statistics
    - chains/: Individual chain JSON files
    - failures.json: Failed generations
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save individual chains
    chains_dir = output_path / "chains"
    chains_dir.mkdir(exist_ok=True)

    successful_results = [r for r in results if r.success]

    for i, result in enumerate(successful_results, 1):
        # Create safe filename
        safe_desc = result.description[:40].replace(" ", "_").replace("/", "-")
        safe_desc = "".join(c for c in safe_desc if c.isalnum() or c in "_-")

        filename = f"{i:03d}_{safe_desc}.json"
        filepath = chains_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result.chain, f, indent=2)

    print(f"\n✓ Saved {len(successful_results)} chains to {chains_dir}/")

    # Save summary
    summary = {
        "total": len(results),
        "successful": len(successful_results),
        "failed": len(results) - len(successful_results),
        "success_rate": len(successful_results) / len(results) * 100,
        "average_duration": sum(r.duration for r in results) / len(results),
        "prompt_version": results[0].prompt_version if results else "unknown",
        "results": [r.to_dict() for r in results]
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary to {summary_file}")

    # Save failures
    failures = [r for r in results if not r.success]
    if failures:
        failures_file = output_path / "failures.json"
        with open(failures_file, 'w') as f:
            json.dump([r.to_dict() for r in failures], f, indent=2)

        print(f"✓ Saved {len(failures)} failures to {failures_file}")

    print(f"\n✓ All results saved to {output_path}/")


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


async def run_default_batch(generator: ParameterGenerator, output_dir: Optional[str]):
    """Run default batch generation example."""
    print("\nRunning default batch generation with example descriptions...")

    results = await batch_generate(
        generator,
        BATCH_DESCRIPTIONS,
        max_concurrent=3
    )

    if output_dir:
        save_results(results, output_dir)

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch generate audio effect parameters from multiple descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default batch
  python examples/batch_generation.py

  # Load descriptions from file
  python examples/batch_generation.py --input descriptions.txt

  # Save results to directory
  python examples/batch_generation.py --output results/

  # Compare prompt versions
  python examples/batch_generation.py --compare-prompts v1 v2

  # Control parallelism
  python examples/batch_generation.py --parallel 5

  # Combine options
  python examples/batch_generation.py --input descriptions.txt --output results/ --parallel 3

Input file format (descriptions.txt):
  warm and intimate vocal | eq,reverb
  bright energetic guitar | eq,compressor
  atmospheric pad         # Uses default effects
  # This is a comment

IMPORTANT: Create a .env file with your API key before running:
  ANTHROPIC_API_KEY=sk-ant-...
  # OR
  OPENROUTER_API_KEY=sk-or-...
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        metavar="FILE",
        help="Input file with descriptions (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="DIR",
        help="Output directory for results (default: output/batch_TIMESTAMP/)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        metavar="N",
        help="Maximum concurrent generations (default: 3)"
    )
    parser.add_argument(
        "--compare-prompts",
        nargs="+",
        metavar="VERSION",
        help="Compare multiple prompt versions (e.g., v1 v2)"
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

    # Determine output directory
    output_dir = args.output
    if not output_dir and not args.compare_prompts:
        # Generate default output directory with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/batch_{timestamp}"

    # Run batch processing
    try:
        if args.compare_prompts:
            # Prompt version comparison
            descriptions = BATCH_DESCRIPTIONS
            if args.input:
                # Load from file for comparison
                with open(args.input, 'r') as f:
                    descriptions = []
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if '|' in line:
                                desc, effects_str = line.split('|', 1)
                                effects = [e.strip() for e in effects_str.split(',')]
                            else:
                                desc = line
                                effects = ["eq", "reverb", "compressor"]

                            descriptions.append({
                                "description": desc.strip(),
                                "effects": effects
                            })

            results_by_version = asyncio.run(
                compare_prompt_versions(
                    provider,
                    descriptions,
                    args.compare_prompts
                )
            )

            # Save comparison results
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                for version, results in results_by_version.items():
                    version_dir = output_path / version
                    save_results(results, str(version_dir))

                # Save comparison summary
                comparison_file = output_path / "comparison.json"
                comparison_data = {
                    "versions": args.compare_prompts,
                    "test_set_size": len(descriptions),
                    "results": {
                        version: {
                            "success_rate": sum(1 for r in results if r.success) / len(results) * 100,
                            "avg_duration": sum(r.duration for r in results) / len(results),
                            "failures": len([r for r in results if not r.success])
                        }
                        for version, results in results_by_version.items()
                    }
                }

                with open(comparison_file, 'w') as f:
                    json.dump(comparison_data, f, indent=2)

                print(f"\n✓ Comparison results saved to {output_path}/")

        elif args.input:
            # Batch from input file
            generator = ParameterGenerator(
                llm_provider=provider,
                prompt_version="v1"
            )

            results = asyncio.run(
                batch_from_file(generator, args.input, args.parallel)
            )

            if output_dir:
                save_results(results, output_dir)

        else:
            # Default batch
            generator = ParameterGenerator(
                llm_provider=provider,
                prompt_version="v1"
            )

            results = asyncio.run(
                run_default_batch(generator, output_dir)
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
