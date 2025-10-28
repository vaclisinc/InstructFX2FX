"""
Real API Integration Tests for Parameter Generation and Scoring System.

Tests both Issue #6 (Parameter Generation) and Issue #8 (Scoring System)
across three different LLM providers:
1. Anthropic Claude
2. OpenRouter (Claude 3.5 Sonnet)
3. OpenRouter (Llama 3.2 90B)

This test suite validates that both modules work correctly with real API calls
and compares the quality of outputs across different providers.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pytest
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.llm_judge import create_provider, LLMProvider
from src.generation.parameter_generator import ParameterGenerator
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest

# Load environment variables
load_dotenv()


# Test descriptions for parameter generation
TEST_DESCRIPTIONS = [
    {
        "description": "warm and intimate vocal sound",
        "effects": ["eq", "reverb"],
        "expected_characteristics": ["low-mid boost", "small room", "subtle reverb"]
    },
    {
        "description": "bright and energetic guitar sound",
        "effects": ["eq", "compressor"],
        "expected_characteristics": ["high-frequency boost", "fast attack", "moderate ratio"]
    },
    {
        "description": "spacious and ethereal pad sound",
        "effects": ["reverb", "eq"],
        "expected_characteristics": ["large room", "long decay", "high-frequency roll-off"]
    },
    {
        "description": "punchy and controlled drum sound",
        "effects": ["compressor", "eq"],
        "expected_characteristics": ["fast attack", "high ratio", "low-end emphasis"]
    },
]


# Provider configurations
PROVIDER_CONFIGS = {
    "anthropic_claude": {
        "provider": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-20250514",  # Correct Claude model name
        "name": "Anthropic Claude Sonnet 4"
    },
    "openrouter_llama": {
        "provider": "openrouter",
        "api_key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.2-3b-instruct:free",  # Working free Llama model
        "name": "OpenRouter Llama 3.2 3B (Free)"
    },
    "openai_gpt": {
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
        "name": "OpenAI GPT-4o Mini"
    },
}


class TestResults:
    """Container for test results and statistics."""

    def __init__(self):
        self.results = []
        self.start_time = datetime.now()

    def add_result(self, provider_name: str, test_type: str,
                   description: str, success: bool,
                   data: Dict[str, Any], error: str = None):
        """Add a test result."""
        self.results.append({
            "timestamp": datetime.now().isoformat(),
            "provider": provider_name,
            "test_type": test_type,
            "description": description,
            "success": success,
            "data": data,
            "error": error
        })

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r["success"])
        failed_tests = total_tests - successful_tests

        # Group by provider
        by_provider = {}
        for result in self.results:
            provider = result["provider"]
            if provider not in by_provider:
                by_provider[provider] = {"total": 0, "success": 0, "failed": 0}
            by_provider[provider]["total"] += 1
            if result["success"]:
                by_provider[provider]["success"] += 1
            else:
                by_provider[provider]["failed"] += 1

        # Group by test type
        by_test_type = {}
        for result in self.results:
            test_type = result["test_type"]
            if test_type not in by_test_type:
                by_test_type[test_type] = {"total": 0, "success": 0, "failed": 0}
            by_test_type[test_type]["total"] += 1
            if result["success"]:
                by_test_type[test_type]["success"] += 1
            else:
                by_test_type[test_type]["failed"] += 1

        duration = (datetime.now() - self.start_time).total_seconds()

        return {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": failed_tests,
            "success_rate": f"{(successful_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%",
            "duration_seconds": duration,
            "by_provider": by_provider,
            "by_test_type": by_test_type,
            "completed_at": datetime.now().isoformat()
        }

    def save_to_file(self, filename: str):
        """Save results to JSON file."""
        output_dir = Path(__file__).parent / "test_results"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / filename
        with open(output_file, "w") as f:
            json.dump({
                "summary": self.get_summary(),
                "results": self.results
            }, f, indent=2)

        print(f"\n✓ Test results saved to: {output_file}")
        return output_file


def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are available."""
    available_keys = {}
    for provider_id, config in PROVIDER_CONFIGS.items():
        key_name = config["api_key_env"]
        api_key = os.getenv(key_name)
        available_keys[provider_id] = bool(api_key)
    return available_keys


def get_available_providers() -> List[str]:
    """Get list of providers with available API keys."""
    available_keys = check_api_keys()
    return [provider_id for provider_id, available in available_keys.items() if available]


def create_llm_provider(provider_id: str) -> LLMProvider:
    """Create LLM provider instance from configuration."""
    config = PROVIDER_CONFIGS[provider_id]
    api_key = os.getenv(config["api_key_env"])

    if not api_key:
        raise ValueError(f"API key {config['api_key_env']} not found in environment")

    provider_config = {
        "provider": config["provider"],
        "api_key": api_key,
        "model": config["model"],
        "retry": {
            "max_attempts": 3,
            "initial_delay": 1.0,
            "max_delay": 30.0
        },
        "rate_limit": {
            "requests_per_minute": 50
        }
    }

    return create_provider(provider_config)


@pytest.mark.asyncio
@pytest.mark.integration
class TestParameterGenerationRealAPI:
    """Test Issue #6: Parameter Generation Module with real APIs."""

    async def test_all_providers_parameter_generation(self):
        """Test parameter generation across all available providers."""
        print("\n" + "="*80)
        print("ISSUE #6: PARAMETER GENERATION MODULE - REAL API TESTS")
        print("="*80)

        results = TestResults()
        available_providers = get_available_providers()

        if not available_providers:
            pytest.skip("No API keys available for testing")

        print(f"\n📋 Testing {len(available_providers)} providers with {len(TEST_DESCRIPTIONS)} scenarios")
        print(f"Providers: {', '.join([PROVIDER_CONFIGS[p]['name'] for p in available_providers])}\n")

        for provider_id in available_providers:
            provider_config = PROVIDER_CONFIGS[provider_id]
            print(f"\n{'─'*80}")
            print(f"🔷 Provider: {provider_config['name']}")
            print(f"   Model: {provider_config['model']}")
            print(f"{'─'*80}\n")

            try:
                # Create provider and generator
                llm_provider = create_llm_provider(provider_id)
                generator = ParameterGenerator(
                    llm_provider=llm_provider,
                    prompt_version="v1"
                )

                # Test each description
                for idx, test_case in enumerate(TEST_DESCRIPTIONS, 1):
                    description = test_case["description"]
                    effects = test_case["effects"]

                    print(f"\n  Test {idx}/{len(TEST_DESCRIPTIONS)}: {description}")
                    print(f"  Effects: {effects}")

                    try:
                        # Generate parameters
                        effect_chain = await generator.generate_parameters(
                            description=description,
                            effects=effects,
                            temperature=0.7,
                            max_tokens=2048
                        )

                        # Validate output
                        assert effect_chain is not None
                        assert len(effect_chain.effects) > 0
                        # Allow LLM to enhance description (check if original is contained)
                        assert description in effect_chain.description, \
                            f"Original description '{description}' not found in LLM output '{effect_chain.description}'"
                        assert set(effect_chain.order) == set(effects)

                        # Collect result data
                        result_data = {
                            "effect_count": len(effect_chain.effects),
                            "effect_types": effect_chain.order,
                            "effects": [
                                {
                                    "type": effect_chain.order[i],
                                    "parameters": effect.model_dump()
                                }
                                for i, effect in enumerate(effect_chain.effects)
                            ]
                        }

                        results.add_result(
                            provider_name=provider_config['name'],
                            test_type="parameter_generation",
                            description=description,
                            success=True,
                            data=result_data
                        )

                        print(f"  ✓ Generated {len(effect_chain.effects)} effects successfully")
                        print(f"    Order: {' → '.join(effect_chain.order)}")

                    except Exception as e:
                        print(f"  ✗ Failed: {str(e)}")
                        results.add_result(
                            provider_name=provider_config['name'],
                            test_type="parameter_generation",
                            description=description,
                            success=False,
                            data={},
                            error=str(e)
                        )

            except Exception as e:
                print(f"✗ Provider initialization failed: {e}")
                continue

        # Print summary
        print(f"\n{'='*80}")
        print("PARAMETER GENERATION TEST SUMMARY")
        print(f"{'='*80}\n")
        summary = results.get_summary()
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']} ({summary['success_rate']})")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Duration: {summary['duration_seconds']:.2f}s\n")

        print("By Provider:")
        for provider, stats in summary['by_provider'].items():
            print(f"  {provider}: {stats['success']}/{stats['total']} passed")

        # Save results
        results.save_to_file(f"parameter_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Assert overall success
        assert summary['successful_tests'] > 0, "No tests passed"
        assert summary['successful_tests'] / summary['total_tests'] >= 0.7, \
            f"Success rate {summary['success_rate']} below 70% threshold"


@pytest.mark.asyncio
@pytest.mark.integration
class TestScoringSystemRealAPI:
    """Test Issue #8: Scoring System Implementation with real APIs."""

    async def test_all_providers_scoring_system(self):
        """Test scoring system across all available providers."""
        print("\n" + "="*80)
        print("ISSUE #8: SCORING SYSTEM IMPLEMENTATION - REAL API TESTS")
        print("="*80)

        results = TestResults()
        available_providers = get_available_providers()

        if not available_providers:
            pytest.skip("No API keys available for testing")

        print(f"\n📋 Testing {len(available_providers)} providers with {len(TEST_DESCRIPTIONS)} scenarios")
        print(f"Providers: {', '.join([PROVIDER_CONFIGS[p]['name'] for p in available_providers])}\n")

        # First, generate parameters using the first available provider
        print("🔧 Generating test parameters...")
        test_provider = create_llm_provider(available_providers[0])
        test_generator = ParameterGenerator(test_provider, prompt_version="v1")

        test_parameters = []
        for test_case in TEST_DESCRIPTIONS:
            try:
                effect_chain = await test_generator.generate_parameters(
                    description=test_case["description"],
                    effects=test_case["effects"],
                    temperature=0.7
                )
                test_parameters.append({
                    "description": test_case["description"],
                    "parameters": effect_chain.model_dump()
                })
                print(f"  ✓ Generated parameters for: {test_case['description']}")
            except Exception as e:
                print(f"  ✗ Failed to generate: {e}")
                test_parameters.append({
                    "description": test_case["description"],
                    "parameters": {"effects": [], "order": []}
                })

        # Now test scoring with each provider
        for provider_id in available_providers:
            provider_config = PROVIDER_CONFIGS[provider_id]
            print(f"\n{'─'*80}")
            print(f"🔷 Provider: {provider_config['name']}")
            print(f"   Model: {provider_config['model']}")
            print(f"{'─'*80}\n")

            try:
                # Create provider and scorer
                llm_provider = create_llm_provider(provider_id)
                scorer = ScoringSystem(llm_provider=llm_provider)

                # Test scoring each parameter set
                for idx, test_data in enumerate(test_parameters, 1):
                    description = test_data["description"]
                    parameters = test_data["parameters"]

                    print(f"\n  Test {idx}/{len(test_parameters)}: {description}")

                    try:
                        # Create scoring request
                        scoring_request = ScoringRequest(
                            description=description,
                            parameters=parameters,
                            iteration=0
                        )

                        # Score parameters
                        scoring_response = await scorer.score_parameters(scoring_request)

                        # Validate output
                        assert scoring_response is not None
                        assert 0 <= scoring_response.overall_score <= 100
                        assert 0 <= scoring_response.confidence <= 1
                        assert len(scoring_response.dimensions) > 0
                        assert len(scoring_response.feedback) > 0

                        # Collect result data
                        result_data = {
                            "overall_score": scoring_response.overall_score,
                            "confidence": scoring_response.confidence,
                            "dimensions": [
                                {
                                    "name": dim.name,
                                    "score": dim.score,
                                    "reasoning": dim.reasoning[:100] + "..."  # Truncate for readability
                                }
                                for dim in scoring_response.dimensions
                            ],
                            "feedback_length": len(scoring_response.feedback),
                            "suggestions_count": len(scoring_response.suggestions)
                        }

                        results.add_result(
                            provider_name=provider_config['name'],
                            test_type="scoring_system",
                            description=description,
                            success=True,
                            data=result_data
                        )

                        print(f"  ✓ Score: {scoring_response.overall_score:.1f}/100")
                        print(f"    Confidence: {scoring_response.confidence:.2f}")
                        print(f"    Dimensions: {len(scoring_response.dimensions)}")
                        for dim in scoring_response.dimensions:
                            print(f"      - {dim.name}: {dim.score:.1f}")

                    except Exception as e:
                        print(f"  ✗ Failed: {str(e)}")
                        results.add_result(
                            provider_name=provider_config['name'],
                            test_type="scoring_system",
                            description=description,
                            success=False,
                            data={},
                            error=str(e)
                        )

            except Exception as e:
                print(f"✗ Provider initialization failed: {e}")
                continue

        # Print summary
        print(f"\n{'='*80}")
        print("SCORING SYSTEM TEST SUMMARY")
        print(f"{'='*80}\n")
        summary = results.get_summary()
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']} ({summary['success_rate']})")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Duration: {summary['duration_seconds']:.2f}s\n")

        print("By Provider:")
        for provider, stats in summary['by_provider'].items():
            print(f"  {provider}: {stats['success']}/{stats['total']} passed")

        # Save results
        results.save_to_file(f"scoring_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Assert overall success
        assert summary['successful_tests'] > 0, "No tests passed"
        assert summary['successful_tests'] / summary['total_tests'] >= 0.7, \
            f"Success rate {summary['success_rate']} below 70% threshold"


@pytest.mark.asyncio
@pytest.mark.integration
class TestEndToEndIntegration:
    """Test complete workflow: Generate → Score → Compare across providers."""

    async def test_full_pipeline_comparison(self):
        """Test full parameter generation + scoring pipeline across all providers."""
        print("\n" + "="*80)
        print("END-TO-END INTEGRATION TEST: GENERATE + SCORE COMPARISON")
        print("="*80)

        results = TestResults()
        available_providers = get_available_providers()

        if len(available_providers) < 2:
            pytest.skip("Need at least 2 providers for comparison testing")

        print(f"\n📋 Comparing {len(available_providers)} providers on full pipeline")
        print(f"Providers: {', '.join([PROVIDER_CONFIGS[p]['name'] for p in available_providers])}\n")

        # Use first description for comparison
        test_description = TEST_DESCRIPTIONS[0]["description"]
        test_effects = TEST_DESCRIPTIONS[0]["effects"]

        print(f"Test case: {test_description}")
        print(f"Effects: {test_effects}\n")

        comparison_results = []

        for provider_id in available_providers:
            provider_config = PROVIDER_CONFIGS[provider_id]
            print(f"\n{'─'*80}")
            print(f"🔷 Provider: {provider_config['name']}")
            print(f"{'─'*80}")

            try:
                # Create provider, generator, and scorer
                llm_provider = create_llm_provider(provider_id)
                generator = ParameterGenerator(llm_provider, prompt_version="v1")
                scorer = ScoringSystem(llm_provider)

                # Step 1: Generate parameters
                print("  Step 1: Generating parameters...")
                effect_chain = await generator.generate_parameters(
                    description=test_description,
                    effects=test_effects,
                    temperature=0.7
                )
                print(f"    ✓ Generated {len(effect_chain.effects)} effects")

                # Step 2: Score the generated parameters
                print("  Step 2: Scoring parameters...")
                scoring_request = ScoringRequest(
                    description=test_description,
                    parameters=effect_chain.model_dump(),
                    iteration=0
                )
                scoring_response = await scorer.score_parameters(scoring_request)
                print(f"    ✓ Score: {scoring_response.overall_score:.1f}/100")
                print(f"    ✓ Confidence: {scoring_response.confidence:.2f}")

                # Collect comparison data
                comparison_data = {
                    "provider": provider_config['name'],
                    "generation": {
                        "effect_count": len(effect_chain.effects),
                        "effect_order": effect_chain.order,
                        "parameters": effect_chain.model_dump()
                    },
                    "scoring": {
                        "overall_score": scoring_response.overall_score,
                        "confidence": scoring_response.confidence,
                        "dimensions": {dim.name: dim.score for dim in scoring_response.dimensions},
                        "feedback": scoring_response.feedback[:200]
                    }
                }
                comparison_results.append(comparison_data)

                results.add_result(
                    provider_name=provider_config['name'],
                    test_type="end_to_end",
                    description=test_description,
                    success=True,
                    data=comparison_data
                )

            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                results.add_result(
                    provider_name=provider_config['name'],
                    test_type="end_to_end",
                    description=test_description,
                    success=False,
                    data={},
                    error=str(e)
                )

        # Print comparison summary
        print(f"\n{'='*80}")
        print("PROVIDER COMPARISON SUMMARY")
        print(f"{'='*80}\n")

        if len(comparison_results) >= 2:
            print("Score Comparison:")
            for result in comparison_results:
                provider = result["provider"]
                score = result["scoring"]["overall_score"]
                confidence = result["scoring"]["confidence"]
                print(f"  {provider}:")
                print(f"    Score: {score:.1f}/100")
                print(f"    Confidence: {confidence:.2f}")
                print(f"    Dimensions: {result['scoring']['dimensions']}")

            # Compare scores
            scores = [r["scoring"]["overall_score"] for r in comparison_results]
            score_variance = max(scores) - min(scores)
            print(f"\n  Score variance: {score_variance:.1f} points")
            print(f"  Average score: {sum(scores)/len(scores):.1f}")

        # Save results
        results.save_to_file(f"end_to_end_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        # Assert success
        summary = results.get_summary()
        assert summary['successful_tests'] >= 2, "Need at least 2 successful providers for comparison"


if __name__ == "__main__":
    """Run tests directly with python."""
    import sys

    print("\n" + "="*80)
    print("REAL API INTEGRATION TESTS - ISSUES #6 AND #8")
    print("="*80)

    # Check API keys
    print("\nChecking API keys...")
    available_keys = check_api_keys()
    for provider_id, config in PROVIDER_CONFIGS.items():
        status = "✓ Available" if available_keys[provider_id] else "✗ Missing"
        print(f"  {config['name']}: {status}")

    available_providers = get_available_providers()
    if not available_providers:
        print("\n✗ No API keys available. Please set up .env file.")
        sys.exit(1)

    print(f"\n✓ {len(available_providers)} provider(s) available for testing\n")

    # Run tests
    pytest.main([
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "integration"
    ])
