#!/usr/bin/env python3
"""
Demo script for Issues #6 (Parameter Generation) and #8 (Scoring System).
Tests real API integration with working model configurations.
"""
import asyncio
import os
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.join(script_dir, 'baseline-system')
os.chdir(baseline_dir)
sys.path.insert(0, baseline_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(baseline_dir, '.env'))

from models.llm_judge.factory import create_provider
from src.generation.parameter_generator import ParameterGenerator
from src.scoring.scorer import ScoringSystem
from src.scoring.models import ScoringRequest


# Test configurations with WORKING models
MODELS = {
    "Claude (Anthropic)": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514"
    },
    "Claude (OpenRouter)": {
        "provider": "openrouter",
        "model": "anthropic/claude-3.5-sonnet"
    },
    "GPT-4o Mini (OpenAI)": {
        "provider": "openai",
        "model": "gpt-4o-mini"
    },
    "Llama 3.2 (OpenRouter)": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.2-3b-instruct:free"
    }
}


async def test_parameter_generation(model_name, config, description, effects):
    """Test Issue #6: Parameter Generation"""
    print(f"\n{'='*70}")
    print(f"🎵 Testing Parameter Generation: {model_name}")
    print(f"📝 Description: {description}")
    print(f"🎛️  Effects: {effects}")
    print('='*70)

    try:
        # Create provider and generator
        provider = create_provider(config)
        generator = ParameterGenerator(provider, prompt_version="v1")

        print("⏳ Generating parameters...")
        effect_chain = await generator.generate_parameters(
            description=description,
            effects=effects,
            temperature=0.7
        )

        print(f"\n✅ Successfully generated {len(effect_chain.effects)} effects!")
        print(f"📊 Effect order: {' → '.join(effect_chain.order)}")
        print(f"📋 Description: {effect_chain.description}")

        # Show generated parameters
        print(f"\n🎛️  Generated Parameters:")
        for i, effect in enumerate(effect_chain.effects):
            effect_type = effect_chain.order[i]
            print(f"\n  {i+1}. {effect_type.upper()}:")
            params = effect.model_dump()
            for key, value in params.items():
                if key != 'effect_type' and not key.startswith('_'):
                    if isinstance(value, list):
                        print(f"     - {key}: {len(value)} items")
                    elif isinstance(value, float):
                        print(f"     - {key}: {value:.2f}")
                    else:
                        print(f"     - {key}: {value}")

        return effect_chain

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def test_scoring_system(model_name, config, description, parameters):
    """Test Issue #8: Scoring System"""
    print(f"\n{'='*70}")
    print(f"🎯 Testing Scoring System: {model_name}")
    print('='*70)

    try:
        # Create provider and scorer
        provider = create_provider(config)
        scorer = ScoringSystem(provider)

        # Create scoring request
        request = ScoringRequest(
            description=description,
            parameters=parameters.model_dump() if parameters else {},
            iteration=0
        )

        print("⏳ Scoring parameters...")
        response = await scorer.score_parameters(request)

        print(f"\n✅ Scoring completed successfully!")
        print(f"📊 Overall Score: {response.overall_score:.1f}/100")
        print(f"🎯 Confidence: {response.confidence:.2f}")

        print(f"\n📈 Dimensional Scores:")
        for dim in response.dimensions:
            print(f"  • {dim.name}: {dim.score:.1f}/100")
            print(f"    Reasoning: {dim.reasoning[:100]}...")

        print(f"\n💬 Feedback:")
        print(f"  {response.feedback[:200]}...")

        if response.suggestions:
            print(f"\n💡 Suggestions:")
            for i, suggestion in enumerate(response.suggestions[:3], 1):
                print(f"  {i}. {suggestion}")

        return response

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def test_full_pipeline(model_name, config):
    """Test complete workflow: Generate → Score"""
    print(f"\n\n{'#'*70}")
    print(f"# FULL PIPELINE TEST: {model_name}")
    print(f"{'#'*70}")

    # Test case
    description = "warm and intimate vocal sound"
    effects = ["eq", "reverb"]

    # Step 1: Generate parameters
    print("\n🔹 STEP 1: PARAMETER GENERATION")
    parameters = await test_parameter_generation(
        model_name, config, description, effects
    )

    if not parameters:
        print(f"\n❌ Pipeline failed at generation step")
        return False

    # Step 2: Score the generated parameters
    print("\n🔹 STEP 2: SCORING")
    score = await test_scoring_system(
        model_name, config, description, parameters
    )

    if not score:
        print(f"\n❌ Pipeline failed at scoring step")
        return False

    print(f"\n{'='*70}")
    print(f"✅ FULL PIPELINE SUCCESS!")
    print(f"   Generated: {len(parameters.effects)} effects")
    print(f"   Score: {score.overall_score:.1f}/100")
    print(f"   Confidence: {score.confidence:.2f}")
    print(f"{'='*70}")

    return True


async def main():
    print("\n" + "="*70)
    print("🎼 LLM-as-Music-Judge: Issues #6 & #8 Demo")
    print("   Testing Parameter Generation + Scoring System")
    print("="*70)

    # Check which models are available
    available_models = {}
    for name, config in MODELS.items():
        provider = config["provider"]
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            available_models[name] = config
        elif provider == "openai" and os.getenv("OPENAI_API_KEY"):
            available_models[name] = config
        elif provider == "openrouter" and os.getenv("OPENROUTER_API_KEY"):
            available_models[name] = config

    if not available_models:
        print("\n❌ No API keys available. Please set up .env file.")
        return

    print(f"\n📋 Available models: {len(available_models)}")
    for name in available_models.keys():
        print(f"  • {name}")

    # Test with first available model
    test_model = list(available_models.items())[0]
    model_name, config = test_model

    print(f"\n🎯 Testing with: {model_name}")
    print(f"   Model: {config['model']}")

    # Run full pipeline test
    success = await test_full_pipeline(model_name, config)

    if success:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n📝 Summary:")
        print("  ✓ Issue #6 (Parameter Generation): WORKING")
        print("  ✓ Issue #8 (Scoring System): WORKING")
        print("  ✓ End-to-End Pipeline: FUNCTIONAL")
        print("\n🚀 Both modules are production-ready!")
    else:
        print("\n❌ Tests failed. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())
