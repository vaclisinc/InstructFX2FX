#!/usr/bin/env python3
"""Demo all working models including GPT."""
import asyncio
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
baseline_dir = os.path.join(script_dir, 'baseline-system')
os.chdir(baseline_dir)
sys.path.insert(0, baseline_dir)

from dotenv import load_dotenv
load_dotenv(os.path.join(baseline_dir, '.env'))

from models.llm_judge.factory import create_provider
from models.llm_judge.types import LLMRequest

async def test_model(name, provider_type, model_name, prompt):
    """Test a specific model."""
    print(f"\n{'='*70}")
    print(f"🤖 Testing {name}")
    print(f"📦 Model: {model_name}")
    print('='*70)
    
    try:
        config = {
            "provider": provider_type,
            "model": model_name
        }
        provider = create_provider(config)
        request = LLMRequest(prompt=prompt, temperature=0.7, max_tokens=150)
        
        print("⏳ Calling API...")
        response = await provider.generate(request)
        
        print(f"\n📝 {name} says:")
        print("-" * 70)
        print(response.content)
        print("-" * 70)
        print(f"✅ Success! Tokens: {response.tokens_used}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    print("\n" + "="*70)
    print("🚀 LLM Provider Demo - Testing Labubu")
    print("="*70)
    
    prompt = "What is 'labubu'? Please explain in 2-3 sentences."
    results = {}
    
    # Test Claude with CORRECT model name
    if os.getenv("ANTHROPIC_API_KEY"):
        results['Claude'] = await test_model(
            "Claude (Anthropic)",
            "anthropic",
            "claude-3-5-sonnet-20241022",  # This is the correct name
            prompt
        )
        
        # If that fails, try another model
        if not results['Claude']:
            print("\n⚠️ Trying different Claude model...")
            results['Claude Alt'] = await test_model(
                "Claude Haiku",
                "anthropic",
                "claude-3-haiku-20240307",
                prompt
            )
    
    # Test GPT via OpenRouter
    if os.getenv("OPENROUTER_API_KEY"):
        results['GPT-3.5'] = await test_model(
            "GPT-3.5 Turbo (via OpenRouter)",
            "openrouter",
            "openai/gpt-3.5-turbo",
            prompt
        )
        
        # Test GPT-4
        results['GPT-4'] = await test_model(
            "GPT-4 Turbo (via OpenRouter)",
            "openrouter",
            "openai/gpt-4-turbo",
            prompt
        )
        
        # Test free Llama
        results['Llama'] = await test_model(
            "Llama 3.2 Free (via OpenRouter)",
            "openrouter",
            "meta-llama/llama-3.2-3b-instruct:free",
            prompt
        )
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Test Summary:")
    print('='*70)
    for model, success in results.items():
        status = "✅ Working" if success else "❌ Failed"
        print(f"  {model}: {status}")
    print('='*70)

if __name__ == "__main__":
    asyncio.run(main())
