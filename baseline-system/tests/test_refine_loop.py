"""
Test suite for refinement loop functionality.

Each test has a specific purpose:
1. Complete workflow test - shows detailed parameter evolution
2. Max iterations test - verifies iteration limit
3. Fallback test - tests behavior without audio file
"""

import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env file at module import time
_env_path = Path(__file__).parent.parent / '.env'
if _env_path.exists():
    load_dotenv(_env_path)


class TestRefinementLoop:
    """Tests for the refine_loop function with detailed output."""

    def test_refine_loop_complete_workflow(self):
        """
        Test complete refinement loop workflow with detailed parameter tracking.
        
        Purpose: Main integration test showing full refinement process
        """
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import refine_loop
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        config['refinement']['max_iterations'] = 4

        user_prompt = "warm and spacious with cathedral reverb"
        audio_path = str(Path(__file__).parent.parent / 'audio_samples' / 'piano.wav')

        print("\n" + "="*80)
        print("REFINEMENT LOOP - COMPLETE WORKFLOW TEST")
        print("="*80)
        print(f"Model: {config['llm']['provider']}/{config['llm']['model']}")
        print(f"User Prompt: '{user_prompt}'")
        print(f"Audio File: {audio_path}")
        print(f"Max Iterations: {config['refinement']['max_iterations']}")
        print(f"Target Score: {config['refinement']['target_score']}")
        print(f"Convergence Threshold: {config['refinement']['convergence_threshold']}")
        print("="*80 + "\n")

        result = refine_loop(user_prompt, audio_path, config)

        print("\n" + "="*80)
        print("SCORE PROGRESSION")
        print("="*80)
        print(f"Total Iterations: {len(result['history'])}")
        for entry in result['history']:
            print(f"  Iteration {entry['iteration']}: {entry['score']:.2f}")

        print(f"\nBest Score: {max(entry['score'] for entry in result['history']):.2f}")
        print(f"Initial Score: {result['history'][0]['score']:.2f}")
        print(f"Final Score: {result['history'][-1]['score']:.2f}")

        print("\n" + "="*80)
        print("DETAILED PARAMETER EVOLUTION")
        print("="*80)
        for entry in result['history']:
            print(f"\n--- Iteration {entry['iteration']} (Score: {entry['score']:.2f}) ---")

            print("  Reverb:")
            for key, value in entry['params']['reverb'].items():
                print(f"    {key}: {value}")

            print("  EQ:")
            for i, band in enumerate(entry['params']['eq']):
                print(f"    Band {i+1}: freq={band['freq']}, gain={band['gain']}, Q={band['Q']}")

            print("  Compressor:")
            for key, value in entry['params']['compressor'].items():
                print(f"    {key}: {value}")

        print("\n" + "="*80)
        print("BEST PARAMETERS (Highest Score)")
        print("="*80)
        print(json.dumps(result['best_params'], indent=2))
        print("\n" + "="*80 + "\n")

        # Assertions
        assert isinstance(result, dict)
        assert 'best_params' in result
        assert 'history' in result
        assert len(result['history']) > 0
        assert len(result['history']) <= config['refinement']['max_iterations']

        for entry in result['history']:
            assert 'iteration' in entry
            assert 'params' in entry
            assert 'score' in entry
            assert isinstance(entry['score'], (int, float))
            assert 0 <= entry['score'] <= 10

    def test_refine_loop_max_iterations(self):
        """
        Test that refinement loop respects max_iterations limit.
        
        Purpose: Verify iteration limit control works correctly
        """
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import refine_loop
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        config['refinement']['convergence_threshold'] = 0.01
        config['refinement']['max_iterations'] = 2

        user_prompt = "bright and energetic"
        audio_path = str(Path(__file__).parent.parent / 'audio_samples' / 'piano.wav')

        print("\n" + "="*80)
        print("MAX ITERATIONS TEST")
        print("="*80)
        print(f"Model: {config['llm']['provider']}/{config['llm']['model']}")
        print(f"Max Iterations: {config['refinement']['max_iterations']}")
        print(f"Convergence Threshold: {config['refinement']['convergence_threshold']} (very low)")
        print("Expected: Should stop at exactly max_iterations")
        print("="*80 + "\n")

        result = refine_loop(user_prompt, audio_path, config)

        print(f"\nActual Iterations: {len(result['history'])}")
        print(f"Expected Max: {config['refinement']['max_iterations']}")

        for entry in result['history']:
            print(f"  Iteration {entry['iteration']}: Score {entry['score']:.2f}")

        print("\n" + "="*80 + "\n")

        assert len(result['history']) <= config['refinement']['max_iterations']

    def test_refine_loop_without_audio_file(self):
        """
        Test refinement loop fallback when audio file doesn't exist.
        
        Purpose: Verify graceful fallback to parameter-based description
        """
        from tests.conftest import require_api_key
        require_api_key('OPENROUTER_API_KEY')

        from src.generation.parameters import refine_loop
        from src.config.loader import load_config

        config_path = Path(__file__).parent.parent / 'configs' / 'default.yaml'
        config = load_config(str(config_path))

        config['refinement']['max_iterations'] = 2

        user_prompt = "warm and spacious"
        audio_path = "nonexistent_audio.wav"

        print("\n" + "="*80)
        print("FALLBACK TEST (No Audio File)")
        print("="*80)
        print(f"Model: {config['llm']['provider']}/{config['llm']['model']}")
        print(f"User Prompt: '{user_prompt}'")
        print(f"Audio Path: {audio_path} (does not exist)")
        print("Expected: Should use parameter-based description as fallback")
        print("="*80 + "\n")

        result = refine_loop(user_prompt, audio_path, config)

        print(f"\nCompleted {len(result['history'])} iterations")
        print("Scores:", [entry['score'] for entry in result['history']])
        print("\n" + "="*80 + "\n")

        assert isinstance(result, dict)
        assert 'best_params' in result
        assert 'history' in result
        assert len(result['history']) > 0
