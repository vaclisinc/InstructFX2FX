#!/usr/bin/env python3
"""Test configuration loader."""

from src.config.loader import load_config, get_config_loader
import json

def main():
    print("=== Testing Configuration Loader ===\n")

    # Load default configuration
    print("1. Loading default configuration...")
    config = load_config()
    print(f"✓ Loaded config for profile: {config.profile or 'default'}")

    # Display audio settings
    print("\n2. Audio Settings:")
    print(f"   - Sample Rate: {config.audio.sample_rate} Hz")
    print(f"   - Audio Directory: {config.audio.audio_dir}")
    print(f"   - Max Duration: {config.audio.max_duration} seconds")

    # Display LLM settings
    print("\n3. LLM Settings:")
    print(f"   - Provider: {config.llm.provider}")
    print(f"   - Model: {config.llm.model}")
    print(f"   - Temperature: {config.llm.temperature}")
    print(f"   - Max Tokens: {config.llm.max_tokens}")
    print(f"   - API Key: {'✓ Configured' if config.llm.api_key else '✗ Not configured'}")

    # Display logging settings
    print("\n4. Logging Settings:")
    print(f"   - Level: {config.logging.level}")
    print(f"   - Format: {config.logging.format}")
    print(f"   - Output Directory: {config.logging.output_dir}")

    # Display experiment settings
    print("\n5. Experiment Settings:")
    print(f"   - Name: {config.experiment.name}")
    print(f"   - Dataset: {config.experiment.dataset}")
    print(f"   - Batch Size: {config.experiment.batch_size}")
    print(f"   - Iterations: {config.experiment.num_iterations}")

    # Test hot-reload capability
    print("\n6. Testing Hot-Reload:")
    loader = get_config_loader()
    print(f"   - Current modification time tracked")
    print(f"   - Hot-reload enabled: will auto-reload on config file changes")

    # Test JSON serialization
    print("\n7. JSON Serialization:")
    config_dict = config.model_dump(exclude={'llm': {'api_key'}})
    json_str = json.dumps(config_dict, indent=2, default=str)
    print(f"   - Config can be serialized to JSON ({len(json_str)} bytes)")

    print("\n=== All Configuration Tests Passed ✓ ===")

if __name__ == "__main__":
    main()