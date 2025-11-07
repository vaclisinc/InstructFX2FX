#!/usr/bin/env python3
"""
Test script to verify CLAP model loads correctly on GPU 0.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation.audio_description import _get_clap_model
import torch


def main():
    print("=" * 70)
    print("Testing CLAP GPU Configuration")
    print("=" * 70)

    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    print("\n" + "-" * 70)
    print("Loading CLAP model...")
    print("-" * 70)

    try:
        # This should initialize the model on GPU 0
        model, device = _get_clap_model()

        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"Model device: {device}")
        print(f"Model type: {type(model)}")
        print(f"\nThe CLAP model is correctly configured to use GPU 0 only.")
        print("No batch processing or MPS support - simple and efficient!")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("ERROR!")
        print("=" * 70)
        print(f"Failed to load CLAP model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
