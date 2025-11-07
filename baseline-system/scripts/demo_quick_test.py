#!/usr/bin/env python3
"""
Quick demo with reduced label set for fast testing.
Tests GPU acceleration and top-k matching.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation.audio_description import (
    generate_audio_description_with_clap_topk,
    load_socialfx_labels
)


def main():
    # Load Social FX labels (reverb only)
    reverb_labels = load_socialfx_labels(effect_type='reverb')
    print(f"    Total reverb labels available: {len(reverb_labels)}")
    print(f"    Using all {len(reverb_labels)} reverb labels for matching")
    print(f"    Sample: {', '.join(reverb_labels[:10])}...")

    # test audio
    test_audio = Path(__file__).parent.parent / 'audio_samples' / 'reverb' /  'spacious.wav'
    if not test_audio.exists():
        print(f"\n✗ Audio file not found: {test_audio}")
        return

    print(f"\n[2] Testing with: {test_audio.name}")
    print("-" * 70)

    # Test top-k matching
    print("\n[3] Running CLAP top-k matching...")

    try:
        results = generate_audio_description_with_clap_topk(
            str(test_audio),
            reverb_labels,
            k=10  # Top-10 labels
        )

        print("\n" + "=" * 70)
        print("RESULTS - Top 10 Matching Labels:")
        print("=" * 70)

        for i, (label, score) in enumerate(results, 1):
            bar_length = int(score * 50)  # Visual bar
            bar = "█" * bar_length
            print(f"{i:2d}. {label:20s} {score:.4f} {bar}")

        print("\n" + "=" * 70)
        print("✓ Success! GPU-accelerated CLAP is working!")
        print("=" * 70)

        # Show formatted output
        labels_only = [label for label, score in results[:5]]
        print(f"\nFormatted output (top-5):")
        print(f'  "The audio is: {", ".join(labels_only)}"')

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
