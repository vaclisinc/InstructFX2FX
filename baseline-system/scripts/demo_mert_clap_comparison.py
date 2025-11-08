#!/usr/bin/env python3
"""
Quick demo to compare CLAP, MERT, and Hybrid retrieval on sample audio.

This script demonstrates the three retrieval modes on a single audio file
using cleaned SocialFX labels.
"""

import sys
import json
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation.audio_description import generate_audio_description_hybrid


def load_cleaned_labels(effect_type: str = 'reverb') -> List[str]:
    """Load cleaned labels for effect type."""
    base_dir = Path(__file__).parent.parent.parent
    labels_file = base_dir / 'baseline-system' / 'data' / 'cleaned_labels' / f'{effect_type}_labels.json'

    with open(labels_file, 'r') as f:
        data = json.load(f)

    return data['labels']


def demo_single_audio(audio_path: str, effect_type: str = 'reverb', device: str = 'cuda:0'):
    """
    Run demo on a single audio file.

    Args:
        audio_path: Path to audio file
        effect_type: Effect type ('reverb', 'eq', 'comp')
        device: Device to use
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return

    print(f"\n{'='*80}")
    print(f"DEMO: Comparing CLAP vs MERT vs Hybrid Retrieval")
    print(f"{'='*80}")
    print(f"Audio: {audio_path.name}")
    print(f"Effect Type: {effect_type}")
    print(f"Device: {device}")

    # Load labels
    print(f"\nLoading cleaned {effect_type} labels...")
    labels = load_cleaned_labels(effect_type)
    print(f"  Loaded {len(labels)} labels")

    # Run each mode
    modes = ['clap', 'mert', 'hybrid']
    k = 10

    results = {}

    for mode in modes:
        print(f"\n{'-'*80}")
        print(f"Mode: {mode.upper()}")
        print(f"{'-'*80}")

        try:
            predictions = generate_audio_description_hybrid(
                audio_path=str(audio_path),
                candidate_descriptions=labels,
                k=k,
                mode=mode,
                device=device
            )

            results[mode] = predictions

            # Display top-10
            print(f"\nTop-{k} predictions:")
            for i, (label, score) in enumerate(predictions):
                print(f"  {i+1:2d}. {label:30s} (score: {score:.4f})")

        except Exception as e:
            print(f"Error in {mode} mode: {e}")
            continue

    # Summary comparison
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Rank':<6} {'CLAP':<30} {'MERT':<30} {'Hybrid':<30}")
    print(f"{'-'*6} {'-'*30} {'-'*30} {'-'*30}")

    for i in range(k):
        clap_label = results.get('clap', [])[i][0] if i < len(results.get('clap', [])) else ''
        mert_label = results.get('mert', [])[i][0] if i < len(results.get('mert', [])) else ''
        hybrid_label = results.get('hybrid', [])[i][0] if i < len(results.get('hybrid', [])) else ''

        print(f"{i+1:<6} {clap_label:<30} {mert_label:<30} {hybrid_label:<30}")

    # Check for consensus in top-5
    if all(mode in results for mode in modes):
        clap_top5 = set(label for label, _ in results['clap'][:5])
        mert_top5 = set(label for label, _ in results['mert'][:5])
        hybrid_top5 = set(label for label, _ in results['hybrid'][:5])

        consensus = clap_top5 & mert_top5 & hybrid_top5

        print(f"\n{'='*80}")
        print(f"Consensus labels (appear in top-5 of all modes): {len(consensus)}")
        print(f"{'='*80}")

        if consensus:
            for label in sorted(consensus):
                print(f"  - {label}")
        else:
            print("  No consensus labels in top-5")

        # Show unique to each mode
        print(f"\nUnique to CLAP top-5: {clap_top5 - mert_top5 - hybrid_top5}")
        print(f"Unique to MERT top-5: {mert_top5 - clap_top5 - hybrid_top5}")
        print(f"Unique to Hybrid top-5: {hybrid_top5 - clap_top5 - mert_top5}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Demo CLAP vs MERT comparison')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--effect-type', type=str, default='reverb',
                        choices=['reverb', 'eq', 'comp'],
                        help='Effect type (default: reverb)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (default: cuda:0)')

    args = parser.parse_args()

    demo_single_audio(args.audio_path, args.effect_type, args.device)


if __name__ == '__main__':
    main()
