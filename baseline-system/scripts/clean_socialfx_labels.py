#!/usr/bin/env python3
"""
Improved SocialFX Label Cleaning Script V2

Uses rule-based consolidation instead of string similarity to avoid bad merges.
"""

import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter
import argparse


def clean_label(label: str) -> Optional[str]:
    """Basic label cleaning."""
    label = label.strip()

    if label.lower() in ['none of the above', 'didnotagree', 'english', '']:
        return None

    label = label.lower()
    label = re.sub(r'[^a-z0-9\-_\s]', '', label)
    label = re.sub(r'[\s_]+', '-', label)
    label = re.sub(r'-+', '-', label)
    label = label.strip('-')

    if len(label) < 2:
        return None

    return label


def get_base_form(label: str) -> str:
    """
    Get base form of label using linguistic rules, not string similarity.

    Rules:
    - Remove common suffixes: -er, -est, -ed, -ing, -s, -es, -ly
    - Only if they form valid patterns
    """
    # Handle hyphenated compounds - keep them as-is unless they have suffixes
    if '-' in label:
        parts = label.split('-')
        # Only process last part for suffixes
        last_part = parts[-1]
        base_last = get_base_form(last_part)
        if base_last != last_part:
            # Only consolidate if suffix removal makes sense
            return '-'.join(parts[:-1] + [base_last])
        return label

    # Comparatives: louder -> loud, softer -> soft
    if label.endswith('er') and len(label) > 4:
        base = label[:-2]
        # Check if it's a valid comparative (ends in consonant)
        if base and base[-1] not in 'aeiou':
            return base

    # Superlatives: loudest -> loud
    if label.endswith('est') and len(label) > 5:
        base = label[:-3]
        if base and base[-1] not in 'aeiou':
            return base

    # Plurals: echoes -> echo, bells -> bell
    if label.endswith('es') and len(label) > 4:
        base = label[:-2]
        # Common -es plurals: echo->echoes, church->churches
        if base.endswith(('ch', 'sh', 'x', 'z', 'o')):
            return base
    elif label.endswith('s') and len(label) > 3:
        # Simple plural: bell -> bells
        base = label[:-1]
        # Don't remove 's' from words that naturally end in 's'
        if not base.endswith('s'):
            return base

    # -ing forms: echoing -> echo
    if label.endswith('ing') and len(label) > 5:
        base = label[:-3]
        # Handle double consonant: running -> run
        if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in 'aeiou':
            return base[:-1]
        return base

    # -ed forms: muffled -> muffle
    if label.endswith('ed') and len(label) > 4:
        base = label[:-2]
        return base

    # -ly forms: lovely -> love (be careful here)
    if label.endswith('ly') and len(label) > 4:
        base = label[:-2]
        return base

    return label


def consolidate_labels(labels: List[str]) -> Dict[str, str]:
    """
    Consolidate labels using rule-based approach.

    Returns:
        Mapping from original label -> base label
    """
    consolidation_map = {}
    base_to_originals = {}

    for label in sorted(set(labels), key=len):  # Process shorter labels first
        base = get_base_form(label)

        if base not in base_to_originals:
            base_to_originals[base] = label

        # Map this label to the first (shortest) label with this base form
        consolidation_map[label] = base_to_originals[base]

    return consolidation_map


def extract_labels_from_csv(csv_path: str, label_column: str, is_comma_separated: bool = True) -> List[str]:
    """Extract labels from CSV file."""
    labels = []

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)

        for row in reader:
            if label_column not in row:
                continue

            raw_value = row[label_column]

            if is_comma_separated:
                raw_labels = raw_value.split(',')
            else:
                raw_labels = [raw_value]

            for raw_label in raw_labels:
                cleaned = clean_label(raw_label)
                if cleaned:
                    labels.append(cleaned)

    return labels


def filter_by_frequency(labels: List[str], min_frequency: int) -> Tuple[Set[str], Counter]:
    """Filter labels by minimum frequency."""
    frequency = Counter(labels)
    filtered = {label for label, count in frequency.items() if count >= min_frequency}
    return filtered, frequency


def process_effect_labels(
    csv_path: str,
    label_column: str,
    is_comma_separated: bool,
    min_frequency: int,
    effect_name: str
) -> Dict:
    """Process labels for a single effect type."""
    print(f"\n{'='*60}")
    print(f"Processing {effect_name.upper()} labels from {csv_path}")
    print(f"{'='*60}")

    # Step 1: Extract and clean
    print(f"Step 1: Extracting labels from '{label_column}' column...")
    raw_labels = extract_labels_from_csv(csv_path, label_column, is_comma_separated)
    print(f"  Found {len(raw_labels)} raw label instances")
    print(f"  Unique labels: {len(set(raw_labels))}")

    # Step 2: Frequency filtering
    print(f"\nStep 2: Filtering by frequency (min={min_frequency})...")
    filtered_labels, frequency = filter_by_frequency(raw_labels, min_frequency)
    print(f"  Labels after frequency filtering: {len(filtered_labels)}")
    print(f"  Removed {len(set(raw_labels)) - len(filtered_labels)} low-frequency labels")

    # Step 3: Rule-based consolidation
    print(f"\nStep 3: Consolidating using linguistic rules...")
    filtered_list = list(filtered_labels)
    consolidation_map = consolidate_labels(filtered_list)

    # Get unique base labels
    base_labels = sorted(set(consolidation_map.values()))
    print(f"  Base labels after consolidation: {len(base_labels)}")

    # Show consolidation examples
    from collections import defaultdict
    consolidations = defaultdict(list)
    for original, base in consolidation_map.items():
        if original != base:
            consolidations[base].append(original)

    if consolidations:
        print(f"\n  Example consolidations (showing first 15):")
        for i, (base, variants) in enumerate(list(consolidations.items())[:15]):
            print(f"    '{base}' <- {variants}")

    # Calculate consolidated frequencies
    consolidated_frequency = Counter()
    for label in raw_labels:
        if label in consolidation_map:
            base = consolidation_map[label]
            consolidated_frequency[base] += 1

    # Get top labels
    top_20 = consolidated_frequency.most_common(20)
    print(f"\n  Top 20 most frequent labels:")
    for label, count in top_20:
        print(f"    {label:30s} : {count:4d}")

    # Prepare output
    result = {
        'effect_type': effect_name,
        'labels': base_labels,
        'metadata': {
            'total_instances': len(raw_labels),
            'unique_raw_labels': len(set(raw_labels)),
            'after_frequency_filter': len(filtered_labels),
            'final_consolidated_labels': len(base_labels),
            'min_frequency_threshold': min_frequency,
            'consolidation_method': 'rule-based (suffixes)',
            'consolidation_count': len(consolidations),
        },
        'frequency': dict(consolidated_frequency),
        'consolidation_map': consolidation_map,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Clean and consolidate SocialFX labels (V2 - rule-based)')
    parser.add_argument('--min-frequency', type=int, default=3,
                        help='Minimum label frequency to keep (default: 3)')
    parser.add_argument('--output-dir', type=str,
                        default='baseline-system/data/cleaned_labels',
                        help='Output directory for cleaned labels')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data_dir = base_dir / 'ref' / 'socialfx_raw' / 'data' / 'raw'

    # Effect configurations
    effects = [
        {
            'name': 'reverb',
            'csv_path': raw_data_dir / 'reverb_contributions.csv',
            'label_column': 'agreed',
            'is_comma_separated': True,
        },
        {
            'name': 'eq',
            'csv_path': raw_data_dir / 'eq_contributions.csv',
            'label_column': 'descriptor',
            'is_comma_separated': False,
        },
        {
            'name': 'comp',
            'csv_path': raw_data_dir / 'comp_contributions.csv',
            'label_column': 'agreed',
            'is_comma_separated': True,
        },
    ]

    # Process each effect
    results = {}
    for effect_config in effects:
        result = process_effect_labels(
            csv_path=str(effect_config['csv_path']),
            label_column=effect_config['label_column'],
            is_comma_separated=effect_config['is_comma_separated'],
            min_frequency=args.min_frequency,
            effect_name=effect_config['name']
        )
        results[effect_config['name']] = result

        # Save individual effect labels
        output_file = output_dir / f"{effect_config['name']}_labels.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✓ Saved to {output_file}")

    # Save summary
    summary = {
        'parameters': {
            'min_frequency': args.min_frequency,
            'consolidation_method': 'rule-based (linguistic suffixes)',
        },
        'summary': {
            effect: {
                'total_instances': results[effect]['metadata']['total_instances'],
                'unique_raw': results[effect]['metadata']['unique_raw_labels'],
                'final_labels': results[effect]['metadata']['final_consolidated_labels'],
            }
            for effect in ['reverb', 'eq', 'comp']
        }
    }

    summary_file = output_dir / 'cleaning_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("CLEANING COMPLETE")
    print(f"{'='*60}")
    print("\nSummary:")
    for effect in ['reverb', 'eq', 'comp']:
        meta = results[effect]['metadata']
        print(f"\n{effect.upper()}:")
        print(f"  Raw instances: {meta['total_instances']}")
        print(f"  Unique raw: {meta['unique_raw_labels']}")
        print(f"  After frequency filter: {meta['after_frequency_filter']}")
        print(f"  Final consolidated: {meta['final_consolidated_labels']}")
        print(f"  Reduction: {100 * (1 - meta['final_consolidated_labels'] / meta['unique_raw_labels']):.1f}%")

    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
