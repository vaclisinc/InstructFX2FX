#!/usr/bin/env python3
"""
Extract unique 'agreed' labels from Social FX dataset.

Fast script to extract all unique perceptual labels from the Social FX
contribution CSVs for use in CLAP-based audio description.
"""

import csv
import json
import re
from pathlib import Path
from collections import Counter
from typing import Optional


def clean_label(label: str) -> Optional[str]:
    """
    Clean and normalize a label string.

    Args:
        label: Raw label string from CSV

    Returns:
        Cleaned label or None if invalid
    """
    if not label or not isinstance(label, str):
        return None

    # Strip whitespace
    label = label.strip()

    # Skip empty or invalid labels
    if not label:
        return None

    # Skip meta-labels
    if label.lower() in ['none of the above', 'didnotagree', 'english', '']:
        return None

    # Convert to lowercase
    label = label.lower()

    # Remove special characters, keep only alphanumeric, hyphens, underscores
    label = re.sub(r'[^a-z0-9\-_]', '-', label)

    # Remove multiple consecutive hyphens
    label = re.sub(r'-+', '-', label)

    # Remove leading/trailing hyphens
    label = label.strip('-')

    # Skip if too short (likely noise)
    if len(label) < 2:
        return None

    return label


def extract_labels_from_csv(csv_path: str, column_name: str = 'agreed') -> list[str]:
    """
    Extract all labels from a Social FX contribution CSV.

    Args:
        csv_path: Path to CSV file
        column_name: Name of the column containing labels (default: 'agreed')

    Returns:
        List of all labels (with duplicates, for frequency counting)
    """
    labels = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                # Extract the specified column
                label_value = row.get(column_name, '')

                if not label_value or not label_value.strip():
                    continue

                # Split comma-separated labels
                raw_labels = label_value.split(',')

                for raw_label in raw_labels:
                    cleaned = clean_label(raw_label)
                    if cleaned:
                        labels.append(cleaned)

    except Exception as e:
        print(f"Warning: Error reading {csv_path}: {e}")

    return labels


def extract_all_unique_labels(contributions_dir: str) -> dict:
    """
    Extract all unique labels from all Social FX contribution CSVs.

    Args:
        contributions_dir: Path to directory containing CSV files

    Returns:
        Dict with keys: 'reverb', 'eq', 'compressor', 'all'
    """
    csv_files = {
        'reverb': ('reverb_contributions.csv', 'agreed'),
        'eq': ('eq_contributions.csv', 'descriptor'),  # EQ uses 'descriptor' column
        'compressor': ('comp_contributions.csv', 'agreed')
    }

    labels_by_effect = {}
    all_labels = []

    for effect_type, (csv_file, column_name) in csv_files.items():
        csv_path = Path(contributions_dir) / csv_file

        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping...")
            labels_by_effect[effect_type] = []
            continue

        print(f"Processing {csv_file} (column: '{column_name}')...")
        labels = extract_labels_from_csv(str(csv_path), column_name=column_name)

        # Store unique labels for this effect type
        unique_labels_for_effect = sorted(set(labels))
        labels_by_effect[effect_type] = unique_labels_for_effect

        # Add to all labels
        all_labels.extend(labels)

        print(f"  Found {len(labels)} labels (with duplicates)")
        print(f"  Unique: {len(unique_labels_for_effect)}")

    # Get all unique labels across all effects
    all_unique_labels = sorted(set(all_labels))

    # Count frequencies for reporting
    label_counts = Counter(all_labels)

    print(f"\n=== Summary ===")
    print(f"Total labels extracted: {len(all_labels)}")
    print(f"Unique labels per effect:")
    print(f"  Reverb: {len(labels_by_effect.get('reverb', []))}")
    print(f"  EQ: {len(labels_by_effect.get('eq', []))}")
    print(f"  Compressor: {len(labels_by_effect.get('compressor', []))}")
    print(f"Total unique labels (all effects): {len(all_unique_labels)}")

    print(f"\nTop 20 most common labels (across all effects):")
    for label, count in label_counts.most_common(20):
        print(f"  {label}: {count}")

    return {
        'reverb': labels_by_effect.get('reverb', []),
        'eq': labels_by_effect.get('eq', []),
        'compressor': labels_by_effect.get('compressor', []),
        'all': all_unique_labels
    }


def main():
    """Main function to extract and save Social FX labels."""

    # Paths
    project_root = Path(__file__).parent.parent.parent
    contributions_dir = project_root / 'ref' / 'socialfx_raw' / 'data' / 'raw'
    output_path = project_root / 'baseline-system' / 'data' / 'socialfx_labels.json'

    print("=== Social FX Label Extraction ===\n")
    print(f"Reading from: {contributions_dir}")
    print(f"Output to: {output_path}\n")

    # Extract labels
    labels_dict = extract_all_unique_labels(str(contributions_dir))

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(labels_dict, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Labels saved to: {output_path}")
    print(f"✓ Ready to use for CLAP top-k matching!")


if __name__ == '__main__':
    main()
