#!/usr/bin/env python3
"""
Comprehensive comparison experiment for CLAP vs MERT retrieval.

This script runs systematic comparisons across:
- Original vs cleaned labels
- CLAP vs MERT vs Hybrid modes
- Different effect types (reverb, eq, comp)

Outputs:
- Accuracy tables (CSV + markdown)
- t-SNE visualizations
- HTML comparison reports
"""

import json
import sys
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation.audio_description import generate_audio_description_hybrid


def load_labels(labels_file: Path) -> List[str]:
    """Load labels from JSON file."""
    with open(labels_file, 'r') as f:
        data = json.load(f)

    if 'labels' in data:
        # Cleaned labels format
        return data['labels']
    else:
        # Original labels format (assuming 'all' or effect-specific key)
        return data.get('all', []) or data.get('reverb', []) or data.get('eq', []) or data.get('compressor', [])


def run_single_experiment(
    audio_files: List[Tuple[str, List[str]]],
    labels: List[str],
    mode: str,
    label_type: str,
    effect_type: str,
    device: str = 'cuda:0'
) -> Dict:
    """
    Run a single experiment configuration.

    Args:
        audio_files: List of (audio_path, ground_truth_labels) tuples
        labels: Candidate labels
        mode: Retrieval mode ('clap', 'mert', 'hybrid')
        label_type: 'original' or 'cleaned'
        effect_type: 'reverb', 'eq', or 'comp'
        device: Device to use

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {effect_type.upper()} | {label_type.upper()} labels | {mode.upper()} mode")
    print(f"{'='*70}")
    print(f"Candidate labels: {len(labels)}")
    print(f"Test samples: {len(audio_files)}")

    results = {
        'config': {
            'effect_type': effect_type,
            'label_type': label_type,
            'mode': mode,
            'num_labels': len(labels),
            'num_samples': len(audio_files),
            'timestamp': datetime.now().isoformat()
        },
        'metrics': {
            'hit@1': 0,
            'hit@5': 0,
            'hit@10': 0,
            'mrr': 0.0,
            'mean_score_top1': 0.0
        },
        'per_sample': []
    }

    reciprocal_ranks = []
    top1_scores = []

    for idx, (audio_path, ground_truth) in enumerate(audio_files):
        print(f"  Processing [{idx+1}/{len(audio_files)}]: {Path(audio_path).name}")

        try:
            # Get top-10 predictions
            predictions = generate_audio_description_hybrid(
                audio_path=audio_path,
                candidate_descriptions=labels,
                k=10,
                mode=mode,
                device=device
            )

            predicted_labels = [label for label, _ in predictions]
            scores = [score for _, score in predictions]

            # Calculate hit rates
            hit_at_1 = int(any(gt in predicted_labels[:1] for gt in ground_truth))
            hit_at_5 = int(any(gt in predicted_labels[:5] for gt in ground_truth))
            hit_at_10 = int(any(gt in predicted_labels[:10] for gt in ground_truth))

            results['metrics']['hit@1'] += hit_at_1
            results['metrics']['hit@5'] += hit_at_5
            results['metrics']['hit@10'] += hit_at_10

            # Reciprocal rank
            rank = None
            for i, pred_label in enumerate(predicted_labels):
                if pred_label in ground_truth:
                    rank = i + 1
                    break

            rr = 1.0 / rank if rank is not None else 0.0
            reciprocal_ranks.append(rr)

            # Top-1 score
            top1_scores.append(scores[0] if scores else 0.0)

            # Store per-sample results
            results['per_sample'].append({
                'audio': Path(audio_path).name,
                'ground_truth': ground_truth,
                'predictions': predicted_labels,
                'scores': scores,
                'hit@1': hit_at_1,
                'hit@5': hit_at_5,
                'hit@10': hit_at_10,
                'reciprocal_rank': rr
            })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    # Calculate final metrics
    n = len(audio_files)
    results['metrics']['hit@1'] /= n
    results['metrics']['hit@5'] /= n
    results['metrics']['hit@10'] /= n
    results['metrics']['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    results['metrics']['mean_score_top1'] = np.mean(top1_scores) if top1_scores else 0.0

    # Print summary
    print(f"\nResults:")
    print(f"  Hit@1:  {results['metrics']['hit@1']:.3f}")
    print(f"  Hit@5:  {results['metrics']['hit@5']:.3f}")
    print(f"  Hit@10: {results['metrics']['hit@10']:.3f}")
    print(f"  MRR:    {results['metrics']['mrr']:.3f}")
    print(f"  Mean Top-1 Score: {results['metrics']['mean_score_top1']:.3f}")

    return results


def create_comparison_table(all_results: List[Dict], output_dir: Path):
    """
    Create comparison tables in CSV and Markdown formats.

    Args:
        all_results: List of result dictionaries from all experiments
        output_dir: Directory to save tables
    """
    # Prepare table data
    table_rows = []

    for result in all_results:
        config = result['config']
        metrics = result['metrics']

        row = {
            'Effect Type': config['effect_type'],
            'Label Type': config['label_type'],
            'Mode': config['mode'],
            'Num Labels': config['num_labels'],
            'Hit@1': f"{metrics['hit@1']:.3f}",
            'Hit@5': f"{metrics['hit@5']:.3f}",
            'Hit@10': f"{metrics['hit@10']:.3f}",
            'MRR': f"{metrics['mrr']:.3f}",
            'Top-1 Score': f"{metrics['mean_score_top1']:.3f}"
        }
        table_rows.append(row)

    # Save as CSV
    csv_file = output_dir / 'comparison_table.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)

    print(f"\n✓ CSV table saved to: {csv_file}")

    # Save as Markdown
    md_file = output_dir / 'comparison_table.md'
    with open(md_file, 'w') as f:
        # Write header
        headers = list(table_rows[0].keys())
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '|'.join(['---' for _ in headers]) + '|\n')

        # Write rows
        for row in table_rows:
            f.write('| ' + ' | '.join(str(row[h]) for h in headers) + ' |\n')

        # Add summary section
        f.write('\n## Summary\n\n')

        # Group by effect type and mode to find best configs
        by_effect = {}
        for row in table_rows:
            effect = row['Effect Type']
            if effect not in by_effect:
                by_effect[effect] = []
            by_effect[effect].append(row)

        for effect, rows in by_effect.items():
            f.write(f'### {effect.upper()}\n\n')

            # Find best configuration by Hit@5
            best_row = max(rows, key=lambda r: float(r['Hit@5']))
            f.write(f"**Best configuration:** {best_row['Label Type']} labels + {best_row['Mode']} mode\n\n")
            f.write(f"- Hit@5: {best_row['Hit@5']}\n")
            f.write(f"- MRR: {best_row['MRR']}\n\n")

    print(f"✓ Markdown table saved to: {md_file}")


def create_summary_report(all_results: List[Dict], output_dir: Path):
    """
    Create a comprehensive HTML summary report.

    Args:
        all_results: List of result dictionaries
        output_dir: Output directory
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <title>CLAP vs MERT Comparison Report</title>",
        "    <style>",
        "        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 30px; background: #f0f2f5; }",
        "        h1 { color: #1a1a1a; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }",
        "        h2 { color: #333; margin-top: 30px; }",
        "        .metadata { background: #e3f2fd; padding: 15px; border-radius: 5px; margin: 20px 0; }",
        "        .comparison-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }",
        "        .result-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "        .result-card h3 { margin-top: 0; color: #1976d2; }",
        "        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; }",
        "        .metric:last-child { border-bottom: none; }",
        "        .metric-name { font-weight: 500; color: #555; }",
        "        .metric-value { font-weight: bold; color: #1a1a1a; }",
        "        .highlight { background: #fffde7; padding: 3px 8px; border-radius: 3px; }",
        "        .best { background: #c8e6c9; padding: 3px 8px; border-radius: 3px; font-weight: bold; }",
        "        table { width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }",
        "        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }",
        "        th { background: #1976d2; color: white; font-weight: 600; }",
        "        tr:hover { background: #f5f5f5; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>CLAP vs MERT Comparison Report</h1>",
        f"    <div class='metadata'>",
        f"        <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>",
        f"        <strong>Total Experiments:</strong> {len(all_results)}",
        f"    </div>",
    ]

    # Group results by effect type
    by_effect = {}
    for result in all_results:
        effect = result['config']['effect_type']
        if effect not in by_effect:
            by_effect[effect] = []
        by_effect[effect].append(result)

    # Create sections for each effect type
    for effect, results in by_effect.items():
        html_parts.append(f"    <h2>{effect.upper()} Effect</h2>")

        # Find best configuration
        best_result = max(results, key=lambda r: r['metrics']['hit@5'])

        html_parts.append(f"    <p><strong>Best Configuration:</strong> ")
        html_parts.append(f"<span class='best'>{best_result['config']['label_type'].title()} labels + ")
        html_parts.append(f"{best_result['config']['mode'].upper()} mode</span></p>")

        # Create comparison grid
        html_parts.append(f"    <div class='comparison-grid'>")

        for result in results:
            config = result['config']
            metrics = result['metrics']

            is_best = result == best_result

            html_parts.append(f"        <div class='result-card'>")
            html_parts.append(f"            <h3>{config['label_type'].title()} + {config['mode'].upper()}</h3>")
            html_parts.append(f"            <p style='color: #666; font-size: 0.9em;'>")
            html_parts.append(f"{config['num_labels']} labels | {config['num_samples']} samples</p>")

            # Metrics
            for metric_name, metric_key in [('Hit@1', 'hit@1'), ('Hit@5', 'hit@5'), ('Hit@10', 'hit@10'), ('MRR', 'mrr')]:
                value = metrics[metric_key]
                value_class = 'best' if is_best and metric_key == 'hit@5' else ''

                html_parts.append(f"            <div class='metric'>")
                html_parts.append(f"                <span class='metric-name'>{metric_name}:</span>")
                html_parts.append(f"                <span class='metric-value {value_class}'>{value:.3f}</span>")
                html_parts.append(f"            </div>")

            html_parts.append(f"        </div>")

        html_parts.append(f"    </div>")

    # Add detailed table
    html_parts.append(f"    <h2>Detailed Results Table</h2>")
    html_parts.append(f"    <table>")
    html_parts.append(f"        <thead>")
    html_parts.append(f"            <tr>")
    html_parts.append(f"                <th>Effect</th><th>Labels</th><th>Mode</th>")
    html_parts.append(f"                <th>Hit@1</th><th>Hit@5</th><th>Hit@10</th><th>MRR</th>")
    html_parts.append(f"            </tr>")
    html_parts.append(f"        </thead>")
    html_parts.append(f"        <tbody>")

    for result in all_results:
        config = result['config']
        metrics = result['metrics']

        html_parts.append(f"            <tr>")
        html_parts.append(f"                <td>{config['effect_type']}</td>")
        html_parts.append(f"                <td>{config['label_type']}</td>")
        html_parts.append(f"                <td>{config['mode']}</td>")
        html_parts.append(f"                <td>{metrics['hit@1']:.3f}</td>")
        html_parts.append(f"                <td>{metrics['hit@5']:.3f}</td>")
        html_parts.append(f"                <td>{metrics['hit@10']:.3f}</td>")
        html_parts.append(f"                <td>{metrics['mrr']:.3f}</td>")
        html_parts.append(f"            </tr>")

    html_parts.append(f"        </tbody>")
    html_parts.append(f"    </table>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    # Write to file
    html_file = output_dir / 'comparison_report.html'
    with open(html_file, 'w') as f:
        f.write('\n'.join(html_parts))

    print(f"✓ HTML report saved to: {html_file}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive comparison experiments')
    parser.add_argument('--effect-types', nargs='+', default=['reverb'],
                        choices=['reverb', 'eq', 'comp'],
                        help='Effect types to test')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--ground-truth', type=str, required=True,
                        help='JSON file with ground truth labels')
    parser.add_argument('--original-labels-dir', type=str,
                        default='baseline-system/data',
                        help='Directory containing original labels')
    parser.add_argument('--cleaned-labels-dir', type=str,
                        default='baseline-system/data/cleaned_labels',
                        help='Directory containing cleaned labels')
    parser.add_argument('--output-dir', type=str,
                        default='baseline-system/results/comparison',
                        help='Output directory for results')
    parser.add_argument('--modes', nargs='+', default=['clap', 'mert', 'hybrid'],
                        choices=['clap', 'mert', 'hybrid'],
                        help='Retrieval modes to test')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--skip-original', action='store_true',
                        help='Skip experiments with original labels')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    audio_dir = Path(args.audio_dir)
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    print(f"Loading ground truth from {args.ground_truth}...")
    with open(args.ground_truth, 'r') as f:
        ground_truth_dict = json.load(f)

    # Build audio files list
    audio_files = []
    for filename, gt_labels in ground_truth_dict.items():
        audio_path = audio_dir / filename
        if audio_path.exists():
            audio_files.append((str(audio_path), gt_labels))
        else:
            print(f"  Warning: {audio_path} not found")

    print(f"Found {len(audio_files)} audio files with ground truth\n")

    if len(audio_files) == 0:
        print("Error: No valid audio files found. Exiting.")
        return

    # Run all experiments
    all_results = []

    for effect_type in args.effect_types:
        print(f"\n{'#'*70}")
        print(f"# EFFECT TYPE: {effect_type.upper()}")
        print(f"{'#'*70}")

        # Load labels
        cleaned_labels_path = base_dir / args.cleaned_labels_dir / f"{effect_type}_labels.json"

        if not cleaned_labels_path.exists():
            print(f"Error: Cleaned labels not found: {cleaned_labels_path}")
            continue

        cleaned_labels = load_labels(cleaned_labels_path)

        # Optionally load original labels
        if not args.skip_original:
            original_labels_path = base_dir / args.original_labels_dir / 'socialfx_labels.json'
            if original_labels_path.exists():
                with open(original_labels_path, 'r') as f:
                    original_data = json.load(f)
                original_labels = original_data.get(effect_type, [])
            else:
                print(f"Warning: Original labels not found: {original_labels_path}")
                original_labels = None
        else:
            original_labels = None

        # Run experiments for each configuration
        for mode in args.modes:
            # Cleaned labels
            result = run_single_experiment(
                audio_files=audio_files,
                labels=cleaned_labels,
                mode=mode,
                label_type='cleaned',
                effect_type=effect_type,
                device=args.device
            )
            all_results.append(result)

            # Save individual result
            result_file = output_dir / f'{effect_type}_cleaned_{mode}.json'
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)

            # Original labels (if available)
            if original_labels and len(original_labels) > 0:
                result = run_single_experiment(
                    audio_files=audio_files,
                    labels=original_labels,
                    mode=mode,
                    label_type='original',
                    effect_type=effect_type,
                    device=args.device
                )
                all_results.append(result)

                result_file = output_dir / f'{effect_type}_original_{mode}.json'
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2)

    # Create comparison tables and reports
    print(f"\n{'='*70}")
    print("GENERATING COMPARISON REPORTS")
    print(f"{'='*70}")

    create_comparison_table(all_results, output_dir)
    create_summary_report(all_results, output_dir)

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"\nKey files:")
    print(f"  - comparison_table.csv: Metrics table (CSV format)")
    print(f"  - comparison_table.md: Metrics table (Markdown format)")
    print(f"  - comparison_report.html: Interactive HTML report")
    print(f"  - Individual JSON files for each configuration")


if __name__ == '__main__':
    main()
