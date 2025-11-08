#!/usr/bin/env python3
"""
Evaluation framework for audio description retrieval.

Evaluates CLAP, MERT, and hybrid approaches using:
1. Ground truth comparison (if labeled test set available)
2. Manual inspection interface (HTML report)
3. t-SNE visualization of embedding spaces
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation.audio_description import (
    generate_audio_description_hybrid,
    get_mert_embeddings,
    _get_clap_model,
)


def load_cleaned_labels(effect_type: str, labels_dir: Path) -> List[str]:
    """Load cleaned labels for a specific effect type."""
    labels_file = labels_dir / f"{effect_type}_labels.json"

    with open(labels_file, 'r') as f:
        data = json.load(f)

    return data['labels']


def evaluate_top_k_accuracy(
    audio_files: List[Tuple[str, List[str]]],
    candidate_labels: List[str],
    mode: str,
    k_values: List[int] = [1, 5, 10],
    device: str = 'cuda:0'
) -> Dict:
    """
    Evaluate top-k accuracy on a labeled test set.

    Args:
        audio_files: List of (audio_path, ground_truth_labels) tuples
        candidate_labels: Full list of candidate labels
        mode: Retrieval mode ('clap', 'mert', 'hybrid')
        k_values: Values of k to evaluate
        device: Device to use

    Returns:
        Dictionary with accuracy metrics
    """
    results = {
        'mode': mode,
        'total_samples': len(audio_files),
        'k_values': k_values,
        'hit_rates': {k: 0 for k in k_values},
        'mrr': 0.0,  # Mean Reciprocal Rank
        'per_sample': []
    }

    reciprocal_ranks = []

    print(f"\n{'='*60}")
    print(f"Evaluating {mode.upper()} mode on {len(audio_files)} samples")
    print(f"{'='*60}")

    for audio_path, ground_truth in tqdm(audio_files, desc=f"Evaluating {mode}"):
        try:
            # Get top-k predictions
            max_k = max(k_values)
            predictions = generate_audio_description_hybrid(
                audio_path=audio_path,
                candidate_descriptions=candidate_labels,
                k=max_k,
                mode=mode,
                device=device
            )

            predicted_labels = [label for label, _ in predictions]

            # Calculate hit rates for each k
            sample_hits = {}
            for k in k_values:
                top_k_preds = predicted_labels[:k]
                # Check if any ground truth label is in top-k
                hit = any(gt in top_k_preds for gt in ground_truth)
                sample_hits[k] = hit
                if hit:
                    results['hit_rates'][k] += 1

            # Calculate reciprocal rank
            rank = None
            for i, pred_label in enumerate(predicted_labels):
                if pred_label in ground_truth:
                    rank = i + 1
                    break

            if rank is not None:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)

            results['per_sample'].append({
                'audio_path': audio_path,
                'ground_truth': ground_truth,
                'predictions': predicted_labels,
                'hits': sample_hits,
                'reciprocal_rank': reciprocal_ranks[-1]
            })

        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue

    # Calculate final metrics
    for k in k_values:
        results['hit_rates'][k] = results['hit_rates'][k] / len(audio_files)

    results['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Print summary
    print(f"\nResults for {mode.upper()}:")
    for k in k_values:
        print(f"  Hit Rate @ {k:2d}: {results['hit_rates'][k]:.3f}")
    print(f"  Mean Reciprocal Rank: {results['mrr']:.3f}")

    return results


def generate_html_report(
    evaluation_results: Dict,
    audio_files: List[Tuple[str, List[str]]],
    output_file: Path
):
    """
    Generate HTML report for manual inspection.

    Args:
        evaluation_results: Dictionary with results from different modes
        audio_files: List of audio files with ground truth
        output_file: Path to save HTML report
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "    <meta charset='utf-8'>",
        "    <title>Audio Retrieval Evaluation Report</title>",
        "    <style>",
        "        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
        "        h1 { color: #333; }",
        "        .summary { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; }",
        "        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }",
        "        .metric-card { background: #e3f2fd; padding: 15px; border-radius: 5px; }",
        "        .metric-card h3 { margin: 0 0 10px 0; color: #1976d2; }",
        "        .sample { background: white; padding: 20px; margin-bottom: 15px; border-radius: 8px; }",
        "        .sample h3 { margin-top: 0; }",
        "        .audio-player { margin: 10px 0; }",
        "        .predictions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }",
        "        .pred-col { background: #f9f9f9; padding: 15px; border-radius: 5px; }",
        "        .pred-col h4 { margin-top: 0; text-align: center; }",
        "        .label { padding: 5px 10px; margin: 3px; display: inline-block; border-radius: 3px; font-size: 14px; }",
        "        .label.gt { background: #c8e6c9; }",
        "        .label.hit { background: #fff9c4; }",
        "        .label.miss { background: #ffccbc; }",
        "        .ground-truth { background: #e8f5e9; padding: 10px; border-left: 4px solid #4caf50; margin: 10px 0; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <h1>Audio Description Retrieval - Evaluation Report</h1>",
    ]

    # Summary section
    html_parts.append("    <div class='summary'>")
    html_parts.append("        <h2>Summary</h2>")
    html_parts.append("        <div class='metrics'>")

    for mode, results in evaluation_results.items():
        html_parts.append(f"            <div class='metric-card'>")
        html_parts.append(f"                <h3>{mode.upper()}</h3>")
        for k, hit_rate in results['hit_rates'].items():
            html_parts.append(f"                <p>Hit@{k}: <strong>{hit_rate:.3f}</strong></p>")
        html_parts.append(f"                <p>MRR: <strong>{results['mrr']:.3f}</strong></p>")
        html_parts.append(f"            </div>")

    html_parts.append("        </div>")
    html_parts.append("    </div>")

    # Per-sample results
    html_parts.append("    <h2>Per-Sample Results</h2>")

    # Get CLAP results for display (assuming it's in evaluation_results)
    if 'clap' in evaluation_results:
        clap_samples = evaluation_results['clap']['per_sample']
    else:
        clap_samples = []

    if 'mert' in evaluation_results:
        mert_samples = evaluation_results['mert']['per_sample']
    else:
        mert_samples = []

    if 'hybrid' in evaluation_results:
        hybrid_samples = evaluation_results['hybrid']['per_sample']
    else:
        hybrid_samples = []

    num_samples = len(audio_files)

    for i in range(num_samples):
        audio_path, ground_truth = audio_files[i]
        audio_name = Path(audio_path).name

        html_parts.append(f"    <div class='sample'>")
        html_parts.append(f"        <h3>Sample {i+1}: {audio_name}</h3>")

        # Audio player
        html_parts.append(f"        <audio class='audio-player' controls>")
        html_parts.append(f"            <source src='{audio_path}' type='audio/wav'>")
        html_parts.append(f"        </audio>")

        # Ground truth
        html_parts.append(f"        <div class='ground-truth'>")
        html_parts.append(f"            <strong>Ground Truth:</strong> ")
        for gt in ground_truth:
            html_parts.append(f"<span class='label gt'>{gt}</span>")
        html_parts.append(f"        </div>")

        # Predictions from each mode
        html_parts.append(f"        <div class='predictions'>")

        for mode_name, samples in [('CLAP', clap_samples), ('MERT', mert_samples), ('Hybrid', hybrid_samples)]:
            if i < len(samples):
                sample = samples[i]
                predictions = sample['predictions'][:10]  # Show top-10

                html_parts.append(f"            <div class='pred-col'>")
                html_parts.append(f"                <h4>{mode_name}</h4>")

                for j, pred in enumerate(predictions):
                    # Check if prediction matches any ground truth
                    if pred in ground_truth:
                        css_class = 'label hit'
                    else:
                        css_class = 'label miss'

                    html_parts.append(f"                <span class='{css_class}'>{j+1}. {pred}</span><br>")

                html_parts.append(f"            </div>")

        html_parts.append(f"        </div>")
        html_parts.append(f"    </div>")

    html_parts.append("</body>")
    html_parts.append("</html>")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(html_parts))

    print(f"\n✓ HTML report saved to: {output_file}")


def visualize_embeddings_tsne(
    audio_files: List[Tuple[str, List[str]]],
    candidate_labels: List[str],
    effect_type: str,
    output_dir: Path,
    device: str = 'cuda:0',
    n_components: int = 2,
    perplexity: int = 30
):
    """
    Create t-SNE visualization of audio and label embeddings.

    Args:
        audio_files: List of audio files with ground truth
        candidate_labels: Full list of candidate labels
        effect_type: Effect type for labeling
        output_dir: Directory to save plots
        device: Device to use
        n_components: Number of t-SNE components (2 or 3)
        perplexity: t-SNE perplexity parameter
    """
    print(f"\n{'='*60}")
    print(f"Generating t-SNE visualizations for {effect_type}")
    print(f"{'='*60}")

    # Get CLAP model
    clap_model, active_device = _get_clap_model(device)

    # Extract audio embeddings (CLAP and MERT)
    clap_audio_embeddings = []
    mert_audio_embeddings = []
    audio_labels = []

    print(f"\nExtracting audio embeddings...")
    for audio_path, ground_truth in tqdm(audio_files):
        try:
            # CLAP audio embedding
            clap_emb = clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)
            clap_audio_embeddings.append(clap_emb.flatten())

            # MERT audio embedding
            mert_emb = get_mert_embeddings(audio_path, device)
            mert_audio_embeddings.append(mert_emb)

            audio_labels.append(f"Audio: {Path(audio_path).stem}")

        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            continue

    # Extract text embeddings from CLAP
    print(f"\nExtracting text embeddings for {len(candidate_labels)} labels...")
    text_embeddings = []
    batch_size = 128

    for i in tqdm(range(0, len(candidate_labels), batch_size)):
        batch = candidate_labels[i:i+batch_size]
        batch_emb = clap_model.get_text_embedding(batch, use_tensor=False)
        text_embeddings.append(batch_emb)

    text_embeddings = np.vstack(text_embeddings)

    # Create t-SNE visualizations
    print(f"\nRunning t-SNE (this may take a while)...")

    # 1. CLAP embeddings (audio + text in same space)
    print(f"  Processing CLAP embeddings...")
    clap_combined = np.vstack([np.array(clap_audio_embeddings), text_embeddings])

    tsne_clap = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    clap_2d = tsne_clap.fit_transform(clap_combined)

    # Split back into audio and text
    n_audio = len(clap_audio_embeddings)
    clap_audio_2d = clap_2d[:n_audio]
    clap_text_2d = clap_2d[n_audio:]

    # Create interactive plot
    fig_clap = go.Figure()

    # Add text labels
    fig_clap.add_trace(go.Scatter(
        x=clap_text_2d[:, 0],
        y=clap_text_2d[:, 1],
        mode='markers',
        marker=dict(size=5, color='lightblue', opacity=0.5),
        text=candidate_labels,
        name='Labels',
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))

    # Add audio points
    fig_clap.add_trace(go.Scatter(
        x=clap_audio_2d[:, 0],
        y=clap_audio_2d[:, 1],
        mode='markers',
        marker=dict(size=12, color='red', symbol='diamond'),
        text=audio_labels,
        name='Audio',
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))

    fig_clap.update_layout(
        title=f't-SNE Visualization: CLAP Embeddings ({effect_type})',
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2',
        width=1200,
        height=800,
        hovermode='closest'
    )

    clap_output = output_dir / f'tsne_clap_{effect_type}.html'
    fig_clap.write_html(str(clap_output))
    print(f"\n✓ CLAP t-SNE saved to: {clap_output}")

    # 2. MERT embeddings (separate space - need to project to same dim as text)
    print(f"  Processing MERT embeddings...")

    # Truncate/pad MERT to match text embedding dimension
    mert_array = np.array(mert_audio_embeddings)
    text_dim = text_embeddings.shape[1]
    mert_dim = mert_array.shape[1]

    if mert_dim > text_dim:
        mert_array = mert_array[:, :text_dim]
    elif mert_dim < text_dim:
        padding = np.zeros((mert_array.shape[0], text_dim - mert_dim))
        mert_array = np.concatenate([mert_array, padding], axis=1)

    mert_combined = np.vstack([mert_array, text_embeddings])

    tsne_mert = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    mert_2d = tsne_mert.fit_transform(mert_combined)

    mert_audio_2d = mert_2d[:n_audio]
    mert_text_2d = mert_2d[n_audio:]

    fig_mert = go.Figure()

    # Add text labels
    fig_mert.add_trace(go.Scatter(
        x=mert_text_2d[:, 0],
        y=mert_text_2d[:, 1],
        mode='markers',
        marker=dict(size=5, color='lightgreen', opacity=0.5),
        text=candidate_labels,
        name='Labels',
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))

    # Add audio points
    fig_mert.add_trace(go.Scatter(
        x=mert_audio_2d[:, 0],
        y=mert_audio_2d[:, 1],
        mode='markers',
        marker=dict(size=12, color='purple', symbol='diamond'),
        text=audio_labels,
        name='Audio (MERT)',
        hovertemplate='<b>%{text}</b><extra></extra>'
    ))

    fig_mert.update_layout(
        title=f't-SNE Visualization: MERT Audio + CLAP Text ({effect_type})',
        xaxis_title='t-SNE Component 1',
        yaxis_title='t-SNE Component 2',
        width=1200,
        height=800,
        hovermode='closest'
    )

    mert_output = output_dir / f'tsne_mert_{effect_type}.html'
    fig_mert.write_html(str(mert_output))
    print(f"✓ MERT t-SNE saved to: {mert_output}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate audio retrieval systems')
    parser.add_argument('--effect-type', type=str, required=True, choices=['reverb', 'eq', 'comp'],
                        help='Effect type to evaluate')
    parser.add_argument('--audio-dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--ground-truth', type=str, required=True,
                        help='JSON file with ground truth labels (format: {"audio_file.wav": ["label1", "label2"]})')
    parser.add_argument('--labels-dir', type=str,
                        default='baseline-system/data/cleaned_labels',
                        help='Directory containing cleaned label files')
    parser.add_argument('--output-dir', type=str,
                        default='baseline-system/results/evaluation',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--modes', nargs='+', default=['clap', 'mert', 'hybrid'],
                        choices=['clap', 'mert', 'hybrid'],
                        help='Retrieval modes to evaluate')
    parser.add_argument('--skip-tsne', action='store_true',
                        help='Skip t-SNE visualization (faster)')

    args = parser.parse_args()

    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    labels_dir = base_dir / args.labels_dir
    audio_dir = Path(args.audio_dir)
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cleaned labels
    print(f"Loading cleaned {args.effect_type} labels...")
    candidate_labels = load_cleaned_labels(args.effect_type, labels_dir)
    print(f"  Loaded {len(candidate_labels)} labels")

    # Load ground truth
    print(f"\nLoading ground truth from {args.ground_truth}...")
    with open(args.ground_truth, 'r') as f:
        ground_truth_dict = json.load(f)

    # Build audio files list
    audio_files = []
    for filename, gt_labels in ground_truth_dict.items():
        audio_path = audio_dir / filename
        if audio_path.exists():
            audio_files.append((str(audio_path), gt_labels))
        else:
            print(f"  Warning: {audio_path} not found, skipping")

    print(f"  Found {len(audio_files)} audio files with ground truth")

    # Run evaluations
    evaluation_results = {}

    for mode in args.modes:
        results = evaluate_top_k_accuracy(
            audio_files=audio_files,
            candidate_labels=candidate_labels,
            mode=mode,
            k_values=[1, 5, 10],
            device=args.device
        )
        evaluation_results[mode] = results

        # Save individual results
        mode_output = output_dir / f'results_{args.effect_type}_{mode}.json'
        with open(mode_output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Saved results to: {mode_output}")

    # Generate HTML report
    report_output = output_dir / f'report_{args.effect_type}.html'
    generate_html_report(evaluation_results, audio_files, report_output)

    # Generate t-SNE visualizations
    if not args.skip_tsne:
        visualize_embeddings_tsne(
            audio_files=audio_files,
            candidate_labels=candidate_labels,
            effect_type=args.effect_type,
            output_dir=output_dir,
            device=args.device
        )

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
