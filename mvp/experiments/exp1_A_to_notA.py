"""
Experiment 1: A → not A

Test whether CLAP can understand negation and reversal.

Example: "too bright" → "not bright"
Expected: Reduce high frequencies
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from clap import load_clap_model
from ddsp import create_fx_chain
from llm import setup_llm_client, generate_initial_params
from refine import refine_with_directional_loss
from utils import (
    params_dict_to_tensor,
    tensor_to_params_dict,
    load_audio,
    save_results,
    plot_optimization_history,
    compute_metrics
)


def run_experiment(
    audio_path: str,
    attribute: str = "bright",
    n_iterations: int = 100,
    output_dir: str = None,
    device: str = None
):
    """
    Run Experiment 1: A → not A

    Args:
        audio_path: Path to reference audio
        attribute: Audio attribute (e.g., "bright", "warm")
        n_iterations: Number of optimization iterations
        output_dir: Output directory for results
        device: Torch device
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("\n" + "="*70)
    print("EXPERIMENT 1: A → not A")
    print(f"Attribute: {attribute}")
    print("="*70)

    # Setup
    print("\n📦 Loading models...")
    clap_model = load_clap_model(device=device)
    fx_chain = create_fx_chain(sample_rate=44100, device=device)
    llm_client = setup_llm_client(provider='anthropic')

    # Load audio
    print(f"\n🎵 Loading audio: {audio_path}")
    audio = load_audio(audio_path, device=device)

    # Step 1: LLM generates initial params
    print(f"\n🤖 Step 1: LLM generates params for '{attribute}'")
    initial_params_dict = generate_initial_params(
        llm_client=llm_client,
        prompt=f"make this sound {attribute}"
    )
    print("✓ LLM params generated")

    # Convert to tensor
    initial_params_tensor = params_dict_to_tensor(
        initial_params_dict,
        fx_chain,
        device=device
    )

    # Step 2: Refine with directional loss
    print(f"\n🎯 Step 2: Refine with directional loss")
    text_anchor = f"this sound is too {attribute}"
    text_target = f"this sound is not {attribute}"

    refined_params, history = refine_with_directional_loss(
        audio=audio,
        fx_chain=fx_chain,
        initial_params=initial_params_tensor,
        text_anchor=text_anchor,
        text_target=text_target,
        clap_model=clap_model,
        n_iterations=n_iterations,
        learning_rate=0.01,
        device=device
    )

    # Convert refined params to dict
    refined_params_dict = tensor_to_params_dict(refined_params, fx_chain)

    # Generate audio samples
    print("\n🔊 Generating audio samples...")
    audio_original = audio
    audio_llm = fx_chain(audio, torch.sigmoid(initial_params_tensor))
    audio_refined = fx_chain(audio, torch.sigmoid(refined_params))

    # Compute metrics
    metrics = compute_metrics(history)

    print("\n📊 Results:")
    print(f"  Initial loss: {metrics['initial_loss']:.4f}")
    print(f"  Final loss: {metrics['final_loss']:.4f}")
    print(f"  Improvement: {metrics['improvement_percent']:.1f}%")

    # Save results
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / 'outputs' / 'results' / 'exp1_A_to_notA' / attribute

    output_dir = Path(output_dir)

    save_results(
        output_dir=output_dir,
        audio_files={
            'original': audio_original.squeeze(0),
            'llm_init': audio_llm.detach().squeeze(0),
            'refined': audio_refined.detach().squeeze(0)
        },
        params={
            'initial': initial_params_dict,
            'refined': refined_params_dict
        },
        history=history,
        experiment_info={
            'experiment': 'A_to_notA',
            'attribute': attribute,
            'text_anchor': text_anchor,
            'text_target': text_target,
            'n_iterations': n_iterations,
            'metrics': metrics
        }
    )

    # Plot
    plot_path = output_dir / 'optimization_curve.png'
    plot_optimization_history(history, save_path=plot_path)

    print("\n✅ Experiment 1 complete!")
    print(f"Results saved to: {output_dir}")

    return {
        'params_initial': initial_params_dict,
        'params_refined': refined_params_dict,
        'history': history,
        'metrics': metrics
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Experiment 1: A → not A')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--attribute', type=str, default='bright', help='Audio attribute')
    parser.add_argument('--iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    run_experiment(
        audio_path=args.audio,
        attribute=args.attribute,
        n_iterations=args.iterations,
        output_dir=args.output,
        device=args.device
    )
