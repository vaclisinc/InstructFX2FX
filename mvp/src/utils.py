"""
Utility functions for parameter conversion and helpers.
"""

import torch
import json
from pathlib import Path


def params_dict_to_tensor(params_dict: dict, fx_chain, device='cpu') -> torch.Tensor:
    """
    Convert LLM parameter dictionary to tensor format for DDSP.

    Args:
        params_dict: Dictionary with 'eq', 'compressor', 'reverb' keys
        fx_chain: FX chain to determine expected parameter count
        device: Device to create tensor on

    Returns:
        Parameter tensor [1, num_params]
    """
    # Extract parameters in order
    eq_params = params_dict.get('eq', [])
    comp_params = params_dict.get('compressor', [])
    reverb_params = params_dict.get('reverb', [])

    # Concatenate all params
    all_params = eq_params + comp_params + reverb_params

    # Convert to tensor
    params_tensor = torch.tensor(all_params, dtype=torch.float32, device=device)

    # Ensure batch dimension
    if params_tensor.ndim == 1:
        params_tensor = params_tensor.unsqueeze(0)

    # Validate size
    expected_size = fx_chain.num_params
    if params_tensor.shape[1] != expected_size:
        raise ValueError(
            f"Parameter size mismatch: expected {expected_size}, got {params_tensor.shape[1]}"
        )

    return params_tensor


def tensor_to_params_dict(params_tensor: torch.Tensor, fx_chain) -> dict:
    """
    Convert parameter tensor back to dictionary format.

    Args:
        params_tensor: Parameter tensor [B, num_params] or [num_params]
        fx_chain: FX chain to determine parameter structure

    Returns:
        Dictionary with 'eq', 'compressor', 'reverb' keys
    """
    # Remove batch dimension if present
    if params_tensor.ndim == 2:
        params_tensor = params_tensor.squeeze(0)

    # Convert to numpy
    params = params_tensor.cpu().detach().numpy()

    # Split based on FX chain structure
    eq_size = fx_chain.eq.num_params
    comp_size = fx_chain.compressor.num_params
    reverb_size = fx_chain.reverb.num_params

    eq_params = params[:eq_size].tolist()
    comp_params = params[eq_size:eq_size + comp_size].tolist()
    reverb_params = params[eq_size + comp_size:].tolist()

    return {
        'eq': eq_params,
        'compressor': comp_params,
        'reverb': reverb_params
    }


def save_results(
    output_dir: Path,
    audio_files: dict,
    params: dict,
    history: list,
    experiment_info: dict
):
    """
    Save experiment results to disk.

    Args:
        output_dir: Directory to save results
        audio_files: Dict mapping names to audio tensors
        params: Dict with initial/refined parameters
        history: Optimization history
        experiment_info: Experiment metadata
    """
    import torchaudio

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save audio files
    for name, audio in audio_files.items():
        audio_path = output_dir / f"{name}.wav"
        torchaudio.save(str(audio_path), audio.cpu(), 44100)

    # Save parameters
    params_path = output_dir / "parameters.json"
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=2)

    # Save optimization history
    history_path = output_dir / "history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save experiment info
    info_path = output_dir / "experiment_info.json"
    with open(info_path, 'w') as f:
        json.dump(experiment_info, f, indent=2)

    print(f"\n✓ Results saved to: {output_dir}")
    print(f"  - {len(audio_files)} audio files")
    print(f"  - parameters.json")
    print(f"  - history.json ({len(history)} iterations)")
    print(f"  - experiment_info.json")


def load_audio(audio_path: str, target_sr: int = 44100, device='cpu') -> torch.Tensor:
    """
    Load audio file and convert to tensor.

    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate
        device: Device to load audio on

    Returns:
        Audio tensor [1, C, T]
    """
    import torchaudio

    audio, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)

    # Ensure batch dimension
    if audio.ndim == 2:
        audio = audio.unsqueeze(0)

    return audio.to(device)


def plot_optimization_history(
    history: list,
    history_baseline: list = None,
    save_path: Path = None
):
    """
    Plot optimization loss curves.

    Args:
        history: List of dicts with 'iteration' and 'loss'
        history_baseline: Optional baseline history for comparison
        save_path: Optional path to save plot
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))

    # Plot main history
    iterations = [h['iteration'] for h in history]
    losses = [h['loss'] for h in history]
    plt.plot(iterations, losses, linewidth=2, label='LLM Init')

    # Plot baseline if provided
    if history_baseline:
        iterations_baseline = [h['iteration'] for h in history_baseline]
        losses_baseline = [h['loss'] for h in history_baseline]
        plt.plot(iterations_baseline, losses_baseline, linewidth=2, alpha=0.7, label='Random Init')

    plt.xlabel('Iteration')
    plt.ylabel('Directional Loss')
    plt.title('Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")

    plt.show()


def compute_metrics(history: list) -> dict:
    """
    Compute optimization metrics.

    Args:
        history: Optimization history

    Returns:
        Dictionary with metrics
    """
    initial_loss = history[0]['loss']
    final_loss = history[-1]['loss']
    improvement = (initial_loss - final_loss) / initial_loss * 100

    # Find iteration with best (lowest) loss
    best_iter = min(range(len(history)), key=lambda i: history[i]['loss'])
    best_loss = history[best_iter]['loss']

    return {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'best_loss': best_loss,
        'best_iteration': best_iter,
        'improvement_percent': improvement,
        'total_iterations': len(history)
    }
