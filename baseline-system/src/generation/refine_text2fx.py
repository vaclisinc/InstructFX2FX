import torch
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any
import sys
import os

# Add text2fx to path if not installed
# Assuming text2fx-repo is at ../../ref/text2fx-repo relative to this file
# This file is at baseline-system/src/generation/refine_text2fx.py
# So we need to go up 3 levels to baseline-system/src/generation -> src -> baseline-system -> text2preset
# Then down to ref/text2fx-repo
REPO_ROOT = Path(__file__).parent.parent.parent.parent
TEXT2FX_PATH = REPO_ROOT / 'ref' / 'text2fx-repo'

if str(TEXT2FX_PATH) not in sys.path:
    sys.path.append(str(TEXT2FX_PATH))

try:
    import text2fx.core as tc
    from text2fx.core import Channel, ParametricEQ_40band, Distortion
    import dasp_pytorch
    from audiotools import AudioSignal
    from text2fx.constants import SAMPLE_RATE, DEVICE
except ImportError:
    print("Warning: text2fx not found. Please ensure it is installed or the path is correct.")
    # Mocking for when dependencies aren't installed yet (e.g. in some test envs)
    tc = None

def clip_directional_loss(
        a1: torch.Tensor, 
        a2: torch.Tensor, 
        b1: torch.Tensor, 
        b2: torch.Tensor
    ):
    """
    Compute directional loss between two pairs of embeddings.
    Loss = 1 - cosine_similarity(a1-a2, b1-b2)
    """
    a_dir = a1 - a2
    a_dir /= a_dir.clone().norm(dim=-1, keepdim=True) + 1e-8

    b_dir = b1 - b2
    b_dir /= b_dir.clone().norm(dim=-1, keepdim=True) + 1e-8

    loss = 1 - torch.cosine_similarity(a_dir, b_dir, dim=-1)
    return loss

def get_default_channel(sample_rate=44100):
    """
    Create a default FX chain similar to the baseline (EQ, Compressor, Reverb).
    Note: Text2FX uses dasp_pytorch modules.
    """
    return Channel(
        dasp_pytorch.ParametricEQ(sample_rate=sample_rate),
        dasp_pytorch.Compressor(sample_rate=sample_rate),
        dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate),
        # Distortion(sample_rate=sample_rate) # Optional, maybe too aggressive for general use
    )

def refine_with_text2fx(
    user_prompt: str, 
    audio_path: str, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Refine audio using Text2FX gradient-based optimization.
    
    Args:
        user_prompt: The target text description.
        audio_path: Path to the input audio file.
        config: Configuration dictionary.
        
    Returns:
        Dictionary containing:
            - 'best_params': The optimized parameters (as a dict).
            - 'output_audio_path': Path to the optimized audio file.
            - 'history': Optimization history (losses).
    """
    if tc is None:
        raise ImportError("text2fx module could not be imported.")

    # Configuration
    refine_config = config.get('refinement', {})
    n_iters = refine_config.get('max_iterations', 100) # Default to 100 for gradient steps
    lr = refine_config.get('learning_rate', 1e-2)
    device = DEVICE # Use device from text2fx constants
    
    # Load CLAP model
    # We can use the one from text2fx or our own wrapper. 
    # For consistency with Text2FX logic, let's use their wrapper if available, 
    # or adapt our existing one. Text2FX relies on specific embedding methods.
    # Let's try to use text2fx's MSCLAPWrapper for now as it's tested with their loss.
    from text2fx.msclap import MSCLAPWrapper
    clap = MSCLAPWrapper()
    
    # Preprocess Audio
    sig = tc.preprocess_audio(audio_path).to(device)
    
    # Create Channel
    channel = get_default_channel(sample_rate=sig.sample_rate)
    
    # Initialize Parameters
    params = torch.nn.parameter.Parameter(
        torch.randn(sig.batch_size, channel.num_params).to(device) 
    )
    params.requires_grad = True
    optimizer = torch.optim.Adam([params], lr=lr)
    
    # Prepare Embeddings
    # 1. Audio Input Embedding (Anchor for audio)
    audio_in_emb = clap.get_audio_embeddings(sig).detach()
    
    # 2. Text Target Embedding
    text_processed = [f"this sound is {user_prompt}"]
    embedding_target = clap.get_text_embeddings(text_processed).detach()
    
    # 3. Text Negative/Anchor Embedding (for directional loss)
    # "this sound is not {user_prompt}" or just "neutral sound"
    # Text2FX uses "this sound is not {t}"
    text_neg_processed = [f"this sound is not {user_prompt}"]
    text_anchor_emb = clap.get_text_embeddings(text_neg_processed).detach()
    
    history = []
    
    # Optimization Loop
    print(f"Starting Text2FX optimization for '{user_prompt}' ({n_iters} steps)...")
    for n in range(n_iters):
        optimizer.zero_grad()
        
        # Apply FX
        # Sigmoid to map params to [0, 1] range expected by dasp_pytorch modules
        signal_effected = channel(sig.clone(), torch.sigmoid(params))
        
        # Get Embedding
        embedding_effected = clap.get_audio_embeddings(signal_effected)
        
        # Calculate Loss (Directional)
        # Direction: (Effected - Input) should match (Target Text - Anchor Text)
        loss = clip_directional_loss(
            embedding_effected, audio_in_emb, 
            embedding_target, text_anchor_emb
        ).mean()
        
        loss.backward()
        optimizer.step()
        
        history.append({'iteration': n, 'loss': loss.item()})
        
        if n % 10 == 0:
            print(f"Step {n}: loss={loss.item():.4f}")
            
    # Final Result
    out_params = params.detach().cpu()
    out_params_dict = channel.save_params_to_dict(out_params)
    
    # Save Output Audio
    # We need a place to save it. Let's use a 'outputs' dir in baseline-system
    output_dir = Path(audio_path).parent.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    output_filename = f"{Path(audio_path).stem}_optimized.wav"
    output_audio_path = output_dir / output_filename
    
    # Render final audio
    final_sig = channel(sig.clone(), torch.sigmoid(params)).detach().cpu()
    final_sig = tc.preprocess_audio(final_sig) # Ensure max norm
    
    # Save
    # AudioSignal.write handles path as str or Path
    final_sig.write(output_audio_path)
    
    return {
        'best_params': out_params_dict,
        'output_audio_path': str(output_audio_path),
        'history': history
    }
