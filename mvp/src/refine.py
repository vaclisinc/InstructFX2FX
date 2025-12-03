"""
Text2FX Refinement with Directional Loss

This module implements the core refinement loop using CLAP embeddings
and differentiable audio effects.
"""

import torch
import torch.nn.functional as F


def directional_loss(
    audio_anchor: torch.Tensor,
    audio_effected: torch.Tensor,
    text_anchor: torch.Tensor,
    text_target: torch.Tensor
) -> torch.Tensor:
    """
    Compute directional loss between audio and text embeddings.

    The loss encourages the direction from audio_anchor to audio_effected
    to align with the direction from text_anchor to text_target.

    Loss = 1 - cosine_similarity(audio_direction, text_direction)

    Args:
        audio_anchor: Embedding of original/current audio [B, D]
        audio_effected: Embedding of processed audio [B, D]
        text_anchor: Embedding of source description (e.g., "too bright") [B, D]
        text_target: Embedding of target description (e.g., "not bright") [B, D]

    Returns:
        Scalar loss value
    """
    # Compute directional vectors
    audio_direction = audio_effected - audio_anchor
    text_direction = text_target - text_anchor

    # Normalize
    audio_direction = F.normalize(audio_direction, dim=-1)
    text_direction = F.normalize(text_direction, dim=-1)

    # Compute loss (1 - cosine similarity)
    loss = 1 - F.cosine_similarity(audio_direction, text_direction, dim=-1)

    return loss.mean()


def refine_with_directional_loss(
    audio: torch.Tensor,
    fx_chain,
    initial_params: torch.Tensor,
    text_anchor: str,
    text_target: str,
    clap_model,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    device: torch.device = None
) -> tuple[torch.Tensor, list]:
    """
    Refine audio effect parameters using directional loss in CLAP embedding space.

    This is the core refinement loop that:
    1. Applies FX parameters to audio
    2. Encodes processed audio with CLAP
    3. Computes directional loss
    4. Updates parameters via gradient descent

    Args:
        audio: Input audio tensor [B, C, T]
        fx_chain: Differentiable FX chain (dasp_pytorch)
        initial_params: Initial parameter tensor [B, num_params]
        text_anchor: Source text description (e.g., "this sound is too bright")
        text_target: Target text description (e.g., "this sound is not bright")
        clap_model: CLAP model wrapper with get_audio_embedding/get_text_embedding methods
        n_iterations: Number of optimization steps
        learning_rate: Learning rate for Adam optimizer
        device: Torch device

    Returns:
        refined_params: Optimized parameter tensor [B, num_params]
        history: List of dicts with 'iteration' and 'loss' for each step
    """
    if device is None:
        device = audio.device

    # Initialize parameters (make a copy to avoid modifying input)
    params = torch.nn.Parameter(initial_params.clone().detach().to(device))
    params.requires_grad = True

    # Setup optimizer
    optimizer = torch.optim.Adam([params], lr=learning_rate)

    # Get text embeddings (fixed throughout optimization)
    text_anchor_emb = clap_model.get_text_embedding(text_anchor).detach()
    text_target_emb = clap_model.get_text_embedding(text_target).detach()

    # Get original audio embedding (anchor)
    audio_anchor_emb = clap_model.get_audio_embedding(audio).detach()

    # Optimization history
    history = []

    print(f"\n🎯 Starting refinement: {n_iterations} iterations")
    print(f"Text direction: '{text_anchor}' → '{text_target}'")

    for i in range(n_iterations):
        optimizer.zero_grad()

        # Apply FX with current parameters
        # Use sigmoid to map params to [0, 1] range expected by dasp_pytorch
        audio_effected = fx_chain(audio.clone(), torch.sigmoid(params))

        # Get audio embedding
        audio_effected_emb = clap_model.get_audio_embedding(audio_effected)

        # Compute directional loss
        loss = directional_loss(
            audio_anchor=audio_anchor_emb,
            audio_effected=audio_effected_emb,
            text_anchor=text_anchor_emb,
            text_target=text_target_emb
        )

        # Backprop and update
        loss.backward()
        optimizer.step()

        # Record history
        history.append({
            'iteration': i,
            'loss': loss.item()
        })

        # Print progress
        if i % 10 == 0 or i == n_iterations - 1:
            print(f"  Iteration {i:3d}: loss = {loss.item():.4f}")

    print(f"\n✓ Refinement complete!")
    print(f"  Initial loss: {history[0]['loss']:.4f}")
    print(f"  Final loss: {history[-1]['loss']:.4f}")
    print(f"  Improvement: {(1 - history[-1]['loss'] / history[0]['loss']) * 100:.1f}%")

    return params.detach(), history


def refine_with_cosine_loss(
    audio: torch.Tensor,
    fx_chain,
    initial_params: torch.Tensor,
    text_target: str,
    clap_model,
    n_iterations: int = 100,
    learning_rate: float = 0.01,
    device: torch.device = None
) -> tuple[torch.Tensor, list]:
    """
    Refine parameters using cosine loss (Text2FX-cosine variant).

    Instead of directional loss, directly minimizes distance between
    audio embedding and text embedding.

    Loss = 1 - cosine_similarity(audio_embedding, text_embedding)

    Args:
        audio: Input audio tensor [B, C, T]
        fx_chain: Differentiable FX chain
        initial_params: Initial parameter tensor [B, num_params]
        text_target: Target text description
        clap_model: CLAP model wrapper
        n_iterations: Number of optimization steps
        learning_rate: Learning rate
        device: Torch device

    Returns:
        refined_params: Optimized parameter tensor
        history: Optimization history
    """
    if device is None:
        device = audio.device

    params = torch.nn.Parameter(initial_params.clone().detach().to(device))
    params.requires_grad = True

    optimizer = torch.optim.Adam([params], lr=learning_rate)

    # Get fixed text embedding
    text_target_emb = clap_model.get_text_embedding(text_target).detach()

    history = []

    print(f"\n🎯 Starting cosine refinement: {n_iterations} iterations")
    print(f"Target: '{text_target}'")

    for i in range(n_iterations):
        optimizer.zero_grad()

        # Apply FX
        audio_effected = fx_chain(audio.clone(), torch.sigmoid(params))

        # Get audio embedding
        audio_emb = clap_model.get_audio_embedding(audio_effected)

        # Compute cosine loss
        loss = 1 - F.cosine_similarity(audio_emb, text_target_emb, dim=-1).mean()

        # Backprop
        loss.backward()
        optimizer.step()

        history.append({
            'iteration': i,
            'loss': loss.item()
        })

        if i % 10 == 0 or i == n_iterations - 1:
            print(f"  Iteration {i:3d}: loss = {loss.item():.4f}")

    print(f"\n✓ Cosine refinement complete!")
    print(f"  Initial loss: {history[0]['loss']:.4f}")
    print(f"  Final loss: {history[-1]['loss']:.4f}")

    return params.detach(), history
