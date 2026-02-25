import torch.nn.functional as F
import torch
from tqdm import tqdm

# ========== Text2FX Refinement ==========


def directional_loss(audio_anchor, audio_effected, text_anchor, text_target):
    """Compute directional loss in CLAP embedding space."""
    # Ensure all inputs are torch tensors
    if not isinstance(audio_anchor, torch.Tensor):
        audio_anchor = torch.from_numpy(audio_anchor)
    if not isinstance(audio_effected, torch.Tensor):
        audio_effected = torch.from_numpy(audio_effected)
    if not isinstance(text_anchor, torch.Tensor):
        text_anchor = torch.from_numpy(text_anchor)
    if not isinstance(text_target, torch.Tensor):
        text_target = torch.from_numpy(text_target)

    audio_dir = F.normalize(audio_effected - audio_anchor, dim=-1)
    text_dir = F.normalize(text_target - text_anchor, dim=-1)
    return (1 - F.cosine_similarity(audio_dir, text_dir, dim=-1)).mean()



def refine_with_directional_loss(
    audio, fx_chain, initial_params, text_anchor, text_target,
    clap_model, n_iterations=100, lr=0.01, device=None,
    snapshot_interval=None
):
    """Refine parameters using gradient descent in CLAP space.

    If snapshot_interval is set, saves FX params every N iterations
    for on-demand audio rendering via slider.
    """
    if device is None:
        device = audio.device

    # Shorten audio for faster CLAP processing (use first 5 seconds)
    max_clap_samples = 5 * 44100
    if audio.shape[-1] > max_clap_samples:
        audio_short = audio[..., :max_clap_samples]
        print(f"⚡ Using shortened audio for CLAP: {audio_short.shape[-1]/44100:.1f}s instead of {audio.shape[-1]/44100:.1f}s")
    else:
        audio_short = audio

    # Setup - ensure initial_params is on correct device and requires grad
    params = torch.nn.Parameter(initial_params.clone().detach().to(device).requires_grad_(True))
    optimizer = torch.optim.Adam([params], lr=lr)

    # Get fixed embeddings (no gradients needed for these)
    text_anchor_emb = clap_model.get_text_embedding(text_anchor)
    text_target_emb = clap_model.get_text_embedding(text_target)
    # Use LLM-processed audio as anchor so it semantically aligns with text_anchor
    audio_anchor_emb = clap_model.get_audio_embedding(
        fx_chain(audio_short.clone(), torch.sigmoid(initial_params.clone().detach().to(device)))
    )

    # Ensure all are tensors and detached
    if isinstance(text_anchor_emb, torch.Tensor):
        text_anchor_emb = text_anchor_emb.detach()
    else:
        text_anchor_emb = torch.from_numpy(text_anchor_emb).to(device)

    if isinstance(text_target_emb, torch.Tensor):
        text_target_emb = text_target_emb.detach()
    else:
        text_target_emb = torch.from_numpy(text_target_emb).to(device)

    if isinstance(audio_anchor_emb, torch.Tensor):
        audio_anchor_emb = audio_anchor_emb.detach()
    else:
        audio_anchor_emb = torch.from_numpy(audio_anchor_emb).to(device)

    history = []
    snapshots = {}  # {iteration: params tensor}

    # Save iteration 0 snapshot (before any optimization)
    if snapshot_interval is not None:
        snapshots[0] = params.detach().clone()

    print(f"\n🎯 Refining: '{text_anchor}' → '{text_target}'")
    if snapshot_interval:
        print(f"📸 Saving param snapshots every {snapshot_interval} iterations")

    for i in tqdm(range(n_iterations)):
        optimizer.zero_grad()

        # Apply FX with gradient tracking
        audio_effected = fx_chain(audio_short.clone(), torch.sigmoid(params))

        # Get embedding - gradients will flow back through audio_effected
        audio_effected_emb = clap_model.get_audio_embedding(audio_effected)

        # Ensure it's a tensor
        if not isinstance(audio_effected_emb, torch.Tensor):
            audio_effected_emb = torch.from_numpy(audio_effected_emb).to(device)

        # Compute loss
        loss = directional_loss(
            audio_anchor_emb, audio_effected_emb,
            text_anchor_emb, text_target_emb
        )

        # Update
        loss.backward()
        optimizer.step()

        history.append({'iteration': i, 'loss': loss.item()})
        if i % 20 == 0 or i == n_iterations - 1:
            print(f"  Iter {i:3d}: loss = {loss.item():.4f}")

        # Save param snapshot at intervals and at the last iteration
        if snapshot_interval is not None:
            if (i + 1) % snapshot_interval == 0 or i == n_iterations - 1:
                snapshots[i + 1] = params.detach().clone()

    print(f"✓ Done! Improved {(1 - history[-1]['loss']/history[0]['loss'])*100:.1f}%")
    if snapshots:
        print(f"📸 Saved {len(snapshots)} param snapshots")
    return params.detach(), history, snapshots