
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from configurations.config import OptimizationMethod
from utilities.fx_processing import EQ_ORDER, COMP_ORDER, REVERB_ORDER

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
    snapshot_interval=None, optimization_method=OptimizationMethod.GRADIENT_DESCENT
):
    """Refine parameters using gradient descent in CLAP space.

    If snapshot_interval is set, saves FX params every N iterations
    for on-demand audio rendering via slider.
    """
    if device is None:
        device = audio.device


    def inverse_sigmoid_torch(y, eps=1e-6):
        y = torch.clamp(y, eps, 1 - eps)
        return torch.log(y / (1 - y))

    # Shorten audio for faster CLAP processing (use first 5 seconds)
    max_clap_samples = 5 * 44100
    if audio.shape[-1] > max_clap_samples:
        audio_short = audio[..., :max_clap_samples]
        print(f"⚡ Using shortened audio for CLAP: {audio_short.shape[-1]/44100:.1f}s instead of {audio.shape[-1]/44100:.1f}s")
    else:
        audio_short = audio

    # Setup - ensure initial_params is on correct device and requires grad
    params = torch.nn.Parameter(inverse_sigmoid_torch(initial_params).clone().detach().to(device).requires_grad_(True))
    optimizer = torch.optim.Adam([params], lr=lr)

    # Get fixed embeddings (no gradients needed for these)
    text_anchor_emb = clap_model.get_text_embedding(text_anchor)
    text_target_emb = clap_model.get_text_embedding(text_target)
    # Use LLM-processed audio as anchor so it semantically aligns with text_anchor

    # Anchor embedding is from the original audio (before effects) to capture the "starting point" in CLAP space
    audio_anchor_emb = clap_model.get_audio_embedding(
        audio_short.clone())

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

    audios = {}
    audios.update({"original": audio_short.detach().cpu()})

    print(f'\n⚡ Starting {optimization_method.name.replace("_", " ").title()}-based refinement for {n_iterations} iterations...')
    if optimization_method.value == OptimizationMethod.GRADIENT_DESCENT.value:
        print(f"⚡ Starting gradient descent refinement...")
        for i in tqdm(range(n_iterations)):
            optimizer.zero_grad()

            # Apply FX with gradient tracking
            audio_effected = fx_chain(audio_short.clone(), torch.sigmoid(params))

            # Get embedding - gradients will flow back through audio_effected
            audio_effected_emb = clap_model.get_audio_embedding(audio_effected)

            if i % 10 == 0:
                audios[f"iter_{i}_in_{optimization_method.name.lower().replace(' ', '_')}"] = audio_effected.detach().cpu()

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

    elif optimization_method.value == OptimizationMethod.BAYESIAN_OPTIMIZATION.value:
        # ---------------------------------------------------------------
        # Bayesian Optimization over directional loss
        # Search space: 49 normalised params in [0, 1] matching FXChain order
        # ---------------------------------------------------------------

        # Build ordered dimension names (same order as FXChain.process_normalized)
        param_order = (
            [f"EQ__{k}" for k in EQ_ORDER]
            + [f"Compressor__{k}" for k in COMP_ORDER]
            + [f"Reverb__{k}" for k in REVERB_ORDER]
        )
        n_params = len(param_order)
        assert n_params == 49, f"Expected 49 params, got {n_params}"

        search_space = [Real(0.0, 1.0, name=name) for name in param_order]
        iterator = Integer(0, n_iterations, name='iteration')

        # Initial point from LLM / preset (already normalised to [0, 1])
        x0 = torch.sigmoid(initial_params.clone().detach()).squeeze().cpu().numpy().tolist()

        # ----- objective -------------------------------------------------
        @use_named_args(search_space)
        def bo_objective(**params_dict):
            param_values = [params_dict[name] for name in param_order]
            param_tensor = torch.tensor(
                param_values, device=device, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                audio_effected = fx_chain(audio_short.clone(), torch.sigmoid(param_tensor))
                audio_effected_emb = clap_model.get_audio_embedding(audio_effected)

                if not isinstance(audio_effected_emb, torch.Tensor):
                    audio_effected_emb = torch.from_numpy(audio_effected_emb).to(device)

                loss = directional_loss(
                    audio_anchor_emb, audio_effected_emb,
                    text_anchor_emb, text_target_emb,
                )

            return float(loss.item())
        # -----------------------------------------------------------------

        # Progress bar + early stopping
        pbar = tqdm(total=n_iterations, desc="Bayesian Optimization", unit="iter")

        best_loss_so_far = np.inf
        stall_counter = 0
        stall_limit = 30

        def _bo_callback(res):
            nonlocal best_loss_so_far, stall_counter
            pbar.update(1)
            current_best = res.fun
            iteration = len(res.func_vals)

            # Record history & snapshots
            history.append({"iteration": iteration, "loss": current_best})

            if snapshot_interval is not None:
                if iteration % snapshot_interval == 0 or iteration == n_iterations:
                    best_params_list = res.x
                    snap = torch.tensor(best_params_list, device=device, dtype=torch.float32).unsqueeze(0)
                    audio_effected = fx_chain(audio_short.clone(), snap)
                    audios[f"iter_{iteration}_in_{optimization_method.name.lower().replace(' ', '_')}"] = audio_effected.detach().cpu()

            if iteration % 20 == 0 or iteration == n_iterations:
                print(f"  Iter {iteration:3d}: best loss = {current_best:.4f}")

            # Early stopping
            if current_best < best_loss_so_far - 1e-4:
                best_loss_so_far = current_best
                stall_counter = 0
            else:
                stall_counter += 1

            if stall_counter >= stall_limit:
                print(f"\n⏹ Early stopping: no improvement in {stall_limit} iterations.")
                return True  # stops gp_minimize

        print(f"\n🎯 Bayesian refinement: '{text_anchor}' → '{text_target}'")
        if snapshot_interval:
            print(f"📸 Saving param snapshots every {snapshot_interval} iterations")

        result = gp_minimize(
            bo_objective,
            search_space,
            n_calls=n_iterations,
            x0=x0,
            acq_func="LCB",
            kappa=5,
            n_initial_points=min(20, n_iterations // 4),
            random_state=42,
            callback=[_bo_callback],
        )
        pbar.close()

        # Build final parameter tensor (already in [0, 1])
        best_params = torch.tensor(
            result.x, device=device, dtype=torch.float32
        ).unsqueeze(0)

        if not history:
            history.append({"iteration": 0, "loss": result.fun})

    if len(history) > 0:
        print(f"✓ Done! Final loss = {history[-1]['loss']:.4f}")
    # if snapshots:
    #     print(f"📸 Saved {len(snapshots)} param snapshots")
    return torch.sigmoid(best_params) if optimization_method == OptimizationMethod.BAYESIAN_OPTIMIZATION else torch.sigmoid(params.detach()), history, audios