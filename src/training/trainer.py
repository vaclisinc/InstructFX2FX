
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from configurations.config import OptimizationMethod, LossFunction
from utilities.fx_processing import EQ_ORDER, COMP_ORDER, REVERB_ORDER
from .loss import directional_loss, forward_loss, guided_forward_loss

def move_in_CLAP(
    audio,
    fx_chain,
    initial_params,
    target: str | torch.Tensor,
    clap_model,
    loss_function,
    n_iterations=100,
    lr=0.01,
    device=None,
    snapshot_interval=None,
    text_anchor=None,
    optimization_method=OptimizationMethod.GRADIENT_DESCENT,

):
    """Refine parameters using gradient descent in CLAP space.

    If snapshot_interval is set, saves FX params every N iterations
    for on-demand audio rendering via slider.
    """
    if device is None:
        device = audio.device

    print('loss function is', loss_function)

    if loss_function.value == LossFunction.DIRECTIONAL_LOSS.value:
        assert text_anchor is not None, "text_anchor must be provided for directional loss"

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
    params = torch.nn.Parameter(inverse_sigmoid_torch(initial_params).clone().detach().to(device).requires_grad_(True)) # INVERSE SIGMOID
    optimizer = torch.optim.Adam([params], lr=lr)

    if loss_function.value == LossFunction.GUIDED_SEMANTIC_LOSS.value:
        text_anchor="A harsh, distorted, muddy, unclear, oversaturated, unpleasant sound"

    # Get fixed embeddings (no gradients needed for these)
    if text_anchor:
        if isinstance(text_anchor, str):
            text_anchor_emb = clap_model.get_text_embedding(text_anchor)
        elif isinstance(text_anchor, torch.Tensor):
            text_anchor_emb = text_anchor.detach()
    else:
        text_anchor_emb = None

    # Use LLM-processed audio as anchor so it semantically aligns with text_anchor

    # Anchor embedding is from the original audio (before effects) to capture the "starting point" in CLAP space
    audio_anchor_emb = clap_model.get_audio_embedding(
        audio_short.clone())

    # Ensure all are tensors and detached
    if text_anchor_emb is not None:
        if isinstance(text_anchor_emb, torch.Tensor):
            text_anchor_emb = text_anchor_emb.detach()
        else:
            text_anchor_emb = torch.from_numpy(text_anchor_emb).to(device)

    if isinstance(target, torch.Tensor):
        target_emb = target.detach()
    else:
        target_emb = clap_model.get_text_embedding(target)

    if isinstance(audio_anchor_emb, torch.Tensor):
        audio_anchor_emb = audio_anchor_emb.detach()
    else:
        audio_anchor_emb = torch.from_numpy(audio_anchor_emb).to(device)

    loss_history = []
    audio_param_snapshots = {}

    # Save start snapshot (before any optimization)
    audio_param_snapshots["start"] = (fx_chain(audio_short.clone(), torch.sigmoid(params)).cpu(), torch.sigmoid(params).detach().cpu())

    print(f"\n🎯 Refining: '{text_anchor}' → '{target}'")
    if snapshot_interval:
        print(f"📸 Saving snapshots every {snapshot_interval} iterations")

    assert optimization_method is not None, "Optimization method undefined."
    print(f'\n⚡ Starting {optimization_method.name.replace("_", " ").title()}-based refinement for {n_iterations} iterations...')
    if optimization_method.value == OptimizationMethod.GRADIENT_DESCENT.value:
        print(f"⚡ Starting gradient descent refinement...")
        for i in tqdm(range(n_iterations)):
            optimizer.zero_grad()

            # Apply FX with gradient tracking
            audio_effected = fx_chain(audio_short.clone(), torch.sigmoid(params)) # SIGMOID

            # Get embedding - gradients will flow back through audio_effected
            audio_effected_emb = clap_model.get_audio_embedding(audio_effected)

            # Ensure it's a tensor
            if not isinstance(audio_effected_emb, torch.Tensor):
                audio_effected_emb = torch.from_numpy(audio_effected_emb).to(device)

            # Compute loss
            if loss_function.value == LossFunction.DIRECTIONAL_LOSS.value:
                loss = directional_loss(
                    audio_anchor_emb, audio_effected_emb,
                    text_anchor_emb, target_emb
                )
            elif loss_function.value == LossFunction.SEMANTIC_SIMILARITY_LOSS.value:
                loss = forward_loss(audio_effected_emb, target_emb)

            elif loss_function.value == LossFunction.GUIDED_SEMANTIC_LOSS.value:
                loss = guided_forward_loss(audio_effected_emb, target_emb, text_anchor_emb)


            # Update
            loss.backward()
            optimizer.step()

            loss_history.append({'iteration': i, 'loss': loss.item()})

            # Save snapshot at intervals and at the last iteration
            if snapshot_interval is not None:
                if (i + 1) % snapshot_interval == 0 or i == n_iterations - 1:
                    print(f"  Iter {i:3d}: loss = {loss.item():.4f}")
                    audio_param_snapshots[f"iter_{i+1}"] = {"audio": audio_effected.detach().cpu(), "params": params.detach().cpu()} # FIXME: store NON-SIGMOID params for later analysis !

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
        x0 = initial_params.detach().cpu().numpy().flatten().tolist() # NO SIGMOID

        print(f"⚡ Starting Bayesian Optimization refinement with {n_iterations} iterations... {params}")

        # ----- objective -------------------------------------------------
        @use_named_args(search_space)
        def bo_objective(**params_dict):
            param_values = [params_dict[name] for name in param_order]
            param_tensor = torch.tensor(
                param_values, device=device, dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():
                audio_effected = fx_chain(audio_short.clone(), param_tensor)
                audio_effected_emb = clap_model.get_audio_embedding(audio_effected)

                if not isinstance(audio_effected_emb, torch.Tensor):
                    audio_effected_emb = torch.from_numpy(audio_effected_emb).to(device)

                loss = directional_loss(
                    audio_anchor_emb, audio_effected_emb,
                    text_anchor_emb, target_emb,
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

            # Record loss_history & parameter_snapshots
            loss_history.append({"iteration": iteration, "loss": current_best})

            if snapshot_interval is not None:
                if (iteration) % snapshot_interval == 0 or iteration == n_iterations:
                    best_params_list = res.x

                    snap = torch.tensor(best_params_list, device=device, dtype=torch.float32).unsqueeze(0)
                    print(f"  Iter {iteration:3d}: loss = {current_best:.4f}")
                    audio_effected = fx_chain(audio_short.clone(), snap)
                    audio_param_snapshots[f"iter_{iteration}"] = (audio_effected.detach().cpu(), snap.cpu())

            # Early stopping
            if current_best < best_loss_so_far - 1e-4:
                best_loss_so_far = current_best
                stall_counter = 0
            else:
                stall_counter += 1

            if stall_counter >= stall_limit:
                print(f"\n⏹ Early stopping: no improvement in {stall_limit} iterations.")
                return True  # stops gp_minimize

        print(f"\n🎯 Bayesian refinement: '{text_anchor}' → '{target}'")
        if snapshot_interval:
            print(f"📸 Saving snapshots every {snapshot_interval} iterations")

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

        if not loss_history:
            loss_history.append({"iteration": 0, "loss": result.fun})


    if len(loss_history) > 0:
        print(f"✓ Done! Final loss = {loss_history[-1]['loss']:.4f}")

    final_params = best_params if optimization_method == OptimizationMethod.BAYESIAN_OPTIMIZATION else torch.sigmoid(params.detach())
    print(f"📊 Final params (normalized): {final_params.squeeze().cpu().numpy()}")

    # Compute and store final audio with final parameters
    with torch.no_grad():
        final_audio = fx_chain(audio.clone(), final_params).detach().cpu()
        audio_param_snapshots['end'] = (final_audio, final_params.detach().cpu())

    return final_params, final_audio, loss_history, audio_param_snapshots