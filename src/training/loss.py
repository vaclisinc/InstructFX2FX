
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from configurations.config import OptimizationMethod, LossFunction
from utilities.fx_processing import EQ_ORDER, COMP_ORDER, REVERB_ORDER

# ========== Loss Functions within CLAP ==========

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

    audio_dir = F.normalize(audio_effected - audio_anchor, dim=-1) # FIXME: should we normalize the direction vectors?
    text_dir = F.normalize(text_target - text_anchor, dim=-1)
    return (1 - F.cosine_similarity(audio_dir, text_dir, dim=-1)).mean()


def forward_loss(audio_effected, text_target):
    """Compute forward loss in CLAP embedding space."""
    if not isinstance(audio_effected, torch.Tensor):
        audio_effected = torch.from_numpy(audio_effected)
    if not isinstance(text_target, torch.Tensor):
        text_target = torch.from_numpy(text_target)

    audio_emb = F.normalize(audio_effected, dim=-1) # FIXME: should we normalize the vectors?
    text_emb = F.normalize(text_target, dim=-1)
    return (1 - F.cosine_similarity(audio_emb, text_emb, dim=-1)).mean()


def guided_forward_loss(audio_effected, text_target, text_anchor):
    """Compute guided forward loss in CLAP embedding space."""
    assert text_anchor is not None, "text anchor should not be None"
    if not isinstance(audio_effected, torch.Tensor):
        audio_effected = torch.from_numpy(audio_effected)
    if not isinstance(text_target, torch.Tensor):
        text_target = torch.from_numpy(text_target)
    if not isinstance(text_anchor, torch.Tensor):
        text_anchor = torch.from_numpy(text_anchor)

    audio_emb = F.normalize(audio_effected, dim=-1) # FIXME: should we normalize the vectors?
    text_target_emb = F.normalize(text_target, dim=-1)
    text_anchor_emb = F.normalize(text_anchor, dim=-1)

    target_sim = F.cosine_similarity(audio_emb, text_target_emb, dim=-1)
    anchor_sim = F.cosine_similarity(audio_emb, text_anchor_emb, dim=-1)

    # Encourage similarity to target and dissimilarity to anchor
    return (1 - target_sim + anchor_sim).mean()


# ========== Refinement Loss ==========

def refinement_loss(audio_prev_emb, audio_new_emb, text_prev_emb, text_target_emb,
                    params_prev, params_new, alpha=0.1):
    """Anchored Refinement Loss for iterative audio effect optimization.

    Designed for the sequential re-prompting scenario: given existing parameters
    from a previous stage, refine them according to a new text instruction while
    suppressing parameter changes that do not contribute to the instructed direction.

    L = L_sufficiency + alpha * L_minimality

    L_sufficiency = 1 - cos(delta_a, delta_t)
        Drives the audio to move in the direction specified by the text,
        relative to the previous state (not dry audio).

    L_minimality  = ||delta_theta||^2 * (1 - |cos(delta_a, delta_t)|)
        Penalizes parameter displacement, gated by alignment quality.
        When audio moves in the correct direction, parameter changes are
        unrestricted. When audio drifts in an unrelated direction, parameter
        changes are penalized.

    Args:
        audio_prev_emb:  CLAP embedding of audio with previous params [B, D]
        audio_new_emb:   CLAP embedding of audio with current params [B, D]
        text_prev_emb:   CLAP embedding of previous instruction [B, D]
        text_target_emb: CLAP embedding of current instruction [B, D]
        params_prev:     previous parameters in normalized [0,1] space [B, P]
        params_new:      current parameters in normalized [0,1] space [B, P]
        alpha:           weight for minimality term (default 0.1)

    Returns:
        Scalar loss tensor with gradient.
    """
    # Ensure tensors
    tensors = [audio_prev_emb, audio_new_emb, text_prev_emb, text_target_emb,
               params_prev, params_new]
    for i, t in enumerate(tensors):
        if not isinstance(t, torch.Tensor):
            tensors[i] = torch.from_numpy(t)
    audio_prev_emb, audio_new_emb, text_prev_emb, text_target_emb, \
        params_prev, params_new = tensors

    # Deltas in embedding space
    delta_a = audio_new_emb - audio_prev_emb      # audio direction
    delta_t = text_target_emb - text_prev_emb      # text direction

    # Sufficiency: directional alignment relative to previous state
    delta_a_norm = F.normalize(delta_a, dim=-1)
    delta_t_norm = F.normalize(delta_t, dim=-1)
    cos_align = F.cosine_similarity(delta_a_norm, delta_t_norm, dim=-1)
    L_suf = (1 - cos_align).mean()

    # Minimality: gated parameter-space regularization
    delta_theta = params_new - params_prev.detach()
    param_displacement = (delta_theta ** 2).sum(dim=-1)  # ||delta_theta||^2 per batch
    alignment_gate = (1 - cos_align.detach().abs())       # detach to not double-penalize
    L_min = (param_displacement * alignment_gate).mean()

    return L_suf + alpha * L_min
