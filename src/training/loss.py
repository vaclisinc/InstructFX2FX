
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
