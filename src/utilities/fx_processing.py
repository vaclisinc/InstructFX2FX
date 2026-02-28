import math
import torch
from effects.fx import ALL_PARAM_RANGES

def normalize_effect_parameters(value, spec):
    lo, hi = spec["lo"], spec["hi"]
    if spec["scale"] == "linear":
        return (value - lo) / (hi - lo)
    elif spec["scale"] == "log":
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        raise ValueError(f"Unknown scale: {spec['scale']}")

EQ_ORDER = [
    "b1_freq","b1_gain","b1_q",
    "b2_freq","b2_gain","b2_q",
    "b3_freq","b3_gain","b3_q",
    "b4_freq","b4_gain","b4_q",
    "b5_freq","b5_gain","b5_q",
    "b6_freq","b6_gain","b6_q",
]

COMP_ORDER = [
    "threshold_db",
    "ratio",
    "attack",
    "release",
    "makeup_gain_db",
    "mix",
]

REVERB_ORDER = [
    # Early
    "early_gain",
    "early_delay",
    "early_diffusion",
    "early_width",
    "early_lowcut",
    "early_highcut",
    "early_mix",

    # Late
    "late_gain",
    "decay_time",
    "late_diffusion",
    "density",
    "mod_rate",
    "mod_depth",
    "late_lowcut",
    "late_highcut",
    "late_width",
    "late_mix",

    # Global / Output
    "pre_delay",
    "damping",
    "lowcut",
    "highcut",
    "wet",
    "dry",
    "width",
    "mix",
]

def fx_initial_params_to_tensor(config, device="cpu", dtype=torch.float32, param_ranges=None):
    """
    Convert grouped FX config dict into a normalized tensor of 49 parameters
    in the exact order expected by FXChain.process_normalized().

    Args:
        config: Dict with effect types as keys ('EQ', 'Compressor', 'Reverb')
                and parameter dicts as values
    """
    if type(config) is torch.Tensor:
        return config.to(device=device, dtype=dtype)

    elif type(config) is not dict:
        raise ValueError(f"Expected config to be a dict or Tensor, got {type(config)}")

    if param_ranges is None:
        param_ranges = ALL_PARAM_RANGES

    eq   = config["EQ"]
    comp = config["Compressor"]
    rev  = config["Reverb"]

    params = []

    # -----------------------------
    # 1) EQ (18)
    # -----------------------------
    for key in EQ_ORDER:
        spec = param_ranges["EQ"][key]
        params.append(normalize_effect_parameters(eq[key], spec))

    # -----------------------------
    # 2) Compressor (6)
    # -----------------------------
    for key in COMP_ORDER:
        spec = param_ranges["Compressor"][key]
        params.append(normalize_effect_parameters(comp[key], spec))

    # -----------------------------
    # 3) Reverb (25)
    # -----------------------------
    for key in REVERB_ORDER:
        spec = param_ranges["Reverb"][key]
        params.append(normalize_effect_parameters(rev[key], spec))

    assert len(params) == 49, f"Expected 49 params, got {len(params)}"

    return torch.tensor(params, device=device, dtype=dtype).unsqueeze(0)