import math
import torch

def normalize(value, spec):
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

def fx_initial_params_to_tensor(config, device="cpu", dtype=torch.float32, ALL_PARAM_RANGES=None):
    """
    Convert grouped FX config JSON into a normalized tensor of 49 parameters
    in the exact order expected by FXChain.process_normalized().
    """

    eq   = next(c for c in config if c["type"] == "EQ")
    comp = next(c for c in config if c["type"] == "Compressor")
    rev  = next(c for c in config if c["type"] == "Reverb")

    params = []

    # -----------------------------
    # 1) EQ (18)
    # -----------------------------
    for key in EQ_ORDER:
        spec = ALL_PARAM_RANGES["EQ"][key]
        params.append(normalize(eq[key], spec))

    # -----------------------------
    # 2) Compressor (6)
    # -----------------------------
    for key in COMP_ORDER:
        spec = ALL_PARAM_RANGES["Compressor"][key]
        params.append(normalize(comp[key], spec))

    # -----------------------------
    # 3) Reverb (25)
    # -----------------------------
    for key in REVERB_ORDER:
        spec = ALL_PARAM_RANGES["Reverb"][key]
        params.append(normalize(rev[key], spec))

    assert len(params) == 49, f"Expected 49 params, got {len(params)}"

    return torch.tensor(params, device=device, dtype=dtype).unsqueeze(0)