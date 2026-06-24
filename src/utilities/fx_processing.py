import math
import torch
from effects.fx import ALL_PARAM_RANGES_DASP

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

# Canonical param order per effect dict-key
EFFECT_PARAM_ORDERS = {
    "EQ":         EQ_ORDER,
    "Compressor": COMP_ORDER,
    "Reverb":     REVERB_ORDER,
}

# Canonical ordering so tensor always matches FXChain order
_CANONICAL_EFFECT_KEYS = ["EQ", "Compressor", "Reverb"]


def fx_tensor_to_params_dict(tensor, effect_keys, param_ranges=None):
    """Inverse of fx_initial_params_to_tensor.

    Converts a normalized [0,1] param tensor back to a nested params dict.

    Args:
        tensor: shape [1, N] normalized params (as returned by move_in_CLAP)
        effect_keys: list of effect dict-keys in chain order (e.g. ['EQ', 'Reverb'])
        param_ranges: param ranges dict (defaults to ALL_PARAM_RANGES_DASP)

    Returns:
        dict like {'EQ': {'b1_freq': 100.0, ...}, 'Reverb': {...}}
    """
    if param_ranges is None:
        param_ranges = ALL_PARAM_RANGES_DASP

    params_flat = tensor.squeeze(0).tolist()
    result = {}
    idx = 0
    for key in effect_keys:
        if key not in EFFECT_PARAM_ORDERS:
            continue
        effect_dict = {}
        for param_name in EFFECT_PARAM_ORDERS[key]:
            spec = param_ranges[key][param_name]
            norm_val = params_flat[idx]
            if spec["scale"] == "linear":
                val = norm_val * (spec["hi"] - spec["lo"]) + spec["lo"]
            elif spec["scale"] == "log":
                val = math.exp(
                    norm_val * (math.log(spec["hi"]) - math.log(spec["lo"])) + math.log(spec["lo"])
                )
            else:
                raise ValueError(f"Unknown scale: {spec['scale']}")
            effect_dict[param_name] = val
            idx += 1
        result[key] = effect_dict
    return result


def fx_initial_params_to_tensor(config, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32, param_ranges=None):
    """Convert a grouped FX config dict into a normalized parameter tensor.

    Iterates over the effects present in `config` in canonical order
    (EQ → Compressor → Reverb → any future effects in _CANONICAL_EFFECT_KEYS).
    Only effects present as keys in `config` are included, so the tensor
    length equals the total params of those effects.

    Args:
        config: Dict mapping effect dict-keys ('EQ', 'Compressor', 'Reverb', …)
                to parameter dicts. May contain any subset of the known effects.
    """
    if type(config) is torch.Tensor:
        return config.to(device=device, dtype=dtype)
    if type(config) is not dict:
        raise ValueError(f"Expected config to be a dict or Tensor, got {type(config)}")

    if param_ranges is None:
        param_ranges = ALL_PARAM_RANGES_DASP

    params = []
    # Iterate in canonical order; skip effects not in the config
    for dict_key in _CANONICAL_EFFECT_KEYS:
        if dict_key not in config:
            continue
        effect_params = config[dict_key]
        for param_name in EFFECT_PARAM_ORDERS[dict_key]:
            spec = param_ranges[dict_key][param_name]
            params.append(normalize_effect_parameters(effect_params[param_name], spec))

    # Clamp to [0,1]: the DASP processors require normalized params in range and
    # raise otherwise. An LLM-emitted value past a param's declared range would
    # normalize outside [0,1]; clamp so it's pinned to the boundary instead.
    return torch.tensor(params, device=device, dtype=dtype).unsqueeze(0).clamp(0.0, 1.0)
