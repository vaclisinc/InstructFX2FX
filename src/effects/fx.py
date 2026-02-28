from dataclasses import dataclass
from abc import ABC, abstractmethod
import dasp_pytorch
import numpy as np

class Effect(ABC):
    @abstractmethod
    def apply(self, audio):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @property
    def type(self):
        return self.__class__.__name__

class EQ(Effect):
    def __init__(self, sample_rate=44100):
        self.effect = dasp_pytorch.ParametricEQ(sample_rate=sample_rate)
        self.num_params = self.effect.num_params

    def apply(self, audio, params):
        return self.effect.process_normalized(audio, params)

    def to_dict(self):
        return {
            "type": self.type()
        }

class Compressor(Effect):
    def __init__(self, sample_rate=44100):
        self.effect = dasp_pytorch.Compressor(sample_rate=sample_rate)
        self.num_params = self.effect.num_params
    def to_dict(self):
        return {
            "type": self.type()
        }

    def apply(self, audio, params):
        return self.effect.process_normalized(audio, params)


class Reverb(Effect):
    def __init__(self, sample_rate=44100):
        self.effect = dasp_pytorch.NoiseShapedReverb(sample_rate=sample_rate)
        self.num_params = self.effect.num_params

    def apply(self, audio, params):
        return self.effect.process_normalized(audio, params)

    def to_dict(self):
        return {
            "type": self.type()
        }

class FXChain:
    def __init__(self, effects):
        self.effects = effects

    def __call__(self, audio, params):
        idx = 0
        for effect in self.effects:
            effect_params = params[:, idx:idx + effect.num_params]
            if effect.type == "Reverb":
                idx += effect.num_params
                continue
            audio = effect.apply(audio, effect_params)
            idx += effect.num_params
        return audio

    @property
    def num_params(self):
        return sum(effect.num_params for effect in self.effects)

class FXChainFactory:
    """Factory to create FX chains."""
    @staticmethod
    def create_fx_chain(sample_rate=44100, device='cpu'):
        """Create default FX chain."""
        eq = EQ(sample_rate=sample_rate)
        comp = Compressor(sample_rate=sample_rate)
        reverb = Reverb(sample_rate=sample_rate)
        fx_chain = FXChain([eq, comp, reverb])
        print(f"✓ FX chain created: {fx_chain.num_params} parameters")
        return fx_chain


ALL_PARAM_RANGES = {

    # =================================================
    # 1. 6-BAND PARAMETRIC EQ (18 params)
    # Order per band: freq, gain_db, Q
    # =================================================
    "EQ": {
        "b1_freq": {"lo": 20.0, "hi": 20000.0, "scale": "log"},
        "b1_gain": {"lo": -24.0, "hi": 24.0, "scale": "linear"},
        "b1_q":    {"lo": 0.1, "hi": 10.0, "scale": "log"},

        "b2_freq": {"lo": 20.0, "hi": 20000.0, "scale": "log"},
        "b2_gain": {"lo": -24.0, "hi": 24.0, "scale": "linear"},
        "b2_q":    {"lo": 0.1, "hi": 10.0, "scale": "log"},

        "b3_freq": {"lo": 20.0, "hi": 20000.0, "scale": "log"},
        "b3_gain": {"lo": -24.0, "hi": 24.0, "scale": "linear"},
        "b3_q":    {"lo": 0.1, "hi": 10.0, "scale": "log"},

        "b4_freq": {"lo": 20.0, "hi": 20000.0, "scale": "log"},
        "b4_gain": {"lo": -24.0, "hi": 24.0, "scale": "linear"},
        "b4_q":    {"lo": 0.1, "hi": 10.0, "scale": "log"},

        "b5_freq": {"lo": 20.0, "hi": 20000.0, "scale": "log"},
        "b5_gain": {"lo": -24.0, "hi": 24.0, "scale": "linear"},
        "b5_q":    {"lo": 0.1, "hi": 10.0, "scale": "log"},

        "b6_freq": {"lo": 20.0, "hi": 20000.0, "scale": "log"},
        "b6_gain": {"lo": -24.0, "hi": 24.0, "scale": "linear"},
        "b6_q":    {"lo": 0.1, "hi": 10.0, "scale": "log"},
    },
        # =================================================
    # 2. COMPRESSOR (6 params)
    # =================================================
    "Compressor": {
        "threshold_db":   {"lo": -60.0, "hi": 0.0,  "scale": "linear"},
        "ratio":          {"lo": 1.0,   "hi": 20.0, "scale": "log"},
        "attack":         {"lo": 0.001, "hi": 0.1,  "scale": "log"},
        "release":        {"lo": 0.01,  "hi": 1.0,  "scale": "log"},
        "makeup_gain_db": {"lo": 0.0,   "hi": 24.0, "scale": "linear"},
        "mix":            {"lo": 0.0,   "hi": 1.0,  "scale": "linear"},
    },
        # =================================================
    # 3. REVERB (25 params)
    # =================================================
    "Reverb": {

        # Early reflections (7)
        "early_gain":      {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "early_delay":     {"lo": 0.0, "hi": 0.1, "scale": "linear"},
        "early_diffusion": {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "early_width":     {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "early_lowcut":    {"lo": 20.0, "hi": 1000.0, "scale": "log"},
        "early_highcut":   {"lo": 2000.0, "hi": 20000.0, "scale": "log"},
        "early_mix":       {"lo": 0.0, "hi": 1.0, "scale": "linear"},

        # Late reverb (10)
        "late_gain":       {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "decay_time":      {"lo": 0.1, "hi": 10.0, "scale": "log"},
        "late_diffusion":  {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "density":         {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "mod_rate":        {"lo": 0.0, "hi": 5.0, "scale": "linear"},
        "mod_depth":       {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "late_lowcut":     {"lo": 20.0, "hi": 1000.0, "scale": "log"},
        "late_highcut":    {"lo": 2000.0, "hi": 20000.0, "scale": "log"},
        "late_width":      {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "late_mix":        {"lo": 0.0, "hi": 1.0, "scale": "linear"},

        # Output / global (8)
        "pre_delay":       {"lo": 0.0, "hi": 0.1, "scale": "linear"},
        "damping":         {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "lowcut":          {"lo": 20.0, "hi": 1000.0, "scale": "log"},
        "highcut":         {"lo": 2000.0, "hi": 20000.0, "scale": "log"},
        "wet":             {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "dry":             {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "width":           {"lo": 0.0, "hi": 1.0, "scale": "linear"},
        "mix":             {"lo": 0.0, "hi": 1.0, "scale": "linear"},
    }
}

def sample_param(lo, hi, scale):
    """
    Sample a single parameter given bounds and scale.
    """
    if scale == "linear":
        return np.random.uniform(lo, hi)
    elif scale == "log":
        return np.exp(np.random.uniform(np.log(lo), np.log(hi)))
    else:
        raise ValueError(f"Unknown scale: {scale}")


def initialize_random_params():
    """
    Initialize all parameters defined in ALL_PARAM_RANGES at random.

    Returns:
        dict: nested dict with the same structure as ALL_PARAM_RANGES,
              containing sampled float values.
    """
    params = {}

    for module_name, module_params in ALL_PARAM_RANGES.items():
        params[module_name] = {}
        for param_name, spec in module_params.items():
            params[module_name][param_name] = sample_param(
                lo=spec["lo"],
                hi=spec["hi"],
                scale=spec["scale"],
            )

    return params

def initialize_uniform_params():
    """
    Initialize all parameters defined in ALL_PARAM_RANGES to zeros.

    Returns:
        dict: nested dict with the same structure as ALL_PARAM_RANGES,
              containing zeros.
    """
    params = {}

    for module_name, module_params in ALL_PARAM_RANGES.items():
        params[module_name] = {}
        for param_name, spec in module_params.items():
            params[module_name][param_name] = 0.5

    return params


llm_params_dict_example = {
    'EQ': {
        'b1_freq': 120,
        'b1_gain': 2.5,
        'b1_q': 1.0,
        'b2_freq': 300,
        'b2_gain': 1.5,
        'b2_q': 1.2,
        'b3_freq': 1000,
        'b3_gain': 3.0,
        'b3_q': 0.8,
        'b4_freq': 3000,
        'b4_gain': 5.0,
        'b4_q': 0.7,
        'b5_freq': 6000,
        'b5_gain': 6.0,
        'b5_q': 0.9,
        'b6_freq': 10000,
        'b6_gain': 4.5,
        'b6_q': 0.7
    },
    'Compressor': {
        'threshold_db': -20,
        'ratio': 3.0,
        'attack': 0.01,
        'release': 0.1,
        'makeup_gain_db': 2.0,
        'mix': 0.8
    },
    'Reverb': {
        'early_gain': 0.6,
        'early_delay': 0.01,
        'early_diffusion': 0.7,
        'early_width': 0.8,
        'early_lowcut': 200,
        'early_highcut': 8000,
        'early_mix': 0.7,
        'late_gain': 0.5,
        'decay_time': 1.5,
        'late_diffusion': 0.8,
        'density': 0.9,
        'mod_rate': 0.3,
        'mod_depth': 0.2,
        'late_lowcut': 250,
        'late_highcut': 9000,
        'late_width': 0.75,
        'late_mix': 0.6,
        'pre_delay': 0.02,
        'damping': 0.5,
        'lowcut': 100,
        'highcut': 12000,
        'wet': 0.5,
        'dry': 0.5,
        'width': 0.9,
        'mix': 0.7
    }
}
llm_params_tensor_example = [0.2594, 0.5521, 0.5000, 0.3920, 0.5312, 0.5396, 0.5663, 0.5625, 0.4515,
         0.7254, 0.6042, 0.4225, 0.8257, 0.6250, 0.4771, 0.8997, 0.5938, 0.4225,
         0.6667, 0.3667, 0.5000, 0.5000, 0.0833, 0.8000, 0.6000, 0.1000, 0.7000,
         0.8000, 0.5886, 0.6021, 0.7000, 0.5000, 0.5880, 0.8000, 0.9000, 0.0600,
         0.2000, 0.6456, 0.6532, 0.7500, 0.6000, 0.2000, 0.5000, 0.4114, 0.7782,
         0.5000, 0.5000, 0.9000, 0.7000]

llm_params_dict_example_pedalboard = {'EQ': {'mode': 'shelf-shelf', 'low_cut': 100.0, 'high_cut': 16000.0, 'q': 1.0, 'gains': {'high_shelf': 4.0}, 'peak1_freq': 200.0, 'peak2_freq': 1000.0, 'peak3_freq': 5000.0}, 'Distortion': {'drive_db': 0.0}, 'Reverb': {'room_size': 0.3, 'damping': 0.2, 'wet_level': 0.1}, 'Delay': {'delay': 0.01}, 'PitchShift': {'semitones': 0}, 'Bitcrush': {'bit_depth': 16}}