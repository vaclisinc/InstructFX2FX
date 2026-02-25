from dataclasses import dataclass
from abc import ABC, abstractmethod
import dasp_pytorch

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