from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional

import dasp_pytorch
import numpy as np
import torch
from pedalboard import (
    Pedalboard,
    Bitcrush as PedalboardBitcrush,
    Delay as PedalboardDelay,
    Distortion as PedalboardDistortion,
    HighpassFilter,
    HighShelfFilter,
    LowpassFilter,
    LowShelfFilter,
    PeakFilter,
    PitchShift as PedalboardPitchShift,
    Reverb as PedalboardReverb,
)

class Effect(ABC):
    @abstractmethod
    def apply(self, audio, params):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @property
    def type(self):
        return getattr(self, "effect_name", self.__class__.__name__)

class FXFamily:
    DASP = "DASP"
    PEDALBOARD = "Pedalboard"

@dataclass(frozen=True)
class EffectSpec:
    name: str
    dict_key: str
    backend: str
    effect_cls: Optional[type] = None
    renderer: Optional[Callable] = None


def _to_numpy_audio(audio):
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
        if audio_np.ndim == 3 and audio_np.shape[0] == 1:
            audio_np = audio_np[0]
        return audio_np
    return np.asarray(audio)


def _from_numpy_audio(audio_np, template_audio):
    if isinstance(template_audio, torch.Tensor):
        restored = torch.from_numpy(np.asarray(audio_np)).to(
            device=template_audio.device,
            dtype=template_audio.dtype,
        )
        if template_audio.ndim == 3 and restored.ndim == 2:
            restored = restored.unsqueeze(0)
        return restored
    return audio_np


def _render_pedalboard_plugins(audio, sample_rate, plugins):
    board = Pedalboard(plugins)
    audio_np = _to_numpy_audio(audio)
    rendered = board(audio_np, sample_rate)
    return _from_numpy_audio(rendered, audio)


def _build_pedalboard_eq_plugins(config):
    default_gains = {
        "low_shelf": 3.0,
        "high_shelf": -2.0,
        "peak1": -1.5,
        "peak2": 2.0,
        "peak3": 1.0,
    }
    gains = {**default_gains, **config.get("gains", {})}

    try:
        low_mode, high_mode = config.get("mode", "pass-pass").split("-")
    except ValueError:
        low_mode, high_mode = "pass", "pass"

    q_value = config.get("q", 1.0)
    plugins = []
    if low_mode == "pass":
        plugins.append(HighpassFilter(cutoff_frequency_hz=config.get("low_cut", 80.0)))
    elif low_mode == "shelf":
        plugins.append(
            LowShelfFilter(
                cutoff_frequency_hz=config.get("low_cut", 80.0),
                gain_db=gains["low_shelf"],
                q=q_value,
            )
        )

    if high_mode == "pass":
        plugins.append(LowpassFilter(cutoff_frequency_hz=config.get("high_cut", 14000.0)))
    elif high_mode == "shelf":
        plugins.append(
            HighShelfFilter(
                cutoff_frequency_hz=config.get("high_cut", 14000.0),
                gain_db=gains["high_shelf"],
                q=q_value,
            )
        )

    plugins.extend(
        [
            PeakFilter(
                cutoff_frequency_hz=config.get("peak1_freq", 200.0),
                gain_db=gains["peak1"],
                q=q_value,
            ),
            PeakFilter(
                cutoff_frequency_hz=config.get("peak2_freq", 1000.0),
                gain_db=gains["peak2"],
                q=q_value,
            ),
            PeakFilter(
                cutoff_frequency_hz=config.get("peak3_freq", 5000.0),
                gain_db=gains["peak3"],
                q=q_value,
            ),
        ]
    )
    return plugins


def _render_pedalboard_eq(audio, sample_rate, params):
    return _render_pedalboard_plugins(audio, sample_rate, _build_pedalboard_eq_plugins(params))


def _render_pedalboard_distortion(audio, sample_rate, params):
    return _render_pedalboard_plugins(
        audio,
        sample_rate,
        [PedalboardDistortion(drive_db=params["drive_db"])],
    )


def _render_pedalboard_reverb(audio, sample_rate, params):
    return _render_pedalboard_plugins(
        audio,
        sample_rate,
        [
            PedalboardReverb(
                room_size=params["room_size"],
                damping=params.get("damping", 0.4),
                wet_level=params.get("wet_level", 0.12),
            )
        ],
    )


def _render_pedalboard_delay(audio, sample_rate, params):
    return _render_pedalboard_plugins(
        audio,
        sample_rate,
        [PedalboardDelay(delay_seconds=params["delay"], feedback=0.25)],
    )


def _render_pedalboard_pitchshift(audio, sample_rate, params):
    return _render_pedalboard_plugins(
        audio,
        sample_rate,
        [PedalboardPitchShift(semitones=params["semitones"])],
    )


def _render_pedalboard_bitcrush(audio, sample_rate, params):
    return _render_pedalboard_plugins(
        audio,
        sample_rate,
        [PedalboardBitcrush(bit_depth=params["bit_depth"])],
    )


class DASPEffectAdapter(Effect):
    def __init__(self, effect_name, dict_key, effect_cls, sample_rate=44100):
        self.effect_name = effect_name
        self.dict_key = dict_key
        self.backend = FXFamily.DASP
        self.effect = effect_cls(sample_rate=sample_rate)
        self.num_params = self.effect.num_params

    def apply(self, audio, params):
        return self.effect.process_normalized(audio, params)

    def __call__(self, audio, params):
        return self.apply(audio, params)

    def to_dict(self):
        return {"type": self.type, "backend": self.backend, "dict_key": self.dict_key}


class PedalboardEffectAdapter(Effect):
    def __init__(self, effect_name, dict_key, renderer, sample_rate=44100):
        self.effect_name = effect_name
        self.dict_key = dict_key
        self.backend = FXFamily.PEDALBOARD
        self.renderer = renderer
        self.sample_rate = sample_rate
        self.num_params = None

    def apply(self, audio, params):
        return self.renderer(audio, self.sample_rate, params)

    def to_dict(self):
        return {"type": self.type, "backend": self.backend, "dict_key": self.dict_key}


class EQ(DASPEffectAdapter):
    def __init__(self, sample_rate=44100):
        super().__init__(effect_name="EQ", dict_key="EQ", effect_cls=dasp_pytorch.ParametricEQ, sample_rate=sample_rate)


class Compressor(DASPEffectAdapter):
    def __init__(self, sample_rate=44100):
        super().__init__(
            effect_name="Compressor",
            dict_key="Compressor",
            effect_cls=dasp_pytorch.Compressor,
            sample_rate=sample_rate,
        )


class Reverb(DASPEffectAdapter):
    def __init__(self, sample_rate=44100):
        super().__init__(
            effect_name="Reverb",
            dict_key="Reverb",
            effect_cls=dasp_pytorch.NoiseShapedReverb,
            sample_rate=sample_rate,
        )


class DASPFXChain:
    def __init__(self, effects):
        self.effects = effects
        self.backend = FXFamily.DASP

    def __call__(self, audio, params):
        idx = 0
        for effect in self.effects:
            effect_params = params[:, idx:idx + effect.num_params] # FIXME: have a more solid approach to storing the parameters
            audio = effect.apply(audio, effect_params)
            idx += effect.num_params
        return audio

    @property
    def num_params(self):
        return sum(effect.num_params for effect in self.effects)


class PedalboardFXChain:
    def __init__(self, effects):
        self.effects = effects
        self.backend = FXFamily.PEDALBOARD

    def __call__(self, audio, params):
        current_audio = audio
        for effect in self.effects:
            effect_params = params.get(effect.dict_key, {})
            current_audio = effect.apply(current_audio, effect_params)
        return current_audio

    @property
    def num_params(self):
        return None


FXChain = DASPFXChain


DASP_EFFECT_SPECS = {
    "eq": EffectSpec("eq", "EQ", FXFamily.DASP, effect_cls=dasp_pytorch.ParametricEQ),
    "compressor": EffectSpec("compressor", "Compressor", FXFamily.DASP, effect_cls=dasp_pytorch.Compressor),
    "reverb": EffectSpec("reverb", "Reverb", FXFamily.DASP, effect_cls=dasp_pytorch.NoiseShapedReverb),
}

PEDALBOARD_EFFECT_SPECS = {
    "eq": EffectSpec("eq", "EQ", FXFamily.PEDALBOARD, renderer=_render_pedalboard_eq),
    "distortion": EffectSpec("distortion", "Distortion", FXFamily.PEDALBOARD, renderer=_render_pedalboard_distortion),
    "reverb": EffectSpec("reverb", "Reverb", FXFamily.PEDALBOARD, renderer=_render_pedalboard_reverb),
    "delay": EffectSpec("delay", "Delay", FXFamily.PEDALBOARD, renderer=_render_pedalboard_delay),
    "pitchshift": EffectSpec("pitchshift", "PitchShift", FXFamily.PEDALBOARD, renderer=_render_pedalboard_pitchshift),
    "bitcrush": EffectSpec("bitcrush", "Bitcrush", FXFamily.PEDALBOARD, renderer=_render_pedalboard_bitcrush),
}

BACKEND_EFFECT_REGISTRY = {
    FXFamily.DASP: DASP_EFFECT_SPECS,
    FXFamily.PEDALBOARD: PEDALBOARD_EFFECT_SPECS,
}

# Backward-compatible DASP registry used by existing callers.
EFFECT_REGISTRY = {
    "eq": {"class": EQ, "dict_key": "EQ", "backend": FXFamily.DASP},
    "compressor": {"class": Compressor, "dict_key": "Compressor", "backend": FXFamily.DASP},
    "reverb": {"class": Reverb, "dict_key": "Reverb", "backend": FXFamily.DASP},
}

class FXChainFactory:
    """Factory to create FX chains."""
    @staticmethod
    def create_fx_chain(sample_rate=44100, device='cpu', backend=FXFamily.DASP):
        """Create a default FX chain for the requested backend."""
        if backend == FXFamily.DASP:
            effects = ["eq", "compressor", "reverb"]
        elif backend == FXFamily.PEDALBOARD:
            effects = ["eq", "distortion", "bitcrush", "pitchshift", "delay", "reverb"]
        else:
            raise ValueError(f"Unknown FX family: {backend}")
        return FXChainFactory.create_fx_chain_from_effects(
            effects,
            sample_rate=sample_rate,
            device=device,
            backend=backend,
        )

    @staticmethod
    def create_fx_chain_from_effects(effects: list, sample_rate=44100, device='cpu', backend=FXFamily.DASP):
        """Create an FX chain from a list of effect names for the requested backend."""
        registry = BACKEND_EFFECT_REGISTRY.get(backend)
        if registry is None:
            raise ValueError(f"Unknown FX family: {backend}")

        unknown = [effect_name for effect_name in effects if effect_name not in registry]
        if unknown:
            raise ValueError(f"Unknown effects for {backend}: {unknown}. Valid: {list(registry)}")

        if backend == FXFamily.DASP:
            effect_objects = [
                DASPEffectAdapter(
                    effect_name=registry[effect_name].dict_key,
                    dict_key=registry[effect_name].dict_key,
                    effect_cls=registry[effect_name].effect_cls,
                    sample_rate=sample_rate,
                )
                for effect_name in effects
            ]
            fx_chain = DASPFXChain(effect_objects)
            print(f"✓ DASP FX chain {effects} created: {fx_chain.num_params} parameters")
            return fx_chain

        effect_objects = [
            PedalboardEffectAdapter(
                effect_name=registry[effect_name].dict_key,
                dict_key=registry[effect_name].dict_key,
                renderer=registry[effect_name].renderer,
                sample_rate=sample_rate,
            )
            for effect_name in effects
        ]
        fx_chain = PedalboardFXChain(effect_objects)
        print(f"✓ Pedalboard FX chain {effects} created")
        return fx_chain


ALL_PARAM_RANGES_DASP = {

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

ALL_PARAM_RANGES_PB = {
    "Distortion": {"drive_db": {"lo": 0, "hi": 15, "res": 0.1, "scale": "linear"}},
    "EQ": {
        "mode": {"choices":
                 ["pass-pass", "pass-shelf", "shelf-pass", "shelf-shelf"], "type": "categorical"},
        "low_cut": {"lo": 50, "hi": 500, "res": 10, "scale": "log"},
        "high_cut": {"lo": 8000, "hi": 16000, "res": 100, "scale": "log"},
        "q": {"lo": 0.1, "hi": 10.0, "res": 0.1, "scale": "linear"},
        "gains.low_shelf": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.high_shelf": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak1": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak2": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "gains.peak3": {"lo": -20.0, "hi": 20.0, "res": 0.2, "scale": "linear"},
        "peak1_freq": {"lo": 100.0, "hi": 500.0, "res": 10.0, "scale": "log"},
        "peak2_freq": {"lo": 500.0, "hi": 4000.0, "res": 100.0, "scale": "log"},
        "peak3_freq": {"lo": 4000.0, "hi": 12000.0, "res": 1000.0, "scale": "log"}
    },
    "Reverb": {
        "room_size": {"lo": 0.0, "hi": 1.0, "res": 0.05, "scale": "linear"},
        "damping": {"lo": 0.0, "hi": 1.0, "res": 0.05, "scale": "linear"},
        "wet_level": {"lo": 0.00, "hi": 1.0, "res": 0.01, "scale": "linear"},
    },
    "Delay": {"delay": {"lo": 0.0, "hi": 0.05, "res": 0.01, "scale": "linear"}},
    "PitchShift": {"semitones": {"lo": -12, "hi": 12, "res": 1, "scale": "linear"}},
    "Bitcrush": {"bit_depth": {"lo": 0, "hi": 16, "res": 1, "scale": "linear"}},
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
    Initialize all parameters defined in ALL_PARAM_RANGES_DASP at random.

    Returns:
        dict: nested dict with the same structure as ALL_PARAM_RANGES_DASP,
              containing sampled float values.
    """
    params = {}

    for module_name, module_params in ALL_PARAM_RANGES_DASP.items():
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
    Initialize all parameters defined in ALL_PARAM_RANGES_DASP to zeros.

    Returns:
        dict: nested dict with the same structure as ALL_PARAM_RANGES_DASP,
              containing zeros.
    """
    params = {}

    for module_name, module_params in ALL_PARAM_RANGES_DASP.items():
        params[module_name] = {}
        for param_name, spec in module_params.items():
            params[module_name][param_name] = 0.5

    return params


llm_params_dict_example = {'EQ': {'b1_freq': 100, 'b1_gain': -3, 'b1_q': 0.7, 'b2_freq': 250, 'b2_gain': -2, 'b2_q': 1.0, 'b3_freq': 1000, 'b3_gain': 2.5, 'b3_q': 0.9, 'b4_freq': 3000, 'b4_gain': 5, 'b4_q': 1.2, 'b5_freq': 6000, 'b5_gain': 4, 'b5_q': 1.1, 'b6_freq': 12000, 'b6_gain': 6, 'b6_q': 0.8}, 'Compressor': {'threshold_db': -18, 'ratio': 4, 'attack': 0.003, 'release': 0.15, 'makeup_gain_db': 5, 'mix': 0.8}, 'Reverb': {'early_gain': 0.8, 'early_delay': 0.05, 'early_diffusion': 0.7, 'early_width': 1.0, 'early_lowcut': 100, 'early_highcut': 8000, 'early_mix': 0.6, 'late_gain': 0.7, 'decay_time': 2.5, 'late_diffusion': 0.8, 'density': 0.85, 'mod_rate': 0.5, 'mod_depth': 0.3, 'late_lowcut': 150, 'late_highcut': 9000, 'late_width': 0.9, 'late_mix': 0.7, 'pre_delay': 0.02, 'damping': 0.5, 'lowcut': 80, 'highcut': 16000, 'wet': 0.6, 'dry': 1.0, 'width': 0.95, 'mix': 0.5}}
llm_params_tensor_example = [0.2330, 0.4375, 0.4225, 0.3656, 0.4583, 0.5000, 0.5663, 0.5521, 0.4771,
         0.7254, 0.6042, 0.5396, 0.8257, 0.5833, 0.5207, 0.9261, 0.6250, 0.4515,
         0.7000, 0.4628, 0.2386, 0.5880, 0.2083, 0.8000, 0.8000, 0.5000, 0.7000,
         1.0000, 0.4114, 0.6021, 0.6000, 0.7000, 0.6990, 0.8000, 0.8500, 0.1000,
         0.3000, 0.5151, 0.6532, 0.9000, 0.7000, 0.2000, 0.5000, 0.3544, 0.9031,
         0.6000, 1.0000, 0.9500, 0.5000]
# LLM-generated example parameters for task "Make the sound brighter."

llm_params_dict_example_pedalboard = {'EQ': {'mode': 'shelf-shelf', 'low_cut': 100.0, 'high_cut': 16000.0, 'q': 1.0, 'gains': {'high_shelf': 4.0}, 'peak1_freq': 200.0, 'peak2_freq': 1000.0, 'peak3_freq': 5000.0}, 'Distortion': {'drive_db': 0.0}, 'Reverb': {'room_size': 0.3, 'damping': 0.2, 'wet_level': 0.1}, 'Delay': {'delay': 0.01}, 'PitchShift': {'semitones': 0}, 'Bitcrush': {'bit_depth': 16}}