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
        print(f"Applying FX chain with {params} parameters")
        idx = 0
        for effect in self.effects:
            effect_params = params[:, idx:idx + effect.num_params]
            if effect.type == "Reverb":
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
