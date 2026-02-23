from dataclasses import dataclass
from abc import ABC, abstractmethod

class Effect(ABC):
    @abstractmethod
    def apply(self, audio):
        pass

    @abstractmethod
    def to_dict(self):
        pass

class EQ(Effect):
    # def __init__(self, mode, low_cut, high_cut, q, gains, peak_freqs):
    #     self.mode = mode
    #     self.low_cut = low_cut
    #     self.high_cut = high_cut
    #     self.q = q
    #     self.gains = gains
    #     self.peak_freqs = peak_freqs

    # def apply(self, audio):
    #     # Placeholder for actual EQ processing logic
    #     pass

    # def to_dict(self):
    #     return {
    #         "type": "EQ",
    #         "mode": self.mode,
    #         "low_cut": self.low_cut,
    #         "high_cut": self.high_cut,
    #         "q": self.q,
    #         "gains": self.gains,
    #         "peak1_freq": self.peak_freqs.get("peak1", None),
    #         "peak2_freq": self.peak_freqs.get("peak2", None),
    #         "peak3_freq": self.peak_freqs.get("peak3", None)
    #     }
