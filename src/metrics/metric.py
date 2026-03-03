import torch
import abc
from typing import Any


class Metric(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        raise NotImplementedError("Must implement compute method in subclass")


class CLAPSimilarity(Metric):
    def __init__(self, device="cpu"):
        super().__init__()
        from embeddings.clap import CLAPWrapper
        self.clap = CLAPWrapper(
            device="cuda" if torch.cuda.is_available() else device
        )

    def compute(
        self,
        original_audio: Any,
        target_audio: Any,
        prompt: Any = None,
    ) -> float:
        raise NotImplementedError("CLAPSimilarity metric")