import torch
from dataclasses import dataclass
import abc
from typing import Any
from embeddings.clap import CLAPWrapper
from prompts.prompt import Prompt

class Metric (abc.ABC):
    @abc.abstractmethod
    def compute(self, original_audio : Any, target_audio: Any, prompt : Prompt = None):
        raise NotImplementedError("Must implement compute method in subclass")

class CLAPSimilarity(Metric):
    def __init__(self, device='cpu'):
        super().__init__()
        self.clap = CLAPWrapper(device='cuda' if torch.cuda.is_available() else device)

    def compute(self, original_audio, target_audio, prompt : Prompt = None):
        raise NotImplementedError("CLAPSimilarity metric")