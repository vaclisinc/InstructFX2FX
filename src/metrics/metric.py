import torch
import abc
from typing import Any


class Metric(abc.ABC):
    @abc.abstractmethod
    def compute(
        self
    ) -> float:
        raise NotImplementedError("Must implement compute method in subclass")