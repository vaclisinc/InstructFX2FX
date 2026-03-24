from .metric import Metric


class FXEncMetric(Metric):
    def __init__(self, name: str = "FXEncMetric") -> None:
        super().__init__(name)